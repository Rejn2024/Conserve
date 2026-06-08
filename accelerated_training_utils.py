#!/usr/bin/env python3
"""Utilities to accelerate tone-pulse training epochs.

Implements four performance-focused capabilities:
1) one-time precompute/cache of deterministic inputs,
2) DataLoader-based input pipeline with worker prefetching,
3) batch-oriented score path helpers,
4) mixed-precision + torch.compile setup helpers.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tx_controller_tone_pulse_stft_varlen_9 import (
    FIRST_PASS_SCALAR_FEATURE_NAMES,
    FIRST_PASS_SCALAR_FEATURE_SCHEMA_VERSION,
    build_controlled_tone_pulse_batch_from_iq_batches,
    build_first_pass_scalar_side_from_iq_batch,
    preprocess_batched_iq_to_stft_feature,
    tone_pulse_action_dim,
)

import advanced_link_skdsp_v7_robust as link7
import score_iq_decode as scorer


def _to_complex_tensor(x: Any) -> torch.Tensor:
    return torch.as_tensor(x, dtype=link7.DEFAULT_COMPLEX_DTYPE)

def repeat_to_length_mod(arr, target_length):
    if arr.ndim != 1:
        raise ValueError("Input tensor must be 1D")
    if arr.numel() == 0:
        raise ValueError("Input tensor must not be empty")

    idx = torch.arange(target_length, device=arr.device) % arr.numel()
    return arr[idx]


def _sample_numeric_suffix(name: str) -> Optional[int]:
    """Return the trailing numeric suffix from a sample file/directory name."""

    match = re.search(r"(\d+)(?:\.[^.]+)?$", name)
    if match is None:
        return None
    return int(match.group(1))




def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Split an ``s3://bucket/prefix`` URI into bucket and normalized prefix."""

    if not isinstance(s3_uri, str) or not s3_uri.startswith("s3://"):
        raise ValueError("s3_uri must be a string beginning with 's3://'")
    without_scheme = s3_uri[5:]
    bucket, sep, prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError("s3_uri must include a bucket name")
    return bucket, prefix.strip("/")


def _s3_join(prefix: str, name: str) -> str:
    return f"{prefix.rstrip('/')}/{name}" if prefix else name


def _get_s3_client(s3_client: Optional[Any] = None) -> Any:
    if s3_client is not None:
        return s3_client
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for S3 cache helpers. Install it in the AWS "
            "JupyterLab environment with `pip install boto3`."
        ) from exc
    return boto3.client("s3")


def _s3_object_exists(client: Any, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        response = getattr(exc, "response", {}) or {}
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return False
        raise


def _s3_list_cache_records(client: Any, bucket: str, prefix: str) -> List[str]:
    records: List[str] = []
    list_prefix = _s3_join(prefix, "sample_") if prefix else "sample_"

    if hasattr(client, "get_paginator"):
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=list_prefix)
    else:
        pages = []
        token: Optional[str] = None
        while True:
            kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": list_prefix}
            if token:
                kwargs["ContinuationToken"] = token
            page = client.list_objects_v2(**kwargs)
            pages.append(page)
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")

    for page in pages:
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            if key.endswith(".pt") and Path(key).name.startswith("sample_"):
                records.append(key)
    return sorted(records)


def _build_training_cache_record(
    sdir: Path,
    jammer_sampling_freq: float,
    section_len: Optional[int],
    resample: Callable[[Any, float, float], Any],
    *,
    cache_stft_features: bool = False,
    stft_device: str = "cpu",
) -> Dict[str, Any]:
    import load_tx_iq_data as loadmod

    whole = loadmod.load_whole_iq(sdir)
    sample_rate_hz = float(whole["meta"]["sample_rate_hz"])
    iq = _to_complex_tensor(resample(whole["iq"], sample_rate_hz, jammer_sampling_freq))
    if section_len is not None:
        if section_len <= 0:
            raise ValueError("section_len must be > 0 when provided")
        iq = iq[:section_len]

    record = {
        "sample_name": sdir.name,
        "source_dir": str(sdir),
        "whole_iq": _to_complex_tensor(whole["iq"]),
        "whole_meta": whole["meta"],
        "whole_sample_rate_hz": sample_rate_hz,
        "jammer_sampling_freq": float(jammer_sampling_freq),
        "iq": iq,
    }
    if cache_stft_features:
        record["stft_feature_list"] = compute_stft_feature_list_for_iq_batch(
            iq=iq,
            intake_sample_rate_hz=jammer_sampling_freq,
            device=stft_device,
            squeeze_batch=True,
            output_device="cpu",
        )
    return record


def _normalize_cached_stft_feature_list(
    stft_feature_list: Sequence[Any], *, device: str = "cpu"
) -> List[Any]:
    """Move the single cached frequency/timing STFT view to ``device``."""
    if len(stft_feature_list) != 1:
        raise ValueError("stft_feature_list must contain one complete-IQ feature view")
    view = stft_feature_list[0]
    if isinstance(view, dict):
        return [{key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in view.items()}]
    return [torch.as_tensor(view, dtype=torch.float32, device=device)]


def compute_stft_feature_list_for_iq_batch(
    *, iq: torch.Tensor, intake_sample_rate_hz: float, device: str = "cpu",
    squeeze_batch: bool = False, output_device: str = "cpu",
) -> List[Dict[str, torch.Tensor]]:
    """Compute native frequency/timing STFT maps for one complete IQ object."""
    iq_t = torch.as_tensor(iq, dtype=link7.DEFAULT_COMPLEX_DTYPE, device=device)
    if iq_t.ndim == 1:
        iq_t = iq_t.unsqueeze(0)
    if iq_t.ndim != 2:
        raise ValueError("IQ tensor must have shape [samples] or [batch, samples]")
    proc = preprocess_batched_iq_to_stft_feature(iq_t, sample_rate_hz=intake_sample_rate_hz)
    view = {
        "frequency_feature": proc["frequency_feature"].to(device=output_device, dtype=torch.float32),
        "timing_feature": proc["timing_feature"].to(device=output_device, dtype=torch.float32),
    }
    if squeeze_batch:
        if iq_t.shape[0] != 1:
            raise ValueError("squeeze_batch=True requires one IQ object")
        view = {key: value.squeeze(0).contiguous() for key, value in view.items()}
    return [view]


def compute_stft_feature_list_for_iq_sections(**kwargs: Any) -> List[Dict[str, torch.Tensor]]:
    """Compatibility wrapper joining legacy iq1/iq2/iq3 inputs into one object."""
    iq = kwargs.pop("iq", None)
    if iq is None:
        parts = [kwargs.pop(key) for key in ("iq1", "iq2", "iq3") if key in kwargs]
        if not parts:
            raise ValueError("an iq tensor is required")
        iq = torch.cat([part.unsqueeze(0) if part.ndim == 1 else part for part in parts], dim=-1)
    return compute_stft_feature_list_for_iq_batch(iq=iq, **kwargs)


def _collate_stft_feature_list(batch: Sequence[Dict[str, Any]]) -> Optional[List[Any]]:
    if not batch or not all("stft_feature_list" in row for row in batch):
        return None
    views = [row["stft_feature_list"] for row in batch]
    if any(len(view) != 1 for view in views):
        raise ValueError("stft_feature_list must contain one complete-IQ feature view")
    def pad_and_stack(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        max_h = max(int(tensor.shape[-2]) for tensor in tensors)
        max_w = max(int(tensor.shape[-1]) for tensor in tensors)
        padded = [
            torch.nn.functional.pad(
                tensor,
                (0, max_w - int(tensor.shape[-1]), 0, max_h - int(tensor.shape[-2])),
            )
            for tensor in tensors
        ]
        return torch.stack(padded, dim=0)

    first = views[0][0]
    if isinstance(first, dict):
        return [{
            key: pad_and_stack([torch.as_tensor(view[0][key], dtype=torch.float32) for view in views])
            for key in first
        }]
    return [pad_and_stack([torch.as_tensor(view[0], dtype=torch.float32) for view in views])]


def _resolve_sample_dirs(dataset_root: Path, max_numeric_suffix: Optional[int]) -> List[Path]:
    import load_tx_iq_data as loadmod

    sample_dirs = loadmod.list_sample_dirs(dataset_root)
    if max_numeric_suffix is not None:
        if max_numeric_suffix < 0:
            raise ValueError("max_numeric_suffix must be non-negative")
        sample_dirs = [
            sdir
            for sdir in sample_dirs
            if (suffix := _sample_numeric_suffix(sdir.name)) is not None
            and suffix <= max_numeric_suffix
        ]
    return sample_dirs

def precompute_training_cache(
    dataset_root: Path,
    cache_root: Path,
    jammer_sampling_freq: float,
    *,
    section_len: Optional[int] = None,
    overwrite: bool = False,
    max_numeric_suffix: Optional[int] = None,
    resample_fn: Optional[Callable[[Any, float, float], Any]] = None,
    cache_stft_features: bool = False,
    stft_device: str = "cpu",
) -> List[Path]:
    """Precompute deterministic sample tensors once and save to cache files.

    Each cached sample stores:
    - whole_iq (complex32 when available, otherwise complex64)
    - whole_sample_rate_hz (float)
    - one complete IQ object resampled to jammer_sampling_freq at its native length
    - metadata + source path for debugging/auditability

    Args:
        dataset_root: Directory containing ``sample_<number>`` sample directories.
        cache_root: Directory where ``.pt`` cache records and the manifest are written.
        jammer_sampling_freq: Target sample rate for the cached complete IQ tensor.
        section_len: Optional maximum number of resampled IQ values to keep. By
            default the complete transmission is retained without padding.
        overwrite: Rebuild existing cache records when True.
        max_numeric_suffix: Optional inclusive upper limit for the trailing numeric
            suffix of sample directory names. For example, ``100`` processes
            ``sample_000100`` and lower while skipping ``sample_000101``.
        resample_fn: Optional dependency injection hook for custom resampling.
        cache_stft_features: When True, also stores deterministic STFT feature
            native frequency/timing maps for the complete IQ object.
        stft_device: Device used while computing STFT features; features are
            moved back to CPU before they are written to the cache record.
    """

    if resample_fn is None:
        import advanced_link_skdsp_v7_robust as link7

        resample = link7.resample_iq
    else:
        resample = resample_fn
    dataset_root = Path(dataset_root)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = _resolve_sample_dirs(dataset_root, max_numeric_suffix)

    produced: List[Path] = []

    for sdir in sample_dirs:
        out_path = cache_root / f"{sdir.name}.pt"
        if out_path.exists() and not overwrite:
            produced.append(out_path)
            continue

        record = _build_training_cache_record(
            sdir=sdir,
            jammer_sampling_freq=jammer_sampling_freq,
            section_len=section_len,
            resample=resample,
            cache_stft_features=cache_stft_features,
            stft_device=stft_device,
        )

        # print(f'record["iq"].shape : {record["iq"].shape}')

        torch.save(record, out_path)
        produced.append(out_path)

    manifest = {
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root),
        "jammer_sampling_freq": float(jammer_sampling_freq),
        "section_len": None if section_len is None else int(section_len),
        "max_numeric_suffix": max_numeric_suffix,
        "cache_stft_features": bool(cache_stft_features),
        "num_samples": len(produced),
        "files": [str(p.name) for p in produced],
    }
    with open(cache_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return produced


def precompute_training_cache_s3(
    dataset_root: Path,
    cache_s3_uri: str,
    jammer_sampling_freq: float,
    *,
    section_len: Optional[int] = None,
    overwrite: bool = False,
    max_numeric_suffix: Optional[int] = None,
    resample_fn: Optional[Callable[[Any, float, float], Any]] = None,
    cache_stft_features: bool = False,
    stft_device: str = "cpu",
    s3_client: Optional[Any] = None,
) -> List[str]:
    """Precompute deterministic sample tensors and upload cache records to S3.

    ``cache_s3_uri`` is supplied by the caller and must be an ``s3://bucket/prefix``
    location.  The helper writes one ``sample_*.pt`` object per sample plus a
    ``manifest.json`` object under that prefix.  It is intended for AWS-hosted
    notebook workflows where cache generation happens on a JupyterLab instance
    and the resulting records should persist in S3.
    """

    if resample_fn is None:
        import advanced_link_skdsp_v7_robust as link7

        resample = link7.resample_iq
    else:
        resample = resample_fn

    dataset_root = Path(dataset_root)
    bucket, prefix = _parse_s3_uri(cache_s3_uri)
    client = _get_s3_client(s3_client)
    sample_dirs = _resolve_sample_dirs(dataset_root, max_numeric_suffix)

    produced: List[str] = []

    for sdir in sample_dirs:
        key = _s3_join(prefix, f"{sdir.name}.pt")
        if not overwrite and _s3_object_exists(client, bucket, key):
            produced.append(f"s3://{bucket}/{key}")
            continue

        record = _build_training_cache_record(
            sdir=sdir,
            jammer_sampling_freq=jammer_sampling_freq,
            section_len=section_len,
            resample=resample,
            cache_stft_features=cache_stft_features,
            stft_device=stft_device,
        )
        buffer = io.BytesIO()
        torch.save(record, buffer)
        buffer.seek(0)
        client.upload_fileobj(buffer, bucket, key)
        produced.append(f"s3://{bucket}/{key}")

    manifest = {
        "dataset_root": str(dataset_root),
        "cache_root": cache_s3_uri.rstrip("/"),
        "cache_s3_uri": cache_s3_uri.rstrip("/"),
        "jammer_sampling_freq": float(jammer_sampling_freq),
        "section_len": None if section_len is None else int(section_len),
        "max_numeric_suffix": max_numeric_suffix,
        "cache_stft_features": bool(cache_stft_features),
        "num_samples": len(produced),
        "files": [Path(uri).name for uri in produced],
        "s3_uris": produced,
    }
    body = json.dumps(manifest, indent=2).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=_s3_join(prefix, "manifest.json"),
        Body=body,
        ContentType="application/json",
    )

    return produced


class CachedIQDataset(Dataset):
    """Dataset that reads per-sample cached .pt records."""

    def __init__(self, cache_root: Path):
        self.cache_root = Path(cache_root)
        self.records = sorted(self.cache_root.glob("sample_*.pt"))
        if not self.records:
            raise ValueError(f"No cache records found in {self.cache_root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return torch.load(self.records[idx],
                          map_location="cpu",
                          weights_only=False)



def collate_cached_iq(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate cached records into a batch dict.

    Both original and resampled IQ objects may have variable lengths. The
    resampled tensors are padded only to the longest item in this batch, and
    their native lengths are returned in ``iq_lengths``.
    """

    iq_list = [x["iq"] for x in batch]
    iq_lengths = torch.as_tensor([int(iq.numel()) for iq in iq_list], dtype=torch.long)
    max_len = int(iq_lengths.max().item())
    padded_iq = [torch.nn.functional.pad(iq, (0, max_len - int(iq.numel()))) for iq in iq_list]
    out = {
        "sample_names": [x["sample_name"] for x in batch],
        "source_dirs": [x["source_dir"] for x in batch],
        "whole_iq_list": [x["whole_iq"] for x in batch],
        "whole_meta_list": [x["whole_meta"] for x in batch],
        "whole_sr_list": [float(x["whole_sample_rate_hz"]) for x in batch],
        "iq_list": iq_list,
        "iq_lengths": iq_lengths,
        "iq": torch.stack(padded_iq, dim=0),
    }
    stft_feature_list = _collate_stft_feature_list(batch)
    if stft_feature_list is not None:
        out["stft_feature_list"] = stft_feature_list
    return out


class CachedIQDatasetS3(Dataset):
    """Dataset that reads per-sample cached ``.pt`` records from S3."""

    def __init__(self, cache_s3_uri: str, *, s3_client: Optional[Any] = None):
        self.cache_s3_uri = cache_s3_uri.rstrip("/")
        self.bucket, self.prefix = _parse_s3_uri(cache_s3_uri)
        self._s3_client = s3_client
        client = _get_s3_client(s3_client)
        self.records = _s3_list_cache_records(client, self.bucket, self.prefix)
        if not self.records:
            raise ValueError(f"No cache records found in {self.cache_s3_uri}")

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        # boto3 clients are not pickle-friendly; each DataLoader worker should
        # create its own client lazily after fork/spawn.
        state["_s3_client"] = None
        return state

    @property
    def s3_client(self) -> Any:
        if self._s3_client is None:
            self._s3_client = _get_s3_client()
        return self._s3_client

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.records[idx]
        buffer = io.BytesIO()
        self.s3_client.download_fileobj(self.bucket, key, buffer)
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu", weights_only=False)


def create_cached_dataloader(
    cache_root: Path,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> DataLoader:
    ds = CachedIQDataset(cache_root)
    kwargs: Dict[str, Any] = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_cached_iq,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers

    return DataLoader(**kwargs)


def create_cached_dataloader_s3(
    cache_s3_uri: str,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    s3_client: Optional[Any] = None,
) -> DataLoader:
    """Create a DataLoader over S3-backed cached IQ records.

    The returned batches use the same ``collate_cached_iq`` structure as
    ``create_cached_dataloader`` (``iq``, ``whole_iq_list``, metadata lists, etc.), so it can be passed directly
    to ``JammerVecEnv`` and the ``train_rl_batched`` workflow in
    ``RL_Jamming_test_02.ipynb``.
    """

    ds = CachedIQDatasetS3(cache_s3_uri, s3_client=s3_client)
    kwargs: Dict[str, Any] = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_cached_iq,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers

    return DataLoader(**kwargs)


def maybe_compile_model(model: torch.nn.Module, enabled: bool = True) -> torch.nn.Module:
    """Compile model when supported (PyTorch 2.x), otherwise return unchanged.

    TorchDynamo + Inductor compilation can fail lazily on first invocation
    (for example when Triton is unavailable). In that case we fallback to the
    original eager model so training can continue.
    """
    if not enabled:
        return model
    try:
        compiled_model = torch.compile(model)
    except Exception:
        return model
    return _CompileFallbackModel(compiled_model=compiled_model, eager_model=model)


class _CompileFallbackModel(torch.nn.Module):
    """Wrapper that falls back to eager mode if compiled execution fails."""

    def __init__(self, *, compiled_model: torch.nn.Module, eager_model: torch.nn.Module):
        super().__init__()
        self._compiled_model = compiled_model
        self._eager_model = eager_model
        self._use_eager = False

    def __getattr__(self, name: str):
        """Proxy unknown attributes to the wrapped eager model.

        This preserves access to custom model attributes (e.g. `max_tones`)
        even when the model is wrapped for compile fallback.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            eager_model = super().__getattr__("_eager_model")
            return getattr(eager_model, name)

    def forward(self, *args, **kwargs):
        if self._use_eager:
            return self._eager_model(*args, **kwargs)
        try:
            return self._compiled_model(*args, **kwargs)
        except Exception as exc:
            if not _should_fallback_to_eager(exc):
                raise
            self._use_eager = True
            return self._eager_model(*args, **kwargs)


def _should_fallback_to_eager(exc: Exception) -> bool:
    """Return True when compiled execution should fallback to eager mode."""
    message = str(exc).lower()

    compile_runtime_markers = (
        "triton",
        "torch._inductor",
        "cuda error",
        "illegal memory access",
        "device-side assert",
        "cuda kernel errors might be asynchronously reported",
    )
    return any(marker in message for marker in compile_runtime_markers)


def autocast_context(device: str, enabled: bool = True, dtype: Optional[torch.dtype] = None):
    if not enabled or device != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    use_dtype = dtype or torch.float16
    return torch.autocast(device_type="cuda", dtype=use_dtype)


def compute_batch_scores(
    *,
    jam_batch: Sequence[Dict[str, Any]],
    whole_iq_list: Sequence[torch.Tensor],
    whole_meta_list: Sequence[Dict[str, Any]],
    whole_sr_list: Sequence[float],
    jammer_sampling_freq: float,
    criterion: Callable[[torch.Tensor, torch.Tensor, Dict[str, Any]], torch.Tensor],
    repeat_to_length_fn: Callable[[Any, int], Any],
    device: str,
    resample_fn: Optional[Callable[[Any, float, float], Any]] = None,
) -> torch.Tensor:
    """Compute per-row scores with a batch-oriented API and shared bookkeeping.

    This keeps metadata-driven decode/scoring in one place and allows future
    criterion vectorization by swapping in a criterion that accepts batched input.
    """

    if resample_fn is None:
        import advanced_link_skdsp_v7_robust as link7

        resample = link7.resample_iq
    else:
        resample = resample_fn
    scores: List[torch.Tensor] = []

    for whole_iq, whole_meta, whole_sr, jam_item in zip(
        whole_iq_list,
        whole_meta_list,
        whole_sr_list,
        jam_batch,
    ):
        jam_iq_rx_resam = resample(jam_item["tx_iq"], jammer_sampling_freq, whole_sr)
        jam_iq_rx_resam = repeat_to_length_fn(jam_iq_rx_resam, whole_iq.shape[0])
        jam_iq_t = torch.as_tensor(jam_iq_rx_resam[: whole_iq.shape[0]], dtype=link7.DEFAULT_COMPLEX_DTYPE, device=device)
        whole_iq_t = whole_iq.to(device=device, non_blocking=True)
        jammed = whole_iq_t + jam_iq_t
        scores.append(criterion(jammed, whole_iq_t, whole_meta))

    if not scores:
        return torch.empty((0,), dtype=torch.float32, device=device)
    return torch.stack(scores)





def _as_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_action(action: Any, *, max_tones: Optional[int] = None, max_pulses: Optional[int] = None) -> Dict[str, Any]:
    """Normalize PPO action payloads into controller kwargs.

    Supported forms:
    - dict with optional keys:
        desired_output_iq_len, user_peak_power_fraction, seed,
        noise_color, fading_mode, burst_color, rf_center_hz, carrier_hz,
        num_tones, base_f, spacing, amp_raw, pulse_count,
        start_offset_samples, pulse_phase_rel_rad
    - sequence/tensor:
        * up to 3 items mapped to desired_output_iq_len, user_peak_power_fraction, seed
        * when max_tones is known, the current ActorCritic layout is
          decoded as 12 + 4*max_tones controls.  The three extra
          Gaussian dimensions govern recurrent pulse phase, length, and power
          controllers without storing per-pulse columns.
        * > 3 legacy items are interpreted as continuous controls in this order:
          noise_color, fading_mode, burst_color, rf_center_hz, carrier_hz,
          num_tones, base_f, spacing, [amp_raw...], legacy pulse_on_samples,
          legacy pulse_off_samples, pulse_count, start_offset_samples.  The
          legacy on/off values are parsed only for backward compatibility and
          are not forwarded as recurrent-controller action overrides.
    - scalar/tensor scalar interpreted as user_peak_power_fraction
    """

    if isinstance(action, dict):
        return dict(action)

    if torch.is_tensor(action):
        if action.ndim == 0:
            return {"user_peak_power_fraction": float(action.detach().cpu().item())}
        action = action.detach().cpu().reshape(-1).tolist()

    if isinstance(action, (list, tuple)):
        if len(action) > 3:
            vec = [float(x) for x in action]
            if max_tones is not None and len(vec) == tone_pulse_action_dim(int(max_tones)):
                mt = int(max_tones)
                idx = 0
                out = {
                    "noise_color": vec[idx],
                    "fading_mode": vec[idx + 1],
                    "rf_center_delta_hz": vec[idx + 2],
                    "carrier_hz_norm": vec[idx + 3],
                    "num_tones": vec[idx + 4],
                }
                idx += 5
                out["tone_freq_mean_norms"] = vec[idx : idx + mt]
                idx += mt
                out["tone_freq_std_norms"] = vec[idx : idx + mt]
                idx += mt
                out["tone_amp_raw"] = vec[idx : idx + mt]
                out["amp_raw"] = out["tone_amp_raw"]
                idx += mt
                out["tone_phase_rel_rad"] = vec[idx : idx + mt]
                idx += mt
                out["tone_phase_offset_rad"] = vec[idx]
                idx += 1
                out["pulse_phase_ar_control"] = vec[idx]
                out["pulse_length_ar_control"] = vec[idx + 1]
                out["pulse_power_ar_control"] = vec[idx + 2]
                idx += 3
                out["pulse_phase_offset_rad"] = vec[idx]
                idx += 1
                out["pulse_count"] = vec[idx]
                out["start_offset_samples"] = vec[idx + 1]
                return out

            if len(vec) < 12:
                raise ValueError("continuous action vector must contain at least 12 values")
            amp_width = len(vec) - 12
            return {
                "noise_color": vec[0],
                "fading_mode": vec[1],
                "burst_color": vec[2],
                "rf_center_hz": vec[3],
                "carrier_hz": vec[4],
                "num_tones": vec[5],
                "base_f": vec[6],
                "spacing": vec[7],
                "amp_raw": vec[8 : 8 + amp_width],
                "pulse_on_samples": vec[8 + amp_width],
                "pulse_off_samples": vec[9 + amp_width],
                "pulse_count": vec[10 + amp_width],
                "start_offset_samples": vec[11 + amp_width],
            }

        out: Dict[str, Any] = {}
        if len(action) >= 1:
            out["desired_output_iq_len"] = action[0]
        if len(action) >= 2:
            out["user_peak_power_fraction"] = action[1]
        if len(action) >= 3:
            out["seed"] = action[2]
        return out

    if action is None:
        return {}

    return {"user_peak_power_fraction": action}


_ACTION_OVERRIDE_KEYS = (
    "noise_color",
    "fading_mode",
    "burst_color",
    "rf_center_hz",
    "carrier_hz",
    "num_tones",
    "base_f",
    "spacing",
    "amp_raw",
    "pulse_count",
    "start_offset_samples",
    "sample_rate_hz",
    "sample_rate_scale",
    "rf_center_delta_hz",
    "carrier_hz_norm",
    "tone_frequencies_hz",
    "tone_freq_mean_norms",
    "tone_frequency_std_hz",
    "tone_freq_std_norms",
    "tone_amplitudes",
    "tone_amp_raw",
    "tone_initial_phases_rad",
    "tone_phase_rel_rad",
    "tone_phase_offset_rad",
    "pulse_phase_rel_rad",
    "pulse_phase_rel_mix_logits",
    "pulse_phase_rel_mix_loc_rad",
    "pulse_phase_rel_mix_concentration",
    "pulse_phase_offset_rad",
    "pulse_phase_rotations_rad",
    "pulse_phase_ar_control",
    "pulse_length_ar_control",
    "pulse_power_ar_control",
    "pulse_length_log",
    "pulse_lengths_samples",
    "pulse_power_logit",
    "pulse_power_amplitudes",
    "snr_db",
    "freq_offset",
    "timing_offset",
    "rician_k_db",
    "burst_probability",
    "burst_power_ratio_db",
    "peak_power",
    "seed",
)


def _action_overrides_from_cfg(action_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: action_cfg[k] for k in _ACTION_OVERRIDE_KEYS if k in action_cfg}


def jammer_controller(
    *,
    model: torch.nn.Module,
    sample: Dict[str, Any],
    action: Any,
    jammer_sampling_freq: float,
    device: str = "cpu",
    default_output_len: int = 100_000,
    default_peak_power_fraction: float = 40.0,
    default_seed: int = 11,
) -> Dict[str, Any]:
    """Concrete (sample, action) adapter around build_controlled_tone_pulse_batch_from_iq_batches."""

    synthesis_model = getattr(model, "backbone", model)
    action_cfg = _normalize_action(
        action,
        max_tones=getattr(synthesis_model, "max_tones", None),
        max_pulses=getattr(synthesis_model, "max_pulses", None),
    )
    desired_output_iq_len = _as_int(action_cfg.get("desired_output_iq_len"), default_output_len)
    user_peak_power_fraction = _as_float(action_cfg.get("user_peak_power_fraction"), default_peak_power_fraction)
    seed = _as_int(action_cfg.get("seed"), default_seed)

    action_overrides = _action_overrides_from_cfg(action_cfg)

    jam_batch = build_controlled_tone_pulse_batch_from_iq_batches(
        model=synthesis_model,
        rx_iq_batches=[sample["iq"].unsqueeze(0)],
        intake_sample_rate_hz=jammer_sampling_freq,
        desired_output_iq_len=desired_output_iq_len,
        user_peak_power_fraction=user_peak_power_fraction,
        action_overrides=[action_overrides if action_overrides else None],
        seed=seed,
        device=device,
    )
    return jam_batch[0]


def jammer_controller_batch(
    *,
    model: torch.nn.Module,
    samples: Sequence[Dict[str, Any]],
    actions: Sequence[Any],
    jammer_sampling_freq: float,
    device: str = "cpu",
    rx_iq_batches: Optional[Sequence[torch.Tensor]] = None,
    action_cfg: Optional[Dict[str, Any]] = None,
    default_output_len: int = 100_000,
    user_peak_power_fraction: float = 40.0,
    default_seed: int = 11,
) -> List[Dict[str, Any]]:
    """Batch adapter for vectorized env rollouts.

    For maximal throughput this performs one model forward call for the whole vector,
    while preserving per-row action overrides for waveform synthesis.
    """

    synthesis_model = getattr(model, "backbone", model)
    max_tones = getattr(synthesis_model, "max_tones", None)
    max_pulses = getattr(synthesis_model, "max_pulses", None)

    if torch.is_tensor(actions):
        if actions.ndim == 1:
            actions_len = 1
            first_action = actions
            action_rows = [actions]
        else:
            actions_len = int(actions.shape[0])
            first_action = actions[0]
            action_rows = [actions[i] for i in range(actions_len)]
    else:
        actions_len = len(actions)
        first_action = actions[0] if actions_len else None
        action_rows = list(actions)

    if len(samples) != actions_len:
        raise ValueError("samples and actions must have the same length")
    if not samples:
        return []

    if action_cfg is None:
        action_cfgs = [
            _normalize_action(row, max_tones=max_tones, max_pulses=max_pulses)
            for row in action_rows
        ]
        action_cfg = action_cfgs[0]
    else:
        # Explicit action_cfg remains a hot-path escape hatch for callers that
        # intentionally do not want to materialize CUDA action tensors on CPU.
        action_cfgs = [dict(action_cfg) for _ in range(actions_len)]

    desired_output_iq_len = _as_int(action_cfg.get("desired_output_iq_len"), default_output_len)
    seed = _as_int(action_cfg.get("seed"), default_seed)
    action_overrides = [
        overrides if overrides else None
        for overrides in (_action_overrides_from_cfg(cfg) for cfg in action_cfgs)
    ]

    if rx_iq_batches is None:
        iq = torch.stack([sample["iq"] for sample in samples], dim=0).to(dtype=link7.DEFAULT_COMPLEX_DTYPE, device=device)
    else:
        if len(rx_iq_batches) != 1:
            raise ValueError("rx_iq_batches must contain one complete IQ batch")
        iq = torch.as_tensor(rx_iq_batches[0], dtype=link7.DEFAULT_COMPLEX_DTYPE, device=device)

    return build_controlled_tone_pulse_batch_from_iq_batches(
        model=synthesis_model,
        rx_iq_batches=[iq],
        intake_sample_rate_hz=jammer_sampling_freq,
        desired_output_iq_len=desired_output_iq_len,
        user_peak_power_fraction=user_peak_power_fraction,
        action_overrides=action_overrides,
        seed=seed,
        device=device,
    )


class _SamplePool:
    """Eager-or-lazy sample source used by JammerVecEnv.

    DataLoader inputs can contain many cached IQ tensors.  Keeping them lazy avoids
    loading the full cache into RAM when constructing the vectorized environment.
    """

    def __init__(
        self,
        samples: Iterable[Dict[str, Any]],
        *,
        lazy: bool,
        expand_batch_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
    ) -> None:
        self.lazy = bool(lazy)
        self._expand_batch_fn = expand_batch_fn
        self._iterable: Optional[Iterable[Dict[str, Any]]] = samples if lazy else None
        self._iterator: Optional[Iterator[Dict[str, Any]]] = None
        self._buffer: List[Dict[str, Any]] = []
        self._batches_seen = 0
        try:
            self._known_batches = int(len(samples))  # type: ignore[arg-type]
        except Exception:
            self._known_batches = 0

        if self.lazy:
            self.items: List[Dict[str, Any]] = []
        else:
            self.items = []
            for item in samples:
                self.items.extend(self._expand_item(item))
            if not self.items:
                raise ValueError("samples must be non-empty")

    def __len__(self) -> int:
        if self.lazy:
            return self._known_batches
        return len(self.items)

    def _expand_item(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(item, dict):
            raise TypeError("samples must contain dict entries or cached batch dicts")
        if "iq" not in item:
            raise ValueError("each sample/batch dict must include one complete iq object")

        iq = item["iq"]
        if torch.is_tensor(iq) and iq.ndim == 2:
            return self._expand_batch_fn(item)
        return [item]

    def _next_iter_item(self) -> Tuple[Dict[str, Any], bool]:
        if self._iterable is None:
            raise RuntimeError("lazy sample pool has no iterable")

        if self._iterator is None:
            self._iterator = iter(self._iterable)

        did_wrap = False
        while True:
            try:
                item = next(self._iterator)
                self._batches_seen += 1
                if self._known_batches and self._batches_seen >= self._known_batches:
                    did_wrap = True
                    self._batches_seen = 0
                return item, did_wrap
            except StopIteration:
                self._iterator = iter(self._iterable)
                did_wrap = True
                self._batches_seen = 0
                try:
                    item = next(self._iterator)
                    self._batches_seen += 1
                    if self._known_batches and self._batches_seen >= self._known_batches:
                        self._batches_seen = 0
                    return item, did_wrap
                except StopIteration as exc:
                    raise ValueError("samples must be non-empty") from exc

    def next_samples(self, count: int, cursor: int) -> Tuple[List[Dict[str, Any]], int, bool]:
        if count <= 0:
            return [], cursor, False

        if not self.lazy:
            did_wrap = False
            out: List[Dict[str, Any]] = []
            for _ in range(count):
                out.append(self.items[cursor % len(self.items)])
                cursor += 1
                if cursor >= len(self.items):
                    did_wrap = True
            return out, cursor, did_wrap

        did_wrap = False
        while len(self._buffer) < count:
            item, wrapped = self._next_iter_item()
            did_wrap = did_wrap or wrapped
            self._buffer.extend(self._expand_item(item))

        out = self._buffer[:count]
        del self._buffer[:count]
        return out, cursor, did_wrap


class JammerVecEnv:
    """Vectorized jammer environment with batched controller calls for PPO rollouts."""

    def __init__(
        self,
        *,
        samples: Iterable[Dict[str, Any]],
        test_samples: Optional[Iterable[Dict[str, Any]]] = None,
        model: torch.nn.Module,
        jammer_sampling_freq: float,
        num_envs: int,
        reward_fn: Optional[Callable[[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]], torch.Tensor]] = None,
        default_output_len: int = 100_000,
        max_steps: int = 1,
        user_peak_power_fraction: float = 40.0,
        device: str = "cuda",
        track_env_grad: bool = False,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")

        try:
            self._source_batches_per_epoch = int(len(samples))  # type: ignore[arg-type]
        except Exception:
            self._source_batches_per_epoch = 0

        self._train_pool = self._build_sample_pool(samples)
        if not self._train_pool.lazy and not self._train_pool.items:
            raise ValueError("samples must be non-empty")
        self._test_pool = self._build_sample_pool(test_samples) if test_samples is not None else None

        # Backwards-compatible aliases for callers that inspect eager sample lists.
        # Lazy DataLoader-backed environments keep these empty to avoid materializing
        # the complete dataset in memory.
        self.samples = self._train_pool.items
        self.test_samples = self._test_pool.items if self._test_pool is not None else []
        self.model = model
        self.jammer_sampling_freq = float(jammer_sampling_freq)
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.device = device
        self.reward_fn = reward_fn or self._default_reward
        self.user_peak_power_fraction = user_peak_power_fraction
        self.track_env_grad = bool(track_env_grad)
        self.default_output_len = default_output_len

        self._mode = "train"
        self._cursor = {"train": 0, "test": 0}
        self._step_count = 0
        self._active_samples: List[Dict[str, Any]] = []
        self._epoch_complete = False

    @staticmethod
    def _expand_cached_batch(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        iq = batch["iq"]
        if not torch.is_tensor(iq):
            raise TypeError("cached batch iq must be a torch tensor")
        if iq.ndim != 2:
            raise ValueError("cached batch iq must have shape [batch, samples]")

        bs = int(iq.shape[0])
        stft_feature_list = batch.get("stft_feature_list")
        out: List[Dict[str, Any]] = []
        for i in range(bs):
            row = {
                "sample_name": batch.get("sample_names", [None] * bs)[i],
                "source_dir": batch.get("source_dirs", [None] * bs)[i],
                "whole_iq": batch.get("whole_iq_list", [None] * bs)[i],
                "whole_meta": batch.get("whole_meta_list", [None] * bs)[i],
                "whole_sample_rate_hz": batch.get("whole_sr_list", [None] * bs)[i],
                "iq": iq[i],
            }
            if stft_feature_list is not None:
                view = stft_feature_list[0]
                if isinstance(view, dict):
                    row["stft_feature_list"] = [{key: value[i] for key, value in view.items()}]
                else:
                    row["stft_feature_list"] = [view[i]]
            out.append(row)
        return out

    @classmethod
    def _coerce_samples(cls, samples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Materialize samples for legacy callers that pass in-memory collections."""

        return _SamplePool(
            samples,
            lazy=False,
            expand_batch_fn=cls._expand_cached_batch,
        ).items

    @classmethod
    def _build_sample_pool(cls, samples: Optional[Iterable[Dict[str, Any]]]) -> _SamplePool:
        if samples is None:
            raise ValueError("samples must be non-empty")

        # DataLoader inputs are the common high-memory path: iterating them here
        # loads every cached tensor into RAM.  Keep them lazy and only retain the
        # active rollout rows plus any partially consumed batch.  Non-sequence
        # iterables are also kept lazy so generators are not accidentally drained.
        lazy = isinstance(samples, DataLoader) or not isinstance(samples, (list, tuple))
        return _SamplePool(
            samples,
            lazy=lazy,
            expand_batch_fn=cls._expand_cached_batch,
        )

    def _active_pool(self) -> _SamplePool:
        if self._mode == "test":
            if self._test_pool is None:
                raise ValueError("test_samples were not provided")
            return self._test_pool
        return self._train_pool

    def set_mode(self, mode: str) -> None:
        if mode not in ("train", "test"):
            raise ValueError("mode must be 'train' or 'test'")
        if mode == "test" and self._test_pool is None:
            raise ValueError("test_samples were not provided")
        self._mode = mode
        self._step_count = 0
        self._active_samples = []
        self._epoch_complete = False

    @property
    def _active(self) -> List[Optional[str]]:
        """Expose UUIDs for the currently active IQ samples."""

        uuids: List[Optional[str]] = []
        for idx, sample in enumerate(self._active_samples):
            uuid_val = None
            for key in ("whole_iq_uuid", "uuid", "sample_uuid", "id"):
                if isinstance(sample, dict):
                    value = sample.get(key)
                    if value is not None:
                        uuid_val = str(value)
                        break
            if uuid_val is None:
                sample_name = sample.get("sample_name") if isinstance(sample, dict) else None
                uuid_val = str(sample_name) if sample_name else f"active_sample_{idx}"
            uuids.append(uuid_val)
        return uuids

    @_active.setter
    def _active(self, value: Sequence[Any]) -> None:
        rows = list(value) if value is not None else []
        if rows and not isinstance(rows[0], dict):
            raise ValueError("_active stores sample dictionaries; received UUID-only values")
        self._active_samples = rows

    @property
    def mode(self) -> str:
        return self._mode

    def _next_samples(self) -> List[Dict[str, Any]]:
        pool = self._active_pool()
        mode = self._mode
        out, cursor, did_wrap = pool.next_samples(self.num_envs, self._cursor[mode])
        self._cursor[mode] = cursor
        self._epoch_complete = did_wrap
        return out

    def _default_reward(self,
                        jam_batch: Sequence[Dict[str, Any]],
                        samples: Sequence[Dict[str, Any]],
                        alpha: float = 1.0) -> Tuple[torch.Tensor, int, int]:
        """Default reward used by PPO loops.

        Mirrors the decode-side objective from `JammerLoss` in
        `generating_sample_transmissions.ipynb`:
            reward = score_decode(rx_result, whole_meta) + alpha * metric_div
        with `alpha=10`.

        If decode/score inputs are unavailable for a row, this falls back to a
        cheap energy proxy for that row.
        """

        # alpha = 10.0
        vals: List[torch.Tensor] = []

        decode_success = 0
        total = 0

        for jam_item, sample in zip(jam_batch, samples):
            tx_iq = jam_item["tx_iq"]
            whole_meta = sample.get("whole_meta")
            whole_iq: torch.Tensor = sample.get("whole_iq")
            jam_iq_rx_resam = link7.resample_iq(tx_iq,
                                                self.jammer_sampling_freq,
                                                whole_meta['sample_rate_hz'])
            jam_iq_rx_resam = repeat_to_length_mod(jam_iq_rx_resam, whole_iq.shape[0])
            jam_iq_rx_resam_t = torch.as_tensor(jam_iq_rx_resam[:whole_iq.shape[0]],
                                                dtype=link7.DEFAULT_COMPLEX_DTYPE,
                                                device=self.device)
            jammed = whole_iq.to(self.device) + jam_iq_rx_resam_t
            rx_result = link7.rx_command_iq(jammed, whole_meta)

            # score = torch.tensor(1.0, dtype=torch.float32)
            total += 1

            score = torch.as_tensor(scorer.score_decode(rx_result, whole_meta), dtype=torch.float32)

            if rx_result.get("message") is not None:
                decode_success += 1

                # metric_div = torch.as_tensor(rx_result.get("metric_div", 0.0), dtype=torch.float32)
                # score = score + (alpha * metric_div)

            vals.append(score)#.detach().cpu())

        return torch.stack(vals), decode_success, total #.to(dtype=torch.float32)

    def _obs_from_samples(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        iq = torch.stack([sample["iq"] for sample in samples], dim=0)
        obs: Dict[str, Any] = {
            "iq": iq,
            "scalar_side": build_first_pass_scalar_side_from_iq_batch(iq, self.jammer_sampling_freq),
            "scalar_feature_names": FIRST_PASS_SCALAR_FEATURE_NAMES,
            "scalar_feature_schema": FIRST_PASS_SCALAR_FEATURE_SCHEMA_VERSION,
        }
        stft_feature_list = _collate_stft_feature_list(samples)
        if stft_feature_list is not None:
            obs["stft_feature_list"] = stft_feature_list
        return obs

    def reset(self) -> Dict[str, Any]:
        self._step_count = 0
        self._active_samples = self._next_samples()
        return self._obs_from_samples(self._active_samples)

    def step(self, actions: Sequence[Any]):
        # Convenience for single-env training loops: if a single action vector
        # (e.g. shape [action_dim]) is provided, wrap it into a batch.  Batched
        # tensors/arrays are accepted directly to avoid an eager per-row CPU list
        # conversion in high-throughput training loops.
        if isinstance(actions, torch.Tensor):
            action_count = 1 if actions.ndim == 1 else int(actions.shape[0])
            if self.num_envs == 1 and actions.ndim == 1:
                actions = actions.unsqueeze(0)
        elif isinstance(actions, np.ndarray):
            action_count = 1 if actions.ndim == 1 else int(actions.shape[0])
            if self.num_envs == 1 and actions.ndim == 1:
                actions = actions.reshape(1, -1)
        else:
            action_count = len(actions)

        if action_count != self.num_envs:
            raise ValueError(f"actions is of length {action_count} but must contain {self.num_envs} entries")
        if not self._active_samples:
            self._active_samples = self._next_samples()

        with torch.set_grad_enabled(self.track_env_grad):
            jam_batch = jammer_controller_batch(
                model=self.model,
                samples=self._active_samples,
                actions=actions,
                jammer_sampling_freq=self.jammer_sampling_freq,
                user_peak_power_fraction = self.user_peak_power_fraction,
                device=self.device,
            )
            rewards_t, success, total = self.reward_fn(jam_batch, self._active_samples)

        rewards = torch.as_tensor(rewards_t)#, dtype=torch.float32)#.cpu().numpy()
        if not self.track_env_grad:
            rewards = rewards.detach()

        self._step_count += 1
        done = self._step_count >= self.max_steps
        dones = [bool(done)] * self.num_envs

        infos = [
            {
                "tx_metadata": jam_item.get("tx_metadata", {}),
                "sample_name": sample.get("sample_name"),
                "mode": self._mode,
                "epoch_complete": bool(self._epoch_complete),
            }
            for jam_item, sample in zip(jam_batch, self._active_samples)
        ]

        if done:
            self._active_samples = self._next_samples()
            next_obs = self._obs_from_samples(self._active_samples)
            self._step_count = 0
        else:
            next_obs = self._obs_from_samples(self._active_samples)

        return next_obs, rewards, dones, infos, success, total


def run_epoch_cached(
    *,
    dataloader: Iterable[Dict[str, Any]],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: Callable[[torch.Tensor, torch.Tensor, Dict[str, Any]], torch.Tensor],
    jammer_sampling_freq: float,
    repeat_to_length_fn: Callable[[Any, int], Any],
    train_mode: bool,
    device: str,
    output_len: int,
    amp_enabled: bool = True,
    amp_dtype: Optional[torch.dtype] = None,
    grad_scaler: Optional[torch.amp.GradScaler] = None,
) -> Optional[float]:
    """Epoch loop over cached DataLoader with optional AMP and grad scaling."""

    if train_mode:
        model.train()
    else:
        model.eval()

    losses: List[float] = []

    def _optimizer_has_any_grad(opt: torch.optim.Optimizer) -> bool:
        for group in opt.param_groups:
            for param in group["params"]:
                if param is not None and param.grad is not None:
                    return True
        return False

    for batch_idx, batch in enumerate(dataloader):
        iq = batch["iq"].to(device=device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            with autocast_context(device=device, enabled=amp_enabled, dtype=amp_dtype):
                jam_batch = build_controlled_tone_pulse_batch_from_iq_batches(
                    model=model,
                    rx_iq_batches=[iq],
                    intake_sample_rate_hz=jammer_sampling_freq,
                    desired_output_iq_len=output_len,
                    user_peak_power_fraction=40.0,
                    seed=11 + batch_idx * iq.shape[0],
                    device=device,
                )

                score_t = compute_batch_scores(
                    jam_batch=jam_batch,
                    whole_iq_list=batch["whole_iq_list"],
                    whole_meta_list=batch["whole_meta_list"],
                    whole_sr_list=batch["whole_sr_list"],
                    jammer_sampling_freq=jammer_sampling_freq,
                    criterion=criterion,
                    repeat_to_length_fn=repeat_to_length_fn,
                    device=device,
                )
                if score_t.numel() == 0:
                    continue
                loss = score_t.mean()

            if train_mode and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if not loss.requires_grad:
                    continue
                if grad_scaler is not None and device == "cuda" and amp_enabled:
                    grad_scaler.scale(loss).backward()
                    did_step = False
                    if _optimizer_has_any_grad(optimizer):
                        grad_scaler.step(optimizer)
                        did_step = True
                    if did_step:
                        grad_scaler.update()
                else:
                    loss.backward()
                    if _optimizer_has_any_grad(optimizer):
                        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    if not losses:
        return None
    return float(sum(losses) / len(losses))


def build_stft_observation_from_iq_batch(
    *, iq: torch.Tensor, intake_sample_rate_hz: float, device: str = "cpu",
    stft_feature_list: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Build an ActorCritic observation from one complete IQ object per row."""
    scalar_side = build_first_pass_scalar_side_from_iq_batch(iq, intake_sample_rate_hz, device=device)
    obs: Dict[str, Any] = {
        "iq": iq,
        "scalar_side": scalar_side,
        "scalar_feature_names": FIRST_PASS_SCALAR_FEATURE_NAMES,
        "scalar_feature_schema": FIRST_PASS_SCALAR_FEATURE_SCHEMA_VERSION,
    }
    if stft_feature_list is not None:
        obs["stft_feature_list"] = _normalize_cached_stft_feature_list(stft_feature_list, device=device)
    else:
        obs["stft_feature_list"] = compute_stft_feature_list_for_iq_batch(
            iq=iq, intake_sample_rate_hz=intake_sample_rate_hz, device=device,
            squeeze_batch=False, output_device=device,
        )
    return obs


def build_stft_observation_from_samples(
    samples: Sequence[Dict[str, Any]], *, intake_sample_rate_hz: float,
    device: str = "cpu", use_cached_stft: bool = True,
) -> Dict[str, Any]:
    """Build a single-IQ ActorCritic observation from sample dictionaries."""
    cached = _collate_stft_feature_list(samples) if use_cached_stft else None
    return build_stft_observation_from_iq_batch(
        iq=torch.stack([sample["iq"] for sample in samples], dim=0),
        intake_sample_rate_hz=intake_sample_rate_hz, device=device, stft_feature_list=cached,
    )
