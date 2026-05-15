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
    build_controlled_tone_pulse_batch_from_iq_batches,
    preprocess_batched_iq_to_stft_feature,
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
    section_len: int,
    resample: Callable[[Any, float, float], Any],
    *,
    cache_stft_features: bool = False,
    stft_device: str = "cpu",
) -> Dict[str, Any]:
    import load_tx_iq_data as loadmod

    whole = loadmod.load_whole_iq(sdir)
    sections = loadmod.load_sections(sdir)
    sample_rate_hz = float(whole["meta"]["sample_rate_hz"])

    iq1 = _to_complex_tensor(
        resample(sections["sections"][0], sample_rate_hz, jammer_sampling_freq)[:section_len]
    )
    iq2 = _to_complex_tensor(
        resample(sections["sections"][1], sample_rate_hz, jammer_sampling_freq)[:section_len]
    )
    iq3 = _to_complex_tensor(
        resample(sections["sections"][2], sample_rate_hz, jammer_sampling_freq)[:section_len]
    )

    record = {
        "sample_name": sdir.name,
        "source_dir": str(sdir),
        "whole_iq": _to_complex_tensor(whole["iq"]),
        "whole_meta": whole["meta"],
        "whole_sample_rate_hz": sample_rate_hz,
        "jammer_sampling_freq": float(jammer_sampling_freq),
        "iq1": iq1,
        "iq2": iq2,
        "iq3": iq3,
    }
    if cache_stft_features:
        record["stft_feature_list"] = compute_stft_feature_list_for_iq_sections(
            iq1=iq1,
            iq2=iq2,
            iq3=iq3,
            intake_sample_rate_hz=jammer_sampling_freq,
            device=stft_device,
            squeeze_batch=True,
            output_device="cpu",
        )
    return record


def _normalize_cached_stft_feature_list(
    stft_feature_list: Sequence[torch.Tensor],
    *,
    device: str = "cpu",
) -> List[torch.Tensor]:
    """Return cached STFT feature maps as float32 tensors on ``device``.

    Cache records store one tensor per IQ view.  Per-sample records normally use
    shape ``[C, F, T]`` while collated batches use ``[B, C, F, T]``; both forms
    are accepted and preserved by this helper.
    """

    if len(stft_feature_list) != 3:
        raise ValueError("stft_feature_list must contain exactly three feature tensors")
    return [
        torch.as_tensor(feature, dtype=torch.float32, device=device)
        for feature in stft_feature_list
    ]


def compute_stft_feature_list_for_iq_sections(
    *,
    iq1: torch.Tensor,
    iq2: torch.Tensor,
    iq3: torch.Tensor,
    intake_sample_rate_hz: float,
    device: str = "cpu",
    squeeze_batch: bool = False,
    output_device: str = "cpu",
) -> List[torch.Tensor]:
    """Compute deterministic STFT feature tensors for cached IQ sections.

    ``iq1``/``iq2``/``iq3`` may be either single examples of shape ``[N]`` or
    already-batched tensors of shape ``[B, N]``.  The output mirrors the actor
    input order and can be saved directly in cache records under
    ``stft_feature_list``.
    """

    iq_batches = []
    for iq in (iq1, iq2, iq3):
        iq_t = torch.as_tensor(iq, dtype=link7.DEFAULT_COMPLEX_DTYPE, device=device)
        if iq_t.ndim == 1:
            iq_t = iq_t.unsqueeze(0)
        if iq_t.ndim != 2:
            raise ValueError("IQ section tensors must have shape [samples] or [batch, samples]")
        iq_batches.append(iq_t)

    features: List[torch.Tensor] = []
    for iq_t in iq_batches:
        proc = preprocess_batched_iq_to_stft_feature(iq_t, sample_rate_hz=intake_sample_rate_hz)
        feature = proc["feature"].to(device=output_device, dtype=torch.float32)
        if squeeze_batch:
            if feature.shape[0] != 1:
                raise ValueError("squeeze_batch=True requires single-example IQ sections")
            feature = feature.squeeze(0).contiguous()
        features.append(feature.contiguous())
    return features


def _collate_stft_feature_list(batch: Sequence[Dict[str, Any]]) -> Optional[List[torch.Tensor]]:
    if not batch or not all("stft_feature_list" in x for x in batch):
        return None

    per_view: List[torch.Tensor] = []
    for view_idx in range(3):
        view_tensors = []
        for row in batch:
            feature_list = row["stft_feature_list"]
            if len(feature_list) != 3:
                raise ValueError("stft_feature_list must contain exactly three feature tensors")
            feature = torch.as_tensor(feature_list[view_idx], dtype=torch.float32)
            if feature.ndim == 4 and feature.shape[0] == 1:
                feature = feature.squeeze(0)
            if feature.ndim != 3:
                raise ValueError("per-sample STFT features must have shape [C, F, T]")
            view_tensors.append(feature)
        per_view.append(torch.stack(view_tensors, dim=0))
    return per_view


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
    section_len: int = 1024,
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
    - iq1/iq2/iq3 resampled to jammer_sampling_freq and cropped to section_len
    - metadata + source path for debugging/auditability

    Args:
        dataset_root: Directory containing ``sample_<number>`` sample directories.
        cache_root: Directory where ``.pt`` cache records and the manifest are written.
        jammer_sampling_freq: Target sample rate for the cached section tensors.
        section_len: Number of resampled IQ values to keep from each section.
        overwrite: Rebuild existing cache records when True.
        max_numeric_suffix: Optional inclusive upper limit for the trailing numeric
            suffix of sample directory names. For example, ``100`` processes
            ``sample_000100`` and lower while skipping ``sample_000101``.
        resample_fn: Optional dependency injection hook for custom resampling.
        cache_stft_features: When True, also stores deterministic STFT feature
            maps for iq1/iq2/iq3 so RL loops can skip repeated preprocessing.
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
        torch.save(record, out_path)
        produced.append(out_path)

    manifest = {
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root),
        "jammer_sampling_freq": float(jammer_sampling_freq),
        "section_len": int(section_len),
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
    section_len: int = 1024,
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
        "section_len": int(section_len),
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

    Note: whole_iq may have variable lengths, so keep as a list.
    Resampled section tensors are fixed-length and stackable.
    """

    out = {
        "sample_names": [x["sample_name"] for x in batch],
        "source_dirs": [x["source_dir"] for x in batch],
        "whole_iq_list": [x["whole_iq"] for x in batch],
        "whole_meta_list": [x["whole_meta"] for x in batch],
        "whole_sr_list": [float(x["whole_sample_rate_hz"]) for x in batch],
        "iq1": torch.stack([x["iq1"] for x in batch], dim=0),
        "iq2": torch.stack([x["iq2"] for x in batch], dim=0),
        "iq3": torch.stack([x["iq3"] for x in batch], dim=0),
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
    ``create_cached_dataloader`` (``iq1``, ``iq2``, ``iq3``, ``whole_iq_list``,
    metadata lists, etc.), so it can be passed directly to ``JammerVecEnv`` and
    the ``train_rl_batched`` workflow in ``RL_Jamming_test_02.ipynb``.
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


def _normalize_action(action: Any) -> Dict[str, Any]:
    """Normalize PPO action payloads into controller kwargs.

    Supported forms:
    - dict with optional keys:
        desired_output_iq_len, user_peak_power_fraction, seed,
        noise_color, fading_mode, burst_color, rf_center_hz, carrier_hz,
        num_tones, base_f, spacing, amp_raw, pulse_on_samples,
        pulse_off_samples, pulse_count, start_offset_samples
    - sequence/tensor:
        * up to 3 items mapped to desired_output_iq_len, user_peak_power_fraction, seed
        * > 3 items interpreted as continuous control action vector in this order:
          noise_color, fading_mode, burst_color, rf_center_hz, carrier_hz,
          num_tones, base_f, spacing, [amp_raw...], pulse_on_samples,
          pulse_off_samples, pulse_count, start_offset_samples
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


def jammer_controller(
    *,
    model: torch.nn.Module,
    sample: Dict[str, Any],
    action: Any,
    jammer_sampling_freq: float,
    device: str = "cpu",
    default_output_len: int = 8_000,
    default_peak_power_fraction: float = 40.0,
    default_seed: int = 11,
) -> Dict[str, Any]:
    """Concrete (sample, action) adapter around build_controlled_tone_pulse_batch_from_iq_batches."""

    action_cfg = _normalize_action(action)
    desired_output_iq_len = _as_int(action_cfg.get("desired_output_iq_len"), default_output_len)
    user_peak_power_fraction = _as_float(action_cfg.get("user_peak_power_fraction"), default_peak_power_fraction)
    seed = _as_int(action_cfg.get("seed"), default_seed)

    action_overrides = {
        k: action_cfg[k]
        for k in (
            "noise_color",
            "fading_mode",
            "burst_color",
            "rf_center_hz",
            "carrier_hz",
            "num_tones",
            "base_f",
            "spacing",
            "amp_raw",
            "pulse_on_samples",
            "pulse_off_samples",
            "pulse_count",
            "start_offset_samples",
        )
        if k in action_cfg
    }

    jam_batch = build_controlled_tone_pulse_batch_from_iq_batches(
        model=model,
        rx_iq_batches=[
            sample["iq1"].unsqueeze(0),
            sample["iq2"].unsqueeze(0),
            sample["iq3"].unsqueeze(0),
        ],
        intake_sample_rate_hz=jammer_sampling_freq,
        desired_output_iq_len=desired_output_iq_len,
        user_peak_power_fraction=user_peak_power_fraction,
        # action_overrides=[action_overrides if action_overrides else None],
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
    default_output_len: int = 8_000,
    default_peak_power_fraction: float = 40.0,
    user_peak_power_fraction: float = 40.0,
    default_seed: int = 11,
) -> List[Dict[str, Any]]:
    """Batch adapter for vectorized env rollouts.

    For maximal throughput this performs one model forward call for the whole vector.
    The first action's scalar controls are applied batch-wide.
    """

    if torch.is_tensor(actions):
        if actions.ndim == 1:
            actions_len = 1
            first_action = actions
        else:
            actions_len = int(actions.shape[0])
            first_action = actions[0]
    else:
        actions_len = len(actions)
        first_action = actions[0] if actions_len else None

    if len(samples) != actions_len:
        raise ValueError("samples and actions must have the same length")
    if not samples:
        return []

    # Only the first action currently controls batch-wide scalar generation
    # settings; per-row action_overrides are not wired into the controlled batch
    # builder below.  Callers that do not need those batch-wide scalar settings
    # can pass ``action_cfg={}`` to avoid normalizing a CUDA action tensor through
    # detach().cpu().tolist() in the rollout hot path.
    if action_cfg is None:
        action_cfg = _normalize_action(first_action)
    desired_output_iq_len = _as_int(action_cfg.get("desired_output_iq_len"), default_output_len)
    # user_peak_power_fraction = _as_float(action_cfg.get("user_peak_power_fraction"), default_peak_power_fraction)
    seed = _as_int(action_cfg.get("seed"), default_seed)

    if rx_iq_batches is None:
        iq1 = torch.stack([s["iq1"] for s in samples], dim=0).to(dtype=link7.DEFAULT_COMPLEX_DTYPE, device=device)
        iq2 = torch.stack([s["iq2"] for s in samples], dim=0).to(dtype=link7.DEFAULT_COMPLEX_DTYPE, device=device)
        iq3 = torch.stack([s["iq3"] for s in samples], dim=0).to(dtype=link7.DEFAULT_COMPLEX_DTYPE, device=device)
    else:
        if len(rx_iq_batches) != 3:
            raise ValueError("rx_iq_batches must contain exactly iq1, iq2, and iq3 batches")
        iq1, iq2, iq3 = [
            torch.as_tensor(x, dtype=link7.DEFAULT_COMPLEX_DTYPE, device=device)
            for x in rx_iq_batches
        ]

    return build_controlled_tone_pulse_batch_from_iq_batches(
        model=model.backbone,
        rx_iq_batches=[iq1, iq2, iq3],
        intake_sample_rate_hz=jammer_sampling_freq,
        desired_output_iq_len=desired_output_iq_len,
        user_peak_power_fraction=user_peak_power_fraction,
        # action_overrides=action_overrides,
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
        if "iq1" not in item or "iq2" not in item or "iq3" not in item:
            raise ValueError("each sample/batch dict must include iq1, iq2, iq3")

        iq1 = item["iq1"]
        if torch.is_tensor(iq1) and iq1.ndim == 2:
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

        self._mode = "train"
        self._cursor = {"train": 0, "test": 0}
        self._step_count = 0
        self._active: List[Dict[str, Any]] = []
        self._epoch_complete = False

    @staticmethod
    def _expand_cached_batch(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        iq1 = batch["iq1"]
        iq2 = batch["iq2"]
        iq3 = batch["iq3"]

        if not torch.is_tensor(iq1) or not torch.is_tensor(iq2) or not torch.is_tensor(iq3):
            raise TypeError("cached batch iq1/iq2/iq3 must be torch tensors")
        if iq1.ndim != 2 or iq2.ndim != 2 or iq3.ndim != 2:
            raise ValueError("cached batch iq1/iq2/iq3 must have shape [batch, samples]")

        bs = int(iq1.shape[0])
        sample_names = batch.get("sample_names")
        source_dirs = batch.get("source_dirs")
        whole_iq_list = batch.get("whole_iq_list")
        whole_meta_list = batch.get("whole_meta_list")
        whole_sr_list = batch.get("whole_sr_list")
        stft_feature_list = batch.get("stft_feature_list")

        out: List[Dict[str, Any]] = []
        for i in range(bs):
            row = {
                "sample_name": sample_names[i] if sample_names is not None else None,
                "source_dir": source_dirs[i] if source_dirs is not None else None,
                "whole_iq": whole_iq_list[i] if whole_iq_list is not None else None,
                "whole_meta": whole_meta_list[i] if whole_meta_list is not None else None,
                "whole_sample_rate_hz": whole_sr_list[i] if whole_sr_list is not None else None,
                "iq1": iq1[i],
                "iq2": iq2[i],
                "iq3": iq3[i],
            }
            if stft_feature_list is not None:
                row["stft_feature_list"] = [
                    torch.as_tensor(view[i], dtype=torch.float32)
                    for view in stft_feature_list
                ]
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
        self._active = []
        self._epoch_complete = False

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

        success = 0
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

            if rx_result.get("message") is None:
                success += 1

                # metric_div = torch.as_tensor(rx_result.get("metric_div", 0.0), dtype=torch.float32)
                # score = score + (alpha * metric_div)

            vals.append(score)#.detach().cpu())

        return torch.stack(vals), success, total #.to(dtype=torch.float32)

    @staticmethod
    def _obs_from_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        obs: Dict[str, Any] = {
            "iq1": torch.stack([s["iq1"] for s in samples], dim=0),
            "iq2": torch.stack([s["iq2"] for s in samples], dim=0),
            "iq3": torch.stack([s["iq3"] for s in samples], dim=0),
        }
        stft_feature_list = _collate_stft_feature_list(samples)
        if stft_feature_list is not None:
            obs["stft_feature_list"] = stft_feature_list
        return obs

    def reset(self) -> Dict[str, Any]:
        self._step_count = 0
        self._active = self._next_samples()
        return self._obs_from_samples(self._active)

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
        if not self._active:
            self._active = self._next_samples()

        with torch.set_grad_enabled(self.track_env_grad):
            jam_batch = jammer_controller_batch(
                model=self.model,
                samples=self._active,
                actions=actions,
                jammer_sampling_freq=self.jammer_sampling_freq,
                user_peak_power_fraction = self.user_peak_power_fraction,
                device=self.device,
            )
            rewards_t, success, total = self.reward_fn(jam_batch, self._active)

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
            for jam_item, sample in zip(jam_batch, self._active)
        ]

        if done:
            self._active = self._next_samples()
            next_obs = self._obs_from_samples(self._active)
            self._step_count = 0
        else:
            next_obs = self._obs_from_samples(self._active)

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
        iq1 = batch["iq1"].to(device=device, non_blocking=True)
        iq2 = batch["iq2"].to(device=device, non_blocking=True)
        iq3 = batch["iq3"].to(device=device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            with autocast_context(device=device, enabled=amp_enabled, dtype=amp_dtype):
                jam_batch = build_controlled_tone_pulse_batch_from_iq_batches(
                    model=model,
                    rx_iq_batches=[iq1, iq2, iq3],
                    intake_sample_rate_hz=jammer_sampling_freq,
                    desired_output_iq_len=8_000,
                    user_peak_power_fraction=40.0,
                    seed=11 + batch_idx * iq1.shape[0],
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
    *,
    iq1: torch.Tensor,
    iq2: torch.Tensor,
    iq3: torch.Tensor,
    intake_sample_rate_hz: float,
    device: str = "cpu",
    stft_feature_list: Optional[Sequence[torch.Tensor]] = None,
) -> Dict[str, List[torch.Tensor]]:
    """Build the ActorCritic observation payload from cached IQ sections.

    The controller observation is image-only and contains:
    - stft_feature_list[0]: STFT feature map for iq1, shape [B, C, F, T1]
    - stft_feature_list[1]: STFT feature map for iq2, shape [B, C, F, T2]
    - stft_feature_list[2]: STFT feature map for iq3, shape [B, C, F, T3]

    Pass ``stft_feature_list`` from a cache record or collated batch to bypass
    repeated STFT preprocessing in RL loops.  When omitted, this function keeps
    the previous behavior and computes features from the IQ batches.
    """

    if stft_feature_list is not None:
        return {"stft_feature_list": _normalize_cached_stft_feature_list(stft_feature_list, device=device)}

    return {
        "stft_feature_list": compute_stft_feature_list_for_iq_sections(
            iq1=iq1,
            iq2=iq2,
            iq3=iq3,
            intake_sample_rate_hz=intake_sample_rate_hz,
            device=device,
            squeeze_batch=False,
            output_device=device,
        )
    }


def build_stft_observation_from_samples(
    samples: Sequence[Dict[str, Any]],
    *,
    intake_sample_rate_hz: float,
    device: str = "cpu",
    use_cached_stft: bool = True,
) -> Dict[str, List[torch.Tensor]]:
    """Build an ActorCritic STFT observation from sample dictionaries.

    If every sample contains ``stft_feature_list`` and ``use_cached_stft`` is
    True, cached features are stacked and moved to ``device``.  Otherwise the
    helper falls back to stacking ``iq1``/``iq2``/``iq3`` and computing STFT
    features on demand.
    """

    cached = _collate_stft_feature_list(samples) if use_cached_stft else None
    if cached is not None:
        return {"stft_feature_list": _normalize_cached_stft_feature_list(cached, device=device)}

    return build_stft_observation_from_iq_batch(
        iq1=torch.stack([s["iq1"] for s in samples], dim=0),
        iq2=torch.stack([s["iq2"] for s in samples], dim=0),
        iq3=torch.stack([s["iq3"] for s in samples], dim=0),
        intake_sample_rate_hz=intake_sample_rate_hz,
        device=device,
    )
