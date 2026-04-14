#!/usr/bin/env python3
"""Utilities to accelerate tone-pulse training epochs.

Implements four performance-focused capabilities:
1) one-time precompute/cache of deterministic inputs,
2) DataLoader-based input pipeline with worker prefetching,
3) batch-oriented score path helpers,
4) mixed-precision + torch.compile setup helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from tx_controller_tone_pulse_stft_varlen_2 import build_controlled_tone_pulse_batch_from_iq_batches


def _to_complex64_tensor(x: Any) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.complex64)


def precompute_training_cache(
    dataset_root: Path,
    cache_root: Path,
    jammer_sampling_freq: float,
    *,
    section_len: int = 1024,
    overwrite: bool = False,
    resample_fn: Optional[Callable[[Any, float, float], Any]] = None,
) -> List[Path]:
    """Precompute deterministic sample tensors once and save to cache files.

    Each cached sample stores:
    - whole_iq (complex64)
    - whole_sample_rate_hz (float)
    - iq1/iq2/iq3 resampled to jammer_sampling_freq and cropped to section_len
    - metadata + source path for debugging/auditability
    """

    if resample_fn is None:
        import advanced_link_skdsp_v6_robust as link6

        resample = link6.resample_iq
    else:
        resample = resample_fn
    dataset_root = Path(dataset_root)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    import load_tx_iq_data as loadmod

    sample_dirs = loadmod.list_sample_dirs(dataset_root)
    produced: List[Path] = []

    for sdir in sample_dirs:
        out_path = cache_root / f"{sdir.name}.pt"
        if out_path.exists() and not overwrite:
            produced.append(out_path)
            continue

        whole = loadmod.load_whole_iq(sdir)
        sections = loadmod.load_sections(sdir)
        sample_rate_hz = float(whole["meta"]["sample_rate_hz"])

        iq1 = resample(sections["sections"][0], sample_rate_hz, jammer_sampling_freq)[:section_len]
        iq2 = resample(sections["sections"][1], sample_rate_hz, jammer_sampling_freq)[:section_len]
        iq3 = resample(sections["sections"][2], sample_rate_hz, jammer_sampling_freq)[:section_len]

        record = {
            "sample_name": sdir.name,
            "source_dir": str(sdir),
            "whole_iq": _to_complex64_tensor(whole["iq"]),
            "whole_meta": whole["meta"],
            "whole_sample_rate_hz": sample_rate_hz,
            "jammer_sampling_freq": float(jammer_sampling_freq),
            "iq1": _to_complex64_tensor(iq1),
            "iq2": _to_complex64_tensor(iq2),
            "iq3": _to_complex64_tensor(iq3),
        }
        torch.save(record, out_path)
        produced.append(out_path)

    manifest = {
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root),
        "jammer_sampling_freq": float(jammer_sampling_freq),
        "section_len": int(section_len),
        "num_samples": len(produced),
        "files": [str(p.name) for p in produced],
    }
    with open(cache_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

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
        return torch.load(self.records[idx], map_location="cpu", weights_only=False)



def collate_cached_iq(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate cached records into a batch dict.

    Note: whole_iq may have variable lengths, so keep as a list.
    Resampled section tensors are fixed-length and stackable.
    """

    return {
        "sample_names": [x["sample_name"] for x in batch],
        "source_dirs": [x["source_dir"] for x in batch],
        "whole_iq_list": [x["whole_iq"] for x in batch],
        "whole_meta_list": [x["whole_meta"] for x in batch],
        "whole_sr_list": [float(x["whole_sample_rate_hz"]) for x in batch],
        "iq1": torch.stack([x["iq1"] for x in batch], dim=0),
        "iq2": torch.stack([x["iq2"] for x in batch], dim=0),
        "iq3": torch.stack([x["iq3"] for x in batch], dim=0),
    }


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
            message = str(exc).lower()
            triton_compile_failure = "triton" in message or "torch._inductor" in message
            if not triton_compile_failure:
                raise
            self._use_eager = True
            return self._eager_model(*args, **kwargs)


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
        import advanced_link_skdsp_v6_robust as link6

        resample = link6.resample_iq
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
        jam_iq_t = torch.as_tensor(jam_iq_rx_resam[: whole_iq.shape[0]], dtype=torch.complex64, device=device)
        whole_iq_t = whole_iq.to(device=device, non_blocking=True)
        jammed = whole_iq_t + jam_iq_t
        scores.append(criterion(jammed, whole_iq_t, whole_meta))

    if not scores:
        return torch.empty((0,), dtype=torch.float32, device=device)
    return torch.stack(scores)


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
                    if _optimizer_has_any_grad(optimizer):
                        grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    if _optimizer_has_any_grad(optimizer):
                        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    if not losses:
        return None
    return float(sum(losses) / len(losses))
