"""Accelerated building blocks for a JSBSim trajectory skill classifier.

The module keeps dense Transformer attention away from complete, potentially
very long flights.  It windows trajectories, optionally downsamples each window
with a strided convolution, masks padding, and exposes batch-level timings so a
slow input pipeline cannot be mistaken for a hung model.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class WindowedTrajectoryDataset(Dataset):
    """Present bounded windows from variable-length ``[time, feature]`` tensors.

    A final short window is retained and padded by :func:`collate_windows`.
    Preprocessing should be completed before constructing this dataset; the
    dataset deliberately performs no file I/O or simulation in ``__getitem__``.
    """

    def __init__(
        self,
        trajectories: Sequence[Tensor],
        labels: Sequence[int],
        *,
        window_size: int = 256,
        window_stride: int | None = None,
    ) -> None:
        if len(trajectories) != len(labels):
            raise ValueError("trajectories and labels must have the same length")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        window_stride = window_size if window_stride is None else window_stride
        if window_stride < 1:
            raise ValueError("window_stride must be positive")

        self.trajectories = trajectories
        self.labels = labels
        self.window_size = window_size
        self._windows: list[tuple[int, int]] = []
        for trajectory_index, trajectory in enumerate(trajectories):
            if trajectory.ndim != 2:
                raise ValueError("each trajectory must have shape [time, feature]")
            if trajectory.shape[0] == 0:
                continue
            starts = range(0, trajectory.shape[0], window_stride)
            self._windows.extend((trajectory_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        trajectory_index, start = self._windows[index]
        window = self.trajectories[trajectory_index][start : start + self.window_size]
        return window, int(self.labels[trajectory_index]), trajectory_index


def collate_windows(
    samples: Sequence[tuple[Tensor, int, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pad a batch and return features, labels, padding mask, and flight IDs."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    feature_count = samples[0][0].shape[1]
    max_length = max(sample[0].shape[0] for sample in samples)
    features = samples[0][0].new_zeros((len(samples), max_length, feature_count))
    padding_mask = torch.ones((len(samples), max_length), dtype=torch.bool)
    labels = torch.empty(len(samples), dtype=torch.long)
    trajectory_ids = torch.empty(len(samples), dtype=torch.long)
    for row, (window, label, trajectory_id) in enumerate(samples):
        if window.shape[1] != feature_count:
            raise ValueError("all trajectories must have the same feature count")
        length = window.shape[0]
        features[row, :length] = window
        padding_mask[row, :length] = False
        labels[row] = label
        trajectory_ids[row] = trajectory_id
    return features, labels, padding_mask, trajectory_ids


class SinusoidalPositionalEncoding(nn.Module):
    """Batch-first sinusoidal encoding that grows only to the required length."""

    def __init__(self, d_model: int, max_length: int = 512) -> None:
        super().__init__()
        if d_model < 1 or max_length < 1:
            raise ValueError("d_model and max_length must be positive")
        self.d_model = d_model
        self.register_buffer("encoding", self._make_encoding(max_length), persistent=False)

    def _make_encoding(self, length: int) -> Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / self.d_model)
        )
        encoding = torch.zeros(1, length, self.d_model)
        encoding[0, :, 0::2] = torch.sin(position * frequency)
        encoding[0, :, 1::2] = torch.cos(position * frequency[: self.d_model // 2])
        return encoding

    def forward(self, inputs: Tensor) -> Tensor:
        length = inputs.shape[1]
        if length > self.encoding.shape[1]:
            self.encoding = self._make_encoding(length).to(inputs.device)
        return inputs + self.encoding[:, :length].to(dtype=inputs.dtype)


class JSBSimSkillClassifier(nn.Module):
    """Conv-downsampled, batch-first Transformer classifier."""

    def __init__(
        self,
        input_features: int,
        num_classes: int,
        *,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        conv_stride: int = 2,
        dropout: float = 0.1,
        max_window_size: int = 512,
    ) -> None:
        super().__init__()
        if conv_stride < 1:
            raise ValueError("conv_stride must be positive")
        self.conv_stride = conv_stride
        self.input_projection = nn.Conv1d(
            input_features, d_model, kernel_size=3, stride=conv_stride, padding=1
        )
        self.position = SinusoidalPositionalEncoding(
            d_model, math.ceil(max_window_size / conv_stride)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def _downsample_mask(self, padding_mask: Tensor) -> Tensor:
        # A token is padding only when every source position in its convolution
        # receptive field is padding.  This matches kernel_size=3/padding=1.
        valid = (~padding_mask).float().unsqueeze(1)
        valid = F.max_pool1d(valid, kernel_size=3, stride=self.conv_stride, padding=1)
        return valid.squeeze(1) == 0

    def forward(self, features: Tensor, padding_mask: Tensor) -> Tensor:
        encoded = self.input_projection(features.transpose(1, 2)).transpose(1, 2)
        encoded = self.position(encoded)
        reduced_mask = self._downsample_mask(padding_mask)
        encoded = self.encoder(encoded, src_key_padding_mask=reduced_mask)
        valid = (~reduced_mask).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.classifier(pooled)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_workers: int = 0
    log_every: int = 10
    use_amp: bool = True


def make_loader(
    dataset: Dataset,
    config: TrainingConfig,
    *,
    shuffle: bool = True,
    device: torch.device | None = None,
) -> DataLoader:
    """Build a loader without invalid worker-only options at worker count zero."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    worker_options = {}
    if config.num_workers:
        worker_options = {"persistent_workers": True, "prefetch_factor": 2}
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_windows,
        **worker_options,
    )


def describe_first_batch(loader: DataLoader, device: torch.device) -> None:
    """Print data/device diagnostics before starting a long epoch."""

    features, labels, padding_mask, _ = next(iter(loader))
    batch, length, feature_count = features.shape
    valid_lengths = (~padding_mask).sum(1)
    print(
        f"device={device} batches={len(loader)} batch={batch} time={length} "
        f"features={feature_count} valid_length=[{valid_lengths.min().item()}, "
        f"{valid_lengths.max().item()}] labels={labels.shape}",
        flush=True,
    )


def benchmark_loader(loader: DataLoader, batches: int = 100) -> float:
    """Return seconds spent producing at most ``batches`` input batches."""

    started = time.perf_counter()
    iterator = iter(loader)
    consumed = 0
    while consumed < batches:
        try:
            next(iterator)
        except StopIteration:
            break
        consumed += 1
    elapsed = time.perf_counter() - started
    print(f"loader_batches={consumed} loader_seconds={elapsed:.3f}", flush=True)
    return elapsed


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    log_every: int = 10,
    use_amp: bool = True,
    progress: Callable[[str], None] = print,
) -> float:
    """Train one epoch with AMP and observable, synchronized batch timings."""

    model.train()
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    total_loss = 0.0
    total_examples = 0
    for step, (features, labels, padding_mask, _) in enumerate(loader, start=1):
        started = time.perf_counter()
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            logits = model(features, padding_mask)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        count = labels.numel()
        total_loss += loss.detach().item() * count
        total_examples += count
        if step == 1 or step % log_every == 0 or step == len(loader):
            progress(
                f"batch={step}/{len(loader)} loss={loss.detach().item():.5f} "
                f"seconds={time.perf_counter() - started:.3f}"
            )
    if total_examples == 0:
        raise ValueError("training loader produced no examples")
    return total_loss / total_examples


@torch.no_grad()
def aggregate_window_predictions(
    logits: Tensor, trajectory_ids: Tensor
) -> dict[int, Tensor]:
    """Average window probabilities into one prediction per trajectory."""

    probabilities = logits.softmax(dim=-1)
    return {
        int(trajectory_id): probabilities[trajectory_ids == trajectory_id].mean(0)
        for trajectory_id in trajectory_ids.unique()
    }
