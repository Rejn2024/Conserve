# Suggestions to accelerate `train_rl_loop`

This note captures practical speedups for `train_rl_loop` in `RL_deugging.py`.

## 1) Iterate by resolved rollout length (already applied)

- Use `steps_per_epoch` inside the inner rollout loop instead of `len(env.samples)`.
- Why this helps:
  - For `DataLoader`/cached inputs, `_resolve_steps_per_epoch` already tracks source batch count.
  - Avoids over-iterating when the backing source is not a flat list.

## 2) Keep decode scoring optional during training

- `env.step(...)` currently returns `success` and `total`, which can include expensive decode paths.
- Add a config flag like `cfg.compute_decode_metrics_every_n_epochs` and only request decode metrics intermittently (or in eval only).
- Expected impact: lower per-step overhead when decode scoring invokes heavy DSP / decode routines.

## 3) Use mixed precision for forward + loss

- Wrap forward/loss in:
  - `torch.autocast(device_type="cuda", dtype=torch.float16 or bfloat16)` and
  - `torch.cuda.amp.GradScaler` (if using float16).
- Keep rewards/advantage normalization in fp32 when needed for stability.
- Expected impact: significant throughput gain on modern GPUs.

## 4) Compile hot model path once

- If PyTorch 2.x is available, call `torch.compile(policy)` (or existing helper) before training.
- Benchmark with a warmup epoch; compile can improve steady-state throughput.

## 5) Minimize Python overhead in env-step input/output plumbing

- Keep action tensors on-device and avoid shape conversions each step.
- If possible, have `env.step` accept batched tensor actions directly without per-item Python loops.
- In reward handling, if rewards are already tensors on target device, avoid redundant `torch.as_tensor(...)`.

## 6) Reduce logging/checkpoint frequency

- TensorBoard logging and frequent checkpoint writes can stall training, especially on network filesystems.
- Log/checkpoint every `k` epochs (e.g., every 5) and still track best model.

## 7) Validate DataLoader throughput knobs

- For cached dataset loading:
  - tune `num_workers`,
  - keep `pin_memory=True` on CUDA,
  - tune `prefetch_factor`,
  - use `persistent_workers=True` when workers > 0.
- Profile CPU loader wait time vs GPU utilization to set these values empirically.

## 8) Profile to confirm bottleneck before deep rewrites

- Use `torch.profiler` around a few iterations to split time between:
  - model forward/backward,
  - env stepping/decode,
  - data movement,
  - logging/checkpointing.
- Prioritize the top 1–2 bottlenecks first.
