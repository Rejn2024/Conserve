# Accelerating each epoch in `generating_sample_transmissions.ipynb` (training cell)

## What currently dominates epoch time

From the training cell (`run_epoch`):

1. **Per-sample disk I/O + JSON/NPY parsing inside every batch loop** (`load_whole_iq`, `load_sections`).
2. **Per-sample repeated resampling** of `iq1/iq2/iq3` every epoch (`link6.resample_iq(...)[:1024]`).
3. **CPU↔GPU handoff for many tiny tensors** (multiple `torch.as_tensor(..., device=device)` calls inside nested loops).
4. **Per-row Python loop over `jam_batch` for score computation**, creating high interpreter overhead.
5. **Heavy sync points from frequent printing** (`print` per batch, full `batch_losses` list).
6. **Evaluating full test set every epoch** with same expensive preprocessing path as training.

---

## Recommended acceleration plan (highest impact first)

### 1) Cache/precompute fixed inputs once (biggest win)

`iq1/iq2/iq3` depend only on source files + sample rate conversion to `jammer_sampling_freq`, not on model weights. Precompute once and save as `.pt`/`.npy` (or an LMDB/WebDataset shard), then training reads directly.

- Precompute artifacts per sample dir:
  - `whole_iq` (complex64)
  - `whole_meta`
  - `iq1_1024`, `iq2_1024`, `iq3_1024` at jammer sample rate
- Expected impact: eliminates repeated resample + most runtime file parsing each epoch.

### 2) Move to `Dataset` + `DataLoader` with workers and pinned memory

Replace manual `iter_batches` + nested loading loops with:

- `num_workers > 0` (start with 4–8)
- `pin_memory=True` (when CUDA)
- `persistent_workers=True`
- `prefetch_factor=2..4`

Then transfer tensors via `.to(device, non_blocking=True)`. This overlaps CPU data prep with GPU compute.

### 3) Batch/vectorize score path as much as possible

Current code loops row-by-row for:
- resampling generated jammer IQ back to RX rate,
- length matching,
- `criterion(...)` evaluation.

Refactor toward batched operations:

- Make `repeat_to_length_mod` batch-aware.
- Extend `criterion` to accept batched tensors and metadata vectors where feasible.
- If metadata dependency blocks full vectorization, at least vectorize tensor math and keep only metadata bookkeeping in Python.

### 4) Mixed precision + compiler/runtime optimizations

On CUDA:

- Use AMP (`torch.autocast(device_type='cuda', dtype=torch.float16 or bfloat16)`) for forward path if numerically stable.
- Use `GradScaler` for training.
- Try `torch.compile(model)` (PyTorch 2.x) for `TonePulseTXControlNetVarLen`.
- Set `torch.backends.cudnn.benchmark = True` (if shapes are mostly static).

These can reduce model-side latency with minimal code restructuring.

### 5) Reduce Python/logging overhead in hot loop

- Remove per-batch `print` calls or throttle to every N batches.
- Avoid saving full `tx_config`/`tx_metadata` every row during normal training; make detailed call logging optional (debug mode).
- Accumulate scalar metrics in tensors/arrays and print epoch summaries only.

### 6) Rebalance evaluation frequency

- Run full test every `k` epochs (e.g., every 2–5) or on a fixed validation subset each epoch.
- Keep full test for checkpoint milestones.

This directly shortens average epoch wall time.

---

## Additional implementation details

### A) Avoid unnecessary graph ops

`batch_loss = torch.stack(scores).mean().requires_grad_()` can be simplified to `batch_loss = torch.stack(scores).mean()`; forcing `requires_grad_()` is unnecessary and can add confusion.

### B) Use realistic learning-rate baseline

`Adam(lr=5e-2)` is unusually high for most nets and can cause instability/retries during tuning. A stable LR (e.g., `1e-3` to `3e-4`) often converges faster in wall-clock terms by avoiding wasted epochs.

### C) Cache deterministic transforms in memory for small datasets

If dataset fits RAM, keep decoded/processed arrays in-memory after first pass (simple dict cache keyed by sample dir).

### D) Profile before/after

Use `torch.profiler` + coarse timers around:
- data load,
- preprocessing/resample,
- model forward,
- scoring,
- backward/optimizer.

Track % time per segment to validate each optimization.

---

## Suggested rollout order

1. **Precompute/cached dataset** format.
2. **DataLoader workers + pinning + non-blocking copies**.
3. **Batched score path refactor**.
4. **AMP / `torch.compile`**.
5. **Logging and evaluation frequency reductions**.
6. **Profile and iterate**.

This order typically yields fast wins early while minimizing risk.
