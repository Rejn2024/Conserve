# DSP step acceleration suggestions

This is a DSP-focused checklist for the current training/inference path.

## Highest-impact bottlenecks in current code path

1. **Per-sample decode loop in reward function**  
   `JammerVecEnv._default_reward` runs `resample_iq -> repeat_to_length_mod -> rx_command_iq -> score_decode` per sample in Python.  
   This serial loop is likely the dominant wall-clock cost for RL training.

2. **Per-row STFT preprocessing loop**  
   `preprocess_batched_iq_to_stft_feature` iterates `for i in range(x_batch.shape[0])` and calls `preprocess_iq_to_stft_feature` one sample at a time.

3. **Repeated window/frequency-grid allocations in STFT path**  
   `preprocess_iq_to_stft_feature` rebuilds `hann_window` and `fftfreq/fftshift` tensors every call.

---

## Priority recommendations

### 1) Batch decode calls in reward computation (largest gain)

- Replace per-item `rx_command_iq(...)` calls with the broadcast helper:
  - `advanced_link_skdsp_v7_robust.rx_command_iq_broadcast(iq_batch, meta_list)`.
- Build one batched complex tensor for jammed IQs, then decode the batch.
- Keep a fallback to single-item decode only for exceptions.

**Why:** moves Python loop overhead from “decode call per row” to “one batch call per step”, reducing dispatch and setup overhead.

### 2) Gate expensive decode objective by schedule

- During training, compute decode reward only every `k` steps/epochs (e.g., `k=4` or `k=8`).
- For skipped steps, use a cheap proxy reward (e.g., energy/divergence metric) and reserve full decode for periodic calibration.

**Why:** decode is expensive and often not needed every step to preserve learning signal.

### 3) Vectorize STFT preprocessing across batch

- Refactor `preprocess_batched_iq_to_stft_feature` to call `torch.stft` on batched input once (or in large chunks), instead of per row.
- Then derive channels (`mag`, `phase`, deltas, centroid/spread/flatness) from batched tensors.

**Why:** batched FFT/STFT kernels are much more efficient than Python loops around single-item calls.

### 4) Cache DSP constants on `(device, nfft, nperseg)`

- Cache and reuse:
  - Hann windows
  - `fftfreq/fftshift` grids
  - any constant planes derived from sample rate

**Why:** avoids repeated tiny tensor allocations that become expensive at high call counts.

### 5) Avoid repeated resample when sample rate is unchanged

- In reward path, short-circuit:
  - if `jammer_sampling_freq == whole_meta["sample_rate_hz"]`, skip `resample_iq`.
- If repeated sample-rate pairs are common, precompute rational resample settings once.

**Why:** resampling is one of the most expensive DSP ops; skip whenever possible.

### 6) Keep IQ data resident on a single device per stage

- Minimize CPU↔GPU transitions in reward path:
  - `whole_iq.to(self.device)` is repeated every step.
- If decode remains CPU-only, keep reward-side tensors CPU and avoid bouncing through GPU first.
- If decode can run on GPU, keep end-to-end on GPU until scalar score extraction.

**Why:** transfer overhead can erase gains from faster compute kernels.

### 7) Separate “train-speed mode” vs “eval-fidelity mode”

- Add config presets:
  - **train-speed:** less frequent decode, smaller FFT params, reduced feature set.
  - **eval-fidelity:** full decode and full feature extraction.

**Why:** lets you maximize throughput during optimization while preserving high-quality evaluation.

---

## Low-risk implementation sequence

1. Add decode scheduling + optional proxy reward (fastest to ship).
2. Switch reward decode to batched decode helper.
3. Cache STFT window/frequency tensors.
4. Vectorize STFT batch preprocessing.
5. Re-profile and tune batch size / num_workers accordingly.

---

## Minimal profiling plan (before/after each step)

- Time per section with `time.perf_counter()`:
  - jammer generation
  - resample/repeat
  - decode + scoring
  - STFT feature build
- Record:
  - seconds per training step
  - samples/sec
  - decode success percentage (to ensure quality is not regressing)

Track these metrics for a fixed number of steps on the same dataset split.
