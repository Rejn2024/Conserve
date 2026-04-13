# RX chain acceleration notes for `advanced_link_skdsp_v4_robust.rx_command_iq`

This note focuses specifically on the `rx_command_iq` pipeline in `advanced_link_skdsp_v4_robust.py`.

## Where time is likely spent (from code inspection)

1. **Coarse CFO acquisition remains the largest stage**
   - `rx_command_iq` always runs coarse acquisition with `n_bins=101` over ±25 kHz.
   - `coarse_frequency_acquire` builds a full `[n_bins, n_samples]` rotator tensor and runs convolution across all bins.
   - This is compute-heavy and also memory-heavy for long IQ captures.

2. **Sample-phase search does full decode attempts per delta**
   - `try_decode_over_sample_deltas` fans out `[-3, 3]` (7 deltas), each potentially running equalization + FEC decode.
   - It currently uses `ThreadPoolExecutor` and waits for all worker results, so successful early deltas still pay for all tasks.

3. **Equalizer matrix construction repeats each decode attempt**
   - `design_symbol_equalizer_ls` and `apply_symbol_equalizer` construct stacked windows (`torch.stack(...)`) each call.
   - This can dominate for many repeated candidate decodes.

4. **CPU/GPU sync and conversion overhead in decode path**
   - `try_decode_from_symbols` converts header/payload soft bits to NumPy (`detach().cpu().numpy()`) for some operations.
   - These host transfers break GPU pipeline efficiency and add latency.

5. **Repeated reconstruction of invariant artifacts**
   - `tx_waveform(ACCESS_BITS, ...)` and RRC taps are rebuilt per invocation.

## Highest-impact acceleration ideas

1. **Two-stage CFO search (coarse -> refine)**
   - Stage 1: fewer bins (e.g., 17–25) over full ±25 kHz range.
   - Stage 2: dense local sweep around top-1/top-2 bins.
   - Keeps robustness while cutting total correlation work.

2. **Chunked/batched CFO correlation to cap memory**
   - Keep vectorized approach, but process bins in chunks (e.g., 8–16 bins/chunk).
   - This prevents large `[n_bins, n_samples]` allocations and reduces allocator pressure.

3. **Early-exit sample-phase search**
   - Replace `list(ex.map(...))` with `as_completed(...)` and cancel pending tasks once a valid payload is found.
   - Alternatively, run sequentially in likely-order deltas (`0, +1, -1, +2, -2, ...`) and stop at first success.

4. **Header-first screening before full FEC path**
   - For each sample delta, run lightweight residual-CFO + header plausibility first.
   - Only run full equalizer/FEC/CRC when header vote is valid.

5. **Cache static objects**
   - Cache access waveform and RRC taps by `(sps, beta, span)`.
   - Cache pilot symbols and any repeated bit patterns used in decode.

## Medium-impact ideas

1. **Minimize Torch↔NumPy crossings**
   - Port header/soft-bit helpers (`choose_valid_header_from_copies`, pilot removal) to torch-native variants.
   - Keep data on one device through decode.

2. **Lower precision where safe**
   - `try_decode_from_symbols` currently uses `complex128` for much of decode.
   - Evaluate `complex64/float32` for non-critical stages and compare BER/FER impact.

3. **Vectorized FIR/equalizer paths**
   - Replace window stacks with convolution/unfold-based implementations when possible.
   - This reduces Python overhead and improves kernel efficiency.

4. **Benchmark harness and perf gates**
   - Add a repeatable benchmark script for `rx_command_iq` with representative packet lengths/SNR/CFO.
   - Track p50/p95 latency and decode success to prevent regressions.

## Suggested rollout order

1. Add benchmark harness + baseline metrics.
2. Implement early-exit for sample-phase search.
3. Add waveform/tap caches.
4. Introduce two-stage + chunked CFO acquisition.
5. Reduce Torch↔NumPy conversions and evaluate precision reductions.
