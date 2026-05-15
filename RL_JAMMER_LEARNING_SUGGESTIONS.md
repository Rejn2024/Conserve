# Suggestions to improve jammer learning

> Scope: these recommendations are intended for authorized, closed-loop lab simulation and defensive robustness testing only. Keep all training, evaluation, and generated waveforms inside controlled datasets or shielded test environments.

## Highest-impact learning issues to fix first

### 1) Make the RL action actually control each jammer waveform

`jammer_controller_batch(...)` accepts one action per sample, but currently only normalizes the first action for batch-wide scalar settings and leaves `action_overrides` disconnected from `build_controlled_tone_pulse_batch_from_iq_batches(...)`. As a result, most per-environment sampled actions may not affect the generated jammer IQ, which makes policy-gradient credit assignment extremely weak or impossible.

Recommendations:

- Decode every row's action vector into per-sample controls for tone count, tone frequencies, amplitudes, pulse timing, start offset, channel/noise options, and power budget.
- Pass those per-row controls into the batch builder rather than using the first action for batch-wide configuration.
- Add a regression test that changes only one row's action and verifies only that row's `tx_iq`/metadata changes.
- If the backbone network is intended to be the deterministic policy, either remove unused sampled actions from the PPO path or make the actor's sampled action override the backbone outputs before waveform synthesis.

### 2) Align the reward sign with the optimizer objective

`JammerVecEnv._default_reward(...)` initializes each score to `1.0`, replaces it with `score_decode(...)` only when a decoded message is present, and counts `score < 1.0` as a jamming success. If the RL loop maximizes rewards, this can reward good decodes instead of failed decodes. If the supervised/direct loop minimizes a score, the opposite convention may be correct, but it should be explicit and separated by training mode.

Recommendations:

- Define a single scalar objective such as `jam_reward = 1.0 - decode_score`, where higher means the receiver decoded less successfully.
- Give full credit for decode failure / `message is None` instead of leaving the score at a neutral or decode-good value.
- Log both raw `decode_score` and transformed `jam_reward` so learning curves are interpretable.
- Add unit tests for the three reward cases: perfect decode, partial/corrupt decode, and no decode.

### 3) Use dense, stable reward shaping instead of a sparse decode-only target

A final decode result is an expensive and often sparse signal. The agent will learn faster if it receives dense feedback that correlates with degraded receiver confidence while still validating against end-to-end decode failure.

Recommendations:

- Combine decode outcome with bounded auxiliary terms such as receiver metric degradation, symbol/bit confidence reduction, packet synchronization loss, constellation error growth, and bandwidth/power penalties.
- Normalize all reward components to comparable ranges before summing.
- Start with dense proxy terms each step, then periodically include the full decode reward as the authoritative objective.
- Penalize degenerate solutions such as always-max-power, overly broad noise, or waveforms outside the authorized simulation constraints.

Definitions for the suggested dense reward terms:

- **Receiver metric degradation:** a bounded measurement of how much worse the receiver's internal quality indicators become after adding the generated waveform, compared with the clean baseline for the same packet. Examples include lower correlation peaks, lower demodulator confidence, higher decoder distance, or larger `metric_div`-style values when exposed by the RX chain. Compute it as a baseline-relative delta and clip/normalize it before adding it to the reward.
- **Symbol/bit confidence reduction:** a reduction in the receiver's confidence for recovered symbols or bits, even when the packet still decodes. If the receiver exposes soft decisions, use lower mean absolute log-likelihood ratio, lower posterior probability for the selected symbol, or a smaller margin between the best and second-best symbol hypotheses. This gives learning signal before full decode failure occurs.
- **Packet synchronization loss:** evidence that the receiver can no longer reliably find or track packet timing/frequency structure. Useful proxies include missed preamble detection, reduced sync correlation peak-to-sidelobe ratio, failed carrier/timing lock, large timing-offset estimate error, or frame-boundary loss. Treat hard sync failure as a high reward component, but use the continuous sync metric as the dense signal.
- **Constellation error growth:** an increase in the distance between received/equalized symbols and their nearest ideal constellation points. Common forms are higher error-vector magnitude, higher mean squared symbol error, or lower cluster separation. Normalize by the clean baseline or by expected symbol power so this term is comparable across SNRs and modulation settings.

### 4) Add a curriculum from easy to hard transmissions

Learning can stall if the initial environment is too hard or too diverse. The generator already supports configurable transmission construction, and the cache records preserve sample metadata, so introduce staged difficulty.

Recommendations:

- Stage 1: fixed modulation/channel settings, short packets, known timing, and low variability.
- Stage 2: randomize start offsets, packet lengths, SNR, and carrier offsets.
- Stage 3: add channel fading, timing errors, multiple protocol/rate settings, and held-out transmitter/channel combinations.
- Advance stages only after the policy reaches a decode-failure target on a validation subset.

### 5) Preserve exploration and avoid premature collapse

The actor-critic model has a learned log-standard-deviation head and entropy can be computed during action evaluation. Use those signals deliberately so the policy explores waveform families before converging.

Recommendations:

- Track per-dimension action standard deviation, entropy, KL divergence, and clipping fraction.
- Anneal entropy bonus slowly rather than disabling it early.
- Increase the standard-deviation floor for action dimensions that saturate too soon.
- Randomize seeds and waveform initial phases during training while keeping deterministic validation seeds.

### 6) Improve observation features for timing and spectral overlap

The STFT feature path already exposes magnitude, phase, power, deltas, centroid, spread, flatness, and frame power. Add side information that makes jamming decisions easier to learn without requiring the CNN to infer every scalar from images.

Recommendations:

- Add scalar features for packet duration, section offsets, observed bandwidth, estimated center frequency, sample-rate ratio, and measured RX power.
- Include a compact estimate of where synchronization/preamble-like energy occurs in time.
- Normalize scalar features with running statistics and save those stats with checkpoints.
- Ablate feature groups to confirm each group improves held-out decode-failure rate.

### 7) Use off-policy RL or hybrid imitation + RL for expensive rewards

The reward function performs resampling and receiver decoding for each generated waveform, so sample efficiency matters. PPO can work, but off-policy methods may learn more from each expensive decode.

Recommendations:

- Keep a replay buffer of `(observation, action, reward, next_observation, done, metadata)` transitions.
- Try SAC/TD3-style continuous-control training for waveform parameters while preserving deterministic full-decode evaluation.
- Seed the buffer with heuristic jammer strategies (tone near estimated carrier, pulse during high-energy frames, narrowband sweep) and let RL improve beyond them.
- Use offline pretraining to predict a good initial action from receiver features, then fine-tune with RL.

### 8) Make evaluation harder than training and report generalization

A policy can overfit to cached packets, fixed seeds, or one receiver configuration. Treat decode-failure rate on unseen conditions as the main metric.

Recommendations:

- Hold out entire transmitter/channel/seed groups, not just random rows.
- Report decode-failure rate, mean transformed reward, power-normalized effectiveness, and confidence intervals.
- Evaluate deterministic and stochastic policies separately.
- Add adversarial validation slices: high SNR, frequency offset extremes, long packets, and channels not seen during training.

## Suggested implementation order

1. Wire per-row actions into waveform synthesis and add tests that prove actions change per-row outputs.
2. Replace ambiguous decode scores with a documented higher-is-better `jam_reward`.
3. Add dense auxiliary reward terms with normalization and power/bandwidth penalties.
4. Add curriculum scheduling over dataset/generator difficulty.
5. Track exploration diagnostics and tune entropy/std schedules.
6. Add observation side features and ablation metrics.
7. Experiment with replay-buffer methods or hybrid imitation + RL.
8. Harden evaluation with held-out transmitter/channel splits.

## Minimal learning diagnostics to log every epoch

- Decode-failure rate on train and held-out validation splits.
- Raw decode score and transformed jammer reward.
- Reward component means/stddevs.
- Entropy, action standard deviation, KL divergence, and PPO clip fraction if PPO is used.
- Power use, occupied bandwidth, tone count, pulse duty cycle, and start-offset distributions.
- Per-scenario metrics grouped by SNR, carrier offset, timing offset, channel/fading mode, and packet length.
