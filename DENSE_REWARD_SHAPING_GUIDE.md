# Dense reward shaping guide for simulated jammer learning

> Scope: this guide is for authorized, closed-loop simulation and defensive receiver-robustness testing only. Keep generated waveforms, training runs, and evaluation datasets inside controlled lab environments or offline IQ datasets.

This expands section 3 of `RL_JAMMER_LEARNING_SUGGESTIONS.md`: **Use dense, stable reward shaping instead of a sparse decode-only target**.

## Why dense shaping is needed

A final packet decode result is the most important evaluation target, but it is a poor per-step learning signal by itself:

- It is often binary or nearly binary: decode succeeds until it suddenly fails.
- It is expensive because it can require resampling, synchronization, demodulation, decoding, and scoring.
- It gives weak credit assignment because many different waveform choices can produce the same final decode score.
- It can encourage brittle shortcuts if the policy only sees one receiver configuration or one dataset split.

Dense reward shaping adds intermediate signals that move smoothly as the receiver becomes less confident. These signals should not replace final decode-failure evaluation; they should make training more sample-efficient while the full decode metric remains the authoritative validation metric.

## Recommended reward structure

Use a higher-is-better jammer reward that is explicit about sign and bounded components:

```text
jam_reward =
    w_decode       * decode_failure_reward
  + w_metric       * receiver_metric_degradation
  + w_confidence   * symbol_bit_confidence_reduction
  + w_sync         * packet_synchronization_loss
  + w_constellation* constellation_error_growth
  - w_power        * power_penalty
  - w_bandwidth    * bandwidth_penalty
  - w_invalid      * invalid_waveform_penalty
```

Recommended defaults for early experiments:

- Start with all positive components clipped to `[0, 1]`.
- Start penalty components clipped to `[0, 1]`.
- Keep `w_decode` nonzero whenever full decode is computed.
- Begin with small auxiliary weights such as `0.05` to `0.25`, then tune after plotting component distributions.
- During fast training, compute the expensive full decode periodically and use dense proxies on skipped decode steps.

## Component definitions

### 1) Decode failure reward

**Definition:** the end-to-end packet outcome transformed so that larger values mean worse receiver decode success.

Common mapping:

```text
decode_failure_reward = 1.0 - decode_score
```

Where `decode_score` should be normalized so:

- `1.0` means the packet decoded perfectly.
- `0.0` means no useful payload was recovered.

If the receiver returns no message, treat that as high decode-failure reward rather than a neutral score. Log both the raw decode score and transformed reward so training curves remain interpretable.

### 2) Receiver metric degradation

**Definition:** the baseline-relative worsening of receiver-internal quality metrics after adding the generated waveform.

Useful receiver metrics include:

- Lower preamble or packet correlation peak.
- Lower correlation peak-to-sidelobe ratio.
- Higher decoder distance or path metric.
- Higher equalizer residual.
- Higher `metric_div`-style divergence when exposed by the RX chain.
- Lower demodulator confidence or lock quality.

Recommended computation:

```text
clean_metric  = receiver_metric(clean_iq)
jammed_metric = receiver_metric(jammed_iq)
raw_delta     = direction * (clean_metric - jammed_metric)
component     = clip(raw_delta / scale, 0.0, 1.0)
```

Use `direction = +1` when larger clean values are better, and flip the sign when larger values are worse. Choose `scale` from validation-set statistics, such as the clean metric standard deviation or a percentile range.

### 3) Symbol/bit confidence reduction

**Definition:** the decrease in receiver confidence for recovered symbols or bits, even when the packet still decodes.

Useful confidence proxies include:

- Lower mean absolute log-likelihood ratio for decoded bits.
- Lower posterior probability assigned to selected symbols.
- Smaller margin between the best and second-best symbol hypothesis.
- Higher entropy of symbol or bit posteriors.
- Lower soft-decision confidence before error correction.

Recommended computation:

```text
clean_confidence  = mean_confidence(clean_iq)
jammed_confidence = mean_confidence(jammed_iq)
component         = clip((clean_confidence - jammed_confidence) / scale, 0.0, 1.0)
```

If soft decisions are unavailable, approximate confidence with demodulator margins or normalized distances to nearest decision boundaries. This component is valuable because it provides gradient-like feedback before complete packet failure.

### 4) Packet synchronization loss

**Definition:** evidence that the receiver can no longer reliably find, align, or track packet timing/frequency structure.

Useful synchronization proxies include:

- Missed preamble or sync-word detection.
- Lower sync correlation peak.
- Lower sync peak-to-sidelobe ratio.
- Failed timing lock or carrier lock.
- Large timing-offset estimate error.
- Large carrier-frequency-offset estimate error.
- Frame-boundary uncertainty or loss.

Recommended computation:

```text
if hard_sync_failure:
    component = 1.0
else:
    clean_sync_quality  = sync_quality(clean_iq)
    jammed_sync_quality = sync_quality(jammed_iq)
    component = clip((clean_sync_quality - jammed_sync_quality) / scale, 0.0, 1.0)
```

Keep the hard failure indicator and continuous sync-quality term separate in logs. A policy that only exploits one brittle sync detector may not generalize to other receiver implementations.

### 5) Constellation error growth

**Definition:** the increase in error between equalized received symbols and their nearest ideal constellation points.

Useful constellation metrics include:

- Error-vector magnitude.
- Mean squared symbol error.
- Lower cluster separation.
- Higher within-cluster variance.
- Higher nearest-neighbor decision ambiguity.

Recommended computation:

```text
clean_error  = constellation_error(clean_iq)
jammed_error = constellation_error(jammed_iq)
component    = clip((jammed_error - clean_error) / scale, 0.0, 1.0)
```

Normalize by expected symbol power or clean baseline error so this term remains comparable across SNRs, modulation settings, and packet lengths.

## Penalties to prevent degenerate learning

Dense rewards can accidentally reward unhelpful behavior if constraints are missing. Add penalties for:

- **Power overuse:** penalize average or peak power above the chosen simulation budget.
- **Excess occupied bandwidth:** penalize waveforms that exceed the allowed test bandwidth.
- **Invalid or clipped waveforms:** penalize NaNs, infinities, heavy clipping, empty IQ, or unsupported synthesis parameters.
- **Always-on trivial solutions:** penalize duty cycles that ignore timing structure when a timing-aware policy is desired.
- **Out-of-distribution controls:** penalize actions outside the configured curriculum or held-out evaluation domain.

Keep penalties transparent and logged separately. If a policy learns to maximize penalties' edge cases, tighten clipping and validation checks before adding more reward terms.

## Normalization and clipping

Recommended process:

1. Run a clean baseline pass over the train and validation splits.
2. Record distributions for every candidate metric.
3. Define each component as a baseline-relative delta.
4. Divide by a robust scale, such as interquartile range, median absolute deviation, or a fixed validation percentile span.
5. Clip each component to a bounded interval, usually `[0, 1]`.
6. Log unclipped and clipped values during early experiments.

Avoid mixing raw metrics with very different units in the same reward. A single unnormalized component can dominate training and hide whether the policy is learning useful behavior.

## Decode scheduling

A practical schedule is:

- **Warmup:** dense proxy reward every step, full decode every `k` steps or every epoch.
- **Main training:** dense proxy reward every step, full decode on a fixed validation subset each epoch.
- **Late training:** increase full-decode frequency for candidate checkpoints.
- **Final evaluation:** full decode on held-out transmitter/channel/seed groups only.

When a full decode is skipped, keep the reward scale similar by either omitting `w_decode` for that step or replacing it with a calibrated proxy estimate. Log which reward mode was used for every batch.

## Implementation checklist

- Define a `RewardComponents` record with one field per component and penalty.
- Return both the scalar reward and the component record from the reward function.
- Store clean baseline metrics in the cache when they are deterministic for a sample.
- Use the same component normalization constants for train and validation during a run.
- Add unit tests for perfect decode, partial decode, no decode, sync failure, missing soft metrics, and invalid waveform penalties.
- Track component means, standard deviations, and clipping rates in TensorBoard or the notebook logs.

## Example logging table

| Metric | Meaning | Desired training trend |
| --- | --- | --- |
| `decode_failure_rate` | Fraction of packets with no useful decode | Up on train, then up on held-out validation |
| `decode_score_raw_mean` | Original receiver decode score | Down |
| `jam_reward_mean` | Final scalar reward after weighting | Up |
| `receiver_metric_degradation_mean` | Baseline-relative receiver metric worsening | Up, without saturating immediately |
| `symbol_bit_confidence_reduction_mean` | Soft-decision confidence reduction | Up |
| `packet_sync_loss_mean` | Sync quality loss or hard sync failure | Up, but validate generalization |
| `constellation_error_growth_mean` | Equalized symbol error increase | Up |
| `power_penalty_mean` | Power budget cost | Stable and below configured limit |
| `bandwidth_penalty_mean` | Occupied bandwidth cost | Stable and below configured limit |
| `component_clip_rate` | Fraction of components clipped at bounds | Low to moderate |

## Acceptance criteria for adding dense shaping

Dense shaping is helping if:

- The policy improves held-out full-decode failure rate, not only proxy reward.
- Component distributions do not saturate in the first few epochs.
- Learned actions remain within configured power and bandwidth budgets.
- Validation gains hold across unseen packets, seeds, and channel settings.
- Removing any one auxiliary term in an ablation does not reveal that the policy depended on a brittle artifact.
