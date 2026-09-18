# JSBSim skill-classifier training: performance checklist

The repository checkout does not currently contain
`03_train_jsbsim_skill_classifier.ipynb`, so the exact bottleneck cannot be
confirmed here.  The class definition for a sinusoidal positional encoding is
normally effectively instantaneous; if a notebook cell beginning with that
definition is still running, later statements in the same cell (usually the
training loop) are doing the work.

## Most likely bottleneck: full-sequence self-attention

A Transformer operating on a trajectory of length `T` constructs an attention
matrix with `T * T` entries **for every head and batch item**.  Runtime is
quadratic in `T`, and the attention-score storage alone is approximately:

```text
batch_size * num_heads * T * T * bytes_per_element
```

For example, batch size 8, 8 heads, 10,000 time steps, and float32 requires
about 25.6 GB just for one attention-score tensor, before gradients and other
activations.  Swapping or repeated allocator retries can make this look like a
hang.  Padding variable-length flights to the longest flight makes the problem
worse, especially when one trajectory is an outlier.

Reduce or bound the token count before applying a Transformer:

* Window trajectories (for example, 128--512 steps) and aggregate window
  predictions for a flight-level label.
* Downsample/resample telemetry, or use a strided 1-D convolution before the
  encoder.
* Bucket examples by length and pad only within each bucket.
* Supply a `src_key_padding_mask` so padded samples do not influence results.
  Note that masking improves correctness but standard dense attention still
  allocates a quadratic matrix, so masking alone does not fix the runtime.

Log `x.shape`, the maximum unpadded length, and the above memory estimate for
the first batch.  This quickly confirms or rejects the quadratic-attention
hypothesis.

## Verify tensor layout

Make the layout explicit.  With batches shaped `[batch, time, feature]`, create
every encoder layer with `batch_first=True`:

```python
layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=4 * d_model,
    batch_first=True,
)
encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
```

Otherwise PyTorch's sequence-first convention may interpret the batch axis as
time.  This can silently train the wrong model and can move attention work to
an unintended dimension.

## Make sure the accelerator is actually used

Move both model and every tensor used in the forward pass to the same device.
Print the result rather than assuming CUDA was selected:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("device:", device, "model:", next(model.parameters()).device)

for features, labels, padding_mask in loader:
    features = features.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    padding_mask = padding_mask.to(device, non_blocking=True)
```

For a CUDA run, use pinned DataLoader memory.  Automatic mixed precision
usually reduces activation memory and speeds up compatible GPUs:

```python
loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, ...)
scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

with torch.autocast(device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda"):
    logits = model(features, src_key_padding_mask=padding_mask)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

Start with a smaller batch if the model is near the memory limit.  Increasing
the batch size is useful only after profiling shows adequate free memory.

## Eliminate input-pipeline stalls

Fourteen hours without an epoch can also mean the model is waiting for data.
Do not parse CSV/JSON files, run JSBSim, resample complete trajectories, or fit
feature scalers in `Dataset.__getitem__`.  Precompute those results once and
load array/tensor shards.  Benchmark the loader independently:

```python
from itertools import islice
from time import perf_counter

t0 = perf_counter()
for _ in islice(loader, 100):
    pass
print("100 loader batches:", perf_counter() - t0, "seconds")
```

After confirming samples are safe to load in worker processes, tune
`num_workers`, `persistent_workers=True`, and `prefetch_factor`.  Begin with
`num_workers=0` to expose exceptions and establish a baseline; more workers can
be slower when each worker duplicates a large in-memory dataset.

## Add progress and time each stage

Epoch-only logging hides whether the delay is data loading, the forward pass,
backpropagation, or validation.  Log every few batches and synchronize CUDA
around timings because GPU operations are asynchronous:

```python
def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()

for step, batch in enumerate(loader):
    sync(); batch_start = perf_counter()
    # transfer, forward, backward, and optimizer step
    sync()
    if step % 10 == 0:
        print(f"step={step}/{len(loader)} seconds={perf_counter()-batch_start:.2f}",
              flush=True)
```

Run one batch before a full epoch, then overfit a tiny subset.  If a single
batch is slow, profile it with `torch.profiler`; if a single batch is fast but
an epoch is huge, inspect `len(loader)`, accidental dataset multiplication,
validation frequency, and loader performance.

## Recommended order of operations

1. Restart the kernel and run only the class definition; it should finish at
   once.  Split model definitions, data preparation, and training into separate
   cells so the active stage is visible.
2. Print device, first-batch shapes, `len(dataset)`, `len(loader)`, maximum
   sequence length, and estimated attention-score bytes.
3. Time 100 loader batches with `num_workers=0`.
4. Time one complete training step, including a CUDA synchronization.
5. Cap/window the sequence length if attention memory is large.
6. Enable GPU transfer and mixed precision, then tune loader workers.
7. Use `torch.profiler` only after the coarse timings identify the slow stage.

These checks distinguish a true positional-encoding issue from the much more
common causes: quadratic attention, CPU execution, excessive padding, on-demand
simulation/preprocessing, or a very large number of batches.

## Implemented reusable pipeline

The recommendations above are implemented in `jsbsim_skill_classifier.py`.
Construct `WindowedTrajectoryDataset` from already-loaded `[time, feature]`
tensors, use `collate_windows` (or `make_loader`), and train
`JSBSimSkillClassifier` with `train_one_epoch`.  The implementation bounds each
attention input with trajectory windows, downsamples before attention, uses a
batch-first encoder and padding masks, automatically enables CUDA pinned memory
and mixed precision where available, and logs synchronized batch timings.
