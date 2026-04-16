from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import accelerated_training_utils as atu


def _write_sample(root: Path, idx: int, n_whole: int = 2048):
    np = pytest.importorskip("numpy")
    sdir = root / f"sample_{idx:06d}"
    sdir.mkdir(parents=True, exist_ok=True)

    whole_iq = (np.random.randn(n_whole) + 1j * np.random.randn(n_whole)).astype(np.complex64)
    sections = np.stack(
        [
            (np.random.randn(1536) + 1j * np.random.randn(1536)).astype(np.complex64),
            (np.random.randn(1536) + 1j * np.random.randn(1536)).astype(np.complex64),
            (np.random.randn(1536) + 1j * np.random.randn(1536)).astype(np.complex64),
        ],
        axis=0,
    )

    np.save(sdir / "whole_iq.npy", whole_iq)
    np.save(sdir / "sections.npy", sections)

    with open(sdir / "whole_meta.json", "w", encoding="utf-8") as f:
        json.dump({"sample_rate_hz": 1_000_000.0, "payload": f"p{idx}"}, f)

    with open(sdir / "sections_meta.json", "w", encoding="utf-8") as f:
        json.dump({"starts": [0, 1, 2]}, f)


def test_precompute_and_dataloader(tmp_path: Path):
    pytest.importorskip("numpy")
    data_root = tmp_path / "dataset"
    cache_root = tmp_path / "cache"
    data_root.mkdir()

    _write_sample(data_root, 0)
    _write_sample(data_root, 1)

    def identity_resample(x, _fs_in, _fs_out):
        return x

    produced = atu.precompute_training_cache(
        dataset_root=data_root,
        cache_root=cache_root,
        jammer_sampling_freq=2e9,
        section_len=1024,
        resample_fn=identity_resample,
    )

    assert len(produced) == 2
    assert (cache_root / "manifest.json").exists()

    loader = atu.create_cached_dataloader(
        cache_root,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(loader))
    assert batch["iq1"].shape == (2, 1024)
    assert batch["iq2"].shape == (2, 1024)
    assert batch["iq3"].shape == (2, 1024)
    assert len(batch["whole_iq_list"]) == 2


def test_run_epoch_cached_eval_path(tmp_path: Path, monkeypatch):
    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    for i in range(2):
        torch.save(
            {
                "sample_name": f"sample_{i:06d}",
                "source_dir": str(tmp_path),
                "whole_iq": torch.ones(2048, dtype=torch.complex64),
                "whole_meta": {"sample_rate_hz": 1_000_000.0},
                "whole_sample_rate_hz": 1_000_000.0,
                "jammer_sampling_freq": 2e9,
                "iq1": torch.ones(1024, dtype=torch.complex64),
                "iq2": torch.ones(1024, dtype=torch.complex64),
                "iq3": torch.ones(1024, dtype=torch.complex64),
            },
            cache_root / f"sample_{i:06d}.pt",
        )

    loader = atu.create_cached_dataloader(
        cache_root,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    def fake_build(**kwargs):
        bs = kwargs["rx_iq_batches"][0].shape[0]
        return [{"tx_iq": torch.zeros(512, dtype=torch.complex64)} for _ in range(bs)]

    monkeypatch.setattr(atu, "build_controlled_tone_pulse_batch_from_iq_batches", fake_build)

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x

    def repeat_to_length(x, n):
        xt = torch.as_tensor(x, dtype=torch.complex64)
        reps = (n + xt.numel() - 1) // xt.numel()
        return xt.repeat(reps)[:n]

    def criterion(jammed, whole, _meta):
        return torch.mean(torch.abs(jammed - whole).float())

    loss = atu.run_epoch_cached(
        dataloader=loader,
        model=DummyModel(),
        optimizer=None,
        criterion=criterion,
        jammer_sampling_freq=2e9,
        repeat_to_length_fn=repeat_to_length,
        train_mode=False,
        device="cpu",
        amp_enabled=False,
    )

    assert loss is not None
    assert loss >= 0.0


def test_maybe_compile_model_falls_back_on_triton_runtime_error(monkeypatch):
    class ToyModel(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class FailingCompiled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            raise RuntimeError("TritonMissing: Cannot find a working triton installation.")

    failing = FailingCompiled()

    def fake_compile(_model):
        return failing

    model = ToyModel()
    compiled = atu.maybe_compile_model(model, enabled=False)
    assert compiled is model

    monkeypatch.setattr(atu.torch, "compile", fake_compile)
    wrapped = atu.maybe_compile_model(model, enabled=True)
    out = wrapped(torch.tensor([1.0]))
    out2 = wrapped(torch.tensor([2.0]))

    assert torch.allclose(out, torch.tensor([2.0]))
    assert torch.allclose(out2, torch.tensor([3.0]))
    assert failing.calls == 1


def test_maybe_compile_model_falls_back_on_cuda_illegal_access_runtime_error(monkeypatch):
    class ToyModel(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class FailingCompiled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    failing = FailingCompiled()

    def fake_compile(_model):
        return failing

    model = ToyModel()
    monkeypatch.setattr(atu.torch, "compile", fake_compile)
    wrapped = atu.maybe_compile_model(model, enabled=True)
    out = wrapped(torch.tensor([1.0]))
    out2 = wrapped(torch.tensor([2.0]))

    assert torch.allclose(out, torch.tensor([2.0]))
    assert torch.allclose(out2, torch.tensor([3.0]))
    assert failing.calls == 1


def test_run_epoch_cached_train_path_skips_backward_when_loss_has_no_grad(tmp_path: Path, monkeypatch):
    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "sample_name": "sample_000000",
            "source_dir": str(tmp_path),
            "whole_iq": torch.ones(512, dtype=torch.complex64),
            "whole_meta": {"sample_rate_hz": 1_000_000.0},
            "whole_sample_rate_hz": 1_000_000.0,
            "jammer_sampling_freq": 2e9,
            "iq1": torch.ones(256, dtype=torch.complex64),
            "iq2": torch.ones(256, dtype=torch.complex64),
            "iq3": torch.ones(256, dtype=torch.complex64),
        },
        cache_root / "sample_000000.pt",
    )

    loader = atu.create_cached_dataloader(
        cache_root,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    def fake_build(**_kwargs):
        return [{"tx_iq": torch.zeros(128, dtype=torch.complex64)}]

    monkeypatch.setattr(atu, "build_controlled_tone_pulse_batch_from_iq_batches", fake_build)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return x * self.w

    def repeat_to_length(x, n):
        xt = torch.as_tensor(x, dtype=torch.complex64)
        reps = (n + xt.numel() - 1) // xt.numel()
        return xt.repeat(reps)[:n]

    # Returns a tensor without grad_fn by construction.
    def criterion(_jammed, _whole, _meta):
        return torch.tensor(0.25)

    model = DummyModel()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    initial = model.w.detach().clone()

    loss = atu.run_epoch_cached(
        dataloader=loader,
        model=model,
        optimizer=opt,
        criterion=criterion,
        jammer_sampling_freq=2e9,
        repeat_to_length_fn=repeat_to_length,
        train_mode=True,
        device="cpu",
        amp_enabled=False,
    )

    assert loss is not None
    assert torch.allclose(model.w.detach(), initial)



def test_jammer_controller_adapter_uses_build_function(monkeypatch):
    called = {}

    def fake_build(**kwargs):
        called.update(kwargs)
        return [{"tx_iq": torch.ones(16, dtype=torch.complex64), "tx_metadata": {"ok": True}}]

    monkeypatch.setattr(atu, "build_controlled_tone_pulse_batch_from_iq_batches", fake_build)

    sample = {
        "iq1": torch.ones(32, dtype=torch.complex64),
        "iq2": torch.ones(32, dtype=torch.complex64),
        "iq3": torch.ones(32, dtype=torch.complex64),
    }

    out = atu.jammer_controller(
        model=torch.nn.Identity(),
        sample=sample,
        action={"desired_output_iq_len": 256, "user_peak_power_fraction": 12.5, "seed": 99},
        jammer_sampling_freq=2e9,
        device="cpu",
    )

    assert out["tx_iq"].shape[0] == 16
    assert called["desired_output_iq_len"] == 256
    assert called["user_peak_power_fraction"] == pytest.approx(12.5)
    assert called["seed"] == 99
    assert called["rx_iq_batches"][0].shape == (1, 32)


def test_jammer_vec_env_batches_rollout(monkeypatch):
    build_calls = {"n": 0}

    def fake_build(**kwargs):
        build_calls["n"] += 1
        bs = kwargs["rx_iq_batches"][0].shape[0]
        return [
            {
                "tx_iq": torch.ones(20, dtype=torch.complex64) * (i + 1),
                "tx_metadata": {"row": i},
            }
            for i in range(bs)
        ]

    monkeypatch.setattr(atu, "build_controlled_tone_pulse_batch_from_iq_batches", fake_build)

    samples = [
        {
            "sample_name": f"s{i}",
            "iq1": torch.ones(32, dtype=torch.complex64) * (i + 1),
            "iq2": torch.ones(32, dtype=torch.complex64) * (i + 2),
            "iq3": torch.ones(32, dtype=torch.complex64) * (i + 3),
        }
        for i in range(4)
    ]

    env = atu.JammerVecEnv(
        samples=samples,
        model=torch.nn.Identity(),
        jammer_sampling_freq=2e9,
        num_envs=3,
        max_steps=1,
    )

    obs = env.reset()
    assert obs["iq1"].shape == (3, 32)

    next_obs, rewards, dones, infos = env.step(actions=[{"seed": 5}] * 3)

    assert build_calls["n"] == 1
    assert next_obs["iq1"].shape == (3, 32)
    assert rewards.shape == (3,)
    assert all(dones)
    assert len(infos) == 3
