from __future__ import annotations

import json
import sys
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


def test_precompute_training_cache_limits_numeric_suffix(tmp_path: Path):
    pytest.importorskip("numpy")
    data_root = tmp_path / "dataset"
    cache_root = tmp_path / "cache"
    data_root.mkdir()

    for idx in range(4):
        _write_sample(data_root, idx)

    def identity_resample(x, _fs_in, _fs_out):
        return x

    produced = atu.precompute_training_cache(
        dataset_root=data_root,
        cache_root=cache_root,
        jammer_sampling_freq=2e9,
        section_len=1024,
        max_numeric_suffix=1,
        resample_fn=identity_resample,
    )

    assert [path.name for path in produced] == ["sample_000000.pt", "sample_000001.pt"]
    assert not (cache_root / "sample_000002.pt").exists()

    with open(cache_root / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["max_numeric_suffix"] == 1
    assert manifest["num_samples"] == 2


def test_precompute_training_cache_rejects_negative_numeric_suffix(tmp_path: Path):
    data_root = tmp_path / "dataset"
    cache_root = tmp_path / "cache"
    data_root.mkdir()

    with pytest.raises(ValueError, match="max_numeric_suffix must be non-negative"):
        atu.precompute_training_cache(
            dataset_root=data_root,
            cache_root=cache_root,
            jammer_sampling_freq=2e9,
            max_numeric_suffix=-1,
            resample_fn=lambda x, _fs_in, _fs_out: x,
        )


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


def test_jammer_vec_env_accepts_cached_dataloader_batches(monkeypatch):
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

    rows = [
        {
            "sample_name": f"s{i}",
            "source_dir": "tmp",
            "whole_iq": torch.ones(64, dtype=torch.complex64),
            "whole_meta": {"sample_rate_hz": 1_000_000.0},
            "whole_sample_rate_hz": 1_000_000.0,
            "iq1": torch.ones(32, dtype=torch.complex64) * (i + 1),
            "iq2": torch.ones(32, dtype=torch.complex64) * (i + 2),
            "iq3": torch.ones(32, dtype=torch.complex64) * (i + 3),
        }
        for i in range(4)
    ]
    loader = torch.utils.data.DataLoader(rows, batch_size=2, shuffle=False, collate_fn=atu.collate_cached_iq)

    env = atu.JammerVecEnv(
        samples=loader,
        model=torch.nn.Identity(),
        jammer_sampling_freq=2e9,
        num_envs=2,
        max_steps=1,
    )

    obs = env.reset()
    assert obs["iq1"].shape == (2, 32)
    assert env.samples[0]["sample_name"] == "s0"

    next_obs, rewards, dones, infos = env.step(actions=[{"seed": 5}] * 2)

    assert build_calls["n"] == 1
    assert next_obs["iq1"].shape == (2, 32)
    assert rewards.shape == (2,)
    assert all(dones)
    assert len(infos) == 2


def test_jammer_vec_env_supports_test_loader_mode_switch(monkeypatch):
    def fake_build(**kwargs):
        bs = kwargs["rx_iq_batches"][0].shape[0]
        return [{"tx_iq": torch.ones(20, dtype=torch.complex64), "tx_metadata": {"row": i}} for i in range(bs)]

    monkeypatch.setattr(atu, "build_controlled_tone_pulse_batch_from_iq_batches", fake_build)

    train_rows = [
        {
            "sample_name": f"train_{i}",
            "source_dir": "tmp",
            "whole_iq": torch.ones(64, dtype=torch.complex64),
            "whole_meta": {"sample_rate_hz": 1_000_000.0},
            "whole_sample_rate_hz": 1_000_000.0,
            "iq1": torch.ones(32, dtype=torch.complex64) * (i + 1),
            "iq2": torch.ones(32, dtype=torch.complex64) * (i + 2),
            "iq3": torch.ones(32, dtype=torch.complex64) * (i + 3),
        }
        for i in range(4)
    ]
    test_rows = [
        {
            "sample_name": f"test_{i}",
            "source_dir": "tmp",
            "whole_iq": torch.ones(64, dtype=torch.complex64),
            "whole_meta": {"sample_rate_hz": 1_000_000.0},
            "whole_sample_rate_hz": 1_000_000.0,
            "iq1": torch.ones(32, dtype=torch.complex64) * (i + 10),
            "iq2": torch.ones(32, dtype=torch.complex64) * (i + 11),
            "iq3": torch.ones(32, dtype=torch.complex64) * (i + 12),
        }
        for i in range(2)
    ]
    train_loader = torch.utils.data.DataLoader(train_rows, batch_size=2, shuffle=False, collate_fn=atu.collate_cached_iq)
    test_loader = torch.utils.data.DataLoader(test_rows, batch_size=2, shuffle=False, collate_fn=atu.collate_cached_iq)

    env = atu.JammerVecEnv(
        samples=train_loader,
        test_samples=test_loader,
        model=torch.nn.Identity(),
        jammer_sampling_freq=2e9,
        num_envs=2,
        max_steps=1,
    )

    train_obs = env.reset()
    assert torch.allclose(train_obs["iq1"][0], torch.ones(32, dtype=torch.complex64) * 1)

    env.set_mode("test")
    test_obs = env.reset()
    assert env.mode == "test"
    assert torch.allclose(test_obs["iq1"][0], torch.ones(32, dtype=torch.complex64) * 10)

    _, _, _, infos = env.step(actions=[{"seed": 5}] * 2)
    assert all(info["mode"] == "test" for info in infos)
    assert all(info["epoch_complete"] for info in infos)


def test_default_reward_matches_jammer_loss_decode_term(monkeypatch):
    class FakeLinkModule:
        @staticmethod
        def rx_command_iq(_jammed, _meta):
            return {"message": "ok", "metric_div": 0.3}

    class FakeScoreModule:
        @staticmethod
        def score_decode(_rx_result, _meta):
            return torch.tensor(0.4)

    monkeypatch.setitem(sys.modules, "advanced_link_skdsp_v6_robust", FakeLinkModule)
    monkeypatch.setitem(sys.modules, "score_iq_decode", FakeScoreModule)

    jam_batch = [{"tx_iq": torch.ones(4, dtype=torch.complex64)}]
    samples = [{"whole_iq": torch.zeros(4, dtype=torch.complex64), "whole_meta": {"sample_rate_hz": 1.0}}]
    reward = atu.JammerVecEnv._default_reward(jam_batch, samples)

    # JammerLoss decode term in notebook: score + alpha * metric_div, alpha=10.
    assert reward.shape == (1,)
    assert torch.allclose(reward, torch.tensor([3.4], dtype=torch.float32))


class _FakeS3Client:
    def __init__(self):
        self.objects = {}

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            exc = RuntimeError("not found")
            exc.response = {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}
            raise exc
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def upload_fileobj(self, fileobj, Bucket, Key):
        self.objects[(Bucket, Key)] = fileobj.read()

    def put_object(self, *, Bucket, Key, Body, **_kwargs):
        self.objects[(Bucket, Key)] = Body

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        contents = [
            {"Key": key}
            for (bucket, key), _body in self.objects.items()
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"Contents": sorted(contents, key=lambda x: x["Key"]), "IsTruncated": False}

    def download_fileobj(self, Bucket, Key, fileobj):
        fileobj.write(self.objects[(Bucket, Key)])


def test_precompute_training_cache_s3_and_dataloader_s3(tmp_path: Path):
    pytest.importorskip("numpy")
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    _write_sample(data_root, 0)
    _write_sample(data_root, 1)
    s3 = _FakeS3Client()

    produced = atu.precompute_training_cache_s3(
        dataset_root=data_root,
        cache_s3_uri="s3://example-bucket/cache/train",
        jammer_sampling_freq=2e9,
        section_len=1024,
        resample_fn=lambda x, _fs_in, _fs_out: x,
        s3_client=s3,
    )

    assert produced == [
        "s3://example-bucket/cache/train/sample_000000.pt",
        "s3://example-bucket/cache/train/sample_000001.pt",
    ]
    manifest = json.loads(s3.objects[("example-bucket", "cache/train/manifest.json")].decode("utf-8"))
    assert manifest["cache_s3_uri"] == "s3://example-bucket/cache/train"
    assert manifest["num_samples"] == 2

    loader = atu.create_cached_dataloader_s3(
        "s3://example-bucket/cache/train",
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        s3_client=s3,
    )
    batch = next(iter(loader))
    assert batch["sample_names"] == ["sample_000000", "sample_000001"]
    assert batch["iq1"].shape == (2, 1024)
    assert batch["iq2"].shape == (2, 1024)
    assert batch["iq3"].shape == (2, 1024)
    assert len(batch["whole_iq_list"]) == 2


def test_precompute_training_cache_s3_reuses_existing_objects(tmp_path: Path):
    pytest.importorskip("numpy")
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    _write_sample(data_root, 0)
    s3 = _FakeS3Client()

    first = atu.precompute_training_cache_s3(
        dataset_root=data_root,
        cache_s3_uri="s3://example-bucket/cache",
        jammer_sampling_freq=2e9,
        section_len=1024,
        resample_fn=lambda x, _fs_in, _fs_out: x,
        s3_client=s3,
    )
    original_body = s3.objects[("example-bucket", "cache/sample_000000.pt")]

    second = atu.precompute_training_cache_s3(
        dataset_root=data_root,
        cache_s3_uri="s3://example-bucket/cache",
        jammer_sampling_freq=2e9,
        section_len=256,
        overwrite=False,
        resample_fn=lambda x, _fs_in, _fs_out: x,
        s3_client=s3,
    )

    assert second == first
    assert s3.objects[("example-bucket", "cache/sample_000000.pt")] == original_body


def test_create_cached_dataloader_s3_rejects_empty_prefix():
    with pytest.raises(ValueError, match="No cache records found"):
        atu.create_cached_dataloader_s3(
            "s3://example-bucket/empty",
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            s3_client=_FakeS3Client(),
        )


def test_collate_cached_iq_stacks_cached_stft_features():
    rows = []
    for i in range(2):
        rows.append(
            {
                "sample_name": f"s{i}",
                "source_dir": "tmp",
                "whole_iq": torch.ones(64, dtype=torch.complex64),
                "whole_meta": {"sample_rate_hz": 1_000_000.0},
                "whole_sample_rate_hz": 1_000_000.0,
                "iq1": torch.ones(32, dtype=torch.complex64),
                "iq2": torch.ones(32, dtype=torch.complex64),
                "iq3": torch.ones(32, dtype=torch.complex64),
                "stft_feature_list": [
                    torch.full((2, 3, 4), float(i + view))
                    for view in range(3)
                ],
            }
        )

    batch = atu.collate_cached_iq(rows)

    assert "stft_feature_list" in batch
    assert len(batch["stft_feature_list"]) == 3
    assert batch["stft_feature_list"][0].shape == (2, 2, 3, 4)
    assert torch.allclose(batch["stft_feature_list"][2][1], torch.full((2, 3, 4), 3.0))


def test_build_stft_observation_from_iq_batch_uses_cached_features(monkeypatch):
    def fail_preprocess(*_args, **_kwargs):
        raise AssertionError("preprocess should not be called when cached STFT is supplied")

    monkeypatch.setattr(atu, "preprocess_batched_iq_to_stft_feature", fail_preprocess)
    cached = [torch.ones(2, 3, 4, 5) * view for view in range(3)]

    obs = atu.build_stft_observation_from_iq_batch(
        iq1=torch.ones(2, 16, dtype=torch.complex64),
        iq2=torch.ones(2, 16, dtype=torch.complex64),
        iq3=torch.ones(2, 16, dtype=torch.complex64),
        intake_sample_rate_hz=2e9,
        stft_feature_list=cached,
        device="cpu",
    )

    assert len(obs["stft_feature_list"]) == 3
    assert torch.allclose(obs["stft_feature_list"][1], torch.ones(2, 3, 4, 5))


def test_build_stft_observation_from_samples_prefers_cached_features(monkeypatch):
    def fail_preprocess(*_args, **_kwargs):
        raise AssertionError("preprocess should not be called when cached STFT is supplied")

    monkeypatch.setattr(atu, "preprocess_batched_iq_to_stft_feature", fail_preprocess)
    samples = [
        {
            "iq1": torch.ones(8, dtype=torch.complex64),
            "iq2": torch.ones(8, dtype=torch.complex64),
            "iq3": torch.ones(8, dtype=torch.complex64),
            "stft_feature_list": [torch.ones(2, 3, 4) * (i + view) for view in range(3)],
        }
        for i in range(2)
    ]

    obs = atu.build_stft_observation_from_samples(samples, intake_sample_rate_hz=2e9, device="cpu")

    assert obs["stft_feature_list"][0].shape == (2, 2, 3, 4)
    assert torch.allclose(obs["stft_feature_list"][1][1], torch.full((2, 3, 4), 2.0))


def test_precompute_training_cache_can_store_stft_features(tmp_path: Path, monkeypatch):
    pytest.importorskip("numpy")
    data_root = tmp_path / "dataset"
    cache_root = tmp_path / "cache"
    data_root.mkdir()
    _write_sample(data_root, 0)

    def fake_preprocess(iq, sample_rate_hz):
        batch = iq.shape[0]
        return {"feature": torch.ones(batch, 2, 3, 4) * float(sample_rate_hz)}

    monkeypatch.setattr(atu, "preprocess_batched_iq_to_stft_feature", fake_preprocess)

    produced = atu.precompute_training_cache(
        dataset_root=data_root,
        cache_root=cache_root,
        jammer_sampling_freq=2e9,
        section_len=16,
        resample_fn=lambda x, _fs_in, _fs_out: x,
        cache_stft_features=True,
    )

    record = torch.load(produced[0], map_location="cpu", weights_only=False)
    assert "stft_feature_list" in record
    assert len(record["stft_feature_list"]) == 3
    assert record["stft_feature_list"][0].shape == (2, 3, 4)

    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cache_stft_features"] is True

def test_jammer_controller_batch_passes_per_row_action_overrides(monkeypatch):
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        overrides = kwargs["action_overrides"]
        return [
            {
                "tx_iq": torch.ones(8, dtype=torch.complex64) * (i + 1),
                "tx_metadata": {"override": overrides[i]},
            }
            for i in range(len(overrides))
        ]

    monkeypatch.setattr(atu, "build_controlled_tone_pulse_batch_from_iq_batches", fake_build)

    samples = [
        {
            "iq1": torch.ones(16, dtype=torch.complex64),
            "iq2": torch.ones(16, dtype=torch.complex64),
            "iq3": torch.ones(16, dtype=torch.complex64),
        }
        for _ in range(2)
    ]
    actions = [
        {"num_tones": 1, "carrier_hz": 10.0, "seed": 101},
        {"num_tones": 3, "carrier_hz": 20.0, "seed": 202},
    ]

    out = atu.jammer_controller_batch(
        model=torch.nn.Identity(),
        samples=samples,
        actions=actions,
        jammer_sampling_freq=1_000_000.0,
        device="cpu",
    )

    assert len(out) == 2
    assert captured["action_overrides"][0]["num_tones"] == 1
    assert captured["action_overrides"][0]["carrier_hz"] == pytest.approx(10.0)
    assert captured["action_overrides"][0]["seed"] == 101
    assert captured["action_overrides"][1]["num_tones"] == 3
    assert captured["action_overrides"][1]["carrier_hz"] == pytest.approx(20.0)
    assert captured["action_overrides"][1]["seed"] == 202


def test_jammer_controller_batch_decodes_actor_action_rows(monkeypatch):
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return [
            {"tx_iq": torch.ones(8, dtype=torch.complex64), "tx_metadata": {}}
            for _ in kwargs["action_overrides"]
        ]

    class DummyBackbone(torch.nn.Module):
        max_tones = 2
        max_pulses = 3

    class DummyActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = DummyBackbone()

    monkeypatch.setattr(atu, "build_controlled_tone_pulse_batch_from_iq_batches", fake_build)

    samples = [
        {
            "iq1": torch.ones(16, dtype=torch.complex64),
            "iq2": torch.ones(16, dtype=torch.complex64),
            "iq3": torch.ones(16, dtype=torch.complex64),
        }
        for _ in range(2)
    ]
    actions = torch.zeros((2, 11 + 4 * 2 + 3), dtype=torch.float32)
    actions[0, 4] = 1.0
    actions[0, 5:7] = torch.tensor([0.1, -0.1])
    actions[1, 4] = 2.0
    actions[1, 5:7] = torch.tensor([0.2, -0.2])
    actions[1, -1] = 7.0

    atu.jammer_controller_batch(
        model=DummyActor(),
        samples=samples,
        actions=actions,
        jammer_sampling_freq=1_000_000.0,
        device="cpu",
    )

    assert captured["model"].max_tones == 2
    assert captured["action_overrides"][0]["num_tones"] == pytest.approx(1.0)
    assert captured["action_overrides"][0]["tone_freq_mean_norms"] == pytest.approx([0.1, -0.1])
    assert captured["action_overrides"][1]["num_tones"] == pytest.approx(2.0)
    assert captured["action_overrides"][1]["tone_freq_mean_norms"] == pytest.approx([0.2, -0.2])
    assert captured["action_overrides"][1]["start_offset_samples"] == pytest.approx(7.0)
