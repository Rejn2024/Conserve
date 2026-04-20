from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import importlib
import numpy as np
import torch
from pathlib import Path
import os

from accelerated_training_utils import JammerVecEnv, build_stft_observation_from_iq_batch
from tx_controller_tone_pulse_stft_varlen_3 import ActorCritic, TonePulseTXControlNetVarLen, build_controlled_tone_pulse_batch_from_iq_batches

@dataclass
class PPOConfig:
    rollout_steps: int = 1#28
    updates: int = 1#00
    epochs: int = 2
    gamma: float = 0.99
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Discrete action space size consumed by tx_controller_tone_pulse_stft_varlen_3.ActorCritic
ACTION_DIM = 20


def obs_to_model_obs(obs: Dict[str, torch.Tensor],
                     jammer_sampling_freq: float,
                     device: str) -> Dict[str, List[torch.Tensor]]:
    """Convert vectorized env IQ observation into the controller-v3 observation payload.

    The model now consumes only 2D STFT feature maps and no scalar_side tensor.
    """
    return build_stft_observation_from_iq_batch(
        iq1=obs["iq1"],
        iq2=obs["iq2"],
        iq3=obs["iq3"],
        intake_sample_rate_hz=jammer_sampling_freq,
        device=device,
    )


@torch.no_grad()
def sample_actions(model: ActorCritic, model_obs: Dict[str, List[torch.Tensor]]):
    actions, values, logp = model.get_action_value_logp(model_obs)
    return actions, values, logp


def _resolve_steps_per_epoch(env: JammerVecEnv, cfg: PPOConfig) -> int:
    """Infer how many env steps are needed to consume one full cached-loader pass."""
    if cfg.rollout_steps > 0:
        return int(cfg.rollout_steps)
    # When `samples` was provided as a DataLoader, use its batch count if available.
    if hasattr(env, "_source_batches_per_epoch") and int(env._source_batches_per_epoch) > 0:
        return int(env._source_batches_per_epoch)
    # Fallback: consume all flattened rows once.
    return max(1, int(np.ceil(len(env.samples) / max(1, env.num_envs))))


def train_rl_loop(policy: ActorCritic,
                  env: JammerVecEnv,
                  cfg: PPOConfig,
                  action_dim: int = ACTION_DIM):

    device = cfg.device

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    obs = env.reset()
    steps_per_epoch = _resolve_steps_per_epoch(env, cfg)
    total_updates = max(1, int(cfg.updates))
    epochs = max(1, int(cfg.epochs))

    returns = torch.zeros((env.num_envs,), dtype=torch.float32, device=device)
    values = torch.zeros((env.num_envs,), dtype=torch.float32, device=device)
    loss = torch.tensor(0.0, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        for update in range(total_updates):
            obs_buf, act_buf, val_buf, logp_buf, rew_buf = [], [], [], [], []

            for _ in range(cfg.rollout_steps):
                model_obs = obs_to_model_obs(obs, env.jammer_sampling_freq, device=device)
                action_t, value_t, logp_t = policy.get_action_value_logp(model_obs)

                # The vectorized env accepts per-env action payloads.
                # actions = [int(a.item()) for a in action_t.detach().cpu()]
                actions = action_t.detach().cpu().squeeze()
                next_obs, rewards, dones, infos = env.step(actions)

                obs_buf.append(model_obs)
                act_buf.append(action_t)
                val_buf.append(value_t)
                logp_buf.append(logp_t)
                rew_buf.append(torch.as_tensor(rewards, dtype=torch.float32, device=device))

                obs = next_obs

        # Example placeholder objective using rewards and value baseline.
        returns = torch.stack(rew_buf, dim=0).mean(dim=0)
        values = torch.stack(val_buf, dim=0).mean(dim=0)
        loss = values.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(
        f"epoch={epoch + 1}/{epochs} "
        f"updates={total_updates} "
        f"steps_per_epoch={steps_per_epoch} "
        f"loss={float(loss.item()):.4f}"
    )

    return policy, returns, values, loss


# --- Accelerated training loop using cached inputs + DataLoader + AMP/compile helpers ---
from accelerated_training_utils import (
    precompute_training_cache,
    create_cached_dataloader,
    run_epoch_cached,
    maybe_compile_model,
)


device = "cuda" if torch.cuda.is_available() else "cpu"

epochs = 2
batch_size = 2
jammer_sampling_freq = 2e9

train_path_dat = Path("C:/Users/theon/Jamming/train_set_00/dataset")
test_path_dat = Path("C:/Users/theon/Jamming/test_set_00/dataset")
checkpoint_dir = Path("checkpoints_accelerated")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# 1) One-time deterministic precompute/cache of whole_iq + resampled iq1/iq2/iq3.
train_cache_dir = checkpoint_dir / "train_cache"
test_cache_dir = checkpoint_dir / "test_cache"
precompute_training_cache(train_path_dat, train_cache_dir, jammer_sampling_freq, section_len=1024, overwrite=False)
precompute_training_cache(test_path_dat, test_cache_dir, jammer_sampling_freq, section_len=1024, overwrite=False)

# 2) DataLoader path with worker prefetch + pinned memory.
num_workers = 0 if os.name == "nt" else 4
pin_memory = (device == "cuda")


train_loader = create_cached_dataloader(
    train_cache_dir,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    prefetch_factor=2,
    persistent_workers=(num_workers > 0),
)
test_loader = create_cached_dataloader(
    test_cache_dir,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    prefetch_factor=2,
    persistent_workers=(num_workers > 0),
)


ac = ActorCritic(
        action_dim = ACTION_DIM,
        in_ch = 14,
        base_ch = 24,
        max_tones = 8,
    )

jve = JammerVecEnv(
        samples = train_loader,
        test_samples = test_loader,
        model = ac,
        jammer_sampling_freq = float(jammer_sampling_freq),
        num_envs = 1,
        reward_fn = None,
        user_peak_power_fraction= 40.0,
        max_steps = 1,
        device = device)

cfg = PPOConfig()
policy = ActorCritic(action_dim=ACTION_DIM,
                     in_ch=14,
                     base_ch=24,
                     max_tones=8).to(device)

p = train_rl_loop(jve.model,
                  jve,
                  cfg)

print(f'p : {p}')
