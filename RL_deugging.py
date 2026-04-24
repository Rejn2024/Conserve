from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import importlib
import numpy as np
import torch
from pathlib import Path
import os
import advanced_link_skdsp_v6_robust as link6
import score_iq_decode as scorer

# TensorBoard versions used by torch can reference `np.bool8`, which may be
# missing on newer NumPy builds. Provide a compatibility alias before
# importing SummaryWriter.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from torch.utils.tensorboard import SummaryWriter

from accelerated_training_utils import (JammerVecEnv,
                                        build_stft_observation_from_iq_batch,
                                        jammer_controller_batch)
from tx_controller_tone_pulse_stft_varlen_3 import (ActorCritic,
                                                    TonePulseTXControlNetVarLen,
                                                    build_controlled_tone_pulse_batch_from_iq_batches)

@dataclass
class PPOConfig:
    rollout_steps: int = 0 #28
    updates: int = 0#1#00
    epochs: int = 50
    gamma: float = 0.99
    lr: float = 3e-4
    value_coef: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tensorboard_log_dir: str = "runs/train_rl_loop_09"
    checkpoint_dir: str = "checkpoints_rl_09"
    checkpoint_name: str = "best_model_09.pt"


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




def _normalize_action_batch(actions: Any, num_envs: int) -> Sequence[Any]:
    if num_envs == 1 and isinstance(actions, torch.Tensor) and actions.ndim >= 1:
        return [actions]
    if num_envs == 1 and isinstance(actions, np.ndarray) and actions.ndim >= 1:
        return [actions]
    return actions


def _decode_success_count(env: JammerVecEnv,
                          samples: Sequence[Dict[str, Any]],
                          actions: Any) -> Tuple[int, int]:
    action_batch = _normalize_action_batch(actions, env.num_envs)
    if len(action_batch) != len(samples):
        return 0, 0

    jam_batch = jammer_controller_batch(
        model=env.model,
        samples=samples,
        actions=action_batch,
        jammer_sampling_freq=env.jammer_sampling_freq,
        user_peak_power_fraction=env.user_peak_power_fraction,
        device=env.device,
    )

    success = 0
    total = 0
    for jam_item, sample in zip(jam_batch, samples):
        whole_iq = sample.get("whole_iq")
        whole_meta = sample.get("whole_metadata")
        if whole_iq is None or whole_meta is None:
            continue

        jammed = whole_iq.to(env.device) + jam_item["tx_iq"].to(env.device)
        rx_result = link6.rx_command_iq(jammed, whole_meta)

        score = 0.0
        if rx_result is not None:
            score = float(scorer.score_decode(rx_result, whole_meta))

        total += 1
        if score > 0.0:
            success += 1

    return success, total

def _resolve_steps_per_epoch(env: JammerVecEnv, cfg: PPOConfig) -> int:
    """Infer how many env steps are needed to consume one full cached-loader pass."""
    if cfg.rollout_steps > 0:
        return int(cfg.rollout_steps)
    # When `samples` was provided as a DataLoader, use its batch count if available.
    if hasattr(env, "_source_batches_per_epoch") and int(env._source_batches_per_epoch) > 0:
        return int(env._source_batches_per_epoch)
    # Fallback: consume all flattened rows once.
    return max(1, int(np.ceil(len(env.samples) / max(1, env.num_envs))))


@torch.no_grad()
def _evaluate_split_metrics(policy: ActorCritic,
                            env: JammerVecEnv,
                            split: str,
                            device: str,
                            value_coef: float = 0.5) -> Tuple[float, float]:
    """Evaluate loss and decode success percentage for the requested split."""
    split_samples = getattr(env, "test_samples", None) if split == "test" else getattr(env, "samples", None)
    if not split_samples:
        return float("inf"), 0.0

    logp_buf, rew_buf, val_buf = [], [], []
    decode_successes = 0
    decode_total = 0
    original_mode = env.mode
    try:
        env.set_mode(split)
        obs = env.reset()
        eval_steps = max(1, len(split_samples))
        for _ in range(eval_steps):
            model_obs = obs_to_model_obs(obs, env.jammer_sampling_freq, device=device)
            action_t, value_t, logp_t = policy.get_action_value_logp(model_obs)
            actions = action_t.squeeze()

            # active_samples = list(env._active)
            next_obs, rewards, dones, infos, success, total = env.step(actions)
            decode_successes += success
            decode_total += total

            logp_buf.append(logp_t)
            val_buf.append(value_t)
            rew_buf.append(torch.as_tensor(rewards, dtype=torch.float32, device=device))
            obs = next_obs

        returns = torch.stack(rew_buf, dim=0)
        values = torch.stack(val_buf, dim=0)
        logps = torch.stack(logp_buf, dim=0)
        if logps.ndim > returns.ndim:
            logps = logps.squeeze(-1)
        if values.ndim > returns.ndim:
            values = values.squeeze(-1)

        adv = returns - values.detach()
        adv = adv / (returns.std(unbiased=False) + 1e-8)
        policy_loss = -(logps * adv.detach()).mean()
        critic_loss = torch.nn.functional.mse_loss(values, returns)
        loss = policy_loss + value_coef * critic_loss
        decode_pct = (100.0 * decode_successes / decode_total) if decode_total else 0.0

        return float(loss.item()), float(decode_pct)
    finally:
        env.set_mode(original_mode)


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
    global_step = 0
    writer = SummaryWriter(log_dir=cfg.tensorboard_log_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / cfg.checkpoint_name
    best_test_loss = float("inf")
    ema_beta = 0.9
    ema_train_loss = None
    ema_test_loss = None

    try:
        for epoch in range(epochs):
            epoch_loss_values: List[float] = []
            epoch_decode_successes = 0
            epoch_decode_total = 0
            for update in range(total_updates):
                obs_buf, act_buf, val_buf, logp_buf, rew_buf = [], [], [], [], []

                for _ in range(len(env.samples)):
                    model_obs = obs_to_model_obs(obs, env.jammer_sampling_freq, device=device)
                    action_t, value_t, logp_t = policy.get_action_value_logp(model_obs)

                    # The vectorized env accepts per-env action payloads.
                    # actions = [int(a.item()) for a in action_t.detach().cpu()]
                    actions = action_t.squeeze() #.detach().cpu().squeeze()
                    # active_samples = list(env._active)

                    next_obs, rewards, dones, infos, success, total = env.step(actions)
                    epoch_decode_successes += success
                    epoch_decode_total += total

                    obs_buf.append(model_obs)
                    act_buf.append(action_t)
                    val_buf.append(value_t)
                    logp_buf.append(logp_t)
                    rew_buf.append(torch.as_tensor(rewards, dtype=torch.float32, device=device))

                    obs = next_obs

                returns = torch.stack(rew_buf, dim=0)
                values = torch.stack(val_buf, dim=0)
                logps = torch.stack(logp_buf, dim=0)
                if logps.ndim > returns.ndim:
                    logps = logps.squeeze(-1)
                if values.ndim > returns.ndim:
                    values = values.squeeze(-1)

                adv = returns - values.detach()
                adv = adv / (returns.std(unbiased=False) + 1e-8)
                policy_loss = -(logps * adv.detach()).mean()
                critic_loss = torch.nn.functional.mse_loss(values, returns)
                loss = policy_loss + cfg.value_coef * critic_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # mean_reward = float(returns.mean().item())
                # mean_value = float(values.mean().item())
                loss_value = float(loss.item())

                epoch_loss_values.append(loss_value)
                # writer.add_scalar("train/loss", loss_value, global_step)
                # # writer.add_scalar("train/mean_reward", mean_reward, global_step)
                # # writer.add_scalar("train/mean_value", mean_value, global_step)
                # writer.add_scalar("train/epoch", epoch, global_step)
                # # writer.add_scalar("train/update", update, global_step)
                global_step += 1

            train_loss_epoch = float(np.mean(epoch_loss_values)) if epoch_loss_values else float(loss.item())
            train_decode_pct = (100.0 * epoch_decode_successes / epoch_decode_total) if epoch_decode_total else 0.0
            test_loss, test_decode_pct = _evaluate_split_metrics(policy, env, "test", device, cfg.value_coef)

            ema_train_loss = train_loss_epoch if ema_train_loss is None else (ema_beta * ema_train_loss + (1.0 - ema_beta) * train_loss_epoch)
            ema_test_loss = test_loss if ema_test_loss is None else (ema_beta * ema_test_loss + (1.0 - ema_beta) * test_loss)

            writer.add_scalar("train/loss_epoch", train_loss_epoch, epoch)
            writer.add_scalar("train/loss_ema", ema_train_loss, epoch)
            writer.add_scalar("test/loss_epoch", test_loss, epoch)
            writer.add_scalar("test/loss_ema", ema_test_loss, epoch)
            writer.add_scalar("train/decode_success_pct", train_decode_pct, epoch)
            writer.add_scalar("test/decode_success_pct", test_decode_pct, epoch)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_test_loss": best_test_loss,
                        "cfg": cfg.__dict__,
                    },
                    best_checkpoint_path,
                )
                print(f"Saved improved checkpoint at epoch={epoch + 1} test_loss={test_loss:.6f} -> {best_checkpoint_path}")

            print(
                f"epoch={epoch + 1}/{epochs} "
                f"updates={total_updates} "
                f"steps_per_epoch={steps_per_epoch} "
                f"loss={float(loss.item()):.4f} "
                f"train_loss_ema={ema_train_loss:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_loss_ema={ema_test_loss:.4f} "
                f"train_decode_pct={train_decode_pct:.2f} "
                f"test_decode_pct={test_decode_pct:.2f} "
                f"best_test_loss={best_test_loss:.4f}"
            )
    finally:
        writer.close()

    return policy, returns, values, loss


# --- Accelerated training loop using cached inputs + DataLoader + AMP/compile helpers ---
from accelerated_training_utils import (
    precompute_training_cache,
    create_cached_dataloader,
    run_epoch_cached,
    maybe_compile_model,
)


device = "cuda" if torch.cuda.is_available() else "cpu"

# epochs = 250
batch_size = 50
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
    ).to("cuda" if torch.cuda.is_available() else "cpu")

jve = JammerVecEnv(
        samples = train_loader,
        test_samples = test_loader,
        model = ac,
        jammer_sampling_freq = float(jammer_sampling_freq),
        num_envs = 1,
        reward_fn = None,
        user_peak_power_fraction= 1.0,
        max_steps = 1,
        device = device)

cfg = PPOConfig()
cfg.rollout_steps = jve._source_batches_per_epoch
policy = ActorCritic(action_dim=ACTION_DIM,
                     in_ch=14,
                     base_ch=24,
                     max_tones=8).to(device)

p = train_rl_loop(jve.model,
                  jve,
                  cfg)

print(f'p : {p}')
