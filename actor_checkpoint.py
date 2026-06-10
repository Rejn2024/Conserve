"""Utilities for restoring a previously saved PyTorch actor model.

Example::

    import torch

    from actor_checkpoint import load_past_actor_model
    from tx_controller_tone_pulse_stft_varlen_9 import ActorCritic

    actor = ActorCritic(in_ch=23, base_ch=24, max_tones=8, max_pulses=33)
    actor, checkpoint_metadata = load_past_actor_model(
        actor,
        "checkpoints_rl/best_model.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

The actor must be constructed with the same architecture used when the
checkpoint was written. The loader accepts this repository's SAC
(``actor_state_dict``) and PPO (``model_state_dict``) checkpoint formats, as
well as generic ``state_dict`` wrappers and raw state dictionaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any, TypeVar, Union

import torch
import torch.nn as nn


ActorT = TypeVar("ActorT", bound=nn.Module)
CheckpointPath = Union[str, PathLike[str]]


def _select_actor_state_dict(checkpoint: Any) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Actor checkpoint must contain a mapping or a raw state dictionary.")

    for key in ("actor_state_dict", "model_state_dict", "state_dict"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, Mapping):
            metadata = {
                name: value
                for name, value in checkpoint.items()
                if name != key and not str(name).endswith("_state_dict")
            }
            return state_dict, metadata

    if checkpoint and all(isinstance(name, str) for name in checkpoint):
        return checkpoint, {}

    raise KeyError(
        "Checkpoint does not contain 'actor_state_dict', 'model_state_dict', "
        "or 'state_dict', and is not a raw state dictionary."
    )


def _strip_distributed_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remove a DataParallel/DDP ``module.`` prefix when every key has it."""

    if state_dict and all(name.startswith("module.") for name in state_dict):
        return {name.removeprefix("module."): value for name, value in state_dict.items()}
    return dict(state_dict)


def load_past_actor_model(
    actor: ActorT,
    checkpoint_path: CheckpointPath,
    *,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> tuple[ActorT, dict[str, Any]]:
    """Load a past actor checkpoint into ``actor`` and prepare it for inference.

    Args:
        actor: A newly constructed actor with the checkpoint's architecture.
        checkpoint_path: Path to a saved checkpoint or raw state dictionary.
        device: Device on which the restored actor should run.
        strict: Forwarded to :meth:`torch.nn.Module.load_state_dict`.

    Returns:
        A tuple containing the restored, evaluation-mode actor and any
        non-state-dictionary checkpoint metadata (for example, ``epoch`` or
        ``global_step``). Other model and optimizer state dictionaries are
        omitted from the returned metadata.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict, metadata = _select_actor_state_dict(checkpoint)
    actor.load_state_dict(_strip_distributed_prefix(state_dict), strict=strict)
    actor.to(device)
    actor.eval()
    return actor, metadata
