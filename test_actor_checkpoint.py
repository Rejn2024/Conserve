from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from actor_checkpoint import load_past_actor_model


class TinyActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy = nn.Linear(2, 1)


def _filled_actor(value: float) -> TinyActor:
    actor = TinyActor()
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.fill_(value)
    return actor


@pytest.mark.parametrize("state_key", ["actor_state_dict", "model_state_dict", "state_dict"])
def test_load_past_actor_model_supports_wrapped_checkpoint_formats(tmp_path, state_key):
    expected = _filled_actor(3.0)
    checkpoint_path = tmp_path / "past_actor.pt"
    torch.save(
        {state_key: expected.state_dict(), "critic_state_dict": {"weight": torch.ones(1)}, "epoch": 12},
        checkpoint_path,
    )

    restored, metadata = load_past_actor_model(TinyActor().train(), checkpoint_path)

    assert metadata == {"epoch": 12}
    assert not restored.training
    for actual, wanted in zip(restored.parameters(), expected.parameters()):
        assert torch.equal(actual, wanted)


def test_load_past_actor_model_supports_raw_and_dataparallel_state_dict(tmp_path):
    expected = _filled_actor(5.0)
    prefixed_state = {f"module.{name}": value for name, value in expected.state_dict().items()}
    checkpoint_path = tmp_path / "past_actor.pt"
    torch.save(prefixed_state, checkpoint_path)

    restored, metadata = load_past_actor_model(TinyActor(), checkpoint_path)

    assert metadata == {}
    for actual, wanted in zip(restored.parameters(), expected.parameters()):
        assert torch.equal(actual, wanted)


def test_load_past_actor_model_rejects_unknown_checkpoint_format(tmp_path):
    checkpoint_path = tmp_path / "bad_checkpoint.pt"
    torch.save({1: "not a state dictionary"}, checkpoint_path)

    with pytest.raises(KeyError, match="actor_state_dict"):
        load_past_actor_model(TinyActor(), checkpoint_path)
