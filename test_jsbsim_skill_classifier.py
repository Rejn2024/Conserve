import pytest

torch = pytest.importorskip("torch")

from jsbsim_skill_classifier import (
    JSBSimSkillClassifier,
    SinusoidalPositionalEncoding,
    WindowedTrajectoryDataset,
    aggregate_window_predictions,
    collate_windows,
)


def test_window_collation_and_padding_mask():
    dataset = WindowedTrajectoryDataset(
        [torch.ones(5, 3), torch.full((2, 3), 2.0)],
        [1, 0],
        window_size=4,
    )
    assert len(dataset) == 3

    features, labels, mask, ids = collate_windows([dataset[1], dataset[2]])
    assert features.shape == (2, 2, 3)
    assert labels.tolist() == [1, 0]
    assert mask.tolist() == [[False, True], [False, False]]
    assert ids.tolist() == [0, 1]


def test_classifier_handles_odd_model_width_and_padding():
    model = JSBSimSkillClassifier(
        input_features=3,
        num_classes=4,
        d_model=15,
        nhead=3,
        num_layers=1,
        conv_stride=2,
    )
    features = torch.randn(2, 7, 3)
    padding_mask = torch.tensor(
        [[False] * 7, [False, False, False, False, True, True, True]]
    )
    output = model(features, padding_mask)
    assert output.shape == (2, 4)
    assert torch.isfinite(output).all()


def test_positional_encoding_grows_and_predictions_aggregate():
    encoding = SinusoidalPositionalEncoding(7, max_length=2)
    assert encoding(torch.zeros(2, 5, 7)).shape == (2, 5, 7)

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    aggregated = aggregate_window_predictions(logits, torch.tensor([4, 4, 9]))
    assert set(aggregated) == {4, 9}
    assert aggregated[4].shape == (2,)
