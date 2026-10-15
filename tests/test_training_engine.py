from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lob_project.training.engine import _run_epoch, train_torch_model


def _log_probabilities(
    labels: torch.Tensor, rare_probability: float, flat_probability: float
) -> torch.Tensor:
    true_probability = torch.where(labels == 1, flat_probability, rare_probability)
    probabilities = ((1 - true_probability) / 2).unsqueeze(1).repeat(1, 3)
    probabilities.scatter_(1, labels.unsqueeze(1), true_probability.unsqueeze(1))
    return probabilities.log()


@pytest.mark.parametrize("batch_size", [1, 2, 5, 12])
@pytest.mark.parametrize("reorder", [False, True], ids=["grouped", "mixed"])
@pytest.mark.parametrize("weight_scale", [None, 1.0, 0.001], ids=["unweighted", "weighted", "tiny"])
def test_epoch_metrics_match_full_dataset_cross_entropy(
    batch_size: int, reorder: bool, weight_scale: float | None
) -> None:
    labels = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2])
    logits = _log_probabilities(labels, rare_probability=0.9, flat_probability=0.2)
    if reorder:
        order = torch.tensor([2, 0, 3, 10, 4, 5, 1, 6, 11, 7, 8, 9])
        labels, logits = labels[order], logits[order]
    weights = None if weight_scale is None else torch.tensor([4.0, 1.0, 4.0]) * weight_scale
    criterion = nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(TensorDataset(logits, labels), batch_size=batch_size)

    loss, accuracy = _run_epoch(nn.Identity(), loader, criterion, torch.device("cpu"))

    assert loss == pytest.approx(criterion(logits, labels).item(), rel=1e-6)
    assert accuracy == pytest.approx((logits.argmax(dim=1) == labels).float().mean().item())


class _EpochPredictions(nn.Module):
    """Supply deterministic epoch predictions while exercising the real training loop."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("logits", logits)
        self.register_buffer("epoch_index", torch.tensor(-1))

    def train(self, mode: bool = True) -> _EpochPredictions:
        super().train(mode)
        if mode:
            self.epoch_index.add_(1)
        return self

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        epoch = min(self.epoch_index.item(), len(self.logits) - 1)
        return self.logits[epoch, indices] + self.bias


def test_weighted_validation_selects_best_checkpoint_and_stops() -> None:
    labels = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2])
    logits_a = _log_probabilities(labels, rare_probability=0.9, flat_probability=0.4)
    logits_b = _log_probabilities(labels, rare_probability=0.4, flat_probability=0.9)
    reference = nn.CrossEntropyLoss(weight=torch.tensor([4.0, 1.0, 4.0]))
    expected_losses = [reference(logits, labels).item() for logits in (logits_a, logits_b)]
    assert expected_losses[0] < expected_losses[1]
    dataset = TensorDataset(torch.arange(len(labels)), labels)
    model = _EpochPredictions(torch.stack([logits_a, logits_b]))

    result = train_torch_model(
        model,
        DataLoader(dataset, batch_size=len(labels)),
        DataLoader(dataset, batch_size=2),
        epochs=3,
        learning_rate=0.0,
        weight_decay=0.0,
        patience=1,
        device=torch.device("cpu"),
        balance_weights=True,
    )

    assert result.best_epoch == 1
    assert result.best_validation_loss == pytest.approx(expected_losses[0], rel=1e-6)
    assert result.model_state["epoch_index"].item() == 0
    assert len(result.history) == 2
    assert [entry["validation_loss"] for entry in result.history] == pytest.approx(
        expected_losses, rel=1e-6
    )


def test_weighted_training_step_matches_direct_cross_entropy() -> None:
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    labels = torch.tensor([0, 1, 1, 2])
    model = nn.Linear(2, 3)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.1], [0.3, 0.4], [-0.2, 0.1]]))
        model.bias.zero_()
    reference_model = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.004, 0.001, 0.004]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
    reference_logits = reference_model(inputs)
    reference_loss = criterion(reference_logits, labels)
    reference_loss.backward()
    nn.utils.clip_grad_norm_(reference_model.parameters(), max_norm=5.0)
    reference_optimizer.step()

    loss, accuracy = _run_epoch(
        model,
        DataLoader(TensorDataset(inputs, labels), batch_size=len(labels)),
        criterion,
        torch.device("cpu"),
        optimizer,
    )

    assert loss == pytest.approx(reference_loss.item(), rel=1e-6)
    assert accuracy == pytest.approx(
        (reference_logits.argmax(dim=1) == labels).float().mean().item()
    )
    for actual, expected in zip(model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(actual, expected)
