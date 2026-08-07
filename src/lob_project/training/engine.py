from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    model_state: dict
    history: list[dict]
    best_epoch: int
    best_validation_loss: float


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch = y.shape[0]
        total_loss += float(loss.detach()) * batch
        correct += int((logits.argmax(dim=1) == y).sum())
        count += batch
    return total_loss / max(count, 1), correct / max(count, 1)


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
) -> TrainResult:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        with torch.no_grad():
            val_loss, val_accuracy = _run_epoch(
                model, validation_loader, criterion, device, optimizer=None
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_accuracy:.3f} val_acc={val_accuracy:.3f}"
        )
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return TrainResult(best_state, history, best_epoch, best_loss)


def predict_probabilities(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    probabilities = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)
