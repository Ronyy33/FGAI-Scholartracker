"""Training loop with early stopping, shared by the MLP and the LSTM.

Each epoch: shuffle the training rows (seeded), take mini-batches, minimise binary
cross-entropy with AdamW, then measure the loss on the *validation* rows. Training stops when the
validation loss has not improved for ``patience`` epochs, and the weights from the best epoch are
restored. The test partition is never seen here.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


@dataclass
class TrainSettings:
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    grad_clip: float | None = None


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    epochs_run: int
    seconds: float
    history: list[dict[str, float]] = field(default_factory=list)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _logits(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(*inputs)


def predict_proba(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> np.ndarray:
    return torch.sigmoid(_logits(model, inputs)).numpy().astype(float)


def train_network(
    model: nn.Module,
    train_inputs: tuple[torch.Tensor, ...],
    y_train: torch.Tensor,
    val_inputs: tuple[torch.Tensor, ...],
    y_val: torch.Tensor,
    settings: TrainSettings,
    seed: int,
) -> TrainResult:
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(y_train)
    best_loss, best_state, best_epoch, waited = float("inf"), None, 0, 0
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    for epoch in range(1, settings.max_epochs + 1):
        model.train()
        order = torch.randperm(n, generator=generator)
        total = 0.0
        for begin in range(0, n, settings.batch_size):
            idx = order[begin : begin + settings.batch_size]
            loss = loss_fn(model(*(t[idx] for t in train_inputs)), y_train[idx])
            optimiser.zero_grad()
            loss.backward()
            if settings.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimiser.step()
            total += loss.item() * len(idx)
        val_loss = float(loss_fn(_logits(model, val_inputs), y_val))
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= settings.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return TrainResult(best_epoch, best_loss, len(history), time.perf_counter() - start, history)
