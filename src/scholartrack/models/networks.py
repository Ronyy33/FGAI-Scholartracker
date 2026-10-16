"""The two neural networks: a multilayer perceptron and an LSTM classifier.

Both output a single *logit* per student-semester; ``sigmoid(logit)`` is the predicted
probability of a difficulty semester next term. Training uses binary cross-entropy on the logit
(``BCEWithLogitsLoss``), which is numerically safer than applying the sigmoid first.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class MLP(nn.Module):
    """Feed-forward network on the snapshot features.

    input (n_inputs) -> [Linear -> ReLU -> Dropout] x len(hidden_layers) -> Linear -> 1 logit
    """

    def __init__(self, n_inputs: int, hidden_layers: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = n_inputs
        for width in hidden_layers:
            layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LSTMClassifier(nn.Module):
    """LSTM over a student's semester-by-semester feature vectors.

    Input: ``x`` of shape [batch, max_length, n_inputs], left-aligned and zero-padded, and the true
    ``lengths``. Sequences are *packed*, so the LSTM never reads the padding; the final hidden
    state at each sequence's own last semester goes through dropout and a linear layer to give
    one logit.
    """

    def __init__(self, n_inputs: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_inputs,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.head(self.dropout(h_n[-1])).squeeze(-1)
