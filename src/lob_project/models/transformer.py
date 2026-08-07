from __future__ import annotations

import torch
from torch import nn


class CompactLOBTransformer(nn.Module):
    def __init__(
        self,
        sequence_length: int = 100,
        n_features: int = 40,
        n_classes: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.input_projection = nn.Linear(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, sequence_length + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.sequence_length:
            raise ValueError(
                f"Transformer expects [batch, {self.sequence_length}, features], "
                f"received {tuple(x.shape)}"
            )
        batch = x.shape[0]
        x = self.input_projection(x)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.position[:, : x.shape[1] + 1]
        x = self.encoder(x)
        return self.classifier(self.norm(x[:, 0]))
