from __future__ import annotations

from torch import nn

from lob_project.models.deeplob import DeepLOB
from lob_project.models.transformer import CompactLOBTransformer


def make_torch_model(name: str, sequence_length: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "deeplob":
        return DeepLOB()
    if name == "transformer":
        allowed = {
            key: kwargs[key]
            for key in ["d_model", "nhead", "num_layers", "dim_feedforward", "dropout"]
            if key in kwargs and kwargs[key] is not None
        }
        return CompactLOBTransformer(sequence_length=sequence_length, **allowed)
    raise ValueError(f"Unknown torch model: {name}")
