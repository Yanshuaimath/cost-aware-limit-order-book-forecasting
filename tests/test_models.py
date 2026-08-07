import torch

from lob_project.models.deeplob import DeepLOB
from lob_project.models.transformer import CompactLOBTransformer


def test_deeplob_output_shape() -> None:
    model = DeepLOB()
    output = model(torch.randn(2, 100, 40))
    assert output.shape == (2, 3)


def test_transformer_output_shape() -> None:
    model = CompactLOBTransformer(sequence_length=100, d_model=32, nhead=4, num_layers=1)
    output = model(torch.randn(2, 100, 40))
    assert output.shape == (2, 3)
