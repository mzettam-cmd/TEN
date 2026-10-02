import torch
from src.learning.free_phase import relax

def test_relaxation_shape():
    x = torch.randn(4, 3)
    W = torch.randn(3, 2)
    b = torch.zeros(2)

    h = relax(x, W, b, steps=3, lr=0.01, temperature=0.0)
    assert h.shape == (4, 2)
