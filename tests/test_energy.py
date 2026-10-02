import torch
from src.models.energy import quadratic_energy

def test_energy_shape():
    x = torch.randn(4, 3)
    h = torch.randn(4, 2)
    W = torch.randn(3, 2)
    b = torch.randn(2)

    E = quadratic_energy(x, h, W, b)
    assert E.shape == (4,)
