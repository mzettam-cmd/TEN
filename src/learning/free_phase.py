import torch
from src.models.energy import quadratic_energy

def relax(x, W, bias, steps=20, lr=0.05, temperature=0.01, h0=None):
    h = torch.zeros(x.size(0), W.size(1), device=x.device) if h0 is None else h0.clone()

    for _ in range(steps):
        h = h.detach().requires_grad_(True)
        energy = quadratic_energy(x, h, W, bias).sum()
        grad_h = torch.autograd.grad(energy, h)[0]

        if temperature > 0:
            noise = torch.randn_like(h) * (2.0 * lr * temperature) ** 0.5
        else:
            noise = torch.zeros_like(h)

        h = h.detach() - lr * grad_h + noise

    return h
