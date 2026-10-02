import torch

def quadratic_energy(x, h, W, bias):
    interaction = (x @ W) * h
    restoring = 0.5 * h.pow(2)
    return (restoring - interaction - h * bias).sum(dim=-1)
