import torch

def accuracy(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
