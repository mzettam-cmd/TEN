import time
import torch

def evaluate(model, loader, device="cpu"):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.flatten(1)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()

    return correct / total

def timed_call(fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed
