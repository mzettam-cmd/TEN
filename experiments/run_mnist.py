import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import yaml
import torch

from src.datasets.mnist import get_mnist
from src.models.ten import TEN
from src.utils.seed import set_seed

def one_hot(y, n_classes=10):
    return torch.nn.functional.one_hot(y, n_classes).float()


def test(model, loader, cfg, device):
    model.eval()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.flatten(1)
        logits = model.predict(
            x, cfg["relax_steps"], cfg["relax_lr"], 0.0
        )
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()

    return correct / total

def main():
    cfg = yaml.safe_load(open("configs/mnist.yaml"))
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_mnist(cfg["batch_size"])

    model = TEN(784, cfg["hidden_dim"], 10).to(device)
    print("Device:", device)

    for epoch in range(cfg["epochs"]):
        model.train()
        start = time.perf_counter()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.flatten(1)
            target = one_hot(y, 10)

            h, pred = model.free_phase(
                x,
                cfg["relax_steps"],
                cfg["relax_lr"],
                cfg["temperature"]
            )

            loss = ((pred - target) ** 2).mean().item()
            model.contrastive_update(
                x, target, h, pred, cfg["weight_lr"]
            )
            total_loss += loss * y.size(0)

        acc = test(model, test_loader, cfg, device)
        elapsed = time.perf_counter() - start

        print(
            f"Epoch {epoch+1:02d} | "
            f"loss={total_loss/len(train_loader.dataset):.5f} | "
            f"test_acc={acc:.4f} | time={elapsed:.2f}s"
        )

if __name__ == "__main__":
    main()
