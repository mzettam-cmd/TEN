import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.datasets.xor import get_xor
from src.models.ten import TEN
from src.utils.seed import set_seed

def one_hot(y, n_classes=2):
    return torch.nn.functional.one_hot(y, n_classes).float()

def main():
    cfg = yaml.safe_load(open("configs/xor.yaml"))
    set_seed(cfg["seed"])

    X, y = get_xor()
    Y = one_hot(y)

    model = TEN(2, cfg["hidden_dim"], 2)

    for epoch in range(cfg["epochs"]):
        h, pred = model.free_phase(
            X,
            cfg["relax_steps"],
            cfg["relax_lr"],
            cfg["temperature"]
        )
        model.contrastive_update(
            X, Y, h, pred, cfg["weight_lr"]
        )

        if epoch % 100 == 0:
            loss = ((pred - Y) ** 2).mean().item()
            acc = (pred.argmax(1) == y).float().mean().item()
            print(f"Epoch {epoch:4d} | MSE={loss:.6f} | Acc={acc:.3f}")

    pred = model.predict(X, cfg["relax_steps"], cfg["relax_lr"], 0.0)
    print("\nFinal predictions:")
    for i in range(len(X)):
        print(X[i].tolist(), "->", int(pred[i].argmax()), "| target =", int(y[i]))

if __name__ == "__main__":
    main()
