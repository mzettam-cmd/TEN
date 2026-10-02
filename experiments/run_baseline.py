import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.datasets.mnist import get_mnist
from src.models.mlp import MLP
from src.evaluation.benchmarks import evaluate
from src.evaluation.metrics import count_parameters
from src.training.trainer import MLPTrainer
from src.utils.seed import set_seed

def main():
    cfg = yaml.safe_load(open("configs/baseline_mlp.yaml"))
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_mnist(cfg["batch_size"])

    model = MLP(784, cfg["hidden_dim"], 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    trainer = MLPTrainer(model, optimizer, torch.nn.CrossEntropyLoss(), device)

    print("Device:", device)
    print("Parameters:", count_parameters(model))

    for epoch in range(cfg["epochs"]):
        loss = trainer.train_epoch(train_loader)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1:02d} | loss={loss:.4f} | test_acc={acc:.4f}")

if __name__ == "__main__":
    main()
