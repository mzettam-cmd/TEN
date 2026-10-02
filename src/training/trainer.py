import torch
from tqdm import tqdm

class MLPTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            x = x.flatten(1)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * y.size(0)

        return total_loss / len(loader.dataset)
