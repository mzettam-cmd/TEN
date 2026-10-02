import torch
import torch.nn as nn

from src.learning.free_phase import relax
from src.learning.contrastive_rule import update_weights

class TEN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.05)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.05)
        self.b2 = nn.Parameter(torch.zeros(output_dim))

    def free_phase(self, x, steps, lr, temperature):
        h = torch.tanh(x @ self.W1 + self.b1)
        y = relax(x=h, W=self.W2, bias=self.b2,
                  steps=steps, lr=lr, temperature=temperature)
        return h, y

    @torch.no_grad()
    def contrastive_update(self, x, target, h_free, y_free, lr):
        # Cible continue: vecteur one-hot.
        # Cette règle est volontairement simple et constitue
        # une hypothèse expérimentale à tester.
        y_target = target.float()

        delta_w2 = h_free.T @ (y_target - y_free) / x.size(0)
        delta_b2 = (y_target - y_free).mean(dim=0)

        self.W2.add_(lr * delta_w2)
        self.b2.add_(lr * delta_b2)

        # Signal local approximatif pour la couche interne.
        error = (y_target - y_free)
        local = error @ self.W2.T
        delta_w1 = x.T @ (local * (1 - h_free.pow(2))) / x.size(0)
        delta_b1 = (local * (1 - h_free.pow(2))).mean(dim=0)

        self.W1.add_(lr * delta_w1)
        self.b1.add_(lr * delta_b1)

    def predict(self, x, steps=20, lr=0.05, temperature=0.0):
        h = torch.tanh(x @ self.W1 + self.b1)
        return relax(x=h, W=self.W2, bias=self.b2,
                     steps=steps, lr=lr, temperature=temperature)
