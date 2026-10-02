import torch

@torch.no_grad()
def update_weights(x, h_free, h_target, W, bias, lr):
    # Règle locale contrastive:
    # les corrélations cible sont renforcées,
    # les corrélations libres sont soustraites.
    delta_W = x.T @ (h_target - h_free) / x.size(0)
    delta_b = (h_target - h_free).mean(dim=0)

    W.add_(lr * delta_W)
    bias.add_(lr * delta_b)
