import torch

def get_xor():
    X = torch.tensor([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])
    y = torch.tensor([0, 1, 1, 0])
    return X, y
