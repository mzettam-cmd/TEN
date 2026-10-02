from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist(batch_size=128, root="data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train = datasets.MNIST(root=root, train=True, download=True,
                            transform=transform)
    test = datasets.MNIST(root=root, train=False, download=True,
                           transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
