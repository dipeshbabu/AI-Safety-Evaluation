import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_data(batch_size=64, test_batch_size=1000):
    """Load MNIST dataset with proper formatting"""
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, test_set
