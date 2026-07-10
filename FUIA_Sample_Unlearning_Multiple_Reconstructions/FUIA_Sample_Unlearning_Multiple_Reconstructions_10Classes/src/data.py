import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms

from config import MNIST_DIGITS


#MNIST dataset. digits=None -> all 10 classes; a tuple (e.g. (0, 1)) restricts
#to those digits (paper Sec VI.A.1 binary setup).
def load_mnist(digits=MNIST_DIGITS):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST(root="data", train=True,  download=False, transform=transform)
    test_full  = datasets.MNIST(root="data", train=False, download=False, transform=transform)

    if digits is None:
        print(f"  MNIST: {len(train_full)} train, {len(test_full)} test samples (all 10 classes)")
        return train_full, test_full

    #filter to the requested digits (labels read from .targets, no image decode)
    digit_set = set(digits)
    train_indices = [i for i, y in enumerate(train_full.targets.tolist()) if y in digit_set]
    test_indices  = [i for i, y in enumerate(test_full.targets.tolist())  if y in digit_set]

    train = Subset(train_full, train_indices)
    test  = Subset(test_full,  test_indices)
    print(f"  MNIST: {len(train)} train, {len(test)} test samples (digits {tuple(sorted(digit_set))})")
    return train, test


def partition_iid(dataset, num_clients, data_per_client):
    #IID partition: assign data_per_client random samples to each client
    indices = np.random.permutation(len(dataset))
    chunks = np.array_split(indices, num_clients)
    return {i: chunks[i][:data_per_client].tolist() for i in range(num_clients)}
