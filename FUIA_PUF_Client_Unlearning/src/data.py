import numpy as np
from torchvision import datasets, transforms


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
    test  = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    return train, test


#considering only IID since it is not clear what the paper does
def partition_iid(dataset, num_clients, data_per_client):
    indices = np.random.permutation(len(dataset))
    chunks = np.array_split(indices, num_clients)
    return {i: chunks[i][:data_per_client].tolist() for i in range(num_clients)}
