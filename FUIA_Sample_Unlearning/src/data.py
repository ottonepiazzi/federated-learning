import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms


#MNIST dataset, binary classification (digits 0 and 1 only, paper Sec VI.A.1)
def load_mnist_binary():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST(root="data", train=True,  download=False, transform=transform)
    test_full  = datasets.MNIST(root="data", train=False, download=False, transform=transform)

    #filter to digits 0 and 1 only
    train_indices = [i for i, (_, label) in enumerate(train_full) if label in (0, 1)]
    test_indices  = [i for i, (_, label) in enumerate(test_full)  if label in (0, 1)]

    train = Subset(train_full, train_indices)
    test  = Subset(test_full,  test_indices)
    print(f"  Binary MNIST: {len(train)} train, {len(test)} test samples (digits 0, 1)")
    return train, test


def partition_iid(dataset, num_clients, data_per_client):
    #IID partition: assign data_per_client random samples to each client
    indices = np.random.permutation(len(dataset))
    chunks = np.array_split(indices, num_clients)
    return {i: chunks[i][:data_per_client].tolist() for i in range(num_clients)}
