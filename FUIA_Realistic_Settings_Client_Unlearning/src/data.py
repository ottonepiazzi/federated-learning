import numpy as np
from torchvision import datasets, transforms

from config import NUM_CLASSES, TARGET_SIZE, TARGET_CLIENT, SEED


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
    test  = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    return train, test


#IID partition with one special target client
#Target client (id = TARGET_CLIENT) holds exactly `target_size` images
#The remaining (num_clients - 1) clients evenly split the rest of the dataset
#with approximately the same per-class distribution (IID)
#Reproducible: driven by a seeded RNG independent of the global state
def partition_iid(dataset, num_clients, target_size=TARGET_SIZE,
                  target_client=TARGET_CLIENT, seed=SEED):
    rng = np.random.RandomState(seed)

    #Pull labels efficiently when available (MNIST exposes .targets)
    if hasattr(dataset, "targets"):
        labels = np.asarray(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    #Per-class index pools, deterministically shuffled
    class_indices = {c: np.where(labels == c)[0].copy() for c in range(NUM_CLASSES)}
    for c in class_indices:
        rng.shuffle(class_indices[c])

    #Reserve target image(s): pick from a fixed class for determinism
    target_indices = []
    target_class_cycle = list(range(NUM_CLASSES))
    for i in range(target_size):
        c = target_class_cycle[i % NUM_CLASSES]
        target_indices.append(int(class_indices[c][0]))
        class_indices[c] = class_indices[c][1:]

    client_data = {cid: [] for cid in range(num_clients)}
    client_data[target_client] = target_indices

    #Distribute each class evenly across non-target clients
    non_target = [cid for cid in range(num_clients) if cid != target_client]
    for c, idx_pool in class_indices.items():
        chunks = np.array_split(idx_pool, len(non_target))
        for cid, chunk in zip(non_target, chunks):
            client_data[cid].extend(chunk.tolist())

    #Shuffle each non-target client's data so batches mix classes
    for cid in non_target:
        rng.shuffle(client_data[cid])

    return client_data
