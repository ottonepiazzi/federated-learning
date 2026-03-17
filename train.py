#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#utilizzo dataset Fashion MNIST

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import numpy as np
import random
from collections import OrderedDict
import copy
import torchvision.transforms as transforms
import wandb


# In[ ]:


#rete neurale usata per dataset Fashion MNIST
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# In[ ]:


#partizionamento dei dati IID
def dataset_partition_IID(dataset, num_clients):
    indices = np.random.permutation(len(dataset))
    chunks = np.array_split(indices, num_clients)
    return {i: chunks[i].tolist() for i in range(num_clients)}


# In[ ]:


#partizionamento dei dati NON_IID (da controllare modifiche per Fashion MNIST)
def dataset_partition_not_IID(dataset, num_clients, shards_per_client=2):
   labels = np.array([dataset.targets[i].item() if torch.is_tensor(dataset.targets[i]) else dataset.targets[i] for i in range(len(dataset))])
   sorted_indices = np.argsort(labels)

   num_shards = num_clients * shards_per_client
   shard_size = len(dataset) // num_shards
   shards = [sorted_indices[i*shard_size:(i+1)*shard_size].tolist() for i in range(num_shards)]

   random.shuffle(shards)

   client_data = {}
   for i in range(num_clients):
      client_data[i] = shards[i*shards_per_client] + shards[i*shards_per_client +1]

   return client_data


# In[ ]:


#ClientUpdate function
def client_update(client_model, dataset, indices, epochs, batch_size, lr):
    client_model.train()
    optimizer = torch.optim.SGD(client_model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    #creazione DataLoader con solo i dati per il singolo client
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    num_batches = 0

    #addestramento presso il singolo client per E epoche
    for epoch in range(epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            output = client_model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches        

    return client_model.state_dict(), len(indices), avg_loss


# In[ ]:


#simulazione del server: algoritmo FedAvg
def federated_averaging(global_model, client_model_weights):
    total_samples = sum(n_k for _, n_k in client_model_weights)

    #inizializzazione dei pesi aggregati a 0
    aggregated = OrderedDict()
    for key in client_model_weights[0][0].keys():
        aggregated[key] = torch.zeros_like(client_model_weights[0][0][key], dtype=torch.float32)

    #media pesata
    for state_dict, n_k in client_model_weights:
        weight = n_k / total_samples
        for key in aggregated:
            aggregated[key] += weight * state_dict[key].float()

    #aggiornamento parametri del modello globale
    global_model.load_state_dict(aggregated)
    return global_model


# In[ ]:


#calcolo accuratezza sul test set
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/total


# In[ ]:


#simulazione algoritmo completo
def run_federated_averaging(num_clients = 2, fraction = 1, num_rounds = 50, local_epochs = 5, batch_size = 10, lr = 0.01, iid = True):

    #impostazione WandB

    wandb.init(project="FedAvg", config={
        "num_clients": num_clients,
        "fraction":fraction,
        "num_rounds":num_rounds,
        "local_epochs":local_epochs,
        "batch_size": batch_size,
        "lr":lr,
        "iid":iid,
    })

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.FashionMNIST(root="data",
                                          train = True,
                                          download = True,
                                          transform = transform,)

    test_dataset = datasets.FashionMNIST(root="data",
                                          train = False,
                                          download = True,
                                          transform = transform,)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


    if iid:
        client_data = dataset_partition_IID(train_dataset, num_clients)
    else:
        client_data = dataset_partition_not_IID(train_dataset, num_clients)


    global_model = NeuralNetwork()

    m = max(1, int(fraction*num_clients))

    for round_t in range(1, num_rounds+1):
        selected_clients = random.sample(range(num_clients), m)

        client_updates = []
        client_losses = []
        for k in selected_clients:
            local_model = copy.deepcopy(global_model)

            updated_weights, n_k, loss = client_update(client_model=local_model, 
                                                 dataset=train_dataset,
                                                 indices=client_data[k],
                                                 epochs=local_epochs,
                                                 batch_size=batch_size,
                                                 lr=lr,
                                                 )
            client_updates.append((updated_weights, n_k))
            client_losses.append(loss)

        avg_round_loss = sum(client_losses) / len(client_losses)
        global_model = federated_averaging(global_model, client_updates)


        accuracy = evaluate(global_model, test_loader)
        wandb.log({"round": round_t, "accuracy":accuracy, "loss":avg_round_loss})

        if round_t % 5==0 or round_t ==1:
            print(f"Round {round_t:3d}/{num_rounds} | Loss: {avg_round_loss:.4f} | Accuratezza test: {accuracy:.4f} ({accuracy*100:.2f}%)")

    final_accuracy = evaluate(global_model, test_loader)
    wandb.log({"final_accuracy" : final_accuracy})
    print(f"\n  Accuratezza finale: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

    wandb.finish()
    return global_model



# In[ ]:


#esecuzione simulazione

print("\n Esperimento 1: Partizionamento IID\n")
run_federated_averaging(iid=True, num_rounds=50, local_epochs=5, batch_size=10, fraction=0.1)

print("\n\n Esperimento 2: Partizionamento Non-IID\n")
run_federated_averaging(iid=False, num_rounds=50, local_epochs=5, batch_size=10, fraction=0.1)


# In[ ]:




