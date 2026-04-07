#!/usr/bin/env python
# coding: utf-8

#FUIA (Federated Unlearning Inversion Attack) implementation -> focusing on Client Unlearning (or Sample Unlearning)
#Dataset used: MNIST
#using FedAvg -> started from the previous implementation of FedAvg made

#possible extensions: using FedSGD (instead of FedAvg) and considering Sample Unlearning

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
import matplotlib.pyplot as plt


#model architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 10) #10 classes instead of 2
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



#IID Data partitioning
def dataset_partition_IID(dataset, num_clients):
    indices = np.random.permutation(len(dataset))
    chunks = np.array_split(indices, num_clients)
    return {i: chunks[i].tolist() for i in range(num_clients)}

#Non-IID Data partitioning
def dataset_partition_not_IID(dataset, num_clients, shards_per_client=2):
   #labels = np.array([dataset.targets[i].item() if torch.is_tensor(dataset.targets[i]) else dataset.targets[i] for i in range(len(dataset))])
   labels = np.array([dataset[i][1] for i in range(len(dataset))])
   sorted_indices = np.argsort(labels)

   num_shards = num_clients * shards_per_client
   shard_size = len(dataset) // num_shards
   shards = [sorted_indices[i*shard_size:(i+1)*shard_size].tolist() for i in range(num_shards)]

   random.shuffle(shards)

   client_data = {}
   for i in range(num_clients):
      client_data[i] = shards[i*shards_per_client] + shards[i*shards_per_client +1]

   return client_data




#ClientUpdate function
def client_update(client_model, dataset, indices, epochs, batch_size, lr):
    client_model.train()
    optimizer = torch.optim.SGD(client_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    #DataLoader creation, with only the data for a single client 
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



#Simulation of the server during FL: FedAvg algorithm
def federated_averaging(global_model, client_model_weights):
    total_samples = sum(n_k for _, n_k in client_model_weights)

    #aggregated weights initialization to 0 
    aggregated = OrderedDict()
    for key in client_model_weights[0][0].keys():
        aggregated[key] = torch.zeros_like(client_model_weights[0][0][key], dtype=torch.float32)

    #weighted average
    for state_dict, n_k in client_model_weights:
        weight = n_k / total_samples
        for key in aggregated:
            aggregated[key] += weight * state_dict[key].float()

    #update of the parameters of the global model
    global_model.load_state_dict(aggregated)
    return global_model




#accuracy on test set
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



#filtering only 0 and 1 classes from MNIST
def filter_to_binary_classes(dataset, class_0=0, class_1=1):
    indices = [i for i, (_, label) in enumerate(dataset) if label in (class_0, class_1)]
    subset = Subset(dataset, indices)
    return subset




#pretraining of the model
def pretrain(model, dataset, epochs=50, batch_size=32, lr=0.01):
    print(f"Dataset size: {len(dataset)}")
    sample_img, sample_label = dataset[0]
    #print(f"Sample shape: {sample_img.shape}, label: {sample_label}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0
        for images, labels in loader:
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Pre-training epoch {epoch}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    return model



#simulation of the whole algorithm
def run_federated_averaging(num_clients = 50, fraction = 0.2, num_rounds = 10, local_epochs = 3, batch_size = 32, lr = 0.01, iid = False):

    #WandB initialization
    wandb.init(project="FUIA", config={
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

    full_train_dataset = datasets.MNIST(root="data", train = True, download = True, transform = transform,)
    full_test_dataset = datasets.MNIST(root="data", train = False, download = True, transform = transform,)

    #train_dataset = filter_to_binary_classes(full_train_dataset)
    #test_dataset = filter_to_binary_classes(full_test_dataset)
    train_dataset = full_train_dataset
    test_dataset = full_test_dataset


    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    #data split following the paper
    n_total = len(train_dataset)
    n_pretrain = int(0.8 * n_total) #80% of the data used in pre-training, 
    n_private = n_total - n_pretrain #and 20% private to the clients
    pretrain_subset, private_subset = torch.utils.data.random_split(train_dataset, [n_pretrain, n_private])


    global_model = CNN()
    print("Started Pre-Training")
    global_model = pretrain(global_model, pretrain_subset, epochs=50, batch_size=32, lr=0.01) #MODIFICATO lr da 0.1
    print("Pre-Training completed")

    pretrained_state_dict = copy.deepcopy(global_model.state_dict())


    if iid:
        client_data = dataset_partition_IID(private_subset, num_clients)
    else:
        client_data = dataset_partition_not_IID(private_subset, num_clients)

    #1 data per client
    #for k in client_data:
    #    client_data[k] = [client_data[k][0]]


    m = max(1, int(fraction*num_clients))

    #dictionary to store the per-round client updates
    stored_updates = {} #key: (round, client_id), value: parameters difference


    for round_t in range(1, num_rounds+1):
        selected_clients = random.sample(range(num_clients), m)

        global_params_before_round = copy.deepcopy(global_model.state_dict())

        client_updates = []
        client_losses = []
        for k in selected_clients:
            local_model = copy.deepcopy(global_model)

            updated_weights, n_k, loss = client_update(client_model=local_model, 
                                                 dataset=private_subset,
                                                 indices=client_data[k],
                                                 epochs=local_epochs,
                                                 batch_size=batch_size,
                                                 lr=lr,
                                                 )
            client_updates.append((updated_weights, n_k))
            client_losses.append(loss)

            param_diff = {}
            for key in updated_weights:
                param_diff[key] = updated_weights[key] - global_params_before_round[key]

            stored_updates[(round_t, k)] = param_diff


        avg_round_loss = sum(client_losses) / len(client_losses)
        global_model = federated_averaging(global_model, client_updates)


        accuracy = evaluate(global_model, test_loader)
        wandb.log({"round": round_t, "accuracy":accuracy, "loss":avg_round_loss})

        if round_t % 5==0 or round_t==1:
            print(f"Round {round_t:3d}/{num_rounds} | Loss: {avg_round_loss:.4f} | Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    final_accuracy = evaluate(global_model, test_loader)
    wandb.log({"final_accuracy" : final_accuracy})
    print(f"\n  Final accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

    wandb.finish()
    return global_model, stored_updates, client_data, private_subset, pretrained_state_dict



#implementation of Retraing as the FU method
def retraining(pretrained_state_dict, private_subset, client_data, target_client, num_clients=50, fraction=0.2, num_rounds=10, local_epochs=3, batch_size=32, lr=0.01):
    unlearned_model = CNN()
    unlearned_model.load_state_dict(pretrained_state_dict)

    remaining_clients = [c for c in range(num_clients) if c != target_client]

    m = max(1, int(fraction * num_clients))

    stored_updates_unlearn = {}

    for round_t in range(1, num_rounds + 1):
        selected_clients = random.sample(remaining_clients, min(m, len(remaining_clients)))

        global_params_before_round = copy.deepcopy(unlearned_model.state_dict())

        client_updates = []
        for k in selected_clients:
            local_model = copy.deepcopy(unlearned_model)

            updated_weights, n_k, loss = client_update(
                client_model = local_model,
                dataset=private_subset,
                indices=client_data[k],
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
            )

            client_updates.append((updated_weights, n_k))

            param_diff = {}
            for key in updated_weights:
                param_diff[key] = updated_weights[key] - global_params_before_round[key]
            stored_updates_unlearn[(round_t, k)] = param_diff

        unlearned_model = federated_averaging(unlearned_model, client_updates)

    return unlearned_model, stored_updates_unlearn



#Step 1: Gradient Separation
def gradient_separation(stored_updates, target_client):
    #extracts the "clean" gradient from the target client from FL
    target_rounds = []
    for (round_t, k) in stored_updates.keys():
        if k == target_client:
            target_rounds.append(round_t)

    if len(target_rounds) == 0:
        raise ValueError("The target client did not participate in any round")

    clean_gradient = None

    for t in target_rounds:
        sum_t = 0.0
        clients_in_round = []
        for (round_t, k) in stored_updates.keys():
            if round_t == t:
                clients_in_round.append(k)
                l1_norm = sum(stored_updates[(t, k)][key].abs().sum().item() for key in stored_updates[(t, k)])
                sum_t += l1_norm

        target_l1_norm = sum(stored_updates[(t, target_client)][key].abs().sum().item() for key in stored_updates[(t, target_client)])

        gamma_t = target_l1_norm / sum_t

        if clean_gradient is None:
            clean_gradient = {}
            for key in stored_updates[(t, target_client)]:
                clean_gradient[key] = gamma_t * stored_updates[(t, target_client)][key].clone()
        else:
            for key in clean_gradient:
                clean_gradient[key] += gamma_t * stored_updates[(t, target_client)][key]

    return clean_gradient




#Step 2: Taget Gradient Acquisition
def target_gradient_acquisition(original_model, unlearned_model):
    #calculate the difference between the original model and the unlearned model

    gradient_star = {}
    original_params = original_model.state_dict()
    unlearned_params = unlearned_model.state_dict()

    for key in original_params:
        gradient_star[key] = original_params[key].float() - unlearned_params[key].float()

    return gradient_star



#Step 3: Gradient Inversion

#regularization term
def total_variation(x):
    diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
    return torch.sum(diff_h**2) + torch.sum(diff_w**2)

def cosine_similarity_gradients(grad_a, grad_b):
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for key in grad_a:
        dot += (grad_a[key] * grad_b[key]).sum()
        norm_a += (grad_a[key]**2).sum()
        norm_b += (grad_b[key]**2).sum()
    return dot / (torch.sqrt(norm_a) * torch.sqrt(norm_b) + 1e-8)

def gradient_inversion(original_model, clean_gradient, grad_star, label, gamma=0.1, alpha=0.001, num_iterations=3000, lr=0.1):
    #random starting image
    dummy_image = torch.randn(1, 1, 28, 28, requires_grad=True)
    dummy_label = torch.tensor([label])

    optimizer = torch.optim.Adam([dummy_image], lr=lr) #MODIFICATO Adam con SGD
    loss_fn = nn.CrossEntropyLoss()

    model = copy.deepcopy(original_model)
    model.eval()

    for iteration in range(1, num_iterations+1):
        optimizer.zero_grad()

        model.zero_grad()
        output = model(dummy_image)
        loss = loss_fn(output, dummy_label)

        grad_virtual = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        grad_virtual_dict = {}
        for (key, _), g in zip(model.named_parameters(), grad_virtual):
            grad_virtual_dict[key] = g

        cos_clean = cosine_similarity_gradients(grad_virtual_dict, clean_gradient)
        cos_star = cosine_similarity_gradients(grad_virtual_dict, grad_star)

        similarity_loss = -((1-gamma)*cos_clean + gamma*cos_star)

        tv_loss = alpha * total_variation(dummy_image)

        total_loss = similarity_loss + tv_loss
        total_loss.backward()

        optimizer.step()

        #clamp
        with torch.no_grad():
            dummy_image.clamp_(-0.4242, 2.8215)

        if iteration % 500 == 0:
            print(f"Iteration {iteration}/{num_iterations} | Loss: {total_loss.item():.4f} | Cos clean: {cos_clean.item():.4f} | Cos star: {cos_star.item():.4f}")

    return dummy_image.detach()



#function that simulates the entire FUIA (in the Client Unlearning scenario)
def run_fuia_client_unlearning(global_model, stored_updates, client_data, private_subset, pretrained_state_dict, target_client, num_clients=50, fraction=0.2, num_rounds=10, local_epochs=3, batch_size=32, lr=0.01):
    print(f"FUIA attack with target client {target_client}\n")

    original_model = copy.deepcopy(global_model)

    print("Retraining without the target client\n")

    unlearned_model, stored_updates_unlearn = retraining(pretrained_state_dict=pretrained_state_dict,
                                                         private_subset=private_subset,
                                                         client_data=client_data,
                                                         target_client=target_client,
                                                         num_clients=num_clients,
                                                         fraction=fraction,
                                                         num_rounds=num_rounds,
                                                         local_epochs=local_epochs,
                                                         batch_size=batch_size,
                                                         lr=lr,)

    print("Retraining completed\n")

    #Step 1
    clean_gradient = gradient_separation(stored_updates, target_client)

    #for debugging
    clean_norm = sum(v.norm().item() for v in clean_gradient.values())
    clean_nan = any(torch.isnan(v).any().item() for v in clean_gradient.values())
    print(f"  Clean gradient norm: {clean_norm:.6f}, contains NaN: {clean_nan}")

    #Step 2
    grad_star = target_gradient_acquisition(original_model, unlearned_model)

    #for debugging
    star_norm = sum(v.norm().item() for v in grad_star.values())
    star_nan = any(torch.isnan(v).any().item() for v in grad_star.values())
    print(f"  Grad star norm: {star_norm:.6f}, contains NaN: {star_nan}")

    #Step 3
    target_indices = client_data[target_client]
    target_label = private_subset[target_indices[0]][1]
    print(f"Target label: {target_label}")
    reconstructed_image = gradient_inversion(
        original_model=original_model,
        clean_gradient=clean_gradient,
        grad_star=grad_star,
        label = target_label,
        gamma=0.0,
        alpha=0.005,
        num_iterations=8000,
        lr=0.1,
    )


    #results
    original_image = private_subset[target_indices[0]][0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original_image.squeeze(), cmap='grey')
    axes[0].set_title("Original image")
    axes[0].axis('off')

    axes[1].imshow(reconstructed_image.squeeze(), cmap='grey')
    axes[1].set_title("Image reconstructed with FUIA")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    #evaluation metrics (MSE, PSNR)
    mse = torch.mean((original_image - reconstructed_image.squeeze())**2).item()

    max_pixel = 1.0
    psnr = 10*torch.log10(max_pixel**2 / torch.tensor(mse)).item()

    print(f"MSE: {mse:.4f}")
    print(f"PSNR: {psnr:.2f} (dB)")

    return reconstructed_image, mse, psnr


#attack execution

#FL
global_model, stored_updates, client_data, private_subset, pretraied_state_dict = run_federated_averaging(num_clients=50, fraction=0.2, num_rounds=10, local_epochs=3, batch_size=32, lr=0.01, iid=True)

#target client
target_client = None
for (round_t, k) in stored_updates.keys():
    target_client = k
    break

#attack
reconstructed_image, mse, psnr = run_fuia_client_unlearning(global_model=global_model,
                                                            stored_updates=stored_updates, 
                                                            client_data=client_data,
                                                            private_subset=private_subset,
                                                            pretrained_state_dict=pretraied_state_dict,
                                                            target_client=target_client,
                                                            num_clients=50,
                                                            )
