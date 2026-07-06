#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import random
import copy
import time
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import wandb


#Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#using Mac MPS as main device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")

#hyperparameters
NUM_CLIENTS      = 50
FRACTION         = 0.2        #20% client participation per round
NUM_ROUNDS       = 10          #paper: "10 training epochs" for MNIST
LOCAL_EPOCHS     = 3
BATCH_SIZE       = 32
FL_LR            = 0.1         #paper: "initial client-side learning rate is set to 0.1"
PRETRAIN_EPOCHS  = 5           #reduced: binary MNIST converges too fast at 50
PRETRAIN_LR      = 0.01        #pre-training LR (0.1 diverges on binary MNIST)
NUM_CLASSES      = 2           #paper: binary classification (digits 0 and 1)
DATA_PER_CLIENT  = 8           #paper Sec VII.A: "set the number of data points per client to 8"

#Gradient inversion (paper Section V.B, Eq. 13)
INV_ITERATIONS   = 5000         #same as before
INV_LR           = 0.1         #reduced from 0.1 to avoid overshooting
INV_ALPHA        = 1e-5         #TV regularization weight (paper Eq. 13)
INV_RESTARTS     = 32            #increased from 3


#model architecture
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.fc1   = nn.Linear(64 * 7 * 7, 512)
        self.fc2   = nn.Linear(512, num_classes)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x, _ = self.pool1(x)               #discard indices, keep grad support
        x = self.relu(self.conv2(x))
        x, _ = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


#MNIST dataset, binary classification (digits 0 and 1 only, paper Sec VI.A.1)
def load_mnist_binary():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
    test_full  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

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

def client_update(model, dataset, indices, epochs, batch_size, lr, device):
    #Local SGD training on a client's data. Returns state_dict on CPU
    model = model.to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=True, num_workers=0)

    total_loss, n_batches = 0.0, 0
    for _ in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

    model.to("cpu")
    return model.state_dict(), len(indices), total_loss / max(n_batches, 1)


def fedavg(global_model, client_results):
    #FedAvg: weighted average of client state dicts
    total_n = sum(n for _, n, _ in client_results)
    agg = OrderedDict()
    for sd, n, _ in client_results:
        w = n / total_n
        for key in sd:
            val = w * sd[key].float()
            agg[key] = val if key not in agg else agg[key] + val
    global_model.load_state_dict(agg)
    return global_model


def evaluate(model, loader, device):
    model.to(device).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += labels.size(0)
    model.to("cpu")
    return correct / total


#Linear decay
def lr_schedule(initial_lr, round_t, total_rounds):
    return initial_lr * (1.0 - (round_t - 1) / total_rounds)


#Pre-training (paper: "80% used to train a pre-trained model")
def pretrain_model(model, dataset, epochs, batch_size, lr, device):
    model.to(device).train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if ep % 10 == 0 or ep == 1:
            print(f"  Pretrain epoch {ep}/{epochs} | Loss: {ep_loss / len(loader):.4f}")

    model.to("cpu")
    return model


#FL round loop
def run_fl_rounds(model, client_data, private_data, round_selections,
                  test_loader=None, log_prefix=None):
    
    stored_updates = {}
    verbose = log_prefix is not None
    t0 = time.time()

    for rnd in range(1, NUM_ROUNDS + 1):
        selected = round_selections[rnd]
        if not selected:
            continue
        lr = lr_schedule(FL_LR, rnd, NUM_ROUNDS)
        global_sd = {k: v.clone() for k, v in model.state_dict().items()}

        results = []
        for k in selected:
            sd, n_k, loss = client_update(copy.deepcopy(model), private_data,
                                          client_data[k], LOCAL_EPOCHS,
                                          BATCH_SIZE, lr, DEVICE)
            results.append((sd, n_k, loss))
            stored_updates[(rnd, k)] = {
                key: sd[key].float() - global_sd[key].float() for key in sd
            }

        model = fedavg(model, results)

        if verbose and (rnd % 10 == 0 or rnd == 1):
            acc = evaluate(model, test_loader, DEVICE)
            avg_loss = np.mean([r[2] for r in results])
            print(f"  Round {rnd:3d}/{NUM_ROUNDS} | LR: {lr:.5f} | "
                  f"Loss: {avg_loss:.4f} | Acc: {acc * 100:.1f}%")
            wandb.log({f"{log_prefix}/round": rnd,
                       f"{log_prefix}/accuracy": acc,
                       f"{log_prefix}/loss": avg_loss,
                       f"{log_prefix}/lr": lr})

    if verbose:
        acc = evaluate(model, test_loader, DEVICE)
        print(f"  Done ({time.time() - t0:.0f}s) | Final accuracy: {acc * 100:.1f}%")
        wandb.log({f"{log_prefix}/final_accuracy": acc,
                   f"{log_prefix}/time_s": time.time() - t0})

    return model, stored_updates


def run_fl_training():
    train_data, test_data = load_mnist_binary()
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    n_total = len(train_data)
    n_pretrain = int(0.8 * n_total)
    pretrain_data, private_data = torch.utils.data.random_split(
        train_data, [n_pretrain, n_total - n_pretrain])

    model = CNN()
    print(f"Pre-training on {n_pretrain} samples ({PRETRAIN_EPOCHS} epochs)...")
    model = pretrain_model(model, pretrain_data, PRETRAIN_EPOCHS, BATCH_SIZE,
                           PRETRAIN_LR, DEVICE)
    acc = evaluate(model, test_loader, DEVICE)
    print(f"  Pretrain accuracy: {acc * 100:.1f}%\n")
    pretrained_sd = {k: v.clone() for k, v in model.state_dict().items()}

    client_data = partition_iid(private_data, NUM_CLIENTS, DATA_PER_CLIENT)
    m = max(1, int(FRACTION * NUM_CLIENTS))
    round_selections = {r: random.sample(range(NUM_CLIENTS), m)
                        for r in range(1, NUM_ROUNDS + 1)}

    print(f"FL Training: {NUM_ROUNDS} rounds, {m} clients/round, "
          f"{DATA_PER_CLIENT} samples/client, LR={FL_LR}")
    model, stored_updates = run_fl_rounds(model, client_data, private_data,
                                          round_selections, test_loader, "fl")

    return (model, stored_updates, client_data, private_data,
            pretrained_sd, round_selections, test_loader)


def select_forgotten_samples(client_data, private_data):
    #Randomly select 1 sample per client to forget (paper Sec VI.B.1)
    forgotten = {}
    for client_id, sample_indices in client_data.items():
        forgotten[client_id] = random.choice(sample_indices)
    return forgotten


def retrain_without_samples(pretrained_sd, private_data, client_data,
                            forgotten_samples, round_selections):
    model = CNN()
    model.load_state_dict(pretrained_sd)
    modified_data = {}
    for client_id, sample_indices in client_data.items():
        if client_id in forgotten_samples:
            modified_data[client_id] = [i for i in sample_indices
                                        if i != forgotten_samples[client_id]]
        else:
            modified_data[client_id] = list(sample_indices)
    return run_fl_rounds(model, modified_data, private_data, round_selections)



#FUIA ATTACK

def gradient_separation(stored_updates, target_client):
    #group all updates by round: {round_id: {client_id: param_update}}
    updates_by_round = {}
    for (round_id, client_id), param_update in stored_updates.items():
        updates_by_round.setdefault(round_id, {})[client_id] = param_update

    #find rounds where target client participated
    rounds_with_target = sorted([round_id for round_id, clients in updates_by_round.items()
                                 if target_client in clients])
    if not rounds_with_target:
        raise ValueError(f"Client {target_client} never participated")

    def l1_norm(param_update):
        #L1 norm of a parameter update vector (Eq. 9)
        return float(sum(v.float().abs().sum() for v in param_update.values()))

    clean_gradient = None
    for round_id in rounds_with_target:
        round_updates = updates_by_round[round_id]

        #Eq. 9: sum of L1 norms of all clients in this round
        l1_norm_sum = sum(l1_norm(update) for update in round_updates.values())

        #Eq. 10: weight for target client in this round
        weight = l1_norm(round_updates[target_client]) / (l1_norm_sum + 1e-12)

        #Eq. 11: accumulate weighted update
        target_update = round_updates[target_client]
        if clean_gradient is None:
            clean_gradient = {key: weight * val.float().clone()
                              for key, val in target_update.items()}
        else:
            for key in clean_gradient:
                clean_gradient[key] += weight * target_update[key].float()

    return clean_gradient


def target_gradient_acquisition(clean_gradient_fl, clean_gradient_fu):
    target_gradient = {key: clean_gradient_fl[key] - clean_gradient_fu[key]
                       for key in clean_gradient_fl}
    return target_gradient


def total_variation(image):
    #Anisotropic total variation (Eq. 15)
    diff_h = (image[:, :, 1:, :] - image[:, :, :-1, :]).pow(2).sum()
    diff_w = (image[:, :, :, 1:] - image[:, :, :, :-1]).pow(2).sum()
    return diff_h + diff_w


def cosine_similarity_gradients(grad_a, grad_b, param_keys):
    #Cosine similarity between two gradient dicts, flattened and concatenated
    flat_a = torch.cat([grad_a[k].flatten() for k in param_keys])
    flat_b = torch.cat([grad_b[k].flatten() for k in param_keys])
    return nn.functional.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0))


def gradient_inversion(original_model, target_gradient, label):
    device = DEVICE
    param_keys = sorted(target_gradient.keys())

    #move target gradient to device and negate
    target_on_device = {k: -v.to(device).detach() for k, v in target_gradient.items()}

    #pixel bounds in normalized space
    pixel_min = (0.0 - 0.1307) / 0.3081
    pixel_max = (1.0 - 0.1307) / 0.3081

    loss_fn = nn.CrossEntropyLoss()
    label_tensor = torch.tensor([label], device=device)

    #use original model W^o for virtual gradient computation (Eq. 14)
    model = copy.deepcopy(original_model).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(True)

    best_image = None
    best_cosine = float('-inf')

    for restart in range(INV_RESTARTS):
        torch.manual_seed(SEED + restart * 7919)
        virtual_image = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([virtual_image], lr=INV_LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=INV_ITERATIONS, eta_min=INV_LR * 0.01)

        restart_best_cosine = float('-inf')
        restart_best_image = virtual_image.detach().clone()

        for iteration in range(1, INV_ITERATIONS + 1):
            optimizer.zero_grad()
            model.zero_grad()

            #compute virtual gradient (Eq. 14)
            output = model(virtual_image)
            classification_loss = loss_fn(output, label_tensor)
            virtual_grads = torch.autograd.grad(
                classification_loss, model.parameters(), create_graph=True)
            virtual_grad_dict = {name: grad for (name, _), grad in
                                 zip(model.named_parameters(), virtual_grads)}

            #Eq. 13
            cos_sim = cosine_similarity_gradients(
                virtual_grad_dict, target_on_device, param_keys)
            inversion_loss = -cos_sim + INV_ALPHA * total_variation(virtual_image)

            inversion_loss.backward()
            optimizer.step()
            scheduler.step()

            #clamp to valid pixel range
            with torch.no_grad():
                virtual_image.clamp_(pixel_min, pixel_max)

            #track best result by cosine similarity
            current_cosine = cos_sim.item()
            if current_cosine > restart_best_cosine:
                restart_best_cosine = current_cosine
                restart_best_image = virtual_image.detach().clone()

            if iteration % 2000 == 0:
                print(f"    [Restart {restart+1}/{INV_RESTARTS}] "
                      f"Iter {iteration:5d}/{INV_ITERATIONS} | "
                      f"loss: {inversion_loss.item():.6f} | "
                      f"cos_sim: {current_cosine:.4f}")

        if restart_best_cosine > best_cosine:
            best_cosine = restart_best_cosine
            best_image = restart_best_image
            print(f"  -> Restart {restart+1}: best cos_sim = {restart_best_cosine:.4f}")

    return best_image


#Metrics and Visualization
def denormalize(image_tensor):
    #Convert from normalized MNIST space back to [0, 1] pixel range
    return (image_tensor.cpu().float() * 0.3081 + 0.1307).clamp(0, 1).squeeze()


def compute_metrics(original_image, reconstructed_image):
    #MSE and PSNR between original and reconstructed images (in [0,1] space)
    original = denormalize(original_image)
    reconstructed = denormalize(reconstructed_image)
    mse = torch.mean((original - reconstructed) ** 2).item()
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))
    return mse, psnr


#FUIA Attack on a single target client (Algorithm 1, one iteration of the loop)
def attack_target_client(original_model, stored_updates_fl, stored_updates_fu,
                         private_data, target_client, forgotten_idx,
                         round_selections):

    target_label = private_data[forgotten_idx][1]
    rounds_with_target = [r for r, cs in round_selections.items()
                          if target_client in cs]

    print(f"\n  Attacking client {target_client}")
    print(f"    Forgotten label:  {target_label}")
    print(f"    Participated in {len(rounds_with_target)}/{NUM_ROUNDS} rounds")

    #Step 1: Gradient Separation (Eq. 8-11)
    print("    [Step 1a] Gradient separation on FL updates -> clean_gradient_fl")
    clean_gradient_fl = gradient_separation(stored_updates_fl, target_client)
    print(f"      L2 norm: {sum(v.norm().item() for v in clean_gradient_fl.values()):.6f}")

    print("    [Step 1b] Gradient separation on FU updates -> clean_gradient_fu")
    clean_gradient_fu = gradient_separation(stored_updates_fu, target_client)
    print(f"      L2 norm: {sum(v.norm().item() for v in clean_gradient_fu.values()):.6f}")

    #Step 2: Target Gradient Acquisition (Eq. 12)
    print("    [Step 2] Target gradient: nabla_k = clean_gradient_fl - clean_gradient_fu")
    nabla_k = target_gradient_acquisition(clean_gradient_fl, clean_gradient_fu)
    nabla_k_norm = sum(v.norm().item() for v in nabla_k.values())
    print(f"      L2 norm: {nabla_k_norm:.6f}")

    #Step 3: Gradient Inversion (Eq. 13-14)
    print(f"    [Step 3] Gradient inversion using W^o "
          f"({INV_RESTARTS} restarts x {INV_ITERATIONS} iters)...")
    inversion_start = time.time()
    reconstructed_image = gradient_inversion(original_model, nabla_k, target_label)
    inversion_time = time.time() - inversion_start
    print(f"    Inversion done ({inversion_time:.0f}s)")

    #Compute metrics
    original_image = private_data[forgotten_idx][0]
    mse, psnr = compute_metrics(original_image, reconstructed_image.squeeze(0))
    print(f"    MSE: {mse:.4f} | PSNR: {psnr:.2f} dB")

    return reconstructed_image, mse, psnr


def select_target_client(stored_updates, round_selections):
    #Pick the participating client with the most rounds (strongest signal)
    round_counts = {}
    for (round_id, client_id) in stored_updates:
        round_counts[client_id] = round_counts.get(client_id, 0) + 1
    best_client = max(round_counts, key=round_counts.get)
    return best_client


#main
if __name__ == "__main__":
    config = {k: v for k, v in globals().items()
              if k.isupper() and isinstance(v, (int, float, str))}
    config.update({"scenario": "sample_unlearning",
                   "unlearning_method": "retraining", "dataset": "MNIST_binary"})
    wandb.init(project="FUIA", config=config)

    total_start = time.time()

    #Phase 1: Federated Learning
    print("=" * 60 + "\nPhase 1: Federated Learning\n" + "=" * 60)
    (original_model, stored_updates_fl, client_data, private_data,
     pretrained_sd, round_selections, test_loader) = run_fl_training()

    #Phase 2: Unlearning — select 1 sample per client, retrain without ALL of them
    print("\n" + "=" * 60 + "\nPhase 2: Sample Unlearning (Retraining)\n" + "=" * 60)
    forgotten_samples = select_forgotten_samples(client_data, private_data)
    print(f"  Selected 1 forgotten sample per client ({len(forgotten_samples)} total)")

    retrain_start = time.time()
    unlearned_model, stored_updates_fu = retrain_without_samples(
        pretrained_sd, private_data, client_data,
        forgotten_samples, round_selections)
    retrain_time = time.time() - retrain_start

    original_acc = evaluate(original_model, test_loader, DEVICE)
    unlearned_acc = evaluate(unlearned_model, test_loader, DEVICE)
    print(f"  Retraining done ({retrain_time:.0f}s)")
    print(f"  Original model accuracy:  {original_acc * 100:.1f}%")
    print(f"  Unlearned model accuracy: {unlearned_acc * 100:.1f}%")

    #Phase 3: FUIA Attack — attack one target client
    print("\n" + "=" * 60 + "\nPhase 3: FUIA Attack on Target Client\n" + "=" * 60)
    target_client = select_target_client(stored_updates_fl, round_selections)
    forgotten_idx = forgotten_samples[target_client]
    target_label = private_data[forgotten_idx][1]
    print(f"  Target client: {target_client} "
          f"(label={target_label}, index={forgotten_idx})")

    reconstructed, mse, psnr = attack_target_client(
        original_model, stored_updates_fl, stored_updates_fu,
        private_data, target_client, forgotten_idx, round_selections)

    #Results
    print(f"\n{'=' * 60}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  (Paper Fig. 5 reference for MNIST Retrain: MSE ~0.0004, PSNR ~34 dB)")
    print(f"{'=' * 60}")

    #Visualization
    original_image = private_data[forgotten_idx][0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(denormalize(original_image).numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original (label={target_label})", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(denormalize(reconstructed.squeeze(0)).numpy(),
                   cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"FUIA Reconstruction\nMSE={mse:.4f}  PSNR={psnr:.2f} dB",
                      fontsize=14)
    axes[1].axis("off")
    plt.suptitle("FUIA Sample Unlearning — MNIST (Retraining)", fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "fuia_sample_result.png")
    plt.savefig(save_path, dpi=150)
    wandb.log({"attack/result": wandb.Image(fig),
               "attack/mse": mse, "attack/psnr": psnr,
               "attack/target_client": target_client,
               "attack/target_label": target_label})
    print(f"  Saved to {save_path}")
    plt.close(fig)

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    wandb.log({"total_time_s": total_time})
    wandb.finish()
