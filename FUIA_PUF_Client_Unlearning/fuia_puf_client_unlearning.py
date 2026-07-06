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


#seed set for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#Using mainly Mac for training
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")

#Hyperparameters following the setting of the paper
#FL training (paper Section VI)
NUM_CLIENTS      = 50
FRACTION         = 0.2        #20% client participation per round
NUM_ROUNDS       = 80         #paper: "80 training epochs"
LOCAL_EPOCHS     = 3
BATCH_SIZE       = 32
FL_LR            = 0.01       #paper: "learning rate is set to 0.01"
PRETRAIN_EPOCHS  = 5          #the paper specify 50
PRETRAIN_LR      = 0.01
NUM_CLASSES      = 10         #full MNIST (instead of only classes 0 and 1)
DATA_PER_CLIENT  = 1          #paper Sec VI.B: "set the number of data points per client to 1"

#Gradient inversion (paper Section V.B & VII.B; client-unlearning loss = Eq. 18)
INV_ITERATIONS   = 20000
INV_LR           = 0.1
INV_GAMMA        = 0.1        #paper-faithful: weight of Psi term in Eq. 18
INV_ALPHA        = 1e-5       #minimal TV: allow fine detail
INV_RESTARTS     = 3

#PUF-Special hyperparameters
ETA_U_VALUES         = [1.0, 1.5, 2.0, 3.0]
PUF_DEFAULT_ETA_U    = 2.0    # eta_u used for the FUIA gradient-inversion run
PUF_LOCAL_EPOCHS     = LOCAL_EPOCHS
PUF_UNLEARN_LR       = FL_LR


#Model architecture
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
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
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)



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



def client_update(model, dataset, indices, epochs, batch_size, lr, device):
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


#FedAvg
def fedavg(global_model, client_results):
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

#learning rate linear decay schedule
def lr_schedule(initial_lr, round_t, total_rounds):
    return initial_lr * (1.0 - (round_t - 1) / total_rounds)


#Pretraining
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


#FL training: stores per-round client selections and parameter updates
def run_fl_training():
    train_data, test_data = load_mnist()
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    #80% - 20% split
    n_total = len(train_data)
    n_pretrain = int(0.8 * n_total)
    pretrain_data, private_data = torch.utils.data.random_split(
        train_data, [n_pretrain, n_total - n_pretrain])

    #Pre-train
    model = CNN()
    print(f"Pre-training on {n_pretrain} samples ({PRETRAIN_EPOCHS} epochs)...")
    t0 = time.time()
    model = pretrain_model(model, pretrain_data, PRETRAIN_EPOCHS, BATCH_SIZE, PRETRAIN_LR, DEVICE)
    acc = evaluate(model, test_loader, DEVICE)
    print(f"  Done ({time.time() - t0:.0f}s) | Accuracy: {acc * 100:.1f}%\n")
    pretrained_sd = copy.deepcopy(model.state_dict())

    #Partition: 1 sample per client
    client_data = partition_iid(private_data, NUM_CLIENTS, DATA_PER_CLIENT)

    #FL rounds
    m = max(1, int(FRACTION * NUM_CLIENTS))
    stored_updates = {}   #(round, client) -> param_diff dict
    round_selections = {} #round -> list of selected clients

    print(f"FL Training: {NUM_ROUNDS} rounds, {NUM_CLIENTS} clients, "
          f"{m} selected/round, LR={FL_LR}")
    t0 = time.time()

    for rnd in range(1, NUM_ROUNDS + 1):
        selected = random.sample(range(NUM_CLIENTS), m)
        round_selections[rnd] = selected
        global_sd = copy.deepcopy(model.state_dict())
        lr = lr_schedule(FL_LR, rnd, NUM_ROUNDS)

        results = []
        for k in selected:
            local = copy.deepcopy(model)
            sd, n_k, loss = client_update(local, private_data, client_data[k],
                                          LOCAL_EPOCHS, BATCH_SIZE, lr, DEVICE)
            results.append((sd, n_k, loss))
            stored_updates[(rnd, k)] = {
                key: sd[key].float() - global_sd[key].float() for key in sd
            }

        model = fedavg(model, results)

        if rnd % 10 == 0 or rnd == 1:
            acc = evaluate(model, test_loader, DEVICE)
            avg_loss = np.mean([r[2] for r in results])
            print(f"  Round {rnd:3d}/{NUM_ROUNDS} | LR: {lr:.5f} | "
                  f"Loss: {avg_loss:.4f} | Acc: {acc * 100:.1f}%")
            wandb.log({"fl/round": rnd, "fl/accuracy": acc,
                       "fl/loss": avg_loss, "fl/lr": lr})

    fl_time = time.time() - t0
    final_acc = evaluate(model, test_loader, DEVICE)
    print(f"  FL done ({fl_time:.0f}s) | Final accuracy: {final_acc * 100:.1f}%")
    wandb.log({"fl/final_accuracy": final_acc, "fl/time_s": fl_time})

    return (model, stored_updates, client_data, private_data,
            pretrained_sd, round_selections, test_loader)


#Retraining (deterministic): replays same round selections minus target
def retrain_without_client(pretrained_sd, private_data, client_data,
                           target, round_selections):
    model = CNN()
    model.load_state_dict(pretrained_sd)

    for rnd in range(1, NUM_ROUNDS + 1):
        selected = [c for c in round_selections[rnd] if c != target]
        if not selected:
            continue
        lr = lr_schedule(FL_LR, rnd, NUM_ROUNDS)
        results = []
        for k in selected:
            local = copy.deepcopy(model)
            sd, n_k, loss = client_update(local, private_data, client_data[k],
                                          LOCAL_EPOCHS, BATCH_SIZE, lr, DEVICE)
            results.append((sd, n_k, loss))
        model = fedavg(model, results)

    return model


#PUF-Special unlearning (paper Algorithm 1 with S_t^+ = empty, Eq. 9).
def puf_special_unlearn(global_model, private_dataset, client_data_indices,
                        target_client, eta_u,
                        local_epochs, batch_size, learning_rate, device):
    global_state_dict = copy.deepcopy(global_model.state_dict())

    #Target client performs one round of regular local training (ClientOpt).
    target_local_model = copy.deepcopy(global_model)
    target_state_dict, _, _ = client_update(
        target_local_model,
        private_dataset,
        client_data_indices[target_client],
        local_epochs,
        batch_size,
        learning_rate,
        device,
    )

    #Pseudo-gradient (target client's model update on w_t).
    pseudo_gradient = {
        parameter_name: target_state_dict[parameter_name].float()
                        - global_state_dict[parameter_name].float()
        for parameter_name in target_state_dict
    }

    #Apply scaled negation: w_unlearned = w_t - eta_u * pseudo_gradient
    unlearned_state_dict = OrderedDict()
    for parameter_name in global_state_dict:
        unlearned_state_dict[parameter_name] = (
            global_state_dict[parameter_name].float()
            - eta_u * pseudo_gradient[parameter_name]
        )

    unlearned_model = CNN()
    unlearned_model.load_state_dict(unlearned_state_dict)
    return unlearned_model


#Forget Accuracy: accuracy on the target client's data (the data being forgotten)
def forget_accuracy(model, private_dataset, target_indices, device):
    forget_loader = DataLoader(
        Subset(private_dataset, target_indices),
        batch_size=64,
        shuffle=False,
    )
    return evaluate(model, forget_loader, device)


#FUIA Step 1: Gradient Separation (paper Eq. 16)
def gradient_separation(stored_updates, target_client):
    target_rounds = sorted({t for (t, k) in stored_updates if k == target_client})
    if not target_rounds:
        raise ValueError(f"Client {target_client} never participated")

    clean_grad = None
    for t in target_rounds:
        #All clients that participated in round t
        round_keys = [(rt, k) for (rt, k) in stored_updates if rt == t]
        total_l1 = sum(
            sum(v.abs().sum().item() for v in stored_updates[(t, k)].values())
            for (_, k) in round_keys
        )
        target_l1 = sum(
            v.abs().sum().item() for v in stored_updates[(t, target_client)].values()
        )
        gamma_t = target_l1 / (total_l1 + 1e-12)

        if clean_grad is None:
            clean_grad = {key: gamma_t * v.clone()
                          for key, v in stored_updates[(t, target_client)].items()}
        else:
            for key in clean_grad:
                clean_grad[key] += gamma_t * stored_updates[(t, target_client)][key]

    return clean_grad


#FUIA Step 2: Target Gradient Acquisition (paper Eq. 17)
def target_gradient_acquisition(original_model, unlearned_model):
    #Psi = W_original - W_unlearned
    wo = original_model.state_dict()
    wu = unlearned_model.state_dict()
    return {k: wo[k].float() - wu[k].float() for k in wo}


#FUIA Step 3: Gradient Inversion (paper Section V.B)
def total_variation(x):
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).sum()
    return dh + dw


def per_layer_cosine_distance(ga, gb, keys):
    #Per-layer cosine distance (from Geiping et al. "Inverting Gradients")
    total = torch.tensor(0.0, device=ga[keys[0]].device)
    for k in keys:
        a = ga[k].flatten().unsqueeze(0)
        b = gb[k].flatten().unsqueeze(0)
        cos = nn.functional.cosine_similarity(a, b)
        total = total + (1.0 - cos)
    return total / len(keys)


def gradient_inversion(model_for_inversion, clean_grad, target_grad, label,
                       gamma=INV_GAMMA, alpha=INV_ALPHA,
                       n_iters=INV_ITERATIONS, lr=INV_LR,
                       n_restarts=INV_RESTARTS):

    #Force CPU for 2nd-order gradient support with MaxPool2d
    device = torch.device("cpu")

    #Both V_k and Psi are PARAMETER-UPDATE directions, not gradient directions:
    #  * V_k = sum of target client's per-round updates (~ -lr * grad).
    #  * Psi = W_orig - W_unlearned: target's training pushed W_orig in the
    #    -grad direction relative to W_unlearned, so Psi ~ -eps * grad.
    #We negate both so that the cosine-distance loss aligns the dummy
    #image's gradient with +grad (the true gradient at W_original on the
    #forgotten sample).
    clean_d  = {k: -v.to(device).detach() for k, v in clean_grad.items()}
    target_d = {k: -v.to(device).detach() for k, v in target_grad.items()}
    keys = sorted(clean_d.keys())

    #Valid pixel range after MNIST normalization
    norm_min = (0.0 - 0.1307) / 0.3081   # ~ -0.4242
    norm_max = (1.0 - 0.1307) / 0.3081   # ~  2.8215

    loss_fn = nn.CrossEntropyLoss()
    dummy_label = torch.tensor([label], device=device)

    best_img = None
    best_loss = float('inf')

    for r in range(n_restarts):
        torch.manual_seed(SEED + r * 7919)
        x = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)
        opt = torch.optim.Adam([x], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_iters, eta_min=lr * 0.01)

        model = copy.deepcopy(model_for_inversion).to(device).eval()

        local_best_loss = float('inf')
        local_best_img = x.detach().clone()

        for it in range(1, n_iters + 1):
            opt.zero_grad()

            #Enable param grads for create_graph
            for p in model.parameters():
                p.requires_grad_(True)
            model.zero_grad()

            out = model(x)
            loss = loss_fn(out, dummy_label)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            g_dict = {}
            for (name, _), g in zip(model.named_parameters(), grads):
                g_dict[name] = g

            #Per-layer cosine distance (each layer contributes equally)
            dist_c = per_layer_cosine_distance(g_dict, clean_d, keys)
            dist_t = per_layer_cosine_distance(g_dict, target_d, keys)

            inv_loss = (1 - gamma) * dist_c + gamma * dist_t + alpha * total_variation(x)

            #Disable param grads before backward to save compute
            for p in model.parameters():
                p.requires_grad_(False)

            inv_loss.backward()
            opt.step()
            sched.step()

            with torch.no_grad():
                x.clamp_(norm_min, norm_max)

            l = inv_loss.item()
            if l < local_best_loss:
                local_best_loss = l
                local_best_img = x.detach().clone()

            if it % 1000 == 0:
                avg_cos_c = 1.0 - dist_c.item()
                avg_cos_t = 1.0 - dist_t.item()
                print(f"    [Restart {r+1}/{n_restarts}] Iter {it:5d}/{n_iters} | "
                      f"Loss: {l:.4f} | avg_cos_clean: {avg_cos_c:.4f} | "
                      f"avg_cos_target: {avg_cos_t:.4f}")

        if local_best_loss < best_loss:
            best_loss = local_best_loss
            best_img = local_best_img
            print(f"  -> Restart {r+1}: best loss = {local_best_loss:.4f}")

    return best_img


#Metrics
def denormalize(img, mean=0.1307, std=0.3081):
    return (img * std + mean).clamp(0, 1)


def compute_metrics(original, reconstructed):
    #MSE and PSNR on denormalized [0,1] images
    orig = denormalize(original)
    recon = denormalize(reconstructed)
    mse = torch.mean((orig - recon) ** 2).item()
    psnr = 10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-10))).item()
    return mse, psnr


#FUIA attack orchestration: takes an already-unlearned model from any
#unlearning method (retraining, PUF-Special, ...) and runs the three FUIA
#steps (Gradient Separation, Target Gradient Acquisition, Gradient Inversion).
def run_fuia_attack(original_model, unlearned_model, stored_updates,
                    target_client, target_indices, target_label,
                    target_rounds, private_dataset, test_loader,
                    unlearning_method_label, save_path):

    print(f"\n{'=' * 60}")
    print(f"FUIA Attack -- unlearning method: {unlearning_method_label}")
    print(f"  Target client: {target_client}")
    print(f"  Target label:  {target_label}")
    print(f"  Participated in {len(target_rounds)}/{NUM_ROUNDS} rounds")
    print(f"{'=' * 60}")

    original_test_accuracy   = evaluate(original_model,   test_loader, DEVICE)
    unlearned_test_accuracy  = evaluate(unlearned_model,  test_loader, DEVICE)
    print(f"  Original  model Test Acc: {original_test_accuracy  * 100:.1f}%")
    print(f"  Unlearned model Test Acc: {unlearned_test_accuracy * 100:.1f}%")

    #Step 1: Gradient Separation
    print("\n[Step 1] Gradient separation (Eq. 16)...")
    clean_gradient = gradient_separation(stored_updates, target_client)
    clean_gradient_l2_norm = sum(v.norm().item() for v in clean_gradient.values())
    print(f"  Clean gradient L2 norm: {clean_gradient_l2_norm:.6f}")

    #Step 2: Target Gradient Acquisition
    print("\n[Step 2] Target gradient acquisition (Psi = W_orig - W_unlearned)...")
    target_gradient = target_gradient_acquisition(original_model, unlearned_model)
    target_gradient_l2_norm = sum(v.norm().item() for v in target_gradient.values())
    print(f"  Target gradient L2 norm: {target_gradient_l2_norm:.6f}")

    #Diagnostic: global cosine sim between V_k and Psi
    parameter_names = sorted(clean_gradient.keys())
    clean_gradient_flat  = torch.cat([clean_gradient[name].flatten()
                                      for name in parameter_names])
    target_gradient_flat = torch.cat([target_gradient[name].flatten()
                                      for name in parameter_names])
    clean_target_cosine_similarity = nn.functional.cosine_similarity(
        clean_gradient_flat.unsqueeze(0),
        target_gradient_flat.unsqueeze(0),
    ).item()
    print(f"  Cosine sim(clean, target): {clean_target_cosine_similarity:.4f}")

    #Diagnostic: per-layer cosine of V_k and Psi vs the true gradient at the
    #target sample, evaluated at both W_original and W_unlearned. Theory
    #(verified empirically) says both should be NEGATIVE -- V_k and Psi are
    #parameter-update directions, opposite to the gradient. The inversion
    #negates them internally so the dummy image's gradient ends up aligned
    #with +grad.
    print("\n[Diagnostic] Gradient alignment with true data:")
    target_image_tensor      = private_dataset[target_indices[0]][0].unsqueeze(0)
    target_label_tensor      = torch.tensor([target_label])
    diagnostic_loss_function = nn.CrossEntropyLoss()

    for diagnostic_model_name, diagnostic_model in [
        ("W_original",  original_model),
        ("W_unlearned", unlearned_model),
    ]:
        diagnostic_model_cpu = copy.deepcopy(diagnostic_model).to("cpu").eval()
        for parameter in diagnostic_model_cpu.parameters():
            parameter.requires_grad_(True)
        diagnostic_model_cpu.zero_grad()
        diagnostic_output    = diagnostic_model_cpu(target_image_tensor)
        diagnostic_loss      = diagnostic_loss_function(
            diagnostic_output, target_label_tensor)
        diagnostic_gradients = torch.autograd.grad(
            diagnostic_loss, diagnostic_model_cpu.parameters())
        true_gradient = {
            parameter_name: gradient_tensor.detach()
            for (parameter_name, _), gradient_tensor
            in zip(diagnostic_model_cpu.named_parameters(), diagnostic_gradients)
        }

        per_layer_clean_cosines = [
            nn.functional.cosine_similarity(
                true_gradient[name].flatten().unsqueeze(0),
                clean_gradient[name].flatten().unsqueeze(0),
            ).item()
            for name in parameter_names
        ]
        per_layer_target_cosines = [
            nn.functional.cosine_similarity(
                true_gradient[name].flatten().unsqueeze(0),
                target_gradient[name].flatten().unsqueeze(0),
            ).item()
            for name in parameter_names
        ]

        average_clean_cosine  = np.mean(per_layer_clean_cosines)
        average_target_cosine = np.mean(per_layer_target_cosines)
        print(f"  {diagnostic_model_name}: "
              f"avg_cos(true_grad, V_k)={average_clean_cosine:.4f}  "
              f"avg_cos(true_grad, Psi)={average_target_cosine:.4f}")
        for layer_index, parameter_name in enumerate(parameter_names):
            print(f"    {parameter_name:30s}: "
                  f"cos(V_k)={per_layer_clean_cosines[layer_index]:.4f}  "
                  f"cos(Psi)={per_layer_target_cosines[layer_index]:.4f}")

    #Step 3: Gradient Inversion (paper Eq. 18 with gamma=INV_GAMMA).
    print(f"\n[Step 3] Gradient inversion using W_original "
          f"({INV_RESTARTS} restart(s) x {INV_ITERATIONS} iters on CPU)...")
    inversion_start_time = time.time()
    reconstructed_image  = gradient_inversion(
        original_model, clean_gradient, target_gradient, target_label)
    inversion_time_seconds = time.time() - inversion_start_time
    print(f"  Inversion done ({inversion_time_seconds:.0f}s)")

    #Reconstruction quality
    target_image                                = private_dataset[target_indices[0]][0]
    reconstruction_mse, reconstruction_psnr_db  = compute_metrics(
        target_image, reconstructed_image.squeeze(0))

    print(f"\n{'-' * 60}")
    print(f"  MSE:  {reconstruction_mse:.6f}")
    print(f"  PSNR: {reconstruction_psnr_db:.2f} dB")
    print(f"  (Paper reference: MSE ~ 0.0005, PSNR ~ 33 dB)")
    print(f"{'-' * 60}")

    wandb.log({
        "attack/mse": reconstruction_mse,
        "attack/psnr_db": reconstruction_psnr_db,
        "attack/inversion_time_seconds": inversion_time_seconds,
        "attack/target_client": target_client,
        "attack/target_label": target_label,
        "attack/n_target_rounds": len(target_rounds),
        "attack/clean_gradient_l2_norm": clean_gradient_l2_norm,
        "attack/target_gradient_l2_norm": target_gradient_l2_norm,
        "attack/clean_target_cosine_similarity": clean_target_cosine_similarity,
        "attack/original_test_accuracy": original_test_accuracy,
        "attack/unlearned_test_accuracy": unlearned_test_accuracy,
    })

    #Visualization
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    target_image_denormalized        = denormalize(target_image).squeeze().numpy()
    reconstructed_image_denormalized = denormalize(
        reconstructed_image.squeeze(0)).squeeze().numpy()

    axes[0].imshow(target_image_denormalized, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original (label={target_label})", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(reconstructed_image_denormalized, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"FUIA Reconstruction\n"
        f"MSE={reconstruction_mse:.6f}  PSNR={reconstruction_psnr_db:.2f} dB",
        fontsize=14)
    axes[1].axis("off")
    plt.suptitle(
        f"FUIA Client Unlearning Attack -- MNIST + {unlearning_method_label}",
        fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    wandb.log({"attack/result": wandb.Image(figure)})
    print(f"  Saved to {save_path}")
    plt.show()

    return reconstructed_image, reconstruction_mse, reconstruction_psnr_db


#PUF-Special eta_u sweep: for each eta_u, apply PUF-Special unlearning and
#evaluate Test Accuracy + Forget Accuracy on the resulting unlearned model.
def run_eta_u_sweep(original_model, private_dataset, client_data_indices,
                    target_client, test_loader):
    target_indices  = client_data_indices[target_client]
    sweep_results   = []
    unlearned_model_at_default_eta_u = None

    print(f"\n{'-' * 60}")
    print(f"PUF-Special eta_u sweep (target client {target_client})")
    print(f"  values         : {ETA_U_VALUES}")
    print(f"  default eta_u  : {PUF_DEFAULT_ETA_U} (used for the FUIA attack)")
    print(f"{'-' * 60}")

    for eta_u in ETA_U_VALUES:
        unlearning_start_time = time.time()
        unlearned_model = puf_special_unlearn(
            original_model,
            private_dataset,
            client_data_indices,
            target_client,
            eta_u,
            PUF_LOCAL_EPOCHS,
            BATCH_SIZE,
            PUF_UNLEARN_LR,
            DEVICE,
        )
        unlearning_time_seconds = time.time() - unlearning_start_time

        unlearned_test_accuracy   = evaluate(unlearned_model, test_loader, DEVICE)
        unlearned_forget_accuracy = forget_accuracy(
            unlearned_model, private_dataset, target_indices, DEVICE)

        print(f"  eta_u = {eta_u:>4}: "
              f"Test Acc = {unlearned_test_accuracy   * 100:6.2f}% | "
              f"Forget Acc = {unlearned_forget_accuracy * 100:6.2f}% | "
              f"unlearn time = {unlearning_time_seconds:.2f}s")

        sweep_results.append({
            "eta_u":                    eta_u,
            "test_accuracy":            unlearned_test_accuracy,
            "forget_accuracy":          unlearned_forget_accuracy,
            "unlearning_time_seconds":  unlearning_time_seconds,
        })

        wandb.log({
            "puf/eta_u":                   eta_u,
            "puf/test_accuracy":           unlearned_test_accuracy,
            "puf/forget_accuracy":         unlearned_forget_accuracy,
            "puf/unlearning_time_seconds": unlearning_time_seconds,
        })

        if eta_u == PUF_DEFAULT_ETA_U:
            unlearned_model_at_default_eta_u = unlearned_model

    if unlearned_model_at_default_eta_u is None:
        unlearned_model_at_default_eta_u = puf_special_unlearn(
            original_model, private_dataset, client_data_indices, target_client,
            PUF_DEFAULT_ETA_U, PUF_LOCAL_EPOCHS, BATCH_SIZE,
            PUF_UNLEARN_LR, DEVICE)

    return sweep_results, unlearned_model_at_default_eta_u


#Pick a target client that actually participated in FL training
def pick_target_client(stored_updates):
    for (round_index, client_index) in sorted(stored_updates.keys()):
        return client_index
    raise RuntimeError("No client ever participated in FL training.")


#Final summary table + eta_u sweep plot
def print_summary_table(original_test_accuracy, original_forget_accuracy,
                        retrain_test_accuracy, retrain_forget_accuracy,
                        retraining_time_seconds,
                        sweep_results,
                        fuia_reconstruction_mse, fuia_reconstruction_psnr_db,
                        fuia_unlearning_method_label):
    print(f"\n{'=' * 78}")
    print("FINAL RESULTS  --  FUIA + PUF-Special on MNIST")
    print(f"{'=' * 78}")
    print(f"{'Method':<40} {'Test Acc':>12} {'Forget Acc':>14}")
    print(f"{'-' * 78}")
    print(f"{'Original (no unlearning)':<40} "
          f"{original_test_accuracy  * 100:>11.2f}% "
          f"{original_forget_accuracy * 100:>13.2f}%")
    if retrain_test_accuracy is not None:
        retraining_label = (
            f"Retraining (gold,  {retraining_time_seconds:.0f}s)"
            if retraining_time_seconds is not None
            else "Retraining (gold)"
        )
        print(f"{retraining_label:<40} "
              f"{retrain_test_accuracy  * 100:>11.2f}% "
              f"{retrain_forget_accuracy * 100:>13.2f}%")
    print(f"{'-' * 78}")
    for sweep_entry in sweep_results:
        method_label = (
            f"PUF-Special  eta_u={sweep_entry['eta_u']}  "
            f"({sweep_entry['unlearning_time_seconds']:.2f}s)"
        )
        print(f"{method_label:<40} "
              f"{sweep_entry['test_accuracy']    * 100:>11.2f}% "
              f"{sweep_entry['forget_accuracy']  * 100:>13.2f}%")
    print(f"{'=' * 78}")
    print(f"FUIA reconstruction  ({fuia_unlearning_method_label})")
    print(f"  MSE  = {fuia_reconstruction_mse:.6f}")
    print(f"  PSNR = {fuia_reconstruction_psnr_db:.2f} dB")
    print(f"{'=' * 78}\n")


def plot_eta_u_sweep(sweep_results,
                     original_test_accuracy, original_forget_accuracy,
                     retrain_test_accuracy, retrain_forget_accuracy,
                     save_path):
    eta_u_values         = [entry["eta_u"]           for entry in sweep_results]
    test_accuracies      = [entry["test_accuracy"]   for entry in sweep_results]
    forget_accuracies    = [entry["forget_accuracy"] for entry in sweep_results]

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    test_accuracy_axis = axes[0]
    test_accuracy_axis.plot(eta_u_values, test_accuracies, "o-",
                            label="PUF-Special", linewidth=2, markersize=8)
    test_accuracy_axis.axhline(
        y=original_test_accuracy, color="green", linestyle=":",
        label=f"Original ({original_test_accuracy * 100:.1f}%)")
    if retrain_test_accuracy is not None:
        test_accuracy_axis.axhline(
            y=retrain_test_accuracy, color="orange", linestyle=":",
            label=f"Retraining ({retrain_test_accuracy * 100:.1f}%)")
    test_accuracy_axis.set_xlabel(r"$\eta_u$ (unlearning rate)")
    test_accuracy_axis.set_ylabel("Test Accuracy")
    test_accuracy_axis.set_title(r"Test Accuracy vs $\eta_u$")
    test_accuracy_axis.legend(loc="best")
    test_accuracy_axis.grid(True, alpha=0.3)

    forget_accuracy_axis = axes[1]
    forget_accuracy_axis.plot(eta_u_values, forget_accuracies, "o-",
                              label="PUF-Special", linewidth=2, markersize=8)
    forget_accuracy_axis.axhline(
        y=original_forget_accuracy, color="green", linestyle=":",
        label=f"Original ({original_forget_accuracy * 100:.1f}%)")
    if retrain_forget_accuracy is not None:
        forget_accuracy_axis.axhline(
            y=retrain_forget_accuracy, color="orange", linestyle=":",
            label=f"Retraining ({retrain_forget_accuracy * 100:.1f}%)")
    forget_accuracy_axis.set_xlabel(r"$\eta_u$ (unlearning rate)")
    forget_accuracy_axis.set_ylabel("Forget Accuracy (target client data)")
    forget_accuracy_axis.set_title(r"Forget Accuracy vs $\eta_u$")
    forget_accuracy_axis.legend(loc="best")
    forget_accuracy_axis.grid(True, alpha=0.3)

    plt.suptitle("PUF-Special Client Unlearning -- MNIST", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    wandb.log({"results/eta_u_sweep_plot": wandb.Image(figure)})
    print(f"Saved eta_u sweep plot to {save_path}")
    plt.show()


#Attack execution
if __name__ == "__main__":
    wandb.init(project="FUIA", config={
        "scenario": "client_unlearning",
        "dataset": "MNIST",
        "unlearning_method": "PUF-Special",
        "num_clients": NUM_CLIENTS,
        "fraction": FRACTION,
        "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "fl_lr": FL_LR,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "pretrain_lr": PRETRAIN_LR,
        "num_classes": NUM_CLASSES,
        "data_per_client": DATA_PER_CLIENT,
        "inv_iterations": INV_ITERATIONS,
        "inv_lr": INV_LR,
        "inv_gamma": INV_GAMMA,
        "inv_alpha": INV_ALPHA,
        "inv_restarts": INV_RESTARTS,
        "eta_u_values": ETA_U_VALUES,
        "puf_default_eta_u": PUF_DEFAULT_ETA_U,
        "puf_local_epochs": PUF_LOCAL_EPOCHS,
        "puf_unlearn_lr": PUF_UNLEARN_LR,
        "seed": SEED,
    })

    total_start_time = time.time()
    output_directory = os.path.dirname(os.path.abspath(__file__))

    #Phase 1: Federated Learning
    print("-" * 60)
    print("Phase 1: Federated Learning")
    print("-" * 60)
    (original_model, stored_updates, client_data_indices, private_dataset,
     pretrained_state_dict, round_selections, test_loader) = run_fl_training()

    target_client    = pick_target_client(stored_updates)
    target_indices   = client_data_indices[target_client]
    target_label     = private_dataset[target_indices[0]][1]
    target_rounds    = [round_index for round_index, selected_clients
                        in round_selections.items()
                        if target_client in selected_clients]

    #Phase 2: Baselines (original model + retraining gold standard)
    print("\n" + "-" * 60)
    print("Phase 2: Baselines (original model + retraining)")
    print("-" * 60)
    original_test_accuracy   = evaluate(original_model, test_loader, DEVICE)
    original_forget_accuracy = forget_accuracy(
        original_model, private_dataset, target_indices, DEVICE)
    print(f"[Original (no unlearning)]")
    print(f"  Test Accuracy   : {original_test_accuracy   * 100:.2f}%")
    print(f"  Forget Accuracy : {original_forget_accuracy * 100:.2f}%")
    wandb.log({
        "baseline/original_test_accuracy":   original_test_accuracy,
        "baseline/original_forget_accuracy": original_forget_accuracy,
    })

    print(f"\n[Retraining (gold standard)] -- this can take a while")
    retraining_start_time = time.time()
    retrained_model = retrain_without_client(
        pretrained_state_dict, private_dataset, client_data_indices,
        target_client, round_selections)
    retraining_time_seconds  = time.time() - retraining_start_time
    retrain_test_accuracy    = evaluate(retrained_model, test_loader, DEVICE)
    retrain_forget_accuracy  = forget_accuracy(
        retrained_model, private_dataset, target_indices, DEVICE)
    print(f"  Test Accuracy   : {retrain_test_accuracy   * 100:.2f}%")
    print(f"  Forget Accuracy : {retrain_forget_accuracy * 100:.2f}%")
    print(f"  Time            : {retraining_time_seconds:.0f}s")
    wandb.log({
        "baseline/retrain_test_accuracy":    retrain_test_accuracy,
        "baseline/retrain_forget_accuracy":  retrain_forget_accuracy,
        "baseline/retrain_time_seconds":     retraining_time_seconds,
    })

    #Phase 3: PUF-Special unlearning -- eta_u sweep
    print("\n" + "-" * 60)
    print("Phase 3: PUF-Special unlearning (eta_u sweep)")
    print("-" * 60)
    sweep_results, unlearned_model_at_default_eta_u = run_eta_u_sweep(
        original_model, private_dataset, client_data_indices,
        target_client, test_loader)

    #Phase 4: FUIA attack on PUF-Special's unlearned model (default eta_u)
    #With the cosine-based inversion loss, the reconstruction depends only
    #on the direction of Psi (and V_k), not its magnitude. Different
    #eta_u values scale Psi but do not change its direction, so a single
    #FUIA run at eta_u = PUF_DEFAULT_ETA_U is representative of the whole
    #sweep for purposes of MSE/PSNR
    print("\n" + "-" * 60)
    print(f"Phase 4: FUIA Attack on PUF-Special unlearned model "
          f"(eta_u = {PUF_DEFAULT_ETA_U})")
    print("-" * 60)
    fuia_reconstruction_save_path = os.path.join(
        output_directory, "fuia_result_puf_special.png")
    (reconstructed_image,
     fuia_reconstruction_mse,
     fuia_reconstruction_psnr_db) = run_fuia_attack(
        original_model,
        unlearned_model_at_default_eta_u,
        stored_updates,
        target_client,
        target_indices,
        target_label,
        target_rounds,
        private_dataset,
        test_loader,
        unlearning_method_label=f"PUF-Special, eta_u={PUF_DEFAULT_ETA_U}",
        save_path=fuia_reconstruction_save_path,
    )

    #Phase 5: Reporting (table + sweep plot)
    print_summary_table(
        original_test_accuracy, original_forget_accuracy,
        retrain_test_accuracy,  retrain_forget_accuracy,
        retraining_time_seconds,
        sweep_results,
        fuia_reconstruction_mse, fuia_reconstruction_psnr_db,
        fuia_unlearning_method_label=f"PUF-Special, eta_u={PUF_DEFAULT_ETA_U}",
    )
    plot_eta_u_sweep(
        sweep_results,
        original_test_accuracy, original_forget_accuracy,
        retrain_test_accuracy,  retrain_forget_accuracy,
        save_path=os.path.join(output_directory, "puf_special_eta_u_sweep.png"),
    )

    total_time_seconds = time.time() - total_start_time
    print(f"\nTotal time: {total_time_seconds:.0f}s "
          f"({total_time_seconds / 60:.1f} min)")
    wandb.log({"total_time_seconds": total_time_seconds})
    wandb.finish()
