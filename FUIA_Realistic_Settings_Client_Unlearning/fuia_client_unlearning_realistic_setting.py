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

#Device selection: prefer CUDA (fastest, supports 2nd-order grads through
#MaxPool2d which the gradient-inversion step needs), else MPS (training only),
#else CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True            #fixed-size inputs -> faster
    torch.set_float32_matmul_precision("high")       #TF32 matmuls on Ampere+
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

#Gradient inversion device: needs create_graph=True through MaxPool2d.
#CUDA: fully supported -> run inversion on GPU
#MPS:  unsupported -> must fall back to CPU
#CPU:  obviously CPU
INV_DEVICE = DEVICE if DEVICE.type == "cuda" else torch.device("cpu")

#DataLoader knobs: pin_memory enables async H2D copies on CUDA. We keep
#num_workers=0 because MNIST is already fully resident in RAM after the
#dataset is built, so worker processes only add fork overhead.
DL_PIN_MEMORY = (DEVICE.type == "cuda")

print(f"Train device: {DEVICE} | Inversion device: {INV_DEVICE} | "
      f"pin_memory={DL_PIN_MEMORY}")

#Hyperparameters (new setting: no pretraining, 10 clients, IID, 100% participation,
#target client has only 1 image to forget)
NUM_CLIENTS      = 10
FRACTION         = 1.0        #100% client participation per round
NUM_ROUNDS       = 30
LOCAL_EPOCHS     = 1
BATCH_SIZE       = 64
FL_LR            = 0.01       #paper: "learning rate is set to 0.01"
NUM_CLASSES      = 10         #full MNIST
TARGET_CLIENT    = 0          #the client whose 1 image must be forgotten
TARGET_SIZE      = 1          #number of images held by the target client

#Gradient inversion (paper Section V.B & VII.B; client-unlearning loss = Eq. 18)
INV_ITERATIONS   = 8000       #loss plateaus by ~iter 7000 in this regime
INV_LR           = 0.1
INV_GAMMA        = 0.1        #paper-faithful: weight of Psi term in Eq. 18
INV_ALPHA        = 1e-5       #minimal TV: allow fine detail
INV_RESTARTS     = 1          #restarts find similar optima


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



def client_update(model, dataset, indices, epochs, batch_size, lr, device):
    model.to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=True, num_workers=0,
                        pin_memory=DL_PIN_MEMORY)

    total_loss, n_batches = 0.0, 0
    for _ in range(epochs):
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=DL_PIN_MEMORY)
            labels = labels.to(device, non_blocking=DL_PIN_MEMORY)
            opt.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

    #Leave the model on `device`: caller (run_fl_training) handles staging
    return model.state_dict(), len(indices), total_loss / max(n_batches, 1)


#Standard size-weighted FedAvg (paper-faithful)
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
            imgs = imgs.to(device, non_blocking=DL_PIN_MEMORY)
            labels = labels.to(device, non_blocking=DL_PIN_MEMORY)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total

#learning rate linear decay schedule
def lr_schedule(initial_lr, round_t, total_rounds):
    return initial_lr * (1.0 - (round_t - 1) / total_rounds)


#FL training: 100% participation, no pretraining
#We snapshot the freshly-initialized weights and use them as the common
#starting point for both the original FL run and the retraining-based
#unlearning, so the comparison is clean.
def run_fl_training():
    train_data, test_data = load_mnist()
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False,
                             pin_memory=DL_PIN_MEMORY)

    #IID partition over the FULL training set
    client_data = partition_iid(train_data, NUM_CLIENTS,
                                target_size=TARGET_SIZE,
                                target_client=TARGET_CLIENT,
                                seed=SEED)

    #Quick sanity print of the partition
    sizes = {cid: len(idx) for cid, idx in client_data.items()}
    print(f"Partition (IID): client sizes = {sizes}")
    print(f"Total assigned: {sum(sizes.values())} / {len(train_data)}")

    #Fresh model. Snapshot the initial weights to CPU so the retraining
    #baseline can rebuild the same starting state on a fresh model object
    model = CNN().to(DEVICE)
    init_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    #FL rounds: 100% participation
    m = max(1, int(FRACTION * NUM_CLIENTS))
    stored_updates = {}   #(round, client) -> param_diff dict
    round_selections = {} #round -> list of selected clients

    print(f"FL Training: {NUM_ROUNDS} rounds, {NUM_CLIENTS} clients, "
          f"{m} selected/round, LR={FL_LR}")
    t0 = time.time()

    for rnd in range(1, NUM_ROUNDS + 1):
        selected = list(range(NUM_CLIENTS))   #all clients participate
        round_selections[rnd] = selected
        #Cheap clone of the round's starting weights (avoids deepcopy overhead).
        global_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        lr = lr_schedule(FL_LR, rnd, NUM_ROUNDS)

        results = []
        for k in selected:
            local = copy.deepcopy(model)
            sd, n_k, loss = client_update(local, train_data, client_data[k],
                                          LOCAL_EPOCHS, BATCH_SIZE, lr, DEVICE)
            results.append((sd, n_k, loss))
            #Stage deltas on CPU: 30 rounds * 10 clients * ~2.5MB would
            #otherwise pin ~750MB of GPU memory for no reason
            stored_updates[(rnd, k)] = {
                key: (sd[key].float() - global_sd[key].float()).detach().cpu()
                for key in sd
            }

        model = fedavg(model, results)

        if rnd % 5 == 0 or rnd == 1:
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

    return (model, stored_updates, client_data, train_data,
            init_sd, round_selections, test_loader)


#Retraining (deterministic): same initial state, same round schedule, target excluded
def retrain_without_client(init_sd, private_data, client_data,
                           target, round_selections):
    model = CNN()
    model.load_state_dict(init_sd)
    model.to(DEVICE)

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
    #Psi = W_original - W_unlearned. Materialize on CPU regardless of where
    #the source models live
    #inversion will move tensors to INV_DEVICE itself
    wo = original_model.state_dict()
    wu = unlearned_model.state_dict()
    return {k: (wo[k].float() - wu[k].float()).detach().cpu() for k in wo}


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

    #Inversion needs create_graph=True through MaxPool2d
    #CUDA: supported -> run on GPU
    #MPS:  unsupported -> INV_DEVICE was set to CPU during startup
    #CPU:  CPU
    device = INV_DEVICE

    #Both V_k and Psi are PARAMETER-UPDATE directions, not gradient directions:
    #  * V_k = sum of target client's per-round updates (~ -lr * grad)
    #  * Psi = W_orig - W_unlearned: target's training pushed W_orig in the
    #    -grad direction relative to W_unlearned, so Psi ~ -eps * grad
    #We negate both so that the cosine-distance loss aligns the dummy
    #image's gradient with +grad (the true gradient at W_original on the
    #forgotten sample)
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
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED + r * 7919)
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

    #Always hand back a CPU tensor: downstream callers compare to the original
    #image (CPU) and pass it to matplotlib, which both require host memory
    return best_img.detach().cpu()


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


#FUIA attack orchestration
def run_fuia_attack(original_model, stored_updates, client_data, private_data,
                    init_sd, round_selections, test_loader):

    #Target is fixed by construction: the client holding the single image to forget
    target = TARGET_CLIENT

    target_idx = client_data[target][0]
    target_label = private_data[target_idx][1]
    target_rounds = [r for r, cs in round_selections.items() if target in cs]

    print(f"\n{'=' * 60}")
    print(f"FUIA Attack")
    print(f"  Target client: {target}")
    print(f"  Target label:  {target_label}")
    print(f"  Participated in {len(target_rounds)}/{NUM_ROUNDS} rounds")
    print(f"{'=' * 60}")

    #Step 0: Retraining (unlearning via retraining from scratch without target)
    print("\n[Step 0] Retraining without target client...")
    t0 = time.time()
    unlearned_model = retrain_without_client(
        init_sd, private_data, client_data, target, round_selections)
    rt_time = time.time() - t0
    orig_acc = evaluate(original_model, test_loader, DEVICE)
    unl_acc = evaluate(unlearned_model, test_loader, DEVICE)
    print(f"  Done ({rt_time:.0f}s)")
    print(f"  Original model accuracy:  {orig_acc * 100:.1f}%")
    print(f"  Unlearned model accuracy: {unl_acc * 100:.1f}%")

    #Step 1: Gradient Separation
    print("\n[Step 1] Gradient separation (Eq. 16)...")
    clean_grad = gradient_separation(stored_updates, target)
    clean_norm = sum(v.norm().item() for v in clean_grad.values())
    print(f"  Clean gradient L2 norm: {clean_norm:.6f}")

    #Step 2: Target Gradient Acquisition
    print("\n[Step 2] Target gradient acquisition (Psi = W_orig - W_unlearned)...")
    target_grad = target_gradient_acquisition(original_model, unlearned_model)
    target_norm = sum(v.norm().item() for v in target_grad.values())
    print(f"  Target gradient L2 norm: {target_norm:.6f}")

    #Diagnostic: cosine sim between clean and target gradients
    all_keys = sorted(clean_grad.keys())
    vc = torch.cat([clean_grad[k].flatten() for k in all_keys])
    vt = torch.cat([target_grad[k].flatten() for k in all_keys])
    ct_cos = nn.functional.cosine_similarity(vc.unsqueeze(0), vt.unsqueeze(0)).item()
    print(f"  Cosine sim(clean, target): {ct_cos:.4f}")

    #Diagnostic: check gradient alignment with true data at both models.
    #Theory says cos(true_grad, V_k) and cos(true_grad, Psi) should both be
    #NEGATIVE (V_k and Psi are parameter-update directions, opposite to grad)
    #The inversion uses -V_k and -Psi internally, so what gets aligned with
    #the dummy image's gradient is the negation of these signals.
    print("\n[Diagnostic] Gradient alignment with true data:")
    true_img_t = private_data[target_idx][0].unsqueeze(0)
    true_label_t = torch.tensor([target_label])
    loss_fn_diag = nn.CrossEntropyLoss()
    all_keys = sorted(clean_grad.keys())

    for model_name, m in [("W_original", original_model), ("W_unlearned", unlearned_model)]:
        m_cpu = copy.deepcopy(m).to("cpu").eval()
        for p in m_cpu.parameters():
            p.requires_grad_(True)
        m_cpu.zero_grad()
        out = m_cpu(true_img_t)
        loss = loss_fn_diag(out, true_label_t)
        grads = torch.autograd.grad(loss, m_cpu.parameters())
        true_g = {name: g.detach() for (name, _), g in zip(m_cpu.named_parameters(), grads)}

        #Per-layer cosine similarity with V_k and Psi
        cos_clean_layers = [nn.functional.cosine_similarity(
            true_g[k].flatten().unsqueeze(0),
            clean_grad[k].flatten().unsqueeze(0)).item() for k in all_keys]
        cos_target_layers = [nn.functional.cosine_similarity(
            true_g[k].flatten().unsqueeze(0),
            target_grad[k].flatten().unsqueeze(0)).item() for k in all_keys]

        avg_cc = np.mean(cos_clean_layers)
        avg_ct = np.mean(cos_target_layers)
        print(f"  {model_name}: avg_cos(true_grad, V_k)={avg_cc:.4f}  "
              f"avg_cos(true_grad, Psi)={avg_ct:.4f}")
        for i, k in enumerate(all_keys):
            print(f"    {k:30s}: cos(V_k)={cos_clean_layers[i]:.4f}  "
                  f"cos(Psi)={cos_target_layers[i]:.4f}")

    #Step 3: Gradient Inversion (paper Eq. 18 with gamma=INV_GAMMA)
    #The inversion negates V_k and Psi internally so that minimizing the
    #per-layer cosine distance pulls the dummy image's gradient toward the
    #true gradient direction at W_original.
    print(f"\n[Step 3] Gradient inversion using W_original "
          f"({INV_RESTARTS} restart(s) x {INV_ITERATIONS} iters on {INV_DEVICE})...")
    t0 = time.time()
    reconstructed = gradient_inversion(original_model, clean_grad, target_grad,
                                       target_label)
    inv_time = time.time() - t0
    print(f"  Inversion done ({inv_time:.0f}s)")

    #Compute metrics
    original_img = private_data[target_idx][0]
    mse, psnr = compute_metrics(original_img, reconstructed.squeeze(0))

    print(f"\n{'-' * 60}")
    print(f"  MSE:  {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  (Paper reference: MSE ~ 0.0005, PSNR ~ 33 dB)")
    print(f"{'-' * 60}")

    #WandB logging
    wandb.log({
        "attack/mse": mse,
        "attack/psnr": psnr,
        "attack/inversion_time_s": inv_time,
        "attack/retrain_time_s": rt_time,
        "attack/target_client": target,
        "attack/target_label": target_label,
        "attack/n_target_rounds": len(target_rounds),
        "attack/clean_grad_norm": clean_norm,
        "attack/target_grad_norm": target_norm,
        "attack/cos_clean_target": ct_cos,
        "attack/orig_acc": orig_acc,
        "attack/unlearned_acc": unl_acc,
    })

    #Visualization: side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    orig_dn = denormalize(original_img).squeeze().numpy()
    recon_dn = denormalize(reconstructed.squeeze(0)).squeeze().numpy()

    axes[0].imshow(orig_dn, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original (label={target_label})", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(recon_dn, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"FUIA Reconstruction\nMSE={mse:.6f}  PSNR={psnr:.2f} dB",
                      fontsize=14)
    axes[1].axis("off")
    plt.suptitle("FUIA Client Unlearning Attack — MNIST + Retraining", fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "fuia_result.png")
    plt.savefig(save_path, dpi=150)
    wandb.log({"attack/result": wandb.Image(fig)})
    print(f"  Saved to {save_path}")
    plt.show()

    return reconstructed, mse, psnr


#Attack execution
if __name__ == "__main__":
    wandb.init(project="FUIA", config={
        "scenario": "client_unlearning",
        "dataset": "MNIST",
        "unlearning_method": "retraining",
        "num_clients": NUM_CLIENTS,
        "fraction": FRACTION,
        "num_rounds": NUM_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "fl_lr": FL_LR,
        "num_classes": NUM_CLASSES,
        "target_client": TARGET_CLIENT,
        "target_size": TARGET_SIZE,
        "partition": "iid",
        "pretraining": False,
        "aggregation": "size_weighted",
        "inv_iterations": INV_ITERATIONS,
        "inv_lr": INV_LR,
        "inv_gamma": INV_GAMMA,
        "inv_alpha": INV_ALPHA,
        "inv_restarts": INV_RESTARTS,
        "seed": SEED,
    })

    total_t0 = time.time()

    #Phase 1: FL Training
    print("-" * 60)
    print("Phase 1: Federated Learning")
    print("-" * 60)
    (original_model, stored_updates, client_data, private_data,
     init_sd, round_selections, test_loader) = run_fl_training()

    #Phase 2: FUIA Attack
    print("\n" + "-" * 60)
    print("Phase 2: FUIA Client Unlearning Attack")
    print("-" * 60)
    reconstructed, mse, psnr = run_fuia_attack(
        original_model, stored_updates, client_data, private_data,
        init_sd, round_selections, test_loader)

    total_time = time.time() - total_t0
    print(f"\nTotal time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    wandb.log({"total_time_s": total_time})
    wandb.finish()
