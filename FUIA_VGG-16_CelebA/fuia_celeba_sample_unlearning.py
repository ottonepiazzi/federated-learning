#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
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
FRACTION         = 0.2         #20% client participation per round
NUM_ROUNDS       = 80          #paper: 80 rounds (FL training is fast, inversion dominates runtime)
LOCAL_EPOCHS     = 3
BATCH_SIZE       = 32
FL_LR            = 0.1         #paper: "initial client-side learning rate is set to 0.1"
PRETRAIN_EPOCHS  = 15          #reduced from 400 for faster runtime
PRETRAIN_LR      = 0.01        #lower LR to preserve pretrained conv features
NUM_CLASSES      = 2           #paper: binary classification (smile vs non-smile)
DATA_PER_CLIENT  = 8           #paper Sec VII.A: "set the number of data points per client to 8"

#image settings
IMG_SIZE         = 64          #CelebA resized to 64x64
IMG_CHANNELS     = 3           #RGB
IMG_MEAN         = (0.5, 0.5, 0.5)
IMG_STD          = (0.5, 0.5, 0.5)

#Gradient inversion
INV_ITERATIONS   = 10000
INV_LR           = 0.01
INV_ALPHA        = 1e-5         #TV regularization weight (paper Eq. 13)
INV_RESTARTS     = 3


#VGG-16 model architecture
class VGG16(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        #Block 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)

        #Block 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)

        #Block 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, return_indices=True)

        #Block 4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, return_indices=True)

        #Block 5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, return_indices=True)

        #Classifier
        self.fc1 = nn.Linear(512 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        if pretrained:
            self._load_imagenet_conv_weights()

    #load pretrained weights from ImageNet
    def _load_imagenet_conv_weights(self):
        
        #code to fix cerficate problem on Mac
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        pretrained_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        #Mapping: torchvision features index to our layer name
        mapping = {
            '0': 'conv1_1', '2': 'conv1_2',
            '5': 'conv2_1', '7': 'conv2_2',
            '10': 'conv3_1', '12': 'conv3_2', '14': 'conv3_3',
            '17': 'conv4_1', '19': 'conv4_2', '21': 'conv4_3',
            '24': 'conv5_1', '26': 'conv5_2', '28': 'conv5_3',
        }
        pretrained_sd = pretrained_vgg.state_dict()
        for idx, our_name in mapping.items():
            our_layer = getattr(self, our_name)
            our_layer.weight.data.copy_(pretrained_sd[f'features.{idx}.weight'])
            our_layer.bias.data.copy_(pretrained_sd[f'features.{idx}.bias'])
        print("  Loaded pretrained ImageNet conv weights into VGG-16")
        del pretrained_vgg

    def forward(self, x):
        #Block 1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x, _ = self.pool1(x)

        #Block 2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x, _ = self.pool2(x)

        #Block 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x, _ = self.pool3(x)

        #Block 4
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x, _ = self.pool4(x)

        #Block 5
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x, _ = self.pool5(x)

        #Classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)



#CelebA dataset loading
import pandas as pd
from PIL import Image

CELEBA_ROOT = os.path.join("data", "celeba")
CELEBA_IMG_DIR = os.path.join(CELEBA_ROOT, "img_align_celeba", "img_align_celeba")

#CelebA smile/non-smile dataset from Kaggle CSV files
class CelebASmile(torch.utils.data.Dataset):
    def __init__(self, filenames, labels, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(CELEBA_IMG_DIR, self.filenames[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_celeba():
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])

    #read attributes and partition CSVs
    attrs = pd.read_csv(os.path.join(CELEBA_ROOT, "list_attr_celeba.csv"))
    partitions = pd.read_csv(os.path.join(CELEBA_ROOT, "list_eval_partition.csv"))

    #merge on image_id
    df = attrs.merge(partitions, on="image_id")

    #smile labels: convert from {-1, 1} to {0, 1}
    df["label"] = (df["Smiling"] == 1).astype(int)

    #split by partition: 0=train, 1=val, 2=test
    train_df = df[df["partition"] == 0]
    test_df = df[df["partition"] == 2]

    train = CelebASmile(train_df["image_id"].tolist(),
                        train_df["label"].tolist(), transform)
    test = CelebASmile(test_df["image_id"].tolist(),
                       test_df["label"].tolist(), transform)

    print(f"  CelebA (smile/non-smile): {len(train)} train, {len(test)} test samples")
    return train, test

    
    
#IID partition: assign data_per_client random samples to each client
def partition_iid(dataset, num_clients, data_per_client):
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

   
#FedAvg: weighted average of client state dicts
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


#Linear decay (as specified in the paper)
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
        if ep % 50 == 0 or ep == 1:
            print(f"  Pretrain epoch {ep}/{epochs} | Loss: {ep_loss / len(loader):.4f}")

    model.to("cpu")
    return model


#FL round loop
def run_fl_rounds(model, client_data, private_data, round_selections,
                  target_client=None, test_loader=None, log_prefix=None):
    """Run FL training. To save memory, only store full updates for target_client;
    for other clients, only store their L1 norms (needed for gradient separation)."""

    stored_updates = {}      #(rnd, target_client) -> full param delta
    round_l1_norms = {}      #rnd -> total L1 norm sum of all clients in that round
    verbose = log_prefix is not None
    t0 = time.time()

    for rnd in range(1, NUM_ROUNDS + 1):
        selected = round_selections[rnd]
        if not selected:
            continue
        lr = lr_schedule(FL_LR, rnd, NUM_ROUNDS)
        global_sd = {k: v.clone() for k, v in model.state_dict().items()}

        results = []
        l1_sum_this_round = 0.0
        for k in selected:
            sd, n_k, loss = client_update(copy.deepcopy(model), private_data,
                                          client_data[k], LOCAL_EPOCHS,
                                          BATCH_SIZE, lr, DEVICE)
            results.append((sd, n_k, loss))
            delta = {key: sd[key].float() - global_sd[key].float() for key in sd}
            l1_norm_k = float(sum(v.abs().sum() for v in delta.values()))
            l1_sum_this_round += l1_norm_k

            if target_client is not None and k == target_client:
                stored_updates[(rnd, k)] = delta
                stored_updates[(rnd, k, 'l1')] = l1_norm_k
            elif target_client is None:
                #No target specified — store everything (used when target unknown)
                stored_updates[(rnd, k)] = delta
            #Otherwise: discard delta to save memory

        round_l1_norms[rnd] = l1_sum_this_round

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

    return model, stored_updates, round_l1_norms


PRETRAIN_CHECKPOINT = "checkpoints/celeba_vgg16_pretrained.pt"

def run_fl_training():
    train_data, test_data = load_celeba()
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    n_total = len(train_data)
    n_pretrain = int(0.8 * n_total)
    pretrain_data, private_data = torch.utils.data.random_split(
        train_data, [n_pretrain, n_total - n_pretrain])

    model = VGG16(pretrained=True)

    if os.path.exists(PRETRAIN_CHECKPOINT):
        print(f"Loading pretrained checkpoint from {PRETRAIN_CHECKPOINT}...")
        model.load_state_dict(torch.load(PRETRAIN_CHECKPOINT, weights_only=True))
    else:
        print(f"Pre-training on {n_pretrain} samples ({PRETRAIN_EPOCHS} epochs)...")
        model = pretrain_model(model, pretrain_data, PRETRAIN_EPOCHS, BATCH_SIZE,
                               PRETRAIN_LR, DEVICE)
        os.makedirs(os.path.dirname(PRETRAIN_CHECKPOINT), exist_ok=True)
        torch.save(model.state_dict(), PRETRAIN_CHECKPOINT)
        print(f"  Saved pretrained checkpoint to {PRETRAIN_CHECKPOINT}")

    acc = evaluate(model, test_loader, DEVICE)
    print(f"  Pretrain accuracy: {acc * 100:.1f}%\n")
    pretrained_sd = {k: v.clone() for k, v in model.state_dict().items()}

    client_data = partition_iid(private_data, NUM_CLIENTS, DATA_PER_CLIENT)
    m = max(1, int(FRACTION * NUM_CLIENTS))
    round_selections = {r: random.sample(range(NUM_CLIENTS), m)
                        for r in range(1, NUM_ROUNDS + 1)}

    #Pre-select target client: the one that participates in the most rounds (to obtain the best result possible)
    round_counts = {}
    for rnd, clients in round_selections.items():
        for c in clients:
            round_counts[c] = round_counts.get(c, 0) + 1
    target_client = max(round_counts, key=round_counts.get)
    print(f"  Pre-selected target client: {target_client} "
          f"({round_counts[target_client]} rounds)")

    print(f"FL Training: {NUM_ROUNDS} rounds, {m} clients/round, "
          f"{DATA_PER_CLIENT} samples/client, LR={FL_LR}")
    model, stored_updates, round_l1_norms = run_fl_rounds(
        model, client_data, private_data, round_selections,
        target_client=target_client, test_loader=test_loader, log_prefix="fl")

    return (model, stored_updates, round_l1_norms, client_data, private_data,
            pretrained_sd, round_selections, test_loader, target_client)

    
#Randomly select 1 sample per client to forget (paper Sec VI.B.1)
def select_forgotten_samples(client_data, private_data):
    forgotten = {}
    for client_id, sample_indices in client_data.items():
        forgotten[client_id] = random.choice(sample_indices)
    return forgotten

    
    
#Retrain from pretrained model, removing ALL forgotten samples at once
def retrain_without_samples(pretrained_sd, private_data, client_data,
                            forgotten_samples, round_selections, target_client):
    model = VGG16()
    model.load_state_dict(pretrained_sd)
    modified_data = {}
    for client_id, sample_indices in client_data.items():
        if client_id in forgotten_samples:
            modified_data[client_id] = [i for i in sample_indices
                                        if i != forgotten_samples[client_id]]
        else:
            modified_data[client_id] = list(sample_indices)
    return run_fl_rounds(model, modified_data, private_data, round_selections,
                         target_client=target_client)




#FUIA ATTACK (FUIA for Sample Unlearning)
#Step 1: Gradient Separation (paper Eq. 8-11)
def gradient_separation(stored_updates, round_l1_norms, target_client):
    """Eq. 8-11: extract clean gradient for target client using stored updates.
    Uses pre-computed L1 norm sums to avoid storing all clients' full updates."""

    #find rounds where target client participated
    rounds_with_target = sorted([rnd for (rnd, cid, *rest) in stored_updates.keys()
                                 if cid == target_client and not rest])
    if not rounds_with_target:
        raise ValueError(f"Client {target_client} never participated")

    clean_gradient = None
    for rnd in rounds_with_target:
        target_update = stored_updates[(rnd, target_client)]

        #Eq. 9: total L1 norm sum (pre-computed during FL training)
        l1_norm_sum = round_l1_norms[rnd]

        #Eq. 10: weight for target client
        l1_key = (rnd, target_client, 'l1')
        if l1_key in stored_updates:
            target_l1 = stored_updates[l1_key]
        else:
            target_l1 = float(sum(v.float().abs().sum() for v in target_update.values()))
        weight = target_l1 / (l1_norm_sum + 1e-12)

        #Eq. 11: accumulate weighted update
        if clean_gradient is None:
            clean_gradient = {key: weight * val.float().clone()
                              for key, val in target_update.items()}
        else:
            for key in clean_gradient:
                clean_gradient[key] += weight * target_update[key].float()

    return clean_gradient


#Step 2: Target Gradient Acquisition (paper Eq. 12)
def target_gradient_acquisition(clean_gradient_fl, clean_gradient_fu):
    target_gradient = {key: clean_gradient_fl[key] - clean_gradient_fu[key]
                       for key in clean_gradient_fl}
    return target_gradient


#Step 3: Gradient Inversion (paper Eq. 13-14)
def total_variation(image):
    #Anisotropic total variation (Eq. 15)
    diff_h = (image[:, :, 1:, :] - image[:, :, :-1, :]).pow(2).sum()
    diff_w = (image[:, :, :, 1:] - image[:, :, :, :-1]).pow(2).sum()
    return diff_h + diff_w


#Cosine similarity between two gradient dicts, flattened and concatenated
def cosine_similarity_gradients(grad_a, grad_b, param_keys):
    flat_a = torch.cat([grad_a[k].flatten() for k in param_keys])
    flat_b = torch.cat([grad_b[k].flatten() for k in param_keys])
    return nn.functional.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0))


def gradient_inversion(original_model, target_gradient, label):
    #Reconstruct the forgotten sample via gradient matching on W^o
    device = DEVICE
    param_keys = sorted(target_gradient.keys())

    #negate target gradient: stored updates are in negative-gradient space (important!)
    target_on_device = {k: -v.to(device).detach() for k, v in target_gradient.items()}

    #pixel bounds in normalized space (per channel)
    pixel_min = (0.0 - IMG_MEAN[0]) / IMG_STD[0]  #all channels same with (0.5,0.5,0.5)
    pixel_max = (1.0 - IMG_MEAN[0]) / IMG_STD[0]

    loss_fn = nn.CrossEntropyLoss()
    label_tensor = torch.tensor([label], device=device)

    #use original model W^o (Eq. 14)
    model = copy.deepcopy(original_model).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(True)

    best_image = None
    best_cosine = float('-inf')

    for restart in range(INV_RESTARTS):
        torch.manual_seed(SEED + restart * 7919)
        virtual_image = torch.randn(1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE,
                                    device=device, requires_grad=True)
        optimizer = torch.optim.Adam([virtual_image], lr=INV_LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=INV_ITERATIONS, eta_min=INV_LR * 0.01)

        restart_best_cosine = float('-inf')
        restart_best_image = virtual_image.detach().clone()

        for iteration in range(1, INV_ITERATIONS + 1):
            optimizer.zero_grad()
            model.zero_grad()

            #Eq. 14: virtual gradient on W^o
            output = model(virtual_image)
            classification_loss = loss_fn(output, label_tensor)
            virtual_grads = torch.autograd.grad(
                classification_loss, model.parameters(), create_graph=True)
            virtual_grad_dict = {name: grad for (name, _), grad in
                                 zip(model.named_parameters(), virtual_grads)}

            #Eq. 13: min -cos_sim + alpha * TV
            cos_sim = cosine_similarity_gradients(
                virtual_grad_dict, target_on_device, param_keys)
            inversion_loss = -cos_sim + INV_ALPHA * total_variation(virtual_image)

            inversion_loss.backward()
            optimizer.step()
            scheduler.step()

            #clamp to valid pixel range
            with torch.no_grad():
                virtual_image.clamp_(pixel_min, pixel_max)

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
    #Convert from normalized CelebA space back to [0, 1] pixel range
    img = image_tensor.cpu().float()
    for c in range(IMG_CHANNELS):
        img[c] = img[c] * IMG_STD[c] + IMG_MEAN[c]
    return img.clamp(0, 1)


#MSE and PSNR between original and reconstructed images (in [0,1] space)
def compute_metrics(original_image, reconstructed_image):
    original = denormalize(original_image)
    reconstructed = denormalize(reconstructed_image)
    mse = torch.mean((original - reconstructed) ** 2).item()
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))
    return mse, psnr


#FUIA Attack on a single target client
def attack_target_client(original_model, stored_updates_fl, round_l1_fl,
                         stored_updates_fu, round_l1_fu,
                         private_data, target_client, forgotten_idx,
                         round_selections):
    target_label = private_data[forgotten_idx][1]
    rounds_with_target = [r for r, cs in round_selections.items()
                          if target_client in cs]

    print(f"\n  Attacking client {target_client}")
    print(f"    Forgotten label:  {target_label}")
    print(f"    Participated in {len(rounds_with_target)}/{NUM_ROUNDS} rounds")

    #Step 1: Gradient Separation
    print("    [Step 1a] Gradient separation on FL updates -> clean_gradient_fl")
    clean_gradient_fl = gradient_separation(stored_updates_fl, round_l1_fl, target_client)
    print(f"      L2 norm: {sum(v.norm().item() for v in clean_gradient_fl.values()):.6f}")

    print("    [Step 1b] Gradient separation on FU updates -> clean_gradient_fu")
    clean_gradient_fu = gradient_separation(stored_updates_fu, round_l1_fu, target_client)
    print(f"      L2 norm: {sum(v.norm().item() for v in clean_gradient_fu.values()):.6f}")

    #Step 2: Target Gradient Acquisition
    print("    [Step 2] Target gradient: nabla_k = clean_gradient_fl - clean_gradient_fu")
    nabla_k = target_gradient_acquisition(clean_gradient_fl, clean_gradient_fu)
    nabla_k_norm = sum(v.norm().item() for v in nabla_k.values())
    print(f"      L2 norm: {nabla_k_norm:.6f}")

    #Step 3: Gradient Inversion
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


#Main
if __name__ == "__main__":
    config = {k: v for k, v in globals().items()
              if k.isupper() and isinstance(v, (int, float, str, tuple))}
    config.update({"scenario": "sample_unlearning",
                   "unlearning_method": "retraining", "dataset": "CelebA"})
    wandb.init(project="FUIA", config=config)

    total_start = time.time()

    #Phase 1: Federated Learning
    print("=" * 60 + "\nPhase 1: Federated Learning\n" + "=" * 60)
    (original_model, stored_updates_fl, round_l1_fl, client_data, private_data,
     pretrained_sd, round_selections, test_loader, target_client) = run_fl_training()

    #Phase 2: Unlearning
    print("\n" + "=" * 60 + "\nPhase 2: Sample Unlearning (Retraining)\n" + "=" * 60)
    forgotten_samples = select_forgotten_samples(client_data, private_data)
    print(f"  Selected 1 forgotten sample per client ({len(forgotten_samples)} total)")

    retrain_start = time.time()
    unlearned_model, stored_updates_fu, round_l1_fu = retrain_without_samples(
        pretrained_sd, private_data, client_data,
        forgotten_samples, round_selections, target_client)
    retrain_time = time.time() - retrain_start

    original_acc = evaluate(original_model, test_loader, DEVICE)
    unlearned_acc = evaluate(unlearned_model, test_loader, DEVICE)
    print(f"  Retraining done ({retrain_time:.0f}s)")
    print(f"  Original model accuracy:  {original_acc * 100:.1f}%")
    print(f"  Unlearned model accuracy: {unlearned_acc * 100:.1f}%")

    #Phase 3: FUIA Attack
    print("\n" + "=" * 60 + "\nPhase 3: FUIA Attack on Target Client\n" + "=" * 60)
    forgotten_idx = forgotten_samples[target_client]
    target_label = private_data[forgotten_idx][1]
    print(f"  Target client: {target_client} "
          f"(label={target_label}, index={forgotten_idx})")

    reconstructed, mse, psnr = attack_target_client(
        original_model, stored_updates_fl, round_l1_fl,
        stored_updates_fu, round_l1_fu,
        private_data, target_client, forgotten_idx, round_selections)

    #Results
    print(f"\n{'=' * 60}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  (Paper Fig. 5 reference for CelebA Retrain: MSE ~0.03, PSNR ~15 dB)")
    print(f"{'=' * 60}")

    #Visualization
    original_image = private_data[forgotten_idx][0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    #for RGB: convert from (C, H, W) to (H, W, C) for matplotlib
    orig_display = denormalize(original_image).permute(1, 2, 0).numpy()
    recon_display = denormalize(reconstructed.squeeze(0)).permute(1, 2, 0).numpy()

    axes[0].imshow(orig_display)
    axes[0].set_title(f"Original (label={target_label})", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(recon_display)
    axes[1].set_title(f"FUIA Reconstruction\nMSE={mse:.4f}  PSNR={psnr:.2f} dB",
                      fontsize=14)
    axes[1].axis("off")
    plt.suptitle("FUIA Sample Unlearning — CelebA (Retraining)", fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "fuia_celeba_result.png")
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
