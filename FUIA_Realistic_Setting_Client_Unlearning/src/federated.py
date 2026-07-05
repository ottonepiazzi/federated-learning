import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict

from config import DL_PIN_MEMORY, WEIGHTED_AGGREGATION


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


#FedAvg aggregation. `weighted=None` falls back to the config default.
#  weighted=True  -> size-weighted average (paper-faithful, original behaviour)
#  weighted=False -> uniform average over participating clients
def fedavg(global_model, client_results, weighted=None):
    if weighted is None:
        weighted = WEIGHTED_AGGREGATION
    total_n = sum(n for _, n, _ in client_results)
    n_clients = len(client_results)
    agg = OrderedDict()
    for sd, n, _ in client_results:
        w = (n / total_n) if weighted else (1.0 / n_clients)
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
