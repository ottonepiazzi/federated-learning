import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import copy
import time
import wandb

from config import (NUM_CLIENTS, FRACTION, NUM_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE,
                    FL_LR, PRETRAIN_EPOCHS, PRETRAIN_LR, DATA_PER_CLIENT, DEVICE)
from model import CNN
from data import load_mnist, partition_iid
from federated import client_update, fedavg, evaluate, lr_schedule, pretrain_model


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
    train_data, test_data = load_mnist()
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
