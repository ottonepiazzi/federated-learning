import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
import time
import wandb

from config import (NUM_CLIENTS, FRACTION, NUM_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE,
                    FL_LR, TARGET_CLIENT, TARGET_SIZE, SEED, DEVICE, DL_PIN_MEMORY)
from model import CNN
from data import load_mnist, partition_iid
from federated import client_update, fedavg, evaluate, lr_schedule


#FL training: 100% participation, no pretraining
#snapshot the freshly-initialized weights and use them as the common
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
