#!/usr/bin/env python3

import time
import wandb

from config import (SEED, NUM_CLIENTS, FRACTION, NUM_ROUNDS, LOCAL_EPOCHS,
                    BATCH_SIZE, FL_LR, PRETRAIN_EPOCHS, PRETRAIN_LR, NUM_CLASSES,
                    DATA_PER_CLIENT, INV_ITERATIONS, INV_LR, INV_GAMMA,
                    INV_ALPHA, INV_RESTARTS)
from fl_training import run_fl_training
from attack import run_fuia_attack


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
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "pretrain_lr": PRETRAIN_LR,
        "num_classes": NUM_CLASSES,
        "data_per_client": DATA_PER_CLIENT,
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
     pretrained_sd, round_selections, test_loader) = run_fl_training()

    #Phase 2: FUIA Attack
    print("\n" + "-" * 60)
    print("Phase 2: FUIA Client Unlearning Attack")
    print("-" * 60)
    reconstructed, mse, psnr = run_fuia_attack(
        original_model, stored_updates, client_data, private_data,
        pretrained_sd, round_selections, test_loader)

    total_time = time.time() - total_t0
    print(f"\nTotal time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    wandb.log({"total_time_s": total_time})
    wandb.finish()
