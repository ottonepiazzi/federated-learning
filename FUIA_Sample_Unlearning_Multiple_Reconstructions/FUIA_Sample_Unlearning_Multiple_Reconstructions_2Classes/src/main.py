#!/usr/bin/env python3

import time
import os
import matplotlib.pyplot as plt
import wandb

import config as cfg
from config import DEVICE
from federated import evaluate
from fl_training import run_fl_training
from unlearning import select_forgotten_samples, retrain_without_samples
from attack import attack_target_client, select_target_client
from metrics import denormalize


#main
if __name__ == "__main__":
    config = {k: v for k, v in vars(cfg).items()
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

    #Save into the project folder (the parent of this src/ package), matching
    #the original script's output location.
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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
