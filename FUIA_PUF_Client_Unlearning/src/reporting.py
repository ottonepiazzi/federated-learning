import time
import matplotlib.pyplot as plt
import wandb

from config import (ETA_U_VALUES, PUF_DEFAULT_ETA_U, PUF_LOCAL_EPOCHS,
                    PUF_UNLEARN_LR, BATCH_SIZE, DEVICE)
from federated import evaluate
from unlearning import puf_special_unlearn, forget_accuracy


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
