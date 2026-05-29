import os
import time
import wandb

from config import (SEED, NUM_CLIENTS, FRACTION, NUM_ROUNDS, LOCAL_EPOCHS,
                    BATCH_SIZE, FL_LR, PRETRAIN_EPOCHS, PRETRAIN_LR, NUM_CLASSES,
                    DATA_PER_CLIENT, INV_ITERATIONS, INV_LR, INV_GAMMA, INV_ALPHA,
                    INV_RESTARTS, ETA_U_VALUES, PUF_DEFAULT_ETA_U,
                    PUF_LOCAL_EPOCHS, PUF_UNLEARN_LR, DEVICE)
from fl_training import run_fl_training
from federated import evaluate
from unlearning import retrain_without_client, forget_accuracy
from attack import run_fuia_attack
from reporting import (run_eta_u_sweep, pick_target_client,
                       print_summary_table, plot_eta_u_sweep)


#Attack execution. The concrete gradient_inversion implementation (CPU or
#CUDA-batched) and its device label are injected by the entry point, so both
#fuia_puf_client_unlearning(.py) and the CUDA variant share this pipeline.
def run_pipeline(gradient_inversion, inversion_device_label):
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
    #Write results into the project folder (the parent of this src/ package),
    #matching the original scripts' output location.
    output_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
        gradient_inversion=gradient_inversion,
        inversion_device_label=inversion_device_label,
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
