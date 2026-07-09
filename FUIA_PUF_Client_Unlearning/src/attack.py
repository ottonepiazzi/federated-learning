import torch
import torch.nn as nn
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import wandb

from config import NUM_ROUNDS, INV_RESTARTS, INV_ITERATIONS, DEVICE
from federated import evaluate
from fuia_steps import gradient_separation, target_gradient_acquisition
from metrics import denormalize, compute_metrics


#FUIA attack orchestration: takes an already-unlearned model from any
#unlearning method (retraining, PUF-Special, ...) and runs the three FUIA
#steps (Gradient Separation, Target Gradient Acquisition, Gradient Inversion).
#The concrete gradient_inversion implementation (CPU or CUDA-batched) and the
#device label used for logging are injected by the caller so the same
#orchestration serves both entry points.
def run_fuia_attack(original_model, unlearned_model, stored_updates,
                    target_client, target_indices, target_label,
                    target_rounds, private_dataset, test_loader,
                    unlearning_method_label, save_path,
                    gradient_inversion, inversion_device_label):

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
          f"({INV_RESTARTS} restart(s) x {INV_ITERATIONS} iters "
          f"on {inversion_device_label})...")
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
