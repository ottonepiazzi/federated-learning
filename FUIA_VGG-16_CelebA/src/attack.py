import time

from config import NUM_ROUNDS, INV_RESTARTS, INV_ITERATIONS
from fuia import (gradient_separation, target_gradient_acquisition,
                  gradient_inversion)
from metrics import compute_metrics


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
