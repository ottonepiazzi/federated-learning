import time

from config import NUM_ROUNDS, INV_RESTARTS, INV_ITERATIONS
from fuia import (gradient_separation, target_gradient_acquisition,
                  gradient_inversion, gradient_inversion_batch)
from metrics import compute_metrics, match_reconstructions


#FUIA Attack on a single target client (Algorithm 1, one iteration of the loop)
def attack_target_client(original_model, stored_updates_fl, stored_updates_fu,
                         private_data, target_client, forgotten_idx,
                         round_selections):

    target_label = private_data[forgotten_idx][1]
    rounds_with_target = [r for r, cs in round_selections.items()
                          if target_client in cs]

    print(f"\n  Attacking client {target_client}")
    print(f"    Forgotten label:  {target_label}")
    print(f"    Participated in {len(rounds_with_target)}/{NUM_ROUNDS} rounds")

    #Step 1: Gradient Separation (Eq. 8-11)
    print("    [Step 1a] Gradient separation on FL updates -> clean_gradient_fl")
    clean_gradient_fl = gradient_separation(stored_updates_fl, target_client)
    print(f"      L2 norm: {sum(v.norm().item() for v in clean_gradient_fl.values()):.6f}")

    print("    [Step 1b] Gradient separation on FU updates -> clean_gradient_fu")
    clean_gradient_fu = gradient_separation(stored_updates_fu, target_client)
    print(f"      L2 norm: {sum(v.norm().item() for v in clean_gradient_fu.values()):.6f}")

    #Step 2: Target Gradient Acquisition (Eq. 12)
    print("    [Step 2] Target gradient: nabla_k = clean_gradient_fl - clean_gradient_fu")
    nabla_k = target_gradient_acquisition(clean_gradient_fl, clean_gradient_fu)
    nabla_k_norm = sum(v.norm().item() for v in nabla_k.values())
    print(f"      L2 norm: {nabla_k_norm:.6f}")

    #Step 3: Gradient Inversion (Eq. 13-14)
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


#FUIA attack reconstructing a SET of forgotten samples of one client at once
#(paper Sec VII.A.2 ablation). Same three steps as attack_target_client, but the
#target gradient now aggregates several forgotten samples and gradient inversion
#recovers them jointly as a batch. clean_gradient_fl may be passed in to avoid
#recomputing it for every sweep point (it depends only on the FL updates, not on
#which/how many samples are forgotten).
def attack_forgotten_set(original_model, stored_updates_fl, stored_updates_fu,
                         private_data, target_client, forgotten_indices,
                         round_selections, clean_gradient_fl=None):

    labels = [int(private_data[idx][1]) for idx in forgotten_indices]
    n = len(forgotten_indices)
    rounds_with_target = [r for r, cs in round_selections.items()
                          if target_client in cs]

    print(f"\n  Attacking client {target_client} | forgetting {n} sample(s)")
    print(f"    Forgotten labels: {labels}")
    print(f"    Participated in {len(rounds_with_target)}/{NUM_ROUNDS} rounds")

    #Step 1: Gradient Separation (Eq. 8-11). FL side is reused across the sweep.
    if clean_gradient_fl is None:
        print("    [Step 1a] Gradient separation on FL updates -> clean_gradient_fl")
        clean_gradient_fl = gradient_separation(stored_updates_fl, target_client)
    print("    [Step 1b] Gradient separation on FU updates -> clean_gradient_fu")
    clean_gradient_fu = gradient_separation(stored_updates_fu, target_client)

    #Step 2: Target Gradient Acquisition (Eq. 12)
    nabla_k = target_gradient_acquisition(clean_gradient_fl, clean_gradient_fu)
    nabla_k_norm = sum(v.norm().item() for v in nabla_k.values())
    print(f"    [Step 2] Target gradient L2 norm: {nabla_k_norm:.6f}")

    #Step 3: Batch Gradient Inversion (Eq. 13-14)
    print(f"    [Step 3] Batch gradient inversion for {n} image(s) "
          f"({INV_RESTARTS} restarts x {INV_ITERATIONS} iters)...")
    t0 = time.time()
    reconstructed_batch, best_cosine = gradient_inversion_batch(
        original_model, nabla_k, labels)
    print(f"    Inversion done ({time.time() - t0:.0f}s) | best cos_sim = {best_cosine:.4f}")

    #Match the N reconstructions to the N originals and score per image
    original_images = [private_data[idx][0] for idx in forgotten_indices]
    reconstructed_images = [reconstructed_batch[j] for j in range(n)]
    match = match_reconstructions(original_images, reconstructed_images)

    print(f"    mean PSNR={match['mean_psnr']:.2f} dB | "
          f"best={match['best_psnr']:.2f} | worst={match['worst_psnr']:.2f} | "
          f"mean MSE={match['mean_mse']:.4f}")

    return {
        "n_forget": n,
        "forgotten_indices": list(forgotten_indices),
        "labels": labels,
        "reconstructed_batch": reconstructed_batch,
        "original_images": original_images,
        "match": match,
        "target_gradient_norm": nabla_k_norm,
        "best_cosine": best_cosine,
        "clean_gradient_fl": clean_gradient_fl,
    }


def select_target_client(stored_updates, round_selections):
    #Pick the participating client with the most rounds (strongest signal)
    round_counts = {}
    for (round_id, client_id) in stored_updates:
        round_counts[client_id] = round_counts.get(client_id, 0) + 1
    best_client = max(round_counts, key=round_counts.get)
    return best_client
