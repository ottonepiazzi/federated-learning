import torch
import torch.nn as nn
import numpy as np
import copy
import time
import os
import matplotlib.pyplot as plt
import wandb

from config import NUM_ROUNDS, INV_RESTARTS, INV_ITERATIONS, DEVICE
from federated import evaluate
from unlearning import retrain_without_client
from fuia import gradient_separation, target_gradient_acquisition, gradient_inversion
from metrics import denormalize, compute_metrics


#FUIA attack orchestration
def run_fuia_attack(original_model, stored_updates, client_data, private_data,
                    pretrained_sd, round_selections, test_loader):

    #Pick a target client that actually participated in FL
    target = None
    for (t, k) in sorted(stored_updates.keys()):
        target = k
        break

    target_idx = client_data[target][0]
    target_label = private_data[target_idx][1]
    target_rounds = [r for r, cs in round_selections.items() if target in cs]

    print(f"\n{'=' * 60}")
    print(f"FUIA Attack")
    print(f"  Target client: {target}")
    print(f"  Target label:  {target_label}")
    print(f"  Participated in {len(target_rounds)}/{NUM_ROUNDS} rounds")
    print(f"{'=' * 60}")

    #Step 0: Retraining (unlearning via retraining from scratch without target)
    print("\n[Step 0] Retraining without target client...")
    t0 = time.time()
    unlearned_model = retrain_without_client(
        pretrained_sd, private_data, client_data, target, round_selections)
    rt_time = time.time() - t0
    orig_acc = evaluate(original_model, test_loader, DEVICE)
    unl_acc = evaluate(unlearned_model, test_loader, DEVICE)
    print(f"  Done ({rt_time:.0f}s)")
    print(f"  Original model accuracy:  {orig_acc * 100:.1f}%")
    print(f"  Unlearned model accuracy: {unl_acc * 100:.1f}%")

    #Step 1: Gradient Separation
    print("\n[Step 1] Gradient separation (Eq. 16)...")
    clean_grad = gradient_separation(stored_updates, target)
    clean_norm = sum(v.norm().item() for v in clean_grad.values())
    print(f"  Clean gradient L2 norm: {clean_norm:.6f}")

    #Step 2: Target Gradient Acquisition
    print("\n[Step 2] Target gradient acquisition (Psi = W_orig - W_unlearned)...")
    target_grad = target_gradient_acquisition(original_model, unlearned_model)
    target_norm = sum(v.norm().item() for v in target_grad.values())
    print(f"  Target gradient L2 norm: {target_norm:.6f}")

    #Diagnostic: cosine sim between clean and target gradients
    all_keys = sorted(clean_grad.keys())
    vc = torch.cat([clean_grad[k].flatten() for k in all_keys])
    vt = torch.cat([target_grad[k].flatten() for k in all_keys])
    ct_cos = nn.functional.cosine_similarity(vc.unsqueeze(0), vt.unsqueeze(0)).item()
    print(f"  Cosine sim(clean, target): {ct_cos:.4f}")

    print("\n[Diagnostic] Gradient alignment with true data:")
    true_img_t = private_data[target_idx][0].unsqueeze(0)
    true_label_t = torch.tensor([target_label])
    loss_fn_diag = nn.CrossEntropyLoss()
    all_keys = sorted(clean_grad.keys())

    for model_name, m in [("W_original", original_model), ("W_unlearned", unlearned_model)]:
        m_cpu = copy.deepcopy(m).to("cpu").eval()
        for p in m_cpu.parameters():
            p.requires_grad_(True)
        m_cpu.zero_grad()
        out = m_cpu(true_img_t)
        loss = loss_fn_diag(out, true_label_t)
        grads = torch.autograd.grad(loss, m_cpu.parameters())
        true_g = {name: g.detach() for (name, _), g in zip(m_cpu.named_parameters(), grads)}

        #Per-layer cosine similarity with V_k and Psi
        cos_clean_layers = [nn.functional.cosine_similarity(
            true_g[k].flatten().unsqueeze(0),
            clean_grad[k].flatten().unsqueeze(0)).item() for k in all_keys]
        cos_target_layers = [nn.functional.cosine_similarity(
            true_g[k].flatten().unsqueeze(0),
            target_grad[k].flatten().unsqueeze(0)).item() for k in all_keys]

        avg_cc = np.mean(cos_clean_layers)
        avg_ct = np.mean(cos_target_layers)
        print(f"  {model_name}: avg_cos(true_grad, V_k)={avg_cc:.4f}  "
              f"avg_cos(true_grad, Psi)={avg_ct:.4f}")
        for i, k in enumerate(all_keys):
            print(f"    {k:30s}: cos(V_k)={cos_clean_layers[i]:.4f}  "
                  f"cos(Psi)={cos_target_layers[i]:.4f}")

    #Step 3: Gradient Inversion (paper Eq. 18 with gamma=INV_GAMMA).
    #The inversion negates V_k and Psi internally so that minimizing the
    #per-layer cosine distance pulls the dummy image's gradient toward the
    #true gradient direction at W_original.
    print(f"\n[Step 3] Gradient inversion using W_original "
          f"({INV_RESTARTS} restart(s) x {INV_ITERATIONS} iters on CPU)...")
    t0 = time.time()
    reconstructed = gradient_inversion(original_model, clean_grad, target_grad,
                                       target_label)
    inv_time = time.time() - t0
    print(f"  Inversion done ({inv_time:.0f}s)")

    #Compute metrics
    original_img = private_data[target_idx][0]
    mse, psnr = compute_metrics(original_img, reconstructed.squeeze(0))

    print(f"\n{'-' * 60}")
    print(f"  MSE:  {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  (Paper reference: MSE ~ 0.0005, PSNR ~ 33 dB)")
    print(f"{'-' * 60}")

    #WandB logging
    wandb.log({
        "attack/mse": mse,
        "attack/psnr": psnr,
        "attack/inversion_time_s": inv_time,
        "attack/retrain_time_s": rt_time,
        "attack/target_client": target,
        "attack/target_label": target_label,
        "attack/n_target_rounds": len(target_rounds),
        "attack/clean_grad_norm": clean_norm,
        "attack/target_grad_norm": target_norm,
        "attack/cos_clean_target": ct_cos,
        "attack/orig_acc": orig_acc,
        "attack/unlearned_acc": unl_acc,
    })

    #Visualization: side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    orig_dn = denormalize(original_img).squeeze().numpy()
    recon_dn = denormalize(reconstructed.squeeze(0)).squeeze().numpy()

    axes[0].imshow(orig_dn, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original (label={target_label})", fontsize=14)
    axes[0].axis("off")
    axes[1].imshow(recon_dn, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"FUIA Reconstruction\nMSE={mse:.6f}  PSNR={psnr:.2f} dB",
                      fontsize=14)
    axes[1].axis("off")
    plt.suptitle("FUIA Client Unlearning Attack — MNIST + Retraining", fontsize=16)
    plt.tight_layout()

    #Save into the project folder (the parent of this src/ package), matching
    #the original script's output location.
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "fuia_result.png")
    plt.savefig(save_path, dpi=150)
    wandb.log({"attack/result": wandb.Image(fig)})
    print(f"  Saved to {save_path}")
    plt.show()

    return reconstructed, mse, psnr
