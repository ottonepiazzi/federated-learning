import os
import torch
import torch.nn as nn
import copy

from config import SEED, INV_GAMMA, INV_ALPHA, INV_ITERATIONS, INV_LR, INV_RESTARTS


#Pick a device for gradient inversion (which requires double-backward through
#MaxPool2d). CUDA supports it; MPS does not, so on Apple Silicon we fall back
#to CPU. Set env var FUIA_INVERSION_CPU=1 to force CPU even when CUDA exists.
def inversion_device():
    if os.environ.get("FUIA_INVERSION_CPU") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gradient_inversion(model_for_inversion, clean_grad, target_grad, label,
                       gamma=INV_GAMMA, alpha=INV_ALPHA,
                       n_iters=INV_ITERATIONS, lr=INV_LR,
                       n_restarts=INV_RESTARTS):

    from torch.func import functional_call, grad, vmap

    #CUDA supports double-backward through MaxPool2d; MPS does not. Use CUDA
    #when available, otherwise CPU. (Apple Silicon stays on CPU here.)
    device = inversion_device()

    #Both V_k and Psi are PARAMETER-UPDATE directions, not gradient directions:
    #  * V_k = sum of target client's per-round updates (~ -lr * grad).
    #  * Psi = W_orig - W_unlearned: target's training pushed W_orig in the
    #    -grad direction relative to W_unlearned, so Psi ~ -eps * grad.
    #We negate both so that the cosine-distance loss aligns the dummy
    #image's gradient with +grad (the true gradient at W_original on the
    #forgotten sample).
    clean_d  = {k: -v.to(device).detach() for k, v in clean_grad.items()}
    target_d = {k: -v.to(device).detach() for k, v in target_grad.items()}
    keys = sorted(clean_d.keys())
    clean_flat  = {k: clean_d[k].flatten().unsqueeze(0)  for k in keys}
    target_flat = {k: target_d[k].flatten().unsqueeze(0) for k in keys}

    #Valid pixel range after MNIST normalization
    norm_min = (0.0 - 0.1307) / 0.3081   # ~ -0.4242
    norm_max = (1.0 - 0.1307) / 0.3081   # ~  2.8215

    loss_fn = nn.CrossEntropyLoss()

    #All n_restarts dummy images are stacked along dim 0 and optimized together.
    #vmap+grad below computes a separate per-parameter gradient for each one,
    #so the restarts remain independent but run as a single batched workload.
    init_xs = []
    for r in range(n_restarts):
        torch.manual_seed(SEED + r * 7919)
        init_xs.append(torch.randn(1, 1, 28, 28, device=device))
    x = torch.cat(init_xs, dim=0).detach().requires_grad_(True)
    labels = torch.full((n_restarts,), label, dtype=torch.long, device=device)

    opt = torch.optim.Adam([x], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_iters, eta_min=lr * 0.01)

    model = copy.deepcopy(model_for_inversion).to(device).eval()
    params  = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def single_loss(p, b, x_one, y_one):
        out = functional_call(model, (p, b), (x_one.unsqueeze(0),))
        return loss_fn(out, y_one.unsqueeze(0))

    per_restart_grads = vmap(grad(single_loss, argnums=0),
                             in_dims=(None, None, 0, 0))

    best_loss = torch.full((n_restarts,), float('inf'), device=device)
    best_imgs = x.detach().clone()
    sync_every = 50  # how often to refresh best-image tracking on the host

    for it in range(1, n_iters + 1):
        opt.zero_grad()

        #Per-restart param-gradient dict (each entry has leading batch dim R).
        g = per_restart_grads(params, buffers, x, labels)

        dist_c = torch.zeros(n_restarts, device=device)
        dist_t = torch.zeros(n_restarts, device=device)
        for k in keys:
            gk = g[k].reshape(n_restarts, -1)
            dist_c = dist_c + (1.0 - nn.functional.cosine_similarity(gk, clean_flat[k],  dim=1))
            dist_t = dist_t + (1.0 - nn.functional.cosine_similarity(gk, target_flat[k], dim=1))
        dist_c = dist_c / len(keys)
        dist_t = dist_t / len(keys)

        dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum(dim=(1, 2, 3))
        dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).sum(dim=(1, 2, 3))
        tv = dh + dw

        per_loss = (1 - gamma) * dist_c + gamma * dist_t + alpha * tv
        per_loss.sum().backward()
        opt.step()
        sched.step()

        with torch.no_grad():
            x.clamp_(norm_min, norm_max)

            #Best-image tracking stays on-device and only syncs every sync_every
            #iters; per-iter .item() calls would force a host sync and serialize
            #the GPU pipeline.
            if it % sync_every == 0 or it == n_iters:
                detached = per_loss.detach()
                improved = detached < best_loss
                best_imgs = torch.where(
                    improved.view(n_restarts, 1, 1, 1),
                    x.detach(),
                    best_imgs,
                )
                best_loss = torch.where(improved, detached, best_loss)

        if it % 1000 == 0:
            with torch.no_grad():
                avg_cos_c   = (1.0 - dist_c.mean()).item()
                avg_cos_t   = (1.0 - dist_t.mean()).item()
                losses_list = per_loss.detach().cpu().tolist()
            losses_str = ", ".join(f"{v:.4f}" for v in losses_list)
            print(f"    Iter {it:5d}/{n_iters} | losses=[{losses_str}] | "
                  f"avg_cos_clean: {avg_cos_c:.4f} | avg_cos_target: {avg_cos_t:.4f}")

    with torch.no_grad():
        best_idx = best_loss.argmin().item()
        best_img = best_imgs[best_idx:best_idx + 1]
        print(f"  -> Best restart: {best_idx + 1}/{n_restarts}, "
              f"loss = {best_loss[best_idx].item():.4f}")

    #Return on CPU so downstream metrics/plotting (which use CPU tensors from
    #the dataset and call .numpy()) don't hit a device mismatch.
    return best_img.detach().cpu()
