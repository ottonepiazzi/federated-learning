import torch
import torch.nn as nn
import copy

from config import (SEED, INV_DEVICE, INV_GAMMA, INV_ALPHA, INV_ITERATIONS,
                    INV_LR, INV_RESTARTS)


#FUIA Step 1: Gradient Separation (paper Eq. 16)
def gradient_separation(stored_updates, target_client):
    target_rounds = sorted({t for (t, k) in stored_updates if k == target_client})
    if not target_rounds:
        raise ValueError(f"Client {target_client} never participated")

    clean_grad = None
    for t in target_rounds:
        #All clients that participated in round t
        round_keys = [(rt, k) for (rt, k) in stored_updates if rt == t]
        total_l1 = sum(
            sum(v.abs().sum().item() for v in stored_updates[(t, k)].values())
            for (_, k) in round_keys
        )
        target_l1 = sum(
            v.abs().sum().item() for v in stored_updates[(t, target_client)].values()
        )
        gamma_t = target_l1 / (total_l1 + 1e-12)

        if clean_grad is None:
            clean_grad = {key: gamma_t * v.clone()
                          for key, v in stored_updates[(t, target_client)].items()}
        else:
            for key in clean_grad:
                clean_grad[key] += gamma_t * stored_updates[(t, target_client)][key]

    return clean_grad


#FUIA Step 2: Target Gradient Acquisition (paper Eq. 17)
def target_gradient_acquisition(original_model, unlearned_model):
    #Psi = W_original - W_unlearned. Materialize on CPU regardless of where
    #the source models live
    #inversion will move tensors to INV_DEVICE itself
    wo = original_model.state_dict()
    wu = unlearned_model.state_dict()
    return {k: (wo[k].float() - wu[k].float()).detach().cpu() for k in wo}


#FUIA Step 3: Gradient Inversion (paper Section V.B)
def total_variation(x):
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).sum()
    return dh + dw


def per_layer_cosine_distance(ga, gb, keys):
    #Per-layer cosine distance (from Geiping et al. "Inverting Gradients")
    total = torch.tensor(0.0, device=ga[keys[0]].device)
    for k in keys:
        a = ga[k].flatten().unsqueeze(0)
        b = gb[k].flatten().unsqueeze(0)
        cos = nn.functional.cosine_similarity(a, b)
        total = total + (1.0 - cos)
    return total / len(keys)


def gradient_inversion(model_for_inversion, clean_grad, target_grad, label,
                       gamma=INV_GAMMA, alpha=INV_ALPHA,
                       n_iters=INV_ITERATIONS, lr=INV_LR,
                       n_restarts=INV_RESTARTS):

    #Inversion needs create_graph=True through MaxPool2d
    #CUDA: supported -> run on GPU
    #MPS:  unsupported -> INV_DEVICE was set to CPU during startup
    #CPU:  CPU
    device = INV_DEVICE

    #Both V_k and Psi are PARAMETER-UPDATE directions, not gradient directions:
    #  * V_k = sum of target client's per-round updates (~ -lr * grad)
    #  * Psi = W_orig - W_unlearned: target's training pushed W_orig in the
    #    -grad direction relative to W_unlearned, so Psi ~ -eps * grad
    #We negate both so that the cosine-distance loss aligns the dummy
    #image's gradient with +grad (the true gradient at W_original on the
    #forgotten sample)
    clean_d  = {k: -v.to(device).detach() for k, v in clean_grad.items()}
    target_d = {k: -v.to(device).detach() for k, v in target_grad.items()}
    keys = sorted(clean_d.keys())

    #Valid pixel range after MNIST normalization
    norm_min = (0.0 - 0.1307) / 0.3081   # ~ -0.4242
    norm_max = (1.0 - 0.1307) / 0.3081   # ~  2.8215

    loss_fn = nn.CrossEntropyLoss()
    dummy_label = torch.tensor([label], device=device)

    best_img = None
    best_loss = float('inf')

    for r in range(n_restarts):
        torch.manual_seed(SEED + r * 7919)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED + r * 7919)
        x = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)
        opt = torch.optim.Adam([x], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_iters, eta_min=lr * 0.01)

        model = copy.deepcopy(model_for_inversion).to(device).eval()

        local_best_loss = float('inf')
        local_best_img = x.detach().clone()

        for it in range(1, n_iters + 1):
            opt.zero_grad()

            #Enable param grads for create_graph
            for p in model.parameters():
                p.requires_grad_(True)
            model.zero_grad()

            out = model(x)
            loss = loss_fn(out, dummy_label)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            g_dict = {}
            for (name, _), g in zip(model.named_parameters(), grads):
                g_dict[name] = g

            #Per-layer cosine distance (each layer contributes equally)
            dist_c = per_layer_cosine_distance(g_dict, clean_d, keys)
            dist_t = per_layer_cosine_distance(g_dict, target_d, keys)

            inv_loss = (1 - gamma) * dist_c + gamma * dist_t + alpha * total_variation(x)

            #Disable param grads before backward to save compute
            for p in model.parameters():
                p.requires_grad_(False)

            inv_loss.backward()
            opt.step()
            sched.step()

            with torch.no_grad():
                x.clamp_(norm_min, norm_max)

            l = inv_loss.item()
            if l < local_best_loss:
                local_best_loss = l
                local_best_img = x.detach().clone()

            if it % 1000 == 0:
                avg_cos_c = 1.0 - dist_c.item()
                avg_cos_t = 1.0 - dist_t.item()
                print(f"    [Restart {r+1}/{n_restarts}] Iter {it:5d}/{n_iters} | "
                      f"Loss: {l:.4f} | avg_cos_clean: {avg_cos_c:.4f} | "
                      f"avg_cos_target: {avg_cos_t:.4f}")

        if local_best_loss < best_loss:
            best_loss = local_best_loss
            best_img = local_best_img
            print(f"  -> Restart {r+1}: best loss = {local_best_loss:.4f}")

    #Always hand back a CPU tensor: downstream callers compare to the original
    #image (CPU) and pass it to matplotlib, which both require host memory
    return best_img.detach().cpu()
