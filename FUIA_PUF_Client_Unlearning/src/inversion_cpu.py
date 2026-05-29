import torch
import torch.nn as nn
import copy

from config import SEED, INV_GAMMA, INV_ALPHA, INV_ITERATIONS, INV_LR, INV_RESTARTS
from fuia_steps import total_variation, per_layer_cosine_distance


def gradient_inversion(model_for_inversion, clean_grad, target_grad, label,
                       gamma=INV_GAMMA, alpha=INV_ALPHA,
                       n_iters=INV_ITERATIONS, lr=INV_LR,
                       n_restarts=INV_RESTARTS):

    #Force CPU for 2nd-order gradient support with MaxPool2d
    device = torch.device("cpu")

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

    #Valid pixel range after MNIST normalization
    norm_min = (0.0 - 0.1307) / 0.3081   # ~ -0.4242
    norm_max = (1.0 - 0.1307) / 0.3081   # ~  2.8215

    loss_fn = nn.CrossEntropyLoss()
    dummy_label = torch.tensor([label], device=device)

    best_img = None
    best_loss = float('inf')

    for r in range(n_restarts):
        torch.manual_seed(SEED + r * 7919)
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

    return best_img
