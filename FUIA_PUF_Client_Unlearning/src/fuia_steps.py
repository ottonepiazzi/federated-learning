import torch
import torch.nn as nn


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
    #Psi = W_original - W_unlearned
    wo = original_model.state_dict()
    wu = unlearned_model.state_dict()
    return {k: wo[k].float() - wu[k].float() for k in wo}


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
