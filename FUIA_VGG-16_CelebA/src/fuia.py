import torch
import torch.nn as nn
import copy

from config import (SEED, DEVICE, IMG_SIZE, IMG_CHANNELS, IMG_MEAN, IMG_STD,
                    INV_ITERATIONS, INV_LR, INV_ALPHA, INV_RESTARTS)


#FUIA ATTACK (FUIA for Sample Unlearning)

#Step 1: Gradient Separation (paper Eq. 8-11)
def gradient_separation(stored_updates, round_l1_norms, target_client):
    """Eq. 8-11: extract clean gradient for target client using stored updates.
    Uses pre-computed L1 norm sums to avoid storing all clients' full updates."""

    #find rounds where target client participated
    rounds_with_target = sorted([rnd for (rnd, cid, *rest) in stored_updates.keys()
                                 if cid == target_client and not rest])
    if not rounds_with_target:
        raise ValueError(f"Client {target_client} never participated")

    clean_gradient = None
    for rnd in rounds_with_target:
        target_update = stored_updates[(rnd, target_client)]

        #Eq. 9: total L1 norm sum (pre-computed during FL training)
        l1_norm_sum = round_l1_norms[rnd]

        #Eq. 10: weight for target client
        l1_key = (rnd, target_client, 'l1')
        if l1_key in stored_updates:
            target_l1 = stored_updates[l1_key]
        else:
            target_l1 = float(sum(v.float().abs().sum() for v in target_update.values()))
        weight = target_l1 / (l1_norm_sum + 1e-12)

        #Eq. 11: accumulate weighted update
        if clean_gradient is None:
            clean_gradient = {key: weight * val.float().clone()
                              for key, val in target_update.items()}
        else:
            for key in clean_gradient:
                clean_gradient[key] += weight * target_update[key].float()

    return clean_gradient


#Step 2: Target Gradient Acquisition (paper Eq. 12)
def target_gradient_acquisition(clean_gradient_fl, clean_gradient_fu):
    target_gradient = {key: clean_gradient_fl[key] - clean_gradient_fu[key]
                       for key in clean_gradient_fl}
    return target_gradient


#Step 3: Gradient Inversion (paper Eq. 13-14)
def total_variation(image):
    #Anisotropic total variation (Eq. 15)
    diff_h = (image[:, :, 1:, :] - image[:, :, :-1, :]).pow(2).sum()
    diff_w = (image[:, :, :, 1:] - image[:, :, :, :-1]).pow(2).sum()
    return diff_h + diff_w


#Cosine similarity between two gradient dicts, flattened and concatenated
def cosine_similarity_gradients(grad_a, grad_b, param_keys):
    flat_a = torch.cat([grad_a[k].flatten() for k in param_keys])
    flat_b = torch.cat([grad_b[k].flatten() for k in param_keys])
    return nn.functional.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0))


def gradient_inversion(original_model, target_gradient, label):
    #Reconstruct the forgotten sample via gradient matching on W^o
    device = DEVICE
    param_keys = sorted(target_gradient.keys())

    #negate target gradient: stored updates are in negative-gradient space (important!)
    target_on_device = {k: -v.to(device).detach() for k, v in target_gradient.items()}

    #pixel bounds in normalized space (per channel)
    pixel_min = (0.0 - IMG_MEAN[0]) / IMG_STD[0]  #all channels same with (0.5,0.5,0.5)
    pixel_max = (1.0 - IMG_MEAN[0]) / IMG_STD[0]

    loss_fn = nn.CrossEntropyLoss()
    label_tensor = torch.tensor([label], device=device)

    #use original model W^o (Eq. 14)
    model = copy.deepcopy(original_model).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(True)

    best_image = None
    best_cosine = float('-inf')

    for restart in range(INV_RESTARTS):
        torch.manual_seed(SEED + restart * 7919)
        virtual_image = torch.randn(1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE,
                                    device=device, requires_grad=True)
        optimizer = torch.optim.Adam([virtual_image], lr=INV_LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=INV_ITERATIONS, eta_min=INV_LR * 0.01)

        restart_best_cosine = float('-inf')
        restart_best_image = virtual_image.detach().clone()

        for iteration in range(1, INV_ITERATIONS + 1):
            optimizer.zero_grad()
            model.zero_grad()

            #Eq. 14: virtual gradient on W^o
            output = model(virtual_image)
            classification_loss = loss_fn(output, label_tensor)
            virtual_grads = torch.autograd.grad(
                classification_loss, model.parameters(), create_graph=True)
            virtual_grad_dict = {name: grad for (name, _), grad in
                                 zip(model.named_parameters(), virtual_grads)}

            #Eq. 13: min -cos_sim + alpha * TV
            cos_sim = cosine_similarity_gradients(
                virtual_grad_dict, target_on_device, param_keys)
            inversion_loss = -cos_sim + INV_ALPHA * total_variation(virtual_image)

            inversion_loss.backward()
            optimizer.step()
            scheduler.step()

            #clamp to valid pixel range
            with torch.no_grad():
                virtual_image.clamp_(pixel_min, pixel_max)

            current_cosine = cos_sim.item()
            if current_cosine > restart_best_cosine:
                restart_best_cosine = current_cosine
                restart_best_image = virtual_image.detach().clone()

            if iteration % 2000 == 0:
                print(f"    [Restart {restart+1}/{INV_RESTARTS}] "
                      f"Iter {iteration:5d}/{INV_ITERATIONS} | "
                      f"loss: {inversion_loss.item():.6f} | "
                      f"cos_sim: {current_cosine:.4f}")

        if restart_best_cosine > best_cosine:
            best_cosine = restart_best_cosine
            best_image = restart_best_image
            print(f"  -> Restart {restart+1}: best cos_sim = {restart_best_cosine:.4f}")

    return best_image
