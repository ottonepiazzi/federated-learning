import torch
import torch.nn as nn
import copy

from config import SEED, DEVICE, INV_ITERATIONS, INV_LR, INV_ALPHA, INV_RESTARTS


def gradient_separation(stored_updates, target_client):
    #group all updates by round: {round_id: {client_id: param_update}}
    updates_by_round = {}
    for (round_id, client_id), param_update in stored_updates.items():
        updates_by_round.setdefault(round_id, {})[client_id] = param_update

    #find rounds where target client participated
    rounds_with_target = sorted([round_id for round_id, clients in updates_by_round.items()
                                 if target_client in clients])
    if not rounds_with_target:
        raise ValueError(f"Client {target_client} never participated")

    def l1_norm(param_update):
        #L1 norm of a parameter update vector (Eq. 9)
        return float(sum(v.float().abs().sum() for v in param_update.values()))

    clean_gradient = None
    for round_id in rounds_with_target:
        round_updates = updates_by_round[round_id]

        #Eq. 9: sum of L1 norms of all clients in this round
        l1_norm_sum = sum(l1_norm(update) for update in round_updates.values())

        #Eq. 10: weight for target client in this round
        weight = l1_norm(round_updates[target_client]) / (l1_norm_sum + 1e-12)

        #Eq. 11: accumulate weighted update
        target_update = round_updates[target_client]
        if clean_gradient is None:
            clean_gradient = {key: weight * val.float().clone()
                              for key, val in target_update.items()}
        else:
            for key in clean_gradient:
                clean_gradient[key] += weight * target_update[key].float()

    return clean_gradient


def target_gradient_acquisition(clean_gradient_fl, clean_gradient_fu):
    target_gradient = {key: clean_gradient_fl[key] - clean_gradient_fu[key]
                       for key in clean_gradient_fl}
    return target_gradient


def total_variation(image):
    #Anisotropic total variation (Eq. 15)
    diff_h = (image[:, :, 1:, :] - image[:, :, :-1, :]).pow(2).sum()
    diff_w = (image[:, :, :, 1:] - image[:, :, :, :-1]).pow(2).sum()
    return diff_h + diff_w


def cosine_similarity_gradients(grad_a, grad_b, param_keys):
    #Cosine similarity between two gradient dicts, flattened and concatenated
    flat_a = torch.cat([grad_a[k].flatten() for k in param_keys])
    flat_b = torch.cat([grad_b[k].flatten() for k in param_keys])
    return nn.functional.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0))


def gradient_inversion_batch(original_model, target_gradient, labels):
    #Batch gradient inversion (paper Sec V.A Step 3, generalised to several
    #forgotten samples per client — the setting studied in the Fig. 9 ablation).
    #
    #When a client forgets N samples, the target gradient nabla_k is the AGGREGATE
    #gradient of those N samples (gradients are additive over samples). We recover
    #them by optimising a BATCH of N virtual images jointly, so that the gradient
    #of their summed loss matches nabla_k in direction (cosine). All N images are
    #optimised at once in one shared objective — there is no sequential pass from
    #one image to the next; distinct images emerge from their (known) labels and
    #their different random initialisations. The slot<->sample correspondence is
    #arbitrary (permutation ambiguity), which is why callers match reconstructions
    #to originals afterwards.
    device = DEVICE
    param_keys = sorted(target_gradient.keys())
    n = len(labels)

    #move target gradient to device and negate
    target_on_device = {k: -v.to(device).detach() for k, v in target_gradient.items()}

    #pixel bounds in normalized space
    pixel_min = (0.0 - 0.1307) / 0.3081
    pixel_max = (1.0 - 0.1307) / 0.3081

    loss_fn = nn.CrossEntropyLoss()
    label_tensor = torch.tensor(list(labels), device=device)

    #use original model W^o for virtual gradient computation (Eq. 14)
    model = copy.deepcopy(original_model).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(True)

    best_batch = None
    best_cosine = float('-inf')

    for restart in range(INV_RESTARTS):
        torch.manual_seed(SEED + restart * 7919)
        virtual_batch = torch.randn(n, 1, 28, 28, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([virtual_batch], lr=INV_LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=INV_ITERATIONS, eta_min=INV_LR * 0.01)

        restart_best_cosine = float('-inf')
        restart_best_batch = virtual_batch.detach().clone()

        for iteration in range(1, INV_ITERATIONS + 1):
            optimizer.zero_grad()
            model.zero_grad()

            #virtual gradient of the summed batch loss (Eq. 14). For a batch the
            #mean-reduced CE gradient is proportional to the sum of per-sample
            #gradients; cosine similarity is scale-invariant so this matches the
            #aggregate target gradient's direction regardless of the 1/N factor.
            output = model(virtual_batch)
            classification_loss = loss_fn(output, label_tensor)
            virtual_grads = torch.autograd.grad(
                classification_loss, model.parameters(), create_graph=True)
            virtual_grad_dict = {name: grad for (name, _), grad in
                                 zip(model.named_parameters(), virtual_grads)}

            #Eq. 13
            cos_sim = cosine_similarity_gradients(
                virtual_grad_dict, target_on_device, param_keys)
            inversion_loss = -cos_sim + INV_ALPHA * total_variation(virtual_batch)

            inversion_loss.backward()
            optimizer.step()
            scheduler.step()

            #clamp to valid pixel range
            with torch.no_grad():
                virtual_batch.clamp_(pixel_min, pixel_max)

            #track best result by cosine similarity
            current_cosine = cos_sim.item()
            if current_cosine > restart_best_cosine:
                restart_best_cosine = current_cosine
                restart_best_batch = virtual_batch.detach().clone()

            if iteration % 2000 == 0:
                print(f"    [Restart {restart+1}/{INV_RESTARTS}] "
                      f"Iter {iteration:5d}/{INV_ITERATIONS} | "
                      f"loss: {inversion_loss.item():.6f} | "
                      f"cos_sim: {current_cosine:.4f}")

        if restart_best_cosine > best_cosine:
            best_cosine = restart_best_cosine
            best_batch = restart_best_batch
            print(f"  -> Restart {restart+1}: best cos_sim = {restart_best_cosine:.4f}")

    return best_batch, best_cosine


def gradient_inversion(original_model, target_gradient, label):
    #Backward-compatible single-image entry point (N=1 baseline). Returns a
    #(1, 1, 28, 28) tensor exactly as before — used by main.py / the monolithic
    #script. For N=1 this is numerically identical to the previous implementation
    #(same seeds, same torch.randn(1, 1, 28, 28) draw).
    best_batch, _ = gradient_inversion_batch(original_model, target_gradient, [label])
    return best_batch
