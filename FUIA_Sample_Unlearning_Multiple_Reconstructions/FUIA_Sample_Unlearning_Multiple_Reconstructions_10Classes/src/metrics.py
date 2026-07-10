import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


#Metrics and Visualization
def denormalize(image_tensor):
    #Convert from normalized MNIST space back to [0, 1] pixel range
    return (image_tensor.cpu().float() * 0.3081 + 0.1307).clamp(0, 1).squeeze()


def compute_metrics(original_image, reconstructed_image):
    #MSE and PSNR between original and reconstructed images (in [0,1] space)
    original = denormalize(original_image)
    reconstructed = denormalize(reconstructed_image)
    mse = torch.mean((original - reconstructed) ** 2).item()
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))
    return mse, psnr


def match_reconstructions(original_images, reconstructed_images):
    #Match a set of N reconstructed images to the N original forgotten images.
    #
    #Batch gradient inversion recovers the N images up to permutation: virtual
    #slot j does not correspond to original i in any fixed way. We therefore find
    #the assignment (a bijection slot->original) that minimises total MSE via the
    #Hungarian algorithm, then report per-original MSE/PSNR under that assignment.
    #This is the aggregation the sweep graph is built on.
    #
    #original_images / reconstructed_images: lists of tensors (normalized space).
    #Returns: {"assignment": {orig_idx: recon_idx}, "per_image": [{mse,psnr}, ...]
    #          in original order, plus aggregates mean/best/worst PSNR and MSE}.
    n = len(original_images)
    cost = np.zeros((n, n))
    for i, orig in enumerate(original_images):
        for j, recon in enumerate(reconstructed_images):
            cost[i, j], _ = compute_metrics(orig, recon)

    row_idx, col_idx = linear_sum_assignment(cost)   #minimise total MSE
    assignment = {int(i): int(j) for i, j in zip(row_idx, col_idx)}

    per_image = []
    for i in range(n):
        j = assignment[i]
        mse, psnr = compute_metrics(original_images[i], reconstructed_images[j])
        per_image.append({"orig": i, "recon": j, "mse": mse, "psnr": psnr})

    psnrs = [p["psnr"] for p in per_image]
    mses = [p["mse"] for p in per_image]
    return {
        "assignment": assignment,
        "per_image": per_image,
        "mean_psnr": float(np.mean(psnrs)),
        "best_psnr": float(np.max(psnrs)),
        "worst_psnr": float(np.min(psnrs)),
        "mean_mse": float(np.mean(mses)),
    }
