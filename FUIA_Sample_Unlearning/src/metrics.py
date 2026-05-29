import torch
import numpy as np


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
