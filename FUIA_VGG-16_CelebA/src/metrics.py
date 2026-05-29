import torch
import numpy as np

from config import IMG_CHANNELS, IMG_MEAN, IMG_STD


#Metrics and Visualization
def denormalize(image_tensor):
    #Convert from normalized CelebA space back to [0, 1] pixel range
    img = image_tensor.cpu().float()
    for c in range(IMG_CHANNELS):
        img[c] = img[c] * IMG_STD[c] + IMG_MEAN[c]
    return img.clamp(0, 1)


#MSE and PSNR between original and reconstructed images (in [0,1] space)
def compute_metrics(original_image, reconstructed_image):
    original = denormalize(original_image)
    reconstructed = denormalize(reconstructed_image)
    mse = torch.mean((original - reconstructed) ** 2).item()
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))
    return mse, psnr
