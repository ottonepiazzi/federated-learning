import torch


#Metrics
def denormalize(img, mean=0.1307, std=0.3081):
    return (img * std + mean).clamp(0, 1)


def compute_metrics(original, reconstructed):
    #MSE and PSNR on denormalized [0,1] images
    orig = denormalize(original)
    recon = denormalize(reconstructed)
    mse = torch.mean((orig - recon) ** 2).item()
    psnr = 10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-10))).item()
    return mse, psnr
