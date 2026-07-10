import os

import numpy as np
import matplotlib
matplotlib.use("Agg")               #headless: only ever saving figures here
import matplotlib.pyplot as plt

from metrics import denormalize


def plot_recon_grid(result, save_path, title=None):
    #Qualitative figure for a single sweep point: top row = the N original
    #forgotten images, bottom row = their matched reconstructions with per-image
    #PSNR. Column i pairs original i with the reconstruction assigned to it.
    n = result["n_forget"]
    originals = result["original_images"]
    batch = result["reconstructed_batch"]
    assignment = result["match"]["assignment"]
    per_image = {p["orig"]: p for p in result["match"]["per_image"]}
    labels = result["labels"]

    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.8), squeeze=False)
    for i in range(n):
        axes[0][i].imshow(denormalize(originals[i]).numpy(), cmap="gray", vmin=0, vmax=1)
        tag = " (anchor)" if i == 0 else ""
        axes[0][i].set_title(f"orig{tag}\nlabel={labels[i]}", fontsize=10)
        axes[0][i].axis("off")

        j = assignment[i]
        p = per_image[i]
        axes[1][i].imshow(denormalize(batch[j]).numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1][i].set_title(f"recon\nPSNR={p['psnr']:.1f} dB  MSE={p['mse']:.4f}", fontsize=10)
        axes[1][i].axis("off")

    axes[0][0].set_ylabel("original", fontsize=11)
    if title is None:
        title = f"FUIA reconstruction — {n} forgotten sample(s) per client"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_sweep_curves(records, psnr_path, mse_path):
    #The headline result: PSNR (and MSE) vs number of forgotten samples per
    #client. Two PSNR curves — mean over all reconstructed images and the fixed
    #anchor image ("at parity of target image") — plus a min/max band.
    ns = [r["n"] for r in records]
    mean_psnr = [r["mean_psnr"] for r in records]
    anchor_psnr = [r["anchor_psnr"] for r in records]
    best_psnr = [r["best_psnr"] for r in records]
    worst_psnr = [r["worst_psnr"] for r in records]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(ns, worst_psnr, best_psnr, alpha=0.15, color="tab:blue",
                    label="min–max over reconstructed images")
    ax.plot(ns, mean_psnr, "o-", color="tab:blue", label="mean PSNR (all N images)")
    ax.plot(ns, anchor_psnr, "s--", color="tab:red", label="anchor image (fixed N=1 target)")
    ax.set_xlabel("Number of forgotten samples per client (N)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("FUIA vs. number of forgotten data (Sample Unlearning, MNIST)")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(psnr_path, dpi=150)
    plt.close(fig)

    mean_mse = [r["mean_mse"] for r in records]
    anchor_mse = [r["anchor_mse"] for r in records]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ns, mean_mse, "o-", color="tab:blue", label="mean MSE (all N images)")
    ax.plot(ns, anchor_mse, "s--", color="tab:red", label="anchor image (fixed N=1 target)")
    ax.set_xlabel("Number of forgotten samples per client (N)")
    ax.set_ylabel("MSE")
    ax.set_title("FUIA vs. number of forgotten data (Sample Unlearning, MNIST)")
    ax.set_xticks(ns)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(mse_path, dpi=150)
    plt.close(fig)


def plot_mean_psnr_mse(records, path):
    #Single chart with just mean PSNR and mean MSE — no anchor line, no band.
    #The two metrics are on very different scales, so PSNR uses the left y-axis
    #and MSE the right y-axis (twin axes), both plotted against N.
    ns = [r["n"] for r in records]
    mean_psnr = [r["mean_psnr"] for r in records]
    mean_mse = [r["mean_mse"] for r in records]
    c_psnr, c_mse = "tab:blue", "tab:red"

    fig, ax1 = plt.subplots(figsize=(7, 5))
    l1, = ax1.plot(ns, mean_psnr, "o-", color=c_psnr, label="mean PSNR")
    ax1.set_xlabel("Number of forgotten samples per client (N)")
    ax1.set_ylabel("mean PSNR (dB)", color=c_psnr)
    ax1.tick_params(axis="y", labelcolor=c_psnr)
    ax1.set_xticks(ns)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    l2, = ax2.plot(ns, mean_mse, "s--", color=c_mse, label="mean MSE")
    ax2.set_ylabel("mean MSE", color=c_mse)
    ax2.tick_params(axis="y", labelcolor=c_mse)

    ax1.set_title("FUIA mean PSNR & MSE vs. number of forgotten data")
    ax1.legend(handles=[l1, l2], loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
