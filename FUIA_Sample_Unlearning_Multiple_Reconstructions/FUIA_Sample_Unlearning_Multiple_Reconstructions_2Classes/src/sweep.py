#!/usr/bin/env python3
#
# Ablation sweep: FUIA against sample unlearning as the NUMBER OF FORGOTTEN DATA
# per client grows (paper Sec VII.A.2 / Fig. 9, reproduced on MNIST).
#
# Pipeline: run federated learning ONCE, then for every N in SWEEP_FORGET_COUNTS
# make each client forget its first N samples (nested, anchored on the N=1
# sample), retrain the unlearned model, and run FUIA to reconstruct the target
# client's N forgotten images jointly (batch gradient inversion). We log per-image
# and aggregate PSNR/MSE and finally plot PSNR-vs-N (mean over the N reconstructed
# images + the fixed anchor image).
#
# Quick smoke test (cheap, low quality) from inside src/:
#   FUIA_INV_ITERATIONS=60 FUIA_INV_RESTARTS=2 WANDB_MODE=disabled python sweep.py
# Full run (reproduces the baseline at N=1) from inside src/:
#   python sweep.py

import os
import csv
import json
import time

import wandb

import config as cfg
from config import DEVICE, SWEEP_FORGET_COUNTS, DATA_PER_CLIENT
from federated import evaluate
from fl_training import run_fl_training
from unlearning import build_forget_order, build_forget_sets, retrain_without_sample_sets
from fuia import gradient_separation
from attack import attack_forgotten_set, select_target_client
from sweep_plots import plot_recon_grid, plot_sweep_curves, plot_mean_psnr_mse


#WandB is optional. By default WANDB_MODE=disabled, which starts a no-op run that
#needs no login — so every wandb.log across the codebase (incl. fl_training) is a
#harmless no-op and the sweep just saves its figures/CSV locally. Set
#WANDB_MODE=online (once `wandb login` is done) to actually log; if that init
#fails we fall back to a disabled run so the sweep still completes.
_WANDB_ON = True


def _wandb_init(config):
    global _WANDB_ON
    mode = os.environ.get("WANDB_MODE", "disabled")
    try:
        wandb.init(project="FUIA", config=config, mode=mode)
        _WANDB_ON = (mode != "disabled")
    except Exception as e:
        print(f"  [WandB '{mode}' failed ({type(e).__name__}: {e}); falling back to disabled]")
        wandb.init(project="FUIA", config=config, mode="disabled")
        _WANDB_ON = False


def _wandb_log(payload):
    #Safe even in disabled mode: wandb.init has always run, so wandb.log no-ops.
    wandb.log(payload)


def _wandb_finish():
    wandb.finish()


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "sweep_results")
CACHE_PATH = os.path.join(OUT_DIR, "records.json")
CSV_PATH = os.path.join(OUT_DIR, "sweep_metrics.csv")
PSNR_PATH = os.path.join(OUT_DIR, "forget_sweep_psnr.png")
MSE_PATH = os.path.join(OUT_DIR, "forget_sweep_mse.png")
MEAN_PATH = os.path.join(OUT_DIR, "forget_sweep_mean_psnr_mse.png")


def _load_cache():
    #Completed sweep points from a previous run, keyed by N. Everything upstream
    #(FL run, target client, forget order) is seeded, so cached N stay valid.
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH) as f:
        return {int(r["n"]): r for r in json.load(f)}


def _write_outputs(records_by_n):
    #Rewrite cache + CSV + curves from all completed points. Called after every N
    #so an interrupted run still leaves a usable CSV and curve.
    records = [records_by_n[n] for n in sorted(records_by_n)]
    with open(CACHE_PATH, "w") as f:
        json.dump(records, f, indent=2)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    plot_sweep_curves(records, PSNR_PATH, MSE_PATH)
    plot_mean_psnr_mse(records, MEAN_PATH)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    config = {k: v for k, v in vars(cfg).items()
              if k.isupper() and isinstance(v, (int, float, str))}
    config.update({"scenario": "sample_unlearning", "unlearning_method": "retraining",
                   "dataset": "MNIST_binary", "experiment": "forget_count_sweep"})
    _wandb_init(config)

    total_start = time.time()

    #Phase 1: Federated Learning (shared by every sweep point)
    print("=" * 60 + "\nPhase 1: Federated Learning (shared across sweep)\n" + "=" * 60)
    (original_model, stored_updates_fl, client_data, private_data,
     pretrained_sd, round_selections, test_loader) = run_fl_training()

    original_acc = evaluate(original_model, test_loader, DEVICE)
    target_client = select_target_client(stored_updates_fl, round_selections)
    print(f"\n  Original model accuracy: {original_acc * 100:.1f}%")
    print(f"  Target client: {target_client}")

    #Fixed nested forget order (anchored so N=1 == the baseline target image)
    forget_order = build_forget_order(client_data)

    #FL-side clean gradient of the target client is independent of N -> compute once
    print("  Precomputing clean_gradient_fl for the target client (reused for all N)")
    clean_gradient_fl = gradient_separation(stored_updates_fl, target_client)

    #Only sweep values that leave each client at least one sample
    forget_counts = sorted({n for n in SWEEP_FORGET_COUNTS if 1 <= n < DATA_PER_CLIENT})
    records_by_n = _load_cache()
    if records_by_n:
        print(f"  Resuming: found cached results for N={sorted(records_by_n)}")
    print(f"  Sweeping N in {forget_counts} "
          f"(DATA_PER_CLIENT={DATA_PER_CLIENT}, anchor idx={forget_order[target_client][0]})")

    for n_forget in forget_counts:
        grid_path = os.path.join(OUT_DIR, f"recon_N{n_forget}.png")
        if n_forget in records_by_n and os.path.exists(grid_path):
            print(f"\n  N = {n_forget}: reusing cached result (skip recompute)")
            continue

        print("\n" + "=" * 60 + f"\nSweep point: N = {n_forget} forgotten sample(s) per client\n" + "=" * 60)

        #Phase 2: retrain the unlearned model with N samples removed per client
        forget_sets = build_forget_sets(forget_order, n_forget)
        unlearned_model, stored_updates_fu = retrain_without_sample_sets(
            pretrained_sd, private_data, client_data, forget_sets, round_selections)
        unlearned_acc = evaluate(unlearned_model, test_loader, DEVICE)
        print(f"  Unlearned model accuracy: {unlearned_acc * 100:.1f}%")

        #Phase 3: FUIA batch attack on the target client's N forgotten samples
        forgotten_indices = forget_sets[target_client]
        result = attack_forgotten_set(
            original_model, stored_updates_fl, stored_updates_fu, private_data,
            target_client, forgotten_indices, round_selections,
            clean_gradient_fl=clean_gradient_fl)

        anchor = result["match"]["per_image"][0]      #original 0 == the anchor image
        record = {
            "n": n_forget,
            "mean_psnr": result["match"]["mean_psnr"],
            "best_psnr": result["match"]["best_psnr"],
            "worst_psnr": result["match"]["worst_psnr"],
            "mean_mse": result["match"]["mean_mse"],
            "anchor_psnr": anchor["psnr"],
            "anchor_mse": anchor["mse"],
            "best_cosine": result["best_cosine"],
            "target_gradient_norm": result["target_gradient_norm"],
            "unlearned_acc": unlearned_acc,
        }
        records_by_n[n_forget] = record

        plot_recon_grid(result, grid_path)
        print(f"  Saved reconstruction grid -> {grid_path}")

        #Persist after every N so an interrupted run keeps its CSV + curve + cache
        _write_outputs(records_by_n)
        print(f"  Updated CSV + curves ({len(records_by_n)} point(s) so far)")

        _wandb_log({"sweep/N": n_forget,
                    "sweep/mean_psnr": record["mean_psnr"],
                    "sweep/anchor_psnr": record["anchor_psnr"],
                    "sweep/best_psnr": record["best_psnr"],
                    "sweep/worst_psnr": record["worst_psnr"],
                    "sweep/mean_mse": record["mean_mse"],
                    "sweep/anchor_mse": record["anchor_mse"],
                    "sweep/best_cosine": record["best_cosine"],
                    "sweep/target_gradient_norm": record["target_gradient_norm"]})

    #Final rewrite (covers the all-cached case where the loop computed nothing)
    _write_outputs(records_by_n)

    print("\n" + "=" * 60 + "\nSweep summary (target image held fixed across N)\n" + "=" * 60)
    print(f"  {'N':>3} | {'mean PSNR':>9} | {'anchor PSNR':>11} | {'mean MSE':>9} | {'cos':>6}")
    for n in sorted(records_by_n):
        r = records_by_n[n]
        print(f"  {r['n']:>3} | {r['mean_psnr']:>9.2f} | {r['anchor_psnr']:>11.2f} | "
              f"{r['mean_mse']:>9.4f} | {r['best_cosine']:>6.3f}")

    if _WANDB_ON:
        _wandb_log({"sweep/psnr_curve": wandb.Image(PSNR_PATH),
                    "sweep/mse_curve": wandb.Image(MSE_PATH),
                    "sweep/mean_psnr_mse_curve": wandb.Image(MEAN_PATH)})
    print(f"\n  CSV:        {CSV_PATH}")
    print(f"  PSNR curve: {PSNR_PATH}")
    print(f"  MSE curve:  {MSE_PATH}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    _wandb_log({"total_time_s": total_time})
    _wandb_finish()


if __name__ == "__main__":
    main()
