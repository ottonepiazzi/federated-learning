# FUIA - Sample Unlearning (CNN on MNIST)

Implementation of the FUIA (Federated Unlearning Inversion Attack) from the paper
"Model Inversion Attack Against Federated Unlearning" (Zhou et al., IEEE TIFS 2026),
sample unlearning scenario with Retraining on MNIST dataset.

## Setup

### 1. Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

### 2. Run

**macOS / Linux:**
```bash
python3 fuia_sample_unlearning.py
```

**Windows**
```bash
python fuia_sample_unlearning.py
```

At the beginning of each run you are asked to log metrics on WandB. If you want to skip this part, simply insert "3".

The src folder contains a modular version of the code.

## Ablation: number of forgotten data per client (paper Sec. VII.A.2, Fig. 9)

`src/sweep.py` reproduces the paper's ablation on the **number of forgotten
samples per client** for the sample-unlearning scenario. It runs federated
learning once, then for every `N` in `SWEEP_FORGET_COUNTS` (default `1..7`) makes
each client forget its first `N` samples, retrains the unlearned model, and runs
FUIA to reconstruct the target client's `N` forgotten images **jointly** (batch
gradient inversion). The forget sets are **nested and anchored**: the `N=1` set
is the single sample reconstructed in the baseline, and larger `N` only *adds*
samples — so the curve measures degradation "at parity of target image".

Because the `N` reconstructions come out in an arbitrary order, they are matched
to the originals with the Hungarian algorithm before scoring. Reported metrics:
mean PSNR/MSE over all `N` reconstructed images, plus the fixed **anchor** image
tracked separately.

Run from inside `src/`:
```bash
cd src
python sweep.py                       # full run (defaults reproduce N=1 baseline)
```

Outputs go to `sweep_results/`:
- `forget_sweep_psnr.png` / `forget_sweep_mse.png` — PSNR/MSE vs N (mean + anchor)
- `forget_sweep_mean_psnr_mse.png` — mean PSNR and mean MSE vs N on one chart (twin y-axes)
- `recon_N{n}.png` — per-N grid of originals vs matched reconstructions
- `sweep_metrics.csv` — all metrics

Environment overrides (no code edits needed):
- `FUIA_INV_ITERATIONS`, `FUIA_INV_RESTARTS` — cheaper/faster inversion (e.g. a
  quick smoke test: `FUIA_INV_ITERATIONS=60 FUIA_INV_RESTARTS=2 python sweep.py`)
- `WANDB_MODE` — WandB is **off by default** for the sweep (no login/prompt). To
  log to WandB, run `wandb login` once and then `WANDB_MODE=online python sweep.py`.
  If online init fails it falls back to disabled so the run still completes.

> The full sweep repeats a 32-restart × 5000-iteration inversion for each `N`, so
> it is considerably heavier than a single-image run. Lower `FUIA_INV_RESTARTS`
> to trade quality for speed while iterating.
