# FUIA — Sample Unlearning — Multiple Reconstructions (Ablation)

Ablation study on the **FUIA (Federated Unlearning Inversion Attack)**, from the paper
*"Model Inversion Attack Against Federated Unlearning"* (Zhou et al., IEEE TIFS 2026),
in a **Sample Unlearning** scenario on MNIST, reproducing the analysis of how
reconstruction quality degrades as the **number of forgotten images per client grows**
(paper Sec. VII.A.2, Fig. 9).

For each `N` in `SWEEP_FORGET_COUNTS` (default `[1, 2, 3, 4]`) every client forgets its
first `N` samples, the model is retrained, and FUIA jointly reconstructs the target
client's `N` forgotten images. The sweep logs per-image and aggregate PSNR/MSE and
plots quality-vs-N.

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

### 2. Run the sweep

```bash
cd src
python sweep.py
```

Results (PSNR/MSE curves, reconstruction grids, `sweep_metrics.csv`) are saved in a
`sweep_results_*` folder named after the number of classes.

The sweep caches completed `N` reconstructions in `records.json` (inside the results folder) to
allow resuming an interrupted run. To force a clean run from scratch, delete that file
(or the whole results folder) first.


## Configuration

Key parameters are in `src/config.py`:

- **`MNIST_DIGITS`** — `None` uses all 10 classes; a tuple like `(0, 1)` restricts to
  those digits (the original binary setup). `NUM_CLASSES` is derived automatically.
- **`SWEEP_FORGET_COUNTS`** — the values of `N` tested in the ablation.
- **`FUIA_INV_ITERATIONS` / `FUIA_INV_RESTARTS`** — gradient-inversion budget,
  overridable via environment variables without editing the file.

WandB logging is off by default. Set `WANDB_MODE=online` (after `wandb login`) to enable it.
