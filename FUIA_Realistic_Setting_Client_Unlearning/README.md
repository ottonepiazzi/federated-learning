# FUIA – Client Unlearning in a Realistic Setting (CNN on MNIST)

Implementation of the **FUIA (Federated Unlearning Inversion Attack)** from the paper
*"Model Inversion Attack Against Federated Unlearning"* (Zhou et al., IEEE TIFS 2026),
**Client Unlearning** scenario with Retraining, in the version with **more realistic
settings** (heavily imbalanced data split, where the target client holds far fewer
samples than the others).

The code is built around a **single implementation** (`src/`); the different experiments
are in **configuration files** in `experiments/` that override a few
parameters.

## Structure

```
.
├── src/                  
│   ├── config.py         # default parameters + per-experiment override loading
│   ├── data.py           # MNIST, IID partition with an imbalanced target client
│   ├── model.py          # CNN
│   ├── federated.py      # client update, FedAvg (weighted/uniform), evaluation
│   ├── fl_training.py    # Federated Learning loop
│   ├── unlearning.py     # unlearning via Retraining
│   ├── fuia.py           # the 3 FUIA steps
│   ├── attack.py         # attack orchestration + result saving
│   ├── metrics.py        # MSE, PSNR
│   └── main.py           # pipeline entry point
├── experiments/          # one config file per experiment
├── results/              # reconstruction images, one per experiment
├── run_experiment.py     # launcher: selects and runs an experiment
└── requirements.txt
```

## Setup

### 1. Virtual environment

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

The `requirements.txt` selects the correct PyTorch build automatically: the CUDA 12.4
build on Linux/Windows and the standard build on macOS.

### 2. Running the experiments

List the available experiments:
```bash
python run_experiment.py
```

Run (example):
```bash
python run_experiment.py exp_2classi_pesata
```

At startup you are asked whether to log metrics to WandB: to skip this, simply enter `3`.
When finished, the reconstruction is saved to `results/<experiment_name>.png`.

Alternatively, to run the pipeline with the default parameters (which match the experiment in difficult settings):
```bash
cd src
python main.py
```

## Available experiments

All experiments share the Client Unlearning scenario on MNIST with an imbalanced split
(10 clients, 100% participation, no pre-training). The "improved" variants use 3 rounds
and a target client holding 8 homogeneous images of a single class.

| Experiment                         | Classes | Aggregation | FL LR | Rounds | PSNR (dB) |
|------------------------------------|---------|-------------|-------|--------|-----------|
| `baseline_difficult`               | 10      | weighted    | 0.01  | 30     | ~7.29     |
| `exp_10classi_pesata`              | 10      | weighted    | 0.01  | 3      | ~12.71    |
| `exp_10classi_nonpesata`           | 10      | uniform     | 0.01  | 3      | ~13.00    |
| `exp_10classi_pesata_fixlr`        | 10      | weighted    | 0.001 | 3      | ~14.97    |
| `exp_10classi_nonpesata_fixlr`     | 10      | uniform     | 0.001 | 3      | ~14.73    |
| `exp_2classi_pesata`               | 2 (0/1) | weighted    | 0.01  | 3      | ~19.17    |
| `exp_2classi_nonpesata`            | 2 (0/1) | uniform     | 0.01  | 3      | ~19.48    |

