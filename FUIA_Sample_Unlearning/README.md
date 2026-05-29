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
