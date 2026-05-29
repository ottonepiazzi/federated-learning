# FUIA - Client Unlearning (CNN on MNIST) with Realistic Settings

Implementation of the FUIA (Federated Unlearning Inversion Attack) from the paper
"Model Inversion Attack Against Federated Unlearning" (Zhou et al., IEEE TIFS 2026),
client unlearning scenario with Retraining (version with more realistic settings).

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
python3 fuia_client_unlearning_realistic_setting.py
```

**Windows**
```bash
python fuia_client_unlearning_realistic_setting.py
```

At the beginning of each run you are asked to log metrics on WandB. If you want to skip this part, simply insert "3".

The src folder contains a modular version of the code.
