# FUIA - PUF - Client Unlearning (CNN on MNIST)

Implementation of the FUIA (Federated Unlearning Inversion Attack) from the paper
"Model Inversion Attack Against Federated Unlearning" (Zhou et al., IEEE TIFS 2026), combined with the PUF algorithm as a new unlearning method from the paper "Federated Unlearning Made Practical: Seamless Integration via Negated Pseudo-Gradients"
client unlearning scenario on MNIST dataset.

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
python3 fuia_client_unlearning_best.py
```

**Windows**
```bash
python fuia_client_unlearning_best.py
```

At the beginning of each run you are asked to log metrics on WandB. If you want to skip this part, simply insert "3".
