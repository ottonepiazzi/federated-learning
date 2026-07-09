# FUIA - Client Unlearning (CNN on MNIST)

Implementation of the FUIA (Federated Unlearning Inversion Attack) from the paper
"Model Inversion Attack Against Federated Unlearning" (Zhou et al., IEEE TIFS 2026),
client unlearning scenario with Retraining (best version achieved).

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

```bash
cd src
python main.py
```

At the beginning of each run you are asked to log metrics on WandB. To skip it, insert "3".
