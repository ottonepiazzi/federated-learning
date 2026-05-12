# FUIA - PUF - Client Unlearning (CNN on MNIST)

Implementation of the FUIA (Federated Unlearning Inversion Attack) from the paper
"Model Inversion Attack Against Federated Unlearning" (Zhou et al., IEEE TIFS 2026), combined with the PUF algorithm as the new unlearning method, from the paper "Federated Unlearning Made Practical: Seamless Integration via Negated Pseudo-Gradients", with client unlearning scenario on MNIST dataset.
One thingh to notice in the current implementations is that the number of samples per client is set to 1, therefore the Forget Accuracy can only be 0% or 100%.

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
If you are using a NVIDIA GPU that supports CUDA, run:

**macOS / Linux:**
```bash
python3 fuia_puf_client_unlearning_CUDA.py
```

**Windows**
```bash
python fuia_puf_client_unlearning_CUDA.py
```
since the code is slightly more optimized for that hardware and so it will result in a faster excecution.

Otherwise run
**macOS / Linux:**
```bash
python3 fuia_puf_client_unlearning.py
```

**Windows**
```bash
python fuia_puf_client_unlearning.py
```

At the beginning of each run you are asked to log metrics on WandB. If you want to skip this part, simply insert "3".
