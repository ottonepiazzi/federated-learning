# FUIA - Sample Unlearning (VGG-16 on CelebA)

Implementation of the FUIA (Federated Unlearning Inversion Attack) from the paper
"Model Inversion Attack Against Federated Unlearning" (Zhou et al., IEEE TIFS 2026),
sample unlearning scenario with Retraining.

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

### 2. Download CelebA

1. Go to https://www.kaggle.com/datasets/jessicali9530/celeba-dataset and click **Download**
2. Unzip the downloaded archive
3. Place the contents inside a `data/celeba/` folder in the project directory

The final structure must be:

```
data/
  celeba/
    list_attr_celeba.csv
    list_eval_partition.csv
    img_align_celeba/
      img_align_celeba/
        000001.jpg
        000002.jpg
        ...
```

The double-nested `img_align_celeba/img_align_celeba/` folder is how Kaggle packages it. Do not rename or flatten it.

### 3. Run

**macOS / Linux:**
```bash
python3 fuia_celeba_sample_unlearning.py
```

**Windows**
```bash
python fuia_celeba_sample_unlearning.py
```

On the first run, VGG-16 pretrained weights (~528 MB) are downloaded automatically from PyTorch.
A pretrained checkpoint is saved to `checkpoints/` so that pretraining is skipped on subsequent runs.
