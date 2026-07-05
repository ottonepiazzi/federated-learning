import torch
import numpy as np
import random


#seed set for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#Device selection: prefer CUDA (fastest, supports 2nd-order grads through
#MaxPool2d which the gradient-inversion step needs), else MPS (training only),
#else CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True            #fixed-size inputs -> faster
    torch.set_float32_matmul_precision("high")       #TF32 matmuls on Ampere+
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

#Gradient inversion device: needs create_graph=True through MaxPool2d.
#CUDA: fully supported -> run inversion on GPU
#MPS:  unsupported -> must fall back to CPU
#CPU:  obviously CPU
INV_DEVICE = DEVICE if DEVICE.type == "cuda" else torch.device("cpu")

#DataLoader knobs: pin_memory enables async H2D copies on CUDA. We keep
#num_workers=0 because MNIST is already fully resident in RAM after the
#dataset is built, so worker processes only add fork overhead.
DL_PIN_MEMORY = (DEVICE.type == "cuda")

print(f"Train device: {DEVICE} | Inversion device: {INV_DEVICE} | "
      f"pin_memory={DL_PIN_MEMORY}")

#Hyperparameters (new setting: no pretraining, 10 clients, IID, 100% participation,
#target client has only 1 image to forget)
NUM_CLIENTS      = 10
FRACTION         = 1.0        #100% client participation per round
NUM_ROUNDS       = 30
LOCAL_EPOCHS     = 1
BATCH_SIZE       = 64
FL_LR            = 0.01       #paper: "learning rate is set to 0.01"
NUM_CLASSES      = 10         #full MNIST (use 2 for the binary digits-0/1 regime)
TARGET_CLIENT    = 0          #the client whose image(s) must be forgotten
TARGET_SIZE      = 1          #number of images held by the target client
TARGET_CLASS     = 0          #class used when the target holds homogeneous images
HOMOGENEOUS_TARGET = False    #True -> target holds `target_size` images all of TARGET_CLASS
WEIGHTED_AGGREGATION = True   #True -> size-weighted FedAvg; False -> uniform average

#Gradient inversion (paper Section V.B & VII.B; client-unlearning loss = Eq. 18)
INV_ITERATIONS   = 8000       #loss plateaus by ~iter 7000 in this regime
INV_LR           = 0.1
INV_GAMMA        = 0.1        #paper-faithful: weight of Psi term in Eq. 18
INV_ALPHA        = 1e-5       #minimal TV: allow fine detail
INV_RESTARTS     = 1          #restarts find similar optima

#---------------------------------------------------------------------------
#Optional per-experiment overrides. When the environment variable
#FUIA_EXPERIMENT is set (e.g. by run_experiment.py), the module
#experiments/<name>.py is loaded and any UPPER_CASE constant it defines
#replaces the default above. With FUIA_EXPERIMENT unset, all defaults stand
#and behaviour is identical to the original code.
#---------------------------------------------------------------------------
import os as _os
import importlib as _importlib

_EXPERIMENT = _os.environ.get("FUIA_EXPERIMENT")
if _EXPERIMENT:
    _overrides = _importlib.import_module(f"experiments.{_EXPERIMENT}")
    for _name in dir(_overrides):
        if _name.isupper() and not _name.startswith("_"):
            globals()[_name] = getattr(_overrides, _name)
    print(f"[config] experiment overrides applied from experiments/{_EXPERIMENT}.py")
