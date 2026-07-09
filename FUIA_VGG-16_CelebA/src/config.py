import torch
import numpy as np
import random


#Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#using Mac MPS as main device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")



#hyperparameters
NUM_CLIENTS      = 50
FRACTION         = 0.2         #20% client participation per round
NUM_ROUNDS       = 80          #paper: 80 rounds (FL training is fast, inversion dominates runtime)
LOCAL_EPOCHS     = 3
BATCH_SIZE       = 32
FL_LR            = 0.1         #paper: "initial client-side learning rate is set to 0.1"
PRETRAIN_EPOCHS  = 15          #reduced from 400 for faster runtime
PRETRAIN_LR      = 0.01        #lower LR to preserve pretrained conv features
NUM_CLASSES      = 2           #paper: binary classification (smile vs non-smile)
DATA_PER_CLIENT  = 8           #paper Sec VII.A: "set the number of data points per client to 8"

#image settings
IMG_SIZE         = 64          #CelebA resized to 64x64
IMG_CHANNELS     = 3           #RGB
IMG_MEAN         = (0.5, 0.5, 0.5)
IMG_STD          = (0.5, 0.5, 0.5)

#Gradient inversion
INV_ITERATIONS   = 10000
INV_LR           = 0.01
INV_ALPHA        = 1e-5         #TV regularization weight (paper Eq. 13)
INV_RESTARTS     = 3

