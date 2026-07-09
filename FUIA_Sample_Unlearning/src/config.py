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
FRACTION         = 0.2        #20% client participation per round
NUM_ROUNDS       = 10          #paper: "10 training epochs" for MNIST
LOCAL_EPOCHS     = 3
BATCH_SIZE       = 32
FL_LR            = 0.1         #paper: "initial client-side learning rate is set to 0.1"
PRETRAIN_EPOCHS  = 5           #reduced: binary MNIST converges too fast at 50
PRETRAIN_LR      = 0.01        #pre-training LR (0.1 diverges on binary MNIST)
NUM_CLASSES      = 2           #paper: binary classification (digits 0 and 1)
DATA_PER_CLIENT  = 8           #paper Sec VII.A: "set the number of data points per client to 8"

#Gradient inversion
INV_ITERATIONS   = 5000         #same as before
INV_LR           = 0.1         #reduced from 0.1 to avoid overshooting
INV_ALPHA        = 1e-5         #TV regularization weight (paper Eq. 13)
INV_RESTARTS     = 32            #increased from 3
