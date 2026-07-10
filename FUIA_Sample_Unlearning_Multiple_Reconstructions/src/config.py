import os
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
DATA_PER_CLIENT  = 8           #paper Sec VII.A: "set the number of data points per client to 8"

#Dataset: which MNIST digits to use. MNIST_DIGITS=None uses all 10 classes
#(closer to the paper's 10-class CIFAR-10 setup, and gives forgotten samples
#distinct labels which helps multi-image separation). A tuple like (0, 1)
#restricts to those digits (the original binary setup). If you use a subset,
#pick digits starting at 0 and contiguous (0,1 / 0,1,2 ...) so labels already
#match CrossEntropy's 0..K-1 indexing. NUM_CLASSES is derived from this.
MNIST_DIGITS     = None        #None = all 10 digits; e.g. (0, 1) for binary
NUM_CLASSES      = 10 if MNIST_DIGITS is None else len(MNIST_DIGITS)

#Gradient inversion (paper Section V.B, Eq. 13)
#INV_ITERATIONS and INV_RESTARTS can be overridden via environment variables
#(FUIA_INV_ITERATIONS / FUIA_INV_RESTARTS) without editing this file.
#Defaults lowered from 5000/32 to 2000/16: run logs show cos_sim plateaus well
#before iter 2000 and the best restart is found within ~16, so this is ~5x
#faster at negligible quality cost. Set 5000/32 to reproduce the original run.
INV_ITERATIONS   = int(os.environ.get("FUIA_INV_ITERATIONS", 2000))
INV_LR           = 0.1         #reduced from 0.1 to avoid overshooting
INV_ALPHA        = 1e-5         #TV regularization weight (paper Eq. 13)
INV_RESTARTS     = int(os.environ.get("FUIA_INV_RESTARTS", 16))

#Forgotten-data sweep (paper Section VII.A.2, Fig. 9: "The Number of Forgotten
#Data on Each Client"). For each N in this list every client forgets its first
#N samples (nested sets, see build_forget_order), the model is retrained, and
#FUIA reconstructs the N forgotten images of the target client. N must stay
#below DATA_PER_CLIENT so each client keeps at least one sample. Default range
#[1..4] matches the paper (Fig. 9 tests N=2 and N=4); extend if you want more
#points (the sweep caches completed N, so extending only computes the new ones).
SWEEP_FORGET_COUNTS = [1, 2, 3, 4]
