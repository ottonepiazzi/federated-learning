#10-class MNIST, imbalanced split, size-weighted FedAvg, lowered FL learning rate. PSNR ~ 14.97 dB.
NUM_CLASSES        = 10
NUM_ROUNDS         = 3
FL_LR              = 0.001
TARGET_SIZE        = 8
HOMOGENEOUS_TARGET = True
TARGET_CLASS       = 0
WEIGHTED_AGGREGATION = True
