#10-class MNIST, imbalanced split, size-weighted FedAvg. PSNR ~ 12.71 dB.
NUM_CLASSES        = 10
NUM_ROUNDS         = 3
FL_LR              = 0.01
TARGET_SIZE        = 8
HOMOGENEOUS_TARGET = True
TARGET_CLASS       = 0
WEIGHTED_AGGREGATION = True
