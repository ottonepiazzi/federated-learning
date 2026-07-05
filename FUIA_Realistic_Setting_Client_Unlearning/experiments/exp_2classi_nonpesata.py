#2-class MNIST (digits 0/1), imbalanced split, uniform (unweighted) FedAvg. PSNR ~ 19.48 dB.
NUM_CLASSES        = 2
NUM_ROUNDS         = 3
FL_LR              = 0.01
TARGET_SIZE        = 8
HOMOGENEOUS_TARGET = True
TARGET_CLASS       = 0
WEIGHTED_AGGREGATION = False
