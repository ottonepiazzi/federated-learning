#10-class MNIST, imbalanced split, uniform (unweighted) FedAvg. PSNR ~ 13.00 dB.
NUM_CLASSES        = 10
NUM_ROUNDS         = 3
FL_LR              = 0.01
TARGET_SIZE        = 8
HOMOGENEOUS_TARGET = True
TARGET_CLASS       = 0
WEIGHTED_AGGREGATION = False
