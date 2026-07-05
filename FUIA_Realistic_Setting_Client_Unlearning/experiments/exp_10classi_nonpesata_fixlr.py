#10-class MNIST, imbalanced split, uniform (unweighted) FedAvg, lowered FL learning rate. PSNR ~ 14.73 dB.
NUM_CLASSES        = 10
NUM_ROUNDS         = 3
FL_LR              = 0.001
TARGET_SIZE        = 8
HOMOGENEOUS_TARGET = True
TARGET_CLASS       = 0
WEIGHTED_AGGREGATION = False
