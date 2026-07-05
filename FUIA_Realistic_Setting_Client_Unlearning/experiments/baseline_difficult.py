#Baseline "difficult" realistic setting (thesis Sec. 4.6): 10 classes, 30 rounds,
#target client holds a single image, size-weighted FedAvg. Reconstruction fails
#(~7 dB PSNR). These values equal the defaults in src/config.py; the file exists
#so the failing baseline is runnable by name like every other experiment.
NUM_CLASSES        = 10
NUM_ROUNDS         = 30
FL_LR              = 0.01
TARGET_SIZE        = 1
HOMOGENEOUS_TARGET = False
WEIGHTED_AGGREGATION = True
