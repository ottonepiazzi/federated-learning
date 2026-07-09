#!/usr/bin/env python3

#Entry point for the CUDA variant: gradient inversion runs as a single batched
#workload on the GPU when available (falling back to CPU otherwise).
from inversion_cuda import gradient_inversion, inversion_device
from pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline(gradient_inversion, inversion_device_label=str(inversion_device()))
