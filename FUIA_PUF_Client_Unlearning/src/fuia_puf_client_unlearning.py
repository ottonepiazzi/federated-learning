#!/usr/bin/env python3

#Entry point for the CPU variant: gradient inversion is forced onto the CPU
#(the only device that supports double-backward through MaxPool2d everywhere).
from inversion_cpu import gradient_inversion
from pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline(gradient_inversion, inversion_device_label="CPU")
