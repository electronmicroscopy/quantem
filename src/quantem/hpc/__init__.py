"""
quantem.hpc: GPU-accelerated 4D-STEM loading and processing.

Examples
--------
>>> from quantem.hpc import load, bin
>>> data = load('lamella_data_000001.h5')
>>> binned = bin(data, factor=2)
"""

from quantem.hpc.io import (
    GPUDecompressor,
    load,
    clear_memory,
    bin,
)

__all__ = [
    'GPUDecompressor',
    'load',
    'clear_memory',
    'bin',
]
