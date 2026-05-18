from contextlib import contextmanager

import torch.cuda.nvtx as nvtx


@contextmanager
def nvtx_range(enabled: bool, name: str):
    if enabled:
        nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            nvtx.range_pop()
