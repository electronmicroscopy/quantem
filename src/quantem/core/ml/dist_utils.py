"""
Standalone distributed training utilities for ptychography.

These are kept separate from ddp.py (which imports tomography types) so they
can be used by diffractive_imaging without circular imports.
"""

from __future__ import annotations

import os
import socket
from typing import Any

import torch
import torch.distributed as dist


def is_distributed_launch() -> bool:
    """True when launched via torchrun / torch.distributed.launch (RANK env var is set)."""
    return "RANK" in os.environ


def find_free_port() -> str:
    """Return a currently-free TCP port (as a string) on the loopback interface.

    Used to pick the rendezvous port for the notebook ``mp.spawn`` path instead of a
    hardcoded ``29500``. A hardcoded port collides across repeated ``reconstruct`` cell
    re-runs (a run that errors before ``destroy_process_group`` leaves the TCPStore server
    socket bound), producing "client socket ... failed to connect" / "address already in
    use" on the next call. Binding to port 0 lets the OS hand back an unused port each time.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


def maybe_configure_fabric_env() -> None:
    """Set the NCCL/libfabric env for the HPE Slingshot (``hsn``) fabric, if present.

    Perlmutter (and other Slingshot-11 systems) need ``NCCL_SOCKET_IFNAME=hsn`` plus
    ``FI_CXI_ATS=0`` / ``NCCL_CROSS_NIC=1`` for NCCL to bring up its communicators cleanly;
    without them multi-GPU init can hang or emit fatal socket errors. Gated on the presence
    of an ``hsn0`` interface and on each var being unset, so this is a no-op on non-Slingshot
    systems (e.g. a local GPU workstation) and never overrides an explicit user setting.
    """
    if not os.path.isdir("/sys/class/net/hsn0"):
        return
    defaults = {
        "NCCL_SOCKET_IFNAME": "hsn",
        "FI_CXI_ATS": "0",
        "NCCL_CROSS_NIC": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def init_process_group(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
    local_device: int | None = None,
) -> None:
    """Initialize the distributed process group from within an mp.spawn worker.

    ``local_device`` is the physical CUDA device index this rank should bind to
    (e.g. with ``GPU_IDS=[2, 3]``, rank 0 should get ``local_device=2``).
    NCCL allocates communicator buffers on the *current* CUDA device at
    ``init_process_group`` time, so the device must be set *before* that call
    or the buffers will land on whichever device was current — typically
    ``cuda:0``. Falling back to ``rank`` matches PyTorch's
    ``LOCAL_RANK == device_index`` convention used by ``torchrun`` when each
    process maps to a contiguous device starting at 0.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    maybe_configure_fabric_env()
    if backend == "nccl":
        device_index = local_device if local_device is not None else rank
        torch.cuda.set_device(device_index)
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def get_rank() -> int:
    """Return the current process rank (0 if not in a distributed context)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Return the world size (1 if not in a distributed context)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def all_reduce_params(*params: torch.Tensor, op: Any = dist.ReduceOp.AVG) -> None:
    """Average the .grad tensors of the given parameters across all ranks in-place."""
    for p in params:
        if p.grad is not None:
            _ = dist.all_reduce(p.grad, op=op)


def broadcast_params(*params: torch.Tensor, src: int = 0) -> None:
    """Broadcast .data of each parameter from rank src to all other ranks."""
    for p in params:
        _ = dist.broadcast(p.data, src=src)


def worker_init_fn(worker_id: int) -> None:
    """Hide CUDA from DataLoader workers so they only touch CPU-resident tensors."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def spawn_distributed_workers(
    worker_fn, devices: list[int], *worker_args, start_method: str = "forkserver"
) -> None:
    """Launch one worker per device via torch.multiprocessing.start_processes.

    worker_fn must be a module-level callable with signature
    (rank, world_size, *worker_args) — matches the mp.start_processes contract,
    which passes rank as the first arg automatically.
    """
    import torch.multiprocessing as mp

    mp.start_processes(  # type: ignore
        worker_fn,
        args=(len(devices), *worker_args),
        nprocs=len(devices),
        join=True,
        start_method=start_method,
    )
