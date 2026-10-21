import os

import matplotlib
import pytest

matplotlib.use("Agg")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def _cap_threads_per_worker(config):
    """Share the cores between pytest-xdist workers.

    torch, and the BLAS and OpenMP libraries under numpy, start one thread per core in every
    process, so ``-n logical`` would otherwise run about cores**2 threads. Each worker gets
    ``cores // workers`` instead, at least one. This runs in the controller before the workers
    start, so they inherit the variables before importing torch; any already set are kept.
    """
    workers = getattr(config.option, "numprocesses", None)
    if not isinstance(workers, int) or workers < 2:
        return
    threads = str(max((os.cpu_count() or 1) // workers, 1))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, threads)


def pytest_configure(config):
    _cap_threads_per_worker(config)
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
