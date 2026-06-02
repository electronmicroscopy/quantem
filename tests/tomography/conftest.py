"""Shared fixtures and markers for the tomography test suite.

The suite is split into three tiers (see the plan / individual modules):

* CPU, always-on   -- radon, utils, object/dataset models, optimizer-param wiring.
* CPU, slow        -- conventional SIRT/FBP reconstruction (``--runslow``).
* GPU, slow        -- INR / KPlanes reconstruction (``requires_gpu`` + ``--runslow``).

``torch`` and ``scikit-image`` are core dependencies of quantem, so they are always
importable; ``requires_torch`` is kept only for parity with the existing test style. The
meaningful gate is ``requires_gpu`` (CI runs CPU-only) combined with ``@pytest.mark.slow``.
"""

import numpy as np
import pytest
import torch

from quantem.core import config
from quantem.tomography.utils import rot_ZXZ

# --- Markers ---------------------------------------------------------------
requires_torch = pytest.mark.skipif(not config.get("has_torch"), reason="requires torch")
requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


@pytest.fixture
def torch_device() -> str:
    """Device for torch-only tests.

    ``cuda:0`` when available, else ``cpu``. Note: object models that go through
    ``setup_distributed`` must be built on ``cuda`` when CUDA is present (the CPU path is
    only valid when no CUDA device exists), so torch-only construction tests follow this
    fixture rather than hard-coding ``"cpu"``.
    """
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# --- Synthetic data --------------------------------------------------------
def make_tilt_series(volume: torch.Tensor, angles: np.ndarray) -> np.ndarray:
    """Project a volume into a tilt series with quantem's own forward model.

    Mirrors ``tomography_00_generate_phantom``: rotate by ``rot_ZXZ`` (Euler ZXZ, tilt on
    the X axis) then sum along the beam axis. Using this rather than ``radon_torch`` keeps
    the synthetic data consistent with the geometry the reconstructors assume.
    """
    projections = []
    vol = volume.unsqueeze(0)  # (1, Z, Y, X)
    for angle in angles:
        rotated = rot_ZXZ(vol, 0.0, float(angle), 0.0, device="cpu")
        projections.append(rotated[0].sum(0))
    return torch.stack(projections).numpy().astype(np.float32)


@pytest.fixture(scope="module")
def phantom_volume() -> np.ndarray:
    """Small deterministic (32, 32, 32) phantom with a couple of solid blocks."""
    vol = np.zeros((32, 32, 32), dtype=np.float32)
    vol[8:24, 10:20, 12:22] = 1.0
    vol[18:26, 6:12, 16:24] = 0.6
    return vol


@pytest.fixture(scope="module")
def tilt_angles() -> np.ndarray:
    """Nine tilt angles spanning -70..70 degrees."""
    return np.linspace(-70, 70, 9).astype(np.float32)


@pytest.fixture(scope="module")
def tilt_series(phantom_volume: np.ndarray, tilt_angles: np.ndarray) -> np.ndarray:
    """Synthetic tilt series, shape (n_angles, 32, 32)."""
    return make_tilt_series(torch.from_numpy(phantom_volume), tilt_angles)
