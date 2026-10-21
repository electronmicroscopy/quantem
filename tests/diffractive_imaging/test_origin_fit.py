"""The smooth fits of the diffraction origin against probe position."""

import numpy as np
import pytest
import torch

from quantem.diffractive_imaging import DirectPtychography
from quantem.diffractive_imaging.origin_models import fit_origin_from_measured

from .conftest import direct_ptycho_kwargs, integer_shift_defocus


def _grid(n=20):
    rows, cols = torch.meshgrid(
        torch.arange(n, dtype=torch.float64), torch.arange(n, dtype=torch.float64), indexing="ij"
    )
    return torch.stack([rows, cols], -1).reshape(-1, 2)


@pytest.mark.parametrize(
    "surface, exact",
    [
        (lambda x, y: 0.3 + 0.01 * x - 0.02 * y, ("plane", "parabola", "bezier_two")),
        (lambda x, y: 0.3 + 1e-3 * x * x - 2e-3 * x * y, ("parabola", "bezier_two")),
        (lambda x, y: 1e-5 * x * x * y * y, ("bezier_two",)),
    ],
)
def test_each_fit_reproduces_its_own_surfaces(surface, exact):
    positions = _grid()
    x, y = positions.T
    measured = torch.stack([surface(x, y), -surface(y, x)], -1)
    for method in ("plane", "parabola", "bezier_two"):
        fitted = fit_origin_from_measured(measured, positions, fit_method=method)
        error = float((fitted - measured).abs().max() / measured.abs().max())
        if method in exact:
            assert error < 1e-8, method
        else:
            assert error > 1e-3, method


def test_an_unknown_fit_raises():
    positions = _grid(4)
    with pytest.raises(ValueError, match="fit_method"):
        fit_origin_from_measured(torch.zeros(16, 2), positions, fit_method="spline")


@pytest.mark.parametrize("fit_method", ["parabola", "bezier_two"])
def test_the_reconstructions_take_them(dataset4d, fit_method):
    kwargs = dict(direct_ptycho_kwargs(integer_shift_defocus(1)), edge_blend_pixels=0)
    kwargs.pop("force_fitted_origin", None)
    recon = DirectPtychography.from_dataset4d(dataset4d, fit_method=fit_method, **kwargs)
    recon.reconstruct(deconvolution_kernel="obf", verbose=False)
    assert np.isfinite(recon.obj).all()
