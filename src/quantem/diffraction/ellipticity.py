import math
import warnings

import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem


def fit_elliptical_distortion(
    data: Dataset4dstem | Dataset2d,
    center: tuple[float, float],
    fit_radii: tuple[float, float],
    p0: torch.Tensor | None = None,
    mask: NDArray | torch.Tensor | None = None,
    device: str = "cpu",
    max_iter: int = 500,
    write_metadata: bool = True,
) -> dict:
    """Fit the elliptical distortion of a diffraction pattern from its amorphous ring.

    The mean pattern (averaged over scan positions for 4D data) is fit, inside the
    annulus ``fit_radii`` around ``center``, with a central-beam Gaussian, a ring with
    different widths inside and outside its radius, and a constant background.

    Parameters
    ----------
    data : Dataset4dstem or Dataset2d
        4D-STEM data (averaged over scan positions) or a single pattern.
    center : (row, col)
        Approximate pattern center in pixels; the fit refines it.
    fit_radii : (inner, outer)
        Radii in pixels of the annulus that is fit. It should contain the ring and
        exclude the central beam.
    p0 : Tensor or None
        Initial model parameters (see :func:`_amorphous_ring_model`). Estimated from
        the data if None.
    mask : array or None
        Boolean mask of pixels to ignore (True = ignore).
    device : str
        Torch device for the fit.
    max_iter : int
        Maximum number of iterations.
    write_metadata : bool
        If True, store the fitted ellipse in ``data.metadata["ellipticity"]``.

    Returns
    -------
    dict
        ``"ellipse_params"``: (a, b, theta_deg), the semiaxes in pixels (a >= b) and
        the tilt of the major axis from the column axis in degrees, in [-90, 90), as
        used by :func:`polar_transform`. ``"center"``: refined (row, col).
        ``"fit_params"``: the 11 fitted model parameters. ``"cost"``: sum of squared
        residuals.
    """
    dp = (
        torch.from_numpy(np.ascontiguousarray(data.array))
        if data.array is not None
        else data.tensor
    )
    if dp.ndim == 4:
        dp = dp.mean(dim=(0, 1), dtype=torch.float32)
    elif dp.ndim == 2:
        dp = dp.float()
    else:
        raise ValueError(
            f"Got data with shape {tuple(data.shape)}. Expected a 2D or 4D-STEM dataset."
        )
    dp = dp.to(device)

    # pixel offsets from the nominal center, restricted to the fit annulus
    row_c, col_c = map(float, center)
    r_in, r_out = map(float, fit_radii)
    d_row, d_col = torch.meshgrid(
        torch.arange(dp.shape[0], device=device) - row_c,
        torch.arange(dp.shape[1], device=device) - col_c,
        indexing="ij",
    )
    r = torch.sqrt(d_col**2 + d_row**2)
    in_fit = (r > r_in) & (r < r_out) & dp.isfinite()
    if mask is not None:
        in_fit &= ~torch.as_tensor(mask, dtype=torch.bool, device=device)
    d_row, d_col, r, values = d_row[in_fit], d_col[in_fit], r[in_fit], dp[in_fit]
    if values.numel() == 0:
        raise ValueError("No pixels in the fitting annulus. Check center and fit_radii.")

    if p0 is None:
        # ring radius = peak of the annulus intensity summed in radial bins
        n_bins = max(int(r_out - r_in), 10)
        bin_width = (r_out - r_in) / n_bins
        bins = ((r - r_in) / bin_width).long().clamp(0, n_bins - 1)
        radial_sum = torch.zeros(n_bins, device=device).scatter_add_(0, bins, values)
        R = r_in + (radial_sum.argmax().item() + 0.5) * bin_width
        near_ring = values[(r - R).abs() < 3.0]
        if near_ring.numel() == 0:
            raise ValueError(
                f"No annulus pixels within 3 px of the estimated ring radius {R:.1f}. "
                "Annulus is too narrow or fit_radii is misaligned with the ring."
            )
        background = max(values.min().item(), 0.0)
        I1 = max(near_ring.mean().item() - background, 0.0)
        I0 = max(dp[int(row_c), int(col_c)].item() - background, 0.0)
        # start from a circle of radius R centered on the nominal center
        p0 = [I0, I1, R * 0.3, R * 0.15, R * 0.2, background, 0.0, 0.0]
        p0 += convert_ellipse_params_r(R, R, 0.0)
    p0 = torch.as_tensor(p0, dtype=torch.float32, device=device)

    # The fit varies x = (I0, I1, sigma0, sigma1, sigma2, background, row0, col0, R, e1, e2),
    # describing the ring ellipse by its mean radius R and ellipticity (e1, e2) (see
    # _conic_from_shape), which keeps the fit well conditioned. Bounds: amplitudes and
    # background >= 0, R inside fit_radii, |e1| and |e2| <= 0.5.
    a0, b0, theta0 = convert_ellipse_params(*p0[8:].tolist())
    e0 = (a0 - b0) / (a0 + b0)
    shape0 = [(a0 + b0) / 2, e0 * math.cos(2 * theta0), e0 * math.sin(2 * theta0)]
    x = torch.cat([p0[:8], p0.new_tensor(shape0)])
    lower = torch.full_like(x, -math.inf)
    upper = torch.full_like(x, math.inf)
    lower[[0, 1, 5]] = 0.0
    lower[8], upper[8] = r_in, r_out
    lower[9:], upper[9:] = -0.5, 0.5

    def model_params(x):
        # the widths enter the model squared, so their sign does not matter
        return torch.cat([x[:2], x[2:5].abs(), x[5:8], _conic_from_shape(*x[8:])])

    def residuals(x):
        return _amorphous_ring_model(model_params(x), d_row, d_col) - values

    x = _least_squares(residuals, x.clamp(lower, upper), lower, upper, max_iter)
    if x[8] <= r_in or x[8] >= r_out:
        warnings.warn(
            f"Fitted ring radius {x[8].item():.1f} px is at the edge of fit_radii {fit_radii}. "
            "Choose fit_radii around the ring."
        )
    params = model_params(x)
    a, b, theta = convert_ellipse_params(*params[8:].tolist())
    ellipse = (a, b, (math.degrees(theta) + 90.0) % 180.0 - 90.0)
    if write_metadata:
        data.metadata["ellipticity"] = ellipse
    return {
        "ellipse_params": ellipse,
        "center": (row_c + params[6].item(), col_c + params[7].item()),
        "fit_params": params,
        "cost": (residuals(x) ** 2).sum().item(),
    }


def _least_squares(residuals, x, lower, upper, max_iter):
    """Minimise sum(residuals(x) ** 2) with lower <= x <= upper (Levenberg-Marquardt).

    Each step is a Gauss-Newton step damped by lam; lam is lowered after a step that lowers
    the cost and raised until one does. The step is solved in units where every column of
    the Jacobian has the same size (the largest seen so far), so that parameters of very
    different scales are damped alike and the float32 solve stays accurate. Parameters on
    a bound that the gradient pushes outward are not stepped.
    """
    res = residuals(x)
    cost = res @ res
    lam = 1e-3
    col_norm = torch.full_like(x, 1e-12)
    for _ in range(max_iter):
        # (torch.func.jacfwd would turn float32 into float64 here, which fails on MPS)
        J = torch.autograd.functional.jacobian(
            residuals, x, vectorize=True, strategy="forward-mode"
        )
        col_norm = torch.maximum(col_norm, J.norm(dim=0))
        J = J / col_norm
        JtJ, Jtr = J.T @ J, J.T @ res
        free = ~(((x <= lower) & (Jtr > 0)) | ((x >= upper) & (Jtr < 0)))
        for _ in range(30):
            A = JtJ + lam * torch.eye(len(x), device=x.device)
            step = torch.zeros_like(x)
            step[free] = torch.linalg.solve(A[free][:, free], Jtr[free])
            x_new = (x - step / col_norm).clamp(lower, upper)
            res_new = residuals(x_new)
            cost_new = res_new @ res_new
            if cost_new < cost:
                break
            lam *= 3.0
        if not cost_new < cost:
            return x  # no step lowers the cost: converged
        converged = cost - cost_new < 1e-6 * cost
        x, res, cost, lam = x_new, res_new, cost_new, lam * 0.3
        if converged:
            return x
    warnings.warn(f"Ellipse fit did not converge in {max_iter} iterations.")
    return x


def _amorphous_ring_model(params, offset_row, offset_col):
    """Pattern model: central-beam Gaussian + amorphous ring + constant background.

    ``params`` are (I0, I1, sigma0, sigma1, sigma2, background, row0, col0, A, B, C):
    beam and ring amplitudes, beam width, ring widths inside and outside its radius,
    background, center offset from the nominal center in pixels, and the ring
    ellipse ``A*x^2 + B*x*y + C*y^2 = 1`` (x, y = column, row offset from the
    center). The beam shares the ring's elliptical radius.
    """
    I0, I1, sigma0, sigma1, sigma2, background, row0, col0, A, B, C = params
    x = offset_col - col0
    y = offset_row - row0
    # elliptical radius in pixels, equal to the mean semiaxis R on the ring
    a, b = _ellipse_semiaxes(A, B, C)
    R = (a + b) / 2.0
    rho2 = R**2 * (A * x**2 + B * x * y + C * y**2)
    dr = torch.sqrt(torch.clamp(rho2, min=1e-12)) - R
    sigma = torch.where(dr < 0, sigma1, sigma2)
    beam = I0 * torch.exp(-rho2 / (2.0 * sigma0**2))
    ring = I1 * torch.exp(-(dr**2) / (2.0 * sigma**2))
    return beam + ring + background


def convert_ellipse_params(A, B, C):
    """Semiaxes a >= b and tilt (radians, from the column axis) of the ellipse
    ``A*x^2 + B*x*y + C*y^2 = 1``."""
    mean = (A + C) / 2
    half_gap = math.hypot(A - C, B) / 2
    a = 1 / math.sqrt(mean - half_gap)
    b = 1 / math.sqrt(mean + half_gap)
    theta = math.atan2(-B, C - A) / 2
    return a, b, theta


def convert_ellipse_params_r(a, b, theta):
    """Coefficients (A, B, C) of the ellipse with semiaxes a, b and tilt theta (inverse
    of :func:`convert_ellipse_params`)."""
    mean = (1 / a**2 + 1 / b**2) / 2
    half_gap = (1 / b**2 - 1 / a**2) / 2
    A = mean - half_gap * math.cos(2 * theta)
    B = -2 * half_gap * math.sin(2 * theta)
    C = mean + half_gap * math.cos(2 * theta)
    return A, B, C


def _ellipse_semiaxes(A, B, C):
    """Differentiable semiaxes (a, b) of the ellipse ``A*x^2 + B*x*y + C*y^2 = 1``.
    The clamps keep the square roots and their gradients finite for a circle."""
    mean = (A + C) / 2
    half_gap = torch.sqrt(torch.clamp((A - C) ** 2 + B**2, min=1e-24)) / 2
    a = torch.rsqrt(torch.clamp(mean - half_gap, min=1e-24))
    b = torch.rsqrt(torch.clamp(mean + half_gap, min=1e-24))
    return a, b


def _conic_from_shape(R, e1, e2):
    """Conic coefficients (A, B, C) of the ellipse with mean radius R = (a + b) / 2 and
    ellipticity e = (a - b) / (a + b) along (e1, e2) = e * (cos 2 theta, sin 2 theta)."""
    e_sq = e1**2 + e2**2
    return torch.stack([1 + e_sq - 2 * e1, -4 * e2, 1 + e_sq + 2 * e1]) / (R * (1 - e_sq)) ** 2
