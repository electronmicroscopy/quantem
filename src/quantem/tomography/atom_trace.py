"""Atom tracing from 3D tomographic volumes via differentiable Gaussian splatting.

This module decomposes a reconstructed 3D volume (``Dataset3d``) into a set of
atomic sites, each modeled as a 3D Gaussian.  Rather than fitting one atom at a
time (the classic Levenberg-Marquardt approach), the whole volume is treated as a
single differentiable model and *all* parameters are optimized jointly with Adam.
Overlap between neighboring atoms is handled automatically by the joint fit.

Forward model
-------------
The volume is decomposed into a sharp atomic part and a smooth background, both
non-negative::

    volume = volume_atoms + volume_background,   volume_atoms >= 0,  volume_background >= 0

``volume_atoms``
    A sum of 3D Gaussians.  Intensities are kept >= 0, so the atomic part is >= 0.
``volume_background``
    A low-degree tensor-product Bezier (Bernstein) field over a small control
    lattice.  With non-negative control points it is non-negative *everywhere*
    (Bernstein partition-of-unity) and slowly varying by construction, so it
    cannot absorb sharp atomic features -- the model class itself separates the
    two.  This is the hook for future background regularization.

Atom models
-----------
isotropic
    5 degrees of freedom per atom: ``x, y, z`` position, intensity ``I``, and a
    single width ``sigma``.
anisotropic (planned)
    10 degrees of freedom: ``x, y, z``, ``I`` and a Cholesky-parameterized
    precision matrix ``Lambda = L @ L.T`` (positive-definite by construction, so
    the Gaussian can never diverge).

Efficiency
----------
The renderer never evaluates every Gaussian over every voxel.  Each Gaussian only
contributes to a small ``(2*window_radius + 1)**3`` window around its rounded
center, accumulated with ``scatter_add``.  Cost is ``O(n_atoms * window_volume)``
rather than ``O(n_atoms * n_voxels)``.  A window of 3-4 sigma is the right
accuracy/speed trade-off for tracing (relative truncation error ~1e-3 at 3 sigma,
~6e-6 at 4 sigma); larger windows are only needed to *measure* truncation.

Conventions
-----------
All internal site coordinates are in **voxel / array-index units** matching the
volume's axes (axis 0, 1, 2).  Conversion to physical units happens only at the
public ``sites`` (Vector) boundary, using the volume's ``sampling``/``origin``.

.. note::
    Under construction. The differentiable forward model below (atom renderer +
    Bezier background) is complete and verified; the ``Atoms`` class, seeding,
    schedule, and regularizers follow.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.datastructures import Dataset3d, Vector
from quantem.core.io.serialize import AutoSerialize

__all__ = [
    "Atoms",
    "render_isotropic",
    "bernstein_basis",
    "render_background",
    "gaussian_blur3d",
    "seed_peaks",
    "estimate_nn_spacing",
]


def render_isotropic(
    positions: Tensor,
    intensities: Tensor,
    sigmas: Tensor | float,
    volume_shape: tuple[int, int, int],
    window_radius: int,
) -> Tensor:
    """Render a sum of isotropic 3D Gaussians into a dense volume by local splatting.

    Each Gaussian contributes only to a ``(2*window_radius + 1)**3`` window around
    its rounded center.  The window *indices* come from ``round(positions)`` (held
    constant w.r.t. gradients), while the Gaussian is evaluated at the continuous
    ``positions``, so gradients flow to ``positions``, ``intensities`` and
    ``sigmas``.  Choose ``window_radius >~ 3-4 * sigma_max`` so that truncation at
    the window edge is negligible for tracing.

    Parameters
    ----------
    positions : Tensor
        ``(N, 3)`` float tensor of site centers in voxel/array-index coordinates.
    intensities : Tensor
        ``(N,)`` float tensor of Gaussian amplitudes (kept >= 0 by the caller).
    sigmas : Tensor or float
        Gaussian width(s) in voxels.  Scalar / shape ``(1,)`` broadcasts to all
        atoms; otherwise shape ``(N,)``.
    volume_shape : tuple[int, int, int]
        Output volume shape ``(D0, D1, D2)``.
    window_radius : int
        Half-width (in voxels) of the cubic splat window per atom.

    Returns
    -------
    Tensor
        ``(D0, D1, D2)`` rendered atomic volume, differentiable w.r.t.
        ``positions``, ``intensities`` and ``sigmas``.
    """
    device = positions.device
    dtype = positions.dtype
    n_atoms = positions.shape[0]
    d0, d1, d2 = volume_shape

    if n_atoms == 0:
        return torch.zeros(volume_shape, device=device, dtype=dtype)

    sigmas = torch.as_tensor(sigmas, device=device, dtype=dtype)
    if sigmas.ndim == 0:
        sigmas = sigmas.reshape(1)
    if sigmas.shape[0] == 1:
        sigmas = sigmas.expand(n_atoms)

    # Integer window centers -- detached so they carry no gradient.
    centers = torch.round(positions.detach()).long()  # (N, 3)

    # Cubic window offsets, (W**3, 3).
    rng = torch.arange(-window_radius, window_radius + 1, device=device)
    o0, o1, o2 = torch.meshgrid(rng, rng, rng, indexing="ij")
    offsets = torch.stack((o0.reshape(-1), o1.reshape(-1), o2.reshape(-1)), dim=1)

    # Integer voxel coordinates for every (atom, window-voxel): (N, W**3, 3).
    vox = centers[:, None, :] + offsets[None, :, :]

    # Continuous displacement from the (sub-voxel) atom center to each voxel.
    diff = vox.to(dtype) - positions[:, None, :]  # (N, W**3, 3)
    r2 = (diff * diff).sum(dim=-1)  # (N, W**3)
    val = intensities[:, None] * torch.exp(-0.5 * r2 / (sigmas[:, None] ** 2))

    # Zero out-of-bounds contributions; clamp their indices to a valid slot so the
    # scatter is safe (those entries add 0.0).
    shape_t = torch.tensor(volume_shape, device=device)
    valid = ((vox >= 0) & (vox < shape_t)).all(dim=-1)  # (N, W**3)
    val = val * valid.to(dtype)
    vox = torch.minimum(torch.maximum(vox, torch.zeros_like(shape_t)), shape_t - 1)

    flat_idx = (vox[..., 0] * (d1 * d2) + vox[..., 1] * d2 + vox[..., 2]).reshape(-1)
    out = torch.zeros(d0 * d1 * d2, device=device, dtype=dtype)
    out = out.scatter_add(0, flat_idx, val.reshape(-1))
    return out.reshape(volume_shape)


def bernstein_basis(
    num_samples: int,
    degree: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Bernstein basis matrix sampled on a uniform grid over ``[0, 1]``.

    ``B[s, i] = C(degree, i) * t**i * (1 - t)**(degree - i)`` with
    ``t = s / (num_samples - 1)``.  Each row sums to 1 (partition of unity), which
    makes a Bezier field a convex combination of its control points.

    Parameters
    ----------
    num_samples : int
        Number of evenly spaced sample points (the axis length of the volume).
    degree : int
        Polynomial degree; the control lattice has ``degree + 1`` points on this
        axis.  Low degree -> smoother, more slowly varying field.
    device, dtype
        Torch device/dtype for the returned matrix.

    Returns
    -------
    Tensor
        ``(num_samples, degree + 1)`` basis matrix.
    """
    t = torch.linspace(0.0, 1.0, num_samples, device=device, dtype=dtype)[:, None]
    i = torch.arange(degree + 1, device=device, dtype=dtype)[None, :]
    log_binom = (
        torch.lgamma(torch.tensor(degree + 1.0, device=device, dtype=dtype))
        - torch.lgamma(i + 1.0)
        - torch.lgamma(degree - i + 1.0)
    )
    # torch.pow gives 0**0 == 1, so the endpoints interpolate the corner controls.
    return torch.exp(log_binom) * t.pow(i) * (1.0 - t).pow(degree - i)


def render_background(
    control_points: Tensor,
    bases: tuple[Tensor, Tensor, Tensor],
) -> Tensor:
    """Evaluate a tensor-product Bezier (Bernstein) field over a dense volume.

    With non-negative ``control_points`` the field is non-negative everywhere and
    bounded by ``[control_points.min(), control_points.max()]`` (partition of
    unity), and is smooth/slowly varying for low control-lattice degree.

    Parameters
    ----------
    control_points : Tensor
        ``(n0 + 1, n1 + 1, n2 + 1)`` control lattice (kept >= 0 by the caller).
    bases : tuple of Tensor
        Per-axis Bernstein bases ``(B0, B1, B2)`` from :func:`bernstein_basis`,
        with shapes ``(D0, n0+1)``, ``(D1, n1+1)``, ``(D2, n2+1)``.

    Returns
    -------
    Tensor
        ``(D0, D1, D2)`` background field, differentiable w.r.t. ``control_points``.
    """
    b0, b1, b2 = bases
    f = torch.einsum("ijk,ai->ajk", control_points, b0)
    f = torch.einsum("ajk,bj->abk", f, b1)
    f = torch.einsum("abk,ck->abc", f, b2)
    return f


def gaussian_blur3d(volume: Tensor, sigma: float) -> Tensor:
    """Separable 3D Gaussian blur with replicate padding.

    Parameters
    ----------
    volume : Tensor
        ``(D0, D1, D2)`` volume.
    sigma : float
        Standard deviation in voxels.  ``sigma <= 0`` returns the input unchanged.

    Returns
    -------
    Tensor
        Blurred ``(D0, D1, D2)`` volume.
    """
    if sigma <= 0:
        return volume
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=volume.device, dtype=volume.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    v = volume[None, None]  # (1, 1, D0, D1, D2)
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        pad = [0, 0, 0, 0, 0, 0]  # F.pad order is (W_lo, W_hi, H_lo, H_hi, D_lo, D_hi)
        pad[(2 - axis) * 2] = radius
        pad[(2 - axis) * 2 + 1] = radius
        v = F.pad(v, pad, mode="replicate")
        v = F.conv3d(v, kernel.reshape(shape))
    return v[0, 0]


def seed_peaks(
    volume: Tensor,
    blur_sigmas: tuple[float, float] | float = (1.0, 2.0),
    threshold: float | None = None,
    threshold_fraction: float = 0.99,
    min_distance: float = 0.0,
    max_peaks: int | None = None,
    progress: bool = False,
) -> tuple[Tensor, Tensor]:
    """Seed candidate atom sites by difference-of-Gaussians peak detection.

    Bandpass-filters the volume (single blur, or a difference of two blurs to also
    suppress smooth background), finds 3x3x3 local maxima above a threshold,
    refines each to sub-voxel accuracy with a per-axis parabolic fit, and
    optionally enforces a minimum spacing (greedy, brightest-first).

    Parameters
    ----------
    volume : Tensor
        ``(D0, D1, D2)`` volume.
    blur_sigmas : tuple[float, float] or float
        Two sigmas -> difference-of-Gaussians (bandpass).  One sigma -> single
        matched-filter blur.  In voxels.
    threshold : float or None
        Absolute cutoff on the filtered (difference-of-Gaussians) response: a voxel
        is a candidate peak only if its response exceeds this.  Overrides
        ``threshold_fraction``.
    threshold_fraction : float
        Detection threshold as a quantile (0-1) of the filtered response, used when
        ``threshold`` is None.  Only voxels above this quantile are kept, so
        ``0.99`` keeps the brightest ~1% of the response (more, weaker sites) and
        ``0.999`` the brightest ~0.1% (fewer, stronger sites).  Robust to outliers
        and to the absolute data scale.
    min_distance : float
        Minimum spacing between seeds in voxels; closer (dimmer) peaks are
        dropped.  0 disables.
    max_peaks : int or None
        Keep at most this many seeds (brightest first).
    progress : bool
        Show a tqdm progress bar over the detection stages.

    Returns
    -------
    positions : Tensor
        ``(M, 3)`` sub-voxel seed coordinates in voxel/array-index units.
    intensities : Tensor
        ``(M,)`` volume value sampled at each seed (integer voxel).
    """
    # Peak detection runs on CPU for cross-device determinism: the local-maximum
    # test relies on exact float equality (resp == max_pool(resp)), which is not
    # reliable on MPS. Seeding is fast and one-time, so this is cheap.
    bar = tqdm(total=4, desc="find_initial", disable=not progress)
    volume = volume.detach().to("cpu")
    dtype = volume.dtype
    if isinstance(blur_sigmas, (tuple, list)):
        s_lo, s_hi = blur_sigmas
        resp = gaussian_blur3d(volume, float(s_lo)) - gaussian_blur3d(volume, float(s_hi))
    else:
        resp = gaussian_blur3d(volume, float(blur_sigmas))
    if threshold is None:
        threshold = float(np.quantile(resp.numpy(), threshold_fraction))
    bar.set_postfix_str("DoG bandpass")
    bar.update(1)

    # 3x3x3 local maxima above threshold (max_pool uses -inf padding, so the
    # equality test is exact for the window argmax).
    pooled = F.max_pool3d(resp[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    is_peak = (resp == pooled) & (resp > threshold)
    # Drop the outer shell so the parabolic fit always has neighbors.
    is_peak[[0, -1], :, :] = False
    is_peak[:, [0, -1], :] = False
    is_peak[:, :, [0, -1]] = False
    idx = is_peak.nonzero(as_tuple=False)  # (M, 3) long
    bar.set_postfix_str(f"{idx.shape[0]} maxima")
    bar.update(1)
    if idx.shape[0] == 0:
        bar.close()
        empty = torch.zeros((0, 3), dtype=dtype)
        return empty, empty[:, 0]

    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    f0 = resp[i, j, k]
    offsets = torch.zeros_like(idx, dtype=dtype)
    eps = torch.finfo(dtype).eps
    neighbor_pairs = (
        (resp[i - 1, j, k], resp[i + 1, j, k]),
        (resp[i, j - 1, k], resp[i, j + 1, k]),
        (resp[i, j, k - 1], resp[i, j, k + 1]),
    )
    for axis, (f_lo, f_hi) in enumerate(neighbor_pairs):
        denom = f_lo - 2.0 * f0 + f_hi
        shift = torch.where(
            denom.abs() > eps, 0.5 * (f_lo - f_hi) / denom, torch.zeros_like(denom)
        )
        offsets[:, axis] = shift.clamp(-0.5, 0.5)

    positions = idx.to(dtype) + offsets
    intensities = volume[i, j, k]
    # Sort brightest-first.
    order = torch.argsort(intensities, descending=True)
    positions, intensities = positions[order], intensities[order]
    bar.set_postfix_str("subpixel refine")
    bar.update(1)

    # Greedy minimum-distance suppression (brightest wins).
    if min_distance > 0 and positions.shape[0] > 1:
        from scipy.spatial import cKDTree

        pts = positions.numpy()
        neighbors = cKDTree(pts).query_ball_point(pts, r=float(min_distance))
        suppressed = np.zeros(pts.shape[0], dtype=bool)
        keep: list[int] = []
        for a in tqdm(range(pts.shape[0]), desc="dedup", disable=not progress, leave=False):
            if suppressed[a]:
                continue
            keep.append(a)
            for b in neighbors[a]:
                if b > a:
                    suppressed[b] = True
        keep_t = torch.as_tensor(keep)
        positions, intensities = positions[keep_t], intensities[keep_t]
    bar.set_postfix_str(f"{positions.shape[0]} sites")
    bar.update(1)
    bar.close()

    if max_peaks is not None and positions.shape[0] > max_peaks:
        positions, intensities = positions[:max_peaks], intensities[:max_peaks]

    return positions, intensities


def estimate_nn_spacing(
    volume: Tensor,
    r_min: float = 2.0,
    r_max: float | None = None,
    return_profile: bool = False,
) -> float | tuple[float, np.ndarray, np.ndarray]:
    """Estimate the nearest-neighbor spacing (in voxels) from the volume autocorrelation.

    The autocorrelation ``IFFT(|FFT(v)|^2)`` is radially averaged; the radius of
    its first peak beyond the central self-peak is the first coordination shell,
    i.e. the typical nearest-neighbor spacing.  Using ``~0.75 *`` this value as the
    seeding ``min_distance`` strongly suppresses duplicate/false-positive sites.

    Parameters
    ----------
    volume : Tensor or ndarray
        3D volume.
    r_min : float
        Ignore peaks closer than this radius (excludes the central self-peak).
    r_max : float or None
        Largest radius to consider.  Default ``min(shape) // 2``.
    return_profile : bool
        If True, also return the radial-autocorrelation profile (normalized so the
        zero-lag value is 1) for plotting/inspection.

    Returns
    -------
    float or tuple
        The nearest-neighbor spacing in voxels; or, if ``return_profile``,
        ``(spacing, radii, radial)`` with ``radii``/``radial`` as 1D arrays.

    Raises
    ------
    RuntimeError
        If no autocorrelation shell peak is found.
    """
    from scipy.signal import find_peaks

    vt = torch.as_tensor(volume).detach().to(device="cpu", dtype=torch.float32)
    vt = vt - vt.mean()
    power = torch.fft.fftn(vt).abs() ** 2
    ac = torch.fft.fftshift(torch.fft.ifftn(power).real).numpy()
    shape = ac.shape
    center = [s // 2 for s in shape]
    sq = [((np.arange(s) - c) ** 2).astype(np.float32) for s, c in zip(shape, center)]
    radius = np.sqrt(sq[0][:, None, None] + sq[1][None, :, None] + sq[2][None, None, :])
    r_int = np.rint(radius).astype(np.int64).ravel()
    counts = np.bincount(r_int)
    radial = np.bincount(r_int, weights=ac.ravel().astype(np.float64)) / np.maximum(counts, 1)
    cutoff = int(r_max) if r_max is not None else min(shape) // 2
    radial = radial[:cutoff]
    if radial[0] > 0:
        radial = radial / radial[0]  # normalize so the zero-lag value is 1
    peaks, _ = find_peaks(radial)
    shell = [int(p) for p in peaks if p >= max(1, int(round(r_min)))]
    if not shell:
        raise RuntimeError(
            "Could not estimate NN spacing from the autocorrelation; "
            "pass min_distance to find_initial() explicitly."
        )
    spacing = float(shell[0])
    if return_profile:
        return spacing, np.arange(len(radial), dtype=float), radial
    return spacing


def _inverse_softplus(y: Tensor) -> Tensor:
    """Numerically stable inverse of ``softplus`` (no overflow for large y).

    ``log(exp(y) - 1) = y + log(-expm1(-y))``; for large y the second term -> 0,
    so the result stays finite (the naive ``log(expm1(y))`` overflows in float32
    around y ~ 88).
    """
    y = y.clamp(min=1e-6)
    return y + torch.log(-torch.expm1(-y))


class Atoms(AutoSerialize):
    """Atomic sites traced from a 3D volume by differentiable Gaussian splatting.

    The volume is modeled as ``volume_atoms + volume_background`` (both >= 0) and
    all parameters are optimized jointly.  Atom parameters are stored internally in
    **voxel/array-index** coordinates; the public ``sites`` Vector reports them in
    the source volume's physical units.

    Construct with :meth:`from_dataset`, then call :meth:`find_initial` (and,
    soon, ``trace``/``optimize``) to populate and refine the sites.

    Parameters
    ----------
    volume : Dataset3d
        Input 3D volume (e.g. a tomographic reconstruction).
    model : {"isotropic"}
        Atom model.  Only isotropic (x, y, z, intensity, sigma) is implemented;
        anisotropic is planned.
    sigma_init : float
        Initial Gaussian width in voxels.
    sigma_cutoff : float
        Splat window half-width in units of sigma (3-4 is the speed/accuracy
        sweet spot).
    background : bool
        If True, model a smooth non-negative Bezier background.
    background_degree : int or tuple[int, int, int]
        Per-axis degree of the background control lattice (lower = smoother).
    device : str, int, or None
        Compute device; None selects the quantem default (cuda > mps > cpu).
    """

    _token = object()

    def __init__(
        self,
        volume: Dataset3d,
        *,
        model: str = "isotropic",
        sigma_init: float = 2.0,
        sigma_cutoff: float = 4.0,
        background: bool = True,
        background_degree: int | tuple[int, int, int] = 4,
        device: str | int | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use Atoms.from_dataset() to instantiate this class.")
        super().__init__()
        if model != "isotropic":
            raise NotImplementedError(
                "Only model='isotropic' is implemented; anisotropic is planned."
            )
        dev, _ = config.validate_device(device)
        self.device = dev
        self._model = model
        self._sigma_init = float(sigma_init)
        if self._sigma_init <= 0:
            raise ValueError(f"sigma_init must be > 0, got {self._sigma_init}.")
        self._sigma_cutoff = float(sigma_cutoff)
        self._dtype = torch.float32

        # Calibration from the source volume (for voxel <-> physical conversion).
        self._source = volume
        self._sampling = np.asarray(volume.sampling, dtype=float)
        self._origin = np.asarray(volume.origin, dtype=float)
        self._units = list(volume.units)
        self._signal_units = volume.signal_units

        # Raw volume as a torch tensor on the chosen device.
        arr = volume.array
        vol = torch.as_tensor(np.asarray(arr)) if arr is not None else volume.tensor
        self._volume = vol.to(device=self.device, dtype=self._dtype)
        self._shape = tuple(int(s) for s in self._volume.shape)

        # Atom parameters (empty until seeded). Stored raw/unconstrained.
        self._positions = torch.zeros((0, 3), dtype=self._dtype, device=self.device)
        self._raw_intensity = torch.zeros((0,), dtype=self._dtype, device=self.device)
        self._raw_sigma = torch.zeros((0,), dtype=self._dtype, device=self.device)

        # Bezier background control lattice + per-axis Bernstein bases.
        self._has_background = bool(background)
        if self._has_background:
            degs = (
                (background_degree,) * 3
                if isinstance(background_degree, int)
                else tuple(int(d) for d in background_degree)
            )
            self._bg_degrees = degs
            self._bases = tuple(
                bernstein_basis(self._shape[a], degs[a], device=self.device, dtype=self._dtype)
                for a in range(3)
            )
            init_bg = max(float(self._volume.min()), 1e-4)
            self._raw_background = _inverse_softplus(
                torch.full(
                    tuple(d + 1 for d in degs), init_bg, dtype=self._dtype, device=self.device
                )
            )
        else:
            self._bg_degrees = None
            self._bases = None
            self._raw_background = None

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dataset(
        cls,
        volume: Dataset3d,
        *,
        model: str = "isotropic",
        sigma_init: float = 2.0,
        sigma_cutoff: float = 4.0,
        background: bool = True,
        background_degree: int | tuple[int, int, int] = 4,
        device: str | int | None = None,
    ) -> "Atoms":
        """Create an :class:`Atoms` tracer from a 3D :class:`Dataset3d` volume."""
        if not isinstance(volume, Dataset3d):
            raise TypeError(f"volume must be a Dataset3d, got {type(volume).__name__}.")
        return cls(
            volume,
            model=model,
            sigma_init=sigma_init,
            sigma_cutoff=sigma_cutoff,
            background=background,
            background_degree=background_degree,
            device=device,
            _token=cls._token,
        )

    # ------------------------------------------------------------------ #
    # Constrained parameter views (raw -> physical-meaning)
    # ------------------------------------------------------------------ #
    @property
    def _intensity(self) -> Tensor:
        return F.softplus(self._raw_intensity)

    @property
    def _sigma(self) -> Tensor:
        return F.softplus(self._raw_sigma)

    @property
    def _background_control(self) -> Tensor | None:
        return None if not self._has_background else F.softplus(self._raw_background)

    def _window_radius(self) -> int:
        smax = self._sigma_init if self.num_sites == 0 else float(self._sigma.detach().max())
        return max(1, int(math.ceil(self._sigma_cutoff * smax)))

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @property
    def num_sites(self) -> int:
        """Number of atomic sites."""
        return int(self._positions.shape[0])

    @property
    def model(self) -> str:
        """Atom model name."""
        return self._model

    # ------------------------------------------------------------------ #
    # Seeding
    # ------------------------------------------------------------------ #
    def estimate_spacing(
        self,
        r_min: float = 2.0,
        recompute: bool = False,
        plot: bool = False,
        returnfig: bool = False,
    ) -> float | tuple:
        """Estimate (and cache) the nearest-neighbor atomic spacing, in voxels.

        Uses the volume autocorrelation (:func:`estimate_nn_spacing`).  The result
        seeds the default ``min_distance`` for :meth:`find_initial`, which strongly
        suppresses duplicate / false-positive sites (especially around weak ones).

        Parameters
        ----------
        r_min : float, default 2.0
            Ignore autocorrelation peaks closer than this radius (excludes the
            central self-peak).
        recompute : bool, default False
            Recompute even if a cached value exists.
        plot : bool, default False
            Plot the radial autocorrelation profile with the detected first-shell
            peak (= the spacing) and the resulting ``min_distance`` marked, to help
            you judge whether the estimate is sensible.
        returnfig : bool, default False
            If True, return ``(spacing, fig, ax)`` instead of just the spacing.

        Returns
        -------
        float or tuple
            The spacing in voxels, or ``(spacing, fig, ax)`` if ``returnfig``.
        """
        if recompute or getattr(self, "_nn_spacing", None) is None:
            self._nn_spacing, radii, radial = estimate_nn_spacing(
                self._volume, r_min=r_min, return_profile=True
            )
            self._nn_profile = (radii, radial)

        if not plot:
            return self._nn_spacing

        import matplotlib.pyplot as plt

        radii, radial = self._nn_profile
        spacing = self._nn_spacing
        # Show the shell structure beyond the central self-peak.
        r_hi = min(len(radial) - 1, max(int(round(4 * spacing)), 12))
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(radii[1 : r_hi + 1], radial[1 : r_hi + 1], color="0.2", lw=1.2)
        ax.axvline(spacing, color="tab:red", ls="--", label=f"NN spacing = {spacing:.1f} voxels")
        ax.axvline(
            0.75 * spacing, color="tab:blue", ls=":",
            label=f"min_distance = {0.75 * spacing:.2f} voxels",
        )
        ax.set_xlabel("radius (voxels)")
        ax.set_ylabel("radial autocorrelation")
        ax.set_title("volume autocorrelation")
        ax.legend(fontsize=9)
        fig.tight_layout()
        return (self._nn_spacing, fig, ax) if returnfig else self._nn_spacing

    def find_initial(
        self,
        blur_sigmas: tuple[float, float] | float | None = None,
        threshold: float | None = None,
        threshold_fraction: float = 0.99,
        min_distance: float | None = None,
        spacing_factor: float = 0.75,
        max_peaks: int | None = None,
        progress: bool = True,
    ) -> "Atoms":
        """Find an initial set of atomic sites by difference-of-Gaussians detection.

        Band-pass filters the volume to enhance atom-sized blobs, keeps the local
        maxima above a threshold as candidate sites, refines each to sub-voxel
        accuracy, and drops duplicates closer than ``min_distance``.  This is the
        first step of tracing and seeds :meth:`refine`.

        Parameters
        ----------
        blur_sigmas : tuple[float, float] or float or None
            Difference-of-Gaussians widths in voxels, ``(small, large)``; the
            band-pass highlights features at the atom scale.  Default brackets
            ``sigma_init`` as ``(0.75, 1.5) * sigma_init``.
        threshold : float or None
            Absolute cutoff on the filtered response (a voxel is a candidate only
            if its response exceeds this).  Overrides ``threshold_fraction``.
        threshold_fraction : float, default 0.99
            Detection threshold as a quantile (0-1) of the filtered response: only
            voxels whose response is above this quantile are kept.  ``0.99`` keeps
            the brightest ~1% of the response (more sites, including weaker ones);
            ``0.999`` keeps the brightest ~0.1% (fewer, stronger sites).  Being a
            quantile, it is robust to outliers and to the absolute data scale.
        min_distance : float or None
            Minimum spacing between sites in voxels; of two sites closer than this,
            the dimmer is removed.  Defaults to ``spacing_factor`` times the
            nearest-neighbor spacing from :meth:`estimate_spacing` -- the main lever
            against duplicate / false-positive sites.
        spacing_factor : float, default 0.75
            Fraction of the estimated nearest-neighbor spacing used for the default
            ``min_distance`` (ignored when ``min_distance`` is given).
        max_peaks : int or None
            If set, keep only the brightest ``max_peaks`` sites.
        progress : bool, default True
            Show a tqdm progress bar.

        Returns
        -------
        Atoms
            ``self``, with ``sites`` populated.
        """
        if blur_sigmas is None:
            blur_sigmas = (0.75 * self._sigma_init, 1.5 * self._sigma_init)
        if min_distance is None:
            try:
                min_distance = spacing_factor * self.estimate_spacing()
            except RuntimeError:
                min_distance = 2.0 * self._sigma_init
        pos, inten = seed_peaks(
            self._volume,
            blur_sigmas=blur_sigmas,
            threshold=threshold,
            threshold_fraction=threshold_fraction,
            min_distance=min_distance,
            max_peaks=max_peaks,
            progress=progress,
        )
        self._positions = pos.to(device=self.device, dtype=self._dtype)
        self._raw_intensity = _inverse_softplus(inten.clamp(min=1e-6)).to(
            device=self.device, dtype=self._dtype
        )
        self._raw_sigma = _inverse_softplus(
            torch.full((pos.shape[0],), self._sigma_init, dtype=self._dtype, device=self.device)
        )
        return self

    # ------------------------------------------------------------------ #
    # Refinement (joint gradient optimization + site add / remove / merge)
    # ------------------------------------------------------------------ #
    def refine(
        self,
        num_iterations: int = 100,
        learning_rate: float = 0.05,
        loss: str = "huber",
        sigma_bounds: tuple[float, float] | None = None,
        intensity_min: float | None = None,
        add: bool = True,
        remove: bool = True,
        merge: bool = True,
        add_threshold_fraction: float = 0.98,
        min_neighbors: int = 2,
        isolation_radius: float | None = None,
        update_every: int = 25,
        min_distance: float | None = None,
        progress: bool = True,
    ) -> "Atoms":
        """Refine the atomic model by joint gradient descent.

        All site parameters (positions, intensities, widths) and the background
        are optimized together against the measured volume.  By default the site
        set is also maintained during refinement: new sites are *added* at peaks
        of the residual, weak sites are *removed*, and duplicates are *merged*.

        Parameters
        ----------
        num_iterations : int
            Number of Adam steps.
        learning_rate : float
            Base step size (positions, in voxels).  Intensity/background steps are
            scaled internally by the data magnitude; widths use half this rate.
        loss : {"huber", "mse"}
            Data-fidelity term.  Huber is robust to reconstruction artifacts.
        sigma_bounds : tuple[float, float] or None
            Hard (min, max) bounds on widths in voxels.  Default
            ``(0.5, 2.0) * sigma_init``.
        intensity_min : float or None
            Sites dimmer than this are removed.  Default 10% of the median site
            intensity.
        add, remove, merge : bool
            Enable adding sites at residual peaks / removing weak (low-intensity)
            sites / merging close sites during refinement.
        add_threshold_fraction : float, default 0.98
            Detection quantile (0-1) for ``add``: lower values recover weaker atoms
            from the residual (raise toward 1 to add only obvious ones).  Pair with
            ``min_neighbors`` to reject the extra noise this admits.
        min_neighbors : int, default 2
            Remove sites with fewer than this many neighbors within
            ``isolation_radius`` -- atoms do not float alone, so isolated detections
            are almost always false positives.  This is what makes a low
            ``add_threshold_fraction`` usable for finding weak atoms (real weak
            atoms sit on the lattice and are kept; isolated noise is dropped).  Set
            0 to disable.
        isolation_radius : float or None
            Neighbor-search radius (voxels) for ``min_neighbors``.  Default
            ``1.5 *`` the estimated nearest-neighbor spacing.
        update_every : int
            Apply the add/remove/merge maintenance every this many iterations.
        min_distance : float or None
            Minimum site spacing for merge/add, in voxels.  Default
            ``2 * sigma_init``.
        progress : bool
            Show a tqdm progress bar (loss and site count).
        """
        if self.num_sites == 0:
            raise RuntimeError("No sites to refine; call find_initial() first.")
        if loss not in ("huber", "mse"):
            raise ValueError("loss must be 'huber' or 'mse'.")
        sig_lo, sig_hi = sigma_bounds or (0.5 * self._sigma_init, 2.0 * self._sigma_init)
        if min_distance is None:
            try:
                min_distance = 0.75 * self.estimate_spacing()
            except RuntimeError:
                min_distance = 2.0 * self._sigma_init
        self._intensity_scale = float(self._intensity.detach().median().clamp(min=1e-3))
        if intensity_min is None:
            intensity_min = 0.1 * self._intensity_scale
        delta = self._intensity_scale

        self._enable_grad()
        opt = self._build_optimizer(learning_rate)
        bar = tqdm(range(num_iterations), desc="refine", disable=not progress)
        for it in bar:
            opt.zero_grad(set_to_none=True)
            model = self.render()
            if loss == "huber":
                data_term = F.huber_loss(model, self._volume, delta=delta)
            else:
                data_term = F.mse_loss(model, self._volume)
            data_term.backward()
            opt.step()
            # Hard constraints: widths in band, positions inside the volume.
            with torch.no_grad():
                clamped = F.softplus(self._raw_sigma).clamp(sig_lo, sig_hi)
                self._raw_sigma.copy_(_inverse_softplus(clamped))
                self._positions.clamp_(min=0.0)
                for a in range(3):
                    self._positions[:, a].clamp_(max=float(self._shape[a] - 1))
            # Periodic site-set maintenance.
            if update_every and (it + 1) % update_every == 0 and it + 1 < num_iterations:
                changed = False
                with torch.no_grad():
                    if add:
                        changed |= self.add_sites(
                            min_distance=min_distance, threshold_fraction=add_threshold_fraction
                        )
                    if merge:
                        changed |= self.merge_sites(min_distance)
                    if min_neighbors > 0:
                        changed |= self.remove_isolated(isolation_radius, min_neighbors)
                    if remove:
                        changed |= self.remove_sites(intensity_min, sigma_bounds=(sig_lo, sig_hi))
                if changed:
                    self._enable_grad()
                    opt = self._build_optimizer(learning_rate)
            bar.set_postfix(loss=f"{float(data_term.detach()):.4g}", n=self.num_sites)
        self._disable_grad()
        return self

    def _enable_grad(self) -> None:
        for p in (self._positions, self._raw_intensity, self._raw_sigma):
            p.requires_grad_(True)
        if self._has_background:
            self._raw_background.requires_grad_(True)

    def _disable_grad(self) -> None:
        for p in (self._positions, self._raw_intensity, self._raw_sigma):
            p.requires_grad_(False)
        if self._has_background:
            self._raw_background.requires_grad_(False)

    def _build_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        # Per-group rates: positions in voxels, intensity/background scaled by the
        # data magnitude, widths slower for stability.
        scale = getattr(self, "_intensity_scale", 1.0)
        groups = [
            {"params": [self._positions], "lr": learning_rate},
            {"params": [self._raw_sigma], "lr": 0.5 * learning_rate},
            {"params": [self._raw_intensity], "lr": learning_rate * scale},
        ]
        if self._has_background:
            groups.append({"params": [self._raw_background], "lr": learning_rate * scale})
        return torch.optim.Adam(groups)

    def remove_sites(self, intensity_min: float, sigma_bounds=None) -> bool:
        """Remove sites dimmer than ``intensity_min`` (and outside ``sigma_bounds``).

        Returns True if any site was removed.
        """
        keep = self._intensity.detach() >= intensity_min
        if sigma_bounds is not None:
            sig = self._sigma.detach()
            keep = keep & (sig >= sigma_bounds[0]) & (sig <= sigma_bounds[1])
        if bool(keep.all()):
            return False
        self._positions = self._positions.detach()[keep]
        self._raw_intensity = self._raw_intensity.detach()[keep]
        self._raw_sigma = self._raw_sigma.detach()[keep]
        return True

    def merge_sites(self, min_distance: float) -> bool:
        """Merge sites closer than ``min_distance`` voxels, keeping the brightest.

        Returns True if any site was merged away.
        """
        if self.num_sites < 2:
            return False
        from scipy.spatial import cKDTree

        pos = self._positions.detach().cpu().numpy()
        inten = self._intensity.detach().cpu().numpy()
        neighbors = cKDTree(pos).query_ball_point(pos, r=float(min_distance))
        suppressed = np.zeros(pos.shape[0], dtype=bool)
        for a in np.argsort(inten)[::-1]:
            if suppressed[a]:
                continue
            for b in neighbors[a]:
                if b != a and inten[b] <= inten[a]:
                    suppressed[b] = True
        if not suppressed.any():
            return False
        keep = torch.as_tensor(np.nonzero(~suppressed)[0], device=self.device)
        self._positions = self._positions.detach()[keep]
        self._raw_intensity = self._raw_intensity.detach()[keep]
        self._raw_sigma = self._raw_sigma.detach()[keep]
        return True

    def remove_isolated(self, radius: float | None = None, min_neighbors: int = 3) -> bool:
        """Remove isolated sites (fewer than ``min_neighbors`` neighbors within ``radius``).

        Counts how many other sites lie within ``radius`` voxels of each site and
        drops those below ``min_neighbors``.  Physically, atoms do not float alone
        in vacuum, so isolated detections are almost always false positives.  This
        is also what makes detecting *weak* atoms practical: lower the
        ``find_initial`` / ``add`` threshold to admit weak sites (and noise), then
        remove the noise here -- real weak atoms sit on the lattice with many
        neighbors and are kept, while spurious peaks are isolated and removed.

        Parameters
        ----------
        radius : float or None
            Neighbor-search radius in voxels.  Default ``1.5 *`` the estimated
            nearest-neighbor spacing (:meth:`estimate_spacing`).
        min_neighbors : int, default 3
            Minimum neighbors within ``radius`` required to keep a site.

        Returns
        -------
        bool
            True if any site was removed.
        """
        if self.num_sites < 2:
            return False
        if radius is None:
            radius = 1.5 * self.estimate_spacing()
        from scipy.spatial import cKDTree

        pts = self._positions.detach().cpu().numpy()
        counts = cKDTree(pts).query_ball_point(pts, r=float(radius), return_length=True)
        keep = (counts - 1) >= min_neighbors  # subtract the site's own match
        if bool(keep.all()):
            return False
        keep_t = torch.as_tensor(np.nonzero(keep)[0], device=self.device)
        self._positions = self._positions.detach()[keep_t]
        self._raw_intensity = self._raw_intensity.detach()[keep_t]
        self._raw_sigma = self._raw_sigma.detach()[keep_t]
        return True

    def add_sites(
        self, min_distance: float, threshold_fraction: float = 0.999, max_add: int | None = None
    ) -> bool:
        """Add new sites at peaks of the positive residual (densification).

        Detects peaks in ``volume - model`` (clamped to >= 0) that lie at least
        ``min_distance`` voxels from existing sites, and appends them.  Used during
        :meth:`refine` to recover atoms the current model is missing.

        Parameters
        ----------
        min_distance : float
            Minimum spacing in voxels, both among new sites and from existing ones.
        threshold_fraction : float, default 0.999
            Detection quantile (0-1) on the residual response (see
            :func:`seed_peaks`); high by default so only clear, missed atoms are
            added.
        max_add : int or None
            If set, cap the number of sites added in this call.

        Returns
        -------
        bool
            True if any site was added.
        """
        residual = (self._volume - self.render()).detach().clamp(min=0.0)
        blur = (0.75 * self._sigma_init, 1.5 * self._sigma_init)
        new_pos, new_int = seed_peaks(
            residual,
            blur_sigmas=blur,
            threshold_fraction=threshold_fraction,
            min_distance=min_distance,
            progress=False,
        )
        if new_pos.shape[0] == 0:
            return False
        if self.num_sites > 0:
            from scipy.spatial import cKDTree

            existing = self._positions.detach().cpu().numpy()
            dist, _ = cKDTree(existing).query(new_pos.cpu().numpy(), k=1)
            far = torch.as_tensor(dist >= float(min_distance))
            new_pos, new_int = new_pos[far], new_int[far]
        if new_pos.shape[0] == 0:
            return False
        if max_add is not None:
            new_pos, new_int = new_pos[:max_add], new_int[:max_add]
        new_pos = new_pos.to(device=self.device, dtype=self._dtype)
        new_raw_int = _inverse_softplus(
            new_int.to(device=self.device, dtype=self._dtype).clamp(min=1e-6)
        )
        new_raw_sig = _inverse_softplus(
            torch.full((new_pos.shape[0],), self._sigma_init, dtype=self._dtype, device=self.device)
        )
        self._positions = torch.cat([self._positions.detach(), new_pos], dim=0)
        self._raw_intensity = torch.cat([self._raw_intensity.detach(), new_raw_int], dim=0)
        self._raw_sigma = torch.cat([self._raw_sigma.detach(), new_raw_sig], dim=0)
        return True

    # ------------------------------------------------------------------ #
    # Rendering / decomposition
    # ------------------------------------------------------------------ #
    def _render_atoms(self) -> Tensor:
        return render_isotropic(
            self._positions, self._intensity, self._sigma, self._shape, self._window_radius()
        )

    def _render_background(self) -> Tensor:
        if not self._has_background:
            return torch.zeros(self._shape, dtype=self._dtype, device=self.device)
        return render_background(self._background_control, self._bases)

    def render(self) -> Tensor:
        """Render the full model ``volume_atoms + volume_background`` as a tensor."""
        return self._render_atoms() + self._render_background()

    def _as_dataset(self, tensor: Tensor, name: str) -> Dataset3d:
        arr = tensor.detach().to("cpu").numpy().astype(np.float32)
        return Dataset3d.from_array(
            arr,
            name=name,
            origin=self._origin,
            sampling=self._sampling,
            units=self._units,
            signal_units=self._signal_units,
        )

    @property
    def volume(self) -> Dataset3d:
        """The source volume."""
        return self._source

    @property
    def volume_atoms(self) -> Dataset3d:
        """Rendered atomic part (>= 0)."""
        return self._as_dataset(self._render_atoms(), "volume_atoms")

    @property
    def volume_background(self) -> Dataset3d:
        """Rendered smooth background (>= 0)."""
        return self._as_dataset(self._render_background(), "volume_background")

    @property
    def volume_model(self) -> Dataset3d:
        """Full model ``volume_atoms + volume_background``."""
        return self._as_dataset(self.render(), "volume_model")

    @property
    def residual(self) -> Dataset3d:
        """Source volume minus the model."""
        return self._as_dataset(self._volume - self.render(), "residual")

    # ------------------------------------------------------------------ #
    # Sites as a Vector (physical units)
    # ------------------------------------------------------------------ #
    @property
    def sites(self) -> Vector:
        """Atomic sites as a :class:`Vector` with fields x, y, z, intensity, sigma.

        Positions and sigma are in the source volume's physical units (sigma uses
        the mean sampling).
        """
        fields = ["x", "y", "z", "intensity", "sigma"]
        units = [self._units[0], self._units[1], self._units[2], self._signal_units, self._units[0]]
        v = Vector.from_shape(shape=(), fields=fields, units=units, name="atoms")
        n = self.num_sites
        if n == 0:
            return v
        pos_vox = self._positions.detach().to("cpu").numpy()
        pos_phys = self._origin[None, :] + pos_vox * self._sampling[None, :]
        inten = self._intensity.detach().to("cpu").numpy()
        sigma_phys = self._sigma.detach().to("cpu").numpy() * float(self._sampling.mean())
        data = np.column_stack([pos_phys, inten, sigma_phys]).astype(np.float32)
        v[...] = data
        return v

    # ------------------------------------------------------------------ #
    # Interactive 3D widget
    # ------------------------------------------------------------------ #
    def show_3d_atoms(self, **kwargs):
        """Open the interactive 3D slice + atom-overlay widget.

        Renders an orthogonal slice (xy / xz / yz) through the volume with the
        atomic sites overlaid (marker size scaled by intensity, opacity fading
        with distance from the slice).  Sites are passed in voxel coordinates so
        they register with the volume.  Requires the ``quantem.widget`` package;
        keyword arguments are forwarded to :class:`quantem.widget.Show3DAtoms`.
        """
        from quantem.widget import Show3DAtoms

        if self.num_sites:
            sites = np.column_stack(
                [
                    self._positions.detach().to("cpu").numpy(),
                    self._intensity.detach().to("cpu").numpy(),
                    self._sigma.detach().to("cpu").numpy(),
                ]
            ).astype(np.float32)
        else:
            sites = np.zeros((0, 5), dtype=np.float32)
        volume = self._volume.detach().to("cpu").numpy().astype(np.float32)
        kwargs.setdefault("title", str(self._source.name))
        kwargs.setdefault("sampling", tuple(float(s) for s in self._sampling))
        return Show3DAtoms(volume, sites=sites, **kwargs)

    def __repr__(self) -> str:
        return (
            f"quantem.Atoms(model={self._model}, num_sites={self.num_sites}, "
            f"volume_shape={self._shape}, device={self.device}, "
            f"background={self._has_background})"
        )
