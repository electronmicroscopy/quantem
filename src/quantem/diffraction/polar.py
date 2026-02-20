from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import (
    Polar4dstem,
    auto_origin_id,
    dataset4dstem_polar_transform,
)
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.validators import ensure_valid_array

# TODO: subpixel origin finding (auto_origin_id currently uses integer pixel search)
# TODO: elliptical distortion correction in origin finding
# TODO: beamstop mask support (mask diffraction-space pixels before azimuthal averaging)


class PairDistributionFunction(AutoSerialize):
    """
    Pair distribution function (PDF) utilities for diffraction / 4D-STEM data.

    This class wraps a 4D-STEM (or 2D diffraction) dataset and stores a
    polar-transformed representation as a Polar4dstem instance in `self.polar`.
    The PDF pipeline provides methods to compute:

    - azimuthal integration to obtain I(k)
    - background fitting using a parametric model in k^2 / k^4
    - formation of F(k) and a windowed sine transform to obtain G(r)
    - optional density estimation and origin correction (Yoshimoto & Omote-style iteration)
    - basic plotting helpers for I(k), background, F(k), G(r), and g(r)
    Some analysis methods (FEM, clustering, etc.) will be implemented in future revisions.
    """

    _token = object()

    def __init__(
        self,
        polar: Polar4dstem,
        input_data: Any | None = None,
        device: str = "cpu",
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use PairDistributionFunction.from_data() to instantiate this class."
            )

        super().__init__()
        self.polar = polar
        self.input_data = input_data
        self.device = device

        self._polar_tensor: torch.Tensor | None = None
        self._r: torch.Tensor | None = None
        self._reduced_pdf: torch.Tensor | None = None
        self._pdf: torch.Tensor | None = None
        self.radial_mean: torch.Tensor | None = None
        self.Sk: torch.Tensor | None = None
        self.Fk: torch.Tensor | None = None
        self.bg: torch.Tensor | None = None
        self.Fk_mask: torch.Tensor | None = None
        self.rho0: float | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_data(
        cls,
        data: NDArray | Dataset2d | Dataset3d | Dataset4dstem | Polar4dstem,
        *,
        find_origin: bool = True,
        origin_row: float | None = None,
        origin_col: float | None = None,
        ellipse_params: tuple[float, float, float] | None = None,
        num_annular_bins: int = 180,
        radial_min: float = 0.0,
        radial_max: float | None = None,
        radial_step: float = 1.0,
        two_fold_rotation_symmetry: bool = False,
        device: str = "cpu",
    ):
        """
         -> "PairDistributionFunction"
        Create a PairDistributionFunction object from various input types.

        Parameters
        ----------
        data
            Supported inputs:
            - 2D numpy array (single diffraction pattern)
            - 4D numpy array (scan_y, scan_x, ky, kx)
            - Dataset2d
            - Dataset4dstem
            - Polar4dstem

            If a :class:`Polar4dstem` is provided, it is used directly and no origin finding
            or polar transform is performed.
        find_origin
            If True, finds the origin for each scan position by calling
            :meth:`find_origin`. If False, `origin_row`/`origin_col` are used (or default
            to the image center).
        origin_row, origin_col
            Diffraction-space origin (in pixels), used only if `find_origin=False`. If None,
            defaults to the central pixel of the diffraction pattern.
        Other parameters
            Passed through to Dataset4dstem.polar_transform when needed.
        """
        # Polar input: use directly
        if isinstance(data, Polar4dstem):
            polar = data
            return cls(polar=polar, input_data=data, device=device, _token=cls._token)

        # Dataset2d input: wrap as a trivial 4D-STEM (1x1 scan) and fall through
        if isinstance(data, Dataset2d):
            arr2d = data.array
            if arr2d.ndim != 2:
                raise ValueError("Dataset2d for PairDistributionFunction must be 2D.")
            arr4 = arr2d[None, None, ...]  # (1, 1, ky, kx)

            data = Dataset4dstem.from_array(
                array=arr4,
                name=f"{data.name}_as4dstem"
                if getattr(data, "name", None)
                else "rdf_4dstem_from_2d",
                origin=np.concatenate(
                    [np.zeros(2, dtype=float), np.asarray(data.origin, dtype=float)]
                ),
                sampling=np.concatenate(
                    [np.ones(2, dtype=float), np.asarray(data.sampling, dtype=float)]
                ),
                units=["pixels", "pixels"] + list(data.units),
                signal_units=data.signal_units,
            )

        # Dataset4dstem input: polar-transform it
        if isinstance(data, Dataset4dstem):
            scan_y, scan_x, ny, nx = data.array.shape
            if find_origin:
                origin_array = auto_origin_id(
                    data,
                    ellipse_params=ellipse_params,
                    num_annular_bins=num_annular_bins,
                    radial_min=radial_min,
                    radial_max=radial_max,
                    radial_step=radial_step,
                    two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                    device=device,
                )
            else:
                if origin_row is None:
                    origin_row = (ny - 1) / 2.0
                if origin_col is None:
                    origin_col = (nx - 1) / 2.0
                origin_array = np.zeros((scan_y, scan_x, 2), dtype=float)
                origin_array[..., 0] = origin_row
                origin_array[..., 1] = origin_col

            polar = dataset4dstem_polar_transform(
                data,
                origin_array=origin_array,
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                device=device,
            )
            return cls(polar=polar, input_data=data, device=device, _token=cls._token)

        # Dataset3d input: not yet specified how to interpret
        if isinstance(data, Dataset3d):
            raise NotImplementedError(
                "PairDistributionFunction.from_data does not yet support Dataset3d inputs."
            )

        # Numpy array input
        arr = ensure_valid_array(data)
        if arr.ndim == 2:
            ds2 = Dataset2d.from_array(arr, name="rdf_input_2d")

            return cls.from_data(
                ds2,
                find_origin=find_origin,
                origin_row=origin_row,
                origin_col=origin_col,
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                device=device,
            )
        elif arr.ndim == 4:
            ds4 = Dataset4dstem.from_array(arr, name="rdf_input_4dstem")
            return cls.from_data(
                ds4,
                find_origin=find_origin,
                origin_row=origin_row,
                origin_col=origin_col,
                ellipse_params=ellipse_params,
                num_annular_bins=num_annular_bins,
                radial_min=radial_min,
                radial_max=radial_max,
                radial_step=radial_step,
                two_fold_rotation_symmetry=two_fold_rotation_symmetry,
                device=device,
            )
        else:
            raise ValueError("PairDistributionFunction.from_data only supports 2D or 4D arrays.")

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def qq(self) -> Any:
        """
        Scattering vector coordinate array along the radial dimension of `self.polar`,
        in physical units (using Polar4dstem.sampling and origin).
        """
        # Polar4dstem dims: (scan_y, scan_x, phi, r)
        # radial axis is 3
        return self.polar.coords_units(3)

    @property
    def radial_bins(self) -> Any:
        """
        Radial bin centers in pixel units (convenience alias).
        """
        return self.polar.coords(3)

    @property
    def r(self) -> NDArray | None:
        """Real-space radial grid as a numpy array."""
        if self._r is None:
            return None
        return self._to_numpy(self._r)

    @property
    def reduced_pdf(self) -> NDArray | None:
        """Reduced pair distribution function G(r) as a numpy array."""
        if self._reduced_pdf is None:
            return None
        return self._to_numpy(self._reduced_pdf)

    @property
    def pdf(self) -> NDArray | None:
        """Pair distribution function g(r) as a numpy array."""
        if self._pdf is None:
            return None
        return self._to_numpy(self._pdf)

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _get_mask_bool(self, mask_realspace):
        """
        Normalize a real-space mask specification to a boolean (rx, ry) mask.

        Returns
        -------
        mask_bool : np.ndarray or None
            Boolean mask of shape (rx, ry), or None if `mask_realspace` is None.
        """
        mask_bool = None
        if mask_realspace is not None:
            rx, ry = self.polar.array.shape[:2]
            mask_realspace = np.asarray(mask_realspace)

            if mask_realspace.dtype == bool and mask_realspace.shape == (rx, ry):
                mask_bool = mask_realspace
            else:
                raise ValueError("mask_realspace must be boolean array.")
        return mask_bool

    # ------------------------------------------------------------------
    # Torch conversion utilities
    # ------------------------------------------------------------------
    @property
    def polar_tensor(self) -> torch.Tensor:
        if self._polar_tensor is None:
            self._polar_tensor = torch.from_numpy(self.polar.array.astype(np.float32)).to(
                device=self.device
            )
        return self._polar_tensor

    def _to_torch(self, arr: NDArray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32)).to(device=self.device)

    def _to_numpy(self, tensor: torch.Tensor) -> NDArray:
        return tensor.detach().cpu().numpy()

    @staticmethod
    def _gaussian_kernel_1d(
        sigma: float, device: str = "cpu", num_sigmas: float = 3.0
    ) -> torch.Tensor:
        """Create 1D Gaussian kernel for torch convolution."""
        radius = int(np.ceil(num_sigmas * sigma))
        support = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-0.5 * (support / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel

    def _gaussian_filter1d_torch(
        self,
        Fk: torch.Tensor,
        sigma: float,
        mode: str = "nearest",
    ) -> torch.Tensor:
        """
        Apply 1D Gaussian filter, replaces scipy.ndimage.gaussian_filter1d.
        """
        kernel = self._gaussian_kernel_1d(sigma, device=self.device)
        padding = len(kernel) // 2
        x = Fk.unsqueeze(0).unsqueeze(0)  # reshape to (batch, channels, length)
        kernel_w = kernel.view(1, 1, -1)
        if mode == "nearest":
            x = F.pad(x, (padding, padding), mode="replicate")
            result = F.conv1d(x, kernel_w)
        else:
            result = F.conv1d(x, kernel_w, padding=padding)
        return result.squeeze(0).squeeze(0)  # reshape to (length)

    @staticmethod
    def _scattering_model_torch(
        k2: torch.Tensor,
        c: torch.Tensor,
        i0: torch.Tensor,
        s0: torch.Tensor,
        i1: torch.Tensor,
        s1: torch.Tensor,
    ) -> torch.Tensor:
        """Torch version of the scattering model."""
        # Add small epsilon to denominators to prevent division by zero during backprop
        # while still allowing s0/s1 to vary freely
        eps = 1e-10
        exp1 = torch.clamp(k2 / (-2.0 * (s0**2 + eps)), min=-100, max=0)
        exp2 = torch.clamp((k2**2) / (-2.0 * (s1**4 + eps)), min=-100, max=0)
        return c + i0 * torch.exp(exp1) + i1 * torch.exp(exp2)

    def _compute_fit_weights(self, k: torch.Tensor, kmin: float, kmax: float) -> torch.Tensor:
        """
        Compute weighting tensor for background fitting.
        Weights downweight low-k region (using sin² taper) and emphasize high-k values.
        """
        dk = k[1] - k[0]
        k_width = kmax - kmin

        # sin² taper for low-k suppression
        mask_low = torch.sin(torch.clamp((k - kmin) / k_width, 0.0, 1.0) * (torch.pi / 2.0)) ** 2
        # high weight where mask_low is small
        # later used to divide, so large weights mean small contribution
        weights = torch.where(
            mask_low > 1e-4,
            1.0 / mask_low,
            torch.tensor(1e6, device=self.device, dtype=k.dtype),
        )
        # emphasize high-k values
        weights = weights * (k[-1] - 0.9 * k + dk)
        return weights

    def _closure(self, optimizer, theta, k2, Ik_norm, weights):
        """match scipy curve_fit behavior"""
        optimizer.zero_grad()
        # Map from unconstrained to constrained (positive) space via softplus
        c = F.softplus(theta[0])
        i0 = F.softplus(theta[1])
        s0 = F.softplus(theta[2])
        i1 = F.softplus(theta[3])
        s1 = F.softplus(theta[4])

        pred = self._scattering_model_torch(k2, c, i0, s0, i1, s1)
        residuals = (pred - Ik_norm) ** 2
        loss = (residuals / (weights**2)).sum()
        loss.backward()
        return loss

    def _frequency_filtering(
        self,
        Fk: torch.Tensor,
        k_lowpass: float | None,
        k_highpass: float | None,
        dk: torch.Tensor,
    ) -> torch.Tensor:
        """Band pass filtering using torch"""
        if (
            k_lowpass is not None
            and k_lowpass > 0.0
            and k_highpass is not None
            and k_highpass > 0.0
        ):
            if k_highpass > k_lowpass:
                raise ValueError("Gaussian band-pass filtering requires k_highpass < k_lowpass.")
            Fk_low = self._gaussian_filter1d_torch(Fk, sigma=k_lowpass / dk.item(), mode="nearest")
            Fk_high = self._gaussian_filter1d_torch(
                Fk, sigma=k_highpass / dk.item(), mode="nearest"
            )
            Fk = Fk_high - Fk_low
        elif k_lowpass is not None and k_lowpass > 0.0:
            Fk = self._gaussian_filter1d_torch(Fk, sigma=k_lowpass / dk.item(), mode="nearest")
        elif k_highpass is not None and k_highpass > 0.0:
            Fk_high = self._gaussian_filter1d_torch(
                Fk, sigma=k_highpass / dk.item(), mode="nearest"
            )
            Fk = Fk - Fk_high
        return Fk

    def _lorch_window(self, k: torch.Tensor, kmin: float, kmax: float) -> torch.Tensor:
        """
        Construct a combined low-q taper and high-q Lorch window.

        The returned window is:
        - zero outside [kmin, kmax]
        - smoothly rises from 0->1 near kmin using a sin^2 ramp over 10% of the band
        - applies a Lorch-style sinc factor over the full in-band region:
            sin(pi * k/kmax) / (pi * k/kmax)
        """
        # low q taper
        edge_frac_low = 0.1  # 10% of range at low-q
        edge_width_low = edge_frac_low * (kmax - kmin)
        low = (k >= kmin) & (k < kmin + edge_width_low)
        t = (k - kmin) / edge_width_low
        wk = torch.ones_like(k)
        wk = torch.where(low, torch.sin(0.5 * torch.pi * t) ** 2, wk)
        wk = torch.where(k < kmin, torch.zeros_like(wk), wk)
        wk = torch.where(k > kmax, torch.zeros_like(wk), wk)

        # High q taper with Lorch window: w(k) = sin(pi*k/kmax)/(pi*k/kmax)
        x = k / kmax
        inband = (k >= kmin) & (k <= kmax)
        # sinc function: sin(pi*x)/(pi*x) with limit 1 at x=0
        sinc_val = torch.where(
            x == 0,
            torch.ones_like(x),
            torch.sin(torch.pi * x) / (torch.pi * x),
        )
        lorch = torch.where(inband, sinc_val, torch.zeros_like(k))
        wk = wk * lorch
        return wk

    def _compute_alpha_beta(
        self,
        Q2d: torch.Tensor,
        r2d: torch.Tensor,
        G_beta: torch.Tensor,
        r_1d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Yoshimoto-Omote alpha(Q) and beta(Q) integrals used for density estimation.
        """
        Qsafe = torch.where(
            Q2d == 0.0,
            torch.tensor(1e-12, device=self.device, dtype=torch.float32),
            Q2d,
        )
        alpha_int = -4 * torch.pi * r2d * torch.sin(Qsafe * r2d) / Qsafe
        beta_int = G_beta.unsqueeze(0) * torch.sin(Qsafe * r2d) / Qsafe
        alpha = torch.trapezoid(alpha_int, x=r_1d, dim=1)
        beta = torch.trapezoid(beta_int, x=r_1d, dim=1)
        return alpha, beta

    # ------------------------------------------------------------------
    # Analysis method stubs
    # ------------------------------------------------------------------

    # TODO: add beamstop mask support (mask diffraction-space pixels before
    #       azimuthal averaging, e.g. to exclude a beam stop shadow)

    def calculate_radial_mean(
        self,
        mask_realspace: NDArray | None = None,
        returnval: bool = False,
    ) -> torch.Tensor | None:
        """
        Calculate the radial mean intensity from the Polar4dSTEM dataset.

        The polar array is assumed to have shape (scan_y, scan_x, phi, k).
        This method computes, for each scan position, the mean over the azimuthal
        axis (phi), then averages across scan positions to produce a single 1D
        radial curve. This result is stored in ``self.radial_mean``.

        If a real-space mask is provided, only the selected scan positions are
        used in the scan-position average.

        Parameters
        ----------
        mask_realspace : NDArray or None, optional
            Boolean mask in real space used to select probe positions.
            If ``None``, all probe positions are used.
            Must have shape (scan_y, scan_x) where True means "include".
        returnval : bool, optional
            If True, return the computed 1D radial mean tensor.

        Returns
        -------
        radial_mean : torch.Tensor or None
            If `returnval=True`, returns the 1D radial mean intensity (Nk,).
            Otherwise returns None.
        """
        polar_data = self.polar_tensor  # shape: (scan_y, scan_x, phi, k)

        if mask_realspace is None:
            # intensity over q-range for each probe position then average
            radial_probe = polar_data.mean(dim=2)  # dim 2: theta
            self.radial_mean = radial_probe.mean(dim=(0, 1))
        else:
            mask_torch = torch.from_numpy(mask_realspace).to(device=self.device)
            masked_polar = polar_data[mask_torch]  # (N_valid, N_theta, N_k)
            # intensity over q-range of each unmasked probe position
            radial_probe = masked_polar.mean(dim=1)
            # average over unmasked probe positions
            self.radial_mean = radial_probe.mean(dim=0)

        if returnval:
            return self.radial_mean
        else:
            return None

    def fit_bg(
        self,
        Ik: torch.Tensor,
        kmin: float,
        kmax: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fit a smooth background B(k) to a radial intensity curve I(k) using
        PyTorch LBFGS optimizer, with weighting that downweights the low-k
        region and emphasizes higher k.

        The fitted function uses the following form (adopted from py4dstem):
            B(k) = c
                + i0 * exp(-k^2 / (2 s0^2))
                + i1 * exp(-k^4 / (2 s1^4))

        Parameters
        ----------
        Ik
            1D radial intensity tensor (Nk,). Produced by
            :meth:`calculate_radial_mean`.
        kmin, kmax
            k-range (in the same units as the internally constructed `k` grid)
            used to build the low-k weighting mask.

        Returns
        -------
        bg : torch.Tensor
            Fitted background curve B(k), shape (Nk,).
        f : torch.Tensor
            Background minus the constant offset, f(k) = B(k) - c, or functionally
            similar to <f>^2(k)
        """
        k = self._to_torch(np.asarray(self.qq))
        k2 = k**2

        # normalize intensity
        int_mean = Ik.mean()
        Ik_norm = Ik / int_mean
        # initial guesses
        const_bg = float(Ik_norm.min())
        int0 = float(Ik_norm.median()) - const_bg
        sigma0 = float(k.mean())
        # ensure positive values
        const_bg = max(const_bg, 1e-6)
        int0 = max(int0, 1e-6)
        sigma0 = max(sigma0, 1e-6)

        init_vals = torch.tensor(
            [const_bg, int0, sigma0, int0, sigma0],
            device=self.device,
            dtype=torch.float32,
        )
        # Map to unconstrained space via inverse softplus: x = y + log(1 - exp(-y))
        # For numerical stability, clamp init_vals away from zero
        init_vals = torch.clamp(init_vals, min=1e-6)
        theta = init_vals + torch.log(-torch.expm1(-init_vals))
        theta = theta.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [theta],
            lr=1.0,
            max_iter=20,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        # k-dependent fitting weights
        weights = self._compute_fit_weights(k, kmin, kmax)

        prev_loss = torch.tensor(float("inf"))
        max_outer_iter = 100
        tol = 1e-8
        for step in range(max_outer_iter):
            loss = optimizer.step(lambda: self._closure(optimizer, theta, k2, Ik_norm, weights))
            if torch.abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

        # final params
        with torch.no_grad():
            c = F.softplus(theta[0])
            i0 = F.softplus(theta[1])
            s0 = F.softplus(theta[2])
            i1 = F.softplus(theta[3])
            s1 = F.softplus(theta[4])
            # undo normalization
            c_scaled = c * int_mean
            i0_scaled = i0 * int_mean
            i1_scaled = i1 * int_mean
            # compute bg
            bg = self._scattering_model_torch(k2, c_scaled, i0_scaled, s0, i1_scaled, s1)
            f = bg - c_scaled
        return bg, f

    def calculate_Gr(
        self,
        k_min: float | None = None,
        k_max: float | None = None,
        k_lowpass: float | None = None,
        k_highpass: float | None = None,
        r_min: float = 0.0,
        r_max: float = 20.0,
        r_step: float = 0.02,
        mask_realspace: NDArray | None = None,
        damp_origin_oscillations: bool = False,
        r_cut: float = 0.8,
        returnval: bool = False,
    ) -> list[NDArray] | None:
        """
        Calculate the reduced pair distribution function G(r) from a 4D-STEM dataset.

        This routine:
        * Computes the radial mean intensity I(k) from self.polar (optionally
            restricted to a real-space mask).
        * Fits a smooth background B(k) and associated f(k) using :meth:`fit_bg`.
        * Constructs the reduced structure factor F(k) with optional low/highpass filtering.
        * Applies a window in k (low-k sin^2 ramp x Lorch high-k taper).
        * Computes the reduced PDF using a discrete sine transform:
           G(r) = sum_k sin(2*pi*k*r) * F_windowed(k)

        If ``damp_origin_oscillations=True``, :meth:`estimate_density` is called
        and the corrected F(k)/G(r) are stored as ``self.Fk_damped`` and
        ``self.reduced_pdf_damped``. The estimated density is cached in
        ``self.rho0`` so that a subsequent :meth:`calculate_gr` call can reuse it.

        Stored attributes:
        * self.radial_mean, self.Ik, self.bg, self.Fk, self.Fk_masked
        * self.Sk, self.r, self.reduced_pdf
        * self.rho0, self.Fk_damped, self.reduced_pdf_damped (if damping)

        Parameters
        ----------
        k_min : float, optional
            Minimum k (A^-1) for masks and transforms.
        k_max : float or None, optional
            Maximum k (A^-1) for masks and transforms.
        k_lowpass : float or None, optional
            Low-pass Gaussian filter sigma in k-space.
        k_highpass : float or None, optional
            High-pass Gaussian filter sigma in k-space.
        r_min : float, optional
            Minimum r (A) for the real-space grid.
        r_max : float, optional
            Maximum r (A) for the real-space grid.
        r_step : float, optional
            Step size in r (A) for the real-space grid.
        mask_realspace : NDArray or None, optional
            Boolean real-space mask selecting probe positions.
        damp_origin_oscillations : bool, optional
            If True, run :meth:`estimate_density` and store corrected F(k)/G(r).
        r_cut : float, optional
            Minimum radial distance (A) for peak search in density estimation.
            Forwarded to :meth:`estimate_density`.
        returnval : bool, optional
            If True, return ``[r, G(r)]`` as numpy arrays.

        Returns
        -------
        list[np.ndarray] or None
        """
        # this is missing a 2pi term that we add back during the pdf calc later
        k_np = np.asarray(self.qq)
        k = self._to_torch(k_np)
        dk = k[1] - k[0]
        # small epsilon to avoid division by very small k values
        k_safe = torch.clamp(k, min=1e-10)
        self.kmax = k_max if k_max is not None else float(k.max())
        self.kmin = k_min if k_min is not None else float(k.min())

        mask_bool = self._get_mask_bool(mask_realspace)
        Ik = self.calculate_radial_mean(mask_realspace=mask_bool, returnval=True)
        bg, f = self.fit_bg(Ik, self.kmin, self.kmax)
        # prevent division by near-zero values which cause NaNs at high k
        f_safe = torch.clamp(f, min=1e-10 * f.max())

        Fk = (Ik - bg) * k_safe / f_safe
        Fk = self._frequency_filtering(Fk, k_lowpass, k_highpass, dk)
        # Compute Sk from Fk BEFORE applying the 2pi scaling,
        # so that estimate_density corrections are on the same scale
        self.Sk = torch.ones_like(k)
        mask = k > 0
        self.Sk = torch.where(mask, 1.0 + (Fk / k_safe), self.Sk)
        # apply that missing 2pi factor
        Fk = Fk * 2 * torch.pi
        # damp edges with lorch window
        wk = self._lorch_window(k, self.kmin, self.kmax)
        Fk_win = Fk * wk

        r = torch.arange(r_min, r_max, r_step, device=self.device, dtype=torch.float32)
        ka, ra = torch.meshgrid(k, r, indexing="ij")
        # compute reduced PDF using discrete sine transform
        reduced_pdf = (
            (2 / torch.pi)
            * dk
            * 2
            * torch.pi
            * torch.sum(
                torch.sin(2 * torch.pi * ra * ka) * Fk_win[:, None],
                dim=0,
            )
        )
        reduced_pdf[0] = 0  # physically must be at 0 when r = 0

        self.Ik = Ik
        self.bg = bg
        self.Fk = Fk
        self.Fk_masked = Fk_win
        self._r = r
        self._reduced_pdf = reduced_pdf

        # Sk was already computed above (before 2pi scaling)

        if damp_origin_oscillations:
            density_est = self.estimate_density(
                r_cut=r_cut,
                max_iter=20,
                tol_percent=1e-1,
            )
            self.rho0 = density_est[0]
            self.Fk_damped = density_est[1]
            self.reduced_pdf_damped = density_est[2]

        if returnval:
            Gr = getattr(self, "reduced_pdf_damped", None)
            if Gr is None:
                Gr = self._reduced_pdf
            return [self._to_numpy(self._r), self._to_numpy(Gr)]
        return None

    def calculate_gr(
        self,
        density: float | None = None,
        r_cut: float = 0.8,
        set_pdf_positive: bool = False,
        returnval: bool = False,
    ) -> list[NDArray] | None:
        """
        Calculate the pair distribution function g(r) from G(r).

        Requires :meth:`calculate_Gr` to have been run first. The density
        rho0 is determined by (in priority order):

        1. The ``density`` argument, if provided.
        2. ``self.rho0``, if already cached from a prior :meth:`estimate_density` call
           (e.g. via ``calculate_Gr(damp_origin_oscillations=True)``).
        3. A fresh call to :meth:`estimate_density` (result cached in ``self.rho0``).

        The G(r) used is ``self.reduced_pdf_damped`` if it exists (i.e. the user
        chose damping in :meth:`calculate_Gr`), otherwise ``self.reduced_pdf``.

        Parameters
        ----------
        density : float or None, optional
            Number density (atoms/A^3). If None, uses cached or estimated value.
        r_cut : float, optional
            Minimum radial distance (A) for peak search in density estimation.
            Only used when density must be estimated. Forwarded to
            :meth:`estimate_density`.
        set_pdf_positive : bool, optional
            If True, clamp negative g(r) values to 0.
        returnval : bool, optional
            If True, return ``[r, g(r)]`` as numpy arrays.

        Returns
        -------
        list[np.ndarray] or None
        """
        if self._reduced_pdf is None or self._r is None:
            raise RuntimeError("Run calculate_Gr() before calculate_gr().")

        # Determine density
        if density is not None:
            rho0 = density
        elif self.rho0 is not None:
            rho0 = self.rho0
        else:
            density_est = self.estimate_density(
                r_cut=r_cut,
                max_iter=20,
                tol_percent=1e-1,
            )
            self.rho0 = density_est[0]
            rho0 = self.rho0

        # Use damped G(r) if the user opted into damping, otherwise undamped
        Gr = getattr(self, "reduced_pdf_damped", None)
        if Gr is None:
            Gr = self._reduced_pdf

        r = self._r
        mask = r > 0
        pdf = torch.ones_like(Gr)
        pdf = torch.where(mask, 1 + Gr / (4 * torch.pi * r * rho0), torch.zeros_like(pdf))
        if set_pdf_positive:  # negative values are unphysical
            pdf = torch.maximum(pdf, torch.zeros_like(pdf))

        self._pdf = pdf
        if returnval:
            return [self._to_numpy(self._r), self._to_numpy(self._pdf)]
        return None

    def estimate_density(
        self,
        r_cut: float = 0.8,
        max_iter: int = 40,
        tol_percent: float = 1e-4,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """
        Estimate number density rho0 (atoms/A^3) and compute a corrected G(r).

        This method implements an iterative Q-space density estimation by
        Yoshimoto & Omote (2022). It uses the structure factor `self.Sk` and
        the reduced PDF `self.reduced_pdf` to iteratively update rho0 and a
        corrected S(k) so that the implied G(r) is more physically consistent
        at low r.

        This method requires that :meth:`calculate_Gr` has already been run,
        because it depends on `self.Sk`, `self.reduced_pdf`, `self.r`,
        and the k-window bounds (`self.kmin`, `self.kmax`).

        Parameters
        ----------
        r_cut : float, optional
            Minimum radial distance (A) for the peak search used to determine
            the correction interval. Peaks below this distance are ignored.
        max_iter : int, optional
            Maximum number of Q-space iterations.
        tol_percent : float, optional
            Convergence threshold on the relative change in rho0 (in %),
            as defined in Eq. (12) of Yoshimoto & Omote (2022).

        Returns
        -------
        rho0 : float
            Estimated microscopic number density (atoms/A^3).
        Fk_win_damped : torch.Tensor
            Windowed corrected reduced structure function used for the transform.
        G_cor : torch.Tensor
            Reduced PDF G(r) with dampened oscillations near origin.
        """
        if self.Sk is None or self._reduced_pdf is None or self._r is None:
            raise RuntimeError("Run calculate_Gr() before estimate_density().")

        k = self._to_torch(np.asarray(self.qq))
        dk = k[1] - k[0]
        k_fit_mask = k >= self.kmin
        k_fit = k[k_fit_mask]
        ka, ra = torch.meshgrid(k, self._r, indexing="ij")

        mask_search = self._r >= r_cut
        r_search = self._r[mask_search]
        G_search = self._reduced_pdf[mask_search]

        # find tallest peak and first local minimum to the left of r_peak
        ind_max = torch.argmax(G_search)
        r_max = r_search[ind_max]
        left = self._r < r_max
        if not torch.any(left):
            # If peak is immediately at cutoff, just use cutoff as rmin
            rmin = r_cut
        else:
            r_left = self._r[left]
            G_left = self._reduced_pdf[left]
            mins_cond = (G_left[1:-1] < G_left[:-2]) & (G_left[1:-1] < G_left[2:])
            # fix indexing from slicing with +1
            mins_indices = torch.where(mins_cond)[0] + 1
            # minimum closest to the peak, else global min in left interval
            if mins_indices.numel() > 0:
                rmin = float(r_left[mins_indices[-1]])
            else:
                rmin = float(r_left[torch.argmin(G_left)])

        # Restrict r to [0, rmin] for the correction
        r_mask = (self._r >= 0.0) & (self._r <= rmin)
        r_short = self._r[r_mask]
        G_short = self._reduced_pdf[r_mask]
        k_fit_scaled = k_fit * 2 * torch.pi
        k2d_fit, r2d_fit = torch.meshgrid(k_fit_scaled, r_short, indexing="ij")

        # Iterative refinement of rho0 and S(k)
        rho0 = 0.0
        rho0_prev = None
        Sk_cor = self.Sk.clone()
        G_cor = self._reduced_pdf.clone()
        Fk_win_damped = self.Fk_masked.clone()
        # start with uncorrected Gr
        G_beta = G_short
        wk = self._lorch_window(k, self.kmin, self.kmax)
        for j in range(max_iter):
            if j > 0:
                G_beta = G_cor[r_mask]

            # calculate alpha/beta for S(k) adjustment
            alpha, beta = self._compute_alpha_beta(k2d_fit, r2d_fit, G_beta, r_short)
            rho0 = float(torch.sum(alpha * beta) / torch.sum(alpha**2))
            if rho0_prev is not None:
                Rj = np.sqrt(((rho0_prev - rho0) ** 2) / (rho0**2)) * 100.0
                if Rj < tol_percent:
                    break

            # Update S_cor(k) and G_cor
            Sk_cor[k_fit_mask] = Sk_cor[k_fit_mask] - beta + rho0 * alpha
            Fk_cor = k * (Sk_cor - 1.0)
            Fk_win_damped = Fk_cor * wk * 2 * torch.pi
            G_cor = (
                (2.0 / torch.pi)
                * dk
                * 2
                * torch.pi
                * torch.sum(torch.sin(2 * torch.pi * ka * ra) * Fk_win_damped[:, None], dim=0)
            )
            G_cor[0] = 0.0
            rho0_prev = rho0

        return rho0, Fk_win_damped, G_cor

    # ------------------------------------------------------------------
    # Plotting functions
    # ------------------------------------------------------------------

    PlotName = Literal[
        "radial_mean",
        "background_fits",
        "reduced_sf",
        "reduced_pdf",
        "pdf",
        "oscillation_damping",
    ]

    def _apply_xrange(
        self,
        x: NDArray,
        y: NDArray,
        xmin: float | None,
        xmax: float | None,
    ) -> tuple[NDArray, NDArray]:
        if xmin is None and xmax is None:
            return x, y
        xmin_eff = x.min() if xmin is None else xmin
        xmax_eff = x.max() if xmax is None else xmax
        if xmax_eff <= xmin_eff:
            raise ValueError(f"xmax must be > xmin (got xmin={xmin_eff}, xmax={xmax_eff}).")
        m = (x >= xmin_eff) & (x <= xmax_eff)
        # avoid empty plots
        if not np.any(m):
            raise ValueError("Requested plot range contains no data.")
        return x[m], y[m]

    def plot_pdf_results(
        self,
        which: Iterable[PlotName] = ("reduced_pdf",),
        *,
        qmin: float | None = None,
        qmax: float | None = None,
        rmin: float | None = None,
        rmax: float | None = None,
        figsize: tuple[float, float] = (6, 4),
        returnfigs: bool = False,
    ):
        """
        Convenience plotting dispatcher.

        Examples
        --------
        pdfc.calculate_Gr(...)
        pdfc.plot(["radial_mean", "background", "reduced_pdf"])
        """
        mapping = {
            "radial_mean": self.plot_radial_mean,
            "background_fits": self.plot_background_fits,
            "reduced_sf": self.plot_reduced_sf,
            "reduced_pdf": self.plot_reduced_pdf,
            "pdf": self.plot_pdf,
            "oscillation_damping": self.plot_oscillation_damping,
        }

        figs = []
        for name in which:
            if name not in mapping:
                raise ValueError(f"Unknown plot '{name}'. Options: {tuple(mapping)}")
            fig = mapping[name](
                qmin=qmin, qmax=qmax, rmin=rmin, rmax=rmax, figsize=figsize, returnfig=returnfigs
            )
            if returnfigs:
                figs.append(fig)

        return figs if returnfigs else None

    def plot_radial_mean(
        self,
        qmin: float | None = None,
        qmax: float | None = None,
        rmin: float | None = None,  # accepted for dispatcher compatibility, unused
        rmax: float | None = None,  # accepted for dispatcher compatibility, unused
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting radial mean intensity vs scattering vector.
        """

        if self.radial_mean is None:
            raise RuntimeError("Radial mean intensity has not been calculated yet.")

        x = np.asarray(self.qq)
        y = self._to_numpy(self.radial_mean)
        x, y = self._apply_xrange(x, y, qmin, qmax)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, label="Radial Mean Intensity I(k)")
        ax.set_xlabel("Scattering Vector q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Radial Mean Intensity vs Scattering Vector")
        ax.legend()
        ax.set_yscale("log")
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_background_fits(
        self,
        qmin: float | None = None,
        qmax: float | None = None,
        rmin: float | None = None,  # accepted for dispatcher compatibility, unused
        rmax: float | None = None,  # accepted for dispatcher compatibility, unused
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting background fit vs radial mean intensity.
        """
        if self.Ik is None or self.bg is None:
            raise RuntimeError("Radial mean intensity or background has not been calculated yet.")

        x = np.asarray(self.qq)
        y1 = self._to_numpy(self.radial_mean)
        x, y1 = self._apply_xrange(x, y1, qmin, qmax)
        x = np.asarray(self.qq)
        y2 = self._to_numpy(self.bg)
        x, y2 = self._apply_xrange(x, y2, qmin, qmax)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y1, label="Radial Mean Intensity I(k)")
        ax.plot(x, y2, label="Background B(k)", linestyle="--")
        ax.set_xlabel("Scattering Vector q (1/Å)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Radial Mean Intensity and Background Fit")
        ax.legend()
        ax.set_yscale("log")
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_reduced_sf(
        self,
        qmin: float | None = None,
        qmax: float | None = None,
        rmin: float | None = None,  # accepted for dispatcher compatibility, unused
        rmax: float | None = None,  # accepted for dispatcher compatibility, unused
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting reduced structure factor F(k).
        """
        if self.Fk_masked is None:
            raise RuntimeError("Reduced structure factor F(k) has not been calculated yet.")

        Fk = getattr(self, "Fk_damped", None)
        if Fk is None:
            Fk = self.Fk_masked

        x = np.asarray(self.qq)
        y = self._to_numpy(Fk)
        x, y = self._apply_xrange(x, y, qmin, qmax)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, label="Reduced Structure Factor F(k)")
        ax.set_xlabel("Scattering Vector q (1/Å)")
        ax.set_ylabel("Reduced Structure Factor F(k)")
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_reduced_pdf(
        self,
        qmin: float | None = None,  # accepted for dispatcher compatibility, unused
        qmax: float | None = None,  # accepted for dispatcher compatibility, unused
        rmin: float | None = None,
        rmax: float | None = None,
        padding_frac: float = 0.1,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting reduced PDF g(r).
        """
        if self._reduced_pdf is None:
            raise RuntimeError("Reduced PDF has not been calculated yet.")
        Gr = getattr(self, "reduced_pdf_damped", None)
        if Gr is None:
            Gr = self._reduced_pdf

        x = self._to_numpy(self._r)
        y = self._to_numpy(Gr)
        x, y = self._apply_xrange(x, y, rmin, rmax)

        # Find radial value of primary peak and trough for y-limits
        # Filter out NaN and Inf values to avoid plot errors
        valid_mask = np.isfinite(y)
        if np.any(valid_mask):
            y_valid = y[valid_mask]
            y_max = np.max(y_valid)
            y_min = np.min(y_valid)
        else:
            # Fallback if all values are invalid
            y_max = 1.0
            y_min = -1.0
        yrange = y_max - y_min
        pad = padding_frac * yrange

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, label="Reduced Pair Distribution Function G(r)")
        ax.set_xlabel("Radial Distance r (Å)")
        ax.set_ylabel("Reduced Pair Distribution Function G(r)")
        ax.set_ylim(y_min - pad, y_max + pad)
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_pdf(
        self,
        qmin: float | None = None,  # accepted for dispatcher compatibility, unused
        qmax: float | None = None,  # accepted for dispatcher compatibility, unused
        rmin: float | None = None,
        rmax: float | None = None,
        padding_frac: float = 0.1,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        """
        Plotting pair distribution function g(r).
        """
        if self._reduced_pdf is None or self._pdf is None:
            raise RuntimeError("Reduced PDF or PDF has not been calculated yet.")

        x = self._to_numpy(self._r)
        y = self._to_numpy(self._pdf)
        x, y = self._apply_xrange(x, y, rmin, rmax)

        # Find radial value of primary peak
        # Filter out NaN and Inf values to avoid plot errors
        valid_mask = np.isfinite(y)
        if np.any(valid_mask):
            y_valid = y[valid_mask]
            y_max = np.max(y_valid)
            y_min = np.min(y_valid)
        else:
            # Fallback if all values are invalid
            y_max = 1.0
            y_min = -1.0
        yrange = y_max - y_min
        pad = padding_frac * yrange

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, label="Pair Distribution Function g(r)")
        ax.set_xlabel("Radial Distance r (Å)")
        ax.set_ylabel("Pair Distribution Function g(r)")
        ax.set_ylim(y_min - pad, y_max + pad)
        plt.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()

    def plot_oscillation_damping(
        self,
        qmin: float | None = None,  # accepted for dispatcher compatibility, unused
        qmax: float | None = None,  # accepted for dispatcher compatibility, unused
        rmin: float | None = None,
        rmax: float | None = None,
        padding_frac: float = 0.1,
        figsize: tuple[float, float] = (8, 4),
        returnfig: bool = False,
    ):
        if (
            self.Fk_masked is None
            or not hasattr(self, "Fk_damped")
            or not hasattr(self, "reduced_pdf_damped")
        ):
            raise RuntimeError(
                "Oscillation damping data not available. "
                "Run calculate_Gr(damp_origin_oscillations=True) first."
            )

        k = np.asarray(self.qq)

        # Convert torch tensors to numpy for plotting
        Fk_masked = self._to_numpy(self.Fk_masked)
        Fk_damped = self._to_numpy(self.Fk_damped)
        r = self._to_numpy(self._r)
        reduced_pdf = self._to_numpy(self._reduced_pdf)
        reduced_pdf_damped = self._to_numpy(self.reduced_pdf_damped)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # F(k)
        axS_top = axes[0, 0]
        axS_res = axes[1, 0]
        axS_top.plot(k, Fk_masked, label="F_obs(k)", color="gray")
        axS_top.plot(k, Fk_damped, label="F_cor(k)", color="red")
        axS_top.set_xlabel("k (A$^{-1}$)")
        axS_top.set_ylabel("F(k)")
        axS_top.legend()

        axS_res.plot(k, Fk_damped - Fk_masked, color="blue")
        axS_res.set_xlabel("k (A$^{-1}$)")
        axS_res.set_ylabel("F_cor - F_obs")

        # G(r)
        axG_top = axes[0, 1]
        axG_res = axes[1, 1]
        axG_top.plot(r, reduced_pdf, label="G_obs(r)", color="gray")
        axG_top.plot(r, reduced_pdf_damped, label="G_cor(r)", color="red")
        axG_top.set_xlabel("r (A)")
        axG_top.set_ylabel("G(r)")
        axG_top.legend()

        axG_res.plot(r, reduced_pdf_damped - reduced_pdf, color="blue")
        axG_res.set_xlabel("r (A)")
        axG_res.set_ylabel("G_cor - G_obs")

        fig.tight_layout()

        if returnfig:
            return fig
        else:
            plt.show()
