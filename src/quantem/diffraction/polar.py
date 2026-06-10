from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.filter import gaussian_filter_1d, gaussian_kernel_1d
from quantem.core.utils.utils import to_numpy
from quantem.diffraction.polar_transform import (
    find_origin_angular_descent,
    find_origin_angular_grid,
    polar_transform,
)

# TODO: elliptical distortion correction in origin finding
# TODO: beamstop mask support (mask diffraction-space pixels before azimuthal averaging)


class PairDistributionFunction(AutoSerialize):
    """Compute the pair distribution function from 4D-STEM diffuse scattering

    The pair distribution function g(r) gives the probability of finding
    pairs of atoms at separation r, and is the standard tool for
    characterizing local atomic structure in amorphous materials where
    Bragg diffraction is unavailable. This class implements the standard
    extraction pipeline from a 4D-STEM scan (or a single averaged
    diffraction pattern):

    - polar transform of the diffraction patterns
    - azimuthal averaging to obtain I(k)
    - parametric background fit B(k) (Gaussian model in k² and k⁴)
    - reduced structure factor F(k) = 2π · k · [S(k) − 1]
    - windowed sine transform of F(k) to recover the reduced PDF G(r)
    - optional density estimation and Yoshimoto–Omote oscillation damping
    - normalization to g(r) = 1 + G(r) / (4π · r · ρ₀)

    Diffraction data is held in two complementary forms. ``Dataset4dstem``
    holds the input scan with each DP in Cartesian coordinates, indexed as
    ``(scan_row, scan_col, n_row, n_col)``. ``Polar4dstem`` holds the
    result of rebinning each DP to polar coordinates ``(phi, r_pix)``.
    The polar transform is expensive and irreversible, so its result is
    cached as a first-class dataset on ``self.polar`` rather than
    recomputed on demand. ``from_data`` runs the polar transform (and
    optional origin finding) once.

    Attributes
    ----------
    polar : Polar4dstem
        Polar-transformed diffraction data wrapped by this instance.
    input_data : Dataset4dstem or None
        Original input dataset that was polar-transformed to produce
        ``self.polar``. A ``Dataset2d`` input to ``from_data`` is wrapped
        as a 1×1 ``Dataset4dstem`` before being stored here.
    device : str
        Torch device used for computation.
    Ik : torch.Tensor or None
        Azimuthally averaged intensity I(k), set by ``calculate_radial_mean``.
    bg : torch.Tensor or None
        Fitted background B(k), set by ``fit_bg``.
    f : torch.Tensor or None
        Empirical approximation of the mean-squared atomic scattering factor
        ⟨f(k)⟩². Set by ``fit_bg``.
    Sk : torch.Tensor or None
        Structure factor S(k) = 1 + [I(k) − B(k)] / f(k). Set by ``calculate_Gr``.
    Fk : torch.Tensor or None
        Reduced structure factor F(k) = 2π · k · [S(k) − 1]. The 2π factor is
        explicitly included (py4dstem omits it). Set by ``calculate_Gr``.
    Fk_mask : torch.Tensor or None
        Window applied to F(k) before the sine transform, combining a bandpass
        and a Lorch taper. Set by ``calculate_Gr``.
    Fk_damped : torch.Tensor or None
        F(k) after iterative low-r oscillation damping, set by ``estimate_density``.
    reduced_pdf_damped : torch.Tensor or None
        G(r) recomputed from the damped F(k), set by ``estimate_density``.
    rho0 : float or None
        Estimated atomic number density ρ₀ (atoms/Å³), set by ``estimate_density``.

    Exposed as read-only numpy-valued properties:

    r : NDArray or None
        Real-space radial axis in Å.
    reduced_pdf : NDArray or None
        Reduced pair distribution function G(r), obtained by windowed sine
        transform of F(k):
        G(r) = (2/π) ∫ F(k) · sin(2π · k · r) dk.
    pdf : NDArray or None
        Pair distribution function g(r) = 1 + G(r) / (4π · r · ρ₀).

    Examples
    --------
    Construct from a 4D-STEM scan and run the standard pipeline:

    >>> import quantem as em
    >>> ds = em.core.io.read_4dstem("scan.h5", file_type="arina")
    >>> rdf = em.diffraction.PairDistributionFunction.from_data(ds)
    >>> rdf.calculate_Gr(k_min_fit=0.05, k_max_fit=2.0, r_max=10.0)
    >>> rdf.calculate_gr(set_pdf_positive=True)

    Inspect intermediate results:

    >>> rdf.plot_pdf_results(["background_fits", "reduced_sf", "reduced_pdf", "pdf"])

    Restrict the radial average to a real-space region of interest:

    >>> mask = np.zeros(ds.array.shape[:2], dtype=bool)
    >>> mask[300:, 300:] = True
    >>> rdf.calculate_Gr(k_min_fit=0.05, k_max_fit=2.0, mask_realspace=mask)
    """

    _token = object()

    def __init__(
        self,
        polar: Polar4dstem,
        input_data: Dataset4dstem | None = None,
        device: str = "cpu",
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Direct instantiation of PairDistributionFunction is not allowed. "
                "Use PairDistributionFunction.from_data() to instantiate this class."
            )

        super().__init__()
        self.polar = polar
        self.input_data = input_data
        self.device = device

        self._r: torch.Tensor | None = None
        self._reduced_pdf: torch.Tensor | None = None
        self._pdf: torch.Tensor | None = None
        self.Ik: torch.Tensor | None = None
        self.Sk: torch.Tensor | None = None
        self.Fk: torch.Tensor | None = None
        self.bg: torch.Tensor | None = None
        self.f: torch.Tensor | None = None
        self.Fk_mask: torch.Tensor | None = None
        self.Fk_damped: torch.Tensor | None = None
        self.reduced_pdf_damped: torch.Tensor | None = None
        self.rho0: float | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_data(
        cls,
        data: Dataset2d | Dataset4dstem,
        *,
        find_origin: bool = True,
        origin_method: str = "grid",
        origin_row: float | None = None,
        origin_col: float | None = None,
        origin_array: NDArray | None = None,
        ellipse_params: tuple[float, float, float] | None = None,
        num_annular_bins: int = 180,
        radial_min: float = 0.0,
        radial_max: float | None = None,
        radial_step: float = 1.0,
        two_fold_rotation_symmetry: bool = False,
        device: str | None = None,
    ) -> Self:
        """Create a PairDistributionFunction from a dataset.

        Parameters
        ----------
        data : Dataset4dstem or Dataset2d
            - ``Dataset4dstem``: triggers origin finding (optional) and polar
              transform.
            - ``Dataset2d``: single averaged diffraction pattern (e.g. SAED
              or a pre-averaged 4DSTEM result); wrapped as a 1x1 scan
              internally.
        find_origin : bool
            If True, find the diffraction origin at each scan position (method chosen
            by ``origin_method``). If False, use ``origin_row`` / ``origin_col`` (or
            the image center if those are None).
        origin_method : {"grid", "descent"}
            Origin finder used when ``find_origin=True``. ``"grid"`` (default) is the
            global angular-variance search (:func:`find_origin_angular_grid`);
            ``"descent"`` is the COM-anchored local descent
            (:func:`find_origin_angular_descent`), better on small detectors.
        origin_row, origin_col : float or None
            Fixed diffraction-space origin in pixels, used only when
            ``find_origin=False`` and ``origin_array`` is None. Defaults to
            the center of the diffraction pattern.
        origin_array : ndarray or None
            Pre-computed per-DP origins of shape ``(scan_row, scan_col, 2)``.
            When provided, ``find_origin_angular_grid`` is skipped and these origins
            are used directly. Takes precedence over ``find_origin`` and
            ``origin_row``/``origin_col``.
        ellipse_params : tuple of (float, float, float) or None
            Elliptical distortion parameters ``(a, b, theta_deg)`` applied
            during origin finding and polar transform.
        num_annular_bins : int
            Number of angular bins in the polar transform.
        radial_min, radial_max : float or None
            Radial range of the polar transform, in pixels.
        radial_step : float
            Radial step size in pixels.
        two_fold_rotation_symmetry : bool
            If True, sample only ``[0, pi)`` in the angular axis.
        device : str or None
            Torch device used for computation. If None (default), it is
            inferred from the input.

        Returns
        -------
        PairDistributionFunction
        """
        if device is None:
            device = data.device if data.array is None else "cpu"

        # Dataset2d input: wrap as a trivial 4D-STEM (1x1 scan) and fall through
        if isinstance(data, Dataset2d):
            # Dataset2d is numpy-only on dev for now
            arr2d = data.array
            if arr2d.ndim != 2:
                raise ValueError(
                    f"Found array with shape: {arr2d.shape}. "
                    "Dataset2d for PairDistributionFunction must be 2D."
                )
            arr4 = arr2d[None, None, ...]  # (1, 1, n_row, n_col)

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
            scan_row, scan_col, n_row, n_col = data.shape
            if origin_array is not None:
                origin_array = np.asarray(origin_array, dtype=float)
                if origin_array.shape != (scan_row, scan_col, 2):
                    raise ValueError(
                        f"origin_array has shape {origin_array.shape}, expected "
                        f"({scan_row}, {scan_col}, 2)."
                    )
            elif find_origin:
                if origin_method == "descent":
                    origin_array = find_origin_angular_descent(
                        data,
                        ellipse_params=ellipse_params,
                        radial_min=radial_min,
                        radial_max=radial_max,
                        radial_step=radial_step,
                        device=device,
                    )
                elif origin_method == "grid":
                    origin_array = find_origin_angular_grid(
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
                    raise ValueError(
                        f"origin_method must be 'grid' or 'descent', got {origin_method!r}."
                    )
            else:
                if origin_row is None:
                    origin_row = (n_row - 1) / 2.0
                if origin_col is None:
                    origin_col = (n_col - 1) / 2.0
                origin_array = np.zeros((scan_row, scan_col, 2), dtype=float)
                origin_array[..., 0] = origin_row
                origin_array[..., 1] = origin_col

            polar = polar_transform(
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

        raise TypeError(
            f"Got {type(data).__name__}. PairDistributionFunction.from_data "
            "accepts Dataset4dstem or Dataset2d. Wrap numpy arrays with "
            "Dataset4dstem.from_array or Dataset2d.from_array first."
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def qq(self) -> NDArray:
        """
        Scattering vector coordinate array along the radial dimension of `self.polar`,
        in physical units (using Polar4dstem.sampling and origin).
        """
        # Polar4dstem dims: (scan_row, scan_col, phi, r_pix)
        # radial axis is 3
        # origin[3] is the physical q-value at bin 0 (radial_min * pixel_size),
        # sampling[3] is the physical step per bin (radial_step * pixel_size).
        n = self.polar.shape[3]
        origin_r = float(np.asarray(self.polar.origin)[3])
        sampling_r = float(np.asarray(self.polar.sampling)[3])
        return np.arange(n, dtype=float) * sampling_r + origin_r

    @property
    def r(self) -> NDArray | None:
        """Real-space radial grid as a numpy array."""
        if self._r is None:
            return None
        return to_numpy(self._r)

    @property
    def reduced_pdf(self) -> NDArray | None:
        """Reduced pair distribution function G(r) as a numpy array."""
        if self._reduced_pdf is None:
            return None
        return to_numpy(self._reduced_pdf)

    @property
    def pdf(self) -> NDArray | None:
        """Pair distribution function g(r) as a numpy array."""
        if self._pdf is None:
            return None
        return to_numpy(self._pdf)

    # ------------------------------------------------------------------
    # Analysis
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

        The polar array is assumed to have shape (scan_row, scan_col, phi, k).
        This method computes, for each scan position, the mean over the azimuthal
        axis (phi), then averages across scan positions to produce a single 1D
        radial curve. This result is stored in ``self.Ik``.

        If a real-space mask is provided, only the selected scan positions are
        used in the scan-position average. The computation streams chunks through torch to keep peak
        memory low.

        Parameters
        ----------
        mask_realspace : NDArray or None, optional
            Boolean mask in real space used to select probe positions.
            If ``None``, all probe positions are used.
            Must have shape (scan_row, scan_col) where True means "include".
        returnval : bool, optional
            If True, return the computed 1D radial mean tensor.

        Returns
        -------
        radial_mean : torch.Tensor or None
            If `returnval=True`, returns the 1D radial mean intensity (Nk,).
            Otherwise returns None.
        """
        polar_data = (
            self.polar.tensor
            if self.polar.array is None
            else torch.from_numpy(np.ascontiguousarray(self.polar.array))
        )  # (scan_row, scan_col, phi, k)
        scan_row, scan_col, n_phi, n_k = polar_data.shape
        intensity_sum = torch.zeros(n_k, device=self.device, dtype=torch.float64)
        n_valid = 0
        chunk_row = 16  # number of scan rows to process at a time
        for row0 in range(0, scan_row, chunk_row):
            row1 = min(row0 + chunk_row, scan_row)
            chunk = polar_data[row0:row1].to(self.device)
            # mean over phi first -> (chunk, scan_col, k)
            radial_mean = chunk.mean(dim=2)
            if mask_realspace is not None:
                mask_chunk = torch.from_numpy(mask_realspace[row0:row1]).to(self.device)
                n_chunk = int(mask_chunk.sum())
                if n_chunk == 0:
                    continue
                # sum unmasked intensities in chunk and count for normalization later
                intensity_sum += radial_mean[mask_chunk].sum(dim=0)
                n_valid += n_chunk
            else:
                # sum all intensities in chunk and count for normalization later
                intensity_sum += radial_mean.sum(dim=(0, 1))
                n_valid += (row1 - row0) * scan_col
        if n_valid == 0:
            raise ValueError(
                "No valid scan positions selected. The real-space mask is "
                "all False or the dataset is empty."
            )
        self.Ik = (intensity_sum / n_valid).float()

        if returnval:
            return self.Ik
        else:
            return None

    def fit_bg(
        self,
        Ik: torch.Tensor,
        kmin: float | None = None,
        kmax: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fit a smooth background B(k) to a radial intensity curve I(k).

        The background model is a constant plus two monotonically decaying
        terms (adopted from py4DSTEM):

            B(k) = c
                 + i0 * exp(-k^2 / (2 s0^2))
                 + i1 * exp(-k^4 / (2 s1^4))

        B(k) is later subtracted from I(k) to isolate the diffuse signal, and
        f(k) = B(k) - c is used as the denominator in the structure factor
        S(k) = 1 + [I(k) − B(k)] / f(k).

        The five parameters are fit by weighted least squares, with ``sigma = weights_fit`` (a sin² low-k taper that downweights the
        central beam, times a linear factor emphasising higher k). This is the
        single-curve convenience over :meth:`fit_bg_batched`, which holds the
        torch-native Levenberg-Marquardt solver.

        Parameters
        ----------
        Ik
            1D radial intensity tensor (Nk,). Produced by
            :meth:`calculate_radial_mean`.
        kmin, kmax
            Restrict the fit to k in [kmin, kmax]. The returned B(k) is
            still evaluated over the full k axis. If None, the full k
            range is used.

        Returns
        -------
        bg : torch.Tensor
            Fitted background curve B(k), shape (Nk,).
        f : torch.Tensor
            Background minus the constant offset, f(k) = B(k) - c, or functionally
            similar to <f>^2(k). Used later to compute the reduced structure factor F(k).
        """
        # A single curve is just a batch of one
        Ik_row = torch.as_tensor(Ik)[None, :]
        bg_stack, f_stack = self.fit_bg_batched(Ik_row, kmin=kmin, kmax=kmax)
        self.bg, self.f = bg_stack[0], f_stack[0]
        return self.bg, self.f

    def fit_bg_batched(
        self,
        Ik_stack: torch.Tensor,
        kmin: float | None = None,
        kmax: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit B(k) for a stack of N radial curves at once (vectorized LM).

        Background fit is fit with weighting ``sigma = weights_fit``, where

            mask_low    = sin^2( clip((k - kmin) / k_width, 0, 1) * pi/2 )
            weights_fit = (1 / mask_low) * (k[-1] - 0.9*k + dk)

        The fit minimises ``sum[ ((B - I) / weights_fit)^2 ]`` over k in
        [kmin, kmax]. All N curves are fit together in a single solve rather
        than one at a time. 

        Parameters
        ----------
        Ik_stack
            (N, Nk) radial means, one row per curve.
        kmin, kmax
            Fit range. ``kmin`` is also the start of the low-k sin^2 taper

        Returns
        -------
        bg_stack, f_stack
            Each (N, Nk) float32: fitted background and f(k) = B(k) - c per row.
        """
        k = torch.as_tensor(self.qq, dtype=torch.float64, device=self.device)
        if kmin is None:
            kmin = float(k.min())
        if kmax is None:
            kmax = float(k.max())
        k2 = k**2
        fit_mask = (k >= kmin) & (k <= kmax)
        k2_fit = k2[fit_mask]  # (Nf,)
        k_fit = k[fit_mask]
        dk = k[1] - k[0]
        # set up weighting: sin^2 low-k taper * linear high-k emphasis.
        # taper width fixed to default (0.25 1/A) 
        k_width = 0.25
        ramp = torch.clamp((k_fit - kmin) / k_width, 0.0, 1.0)
        mask_low = torch.sin(ramp * (torch.pi / 2.0)) ** 2
        high_k_weight = k_fit[-1] - 0.9 * k_fit + dk
        # inv_weights = 1 / weights_fit = mask_low / kfac
        # store inverse because can just multiply it later
        inv_weights = torch.where(
            mask_low > 1e-4, mask_low / high_k_weight, torch.zeros_like(mask_low)
        )  # length of fit window (Nf,)
        Ik = torch.as_tensor(Ik_stack, dtype=torch.float64, device=self.device)
        Ik_fit = Ik[:, fit_mask].clamp(min=1e-10)  # (N, Nf)
        n = Ik_fit.shape[0] #number of Ik in batch

        # Build Jacobian
        # Initial guesses: constant = min(I), amplitudes = median(I) - min(I), widths = mean(k).
        c0 = Ik_fit.amin(dim=1)  # (N,)
        amp = (Ik_fit.median(dim=1).values - c0).clamp(min=1e-10)  # (N,)
        sig = torch.full(
            (n,), max(float(k.mean()), 1e-3), dtype=torch.float64, device=self.device
        )
        init = torch.stack([c0.clamp(min=1e-10), amp, sig, amp, sig], dim=1)  # (N, 5)
        # exp(theta) = parameters ensures that they remain positive
        theta = torch.log(init)

        def _rj(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Batched residual (N, Nf) and analytic Jacobian (N, Nf, 5).

            Model B(k):
                B(k) = c + i0 * E0 + i1 * E1
                where:
                    E0 = exp(-k^2 / (2 s0^2))
                    E1 = exp(-k^4 / (2 s1^4))
            """
            c, i0, s0, i1, s1 = torch.exp(theta).unbind(dim=1)  # each (N,)
            E0 = torch.exp(
                torch.clamp(-k2_fit[None, :] / (2.0 * s0[:, None] ** 2), min=-100.0, max=0.0)
            )
            E1 = torch.exp(
                torch.clamp(
                    -(k2_fit[None, :] ** 2) / (2.0 * s1[:, None] ** 4), min=-100.0, max=0.0
                )
            )
            B = c[:, None] + i0[:, None] * E0 + i1[:, None] * E1
            w = inv_weights[None, :]  # 1 / weights_fit, broadcast over the batch
            r = (B - Ik_fit) * w
            t0 = i0[:, None] * E0
            t1 = i1[:, None] * E1
            jac = torch.stack(
                [
                    c[:, None] * w,
                    t0 * w,
                    t0 * (k2_fit[None, :] / s0[:, None] ** 2) * w,
                    t1 * w,
                    t1 * (2.0 * k2_fit[None, :] ** 2 / s1[:, None] ** 4) * w,
                ],
                dim=2,
            )
            return r, jac

        def _loss(theta: torch.Tensor) -> torch.Tensor:
            r, _ = _rj(theta)
            return (r * r).sum(dim=1)  # (N,)

        # Batched Levenberg-Marquardt, replaces scipy.curve_fit.
        # recompute Jacobian every iteration and take one step, accepting
        # per-curve where the loss improved. lamba is adapted per-curve
        # over the course of fitting (decrease on accept toward Gauss-Newton, 
        # increase on reject toward gradient descent). Converged curves simply stop being updated.
        lam = torch.ones(n, dtype=torch.float64, device=self.device)
        loss = _loss(theta)
        eye = 1e-12 * torch.eye(5, dtype=torch.float64, device=self.device)
        max_log_step = 1.0
        for _ in range(200):  # outer loop: recompute the Jacobian at the current theta
            r, jac = _rj(theta)
            JtJ = torch.einsum("nki,nkj->nij", jac, jac)  # (N, 5, 5)
            Jtr = torch.einsum("nki,nk->ni", jac, r)  # (N, 5)
            diag = torch.diagonal(JtJ, dim1=1, dim2=2).clamp(min=1e-12)  # (N, 5)
            theta_base = theta  # all trial steps start from here this iter
            prev_loss = loss
            accepted = torch.zeros(n, dtype=torch.bool, device=self.device)
            # inner loop: per-curve lambda search. increase damping and retry until
            # each curve finds a downhill step
            for _ in range(30):
                A = JtJ + lam[:, None, None] * torch.diag_embed(diag) + eye
                delta = torch.linalg.solve(A, Jtr)  # (N, 5)
                # shrink the step if largest component exceeds max_log_step
                step_scale = (
                    max_log_step / delta.abs().amax(dim=1).clamp(min=1e-30)
                ).clamp(max=1.0)
                theta_trial = theta_base - delta * step_scale[:, None]
                loss_trial = _loss(theta_trial)
                # accept improvements per-curve
                improved = (
                    torch.isfinite(loss_trial) & (loss_trial < loss) & (~accepted)
                )
                theta = torch.where(improved[:, None], theta_trial, theta)
                loss = torch.where(improved, loss_trial, loss)
                accepted = accepted | improved
                # Accepted curves relax lambda (toward Gauss-Newton), curves not accepted tighten it (toward gradient descent).
                lam = torch.where(
                    improved,
                    (lam * 0.3).clamp(min=1e-12),
                    torch.where(~accepted, (lam * 3.0).clamp(max=1e12), lam),
                )
                if bool(accepted.all()):
                    break
            if not bool(accepted.any()):
                break  # no curve found a downhill step -> all converged / stuck
            # Stop once every curve's relative improvement is negligible.
            rel = (prev_loss - loss) / (prev_loss + 1e-30)
            if float(rel.max()) < 1e-9:
                break

        # Evaluate B(k) over the full k-range (not just k_fit) for every curve
        with torch.no_grad():
            c, i0, s0, i1, s1 = torch.exp(theta).unbind(dim=1)
            E0 = torch.exp(
                torch.clamp(-k2[None, :] / (2.0 * s0[:, None] ** 2), min=-100.0, max=0.0)
            )
            E1 = torch.exp(
                torch.clamp(
                    -(k2[None, :] ** 2) / (2.0 * s1[:, None] ** 4), min=-100.0, max=0.0
                )
            )
            bg = c[:, None] + i0[:, None] * E0 + i1[:, None] * E1  # (N, Nk)
            f = (bg - c[:, None]).clamp(min=1e-10 * bg.amax(dim=1, keepdim=True))
        return bg.to(dtype=torch.float32), f.to(dtype=torch.float32)

    def calculate_Gr(
        self,
        k_min_fit: float | None = None,
        k_max_fit: float | None = None,
        k_min_window: float | None = None,
        k_max_window: float | None = None,
        k_lowpass: float | None = None,
        k_highpass: float | None = None,
        r_min: float = 0.0,
        r_max: float = 20.0,
        r_step: float = 0.02,
        mask_realspace: NDArray | None = None,
        damp_origin_oscillations: bool = False,
        density: float | None = None,
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
        * self.Ik, self.bg, self.Fk, self.Fk_masked
        * self.Sk, self.r, self.reduced_pdf
        * self.rho0, self.Fk_damped, self.reduced_pdf_damped (if damping)

        Parameters
        ----------
        k_min_fit : float, optional
            Minimum k (A^-1) for the background fit.
        k_max_fit : float or None, optional
            Maximum k (A^-1) for the background fit.
        k_min_window : float or None, optional
            Minimum k (A^-1) for the structure-factor Lorch window.
            If None, falls back to ``k_min_fit``.
        k_max_window : float or None, optional
            Maximum k (A^-1) for the structure-factor Lorch window.
            If None, falls back to ``k_max_fit``.
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
        density : float or None, optional
            Known number density (atoms/A^3). If provided together with
            ``damp_origin_oscillations=True``, the S(k)/G(r) correction uses
            this value instead of estimating it.
        r_cut : float, optional
            Minimum radial distance (A) for peak search in density estimation.
            Forwarded to :meth:`estimate_density`.
        returnval : bool, optional
            If True, return ``[r, G(r)]`` as numpy arrays.

        Returns
        -------
        list[np.ndarray] or None
        """
        # clear results from any previous run so stale state doesn't leak
        self.Fk_damped = None
        self.reduced_pdf_damped = None
        self.rho0 = None
        # this is missing a 2pi term that we add back during the pdf calc later
        k_np = np.asarray(self.qq)
        k = torch.from_numpy(k_np.astype(np.float32)).to(device=self.device)
        dk = k[1] - k[0]
        # small epsilon to avoid division by very small k values
        k_safe = torch.clamp(k, min=1e-10)
        self.kmax_fit = k_max_fit if k_max_fit is not None else float(k.max())
        self.kmin_fit = k_min_fit if k_min_fit is not None else float(k.min())
        # window range defaults to bg-fit range when not specified
        self.kmin_window = k_min_window if k_min_window is not None else self.kmin_fit
        self.kmax_window = k_max_window if k_max_window is not None else self.kmax_fit

        # Validate the real-space mask, if provided, before using it downstream
        mask_bool = None
        if mask_realspace is not None:
            scan_row, scan_col = self.polar.shape[:2]
            mask_realspace = np.asarray(mask_realspace)
            if mask_realspace.dtype == bool and mask_realspace.shape == (scan_row, scan_col):
                mask_bool = mask_realspace
            else:
                raise ValueError(
                    f"Got shape {mask_realspace.shape}. "
                    "mask_realspace must be boolean array of shape "
                    f"({scan_row}, {scan_col})."
                )
        # Recompute the radial mean whenever a real-space mask is given 
        # only reuse the cache when no mask is passed, otherwise calculate_Gr 
        # would silently ignore mask_realspace.
        if self.Ik is not None and mask_bool is None:
            Ik = self.Ik
        else:
            Ik = self.calculate_radial_mean(mask_realspace=mask_bool, returnval=True)
        # Likewise re-fit the background when the region changed.
        if self.bg is not None and self.f is not None and mask_bool is None:
            bg, f = self.bg, self.f
        else:
            bg, f = self.fit_bg(Ik, self.kmin_fit, self.kmax_fit)
        # prevent division by near-zero values which cause NaNs at high k
        f_safe = torch.clamp(f, min=1e-10 * f.max())

        # below is the standard definition of F(k) used in PDF analysis, except for missing 2pi factor
        Fk = (Ik - bg) * k_safe / f_safe
        # apply optional frequency filtering for noise reduction
        Fk = self._frequency_filtering(Fk, k_lowpass, k_highpass, dk)
        # Compute Sk from Fk BEFORE applying the 2pi scaling,
        # so that estimate_density corrections are on the same scale
        self.Sk = torch.ones_like(k)
        mask = k > 0
        self.Sk = torch.where(mask, 1.0 + (Fk / k_safe), self.Sk)
        # apply that missing 2pi factor
        Fk = Fk * 2 * torch.pi
        # damp edges with lorch window
        wk = self._lorch_window(k, self.kmin_window, self.kmax_window)
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

        # optionally damped unphysical oscillations near the origin by iteratively estimating density and correcting F(k)
        if damp_origin_oscillations:
            density_est = self.estimate_density(
                density=density,
                r_cut=r_cut,
                max_iter=20,
                tol_percent=1e-1,
            )
            self.rho0 = density_est[0]
            self.Fk_damped = density_est[1]
            self.reduced_pdf_damped = density_est[2]

        if returnval:
            Gr = (
                self.reduced_pdf_damped
                if self.reduced_pdf_damped is not None
                else self._reduced_pdf
            )
            return [to_numpy(self._r), to_numpy(Gr)]
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
            raise RuntimeError(
                "Reduced PDF not computed."
                "Run PairDistributionFunction.calculate_Gr() before calculate_gr()."
            )

        # Determine density
        if density is not None:
            rho0 = density
        elif self.rho0 is not None:
            rho0 = self.rho0
            print(f"  Using estimated rho0 = {rho0:.6f} atoms/A^3", flush=True)
        else:
            # the oscillation correction simultaneously produces a density estimate
            # if the user didn't run damping in calculate_Gr, we can still run the density estimation without using the corrected Fk/G(r)
            density_est = self.estimate_density(
                r_cut=r_cut,
                max_iter=20,
                tol_percent=1e-1,
            )
            self.rho0 = density_est[0]
            rho0 = self.rho0
            print(f"  Estimated rho0 = {rho0:.6f} atoms/A^3", flush=True)

        # Use damped G(r) if the user opted into damping, otherwise undamped
        Gr = self.reduced_pdf_damped if self.reduced_pdf_damped is not None else self._reduced_pdf
        Gr = Gr.clone()

        r = self._r
        mask = r > 0
        pdf = torch.ones_like(Gr)
        # the formula for g(r) from G(r) is: g(r) = 1 + G(r) / (4 * pi * r * rho0)
        pdf = torch.where(mask, 1 + Gr / (4 * torch.pi * r * rho0), torch.zeros_like(pdf))
        if set_pdf_positive:  # negative values are unphysical
            pdf = torch.maximum(pdf, torch.zeros_like(pdf))

        self._pdf = pdf
        if returnval:
            return [to_numpy(self._r), to_numpy(self._pdf)]
        return None

    def estimate_density(
        self,
        density: float | None = None,
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

        If ``density`` is provided, the given value is used as a fixed rho0
        for the S(k)/G(r) correction instead of estimating it iteratively.

        This method requires that :meth:`calculate_Gr` has already been run,
        because it depends on `self.Sk`, `self.reduced_pdf`, `self.r`,
        and the k-window bounds (`self.kmin_fit`, `self.kmin_window`,
        `self.kmax_window`).

        Parameters
        ----------
        density : float or None, optional
            Known number density (atoms/A^3). If provided, used as a fixed
            rho0 — the iterative estimation is skipped and only the S(k)/G(r)
            correction is performed.
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
            Number density (atoms/A^3), either provided or estimated.
        Fk_win_damped : torch.Tensor
            Windowed corrected reduced structure function used for the transform.
        G_cor : torch.Tensor
            Reduced PDF G(r) with dampened oscillations near origin.
        """
        # we need the non-reduced structure factor (S(k) = 1 + F(k)/k) for the density estimation correction,
        # so we compute it here from the Fk we already have
        if self.Sk is None or self._reduced_pdf is None or self._r is None:
            raise RuntimeError(
                "This method depends on Sk, reduced_pdf, and r from calculate_Gr. "
                "Run PairDistributionFunction.calculate_Gr() before estimate_density()."
            )

        k = torch.from_numpy(np.asarray(self.qq).astype(np.float32)).to(device=self.device)
        dk = k[1] - k[0]
        k_fit_mask = (k >= self.kmin_fit) & (k <= self.kmax_window)
        k_fit = k[k_fit_mask]
        ka, ra = torch.meshgrid(k, self._r, indexing="ij")

        # r_cut sets the minimum r for the peak search used to determine the correction interval
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
        k_fit_scaled = k_fit * 2 * torch.pi
        k2d_fit, r2d_fit = torch.meshgrid(k_fit_scaled, r_short, indexing="ij")

        # Iterative refinement of rho0 and S(k)
        fixed_density = density is not None
        rho0 = density if fixed_density else 0.0
        rho0_prev = None
        Sk_cor = self.Sk.clone()
        # calculate lorch function once bc it doesn't change during the iteration
        wk = self._lorch_window(k, self.kmin_window, self.kmax_window)
        # windowed G(r) for the iteration
        Fk_win = k * (Sk_cor - 1.0) * wk * 2 * torch.pi
        G_iter = (
            (2.0 / torch.pi)
            * dk
            * 2
            * torch.pi
            * torch.sum(torch.sin(2 * torch.pi * ka * ra) * Fk_win[:, None], dim=0)
        )
        G_iter[0] = 0.0
        G_beta = G_iter[r_mask]
        beta_prev = None
        for j in range(max_iter):
            if j > 0:
                G_beta = G_iter[r_mask]
            # calculate alpha/beta for S(k) adjustment
            # alpha and beta are the ideal and actual contributions to G(r) in the short-r range
            # from the current S(k) and G(r)
            alpha, beta = self._compute_alpha_beta(k2d_fit, r2d_fit, G_beta, r_short)
            if not fixed_density:
                rho0 = float(torch.sum(alpha * beta) / torch.sum(alpha**2))
                if rho0_prev is not None:
                    Rj = abs(rho0_prev - rho0) / abs(rho0) * 100.0
                    if Rj < tol_percent:
                        break
            else:
                # fixed density: converge on the S(k) correction magnitude
                if beta_prev is not None:
                    delta = float(torch.max(torch.abs(beta - beta_prev)))
                    if delta < tol_percent * 1e-2:
                        break
                beta_prev = beta.clone()
            # Update S_cor(k) and G_cor
            Sk_cor[k_fit_mask] = Sk_cor[k_fit_mask] - beta + rho0 * alpha
            Fk_win = k * (Sk_cor - 1.0) * wk * 2 * torch.pi
            G_iter = (
                (2.0 / torch.pi)
                * dk
                * 2
                * torch.pi
                * torch.sum(torch.sin(2 * torch.pi * ka * ra) * Fk_win[:, None], dim=0)
            )
            G_iter[0] = 0.0
            rho0_prev = rho0
        return rho0, Fk_win, G_iter

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

        if self.Ik is None:
            raise RuntimeError(
                "Radial mean intensity has not been calculated yet."
                "Run PairDistributionFunction.calculate_Gr() or PairDistributionFunction.calculate_radial_mean() before plotting."
            )

        x = np.asarray(self.qq)
        y = to_numpy(self.Ik)
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
            raise RuntimeError(
                "Radial mean intensity or background has not been calculated yet."
                "Run PairDistributionFunction.calculate_Gr() or both calculate_radial_mean() and calculate_background() before plotting."
            )

        x = np.asarray(self.qq)
        y1 = to_numpy(self.Ik)
        x, y1 = self._apply_xrange(x, y1, qmin, qmax)
        x = np.asarray(self.qq)
        y2 = to_numpy(self.bg)
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
            raise RuntimeError(
                "Reduced structure factor F(k) has not been calculated yet."
                "Run PairDistributionFunction.calculate_Gr() before plotting."
            )

        Fk = getattr(self, "Fk_damped", None)
        if Fk is None:
            Fk = self.Fk_masked

        x = np.asarray(self.qq)
        y = to_numpy(Fk)
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
            raise RuntimeError(
                "Reduced PDF has not been calculated yet."
                "Run PairDistributionFunction.calculate_Gr() before plotting."
            )
        Gr = self.reduced_pdf_damped if self.reduced_pdf_damped is not None else self._reduced_pdf

        x = to_numpy(self._r)
        y = to_numpy(Gr)
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
            raise RuntimeError(
                "PDF has not been calculated yet."
                "Run PairDistributionFunction.calculate_gr() before plotting."
            )

        x = to_numpy(self._r)
        y = to_numpy(self._pdf)
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
        if self.Fk_masked is None or self.Fk_damped is None or self.reduced_pdf_damped is None:
            raise RuntimeError(
                "Oscillation damping data not available. "
                "Run calculate_Gr(damp_origin_oscillations=True) first."
            )

        k = np.asarray(self.qq)

        # Convert torch tensors to numpy for plotting
        Fk_masked = to_numpy(self.Fk_masked)
        Fk_damped = to_numpy(self.Fk_damped)
        r = to_numpy(self._r)
        reduced_pdf = to_numpy(self._reduced_pdf)
        reduced_pdf_damped = to_numpy(self.reduced_pdf_damped)

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

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

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
                raise ValueError(
                    "k_highpass is greater than k_lowpass."
                    "Gaussian band-pass filtering requires k_highpass < k_lowpass."
                )
            low_kernel = gaussian_kernel_1d(k_lowpass / dk.item()).to(self.device)
            high_kernel = gaussian_kernel_1d(k_highpass / dk.item()).to(self.device)
            Fk_low = gaussian_filter_1d(Fk, low_kernel)
            Fk_high = gaussian_filter_1d(Fk, high_kernel)
            Fk = Fk_high - Fk_low
        elif k_lowpass is not None and k_lowpass > 0.0:
            low_kernel = gaussian_kernel_1d(k_lowpass / dk.item()).to(self.device)
            Fk = gaussian_filter_1d(Fk, low_kernel)
        elif k_highpass is not None and k_highpass > 0.0:
            high_kernel = gaussian_kernel_1d(k_highpass / dk.item()).to(self.device)
            Fk_high = gaussian_filter_1d(Fk, high_kernel)
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
