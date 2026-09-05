import gc
import math
import warnings
from typing import TYPE_CHECKING, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.datastructures import Dataset2d, Dataset3d, Dataset4d
from quantem.core.utils.validators import (
    validate_aberration_coefficients,
    validate_tensor,
)
from quantem.diffractive_imaging.complex_probe import (
    FourierProbe,
    aberration_surface,
    aberration_surface_cartesian_basis,
    aberration_surface_cartesian_gradients,
    aperture,
    gamma_factor,
    merge_aberration_coefficients,
    polar_coordinates,
    spatial_frequencies,
)
from quantem.diffractive_imaging.ptycho_utils import SimpleBatcher

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch

from quantem.diffractive_imaging.direct_ptycho_utils import (
    ABERRATION_PRESETS,
    _crop_corner_centered_mask,
    _rotation_degrees_to_radians,
    align_vbf_stack_multiscale,
    build_vbf_stack_from_dataset3d,
    build_vbf_stack_from_dataset4d,
    fit_aberrations_from_shifts,
    group_basis_by_method,
    regrid_vbf_stack,
    unwrap_bf_overlap_phase_torch,
)
from quantem.diffractive_imaging.direct_ptychography_base import (
    DirectPtychographyBase,
    HyperparameterState,
)

# re-exported because this was `OptimizationParameter`'s public path in v0.1.9, and
# published notebooks still import it from this module
from quantem.diffractive_imaging.direct_ptychography_base import (
    OptimizationParameter as OptimizationParameter,
)


class DirectPtychography(DirectPtychographyBase):
    """
    Direct ptychography in the scan-space Fourier domain.

    Every kernel here -- SSB, OBF, matched-filter, iCoM and parallax -- is a multiplier on
    ``G(k, q)``, the virtual bright-field stack Fourier transformed over the scan. The
    object is recovered by summing those multipliers over bright-field pixels and
    transforming back once.

    Because the multiplication happens in ``q``, the scan has to lie on a regular grid.
    :meth:`from_dataset4d` and :meth:`from_virtual_bfs` are given one; :meth:`from_dataset3d`
    resamples ungridded positions onto one, which is where its caveats come from.

    The alternative is
    :class:`~quantem.diffractive_imaging.direct_ptychography_montage.DirectPtychographyMontage`,
    which accumulates the scan onto a real-space canvas and needs no grid. See its class
    docstring for the full comparison; in short, this class is exact and cheapest on a
    gridded scan, but a bright-field mask beyond a few tens of thousands of pixels will not
    fit in memory here.

    Instantiate with :meth:`from_dataset4d`, :meth:`from_virtual_bfs` or
    :meth:`from_dataset3d`.
    """

    _token = object()

    def __init__(
        self,
        vbf_dataset: Dataset3d,
        bf_mask_dataset: Dataset2d,
        energy: float | None,
        rotation_angle: float,
        aberration_coefs: dict,
        semiangle_cutoff: float,
        soft_edges: bool,
        crop_bf_mask: bool,
        bf_mask_padding_px: int,
        rng: np.random.Generator | int | None,
        device: str | int,
        verbose: int | bool,
        wavelength: float | None = None,
        fourier_probe: "FourierProbe | None" = None,
        _token: object | None = None,
    ):
        """ """
        if _token is not self._token:
            raise RuntimeError(
                "Use DirectPtychography.from_dataset4dstem() or DirectPtychography.from_virtual_bfs() to instantiate this class."
            )

        self.device = device
        self.verbose = verbose
        self.vbf_stack = vbf_dataset.array  # ty:ignore[invalid-assignment]
        self.bf_mask = bf_mask_dataset.array  # ty:ignore[invalid-assignment]
        if crop_bf_mask:
            self.bf_mask = _crop_corner_centered_mask(self.bf_mask, bf_mask_padding_px)

        self.wavelength = self._resolve_wavelength(energy, wavelength)
        self.scan_units = vbf_dataset.units[-2:]
        self.detector_units = bf_mask_dataset.units

        self.scan_gpts = tuple(int(n) for n in vbf_dataset.shape[-2:])
        self.scan_sampling = vbf_dataset.sampling[-2:]
        self.reciprocal_sampling = bf_mask_dataset.sampling
        self.angular_sampling = tuple(d * 1e3 * self.wavelength for d in self.reciprocal_sampling)

        self.num_bf = vbf_dataset.shape[0]
        self.gpts = tuple(int(n) for n in self.bf_mask.shape[:2])
        self.sampling = tuple(1 / s / n for n, s in zip(self.reciprocal_sampling, self.gpts))

        self.fourier_probe = fourier_probe
        self.semiangle_cutoff = semiangle_cutoff  # ty:ignore[invalid-assignment]
        self.soft_edges = soft_edges
        self.rng = rng

        self.hyperparameter_state = HyperparameterState(
            initial_aberrations=aberration_coefs, initial_rotation_angle=rotation_angle
        )

        self._preprocess()

    @classmethod
    def from_virtual_bfs(
        cls,
        vbf_dataset: Dataset3d,
        bf_mask_dataset: Dataset2d,
        energy: float | None = None,
        rotation_angle: float | None = None,
        semiangle_cutoff: float | None = None,
        aberration_coefs: dict = {},
        wavelength: float | None = None,
        fourier_probe: "FourierProbe | None" = None,
        soft_edges: bool = True,
        crop_bf_mask: bool = True,
        bf_mask_padding_px: int = 1,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
    ):
        """ """

        return cls(
            vbf_dataset=vbf_dataset,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            wavelength=wavelength,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            semiangle_cutoff=semiangle_cutoff,
            fourier_probe=fourier_probe,
            soft_edges=soft_edges,
            crop_bf_mask=crop_bf_mask,
            bf_mask_padding_px=bf_mask_padding_px,
            rng=rng,
            device=device,
            verbose=verbose,
            _token=cls._token,
        )

    @classmethod
    def from_dataset4d(
        cls,
        dataset: Dataset4d,
        energy: float | None = None,
        semiangle_cutoff: float | None = None,
        aberration_coefs: dict = {},
        wavelength: float | None = None,
        fourier_probe: "FourierProbe | None" = None,
        rotation_angle: float | None = None,
        max_batch_size: int | None = None,
        fit_method: str = "plane",
        mode: str = "bilinear",
        force_measured_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        force_fitted_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        intensity_threshold: float = 0.5,
        soft_edges: bool = True,
        crop_bf_mask: bool = True,
        bf_mask_padding_px: int = 1,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
        normalization_order: int = 0,
        edge_blend_pixels: int = 0,
    ):
        """ """

        vbf_dataset, bf_mask_dataset, rotation_angle = build_vbf_stack_from_dataset4d(
            dataset,
            device=device,
            max_batch_size=max_batch_size,
            fit_method=fit_method,
            mode=mode,
            force_measured_origin=force_measured_origin,
            force_fitted_origin=force_fitted_origin,
            rotation_angle=rotation_angle,
            intensity_threshold=intensity_threshold,
            normalization_order=normalization_order,
            edge_blend_pixels=edge_blend_pixels,
        )

        return cls(
            vbf_dataset=vbf_dataset,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            wavelength=wavelength,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            semiangle_cutoff=semiangle_cutoff,
            fourier_probe=fourier_probe,
            soft_edges=soft_edges,
            crop_bf_mask=crop_bf_mask,
            bf_mask_padding_px=bf_mask_padding_px,
            rng=rng,
            device=device,
            verbose=verbose,
            _token=cls._token,
        )

    @classmethod
    def from_dataset3d(
        cls,
        dataset: Dataset3d,
        positions: Dataset2d | torch.Tensor | NDArray,
        energy: float | None = None,
        semiangle_cutoff: float | None = None,
        rotation_angle: float | None = None,
        scan_sampling: Tuple[float, float] | Literal["auto"] = "auto",
        aberration_coefs: dict = {},
        wavelength: float | None = None,
        fourier_probe: "FourierProbe | None" = None,
        bf_mask: torch.Tensor | NDArray | None = None,
        scan_gpts: Tuple[int, int] | None = None,
        interpolation: Literal["nearest", "bilinear"] = "nearest",
        hole_fill: Literal["mean", "zero"] = "mean",
        max_batch_size: int | None = None,
        fit_method: str = "plane",
        mode: str = "bilinear",
        force_measured_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        force_fitted_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        intensity_threshold: float = 0.5,
        soft_edges: bool = True,
        crop_bf_mask: bool = True,
        bf_mask_padding_px: int = 1,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
        normalization_order: int = 0,
    ):
        """
        Build from an ungridded stack of diffraction patterns and their probe positions.

        Every kernel here is a multiplier on the scan-space Fourier transform, which needs a
        regular grid. So the bright-field images are resampled onto one first, by splatting
        them at their probe positions with no aberration shift and dividing by the
        accumulated weight. Once gridded, SSB, OBF, matched-filter and iCoM all run exactly
        as they do for a raster scan.

        Parameters
        ----------
        dataset : Dataset3d
            ``(N, Qx, Qy)`` diffraction patterns, reciprocal units ``"A^-1"`` or ``"mrad"``.
        positions : Dataset2d, torch.Tensor or ndarray
            ``(N, 2)`` probe positions in Angstrom, ordered ``(row, col)`` to match the
            diffraction axes. A ``Dataset2d`` must carry units ``"A"``.
        rotation_angle : float
            Detector rotation in degrees. Required: rotation is otherwise estimated from the
            curl of the center of mass over a 2D scan grid, which an ungridded scan lacks.
        scan_sampling : tuple of float or "auto"
            Grid pixel size in Angstrom. ``"auto"`` uses the median nearest-neighbour
            position spacing and warns with the inferred value.

            This, rather than ``reconstruct(upsampling_factor=...)``, is how to sample more
            finely than the scan. Pass a fraction of the position spacing and the empty
            pixels in between become the sparse comb the deconvolution unfolds, with its
            teeth on the real probe positions. Measured against the analytical parallax CTF
            on a scan scattered by a full pixel, correlation over the band above the scan
            Nyquist: 0.720 for binning then ``upsampling_factor=2``, against 0.997 for
            halving ``scan_sampling`` instead, and 0.982 for the montage.
        scan_gpts : tuple of int, optional
            Grid size. Defaults to whatever just covers the positions at ``scan_sampling``,
            and may only be larger than that -- ``scan_sampling`` stays authoritative, so
            this pads the canvas rather than rescaling the positions into it.
        interpolation : {"nearest", "bilinear"}
            Deposition scheme, matching ``DirectPtychographyMontage.reconstruct``.
            ``"nearest"`` by default; see :func:`regrid_vbf_stack`.
        hole_fill : {"mean", "zero"}
            What to put in grid pixels no probe position reached. Defaults to each
            bright-field image's mean over the visited pixels, which matters on a masked or
            irregular scan and is also what centers a comb's gaps -- see
            :func:`regrid_vbf_stack`.

        Notes
        -----
        Regridding has two failure modes, both of which raise a warning.

        Avoid combining this with ``reconstruct(upsampling_factor > 1)`` on an irregular
        scan; pass a finer ``scan_sampling`` instead. Upsampling at reconstruct time recovers
        detail above the scan Nyquist from where each probe sat, which binning onto a grid
        discards, so the extra band returns as a replica of the contrast-transfer function
        rather than an extension of it. With no holes, correlation over the extension band
        falls 0.997 -> 0.822 -> 0.711 as the sub-pixel scatter grows from 0 to 0.5 to 1.0
        grid pixels.

        The second is holes: a grid pixel no probe reached has to be invented. ``hole_fill``
        keeps that from becoming a hard-edged step, but a large filled region still biases
        the result, and a coarser ``scan_gpts`` is the usual fix. Contiguous holes are worse
        than scattered ones at equal fraction -- scattered holes behave like noise, while an
        excluded region is a low-frequency mask the deconvolution spreads everywhere. Note
        that a masked scan and an upsampled one want opposite treatments, filled holes versus
        deliberate gaps, and ``hole_fill`` cannot do both at once.

        Both are properties of the regridding rather than the deconvolution, so neither
        applies to
        :class:`~quantem.diffractive_imaging.direct_ptychography_montage.DirectPtychographyMontage`,
        which uses the positions as measured. Prefer it for a scan that is sparse, strongly
        irregular, or both masked and upsampled; see its class docstring for the comparison.
        """
        (
            vbf_stack,
            positions_px,
            bf_mask_dataset,
            fitted_gpts,
            scan_sampling,
            rotation_angle,
            scan_origin,
        ) = build_vbf_stack_from_dataset3d(
            dataset,
            positions,
            scan_sampling,
            device=device,
            max_batch_size=max_batch_size,
            fit_method=fit_method,
            mode=mode,
            force_measured_origin=force_measured_origin,
            force_fitted_origin=force_fitted_origin,
            rotation_angle=rotation_angle,
            intensity_threshold=intensity_threshold,
            normalization_order=normalization_order,
            bf_mask=bf_mask,
        )

        if scan_gpts is None:
            scan_gpts = fitted_gpts
        else:
            # `scan_sampling` stays authoritative, so the grid is padded, never rescaled:
            # resizing the pixel to fill a requested shape would contradict the caller
            scan_gpts = tuple(int(n) for n in scan_gpts)
            if any(n < f for n, f in zip(scan_gpts, fitted_gpts)):
                raise ValueError(
                    f"`scan_gpts={scan_gpts}` is smaller than the {fitted_gpts} needed to "
                    f"cover the positions at scan_sampling={scan_sampling}; positions would "
                    "be dropped. Pass a coarser `scan_sampling` instead."
                )

        gridded, hole_fraction, occupied = regrid_vbf_stack(
            vbf_stack,
            positions_px,
            scan_gpts,
            interpolation=interpolation,
            hole_fill=hole_fill,
        )
        if verbose:
            occupancy = vbf_stack.shape[1] / (scan_gpts[0] * scan_gpts[1])
            detail = (
                f"grid is finer than the scan ({occupancy:.2f} positions per pixel), so "
                f"{hole_fraction:.1%} of it is comb gaps for the deconvolution to unfold"
                if occupancy < 1
                else f"{hole_fraction:.1%} of grid pixels unvisited"
            )
            print(
                f"Regridded {vbf_stack.shape[1]} positions onto "
                f"{scan_gpts[0]}x{scan_gpts[1]}; {detail}."
            )

        vbf_dataset = Dataset3d.from_array(
            gridded.cpu().numpy(),
            name="regridded virtual BF stack",
            sampling=(1.0, *scan_sampling),
            units=("index", "A", "A"),
        )

        reconstruction = cls.from_virtual_bfs(
            vbf_dataset=vbf_dataset,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            wavelength=wavelength,
            rotation_angle=rotation_angle,
            semiangle_cutoff=semiangle_cutoff,
            aberration_coefs=aberration_coefs,
            fourier_probe=fourier_probe,
            soft_edges=soft_edges,
            crop_bf_mask=crop_bf_mask,
            bf_mask_padding_px=bf_mask_padding_px,
            rng=rng,
            device=device,
            verbose=verbose,
        )
        # the grid was anchored at the position bounding box, so record where that was --
        # this is what lets `obj_origin` be read in the caller's own coordinates
        reconstruction.scan_origin = scan_origin

        # how far the positions sit from the grid they were binned onto. Upsampling unfolds
        # aliased detail from where each probe actually was, which regridding discards, so
        # this predicts whether `upsampling_factor` can work at all
        subpixel = positions_px - np.round(positions_px)
        lattice_rms_px = float(np.sqrt((subpixel**2).sum(axis=1).mean()))

        reconstruction._regrid_info = {
            "hole_fraction": hole_fraction,
            "occupied": occupied,
            "positions_px": positions_px,
            "scan_gpts": scan_gpts,
            "scan_sampling": scan_sampling,
            "lattice_rms_px": lattice_rms_px,
        }
        return reconstruction

    #: sub-pixel scatter, in grid pixels, above which regridding has destroyed enough of the
    #: probe positions that upsampling replicates the band instead of extending it
    _UNFOLDING_RMS_LIMIT = 0.1

    def _warn_if_upsampling_cannot_unfold(self, upsampling_factor):
        """Warn when upsampling an irregular regridded scan, which cannot unfold.

        ``upsampling_factor`` recovers detail above the scan Nyquist from the aliased content
        of the bright-field images, which depends on where each probe actually sat. Binning
        those measurements onto a grid throws that away, so on an irregular scan the extra
        band comes back as a replica of the CTF rather than an extension of it.

        Measured on a white-noise object against the analytical parallax CTF, with no holes
        at all: correlation over the band above the scan Nyquist falls 0.997 -> 0.965 ->
        0.822 -> 0.711 as the sub-pixel scatter grows 0 -> 0.25 -> 0.5 -> 1.0 grid pixels,
        while the montage holds 0.995 -> 0.983 on the same data.
        """
        info = getattr(self, "_regrid_info", None)
        if upsampling_factor <= 1 or info is None:
            return
        if info["lattice_rms_px"] <= self._UNFOLDING_RMS_LIMIT:
            return

        warnings.warn(
            f"This reconstruction was regridded from ungridded positions that sit "
            f"{info['lattice_rms_px']:.2f} grid pixels from the grid on average, and "
            f"`upsampling_factor={upsampling_factor}` cannot unfold that. Upsampling recovers "
            "detail above the scan Nyquist from where each probe actually was, which the "
            "regridding has already discarded, so the extra band will come back as a replica "
            "of the contrast-transfer function rather than an extension of it. Rebuild with "
            f"a `scan_sampling` {upsampling_factor}x finer instead, which keeps the probe "
            "positions on the grid rather than binning them away first, or use "
            "DirectPtychographyMontage, which needs no grid at all.",
            stacklevel=2,
        )

    def _preprocess(
        self,
    ):
        """ """

        self._vbf_fourier = torch.fft.fft2(self.vbf_stack, dim=(-2, -1))
        self._dc_per_image = self._vbf_fourier[..., 0, 0].mean(0)
        self._vbf_fourier[..., 0, 0] = 0  # zero DC
        self._corrected_stack = None
        self._q_signal_power = self._vbf_fourier.abs().square().sum(dim=0)  # (N_qx, N_qy)

        return self

    def _return_upsampled_qgrid(
        self,
        upsampling_factor=None,
    ):
        """
        Assumes integer upsampling factor.
        """

        if upsampling_factor is None:
            scan_gpts = self.scan_gpts
            scan_sampling = self.scan_sampling
        else:
            scan_gpts = tuple(n * upsampling_factor for n in self.scan_gpts)
            scan_sampling = tuple(s / upsampling_factor for s in self.scan_sampling)

        qxa, qya = spatial_frequencies(scan_gpts, scan_sampling, device=self.device)

        return qxa, qya

    @property
    def vbf_stack(self) -> torch.Tensor:
        return self._vbf_stack

    @vbf_stack.setter
    def vbf_stack(self, value: torch.Tensor):
        self._vbf_stack = validate_tensor(value, "vbf_stack", dtype=torch.float).to(
            device=self.device
        )

    def _return_kernel_contributions(
        self,
        bf,
        deconvolution_kernel,
        vbf_fourier,
        kxa,
        kya,
        qxa,
        qya,
        cmplx_probe_k,
        grad_k,
        sign_sin_chi_q,
        aberration_coefs,
        batch_idx,
    ):
        """ """
        ind_i = bf.bf_inds_i[batch_idx]
        ind_j = bf.bf_inds_j[batch_idx]

        kx = kxa[ind_i, ind_j].view(-1, 1, 1)
        ky = kya[ind_i, ind_j].view(-1, 1, 1)

        power = None

        if deconvolution_kernel in ("ssb", "obf", "mf"):
            qmkxa = qxa.unsqueeze(0) - kx
            qmkya = qya.unsqueeze(0) - ky
            qpkxa = qxa.unsqueeze(0) + kx
            qpkya = qya.unsqueeze(0) + ky

            cmplx_probe_at_k = cmplx_probe_k[ind_i, ind_j].view(-1, 1, 1)

            gamma = gamma_factor(
                (qmkxa, qmkya),
                (qpkxa, qpkya),
                cmplx_probe_at_k,
                self._return_probe(aberration_coefs),
                normalize=False,
            )

            fourier_factor = -1.0j * vbf_fourier * gamma.conj()
            abs_gamma = gamma.abs()

            if deconvolution_kernel == "ssb":
                fourier_factor = fourier_factor / abs_gamma.clip(1e-8)
            else:
                power = abs_gamma.square().sum(0)

        elif deconvolution_kernel == "prlx":
            qvec = torch.stack((qxa, qya), 0)
            grad_kq = torch.einsum("na,amp->nmp", grad_k[batch_idx], qvec)
            operator = torch.exp(-1j * grad_kq) * sign_sin_chi_q
            fourier_factor = vbf_fourier * operator

        else:
            q2 = qxa.square() + qya.square()
            qx_op = -1.0j * qxa / q2
            qy_op = -1.0j * qya / q2
            qx_op[0, 0] = 0.0
            qy_op[0, 0] = 0.0

            operator = kx * qx_op + ky * qy_op
            fourier_factor = vbf_fourier * operator

        return fourier_factor, power

    def reconstruct(
        self,
        bf_mask=None,
        override_aberration_coefs=None,
        upsampling_factor=None,
        override_rotation_angle=None,
        max_batch_size=None,
        deconvolution_kernel="single-sideband",
        q_highpass=None,
        q_lowpass=None,
        butterworth_order=12,
        matched_filter_norm_epsilon=1e-1,
        parallax_flip_phase=True,
        verbose=None,
        use_initial_state=False,
    ):
        """
        Unified reconstruction method supporting multiple deconvolution techniques.

        Parameters
        ----------
        bf_mask: torch.Tensor, optional
            Subset of bright field mask to use for reconstruction. Note this must be
            strictly smaller than the bf_mask used for initialization.
        override_aberration_coefs : dict, optional
            Aberration coefficients for the probe
        upsampling_factor : int, optional
            Factor by which to upsample the reconstruction
        override_rotation_angle : float, optional
            Rotation angle for coordinate system, in degrees
        max_batch_size : int, optional
            Maximum batch size for processing
        deconvolution_kernel : str, one of ['ssb', 'obf', 'mf','prlx','icom']
        q_highpass : float, optional
            High-pass filter cutoff
        q_lowpass : float, optional
            Low-pass filter cutoff
        verbose : bool, optional
            If True, show progress bar

        Returns
        -------
        self
            Returns self with corrected_stack attribute set
        """

        state = self.hyperparameter_state

        if verbose is None:
            verbose = self.verbose

        if use_initial_state:
            if verbose:
                print("Reconstructing with:\n\n", state.summarize(which="initial"))
            aberration_coefs = state.initial_aberrations
            rotation_angle = state.initial_rotation_angle
        else:
            if verbose:
                print(
                    "Reconstructing with:\n\n",
                    state.summarize(
                        which="current",
                        override_aberration_coefs=override_aberration_coefs,
                        override_rotation_angle=override_rotation_angle,
                    ),
                )
            aberration_coefs = state.current_aberrations(override_aberration_coefs)
            rotation_angle = state.current_rotation_angle(override_rotation_angle)

        if upsampling_factor is None:
            upsampling_factor = 1
        upsampling_factor = math.ceil(upsampling_factor)
        self._warn_if_upsampling_cannot_unfold(upsampling_factor)

        if bf_mask is None:
            bf_mask = self.bf_mask
        bf = self._return_bf_context(bf_mask)

        num_bf = bf.num_bf
        bf_mask = bf.bf_mask
        vbf_index_mapping = bf.vbf_index_mapping

        if max_batch_size is None:
            max_batch_size = num_bf

        deconvolution_kernel = self._normalize_kernel_name(deconvolution_kernel)
        if deconvolution_kernel == "prlx":
            # iCoM is exempt: `k . q / |q|**2` never reads the probe
            self._require_analytic_probe("The prlx kernel")

        # Get upsampled q-space grid
        qxa, qya = self._return_upsampled_qgrid(upsampling_factor)
        q, theta = polar_coordinates(qxa, qya)

        # Get k-space grid
        kxa, kya = spatial_frequencies(
            self.gpts,
            self.sampling,
            rotation_angle=_rotation_degrees_to_radians(rotation_angle),
            device=self.device,
        )
        k, phi = polar_coordinates(kxa, kya)

        # compute global / cheap functions for prlx
        if deconvolution_kernel == "prlx":
            dx, dy = aberration_surface_cartesian_gradients(
                k * self.wavelength,
                phi,
                aberration_coefs=aberration_coefs,
            )
            grad_k = torch.stack((dx[bf_mask], dy[bf_mask]), -1)

            if parallax_flip_phase:
                chi_q = aberration_surface(
                    q * self.wavelength,
                    theta,
                    self.wavelength,
                    aberration_coefs=aberration_coefs,
                )
                sign_sin_chi_q = torch.sign(torch.sin(chi_q))
            else:
                sign_sin_chi_q = torch.ones_like(q)
        else:
            grad_k = None
            sign_sin_chi_q = None

        # compute global / cheap functions for all
        cmplx_probe_k = self._return_probe_on_grid(k, phi, aberration_coefs)
        BF_weights = cmplx_probe_k[bf_mask].abs().square().sum()

        butterworth_env = torch.ones_like(q)
        if q_lowpass:
            butterworth_env *= 1 / (1 + (q / q_lowpass) ** (2 * butterworth_order))
        if q_highpass:
            butterworth_env *= 1 - 1 / (1 + (q / q_highpass) ** (2 * butterworth_order))

        # Process batches
        pbar = tqdm(range(num_bf), disable=not verbose)
        batcher = SimpleBatcher(num_bf, batch_size=max_batch_size, shuffle=False, rng=self.rng)

        fourier_factor = torch.empty(
            (num_bf,) + qxa.shape, device=self.device, dtype=torch.complex64
        )
        if deconvolution_kernel in ("obf", "mf"):
            power = torch.zeros(qxa.shape, device=self.device)
        else:
            power = None

        # first pass
        for batch_idx in batcher:
            mapped_idx = vbf_index_mapping[batch_idx]
            vbf_fourier = self._vbf_fourier[mapped_idx]

            # Fourier-space tiling
            vbf_fourier = torch.cat(
                [torch.cat([vbf_fourier] * upsampling_factor, dim=-1)] * upsampling_factor,
                dim=-2,
            )

            num, pow = self._return_kernel_contributions(
                bf,
                deconvolution_kernel,
                vbf_fourier,
                kxa,
                kya,
                qxa,
                qya,
                cmplx_probe_k,
                grad_k,
                sign_sin_chi_q,
                aberration_coefs,
                batch_idx,
            )
            if power is None:
                num *= butterworth_env
                # num[:, 0, 0] = self._dc_per_image

                fourier_factor[batch_idx] = torch.fft.ifft2(num)
            else:
                fourier_factor[batch_idx] = num
                power += pow

            pbar.update(len(batch_idx))
        pbar.close()

        if power is not None:
            power /= BF_weights

            if deconvolution_kernel == "obf":
                norm = power.sqrt().clamp_min(1e-8)
            elif deconvolution_kernel == "mf":
                norm = (power + matched_filter_norm_epsilon * power.max()).clamp_min(1e-8)

            # second pass
            for batch_idx in batcher:
                ff = fourier_factor[batch_idx]

                if power is not None:
                    ff /= norm

                ff *= butterworth_env
                # ff[:, 0, 0] = self._dc_per_image

                fourier_factor[batch_idx] = torch.fft.ifft2(ff)

        self.corrected_stack = fourier_factor.real / BF_weights

        # memory management
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return self

    @property
    def corrected_stack(self):
        return self._corrected_stack

    @corrected_stack.setter
    def corrected_stack(self, value: torch.Tensor):
        self._corrected_stack = validate_tensor(value, "corrected_stack", dtype=torch.float32).to(
            device=self.device
        )

    @property
    def corrected_bf(self):
        if self.corrected_stack is None:
            return None
        return self.corrected_stack.sum(dim=0)

    def variance_loss(self):
        """ """
        if self.corrected_stack is None:
            return None
        if self.corrected_stack.abs().sum() > 0:
            mean_corrected_bf = self.corrected_stack.mean(dim=0)
            variance_loss = ((self.corrected_stack - mean_corrected_bf).abs().square()).mean()
        else:
            variance_loss = torch.tensor(
                torch.inf, dtype=self.corrected_stack.dtype, device=self.device
            )
        return variance_loss

    def fit_hyperparameters_cross_correlation(
        self,
        bf_mask: torch.Tensor | None = None,
        rotation_angle: float | None = None,
        aberration_coefs: dict[str, int | float | torch.Tensor] = {},
        bin_factors: tuple[int, ...] = (3, 2, 1),
        pair_connectivity: int = 4,
        alignment_method: str = "reference",
        reference: torch.Tensor | NDArray | None = None,
        running_average: bool = False,
        regularize_shifts: bool = True,
        dft_upsample_factor: int = 4,
        verbose=None,
        use_initial_state=False,
        **reconstruct_kwargs,
    ):
        """
        Fit aberrations and rotation angle from virtual BF stack.

        Parameters
        ----------
        bf_mask: torch.Tensor, optional
            Subset of bright field mask to use for reconstruction. Note this must be
            strictly smaller than the bf_mask used for initialization.
        bin_factors : tuple of int
            Sequence of binning factors from coarse to fine
        pair_connectivity : int
            Neighbor connectivity for pairwise alignment (4 or 8)
        alignment_method : str
            Alignment strategy:
            - "pairwise": Graph-based pairwise alignment (most robust)
            - "reference": Align all images to a reference
        reference : array-like, optional
            Reference image for alignment_method="reference".
            If None, uses mean of initial reconstruction.
        running_average : bool
            If True and alignment_method="reference", updates reference as running
            average during alignment. Can help with noisy data.
        regularize_shifts : bool
            If True, constrains shifts to physical aberration model at each iteration.
        **reconstruct_kwargs
            Additional arguments passed to reconstruct methods.
            If aberration coefficients are provided (e.g., C10, C12, phi12) or rotation_angle,
            performs initial reconstruction to seed the alignment.

        Returns
        -------
        self : object
            Returns self with fitted parameters stored in self._fitted_parameters

        The running_average option updates the reference at each bin level as:
            ref_new = ref_old * n/(n+1) + aligned_mean / (n+1)
        """

        if verbose is None:
            verbose = self.verbose

        if bf_mask is None:
            bf_mask = self.bf_mask
        bf = self._return_bf_context(bf_mask)
        bf_mask = bf.bf_mask
        inds_i, inds_j = bf.bf_inds_i, bf.bf_inds_j

        scan_sampling = torch.as_tensor(
            self.scan_sampling, device=self.device, dtype=torch.float32
        )

        # initial reconstruction
        safe_kwargs = {
            k: v
            for k, v in reconstruct_kwargs.items()
            if k not in ["deconvolution_kernel", "parallax_flip_phase"]
        }

        state = self.hyperparameter_state
        state.clear_optimized()

        self.reconstruct(
            override_rotation_angle=rotation_angle,
            override_aberration_coefs=aberration_coefs,
            deconvolution_kernel="parallax",
            parallax_flip_phase=False,
            verbose=False,
            use_initial_state=use_initial_state,
            **safe_kwargs,
        )

        vbf_stack = self.corrected_stack.clone()

        # Get initial shifts
        lateral_shifts = self._return_lateral_shifts(rotation_angle, aberration_coefs, bf_mask)
        initial_shifts = lateral_shifts / scan_sampling

        if alignment_method == "reference":
            reference = (
                vbf_stack.mean(0)
                if reference is None
                else torch.as_tensor(reference, dtype=torch.float32, device=self.device)
            )
        else:
            reference = None

        if regularize_shifts:
            kxa, kya = spatial_frequencies(self.gpts, self.sampling, device=self.device)
            kvec = torch.dstack((kxa[bf_mask], kya[bf_mask])).view((-1, 2))
            basis = kvec * self.wavelength / scan_sampling
        else:
            basis = None

        shifts_px, vbf_stack = align_vbf_stack_multiscale(
            vbf_stack,
            bf_mask,
            inds_i,
            inds_j,
            bin_factors,
            pair_connectivity=pair_connectivity,
            upsample_factor=dft_upsample_factor,
            reference=reference,
            initial_shifts=initial_shifts,
            running_average=running_average,
            basis=basis,
            verbose=verbose,
        )

        lateral_shifts = shifts_px * scan_sampling

        fit_results = fit_aberrations_from_shifts(
            lateral_shifts,
            bf_mask,
            self.wavelength,
            self.gpts,
            self.sampling,
        )

        self.corrected_stack = vbf_stack

        fitted_aberration_coefs = fit_results.copy()
        fitted_rotation_angle = fitted_aberration_coefs.pop("rotation_angle", None)

        state.optimized_aberrations = validate_aberration_coefficients(fitted_aberration_coefs)
        state.optimized_rotation_angle = fitted_rotation_angle
        state.optimized_keys = {"C10", "C12", "phi12", "rotation_angle"}

        if verbose:
            print("Optimized state:\n\n", self.hyperparameter_state)

        self.reconstruct(verbose=False, **reconstruct_kwargs)

        return self

    def _select_q_modes(self, num_q_modes, signal_weight=0.5, proximity_sharpness=2.0):
        """
        Args:
            signal_weight: 0 = pure proximity, 1 = pure signal, 0.5 = balanced
            proximity_sharpness: higher = more concentrated around alpha_optimal
        """
        qxa, qya = self._return_upsampled_qgrid()
        alpha = torch.sqrt(qxa**2 + qya**2) * self.wavelength
        alpha_optimal = self.semiangle_cutoff * 1e-3

        # Normalize both metrics to [0, 1]
        signal_normalized = self._q_signal_power / self._q_signal_power.max()

        distance_penalty = torch.abs(alpha - alpha_optimal) / alpha_optimal
        proximity_score = torch.exp(-proximity_sharpness * distance_penalty**2)

        # Weighted geometric mean (more robust than arithmetic)
        selection_metric = (signal_normalized**signal_weight) * (
            proximity_score ** (1 - signal_weight)
        )

        flat_inds = torch.argsort(selection_metric.flatten(), descending=True)
        qi = flat_inds // selection_metric.shape[1]
        qj = flat_inds % selection_metric.shape[1]

        return qi[:num_q_modes], qj[:num_q_modes]

    def _fit_hyperparameters_least_squares_inner(
        self,
        bf_mask: torch.Tensor,
        aberration_coefs: dict[str, int | float | torch.Tensor] = {},
        rotation_angle: float | None = None,
        cartesian_basis: str | list[str] = "low_order",
        num_q_modes: int = 6,
        q_signal_weight: float = 0.05,
        unwrap_method: Literal["reliability-sorting", "poisson"] = "reliability-sorting",
        two_pass_unwrapping: bool = False,
    ):
        """
        Phase-only least-squares aberration fit from vBF Fourier data.
        Returns delta aberrations in cartesian form.
        """
        if isinstance(cartesian_basis, str):
            cartesian_basis = ABERRATION_PRESETS[cartesian_basis]

        device = self.device
        wavelength = self.wavelength

        # ---------------------------------------------------------
        # Select strongest spatial frequencies
        # ---------------------------------------------------------
        qi, qj = self._select_q_modes(num_q_modes, signal_weight=q_signal_weight)
        qxa, qya = self._return_upsampled_qgrid()
        qx = qxa[qi, qj]  # (N_q,)
        qy = qya[qi, qj]  # (N_q,)

        N_q = qx.numel()
        P = len(cartesian_basis)

        # ---------------------------------------------------------
        # k-space BF pixels
        # ---------------------------------------------------------
        kxa, kya = spatial_frequencies(
            self.gpts,
            self.sampling,
            rotation_angle=_rotation_degrees_to_radians(rotation_angle),
            device=device,
        )

        kx = kxa[bf_mask]  # (N_k,)
        ky = kya[bf_mask]  # (N_k,)

        # reshape for broadcasting
        kx = kx[:, None]  # (N_k, 1)
        ky = ky[:, None]
        qx = qx[None, :]  # (1, N_q)
        qy = qy[None, :]

        # ---------------------------------------------------------
        # k, k+q, k−q coordinates
        # ---------------------------------------------------------
        k0, phi0 = polar_coordinates(kx, ky)
        kp, phip = polar_coordinates(kx + qx, ky + qy)
        km, phim = polar_coordinates(kx - qx, ky - qy)

        # ---------------------------------------------------------
        # Aperture + chi (prior aberrations)
        # ---------------------------------------------------------
        ap0 = aperture(
            k0 * wavelength, phi0, self.semiangle_cutoff, self.angular_sampling, soft_edges=True
        )
        app = aperture(
            kp * wavelength, phip, self.semiangle_cutoff, self.angular_sampling, soft_edges=True
        )
        apm = aperture(
            km * wavelength, phim, self.semiangle_cutoff, self.angular_sampling, soft_edges=True
        )

        chi0 = aberration_surface(k0 * wavelength, phi0, wavelength, aberration_coefs)
        chip = aberration_surface(kp * wavelength, phip, wavelength, aberration_coefs)
        chim = aberration_surface(km * wavelength, phim, wavelength, aberration_coefs)

        # ---------------------------------------------------------
        # Overlap-only masks
        # ---------------------------------------------------------
        def ap_to_mask(ap, eps=1e-6):
            return ap > eps

        m0 = ap_to_mask(ap0)
        mp = ap_to_mask(app)
        mm = ap_to_mask(apm)

        plus_mask = m0 & mp & ~mm
        minus_mask = m0 & mm & ~mp

        # ---------------------------------------------------------
        # Gamma and phase deconvolution
        # ---------------------------------------------------------
        psi0 = ap0 * torch.exp(-1j * chi0)
        psip = app * torch.exp(-1j * chip)
        psim = apm * torch.exp(-1j * chim)

        gamma = psim * psi0.conj() - psip.conj() * psi0

        valid = gamma.abs() > 1e-6
        plus_mask &= valid
        minus_mask &= valid

        gamma_angle = torch.angle(gamma)
        measured = self._vbf_fourier[:, qi, qj]  # (N_k, N_q)
        deconv = measured * torch.exp(-1j * gamma_angle)

        # ---------------------------------------------------------
        # Pre-compute all basis functions (vectorized)
        # ---------------------------------------------------------
        chi_0_basis = aberration_surface_cartesian_basis(
            k0[:, 0] * wavelength,
            phi0[:, 0],
            wavelength,
            cartesian_basis,
        )  # (N_k, P)

        # For k+q and k-q (depends on q)
        chi_p_basis = aberration_surface_cartesian_basis(
            kp.flatten() * wavelength,
            phip.flatten(),
            wavelength,
            cartesian_basis,
        ).reshape(kp.shape[0], kp.shape[1], P)  # (N_k, N_q, P)

        chi_m_basis = aberration_surface_cartesian_basis(
            km.flatten() * wavelength,
            phim.flatten(),
            wavelength,
            cartesian_basis,
        ).reshape(km.shape[0], km.shape[1], P)  # (N_k, N_q, P)

        # Pre-allocate unwrapped phases
        phi_plus_all = torch.zeros_like(deconv, dtype=torch.float32)
        phi_minus_all = torch.zeros_like(deconv, dtype=torch.float32)

        for j in range(N_q):
            phi_plus_all[:, j] = unwrap_bf_overlap_phase_torch(
                complex_data_bf=deconv[:, j],
                mask_bf=plus_mask[:, j],
                bf_mask=bf_mask,
                method=unwrap_method,
                two_pass=two_pass_unwrapping,
            )

            phi_minus_all[:, j] = unwrap_bf_overlap_phase_torch(
                complex_data_bf=deconv[:, j],
                mask_bf=minus_mask[:, j],
                bf_mask=bf_mask,
                method=unwrap_method,
                two_pass=two_pass_unwrapping,
            )

        mask_plus_flat = plus_mask.reshape(-1)
        mask_minus_flat = minus_mask.reshape(-1)

        # dchi_plus_flat: (N_k*N_q, P)
        dchi_plus_flat = (chi_p_basis - chi_0_basis[:, None, :]).reshape(-1, P)
        dchi_minus_flat = (chi_0_basis[:, None, :] - chi_m_basis).reshape(-1, P)

        # rhs
        phi_plus_flat = phi_plus_all.reshape(-1)
        phi_minus_flat = phi_minus_all.reshape(-1)

        A_plus = torch.cat(
            [
                dchi_plus_flat[mask_plus_flat],
                torch.eye(N_q, device=device).repeat_interleave(plus_mask.sum(0), dim=0),
            ],
            dim=1,
        )

        b_plus = phi_plus_flat[mask_plus_flat]

        A_minus = torch.cat(
            [
                dchi_minus_flat[mask_minus_flat],
                torch.eye(N_q, device=device).repeat_interleave(minus_mask.sum(0), dim=0),
            ],
            dim=1,
        )

        b_minus = phi_minus_flat[mask_minus_flat]

        A = torch.cat([A_plus, A_minus], dim=0)
        b = torch.cat([b_plus, b_minus], dim=0)

        # ---------------------------------------------------------
        # Solve LS
        # ---------------------------------------------------------
        # Fall back to CPU (to support MPS), as fit_linear_plane does for eigh
        sol = torch.linalg.lstsq(A.cpu(), b.cpu()).solution.to(A.device)

        delta_cartesian = {name: sol[i] for i, name in enumerate(cartesian_basis)}

        return merge_aberration_coefficients(aberration_coefs, delta_cartesian)

    def fit_hyperparameters_least_squares(
        self,
        aberration_coefs: dict[str, int | float] | None = None,
        rotation_angle: float | None = None,
        cartesian_basis: str | list[str] | list[list[str]] = "low_order",
        num_q_modes: int = 6,
        q_signal_weight: float = 0.5,
        fit_method: Literal["global", "recursive", "sequential"] = "recursive",
        unwrap_method: Literal["reliability-sorting", "poisson"] = "reliability-sorting",
        two_pass_unwrapping: bool = False,
        verbose=None,
        use_initial_state=False,
        bf_mask=None,
        **reconstruct_kwargs,
    ):
        """
        Fit aberration coefficients using least squares.

        Args:
            aberration_coefs: Initial aberration coefficients to deconvolve
            rotation_angle: Rotation angle for basis functions, in degrees
            cartesian_basis: Aberration basis to fit. Can be:
                - str: preset name like "low_order"
                - list[str]: explicit list like ["C10", "C12_a", "C12_b", ...]
                - list[list[str]]: custom grouping for sequential fitting
            num_q_modes: Number of spatial frequency modes to use
            q_signal_weight: Balance between signal (1.0) and proximity to α_optimal (0.0)
            fit_method: How to iterate through basis functions:
                - "global": fit all coefficients simultaneously
                - "recursive": fit by radial order, accumulating previous orders
                - "sequential": fit by radial order, only current order each pass
            unwrap_method: Phase unwrapping algorithm
            two_pass_unwrapping: Whether to use two-pass unwrapping

        Returns:
            Updated aberration coefficients dictionary
        """

        state = self.hyperparameter_state

        if verbose is None:
            verbose = self.verbose

        if use_initial_state:
            aberration_coefs = state.initial_aberrations
            rotation_angle = state.initial_rotation_angle
        else:
            aberration_coefs = state.current_aberrations(aberration_coefs)
            rotation_angle = state.current_rotation_angle(rotation_angle)

        if bf_mask is None:
            bf_mask = self.bf_mask
        bf = self._return_bf_context(bf_mask)
        bf_mask = bf.bf_mask

        # Parse cartesian_basis into list of coefficient groups
        if isinstance(cartesian_basis, str):
            cartesian_basis = ABERRATION_PRESETS[cartesian_basis]

        if len(cartesian_basis) > 0 and isinstance(cartesian_basis[0], list):
            basis_groups = cartesian_basis
        else:
            basis_groups = group_basis_by_method(
                cartesian_basis,  # ty:ignore[invalid-argument-type]
                fit_method,
            )

        state.clear_optimized()
        current_coefs = aberration_coefs.copy()
        # Convert all values to tensors
        current_coefs = {
            k: torch.as_tensor(v, device=self.device, dtype=torch.float32)
            for k, v in aberration_coefs.items()
        }

        # Iterate through basis groups
        pbar = tqdm(basis_groups, desc="Fitting aberrations", unit="order", disable=not verbose)
        for i, basis_group in enumerate(pbar):
            pbar.set_postfix_str(f"{basis_group}"[:50])

            current_coefs = self._fit_hyperparameters_least_squares_inner(
                bf_mask,
                aberration_coefs=current_coefs,
                rotation_angle=rotation_angle,
                cartesian_basis=basis_group,
                num_q_modes=num_q_modes,
                q_signal_weight=q_signal_weight,
                unwrap_method=unwrap_method,
                two_pass_unwrapping=two_pass_unwrapping,
            )
        pbar.close()

        optimized_coefs = {k: float(v) for k, v in current_coefs.items() if v != 0.0}
        state.optimized_aberrations = validate_aberration_coefficients(optimized_coefs)

        if verbose:
            print("Optimized state:\n\n", self.hyperparameter_state)

        self.reconstruct(verbose=False, **reconstruct_kwargs)

        return self

    def _reconstruct_all_permutations(self, verbose=None, **reconstruct_kwargs):
        """ """

        if verbose is None:
            verbose = self.verbose

        safe_kwargs = {
            k: v for k, v in reconstruct_kwargs.items() if k not in ["deconvolution_kernel"]
        }

        kernels = ["ssb", "obf", "mf", "prlx", "icom"]

        recons = [
            self.reconstruct(
                deconvolution_kernel=kernel,
                verbose=False,
                **safe_kwargs,
            ).obj
            for kernel in tqdm(kernels, disable=not verbose)
        ]

        return recons

    def _reconstruct_with_halfsets(self, verbose=None, **reconstruct_kwargs):
        """
        Compute two half-set reconstructions using alternating BF pixels (checkerboard pattern).

        Returns
        -------
        halfset_1 : torch.Tensor
            Reconstruction using first half of BF pixels
        halfset_2 : torch.Tensor
            Reconstruction using second half of BF pixels
        """
        if verbose is None:
            verbose = False

        bf1, bf2 = self._make_checkerboard_bf_masks(self.gpts, self.bf_mask)
        safe_kwargs = {k: v for k, v in reconstruct_kwargs.items() if k not in ["bf_mask"]}

        self.reconstruct(**safe_kwargs, bf_mask=bf1, verbose=verbose)
        halfset_1 = self.corrected_bf

        self.reconstruct(**safe_kwargs, bf_mask=bf2, verbose=verbose)
        halfset_2 = self.corrected_bf

        return [halfset_1, halfset_2]
