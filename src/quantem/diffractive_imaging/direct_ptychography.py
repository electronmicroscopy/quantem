import gc
import math
import warnings
from typing import TYPE_CHECKING, Literal, Sequence, Tuple

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
    aberration_surface_cartesian_gradients,
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
    EIGENVALUE_FLOOR,
    OVERLAP_FLOOR,
    WDD_PADDING,
    WDD_RIDGE,
    _crop_corner_centered_mask,
    _rotation_degrees_to_radians,
    align_vbf_stack_multiscale,
    build_vbf_stack_from_dataset3d,
    build_vbf_stack_from_dataset4d,
    fit_aberrations_from_shifts,
    preferred_float_dtype,
    regrid_vbf_stack,
    slice_depths,
    wdd_upsilon,
    woa_kernel_terms,
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

    Every kernel here -- SSB, OBF, iCoM and parallax -- is a multiplier on
    ``G(k, q)``, the virtual bright-field stack Fourier transformed over the scan. The
    object is recovered by summing those multipliers over bright-field pixels and
    transforming back once.

    ``obj_type="complex"`` adds a log-amplitude channel, and ``num_slices > 1`` reconstructs
    several depths: by default one plane per depth, and with ``slice_mode="joint"`` ``"obf"`` and
    ``"wiener"`` solve a small least-squares problem at every frequency, with a column per slice
    for the phase and, for a complex object, one for the log-amplitude. :meth:`residual_loss` scores any reconstruction by how much of the data its model
    leaves unexplained, and :meth:`fit_hyperparameters_gradient` minimizes it by autograd.

    Partial coherence is described either empirically, by a mixed-state :attr:`fourier_probe`,
    or by two parameters, :attr:`focal_spread` and :attr:`source_size`, which every constructor
    takes. ``"wiener"`` is the kernel that divides it back out.

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
        focal_spread: float = 0.0,
        source_size: float = 0.0,
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
        self.focal_spread = focal_spread
        self.source_size = source_size
        self.rng = rng
        # each bright-field pixel's scan mean before normalization, which WDD multiplies back
        means = vbf_dataset.metadata.get("bright_field_means")
        self._bright_field_means = (
            None if means is None else torch.as_tensor(means, dtype=torch.float32)
        )

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
        focal_spread: float = 0.0,
        source_size: float = 0.0,
        soft_edges: bool = True,
        crop_bf_mask: bool = True,
        bf_mask_padding_px: int = 1,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
    ):
        """ """

        device, _ = config.validate_device(device)
        return cls(
            vbf_dataset=vbf_dataset,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            wavelength=wavelength,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            semiangle_cutoff=semiangle_cutoff,
            fourier_probe=fourier_probe,
            focal_spread=focal_spread,
            source_size=source_size,
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
        focal_spread: float = 0.0,
        source_size: float = 0.0,
        rotation_angle: float | None = None,
        max_batch_size: int | None = None,
        fit_method: str = "plane",
        mode: str = "bilinear",
        force_measured_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        force_fitted_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        intensity_threshold: float | None = None,
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

        device, _ = config.validate_device(device)
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
            aperture=cls._model_aperture_for(
                energy, wavelength, semiangle_cutoff, fourier_probe, device
            ),
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
            focal_spread=focal_spread,
            source_size=source_size,
            soft_edges=soft_edges,
            crop_bf_mask=crop_bf_mask,
            bf_mask_padding_px=bf_mask_padding_px,
            rng=rng,
            device=device,
            verbose=verbose,
            _token=cls._token,
        )

    @classmethod
    def from_tilt_series(
        cls,
        datasets: Sequence[Dataset4d],
        tilt_angles,
        per_dataset_kwargs: Sequence[dict] | None = None,
        **kwargs,
    ):
        """A tilt series: one reconstruction per acquisition, solved jointly for its slices.

        Returns a
        :class:`~quantem.diffractive_imaging.direct_ptychography_tilt_series.DirectPtychographyTiltSeries`,
        whose ``reconstruct`` solves every acquisition's normal equations together and whose
        acquisitions are ordinary reconstructions that can be fitted on their own first.

        Parameters
        ----------
        datasets : sequence of Dataset4d
            The acquisitions, on one scan grid.
        tilt_angles : array-like
            ``(T, 2)`` tilts in mrad, in the scan frame: a layer ``z`` Angstrom downstream of
            the reference plane appears displaced by ``z * theta * 1e-3`` Angstrom.
        per_dataset_kwargs : sequence of dict, optional
            Keyword arguments for each dataset's :meth:`from_dataset4d` that override the
            shared ``kwargs``, for example its own ``aberration_coefs``.
        **kwargs
            Passed to :meth:`from_dataset4d` for every dataset.
        """
        from quantem.diffractive_imaging.direct_ptychography_tilt_series import (
            DirectPtychographyTiltSeries,
        )

        datasets = list(datasets)
        if per_dataset_kwargs is None:
            per_dataset_kwargs = [{}] * len(datasets)
        if len(per_dataset_kwargs) != len(datasets):
            raise ValueError(
                f"`per_dataset_kwargs` has {len(per_dataset_kwargs)} entries for "
                f"{len(datasets)} datasets."
            )
        reconstructions = [
            cls.from_dataset4d(dataset, **{**kwargs, **own})
            for dataset, own in zip(datasets, per_dataset_kwargs)
        ]
        return DirectPtychographyTiltSeries(reconstructions, tilt_angles)

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
        focal_spread: float = 0.0,
        source_size: float = 0.0,
        bf_mask: torch.Tensor | NDArray | None = None,
        scan_gpts: Tuple[int, int] | None = None,
        interpolation: Literal["nearest", "bilinear"] = "nearest",
        hole_fill: Literal["mean", "zero"] = "mean",
        max_batch_size: int | None = None,
        fit_method: str = "plane",
        mode: str = "bilinear",
        force_measured_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        force_fitted_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        intensity_threshold: float | None = None,
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
        accumulated weight. Once gridded, SSB, OBF and iCoM all run exactly
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
        intensity_threshold : float, optional
            By default the bright-field mask is every detector pixel the probe model reaches,
            from ``semiangle_cutoff`` or ``fourier_probe``, which holds the aperture's soft
            edge and no dark field whatever the specimen scatters (see
            :func:`~quantem.diffractive_imaging.direct_ptycho_utils.bright_field_mask`). A
            value instead keeps the pixels whose mean intensity exceeds that fraction of the
            mean pattern's peak, as before.
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
        device, _ = config.validate_device(device)
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
            aperture=cls._model_aperture_for(
                energy, wavelength, semiangle_cutoff, fourier_probe, device
            ),
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
            focal_spread=focal_spread,
            source_size=source_size,
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
        self._corrected_bf = None
        self._corrected_amplitude = None
        self._residual = None
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

    #: elements of one ``(batch, Ny, Nx, columns)`` buffer, which bounds the default batch size
    _ELEMENTS_PER_BATCH = 2**25

    def _reconstruction_context(
        self,
        bf_mask,
        override_aberration_coefs,
        upsampling_factor,
        override_rotation_angle,
        deconvolution_kernel,
        q_highpass,
        q_lowpass,
        butterworth_order,
        band_limit,
        parallax_flip_phase,
        verbose,
        use_initial_state,
    ) -> dict:
        """Everything a pass over the bright-field pixels needs except the data itself."""
        state = self.hyperparameter_state

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

        kernel = self._normalize_kernel_name(deconvolution_kernel)
        if kernel == "prlx":
            # iCoM is exempt: `k . q / |q|**2` never reads the probe
            self._require_analytic_probe("The prlx kernel", aberration_coefs)

        # upsampled q-space grid and the (rotated) k-space grid
        qxa, qya = self._return_upsampled_qgrid(upsampling_factor)
        q, theta = polar_coordinates(qxa, qya)
        kxa, kya = spatial_frequencies(
            self.gpts,
            self.sampling,
            rotation_angle=_rotation_degrees_to_radians(rotation_angle),
            device=self.device,
        )
        k, phi = polar_coordinates(kxa, kya)

        grad_k = sign_sin_chi_q = None
        if kernel == "prlx":
            dx, dy = aberration_surface_cartesian_gradients(
                k * self.wavelength,
                phi,
                aberration_coefs=aberration_coefs,
            )
            grad_k = torch.stack((dx[bf.bf_mask], dy[bf.bf_mask]), -1)
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

        cmplx_probe_k = self._return_probe_on_grid(k, phi, aberration_coefs)
        bf_weights = cmplx_probe_k[bf.bf_mask].abs().square().sum()

        q_lowpass = self._resolve_q_lowpass(q_lowpass, kernel, band_limit)
        butterworth_env = torch.ones_like(q)
        if q_lowpass:
            butterworth_env *= 1 / (1 + (q / q_lowpass) ** (2 * butterworth_order))
        if q_highpass:
            butterworth_env *= 1 - 1 / (1 + (q / q_highpass) ** (2 * butterworth_order))

        # the weak-object kernels solve the alias tiles of each scan frequency jointly, so their
        # normal equations, and with them the residual, live on the scan's own grid
        alias_factor = upsampling_factor if kernel in self._WOA_KERNELS else 1
        solve_qxa, solve_qya = (
            self._return_upsampled_qgrid(None) if alias_factor > 1 else (qxa, qya)
        )
        residual_mask = self._residual_mask(solve_qxa, solve_qya)
        peak_intensity = float(cmplx_probe_k.detach().abs().max()) ** 2

        return dict(
            aberration_coefs=aberration_coefs,
            rotation_angle=rotation_angle,
            upsampling_factor=upsampling_factor,
            bf=bf,
            kernel=kernel,
            qxa=qxa,
            qya=qya,
            kxa=kxa,
            kya=kya,
            grad_k=grad_k,
            sign_sin_chi_q=sign_sin_chi_q,
            cmplx_probe_k=cmplx_probe_k,
            bf_weights=bf_weights,
            butterworth_env=butterworth_env,
            residual_mask=residual_mask,
            peak_intensity=peak_intensity,
            alias_factor=alias_factor,
        )

    def _tiled_vbf_fourier(self, bf, batch_idx, upsampling_factor):
        """The bright-field spectra of one batch, tiled onto the upsampled q-grid."""
        vbf_fourier = self._vbf_fourier[bf.vbf_index_mapping[batch_idx]]
        return torch.cat(
            [torch.cat([vbf_fourier] * upsampling_factor, dim=-1)] * upsampling_factor,
            dim=-2,
        )

    def _accumulate_kernel_sums(
        self, ctx, depths, obj_type, max_batch_size, verbose, slice_phases=None
    ):
        """One pass over the bright-field pixels, summing what the kernel needs in Fourier space.

        Nothing per bright-field pixel is kept: every kernel is a multiplier on ``G(q, k)``
        summed over ``k``, so the sums are taken here and transformed back once.
        ``slice_phases`` (``(num_slices, Ny, Nx)``) displaces each slice laterally, as a tilt
        series needs; see :func:`~quantem.diffractive_imaging.direct_ptycho_utils.woa_kernel_terms`.
        """
        bf, kernel = ctx["bf"], ctx["kernel"]
        qxa, qya = ctx["qxa"], ctx["qya"]
        kxa, kya = ctx["kxa"], ctx["kya"]
        cmplx_probe_k = ctx["cmplx_probe_k"]
        upsampling_factor = ctx["upsampling_factor"]
        shape = qxa.shape
        complex_obj = obj_type == "complex"
        num_columns = len(depths) * (2 if complex_obj else 1)
        alias_factor = ctx["alias_factor"]
        solve_shape = tuple(n // alias_factor for n in shape)
        solve_columns = num_columns * alias_factor**2

        if max_batch_size is None:
            max_batch_size = max(
                1, min(bf.num_bf, self._ELEMENTS_PER_BATCH // (shape[0] * shape[1] * num_columns))
            )

        # the widest precision the device has: float64 sums, except on MPS
        real_dtype = preferred_float_dtype(self.device)
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

        sums = {"energy": torch.zeros(shape, device=self.device, dtype=real_dtype)}
        woa = kernel in self._WOA_KERNELS
        if woa:
            sums["gram"] = torch.zeros(
                (*solve_shape, solve_columns, solve_columns),
                device=self.device,
                dtype=complex_dtype,
            )
            sums["numerator"] = torch.zeros(
                (*solve_shape, solve_columns), device=self.device, dtype=complex_dtype
            )
            if solve_columns == 1:
                sums["residual_gram"] = torch.zeros_like(sums["gram"])
            sums["weighted_energy"] = torch.zeros(
                solve_shape, device=self.device, dtype=real_dtype
            )
            probe = self._return_probe(ctx["aberration_coefs"])
            peak_amplitude = float(cmplx_probe_k.detach().abs().max())
            aperture_floor = 1e-6 * peak_amplitude
        if kernel in ("ssb", "prlx", "icom"):
            sums["phase"] = torch.zeros(shape, device=self.device, dtype=complex_dtype)
        if complex_obj and kernel in ("ssb", "icom"):
            sums["amplitude"] = torch.zeros(shape, device=self.device, dtype=complex_dtype)
        if kernel == "icom":
            sums["power"] = torch.zeros(shape, device=self.device, dtype=real_dtype)
            q2 = qxa.square() + qya.square()
            qx_op = -1.0j * qxa / q2
            qy_op = -1.0j * qya / q2
            qx_op[0, 0] = 0.0
            qy_op[0, 0] = 0.0
        if kernel == "prlx":
            qvec = torch.stack((qxa, qya), 0)

        keys = list(sums)

        def batch_terms(batch_idx):
            """Every sum's contribution from one batch of bright-field pixels, in ``keys`` order."""
            vbf_fourier = self._tiled_vbf_fourier(bf, batch_idx, upsampling_factor)
            terms = {"energy": vbf_fourier.abs().square().sum(0)}

            ind_i = bf.bf_inds_i[batch_idx]
            ind_j = bf.bf_inds_j[batch_idx]
            kx = kxa[ind_i, ind_j].view(-1, 1, 1)
            ky = kya[ind_i, ind_j].view(-1, 1, 1)

            if woa:
                terms.update(
                    woa_kernel_terms(
                        kernel,
                        vbf_fourier,
                        probe,
                        cmplx_probe_k[ind_i, ind_j].view(-1, 1, 1),
                        kx,
                        ky,
                        qxa,
                        qya,
                        self.wavelength,
                        depths,
                        complex_obj,
                        aperture_floor,
                        peak_amplitude**2,
                        alias_factor,
                        slice_phases,
                    )
                )

            elif kernel == "prlx":
                grad_kq = torch.einsum("na,amp->nmp", ctx["grad_k"][batch_idx], qvec)
                terms["phase"] = (vbf_fourier * torch.exp(-1j * grad_kq)).sum(0)

            else:
                operator = kx * qx_op + ky * qy_op
                terms["phase"] = (vbf_fourier * operator).sum(0)
                terms["power"] = operator.abs().square().sum(0)
                if complex_obj:
                    terms["amplitude"] = vbf_fourier.sum(0)

            return tuple(terms[key] for key in keys)

        # recompute each batch in the backward pass rather than keeping it, so a gradient
        # costs one batch of memory instead of one (num_bf, Ny, Nx, columns) buffer
        differentiable = torch.is_grad_enabled() and any(
            isinstance(value, torch.Tensor) and value.requires_grad
            for value in (cmplx_probe_k, kxa, depths, ctx["grad_k"])
        )

        pbar = tqdm(range(bf.num_bf), disable=not verbose)
        batcher = SimpleBatcher(bf.num_bf, batch_size=max_batch_size, shuffle=False, rng=self.rng)
        for batch_idx in batcher:
            if differentiable:
                terms = torch.utils.checkpoint.checkpoint(
                    batch_terms, batch_idx, use_reentrant=False
                )
            else:
                terms = batch_terms(batch_idx)
            for key, term in zip(keys, terms):
                sums[key] = sums[key] + term
            pbar.update(len(batch_idx))
        pbar.close()
        return sums

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
        band_limit=True,
        parallax_flip_phase=True,
        verbose=None,
        use_initial_state=False,
        obj_type="pure_phase",
        num_slices=1,
        slice_thicknesses=None,
        differentiable=False,
        ridge=None,
        prior=None,
        noise_variance=None,
        slice_mode="sectioned",
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
            Factor by which to upsample the reconstruction. A scan samples the spectrum only
            modulo its own grid, so each scan frequency holds ``U^2`` frequencies of the finer
            grid, its alias tiles, which the bright-field pixels see differently. ``"obf"`` and
            ``"wiener"`` solve for all of them jointly at each scan frequency, as does the
            residual of every weak-object kernel; SSB, parallax, iCoM and WDD take each fine
            frequency on its own. From a scan whose Nyquist was half the band, ``U = 2``
            recovered the band above it as a full scan did: correlations of 0.90 and 0.999 for
            the OBF and Wiener phase and 0.89 and 0.98 for their log-amplitude, against 0.90,
            0.999, 0.89 and 0.99 from the full scan. Solving the tiles one at a time had left
            the log-amplitude at 0.81 and 0.78; WDD's stays at 0.68 against 0.92.
        override_rotation_angle : float, optional
            Rotation angle for coordinate system, in degrees
        max_batch_size : int, optional
            Number of bright-field pixels processed at once. Defaults to as many as keep one
            ``(batch, Ny, Nx, columns)`` buffer near 32 million elements.
        deconvolution_kernel : str, one of ['ssb', 'obf', 'wiener', 'wdd', 'prlx', 'icom']
            ``"obf"`` solves the per-frequency least-squares problem and divides each channel
            by its noise standard deviation, so the noise is white and the transfer follows the
            signal-to-noise ratio. Its values are not calibrated radians: on a weak
            white-noise object the median transfer over the band was 0.7-0.9 for OBF, 0.5-0.8
            for SSB, and depended on the aberrations for both. At ``upsampling_factor > 1``
            the output of every kernel but ``"wiener"`` and ``"wdd"`` carries an extra
            ``1 / U**2``.

            ``"wiener"`` solves the same normal equations and returns the posterior mean under a
            Gaussian prior whose spectrum is the data's own: per ring of ``|q|`` and channel,
            the power the least-squares estimate holds beyond the noise, kept only where that
            excess is significant (see
            :func:`~quantem.diffractive_imaging.direct_ptycho_utils.wiener_estimate`). It
            estimates the object rather than whitening the noise, so its values are radians and
            nepers: on the same white-noise object its transfer was 0.97-1.02 without noise
            and had a median of 0.996-0.999 at 1e4 electrons per Angstrom^2, for any aberration.
            It also divides out the partial coherence of :attr:`source_size` and
            :attr:`focal_spread` where the signal allows, which OBF cannot: on one simulated
            WSe2 layer with a 0.5 A source and 50 A of focal spread it reached a correlation of
            0.95 with the layer at 1e5 electrons per pattern where OBF reached 0.76 (0.99
            against 0.98 without partial coherence, where the difference is only that OBF does
            not aim at the object). It supports a complex object and several slices; on two
            twisted layers 7 A apart at 1e4 electrons per pattern it reached 0.90 on a crystal
            and 0.83 on random atoms, where ``prior="ard"`` reached 0.93 and 0.81. The noise
            variance comes from the residual unless ``noise_variance`` is given. The result is
            nonlinear in the data and carries no gradient.

            It reports the lowest frequencies at full strength, where the phase column is weak.
            Where the data hold a slow variation the weak-object model does not explain, it
            shows as a gradient across the field that OBF fades: on gold nanoparticles the two
            longest wavelengths were left 300-450 times the noise unexplained.

            ``"wdd"`` is Wigner distribution deconvolution over the bright field. It models the
            intensity itself, a convolution over the detector of the overlap function with the
            object's Wigner distribution, with no weak-object approximation, so it recovers a
            strong single-slice transmission ``T(r)``: :attr:`corrected_bf` is ``angle(T)``,
            wrapped, and :attr:`corrected_amplitude` is ``ln|T|``. On 78 A of simulated SrTiO3
            its phase correlated 0.77-0.84 with the projected potential where OBF's did
            0.34-0.49. Its deconvolution is regularized by ``ridge``, so its values are not
            calibrated either: for a weak object the transfer is the regularized filter's weight
            at zero offset, a median of 0.68-0.74 over the band at the default ridge and as low
            as 0.39 near its edge. The convolution needs every detector pixel the aperture
            reaches, which the default bright-field mask, the model aperture, holds; it warns
            when a mask built with ``intensity_threshold`` does not. It solves for one slice,
            and multiplies the normalized stack back by the bright-field means
            :meth:`from_dataset4d` recorded, or by ``|psi(k)|^2`` when there are none. :meth:`residual_loss` is the weak-object model's, as for SSB
            and OBF. A mixed-state :attr:`fourier_probe`, :attr:`focal_spread` and
            :attr:`source_size` all enter its overlap function, but its fixed ridge suppresses
            the frequencies they weaken rather than restoring them: on one WSe2 layer with a
            0.5 A source and 50 A of focal spread, its correlation with the layer went from 0.73
            to 0.65 at 1e4 electrons per pattern and stayed at 0.76 at 1e5 when they were
            modelled, where ``"wiener"`` went from 0.79 to 0.95 at 1e5.
        q_highpass : float, optional
            High-pass filter cutoff
        q_lowpass : float, optional
            Low-pass filter cutoff. For ``"prlx"`` and ``"icom"`` it is clamped to at most
            ``2 * semiangle_cutoff / wavelength``, past which ``G(k, q)`` holds no signal.
            ``"ssb"`` and ``"obf"`` impose that cutoff through ``gamma`` already.
        band_limit : bool
            Apply that clamp. Turn it off to inspect the raw kernel.
        ridge : float, optional
            The deconvolution's ridge, relative to the scale of what it inverts. Defaults to
            1e-5 for ``"obf"`` and ``"wiener"`` and to 1e-3 for ``"wdd"``.

            For ``"wdd"`` it is the Wiener constant, relative to the largest ``|Gamma^|^2``. On
            simulated data it was best near 1e-3 at 1e5 electrons per Angstrom^2 and near 1e-2
            at 1e4; much smaller values amplify the noise wherever the probe's ambiguity
            function is weak.

            For ``"obf"`` and ``"wiener"`` it is the ridge on the unit-diagonal Gram matrix when
            they solve for several columns: several slices, or phase and amplitude. Near zero, the
            default, is the joint least-squares solve, which separates the columns wherever
            they differ at all, at a noise cost of ``1 / (1 - rho^2)`` for columns with
            correlation ``rho``. Larger values shrink the combinations the data barely tell
            apart; as it grows every slice tends to its own plane's reconstruction, a sweep of
            independent planes with no separation and no noise cost. Raise it at low dose:
            for two slices 0.1 ``lambda / alpha^2`` apart, 0.01 kept a purity of 0.94 and
            lifted each slice's correlation with its layer from 0.78 to 0.85 at 1e4 electrons
            per pattern, while at 1e5 the default was best.
        prior : {None, "ard"}
            ``"ard"`` (``"obf"`` only) replaces the least-squares estimate with the
            posterior mean under automatic relevance determination: every coefficient, each
            slice and channel at each frequency, gets its own Gaussian prior variance, learned
            from the data, and those the data do not support shrink to zero (see
            :func:`~quantem.diffractive_imaging.direct_ptycho_utils.ard_estimate`). A
            frequency where only one slice carries signal is then fitted with that slice alone,
            at the noise of a single plane. On two twisted WSe2 layers 7 A apart at 1e4
            electrons per pattern it raised each slice's correlation with its layer from 0.77
            to 0.93 at a purity of 1.00, where the best ridge reached 0.85 at 0.94; on randomly
            placed atoms it matched the best ridge without tuning. It is a sparsity prior, so
            it helps most where the spectrum is sparse, as a crystal's is. The OBF gain is then
            applied to the posterior mean, so the images compare directly with least squares. :meth:`residual_loss`, and so every hyperparameter fit, still uses
            least squares.
            The result is nonlinear in the data and carries no gradient.

            On a single plane it is a data-driven Fourier filter and changes little: the phase
            of one simulated layer moved by less than 0.005 in correlation either way, while a
            weak log-amplitude gained most (0.76 to 0.89 on a WSe2 layer at 1e4). On gold
            nanoparticles a pure-phase reconstruction lost 74% of its power at the particles'
            scale, where the pure-phase column is weak and model error inflates the noise
            estimate; the complex one kept 99.6% of it. Hence it is off by default.
        noise_variance : float, optional
            The noise level ``prior="ard"`` and ``"wiener"`` need, per unit weight of the mean-normalized
            bright-field data. Defaults to an estimate from the residual at each frequency,
            stored in :attr:`noise_variance`, which was within 7% of the truth on simulated
            Poisson data. Model error raises the estimate, which makes the prior shrink more.
        parallax_flip_phase : bool
            Apply the ``sign(sin(chi))`` phase flip to ``"prlx"``.
        verbose : bool, optional
            If True, show progress bar
        use_initial_state : bool
            Reconstruct with the initial rather than the optimized hyperparameters.
        obj_type : {"pure_phase", "complex"}
            ``"complex"`` also recovers the weak-object log-amplitude ``ln|t|``, stored in
            :attr:`corrected_amplitude`, from the symmetric overlap ``Gamma+``. Both channels
            are zero-mean.

            - ``"obf"`` solves for the two channels jointly at each frequency, so
              neither leaks into the other. In the triple overlap of an aberrated probe the
              two overlap functions are nearly parallel, which costs some signal-to-noise
              ratio there but not separation.
            - ``"ssb"`` uses only the two double-overlap sidebands, where
              ``Gamma+ = +-Gamma-``: their phase-compensated sum is the amplitude and their
              difference the phase. This separates exactly for any aberration, but drops the
              triple overlap, so its phase differs slightly from the pure-phase SSB.
            - ``"icom"`` returns the zeroth moment, the plain bright-field sum, which carries
              no phase for any aberration but is not deconvolved.
            - ``"prlx"`` raises: tilt-corrected bright field sums the two sidebands and records
              only ``cos(chi) Psi_a + sin(chi) Psi_phi`` at each frequency.

            At low frequencies the two channels are told apart by the double overlaps at the
            aperture edge, a crescent about one detector pixel wide, so two things in the data
            matter more here than for the phase alone. Fitting the diffraction origin
            resamples every pattern bilinearly, which blurs that crescent; on a white-noise
            object this raised the amplitude-to-phase leakage from 0.3% with
            ``force_fitted_origin`` to 4.7% without. And a scan coarser than the band
            ``2 * semiangle_cutoff / wavelength`` aliases frequencies the per-frequency model
            does not include, which leaked 1-10% at a scan twice too coarse even with
            upsampling.
        num_slices : int
            Number of thin slices. A slice a distance ``z`` downstream of the plane the
            aberrations refer to sees the probe at ``C10 + z``. :attr:`corrected_bf` and
            :attr:`corrected_amplitude` are then ``(num_slices, Ny, Nx)``, and how the slices
            are found is set by ``slice_mode``.
        slice_thicknesses : float or sequence of float, optional
            The ``num_slices - 1`` gaps between consecutive slices in Angstrom, or one value
            for all of them. Defaults to those a hyperparameter fit stored, if any. The stack is centred on the plane ``C10`` refers to, so one slice
            sits exactly there. The reference plane matters most: it must be right to about a
            quarter of the spacing, while a spacing wrong by 0.5-1.5 times costs little. Fit
            ``C10`` with one slice and :meth:`residual_loss`, which places the mean plane of
            the specimen to a fraction of an Angstrom, then add slices about it. With several
            jointly solved slices the residual barely depends on the offset or the spacing, because slices
            far closer than ``lambda / alpha**2`` span nearly the same columns whatever their
            positions: an assumed spacing ``s`` times too small mostly rescales the difference
            between the slices by ``1 / s``.
        slice_mode : {"sectioned", "joint"}
            ``"sectioned"``, the default (also ``"depth-section"`` or ``"sweep"``), is depth
            sectioning: one plane per slice, slice ``n`` reconstructed with the probe at
            ``C10 + z_n``, with any kernel but ``"icom"``, which does not depend on the
            aberrations. Each plane is an ordinary single-plane reconstruction, so it costs no
            extra noise, but it sees every layer within the depth of field
            ``lambda / alpha**2``: on two layers 0.25 ``lambda / alpha**2`` apart, both sections
            correlated about equally with both layers. Each section's residual is stored in
            :attr:`section_residuals`, and :meth:`residual_loss` is ``None``, since the stack
            is not one model of the data.

            ``"joint"`` solves for all the slices together at each frequency, ``"obf"`` and
            ``"wiener"`` only. That separates layers far closer than the depth of field, at a
            noise cost set by how alike the slices' columns are, and gives the stack one
            residual, which a hyperparameter fit with several slices needs.
        differentiable : bool
            Keep the autograd graph, so that :attr:`corrected_bf`, :attr:`corrected_amplitude`
            and :meth:`residual_loss` (then a tensor) can be backpropagated to aberration
            coefficients, a rotation angle or slice thicknesses passed as tensors that require
            grad. Each batch of bright-field pixels is recomputed in the backward pass rather
            than stored. :meth:`fit_hyperparameters_gradient` drives this.

        Returns
        -------
        self
        """
        arguments = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        if verbose is None:
            verbose = self.verbose

        if self._normalize_slice_mode(slice_mode) == "sectioned" and int(num_slices) > 1:
            with torch.set_grad_enabled(bool(differentiable)):
                return self._reconstruct_sections(self.reconstruct, arguments)

        with torch.set_grad_enabled(bool(differentiable)):
            return self._reconstruct(
                bf_mask,
                override_aberration_coefs,
                upsampling_factor,
                override_rotation_angle,
                max_batch_size,
                deconvolution_kernel,
                q_highpass,
                q_lowpass,
                butterworth_order,
                band_limit,
                parallax_flip_phase,
                verbose,
                use_initial_state,
                obj_type,
                num_slices,
                slice_thicknesses,
                differentiable,
                ridge,
                prior,
                noise_variance,
            )

    def _reconstruct(
        self,
        bf_mask,
        override_aberration_coefs,
        upsampling_factor,
        override_rotation_angle,
        max_batch_size,
        deconvolution_kernel,
        q_highpass,
        q_lowpass,
        butterworth_order,
        band_limit,
        parallax_flip_phase,
        verbose,
        use_initial_state,
        obj_type,
        num_slices,
        slice_thicknesses,
        differentiable,
        ridge=None,
        prior=None,
        noise_variance=None,
    ):
        """The body of :meth:`reconstruct`, run with gradients on or off."""
        ctx = self._reconstruction_context(
            bf_mask,
            override_aberration_coefs,
            upsampling_factor,
            override_rotation_angle,
            deconvolution_kernel,
            q_highpass,
            q_lowpass,
            butterworth_order,
            band_limit,
            parallax_flip_phase,
            verbose,
            use_initial_state,
        )
        kernel = ctx["kernel"]
        self._validate_reconstruction_mode(kernel, obj_type, num_slices)
        self._validate_prior(kernel, prior, noise_variance)
        if ridge is None:
            ridge = WDD_RIDGE if kernel == "wdd" else EIGENVALUE_FLOOR
        if num_slices > 1:
            slice_thicknesses = self.hyperparameter_state.current_slice_thicknesses(
                slice_thicknesses
            )
        depths = slice_depths(num_slices, slice_thicknesses, device=self.device)

        sums = self._accumulate_kernel_sums(ctx, depths, obj_type, max_batch_size, verbose)
        env = ctx["butterworth_env"]
        if kernel == "wdd":
            self._noise_variance = None
            phase, amplitude = self._wdd_channels(ctx, obj_type, ridge, verbose)
        else:
            phase_q, amplitude_q = self._fourier_channels(
                ctx,
                sums,
                num_slices,
                obj_type,
                ridge,
                prior,
                noise_variance,
            )
            phase = torch.fft.ifft2(phase_q * env).real.to(torch.float32)
            amplitude = (
                None
                if amplitude_q is None
                else torch.fft.ifft2(amplitude_q * env).real.to(torch.float32)
            )
        if num_slices == 1:
            phase = phase[0]
            amplitude = None if amplitude is None else amplitude[0]

        self._corrected_bf = phase
        self._corrected_amplitude = amplitude
        self._obj_type = obj_type
        self._num_slices = int(num_slices)
        self._slice_depths = [float(z) for z in depths.detach().cpu()]
        self._slice_mode = "joint"
        self._section_residuals = None
        self._kernel = kernel
        residual = self._residual_from_sums(ctx, sums)
        self._residual = residual if differentiable else float(residual)

        # memory management
        del sums
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return self

    #: fraction of the probe intensity outside the bright-field mask above which WDD warns
    _WDD_CLIPPED_INTENSITY = 5e-3

    def _warn_if_mask_clips_aperture(self, cmplx_probe_k, bf_mask):
        """WDD models the convolution over every pixel the aperture reaches, so a mask that cuts
        its soft edge leaves the model without data it predicts.

        On a weak white-noise object, a mask at half the mean pattern's peak left 1.8% of the
        probe intensity outside and the weak-object transfer did not converge to one as the
        ridge vanished (1.23); at 0.005 of the peak it left none and converged (0.998). On 78 A
        of SrTiO3, a mask of exactly the aperture's pixels raised the correlation with the exit
        wave from 0.49-0.51 to 0.51-0.52 in focus and from 0.41 to 0.42-0.44 defocused.

        A lower threshold on the mean pattern is not the same thing on a strongly scattering
        specimen: at 0.05 of the peak the mask reached 1.4 alpha on that SrTiO3 and 1.2 alpha on
        gold nanoparticles, so it also admitted 350-380 dark-field pixels, which WDD then uses.
        """
        intensity = cmplx_probe_k.detach().abs().square()
        outside = float(intensity[~bf_mask].sum() / intensity.sum())
        if outside > self._WDD_CLIPPED_INTENSITY:
            warnings.warn(
                f"The bright-field mask leaves {outside:.1%} of the probe intensity outside it, "
                "and WDD's convolution model needs every pixel the aperture reaches. Rebuild "
                "with `intensity_threshold=None`, the default, whose mask is the probe "
                "model's aperture.",
                stacklevel=5,
            )

    def _wdd_channels(self, ctx, obj_type, ridge, verbose):
        """``(phase, log_amplitude)``, each ``(1, Ny, Nx)``, of the object WDD recovers.

        The object is ``T(r)`` itself rather than a linearized phase, so the phase is
        ``angle(T)``, wrapped to ``(-pi, pi]``, and the log-amplitude ``ln|T|``. Both are
        normalized as the other kernels' are: ``T`` is scaled to unit geometric-mean modulus and
        its mean rotated onto the real axis, which WDD cannot fix itself since it recovers
        ``T~(0) T~*(-Q)``.
        """
        bf = ctx["bf"]
        probe = self._return_probe(ctx["aberration_coefs"])
        real_dtype = preferred_float_dtype(self.device)
        padded = tuple(WDD_PADDING * n for n in self.gpts)
        kx, ky = spatial_frequencies(
            padded,
            tuple(s / WDD_PADDING for s in self.sampling),
            rotation_angle=_rotation_degrees_to_radians(ctx["rotation_angle"]),
            device=self.device,
        )

        def padded_indices(rows, cols):
            """Detector pixels' signed frequency indices, on the padded grid."""
            return tuple(
                torch.where(index >= (n + 1) // 2, index - n, index) % p
                for index, n, p in zip((rows, cols), self.gpts, padded)
            )

        rows, cols = padded_indices(bf.bf_inds_i, bf.bf_inds_j)
        # Gamma vanishes wherever psi(k) does, so it is evaluated on the aperture alone
        amplitude = ctx["cmplx_probe_k"].detach().abs()
        aperture = torch.nonzero(amplitude > OVERLAP_FLOOR * amplitude.max(), as_tuple=True)
        support = torch.zeros(padded, dtype=torch.bool, device=self.device)
        support[padded_indices(*aperture)] = True
        self._warn_if_mask_clips_aperture(ctx["cmplx_probe_k"], bf.bf_mask)
        vbf_fourier = self._vbf_fourier[bf.vbf_index_mapping]
        if self._bright_field_means is not None:
            intensity = self._bright_field_means.to(self.device)[bf.vbf_index_mapping]
        else:
            intensity = ctx["cmplx_probe_k"][bf.bf_inds_i, bf.bf_inds_j].abs().square()

        upsilon = wdd_upsilon(
            probe,
            kx.to(real_dtype),
            ky.to(real_dtype),
            support,
            rows,
            cols,
            vbf_fourier,
            intensity.detach(),
            float(self._dc_per_image.real),
            ctx["qxa"],
            ctx["qya"],
            ridge,
            verbose=verbose,
        )
        # Upsilon(Q, 0) = T~(0) T~*(-Q), so T~(Q) T~*(0) is its conjugate at -Q
        height, width = upsilon.shape
        flip_rows = (-torch.arange(height, device=upsilon.device)) % height
        flip_cols = (-torch.arange(width, device=upsilon.device)) % width
        obj = torch.fft.ifft2(upsilon[flip_rows][:, flip_cols].conj())
        log_modulus = obj.abs().clamp_min(1e-12 * float(obj.abs().max())).log()
        log_modulus = log_modulus - log_modulus.mean()
        phase = torch.angle(obj * torch.exp(-1j * torch.angle(obj.mean())))

        env = ctx["butterworth_env"]
        filtered = not bool((env == 1).all())

        def output(image):
            if filtered:
                image = torch.fft.ifft2(torch.fft.fft2(image) * env).real
            return image.to(torch.float32)[None]

        amplitude = output(log_modulus) if obj_type == "complex" else None
        return output(phase), amplitude

    def _return_parallax_stack(self, bf_mask=None, max_batch_size=None, **reconstruct_kwargs):
        """``(num_bf, Ny, Nx)`` parallax-corrected bright-field images, without the phase flip.

        The one place a per-pixel stack is still needed: the cross-correlation fit aligns the
        images to each other.
        """
        ctx = self._reconstruction_context(
            bf_mask,
            reconstruct_kwargs.get("override_aberration_coefs"),
            reconstruct_kwargs.get("upsampling_factor"),
            reconstruct_kwargs.get("override_rotation_angle"),
            "prlx",
            reconstruct_kwargs.get("q_highpass"),
            reconstruct_kwargs.get("q_lowpass"),
            reconstruct_kwargs.get("butterworth_order", 12),
            reconstruct_kwargs.get("band_limit", True),
            False,
            False,
            reconstruct_kwargs.get("use_initial_state", False),
        )
        bf = ctx["bf"]
        qvec = torch.stack((ctx["qxa"], ctx["qya"]), 0)
        stack = torch.empty(
            (bf.num_bf,) + ctx["qxa"].shape, device=self.device, dtype=torch.float32
        )
        batcher = SimpleBatcher(
            bf.num_bf, batch_size=max_batch_size or bf.num_bf, shuffle=False, rng=self.rng
        )
        for batch_idx in batcher:
            vbf_fourier = self._tiled_vbf_fourier(bf, batch_idx, ctx["upsampling_factor"])
            grad_kq = torch.einsum("na,amp->nmp", ctx["grad_k"][batch_idx], qvec)
            shifted = vbf_fourier * torch.exp(-1j * grad_kq) * ctx["butterworth_env"]
            stack[batch_idx] = torch.fft.ifft2(shifted).real / ctx["bf_weights"]
        return stack

    @property
    def corrected_bf(self):
        """The reconstructed phase: ``(Ny, Nx)``, or ``(num_slices, Ny, Nx)`` for several slices."""
        return getattr(self, "_corrected_bf", None)

    def fit_hyperparameters_gradient(
        self,
        parameters=("C10",),
        loss="residual",
        optimizer: Literal["lbfgs", "adam"] = "lbfgs",
        num_iterations: int | None = None,
        learning_rate: float | None = None,
        use_initial_state: bool = False,
        verbose=None,
        **reconstruct_kwargs,
    ):
        """
        Fit aberrations, the rotation angle and slice thicknesses by gradient descent.

        The reconstruction is differentiable in all three, so the loss is minimized with
        autograd rather than sampled as :meth:`optimize_hyperparameters` does. Each step
        reconstructs once with gradients on and recomputes every batch of bright-field pixels
        in the backward pass, which costs about three reconstructions in time and one batch in
        memory.

        Every parameter is moved in natural units, which keeps the optimizer well scaled: an
        aberration ``C_nm`` by the amount that changes ``chi`` by one radian at the aperture
        edge, an angle ``phi_nm`` by 0.1 rad, the rotation by one degree, and each slice
        thickness by a factor ``exp(1)`` (it is fitted on a log scale, so it stays positive).

        Parameters
        ----------
        parameters : sequence of str
            What to fit, each starting from the current state: aberration coefficients such
            as ``"C10"``, ``"C12"`` or ``"phi12"``, ``"rotation_angle"`` for the detector
            rotation, and ``"slice_thicknesses"`` for the gaps between slices when
            ``reconstruct_kwargs`` has ``num_slices > 1`` and ``slice_mode="joint"``. The gaps start from
            ``reconstruct_kwargs["slice_thicknesses"]`` or the state. When the spacing is far
            below ``lambda / alpha**2`` the residual barely depends on it, so expect a slow,
            weakly determined fit there, and fit ``C10`` with one slice first.
        loss : {"residual", "rms_gradient"} or callable
            Objective to minimize. A callable is passed the reconstruction after a
            differentiable :meth:`reconstruct` and must return a scalar tensor.
        optimizer : {"lbfgs", "adam"}
            L-BFGS with a strong-Wolfe line search, or Adam.
        num_iterations : int, optional
            Iterations of L-BFGS (default 20) or Adam steps (default 100).
        learning_rate : float, optional
            Step in the natural units above: 1.0 for L-BFGS, 0.1 for Adam.
        use_initial_state : bool
            Start from the initial rather than the current hyperparameters.
        verbose : bool, optional
            Print the fitted values.
        **reconstruct_kwargs
            Passed to every :meth:`reconstruct`, e.g. ``deconvolution_kernel``, ``obj_type``,
            ``num_slices``, ``bf_mask`` or ``upsampling_factor``.

        Returns
        -------
        self
            With the fitted values written to :attr:`hyperparameter_state` and a final
            reconstruction made with them. ``_gradient_fit_history`` holds the loss and
            parameters after every evaluation.
        """
        self._check_loss_for_slice_mode(loss, reconstruct_kwargs)
        if verbose is None:
            verbose = self.verbose
        state = self.hyperparameter_state
        reconstruct_kwargs = dict(reconstruct_kwargs)
        reconstruct_kwargs.pop("use_initial_state", None)
        reconstruct_kwargs.pop("differentiable", None)

        if use_initial_state:
            start = dict(state.initial_aberrations)
            start_rotation = state.initial_rotation_angle
        else:
            start = state.current_aberrations()
            start_rotation = state.current_rotation_angle()
        start_rotation = 0.0 if start_rotation is None else float(start_rotation)
        parameters = [parameters] if isinstance(parameters, str) else list(parameters)
        rotation_angle = "rotation_angle" in parameters
        slice_thicknesses = "slice_thicknesses" in parameters
        names = list(
            validate_aberration_coefficients(
                {name: 0.0 for name in parameters if name not in HyperparameterState._OTHER_KEYS}
            )
        )
        if names and self.fourier_probe is not None:
            self._require_analytic_probe("A gradient fit of the aberrations", start)

        num_slices = int(reconstruct_kwargs.get("num_slices", 1))
        start_thicknesses = state.current_slice_thicknesses(
            reconstruct_kwargs.pop("slice_thicknesses", None)
        )
        if slice_thicknesses:
            if num_slices < 2:
                raise ValueError("Fitting `slice_thicknesses` needs `num_slices > 1`.")
            if start_thicknesses is None:
                raise ValueError(
                    "Pass a starting `slice_thicknesses` to fit, or run a search that stores one."
                )

        # natural units: the change in each coefficient that moves chi by one radian at alpha
        alpha = (
            self.semiangle_cutoff * 1e-3
            if self.semiangle_cutoff is not None
            else self._return_band_limit() / 2 * self.wavelength
        )

        def unit(name):
            if name.startswith("phi"):
                return 0.1
            order = int(name[1])
            return self.wavelength * (order + 1) / (2 * math.pi * alpha ** (order + 1))

        offsets = {name: torch.zeros((), device=self.device, requires_grad=True) for name in names}
        trainable = list(offsets.values())
        rotation_offset = None
        if rotation_angle:
            rotation_offset = torch.zeros((), device=self.device, requires_grad=True)
            trainable.append(rotation_offset)
        log_thicknesses = None
        if slice_thicknesses:
            base = torch.as_tensor(start_thicknesses, dtype=torch.float32, device=self.device)
            base = base.reshape(-1).expand(num_slices - 1).clone()
            log_thicknesses = torch.zeros_like(base, requires_grad=True)
            trainable.append(log_thicknesses)
        if not trainable:
            raise ValueError(
                "Nothing to fit: name aberrations, 'rotation_angle' or 'slice_thicknesses'."
            )

        def current():
            coefs = dict(start)
            for name, offset in offsets.items():
                coefs[name] = float(start.get(name, 0.0)) + offset * unit(name)
            rotation = (
                start_rotation if rotation_offset is None else start_rotation + rotation_offset
            )
            thicknesses = (
                start_thicknesses if log_thicknesses is None else base * torch.exp(log_thicknesses)
            )
            return coefs, rotation, thicknesses

        def scalar(value):
            return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)

        def snapshot(coefs, rotation, thicknesses):
            values = {name: scalar(coefs[name]) for name in names}
            if rotation_offset is not None:
                values["rotation_angle"] = scalar(rotation)
            if log_thicknesses is not None:
                values["slice_thicknesses"] = [float(t) for t in thicknesses.detach().cpu()]
            return values

        history = []

        def evaluate():
            coefs, rotation, thicknesses = current()
            kwargs = dict(reconstruct_kwargs)
            if thicknesses is not None:
                kwargs["slice_thicknesses"] = thicknesses
            self.reconstruct(
                override_aberration_coefs=coefs,
                override_rotation_angle=rotation,
                differentiable=True,
                verbose=False,
                **kwargs,
            )
            if callable(loss):
                value = loss(self)
            elif loss in ("residual", "rms_gradient"):
                value = self.residual_loss() if loss == "residual" else self.rms_gradient_loss()
            else:
                raise ValueError(
                    f"`loss` must be 'residual', 'rms_gradient' or a callable, got {loss!r}"
                )
            if not isinstance(value, torch.Tensor) or not value.requires_grad:
                raise ValueError(
                    "The loss is not connected to the fitted parameters; a callable loss must "
                    "return a tensor computed from the reconstruction."
                )
            history.append({"loss": scalar(value), **snapshot(coefs, rotation, thicknesses)})
            return value

        if optimizer == "lbfgs":
            opt = torch.optim.LBFGS(
                trainable,
                lr=1.0 if learning_rate is None else learning_rate,
                max_iter=20 if num_iterations is None else num_iterations,
                line_search_fn="strong_wolfe",
            )

            def closure():
                opt.zero_grad()
                value = evaluate()
                value.backward()
                return value

            opt.step(closure)
        elif optimizer == "adam":
            opt = torch.optim.Adam(trainable, lr=0.1 if learning_rate is None else learning_rate)
            for _ in tqdm(
                range(100 if num_iterations is None else num_iterations), disable=not verbose
            ):
                opt.zero_grad()
                evaluate().backward()
                opt.step()
        else:
            raise ValueError(f"`optimizer` must be 'lbfgs' or 'adam', got {optimizer!r}")

        best = min(history, key=lambda entry: entry["loss"])
        fitted = {name: best[name] for name in names}
        state.clear_optimized()
        state.optimized_aberrations = {**start, **fitted}
        state.optimized_keys = set(names)
        if rotation_offset is not None:
            state.optimized_rotation_angle = best["rotation_angle"]
            state.optimized_keys.add("rotation_angle")
        elif not use_initial_state:
            state.optimized_rotation_angle = start_rotation
        if log_thicknesses is not None:
            state.optimized_slice_thicknesses = best["slice_thicknesses"]
            state.optimized_keys.add("slice_thicknesses")
        elif start_thicknesses is not None:
            state.optimized_slice_thicknesses = [
                float(t) for t in np.atleast_1d(np.asarray(start_thicknesses, dtype=float))
            ]
        self._gradient_fit_history = history

        if verbose:
            print("Optimized state:\n\n", self.hyperparameter_state)

        self.reconstruct(verbose=False, **reconstruct_kwargs)
        return self

    def _flat_aperture_mask(self, rotation_angle=None, use_initial_state=False):
        """The bright-field pixels where the aperture is at least half its peak intensity."""
        state = self.hyperparameter_state
        if use_initial_state:
            aberrations, rotation = state.initial_aberrations, state.initial_rotation_angle
        else:
            aberrations, rotation = state.current_aberrations(), state.current_rotation_angle()
        if rotation_angle is not None:
            rotation = rotation_angle
        kxa, kya = spatial_frequencies(
            self.gpts,
            self.sampling,
            rotation_angle=_rotation_degrees_to_radians(rotation),
            device=self.device,
        )
        k, phi = polar_coordinates(kxa, kya)
        intensity = self._return_probe_on_grid(k, phi, aberrations).detach().abs().square()
        return self.bf_mask & (intensity >= 0.5 * intensity.max())

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
        rotation_angle : float, optional
            Starting rotation in degrees; the stored one by default. The shifts cannot tell a
            solution from its twin, the detector turned by 180 degrees with ``C10`` and ``C12``
            negated, so the fit returns the twin nearer this rotation.
        aberration_coefs : dict
            Starting aberrations, overriding the stored ones. The bright-field images
            are first shifted back by this starting point, and the alignment measures what is
            left.
        bf_mask: torch.Tensor, optional
            Subset of the bright-field mask to align. Defaults to the pixels where the
            aperture is at least half its peak intensity: the alignment treats each image as a
            shifted copy of the object, which pixels on the aperture's soft edge are not, and
            with them included the defocus fit on the test fixture landed at -600 A instead of
            1625 A.
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
            bf_mask = self._flat_aperture_mask(rotation_angle, use_initial_state)
        bf = self._return_bf_context(bf_mask)
        bf_mask = bf.bf_mask
        inds_i, inds_j = bf.bf_inds_i, bf.bf_inds_j

        scan_sampling = torch.as_tensor(
            self.scan_sampling, device=self.device, dtype=torch.float32
        )

        # initial parallax-corrected stack, which the alignment refines
        safe_kwargs = {
            k: v
            for k, v in reconstruct_kwargs.items()
            if k
            not in [
                "deconvolution_kernel",
                "parallax_flip_phase",
                "obj_type",
                "num_slices",
                "slice_thicknesses",
                "verbose",
            ]
        }

        state = self.hyperparameter_state
        # the starting point, used both to pre-shift the stack and as the shifts already applied
        # to it: the alignment only adds to these, so the two must be the same aberrations. It
        # is read before the optimized state is cleared, so a stored fit or rotation is kept.
        if use_initial_state:
            aberration_coefs = {
                **state.initial_aberrations,
                **validate_aberration_coefficients(aberration_coefs),
            }
            if rotation_angle is None:
                rotation_angle = state.initial_rotation_angle or 0.0
        else:
            aberration_coefs = state.current_aberrations(aberration_coefs)
            rotation_angle = state.current_rotation_angle(rotation_angle)
        state.clear_optimized()

        vbf_stack = self._return_parallax_stack(
            bf_mask=bf_mask,
            override_rotation_angle=rotation_angle,
            override_aberration_coefs=aberration_coefs,
            use_initial_state=use_initial_state,
            **safe_kwargs,
        )

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
            reference_rotation=rotation_angle,
        )

        fitted_aberration_coefs = fit_results.copy()
        fitted_rotation_angle = fitted_aberration_coefs.pop("rotation_angle", None)

        state.optimized_aberrations = validate_aberration_coefficients(fitted_aberration_coefs)
        state.optimized_rotation_angle = fitted_rotation_angle
        state.optimized_keys = {"C10", "C12", "phi12", "rotation_angle"}

        if verbose:
            print("Optimized state:\n\n", self.hyperparameter_state)

        self.reconstruct(verbose=False, **reconstruct_kwargs)

        return self

    #: every kernel, in the order :meth:`_reconstruct_all_permutations` returns them
    _PERMUTATION_KERNELS = ("ssb", "obf", "wiener", "wdd", "prlx", "icom")

    def _reconstruct_all_permutations(self, verbose=None, **reconstruct_kwargs):
        """Every kernel's :attr:`obj` with the same settings, in :attr:`_PERMUTATION_KERNELS`
        order, leaving out those that cannot take them: ``"prlx"`` for a complex object, all
        but ``"obf"`` and ``"wiener"`` for several joint slices, ``"icom"`` for several
        sectioned ones (the default), and all but ``"obf"`` for a prior."""

        if verbose is None:
            verbose = self.verbose

        safe_kwargs = {
            k: v for k, v in reconstruct_kwargs.items() if k not in ["deconvolution_kernel"]
        }

        # only the kernels that support the requested object type and number of slices
        kernels = list(self._PERMUTATION_KERNELS)
        if safe_kwargs.get("obj_type", "pure_phase") == "complex":
            kernels.remove("prlx")
        if safe_kwargs.get("num_slices", 1) > 1:
            if (
                self._normalize_slice_mode(safe_kwargs.get("slice_mode", "sectioned"))
                == "sectioned"
            ):
                kernels = [k for k in kernels if k != "icom"]
            else:
                kernels = [k for k in kernels if k in self._MULTISLICE_KERNELS]
        if safe_kwargs.get("prior") is not None:
            kernels = [k for k in kernels if k == "obf"]
        elif safe_kwargs.get("noise_variance") is not None:
            kernels = [k for k in kernels if k == "wiener"]

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
