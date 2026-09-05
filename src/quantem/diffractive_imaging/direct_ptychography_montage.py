import gc
import math
import warnings
from typing import TYPE_CHECKING, Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.datastructures import Dataset2d, Dataset3d, Dataset4d
from quantem.core.utils.utils import to_numpy
from quantem.core.utils.validators import (
    validate_aberration_coefficients,
    validate_tensor,
)
from quantem.diffractive_imaging.complex_probe import (
    FourierProbe,
    aberration_surface,
    aberration_surface_cartesian_gradients,
    gamma_factor,
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
    _crop_corner_centered_mask,
    _rotation_degrees_to_radians,
    allocate_splat_buffers,
    build_vbf_stack_from_dataset3d,
    build_vbf_stack_from_dataset4d,
    convolve_stack_fourier,
    preferred_float_dtype,
    scatter_add_splat,
    splat_and_convolve,
    splat_stack,
)
from quantem.diffractive_imaging.direct_ptychography_base import DirectPtychographyBase

# target number of (BF pixel, scan position) points per splat batch
_DEFAULT_POINTS_PER_BATCH = 4_194_304


def _snap_to_integer(values: torch.Tensor, tolerance: float = 1e-4) -> torch.Tensor:
    """Round values that are integers to within ``tolerance``, leave the rest alone.

    Canvas bounds go through ``floor``/``ceil``, where a shift of exactly 4 arriving as
    4.0000001 costs a whole pixel. The k-grid is float32 and positions are float32 on MPS
    (which has no float64), so that noise is unavoidable -- and without snapping the same
    data yields a canvas one pixel larger on CPU and a different one again on MPS.
    """
    rounded = torch.round(values)
    return torch.where((values - rounded).abs() < tolerance, rounded, values)


class DirectPtychographyMontage(DirectPtychographyBase):
    """
    Direct ptychography that montages the scan onto a shared canvas.

    Every kernel of
    :class:`~quantem.diffractive_imaging.direct_ptychography.DirectPtychography` is a
    multiplier on the scan-space Fourier transform. Here that transform is not taken:
    each virtual bright-field image is deposited onto one canvas at its own probe position.
    The detector axis is handled the same way in both classes, summed over bright-field
    pixels each carrying its own kernel.

    Kernels
    -------
    ``prlx``
        A pure translation by ``grad_chi / (2 * pi)`` Angstrom, exact, one deposit per
        point. The shadow-montage (tilt-corrected bright field) construction [1]_.
    ``ssb``, ``obf``, ``mf``
        Convolutions rather than translations. Exact with ``convolution_mode="fft"`` (the
        default); truncated to a box stencil with ``"stencil"``.
    ``icom``
        Exact by FFT. Truncated it is riCOM [2]_, where the radius is a high-pass cutoff
        rather than an error. Being linear in ``k``, it collapses to two convolutions of
        the centre-of-mass shift regardless of the number of bright-field pixels.

    The parallax equivalence holds both ways: ``exp(-1j * grad_chi . q)`` is a translation,
    and Fourier-space tiling by ``U`` is real-space zero-insertion at every ``U``-th pixel.
    Working in real space instead gives:

    - no scan-space FFT, so the scan positions need not lie on a grid --
      see :meth:`from_dataset3d`;
    - the phase-flip and Butterworth filters, which do not depend on the bright-field index,
      collapse into a single post-hoc filter on the finished image;
    - a per-position defocus, which models a tilted sample -- see :attr:`defocus_gradient`
      and :meth:`fit_defocus_gradient`. A Fourier multiplier is global over the scan and
      cannot express this.

    Choosing between the two classes
    --------------------------------
    ==================================  ==========================================
    gridded scan                        ``DirectPtychography`` -- exact, and
                                        cheapest at one scan FFT
    ungridded scan                      either: ``from_dataset3d`` regrids onto a
                                        lattice first, this class never grids
    sub-pixel positions matter          here -- regridding discards them (0.997
                                        against 0.982 CTF correlation, quoted in
                                        ``DirectPtychography.from_dataset3d``)
    scan both masked and upsampled      here -- ``hole_fill`` cannot serve filled
                                        holes and deliberate gaps at once
    position-dependent defocus          here only
    large bright-field mask             here -- see below
    ==================================  ==========================================

    The last is a memory limit rather than a preference. ``DirectPtychography._preprocess``
    materializes the scan transform as ``(N_bf, Ry, Rx)`` complex64 -- 34 GB at 167k
    bright-field pixels on a 128x170 canvas -- where this class streams over detector pixels
    into one canvas for roughly 800 MB.

    Instantiate with :meth:`from_dataset4d`, :meth:`from_virtual_bfs` or
    :meth:`from_dataset3d`.

    References
    ----------
    .. [1] *Microscopy and Microanalysis* 32(1), ozaf126 (2026).
       https://doi.org/10.1093/mam/ozaf126
    .. [2] Yu et al., *Microscopy and Microanalysis* 28, 1526 (2022). riCOM.

    Related, though neither is the kernel implemented here: the first convolves a WDD
    kernel where this convolves SSB/OBF/MF kernels, and the second uses a segmented
    detector rather than a pixelated one.

    .. [3] Convolution WDD: *Ultramicroscopy* 285, 114411 (2026).
       https://doi.org/10.1016/j.ultramic.2026.114411
    .. [4] Segmented-detector OBF: *Ultramicroscopy* 220, 113133 (2021).
       https://doi.org/10.1016/j.ultramic.2020.113133
    """

    _token = object()

    def __init__(
        self,
        vbf_stack: torch.Tensor | NDArray,
        positions_px: torch.Tensor | NDArray,
        bf_mask_dataset: Dataset2d,
        energy: float | None,
        rotation_angle: float,
        aberration_coefs: dict,
        semiangle_cutoff: float,
        scan_sampling: Tuple[float, float],
        scan_units: Tuple[str, str],
        scan_gpts: Tuple[int, int],
        boundary: Literal["wrap", "pad"],
        gridded_scan: bool,
        subtract_frame_mean: bool,
        soft_edges: bool,
        crop_bf_mask: bool,
        bf_mask_padding_px: int,
        rng: np.random.Generator | int | None,
        device: str | int,
        verbose: int | bool,
        defocus_gradient: Tuple[float, float] | None = None,
        scan_origin: Tuple[float, float] | None = None,
        wavelength: float | None = None,
        fourier_probe: "FourierProbe | None" = None,
        _token: object | None = None,
    ):
        """ """
        if _token is not self._token:
            raise RuntimeError(
                "Use DirectPtychographyMontage.from_dataset4d(), .from_virtual_bfs() or "
                ".from_dataset3d() to instantiate this class."
            )

        self.device = device
        self.verbose = verbose
        self.vbf_stack = vbf_stack
        self.positions_px = positions_px
        self.bf_mask = bf_mask_dataset.array  # ty:ignore[invalid-assignment]
        if crop_bf_mask:
            self.bf_mask = _crop_corner_centered_mask(self.bf_mask, bf_mask_padding_px)

        if rotation_angle is None:
            raise ValueError(
                "`rotation_angle` is required, in degrees: it sets the detector rotation "
                "relative to the scan. Pass 0.0 if the two frames already agree."
            )

        self.wavelength = self._resolve_wavelength(energy, wavelength)
        self.scan_units = scan_units
        self.detector_units = bf_mask_dataset.units

        self.scan_gpts = tuple(int(n) for n in scan_gpts)
        self.scan_sampling = scan_sampling
        self.scan_origin = scan_origin
        self.reciprocal_sampling = bf_mask_dataset.sampling
        self.angular_sampling = tuple(d * 1e3 * self.wavelength for d in self.reciprocal_sampling)

        self.num_bf = int(self.vbf_stack.shape[0])
        self.num_positions = int(self.vbf_stack.shape[1])
        self.gpts = tuple(int(n) for n in self.bf_mask.shape[:2])
        self.sampling = tuple(1 / s / n for n, s in zip(self.reciprocal_sampling, self.gpts))

        self.fourier_probe = fourier_probe
        self.semiangle_cutoff = semiangle_cutoff
        self.soft_edges = soft_edges
        self.boundary = boundary
        #: whether the positions lie on a regular lattice; drives the sampling-density
        #: correction that `weight_normalize` applies
        self.gridded_scan = gridded_scan
        self.subtract_frame_mean = subtract_frame_mean
        self.defocus_gradient = defocus_gradient
        self.rng = rng

        if self.positions_px.shape[0] != self.num_positions:
            raise ValueError(
                f"`positions_px` has {self.positions_px.shape[0]} rows but `vbf_stack` has "
                f"{self.num_positions} scan positions."
            )

        self.hyperparameter_state = self._make_hyperparameter_state(
            aberration_coefs, rotation_angle
        )

        self._preprocess()

    @staticmethod
    def _make_hyperparameter_state(aberration_coefs, rotation_angle):
        from quantem.diffractive_imaging.direct_ptychography_base import HyperparameterState

        return HyperparameterState(
            initial_aberrations=aberration_coefs, initial_rotation_angle=rotation_angle
        )

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

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
        boundary: Literal["wrap", "pad"] = "wrap",
        defocus_gradient: Tuple[float, float] | None = None,
        subtract_frame_mean: bool = False,
        soft_edges: bool = True,
        crop_bf_mask: bool = True,
        bf_mask_padding_px: int = 1,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
    ):
        """
        Build from a gridded virtual bright-field stack.

        Accepts exactly the ``(N_bf, Rx, Ry)`` ``Dataset3d`` that
        :meth:`DirectPtychography.from_virtual_bfs` takes, so the same stack can be fed to
        both classes; the trailing scan axes are flattened internally.
        """
        scan_gpts = tuple(int(n) for n in vbf_dataset.shape[-2:])
        vbf_stack = np.asarray(vbf_dataset.array).reshape(vbf_dataset.shape[0], -1)

        return cls(
            vbf_stack=vbf_stack,
            positions_px=cls._raster_positions_px(scan_gpts),
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            wavelength=wavelength,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            semiangle_cutoff=semiangle_cutoff,
            scan_sampling=tuple(vbf_dataset.sampling[-2:]),
            scan_units=tuple(vbf_dataset.units[-2:]),
            scan_gpts=scan_gpts,
            fourier_probe=fourier_probe,
            boundary=boundary,
            gridded_scan=True,
            defocus_gradient=defocus_gradient,
            subtract_frame_mean=subtract_frame_mean,
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
        boundary: Literal["wrap", "pad"] = "wrap",
        defocus_gradient: Tuple[float, float] | None = None,
        subtract_frame_mean: bool = False,
        soft_edges: bool = True,
        crop_bf_mask: bool = True,
        bf_mask_padding_px: int = 1,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
        normalization_order: int = 0,
        edge_blend_pixels: int = 0,
    ):
        """
        Build from a raster-scanned 4D-STEM dataset.

        Runs the same origin-correction, bright-field masking and normalization pipeline as
        :meth:`DirectPtychography.from_dataset4d` (they share
        :func:`~quantem.diffractive_imaging.direct_ptycho_utils.build_vbf_stack_from_dataset4d`),
        then flattens the scan axes onto an integer position grid.
        """
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

        return cls.from_virtual_bfs(
            vbf_dataset=vbf_dataset,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            wavelength=wavelength,
            rotation_angle=rotation_angle,
            semiangle_cutoff=semiangle_cutoff,
            aberration_coefs=aberration_coefs,
            fourier_probe=fourier_probe,
            boundary=boundary,
            defocus_gradient=defocus_gradient,
            subtract_frame_mean=subtract_frame_mean,
            soft_edges=soft_edges,
            crop_bf_mask=crop_bf_mask,
            bf_mask_padding_px=bf_mask_padding_px,
            rng=rng,
            device=device,
            verbose=verbose,
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
        max_batch_size: int | None = None,
        fit_method: str = "plane",
        mode: str = "bilinear",
        force_measured_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        force_fitted_origin: Tuple[float, float] | torch.Tensor | NDArray | None = None,
        intensity_threshold: float = 0.5,
        boundary: Literal["wrap", "pad"] = "pad",
        defocus_gradient: Tuple[float, float] | None = None,
        subtract_frame_mean: bool = False,
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
            Canvas pixel size in Angstrom. ``"auto"`` uses the median nearest-neighbour
            position spacing and warns with the inferred value.

        Notes
        -----
        Positions are *not* rotated: the detector rotation already enters through the
        bright-field k-grid, and rotating the positions as well would double-count it.
        """
        (
            vbf_stack,
            positions_px,
            bf_mask_dataset,
            scan_gpts,
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

        return cls(
            vbf_stack=vbf_stack,
            positions_px=positions_px,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            wavelength=wavelength,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            semiangle_cutoff=semiangle_cutoff,
            scan_sampling=scan_sampling,
            scan_units=("A", "A"),
            scan_gpts=scan_gpts,
            scan_origin=scan_origin,
            fourier_probe=fourier_probe,
            boundary=boundary,
            gridded_scan=False,
            defocus_gradient=defocus_gradient,
            subtract_frame_mean=subtract_frame_mean,
            soft_edges=soft_edges,
            crop_bf_mask=crop_bf_mask,
            bf_mask_padding_px=bf_mask_padding_px,
            rng=rng,
            device=device,
            verbose=verbose,
            _token=cls._token,
        )

    @staticmethod
    def _raster_positions_px(scan_gpts: Tuple[int, int]) -> NDArray:
        """Integer ``(Rx*Ry, 2)`` raster positions in scan pixels, "ij" ordered."""
        ii, jj = np.meshgrid(np.arange(scan_gpts[0]), np.arange(scan_gpts[1]), indexing="ij")
        return np.stack((ii.ravel(), jj.ravel()), axis=-1).astype(np.float64)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def vbf_stack(self) -> torch.Tensor:
        """``(N_bf, N_pos)`` virtual bright-field stack, flattened over scan positions."""
        return self._vbf_stack

    @vbf_stack.setter
    def vbf_stack(self, value):
        stack = validate_tensor(value, "vbf_stack", dtype=torch.float).to(device=self.device)
        if stack.ndim != 2:
            raise ValueError(
                f"`vbf_stack` must have shape (N_bf, N_pos), got {tuple(stack.shape)}"
            )
        self._vbf_stack = stack

    @property
    def _float_dtype(self) -> torch.dtype:
        """Widest float this device supports. MPS has no float64, so everything positional
        -- coordinates, shifts, canvas origins -- has to follow the device rather than
        hardcode float64."""
        return preferred_float_dtype(self.device)

    @property
    def positions_px(self) -> torch.Tensor:
        """``(N_pos, 2)`` scan positions in canvas pixels at ``upsampling_factor=1``."""
        return self._positions_px

    @positions_px.setter
    def positions_px(self, value):
        positions = validate_tensor(value, "positions_px", dtype=self._float_dtype).to(
            device=self.device
        )
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                f"`positions_px` must have shape (N_pos, 2), got {tuple(positions.shape)}"
            )
        self._positions_px = positions

    @property
    def defocus_gradient(self) -> Tuple[float, float] | None:
        """``(d C10 / d row, d C10 / d col)`` in Angstrom per Angstrom, or ``None``.

        Models a tilted sample, whose defocus varies linearly across the field of view as
        ``C10(r) = C10_global + g . (r - r_centroid)``. The magnitude is the tangent of the
        sample tilt, so a 5 degree tilt is ``|g| = 0.087``.

        Measuring from the centroid of the scan positions makes ``mean(delta C10) = 0``
        exactly, so the gradient is orthogonal to the global ``C10``: a hyperparameter search
        over ``C10`` stays well posed with a gradient set.

        This only matters when the defocus swing across the field of view is comparable to
        the depth of field ``wavelength / semiangle**2`` -- about 1200 Angstrom at 4 mrad,
        where it is irrelevant, but only 22 Angstrom at 30 mrad, where it dominates.
        """
        return self._defocus_gradient

    @defocus_gradient.setter
    def defocus_gradient(self, value):
        if value is None:
            self._defocus_gradient = None
            return
        value = tuple(float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1))
        if len(value) != 2:
            raise ValueError(
                f"`defocus_gradient` must be a (row, col) pair or None, got {value!r}"
            )
        self._defocus_gradient = value

    @property
    def defocus_map_results(self) -> dict | None:
        """What the last :meth:`defocus_map` measured, or ``None`` if it has not run.

        :meth:`fit_defocus_gradient` leaves its map here, so the per-patch loss curves can be
        plotted without paying for a second pass -- and a fit should be looked at before it
        is trusted, since a patch whose minimum sits on an endpoint of ``c10_values`` is
        dropped from the plane silently apart from the reported count.
        """
        return getattr(self, "_defocus_map_results", None)

    @property
    def positions_centroid_px(self) -> torch.Tensor:
        """Centroid of the scan positions, in canvas pixels. Where ``delta C10`` vanishes."""
        return self._positions_px.mean(dim=0)

    @property
    def corrected_bf(self) -> torch.Tensor | None:
        """Reconstructed phase image, or ``None`` before :meth:`reconstruct`."""
        return self._corrected_bf

    @property
    def weights(self) -> torch.Tensor | None:
        """Accumulated splat weight per canvas pixel -- the montage's local support."""
        if self._sum_w is None:
            return None
        return self._sum_w.reshape(self._canvas_shape)

    @property
    def variance_map(self) -> torch.Tensor | None:
        """Per-pixel variance across bright-field images (see :meth:`variance_loss`)."""
        if self._sum_wv2 is None:
            return None
        _, var, _ = self._weighted_moments()
        return var.reshape(self._canvas_shape)

    @property
    def _obj_fov(self) -> tuple[float, float]:
        """Field of view of the canvas, in Angstrom.

        With ``boundary="pad"`` the canvas grows past the scan to cover the shifted
        positions, so it spans more than :attr:`fov`. Computed by :meth:`_return_canvas`
        alongside the canvas shape, so the two always agree.
        """
        if self._canvas_fov is None:
            return self.fov
        return self._canvas_fov

    @property
    def obj_origin(self) -> tuple[float, float]:
        """Position of object pixel ``(0, 0)``, in Angstrom, in the caller's coordinates.

        The canvas corner, which ``"pad"`` places below the scan origin to make room for the
        aberration shifts. With :attr:`_obj_sampling` this maps the reconstruction back onto
        the probe positions that were passed in, and hence onto any other reconstruction of
        the same region -- see :meth:`reconstruct`'s ``obj_origin`` and ``obj_fov``.
        """
        if self._canvas_origin_px is None:
            return self.scan_origin
        origin_px = to_numpy(self._canvas_origin_px)
        return tuple(
            float(o + p * s) for o, p, s in zip(self.scan_origin, origin_px, self._obj_sampling)
        )

    # ------------------------------------------------------------------
    # preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self):
        """
        Remove the scan mean of each bright-field image.

        This is the real-space equivalent of zeroing the DC bin of each image's scan-space
        Fourier transform, which is what ``DirectPtychography._preprocess`` does. It also
        centers the accumulated values on zero, which keeps the ``E[v^2] - E[v]^2`` variance
        accumulation well conditioned.
        """
        self._dc_per_image = self._vbf_stack.mean(dim=1)
        self._vbf_stack = self._vbf_stack - self._dc_per_image[:, None]

        if self.subtract_frame_mean:
            self._vbf_stack = self._vbf_stack - self._vbf_stack.mean(dim=0, keepdim=True)

        self._reset_reconstruction()
        return self

    def _reset_reconstruction(self):
        self._sum_w = None
        self._sum_wv = None
        self._sum_wv2 = None
        self._corrected_bf = None
        self._canvas_shape = None
        self._canvas_origin_px = None
        self._canvas_fov = None
        self._bf_weights = None
        self._kernel = "prlx"
        self._stencil_info = None

    # ------------------------------------------------------------------
    # reconstruction
    # ------------------------------------------------------------------

    def _return_k_grid(self, rotation_angle):
        """``(kxa, kya, k, phi)`` on the rotated detector k-grid."""
        kxa, kya = spatial_frequencies(
            self.gpts,
            self.sampling,
            rotation_angle=_rotation_degrees_to_radians(rotation_angle),
            device=self.device,
        )
        k, phi = polar_coordinates(kxa, kya)
        return kxa, kya, k, phi

    def _upsampled_sampling(self, upsampling_factor) -> torch.Tensor:
        return torch.as_tensor(
            [s / upsampling_factor for s in self.scan_sampling],
            device=self.device,
            dtype=self._float_dtype,
        )

    def _return_defocus_rate_px(self, rotation_angle, bf_mask, upsampling_factor):
        """``(num_bf, 2)`` change in lateral shift per Angstrom of defocus, in canvas pixels.

        ``chi`` is linear in every aberration magnitude, so the rate is exactly the shift of
        a unit ``C10`` and is *independent of all other aberrations* -- hence the absent
        ``aberration_coefs`` argument. Evaluating it through
        ``aberration_surface_cartesian_gradients`` rather than hand-coding the analytic
        ``wavelength * k`` keeps it tied to the same expression the Fourier class uses.
        """
        self._require_analytic_probe("A defocus gradient")
        _, _, k, phi = self._return_k_grid(rotation_angle)
        dx, dy = aberration_surface_cartesian_gradients(
            k * self.wavelength, phi, aberration_coefs={"C10": 1.0}
        )
        rate = torch.stack((dx[bf_mask], dy[bf_mask]), -1)
        return (
            rate.to(self._float_dtype)
            / (2 * math.pi)
            / self._upsampled_sampling(upsampling_factor)
        )

    def _return_delta_c10(self, defocus_gradient) -> torch.Tensor | None:
        """``(N_pos,)`` local defocus offset in Angstrom, or ``None`` for no gradient.

        The positions are taken in the unrotated scan frame: defocus varies with physical
        position on the specimen, whereas the detector rotation belongs to the k-grid in
        :meth:`_return_defocus_rate_px`. Rotating here as well would double-count it.
        """
        if defocus_gradient is None or (defocus_gradient[0] == 0.0 and defocus_gradient[1] == 0.0):
            return None
        self._require_analytic_probe("A defocus gradient")

        scan_sampling = torch.as_tensor(
            tuple(self.scan_sampling), device=self.device, dtype=self._float_dtype
        )
        offsets_ang = (self._positions_px - self.positions_centroid_px) * scan_sampling
        gradient = torch.as_tensor(defocus_gradient, device=self.device, dtype=self._float_dtype)
        return offsets_ang @ gradient

    def _return_shifts_px(self, rotation_angle, aberration_coefs, bf_mask, upsampling_factor):
        """``(num_bf, 2)`` parallax shifts in upsampled canvas pixels, plus the BF weight."""
        _, _, k, phi = self._return_k_grid(rotation_angle)

        dx, dy = aberration_surface_cartesian_gradients(
            k * self.wavelength,
            phi,
            aberration_coefs=aberration_coefs,
        )
        grad_k = torch.stack((dx[bf_mask], dy[bf_mask]), -1)

        upsampled_sampling = self._upsampled_sampling(upsampling_factor)
        shifts_px = grad_k.to(self._float_dtype) / (2 * math.pi) / upsampled_sampling

        # matches DirectPtychography.reconstruct: soft_edges is left at evaluate_probe's
        # default rather than taking self.soft_edges, so the two normalizations agree
        cmplx_probe_k = self._return_probe_on_grid(k, phi, aberration_coefs)
        bf_weights = cmplx_probe_k[bf_mask].abs().square().sum()

        return shifts_px, bf_weights

    @staticmethod
    def _return_shift_extrema(shifts_px, defocus_rate_px, delta_c10):
        """``(lo, hi)`` over every ``(bright-field pixel, scan position)`` shift.

        The total shift is ``shifts[m] + rate[m] * delta_c10[n]``, which is monotone in
        ``delta_c10``, so its extrema over ``n`` are attained at the extremes of
        ``delta_c10`` and there is no need to materialize the ``(num_bf, N_pos, 2)`` array.
        """
        if delta_c10 is None:
            return shifts_px.amin(0), shifts_px.amax(0)

        at_min = shifts_px + defocus_rate_px * delta_c10.min()
        at_max = shifts_px + defocus_rate_px * delta_c10.max()
        return (
            torch.minimum(at_min, at_max).amin(0),
            torch.maximum(at_min, at_max).amax(0),
        )

    def _return_kernel_fourier(
        self, batch_idx, bf, kernel, qxa, qya, kxa, kya, cmplx_probe_k, probe, norm
    ):
        """``(B, Ny, Nx)`` Fourier deconvolution kernel for a batch of bright-field pixels.

        Mirrors ``DirectPtychography._return_kernel_contributions`` term for term, minus the
        data, so the two classes cannot drift apart.
        """
        ind_i = bf.bf_inds_i[batch_idx]
        ind_j = bf.bf_inds_j[batch_idx]
        kx = kxa[ind_i, ind_j].view(-1, 1, 1)
        ky = kya[ind_i, ind_j].view(-1, 1, 1)

        if kernel == "icom":
            # `k . q / |q|**2`, which never reads the probe -- so it needs no overlap
            # function, and an empirical probe raises no sampling question here
            q_square = qxa.square() + qya.square()
            qx_op = -1.0j * qxa / q_square
            qy_op = -1.0j * qya / q_square
            qx_op[0, 0] = 0.0
            qy_op[0, 0] = 0.0
            return kx * qx_op.unsqueeze(0) + ky * qy_op.unsqueeze(0), None

        gamma = gamma_factor(
            (qxa.unsqueeze(0) - kx, qya.unsqueeze(0) - ky),
            (qxa.unsqueeze(0) + kx, qya.unsqueeze(0) + ky),
            cmplx_probe_k[ind_i, ind_j].view(-1, 1, 1),
            probe,
            normalize=False,
        )

        if kernel == "ssb":
            return -1.0j * gamma.conj() / gamma.abs().clip(1e-8), gamma
        return -1.0j * gamma.conj() / (1.0 if norm is None else norm), gamma

    @staticmethod
    def _resolve_convolution_mode(convolution_mode, kernel, stencil_radius):
        """Which convolution route to take for a non-parallax kernel.

        ``"auto"`` reads ``stencil_radius``: naming one is a request to truncate, so it takes
        the stencil; leaving it at ``"auto"`` takes the exact FFT. That is cheap to decide,
        where actually measuring which is faster would cost a full pass over the kernels.
        """
        if kernel == "prlx":
            return "splat"
        if convolution_mode not in ("auto", "fft", "stencil"):
            raise ValueError(
                f"`convolution_mode` must be 'auto', 'fft' or 'stencil', got {convolution_mode!r}"
            )
        if convolution_mode != "auto":
            return convolution_mode
        return "stencil" if stencil_radius != "auto" else "fft"

    def _probe_rotation_is_exact(self, rotation_angle) -> bool:
        """Whether a rotation leaves `k -/+ q` on the probe's own reciprocal lattice.

        `_return_k_grid` rotates the k-grid into the scan frame, which is what the analytic
        aperture wants -- it is *evaluated* there. An array can only be *read* on its lattice,
        and while `k` itself stays put, the offset `q` arrives rotated. Only rotations that
        map the lattice onto itself keep it there; everything else has to be interpolated,
        which `_resample_probe` handles by refining the grid first.
        """
        turns = float(rotation_angle) / 90.0
        square = abs(self.reciprocal_sampling[0] - self.reciprocal_sampling[1]) < 1e-9
        return abs(turns - round(turns)) < 1e-6 and (round(turns) % 2 == 0 or square)

    def _resample_probe(self, probe, q_step, oversample=1):
        """`probe` on the canvas's reciprocal grid, cached across reconstructions.

        Two transforms over a large detector are not free, and the canvas rarely changes
        between calls -- a defocus sweep or a hyperparameter search repeats the same one.

        ``oversample`` refines further, which is what makes an off-lattice sampling accurate:
        bilinear error falls as the square of the refinement. Measured on a speckled X-ray
        probe at a generic sub-pixel offset: 21% unrefined, 7.5% at 2x, 1.7% at 4x, 0.52% at
        8x, 0.064% at 16x -- against a memory cost that grows as the square.
        """
        key = (q_step, int(oversample))
        cached = getattr(self, "_resampled_probe", None)
        if cached is not None and cached[0] == key and cached[1] is probe:
            return cached[2]
        refined = probe.resampled_to(tuple(q / oversample for q in q_step))
        self._resampled_probe = (key, probe, refined)
        return refined

    def _return_com_shift(self, bf, rotation_angle, upsampling_factor):
        """``(2, N_pos)`` centre-of-mass shift, in upsampled canvas pixels.

        The k-weighted first moment of each diffraction pattern, summed over the detector.
        This is the collapse that makes riCOM cheap: the iCoM kernel is linear in ``k``, so

            sum_m FFT(V_m) K_m(q) = A(q) FFT(splat(com_x)) + B(q) FFT(splat(com_y))

        with ``A``, ``B`` the two components of ``-i q / |q|**2``. Summing over the detector
        *before* convolving turns ``num_bf`` transforms into two -- 167k into two on the
        X-ray data this was written against -- and is what the riCOM paper does.
        """
        kxa, kya, _, _ = self._return_k_grid(rotation_angle)
        k_vectors = torch.stack((kxa[bf.bf_mask], kya[bf.bf_mask]), dim=-1).to(self._float_dtype)
        values = self._vbf_stack[bf.vbf_index_mapping].to(self._float_dtype)
        return torch.einsum("mn,md->dn", values, k_vectors)

    def _return_icom_operators(self, qxa, qya):
        """``(2, Ny, Nx)`` complex ``-i q / |q|**2``, the two halves of the iCoM kernel."""
        q_square = qxa.square() + qya.square()
        operators = torch.stack((-1.0j * qxa / q_square, -1.0j * qya / q_square), dim=0)
        operators[:, 0, 0] = 0.0
        return operators

    def _truncate_icom_operators(self, operators, canvas_shape, stencil_radius, max_radius):
        """``((2, S*S) weights, info)`` -- the riCOM box stencil, from the iCoM operators.

        The real-space kernel is ``r / (2 * pi * |r|**2)``, centred at the origin, so the box
        is taken about the origin rather than about a parallax shift: iCoM carries no shift
        to divide out. The radius is riCOM's ``(n - 1) / 2``, and it is a high-pass cutoff by
        intent, not a truncation error -- so nothing is reported as one.
        """
        n_rows, n_cols = canvas_shape
        limit = max(1, min(n_rows, n_cols) // 2 - 1)
        radius = limit if stencil_radius == "auto" else min(int(stencil_radius), limit)
        radius = min(radius, max_radius) if stencil_radius == "auto" else radius

        kappa = torch.fft.fftshift(torch.fft.ifft2(operators), dim=(-2, -1))
        centre = (n_rows // 2, n_cols // 2)
        window = (
            slice(centre[0] - radius, centre[0] + radius + 1),
            slice(centre[1] - radius, centre[1] + radius + 1),
        )
        weights = kappa[:, window[0], window[1]].reshape(2, -1)

        inside = weights.abs().square().sum()
        total = kappa.abs().square().sum()
        return weights, {
            "stencil_radius": radius,
            "mean_error": float((1 - inside / total).clamp_min(0).sqrt()),
            "max_error": float((1 - inside / total).clamp_min(0).sqrt()),
        }

    def _return_kernel_context(
        self,
        bf,
        *,
        kernel,
        rotation_angle,
        aberration_coefs,
        canvas_shape,
        upsampling_factor,
        matched_filter_norm_epsilon,
        kernel_batch_size,
        probe_oversample=1,
    ):
        """``(kernel_args, norm, bf_weights)``, everything a Fourier kernel needs but the batch.

        Shared by the stencil and the FFT convolution paths, so the two cannot build
        different kernels from the same settings.
        """
        upsampled_sampling = tuple(s / upsampling_factor for s in self.scan_sampling)
        qxa, qya = spatial_frequencies(canvas_shape, upsampled_sampling, device=self.device)
        kxa, kya, k, phi = self._return_k_grid(rotation_angle)

        cmplx_probe_k = self._return_probe_on_grid(k, phi, aberration_coefs)
        bf_weights = cmplx_probe_k[bf.bf_mask].abs().square().sum()

        probe = self._return_probe(aberration_coefs)
        if probe.array is not None:
            # an empirical probe can only be read on its own reciprocal grid, so refine it
            # onto the canvas's -- exactly, by zero-padding in real space
            q_step = tuple(1 / (n * d) for n, d in zip(canvas_shape, upsampled_sampling))
            oversample = 1
            if not self._probe_rotation_is_exact(rotation_angle):
                # the rotation carries q off the lattice, so the probe has to be interpolated
                oversample = max(1, int(probe_oversample))
                probe = FourierProbe(
                    probe.wavelength,
                    array=probe.array,
                    reciprocal_sampling=probe.reciprocal_sampling,
                    interpolation="bilinear",
                )
                if oversample < 8:
                    warnings.warn(
                        f"rotation_angle={rotation_angle} takes q off the probe's reciprocal "
                        f"lattice, so psi is interpolated. At probe_oversample={oversample} "
                        "that costs roughly "
                        f"{ {1: '20%', 2: '7%', 4: '2%'}.get(oversample, '<1%') } rms on a "
                        "speckled probe; raise it (memory grows as its square) or use a "
                        "rotation that maps the lattice onto itself.",
                        stacklevel=3,
                    )
            probe = self._resample_probe(probe, q_step, oversample)

        kernel_args = (bf, kernel, qxa, qya, kxa, kya, cmplx_probe_k, probe)

        # obf and mf normalize by a power spectrum summed over every bright-field pixel, so
        # they need a pass over all of them before any kernel is final
        norm = None
        if kernel in ("obf", "mf"):  # icom needs no normalization pass
            power = torch.zeros(canvas_shape, device=self.device)
            batcher = SimpleBatcher(
                bf.num_bf, batch_size=kernel_batch_size, shuffle=False, rng=self.rng
            )
            for batch_idx in batcher:
                _, gamma = self._return_kernel_fourier(batch_idx, *kernel_args, None)
                power += gamma.abs().square().sum(0)
            power /= bf_weights
            if kernel == "obf":
                norm = power.sqrt().clamp_min(1e-8)
            else:
                norm = (power + matched_filter_norm_epsilon * power.max()).clamp_min(1e-8)

        return kernel_args, norm, bf_weights

    def _return_kernel_stencil(
        self,
        bf,
        *,
        kernel,
        rotation_angle,
        aberration_coefs,
        canvas_shape,
        upsampling_factor,
        shift_centers,
        stencil_radius,
        truncation_tolerance,
        max_stencil_radius,
        matched_filter_norm_epsilon,
        kernel_batch_size,
        verbose,
        probe_oversample=1,
    ):
        """``(stencil_offsets, stencil_weights, bf_weights, info)`` for a convolution kernel.

        Builds each bright-field pixel's Fourier kernel on the canvas grid, transforms it to
        real space and truncates it to a box stencil. Costs ``num_bf`` canvas FFTs once per
        reconstruction, negligible beside the scatter that follows.

        ``shift_centers`` is the integer parallax shift, which is divided out of the kernel
        by a phase ramp and added back to the deposit coordinates. Without it the stencil
        would have to span the shift itself -- tens of pixels at realistic defocus -- rather
        than just the residual chirp and aperture ringing.
        """
        kernel_args, norm, bf_weights = self._return_kernel_context(
            bf,
            kernel=kernel,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            canvas_shape=canvas_shape,
            upsampling_factor=upsampling_factor,
            matched_filter_norm_epsilon=matched_filter_norm_epsilon,
            kernel_batch_size=kernel_batch_size,
            probe_oversample=probe_oversample,
        )

        n_rows, n_cols = canvas_shape
        radius_limit = max(1, min(n_rows, n_cols) // 2 - 1)

        def batches():
            return SimpleBatcher(
                bf.num_bf, batch_size=kernel_batch_size, shuffle=False, rng=self.rng
            )

        # exp(2i.pi.c.m/N) rolls kappa by -c, undoing the parallax shift
        freq_row = torch.fft.fftfreq(n_rows, device=self.device).view(-1, 1)
        freq_col = torch.fft.fftfreq(n_cols, device=self.device).view(1, -1)

        def recentered_kappa(batch_idx):
            kernel_fourier, _ = self._return_kernel_fourier(batch_idx, *kernel_args, norm)
            centers = shift_centers[batch_idx].to(torch.float32)
            ramp = torch.exp(
                2j
                * math.pi
                * (centers[:, 0, None, None] * freq_row + centers[:, 1, None, None] * freq_col)
            )
            return torch.fft.fftshift(torch.fft.ifft2(kernel_fourier * ramp), dim=(-2, -1))

        center = (n_rows // 2, n_cols // 2)
        rows = torch.arange(n_rows, device=self.device).view(-1, 1) - center[0]
        cols = torch.arange(n_cols, device=self.device).view(1, -1) - center[1]
        chebyshev = torch.maximum(rows.abs(), cols.abs())

        # first pass: cumulative energy inside each candidate box, per bright-field pixel.
        # only these scalars are kept, so the full-canvas kernels never all exist at once
        candidates = list(range(1, min(max_stencil_radius, radius_limit) + 1))
        if stencil_radius != "auto":
            candidates = [min(int(stencil_radius), radius_limit)]

        inside = torch.zeros((bf.num_bf, len(candidates)), device=self.device)
        total = torch.zeros(bf.num_bf, device=self.device)
        for batch_idx in tqdm(list(batches()), disable=not verbose, desc=f"{kernel} kernel"):
            energy = recentered_kappa(batch_idx).abs().square()
            total[batch_idx] = energy.sum((-2, -1))
            for column, candidate in enumerate(candidates):
                inside[batch_idx, column] = (energy * (chebyshev <= candidate)).sum((-2, -1))

        errors = 1 - inside / total.clamp_min(torch.finfo(total.dtype).tiny)[:, None]
        errors = errors.clamp_min(0).sqrt()

        mean_errors = errors.mean(0)
        if stencil_radius == "auto":
            meets = (mean_errors <= truncation_tolerance).nonzero()
            column = int(meets[0]) if meets.numel() else len(candidates) - 1
        else:
            column = 0
        radius = candidates[column]

        info = {
            "stencil_radius": radius,
            "mean_error": float(mean_errors[column]),
            "max_error": float(errors[:, column].max()),
        }

        if info["mean_error"] > truncation_tolerance and kernel != "icom":
            warnings.warn(
                f"A stencil radius of {radius} px leaves an estimated "
                f"{info['mean_error']:.0%} truncation error (worst bright-field pixel "
                f"{info['max_error']:.0%}). The {kernel.upper()} kernel is not compact in "
                "real space -- dividing by |gamma| leaves a phase on a hard-edged support, "
                "whose transform has tails decaying as r**-1.5, so the error falls only like "
                "1/radius and being in focus does not help. DirectPtychography computes the "
                "same kernel exactly by FFT; prefer it unless the scan is ungridded. This "
                "estimate assumes a white object spectrum, so it is pessimistic for a real "
                "one.",
                stacklevel=3,
            )
        # iCoM is excluded above: truncating it is riCOM, not an approximation of iCoM.

        # second pass: crop to the chosen box
        window = (
            slice(center[0] - radius, center[0] + radius + 1),
            slice(center[1] - radius, center[1] + radius + 1),
        )
        side = 2 * radius + 1
        stencil_weights = torch.empty(
            (bf.num_bf, side * side), device=self.device, dtype=torch.complex64
        )
        for batch_idx in batches():
            stencil_weights[batch_idx] = recentered_kappa(batch_idx)[
                :, window[0], window[1]
            ].reshape(len(batch_idx), -1)

        span = torch.arange(-radius, radius + 1, device=self.device)
        offsets = torch.stack(torch.meshgrid(span, span, indexing="ij"), dim=-1).reshape(-1, 2)

        return offsets, stencil_weights, bf_weights, info

    def _return_canvas(
        self,
        shifts_px,
        upsampling_factor,
        boundary,
        pad_px,
        defocus_rate_px=None,
        delta_c10=None,
        obj_origin=None,
        obj_fov=None,
    ):
        """``(canvas_shape, canvas_origin_px, canvas_fov)`` for the requested boundary.

        The field of view is returned alongside the shape, rather than recomputed later,
        so the two cannot disagree about the upsampling factor.
        """
        positions_up = self._positions_px * upsampling_factor
        shift_lo, shift_hi = self._return_shift_extrema(shifts_px, defocus_rate_px, delta_c10)

        def with_fov(canvas_shape, origin):
            canvas_fov = tuple(
                n * s / upsampling_factor for n, s in zip(canvas_shape, self.scan_sampling)
            )
            return canvas_shape, origin, canvas_fov

        if boundary == "wrap" and obj_origin is None and obj_fov is None:
            # spans exactly the scan field of view, at any upsampling factor
            canvas_shape = tuple(int(n) * upsampling_factor for n in self.scan_gpts)
            origin = torch.zeros(2, device=self.device, dtype=self._float_dtype)
            return with_fov(canvas_shape, origin)

        if boundary not in ("wrap", "pad"):
            raise ValueError(f"`boundary` must be 'wrap' or 'pad', got {boundary!r}")

        if pad_px is not None and obj_fov is not None:
            raise ValueError("`pad_px` and `obj_fov` both size the canvas; pass one or the other.")

        if pad_px is None:
            lo = torch.floor(_snap_to_integer(positions_up.amin(0) + shift_lo))
            hi = torch.ceil(_snap_to_integer(positions_up.amax(0) + shift_hi))
        else:
            lo = torch.floor(_snap_to_integer(positions_up.amin(0))) - pad_px
            hi = torch.ceil(_snap_to_integer(positions_up.amax(0))) + pad_px

        # +2 leaves room for the upper bilinear corner at the far edge
        canvas_shape = tuple(int(v) + 2 for v in (hi - lo))

        if obj_origin is not None:
            # Angstrom in the caller's frame -> upsampled canvas pixels, deliberately *not*
            # rounded: each frame anchors its grid at its own bounding box, so snapping
            # would leave the same window a fraction of a pixel apart per frame -- the very
            # misregistration this removes. Fractional coordinates are fine for the splat.
            offset = np.asarray(obj_origin, dtype=np.float64) - np.asarray(self.scan_origin)
            lo_np = offset / np.asarray(self.scan_sampling) * upsampling_factor
            lo = torch.as_tensor(lo_np, device=self.device, dtype=self._float_dtype)

        if obj_fov is not None:
            fov = np.asarray(obj_fov, dtype=np.float64)
            if fov.size != 2 or np.any(fov <= 0):
                raise ValueError(f"`obj_fov` must be a positive (row, col) pair, got {obj_fov!r}")
            canvas_shape = tuple(
                max(1, int(round(f / s * upsampling_factor)))
                for f, s in zip(fov, self.scan_sampling)
            )

        if boundary == "wrap":
            self._warn_if_positions_wrap(positions_up, lo, canvas_shape, upsampling_factor)

        return with_fov(canvas_shape, lo)

    def _warn_if_positions_wrap(self, positions_up, lo, canvas_shape, upsampling_factor):
        """Warn when ``"wrap"`` folds part of the scan back over the object.

        Wrapping a canvas that spans the whole scan is harmless, and is what makes the
        montage reproduce ``DirectPtychography``, which is periodic in the scan. Wrapping a
        smaller one is not: the positions outside come back on the opposite side and lay a
        second, offset copy of the specimen over the first.
        """
        shape = torch.as_tensor(canvas_shape, device=positions_up.device, dtype=lo.dtype)
        below = (lo - positions_up.amin(0)).clamp_min(0)
        above = (positions_up.amax(0) - (lo + shape)).clamp_min(0)
        outside = torch.maximum(below, above)
        if not bool((outside > 0.5).any()):
            return

        overhang = to_numpy(outside) / upsampling_factor * np.asarray(self.scan_sampling)
        warnings.warn(
            f"boundary='wrap' with a canvas of {tuple(canvas_shape)} px leaves scan positions "
            f"up to {np.round(overhang, 1)} Angstrom outside it, which wrap around and lay a "
            "second copy of the specimen over the reconstruction. Either widen `obj_fov` to "
            "cover the scan -- for an empirical probe, to the next whole multiple of the "
            "probe's field of view -- or use boundary='pad', which drops them instead.",
            stacklevel=4,
        )

    def reconstruct(
        self,
        bf_mask=None,
        override_aberration_coefs=None,
        upsampling_factor=None,
        override_rotation_angle=None,
        max_batch_size=None,
        deconvolution_kernel="parallax",
        q_highpass=None,
        q_lowpass=None,
        butterworth_order=12,
        parallax_flip_phase=True,
        verbose=None,
        use_initial_state=False,
        boundary=None,
        defocus_gradient=None,
        interpolation="nearest",
        weight_normalize=None,
        weight_threshold=1e-2,
        pad_px=None,
        obj_origin=None,
        obj_fov=None,
        compute_variance=True,
        suppress_nyquist=False,
        convolution_mode="auto",
        probe_oversample=8,
        stencil_radius="auto",
        truncation_tolerance=0.1,
        max_stencil_radius=32,
        matched_filter_norm_epsilon=1e-1,
        kernel_batch_size=16,
    ):
        """
        Accumulate the canvas and apply the post-hoc Fourier filters.

        Parameters
        ----------
        bf_mask : torch.Tensor, optional
            Subset of the bright-field mask to use. Must be strictly smaller than the mask
            used at initialization.
        override_aberration_coefs : dict, optional
            Aberration coefficients, overriding the hyperparameter state.
        upsampling_factor : int, optional
            Integer factor by which to refine the canvas relative to the scan sampling.
        override_rotation_angle : float, optional
            Detector rotation in degrees, overriding the hyperparameter state.
        max_batch_size : int, optional
            Number of bright-field pixels splatted at once. Defaults to a memory-bounded
            chunk of roughly four million ``(BF pixel, scan position)`` points.
        deconvolution_kernel : str
            ``"prlx"`` (and its aliases) is a pure translation and is exact.

            ``"ssb"``, ``"obf"``, ``"mf"`` and ``"icom"`` are convolutions. They are exact
            by default, evaluated by FFT on the canvas; ``convolution_mode="stencil"``
            truncates them to a box instead.

            Truncating ``"icom"`` is the exception that is not an approximation: it gives
            riCOM, whose real-space kernel is ``r / (2 * pi * |r|**2)`` and whose radius is a
            high-pass cutoff, so no truncation warning is raised for it. Being linear in
            ``k``, it also collapses the detector sum into two convolutions of the
            centre-of-mass shift.
        q_highpass, q_lowpass : float, optional
            Butterworth filter cutoffs, applied once to the finished image.
        parallax_flip_phase : bool
            Apply the ``sign(sin(chi(q)))`` phase-flip filter.
        boundary : {"wrap", "pad"}, optional
            ``"wrap"`` wraps the montage periodically over the scan grid and reproduces
            ``DirectPtychography``; ``"pad"`` grows the canvas to cover the shifted positions
            and drops nothing. Defaults to the value chosen at construction.
        defocus_gradient : tuple of float, optional
            ``(d C10 / d row, d C10 / d col)`` in Angstrom per Angstrom, for a tilted sample
            whose defocus varies across the field of view. Defaults to
            :attr:`defocus_gradient`; pass ``(0.0, 0.0)`` to disable it for one call.

            Each scan position is then shifted by its own local defocus, which a Fourier
            formulation cannot express: ``exp(-1j * grad_chi . q)`` is global over the scan.
            The post-hoc phase flip still uses the aplanatic ``C10``; a space-variant
            contrast-transfer correction is not attempted.
        interpolation : {"nearest", "bilinear"}
            Sub-pixel deposition scheme.

            ``"nearest"`` (default) snaps each shift to the closest canvas pixel, a roll of
            the bright-field image by ``round(shift)`` on a raster scan. The quantization
            error is ``1/(2*upsampling_factor)`` scan pixels, so it shrinks as you upsample:
            against the exact Fourier shift on a 4 mrad apoferritin dataset it retains
            0.81 / 0.95 / 0.99 / 1.00 of the in-band power at ``upsampling_factor``
            1 / 2 / 4 / 8.

            ``"bilinear"`` spreads each shift over the four neighbouring pixels, trading
            some smoothing (0.67 / 0.90 / 0.97 / 0.99 over the same series) for coverage.
            Prefer it for positions off a lattice, where snapping leaves parts of the canvas
            unvisited -- 17-31% empty against 10-15% on a jittered scan.
        weight_normalize : bool, optional
            Divide by the accumulated weight rather than by the total bright-field weight.
            Defaults to ``True`` for an ungridded scan and ``False`` otherwise.

            This corrects for uneven sampling density, which an ungridded scan needs. On a
            raster scan the density is already uniform, so it only rescales the edges of a
            padded canvas, amplifying the noise of the few contributions there; leaving it
            ``False`` lets those edges fade out instead.

            For ``"wrap"`` with ``upsampling_factor > 1`` the accumulated weight is a comb of
            ones and zeros, so normalizing by it is meaningless -- leave it ``False``.
        weight_threshold : float
            Fraction of the peak weight below which the normalized image is tapered to zero,
            following ``bilinear_kde``. Only used when ``weight_normalize`` is true.
        pad_px : int, optional
            Freeze the ``"pad"`` canvas to the position bounding box plus this many pixels,
            instead of sizing it from the (aberration-dependent) shifts. Use this to keep the
            canvas a fixed size across hyperparameter trials or a defocus series, where the
            automatic size would otherwise change with the shifts. Contributions landing
            beyond ``pad_px`` are dropped, so choose it larger than the shifts you expect.
        obj_origin : tuple of float, optional
            Pin the canvas corner to this ``(row, col)`` coordinate in Angstrom, in the same
            frame as the probe positions passed to :meth:`from_dataset3d`. Defaults to
            whatever ``pad_px`` or the shifts imply, which follows each acquisition's own
            bounding box and so differs between them.
        obj_fov : tuple of float, optional
            Pin the canvas extent to ``(rows, cols)`` Angstrom, rather than sizing it from
            the positions. Mutually exclusive with ``pad_px``.

            Together these name a fixed window in the specimen's coordinates, so separate
            acquisitions -- successive frames of a multi-frame scan, say -- reconstruct onto
            pixel-identical canvases that can be stacked, differenced or cross-correlated.
            Without them each frame's canvas follows its own bounding box, leaving two frames
            of the same region a few pixels apart. Both are read back from
            :attr:`obj_origin` and :attr:`_obj_sampling`.

            Neither applies to ``boundary="wrap"``, whose canvas is the scan grid.
        compute_variance : bool
            Accumulate the sum of squares needed by :meth:`variance_loss`.
        suppress_nyquist : bool
            Zero the Nyquist row and column of the phase-flip filter. Off by default, to
            match ``DirectPtychography``; turn it on for odd-order aberrations, where
            ``sign(sin(chi))`` is not symmetric and leaves a checkerboard artifact.
        convolution_mode : {"auto", "fft", "stencil"}
            How the ``ssb`` / ``obf`` / ``mf`` convolutions are evaluated.

            ``"fft"`` splats each bright-field image onto the canvas and multiplies by the
            kernel in ``q``. Nothing is truncated, and it is asymptotically cheaper for a
            kernel that spans the canvas: one transform per bright-field image against
            ``(2 * stencil_radius + 1) ** 2`` taps per scan point. On a gridded scan it
            reproduces ``DirectPtychography`` to float precision, where a radius-5 stencil is
            20-34% off and a radius-12 one still 2-5% off.

            ``"stencil"`` keeps the truncated box, evaluated as a grouped convolution.
            Worth it when the kernel is local, where it avoids a canvas-sized transform.

            ``"auto"`` reads ``stencil_radius``: naming one takes the stencil, leaving it at
            ``"auto"`` takes the FFT. Measuring which is faster would cost a full pass over
            the kernels, so this reads intent rather than benchmarking.

            With ``boundary="pad"`` the FFT route doubles the canvas and crops back, since a
            Fourier convolution is otherwise circular. That costs four times the transform
            area; ``boundary="wrap"`` wants the circular one anyway and pays nothing.
        probe_oversample : int
            How finely an empirical ``fourier_probe`` is refined before being sampled off its
            own reciprocal lattice, which a ``rotation_angle`` that is not a multiple of 90
            degrees forces. Both the accuracy and the memory cost scale as its square:
            measured on a speckled X-ray probe at a generic sub-pixel offset, 21% unrefined,
            7.5% at 2, 1.7% at 4, 0.52% at 8, 0.064% at 16. Ignored for an analytic probe,
            which is evaluated rather than sampled, and for a rotation that maps the lattice
            onto itself.
        stencil_radius : int or "auto"
            Half-width of the box stencil used by the ``ssb`` / ``obf`` / ``mf`` / ``icom``
            kernels, in canvas pixels. ``"auto"`` grows it until the estimated truncation
            error meets ``truncation_tolerance``, capped at ``max_stencil_radius``. For
            ``icom`` this is riCOM's ``(n - 1) / 2``, where setting it is the intent rather
            than a compromise.

            The box carries no taper: tapering measures worse at equal radius (0.40 against
            0.29 relative error at radius 8, 20 mrad in focus), since it discards mid-radius
            content that matters more than the ringing it suppresses.
        truncation_tolerance : float
            Target relative operator error for ``stencil_radius="auto"``. A warning reports
            the achieved error whenever it cannot be met.
        max_stencil_radius : int
            Cap on the automatic radius. Cost scales with its square.
        matched_filter_norm_epsilon : float
            Regularization of the ``mf`` power normalization, as in ``DirectPtychography``.
        kernel_batch_size : int
            Bright-field pixels whose real-space kernels are built at once.

        Returns
        -------
        self

        Notes
        -----
        The convolution kernels cost more than the parallax one either way: an FFT per
        bright-field image, or ``(2 * stencil_radius + 1) ** 2`` deposits per point. On a
        gridded scan ``DirectPtychography`` computes the same thing with a single scan FFT
        and is faster; see the class docstring for when to prefer which.
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

        kernel = self._normalize_kernel_name(deconvolution_kernel)
        if kernel == "prlx":
            # zero aberrations would give zero shifts and quietly sum the bright-field stack
            # into a plain incoherent image, which is not a parallax reconstruction
            self._require_analytic_probe("The parallax kernel")
        if upsampling_factor is None:
            upsampling_factor = 1
        upsampling_factor = math.ceil(upsampling_factor)

        if bf_mask is None:
            bf_mask = self.bf_mask
        bf = self._return_bf_context(bf_mask)

        if boundary is None:
            boundary = self.boundary
        if defocus_gradient is None:
            defocus_gradient = self.defocus_gradient
        if verbose and defocus_gradient is not None:
            print(f"  defocus_gradient={tuple(defocus_gradient)!r} A/A,")
        if weight_normalize is None:
            # density correction matters for an ungridded scan; on a raster it would only
            # amplify noise at the low-weight edges of a padded canvas
            weight_normalize = not self.gridded_scan

        shifts_px, bf_weights = self._return_shifts_px(
            rotation_angle, aberration_coefs, bf.bf_mask, upsampling_factor
        )

        delta_c10 = self._return_delta_c10(defocus_gradient)
        if delta_c10 is None:
            defocus_rate_px = None
        else:
            defocus_rate_px = self._return_defocus_rate_px(
                rotation_angle, bf.bf_mask, upsampling_factor
            )

        canvas_shape, canvas_origin, canvas_fov = self._return_canvas(
            shifts_px,
            upsampling_factor,
            boundary,
            pad_px,
            defocus_rate_px,
            delta_c10,
            obj_origin=obj_origin,
            obj_fov=obj_fov,
        )

        if max_batch_size is None:
            max_batch_size = max(1, _DEFAULT_POINTS_PER_BATCH // max(self.num_positions, 1))

        coords_base = self._positions_px * upsampling_factor - canvas_origin
        self._reset_reconstruction()
        self._kernel = kernel

        mode = self._resolve_convolution_mode(convolution_mode, kernel, stencil_radius)
        stencil_offsets = stencil_weights = kernel_args = norm = None
        fft_shape = canvas_shape

        # riCOM: the iCoM kernel is linear in k, so summing the detector first turns the
        # whole reconstruction into two convolutions of the centre-of-mass shift. Needs
        # every bright-field pixel to deposit at the same place, which a per-position
        # defocus breaks.
        collapse_icom = kernel == "icom" and delta_c10 is None

        if kernel == "prlx":
            deposit_shifts = shifts_px
            self._stencil_info = None
        elif collapse_icom:
            deposit_shifts = torch.zeros_like(shifts_px)
            self._stencil_info = None
            if boundary == "pad" and mode == "fft":
                fft_shape = (canvas_shape[0] * 2, canvas_shape[1] * 2)
            upsampled_sampling = tuple(s / upsampling_factor for s in self.scan_sampling)
            qxa, qya = spatial_frequencies(
                fft_shape if mode == "fft" else canvas_shape,
                upsampled_sampling,
                device=self.device,
            )
            icom_operators = self._return_icom_operators(qxa, qya)
            icom_values = self._return_com_shift(bf, rotation_angle, upsampling_factor)
            _, _, bf_weights = self._return_kernel_context(
                bf,
                kernel=kernel,
                rotation_angle=rotation_angle,
                aberration_coefs=aberration_coefs,
                canvas_shape=canvas_shape,
                upsampling_factor=upsampling_factor,
                matched_filter_norm_epsilon=matched_filter_norm_epsilon,
                kernel_batch_size=kernel_batch_size,
                probe_oversample=probe_oversample,
            )
            if mode == "stencil":
                icom_operators, self._stencil_info = self._truncate_icom_operators(
                    icom_operators, canvas_shape, stencil_radius, max_stencil_radius
                )
        elif mode == "fft":
            # nothing is truncated, so there is no reason to divide the shift out of the
            # kernel and add it back to the deposits
            deposit_shifts = torch.zeros_like(shifts_px)
            self._stencil_info = None
            if boundary == "pad":
                # a Fourier convolution is circular; doubling the canvas and cropping back
                # makes it linear, which is what "pad" asks for
                fft_shape = (canvas_shape[0] * 2, canvas_shape[1] * 2)
            kernel_args, norm, bf_weights = self._return_kernel_context(
                bf,
                kernel=kernel,
                rotation_angle=rotation_angle,
                aberration_coefs=aberration_coefs,
                canvas_shape=fft_shape,
                upsampling_factor=upsampling_factor,
                matched_filter_norm_epsilon=matched_filter_norm_epsilon,
                kernel_batch_size=kernel_batch_size,
                probe_oversample=probe_oversample,
            )
            # each bright-field pixel now needs a canvas of its own, so size the batch by
            # canvas area rather than by scan positions
            max_batch_size = max(1, _DEFAULT_POINTS_PER_BATCH // max(np.prod(fft_shape), 1))
        else:
            # the kernel carries the parallax shift in its phase; deposit at its integer part
            # and leave only the residual in the stencil
            deposit_shifts = shifts_px.round()
            stencil_offsets, stencil_weights, bf_weights, self._stencil_info = (
                self._return_kernel_stencil(
                    bf,
                    kernel=kernel,
                    rotation_angle=rotation_angle,
                    aberration_coefs=aberration_coefs,
                    canvas_shape=canvas_shape,
                    upsampling_factor=upsampling_factor,
                    shift_centers=deposit_shifts,
                    stencil_radius=stencil_radius,
                    truncation_tolerance=truncation_tolerance,
                    max_stencil_radius=max_stencil_radius,
                    matched_filter_norm_epsilon=matched_filter_norm_epsilon,
                    kernel_batch_size=kernel_batch_size,
                    verbose=verbose,
                    probe_oversample=probe_oversample,
                )
            )
            # a stencil of S taps costs S deposits per point, so shrink the batch to match
            max_batch_size = max(1, max_batch_size // max(len(stencil_offsets), 1))

        buffers = (
            allocate_splat_buffers(canvas_shape, self.device, accumulate_squares=compute_variance)
            if kernel == "prlx"
            else None
        )
        accumulator = (
            None
            if kernel == "prlx"
            else torch.zeros(fft_shape, device=self.device, dtype=torch.complex64)
        )

        n_components = 0 if collapse_icom else bf.num_bf
        pbar = tqdm(range(n_components), disable=not verbose)
        batcher = SimpleBatcher(
            n_components, batch_size=max_batch_size, shuffle=False, rng=self.rng
        )

        for batch_idx in batcher:
            mapped_idx = bf.vbf_index_mapping[batch_idx]
            values = self._vbf_stack[mapped_idx]  # (B, N_pos)
            coords = coords_base[None] + deposit_shifts[batch_idx][:, None]  # (B, N_pos, 2)
            if delta_c10 is not None:
                # each position gets its own defocus, hence its own shift; the shift is
                # exactly linear in C10, so this is one broadcast add rather than a re-fit
                coords = coords + defocus_rate_px[batch_idx][:, None, :] * delta_c10[None, :, None]

            if kernel == "prlx":
                scatter_add_splat(
                    values,
                    coords,
                    canvas_shape,
                    boundary=boundary,
                    interpolation=interpolation,
                    out=buffers,
                )
            elif collapse_icom:
                pass  # handled in one shot below, outside the bright-field loop
            elif mode == "fft":
                stack = splat_stack(
                    values,
                    coords,
                    fft_shape,
                    boundary="pad" if fft_shape != canvas_shape else boundary,
                    interpolation=interpolation,
                )
                kernel_fourier, _ = self._return_kernel_fourier(batch_idx, *kernel_args, norm)
                accumulator += convolve_stack_fourier(stack, kernel_fourier)
            else:
                accumulator += splat_and_convolve(
                    values,
                    coords,
                    canvas_shape,
                    stencil_weights[batch_idx],
                    self._stencil_info["stencil_radius"],
                    boundary=boundary,
                    interpolation=interpolation,
                ).sum(0)
            pbar.update(len(batch_idx))
        pbar.close()

        if collapse_icom:
            coords = coords_base[None].expand(2, -1, -1)
            if mode == "fft":
                stack = splat_stack(
                    icom_values,
                    coords,
                    fft_shape,
                    boundary="pad" if fft_shape != canvas_shape else boundary,
                    interpolation=interpolation,
                )
                accumulator = convolve_stack_fourier(stack, icom_operators)
            else:
                accumulator = splat_and_convolve(
                    icom_values,
                    coords,
                    canvas_shape,
                    icom_operators,
                    self._stencil_info["stencil_radius"],
                    boundary=boundary,
                    interpolation=interpolation,
                ).sum(0)

        self._canvas_shape = canvas_shape
        self._canvas_origin_px = canvas_origin
        self._canvas_fov = canvas_fov
        self._bf_weights = bf_weights

        if kernel != "prlx":
            if mode == "fft":
                accumulator = torch.fft.ifft2(accumulator)
            # matches DirectPtychography, which takes the real part of the summed stack
            obj = accumulator[: canvas_shape[0], : canvas_shape[1]].real / bf_weights
        else:
            self._sum_w, self._sum_wv, self._sum_wv2 = buffers
            # normalization must precede filtering: dividing by the (spatially varying)
            # weight map is not linear, so it does not commute with the Fourier filters below
            if weight_normalize:
                mean, _, support = self._weighted_moments(weight_threshold)
                obj = (mean * support).reshape(canvas_shape)
            else:
                obj = self._sum_wv.reshape(canvas_shape) / bf_weights

        obj = self._apply_fourier_filters(
            obj,
            aberration_coefs=aberration_coefs,
            upsampling_factor=upsampling_factor,
            q_lowpass=q_lowpass,
            q_highpass=q_highpass,
            butterworth_order=butterworth_order,
            # only `prlx` needs the phase flip -- the deconvolution kernels already invert
            # the contrast transfer, and DirectPtychography draws the same line
            parallax_flip_phase=parallax_flip_phase and kernel == "prlx",
            suppress_nyquist=suppress_nyquist,
        )
        self._corrected_bf = obj.to(torch.float32)

        # memory management
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return self

    def _apply_fourier_filters(
        self,
        obj,
        *,
        aberration_coefs,
        upsampling_factor,
        q_lowpass,
        q_highpass,
        butterworth_order,
        parallax_flip_phase,
        suppress_nyquist,
    ):
        """
        Apply the bright-field-index-independent filters once, on the summed image.

        ``DirectPtychography`` multiplies these into every bright-field image before its
        inverse transform; because they do not depend on the bright-field index, doing it
        once on the sum is exactly equivalent.
        """
        if not (parallax_flip_phase or q_lowpass or q_highpass or suppress_nyquist):
            return obj

        upsampled_sampling = tuple(s / upsampling_factor for s in self.scan_sampling)
        qxa, qya = spatial_frequencies(obj.shape, upsampled_sampling, device=self.device)
        q, theta = polar_coordinates(qxa, qya)

        # built at the grid's native precision, not the accumulator's: chi(q) reaches tens
        # of radians, so sign(sin(chi)) is ill-conditioned at its zero crossings and float64
        # would flip a handful of pixels relative to DirectPtychography
        filt = torch.ones_like(q)
        if parallax_flip_phase:
            chi_q = aberration_surface(
                q * self.wavelength,
                theta,
                self.wavelength,
                aberration_coefs=aberration_coefs,
            )
            filt = filt * torch.sign(torch.sin(chi_q))
        if q_lowpass:
            filt = filt / (1 + (q / q_lowpass) ** (2 * butterworth_order))
        if q_highpass:
            filt = filt * (1 - 1 / (1 + (q / q_highpass) ** (2 * butterworth_order)))
        if suppress_nyquist:
            n_rows, n_cols = obj.shape
            if n_rows % 2 == 0:
                filt[n_rows // 2, :] = 0.0
            if n_cols % 2 == 0:
                filt[:, n_cols // 2] = 0.0

        return torch.fft.ifft2(torch.fft.fft2(obj) * filt.to(obj.dtype)).real

    @staticmethod
    def _moments_from_buffers(sum_w, sum_wv, sum_wv2, weight_threshold: float = 1e-2):
        """``(mean, variance, support)`` per canvas pixel, from raw splat accumulators."""
        w = sum_w
        tiny = torch.finfo(w.dtype).tiny
        inv_w = 1 / w.clamp_min(tiny)

        mean = sum_wv * inv_w
        if sum_wv2 is None:
            var = torch.zeros_like(mean)
        else:
            var = (sum_wv2 * inv_w - mean.square()).clamp_min(0)

        w_max = w.max()
        if w_max <= 0:
            support = torch.zeros_like(w)
        else:
            support = (w / (weight_threshold * w_max)).clamp(max=1.0)

        return mean * (w > 0), var * (w > 0), support

    @classmethod
    def _variance_loss_from_buffers(cls, sum_w, sum_wv, sum_wv2):
        """Weight-averaged per-pixel variance across bright-field images."""
        _, var, _ = cls._moments_from_buffers(sum_w, sum_wv, sum_wv2)
        denom = sum_w.sum()
        if denom <= 0:
            return torch.tensor(torch.inf, dtype=sum_w.dtype, device=sum_w.device)
        return (var * sum_w).sum() / denom

    def _weighted_moments(self, weight_threshold: float = 1e-2):
        """``(mean, variance, support)`` per canvas pixel, as flat tensors."""
        if self._sum_w is None or self._sum_wv is None:
            raise RuntimeError("Run reconstruct() before asking for the accumulated moments.")
        return self._moments_from_buffers(
            self._sum_w, self._sum_wv, self._sum_wv2, weight_threshold
        )

    def variance_loss(self):
        """
        Weight-averaged variance across bright-field images, without storing the stack.

        Accumulating ``sum(w)``, ``sum(w*v)`` and ``sum(w*v**2)`` during the splat gives the
        per-pixel population variance across bright-field images directly, so no
        ``(N_bf, Ny, Nx)`` stack is needed.

        It differs from ``DirectPtychography.variance_loss`` in four documented ways:

        1. It is a weight-averaged mean over pixels rather than an unweighted one. The two
           coincide for ``boundary="wrap"`` on a complete grid with ``upsampling_factor=1``,
           where the accumulated weight is exactly ``num_bf`` everywhere.
        2. It lacks the ``1 / bf_weights**2`` scale, being computed on the raw values.
        3. It is computed *before* the phase-flip and Butterworth filters, which are applied
           post-hoc here but per bright-field image there. With ``parallax_flip_phase=True``
           this is a genuinely different objective.
        4. For ``upsampling_factor > 1`` it ignores unvisited canvas pixels instead of
           counting them as zeros.

        With ``interpolation="bilinear"`` and non-integer shifts it also folds in the
        within-interpolation spread, since ``sum(w*v**2)`` averages ``v**2`` over the
        neighbouring source pixels. Prefer ``upsampling_factor=1`` and
        ``parallax_flip_phase=False`` when driving a hyperparameter search.
        """
        if self._kernel != "prlx":
            raise NotImplementedError(
                f"variance_loss is only defined for the parallax kernel, not {self._kernel!r}: "
                "the convolution kernels deposit complex weights that are not a partition of "
                "unity, so there is no per-pixel spread across bright-field images to take. "
                "To drive a hyperparameter search on this kernel pass loss='rms_gradient', "
                "which measures the reconstructed image instead and is defined for every "
                "kernel."
            )
        if self._sum_w is None or self._sum_wv2 is None:
            return None
        return self._variance_loss_from_buffers(self._sum_w, self._sum_wv, self._sum_wv2)

    # ------------------------------------------------------------------
    # position-dependent defocus
    # ------------------------------------------------------------------

    def _return_patch_indices(self, patch_grid) -> list[torch.Tensor]:
        """Position indices for each tile of a ``patch_grid`` partition of the scan bbox."""
        pos = self._positions_px
        lo = pos.amin(0)
        span = (pos.amax(0) - lo).clamp_min(torch.finfo(pos.dtype).tiny)

        tile = torch.stack(
            [
                ((pos[:, i] - lo[i]) / span[i] * patch_grid[i]).floor().clamp(0, patch_grid[i] - 1)
                for i in range(2)
            ],
            dim=-1,
        ).to(torch.int64)

        flat = tile[:, 0] * patch_grid[1] + tile[:, 1]
        return [
            torch.nonzero(flat == p, as_tuple=True)[0]
            for p in range(int(patch_grid[0]) * int(patch_grid[1]))
        ]

    def _patch_canvas(self, pos_idx, margin_px, upsampling_factor):
        """``(canvas_shape, origin)`` for a patch, sized independently of the trial defocus."""
        positions_up = self._positions_px[pos_idx] * upsampling_factor
        lo = torch.floor(positions_up.amin(0) - margin_px)
        hi = torch.ceil(positions_up.amax(0) + margin_px)
        return tuple(int(v) + 2 for v in (hi - lo)), lo

    def _patch_variance_loss(
        self,
        pos_idx,
        canvas_shape,
        canvas_origin,
        *,
        bf,
        rotation_angle,
        aberration_coefs,
        upsampling_factor,
        interpolation,
        max_batch_size,
    ):
        """Variance loss of a montage built from a spatial subset of the scan positions.

        Deliberately does not go through :meth:`reconstruct`: threading a position subset
        through the public signature would make its canvas logic branch for a purely internal
        use. It is otherwise the same functional as :meth:`variance_loss` -- the same
        weight-averaged per-pixel spread over the same accumulators -- so the patch fit and
        a hyperparameter search cannot disagree about what a good defocus is.

        The canvas is passed in frozen, rather than sized from the shifts, which is what
        makes the value comparable *across trial defocus values*: a canvas that grew with
        the defocus would add low-weight edge pixels, whose variance is small simply because
        few images reach them, and the loss would then fall monotonically with defocus
        rather than have a minimum at the right one. Weighting by ``sum_w`` is what keeps
        those edges from dominating once the canvas is fixed.
        """
        shifts_px, _ = self._return_shifts_px(
            rotation_angle, aberration_coefs, bf.bf_mask, upsampling_factor
        )

        buffers = allocate_splat_buffers(canvas_shape, self.device, accumulate_squares=True)
        coords_base = self._positions_px[pos_idx] * upsampling_factor - canvas_origin

        batcher = SimpleBatcher(bf.num_bf, batch_size=max_batch_size, shuffle=False, rng=self.rng)
        for batch_idx in batcher:
            values = self._vbf_stack[bf.vbf_index_mapping[batch_idx]][:, pos_idx]
            coords = coords_base[None] + shifts_px[batch_idx][:, None]
            scatter_add_splat(
                values,
                coords,
                canvas_shape,
                boundary="pad",
                interpolation=interpolation,
                out=buffers,
            )

        # Score the whole patch, not its densest spots. An ungridded weight map is uneven
        # everywhere, so a 90%-of-peak cut kept 14% of the canvas and moved with the
        # defocus -- putting the apoferritin minimum at C10 = 9.5 kA against a true 13.0 kA.
        return float(self._variance_loss_from_buffers(*buffers))

    @staticmethod
    def _refine_minimum(values, losses):
        """Sub-grid minimum by a 3-point parabolic fit, or ``None`` if pinned at an end."""
        i = int(np.argmin(losses))
        if i == 0 or i == len(losses) - 1:
            return None

        y0, y1, y2 = losses[i - 1], losses[i], losses[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom <= 0:  # not a minimum -- flat or concave
            return float(values[i])

        # vertex offset in units of the (possibly uneven) local step
        offset = 0.5 * (y0 - y2) / denom
        step = values[i + 1] - values[i] if offset > 0 else values[i] - values[i - 1]
        return float(values[i] + offset * abs(step))

    def defocus_map(
        self,
        c10_values,
        patch_grid: Tuple[int, int] = (3, 3),
        bf_mask=None,
        override_aberration_coefs=None,
        override_rotation_angle=None,
        upsampling_factor: int = 1,
        interpolation: str = "bilinear",
        min_patch_positions: int = 64,
        max_batch_size=None,
        verbose=None,
    ) -> dict:
        """
        Best-fit defocus in each of a grid of spatial patches.

        Reconstructs each patch on its own small canvas over a range of trial ``C10`` values
        and takes the variance-loss minimum, refined by a parabolic fit so a coarse
        ``c10_values`` grid still resolves sub-step differences. This is the measurement
        :meth:`fit_defocus_gradient` fits a plane to; call it directly to inspect the loss
        curves before trusting the fit.

        Parameters
        ----------
        c10_values : array-like
            Trial defocus values in Angstrom, ascending. Must bracket the true local defocus
            in every patch -- a patch whose minimum sits on an endpoint is flagged invalid.
        patch_grid : tuple of int
            Number of patches along each scan axis. Any positive pair; a 1D grid such as
            ``(4, 1)`` gives a defocus profile along one axis. Fitting a *plane* needs at
            least three patches -- :meth:`fit_defocus_gradient` enforces that.
        interpolation : {"bilinear", "nearest"}
            Defaults to ``"bilinear"`` here, unlike :meth:`reconstruct`. Snapping to the
            nearest pixel makes the loss a staircase in ``C10``, so small defocus changes
            produce no change at all and the minimum cannot be located.
        min_patch_positions : int
            Patches with fewer positions than this are flagged invalid.

        Returns
        -------
        dict with keys
            ``centers_A`` ``(P, 2)`` patch centers in Angstrom, measured from the position
            centroid so they share the frame of :attr:`defocus_gradient`; ``c10_best``
            ``(P,)``; ``losses`` ``(P, n_c10)``; ``valid`` ``(P,)`` bool; ``c10_values``.

        Notes
        -----
        The estimator carries a small offset that is uniform across patches -- a few percent
        of ``C10`` on synthetic data. That cancels out of the *gradient*, which is a
        difference between patches, but it does bias the absolute defocus, so treat the
        offset from :meth:`fit_defocus_gradient` as approximate.

        Patches need enough positions for the loss to have a clear minimum, and how many is
        data dependent -- the warning here only catches patches smaller than the shifts
        themselves. The reliable check is to run this on a region you believe is flat and
        confirm ``c10_best`` comes back constant.
        """
        if verbose is None:
            verbose = self.verbose

        c10_values = np.asarray(c10_values, dtype=np.float64).ravel()
        if c10_values.size < 3:
            raise ValueError(
                f"`c10_values` needs at least 3 points to bracket a minimum, got "
                f"{c10_values.size}."
            )
        if min(int(patch_grid[0]), int(patch_grid[1])) < 1:
            raise ValueError(f"`patch_grid` entries must be positive, got {patch_grid!r}.")

        state = self.hyperparameter_state
        base_coefs = state.current_aberrations(override_aberration_coefs)
        rotation_angle = state.current_rotation_angle(override_rotation_angle)

        bf = self._return_bf_context(self.bf_mask if bf_mask is None else bf_mask)
        if max_batch_size is None:
            max_batch_size = max(1, _DEFAULT_POINTS_PER_BATCH // max(self.num_positions, 1))

        patches = self._return_patch_indices(patch_grid)
        scan_sampling = torch.as_tensor(
            tuple(self.scan_sampling), device=self.device, dtype=self._float_dtype
        )

        # size every patch canvas for the largest trial defocus, so it stays fixed across
        # the scan below -- see _patch_variance_loss
        max_shift = 0.0
        for c10 in (c10_values.min(), c10_values.max()):
            trial, _ = self._return_shifts_px(
                rotation_angle, {**base_coefs, "C10": float(c10)}, bf.bf_mask, upsampling_factor
            )
            max_shift = max(max_shift, float(trial.abs().max()))
        margin_px = math.ceil(max_shift) + 1

        centers, best, all_losses, valid = [], [], [], []
        pbar = tqdm(patches, disable=not verbose, desc="defocus map")
        for pos_idx in pbar:
            center = (
                (self._positions_px[pos_idx].mean(0) - self.positions_centroid_px) * scan_sampling
                if pos_idx.numel()
                else torch.zeros(2, device=self.device, dtype=self._float_dtype)
            )
            centers.append(center.cpu().numpy())

            if pos_idx.numel() < min_patch_positions:
                all_losses.append(np.full(c10_values.size, np.nan))
                best.append(np.nan)
                valid.append(False)
                continue

            extent = self._positions_px[pos_idx].amax(0) - self._positions_px[pos_idx].amin(0)
            if float(extent.min()) < 2 * max_shift:
                warnings.warn(
                    f"A patch spans {float(extent.min()):.0f} scan pixels but the parallax "
                    f"shifts reach {max_shift:.0f}, so its bright-field images barely "
                    "overlap and the variance loss has little to compare. Use a coarser "
                    "`patch_grid`.",
                    stacklevel=2,
                )

            canvas_shape, canvas_origin = self._patch_canvas(pos_idx, margin_px, upsampling_factor)
            losses = np.array(
                [
                    self._patch_variance_loss(
                        pos_idx,
                        canvas_shape,
                        canvas_origin,
                        bf=bf,
                        rotation_angle=rotation_angle,
                        aberration_coefs={**base_coefs, "C10": float(c10)},
                        upsampling_factor=upsampling_factor,
                        interpolation=interpolation,
                        max_batch_size=max_batch_size,
                    )
                    for c10 in c10_values
                ]
            )
            all_losses.append(losses)

            refined = self._refine_minimum(c10_values, losses)
            best.append(np.nan if refined is None else refined)
            valid.append(refined is not None)
        pbar.close()

        results = {
            "centers_A": np.stack(centers),
            "c10_best": np.array(best),
            "losses": np.stack(all_losses),
            "valid": np.array(valid),
            "c10_values": c10_values,
        }
        self._defocus_map_results = results
        return results

    def fit_defocus_gradient(
        self,
        c10_values,
        patch_grid: Tuple[int, int] = (3, 3),
        update_defocus: bool = True,
        verbose=None,
        **defocus_map_kwargs,
    ):
        """
        Fit a defocus plane across the field of view and store it as :attr:`defocus_gradient`.

        Runs :meth:`defocus_map` and least-squares fits ``C10 = offset + g . r`` through the
        valid patch centers. Because the centers are measured from the position centroid, the
        offset is the mean defocus and the gradient is exactly orthogonal to it -- so a
        subsequent :meth:`grid_search_hyperparameters` over ``C10`` stays well posed.

        Parameters
        ----------
        update_defocus : bool
            Also write the fitted offset into the optimized ``C10``. On by default: the offset
            and the gradient are fit jointly, so keeping a stale global ``C10`` alongside a
            fresh gradient would be inconsistent.

        Returns
        -------
        self
        """
        if verbose is None:
            verbose = self.verbose

        results = self.defocus_map(
            c10_values, patch_grid=patch_grid, verbose=verbose, **defocus_map_kwargs
        )

        valid = results["valid"]
        if valid.sum() < 3:
            raise RuntimeError(
                f"Only {int(valid.sum())} of {valid.size} patches gave a bracketed minimum, "
                "and a plane needs 3. Widen `c10_values`, or use a coarser `patch_grid` so "
                "each patch has more positions."
            )

        centers = results["centers_A"][valid]
        design = np.column_stack([np.ones(len(centers)), centers])
        offset, g_row, g_col = np.linalg.lstsq(design, results["c10_best"][valid], rcond=None)[0]

        self.defocus_gradient = (float(g_row), float(g_col))

        if update_defocus:
            state = self.hyperparameter_state
            coefs = state.current_aberrations()
            coefs["C10"] = float(offset)
            state.optimized_aberrations = validate_aberration_coefficients(coefs)
            state.optimized_keys.add("C10")

        if verbose:
            residual = results["c10_best"][valid] - design @ [offset, g_row, g_col]
            print(
                f"Fitted defocus plane over {int(valid.sum())}/{valid.size} patches:\n"
                f"  C10 offset      = {offset:.1f} A\n"
                f"  defocus_gradient= ({g_row:.4g}, {g_col:.4g}) A/A "
                f"(|g| = {math.hypot(g_row, g_col):.4g}, "
                f"tilt = {math.degrees(math.atan(math.hypot(g_row, g_col))):.2f} deg)\n"
                f"  residual RMS    = {float(np.sqrt((residual**2).mean())):.1f} A"
            )

        return self
