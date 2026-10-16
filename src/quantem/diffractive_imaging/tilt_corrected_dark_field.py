"""Tilt-corrected dark field (tcDF), and its fusion with a bright-field reconstruction."""

import gc
import math
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from quantem.core import config
from quantem.core.datastructures import Dataset4d
from quantem.core.utils.utils import to_numpy
from quantem.diffractive_imaging.complex_probe import (
    aberration_surface_cartesian_gradients,
    polar_coordinates,
    spatial_frequencies,
)
from quantem.diffractive_imaging.direct_ptycho_utils import (
    _rotation_degrees_to_radians,
    bright_field_mask,
    explained_energy,
    fit_and_shift_diffraction_origin,
    overlap_terms,
    preferred_float_dtype,
)
from quantem.diffractive_imaging.direct_ptychography_base import (
    DirectPtychographyBase,
    HyperparameterState,
)
from quantem.diffractive_imaging.ptycho_utils import SimpleBatcher

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class TiltCorrectedDarkField(DirectPtychographyBase):
    """
    Tilt-corrected dark field: the incoherent dark-field signal, shift-corrected by segment.

    Electrons scattered outside the bright-field disk carry incoherent contrast. Each
    dark-field pixel's image is displaced by the probe's aberrations just as a bright-field
    pixel's is, but the displacement depends on the specimen's scattering factor as well as
    on the detector position, since the intensity at a dark-field pixel comes mostly from the
    nearest part of the illumination cone. Following Ma et al. [1]_ (section 2.3), the
    dark-field annulus is cut into narrow azimuthal segments, each segment's pixels are summed
    into one image, and each image is shifted as a bright-field pixel would be at a fixed
    radius in that segment's direction. For pure defocus the shift at two thirds of the
    aperture radius, ``(2/3) C10 alpha``, is what Ma et al. found satisfactory; the exact
    value depends on the scattering factor, which is why it is a parameter here.

    The image is the mean of the shifted segment images, with no phase flip, since the
    contrast is incoherent. It carries information to ``2 alpha / lambda`` with an envelope
    set by the illumination cone, and complements a bright-field reconstruction at low
    spatial frequencies, where :func:`fuse_full_field` combines the two.

    Instantiate with :meth:`from_dataset4d`, or :meth:`from_direct_ptychography` to start from
    a bright-field reconstruction's fitted aberrations. The result is
    :attr:`corrected_bf`, named for the shared base class; here it is the tcDF image.

    References
    ----------
    .. [1] D. Ma et al., *Ultramicroscopy* 283, 114351 (2026).
    """

    _token = object()

    def __init__(
        self,
        vdf_fourier,
        segment_masks,
        bf_mask,
        dataset: Dataset4d,
        wavelength: float,
        semiangle_cutoff: float,
        rotation_angle: float,
        aberration_coefs: dict,
        rng,
        device,
        verbose,
        _token=None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use TiltCorrectedDarkField.from_dataset4d() or "
                "TiltCorrectedDarkField.from_direct_ptychography() to instantiate this class."
            )
        self.device = device
        self.verbose = verbose
        self.rng = rng
        self.wavelength = wavelength
        self.scan_units = tuple(dataset.units[:2])
        self.detector_units = tuple(dataset.units[-2:])
        self.scan_sampling = dataset.sampling[:2]
        self.reciprocal_sampling = dataset.sampling[-2:]
        self.angular_sampling = tuple(d * 1e3 * self.wavelength for d in self.reciprocal_sampling)
        self.scan_gpts = tuple(int(n) for n in dataset.shape[:2])
        self.gpts = tuple(int(n) for n in dataset.shape[-2:])
        self.sampling = tuple(1 / s / n for n, s in zip(self.reciprocal_sampling, self.gpts))
        self.bf_mask = bf_mask
        self.semiangle_cutoff = semiangle_cutoff
        self.soft_edges = True
        self.hyperparameter_state = HyperparameterState(
            initial_aberrations=aberration_coefs, initial_rotation_angle=rotation_angle
        )
        self._vdf_fourier = vdf_fourier
        self._segment_masks = segment_masks
        self._corrected_bf = None
        self._residual = None

    @classmethod
    def from_dataset4d(
        cls,
        dataset: Dataset4d,
        energy: float | None = None,
        semiangle_cutoff: float | None = None,
        aberration_coefs: dict = {},
        wavelength: float | None = None,
        rotation_angle: float | None = None,
        n_angular_bins: int | None = 32,
        inner_radius: float | None = None,
        outer_radius: float | None = None,
        intensity_threshold: float | None = None,
        max_batch_size: int | None = None,
        fit_method: str = "plane",
        mode: str = "bilinear",
        force_measured_origin=None,
        force_fitted_origin=None,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
        verbose: int | bool = True,
    ):
        """
        Build the segment images from a 4D-STEM dataset, which must include the dark field.

        Parameters
        ----------
        dataset : Dataset4d
            ``(Rx, Ry, Qx, Qy)``, with the detector extending beyond the bright-field disk.
        energy, wavelength : float
            Exactly one: the electron energy in volts, or the wavelength in Angstrom.
        semiangle_cutoff : float
            Probe semi-angle in mrad.
        aberration_coefs : dict
            Initial aberration coefficients, as for ``DirectPtychography``.
        rotation_angle : float, optional
            Detector rotation in degrees; estimated from the centre of mass when omitted.
        n_angular_bins : int or None
            Number of azimuthal segments. ``None`` makes every dark-field pixel its own segment
            (the "all" mode of Ma et al.), which is the slowest and noisiest choice.
        inner_radius, outer_radius : float, optional
            The annulus, in inverse Angstrom. Defaults to one detector pixel beyond the
            bright-field disk, and to the detector edge.
        intensity_threshold, max_batch_size, fit_method, mode, force_measured_origin, \
force_fitted_origin
            Diffraction-origin fitting, as for ``DirectPtychography.from_dataset4d``.
        """
        device, _ = config.validate_device(device)
        resolved_wavelength = cls._resolve_wavelength(energy, wavelength)
        shifted, rotation_angle = fit_and_shift_diffraction_origin(
            dataset,
            device=device,
            max_batch_size=max_batch_size,
            fit_method=fit_method,
            mode=mode,
            force_measured_origin=force_measured_origin,
            force_fitted_origin=force_fitted_origin,
            rotation_angle=rotation_angle,
        )
        aperture = cls._model_aperture_for(energy, wavelength, semiangle_cutoff, None, device)
        bf_mask = bright_field_mask(
            shifted.mean(dim=(0, 1)),
            intensity_threshold,
            None
            if aperture is None
            else aperture(
                tuple(shifted.shape[-2:]),
                tuple(float(v) for v in dataset.sampling[-2:]),
                tuple(dataset.units[-2:]),
                rotation_angle,
            ),
        )

        reciprocal = np.asarray(dataset.sampling[-2:], dtype=np.float64)
        if tuple(dataset.units[-2:])[0] == "mrad":
            reciprocal = reciprocal / resolved_wavelength / 1e3
        gpts = tuple(int(n) for n in dataset.shape[-2:])
        sampling = tuple(1 / s / n for n, s in zip(reciprocal, gpts))
        kxa, kya = spatial_frequencies(gpts, sampling, device=shifted.device)
        k, theta = polar_coordinates(kxa, kya)

        if inner_radius is None:
            inner_radius = float(k[bf_mask].max()) + float(reciprocal.max())
        annulus = k >= inner_radius
        if outer_radius is not None:
            annulus &= k < outer_radius
        if not bool(annulus.any()):
            raise ValueError(
                "The dark-field annulus is empty: the detector must extend beyond the "
                f"bright-field disk (inner radius {inner_radius:.3f} 1/A, detector edge "
                f"{float(k.max()):.3f} 1/A)."
            )

        if n_angular_bins is None:
            segment_masks = torch.zeros(
                (int(annulus.sum()), *gpts), dtype=torch.bool, device=shifted.device
            )
            rows, cols = torch.nonzero(annulus, as_tuple=True)
            segment_masks[torch.arange(len(rows)), rows, cols] = True
        else:
            angle = torch.remainder(theta, 2 * math.pi)
            # pixels on a bin edge, along the axes and diagonals, go to the bin above whatever
            # precision theta was computed in: float32 and float64 otherwise split them differently
            index = torch.floor(angle / (2 * math.pi) * n_angular_bins + 1e-4).long()
            index = torch.remainder(index, int(n_angular_bins))
            segment_masks = torch.stack(
                [annulus & (index == b) for b in range(int(n_angular_bins))]
            )
            segment_masks = segment_masks[segment_masks.flatten(1).any(1)]

        # mean intensity per segment at every scan position, normalized to unit mean
        flat = shifted.reshape(*shifted.shape[:2], -1).to(torch.float32)
        weights = segment_masks.flatten(1).to(torch.float32)
        weights = weights / weights.sum(1, keepdim=True)
        vdf = torch.einsum("rcp,sp->src", flat, weights)
        vdf = vdf / vdf.mean((-2, -1), keepdim=True).clamp_min(torch.finfo(vdf.dtype).tiny)
        vdf_fourier = torch.fft.fft2(vdf)
        vdf_fourier[..., 0, 0] = 0
        del shifted, flat
        gc.collect()

        return cls(
            vdf_fourier=vdf_fourier,
            segment_masks=segment_masks,
            bf_mask=bf_mask,
            dataset=dataset,
            wavelength=resolved_wavelength,
            semiangle_cutoff=semiangle_cutoff,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            rng=rng,
            device=device,
            verbose=verbose,
            _token=cls._token,
        )

    @classmethod
    def from_direct_ptychography(cls, reconstruction, dataset: Dataset4d, **kwargs):
        """Start from a bright-field reconstruction's current aberrations and rotation.

        ``dataset`` is the same 4D-STEM data, with its dark field, since the bright-field
        classes keep only the bright-field pixels.
        """
        state = reconstruction.hyperparameter_state
        defaults = dict(
            wavelength=reconstruction.wavelength,
            semiangle_cutoff=reconstruction.semiangle_cutoff,
            aberration_coefs=state.current_aberrations(),
            rotation_angle=state.current_rotation_angle(),
            device=reconstruction.device,
            verbose=reconstruction.verbose,
        )
        defaults.update(kwargs)
        return cls.from_dataset4d(dataset, **defaults)

    @property
    def num_segments(self) -> int:
        return int(self._segment_masks.shape[0])

    @property
    def corrected_bf(self):
        """The tcDF image, or ``None`` before :meth:`reconstruct`."""
        return self._corrected_bf

    def _segment_shifts(self, aberration_coefs, rotation_angle, shift) -> "torch.Tensor":
        """``(num_segments, 2)`` image shift of every segment, in the units of ``grad chi``."""
        kxa, kya = spatial_frequencies(
            self.gpts,
            self.sampling,
            rotation_angle=_rotation_degrees_to_radians(rotation_angle),
            device=self.device,
        )
        masks = self._segment_masks.to(self.device)
        if shift == "segment":
            k, phi = polar_coordinates(kxa, kya)
            dx, dy = aberration_surface_cartesian_gradients(
                k * self.wavelength, phi, aberration_coefs=aberration_coefs
            )
            weights = masks.flatten(1).to(dx.dtype)
            weights = weights / weights.sum(1, keepdim=True)
            return torch.stack((weights @ dx.reshape(-1), weights @ dy.reshape(-1)), -1)

        radius = 2 / 3 if shift == "two_thirds" else float(shift)
        k_alpha = self.semiangle_cutoff * 1e-3 / self.wavelength
        weights = masks.flatten(1).to(kxa.dtype)
        direction = torch.stack((weights @ kxa.reshape(-1), weights @ kya.reshape(-1)), -1)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        k_segment = direction * radius * k_alpha
        k, phi = polar_coordinates(k_segment[:, 0], k_segment[:, 1])
        dx, dy = aberration_surface_cartesian_gradients(
            k * self.wavelength, phi, aberration_coefs=aberration_coefs
        )
        return torch.stack((dx, dy), -1)

    def reconstruct(
        self,
        override_aberration_coefs=None,
        override_rotation_angle=None,
        shift="two_thirds",
        upsampling_factor=None,
        q_highpass=None,
        q_lowpass=None,
        butterworth_order=12,
        band_limit=True,
        max_batch_size=None,
        verbose=None,
        use_initial_state=False,
    ):
        """
        Shift every segment image and average them.

        Parameters
        ----------
        override_aberration_coefs : dict, optional
            Aberrations, overriding the hyperparameter state. ``C10 + z`` sections at depth
            ``z``.
        override_rotation_angle : float, optional
            Detector rotation in degrees, overriding the hyperparameter state.
        shift : "two_thirds", float or "segment"
            Where each segment's shift is evaluated. ``"two_thirds"`` and a float evaluate the
            aberration gradient at that fraction of the aperture radius along the segment's
            direction; Ma et al. found two thirds satisfactory for pure defocus. ``"segment"``
            averages the gradient over the segment's own pixels, beyond the aperture, which
            over-shifts.
        upsampling_factor : int, optional
            Interpolate the image onto a grid this many times finer, by zero-padding its
            spectrum. Unlike the bright-field classes this recovers no frequency beyond the
            scan's Nyquist limit: their shifts fill a disk and unfold the aliases, while the
            segment shifts all lie on one ring and cannot.
        q_highpass, q_lowpass, butterworth_order : optional
            Butterworth filters. ``q_lowpass`` is clamped to ``2 * semiangle_cutoff /
            wavelength`` unless ``band_limit=False``.
        max_batch_size : int, optional
            Segments processed at once.

        Returns
        -------
        self
        """
        if verbose is None:
            verbose = self.verbose
        state = self.hyperparameter_state
        if use_initial_state:
            aberration_coefs = state.initial_aberrations
            rotation_angle = state.initial_rotation_angle
        else:
            aberration_coefs = state.current_aberrations(override_aberration_coefs)
            rotation_angle = state.current_rotation_angle(override_rotation_angle)

        upsampling_factor = 1 if upsampling_factor is None else math.ceil(upsampling_factor)
        qxa, qya = spatial_frequencies(self.scan_gpts, self.scan_sampling, device=self.device)
        qvec = torch.stack((qxa, qya), 0)

        shifts = self._segment_shifts(aberration_coefs, rotation_angle, shift)

        num = self.num_segments
        # the widest precision the device has: MPS has no float64
        real_dtype = preferred_float_dtype(self.device)
        complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        shifted_sum = torch.zeros(self.scan_gpts, device=self.device, dtype=complex_dtype)
        energy = torch.zeros(self.scan_gpts, device=self.device, dtype=real_dtype)
        batcher = SimpleBatcher(num, batch_size=max_batch_size or num, shuffle=False, rng=self.rng)
        for batch_idx in tqdm(batcher, disable=not verbose):
            spectra = self._vdf_fourier[batch_idx].to(self.device)
            operator = torch.exp(-1j * torch.einsum("na,amp->nmp", shifts[batch_idx], qvec))
            shifted_sum += (spectra * operator).sum(0)
            energy += spectra.abs().square().sum(0)

        # Upsampling interpolates rather than tiling the spectra as the bright-field classes
        # do. Tiling relies on the shifts being diverse enough to unfold the aliases: the
        # bright-field shifts fill a disk and leave under 1% of each alias, but the segment
        # shifts all lie on one ring and leave 15-45%, which prints the scan grid on the image.
        spectrum = _zero_pad_spectrum(shifted_sum / num, upsampling_factor)
        fine_gpts = tuple(n * upsampling_factor for n in self.scan_gpts)
        fine_sampling = tuple(s / upsampling_factor for s in self.scan_sampling)
        q = torch.hypot(*spatial_frequencies(fine_gpts, fine_sampling, device=self.device))
        if band_limit:
            cutoff = self._return_band_limit()
            q_lowpass = cutoff if q_lowpass is None else min(q_lowpass, cutoff)
        envelope = torch.ones_like(q)
        if q_lowpass:
            envelope = envelope / (1 + (q / q_lowpass) ** (2 * butterworth_order))
        if q_highpass:
            envelope = envelope * (1 - 1 / (1 + (q / q_highpass) ** (2 * butterworth_order)))

        image = torch.fft.ifft2(spectrum * envelope).real
        self._corrected_bf = image.to(torch.float32)
        self._obj_type = "pure_phase"
        self._num_slices = 1
        self._slice_depths = [0.0]

        # the shift-and-average model's residual: the spread of the shifted segment images
        mask = self._residual_mask(qxa, qya)
        total = (energy * mask).sum()
        explained = explained_energy(torch.full_like(energy, float(num)), shifted_sum, mask)
        self._residual = 1 - float(explained) / float(total) if total > 0 else float("nan")
        return self


def _zero_pad_spectrum(spectrum, factor: int):
    """A corner-centred spectrum on a grid ``factor`` times larger, scaled so the image keeps
    its values: band-limited interpolation of the image it transforms to."""
    if factor == 1:
        return spectrum
    rows, cols = spectrum.shape[-2:]
    padded = spectrum.new_zeros((*spectrum.shape[:-2], rows * factor, cols * factor))
    for row_out, row_in in (
        (slice(0, (rows + 1) // 2), slice(0, (rows + 1) // 2)),
        (slice(rows * factor - rows // 2, rows * factor), slice((rows + 1) // 2, rows)),
    ):
        for col_out, col_in in (
            (slice(0, (cols + 1) // 2), slice(0, (cols + 1) // 2)),
            (slice(cols * factor - cols // 2, cols * factor), slice((cols + 1) // 2, cols)),
        ):
            padded[..., row_out, col_out] = spectrum[..., row_in, col_in]
    return padded * factor**2


def bright_field_transfer(reconstruction, upsampling_factor: int = 1) -> np.ndarray:
    """Mean ``|Gamma-(q, k)|`` over the bright-field pixels, scaled to a maximum of one.

    The direct reconstruction's transfer, in the shape of its image; :func:`fuse_full_field`
    uses it to weight the bright- and dark-field images frequency by frequency.
    """
    ctx = reconstruction._reconstruction_context(
        None, None, upsampling_factor, None, "obf", None, None, 12, True, False, False, False
    )
    bf = ctx["bf"]
    probe = reconstruction._return_probe(ctx["aberration_coefs"])
    qxa, qya = ctx["qxa"], ctx["qya"]
    total = torch.zeros(qxa.shape, device=qxa.device, dtype=preferred_float_dtype(qxa.device))
    for batch_idx in SimpleBatcher(bf.num_bf, batch_size=256, shuffle=False):
        ind_i, ind_j = bf.bf_inds_i[batch_idx], bf.bf_inds_j[batch_idx]
        kx = ctx["kxa"][ind_i, ind_j].view(-1, 1, 1)
        ky = ctx["kya"][ind_i, ind_j].view(-1, 1, 1)
        t_minus, t_plus, _, _ = overlap_terms(
            probe, ctx["cmplx_probe_k"][ind_i, ind_j].view(-1, 1, 1), kx, ky, qxa, qya
        )
        total += (t_minus - t_plus).abs().sum(0)
    total /= bf.num_bf
    return to_numpy(total / total.max().clamp_min(torch.finfo(total.dtype).tiny))


def fuse_full_field(direct_image, tcdf_image, transfer) -> np.ndarray:
    """Combine a bright-field and a tcDF image, weighting each frequency by the BF transfer.

    ``F^-1[ T F(direct) + (1 - T) F(tcdf) ]`` with ``T`` the bright-field transfer from
    :func:`bright_field_transfer`, after both images are standardized to zero mean and unit
    variance. The bright-field reconstruction transfers poorly at low spatial frequencies,
    where the incoherent dark field is strong, so the two complement each other.

    The two channels have different contrast mechanisms: the bright field is linear in the
    phase, the dark field quadratic in the scattering potential. A linear Fourier weighting
    cannot reconcile them exactly, and Ma et al. note that a nonlinear scaling of the tcDF
    image would be needed for a quantitative combination; this is the simple linear blend.
    """
    direct_image = np.asarray(direct_image, dtype=np.float64)
    tcdf_image = np.asarray(tcdf_image, dtype=np.float64)
    transfer = np.asarray(transfer, dtype=np.float64)
    if not direct_image.shape == tcdf_image.shape == transfer.shape:
        raise ValueError(
            f"The images and the transfer must share a shape, got {direct_image.shape}, "
            f"{tcdf_image.shape} and {transfer.shape}; reconstruct both on the same canvas."
        )

    def standardize(image):
        image = image - image.mean()
        spread = image.std()
        return image / spread if spread > 0 else image

    fused = np.fft.fft2(standardize(direct_image)) * transfer + np.fft.fft2(
        standardize(tcdf_image)
    ) * (1 - transfer)
    return np.fft.ifft2(fused).real
