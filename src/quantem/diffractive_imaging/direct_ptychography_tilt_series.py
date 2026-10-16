from dataclasses import replace
from typing import TYPE_CHECKING, Sequence

import numpy as np

from quantem.core import config
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import to_numpy
from quantem.diffractive_imaging.direct_ptycho_utils import EIGENVALUE_FLOOR, slice_depths
from quantem.diffractive_imaging.direct_ptychography import DirectPtychography
from quantem.diffractive_imaging.direct_ptychography_base import DirectPtychographyBase

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class DirectPtychographyTiltSeries(AutoSerialize):
    """Several acquisitions of one specimen at small tilts, solved jointly for its slices.

    Tilting the specimen by ``theta`` leaves the probe as it is and displaces a layer at depth
    ``z`` sideways by ``z theta``, which multiplies that slice's columns at scan frequency ``q``
    by ``exp(-2 pi i z theta . q)``: it moves every column's ``q_z`` by ``theta . q``. Each
    acquisition's normal equations take those phases and the acquisitions are summed, so slices
    that one acquisition can barely tell apart are told apart by how they move between tilts.
    With two slices 7 A apart at 80 kV and 20 mrad and the dose shared equally between the
    tilts, five tilts of +-5 mrad were worth about 1.4 times the dose of one acquisition and nine
    about 1.6 times.

    Each acquisition is an ordinary :class:`DirectPtychography`, with its own diffraction
    origin, bright-field mask, aberrations and rotation, which can be fitted on their own before
    the joint solve. They must share the scan grid. Build with
    :meth:`DirectPtychography.from_tilt_series` or from reconstructions already made.
    ``save`` writes the acquisitions, tilts, drifts and the last joint reconstruction, and
    :func:`~quantem.core.io.serialize.load` restores them.

    Parameters
    ----------
    reconstructions : sequence of DirectPtychography
        One per acquisition, on the same scan grid.
    tilt_angles : array-like
        ``(T, 2)`` tilts in mrad, in the scan frame: a layer ``z`` Angstrom downstream of the
        reference plane appears displaced by ``z * theta * 1e-3`` Angstrom along the scan's row
        and column axes. A specimen tilt about an axis displaces the layers perpendicular to it.
    shifts : array-like, optional
        ``(T, 2)`` lateral drift of each acquisition in Angstrom, the displacement of its
        content relative to the first's. :meth:`register` measures it.
    """

    def __init__(self, reconstructions: Sequence[DirectPtychography], tilt_angles, shifts=None):
        reconstructions = list(reconstructions)
        if not reconstructions:
            raise ValueError("A tilt series needs at least one reconstruction.")
        if not all(isinstance(r, DirectPtychography) for r in reconstructions):
            raise TypeError(
                "Every acquisition must be a DirectPtychography: the joint solve sums their "
                "per-frequency normal equations on one scan grid."
            )
        first = reconstructions[0]
        for r in reconstructions[1:]:
            if tuple(r.scan_gpts) != tuple(first.scan_gpts) or not np.allclose(
                r.scan_sampling, first.scan_sampling
            ):
                raise ValueError(
                    "Every acquisition must share the scan grid, got "
                    f"{tuple(r.scan_gpts)} at {tuple(r.scan_sampling)} against "
                    f"{tuple(first.scan_gpts)} at {tuple(first.scan_sampling)}."
                )
            if not np.isclose(r.wavelength, first.wavelength):
                raise ValueError("Every acquisition must share the wavelength.")
        self.reconstructions = reconstructions
        self.tilt_angles = self._per_acquisition(tilt_angles, "tilt_angles")
        self.shifts = (
            np.zeros((len(reconstructions), 2))
            if shifts is None
            else self._per_acquisition(shifts, "shifts")
        )
        self._corrected_bf = None
        self._corrected_amplitude = None
        self._residual = None
        self._noise_variance = None

    def _per_acquisition(self, values, name):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(self.reconstructions), 2):
            raise ValueError(
                f"`{name}` must be ({len(self.reconstructions)}, 2), one row per acquisition, "
                f"got shape {values.shape}."
            )
        return values

    def __len__(self):
        return len(self.reconstructions)

    def __getitem__(self, index):
        return self.reconstructions[index]

    def __iter__(self):
        return iter(self.reconstructions)

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def register(
        self,
        max_shift: float | None = None,
        reference: int = 0,
        upsample_factor: int = 16,
        **reconstruct_kwargs,
    ):
        """Measure each acquisition's lateral drift from its single-plane reconstruction.

        The single-plane image is the projection, which a small tilt barely changes and which
        every acquisition determines well, so its shift against the ``reference`` acquisition's
        is the drift. The cross-correlation peak is found within ``max_shift`` Angstrom and
        refined to ``1 / upsample_factor`` of a pixel. The drift is stored in :attr:`shifts`,
        relative to the first acquisition, and the joint solve gives each acquisition's columns
        its phase. ``reconstruct_kwargs`` go to each acquisition's
        :meth:`DirectPtychography.reconstruct`, OBF by default.

        A crystal's projection is periodic, so its cross-correlation peaks repeat with the
        lattice and noise can pick a neighbour: on two twisted square lattices with a 3.2 A
        period and a 0.7 A drift, the unrestricted search landed two periods away. For a
        crystal, keep ``max_shift`` below half the lattice period, or set :attr:`shifts` from
        the stage or scan record.
        """
        kwargs = dict(deconvolution_kernel="obf", verbose=False)
        kwargs.update(reconstruct_kwargs)
        images = [r.reconstruct(**kwargs).corrected_bf.detach() for r in self.reconstructions]
        sampling = np.asarray(self.reconstructions[0]._obj_sampling, dtype=np.float64)
        limit = None if max_shift is None else np.asarray(max_shift) / sampling
        shifts = np.stack(
            [
                -_peak_shift(images[reference], image, limit, upsample_factor) * sampling
                for image in images
            ]
        )
        self.shifts = shifts - shifts[0]
        return self

    # ------------------------------------------------------------------
    # reconstruction
    # ------------------------------------------------------------------

    _KERNELS = ("obf", "wiener")

    def reconstruct(
        self,
        deconvolution_kernel="obf",
        obj_type="pure_phase",
        num_slices=2,
        slice_thicknesses=None,
        upsampling_factor=None,
        ridge=None,
        prior=None,
        noise_variance=None,
        max_batch_size=None,
        q_highpass=None,
        q_lowpass=None,
        butterworth_order=12,
        band_limit=True,
        verbose=None,
        use_initial_state=False,
        slice_mode="joint",
    ):
        """Solve every acquisition's normal equations together for ``num_slices`` slices.

        The arguments mean what they do for :meth:`DirectPtychography.reconstruct`. Only the
        kernels that solve the normal equations, ``"obf"`` and ``"wiener"``, combine
        acquisitions; ``prior="ard"`` works with ``"obf"``. Slice thicknesses default to those
        the first acquisition's hyperparameter state holds. The results are stored here as they
        are on a single reconstruction: :attr:`corrected_bf`, :attr:`corrected_amplitude`,
        :attr:`obj`, :meth:`residual_loss` and :attr:`noise_variance`. ``slice_mode`` is always
        ``"joint"``, since the tilts tell slices apart only when they are fitted together.
        """
        if self.reconstructions[0]._normalize_slice_mode(slice_mode) != "joint":
            raise ValueError(
                "A tilt series is solved jointly: the tilts tell slices apart only when they "
                "are fitted together. Section each acquisition on its own with "
                "`series[n].reconstruct(slice_mode='sectioned')`."
            )
        first = self.reconstructions[0]
        kernel = first._normalize_kernel_name(deconvolution_kernel)
        if kernel not in self._KERNELS:
            raise ValueError(
                f"A tilt series combines acquisitions through the per-frequency normal "
                f"equations, which only {self._KERNELS} solve; got {kernel!r}."
            )
        first._validate_reconstruction_mode(kernel, obj_type, num_slices)
        first._validate_prior(kernel, prior, noise_variance)
        if verbose is None:
            verbose = first.verbose
        if ridge is None:
            ridge = EIGENVALUE_FLOOR
        if num_slices > 1:
            slice_thicknesses = first.hyperparameter_state.current_slice_thicknesses(
                slice_thicknesses
            )

        with torch.no_grad():
            depths = slice_depths(num_slices, slice_thicknesses, device=first.device)
            total, ctx, num_bf, bf_weights = None, None, 0, 0.0
            for r, tilt, shift in zip(self.reconstructions, self.tilt_angles, self.shifts):
                context = r._reconstruction_context(
                    None,
                    None,
                    upsampling_factor,
                    None,
                    kernel,
                    q_highpass,
                    q_lowpass,
                    butterworth_order,
                    band_limit,
                    True,
                    verbose,
                    use_initial_state,
                )
                phases = self._slice_phases(context, depths, tilt, shift)
                sums = r._accumulate_kernel_sums(
                    context, depths, obj_type, max_batch_size, verbose, slice_phases=phases
                )
                if total is None:
                    total, ctx = sums, context
                else:
                    if sums["gram"].shape != total["gram"].shape:
                        raise ValueError("Every acquisition must reconstruct on the same grid.")
                    total = {key: total[key] + sums[key] for key in total}
                num_bf += context["bf"].num_bf
                bf_weights = bf_weights + context["bf_weights"]

            # the noise and the gains count every acquisition's bright-field pixels
            ctx = dict(ctx, bf=replace(ctx["bf"], num_bf=num_bf), bf_weights=bf_weights)
            phase_q, amplitude_q = DirectPtychographyBase._fourier_channels(
                self, ctx, total, num_slices, obj_type, ridge, prior, noise_variance
            )
            env = ctx["butterworth_env"]
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
            self._slice_depths = [float(z) for z in depths.cpu()]
            self._residual = float(first._residual_from_sums(ctx, total))
        return self

    @staticmethod
    def _slice_phases(ctx, depths, tilt_mrad, shift):
        """``(N, H, W)``: ``exp(-2 pi i q . (z_n theta + shift))`` for each slice."""
        qxa, qya = ctx["qxa"], ctx["qya"]
        theta = torch.as_tensor(np.asarray(tilt_mrad) * 1e-3, dtype=torch.float64)
        displacement = depths.cpu().to(torch.float64)[:, None] * theta[None, :]
        displacement = displacement + torch.as_tensor(shift, dtype=torch.float64)[None, :]
        argument = displacement[:, 0, None, None] * qxa.cpu().to(torch.float64)[None]
        argument = argument + displacement[:, 1, None, None] * qya.cpu().to(torch.float64)[None]
        # narrowed before the move: MPS has no complex128, and the columns are complex64
        return torch.exp(-2j * np.pi * argument).to(torch.complex64).to(qxa.device)

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------

    @property
    def corrected_bf(self):
        """The phase of each slice, ``(num_slices, Ny, Nx)``, or ``(Ny, Nx)`` for one."""
        return self._corrected_bf

    @property
    def corrected_amplitude(self):
        """The log-amplitude, shaped like :attr:`corrected_bf`, for ``obj_type="complex"``."""
        return self._corrected_amplitude

    @property
    def obj(self):
        """``exp(a + i phi)`` for a complex object, the phase for a pure-phase one."""
        if self._corrected_bf is None:
            return None
        phase = to_numpy(self._corrected_bf)
        if self._corrected_amplitude is None:
            return phase
        return np.exp(to_numpy(self._corrected_amplitude) + 1j * phase).astype(np.complex64)

    @property
    def slice_depths(self):
        """Depth of each slice in Angstrom, downstream positive, centred on the C10 plane."""
        depths = getattr(self, "_slice_depths", None)
        return None if depths is None else np.asarray(depths)

    @property
    def noise_variance(self):
        """The noise variance the last ``"wiener"`` or ``prior="ard"`` solve used."""
        return self._noise_variance

    def residual_loss(self):
        """The fraction of all acquisitions' data the joint model leaves unexplained."""
        return self._residual


def _peak_shift(reference, image, limit=None, upsample_factor=16):
    """``(2,)`` displacement of ``reference`` relative to ``image`` in pixels, from the peak of
    their cross-correlation within ``limit`` pixels, refined on a grid ``1 / upsample_factor``
    of a pixel fine by evaluating the correlation there directly."""
    reference = reference.cpu().to(torch.float64)
    image = image.cpu().to(torch.float64)
    product = (
        torch.fft.fft2(reference - reference.mean()) * torch.fft.fft2(image - image.mean()).conj()
    )
    correlation = torch.fft.ifft2(product).real
    rows, cols = correlation.shape
    signed_rows = torch.fft.fftfreq(rows, dtype=torch.float64) * rows
    signed_cols = torch.fft.fftfreq(cols, dtype=torch.float64) * cols
    if limit is not None:
        inside = torch.hypot(signed_rows[:, None], signed_cols[None, :]) <= float(np.max(limit))
        correlation = torch.where(inside, correlation, torch.full_like(correlation, -np.inf))
    peak = int(torch.argmax(correlation))
    row0, col0 = signed_rows[peak // cols], signed_cols[peak % cols]
    offsets = (
        torch.arange(-upsample_factor, upsample_factor + 1, dtype=torch.float64) / upsample_factor
    )
    fine_rows, fine_cols = row0 + offsets, col0 + offsets
    row_kernel = torch.exp(
        2j * np.pi * torch.fft.fftfreq(rows, dtype=torch.float64)[:, None] * fine_rows[None]
    )
    col_kernel = torch.exp(
        2j * np.pi * torch.fft.fftfreq(cols, dtype=torch.float64)[:, None] * fine_cols[None]
    )
    fine = (row_kernel.T @ product @ col_kernel).real
    best = int(torch.argmax(fine))
    return np.array(
        [float(fine_rows[best // fine.shape[1]]), float(fine_cols[best % fine.shape[1]])]
    )
