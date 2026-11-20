"""Bragg peaks of semicrystalline polymers, detected with a trained U-Net.

:class:`BraggPeaksPolymer` is a :class:`~quantem.diffraction.BraggVectors` whose
peaks come from a neural network rather than template matching. Polymer
reflections are diffuse arcs on a strong amorphous ring, recorded at low dose,
which template matching handles poorly. The network, trained on simulated
patterns, maps each pattern (resized to 256 x 256) to a peak-position map and a
peak-intensity map; peaks are the local maxima of the smoothed position map.

Everything after detection is shared with :class:`BraggVectors`: origins
(:meth:`measure_origins`), calibration (:class:`DiffractionCalibration`), plots
and saving. :meth:`calibrate_amorphous_ring` measures the elliptic distortion
from the amorphous ring, and :meth:`polar_peaks` returns the calibrated peaks as
(q, theta, intensity) for the orientation and ice analyses.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.vector import Vector
from quantem.core.ml.cnn import CNN2d
from quantem.core.utils.filter import gaussian_kernel_1d
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffraction.bragg_vectors import PEAK_FIELDS, BraggVectors
from quantem.diffraction.bragg_vectors_visualization import plot_detection
from quantem.diffraction.calibration import DiffractionCalibration, ellipse_from_axes
from quantem.diffraction.disk_detection import (
    _bilinear_batched,
    _detect_peaks_batched,
    _refine_parabolic_batched,
    _split_by_counts,
)
from quantem.diffraction.ellipticity import fit_elliptical_distortion

MODEL_SHAPE = (256, 256)
_INVERSE_ANGSTROM = {"a^-1", "1/a", "a-1", "1/angstrom", "angstrom^-1"}
_INVERSE_NM = {"nm^-1", "1/nm", "nm-1"}


def polymer_model() -> CNN2d:
    """The U-Net of the published polymer peak detector (load weights into it)."""
    return CNN2d(
        in_channels=1,
        out_channels=2,
        start_filters=32,
        num_layers=4,
        num_per_layer=3,
        final_activation="sigmoid",
    )


class BraggPeaksPolymer(BraggVectors):
    """Bragg peaks of a polymer 4D-STEM scan, detected with a trained network.

    Typical use::

        bp = BraggPeaksPolymer.from_dataset(dataset)
        bp.load_model("polymer_peaks.pth")
        bp.find_peaks_model(scan_mask=mask)
        bp.calibrate_amorphous_ring(fit_radii=(40, 90))
        peaks = bp.polar_peaks()  # (q, theta, intensity)

    Attributes
    ----------
    model : CNN2d or None
        The peak-detection network, set by :meth:`load_model`.
    scan_mask : np.ndarray or None
        ``(scan_row, scan_col)`` positions to analyze; the normalization is fit
        on these only. None analyzes every position.
    origins : np.ndarray or None
        ``(scan_row, scan_col, 2)`` beam positions in detector pixels.
    normalization : tuple of float or None
        Global percentiles of the min-max-scaled patterns, fit once per mask.
    """

    def __init__(self, dataset, device=None, _token=None):
        super(BraggPeaksPolymer, self).__init__(dataset, device=device, _token=_token)
        self.model: CNN2d | None = None
        self.scan_mask: np.ndarray | None = None
        self.origins: np.ndarray | None = None
        self.normalization: tuple[float, float] | None = None
        self._normalization_key: np.ndarray | None = None

    def load_model(self, path: str, model: CNN2d | None = None) -> "BraggPeaksPolymer":
        """Load network weights, by default into :func:`polymer_model`.

        Parameters
        ----------
        path : str
            Path to a state dict saved with ``torch.save(model.state_dict(), path)``.
        model : CNN2d, optional
            Network to load the weights into, for a retrained architecture.
        """
        model = polymer_model() if model is None else model
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model = model.to(self.device)
        return self

    def find_peaks_model(
        self,
        positions=None,
        scan_mask: np.ndarray | None = None,
        threshold: float = 0.5,
        sigma: float = 1.0,
        batch_size: int = 100,
        percentiles: tuple[float, float] = (1.0, 99.0),
        num_normalize: int = 1000,
        progressbar: bool = True,
    ) -> Vector:
        """Detect peaks with the network.

        The network runs in training mode, so its BatchNorm layers normalize each
        batch with the statistics of the experimental patterns rather than of the
        simulated training set; this detects far more reliably on real data. A
        pattern's peaks therefore depend on the batch it is run in. Patterns are
        batched in raster order over the scan mask, and ``positions`` are run
        inside the same batches, so a preview matches the full scan exactly.

        Parameters
        ----------
        positions : list of (row, col), optional
            Detect only here and return the peaks without storing them.
        scan_mask : np.ndarray, optional
            Boolean ``(scan_row, scan_col)`` region to analyze; stored on
            :attr:`scan_mask`. None keeps the stored mask (or the whole scan).
        threshold : float, default=0.5
            Minimum smoothed position-map value of a peak.
        sigma : float, default=1.0
            Gaussian smoothing of the position map, in model pixels.
        batch_size : int, default=100
            Patterns per network batch. Changing it changes the BatchNorm
            statistics and so, slightly, the peaks.
        percentiles : (float, float), default=(1, 99)
            Clipping percentiles of the input normalization.
        num_normalize : int, default=1000
            Patterns sampled to fit the normalization.
        progressbar : bool, default=True
            Show progress over batches.

        Returns
        -------
        Vector
            Peaks with fields ``(q_row, q_col, intensity)`` in detector pixels;
            shape ``(scan_row, scan_col)``, or ``(len(positions),)``.
        """
        if self.model is None:
            raise RuntimeError("Load the network weights first with load_model().")
        scan_r, scan_c = int(self.dataset.shape[0]), int(self.dataset.shape[1])
        if scan_mask is not None:
            self.scan_mask = np.asarray(scan_mask, dtype=bool)
        mask = np.ones((scan_r, scan_c), bool) if self.scan_mask is None else self.scan_mask
        if mask.shape != (scan_r, scan_c):
            raise ValueError(f"scan_mask shape {mask.shape} must be {(scan_r, scan_c)}.")
        order = np.argwhere(mask)
        if len(order) == 0:
            raise ValueError("scan_mask selects no positions.")
        key = np.concatenate([mask.ravel(), [*percentiles, num_normalize]]).astype(float)
        if self.normalization is None or not np.array_equal(key, self._normalization_key):
            self._fit_normalization(order, percentiles, num_normalize)
            self._normalization_key = key

        # (positions to run together, how many of them to keep)
        if positions is None:
            batches = [
                (order[i : i + batch_size], batch_size) for i in range(0, len(order), batch_size)
            ]
        else:
            positions = np.asarray(positions, dtype=int).reshape(-1, 2)
            index = np.full((scan_r, scan_c), -1)
            index[mask] = np.arange(len(order))
            starts, batches = set(), []
            for r, c in positions:
                i = index[r, c]
                if i < 0:  # outside the mask: run it with the first masked patterns
                    batches.append((np.vstack([[r, c], order[: batch_size - 1]]), 1))
                elif (start := i // batch_size * batch_size) not in starts:
                    starts.add(start)
                    batches.append((order[start : start + batch_size], batch_size))

        found = {}
        for batch, keep in tqdm(batches, desc="Detecting peaks", disable=not progressbar):
            dps = torch.as_tensor(
                np.asarray(self.dataset.array[batch[:, 0], batch[:, 1]]),
                dtype=torch.float32,
                device=self.device,
            )
            for (r, c), p in zip(batch[:keep], self._detect(dps, threshold, sigma)):
                found[(int(r), int(c))] = p

        if positions is not None:
            cells = [found[(int(r), int(c))] for r, c in positions]
            return Vector.from_data(cells, fields=PEAK_FIELDS, name="bragg_peaks_test")
        empty = np.zeros((0, 3))
        nested = [[found.get((r, c), empty) for c in range(scan_c)] for r in range(scan_r)]
        peaks = Vector.from_data(nested, fields=PEAK_FIELDS, name="bragg_peaks")
        peaks.metadata.update(self._scan_calibration())
        self.peaks = peaks
        self.metadata["detect"] = {
            "method": "model",
            "threshold": threshold,
            "sigma": sigma,
            "batch_size": batch_size,
            "percentiles": tuple(percentiles),
        }
        self.compute_bvm()
        return peaks

    def show_detection(
        self,
        positions=None,
        image=None,
        *,
        threshold: float = 0.5,
        sigma: float = 1.0,
        batch_size: int = 100,
        peak_radius: float = 4.0,
        returnfig: bool = False,
        **plot_kwargs,
    ):
        """Detect peaks at a few positions and draw them on the patterns.

        Parameters
        ----------
        positions : list of (row, col), optional
            Positions to show; defaults to four spread over the scan.
        image : np.ndarray or Dataset2d, optional
            Navigation image drawn on the left with the positions marked.
        threshold, sigma, batch_size
            As in :meth:`find_peaks_model`.
        peak_radius : float, default=4.0
            Radius of the peak markers in detector pixels.
        returnfig : bool, default=False
            Return ``(fig, ax)``.
        **plot_kwargs
            Passed to :func:`~quantem.diffraction.bragg_vectors_visualization.plot_detection`.
        """
        positions = self._sample_positions() if positions is None else positions
        found = self.find_peaks_model(
            positions, threshold=threshold, sigma=sigma, batch_size=batch_size, progressbar=False
        )
        image_arr = None if image is None else np.asarray(getattr(image, "array", image))
        fig, ax = plot_detection(
            image_arr,
            [np.asarray(self.dataset.array[r, c], dtype=float) for r, c in positions],
            [found[i].numpy().astype(np.float64) for i in range(len(positions))],
            positions,
            peak_radius=peak_radius,
            **plot_kwargs,
        )
        if returnfig:
            return fig, ax

    def calibrate_amorphous_ring(
        self,
        fit_radii: tuple[float, float],
        origins: np.ndarray | None = None,
        n_harmonics: int = 2,
        pixel_size: float | None = None,
        energy: float | None = None,
        mask: np.ndarray | None = None,
    ) -> DiffractionCalibration:
        """Calibrate from the amorphous ring of the origin-aligned mean pattern.

        Each pattern is shifted (to the nearest pixel) so its origin lands on the
        mean origin before averaging, since a descan smear would read as
        ellipticity. The ring is fit by
        :func:`~quantem.diffraction.ellipticity.fit_elliptical_distortion`, with
        ``n_harmonics`` angular harmonics in its amplitude so texture is not read
        as ellipticity. The scan rotation comes from the dataset metadata.

        Parameters
        ----------
        fit_radii : (float, float)
            Inner and outer radius, in detector pixels, of the annulus fit; it
            should hold the amorphous ring and exclude the central beam.
        origins : np.ndarray, optional
            ``(scan_row, scan_col, 2)`` beam positions; defaults to
            :attr:`origins`, else :meth:`measure_origins`. Stored on :attr:`origins`.
        n_harmonics : int, default=2
            Angular harmonics of the ring amplitude.
        pixel_size : float, optional
            Detector pixel size in 1/Angstrom; defaults to the dataset sampling.
        energy : float, optional
            Beam energy in eV, needed when the detector is calibrated in mrad.
        mask : np.ndarray, optional
            Detector pixels to leave out of the fit (True = ignore), such as a
            beam stop.

        Returns
        -------
        DiffractionCalibration
            Also stored on :attr:`calibration`.
        """
        if origins is not None:
            self.origins = np.asarray(origins, dtype=float)
        elif self.origins is None:
            self.origins = self.measure_origins()
        mean_dp, center = self._centered_mean(self.origins)
        fit = fit_elliptical_distortion(
            Dataset2d.from_array(mean_dp),
            center=center,
            fit_radii=fit_radii,
            mask=mask,
            n_harmonics=n_harmonics,
            write_metadata=False,
        )
        rotation, transpose, _ = self._scan_rotation(None, None)
        self.calibration = DiffractionCalibration(
            self._pixel_size(pixel_size, energy),
            ellipse=ellipse_from_axes(*fit["ellipse_params"]),
            rotation_ccw_deg=rotation,
            transpose=transpose,
            metadata={
                "method": "amorphous ring",
                "ellipse_params": fit["ellipse_params"],
                "ring_center": fit["center"],
                "mean_origin": center,
            },
        )
        return self.calibration

    def polar_peaks(self, calibration: DiffractionCalibration | None = None) -> Vector:
        """Calibrated peaks as ``(q, theta, intensity)``, one-to-one with :attr:`peaks`.

        The peaks are referred to :attr:`origins` (measured if unset) and passed
        through the calibration, which corrects the ellipse and rotates them
        into the scan frame. ``q`` is in 1/Angstrom and ``theta =
        atan2(q_row, q_col)`` in radians in ``[0, 2 pi)``, so a reflection and its
        Friedel partner are pi apart.

        Parameters
        ----------
        calibration : DiffractionCalibration, optional
            Defaults to :attr:`calibration`, else the dataset pixel size with no
            ellipse correction.
        """
        if self.peaks is None:
            raise RuntimeError("Run find_peaks_model() first.")
        if self.origins is None:
            self.origins = self.measure_origins()
        if calibration is None:
            calibration = self.calibration
        if calibration is None:
            rotation, transpose, _ = self._scan_rotation(None, None)
            calibration = DiffractionCalibration(
                self._pixel_size(), rotation_ccw_deg=rotation, transpose=transpose
            )
        centered = self.peaks.copy()
        flat = centered.numpy().astype(np.float64)
        counts = np.asarray(centered.row_counts(), dtype=int)
        flat[:, :2] -= np.repeat(self.origins.reshape(-1, 2), counts, axis=0)
        centered.set_flattened(flat)
        q = calibration.apply(centered).numpy().astype(np.float64)
        polar = np.column_stack(
            [np.hypot(q[:, 0], q[:, 1]), np.mod(np.arctan2(q[:, 0], q[:, 1]), 2 * np.pi), q[:, 2]]
        )
        out = _nest(
            polar,
            self.peaks,
            fields=["q", "theta", "intensity"],
            units=["A^-1", "rad", "counts"],
            name="polar_peaks",
        )
        out.metadata.update({k: v for k, v in self.peaks.metadata.items() if "scan" in k})
        return out

    def detect_ice(self, params=None, measure_widths: bool | None = None):
        """Flag crystalline-ice peaks; see :func:`~quantem.diffraction.polymer_ice.detect_ice`.

        Parameters
        ----------
        params : IceFlaggerParams, optional
            Detection parameters.
        measure_widths : bool, optional
            Measure each candidate's radial and annular width from the patterns,
            for the sharpness gate. Defaults to whether ``params`` sets a width
            ceiling.
        """
        from quantem.diffraction import polymer_ice

        params = polymer_ice.IceFlaggerParams() if params is None else params
        peaks = self.polar_peaks()
        if measure_widths is None:
            measure_widths = (
                params.max_width_q_invA is not None or params.max_width_theta_deg is not None
            )
        if measure_widths:
            peaks = self._add_peak_widths(peaks, params)
        return polymer_ice.detect_ice(peaks, params=params, scan_mask=self.scan_mask)

    # ---- helpers ----

    def _add_peak_widths(self, peaks: Vector, params) -> Vector:
        """``peaks`` with ``width_q`` and ``width_theta`` measured on the patterns."""
        from quantem.diffraction.polymer_ice import measure_peak_widths

        polar = peaks.numpy().astype(np.float64)
        widths = np.full((polar.shape[0], 2), np.nan)
        counts = np.asarray(peaks.row_counts(), dtype=int)
        starts = np.concatenate([[0], np.cumsum(counts)])
        near = np.abs(polar[:, 0] - params.q_target_invA) <= params.dq_invA
        for cell in np.flatnonzero(counts):
            local = np.flatnonzero(near[starts[cell] : starts[cell + 1]])
            if len(local) == 0:
                continue
            r, c = divmod(int(cell), peaks.shape[1])
            widths[starts[cell] + local] = np.column_stack(
                measure_peak_widths(
                    np.asarray(self.dataset.array[r, c], dtype=float),
                    self.peaks[r, c].numpy().astype(np.float64)[local, :2],
                    self.origins[r, c],
                    self._pixel_size(),
                    params,
                )
            )
        return _nest(
            np.column_stack([polar, widths]),
            peaks,
            fields=[*peaks.fields, "width_q", "width_theta"],
            units=[*peaks.units, "A^-1", "rad"],
            name=peaks.name,
        )

    def _pixel_size(self, pixel_size: float | None = None, energy: float | None = None) -> float:
        """Detector pixel size in 1/Angstrom, from the argument or the dataset."""
        if pixel_size is not None:
            return float(pixel_size)
        if self.calibration is not None:
            return float(self.calibration.pixel_size)
        sampling = float(np.atleast_1d(self.dataset.sampling)[2])
        unit = str(self.dataset.units[2]).strip().lower().replace("å", "a").replace(" ", "")
        if unit in _INVERSE_ANGSTROM:
            return sampling
        if unit in _INVERSE_NM:
            return sampling / 10.0
        if unit == "mrad":
            if energy is None:
                raise ValueError("The detector is calibrated in mrad: pass energy (eV).")
            return sampling * 1e-3 / electron_wavelength_angstrom(energy)
        raise ValueError(
            f"Cannot convert detector units {self.dataset.units[2]!r} to 1/Angstrom; "
            "pass pixel_size."
        )

    def _centered_mean(self, origins: np.ndarray, chunk: int = 512):
        """Mean pattern after shifting every origin onto the mean origin."""
        mask = np.ones(origins.shape[:2], bool) if self.scan_mask is None else self.scan_mask
        H, W = int(self.dataset.shape[-2]), int(self.dataset.shape[-1])
        ref = origins[mask].mean(axis=0)
        shifts = np.round(origins[mask] - ref).astype(int)
        pad = int(np.abs(shifts).max())
        total = np.zeros((H + 2 * pad, W + 2 * pad))
        weight = np.zeros_like(total)
        positions = np.argwhere(mask)
        unique, inverse = np.unique(shifts, axis=0, return_inverse=True)
        for k, (sr, sc) in enumerate(unique):
            sel = positions[inverse.ravel() == k]
            window = (slice(pad - sr, pad - sr + H), slice(pad - sc, pad - sc + W))
            for i in range(0, len(sel), chunk):
                block = np.asarray(
                    self.dataset.array[sel[i : i + chunk, 0], sel[i : i + chunk, 1]]
                )
                total[window] += block.sum(axis=0, dtype=np.float64)
            weight[window] += len(sel)
        mean = total / np.maximum(weight, 1)
        return mean[pad : pad + H, pad : pad + W], (float(ref[0]), float(ref[1]))

    def _prepare(self, dps: torch.Tensor) -> torch.Tensor:
        """Resize to the model shape and min-max scale each pattern to [0, 1]."""
        x = F.interpolate(dps[:, None], size=MODEL_SHAPE, mode="bilinear", align_corners=False)
        low = x.amin(dim=(-2, -1), keepdim=True)
        span = x.amax(dim=(-2, -1), keepdim=True) - low
        return torch.where(span > 0, (x - low) / span, torch.zeros_like(x))

    def _fit_normalization(self, order: np.ndarray, percentiles, num: int, seed: int = 0):
        """Global clipping percentiles of the min-max-scaled patterns, from a sample."""
        rng = np.random.default_rng(seed)
        sample = np.sort(rng.choice(len(order), size=min(num, len(order)), replace=False))
        values = []
        for i in range(0, len(sample), 256):
            idx = order[sample[i : i + 256]]
            dps = torch.as_tensor(
                np.asarray(self.dataset.array[idx[:, 0], idx[:, 1]]),
                dtype=torch.float32,
                device=self.device,
            )
            values.append(self._prepare(dps).cpu().numpy().ravel())
        low, high = np.percentile(np.concatenate(values), percentiles)
        self.normalization = (float(low), float(high))

    def _detect(self, dps: torch.Tensor, threshold: float, sigma: float) -> list[np.ndarray]:
        """Peaks ``[q_row, q_col, intensity]`` (detector pixels) of a batch of patterns."""
        low, high = self.normalization
        x = (self._prepare(dps).clamp(low, high) - low) / (high - low + 1e-8)
        self.model.train()
        with torch.no_grad():
            out = self.model(x)
        position, intensity = out[:, 0], out[:, 1]
        if sigma > 0:
            kernel = gaussian_kernel_1d(sigma, num_sigmas=4.0).to(position)
            p = len(kernel) // 2
            x = position[:, None]  # mirror padding that repeats the edge, as scipy's "reflect"
            x = torch.cat([x[..., :p, :].flip(-2), x, x[..., -p:, :].flip(-2)], dim=-2)
            x = torch.cat([x[..., :p].flip(-1), x, x[..., -p:].flip(-1)], dim=-1)
            x = F.conv2d(F.conv2d(x, kernel.view(1, 1, -1, 1)), kernel.view(1, 1, 1, -1))
            position = x[:, 0]
        peaks, bidx, counts = _detect_peaks_batched(position, 1, threshold, 0.0, None)
        peaks = _refine_parabolic_batched(position, peaks, bidx)
        peaks[:, 2] = _bilinear_batched(intensity, bidx, peaks[:, 0], peaks[:, 1])
        # model pixel centers to detector pixels (align_corners=False sampling)
        scale = peaks.new_tensor([dps.shape[-2] / MODEL_SHAPE[0], dps.shape[-1] / MODEL_SHAPE[1]])
        peaks[:, :2] = (peaks[:, :2] + 0.5) * scale - 0.5
        return _split_by_counts(peaks, counts)


def _nest(flat: np.ndarray, like: Vector, **kwargs) -> Vector:
    """A Vector with ``like``'s scan shape and per-position counts holding ``flat`` rows."""
    cells = np.split(flat, np.cumsum(like.row_counts())[:-1])
    scan_c = like.shape[1]
    return Vector.from_data(
        [cells[r * scan_c : (r + 1) * scan_c] for r in range(like.shape[0])], **kwargs
    )
