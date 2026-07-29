from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence, Union

import numpy as np
import torch
from numpy.typing import NDArray

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.bragg_vectors_visualization import (
    plot_basis_vectors,
    plot_bvm,
    plot_detection,
    plot_diffraction_grid,
    plot_lattice_fit,
    plot_reference_lattice,
    plot_template,
)
from quantem.diffraction.disk_detection import (
    cross_correlation,
    detect_disks_batch,
    estimate_central_beam,
    make_template,
    probe_centroid,
    synthetic_probe,
    template_fourier,
)
from quantem.diffraction.strain import StrainMap

PEAK_FIELDS = ("q_row", "q_col", "intensity")


class BraggVectors(AutoSerialize):
    """Correlation-based Bragg disk detection and lattice fitting for 4D-STEM.

    Workflow (each step writes state consumed by the next):

    1. ``make_template_*`` – build a cross-correlation template, either from a
       synthetic soft disk (:meth:`make_template_synthetic`), by averaging data
       over an ROI (:meth:`make_template_from_data`), or from an explicit probe
       image (:meth:`make_template_from_probe`).
    2. :meth:`detect_disks` – template-match every scan position; detected peaks
       are stored in :attr:`peaks` (a :class:`Vector` of ``[q_row, q_col,
       intensity]``, numpy-backed) and accumulated into the Bragg vector map
       :attr:`bvm`.
    3. :meth:`choose_basis_vectors` – pick the lattice basis ``(origin, g1, g2)``
       from the BVM, automatically or by hand; also stores the numbered candidate
       peaks.
    4. :meth:`index_peaks` – quick, lightweight: index just the picked candidate
       peaks into a reference lattice (``reference_ab``/``reference_qpos``).
    5. :meth:`fit_lattice` – the heavy step: at every scan position, match the
       detections to the reference within ``max_peak_shift``, intensity-weighted
       least-squares fit the lattice vectors into ``u_array``/``v_array`` of shape
       ``(scan_row, scan_col, 2)``, and compute the per-position ``mask_weight``.
    6. :meth:`calculate_strain_map` – hand the lattice vectors (and
       ``mask_weight``) to a :class:`~quantem.diffraction.strain.StrainMap`.

    Detection runs in torch (CPU now, CUDA later); the ragged peak table is held
    in a numpy-backed :class:`Vector`. The detector→scan rotation is read from
    the parent dataset metadata (``q_to_r_rotation_ccw_deg`` + ``q_transpose``),
    the single source of truth shared with the DPC/CoM workflow.

    Use :meth:`from_dataset` to construct an instance.

    Parameters
    ----------
    dataset : Dataset4dstem
        The 4D-STEM dataset to analyze.
    device : str, default="cpu"
        Torch device used for detection (e.g. ``"cpu"`` or ``"cuda"``).
    """

    _token = object()

    # nanobeam lattice vectors are measured in reciprocal space -> sign flip
    real_space: bool = False

    def __init__(
        self,
        dataset: Dataset4dstem,
        device: str = "cpu",
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use BraggVectors.from_dataset() to instantiate this class.")
        super(BraggVectors, self).__init__()
        self.dataset = dataset
        self.device = device

        self.peaks: Vector | None = None
        self.bvm: Dataset2d | None = None

        self.origin: np.ndarray | None = None
        self.g1: np.ndarray | None = None
        self.g2: np.ndarray | None = None

        # candidate peaks picked from the BVM in choose_basis_vectors(), and the
        # reference lattice (those candidates indexed) set by index_peaks().
        self.candidates_rc: np.ndarray | None = None
        self.candidates_intensity: np.ndarray | None = None
        self.reference_ab: np.ndarray | None = None
        self.reference_qpos: np.ndarray | None = None
        self.reference_intensity: np.ndarray | None = None

        self.u_array: np.ndarray | None = None
        self.v_array: np.ndarray | None = None
        # per-position diagnostics from fit_lattice()
        self.mask_weight: np.ndarray | None = None
        self.fit_error: np.ndarray | None = None

        self._template: torch.Tensor | None = None
        self._template_ft: torch.Tensor | None = None
        self.metadata: dict[str, Any] = {}

    @classmethod
    def from_dataset(
        cls, dataset: Dataset4dstem, *, device: str = "cpu", name: str | None = None
    ) -> "BraggVectors":
        """Create a BraggVectors workflow bound to a 4D-STEM dataset.

        Parameters
        ----------
        dataset : Dataset4dstem
            The 4D-STEM dataset to analyze.
        device : str, default="cpu"
            Torch device used for detection (e.g. ``"cpu"`` or ``"cuda"``).
        name : str, optional
            If given, sets ``dataset.name``.

        Returns
        -------
        BraggVectors
            A new workflow instance bound to ``dataset``.
        """
        if not isinstance(dataset, Dataset4dstem):
            raise TypeError("BraggVectors.from_dataset expects a Dataset4dstem instance.")
        if name is not None:
            dataset.name = name
        return cls(dataset=dataset, device=device, _token=cls._token)

    def save(
        self,
        path: str | Path,
        mode: Literal["w", "o"] = "w",
        store: Literal["auto", "zip", "dir"] = "auto",
        skip: Union[str, type, Sequence[Union[str, type]]] = (),
        compression_level: int | None = 4,
        *,
        include_dataset: bool = False,
    ) -> None:
        """Save the workflow to disk, excluding the raw 4D-STEM dataset by default.

        Overrides :meth:`~quantem.core.io.serialize.AutoSerialize.save` to drop
        :attr:`dataset` — the raw 4D-STEM cube, which dominates the file size — from
        serialization by default. The detected :attr:`peaks`, lattice fit
        (:attr:`u_array`/:attr:`v_array`), Bragg vector map and all diagnostics are
        kept, so the file holds the *results* of the workflow (orders of magnitude
        smaller than the data) rather than the data itself.

        ``"dataset"`` is recorded in the file's skip metadata, so a reloaded workflow
        simply has no ``dataset`` attribute. Re-attach one (``bv.dataset = ds``) before
        calling methods that read the raw cube — :meth:`detect_disks`,
        :meth:`correlation_map`, :meth:`make_template_from_data`,
        :meth:`calculate_strain_map`, etc. Pass ``include_dataset=True`` to keep the
        dataset in the file instead.

        Parameters
        ----------
        path : str or Path
            Target file path. Use a ``.zip`` extension for zip format, otherwise a
            directory is written.
        mode : {'w', 'o'}, default='w'
            ``'w'`` writes only if the path does not exist; ``'o'`` overwrites.
        store : {'auto', 'zip', 'dir'}, default='auto'
            Storage format; ``'auto'`` infers from the file extension.
        skip : str, type, or sequence of (str or type), default=()
            Additional attribute names/types to skip during serialization, merged with
            the default ``dataset`` exclusion.
        compression_level : int or None, default=4
            Zstandard/Blosc compression level (0–9); ``0`` disables compression.
        include_dataset : bool, default=False
            If ``True``, keep the raw 4D-STEM :attr:`dataset` in the file (large). The
            default ``False`` excludes it.
        """
        if isinstance(skip, (str, type)):
            skip = [skip]
        else:
            skip = list(skip)
        if not include_dataset and "dataset" not in skip:
            skip.append("dataset")
        # Explicit (two-arg) super() rather than the bare super(): the zero-arg form
        # needs a compiler-created __class__ closure cell that is absent when this
        # method's source is re-exec'd from a string (Jupyter autoreload), which
        # raises "super(): __class__ cell not found". The explicit form is immune.
        super(BraggVectors, self).save(
            path,
            mode=mode,
            store=store,
            skip=skip,
            compression_level=compression_level,
        )

    # ---- main methods ----

    def make_template_synthetic(
        self,
        radius: float | None = None,
        edge: float = 1.0,
        center: tuple[float, float] | None = None,
        subtract_mean: bool = True,
    ) -> "BraggVectors":
        """Build the template from a synthetic soft-edged disk.

        Parameters
        ----------
        radius : float, optional
            Disk radius in pixels. Defaults to a rough estimate from the mean
            diffraction pattern
            (:func:`~quantem.diffraction.disk_detection.estimate_central_beam`);
            pass it explicitly when several disks share comparable intensity.
        edge : float, default=1.0
            Width in pixels of the ``tanh`` edge falloff.
        center : tuple of float, optional
            ``(row, col)`` disk center; defaults to the detector center
            ``(H // 2, W // 2)``.
        subtract_mean : bool, default=True
            If ``True``, make the template zero-sum — a band-pass kernel that
            suppresses uniform background in the correlation.

        Returns
        -------
        BraggVectors
            ``self``, for method chaining.
        """
        H, W = int(self.dataset.shape[-2]), int(self.dataset.shape[-1])
        if radius is None:
            dp_mean = torch.as_tensor(
                np.asarray(self.dataset.dp_mean.array), dtype=torch.float, device=self.device
            )
            _, radius = estimate_central_beam(dp_mean)
        if center is None:
            center = (H // 2, W // 2)
        probe = synthetic_probe((H, W), float(radius), edge=edge, center=center)
        self._set_template(probe, center=center, subtract_mean=subtract_mean)
        self.metadata["template"] = {
            "kind": "synthetic",
            "radius": float(radius),
            "edge": float(edge),
            "center": (float(center[0]), float(center[1])),
        }
        return self

    def make_template_from_data(
        self,
        roi: NDArray | None = None,
        subtract_mean: bool = True,
        center: tuple[float, float] | None = None,
    ) -> "BraggVectors":
        """Build the template by averaging diffraction patterns from the data.

        Parameters
        ----------
        roi : np.ndarray, optional
            ``(scan_row, scan_col)`` mask selecting scan positions to average —
            ideally a vacuum / single-disk region so the unscattered probe is
            isolated. ``None`` (default) averages the whole scan (the mean
            diffraction pattern).
        subtract_mean : bool, default=True
            If ``True``, make the template zero-sum — a band-pass kernel that
            suppresses uniform background in the correlation.
        center : tuple of float, optional
            ``(row, col)`` probe center rolled to the origin; defaults to the
            probe's intensity centroid.

        Returns
        -------
        BraggVectors
            ``self``, for method chaining.
        """
        data = torch.as_tensor(
            np.asarray(self.dataset.array), dtype=torch.float, device=self.device
        )
        if roi is None:
            probe = data.mean(dim=(0, 1))
        else:
            m = torch.as_tensor(np.asarray(roi) > 0, device=self.device)
            if not bool(m.any()):
                raise ValueError("roi selects no scan positions.")
            probe = data[m].mean(dim=0)
        if center is None:
            center = probe_centroid(probe)
        self._set_template(probe, center=center, subtract_mean=subtract_mean)
        self.metadata["template"] = {
            "kind": "from_data",
            "roi": roi is not None,
            "center": (float(center[0]), float(center[1])),
        }
        return self

    def make_template_from_probe(
        self,
        probe: NDArray | torch.Tensor,
        center: tuple[float, float] | None = None,
        subtract_mean: bool = True,
    ) -> "BraggVectors":
        """Build the template from an explicit probe image (e.g. a measured vacuum probe).

        Parameters
        ----------
        probe : np.ndarray or torch.Tensor
            Probe image; must match the diffraction-pattern shape.
        center : tuple of float, optional
            ``(row, col)`` probe center rolled to the origin; defaults to the
            probe's intensity centroid.
        subtract_mean : bool, default=True
            If ``True``, make the template zero-sum — a band-pass kernel that
            suppresses uniform background in the correlation.

        Returns
        -------
        BraggVectors
            ``self``, for method chaining.
        """
        probe_t = torch.as_tensor(probe, dtype=torch.float, device=self.device)
        if center is None and tuple(probe_t.shape) == tuple(self.dataset.shape[-2:]):
            center = probe_centroid(probe_t)
        self._set_template(probe_t, center=center, subtract_mean=subtract_mean)
        self.metadata["template"] = {
            "kind": "from_probe",
            "center": (float(center[0]), float(center[1])),
        }
        return self

    @property
    def template(self) -> np.ndarray | None:
        """The correlation template, fftshifted to the image center for display (numpy).

        The ``make_template_*`` methods store the template corner-shifted (center
        at the ``[0, 0]`` FFT origin) so correlation peaks land at absolute disk
        positions; this property shifts it back to the center for plotting.

        Returns
        -------
        np.ndarray or None
            ``(H, W)`` center-shifted template, or ``None`` if no template has been
            built yet.
        """
        if self._template is None:
            return None
        return torch.fft.fftshift(self._template).detach().cpu().numpy()

    def correlation_map(
        self, row: int, col: int, background_sigma: float | str | None = "auto"
    ) -> np.ndarray:
        """Cross-correlation map of one diffraction pattern with the template (numpy).

        Peaks in the returned map sit at absolute disk positions (no fftshift
        needed), matching what :meth:`detect_disks` searches.

        Parameters
        ----------
        row : int
            Scan row of the diffraction pattern to correlate.
        col : int
            Scan column of the diffraction pattern to correlate.

        Returns
        -------
        np.ndarray
            ``(H, W)`` real-space correlation map.
        """
        if self._template_ft is None:
            raise ValueError("Run a make_template_* method before correlation_map().")
        dp = torch.as_tensor(
            np.asarray(self.dataset.array[row, col]), dtype=torch.float, device=self.device
        )
        corr, _ = cross_correlation(
            dp, self._template_ft, self._resolve_background_sigma(background_sigma)
        )
        return corr.detach().cpu().numpy()

    def detect_disks(
        self,
        *,
        positions: list[tuple[int, int]] | None = None,
        min_abs_intensity: float = 0.0,
        min_spacing: float = 0.0,
        edge_boundary: int = 1,
        subpixel: str = "upsample",
        upsample_factor: int = 16,
        max_num_peaks: int = 1000,
        background_sigma: float | str | None = "auto",
        batch_size: int | None = None,
        progressbar: bool = True,
    ) -> Vector:
        """Detect Bragg disks at every scan position (or a subset for testing).

        Pass ``positions`` to test detection hyperparameters on a handful of
        patterns without scanning the full grid; the returned :class:`Vector` then
        has shape ``(len(positions),)`` and the workflow state is left untouched.
        With ``positions=None`` the full scan is processed, :attr:`peaks` and
        :attr:`bvm` are populated, and the same Vector is returned. Patterns are
        processed in batches (the FFTs and subpixel refinement run together across
        the batch, which is far faster on a GPU); results are identical to detecting
        each pattern on its own.

        Parameters
        ----------
        positions : list of tuple of int, optional
            ``(row, col)`` scan positions to test on. ``None`` (default) processes
            the full scan and updates the workflow state.
        min_abs_intensity : float, default=0.0
            Drop correlation peaks below this absolute intensity.
        min_spacing : float, default=0.0
            Minimum spacing in pixels between kept peaks; closer / dimmer peaks are
            suppressed.
        edge_boundary : int, default=1
            Width in pixels of the border in which peaks are ignored.
        subpixel : {"none", "parabolic", "upsample"}, default="upsample"
            Subpixel refinement mode; see
            :func:`~quantem.diffraction.disk_detection.detect_disks`.
        upsample_factor : int, default=16
            Upsampling factor for the ``"upsample"`` subpixel refinement.
        max_num_peaks : int, default=1000
            Maximum number of peaks to keep per pattern.
        batch_size : int, optional
            Number of patterns per batch. ``None`` (default) picks a size from the
            detector dimensions.
        progressbar : bool, default=True
            If ``True``, show a tqdm progress bar over the full-scan detection.

        Returns
        -------
        Vector
            Detected peaks (``[q_row, q_col, intensity]``): shape ``(scan_row,
            scan_col)`` for the full scan, or ``(len(positions),)`` for a test run.
        """
        if self._template_ft is None:
            raise ValueError("Run a make_template_* method before detect_disks().")

        detect_kwargs = dict(
            min_abs_intensity=min_abs_intensity,
            min_spacing=min_spacing,
            edge_boundary=edge_boundary,
            subpixel=subpixel,
            upsample_factor=upsample_factor,
            max_num_peaks=max_num_peaks,
            background_sigma=self._resolve_background_sigma(background_sigma),
        )

        if positions is not None:
            if len(positions) == 0:
                raise ValueError("positions must contain at least one (row, col) to test on.")
            coords = [(int(r), int(c)) for r, c in positions]
            results = self._detect_positions(coords, detect_kwargs, batch_size, progressbar=False)
            return Vector.from_data(results, fields=PEAK_FIELDS, name="bragg_peaks_test")

        scan_r, scan_c = int(self.dataset.shape[0]), int(self.dataset.shape[1])
        coords = list(np.ndindex(scan_r, scan_c))
        results = self._detect_positions(
            coords, detect_kwargs, batch_size, progressbar=progressbar
        )
        # Store every cell in a single bulk pass. Per-cell assignment
        # (peaks[r, c] = arr) re-concatenates the entire backing buffer on every
        # write -- O(N^2) over the scan, which is the stall at the end of
        # detection. from_data stacks all cells with one _replace_cells call.
        nested = [results[r * scan_c : (r + 1) * scan_c] for r in range(scan_r)]
        peaks = Vector.from_data(nested, fields=PEAK_FIELDS, name="bragg_peaks")

        self.peaks = peaks
        self.metadata["detect"] = detect_kwargs
        self.compute_bvm()
        return peaks

    def compute_bvm(self, sampling: float = 1.0) -> Dataset2d:
        """Accumulate all detected peaks into a Bragg vector map (intensity histogram).

        Parameters
        ----------
        sampling : float, default=1.0
            Reciprocal-space sampling (per pixel) stored on the returned dataset.

        Returns
        -------
        Dataset2d
            ``(H, W)`` Bragg vector map, also stored on :attr:`bvm`.
        """
        if self.peaks is None:
            raise ValueError("Run detect_disks() before compute_bvm().")
        H, W = (int(self.dataset.shape[-2]), int(self.dataset.shape[-1]))
        flat = self.peaks.select_fields("q_row", "q_col", "intensity").flatten()

        bvm = np.zeros((H, W), dtype=float)
        if flat.shape[0] > 0:
            rows = np.clip(np.round(flat[:, 0]).astype(int), 0, H - 1)
            cols = np.clip(np.round(flat[:, 1]).astype(int), 0, W - 1)
            np.add.at(bvm, (rows, cols), flat[:, 2])

        self.bvm = Dataset2d.from_array(
            bvm, name="bragg_vector_map", sampling=(sampling, sampling), signal_units="intensity"
        )
        return self.bvm

    def choose_basis_vectors(
        self,
        origin: int | tuple[float, float] | NDArray | None = None,
        g1: int | tuple[float, float] | NDArray | None = None,
        g2: int | tuple[float, float] | NDArray | None = None,
        *,
        num_candidates: int = 100,
        min_spacing: float = 2.0,
        min_abs_intensity: float = 0.0,
        plot: bool = True,
        returnfig: bool = False,
        **show_kwargs,
    ):
        """Select the lattice basis ``(origin, g1, g2)`` from the Bragg vector map.

        Any of ``origin``/``g1``/``g2`` may be given explicitly; the rest are picked
        automatically. The origin is the **brightest** candidate peak (the
        unscattered central beam). With ``quality = intensity / distance`` rewarding
        short, bright vectors, ``g1`` is then the highest-``quality`` peak and ``g2``
        is the highest ``quality * sin^2(theta)`` peak, where ``theta`` is its angle
        to ``g1`` (the ``sin^2`` factor vanishes for peaks collinear with ``g1``).

        Each override accepts **either** form, told apart by shape:

        - a scalar **candidate index** (an ``int``) picks one of the numbered
          candidate peaks drawn on the plot;
        - a **``(row, col)`` vector** is taken literally — an absolute position for
          ``origin``, an offset *from the origin* for ``g1``/``g2``.

        With ``plot=True`` (default) the Bragg vector map is shown with the
        candidate peaks numbered and the chosen basis overlaid, so the index to
        pass back here can be read straight off the figure.

        Parameters
        ----------
        origin : int or tuple of float or np.ndarray, optional
            Candidate index, or absolute ``(row, col)`` lattice origin. Picked
            automatically if omitted.
        g1 : int or tuple of float or np.ndarray, optional
            Candidate index (vector taken as ``peak - origin``), or a ``(row, col)``
            offset *from the origin*. Picked automatically if omitted.
        g2 : int or tuple of float or np.ndarray, optional
            Candidate index (vector taken as ``peak - origin``), or a ``(row, col)``
            offset *from the origin*. Picked automatically if omitted.
        num_candidates : int, default=100
            Number of brightest candidate peaks to consider (and to number on the
            plot).
        min_spacing : float, default=2.0
            Minimum spacing in pixels between candidate peaks.
        min_abs_intensity : float, default=0.0
            Drop candidate peaks below this absolute intensity.
        plot : bool, default=True
            If ``True``, show the Bragg vector map with the numbered candidates and
            the chosen basis overlaid via
            :func:`~quantem.diffraction.bragg_vectors_visualization.plot_basis_vectors`.
        returnfig : bool, default=False
            If ``True``, return ``(fig, ax)`` from the overlay plot instead of
            ``self`` (implies ``plot=True``).
        **show_kwargs
            Display-scaling options forwarded to the overlay plot's
            :func:`~quantem.core.visualization.show_2d` call (e.g. ``norm``,
            ``vmin``, ``vmax``, ``cmap``, ``lower_quantile``, ``upper_quantile``).

        Returns
        -------
        BraggVectors or tuple
            ``self`` for method chaining; or ``(fig, ax)`` when ``returnfig=True``.
        """
        if self.bvm is None:
            raise ValueError("Run detect_disks()/compute_bvm() before choosing basis vectors.")

        cand_rc, cand_int = self._bvm_candidates(num_candidates, min_spacing, min_abs_intensity)
        if cand_rc.shape[0] == 0:
            raise RuntimeError("No candidate peaks found in the Bragg vector map.")
        # remember the numbered candidates so index_peaks() reuses this exact set.
        self.candidates_rc = cand_rc
        self.candidates_intensity = cand_int

        def _candidate(idx: int) -> NDArray:
            i = int(idx)
            if not -cand_rc.shape[0] <= i < cand_rc.shape[0]:
                raise IndexError(
                    f"candidate index {i} out of range for {cand_rc.shape[0]} candidates "
                    f"(increase num_candidates or loosen min_spacing/min_abs_intensity)."
                )
            return cand_rc[i]

        if origin is None:
            # candidates are returned brightest-first, so [0] is the central beam.
            origin_rc = cand_rc[0]
        elif np.ndim(origin) == 0:
            origin_rc = _candidate(origin)
        else:
            origin_rc = np.asarray(origin, dtype=float).reshape(2)

        rel = cand_rc - origin_rc
        dist = np.linalg.norm(rel, axis=1)
        valid = dist > max(1e-6, min_spacing)
        # "shortest/brightest": reward bright peaks close to the origin.
        quality = cand_int / (dist + 1e-12)

        if g1 is None:
            if not valid.any():
                raise RuntimeError("Could not find a g1 candidate distinct from the origin.")
            g1_rc = rel[int(np.argmax(np.where(valid, quality, -np.inf)))]
        elif np.ndim(g1) == 0:
            g1_rc = _candidate(g1) - origin_rc
        else:
            g1_rc = np.asarray(g1, dtype=float).reshape(2)

        if g2 is None:
            # g2 = highest quality * sin^2(theta) where theta is the angle to g1;
            # sin^2 = 1 - cos^2 vanishes for peaks collinear with g1.
            g1n = g1_rc / (np.linalg.norm(g1_rc) + 1e-12)
            cos = rel @ g1n / (dist + 1e-12)
            sin2 = np.clip(1.0 - cos**2, 0.0, 1.0)
            g2_score = np.where(valid, quality * sin2, -np.inf)
            if g2_score.max() <= 0.0:
                raise RuntimeError("Could not find a g2 candidate non-collinear with g1.")
            g2_rc = rel[int(np.argmax(g2_score))]
        elif np.ndim(g2) == 0:
            g2_rc = _candidate(g2) - origin_rc
        else:
            g2_rc = np.asarray(g2, dtype=float).reshape(2)

        self.origin = np.asarray(origin_rc, dtype=float).reshape(2)
        self.g1 = np.asarray(g1_rc, dtype=float).reshape(2)
        self.g2 = np.asarray(g2_rc, dtype=float).reshape(2)

        if plot or returnfig:
            fig, ax = plot_basis_vectors(
                np.asarray(self.bvm.array),
                cand_rc,
                cand_int,
                self.origin,
                self.g1,
                self.g2,
                **show_kwargs,
            )
            if returnfig:
                return fig, ax
        return self

    def index_peaks(
        self,
        *,
        plot: bool = True,
        returnfig: bool = False,
        **show_kwargs,
    ):
        """Index the chosen candidate peaks into a reference lattice.

        Assigns integer Miller indices ``(a, b)`` to the numbered candidate peaks
        picked in :meth:`choose_basis_vectors` — not the full per-position
        detections; that heavy lifting happens in :meth:`fit_lattice`. Each
        candidate gets ``[a, b] = round(B^-1 (q - origin))`` with ``B`` columns
        ``g1, g2``; when two candidates round to the same ``(a, b)`` only the
        brightest is kept. The result is a compact reference lattice stored on
        :attr:`reference_ab` / :attr:`reference_qpos` / :attr:`reference_intensity`
        which :meth:`fit_lattice` matches against at every scan position. This step
        is quick and lightweight.

        With ``plot=True`` (default) the reference lattice is drawn over the Bragg
        vector map, each site ringed and labelled with its ``(a, b)`` index. The ring
        color encodes how far the picked candidate sits from its ideal lattice site
        ``origin + a*g1 + b*g2``, so a mis-picked or duplicate candidate (which rings
        far from zero offset) is easy to spot.

        Parameters
        ----------
        plot : bool, default=True
            If ``True``, show the reference lattice via
            :func:`~quantem.diffraction.bragg_vectors_visualization.plot_reference_lattice`.
        returnfig : bool, default=False
            If ``True``, return ``(fig, ax)`` instead of ``self`` (implies ``plot``).
        **show_kwargs
            Display-scaling options forwarded to the plot's
            :func:`~quantem.core.visualization.show_2d` call (e.g. ``norm``,
            ``vmin``, ``vmax``, ``cmap``).

        Returns
        -------
        BraggVectors or tuple
            ``self`` for method chaining; or ``(fig, ax)`` when ``returnfig=True``.
        """
        if self.candidates_rc is None or self.candidates_intensity is None:
            raise ValueError("Run choose_basis_vectors() before index_peaks().")
        if self.origin is None or self.g1 is None or self.g2 is None:
            raise ValueError("Run choose_basis_vectors() before index_peaks().")

        cand_rc = self.candidates_rc
        cand_int = self.candidates_intensity
        ab = _index_directions(cand_rc, self.origin, self.g1, self.g2)

        # Dedupe by (a, b): candidates are brightest-first, so the first occurrence
        # of each index is the brightest -- keep it, drop the dimmer duplicates.
        seen: dict[tuple[int, int], int] = {}
        for i, (a, b) in enumerate(ab):
            key = (int(a), int(b))
            if key not in seen:
                seen[key] = i
        keep = np.array(sorted(seen.values()), dtype=int)

        self.reference_ab = ab[keep]
        self.reference_qpos = cand_rc[keep]
        self.reference_intensity = cand_int[keep]

        if not (plot or returnfig):
            return self

        fig, ax = plot_reference_lattice(
            np.asarray(self.bvm.array),
            self.reference_qpos,
            self.reference_ab,
            self.origin,
            self.g1,
            self.g2,
            **show_kwargs,
        )
        if returnfig:
            return fig, ax
        return self

    def fit_lattice(
        self,
        min_num_peaks: int = 5,
        max_peak_shift: float | None = None,
        *,
        progressbar: bool = True,
        plot: bool = True,
        returnfig: bool = False,
    ):
        """Per-position weighted least-squares fit of the lattice vectors.

        This is the heavy step. At every scan position the detected peaks are
        matched to the *ideal* lattice sites from :meth:`index_peaks` —
        ``origin + a*g1 + b*g2`` for each reference ``(a, b)`` — keeping a peak only
        when it lands within ``max_peak_shift`` of its nearest ideal site (not the
        measured candidate position), then ``q = x0 + a*g1 + b*g2`` is fit by
        intensity-weighted least squares over the matched peaks. The fitted ``g1``/``g2`` go into :attr:`u_array`/
        :attr:`v_array` (shape ``(scan_row, scan_col, 2)``, row/col components);
        positions with fewer than ``min_num_peaks`` matched peaks are left ``nan``.

        Two diagnostics are stored per position. :attr:`fit_error` is the RMS fit
        residual over the *matched* peaks, in pixels. :attr:`mask_weight` is a lattice
        *order parameter* in ``0``–``1``: every detected peak is snapped to the
        nearest site of the just-fitted lattice and the intensity-weighted RMS of
        those displacements (the zero beam excluded) is normalized by
        ``sqrt(|g1 x g2| / 2*pi)`` — the RMS displacement expected from intensity
        scattered at random with no lattice — as ``1 - RMS_all / rms_rand`` clipped to
        ``[0, 1]``. Because it weighs *all* detected intensity against the lattice
        (not just the matched peaks, as :attr:`fit_error` does), a clean single
        crystal approaches ``1`` while positions with strong off-lattice intensity (a
        second grain, a mis-index, many spurious peaks) fall toward ``0``; weak false
        positives carry little intensity and barely move it. Positions with no valid
        fit (vacuum, fewer than ``min_num_peaks``) are ``0``. It is the default
        reference weighting handed to :meth:`calculate_strain_map`.

        Parameters
        ----------
        min_num_peaks : int, default=5
            Minimum number of matched peaks required to fit a position; positions
            with fewer are left ``nan``.
        max_peak_shift : float, optional
            Inclusion radius in pixels: a detected peak is kept only if it lands
            within this distance of its nearest *ideal* lattice site
            (``origin + a*g1 + b*g2``), excluding peaks that stray too far from where
            the best-fit lattice predicts. Defaults to ``0.5 * min(|g1|, |g2|)`` —
            half the shorter lattice spacing.
        progressbar : bool, default=True
            If ``True``, show a tqdm progress bar over the scan positions.
        plot : bool, default=True
            If ``True``, show the fit diagnostics (mask weight + RMS error) via
            :func:`~quantem.diffraction.bragg_vectors_visualization.plot_lattice_fit`.
        returnfig : bool, default=False
            If ``True``, return ``(fig, ax)`` instead of ``self`` (implies ``plot``).

        Returns
        -------
        BraggVectors or tuple
            ``self`` for method chaining; or ``(fig, ax)`` when ``returnfig=True``.
        """
        if self.peaks is None:
            raise ValueError("Run detect_disks() before fit_lattice().")
        if self.reference_ab is None or self.reference_qpos is None:
            raise ValueError("Run index_peaks() before fit_lattice().")

        ref_ab = self.reference_ab.astype(float)

        # Ideal lattice sites for the reference (a, b) set: origin + a*g1 + b*g2.
        # Per-position detections are matched to these *ideal* points (not the measured
        # candidate positions), so a peak is included only when it lands within
        # max_peak_shift of where the best-fit lattice predicts it should be.
        o = np.asarray(self.origin, dtype=float).reshape(2)
        g1 = np.asarray(self.g1, dtype=float).reshape(2)
        g2 = np.asarray(self.g2, dtype=float).reshape(2)
        ideal_qpos = o[None, :] + ref_ab[:, 0:1] * g1[None, :] + ref_ab[:, 1:2] * g2[None, :]

        if max_peak_shift is None:
            max_peak_shift = 0.5 * float(min(np.linalg.norm(g1), np.linalg.norm(g2)))

        # Normalization scale for the mask weight: the intensity-weighted RMS
        # peak-to-nearest-site displacement expected from intensity scattered at
        # random (uniformly over a unit cell of area |g1 x g2|) -- i.e. with no
        # lattice order at all. sqrt(A / 2pi) is the equal-area-disk value; the mask
        # weight is then 1 - RMS_all / rms_rand, a lattice order parameter in [0, 1].
        cell_area = abs(float(g1[0] * g2[1] - g1[1] * g2[0]))
        rms_rand = float(np.sqrt(cell_area / (2.0 * np.pi))) if cell_area > 0 else 1.0

        scan_r, scan_c = int(self.dataset.shape[0]), int(self.dataset.shape[1])
        u_array = np.full((scan_r, scan_c, 2), np.nan, dtype=float)
        v_array = np.full((scan_r, scan_c, 2), np.nan, dtype=float)
        mask_weight = np.zeros((scan_r, scan_c), dtype=float)
        fit_error = np.full((scan_r, scan_c), np.nan, dtype=float)

        fields = self.peaks.fields
        i_qr, i_qc = fields.index("q_row"), fields.index("q_col")
        i_int = fields.index("intensity")

        coords: Any = list(np.ndindex(scan_r, scan_c))
        if progressbar:
            try:
                from tqdm.auto import tqdm

                coords = tqdm(coords, desc="fit_lattice", leave=True)
            except Exception:
                pass

        for r, c in coords:
            cell = self.peaks[r, c].array
            if cell.shape[0] == 0:
                continue
            qpos = cell[:, [i_qr, i_qc]]
            inten = cell[:, i_int]

            # nearest *ideal* lattice site for each detected peak, kept if close enough
            d = np.linalg.norm(qpos[:, None, :] - ideal_qpos[None, :, :], axis=2)
            nearest = np.argmin(d, axis=1)
            matched = d[np.arange(d.shape[0]), nearest] <= max_peak_shift

            if int(matched.sum()) < min_num_peaks:
                continue

            beta, rms = _fit_lattice_vectors(
                qpos[matched, 0],
                qpos[matched, 1],
                ref_ab[nearest[matched], 0],
                ref_ab[nearest[matched], 1],
                inten[matched],
            )
            if beta is None:
                continue
            u_array[r, c] = beta[1]
            v_array[r, c] = beta[2]
            fit_error[r, c] = rms

            # mask weight = lattice "order parameter": snap EVERY detected peak to the
            # nearest site of the just-fitted lattice (beta = [x0, g1, g2]) and take
            # the intensity-weighted RMS of those displacements, excluding the zero
            # beam. Unlike fit_error (matched peaks only), this sees OFF-lattice
            # intensity -- a second grain, a mis-index, or many spurious peaks drive
            # it up -- while weak false positives, carrying little intensity, barely
            # move it. Normalized by rms_rand: 1 (all intensity on the lattice) down
            # to 0 (scattered as if there were no lattice).
            x0 = beta[0]
            mat = np.stack([beta[1], beta[2]])  # (2, 2): rows g1, g2
            ab = np.rint((qpos - x0[None, :]) @ np.linalg.inv(mat))
            sites = x0[None, :] + ab @ mat
            disp = np.linalg.norm(qpos - sites, axis=1)
            nonzero = ~np.all(ab == 0, axis=1)  # drop the central (zero) beam
            w = np.clip(inten[nonzero], 0.0, None)
            wsum = float(w.sum())
            if wsum > 0:
                rms_all = float(np.sqrt(np.sum(w * disp[nonzero] ** 2) / wsum))
                mask_weight[r, c] = float(np.clip(1.0 - rms_all / rms_rand, 0.0, 1.0))

        self.u_array = u_array
        self.v_array = v_array
        self.mask_weight = mask_weight
        self.fit_error = fit_error
        self.metadata["fit"] = {
            "min_num_peaks": int(min_num_peaks),
            "max_peak_shift": float(max_peak_shift),
        }

        if not (plot or returnfig):
            return self

        fig, ax = plot_lattice_fit(mask_weight, fit_error)
        if returnfig:
            return fig, ax
        return self

    def calculate_strain_map(
        self,
        u_ref: np.ndarray | None = None,
        v_ref: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> StrainMap:
        """Build a :class:`StrainMap` from the fitted per-position lattice vectors.

        Parameters
        ----------
        u_ref : np.ndarray, optional
            ``(2,)`` reference for the first lattice vector. Defaults to the median
            over the scan inside :class:`StrainMap`.
        v_ref : np.ndarray, optional
            ``(2,)`` reference for the second lattice vector. Defaults to the median
            over the scan inside :class:`StrainMap`.
        mask : np.ndarray, optional
            ``(scan_row, scan_col)`` per-position weighting used when computing the
            reference lattice. Defaults to :attr:`mask_weight` from
            :meth:`fit_lattice` (the lattice order parameter — how well all detected
            intensity snaps to the fitted lattice), so clean single-crystal positions
            dominate the reference and positions with off-lattice intensity are
            down-weighted.

        Returns
        -------
        StrainMap
            A strain map initialized from the fitted lattice vectors.
        """
        if self.u_array is None or self.v_array is None:
            raise ValueError("Run fit_lattice() before calculate_strain_map().")

        if mask is None:
            mask = self.mask_weight

        ds_sampling = float(self.dataset.sampling[0])
        ds_units = str(self.dataset.units[0])

        return StrainMap(
            u_array=self.u_array,
            v_array=self.v_array,
            ds_shape=tuple(self.dataset.shape),
            real_space=self.real_space,
            u_ref=u_ref,
            v_ref=v_ref,
            mask=mask,
            ds_sampling=ds_sampling,
            ds_units=ds_units,
        )

    # ---- visualization ----

    def show_template(
        self,
        position: tuple[int, int] = (0, 0),
        *,
        crop_factor: float | None = None,
        returnfig: bool = False,
        **kwargs,
    ):
        """Plot the mean diffraction pattern, the template, and one correlation map.

        Parameters
        ----------
        position : tuple of int, default=(0, 0)
            ``(row, col)`` scan position whose correlation map is shown.
        crop_factor : float, optional
            If given, zoom to a square window of half-width ``crop_factor * radius``
            about the central-beam center, where ``radius`` is the central-beam
            radius (the synthetic template radius if known, else estimated from the
            mean diffraction pattern). The mean-diffraction and correlation panels are
            centered on the beam; the template panel on its own (fftshifted) center.
            For example, ``crop_factor=2.0`` shows two beam radii either side of the
            center. The window is clamped to the detector, so a large factor shows the
            full image. ``None`` (default) shows the full panels.
        returnfig : bool, default=False
            If ``True``, return the ``(fig, ax)`` for further customization.
        **kwargs
            Extra keyword arguments forwarded to
            :func:`~quantem.diffraction.bragg_vectors_visualization.plot_template`.

        Returns
        -------
        tuple
            ``(fig, ax)`` when ``returnfig=True``; otherwise nothing.
        """
        if self._template is None:
            raise ValueError("Run a make_template_* method before show_template().")
        r, c = int(position[0]), int(position[1])
        dp_mean = np.asarray(self.dataset.dp_mean.array)

        crop = None
        if crop_factor is not None:
            # Center the mean-diffraction and correlation panels on the actual
            # central-beam position rather than the geometric center (H//2, W//2):
            # the unscattered beam -- and the correlation peak that matches it -- are
            # generally offset by a few pixels from the detector center.
            center, radius_est = estimate_central_beam(dp_mean)
            radius = self.metadata.get("template", {}).get("radius")
            if radius is None:
                radius = radius_est
            crop = (float(center[0]), float(center[1]), float(crop_factor) * float(radius))

        fig, ax = plot_template(
            dp_mean,
            self.template,
            self.correlation_map(r, c),
            (r, c),
            crop=crop,
            **kwargs,
        )
        if returnfig:
            return fig, ax

    def show_diffraction(
        self,
        inds: list[tuple[int, int]],
        image: np.ndarray | None = None,
        *,
        ncols: int = 4,
        image_kwargs: dict | None = None,
        marker_radius: float | None = None,
        linewidth: float = 0.5,
        sigma_plot: float | None = None,
        returnfig: bool = False,
        **show_kwargs,
    ):
        """Preview the diffraction patterns at ``inds``, optionally beside a navigation image.

        No detection is run; this is for choosing scan positions to tune on. The
        patterns are tiled ``ncols`` wide and rendered with
        :func:`~quantem.core.visualization.show_2d`; the navigation image is styled
        independently (real space and reciprocal space rarely want the same
        scaling).

        Parameters
        ----------
        inds : list of tuple of int
            ``(row, col)`` scan positions to preview.
        image : np.ndarray or Dataset2d, optional
            Real-space navigation image (e.g. a virtual dark-field image) shown on
            the left with the positions marked. ``None`` (default) shows only the
            diffraction tiles.
        ncols : int, default=4
            Number of diffraction tiles per row.
        image_kwargs : dict, optional
            Keyword arguments styling the navigation image (passed to
            :func:`~quantem.core.visualization.show_2d`).
        marker_radius : float, optional
            Radius in image pixels of the scan-position marker rings.
        linewidth : float, default=0.5
            Stroke width of the scan-position markers.
        sigma_plot : float, optional
            Gaussian blur (sigma) applied to the *displayed* patterns only.
        returnfig : bool, default=False
            If ``True``, return the ``(fig, ax)`` for further customization.
        **show_kwargs
            Extra keyword arguments (e.g. ``norm``, ``cmap``, ``cbar``, ``axsize``)
            styling the diffraction tiles via
            :func:`~quantem.core.visualization.show_2d`.

        Returns
        -------
        tuple
            ``(fig, ax)`` when ``returnfig=True``; otherwise nothing.
        """
        if image is None:
            image_arr = None
            image_title = "navigation image"
        else:
            image_title = getattr(image, "name", None) or "navigation image"
            image_arr = np.asarray(image.array if hasattr(image, "array") else image)
        dps = [np.asarray(self.dataset.array[r, c], dtype=float) for r, c in inds]
        fig, ax = plot_diffraction_grid(
            image_arr,
            dps,
            inds,
            ncols=ncols,
            image_title=image_title,
            image_kwargs=image_kwargs,
            marker_radius=marker_radius,
            linewidth=linewidth,
            sigma_plot=sigma_plot,
            **show_kwargs,
        )
        if returnfig:
            return fig, ax

    def show_detection(
        self,
        positions: list[tuple[int, int]] | None = None,
        *,
        min_abs_intensity: float = 0.0,
        min_spacing: float = 0.0,
        edge_boundary: int = 1,
        subpixel: str = "upsample",
        upsample_factor: int = 16,
        max_num_peaks: int = 1000,
        background_sigma: float | str | None = "auto",
        image: np.ndarray | None = None,
        peak_radius: float = 6.0,
        marker_radius: float | None = None,
        linewidth: float = 1.0,
        sigma_plot: float | None = None,
        image_kwargs: dict | None = None,
        returnfig: bool = False,
        **plot_kwargs,
    ):
        """Detect on a few patterns and overlay the peaks, for tuning hyperparameters.

        The detection keywords match :meth:`detect_disks`. The workflow state
        (:attr:`peaks`/:attr:`bvm`) is left untouched, so this is safe to re-run
        while tuning; for the raw peaks, call :meth:`detect_disks` with the same
        ``positions``.

        Parameters
        ----------
        positions : list of tuple of int, optional
            ``(row, col)`` scan positions to detect on. ``None`` (default)
            auto-samples four positions spread across the scan.
        min_abs_intensity : float, default=0.0
            Drop correlation peaks below this absolute intensity.
        min_spacing : float, default=0.0
            Minimum spacing in pixels between kept peaks.
        edge_boundary : int, default=1
            Width in pixels of the border in which peaks are ignored.
        subpixel : {"none", "parabolic", "upsample"}, default="upsample"
            Subpixel refinement mode; see :meth:`detect_disks`.
        upsample_factor : int, default=16
            Upsampling factor for the ``"upsample"`` subpixel refinement.
        max_num_peaks : int, default=1000
            Maximum number of peaks to keep per pattern.
        image : np.ndarray or Dataset2d, optional
            Real-space navigation image (e.g. a virtual dark-field image) shown on
            the left with the chosen positions marked. ``None`` (default) shows only
            the diffraction tiles.
        peak_radius : float, default=6.0
            Radius in diffraction pixels of the cyan rings drawn at detected peaks.
        marker_radius : float, optional
            Radius in image pixels of the scan-position marker rings.
        linewidth : float, default=1.0
            Stroke width of the peak and scan-position rings.
        sigma_plot : float, optional
            Gaussian blur (sigma) applied to the *displayed* patterns only;
            detection still uses the raw data.
        image_kwargs : dict, optional
            Keyword arguments styling the navigation image (passed to
            :func:`~quantem.core.visualization.show_2d`).
        returnfig : bool, default=False
            If ``True``, return the ``(fig, ax)`` for further customization.
        **plot_kwargs
            Extra keyword arguments (e.g. ``ncols``, ``norm``, ``cmap``, ``cbar``,
            ``axsize``) styling the diffraction tiles via
            :func:`~quantem.core.visualization.show_2d`.

        Returns
        -------
        tuple
            ``(fig, ax)`` when ``returnfig=True``; otherwise nothing.
        """
        if positions is None:
            positions = self._sample_positions()
        sub = self.detect_disks(
            positions=positions,
            min_abs_intensity=min_abs_intensity,
            min_spacing=min_spacing,
            edge_boundary=edge_boundary,
            subpixel=subpixel,
            upsample_factor=upsample_factor,
            max_num_peaks=max_num_peaks,
            background_sigma=background_sigma,
            progressbar=False,
        )
        if image is None:
            image_arr = None
            image_title = "virtual image"
        else:
            image_title = getattr(image, "name", None) or "virtual image"
            image_arr = np.asarray(image.array if hasattr(image, "array") else image)
        dps = [np.asarray(self.dataset.array[r, c], dtype=float) for r, c in positions]
        peaks = [sub[i].array for i in range(len(positions))]
        fig, ax = plot_detection(
            image_arr,
            dps,
            peaks,
            positions,
            peak_radius=peak_radius,
            marker_radius=marker_radius,
            linewidth=linewidth,
            sigma_plot=sigma_plot,
            image_title=image_title,
            image_kwargs=image_kwargs,
            **plot_kwargs,
        )
        if returnfig:
            return fig, ax

    def peak_histogram(self, *, returnfig: bool = False, **kwargs):
        """Plot the Bragg vector map beside the per-position peak count.

        The Bragg vector map is the 2-D histogram of all detected peak positions
        (intensity-weighted) accumulated over the scan; the second panel shows the
        number of peaks detected at each scan position.

        Parameters
        ----------
        returnfig : bool, default=False
            If ``True``, return the ``(fig, ax)`` for further customization.
        **kwargs
            Extra keyword arguments forwarded to
            :func:`~quantem.diffraction.bragg_vectors_visualization.plot_bvm`.

        Returns
        -------
        tuple
            ``(fig, ax)`` when ``returnfig=True``; otherwise nothing.
        """
        if self.peaks is None or self.bvm is None:
            raise ValueError("Run detect_disks() before peak_histogram().")
        scan_r, scan_c = int(self.dataset.shape[0]), int(self.dataset.shape[1])
        # row_counts() reads cell_lengths directly; far cheaper than building a
        # per-cell view (self.peaks[r, c]) just to read its row count.
        counts = np.asarray(self.peaks.row_counts(), dtype=int).reshape(scan_r, scan_c)
        fig, ax = plot_bvm(np.asarray(self.bvm.array), counts, **kwargs)
        if returnfig:
            return fig, ax

    # ---- helpers ----

    def _resolve_background_sigma(
        self, background_sigma: float | str | None
    ) -> float | None:
        """Resolve the ``background_sigma`` argument to a value in pixels.

        ``"auto"`` (the default everywhere) maps to twice the central-beam
        radius: wide enough that the disk-scale correlation peaks pass
        untouched, narrow enough to remove the zero-sum template's negative
        moat around a bright unscattered beam -- which otherwise pushes weak
        disk peaks below zero, where the correlation clamp erases them before
        peak finding. Pass ``None`` to disable the background subtraction or a
        float to set the scale explicitly.
        """
        if background_sigma is None:
            return None
        if isinstance(background_sigma, str):
            if background_sigma != "auto":
                raise ValueError("background_sigma must be a float, None, or 'auto'.")
            radius = self.metadata.get("template", {}).get("radius")
            if radius is None:
                dp_mean = np.asarray(self.dataset.dp_mean.array)
                _, radius = estimate_central_beam(dp_mean)
            return 2.0 * float(radius)
        return float(background_sigma)

    def _set_template(
        self,
        probe: torch.Tensor,
        center: tuple[float, float] | None,
        subtract_mean: bool,
    ) -> None:
        """Validate the probe shape, then store the template and its conjugate FT.

        Parameters
        ----------
        probe : torch.Tensor
            Probe image; must match the diffraction-pattern shape.
        center : tuple of float or None
            ``(row, col)`` probe center rolled to the origin, or ``None`` for the
            geometric center.
        subtract_mean : bool
            If ``True``, make the template zero-sum.
        """
        dp_shape = tuple(self.dataset.shape[-2:])
        probe_t = torch.as_tensor(probe, dtype=torch.float, device=self.device)
        if tuple(probe_t.shape) != dp_shape:
            raise ValueError(
                f"probe shape {tuple(probe_t.shape)} does not match diffraction pattern "
                f"shape {dp_shape}."
            )
        self._template = make_template(probe_t, center=center, subtract_mean=subtract_mean)
        self._template_ft = template_fourier(self._template)

    def _sample_positions(self) -> list[tuple[int, int]]:
        """Four scan positions spread across the field (quadrant centers).

        Returns
        -------
        list of tuple of int
            Up to four ``(row, col)`` scan positions at the quadrant centers.
        """
        R, C = int(self.dataset.shape[0]), int(self.dataset.shape[1])
        rs = sorted({min(max(R // 4, 0), R - 1), min(max(3 * R // 4, 0), R - 1)})
        cs = sorted({min(max(C // 4, 0), C - 1), min(max(3 * C // 4, 0), C - 1)})
        return [(r, c) for r in rs for c in cs]

    def _detect_positions(
        self,
        coords: list[tuple[int, int]],
        detect_kwargs: dict[str, Any],
        batch_size: int | None,
        *,
        progressbar: bool,
    ) -> list[NDArray]:
        """Batched detection over a list of ``(row, col)`` scan positions.

        Patterns are stacked into chunks of ``batch_size`` and passed to
        :func:`~quantem.diffraction.disk_detection.detect_disks_batch`.

        Parameters
        ----------
        coords : list of tuple of int
            ``(row, col)`` scan positions to detect on.
        detect_kwargs : dict
            Keyword arguments forwarded to
            :func:`~quantem.diffraction.disk_detection.detect_disks_batch`.
        batch_size : int or None
            Number of patterns per batch. ``None`` picks a size from the detector
            dimensions.
        progressbar : bool
            If ``True``, show a tqdm progress bar over the patterns.

        Returns
        -------
        list of np.ndarray
            One ``(M, 3)`` array of ``[q_row, q_col, intensity]`` per position, in
            ``coords`` order.
        """
        H, W = int(self.dataset.shape[-2]), int(self.dataset.shape[-1])
        if batch_size is None:
            batch_size = int(min(1024, max(1, 16_000_000 // (H * W))))

        it = range(0, len(coords), batch_size)
        if progressbar:
            try:
                from tqdm.auto import tqdm

                bar = tqdm(total=len(coords), desc="detect_disks", leave=True)
            except Exception:
                bar = None
        else:
            bar = None

        results: list[NDArray] = []
        for start in it:
            chunk = coords[start : start + batch_size]
            dps = torch.stack(
                [
                    torch.as_tensor(
                        np.asarray(self.dataset.array[r, c]),
                        dtype=torch.float,
                        device=self.device,
                    )
                    for r, c in chunk
                ],
                dim=0,
            )
            out = detect_disks_batch(dps, self._template_ft, **detect_kwargs)
            results.extend(
                arr if arr.shape[0] else np.empty((0, len(PEAK_FIELDS)), dtype=float)
                for arr in out
            )
            if bar is not None:
                bar.update(len(chunk))

        if bar is not None:
            bar.close()
        return results

    def _bvm_candidates(
        self, num_candidates: int, min_spacing: float, min_abs_intensity: float
    ) -> tuple[NDArray, NDArray]:
        """Find the brightest, well-separated local maxima in the Bragg vector map.

        Parameters
        ----------
        num_candidates : int
            Maximum number of candidate peaks to return.
        min_spacing : float
            Minimum spacing in pixels between candidates.
        min_abs_intensity : float
            Drop candidates below this absolute intensity.

        Returns
        -------
        cand_rc : np.ndarray
            ``(N, 2)`` ``[row, col]`` candidate positions, brightest first.
        cand_int : np.ndarray
            ``(N,)`` candidate intensities.
        """
        from quantem.diffraction.disk_detection import _filter_maxima, _local_maxima

        bvm = torch.as_tensor(self.bvm.array, dtype=torch.float)
        peaks = _local_maxima(bvm, edge_boundary=1)
        peaks = _filter_maxima(peaks, min_abs_intensity, min_spacing, num_candidates)
        arr = peaks.detach().cpu().numpy()
        return arr[:, :2].astype(float), arr[:, 2].astype(float)


def _fractional_indices(q: NDArray, origin: NDArray, g1: NDArray, g2: NDArray) -> NDArray:
    """Continuous (unrounded) ``(a, b)`` lattice coordinates of peaks ``q``.

    Solves ``B [a, b]^T = q - origin`` (least squares), with
    ``B = [[g1_row, g2_row], [g1_col, g2_col]]``.

    Parameters
    ----------
    q : np.ndarray
        ``(N, 2)`` ``[row, col]`` peak positions.
    origin : np.ndarray
        ``(2,)`` lattice origin ``[row, col]``.
    g1 : np.ndarray
        ``(2,)`` first lattice vector ``[row, col]``.
    g2 : np.ndarray
        ``(2,)`` second lattice vector ``[row, col]``.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` float array of continuous ``[a, b]`` coordinates.
    """
    if q.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    beta = np.array([[g1[0], g2[0]], [g1[1], g2[1]]], dtype=float)
    alpha = (q - origin[None, :]).T  # (2, N)
    return np.linalg.lstsq(beta, alpha, rcond=None)[0].T  # (N, 2)


def _index_directions(q: NDArray, origin: NDArray, g1: NDArray, g2: NDArray) -> NDArray:
    """Integer Miller indices for peaks ``q`` against basis ``(g1, g2)`` about ``origin``.

    Rounds the continuous coordinates from :func:`_fractional_indices`.

    Parameters
    ----------
    q : np.ndarray
        ``(N, 2)`` ``[row, col]`` peak positions.
    origin : np.ndarray
        ``(2,)`` lattice origin ``[row, col]``.
    g1 : np.ndarray
        ``(2,)`` first lattice vector ``[row, col]``.
    g2 : np.ndarray
        ``(2,)`` second lattice vector ``[row, col]``.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` int array of ``[a, b]`` Miller indices.
    """
    if q.shape[0] == 0:
        return np.empty((0, 2), dtype=int)
    return np.round(_fractional_indices(q, origin, g1, g2)).astype(int)


def _fit_lattice_vectors(
    q_row: NDArray,
    q_col: NDArray,
    a: NDArray,
    b: NDArray,
    intensity: NDArray,
) -> tuple[NDArray | None, NDArray | None]:
    """Intensity-weighted lattice fit ``q = x0 + a*g1 + b*g2`` for one pattern.

    Parameters
    ----------
    q_row : np.ndarray
        ``(N,)`` peak row positions.
    q_col : np.ndarray
        ``(N,)`` peak column positions.
    a : np.ndarray
        ``(N,)`` Miller index along ``g1``.
    b : np.ndarray
        ``(N,)`` Miller index along ``g2``.
    intensity : np.ndarray
        ``(N,)`` peak intensities, used as fit weights (``sqrt`` of the clamped
        intensity).

    Returns
    -------
    beta : np.ndarray or None
        ``(3, 2)`` fit ``[x0; g1; g2]`` (row/col components per row), or ``None`` if
        the fit is rank-deficient (e.g. all peaks share one lattice row).
    rms : float
        RMS fit residual in pixels (``nan`` when ``beta`` is ``None``).
    """
    design = np.stack([np.ones_like(a), a, b], axis=1)  # (N, 3)
    target = np.stack([q_row, q_col], axis=1)  # (N, 2)
    w = np.sqrt(np.clip(intensity, 0.0, None))[:, None]

    if np.linalg.matrix_rank(design * w) < 3:
        return None, float("nan")

    beta = np.linalg.lstsq(design * w, target * w, rcond=None)[0]  # (3, 2): x0, g1, g2
    resid = target - design @ beta
    rms = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
    return beta, rms
