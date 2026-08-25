"""Orchestration for :class:`DriftCorrection`.

This module owns the public workflow class. Numerical stages live in
``quantem.imaging.drift.core``, plotting in ``plot``, reporting in ``report``,
and 4D-STEM helpers in ``fourdstem``. Notebooks should import
``DriftCorrection`` from ``quantem.imaging`` or ``quantem.imaging.drift`` and
use ``from_emd`` / ``correct_affine`` / ``plot_combined`` / ``show`` / ``save``.
Manual rigid registration remains available through ``align_translation``.
"""

from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray

import quantem.imaging.drift.apply as drift_apply
import quantem.imaging.drift.core.affine as affine
import quantem.imaging.drift.core.nonrigid as nonrigid
import quantem.imaging.drift.core.strip as strip
import quantem.imaging.drift.core.warping as warping
import quantem.imaging.drift.diagnostics as diagnostics
import quantem.imaging.drift.fourdstem as fourdstem
import quantem.imaging.drift.plot as drift_plot
import quantem.imaging.drift.preparation as preparation
import quantem.imaging.drift.report as drift_report
from quantem.core.config import validate_device
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize


class DriftCorrection(AutoSerialize):
    """GPU-accelerated multi-angle scan drift correction.

    Aligns scans at different scan directions, recovers per-scanline drift,
    and forms a corrected product (HAADF pairs, reference EDS/EELS, 4D-STEM).

    Typical chain: :meth:`from_emd` → :meth:`correct_affine` →
    :meth:`plot_combined` / :meth:`show` / :meth:`save`. Use
    :meth:`align_translation` when a manual rigid-registration stage is
    required. Residual polish: :meth:`correct_strip`,
    :meth:`correct_nonrigid` (use small ``max_image_shift`` on lattices).
    """

    def __init__(
        self,
        *datasets: Dataset2d | Dataset3d | NDArray,
        scan_direction_degrees: list[float] | NDArray | float | None = None,
        alignment_image: NDArray | None = None,
        device: str | None = None,
    ):
        """Parameters
        ----------
        *datasets : ndarray or Dataset2d or Dataset3d or Dataset4d
            Two or more inputs (first = reference / first scan).
        scan_direction_degrees : float or sequence of float, optional
            Angle per dataset. Required for bare arrays; omit when Datasets
            already have ``metadata["scan_rotation_deg"]``.
        alignment_image : 2-D ndarray, optional
            ≥3-D reference mode: 2-D partner for alignment (else auto virtual image).
        device : str, optional
            ``None`` → cuda/mps/cpu; ``"cpu"`` / ``"gpu"`` / ``"cuda:N"`` to pin.
        """
        # Core state (always set so all code paths can rely on these).
        self._datasets: list[np.ndarray | None] | None = None
        self._datasets_consumed: bool = False
        # Records that this instance was constructed from a heavy dataset
        # (4D-STEM collection or reference mode). save() drops _datasets, so
        # this flag is the only surviving signal that a reloaded instance
        # needs its datasets re-attached before apply_correction() can run.
        self._built_from_datasets: bool = False
        self._normalized: bool = False
        self._reference_mode: bool = False
        # device=None auto-picks (cuda -> mps -> cpu). Pass "cuda:1" etc. to
        # pin a specific GPU when one process drives several (HPC schedulers
        # that set CUDA_VISIBLE_DEVICES per job need nothing here).
        device, _ = validate_device(device)
        self._device = device
        self._dtype = torch.float32

        if not datasets and alignment_image is None:
            return
        preparation.prepare_inputs(
            self, datasets, scan_direction_degrees, alignment_image
        )

    @property
    def device(self) -> str:
        """Normalized compute device used by this correction.

        Returns ``"cuda:N"``, ``"mps"``, or ``"cpu"``. CUDA indices follow
        PyTorch's visible-device numbering, including any remapping performed
        by ``CUDA_VISIBLE_DEVICES``.
        """
        return str(self._device)

    @classmethod
    def from_images(
        cls,
        *images,
        scan_direction_degrees: list[float] | NDArray | tuple[float, ...] | None = None,
        device: str | None = None,
    ) -> Self:
        """Create drift correction from already-loaded 2-D scans.

        Prefer :meth:`from_emd` for Velox files. Use this when data is already
        in memory. Scan angles resolve as:

        1. Explicit ``scan_direction_degrees`` (always wins).
        2. Each Dataset's ``metadata["scan_rotation_deg"]`` (from ``read_emd``).
        3. Neither → ``TypeError`` (bare arrays never get a silent default).

        Parameters
        ----------
        *images
            Two or more 2-D ``ndarray`` or :class:`Dataset2d` objects.
        scan_direction_degrees : sequence of float or None, default None
            One angle per image when metadata is missing. Omit when every
            Dataset already carries ``scan_rotation_deg``.
        device : str or None, default None
            ``None`` auto-selects ``cuda`` → ``mps`` → ``cpu``. Use ``"cpu"`` to
            force CPU, ``"gpu"`` to require a GPU, or ``"cuda:N"`` for multi-GPU.

        Returns
        -------
        DriftCorrection

        Examples
        --------
        >>> from quantem.imaging.drift.io import read_emd
        >>> from quantem.gpu.device import profile
        >>> device = profile()["device"]  # or device="cpu"
        >>> d0, d1 = read_emd(f0), read_emd(f90)
        >>> dc = DriftCorrection.from_images(d0, d1, device=device)

        Bare arrays require explicit angles:

        >>> dc = DriftCorrection.from_images(
        ...     arr0, arr90, scan_direction_degrees=(0, 90), device="cpu")
        """
        return cls(*images, scan_direction_degrees=scan_direction_degrees, device=device)

    @classmethod
    def from_emd(cls, *paths, verbose: bool = True, device: str | None = None) -> Self:
        """Load Velox EMD files and build a :class:`DriftCorrection`.

        Each path is read with ``read_emd`` (scan angle from Velox metadata), then
        handed to :meth:`from_images`. Never type angles for EMD files. Path order
        does not matter.

        Parameters
        ----------
        *paths : str or Path
            Two or more EMD files with distinct scan angles.
        verbose : bool, default True
            Print shape, pixel size, and scan angle for each file.
        device : str or None, default None
            Compute device. ``None`` auto-selects ``cuda`` → ``mps`` → ``cpu``.
            ``"gpu"`` requires CUDA/MPS; ``"cpu"`` forces CPU; ``"cuda:1"`` pins a
            device in multi-GPU processes.

            Device helpers: ``from quantem.gpu.device import profile`` then
            ``device=profile()["device"]``.

        Returns
        -------
        DriftCorrection
            Ready for :meth:`correct_affine`.

        Examples
        --------
        >>> from quantem.gpu.device import profile
        >>> device = profile()["device"]  # or device="cpu"
        >>> dc = DriftCorrection.from_emd(
        ...     "scan_0.emd", "scan_90.emd", device=device, verbose=False)
        >>> dc.correct_affine(show_combined=False, verbose=False)
        >>> dc.plot_combined(stage=("initial", "affine"), interactive=True)
        >>> dc.save("drift.zip", mode="o")
        """
        if len(paths) < 2:
            raise TypeError(f"from_emd requires at least 2 EMD files, got {len(paths)}")
        # Local import: drift.io imports quantem at module load, so importing it
        # at correction module scope would be circular.
        from quantem.imaging.drift.io import read_emd

        data = [read_emd(p) for p in paths]
        if len(data) == 2:
            data[0], data[1] = preparation.match_scan_shapes(
                data[0], data[1], verbose=verbose
            )
        if verbose:
            for ds in data:
                print(
                    f"{tuple(ds.shape)}  {float(ds.sampling[0]) * 1e3:.2f} pm  "
                    f"scan {ds.metadata['scan_rotation_deg']:.1f} deg  (read from EMD)"
                )
        return cls.from_images(*data, device=device)

    @classmethod
    def from_4dstem(
        cls,
        *datasets,
        scan_direction_degrees: list[float] | NDArray | tuple[float, ...] = (0.0, 90.0),
        scan_sampling: float | tuple[float, float] | None = None,
        scan_units: str | tuple[str, str] = "pixels",
        device: str | None = None,
    ) -> Self:
        """Build a 0°/90° 4D-STEM *collection* correction (two drifted scans).

        Alignment uses auto-extracted virtual images; apply fields with
        :meth:`corrected_4dstem`. Distinct from :meth:`from_reference` (fixed 2-D
        reference + one multi-D target). Currently requires exactly two orthogonal
        4D-STEM datasets.

        Parameters
        ----------
        *datasets
            Two 4D-STEM datasets with leading scan axes
            ``(scan_row, scan_col, det_row, det_col)``.
        scan_direction_degrees : sequence of float, default (0.0, 90.0)
            Scan direction per dataset in degrees.
        scan_sampling : float or 2-tuple of float, optional
            Real-space scan sampling. Scalar applies to both axes. ``None`` leaves
            uncalibrated sampling.
        scan_units : str or 2-tuple of str, default "pixels"
            Units for ``scan_sampling`` (for example ``"nm"``).
        device : str or None, default None
            ``None`` → cuda/mps/cpu. ``"cpu"`` / ``"gpu"`` / ``"cuda:N"`` as in
            :meth:`from_emd`.

        Returns
        -------
        DriftCorrection

        Examples
        --------
        >>> from quantem.gpu.device import profile
        >>> device = profile()["device"]  # or "cpu"
        >>> dc = DriftCorrection.from_4dstem(
        ...     data_0, data_1,
        ...     scan_direction_degrees=(0.0, 90.0),
        ...     scan_sampling=0.05,
        ...     scan_units="nm",
        ...     device=device,
        ... )
        >>> dc.correct_affine(show_combined=False, verbose=False)
        >>> result = dc.corrected_4dstem(merge=True, verbose=True)
        """
        result = cls(
            *datasets,
            scan_direction_degrees=scan_direction_degrees,
            device=device,
        )
        if scan_sampling is not None:
            units = (scan_units, scan_units) if isinstance(scan_units, str) else scan_units
            for image in result.imgs:
                image.sampling = scan_sampling
                image.units = units
        return result

    @classmethod
    def from_reference(
        cls,
        reference_image,
        drifted_dataset,
        *,
        alignment_image: NDArray | None = None,
        scan_direction_degrees: list[float] | NDArray | float = 0.0,
        device: str | None = None,
    ) -> Self:
        """Build a *reference-anchored* correction (fixed frame + drifted data).

        ``reference_image`` is a 2-D HAADF (or a solved :class:`DriftCorrection`
        whose merge becomes the frame). ``drifted_dataset`` may be 2-D, 3-D
        (EDS/EELS), or 4-D STEM. Alignment uses ``alignment_image`` or an auto
        virtual image. Image 0 is fixed automatically.

        Parameters
        ----------
        reference_image
            Fixed 2-D reference, or a solved :class:`DriftCorrection` used as a
            provenance-chained reference.
        drifted_dataset
            Drifted 2-D image, 3-D spectrum image, or 4-D STEM dataset.
        alignment_image : ndarray, optional
            2-D image used to estimate target drift (for example session HAADF).
            If omitted, a virtual image is extracted automatically.
        scan_direction_degrees : float or sequence of float, default 0.0
            Target scan direction(s) in degrees.
        device : str or None, default None
            ``None`` auto-selects device or inherits from a chained reference.
            ``"cpu"`` forces CPU; ``"gpu"`` requires GPU.

        Returns
        -------
        DriftCorrection

        Examples
        --------
        >>> from quantem.gpu.device import profile
        >>> device = profile()["device"]
        >>> dc = DriftCorrection.from_reference(
        ...     haadf_ref,
        ...     eds_cube,
        ...     alignment_image=eds_session_haadf,
        ...     device=device,
        ... )
        >>> dc.correct_affine(show_combined=False)
        >>> eds_corrected = dc.corrected()  # Dataset3d

        Chained reference from a prior pair solve:

        >>> dc_pair = DriftCorrection.from_emd(f0, f90).correct_affine()
        >>> dc_eds = DriftCorrection.from_reference(
        ...     dc_pair, eds_cube, alignment_image=eds_haadf)
        """
        reference_dc = None
        if isinstance(
            reference_image, DriftCorrection
        ) or AutoSerialize._is_autoserialize_instance(reference_image):
            # Chained reference: derive the fixed frame from the solved pair's
            # corrected merge, cropped to the drifted scan field of view.
            reference_dc = reference_image
            if device is None:
                device = reference_dc.device
            drifted_shape = preparation.input_array(drifted_dataset).shape[:2]
            reference_downsample = preparation.reference_downsample(
                tuple(int(value) for value in reference_dc.imgs[0].shape[:2]),
                tuple(int(value) for value in drifted_shape),
                reference_sampling=getattr(reference_dc.imgs[0], "sampling", None),
                target_sampling=getattr(drifted_dataset, "sampling", None),
            )
            if reference_downsample > 1:
                # Solve the reference pair on the target acquisition grid. On
                # atomic images, aligning at the finer grid and averaging only
                # the final merge can preserve a different lattice phase than
                # the coarser EDS scan. Re-solving the already-small virtual
                # images avoids that alias while keeping the operation fully
                # automatic and costs well under a second on the microscope GPU.
                scaled_reference_images = [
                    preparation.average_downsample_2d(
                        np.asarray(image.array),
                        reference_downsample,
                    )
                    for image in reference_dc.imgs
                ]
                reference_dc = cls.from_images(
                    *scaled_reference_images,
                    scan_direction_degrees=tuple(reference_dc.scan_direction_degrees),
                    device=device,
                )
                reference_dc.correct_affine(
                    show_combined=False,
                    show_scans=False,
                    show_knots=False,
                    verbose=False,
                )
            reference_image = preparation.match_reference_image(
                reference_dc.corrected(output_frame="canvas").array,
                tuple(int(value) for value in reference_dc.imgs[0].shape[:2]),
                tuple(int(value) for value in drifted_shape),
            )
            reference_image = reference_image.astype(np.float32, copy=False)
        result = cls(
            reference_image,
            drifted_dataset,
            scan_direction_degrees=scan_direction_degrees,
            alignment_image=alignment_image,
            device=device,
        )
        # The corrected product must carry the drifted dataset's calibration
        # (sampling, units, metadata). Without this, reference-mode corrected()
        # falls back to {} and the EDS/4D output ships uncalibrated.
        result._reference_dataset_info = drift_apply.dataset_info(drifted_dataset)
        if not result._reference_mode:
            raise ValueError(
                "from_reference() requires a 2-D reference image and one "
                "drifted target dataset. For a 2-D target, pass a scalar "
                "scan_direction_degrees value or matching reference/target "
                "scan directions so the call is unambiguously single-sided. "
                "Use DriftCorrection.from_4dstem() for 0/90 4D-STEM "
                "collection correction."
            )
        return result

    drift_field = fourdstem.drift_field
    probe_positions = fourdstem.probe_positions

    preprocess = preparation.preprocess
    align_translation = warping.align_translation
    correct_affine = affine.correct_affine

    correct_strip = strip.correct_strip

    report = drift_report.report

    correct_nonrigid = nonrigid.correct_nonrigid

    diagnose_affine = diagnostics.diagnose_affine
    diagnose_nonrigid = diagnostics.diagnose_nonrigid

    corrected = drift_apply.corrected
    apply_correction = drift_apply.apply_correction

    crop = drift_apply.crop
    crop_slices = drift_apply.crop_slices
    coverage_mask = drift_apply.coverage_mask
    show = drift_plot.show
    show_4dstem = drift_plot.show_4dstem

    corrected_virtual_images = fourdstem.corrected_virtual_images
    regional_diffraction_patterns = fourdstem.regional_diffraction_patterns
    corrected_4dstem = fourdstem.corrected_4dstem

    integrate_virtual_detector = staticmethod(fourdstem.integrate_virtual_detector)

    # -- serialization -------------------------------------------------------

    def save(self, path, mode="w", store="auto", skip=(), compression_level=4):
        """Save a solved correction for later analysis and figure rendering.

        Persists knots and 2-D alignment state so figure notebooks can ``load``
        without re-solving. Large 4D-STEM cubes under ``_datasets`` are always
        skipped.

        Parameters
        ----------
        path : str or Path
            Output path (typically ``drift.zip`` next to the raw data).
        mode : str, default "w"
            File mode for AutoSerialize. Tutorials often use ``"o"`` to overwrite.
        store : str, default "auto"
            Storage backend selection.
        skip : sequence, default ()
            Extra attributes to omit. ``_datasets`` is always appended.
        compression_level : int, default 4
            Archive compression level.

        Returns
        -------
        None

        Examples
        --------
        >>> dc = DriftCorrection.from_emd(f0, f90)
        >>> dc.correct_affine(show_combined=False, verbose=False)
        >>> dc.save("data/sample/drift.zip", mode="o")
        >>> from quantem.core.io import load
        >>> dc2 = load("data/sample/drift.zip")

        """
        if isinstance(skip, (str, type)):
            skip = [skip]
        skip = list(skip) + ["_datasets"]
        super().save(
            path,
            mode=mode,
            store=store,
            skip=skip,
            compression_level=compression_level,
        )

    drift_rate = property(affine.drift_rate)

    # -- visualization methods bound directly from drift_plot so hover
    #    shows the real signature + docstring (no `**kw` indirection).
    plot_warped_images = drift_plot.plot_warped_images
    plot_convergence = drift_plot.plot_convergence
    # Primary registration QA plot (tutorials + paper combined wording).
    plot_combined = drift_plot.plot_combined
    plot_knots = drift_plot.plot_knots
    plot_probe_positions = drift_plot.plot_probe_positions
