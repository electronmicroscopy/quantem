from collections.abc import Sequence
from typing import Self

import torch
from numpy.typing import NDArray

from quantem.core.config import validate_device
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.compound_validators import (
    validate_list_of_dataset2d,
)
from quantem.core.utils.validators import ensure_valid_array

from . import apply as drift_apply
from . import plot as drift_plot
from . import preparation
from .core import affine, nonrigid


class DriftCorrection(AutoSerialize):
    """
        DriftCorrection provides translation, affine, and non-rigid drift correction for
        sequential 2D images using scan direction metadata and flexible spatial interpolation.

        This class supports input data as numpy arrays, Dataset2d, or Dataset3d instances,
        with various padding strategies and configurable spline interpolation of scanline
        trajectories via Bézier knot control.

        Features
        --------
        - Load data from arrays or files
        - Apply initial scanline resampling using Bézier curves
        - Align images using translation, affine, or non-rigid optimization
        - Visualize intermediate and final results with optional knot overlays
        - Serialize state with `.save()` and restore with `.load()`

        Parameters (via `from_data` or `from_file`)
        -------------------------------------------
        images : list of 2D arrays, Dataset2d, Dataset3d, or file names, or a 3D numpy array
            The image stack to correct for drift.
        scan_direction_degrees : list of float
            The scan direction angle (in degrees) for each image, measured relative to vertical.
        pad_fraction : float, default 0.25
            Fraction of padding to add around each image during interpolation.
        pad_value : str, float, or list of float, default 'median'
            How to pad outside the image area during warping. Can be:
            - One of: 'median', 'mean', 'min', 'max'
            - A float quantile value (e.g., 0.25)
            - A list of per-image float values
        number_knots : int, default 1
            Number of knots to use for Bézier interpolation of scanline trajectories.
            We strongly recommend using `number_knots = 1` unless the fast scan direction is
            expected to vary within the image.

        Example
        -------
        Instantiate the DriftCorrection class, run preprocessing and alignment, and save/load results:

        >>> drift = DriftCorrection.from_data(
        ...     images=[
        ...         image0,  # 2D numpy array or Dataset2d
        ...         image1,
        ...     ],
        ...     scan_direction_degrees=[0, 90],
        ... ).preprocess(
        ...     pad_fraction=0.25,
        ...     pad_value='median',
        ...     number_knots=1,
        ... )

        >>> drift.align_affine()
        >>> drift.align_nonrigid()
        >>> drift.plot_merged_images()
        >>> image_corr = drift.generate_corrected_image()

        >>> drift.save("drift_result.zip")
        >>> drift_reloaded = quantem.io.load("drift_result.zip")

        >>> image_corr.save("image_corrected.zip")
        >>> image_corr_reloaded = quantem.io.load("image_corrected.zip")

        Notes
        -----
        - Use `align_translation()` for rigid shifts, `align_affine()` for scan-shear or uniform drift,
          and `align_nonrigid()` for flexible per-row or per-image correction.
        - The class stores resampled images in `self.images_warped` and the control knots in `self.knots`.
        - Visualization is supported through `plot_merged_images()` and `plot_transformed_images()`.

        Performance
        -----------
        ``align_affine`` uses PyTorch to run all heavy operations on GPU
        (works on CUDA, MPS, and CPU). The key optimizations are:

        - **Batched grid search**: all ~97 candidate drift vectors are warped
          and scored in parallel, instead of one-at-a-time in a Python loop.
        - **Batched bilinear KDE** (``core.knots.bilinear_kde_batch``):
          scatter-based image warping via ``scatter_add_`` with int32 indices.
        - **Batched FFT cross-correlation** (``core.warping.cross_corr_batch``):
          sub-pixel alignment using DFT upsampling across all candidates at once.
        - **Zero CPU round-trips**: coordinate transforms, Gaussian smoothing,
          translation alignment, and error computation all stay on GPU until
          the final sync.

        This gives ~300× speedup over the original NumPy implementation
        (e.g. 436 s → 1.5 s on 2048×2048 image pairs).

        Memory is automatically chunked when the full batch doesn't fit.
        Approximate memory per candidate at common sizes:

        ========== =========== ================
        Input size Canvas size Mem / candidate
        ========== =========== ================
        1024×1024  1280×1280     85 MB
        2048×2048  2560×2560    341 MB
        4096×4096  5120×5120   1.36 GB
        ========== =========== ================
        """

    _token = object()

    def __init__(
        self,
        images: list[Dataset2d],
        scan_direction_degrees: NDArray,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use DriftCorrection.from_data() or .from_file() to instantiate this class."
            )

        self.images = images
        self.scan_direction_degrees = ensure_valid_array(scan_direction_degrees, ndim=1)
        self._reference_mode = False

        device, _ = validate_device(None)
        self._device = device
        self._dtype = torch.float32

    @classmethod
    def from_file(
        cls,
        file_paths: Sequence[str],
        scan_direction_degrees: Sequence[float] | NDArray,
        file_type: str | None = None,
    ) -> Self:
        image_list = [Dataset2d.from_file(fp, file_type=file_type) for fp in file_paths]
        return cls.from_data(
            image_list,
            scan_direction_degrees,
        )

    @classmethod
    def from_data(
        cls,
        images: list[Dataset2d] | list[NDArray] | Dataset3d | NDArray,
        scan_direction_degrees: list[float] | NDArray,
    ) -> Self:
        validated_images = validate_list_of_dataset2d(images)

        return cls(
            images=validated_images,
            scan_direction_degrees=scan_direction_degrees,
            _token=cls._token,
        )

    @classmethod
    def from_reference(
        cls,
        reference_image: Dataset2d | NDArray | Self,
        alignment_image: Dataset2d | NDArray,
        *,
        scan_direction_degrees: float | None = None,
    ) -> Self:
        """Fit a fixed-reference drift field from two-dimensional images.

        The two-dimensional ``reference_image`` defines the output frame.
        ``alignment_image`` is the HAADF image acquired with a spectrum image.
        Pass the spectrum image to :meth:`apply_correction` after fitting.

        Parameters
        ----------
        reference_image : Dataset2d, numpy.ndarray, or DriftCorrection
            Fixed two-dimensional reference image. A solved correction is
            converted to its corrected native-size image automatically.
        alignment_image : Dataset2d or numpy.ndarray
            HAADF image acquired with the spectrum image.
        scan_direction_degrees : float or None, default None
            Recorded scan rotation. Omit when ``alignment_image`` carries
            ``metadata["scan_rotation_deg"]``.

        Returns
        -------
        DriftCorrection
            Correction ready for ``preprocess`` and ``align_affine``.

        Examples
        --------
        >>> drift = DriftCorrection.from_reference(
        ...     reference,
        ...     haadf,
        ...     scan_direction_degrees=0.0,
        ... )
        >>> drift.preprocess(show_merged=False, show_images=False)
        >>> drift.align_affine(show_merged=False, show_images=False)
        >>> corrected = drift.apply_correction(spectrum_image)
        """
        if isinstance(reference_image, cls):
            reference_image = reference_image.generate_corrected(
                upsample_factor=1,
                strip_padding=True,
                mask_output=False,
                show_merged=False,
            )
        reference_array = (
            reference_image.array
            if isinstance(reference_image, Dataset2d)
            else ensure_valid_array(reference_image, ndim=2)
        )
        alignment_array = (
            alignment_image.array
            if isinstance(alignment_image, Dataset2d)
            else ensure_valid_array(alignment_image, ndim=2)
        )
        if reference_array.shape != alignment_array.shape:
            raise ValueError(
                "reference_image and alignment_image must have the same "
                f"shape; got {reference_array.shape} and {alignment_array.shape}."
            )
        if scan_direction_degrees is None:
            metadata = getattr(alignment_image, "metadata", {})
            scan_direction_degrees = metadata.get("scan_rotation_deg")
        if scan_direction_degrees is None:
            raise TypeError(
                "scan_direction_degrees is required when alignment_image "
                "does not carry scan_rotation_deg metadata."
            )
        result = cls.from_data(
            images=[reference_image, alignment_image],
            scan_direction_degrees=[scan_direction_degrees, scan_direction_degrees],
        )
        result._reference_mode = True
        return result

    preprocess = preparation.preprocess
    align_translation = preparation.align_translation
    align_affine = affine.align_affine
    _affine_grid_search_batch = affine._affine_grid_search_batch
    _auto_chunk_size = staticmethod(affine._auto_chunk_size)
    _warp_and_translate_torch = affine._warp_and_translate_torch
    align_nonrigid = nonrigid.align_nonrigid
    _optimize_knots_adam = nonrigid._optimize_knots_adam
    _compiled_loss_fn = staticmethod(nonrigid._compiled_loss_fn)
    _optimize_knots_lbfgs = nonrigid._optimize_knots_lbfgs
    _regularize_knots = nonrigid._regularize_knots
    _optimize_knots_scipy = nonrigid._optimize_knots_scipy
    generate_corrected = drift_apply.generate_corrected
    generate_corrected_image = drift_apply.generate_corrected_image
    apply_correction = drift_apply.apply_correction
    calculate_error = drift_apply.calculate_error
    plot_transformed_images = drift_plot.plot_transformed_images
    plot_convergence = drift_plot.plot_convergence
    _ensure_warped_images = drift_plot._ensure_warped_images
    plot_merged_images = drift_plot.plot_merged_images
