from collections.abc import Sequence
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from tqdm import tqdm

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.compound_validators import (
    validate_list_of_dataset2d,
    validate_pad_value,
)
from quantem.core.utils.imaging_utils import (
    bilinear_kde,
    cross_correlation_shift,
    fourier_cropping,
)
from quantem.core.utils.validators import ensure_valid_array
from quantem.core.visualization import show_2d


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
    - Interactive visualization is supported through `plot_merged_images()` and `plot_transformed_images()`.
    """

    _token = object()

    def __init__(
        self,
        images: List[Dataset2d],
        scan_direction_degrees: NDArray,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use DriftCorrection.from_data() or .from_file() to instantiate this class."
            )

        self._images = images
        self.scan_direction_degrees = scan_direction_degrees

    @classmethod
    def from_file(
        cls,
        file_paths: Sequence[str],
        scan_direction_degrees: Union[Sequence[float], NDArray],
        file_type: str | None = None,
    ) -> "DriftCorrection":
        image_list = [Dataset2d.from_file(fp, file_type=file_type) for fp in file_paths]
        return cls.from_data(
            image_list,
            scan_direction_degrees,
        )

    @classmethod
    def from_data(
        cls,
        images: Union[List[Dataset2d], List[NDArray], Dataset3d, NDArray],
        scan_direction_degrees: Union[List[float], NDArray],
    ) -> "DriftCorrection":
        validated_images = validate_list_of_dataset2d(images)

        return cls(
            images=validated_images,
            scan_direction_degrees=scan_direction_degrees,
            _token=cls._token,
        )

    # --- Properties ---
    @property
    def images(self) -> List[Dataset2d]:
        return self._images

    @images.setter
    def images(self, value: Union[List[Dataset2d], List[NDArray], Dataset3d, NDArray]):
        self._images = validate_list_of_dataset2d(value)
        self.pad_value = self.pad_value

    @property
    def pad_value(self) -> List[float]:
        return self._pad_value

    @pad_value.setter
    def pad_value(self, value: Union[float, str, List[float]]):
        self._pad_value = validate_pad_value(value, self.images)

    @property
    def scan_direction_degrees(self) -> NDArray:
        return self._scan_direction_degrees

    @scan_direction_degrees.setter
    def scan_direction_degrees(self, value: Union[List[float], NDArray]):
        self._scan_direction_degrees = ensure_valid_array(value, ndim=1)

    @property
    def pad_fraction(self) -> float:
        return self._pad_fraction

    @pad_fraction.setter
    def pad_fraction(self, value: float):
        self._pad_fraction = float(value)

    @property
    def kde_sigma(self) -> float:
        return self._kde_sigma

    @kde_sigma.setter
    def kde_sigma(self, value: float):
        self._kde_sigma = float(value)

    @property
    def number_knots(self) -> int:
        return self._number_knots

    @number_knots.setter
    def number_knots(self, value: float):
        self._number_knots = int(value)

    def preprocess(
        self,
        pad_fraction: float = 0.25,
        pad_value: Union[float, str, List[float]] = "median",
        kde_sigma: float = 0.5,
        number_knots: int = 1,
        show_merged: bool = False,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        # Validators
        validated_pad_value = validate_pad_value(pad_value, self._images)

        # Input data
        self.pad_fraction = pad_fraction
        self._pad_value = validated_pad_value
        self.kde_sigma = kde_sigma
        self.number_knots = number_knots

        # Derived data
        self.scan_direction = np.deg2rad(self.scan_direction_degrees)
        self.scan_fast = np.stack(
            [
                np.sin(-self.scan_direction),
                np.cos(-self.scan_direction),
            ],
            axis=1,
        )
        self.scan_slow = np.stack(
            [
                np.cos(-self.scan_direction),
                -np.sin(-self.scan_direction),
            ],
            axis=1,
        )
        self.shape = (
            len(self.images),
            int(np.round(self.images[0].shape[0] * (1 + self.pad_fraction) / 2) * 2),
            int(np.round(self.images[1].shape[1] * (1 + self.pad_fraction) / 2) * 2),
        )

        # Initialize Bezier knots and scan vectors for scanlines
        self.knots = []
        for a0 in range(self.shape[0]):
            shape = self.images[a0].shape

            v_slow = np.linspace(-(shape[0] - 1) / 2, (shape[0] - 1) / 2, shape[0])
            u_fast = np.linspace(
                -(shape[1] - 1) / 2, (shape[1] - 1) / 2, self.number_knots
            )

            xa = (
                (self.shape[1] - 1) / 2
                + u_fast[None, :] * self.scan_fast[a0, 0]
                + v_slow[:, None] * self.scan_slow[a0, 0]
            )
            ya = (
                (self.shape[2] - 1) / 2
                + u_fast[None, :] * self.scan_fast[a0, 1]
                + v_slow[:, None] * self.scan_slow[a0, 1]
            )

            self.knots.append(np.stack([xa, ya], axis=0))

        # Precompute the interpolator for all images
        self.interpolator = []
        for a0 in range(self.shape[0]):
            self.interpolator.append(
                DriftInterpolator(
                    input_shape=self.images[a0].shape,
                    output_shape=self.shape[1:],
                    scan_fast=self.scan_fast[a0],
                    scan_slow=self.scan_slow[a0],
                    pad_value=self.pad_value[a0],
                    kde_sigma=self.kde_sigma,
                )
            )

        # Generate initial resampled images
        self.images_warped = Dataset3d.from_shape(self.shape)
        self.weights_warped = Dataset3d.from_shape(self.shape)
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = (
                self.interpolator[ind].warp_image(
                    self.images[ind].array,
                    self.knots[ind],
                )
            )

        # Error tracking
        self.calculate_error(0)

        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots, title="Merged: initial", **kwargs
            )
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: initial" for i in range(self.shape[0])],
                **kwargs,
            )

        return self

    # Translation alignment
    def align_translation(
        self,
        upsample_factor: int = 8,
        max_image_shift: int = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Solve for the translation between all images in DriftCorrection.images_warped
        """

        if not hasattr(self, "knots"):
            print(
                "\033[91mNo knots found — running .preprocess() with default settings.\033[0m"
            )
            self.preprocess()

        # init
        dxy = np.zeros((self.shape[0], 2))

        # loop over images
        F_ref = np.fft.fft2(self.images_warped.array[0])
        for ind in range(1, self.shape[0]):
            shifts, image_shift = cross_correlation_shift(
                F_ref,
                np.fft.fft2(self.images_warped.array[ind]),
                upsample_factor=upsample_factor,
                max_shift=max_image_shift,
                fft_input=True,
                fft_output=True,
                return_shifted_image=True,
            )

            dxy[ind, :] = shifts
            F_ref = F_ref * ind / (ind + 1) + image_shift / (ind + 1)

        # Normalize dxy
        dxy -= np.mean(dxy, axis=0)

        # Apply shifts to knots
        for ind in range(self.shape[0]):
            self.knots[ind][0] += dxy[ind, 0]
            self.knots[ind][1] += dxy[ind, 1]

        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = (
                self.interpolator[ind].warp_image(
                    self.images[ind].array,
                    self.knots[ind],
                )
            )

        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots, title="Merged: translation", **kwargs
            )
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: translation" for i in range(self.shape[0])],
                **kwargs,
            )

        return self

    # Affine alignment
    def align_affine(
        self,
        step: float = 0.01,
        num_tests: int = 9,
        refine: bool = True,
        upsample_factor: int = 8,
        max_image_shift: float | None = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Estimate affine drift from the first 2 images.
        """

        if not hasattr(self, "knots"):
            print(
                "\033[91mNo knots found — running .preprocess() with default settings.\033[0m"
            )
            self.preprocess()

        if num_tests % 2 == 0:
            raise ValueError("num_tests should be odd.")

        # Potential drift vectors
        vec = np.arange(-(num_tests - 1) / 2, (num_tests + 1) / 2)
        xx, yy = np.meshgrid(vec, vec, indexing="ij")
        keep = xx**2 + yy**2 <= (num_tests / 2) ** 2
        dxy = (
            np.vstack(
                (
                    xx[keep],
                    yy[keep],
                )
            ).T
            * step
        )

        # Measure cost function for linear drift vectors
        cost = np.zeros(dxy.shape[0])
        for a0 in tqdm(range(dxy.shape[0]), desc="Solving affine drift"):
            # updated knots
            knot_0 = self.knots[0].copy()
            u = np.arange(knot_0.shape[1]) - (knot_0.shape[1] - 1) / 2
            knot_0[0] += dxy[a0, 0] * u[:, None]
            knot_0[1] += dxy[a0, 1] * u[:, None]

            knot_1 = self.knots[1].copy()
            u = np.arange(knot_1.shape[1]) - (knot_1.shape[1] - 1) / 2
            knot_1[0] += dxy[a0, 0] * u[:, None]
            knot_1[1] += dxy[a0, 1] * u[:, None]

            im0, w0 = self.interpolator[0].warp_image(
                self.images[0].array,
                knot_0,
            )
            im1, w1 = self.interpolator[1].warp_image(
                self.images[1].array,
                knot_1,
            )
            # Cross correlation alignment
            shifts, image_shift = cross_correlation_shift(
                im0,
                im1,
                upsample_factor=upsample_factor,
                fft_input=False,
                fft_output=False,
                return_shifted_image=True,
                max_shift=max_image_shift,
            )
            cost[a0] = np.mean(np.abs(im0 - image_shift))

        # update all knots
        ind = np.argmin(cost)
        for a0 in range(self.shape[0]):
            u = np.arange(self.knots[a0].shape[1]) - (self.knots[a0].shape[1] - 1) / 2
            self.knots[a0][0] += dxy[ind, 0] * u[:, None]
            self.knots[a0][1] += dxy[ind, 1] * u[:, None]

        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[a0], self.weights_warped.array[ind] = (
                self.interpolator[ind].warp_image(
                    self.images[ind].array,
                    self.knots[ind],
                )
            )

        # Translation alignment
        self.align_translation(
            max_image_shift=max_image_shift,
            show_images=False,
            show_merged=False,
            show_knots=False,
        )

        # Error tracking
        self.calculate_error(1)

        # Affine drift refinement
        if refine:
            # Potential drift vectors
            dxy /= num_tests - 1

            # Measure cost function
            cost = np.zeros(dxy.shape[0])
            for a0 in tqdm(range(dxy.shape[0]), desc="Refining affine drift"):
                # updated knots

                knot_0 = self.knots[0].copy()
                u = np.arange(knot_0.shape[1]) - (knot_0.shape[1] - 1) / 2
                knot_0[0] += dxy[a0, 0] * u[:, None]
                knot_0[1] += dxy[a0, 1] * u[:, None]

                knot_1 = self.knots[1].copy()
                u = np.arange(knot_1.shape[1]) - (knot_1.shape[1] - 1) / 2
                knot_1[0] += dxy[a0, 0] * u[:, None]
                knot_1[1] += dxy[a0, 1] * u[:, None]

                im0, w0 = self.interpolator[0].warp_image(
                    self.images[0].array,
                    knot_0,
                )
                im1, w1 = self.interpolator[1].warp_image(
                    self.images[1].array,
                    knot_1,
                )
                # Cross correlation alignment
                shifts, image_shift = cross_correlation_shift(
                    im0,
                    im1,
                    upsample_factor=upsample_factor,
                    fft_input=False,
                    fft_output=False,
                    return_shifted_image=True,
                    max_shift=max_image_shift,
                )
                cost[a0] = np.mean(np.abs(im0 - image_shift))

            # update all knots
            ind = np.argmin(cost)
            for a0 in range(self.shape[0]):
                u = (
                    np.arange(self.knots[a0].shape[1])
                    - (self.knots[a0].shape[1] - 1) / 2
                )
                self.knots[a0][0] += dxy[ind, 0] * u[:, None]
                self.knots[a0][1] += dxy[ind, 1] * u[:, None]

        # Regenerate images
        for ind in range(self.shape[0]):
            self.images_warped.array[ind], self.weights_warped.array[ind] = (
                self.interpolator[ind].warp_image(
                    self.images[ind].array,
                    self.knots[ind],
                )
            )

        # Translation alignment
        self.align_translation(
            max_image_shift=max_image_shift,
            show_images=False,
            show_merged=False,
            show_knots=False,
        )

        # Error tracking
        self.calculate_error(1)

        # Plots
        kwargs.pop("title", None)
        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: affine",
                **kwargs,
            )
        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: affine" for i in range(self.shape[0])],
                **kwargs,
            )

        return self

    # non-rigid alignment
    def align_nonrigid(
        self,
        num_iterations: int = 4,
        max_optimize_iterations: int = 10,
        regularization_sigma_px=1.0,
        regularization_poly_order: int = 1,
        regularization_max_image_shift_px: Optional[float] = None,
        solve_individual_rows: bool = True,
        max_image_shift: float | None = 32,
        show_merged: bool = True,
        show_images: bool = False,
        show_knots: bool = True,
        **kwargs,
    ):
        """
        Non-rigid drift correction.
        """

        if not hasattr(self, "knots"):
            print(
                "\033[91mNo knots found — running .preprocess() with default settings.\033[0m"
            )
            self.preprocess()

        for iterations in tqdm(
            range(num_iterations),
            desc="Solving nonrigid drift",
        ):
            for ind in range(self.shape[0]):
                image_ref = np.delete(self.images_warped.array, ind, axis=0).mean(
                    axis=0
                )

                knots_init = self.knots[ind]
                shape_knots = knots_init.shape

                if solve_individual_rows:
                    knots_updated = np.zeros_like(knots_init)

                    for row_ind in range(knots_init.shape[1]):
                        x0 = knots_init[:, row_ind, :].ravel()

                        def cost_function(x):
                            knots_row = x.reshape(shape_knots[0], shape_knots[2])
                            xa, ya = self.interpolator[ind].transform_rows(knots_row)

                            xf = np.clip(np.floor(xa).astype(int), 0, self.shape[1] - 2)
                            yf = np.clip(np.floor(ya).astype(int), 0, self.shape[2] - 2)
                            dx = xa - xf
                            dy = ya - yf

                            warped = (
                                image_ref[xf, yf] * (1 - dx) * (1 - dy)
                                + image_ref[xf + 1, yf] * dx * (1 - dy)
                                + image_ref[xf, yf + 1] * (1 - dx) * dy
                                + image_ref[xf + 1, yf + 1] * dx * dy
                            )

                            residual = warped - self.images[ind].array[row_ind, :]
                            return np.sum(residual**2)

                        # Run optimization
                        options = (
                            {"maxiter": max_optimize_iterations}
                            if max_optimize_iterations is not None
                            else {}
                        )
                        result = minimize(
                            cost_function, x0, method="L-BFGS-B", options=options
                        )
                        knots_updated[:, row_ind, :] = result.x.reshape((2, -1))

                else:
                    x0 = knots_init.ravel()

                    def cost_function(x):
                        knots = x.reshape(shape_knots)
                        xa, ya = self.interpolator[ind].transform_coordinates(knots)

                        xf = np.clip(np.floor(xa).astype(int), 0, self.shape[1] - 2)
                        yf = np.clip(np.floor(ya).astype(int), 0, self.shape[2] - 2)
                        dx = xa - xf
                        dy = ya - yf

                        warped = (
                            image_ref[xf, yf] * (1 - dx) * (1 - dy)
                            + image_ref[xf + 1, yf] * dx * (1 - dy)
                            + image_ref[xf, yf + 1] * (1 - dx) * dy
                            + image_ref[xf + 1, yf + 1] * dx * dy
                        )

                        residual = warped - self.images[ind].array
                        return np.sum(residual**2)

                    # Run optimization
                    options = (
                        {"maxiter": max_optimize_iterations}
                        if max_optimize_iterations is not None
                        else {}
                    )
                    result = minimize(
                        cost_function, x0, method="L-BFGS-B", options=options
                    )
                    knots_updated = result.x.reshape(shape_knots)

                # apply max shift regularization if needed
                if regularization_max_image_shift_px is not None:
                    knots_shift = knots_updated - self.knots[ind]
                    knots_dist = np.sqrt(np.sum(knots_shift**2, axis=0))
                    sub = knots_dist > regularization_max_image_shift_px
                    knots_updated[0][sub] = (
                        self.knots[ind][0][sub]
                        + knots_shift[0][sub]
                        * regularization_max_image_shift_px
                        / knots_dist[sub]
                    )
                    knots_updated[1][sub] = (
                        self.knots[ind][1][sub]
                        + knots_shift[1][sub]
                        * regularization_max_image_shift_px
                        / knots_dist[sub]
                    )

                # apply smoothness regularization if needed
                if regularization_sigma_px is not None and regularization_sigma_px > 0:
                    knots_smoothed = knots_updated.copy()

                    for dim in range(knots_updated.shape[0]):
                        x = np.arange(knots_updated.shape[1])
                        for knot_ind in range(knots_updated.shape[2]):
                            y = knots_updated[dim, :, knot_ind]

                            coefs = np.polyfit(x, y, deg=regularization_poly_order)
                            trend = np.polyval(coefs, x)

                            # Remove trend, filter, add back
                            residual = y - trend
                            residual_smooth = gaussian_filter(
                                residual, sigma=regularization_sigma_px
                            )
                            knots_smoothed[dim, :, knot_ind] = residual_smooth + trend

                    knots_updated = knots_smoothed

                # Update knots with optimized values
                self.knots[ind] = knots_updated

            # Update images
            for ind in range(self.shape[0]):
                self.images_warped.array[ind], self.weights_warped.array[ind] = (
                    self.interpolator[ind].warp_image(
                        self.images[ind].array,
                        self.knots[ind],
                    )
                )

            # Translation alignment
            self.align_translation(
                max_image_shift=max_image_shift,
                show_images=False,
                show_merged=False,
                show_knots=False,
            )

            # Error tracking
            self.calculate_error(2)

        if show_merged:
            self.plot_merged_images(
                show_knots=show_knots,
                title="Merged: non-rigid",
                **kwargs,
            )

        if show_images:
            self.plot_transformed_images(
                show_knots=show_knots,
                title=[f"Image {i}: non-rigid" for i in range(self.shape[0])],
                **kwargs,
            )

        return self

    def generate_corrected_image(
        self,
        upsample_factor: int = 2,
        output_original_shape: bool = True,
        fourier_filter: bool = True,
        kde_sigma: float = 0.5,
        show_image: bool = True,
        **kwargs,
    ):
        """
        Generate the final output image, after drift correction.
        """

        # init
        stack_corr = np.zeros(
            (
                self.shape[0],
                np.round(self.shape[1] * upsample_factor).astype("int"),
                np.round(self.shape[2] * upsample_factor).astype("int"),
            )
        )

        if kde_sigma is None:
            kde_sigma = self.kde_sigma

        # Update images
        for ind in range(self.shape[0]):
            stack_corr[ind], _ = self.interpolator[ind].warp_image(
                self.images[ind].array,
                self.knots[ind],
                kde_sigma=kde_sigma,
                upsample_factor=upsample_factor,
            )

        if fourier_filter:
            # Apply fourier filtering
            kx = np.fft.fftfreq(stack_corr.shape[1])[:, None]
            ky = np.fft.fftfreq(stack_corr.shape[2])[None, :]
            kr = np.sqrt(kx**2 + ky**2)

            stack_fft = np.fft.fft2(stack_corr)
            weights = np.zeros_like(stack_corr)

            for ind in range(stack_corr.shape[0]):
                weights[ind] = np.divide(
                    np.abs(self.scan_fast[ind, 0] * kx + self.scan_fast[ind, 1] * ky),
                    kr,
                    where=kr > 0.0,
                )
                weights[ind][0, 0] = 1.0
                stack_fft[ind] *= weights[ind]

            weights_sum = np.sum(weights, axis=0)
            image_corr_fft = np.divide(
                np.sum(stack_fft, axis=0),
                weights_sum,
                where=weights_sum > 0.0,
            )

        else:
            image_corr_fft = np.fft.fft2(np.mean(stack_corr, axis=0))

        if output_original_shape:
            image_corr_fft = (
                fourier_cropping(image_corr_fft, self.shape[-2:]) / upsample_factor**2
            )

        image_corr = Dataset2d.from_array(
            np.real(np.fft.ifft2(image_corr_fft)),
            name="drift corrected image",
            origin=self.images[0].origin,
            sampling=self.images[0].sampling,
            units=self.images[0].units,
        )

        if show_image:
            fig, ax = image_corr.show(**kwargs)

        return image_corr

    def calculate_error(
        self,
        mode,
    ):
        # Estimate current error
        images_mean = np.mean(self.images_warped.array, axis=0)
        sig_diff = np.mean(
            np.abs(self.images_warped.array - images_mean[None, :, :]), axis=(1, 2)
        )

        # Error vector
        error_current = np.hstack((mode, np.mean(sig_diff), sig_diff))

        # Initialize or append to error tracking array
        if not hasattr(self, "error_track"):
            self.error_track = error_current[None, :]  # initialize with first row
        else:
            self.error_track = np.vstack((self.error_track, error_current))

    def plot_transformed_images(self, show_knots: bool = True, **kwargs):
        fig, ax = show_2d(
            list(self.images_warped.array),
            **kwargs,
        )
        if show_knots:
            for a0 in range(self.shape[0]):
                x = self.knots[a0][0]
                y = self.knots[a0][1]
                ax[a0].plot(
                    y,
                    x,
                    color="r",
                )

    def plot_merged_images(self, show_knots: bool = True, **kwargs):
        """
        Plot the current transformed images, with knot overlays.
        """
        fig, ax = show_2d(
            self.images_warped.array.mean(0),
            **kwargs,
        )
        if show_knots:
            for a0 in range(self.shape[0]):
                x = self.knots[a0][0]
                y = self.knots[a0][1]
                ax.plot(
                    y,
                    x,
                )


class DriftInterpolator:
    def __init__(
        self,
        input_shape,
        output_shape,
        scan_fast,
        scan_slow,
        pad_value,
        kde_sigma,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.scan_fast = scan_fast
        self.scan_slow = scan_slow
        self.pad_value = pad_value
        self.kde_sigma = kde_sigma

        self.rows_input = np.arange(input_shape[0])
        self.cols_input = np.arange(input_shape[1])
        self.u = np.linspace(0, 1, input_shape[1])

    def transform_rows(
        self,
        knots_row: NDArray,
    ):
        num_knots = knots_row.shape[-1]
        basis = np.linspace(0, 1, num_knots)

        if num_knots == 1:
            xa = knots_row[0] + self.u[None, :] * self.scan_fast[0] * (
                self.input_shape[0] - 1
            )
            ya = knots_row[1] + self.u[None, :] * self.scan_fast[1] * (
                self.input_shape[1] - 1
            )
        elif num_knots == 2:
            xa = interp1d(basis, knots_row[0], kind="linear", assume_sorted=True)(
                self.u
            )
            ya = interp1d(basis, knots_row[1], kind="linear", assume_sorted=True)(
                self.u
            )
        else:
            kind = "quadratic" if num_knots == 3 else "cubic"
            xa = interp1d(
                basis,
                knots_row[0],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)
            ya = interp1d(
                basis,
                knots_row[1],
                kind=kind,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self.u)

        return xa, ya

    def transform_coordinates(
        self,
        knots: NDArray,
    ):
        num_knots = knots.shape[-1]

        if num_knots == 1:
            # vectorized version for speed
            xa, ya = self.transform_rows(knots)
        else:
            xa = np.zeros(self.input_shape)
            ya = np.zeros(self.input_shape)
            for i in range(self.input_shape[0]):
                xa[i], ya[i] = self.transform_rows(knots[:, i])

        return xa, ya

    def warp_image(
        self,
        image: NDArray,
        knots: NDArray,  # shape: (2, rows, num_knots)
        kde_sigma=None,
        output_shape=None,
        pad_value=None,
        upsample_factor=None,
    ) -> NDArray:
        xa, ya = self.transform_coordinates(
            knots,
        )

        if kde_sigma is None:
            kde_sigma = self.kde_sigma

        if output_shape is None:
            output_shape = self.output_shape

        if pad_value is None:
            pad_value = self.pad_value

        if upsample_factor is None:
            upsample_factor = 1.0

        image_interp, weight_interp = bilinear_kde(
            xa=xa * upsample_factor,  # rows
            ya=ya * upsample_factor,  # cols
            values=image,
            output_shape=np.round(np.array(output_shape) * upsample_factor).astype(
                "int"
            ),
            kde_sigma=kde_sigma * upsample_factor,
            pad_value=pad_value,
            return_pix_count=True,
        )

        return image_interp, weight_interp
