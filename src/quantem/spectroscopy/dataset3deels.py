from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

from quantem.core.visualization import show_2d
from quantem.spectroscopy.dataset3dspectroscopy import Dataset3dspectroscopy
from quantem.spectroscopy.spectroscopy_visualzitions import (
    interpret_thickness_quality as _visualize_thickness_quality,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    plot_absolute_thickness as _visualize_absolute_thickness,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    plot_absolute_zlp_shift as _visualize_absolute_zlp_shift,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    plot_dual_eels_picker as _visualize_dual_eels_picker,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    plot_quantem_diagnostic as _visualize_quantem_diagnostic,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    plot_zlp_drift_diagnostics as _visualize_zlp_drift_diagnostics,
)
from quantem.spectroscopy.spectroscopy_visualzitions import (
    visualize_thickness_windows as _visualize_thickness_windows,
)


class Dataset3deels(Dataset3dspectroscopy):
    """An EELS dataset class that inherits from Dataset3dspectroscopy.

    This class represents a scanning transmission electron microscopy (STEM) dataset,
    where the data consists of a 3D array with dimensions (scan_row, scan_col, energy).
    The first two dimensions represent real space sampling, while the last dimension
    represents the energy axis.

    """

    element_info = None
    element_info_path = "eels_edges.csv"
    dataset_type = "EELS"

    plot_absolute_zlp_shift = _visualize_absolute_zlp_shift
    visualize_thickness_windows = _visualize_thickness_windows
    interpret_thickness_quality = _visualize_thickness_quality
    plot_absolute_thickness = _visualize_absolute_thickness
    plot_dual_eels_picker = _visualize_dual_eels_picker
    plot_quantem_diagnostic = _visualize_quantem_diagnostic
    plot_zlp_drift_diagnostics = _visualize_zlp_drift_diagnostics

    def __init__(
        self,
        array: NDArray | Any,
        name: str,
        origin: NDArray | tuple | list | float | int,
        sampling: NDArray | tuple | list | float | int,
        units: list[str] | tuple | list,
        signal_units: str = "arb. units",
        _token: object | None = None,
    ):
        """Initialize a 3D EELS dataset.

        Parameters
        ----------
        array : NDArray | Any
            The underlying 3D array data
        name : str
            A descriptive name for the dataset
        origin : NDArray | tuple | list | float | int
            The origin coordinates for each dimension
        sampling : NDArray | tuple | list | float | int
            The sampling rate/spacing for each dimension
        units : list[str] | tuple | list
            Units for each dimension
        signal_units : str, optional
            Units for the array values, by default "arb. units"
        _token : object | None, optional
            Token to prevent direct instantiation, by default None
        """
        super().__init__(
            array=array,
            name=name,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
            _token=_token,
        )
        self._virtual_images = {}
        self.dataset_type = "eels"

    def calculate_background_iterative(
        self, spectrum, smoothing_kernel_sigma=1.0, sigma_cutoff=3.0
    ):
        """
        Subtract background typical for EELS using iterative Gaussian fitting.
        This method isolates the continuum background from the low-loss region.

        WARNING: Only use with EELS data! Will remove peaks if used with XEDS.

        Parameters
        ----------
        spectrum : ndarray
            1D EELS spectrum
        smoothing_kernel_sigma:

        Returns
        -------
        ndarray
            Background-subtracted spectrum
        """

        from scipy.ndimage import gaussian_filter
        from scipy.stats import norm

        # Smooth for better fitting
        spec_smooth = gaussian_filter(spectrum, smoothing_kernel_sigma)
        pixel_vals = spec_smooth.copy()

        # Iteratively fit Gaussian to low-intensity values (the continuum)
        # Remove outliers (edge peaks) iteratively
        num_iterations = 10

        for _ in range(num_iterations):
            mu, std = norm.fit(pixel_vals)
            if std == 0:
                break
            # Keep only values within +/- the number of standard deviations specificed by sigma_cutoff(removes edge contributions)
            lower = mu - sigma_cutoff * std
            upper = mu + sigma_cutoff * std
            pixel_vals = pixel_vals[(pixel_vals >= lower) & (pixel_vals <= upper)]

        # Subtract the estimated background level
        background_fit = mu

        return background_fit

    # ========== NEW METHOD: Background subtraction for limited pre-edge data ==========

    def subtract_background_limited_preedge(
        self,
        target_edge,
        pre_edge_range=None,
        method="polynomial",
        polynomial_degree=2,
        show=True,
        return_dataset=True,
    ):
        """
        Background subtraction optimized for limited pre-edge data.

        This method bypasses the 10-30% window_size constraint in the standard
        subtract_background() method, allowing background fitting when only a
        small pre-edge region is available (common in high-loss only acquisitions).

        Parameters
        ----------
        target_edge : float
            Energy of the edge onset (eV)
            Examples: 285 for C K-edge, 532 for O K-edge, 284 for C K-edge
        pre_edge_range : tuple of float, optional
            Explicit (start, end) energies in eV for pre-edge fitting window.
            If None, automatically uses all available data before edge.
            Example: (519, 527) for O K-edge when data starts at 518 eV
        method : str, optional
            Background fitting method:
            - 'polynomial': Polynomial fit (default, most stable for short ranges)
            - 'linear': Linear fit (equivalent to polynomial degree=1)
            - 'powerlaw': Power-law A*E^(-r) (needs longer pre-edge, may fail)
        polynomial_degree : int, optional
            Degree of polynomial (1=linear, 2=quadratic, 3=cubic). Default is 2.
            Only used when method='polynomial'.
        show : bool, optional
            Display before/after visualization. Default True.
        return_dataset : bool, optional
            If True, return Dataset3deels. If False, return numpy array. Default True.

        Returns
        -------
        Dataset3deels or ndarray
            Background-subtracted data

        Raises
        ------
        ValueError
            If pre-edge region is insufficient or target_edge is out of range
        RuntimeError
            If fitting fails (typically with powerlaw on limited data)

        Notes
        -----
        **When to use this method:**
        - Data starts close to the edge (limited pre-edge region)
        - Standard subtract_background() fails with window_size error
        - High-loss only acquisitions (no low-loss data)
        - Cropped energy ranges

        **Recommended methods by pre-edge size:**
        - < 10 eV: method='linear' (most stable)
        - 10-20 eV: method='polynomial', degree=2
        - > 20 eV: method='polynomial', degree=2-3, or 'powerlaw'

        **Comparison to GMS background subtraction:**
        This mimics the GMS "Fit Background" function but without the
        window percentage constraint, using direct energy range specification.

        Examples
        --------
        >>> # O K-edge at 532 eV, data starts at 518 eV (only 14 eV pre-edge)
        >>> eels_sub = eels_hl.subtract_background_limited_preedge(
        ...     target_edge=532,
        ...     method='polynomial',
        ...     polynomial_degree=2
        ... )

        >>> # Specify exact pre-edge window
        >>> eels_sub = eels_hl.subtract_background_limited_preedge(
        ...     target_edge=532,
        ...     pre_edge_range=(519, 527),  # 8 eV window
        ...     method='linear',
        ...     show=True
        ... )

        >>> # C K-edge with enough pre-edge for power-law
        >>> eels_sub = eels_hl.subtract_background_limited_preedge(
        ...     target_edge=285,
        ...     pre_edge_range=(200, 280),  # 80 eV window
        ...     method='powerlaw'
        ... )

        See Also
        --------
        subtract_background : Standard method with window_size percentage
        powerlaw_backgroundfit_eels : Direct power-law fitting function
        """

        import warnings

        from scipy.optimize import curve_fit

        energy = self.energy_axis
        mean_spec = self.calculate_mean_spectrum()

        # ===== 1. Determine pre-edge fitting window =====
        if pre_edge_range is None:
            pre_edge_start = float(energy[0])
            pre_edge_end = float(target_edge - 5)
            print(f"Auto-detected pre-edge: {pre_edge_start:.1f} - {pre_edge_end:.1f} eV")
        else:
            pre_edge_start, pre_edge_end = float(pre_edge_range[0]), float(pre_edge_range[1])
            print(f"Using specified pre-edge: {pre_edge_start:.1f} - {pre_edge_end:.1f} eV")

        # ===== 2. Validate inputs =====
        if target_edge < energy[0] or target_edge > energy[-1]:
            raise ValueError(
                f"Target edge {target_edge} eV is outside data range "
                f"[{energy[0]:.1f}, {energy[-1]:.1f}] eV"
            )

        if pre_edge_start < energy[0]:
            raise ValueError(
                f"Pre-edge start {pre_edge_start:.1f} eV is before data start {energy[0]:.1f} eV"
            )

        if pre_edge_end >= target_edge:
            raise ValueError(
                f"Pre-edge end {pre_edge_end:.1f} eV must be before target edge {target_edge:.1f} eV"
            )

        available_preedge = pre_edge_end - pre_edge_start

        if available_preedge < 1:
            raise ValueError(
                f"Insufficient pre-edge region: only {available_preedge:.1f} eV available. "
                f"Need at least 1 eV for fitting."
            )

        # Warn if pre-edge is very limited
        if available_preedge < 10:
            warnings.warn(
                f"Limited pre-edge region ({available_preedge:.1f} eV). "
                f"Background fit may be unreliable. Consider method='linear' for stability.",
                UserWarning,
            )

        # ===== 3. Extract pre-edge data =====
        mask = (energy >= pre_edge_start) & (energy <= pre_edge_end)
        E_window = energy[mask]
        I_window = mean_spec[mask]

        n_points = len(E_window)
        print(f"Pre-edge region: {available_preedge:.1f} eV ({n_points} data points)")

        if n_points < 3:
            raise ValueError(
                f"Insufficient data points in pre-edge window: only {n_points} points. "
                f"Need at least 3 for fitting."
            )

        # ===== 4. Fit background using selected method =====
        if method == "linear" or (method == "polynomial" and polynomial_degree == 1):
            # Linear fit: y = m*x + b
            coeffs = np.polyfit(E_window, I_window, deg=1)
            background = np.polyval(coeffs, energy)
            fit_info = f"Linear: y = {coeffs[0]:.2e}*E + {coeffs[1]:.2e}"

        elif method == "polynomial":
            # Polynomial fit
            if polynomial_degree > n_points - 1:
                warnings.warn(
                    f"Polynomial degree {polynomial_degree} too high for {n_points} points. "
                    f"Using degree {n_points - 1} instead.",
                    UserWarning,
                )
                polynomial_degree = n_points - 1

            coeffs = np.polyfit(E_window, I_window, deg=polynomial_degree)
            background = np.polyval(coeffs, energy)
            fit_info = f"Polynomial (degree {polynomial_degree})"

        elif method == "powerlaw":
            # Power-law fit: A * E^(-r)
            def powerlaw(E, A, r):
                return A * (E ** (-r))

            # Initial guess
            A0 = I_window[0] * (E_window[0] ** 3)
            r0 = 3.0

            try:
                popt, _ = curve_fit(
                    powerlaw,
                    E_window,
                    I_window,
                    p0=[A0, r0],
                    bounds=([0, 0], [np.inf, 10]),
                    maxfev=5000,
                )
                background = powerlaw(energy, popt[0], popt[1])
                fit_info = f"Power-law: A={popt[0]:.2e}, r={popt[1]:.2f}"
            except RuntimeError as e:
                raise RuntimeError(
                    f"Power-law fit failed to converge with {available_preedge:.1f} eV pre-edge. "
                    f"Try method='polynomial' or 'linear' instead. Error: {e}"
                )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'linear', 'polynomial', or 'powerlaw'."
            )

        print(f"✓ Fit method: {fit_info}")

        # ===== 5. Subtract background from 3D data =====
        data_sub = np.maximum(self.array - background[None, None, :], 0)

        # ===== 6. Visualize if requested =====
        if show:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                f"Background Subtraction: {self.name}\nEdge at {target_edge} eV",
                fontsize=14,
                fontweight="bold",
            )

            # Before subtraction
            ax1.plot(energy, mean_spec, "k-", lw=1.5, label="Raw spectrum")
            ax1.plot(energy, background, "r--", lw=2, label=f"Background ({fit_info})")
            ax1.axvspan(
                pre_edge_start,
                pre_edge_end,
                alpha=0.2,
                color="green",
                label=f"Fit region ({available_preedge:.1f} eV)",
            )
            ax1.axvline(target_edge, color="orange", ls=":", lw=2, label="Edge onset")
            ax1.set_xlabel("Energy (eV)", fontsize=12)
            ax1.set_ylabel("Intensity", fontsize=12)
            ax1.set_title("Before Background Subtraction")
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

            # After subtraction
            subtracted_spec = mean_spec - background
            ax2.plot(energy, subtracted_spec, "b-", lw=1.5, label="Background-subtracted")
            ax2.axvline(target_edge, color="orange", ls=":", lw=2, label="Edge onset")
            ax2.axhline(0, color="gray", ls="--", alpha=0.5)
            ax2.set_xlabel("Energy (eV)", fontsize=12)
            ax2.set_ylabel("Intensity", fontsize=12)
            ax2.set_title("After Background Subtraction")
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        # ===== 7. Return result =====
        if return_dataset:
            result = Dataset3deels.from_array(
                data_sub,
                sampling=self.sampling,
                origin=self.origin,
                units=self.units,
                name=f"{self.name} (background subtracted)",
            )
            print(f"✓ Created background-subtracted dataset: {result.shape}")
            return result
        else:
            return data_sub

    def powerlaw_backgroundfit_eels(self, spectrum, energy_range, target_edge, window_size):
        """
        Using a window of the energy axis preceding the target edge, fit a power law function to use for background subtraction.
        The input window size should be 10-30% of the target edge energy.
        """

        energy_axis = self.energy_axis

        if energy_range is not None:
            energy_range[0] = np.maximum(energy_range[0], energy_axis[0])
            energy_range[1] = np.minimum(energy_range[1], energy_axis[-1])

            indices = np.where(
                (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
            )[0]
            energy_axis = energy_axis[indices]
        else:
            indices = np.arange(self.shape[2])

        # Check that input window size is between 10% and 30%

        if window_size < 10 or window_size > 30:
            raise ValueError("Invalid window size. Please input a value of between 10 and 30.")

        # Check that the target edge is within the energy range of the spectrum
        # and that a pre-edge region of size at least 10% of the target edge, ending 5 eV before the target edge
        # exists for pre-edge fitting.

        if target_edge < energy_axis[0] or target_edge > energy_axis[-1]:
            raise ValueError("Target edge is outside of energy range.")
        elif ((target_edge - 5) - target_edge * (window_size / 100)) < energy_axis[0]:
            raise ValueError(
                "Insufficient pre-edge background fitting region for this target edge and window size within given energy range."
            )

        # Fit power law function to spectrum within window region of the energy exis

        window_min_E = (target_edge - 5) - target_edge * (window_size / 100)
        window_max_E = target_edge - 5

        window_indices = np.where((energy_axis >= window_min_E) & (energy_axis <= window_max_E))[0]

        window_E = energy_axis[window_indices]
        window_I = spectrum[window_indices]

        def powerlaw_function(E, A, r):
            return A * (E ** (-r))

        popt, _ = curve_fit(powerlaw_function, window_E, window_I, maxfev=2000)
        background_fit = powerlaw_function(energy_axis, popt[0], popt[1])

        # Plot the region of the spectrum between user-specified energy range, overlaid with the background fit curve, with background estimation
        # window boundaries indicated

        fig, ax = plt.subplots()
        ax.plot(energy_axis, spectrum, label="spectrum", color="b")
        ax.plot(energy_axis, background_fit, label="background", color="r")
        ax.vlines(
            x=[window_min_E, window_max_E],
            ymin=0,
            ymax=np.max(spectrum),
            label="window limits",
            color="k",
            linestyle="dashed",
        )
        ax.legend()

        return background_fit

    def smooth_eels_rolling_average(self, roi=None, energy_range=None, mask=None, kernel_size=10):
        energy_axis = self.energy_axis

        if energy_range is not None:
            energy_range[0] = np.maximum(energy_range[0], energy_axis[0])
            energy_range[1] = np.minimum(energy_range[1], energy_axis[-1])

            indices = np.where(
                (energy_axis >= energy_range[0]) & (energy_axis <= energy_range[1])
            )[0]
            energy_axis = energy_axis[indices]
        else:
            indices = np.arange(self.shape[2])

        array3d_subrange = self.array[:, :, indices]

        kernel = np.ones(kernel_size) / kernel_size

        # For each probe position, convolve spectral data with smoothing kernel

        array3d_smoothed = np.zeros(array3d_subrange.shape)

        scan_row, scan_col, _n_energy = array3d_subrange.shape
        for i_row in range(scan_row):
            for i_col in range(scan_col):
                probe_spectrum = array3d_subrange[i_row, i_col, :]
                spectrum_smoothed = np.convolve(probe_spectrum, kernel, mode="same")
                array3d_smoothed[i_row, i_col, :] = spectrum_smoothed

        output_origin = np.array(self.origin, dtype=float, copy=True)
        output_origin[2] = energy_axis[0]
        smoothed_data3d = Dataset3deels.from_array(
            array=array3d_smoothed,
            sampling=self.sampling,
            origin=output_origin,
            units=self.units,
        )

        # Plot raw and smoothed mean spectra on the same set of axes

        mean_spectrum_raw = self.calculate_mean_spectrum(
            roi=roi,
            energy_range=energy_range,
            mask=mask,
        )
        mean_spectrum_smoothed = smoothed_data3d.calculate_mean_spectrum(
            roi=roi,
            energy_range=energy_range,
            mask=mask,
        )

        fig, ax = plt.subplots()
        ax.plot(energy_axis, mean_spectrum_raw, label="raw spectrum", color="b")
        ax.plot(energy_axis, mean_spectrum_smoothed, label="kernel-smoothed spectrum", color="r")
        ax.legend()

        return smoothed_data3d

    def measure_zlp_offset(
        self,
        zlp_guess_x=None,
        search_window=10,
        fit_window=0.8,
        median_filter_pixels=3,
        polynomial_order_rows=3,
        polynomial_order_columns=3,
        fit_to_plane=False,
        fit_to_polynomial=False,
        fit_zlp=True,
    ):
        """
        Measure ZLP offset at each pixel position by using a guess of ZLP posfitting each spectrum to a Gaussian

        Finds the difference between the maximum of the ZLP Gaussian fit and 0 eV at every pixel,
        and fits a 2D plane to measured ZLP offsets if fit_to_plane=True.

        Parameters
        ----------
        zlp_guess_x : float or None
            Expected energy position of the ZLP in eV. If None, uses the
            tallest peak in each spectrum as the ZLP. If provided, searches
            for the tallest peak within the search window around that energy.
        search_window : int
            Number of channels to search on either side of center_guess.
            Only used when center_guess is not None. Default is 10.

        Returns
        -------
        Dataset3deels
            New dataset with corrected energy calibration.

        """

        # Define Gaussian constraint to fit ZLP to
        def _gaussian_fit(x, A, mu, sigma):
            return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        def _plane_fit_2d(M, a, b, c):
            row, col = M
            return (a * row) + (b * col) + c

        def _polynomial_fit_2d(M, c00, c10, c01, c20, c11, c02):
            row, col = M
            return (
                c00
                + (c10 * row)
                + (c01 * col)
                + (c20 * row**2)
                + (c11 * row * col)
                + (c02 * col**2)
            )

        scan_row, scan_col, n_energy = self.array.shape
        energy_axis = self.energy_axis

        # For each pixel, measure the zlp position by fitting a Gaussian to the measured zero-loss signal and taking its center as the zlp position.

        zlp_measured = np.zeros((scan_row, scan_col))

        for i_row in range(scan_row):
            for i_col in range(scan_col):
                # Apply median filter to discount hot pixels that might spuriously produce the maximum intensity of the spectrum
                if median_filter_pixels > 0:
                    spec_filt = median_filter(self.array[i_row, i_col, :], median_filter_pixels)
                else:
                    spec_filt = self.array[i_row, i_col, :]

                if fit_zlp:
                    # Use initial guess for ZLP to define window for Gaussian fitting. If zlp_guess_x=None (default) use the maximum value of the spectrum
                    if zlp_guess_x is not None:
                        zlp_crude_idx = int(np.argmin(np.abs(energy_axis - zlp_guess_x)))
                    else:
                        zlp_crude_idx = int(np.argmax(spec_filt))

                    mu0 = energy_axis[zlp_crude_idx]

                    lo = mu0 - fit_window
                    hi = mu0 + fit_window

                    x_mask = (energy_axis >= lo) & (energy_axis <= hi)

                    xw = energy_axis[x_mask]
                    yw = spec_filt[x_mask]

                    A0 = spec_filt[zlp_crude_idx]
                    sigma0 = fit_window / 2

                    p0 = (A0, mu0, sigma0)

                    bounds = (
                        (
                            0.0,
                            lo,
                            1e-12,
                        ),
                        (
                            np.inf,
                            hi,
                            np.inf,
                        ),
                    )

                    popt, _ = curve_fit(_gaussian_fit, xw, yw, p0=p0, bounds=bounds)

                    zlp_measured[i_row, i_col] = popt[1]
                else:
                    zlp_crude_idx = int(np.argmax(spec_filt))
                    zlp_measured[i_row, i_col] = energy_axis[zlp_crude_idx]

        if fit_to_plane:
            # Fit a 2D plane to the array of measured ZLPs
            row_data, col_data = np.meshgrid(
                np.arange(scan_row), np.arange(scan_col), indexing="ij"
            )

            coord_data_unpacked = np.vstack((row_data.ravel(), col_data.ravel()))
            ydata_unpacked = zlp_measured.ravel()

            popt, _ = curve_fit(_plane_fit_2d, coord_data_unpacked, ydata_unpacked)

            zlp_plane_1d = _plane_fit_2d(coord_data_unpacked, popt[0], popt[1], popt[2])
            zlp_plane_2d = zlp_plane_1d.reshape(scan_row, scan_col)

            fig, _ = show_2d(
                [zlp_measured, zlp_plane_2d],
                cmap="magma",
                cbar=True,
                title=["Measured ZLP\n(mean of Gaussian fit)", "ZLP plane fit"],
                tight_layout=False,
            )
            fig.subplots_adjust(top=0.88, wspace=0.35)
            return zlp_plane_2d
        elif fit_to_polynomial:
            # Fit a 2D polynomial to the array of measured ZLPs
            row_data, col_data = np.meshgrid(
                np.arange(scan_row), np.arange(scan_col), indexing="ij"
            )

            coord_data_unpacked = np.vstack((row_data.ravel(), col_data.ravel()))
            ydata_unpacked = zlp_measured.ravel()

            popt, _ = curve_fit(_polynomial_fit_2d, coord_data_unpacked, ydata_unpacked)

            zlp_plane_1d = _polynomial_fit_2d(
                coord_data_unpacked, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
            )
            zlp_plane_2d = zlp_plane_1d.reshape(scan_row, scan_col)

            fig, _ = show_2d(
                [zlp_measured, zlp_plane_2d],
                cmap="magma",
                cbar=True,
                title=["Measured ZLP\n(mean of Gaussian fit)", "ZLP polynomial fit"],
                tight_layout=False,
            )
            fig.subplots_adjust(top=0.88, wspace=0.35)

        else:
            fig, _ = show_2d(
                [zlp_measured],
                cmap="magma",
                cbar=True,
                title=["Measured ZLP\n(mean of Gaussian fit)"],
                tight_layout=False,
            )
            fig.subplots_adjust(top=0.88)
            return zlp_measured

    def apply_zlp_correction(
        self,
        zlp_guess_x=None,
        zlp_shifts_array=None,
        fit_window=0.8,
        measure_offset=True,
        fit_to_plane=True,
        fit_to_polynomial=False,
        fit_zlp=True,
        return_3d_dataset=True,
        return_shifts=False,
        in_place=False,
    ):
        # Default behavior is to automatically call measure_zlp_offset to generate an array of ZLP shifts for each scan position.
        # Alternatively, a 2D array matching the scan_row and scan_col dimensions of the 3D dataset can be supplied as the value of zlp_shifts_array to skip this step.
        # If measure_offset is False and no 2D ZLP shifts array is provided, a scalar input for zlp_guess_x can be used to shift the energy axis at every scan position by that amount.
        if measure_offset:
            zlp_array = self.measure_zlp_offset(
                zlp_guess_x=zlp_guess_x,
                fit_window=fit_window,
                fit_to_plane=fit_to_plane,
                fit_to_polynomial=fit_to_polynomial,
                fit_zlp=fit_zlp,
            )
        elif zlp_shifts_array is not None:
            zlp_array = np.asarray(zlp_shifts_array, dtype=float)
            if zlp_array.shape != self.array.shape[0:2]:
                raise ValueError(
                    "Dimensions of input array for ZLP shifts do not match scan_row and scan_col dimensions of 3D spectroscopy dataset."
                )
        elif zlp_guess_x is not None:
            zlp_array = np.ones(self.array.shape[0:2], dtype=float) * zlp_guess_x
        else:
            raise ValueError(
                "measure_offset was set to False and no input argument for ZLP shifts was provided."
            )

        zlp_array = np.asarray(zlp_array, dtype=float)
        if not np.all(np.isfinite(zlp_array)):
            raise ValueError("ZLP shifts must contain only finite values.")

        # Initialize 3D array to populate with spectra aligned along the energy axis
        corrected_array = np.empty(self.array.shape, dtype=np.result_type(self.array.dtype, float))

        scan_row, scan_col, n_energy = self.array.shape

        energy_axis = self.energy_axis
        if np.all((zlp_array >= 0) & (zlp_array <= n_energy - 1)) and (
            np.min(zlp_array) < energy_axis[0] or np.max(zlp_array) > energy_axis[-1]
        ):
            zlp_array = np.interp(zlp_array, np.arange(n_energy), energy_axis)

        # Apply sub-channel ZLP shifts using 1D linear interpolation along the energy axis.
        for i_row in range(scan_row):
            for i_col in range(scan_col):
                spec = self.array[i_row, i_col, :]
                corrected_array[i_row, i_col, :] = np.interp(
                    energy_axis + zlp_array[i_row, i_col],
                    energy_axis,
                    spec,
                    left=np.nan,
                    right=np.nan,
                )

        # Remove all planes along energy axis containing NaN, to equalize spectra lengths across all scan positions
        mask = np.isnan(corrected_array).any(axis=(0, 1))
        aligned_data_3d = corrected_array[:, :, ~mask]
        new_Eaxis = energy_axis[~mask]

        if aligned_data_3d.shape[2] == 0:
            raise ValueError(
                "ZLP shifts leave no shared energy range after alignment. "
                "Check that zlp_shifts_array is in energy units, not channel indices."
            )

        new_origin = new_Eaxis[0]

        # Calculate mean spectra before and after correction for plotting
        mean_spectrum_raw = self.array.mean(axis=(0, 1))
        mean_spectrum_corrected = aligned_data_3d.mean(axis=(0, 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(energy_axis, mean_spectrum_raw, label="Raw mean spectrum", color="r")
        ax2.plot(new_Eaxis, mean_spectrum_corrected, label="ZLP-corrected spectrum", color="b")
        ax1.set_xlabel("Energy (eV)")
        ax1.set_ylabel("Intensity")
        ax1.grid(True, alpha=0.1)
        ax1.legend()
        ax2.set_xlabel("Energy (eV)")
        ax2.set_ylabel("Intensity")
        ax2.grid(True, alpha=0.1)
        ax2.legend()

        fig.tight_layout()

        if return_3d_dataset:
            corrected_dataset = Dataset3deels.from_array(
                array=aligned_data_3d,
                name=self.name,
                sampling=self.sampling,
                origin=new_origin,
                units=self.units,
            )
            if return_shifts:
                return corrected_dataset, zlp_array
            else:
                return corrected_dataset
        elif in_place:
            self.array = aligned_data_3d
            if return_shifts:
                return aligned_data_3d, zlp_array
            else:
                return aligned_data_3d
        else:
            if return_shifts:
                return aligned_data_3d, zlp_array
            else:
                return aligned_data_3d

    def correct_high_loss_energy_axis(
        self,
        ll_3d_dataset=None,
        zlp_guess_x=None,
        zlp_shifts_array=None,
        fit_window=0.8,
        measure_offset=True,
        fit_to_plane=True,
        fit_to_polynomial=False,
        fit_zlp=True,
        return_3d_dataset=True,
        return_shifts=False,
        in_place=False,
    ):
        """
        Applies ZLP correction to low-loss 3D EELS dataset and extends the computed shift at each
        pixel position to correct the corresponding high-loss 3D EELS dataset
        """
        if ll_3d_dataset is None:
            raise ValueError("No ll_3d_dataset provided for ZLP alignment")
        elif ll_3d_dataset.__class__ != Dataset3deels:
            raise ValueError("ll_3d_dataset input is not a Dataset3deels object")

        ll_corrected, ll_shifts = ll_3d_dataset.apply_zlp_correction(
            zlp_guess_x=zlp_guess_x,
            fit_window=fit_window,
            fit_to_plane=fit_to_plane,
            fit_to_polynomial=fit_to_polynomial,
            fit_zlp=fit_zlp,
            return_3d_dataset=False,
            return_shifts=True,
        )

        # Synchronize High-Loss energy origin based on median shift
        hl_corrected, hl_shifts = self.apply_zlp_correction(
            zlp_shifts_array=ll_shifts,
            measure_offset=False,
            return_3d_dataset=return_3d_dataset,
            return_shifts=True,
        )

        if return_shifts:
            return hl_corrected, hl_shifts
        else:
            return hl_corrected

    def calculate_thickness_log_ratio(
        self,
        zlp_window=10,
        median_filter_pixels=3,
        fit_zlp=True,
        zlp_guess_x=None,
        plot=True,
    ):
        """
        Calculates the relative thickness map (t/lambda) using the Log-Ratio method.
        """

        def _gaussian_fit(x, A, mu, sigma):
            return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        scan_row, scan_col, n_energy = self.array.shape
        energy_axis = self.energy_axis

        zlp_measured = np.zeros((scan_row, scan_col))

        for i_row in range(scan_row):
            for i_col in range(scan_col):
                # Apply median filter to discount hot pixels that might spuriously produce the maximum intensity of the spectrum
                if median_filter_pixels > 0:
                    spec_filt = median_filter(self.array[i_row, i_col, :], median_filter_pixels)
                else:
                    spec_filt = self.array[i_row, i_col, :]

                if fit_zlp:
                    # Use initial guess for ZLP to define window for Gaussian fitting. If zlp_guess_x=None (default) use the maximum value of the spectrum
                    if zlp_guess_x is not None:
                        zlp_crude_idx = int(np.argmin(np.abs(energy_axis - zlp_guess_x)))
                    else:
                        zlp_crude_idx = int(np.argmax(spec_filt))

                    mu0 = energy_axis[zlp_crude_idx]

                    lo = mu0 - zlp_window
                    hi = mu0 + zlp_window

                    x_mask = (energy_axis >= lo) & (energy_axis <= hi)

                    xw = energy_axis[x_mask]
                    yw = spec_filt[x_mask]

                    A0 = spec_filt[zlp_crude_idx]
                    sigma0 = zlp_window / 2

                    p0 = (A0, mu0, sigma0)

                    bounds = (
                        (
                            0.0,
                            lo,
                            1e-12,
                        ),
                        (
                            np.inf,
                            hi,
                            np.inf,
                        ),
                    )

                    popt, _ = curve_fit(_gaussian_fit, xw, yw, p0=p0, bounds=bounds)

                    zlp_measured[i_row, i_col] = popt[1]
                else:
                    zlp_crude_idx = int(np.argmax(spec_filt))
                    zlp_measured[i_row, i_col] = energy_axis[zlp_crude_idx]

        I_zlp = np.zeros((scan_row, scan_col))

        for i_row in range(scan_row):
            for i_col in range(scan_col):
                I_zlp[i_row, i_col] = np.sum(
                    self.array[
                        i_row,
                        i_col,
                        np.where(energy_axis >= (zlp_measured[i_row, i_col] - zlp_window / 2))[0][
                            0
                        ] : np.where(energy_axis <= (zlp_measured[i_row, i_col] + zlp_window / 2))[
                            0
                        ][-1],
                    ]
                )

        # print(f"Calculating thickness for {self.name}...")

        # Integrate intensity of ZLP and entire spectrum separately, and calculate t/lambda
        I_total = np.sum(self.array, axis=2)

        t_over_lambda = np.log1p((I_total) / (I_zlp))

        # Remove NaN matrix elements
        t_over_lambda = np.nan_to_num(t_over_lambda, nan=0.0, posinf=0.0, neginf=0.0)
        t_over_lambda = np.clip(t_over_lambda, 0, 4.0)

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            im = ax1.imshow(t_over_lambda, cmap="viridis", origin="upper")
            ax1.set_title(r"Relative Thickness Map ($t/\lambda$)")
            plt.colorbar(im, ax=ax1, label=r"$t/\lambda$")

            ax2.hist(t_over_lambda.flatten(), bins=50, color="steelblue", alpha=0.7, ec="k")
            ax2.axvline(
                np.mean(t_over_lambda),
                color="red",
                ls="--",
                label=f"Mean: {np.mean(t_over_lambda):.2f}",
            )
            ax2.set_title("Thickness Distribution")
            ax2.set_xlabel(r"$t/\lambda$")
            ax2.legend()

            plt.tight_layout()
            plt.show()

        return t_over_lambda
