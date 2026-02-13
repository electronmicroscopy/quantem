from typing import Any

import numpy as np

from scipy.interpolate import interp1d

from scipy.ndimage import median_filter

from numpy.typing import NDArray

from quantem.spectroscopy import Dataset3dspectroscopy


class Dataset3deels(Dataset3dspectroscopy):
    """An EELS dataset class that inherits from Dataset3dspectroscopy.

    This class represents a scanning transmission electron microscopy (STEM) dataset,
    where the data consists of a 3D array with dimensions (energy, scan_y, scan_x).
    The first dimension represents the energy, while the latter
    two dimensions represent real space sampling.

    """

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

    def calculate_background_iterative(self, spectrum):
        """
        Subtract background typical for EELS using iterative Gaussian fitting.
        This method isolates the continuum background from the low-loss region.

        WARNING: Only use with EELS data! Will remove peaks if used with EDS.

        Parameters
        ----------
        spectrum : ndarray
            1D EELS spectrum
        energy_axis : ndarray
            Energy axis corresponding to spectrum

        Returns
        -------
        ndarray
            Background-subtracted spectrum
        """

        from scipy.ndimage import gaussian_filter
        from scipy.stats import norm

        # Smooth for better fitting
        spec_smooth = gaussian_filter(spectrum, sigma=1.0)
        pixel_vals = spec_smooth.copy()

        # Iteratively fit Gaussian to low-intensity values (the continuum)
        # Remove outliers (edge peaks) iteratively
        num_iterations = 10
        cutoff = 3  # +/- 3 sigma

        for _ in range(num_iterations):
            mu, std = norm.fit(pixel_vals)
            if std == 0:
                break
            # Keep only values within +/- 3 sigma (removes edge contributions)
            lower = mu - cutoff * std
            upper = mu + cutoff * std
            pixel_vals = pixel_vals[(pixel_vals >= lower) & (pixel_vals <= upper)]

        # Subtract the estimated background level
        background_fit = mu

        return background_fit

    def calibrate_zero_loss_peak(self, center_guess=None, search_window=10):
        """
        Calibrate the energy axis by centering the zero loss peak at 0 eV.
        Finds the ZLP at every pixel, fits a 2D plane to the ZLP positions,
        and shifts each spectrum individually so the ZLP sits at 0, while aligning
        all ZLPs to the same channel index, allowing a single origin to correctly 
        calibrate the entire dataset. 

        Parameters
        ----------
        center_guess : float or None
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

        n_energy, n_y, n_x = self.array.shape

        dE = float(self.sampling[0])
        E0 = float(self.origin[0])
        energy_axis = E0 + np.arange(n_energy) * dE

        # --- Build ZLP position map ---
        # For every pixel, find the energy where the ZLP sits.
        # A median filter is applied to each spectrum first to remove
        # hot pixels (cosmic rays, detector glitches) that could be
        # brighter than the ZLP and fool the peak finder.
        # If center_guess is provided, only look within a window
        # of search_window channels around that energy.
        # If center_guess is None, just find the tallest peak.

        zlp_map = np.zeros((n_y, n_x))

        if center_guess is not None:
            guess_index = int(round((center_guess - E0) / dE))
            lo = max(guess_index - search_window, 0)
            hi = min(guess_index + search_window + 1, n_energy)

        for iy in range(n_y):
            for ix in range(n_x):
                spectrum = median_filter(self.array[:, iy, ix], size=3)

                if center_guess is None:
                    peak_index = np.argmax(spectrum)
                else:
                    peak_index = lo + np.argmax(spectrum[lo:hi])

                zlp_map[iy, ix] = E0 + peak_index * dE

        # --- Fit a 2D plane to the ZLP map ---
        # The plane equation is: zlp_energy(y, x) = a*y + b*x + c
        # This smooths out noisy per-pixel ZLP measurements by assuming
        # the drift varies linearly across the scan area.

        y_coords, x_coords = np.meshgrid(
            np.arange(n_y), np.arange(n_x), indexing="ij"
        )
        y_flat = y_coords.ravel()
        x_flat = x_coords.ravel()
        z_flat = zlp_map.ravel()

        A = np.column_stack([y_flat, x_flat, np.ones(len(y_flat))])
        coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
        a, b, c = coeffs

        zlp_plane = a * y_coords + b * x_coords + c

        # --- Shift each spectrum so the ZLP lands at 0 eV ---
        # For each pixel, subtract its plane-predicted ZLP position from
        # the energy axis, then interpolate the spectrum back onto the
        # original energy grid. This physically moves the data so all
        # ZLPs align at the same channel index.

        corrected_array = np.zeros_like(self.array)

        for iy in range(n_y):
            for ix in range(n_x):
                shift = zlp_plane[iy, ix]
                shifted_energy = energy_axis - shift
                interpolator = interp1d(
                    shifted_energy,
                    self.array[:, iy, ix],
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                corrected_array[:, iy, ix] = interpolator(energy_axis)

        return Dataset3deels.from_array(
            array=corrected_array,
            name=self.name,
            sampling=self.sampling,
            origin=self.origin,
            units=self.units,
        )