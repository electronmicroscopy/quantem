from typing import Any

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