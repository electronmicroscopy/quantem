from typing import Any

from numpy.typing import NDArray

from quantem.spectroscopy import Dataset3dspectroscopy


class Dataset3deds(Dataset3dspectroscopy):
    """An EDS dataset class that inherits from Dataset3dspectroscopy.

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
        """Initialize a 3D EDS dataset.

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

#Separate function for background subtraction to return a subtracted EDS spectra

    def calculate_background_powerlaw(self, roi=None, energy_range=None, ignore_range=None, mask=None):

            import numpy as np
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            from sklearn.decomposition import PCA

            from quantem.core.datastructures.dataset3d import Dataset3d
            from quantem.core.utils.validators import ensure_valid_array

    # TEMP- COPY OF ROI SELECTION CODE FROM SHOW_MEAN_SPECTRUM---------------------------

        # ADJUST ROI BASED ON GIVEN FLAGS -----------------------------------------------
            # Parse ROI parameter
            if roi is None:
                # Full image
                y, x, dy, dx = 0, 0, int(self.shape[1]), int(self.shape[2])
            elif len(roi) == 2:
                # Single pixel [y, x]
                y, x, dy, dx = int(roi[0]), int(roi[1]), 1, 1
            elif len(roi) == 4:
                # Full ROI [y, x, dy, dx] with None support for defaults
                y_val, x_val, dy_val, dx_val = roi
                
                # Handle None values with defaults
                y = 0 if y_val is None else int(y_val)
                x = 0 if x_val is None else int(x_val)
                dy = int(self.shape[1]) - y if dy_val is None else int(dy_val)
                dx = int(self.shape[2]) - x if dx_val is None else int(dx_val)
            else:
                raise ValueError("roi must be None, [y, x], or [y, x, dy, dx] (with None for defaults)")


        # VALIDATE ROI BOUNDS ---------------------------------------------------------------------------
            errs = []
            Ymax = int(self.shape[1])
            Xmax = int(self.shape[2])

            # type/NaN checks (optional if you already cast to int above)
            for name, val in (("y", y), ("x", x), ("dy", dy), ("dx", dx)):
                if val is None:
                    errs.append(f"{name} is None (missing after normalization).")
           
            # if any None, bail early to avoid arithmetic errors
            if errs:
                raise ValueError("Invalid ROI:\n - " + "\n - ".join(errs))

            # basic constraints
            if y < 0:            errs.append(f"y={y} < 0")
            if x < 0:            errs.append(f"x={x} < 0")
            if dy < 1:           errs.append(f"dy={dy} < 1")
            if dx < 1:           errs.append(f"dx={dx} < 1")

            # starts within image
            if y >= Ymax:        errs.append(f"y start {y} out of bounds [0, {Ymax-1}]")
            if x >= Xmax:        errs.append(f"x start {x} out of bounds [0, {Xmax-1}]")

            # ends within image
            end_y = y + dy
            end_x = x + dx
            if end_y > Ymax:     errs.append(f"y+dy = {end_y} exceeds height {Ymax}")
            if end_x > Xmax:     errs.append(f"x+dx = {end_x} exceeds width {Xmax}")

            if errs:
                raise ValueError("Invalid ROI:\n - " + "\n - ".join(errs))

    #TEMP- COPY OF SPECTRUM CALCULATION FROM SHOW_MEAN_SPECTRUM---------------------------
            
        # SPECTRUM CALCULATION --------------------------------------------------------------

            dE = float(self.sampling[0])
            E0 = float(self.origin[0]) if hasattr(self, "origin") else 0.0
            E  = E0 + dE * np.arange(self.shape[0])

            # MASK HANDLING ---------------------------------------------------------------------
            if mask is not None:
                # Convert to ndarray and validate
                mask = np.asarray(mask)
                
                # Check that it's a proper ndarray
                if not isinstance(mask, np.ndarray):
                    raise TypeError(f"Mask must be a numpy ndarray, got {type(mask)}")
                
                # Check dimensions - must be 1D
                if mask.ndim != 1:
                    raise ValueError(f"Mask must be 1-dimensional, got {mask.ndim}D array with shape {mask.shape}")
                
                # Convert to bool dtype and validate
                if mask.dtype != bool:
                    try:
                        mask = mask.astype(bool)
                    except (ValueError, TypeError):
                        raise TypeError(f"Mask cannot be converted to boolean dtype from {mask.dtype}")
                
                # Check shape matches energy axis
                arr = np.asarray(self.array, dtype=float)
                if mask.shape != (arr.shape[0],):
                    raise ValueError(f"Mask shape {mask.shape} does not match energy axis shape ({arr.shape[0]},)")
                
                arr = arr[mask, :, :]  # select only masked energy channels
                spec = arr.sum(axis=(1,2)) if arr.shape[0] > 0 else np.zeros(0)
                E = E[mask]  # Mask the energy axis as well
            else:
                spec = np.empty(self.shape[0], dtype=float)
                for k in range(self.shape[0]):
                    img = np.asarray(self.array[k], dtype=float)
                    roi = img[y:y+dy, x:x+dx]
                    if roi.size == 0:
                        raise ValueError("ROI is empty; check y/x/dy/dx.")
                    spec[k] = roi.mean()

            # Store ignore_range for later use in element line filtering
            if ignore_range is None:
                ignore_range = [0, 0.25]  # Default: ignore 0-0.25 keV for element lines only

    #-----------------------------------------------------------------------------------

    #POWER LAW BACKGROUND SUBTRACTION

    #TEMP- PORT OF SUBTRACT_BACKGROUND_EDS BODY

            """
            Subtract power-law background typical for EDS Bremsstrahlung.
            Uses a conservative approach with heavy smoothing to avoid creating artifacts.
            
            Parameters
            ----------
            spectrum : ndarray
                1D spectrum
            energy_axis : ndarray
                Energy axis corresponding to spectrum
                
            Returns
            -------
            ndarray
                Background-subtracted spectrum
            """
            from scipy.ndimage import gaussian_filter
            
            # Use a larger window for more conservative background estimation
            window_size = 15  # Larger window = smoother, less aggressive
            background = np.zeros_like(spec)
            half_window = window_size // 2
            
            # Estimate background from sliding minimum
            for i in range(len(spec)):
                start = max(0, i - half_window)
                end = min(len(spec), i + half_window + 1)
                # Use percentile instead of minimum for more robustness
                background[i] = np.percentile(spec[start:end], 10)
            
            # Apply heavy smoothing to avoid creating artificial features
            background = gaussian_filter(background, sigma=5.0)
            
            # Be very conservative - only subtract 80% of estimated background
            # This prevents over-subtraction that creates artificial peaks
            background = background * 0.8
            
            # Ensure background doesn't exceed spectrum
            background = np.minimum(background, spec* 0.9)

            subtracted_mean_spectrum = np.maximum(spec - background, 0)
            
    #TEMP- PORT OF SPECTRUM PLOTTING CODE FROM SHOW_MEAN_SPECTRUM
        # PLOTTING ---------------------------------------------------------------------------
            
            # Create subplot layout: image on left, spectrum on right
            fig, (ax_spec) = plt.subplots(1, 1, figsize=(12, 4))

            
            
            # RIGHT PLOT: Show spectrum
            ax_spec.plot(E, subtracted_mean_spectrum, linewidth=1.5)
            ax_spec.set_xlabel("Energy (keV)")
            ax_spec.set_ylabel("Intensity")
            ax_spec.set_title(f"Spectrum from ROI [{y}:{y+dy}, {x}:{x+dx}]")
            ax_spec.grid(True, alpha=0.1)

            fig.tight_layout()
            plt.show()

            return background