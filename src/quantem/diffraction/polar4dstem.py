# from collections.abc import Sequence
from typing import List, Optional, Union

# import matplotlib.pyplot as plt
import numpy as np
# from numpy.typing import NDArray
# from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, map_coordinates
# from scipy.optimize import minimize
from tqdm import tqdm
import torch

# from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4d import Dataset4d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.cnn2d import MultiChannelCNN2d
# from quantem.core.utils.compound_validators import (
#     validate_list_of_dataset2d,
#     validate_pad_value,
# )
# from quantem.core.utils.imaging_utils import (
#     bilinear_kde,
#     cross_correlation_shift,
#     fourier_cropping,
# )
# from quantem.core.utils.validators import ensure_valid_array
# from quantem.core.visualization import show_2d


class Polar4DStem(AutoSerialize):
    """
    
    """

    _token = object()

    def __init__(
        self,
        dataset_cartesian: Dataset4d,
        peaks_cartesian: List[float, float],  # cartesian peaks
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use Polar4DStem.from_data() or .from_file() to instantiate this class."
            )

        self._dataset_cartesian = dataset_cartesian

    @classmethod
    def from_file(
        cls,
        file_path: str,
        file_type: str | None = None,
    ) -> "Polar4DStem":
        dataset_cartesian = Dataset4d.from_file(file_path, file_type=file_type)
        return cls.from_data(
            dataset_cartesian,
        )

    @classmethod
    def from_data(
        cls,
        dataset_cartesian: Dataset4d,
    ) -> "Polar4DStem":
        return cls(
            dataset_cartesian=dataset_cartesian,
            _token=cls._token,
        )
    
    def find_peaks_model(self):
        pass

    def preprocess(
        self, all_peaks,
    ):
        """ Find center of image through brightest peak, return polar transform of data and peaks"""
        centers = self.find_central_beams_4d(self._dataset_cartesian)
        self.polar_data = self.polar_transform_4d(self._dataset_cartesian, centers=centers)
        self.polar_peaks = self.polar_transform_peaks(all_peaks)

    def refine_peaks(self,):
        pass
    
    def find_central_beam_vectorized(self, image, sigma=2, search_radius=None, com_radius=5):
        """
        Find central beam by smoothing image, taking search window near geometrical center of image,
        and then taking a center of mass around this point within a certain radius for refining center.

        Parameters:
        -----------
        image : ndarray, shape (det_y, det_x)
            2D image
        sigma : float
            Gaussian smoothing sigma
        search_radius : int, optional
            Search radius around geometric center. Defaults to +/- 1/4 of smallest image shape dimension
        com_radius : int
            Radius for center of mass refinement
        """

        H, W = image.shape

        # Gaussian smoothing
        smoothed = gaussian_filter(image, sigma=sigma)

        # Geometric center
        geo_center_y = H // 2
        geo_center_x = W // 2

        # If no search radius provided take window based on shape of image
        if search_radius is None:
            search_radius = min(H, W) // 4

        # Search region bounds
        y_min = max(0, geo_center_y - search_radius)
        y_max = min(H, geo_center_y + search_radius)
        x_min = max(0, geo_center_x - search_radius)
        x_max = min(W, geo_center_x + search_radius)

        # Find max in search region
        search_region = smoothed[y_min:y_max, x_min:x_max]
        # Gives x and y coordinates in terms of search region shape
        local_max_y, local_max_x = np.unravel_index(
            np.argmax(search_region), 
            search_region.shape
        )
        # getting absolute x and y positions in terms of image shape
        max_y = y_min + local_max_y
        max_x = x_min + local_max_x

        # Create coordinate grids
        y_grid = np.arange(H)[:, np.newaxis]
        x_grid = np.arange(W)[np.newaxis, :]

        # Circular mask
        distances_sq = (y_grid - max_y)**2 + (x_grid - max_x)**2
        circular_mask = distances_sq <= com_radius**2

        # Center of mass
        masked_image = image * circular_mask
        total_intensity = masked_image.sum()

        if total_intensity > 0:
            com_y = (masked_image * y_grid).sum() / total_intensity
            com_x = (masked_image * x_grid).sum() / total_intensity
            return (com_y, com_x)
        else:
            # If divide by zero error then just return the max position in smoothed image
            return (float(max_y), float(max_x))

    def find_central_beams_4d(self, data, sigma=2, search_radius=None, com_radius=5, 
                                    use_tqdm=True):
        """
        Fast central beam finding for entire 4D dataset.
        
        Parameters:
        -----------
        data : ndarray, shape (scan_y, scan_x, det_y, det_x)
            4D-STEM dataset
        sigma : float
            Gaussian smoothing sigma
        search_radius : int, optional
            Search radius around geometric center
        com_radius : int
            Radius for center of mass refinement
        use_tqdm : bool
            Show progress bar
        
        Returns:
        --------
        centers : ndarray, shape (scan_y, scan_x, 2)
            Center coordinates (y, x) for each scan position
        """
        from tqdm import tqdm
        
        scan_y, scan_x, det_y, det_x = data.shape
        centers = np.zeros((scan_y, scan_x, 2))
        
        iterator = tqdm(range(scan_y), disable=not use_tqdm, desc="Finding centers")
        
        for i in iterator:
            for j in range(scan_x):
                centers[i, j] = self.find_central_beam_vectorized(
                    data[i, j], 
                    sigma=sigma,
                    search_radius=search_radius,
                    com_radius=com_radius
                )
        
        return centers

    def polar_transform_4d(self, data, centers, num_r=None, num_theta=360):
        """
        Perform polar transform on the last two axes of a 4D array.
        
        Parameters:
        -----------
        data : ndarray, shape (N, M, H, W)
            4D input array where H, W are the axes to transform
        centers : ndarray, shape (N, M)
            Center of each diffraction pattern (usually determined by central beam)
        num_r : int, optional
            Number of radial bins. If None, uses max radius
        num_theta : int, optional
            Number of angular bins (default: 360)
        
        Returns:
        --------
        polar_data : ndarray, shape (N, M, num_r, num_theta)
            Polar-transformed array
        """
        N, M, H, W = data.shape
        

        
        # Transform each 2D slice
        for i in range(N):
            for j in range(M):
                center_y, center_x = centers[i, j]        
                # Maximum radius
                max_radius = np.sqrt(center_y**2 + center_x**2)
                if num_r is None:
                    num_r = int(max_radius)
                
                # Create polar coordinate arrays
                theta = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
                r = np.linspace(0, max_radius, num_r)
                
                # Create meshgrid
                r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
                
                # Convert polar to Cartesian coordinates
                y_coords = center_y + r_grid * np.sin(theta_grid)
                x_coords = center_x + r_grid * np.cos(theta_grid)
                
                # Initialize output
                polar_data = np.zeros((N, M, num_r, num_theta))
                # Use map_coordinates for interpolation
                polar_data[i, j] = map_coordinates(
                    data[i, j], 
                    [y_coords, x_coords], 
                    order=1,  # linear interpolation
                    mode='constant',
                    cval=0.0
                )
        
        return polar_data

    def polar_transform_peaks(self, all_peaks, centers):
        N, M = all_peaks.shape
        polar_peaks = np.empty((N, M), dtype=object)  # Will be a ragged array. N, M, 1 where entry is a (L, 2) array with L peaks, 2 coordinates for each
        for i in range(N):
            for j in range(M):
                center_y, center_x = centers[i, j]    
                peaks = all_peaks[i, j]

                # Check if no peaks for DP
                if peaks is None or len(peaks) == 0:
                    polar_peaks[i, j] = np.zeros((0, 2))  
                    continue

                dy = peaks[:, 0] - center_y
                dx = peaks[:, 1] - center_x
                r = np.sqrt((dy)**2 + (dx)**2)
                theta = np.arctan2(dy, dx)
                polar_peaks[i, j] = np.column_stack([r, theta])

        return polar_peaks

#     def get_origin_single_dp(dp, r, rscale=1.2):
#     """
#     Find the origin for a single diffraction pattern, assuming (a) there is no beam stop,
#     and (b) the center beam contains the highest intensity.

#     Args:
#         dp (ndarray): the diffraction pattern
#         r (number): the approximate disk radius
#         rscale (number): factor by which `r` is scaled to generate a mask

#     Returns:
#         (2-tuple): The origin
#     """
#     Q_Nx, Q_Ny = dp.shape
#     _qx0, _qy0 = np.unravel_index(np.argmax(gaussian_filter(dp, r)), (Q_Nx, Q_Ny))
#     qyy, qxx = np.meshgrid(np.arange(Q_Ny), np.arange(Q_Nx))
#     mask = np.hypot(qxx - _qx0, qyy - _qy0) < r * rscale
#     qx0, qy0 = get_CoM(dp * mask)
#     return qx0, qy0


# # for a datacube


# def get_origin(
#     datacube,
#     r=None,
#     rscale=1.2,
#     dp_max=None,
#     mask=None,
#     fast_center=False,
#     remove_NaN=False,
# ):
#     """
#     Find the origin for all diffraction patterns in a datacube, assuming (a) there is no
#     beam stop, and (b) the center beam contains the highest intensity. Stores the origin
#     positions in the Calibration associated with datacube, and optionally also returns
#     them.

#     Args:
#         datacube (DataCube): the data
#         r (number or None): the approximate radius of the center disk. If None (default),
#             tries to compute r using the get_probe_size method.  The data used for this
#             is controlled by dp_max.
#         rscale (number): expand 'r' by this amount to form a mask about the center disk
#             when taking its center of mass
#         dp_max (ndarray or None): the diffraction pattern or dp-shaped array used to
#             compute the center disk radius, if r is left unspecified. Behavior depends
#             on type:

#                 * if ``dp_max==None`` (default), computes and uses the maximal
#                   diffraction pattern. Note that for a large datacube, this may be a
#                   slow operation.
#                 * otherwise, this should be a (Q_Nx,Q_Ny) shaped array
#         mask (ndarray or None): if not None, should be an (R_Nx,R_Ny) shaped
#                     boolean array. Origin is found only where mask==True, and masked
#                     arrays are returned for qx0,qy0
#         fast_center: (bool)
#             Skip the center of mass refinement step.
#         remove_NaN: (bool)
#             If True, sets NaN to mean value

#     Returns:
#         (2-tuple of (R_Nx,R_Ny)-shaped ndarrays): the origin, (x,y) at each scan position
#     """
#     if r is None:
#         if dp_max is None:
#             dp_max = np.max(datacube.data, axis=(0, 1))
#         else:
#             assert dp_max.shape == (datacube.Q_Nx, datacube.Q_Ny)
#         r, _, _ = get_probe_size(dp_max)

#     qx0 = np.zeros((datacube.R_Nx, datacube.R_Ny))
#     qy0 = np.zeros((datacube.R_Nx, datacube.R_Ny))
#     qyy, qxx = np.meshgrid(np.arange(datacube.Q_Ny), np.arange(datacube.Q_Nx))

#     if mask is None:
#         for rx, ry in tqdmnd(
#             datacube.R_Nx,
#             datacube.R_Ny,
#             desc="Finding origins",
#             unit="DP",
#             unit_scale=True,
#         ):
#             dp = datacube.data[rx, ry, :, :]
#             _qx0, _qy0 = np.unravel_index(
#                 np.argmax(gaussian_filter(dp, r, mode="nearest")),
#                 (datacube.Q_Nx, datacube.Q_Ny),
#             )
#             if fast_center:
#                 qx0[rx, ry], qy0[rx, ry] = _qx0, _qy0
#             else:
#                 _mask = np.hypot(qxx - _qx0, qyy - _qy0) < r * rscale
#                 qx0[rx, ry], qy0[rx, ry] = get_CoM(dp * _mask)

#     else:
#         assert mask.shape == (datacube.R_Nx, datacube.R_Ny)
#         assert mask.dtype == bool
#         qx0 = np.ma.array(
#             data=qx0, mask=np.zeros((datacube.R_Nx, datacube.R_Ny), dtype=bool)
#         )
#         qy0 = np.ma.array(
#             data=qy0, mask=np.zeros((datacube.R_Nx, datacube.R_Ny), dtype=bool)
#         )
#         for rx, ry in tqdmnd(
#             datacube.R_Nx,
#             datacube.R_Ny,
#             desc="Finding origins",
#             unit="DP",
#             unit_scale=True,
#         ):
#             if mask[rx, ry]:
#                 dp = datacube.data[rx, ry, :, :]
#                 _qx0, _qy0 = np.unravel_index(
#                     np.argmax(gaussian_filter(dp, r, mode="nearest")),
#                     (datacube.Q_Nx, datacube.Q_Ny),
#                 )
#                 if fast_center:
#                     qx0[rx, ry], qy0[rx, ry] = _qx0, _qy0
#                 else:
#                     _mask = np.hypot(qxx - _qx0, qyy - _qy0) < r * rscale
#                     qx0.data[rx, ry], qy0.data[rx, ry] = get_CoM(dp * _mask)
#             else:
#                 qx0.mask, qy0.mask = True, True

#     if remove_NaN:
#         qx0[np.isnan(qx0)] = np.mean(qx0[~np.isnan(qx0)])
#         qy0[np.isnan(qy0)] = np.mean(qy0[~np.isnan(qy0)])

#     # return
#     mask = np.ones(datacube.Rshape, dtype=bool)
#     return qx0, qy0, mask