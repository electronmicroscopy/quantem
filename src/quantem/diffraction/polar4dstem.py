# from collections.abc import Sequence
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
# from numpy.typing import NDArray
# from scipy.interpolate import interp1d
from numpy._core.multiarray import NEEDS_INIT
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, label, sum as label_sum, center_of_mass
# from scipy.optimize import minimize
from tqdm import tqdm
import torch
from skimage.feature import blob_dog, blob_log, blob_doh

# from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4d import Dataset4d
from quantem.core.datastructures import Vector
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.cnn2d import MultiChannelCNN2d
from quantem.core.utils.polar import polar_transform_vector, cartesian_transform_vector

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
        resized_cartesian_data: Union[List, NDArray, Vector],
        peaks_cartesian: Union[List, NDArray, Vector],  # cartesian peaks
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use Polar4DStem.from_data() or .from_file() to instantiate this class."
            )

        self._dataset_cartesian = dataset_cartesian
        self.resized_cartesian_data = resized_cartesian_data
        self._peaks_cartesian = peaks_cartesian

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
        resized_cartesian_data: Union[List, NDArray],
        peaks_cartesian: Union[List, NDArray],
    ) -> "Polar4DStem":
        return cls(
            dataset_cartesian=dataset_cartesian,
            resized_cartesian_data=resized_cartesian_data,
            peaks_cartesian=peaks_cartesian,
            _token=cls._token,
        )
    
    def find_peaks_model(self):
        pass

    def preprocess(self):
        """ Find center of image through brightest peak, return polar transform of data and peaks"""
        self.image_centers = self.find_central_beams_4d(self.resized_cartesian_data)
        self.polar_data = self.polar_transform_4d(self.resized_cartesian_data, centers=self.image_centers)
        self.polar_peaks = self.polar_transform_peaks(cartesian_peaks=self._peaks_cartesian, centers=self.image_centers)

    def refine_peaks(self,):
        pass
    
    def find_central_beam_with_size_check(self, image, min_size=9, brightness_threshold=0.99999, com_radius=5):
        """
        Find central beam by taking the maximum value, checking its size, and refining with center of mass.

        Parameters:
        -----------
        image : ndarray, shape (det_y, det_x)
            2D image
        min_size : int
            Minimum size (in pixels) for a spot to be considered the central beam
        brightness_threshold : float
            Fraction of max intensity to use for segmentation
        com_radius : int
            Radius around the brightest pixel to use for center of mass calculation

        Returns:
        --------
        center : tuple
            (y, x) coordinates of the beam center
        """
        # Find the maximum intensity
        max_intensity = np.max(image)
        
        # Create a binary image of bright spots
        binary = image > (max_intensity * brightness_threshold)
        
        # Label connected components
        labeled, num_features = label(binary)
        
        # Find sizes of all labeled regions
        sizes = label_sum(binary, labeled, range(1, num_features + 1))
        
        # Find the label of the largest region that meets the minimum size
        valid_labels = np.where(sizes >= min_size)[0] + 1  # +1 because labels start at 1
        
        if len(valid_labels) == 0:
            # If no valid regions found, fall back to brightest pixel
            center_y, center_x = np.unravel_index(np.argmax(image), image.shape)
        else:
            largest_valid_label = valid_labels[np.argmax(sizes[valid_labels - 1])]
            
            # Find the brightest pixel within this region
            mask = (labeled == largest_valid_label)
            masked_image = image * mask
            center_y, center_x = np.unravel_index(np.argmax(masked_image), image.shape)
        
        # Refine position using center of mass
        y_min = max(0, int(center_y) - com_radius)
        y_max = min(image.shape[0], int(center_y) + com_radius + 1)
        x_min = max(0, int(center_x) - com_radius)
        x_max = min(image.shape[1], int(center_x) + com_radius + 1)
        
        local_region = image[y_min:y_max, x_min:x_max]
        local_com_y, local_com_x = center_of_mass(local_region)
        
        refined_center_y = y_min + local_com_y
        refined_center_x = x_min + local_com_x

        # # Visualization
        # plt.figure(figsize=(15, 10))
        
        # plt.subplot(2, 3, 1)
        # plt.imshow(binary, cmap='gray')
        # plt.title('Binary Image')
        
        # plt.subplot(2, 3, 2)
        # plt.imshow(labeled, cmap='nipy_spectral')
        # plt.title('Labeled Image')
        
        # plt.subplot(2, 3, 3)
        # plt.imshow(masked_image, cmap='viridis')
        # plt.title('Masked Image')
        # plt.scatter(center_x, center_y, color='red', s=50, marker='x')
        
        # plt.subplot(2, 3, 4)
        # plt.imshow(image, cmap='viridis')
        # plt.title('Original Image')
        # plt.scatter(center_x, center_y, color='red', s=50, marker='x', label='Max Intensity')
        # plt.scatter(refined_center_x, refined_center_y, color='white', s=50, marker='o', label='Refined (CoM)')
        # plt.legend()
        
        # plt.subplot(2, 3, 5)
        # plt.imshow(local_region, cmap='viridis')
        # plt.title('Local Region for CoM')
        # plt.scatter(local_com_x, local_com_y, color='white', s=50, marker='o')
        
        # plt.tight_layout()
        # plt.show()
        
        # print(f"Initial center: ({center_y:.2f}, {center_x:.2f})")
        # print(f"Refined center: ({refined_center_y:.2f}, {refined_center_x:.2f})")
        # input()
        
        return (float(refined_center_y), float(refined_center_x))

    def find_central_beams_4d(self, data, min_size=9, brightness_threshold=0.5, use_tqdm=True):
        """
        Fast central beam finding for entire 4D dataset.
        
        Parameters:
        -----------
        data : ndarray, shape (scan_y, scan_x, det_y, det_x)
            4D-STEM dataset
        min_size : int
            Minimum size (in pixels) for a spot to be considered the central beam
        brightness_threshold : float
            Fraction of max intensity to use for segmentation
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
                centers[i, j] = self.find_central_beam_with_size_check(
                    data[i, j],
                    min_size=min_size,
                    brightness_threshold=brightness_threshold
                )
        
        return centers

    def polar_transform_4d(self, data, centers, num_r=None, num_theta=360, use_tqdm: bool=True):
        """
        Perform polar transform on the last two axes of a 4D array.
        
        Parameters:
        -----------
        data : ndarray, shape (N, M, H, W)
            4D input array where H, W are the axes to transform
        centers : ndarray, shape (N, M, 2)
            Center of each diffraction pattern (usually determined by central beam)
        num_r : int, optional
            Number of radial bins. If None, uses max radius across all patterns
        num_theta : int, optional
            Number of angular bins (default: 360)
        
        Returns:
        --------
        polar_data : ndarray, shape (N, M, num_r, num_theta)
            Polar-transformed array
        """
        N, M, H, W = data.shape
        
        # Calculate consistent max_radius across entire dataset
        # This distance will be used to store and display data in terms of bins as well as natural units
        # Distance from closest center to (0,0)
        dist_to_origin = np.sqrt(centers[..., 0]**2 + centers[..., 1]**2).min()
        # Distance from furthest center to bottom-right corner
        dist_to_corner = np.sqrt((H-1 - centers[..., 0])**2 + (W-1 - centers[..., 1])**2).max()
        max_radius = max(dist_to_origin, dist_to_corner)
        # If no radial bins set, use this distance to be number of radial bins
        if num_r is None:
            num_r = int(np.ceil(max_radius))
        
        # Store value for reference later in use when displaying data or returning values
        self.max_radius = max_radius
        self.num_radial_bins = num_r
        self.num_annular_bins = num_theta
        # Pre-calculate radial array (same for all patterns now)
        r = np.linspace(0, max_radius, num_r)
        theta = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
        
        polar_data = np.zeros((N, M, num_r, num_theta))
        
        # Transform each 2D slice
        iterator = tqdm(range(N), disable=not use_tqdm, desc="Transforming data")
        for i in iterator:
            for j in range(M):
                center_y, center_x = centers[i, j]
                
                # Create meshgrid
                r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
                
                # Convert polar to Cartesian coordinates
                y_coords = center_y + r_grid * np.sin(theta_grid)
                x_coords = center_x + r_grid * np.cos(theta_grid)
                
                # Use map_coordinates for interpolation
                polar_data[i, j] = map_coordinates(
                    data[i, j], 
                    [y_coords, x_coords], 
                    order=1,  # linear interpolation
                    mode='constant',
                    cval=0.0
                )
        
        return polar_data

    def polar_transform_peaks(self, cartesian_peaks, centers, use_tqdm: bool=True):
        polar_peaks = polar_transform_vector(cartesian_vector=cartesian_peaks, centers=centers, use_tqdm=use_tqdm)

        # Does polar transform of peaks in native units
        # N, M = cartesian_peaks.shape
        # polar_peaks = Vector.from_shape(
        #     shape=(N, M),
        #     fields=["r", "theta"],
        #     name="polar_peaks_vector",
        #     units=["Pixels", "Radians"],
        # )
        # # polar_peaks = np.empty((N, M), dtype=object)  # Will be a ragged array. N, M, 1 where entry is a (L, 2) array with L peaks, 2 coordinates for each
        # iterator = tqdm(range(N), disable=not use_tqdm, desc="Transforming peaks")
        # for i in iterator:
        #     for j in range(M):
        #         center_y, center_x = centers[i, j]    
        #         peaks = cartesian_peaks[i, j]

        #         # Check if no peaks for DP
        #         if peaks is None or len(peaks) == 0:
        #             polar_peaks.set_data(np.zeros((0, 2)), i, j)
        #             # polar_peaks[i, j] = np.zeros((0, 2))  
        #             continue

        #         dy = peaks[:, 0] - center_y
        #         dx = peaks[:, 1] - center_x
        #         r = np.sqrt((dy)**2 + (dx)**2)
        #         theta = np.arctan2(dy, dx)
        #         theta = np.mod(theta + np.pi, 2 * np.pi)
        #         polar_coords = np.column_stack([r, theta])
        #         polar_peaks.set_data(polar_coords, i, j)
        #         # polar_peaks[i, j] = np.column_stack([r, theta])

        return polar_peaks

    def save_polar_peaks(self, filepath):
        np.save(filepath, self.polar_peaks)

    def save_polar_data(self, filepath):
        np.save(filepath, self.polar_data)

    def load_polar_peaks(self, filepath):
        polar_peaks = np.load(filepath, allow_pickle=True)
        if isinstance(polar_peaks, np.ndarray) and polar_peaks.dtype == object and polar_peaks.size == 1:
            polar_peaks = polar_peaks.item()
        self.polar_peaks = polar_peaks

    def load_polar_data(self, filepath):
        polar_data = np.load(filepath, allow_pickle=True)
        if isinstance(polar_data, np.ndarray) and polar_data.dtype == object and polar_data.size == 1:
            polar_data = polar_data.item()
        self.polar_data = np.load(filepath, allow_pickle=True)


    # Facade function
    # "detect_peaks_ML" function, no need to do this

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