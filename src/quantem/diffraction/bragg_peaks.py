# from collections.abc import Sequence
from typing import List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, label, sum as label_sum, center_of_mass
from tqdm import tqdm
import torch
from quantem.core.datastructures.dataset4d import Dataset4d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.core.ml.cnn2d import MultiChannelCNN2d
from quantem.core.datastructures import Vector
from quantem.core.utils.polar import polar_transform_vector, cartesian_transform_vector
from quantem.diffraction.peak_detection import detect_blobs


# TODO: Likely dataset4dSTEM rather than dataset4d input class
# Bragg peaks from crystalline vs polymer
# 
# TODO: "BraggPeaksPolymer" vs "BraggPeaksCrystal"
class BraggPeaksPolymer(AutoSerialize):
    """
    
    """

    _token = object()

    def __init__(
        self,
        dataset_cartesian: Dataset4dstem,
        compute_parameters: callable = None,
        normalize_data: callable = None,
        model: MultiChannelCNN2d = None,
        final_shape: Tuple[int, int] = (256, 256),
        device: str = 'cpu',
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use BraggPeaks.from_data() or .from_file() to instantiate this class."
            )

        self._dataset_cartesian = dataset_cartesian
        self._device = device
        self._final_shape = final_shape
        # Setting functions for normalization
        self.compute_parameters = compute_parameters
        self.normalize_data = normalize_data
        # To be set by class methods
        self.resized_cartesian_data = None
        self.peak_coordinates_cartesian = None
        self.peak_intensities = None
        self.image_centers = None
        self.polar_data = None
        self.polar_peaks = None
        self.max_radius = None
        self.num_radial_bins = None
        self.num_annular_bins = None

        if model is None:
            # Setup model
            input_channels = 1  # 1 for a greyscale image, 3 for RGB, 4 for RGBA, etc.
            k_size = 3
            # k_size = 7
            num_layers = 4
            start_filters = 16
            num_per_layer = 3
            # num_per_layer = 2
            use_skip_connections = True
            dtype = torch.float32
            dropout = 0.2     
            model = MultiChannelCNN2d(
                in_channels=input_channels,
                out_channels=2,
                start_filters=start_filters,
                num_layers=num_layers,
                num_per_layer=num_per_layer,
                use_skip_connections=use_skip_connections,
                dtype=dtype,
                dropout=dropout,
                final_activations=["sigmoid", "sigmoid"],
                conv_kernel_size=k_size,
            )
        self._model = model

    @property
    def model(self) -> MultiChannelCNN2d:
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model

    @classmethod
    def from_file(
        cls,
        file_path: str,
        file_type: str | None = None,
    ) -> "BraggPeaksPolymer":
        dataset_cartesian = Dataset4dstem.from_file(file_path, file_type=file_type)
        return cls.from_data(
            dataset_cartesian,
        )

    @classmethod
    def from_data(
        cls,
        dataset_cartesian: Dataset4dstem,
        device: str,
    ) -> "BraggPeaksPolymer":
        return cls(
            dataset_cartesian=dataset_cartesian,
            _token=cls._token,
            device=device,
        )
    
    def preprocess(self):
        self.resize_data()

    def resize_data(self, device:str = "cuda:1"):
        Ry, Rx, Qy, Qx = self._dataset_cartesian.shape
        scale_factor = (self._final_shape[0] * self._final_shape[1]) / (Qy * Qx)
        resized_data = np.zeros((Ry, Rx, self._final_shape[0], self._final_shape[1]))
        for i in tqdm(range(Ry), desc='rows'):
            inp = torch.tensor(self._dataset_cartesian[i].array, dtype=torch.float32).to(device)
            inp = torch.nn.functional.interpolate(inp[None, ...], size=self._final_shape, mode='bilinear', align_corners=False) * scale_factor
            resized_data[i, :, :, :] = inp.squeeze().detach().cpu().numpy()
        self.resized_cartesian_data = resized_data

    def set_model_weights(
        self,
        # path_to_model: str = None,
        path_to_weights: str = None,
        gpu_id: int = 1,
    ) -> "BraggPeaksPolymer":
        # if path_to_model is None:
            # path_to_model = ""
        if path_to_weights is None:
            path_to_weights = ""  # TODO: Load weights from cloud
        self._model.load_state_dict(torch.load(path_to_weights, weights_only=True, map_location=f"cuda:{gpu_id}"))
        self._model.to(gpu_id)

    def _postprocess_single(self, position_map, intensity_map, show=False):
        """Process a single 2D image"""
        # Find peaks with subpixel-refinement
        peak_coords, peak_position_signal_intensities, refinement_success = detect_blobs(
            position_map,
            sigma=1.0,  # Sigma for Gaussian smoothing used in processing
            threshold=0.25  # Threshold for strength of peak position signal to be valid peak
        )

        # If no peaks found, return empty lists
        if len(peak_coords) == 0:
            return np.array([]), np.array([])

        # map_coordinates expects coordinates in (row, col) = (y, x) order
        # peak_coords is already in [row, col] format from detect_blobs
        interpolated_intensities = map_coordinates(
            intensity_map, 
            peak_coords.T,  # Transpose to get [[all_y], [all_x]]
            order=1,  # 1 = bilinear interpolation
            mode='nearest'  # How to handle edges
        )
        
        # Optional: filter out peaks that were not successfully refined
        if np.any(refinement_success):
            pass
        
        if show:
            # Peak positions only
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(position_map, cmap='gray', alpha=0.8)
            ax.set_title("Input Position Map with Marked Peaks")
            ax.scatter(peak_coords[:, 1], peak_coords[:, 0], s=10, c='r', label="Peaks")
            ax.legend()
            plt.tight_layout()
            plt.show()

            # Peak positions with color representing intensity
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(position_map, cmap='gray', alpha=0.8)
            scatter = ax.scatter(
                peak_coords[:, 1],  # x coordinates
                peak_coords[:, 0],  # y coordinates
                c=interpolated_intensities,      # color by intensity
                s=10,
                cmap='turbo',    
                edgecolors='black', # white border for visibility
                linewidths=2,
                alpha=0.9,
                marker='o'
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Intensity', fontsize=12)
            ax.set_title('Peak Positions and Intensities', fontsize=14)
            ax.axis('off')
            plt.tight_layout()
            plt.show()

        return peak_coords, interpolated_intensities

    def find_peaks_model(
        self, 
        device: str = "cuda:1",
        n_normalize_samples: int = 1000,
        show_plots=False,
    ):
        Ry, Rx, Qy, Qx = self.resized_cartesian_data.shape
        peaks = Vector.from_shape(
            shape=(Ry, Rx),
            fields=["y", "x"],
            name="peaks_vector",
            units=["Pixels", "Pixels"],
        )
        # peaks = np.empty((Ry, Rx), dtype='object')
        intensities = Vector.from_shape(
            shape=(Ry, Rx),
            fields=["intensities"],
            name="intensities_vector",
            units=["Normalized"],
        )
        # intensities = np.empty((Ry, Rx), dtype='object')
        
        # Normalize - compute parameters from sample
        scan_shape = (Ry, Rx)
        n_positions = scan_shape[0] * scan_shape[1]
        n_normalize_samples = min(n_normalize_samples, n_positions)
        
        # Get stats patterns and move to CPU for parameter computation
        # Sample indices (use numpy instead of torch for this)
        flat_indices = np.random.permutation(n_positions)[:n_normalize_samples]
        scan_y_indices = flat_indices // scan_shape[1]
        scan_x_indices = flat_indices % scan_shape[1]

        # Get stats patterns directly as numpy (never touches GPU)
        stats_patterns = self.resized_cartesian_data[scan_y_indices, scan_x_indices]

        # Move to CPU and convert to numpy for compute_parameters
        # stats_patterns = torch.squeeze(stats_patterns).detach().cpu().numpy()
        median, iqr = self.compute_parameters(stats_patterns)
        normalized_dps_array = np.zeros((Ry, Rx, Qy, Qx))
        # Process row by row
        for i in tqdm(range(Ry), desc="rows"):
            # Get row data
            ins = torch.tensor(
                self.resized_cartesian_data[i], 
                dtype=torch.float32
            ).to(device).squeeze()  # Shape: (Rx, Qy, Qx)
            
            # Normalize using new function (works with tensors on GPU)
            dps_norm = self.normalize_data(ins, median, iqr)
            normalized_dps_array[i, :, :, :] = dps_norm.detach().cpu().numpy()
            ins = dps_norm[:, None, ...]  # Add channel dimension: (Rx, 1, Qy, Qx)
            
            # Pass through model
            outs = self.model(ins).detach().cpu().numpy()
            
            # Post-process each pattern in the row
            for r0 in range(outs.shape[0]):
                peak_coords, peak_intensities = self._postprocess_single(
                    outs[r0, 0], 
                    outs[r0, 1],
                    show=show_plots,
                )
                if len(peak_coords) > 0:
                    peaks.set_data(np.array(peak_coords), i, r0)
                    # peaks[i, r0] = np.array(peak_coords)
                    intensities.set_data(np.array(peak_intensities).reshape(-1, 1), i, r0)
                    # intensities[i, r0] = np.array(peak_intensities)
        
        print('done')
        self.peak_coordinates_cartesian = peaks
        self.peak_intensities = intensities
        self.normalized_dps_array = normalized_dps_array

    def save_peaks(self, filepath):
        np.save(filepath, self.peak_coordinates_cartesian)

    def load_peaks(self, filepath):
        peak_coordinates_cartesian = np.load(filepath, allow_pickle=True)
        if isinstance(peak_coordinates_cartesian, np.ndarray) and peak_coordinates_cartesian.dtype == object and peak_coordinates_cartesian.size == 1:
            peak_coordinates_cartesian = peak_coordinates_cartesian.item()
        self.peak_coordinates_cartesian = peak_coordinates_cartesian
    
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

    def process_polar(self):
        """ Find center of image through brightest peak, return polar transform of data and peaks"""
        self.image_centers = self.find_central_beams_4d(self.resized_cartesian_data)
        self.polar_data = self.polar_transform_4d(self.resized_cartesian_data, centers=self.image_centers)
        self.polar_peaks = self.polar_transform_peaks(cartesian_peaks=self.peak_coordinates_cartesian, centers=self.image_centers)

    def find_central_beam_with_size_check(self, image, min_size=9, brightness_threshold=0.99999, com_radius=5, show_plots=False):
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
        if show_plots:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.imshow(binary, cmap='gray')
            plt.title('Binary Image')
            
            plt.subplot(2, 3, 2)
            plt.imshow(labeled, cmap='nipy_spectral')
            plt.title('Labeled Image')
            
            plt.subplot(2, 3, 3)
            plt.imshow(masked_image, cmap='viridis')
            plt.title('Masked Image')
            plt.scatter(center_x, center_y, color='red', s=50, marker='x')
            
            plt.subplot(2, 3, 4)
            plt.imshow(image, cmap='viridis')
            plt.title('Original Image')
            plt.scatter(center_x, center_y, color='red', s=50, marker='x', label='Max Intensity')
            plt.scatter(refined_center_x, refined_center_y, color='white', s=50, marker='o', label='Refined (CoM)')
            plt.legend()
            
            plt.subplot(2, 3, 5)
            plt.imshow(local_region, cmap='viridis')
            plt.title('Local Region for CoM')
            plt.scatter(local_com_x, local_com_y, color='white', s=50, marker='o')
            
            plt.tight_layout()
            plt.show()
            
            print(f"Initial center: ({center_y:.2f}, {center_x:.2f})")
            print(f"Refined center: ({refined_center_y:.2f}, {refined_center_x:.2f})")
            input()
        
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

    def polar_transform_peaks(self, cartesian_peaks, centers, use_tqdm: bool=True):
        polar_peaks = polar_transform_vector(cartesian_vector=cartesian_peaks, centers=centers, use_tqdm=use_tqdm)
        return polar_peaks

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

