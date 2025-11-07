# from collections.abc import Sequence
from typing import List, Optional, Union, Tuple

# import matplotlib.pyplot as plt
import numpy as np
# from numpy.typing import NDArray
# from scipy.interpolate import interp1d
# from scipy.ndimage import gaussian_filter
# from scipy.optimize import minimize
from tqdm import tqdm
import torch
from skimage.feature import blob_dog, blob_log, blob_doh

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


def percentile_calc(data, lower_percentile=1, upper_percentile=99):
    # Flatten the data for global percentile calculation
    data_flat = data.flatten()
    # Convert percentiles to quantiles (0-1 range)
    p_lower = torch.quantile(data_flat, lower_percentile / 100.0)
    p_upper = torch.quantile(data_flat, upper_percentile / 100.0)
    
    return p_lower, p_upper

def percentile_normalize(data, p_lower, p_upper):
    return torch.clamp((data - p_lower) / (p_upper - p_lower), 0, 1)  


class BraggPeaks(AutoSerialize):
    """
    
    """

    _token = object()

    def __init__(
        self,
        dataset: Dataset4d,
        model: MultiChannelCNN2d = None,
        final_shape: Tuple[int, int] = (256, 256),
        device: str = 'cpu',
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use BraggPeaks.from_data() or .from_file() to instantiate this class."
            )

        self._dataset = dataset
        self._device = device
        self._final_shape = final_shape
        # Setup model
        input_channels = 1  # 1 for a greyscale image, 3 for RGB, 4 for RGBA, etc.
        k_size = 7
        num_layers = 4
        start_filters = 16
        num_per_layer = 2
        use_skip_connections = True
        dtype = torch.float32
        dropout = 0     
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
    ) -> "BraggPeaks":
        dataset = Dataset4d.from_file(file_path, file_type=file_type)
        return cls.from_data(
            dataset,
        )

    @classmethod
    def from_data(
        cls,
        dataset: Dataset4d,
        device: str,
    ) -> "BraggPeaks":
        return cls(
            dataset=dataset,
            _token=cls._token,
            device=device,
        )
    
    def preprocess(self):
        self.resize_data()

    def resize_data(self, device:str = "cuda:1"):
        Ry, Rx, Qy, Qx = self._dataset.shape
        scale_factor = (self._final_shape[0] * self._final_shape[1]) / (Qy * Qx)
        resized_data = np.zeros((Ry, Rx, self._final_shape[0], self._final_shape[1]))
        for i in tqdm(range(Ry), desc='rows'):
            inp = torch.tensor(self._dataset[i].array, dtype=torch.float32).to(device)
            inp = torch.nn.functional.interpolate(inp[None, ...], size=self._final_shape, mode='bilinear', align_corners=False) * scale_factor
            resized_data[i, :, :, :] = inp.squeeze().detach().cpu().numpy()
        self.resized_cartesian_data = resized_data

    def set_model_weights(
        self,
        # path_to_model: str = None,
        path_to_weights: str = None,
        gpu_id: int = 1,
    ) -> "BraggPeaks":
        # if path_to_model is None:
            # path_to_model = ""
        if path_to_weights is None:
            path_to_weights = ""  # TODO: Load weights from cloud
        self._model.load_state_dict(torch.load(path_to_weights, weights_only=True, map_location=f"cuda:{gpu_id}"))
        self._model.to(gpu_id)

    def _postprocess_single(self, position_map, intensity_map, show=False, im_comp=None):
        """Process a single 2D image"""
        # Detect blobs
        # blobs_log = blob_log(
        #     position_map, 
        #     max_sigma=30, 
        #     num_sigma=10, 
        #     threshold=0.1
        # )
        
        blobs_dog = blob_dog(
            position_map, 
            max_sigma=30, 
            threshold=0.1
        )
        if len(blobs_dog) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert sigma to radius
        blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
        
        # Extract coordinates and radii
        coordinates = blobs_dog[:, :2]  # (y, x)
        sigma_gaussian = blobs_dog[:, 2]
        
        # Extract intensities (vectorized)
        y_coords = np.clip(coordinates[:, 0].astype(int), 0, intensity_map.shape[0] - 1)
        x_coords = np.clip(coordinates[:, 1].astype(int), 0, intensity_map.shape[1] - 1)
        intensities = intensity_map[y_coords, x_coords]
        
        # if show:
        #     self._visualize(position_map, intensity_map, blobs_log, 
        #                 coordinates, intensities, im_comp)
        
        return coordinates, intensities

    def postprocessing(self, position_map, intensity_map, show=False, im_comp=None):
        """
        Detect blobs and extract intensities
        
        Parameters
        ----------
        position_map : np.ndarray
            Binary or grayscale image for blob detection
            Shape: (H, W) or (N, H, W)
        intensity_map : np.ndarray
            Image from which to extract intensities at blob positions
            Shape: (H, W) or (N, H, W)
        show : bool
            Whether to visualize results
        im_comp : np.ndarray, optional
            Comparison image for visualization
            
        Returns
        -------
        If 2D input:
            coordinates, sigma_gaussian, intensities
        If 3D input:
            List of (coordinates, sigma_gaussian, intensities) for each image
        """
        # Handle 2D case (single image)
        if position_map.ndim == 2:
            return self._postprocess_single(position_map, intensity_map, show, im_comp)
        
        # Handle 3D case (batch of images)
        elif position_map.ndim == 3:
            N = position_map.shape[0]
            results = []
            
            for i in range(N):
                coords, radii, intensities = self._postprocess_single(
                    position_map[i], 
                    intensity_map[i],
                    show=show and i == 0,  # Only show first image
                    im_comp=im_comp[i] if im_comp is not None else None
                )
                results.append((coords, intensities))
            
            return results
        
        else:
            raise ValueError(f"Expected 2D or 3D array, got {position_map.ndim}D")
        # intensities = [np.average(intensity_map[int(coordinates[i, 0])-2:int(coordinates[i, 0])+2, int(coordinates[i, 1])-2:int(coordinates[i, 1])+2]) for i in range(len(coordinates))]
        # print(f"peak coordinates: {coordinates}")
        # print(f"intensities: {intensities}")

        # if show:
        #     blobs_dog = blob_dog(position_map, max_sigma=30, threshold=0.1)
        #     blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

        #     blobs_doh = blob_doh(position_map, max_sigma=30, threshold=0.01)

        #     blobs_list = [blobs_log, blobs_dog, blobs_doh]
        #     colors = ['yellow', 'lime', 'red']
        #     titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
        #     sequence = zip(blobs_list, colors, titles)

        #     fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
        #     ax = axes.ravel()
        #     for idx, (blobs, color, title) in enumerate(sequence):
        #         ax[idx].set_title(title)
        #         ax[idx].imshow(position_map)
        #         for blob in blobs:
        #             y, x, r = blob
        #             c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        #             ax[idx].add_patch(c)
        #         ax[idx].set_axis_off()
        #     fig.tight_layout() 
            
        #     if im_comp is not None:
        #         fig2, axes2 = plt.subplots(1, 4, figsize=(9, 3), sharex=True, sharey=True)
        #         ax2 = axes2.ravel()
        #         sequence = zip(blobs_list, colors, titles)
        #         ax2[0].set_title("Original Image")
        #         ax2[0].imshow(im_comp, cmap='magma', norm='linear', vmin=0.02, vmax=0.98)
        #         ax2[0].set_axis_off()
        #         for idx, (blobs, color, title) in enumerate(sequence):
        #             ax2[idx+1].set_title(title)
        #             ax2[idx+1].imshow(im_comp, cmap='gray', norm='linear', vmin=0.02, vmax=0.98)  # Display the im_comp image
        #             for blob in blobs:
        #                 y, x, r = blob
        #                 # Check if the blob coordinates are valid
        #                 if x >= 0 and x < im_comp.shape[1] and y >= 0 and y < im_comp.shape[0]:
        #                     c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        #                     ax2[idx+1].add_patch(c)
        #             ax2[idx+1].set_axis_off()   
        #         fig2.tight_layout() 

    def find_peaks_model(
        self, 
        device: str = "cuda:1",
        n_normalize_samples: int = 100,
        ):
        Ry, Rx, Qy, Qx = self.resized_cartesian_data.shape
        # outs = np.zeros((Ry, Rx, 2, Qy, Qx))  # 2 output channels
        peaks = np.empty((Ry, Rx), dtype='object')  # Ragged array for peak coordinates
        intensities = np.empty((Ry, Rx), dtype='object')  # Ragged array for peak intensities
        # Normalize
        scan_shape = (Ry, Rx)
        dp_shape = (Qy, Qx)
        n_positions = scan_shape[0] * scan_shape[1]
        n_normalize_samples = min(n_normalize_samples, n_positions)
        flat_indices = torch.randperm(n_positions, device=device)[:n_normalize_samples]
        scan_y_indices = flat_indices // scan_shape[1]  # row indices, given by floor division by number in row
        scan_x_indices = flat_indices % scan_shape[1]   # column indices, given by remainder of number in row
        stats_patterns = torch.tensor(self.resized_cartesian_data[scan_y_indices.cpu(), scan_x_indices.cpu()], dtype=torch.float32).to(device)
        percentile_lower, percentile_upper = percentile_calc(stats_patterns)
        for i in tqdm(range(Ry), desc="rows"):
            # Go row-by-row with model outputs to prevent memory issues
            ins = torch.tensor(self.resized_cartesian_data[i], dtype=torch.float32).to(device).squeeze()  # Should be of shape (Rx, Ry, Qx, Qy)
            dps_norm = percentile_normalize(ins, percentile_lower, percentile_upper)
            ins = dps_norm[:, None, ...]
            # Pass through model
            outs = self.model(ins).detach().cpu().numpy()
            # outs[i] = self.model(ins).detach().cpu().numpy()
            for r0 in range(outs.shape[0]):  # Expect shape N x 2 x Qy x Qx
            # for r0 in range(outs[i].shape[0]):  # Expect shape N x 2 x Qy x Qx
                peak_coords, peak_intensities = self._postprocess_single(outs[r0, 0], outs[r0, 1])
                peaks[i, r0] = np.array(peak_coords)
                intensities[i, r0] = np.array(peak_intensities)
        print('done')
        self.peak_coordinates_cartesian = peaks
        self.peak_intensities = intensities

    def save_peaks(self, filepath):
        np.save(filepath, self.peak_coordinates_cartesian)

    def load_peaks(self, filepath):
        self.peak_coordinates_cartesian = np.load(filepath, allow_pickle=True)