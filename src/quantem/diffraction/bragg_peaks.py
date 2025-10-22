# from collections.abc import Sequence
# from typing import List, Optional, Union

# import matplotlib.pyplot as plt
# import numpy as np
# from numpy.typing import NDArray
# from scipy.interpolate import interp1d
# from scipy.ndimage import gaussian_filter
# from scipy.optimize import minimize
# from tqdm import tqdm
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


class BraggPeaks(AutoSerialize):
    """
    
    """

    _token = object()

    def __init__(
        self,
        dataset: Dataset4d,
        model: MultiChannelCNN2d = None,
        device: str = 'cpu',
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use BraggPeaks.from_data() or .from_file() to instantiate this class."
            )

        self._dataset = dataset
        self._device = device
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

    def find_peaks_model(self):
        pass