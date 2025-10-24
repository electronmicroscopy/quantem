from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_functions import get_activation_function
from .blocks import Conv2dBlock, Upsample2dBlock, complex_pool, passfunc


class ConvAutoencoder2d(nn.Module):
    def __init__(
        self,
        input_size: int | tuple[int, int],
        input_channels: int = 1,
        latent_dim: int = 128,
        start_filters: int = 16,
        num_layers: int = 4,
        num_per_layer: int = 2,
        dtype: torch.dtype = torch.float32,
        dropout: float = 0.0,
        activation: str | Callable = "relu",
        final_activation: str | Callable = "relu",
        use_batchnorm: bool = True,
        latent_normalization: Literal["none", "l2", "layer", "tanh"] = "none",
    ):
        """
        Convolutional autoencoder for 4DSTEM diffraction pattern analysis.

        Parameters
        ----------
        input_size : int | tuple[int, int]
            Input image size. If int, assumes square image.
        input_channels : int, default=1
            Number of input channels (typically 1 for grayscale diffraction patterns)
        latent_dim : int, default=128
            Dimensionality of the latent representation for clustering
        final_activation : str | Callable, default="relu"
            Output activation. Common choices:
            - "relu": For positive-valued intensities [0,∞)
            - "sigmoid": For normalized intensities [0,1]
            - "softplus": Smooth positive activation
            - "identity": For preprocessed data (mean=0, std=1)
        latent_normalization : Literal["none", "l2", "layer", "tanh"], default="none"
            Latent space normalization for clustering:
            - "l2": For DBScan, K-means (unit hypersphere)
            - "layer": For GMM (Gaussian-like distributions)
            - "tanh": Bounded latent space [-1,1]
            - "none": Raw learned representations
        """
        super().__init__()
        self.input_size = input_size
        self.input_channels = int(input_channels)
        self.latent_dim = int(latent_dim)
        self.start_filters = int(start_filters)
        self.num_layers = int(num_layers)
        self._num_per_layer = int(num_per_layer)
        self.dtype = dtype
        self.dropout = float(dropout)
        self._use_batchnorm = bool(use_batchnorm)
        self.latent_normalization = latent_normalization

        if self.dtype.is_complex:
            self.pool = complex_pool
        else:
            self.pool = passfunc
        self._pooler = nn.MaxPool2d(kernel_size=2, stride=2)

        self.activation = activation
        self.final_activation = final_activation

        self._build()

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    @input_size.setter
    def input_size(self, size: int | tuple[int, int]):
        if isinstance(size, int):
            self._input_size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError("input_size must be a tuple of two integers")
            self._input_size = (int(size[0]), int(size[1]))

    @property
    def activation(self) -> Callable:
        return self._activation

    @activation.setter
    def activation(self, act: str | Callable):
        self._activation = get_activation_function(act, self.dtype) if not callable(act) else act

    @property
    def final_activation(self) -> Callable:
        return self._final_activation

    @final_activation.setter
    def final_activation(self, act: str | Callable):
        self._final_activation = (
            get_activation_function(act, self.dtype) if not callable(act) else act
        )

    def _build(self):
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        # Encoder
        in_channels = self.input_channels
        out_channels = self.start_filters
        for layer_idx in range(self.num_layers):
            if layer_idx > 0:
                out_channels = in_channels * 2

            self.encoder_blocks.append(
                Conv2dBlock(
                    nb_layers=self._num_per_layer,
                    input_channels=in_channels,
                    output_channels=out_channels,
                    use_batchnorm=self._use_batchnorm,
                    dropout=self.dropout,
                    dtype=self.dtype,
                    activation=self.activation,
                )
            )
            in_channels = out_channels

        # Calculate spatial dimensions after encoding
        self._encoded_spatial_dim = (
            self.input_size[0] // (2**self.num_layers),
            self.input_size[1] // (2**self.num_layers),
        )
        self._encoded_channels = in_channels
        self._encoded_features = (
            self._encoded_channels * self._encoded_spatial_dim[0] * self._encoded_spatial_dim[1]
        )

        # Latent space
        self.to_latent = nn.Linear(self._encoded_features, self.latent_dim, dtype=self.dtype)
        self.from_latent = nn.Linear(self.latent_dim, self._encoded_features, dtype=self.dtype)

        # Latent normalization layer
        if self.latent_normalization == "layer":
            if self.dtype.is_complex:
                self.latent_norm = nn.LayerNorm([self.latent_dim], dtype=torch.float32)
            else:
                self.latent_norm = nn.LayerNorm([self.latent_dim], dtype=self.dtype)
        elif self.latent_normalization == "tanh":
            self.latent_norm = nn.Tanh()

        # Decoder
        in_channels = self._encoded_channels
        for layer_idx in range(self.num_layers):
            if layer_idx == self.num_layers - 1:
                out_channels = self.input_channels
            else:
                out_channels = max(in_channels // 2, self.start_filters)

            self.upsample_blocks.append(
                Upsample2dBlock(
                    input_channels=in_channels,
                    output_channels=out_channels,
                    use_batchnorm=self._use_batchnorm and layer_idx < self.num_layers - 1,
                    dtype=self.dtype,
                )
            )

            # Final layer uses different activation
            layer_activation = (
                self.final_activation if layer_idx == self.num_layers - 1 else self.activation
            )

            self.decoder_blocks.append(
                Conv2dBlock(
                    nb_layers=self._num_per_layer,
                    input_channels=out_channels,
                    output_channels=out_channels,
                    use_batchnorm=self._use_batchnorm and layer_idx < self.num_layers - 1,
                    dropout=self.dropout if layer_idx < self.num_layers - 1 else 0.0,
                    dtype=self.dtype,
                    activation=layer_activation,
                )
            )
            in_channels = out_channels

    def _apply_latent_normalization(self, latent: torch.Tensor) -> torch.Tensor:
        if self.latent_normalization == "l2":
            return F.normalize(latent, p=2, dim=-1)
        elif self.latent_normalization == "layer":
            if self.dtype.is_complex:
                latent_real = self.latent_norm(latent.real)
                latent_imag = self.latent_norm(latent.imag)
                return torch.complex(latent_real, latent_imag)
            else:
                return self.latent_norm(latent)
        elif self.latent_normalization == "tanh":
            return self.latent_norm(latent)
        else:  # "none"
            return latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            x = self.pool(x, self._pooler)

        # Flatten and map to latent space
        x = x.flatten(start_dim=1)
        latent = self.to_latent(x)

        # Apply normalization
        latent = self._apply_latent_normalization(latent)

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(latent)
        x = x.view(-1, self._encoded_channels, *self._encoded_spatial_dim)

        for upsample_block, decoder_block in zip(self.upsample_blocks, self.decoder_blocks):
            x = upsample_block(x)
            x = decoder_block(x)

        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def reset_weights(self):
        def _reset(m: nn.Module) -> None:
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

        self.apply(_reset)
