from typing import Union

import numpy as np
import torch

DeviceType = Union[str, "torch.device", int]


class RNGMixin:
    """
    Mixin class providing consistent RNG functionality with both numpy and torch generators.
    If you do not provide a seed, the RNG will be initialized with a random seed, and subsequently
    resetting the RNG will use a _new_ random seed. Setting with a fixed seed or generator and then
    resetting the RNG will use the same seed.

    Provides:
    - self.rng: np.random.Generator property
    - self._rng_torch: torch.Generator for torch operations
    - self._reset_rng(): reset the RNG to the current seed
    """

    def __init__(
        self,
        rng: np.random.Generator | int | None = None,
        device: str | int | torch.device = "cpu",
        *args,
        **kwargs,
    ):
        self._device = device
        self.rng = rng

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, rng: int | np.random.Generator | torch.Generator | None):
        if rng is None:
            self._rng_seed = None
            rng = np.random.default_rng()
        elif isinstance(rng, (int, float)):
            self._rng_seed = rng
            rng = np.random.default_rng(rng)
        elif isinstance(rng, np.random.Generator):
            self._rng_seed = rng.bit_generator._seed_seq.entropy  # type:ignore ## get the seed
        elif isinstance(rng, torch.Generator):
            self._rng_seed = rng.initial_seed()
            rng = np.random.default_rng(self._rng_seed)
        else:
            raise TypeError(f"rng should be a np.random.Generator or a seed, got {type(rng)}")

        self._rng = rng
        self._update_torch_rng()

    def _update_torch_rng(self):
        """Update the torch generator with current seed and device."""
        if self._rng_seed is None:
            self._rng_torch = torch.Generator(device=self._device)
        else:
            self._rng_torch = torch.Generator(device=self._device).manual_seed(
                self._rng_seed % 2**32
            )

    def _reset_rng(self):
        """Reset RNG to current seed, useful for reproducible iterations."""
        if self._rng_seed is not None:
            self.rng = self._rng_seed  # sets rng and _rng_torch

    def _rng_to_device(self, device: "DeviceType"):
        ## Could consider renaming this as just to, allowing super calls
        """Update torch RNG when device changes.
        Currently resets the seed... not sure of a way around that."""
        self._device = device
        rng_torch = torch.Generator(device=self._device)
        if self._rng_seed is not None:
            rng_torch = rng_torch.manual_seed(self._rng_seed % 2**32)
        self._rng_torch = rng_torch
