import math
from typing import TYPE_CHECKING

import numpy as np

from quantem.core import config
from quantem.core.datastructures import Dataset2d, Dataset3d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.rng import RNGMixin
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.core.utils.validators import (
    validate_aberration_coefficients,
    validate_gt,
    validate_tensor,
)
from quantem.diffractive_imaging.complex_probe import (
    aberration_surface,
    aberration_surface_cartesian_gradients,
    polar_spatial_frequencies,
    spatial_frequencies,
)
from quantem.diffractive_imaging.ptycho_utils import SimpleBatcher

if TYPE_CHECKING:
    import torch
else:
    if config.get("has_torch"):
        import torch


class DirectPtychography(RNGMixin, AutoSerialize):
    """ """

    _token = object()

    def __init__(
        self,
        vbf_dataset: Dataset3d,
        bf_mask_dataset: Dataset2d,
        energy: float,
        rotation_angle: float,
        aberration_coefs: dict,
        semiangle_cutoff: float | None,
        vacuum_probe_intensity: torch.Tensor | None,
        rng: np.random.Generator | int | None,
        device: str | int,
        _token: object | None = None,
    ):
        """ """
        if _token is not self._token:
            raise RuntimeError(
                "Use DirectPtychography.from_dataset4dstem() or DirectPtychography.from_virtual_bfs() to instantiate this class."
            )

        self.vbf_stack = vbf_dataset.array
        self.bf_mask = bf_mask_dataset.array

        self.reciprocal_sampling = bf_mask_dataset.sampling

        self.scan_sampling = vbf_dataset.sampling[-2:]
        self.scan_gpts = vbf_dataset.shape[-2:]
        self.num_bf = vbf_dataset.shape[0]

        if semiangle_cutoff is None and vacuum_probe_intensity is None:
            raise ValueError(
                "one of semiangle_cutoff or vacuum_probe_intensity needs to be specified"
            )

        self.semiangle_cutoff = semiangle_cutoff
        self.vacuum_probe_intensity = vacuum_probe_intensity
        self.device = device
        self.rng = rng

        self.gpts = bf_mask_dataset.shape
        self.sampling = tuple(1 / s / n for n, s in zip(self.reciprocal_sampling, self.gpts))
        self.wavelength = electron_wavelength_angstrom(energy)

        self.rotation_angle = rotation_angle
        self.aberration_coefs = aberration_coefs

        self._preprocess()

    @classmethod
    def from_virtual_bfs(
        cls,
        vbf_dataset: Dataset3d,
        bf_mask_dataset: Dataset2d,
        energy: float,
        rotation_angle: float,
        aberration_coefs: dict,
        semiangle_cutoff: float | None = None,
        vacuum_probe_intensity: torch.Tensor | None = None,
        rng: np.random.Generator | int | None = None,
        device: str | int = "cpu",
    ):
        """ """

        return cls(
            vbf_dataset=vbf_dataset,
            bf_mask_dataset=bf_mask_dataset,
            energy=energy,
            rotation_angle=rotation_angle,
            aberration_coefs=aberration_coefs,
            semiangle_cutoff=semiangle_cutoff,
            vacuum_probe_intensity=vacuum_probe_intensity,
            rng=rng,
            device=device,
            _token=cls._token,
        )

    def _preprocess(
        self,
    ):
        """ """

        self._bf_inds_i, self._bf_inds_j = torch.where(self.bf_mask)
        self._vbf_fourier = torch.fft.fft2(self.vbf_stack)
        self._corrected_stack = None

        return self

    def _return_upsampled_qgrid(
        self,
        upsampling_factor=None,
    ):
        """
        Assumes integer upsampling factor.
        """

        if upsampling_factor is None:
            scan_gpts = self.scan_gpts
            scan_sampling = self.scan_sampling
        else:
            scan_gpts = tuple(n * upsampling_factor for n in self.scan_gpts)
            scan_sampling = tuple(s / upsampling_factor for s in self.scan_sampling)

        qxa, qya = spatial_frequencies(scan_gpts, scan_sampling, device=self.device)

        return qxa, qya

    @property
    def vbf_stack(self) -> torch.Tensor:
        return self._vbf_stack

    @vbf_stack.setter
    def vbf_stack(self, value: torch.Tensor):
        self._vbf_stack = validate_tensor(value, "vbf_stack", dtype=torch.float).to(self.device)

    @property
    def bf_mask(self) -> torch.Tensor:
        return self._bf_mask

    @bf_mask.setter
    def bf_mask(self, value: torch.Tensor):
        self._bf_mask = validate_tensor(value, "bf_mask", dtype=torch.bool).to(self.device)

    @property
    def rotation_angle(self) -> float:
        return self._rotation_angle

    @rotation_angle.setter
    def rotation_angle(self, value: float):
        self._rotation_angle = float(value)

    @property
    def aberration_coefs(self) -> dict:
        return self._aberration_coefs

    @aberration_coefs.setter
    def aberration_coefs(self, value: dict):
        value = validate_aberration_coefficients(value)
        self._aberration_coefs = value

    @property
    def semiangle_cutoff(self) -> dict:
        return self._semiangle_cutoff

    @semiangle_cutoff.setter
    def semiangle_cutoff(self, value: float):
        validate_gt(value, 0.0, "semiangle_cutoff")
        self._semiangle_cutoff = value

    @property
    def device(self) -> str:
        """This should be of form 'cuda:X' or 'cpu', as defined by quantem.config"""
        if hasattr(self, "_device"):
            return self._device
        else:
            return config.get("device")

    @device.setter
    def device(self, device: str | int | None):
        # allow setting gpu/cpu, but not changing the device from the config gpu device
        if device is not None:
            dev, _id = config.validate_device(device)
            self._device = dev
            try:
                self.to(dev)
            except AttributeError:
                pass

    def _parallax_approximation(
        self,
        aberration_coefs=None,
        upsampling_factor=None,
        rotation_angle=None,
        max_batch_size=None,
        flip_phase=True,
    ):
        """ """
        if aberration_coefs is None:
            aberration_coefs = self.aberration_coefs
        else:
            aberration_coefs = validate_aberration_coefficients(aberration_coefs)

        if rotation_angle is None:
            rotation_angle = self.rotation_angle
        else:
            rotation_angle = float(rotation_angle)

        if upsampling_factor is None:
            upsampling_factor = 1
        upsampling_factor = math.ceil(upsampling_factor)

        if aberration_coefs:
            k, phi = polar_spatial_frequencies(
                self.gpts, self.sampling, rotation_angle=rotation_angle, device=self.device
            )
            alpha = k * self.wavelength

            dx, dy = aberration_surface_cartesian_gradients(
                alpha,
                phi,
                aberration_coefs,
            )

            grad_k = torch.stack((dx[self.bf_mask], dy[self.bf_mask]), -1)

            qxa, qya = self._return_upsampled_qgrid(upsampling_factor)
            qvec = torch.stack((qxa, qya), 0)

            grad_kq = torch.einsum("na,amp->nmp", grad_k, qvec)
            operator = torch.exp(-1j * grad_kq)

            if flip_phase:
                q = torch.sqrt(qxa.square() + qya.square())
                theta = torch.arctan2(qya, qxa)

                chi_q = aberration_surface(
                    q * self.wavelength,
                    theta,
                    self.wavelength,
                    aberration_coefs,
                )
                sign_sign_chi_q = torch.sign(torch.sin(chi_q))
                operator = operator * sign_sign_chi_q

            if max_batch_size is None:
                max_batch_size = self.num_bf

            batcher = SimpleBatcher(
                self.num_bf, batch_size=max_batch_size, shuffle=False, rng=self.rng
            )

            corrected_stack = torch.empty((self.num_bf,) + qxa.shape, device=self.device)
            for batch_idx in batcher:
                vbf_fourier = torch.tile(
                    self._vbf_fourier[batch_idx],
                    (1, upsampling_factor, upsampling_factor),
                )
                corrected_stack[batch_idx] = (
                    torch.fft.ifft2(vbf_fourier * operator[batch_idx]).real * upsampling_factor
                )

            self.corrected_stack = corrected_stack

        return self
