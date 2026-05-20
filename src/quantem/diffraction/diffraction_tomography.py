from itertools import permutations, product
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation
import torch

from quantem.core.datastructures.dataset import Dataset
from quantem.core.datastructures.dataset6d import Dataset6d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.core.utils.validators import validate_gt


class DiffractionTomography(AutoSerialize):
    """
    Container for 6D diffraction tomography data.

    The expected axis order is `[z, y, x, kz, ky, kx]`, where the first three
    axes correspond to real space and the last three correspond to reciprocal
    space.
    """

    _token = object()
    default_energy = 300e3
    axis_labels = ("x", "y", "z", "kx", "ky", "kz")

    def __init__(
        self,
        dataset: Dataset6d,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use DiffractionTomography.from_array(), "
                ".from_dataset(), or .from_test() to instantiate this class."
            )

        self.dataset = dataset

    @classmethod
    def from_array(
        cls,
        array: torch.tensor,
        name: str | None = None,
        origin: torch.tensor | tuple | list | float | int | None = None,
        sampling: torch.tensor | tuple | list | float | int | None = None,
        units: list[str] | tuple[str, ...] | str | None = None,
        signal_units: str = "arb. units",
        energy: float | None = None,
        wavelength: float | None = None,
    ) -> Self:
        """
        Create a DiffractionTomography instance from a 6D torch tensor.
        """
        metadata = cls._merge_beam_metadata(
            metadata=None,
            energy=energy,
            wavelength=wavelength,
        )
        dataset = Dataset6d.from_array(
            array=array,
            name=name if name is not None else "Diffraction tomography dataset",
            origin=origin,
            sampling=sampling,
            units=units if units is not None else list(cls.axis_labels),
            signal_units=signal_units,
            metadata=metadata,
        )
        return cls.from_dataset(dataset, energy=energy, wavelength=wavelength)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset | Dataset6d,
        energy: float | None = None,
        wavelength: float | None = None,
    ) -> Self:
        """
        Create a DiffractionTomography instance from an existing 6D Dataset.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be a Dataset, got {type(dataset)}")
        if dataset.ndim != 6:
            raise ValueError(f"dataset must be 6D, got ndim={dataset.ndim}")

        metadata = cls._merge_beam_metadata(
            metadata=dataset.metadata,
            energy=energy,
            wavelength=wavelength,
        )
        
        dataset1 = Dataset6d.from_array(
            array=dataset.array,
            name=dataset.name,
            origin=dataset.origin,
            sampling=dataset.sampling,
            units=dataset.units,
            signal_units=dataset.signal_units,
            metadata=metadata,
        )

        return cls(dataset=dataset1, _token=cls._token)

    @staticmethod
    def _resolve_beam_parameters(
        energy: float | None = None,
        wavelength: float | None = None,
    ) -> tuple[float | None, float | None]:
        if energy is not None:
            energy = float(validate_gt(float(energy), 0, "energy"))
        if wavelength is not None:
            wavelength = float(validate_gt(float(wavelength), 0, "wavelength"))

        if energy is not None:
            wavelength_from_energy = float(electron_wavelength_angstrom(energy))
            # print("wavelength_from_energy", wavelength_from_energy)
            if wavelength is None:
                wavelength = wavelength_from_energy
            elif not torch.isclose(torch.tensor(wavelength), torch.tensor(wavelength_from_energy), rtol=1e-6, atol=1e-12):
                raise ValueError(
                    "Provided energy and wavelength are inconsistent. "
                    f"Expected wavelength {wavelength_from_energy} A for energy {energy} eV, "
                    f"got {wavelength} A."
                )

        return energy, wavelength

    @classmethod
    def _merge_beam_metadata(
        cls,
        metadata: dict | None,
        energy: float | None = None,
        wavelength: float | None = None,
    ) -> dict:
        merged = {} if metadata is None else dict(metadata)
        if energy is None and wavelength is None:
            energy = merged.get("energy")
            wavelength = merged.get("wavelength")
            if energy is None and wavelength is None:
                energy = cls.default_energy
        energy, wavelength = cls._resolve_beam_parameters(
            energy=energy,
            wavelength=wavelength,
        )
        merged["energy"] = energy
        merged["wavelength"] = wavelength
        return merged

    def set_beam_parameters(
        self,
        energy: float | None = None,
        wavelength: float | None = None,
    ) -> Self:
        metadata = self._merge_beam_metadata(
            metadata=self.dataset.metadata,
            energy=energy,
            wavelength=wavelength,
        )
        self.dataset.metadata.clear()
        self.dataset.metadata.update(metadata)
        return self

    @staticmethod
    def _make_reciprocal_grids(
        shape: tuple[int, int, int] | list[int] | torch.Tensor,
        sampling: tuple[float, float, float] | list[float] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = tuple(int(n) for n in shape)
        dk = torch.asarray(sampling, dtype=float)
        if len(shape) != 3:
            raise ValueError(f"shape must have length 3, got {shape}")
        if dk.shape != (3,):
            raise ValueError(f"sampling must have length 3, got shape={dk.shape}")
        kx = torch.fft.fftfreq(shape[0], d=1 / (shape[0] * dk[0]))[:, None, None]
        ky = torch.fft.fftfreq(shape[1], d=1 / (shape[1] * dk[1]))[None, :, None]
        kz = torch.fft.fftfreq(shape[2], d=1 / (shape[2] * dk[2]))[None, None, :]
        return kx, ky, kz

    def _update_reciprocal_coordinates(self) -> None:
        self.kx, self.ky, self.kz = self._make_reciprocal_grids(
            self.diffraction_shape,
            self.dataset.sampling[3:],
        )

    @staticmethod
    def _sample_complex_volume_trilinear(
        volume: torch.tensor,
        coords: torch.tensor,
        mode: str = "wrap",
        device: str | int = "cpu",
    ) -> torch.tensor:
        """Sample a complex 3D volume with trilinear interpolation."""
        volume_np = volume.to(torch.complex128).detach().cpu().numpy()
        coords_np = coords.detach().cpu().numpy()
        
        for i in range(coords_np.shape[0]):
            coords_np[i] = np.remainder(coords_np[i], volume_np.shape[i])

        real_mapped = ndi.map_coordinates(
            volume_np.real,
            coords_np,
            order=1,
            mode=mode,
        )

        imag_mapped = ndi.map_coordinates(
            volume_np.imag,
            coords_np,
            order=1,
            mode=mode
        )
        return (torch.from_numpy(real_mapped) + 1j * torch.from_numpy(imag_mapped)).to(device)

    @classmethod
    def _rotate_complex_volume_zxz(
        cls,
        volume: torch.tensor,
        sampling: tuple[float, float, float] | list[float] | torch.tensor,
        zxz: torch.tensor,
    ) -> torch.tensor:
        """Rotate a complex reciprocal-space volume using ZXZ Euler angles."""
        if volume.dim() != 3:
            raise ValueError(f"volume must be 3D, got ndim={volume.dim()}")

        dk = torch.tensor(sampling, dtype=torch.float32)
        if dk.shape != (3,):
            raise ValueError(f"sampling must have length 3, got shape={dk.shape}")

        shape = torch.tensor(volume.shape, dtype=torch.int32)
        kx, ky, kz = cls._make_reciprocal_grids(shape, dk)
        k_grid = torch.stack(torch.broadcast_tensors(kx, ky, kz), axis=-1)

        rot = Rotation.from_euler("zxz", zxz)
        k_source = rot.inv().apply(k_grid.reshape(-1, 3)).reshape(*volume.shape, 3)
        coords = torch.stack(
            [
                torch.remainder(k_source[..., axis] / dk[axis], shape[axis])
                for axis in range(3)
            ],
            axis=0,
        )

        return cls._sample_complex_volume_trilinear(volume, coords, mode="wrap")

    @classmethod
    def from_test(
        cls,
        name = 'test_data',
        origin = None,
        sampling = (10,10,10,0.05,0.05,0.05),
        units = ('A','A','A','A^-1','A^-1','A^-1'),
        signal_units: str = "SF",
        energy: float | None = None,
        wavelength: float | None = None,
    ) -> Self:
        """
        Create test data for development.
        """


        # Test structure factor 
        SF = torch.zeros((41,41,41),dtype=torch.complex128)
        SF[0,0,0] = 1

        # units
        kx = torch.fft.fftfreq(SF.shape[0],d=1/SF.shape[0]/sampling[3])
        ky = torch.fft.fftfreq(SF.shape[1],d=1/SF.shape[1]/sampling[4])
        kz = torch.fft.fftfreq(SF.shape[2],d=1/SF.shape[2]/sampling[5])

        kx = kx[:,None,None]
        ky = ky[None,:,None]
        kz = kz[None,None,:]
        
        # Approx. SF of gold
        a_Au = 4.08 
        hkl_SF = torch.tensor((
            (1,1,1,0.10),
            (2,0,0,0.06),
            (2,2,0,0.03),
            (3,1,1,0.05),
            (2,2,2,0.04),
            ),
            dtype=torch.float32
            )
        dk = torch.tensor([sampling[3], sampling[4], sampling[5]])
        for a0 in range(hkl_SF.shape[0]):
            hkl = hkl_SF[a0, :3]
            sf_amp = hkl_SF[a0, 3]

            # symmetry equivalent vectors
            vec = torch.tensor(
                sorted(
                    {
                        tuple(torch.multiply(
                            torch.tensor(sign), 
                            torch.tensor(perm)
                            ))
                        for perm in set(permutations(hkl.tolist()))
                        for sign in product((-1, 1), repeat=3)
                    }
                ),
                dtype=torch.float32,
            )

            for a1 in range(vec.shape[0]):
                # peak position
                k_peak = vec[a1] / a_Au

                # trilinear interpolation
                grid_coord = k_peak / dk
                base = torch.floor(grid_coord)
                frac = grid_coord - base

                for dx in range(2):
                    wx = frac[0] if dx else 1.0 - frac[0]
                    ix = ((base[0] + dx) % SF.shape[0]).to(torch.int32)
                    for dy in range(2):
                        wy = frac[1] if dy else 1.0 - frac[1]
                        iy = ((base[1] + dy) % SF.shape[1]).to(torch.int32)
                        for dz in range(2):
                            wz = frac[2] if dz else 1.0 - frac[2]
                            iz = ((base[2] + dz) % SF.shape[2]).to(torch.int32)
                            SF[ix, iy, iz] += sf_amp * wx * wy * wz
        

        # Generate test dataset by rotating test SF to different orientations
        tensor = torch.zeros((
            10,10,5,
            SF.shape[0],SF.shape[1],SF.shape[2],
        ),
            dtype = torch.complex128,
        )

        # 001 grain
        tensor[:5,:5,:] = SF


        # 111 grain
        zxz = torch.deg2rad(torch.tensor((45,54.7,15)))
        tensor[5:,:5,:] = cls._rotate_complex_volume_zxz(SF, sampling[3:], zxz)


        # 110 grain
        zxz = torch.deg2rad(torch.tensor((0,45,-10)))
        tensor[:5,5:,:] = cls._rotate_complex_volume_zxz(SF, sampling[3:], zxz)

        # random grain
        zxz = torch.deg2rad(torch.tensor((12,23,54)))
        tensor[5:,5:,:] = cls._rotate_complex_volume_zxz(SF, sampling[3:], zxz)


        fig,ax = plt.subplots(2,2)
        ax[0,0].imshow(
            torch.fft.fftshift(
                torch.abs(tensor[:5,:5,:]).sum((0,1,2,5))
            )**0.5
        )
        ax[1,0].imshow(
            torch.fft.fftshift(
                torch.abs(tensor[5:,:5,:]).sum((0,1,2,5))
            )**0.5
        )
        ax[0,1].imshow(
            torch.fft.fftshift(
                torch.abs(tensor[:5,5:,:]).sum((0,1,2,5))
            )**0.5
        )
        ax[1,1].imshow(
            torch.fft.fftshift(
                torch.abs(tensor[5:,5:,:]).sum((0,1,2,5))
            )**0.5
        )
        ax[0,0].axis('off')
        ax[1,0].axis('off')
        ax[0,1].axis('off')
        ax[1,1].axis('off')

        # Output
        dataset = Dataset6d.from_array(
            array=np.array(tensor),
            name=name if name is not None else "Diffraction tomography dataset",
            origin=origin,
            sampling=sampling,
            units=units if units is not None else list(cls.axis_labels),
            signal_units=signal_units,
            metadata=cls._merge_beam_metadata(
                metadata=None,
                energy=energy,
                wavelength=wavelength,
            ),
        )
        return cls.from_dataset(dataset, energy=energy, wavelength=wavelength)


    @property
    def dataset(self) -> Dataset6d:
        return self._dataset

    @dataset.setter
    def dataset(self, value: Dataset6d):
        if not isinstance(value, Dataset6d):
            raise TypeError(f"dataset must be a Dataset6d, got {type(value)}")
        self._dataset = value
        self._update_reciprocal_coordinates()

    @property
    def array(self) -> torch.tensor:
        return torch.tensor(self.dataset.array)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.dataset.shape

    @property
    def energy(self) -> float | None:
        return self.dataset.metadata.get("energy")

    @energy.setter
    def energy(self, value: float | None):
        self.set_beam_parameters(energy=value)

    @property
    def wavelength(self) -> float | None:
        wavelength = self.dataset.metadata.get("wavelength")
        if wavelength is None and self.energy is not None:
            wavelength = float(electron_wavelength_angstrom(self.energy))
            self.dataset.metadata["wavelength"] = wavelength
        return wavelength

    @wavelength.setter
    def wavelength(self, value: float | None):
        self.set_beam_parameters(wavelength=value)

    @property
    def real_shape(self) -> tuple[int, int, int]:
        return self.shape[:3]

    @property
    def diffraction_shape(self) -> tuple[int, int, int]:
        return self.shape[3:]
    
    @property
    def diffraction_sampling(self) -> tuple[float, float, float]:
        return self.dataset.sampling[3:]


    def make_prop(
        self,
        prop_distance = None,
        shape: tuple[int, int] | list[int] | torch.tensor | None = None,
    ):
        if prop_distance is None:
            prop_distance = self.dataset.sampling[2]
        self.prop_distance = prop_distance
        if shape is None:
            shape = self.diffraction_shape[:2]
        ku, kv = self._make_planar_reciprocal_grids(shape, self.dataset.sampling[3:5])
        self.prop = torch.exp(-1j * torch.pi * self.wavelength * (ku**2 + kv**2) * prop_distance)

    @staticmethod
    def _make_planar_reciprocal_grids(
        shape: tuple[int, int] | list[int] | torch.tensor,
        sampling: tuple[float, float] | list[float] | torch.tensor,
    ) -> tuple[torch.tensor, torch.tensor]:
        shape = tuple(int(n) for n in shape)
        dk = torch.asarray(sampling, dtype=float)
        if len(shape) != 2:
            raise ValueError(f"shape must have length 2, got {shape}")
        if dk.shape != (2,):
            raise ValueError(f"sampling must have length 2, got shape={dk.shape}")

        ku = torch.fft.fftfreq(shape[0], d=1 / (shape[0] * dk[0]))[:, None]
        kv = torch.fft.fftfreq(shape[1], d=1 / (shape[1] * dk[1]))[None, :]
        return ku, kv

    @classmethod
    def _make_projected_k_slice_coords(
        cls,
        shape_2d: tuple[int, int] | list[int] | torch.Tensor,
        sampling_2d: tuple[float, float] | list[float] | torch.Tensor,
        u_proj: torch.Tensor,
        v_proj: torch.Tensor,
        sampling_3d: tuple[float, float, float] | list[float] | torch.Tensor,
        shape_3d: tuple[int, int, int] | list[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ku, kv = cls._make_planar_reciprocal_grids(shape_2d, sampling_2d)
        ku2, kv2 = torch.broadcast_tensors(ku, kv)
        u_proj = torch.asarray(u_proj, dtype=float)
        v_proj = torch.asarray(v_proj, dtype=float)
        k_xyz = ku2[..., None] * u_proj[None, None, :] + kv2[..., None] * v_proj[None, None, :]

        dk = torch.asarray(sampling_3d, dtype=float)
        shape_k = torch.asarray(shape_3d, dtype=int)
        coords = torch.stack(
            [
                torch.remainder(k_xyz[..., axis] / dk[axis], shape_k[axis])
                for axis in range(3)
            ],
            axis=0,
        )
        return coords, ku2, kv2

    @staticmethod
    def _generate_slab_ray_coordinates(
        position: torch.Tensor,
        direction: torch.Tensor,
        shape: tuple[int, int, int] | list[int] | torch.Tensor,
        sampling: tuple[float, float, float] | list[float] | torch.Tensor,
        prop_distance: float,
    ) -> torch.Tensor:
        """Generate equally spaced index-space samples along a ray inside the volume."""
        r0 = torch.asarray(position, dtype=torch.float32)
        w_proj = torch.asarray(direction, dtype=torch.float32)
        shape = torch.asarray(shape, dtype=torch.int32)
        sampling = torch.asarray(sampling, dtype=torch.float32)

        if r0.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {r0.shape}")
        if w_proj.shape != (3,):
            raise ValueError(f"direction must have shape (3,), got {w_proj.shape}")
        if shape.shape != (3,):
            raise ValueError(f"shape must have length 3, got {shape}")
        if sampling.shape != (3,):
            raise ValueError(f"sampling must have length 3, got {sampling.shape}")

        w_norm = torch.linalg.norm(w_proj)
        if w_norm == 0:
            raise ValueError("direction must be non-zero")
        w_proj = w_proj / w_norm

        dr = prop_distance * w_proj / sampling

        t_min = -torch.inf
        t_max = torch.inf
        for axis in range(3):
            if torch.isclose(dr[axis], torch.tensor(0.0)):
                if not (0.0 <= r0[axis] <= shape[axis] - 1):
                    return torch.empty((0, 3), dtype=float)
                continue

            t0 = (0.0 - r0[axis]) / dr[axis]
            t1 = ((shape[axis] - 1) - r0[axis]) / dr[axis]
            t_min = max(t_min, min(t0, t1))
            t_max = min(t_max, max(t0, t1))

        if t_max < t_min:
            return torch.empty((0, 3), dtype=float)

        # Sample the full line segment through the slab, including both negative
        # and positive steps from the itorchut position when they remain in bounds.
        eps = 1e-9
        n_min = int(torch.ceil(t_min - eps))
        n_max = int(torch.floor(t_max + eps))
        if n_max < n_min:
            return torch.empty((0, 3), dtype=float)

        steps = torch.arange(n_min, n_max + 1, dtype=float)[:, None]
        return r0[None, :] + steps * dr[None, :]

    @staticmethod
    def _trilinear_real_weights(position: torch.Tensor) -> list[tuple[tuple[int, int, int], float]]:
        """Return real-space trilinear neighbors and weights for one sample position."""
        base = torch.floor(position).to(torch.int32)
        frac = position - base
        weights: list[tuple[tuple[int, int, int], float]] = []
        for dx in range(2):
            wx = frac[0] if dx else 1.0 - frac[0]
            for dy in range(2):
                wy = frac[1] if dy else 1.0 - frac[1]
                for dz in range(2):
                    wz = frac[2] if dz else 1.0 - frac[2]
                    weights.append(((base[0] + dx, base[1] + dy, base[2] + dz), (wx * wy * wz).to(torch.float32)))
        return weights

    def forward_prop(
        self,
        Psi0,
        position,
        u_proj = (1,0,0),
        v_proj = (0,1,0),
        
    ):
        # projection vectors
        u_proj = torch.asarray(u_proj,dtype=torch.float32)
        v_proj = torch.asarray(v_proj,dtype=torch.float32)
        u_proj /= torch.linalg.norm(u_proj)
        v_proj /= torch.linalg.norm(v_proj)
        w_proj = torch.cross(u_proj,v_proj)
        if torch.isclose(torch.linalg.norm(w_proj), torch.tensor(0.0)):
            raise ValueError("u_proj and v_proj must not be parallel")
        w_proj /= torch.linalg.norm(w_proj)


        # generate real space coordinates passing through volume
        if not hasattr(self, "prop_distance"):
            self.make_prop()

        position = torch.asarray(position, dtype=float).squeeze()
        if position.shape != (3,):
            raise ValueError(
                f"position must have shape (3,), got {position.shape}"
            )
        Psi0 = torch.tensor(Psi0)
        if Psi0.ndim != 2:
            raise ValueError(f"Psi0 must be 2D, got ndim={Psi0.ndim}")

        k_slice_coords, _, _ = self._make_projected_k_slice_coords(
            shape_2d=Psi0.shape,
            sampling_2d=self.dataset.sampling[3:5],
            u_proj=u_proj,
            v_proj=v_proj,
            sampling_3d=self.dataset.sampling[3:],
            shape_3d=self.diffraction_shape,
        )

        if not hasattr(self, "prop") or self.prop.shape != Psi0.shape:
            self.make_prop(prop_distance=self.prop_distance, shape=Psi0.shape)

        r_all = self._generate_slab_ray_coordinates(
            position=position,
            direction=w_proj,
            shape=self.array.shape[:3],
            sampling=self.dataset.sampling[:3],
            prop_distance=self.prop_distance,
        )

        # multislice propagation through volume
        Psi = Psi0.clone()

        nonzero_slices = 0
        total_sf_sum = 0.0

        for ind, r in enumerate(r_all):
            # trilinear interpolation of structure factor slice from 6D volume
            SF = torch.zeros(Psi0.shape, dtype=self.array.dtype)
            for (ix, iy, iz), weight in self._trilinear_real_weights(r):
                if torch.isclose(weight, torch.tensor(0.0)):
                    continue
                if not (
                    0 <= ix < self.real_shape[0]
                    and 0 <= iy < self.real_shape[1]
                    and 0 <= iz < self.real_shape[2]
                ):
                    continue
                SF += weight * self._sample_complex_volume_trilinear(
                    self.array[ix, iy, iz],
                    k_slice_coords,
                    mode="wrap",
                ).to(torch.float32)
            
            sf_magnitude = torch.abs(SF).sum()
            if sf_magnitude > 1e-6:
                nonzero_slices += 1
                total_sf_sum += float(sf_magnitude)

            # transmit
            if not torch.allclose(SF, torch.zeros_like(SF), atol = 1e-2, rtol = 1e-1):
                Psi = torch.fft.fft2(torch.fft.ifft2(Psi) * torch.fft.ifft2(SF))

            # else:
            #     Psi = torch.fft.fft2(torch.fft.ifft2(Psi) * torch.fft.ifft2(SF))

            # propagate
            if ind < len(r_all) - 1:
                Psi *= self.prop

        return Psi
