from itertools import permutations, product
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation

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
        array: NDArray,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
        units: list[str] | tuple[str, ...] | str | None = None,
        signal_units: str = "arb. units",
        energy: float | None = None,
        wavelength: float | None = None,
    ) -> Self:
        """
        Create a DiffractionTomography instance from a 6D array.
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
        dataset = Dataset6d.from_array(
            array=dataset.array,
            name=dataset.name,
            origin=dataset.origin,
            sampling=dataset.sampling,
            units=dataset.units,
            signal_units=dataset.signal_units,
            metadata=metadata,
        )

        return cls(dataset=dataset, _token=cls._token)

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
            if wavelength is None:
                wavelength = wavelength_from_energy
            elif not np.isclose(wavelength, wavelength_from_energy, rtol=1e-6, atol=1e-12):
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
        shape: tuple[int, int, int] | list[int] | NDArray,
        sampling: tuple[float, float, float] | list[float] | NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:
        shape = tuple(int(n) for n in shape)
        dk = np.asarray(sampling, dtype=float)
        if len(shape) != 3:
            raise ValueError(f"shape must have length 3, got {shape}")
        if dk.shape != (3,):
            raise ValueError(f"sampling must have length 3, got shape={dk.shape}")

        kx = np.fft.fftfreq(shape[0], d=1 / (shape[0] * dk[0]))[:, None, None]
        ky = np.fft.fftfreq(shape[1], d=1 / (shape[1] * dk[1]))[None, :, None]
        kz = np.fft.fftfreq(shape[2], d=1 / (shape[2] * dk[2]))[None, None, :]
        return kx, ky, kz

    def _update_reciprocal_coordinates(self) -> None:
        self.kx, self.ky, self.kz = self._make_reciprocal_grids(
            self.diffraction_shape,
            self.dataset.sampling[3:],
        )

    @staticmethod
    def _sample_complex_volume_trilinear(
        volume: NDArray,
        coords: NDArray,
        mode: str = "wrap",
    ) -> NDArray:
        """Sample a complex 3D volume with trilinear interpolation."""
        return ndi.map_coordinates(
            volume.real,
            coords,
            order=1,
            mode=mode,
        ) + 1j * ndi.map_coordinates(
            volume.imag,
            coords,
            order=1,
            mode=mode,
        )

    @classmethod
    def _rotate_complex_volume_zxz(
        cls,
        volume: NDArray,
        sampling: tuple[float, float, float] | list[float] | NDArray,
        zxz: NDArray,
    ) -> NDArray:
        """Rotate a complex reciprocal-space volume using ZXZ Euler angles."""
        if volume.ndim != 3:
            raise ValueError(f"volume must be 3D, got ndim={volume.ndim}")

        dk = np.asarray(sampling, dtype=float)
        if dk.shape != (3,):
            raise ValueError(f"sampling must have length 3, got shape={dk.shape}")

        shape = np.array(volume.shape, dtype=int)
        kx, ky, kz = cls._make_reciprocal_grids(shape, dk)
        k_grid = np.stack(np.broadcast_arrays(kx, ky, kz), axis=-1)

        rot = Rotation.from_euler("zxz", zxz)
        k_source = rot.inv().apply(k_grid.reshape(-1, 3)).reshape(*volume.shape, 3)
        coords = np.stack(
            [
                np.mod(k_source[..., axis] / dk[axis], shape[axis])
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
        SF = np.zeros((41,41,41),dtype='complex')
        SF[0,0,0] = 1

        # units
        kx = np.fft.fftfreq(SF.shape[0],d=1/SF.shape[0]/sampling[3])
        ky = np.fft.fftfreq(SF.shape[1],d=1/SF.shape[1]/sampling[4])
        kz = np.fft.fftfreq(SF.shape[2],d=1/SF.shape[2]/sampling[5])
        kx = kx[:,None,None]
        ky = ky[None,:,None]
        kz = kz[None,None,:]
        
        # Approx. SF of gold
        a_Au = 4.08 
        hkl_SF = np.array((
            (1,1,1,0.10),
            (2,0,0,0.06),
            (2,2,0,0.03),
            (3,1,1,0.05),
            (2,2,2,0.04),
        ))
        dk = np.array([sampling[3], sampling[4], sampling[5]], dtype=float)
        for a0 in range(hkl_SF.shape[0]):
            hkl = hkl_SF[a0, :3].astype(int)
            sf_amp = hkl_SF[a0, 3]

            # symmetry equivalent vectors
            vec = np.array(
                sorted(
                    {
                        tuple(np.multiply(sign, perm))
                        for perm in set(permutations(hkl.tolist()))
                        for sign in product((-1, 1), repeat=3)
                    }
                ),
                dtype=int,
            )

            for a1 in range(vec.shape[0]):
                # peak position
                k_peak = vec[a1].astype(float) / a_Au

                # trilinear interpolation
                grid_coord = k_peak / dk
                base = np.floor(grid_coord).astype(int)
                frac = grid_coord - base

                for dx in range(2):
                    wx = frac[0] if dx else 1.0 - frac[0]
                    ix = (base[0] + dx) % SF.shape[0]
                    for dy in range(2):
                        wy = frac[1] if dy else 1.0 - frac[1]
                        iy = (base[1] + dy) % SF.shape[1]
                        for dz in range(2):
                            wz = frac[2] if dz else 1.0 - frac[2]
                            iz = (base[2] + dz) % SF.shape[2]
                            SF[ix, iy, iz] += sf_amp * wx * wy * wz


        # Generate test dataset by rotating test SF to different orientations
        array = np.zeros((
            10,10,5,
            SF.shape[0],SF.shape[1],SF.shape[2],
        ),
            dtype = 'complex',
        )

        # 001 grain
        array[:5,:5,:] = SF

        # 111 grain
        zxz = np.deg2rad(np.array((45,54.7,15)))
        array[5:,:5,:] = cls._rotate_complex_volume_zxz(SF, sampling[3:], zxz)

        # 110 grain
        zxz = np.deg2rad(np.array((0,45,-10)))
        array[:5,5:,:] = cls._rotate_complex_volume_zxz(SF, sampling[3:], zxz)

        # random grain
        zxz = np.deg2rad(np.array((12,23,54)))
        array[5:,5:,:] = cls._rotate_complex_volume_zxz(SF, sampling[3:], zxz)



        fig,ax = plt.subplots(2,2)
        ax[0,0].imshow(
            np.fft.fftshift(
                np.abs(array[:5,:5,:]).sum((0,1,2,5))
            )**0.5
        )
        ax[1,0].imshow(
            np.fft.fftshift(
                np.abs(array[5:,:5,:]).sum((0,1,2,5))
            )**0.5
        )
        ax[0,1].imshow(
            np.fft.fftshift(
                np.abs(array[:5,5:,:]).sum((0,1,2,5))
            )**0.5
        )
        ax[1,1].imshow(
            np.fft.fftshift(
                np.abs(array[5:,5:,:]).sum((0,1,2,5))
            )**0.5
        )
        ax[0,0].axis('off')
        ax[1,0].axis('off')
        ax[0,1].axis('off')
        ax[1,1].axis('off')



        # Output
        dataset = Dataset6d.from_array(
            array=array,
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
    def array(self) -> NDArray:
        return self.dataset.array

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


    def make_prop(
        self,
        prop_distance = None,
        shape: tuple[int, int] | list[int] | NDArray | None = None,
    ):
        if prop_distance is None:
            prop_distance = self.dataset.sampling[2]
        self.prop_distance = prop_distance
        if shape is None:
            shape = self.diffraction_shape[:2]
        ku, kv = self._make_planar_reciprocal_grids(shape, self.dataset.sampling[3:5])
        self.prop = np.exp(-1j * np.pi * self.wavelength * (ku**2 + kv**2) * prop_distance)

    @staticmethod
    def _make_planar_reciprocal_grids(
        shape: tuple[int, int] | list[int] | NDArray,
        sampling: tuple[float, float] | list[float] | NDArray,
    ) -> tuple[NDArray, NDArray]:
        shape = tuple(int(n) for n in shape)
        dk = np.asarray(sampling, dtype=float)
        if len(shape) != 2:
            raise ValueError(f"shape must have length 2, got {shape}")
        if dk.shape != (2,):
            raise ValueError(f"sampling must have length 2, got shape={dk.shape}")

        ku = np.fft.fftfreq(shape[0], d=1 / (shape[0] * dk[0]))[:, None]
        kv = np.fft.fftfreq(shape[1], d=1 / (shape[1] * dk[1]))[None, :]
        return ku, kv

    @classmethod
    def _make_projected_k_slice_coords(
        cls,
        shape_2d: tuple[int, int] | list[int] | NDArray,
        sampling_2d: tuple[float, float] | list[float] | NDArray,
        u_proj: NDArray,
        v_proj: NDArray,
        sampling_3d: tuple[float, float, float] | list[float] | NDArray,
        shape_3d: tuple[int, int, int] | list[int] | NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:
        ku, kv = cls._make_planar_reciprocal_grids(shape_2d, sampling_2d)
        ku2, kv2 = np.broadcast_arrays(ku, kv)
        u_proj = np.asarray(u_proj, dtype=float)
        v_proj = np.asarray(v_proj, dtype=float)
        k_xyz = ku2[..., None] * u_proj[None, None, :] + kv2[..., None] * v_proj[None, None, :]

        dk = np.asarray(sampling_3d, dtype=float)
        shape_k = np.asarray(shape_3d, dtype=int)
        coords = np.stack(
            [
                np.mod(k_xyz[..., axis] / dk[axis], shape_k[axis])
                for axis in range(3)
            ],
            axis=0,
        )
        return coords, ku2, kv2

    @staticmethod
    def _generate_slab_ray_coordinates(
        position: NDArray,
        direction: NDArray,
        shape: tuple[int, int, int] | list[int] | NDArray,
        sampling: tuple[float, float, float] | list[float] | NDArray,
        prop_distance: float,
    ) -> NDArray:
        """Generate equally spaced index-space samples along a ray inside the volume."""
        r0 = np.asarray(position, dtype=float)
        w_proj = np.asarray(direction, dtype=float)
        shape = np.asarray(shape, dtype=int)
        sampling = np.asarray(sampling, dtype=float)

        if r0.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {r0.shape}")
        if w_proj.shape != (3,):
            raise ValueError(f"direction must have shape (3,), got {w_proj.shape}")
        if shape.shape != (3,):
            raise ValueError(f"shape must have length 3, got {shape}")
        if sampling.shape != (3,):
            raise ValueError(f"sampling must have length 3, got {sampling.shape}")

        w_norm = np.linalg.norm(w_proj)
        if w_norm == 0:
            raise ValueError("direction must be non-zero")
        w_proj = w_proj / w_norm

        dr = prop_distance * w_proj / sampling

        t_min = -np.inf
        t_max = np.inf
        for axis in range(3):
            if np.isclose(dr[axis], 0.0):
                if not (0.0 <= r0[axis] <= shape[axis] - 1):
                    return np.empty((0, 3), dtype=float)
                continue

            t0 = (0.0 - r0[axis]) / dr[axis]
            t1 = ((shape[axis] - 1) - r0[axis]) / dr[axis]
            t_min = max(t_min, min(t0, t1))
            t_max = min(t_max, max(t0, t1))

        if t_max < t_min:
            return np.empty((0, 3), dtype=float)

        # Sample the full line segment through the slab, including both negative
        # and positive steps from the input position when they remain in bounds.
        eps = 1e-9
        n_min = int(np.ceil(t_min - eps))
        n_max = int(np.floor(t_max + eps))
        if n_max < n_min:
            return np.empty((0, 3), dtype=float)

        steps = np.arange(n_min, n_max + 1, dtype=float)[:, None]
        return r0[None, :] + steps * dr[None, :]

    @staticmethod
    def _trilinear_real_weights(position: NDArray) -> list[tuple[tuple[int, int, int], float]]:
        """Return real-space trilinear neighbors and weights for one sample position."""
        base = np.floor(position).astype(int)
        frac = position - base
        weights: list[tuple[tuple[int, int, int], float]] = []
        for dx in range(2):
            wx = frac[0] if dx else 1.0 - frac[0]
            for dy in range(2):
                wy = frac[1] if dy else 1.0 - frac[1]
                for dz in range(2):
                    wz = frac[2] if dz else 1.0 - frac[2]
                    weights.append(((base[0] + dx, base[1] + dy, base[2] + dz), wx * wy * wz))
        return weights

    def forward_prop(
        self,
        Psi0,
        position,
        u_proj = (1,0,0),
        v_proj = (0,1,0),
        
    ):
        # projection vectors
        u_proj = np.array(u_proj,dtype='float')
        v_proj = np.array(v_proj,dtype='float')
        u_proj /= np.linalg.norm(u_proj)
        v_proj /= np.linalg.norm(v_proj)
        w_proj = np.cross(u_proj,v_proj)
        if np.isclose(np.linalg.norm(w_proj), 0.0):
            raise ValueError("u_proj and v_proj must not be parallel")
        w_proj /= np.linalg.norm(w_proj)

        # generate real space coordinates passing through volume
        if not hasattr(self, "prop_distance"):
            self.make_prop()

        position = np.asarray(position, dtype=float)
        if position.shape != (3,):
            raise ValueError(
                f"position must have shape (3,), got {position.shape}"
            )
        Psi0 = np.asarray(Psi0)
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
        Psi = Psi0.copy()
        for ind, r in enumerate(r_all):
            # trilinear interpolation of structure factor slice from 6D volume
            SF = np.zeros(Psi0.shape, dtype=self.array.dtype)
            for (ix, iy, iz), weight in self._trilinear_real_weights(r):
                if np.isclose(weight, 0.0):
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
                )

            # transmit
            Psi = np.fft.fft2(np.fft.ifft2(Psi) * np.fft.ifft2(SF))

            # propagate
            if ind < len(r_all) - 1:
                Psi *= self.prop


        return Psi
