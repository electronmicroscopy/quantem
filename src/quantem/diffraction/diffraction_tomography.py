from itertools import permutations, product
from typing import Self, Any, Literal, Sequence

from numpy.typing import NDArray
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from quantem.core.datastructures.dataset import Dataset
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.dataset6d import Dataset6d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import electron_wavelength_angstrom, tqdmnd
from quantem.core.utils.validators import validate_gt
from quantem.tomography.dataset_models import (
    DatasetModelType,
    DatasetConstraintsType,
    DatasetConstraintParams,
)
from quantem.tomography.object_models import (
    ObjectModelType,
    ObjectPixelated,
    ObjConstraintsType,
    ObjConstraintParams,
)


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
        device: str | int = "cpu",
        obj_model: ObjectPixelated | None = None,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError(
                "Use DiffractionTomography.from_array(), "
                ".from_dataset(), or .from_test() to instantiate this class."
            )

        self.dataset = dataset
        self.device = device
        self.obj_model = obj_model

    @classmethod
    def from_array(
        cls,
        array: NDArray,
        obj_model: ObjectPixelated | None = None,
        name: str | None = None,
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: NDArray | tuple | list | float | int | None = None,
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
            # obj_model=obj_model,
            name=name if name is not None else "Diffraction tomography dataset",
            origin=origin,
            sampling=sampling,
            units=units if units is not None else list(cls.axis_labels),
            signal_units=signal_units,
            metadata=metadata,
        )
        return cls.from_dataset(dataset, obj_model=obj_model, energy=energy, wavelength=wavelength)

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset | Dataset6d,
        obj_model: ObjectPixelated | None = None,
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
            # obj_model=obj_model,
            name=dataset.name,
            origin=dataset.origin,
            sampling=dataset.sampling,
            units=dataset.units,
            signal_units=dataset.signal_units,
            metadata=metadata,
        )

        return cls(dataset=dataset1, obj_model= obj_model,_token=cls._token)

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
        """
        Creates reciprocal space grids based on diffraction shape and sampling.

        Parameters
        ----------
        shape: torch.Tensor
            3D diffraction shape (kz,ky,kx)
        sampling: torch.Tensor
            3D diffraction space sampling (kz,ky,kx)

        Returns
        -------
        (kz, ky, kx): tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Reciprocal space grid
        """
        shape = tuple(int(n) for n in shape)
        dk = torch.asarray(sampling, dtype=float)
        if len(shape) != 3:
            raise ValueError(f"shape must have length 3, got {shape}")
        if dk.shape != (3,):
            raise ValueError(f"sampling must have length 3, got shape={dk.shape}")
        kx = torch.fft.fftfreq(shape[2], d=1 / (shape[2] * dk[2]))[None, None, :]
        ky = torch.fft.fftfreq(shape[1], d=1 / (shape[1] * dk[1]))[None, :, None]
        kz = torch.fft.fftfreq(shape[0], d=1 / (shape[0] * dk[0]))[:, None, None]
        return kz, ky, kx

    def _update_reciprocal_coordinates(self) -> None:
        self.kx, self.ky, self.kz = self._make_reciprocal_grids(
            self.diffraction_shape,
            self.dataset.sampling[3:],
        )

    @staticmethod
    def _sample_complex_volume_trilinear(
        volume: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Samples (6D) volume at specified grid points ((-1,-1): top left corner, (1,1): bottom right corner)
        
        Parameters
        ----------
        volume: torch.Tensor
            Entire 6D volume 
        coords: torch.Tensor
            3D location of structure factor(s) being sampled in the full volume (x,y,z)

        Returns
        -------
        mapped_grid: torch.Tensor
            Sampled intensity value at each grid point
        """
        # separates volume into real and imaginary for coordinate mapping
        # pads volume for trilinear interpolation in grid_sample
        vol = volume.to(torch.complex64)
        real = vol.real[None, None, ...]
        imag = vol.imag[None, None, ...]

        # puts grid in order (x,y,z) mapped between values (-1, 1) for grid_sample
        Nz, Ny, Nx = volume.shape
        # z = coords[0]
        # y = coords[1]
        # x = coords[2]
        # grid = torch.stack(
        #     (
        #         2.0 * (coords[2] / (Nx-1.0))-1.0,
        #         2.0 * (coords[1] / (Ny-1.0))-1.0,
        #         2.0 * (coords[0] / (Nz-1.0))-1.0
        #     ),
        #     dim = -1
        # )[None, None, ...]

        # puts grid in order x,y,z --> output from coords is in shape x,y,z
        grid = torch.stack(
            (
                2.0 * (coords[0] / (Nx-1.0))-1.0,
                2.0 * (coords[1] / (Ny-1.0))-1.0,
                2.0 * (coords[2] / (Nz-1.0))-1.0
            ),
            dim = -1
        )[None, None, ...]

        # gx = 2.0 * (x / (Nx-1.0))-1.0
        # gy = 2.0 * (y / (Ny-1.0))-1.0
        # gz = 2.0 * (z / (Nz-1.0))-1.0

        # grid = torch.stack((gx,gy,gz), dim = -1)[None, None, ...]
        map_real = F.grid_sample(real, grid, mode='bilinear', padding_mode='zeros',align_corners=True)
        map_imag = F.grid_sample(imag, grid, mode='bilinear', padding_mode='zeros',align_corners=True)

        return (map_real + 1j * map_imag).squeeze()
        # return map_real + 1j * map_imag

    @classmethod
    def _rotate_complex_volume_zxz(
        cls,
        volume: torch.Tensor,
        sampling: tuple[float, float, float] | list[float] | torch.Tensor,
        zxz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rotate a complex reciprocal-space volume using ZXZ Euler angles.
        
        Parameters
        ----------
        volume: torch.Tensor
            3D reciprocal space volume (z,y,x)
        sampling: torch.Tensor
            3D reciprocal space sampling (z,y,x)
        zxz: torch.Tensor
            Euler angles for rotation (zxz)

        Returns
        -------
        sampled_grid: torch.Tensor
            Sampled intensity values at each point specified by reciprocal grid from volume and sampling
        """
        if volume.dim() != 3:
            raise ValueError(f"volume must be 3D, got ndim={volume.dim()}")

        dk = torch.tensor(sampling, dtype=torch.float32)
        if dk.shape != (3,):
            raise ValueError(f"sampling must have length 3, got shape={dk.shape}")

        shape = torch.tensor(volume.shape, dtype=torch.int32)
        kz, ky, kx = cls._make_reciprocal_grids(shape, dk)
        # kx, ky, kz = cls._make_reciprocal_grids(shape, dk)
        k_grid = torch.stack(torch.broadcast_tensors(kx, ky, kz), axis=-1)

        rot = Rotation.from_euler("zxz", zxz)
        k_source = rot.inv().apply(k_grid.reshape(-1, 3)).reshape(*volume.shape, 3)
        coords = torch.stack(
            [
                torch.remainder(k_source[..., 0] / dk[2], shape[2]),
                torch.remainder(k_source[..., 1] / dk[1], shape[1]),
                torch.remainder(k_source[..., 2] / dk[0], shape[0]),
            ],
            axis=0,
        )

        return cls._sample_complex_volume_trilinear(volume, coords)

    @staticmethod
    def _make_vacuum_sf(
        diffraction_shape: tuple[int, int, int],
        dtype=torch.complex128,
    ) -> torch.Tensor:
        """
        Structure factor of a vacuum cell — identity transmission in k-space.
        """
        sf = torch.zeros(diffraction_shape, dtype=dtype)
        sf[0, 0, 0] = 1.0
        return sf

    @property
    def dataset(self) -> Dataset6d:
        return self._dataset

    @dataset.setter
    def dataset(self, value: Dataset6d):
        if not isinstance(value, Dataset6d):
            raise TypeError(f"dataset must be a Dataset6d, got {type(value)}")
        self._dataset = value
        self._update_reciprocal_coordinates()
        # Invalidate the material-vs-vacuum cell cache used by forward_prop.
        self._material_mask_cache: torch.Tensor | None = None

    def _get_material_mask(self, tol: float = 1e-9) -> torch.Tensor:
        """Bool array marking which (ix, iy, iz) real-space cells contain material.

        A vacuum cell stores `SF[0, 0, 0] = 1` and zeros elsewhere, so the
        absolute-sum over the diffraction axes equals 1. Material cells exceed
        this baseline. The result is cached and invalidated when `dataset` is
        reassigned.
        """
        if getattr(self, "_material_mask_cache", None) is None:
            abs_sum = torch.abs(self.array).sum(axis=(3, 4, 5))
            self._material_mask_cache = abs_sum > 1.0 + tol
        return self._material_mask_cache

    @property
    def array(self) -> torch.Tensor:
        device = torch.device(self.device)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device(device)
        cached = getattr(self, "_array_cache", None)

        if cached is None or cached.device != device:
            data = self.dataset.array
            if not isinstance(data, torch.Tensor):
                data = torch.from_numpy(data)
                self._array_cache = data.to(device)
        
        return self._array_cache

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
    
    @property
    def real_sampling(self) -> tuple[float, float, float]:
        return self.dataset.sampling[:3]


    def make_prop(
        self,
        prop_distance = None,
        shape: tuple[int, int] | list[int] | torch.Tensor | None = None,
        antialias_fraction: float = 0.9,
        antialias_softness: float = 0.05,
    ):
        """Build the Fresnel propagator with anti-aliasing folded in to prevent undersampling.

        The anti-aliasing band-limit (default cutoff = 0.9 * k_Nyquist) is
        applied as part of the propagator only, so it is multiplied into the
        wave each propagation step and never separately. There is no
        per-slice mask on the transmission step.

        Parameters
        ----------
        prop_distance: float
            Z distance for multislice wave propagation
        shape: torch.Tensor
            3D diffraction shape
        antialias_fraction: float
            Cutoff relative to the Nyquist limit (2/3 by default, matching the standard multislice anti-aliasing rule)
        antialias_softness: float
            Amount of blurring for edges (roll-off width)

        Returns
        -------
        prop: torch.Tensor
            Returns propagator
        """
        device = torch.device(self.device)
        # device = torch.device(device)
        if prop_distance is None:
            # prop_distance = self.dataset.sampling[2]
            prop_distance = self.dataset.sampling[0]
        self.prop_distance = prop_distance
        if shape is None:
            # shape = self.diffraction_shape[:2]
            shape = self.diffraction_shape[1:]
        ku, kv = self._make_planar_reciprocal_grids(shape, self.dataset.sampling[3:5])
        ku = ku.to(device)
        kv = kv.to(device)
        bare_prop = (torch.exp(-1j * torch.pi * self.wavelength * (ku**2 + kv**2) * prop_distance)).to(device)
        self.antialias_mask = self._make_antialias_mask(
            shape = shape,
            sampling = self.dataset.sampling[3:5],
            fraction = antialias_fraction,
            softness = antialias_softness,
        ).to(device)
        self.prop = bare_prop * self.antialias_mask
        return self.prop

    @staticmethod
    def _make_antialias_mask(
        shape: tuple[int, int] | list[int] | torch.Tensor,
        sampling: tuple[float, float] | list[float] | torch.Tensor,
        fraction: float = 2.0 / 3.0,
        softness: float = 0.05,
    ) -> torch.Tensor:
        """
        Creates circular band-limit mask in k-space with a soft cosine roll-off.

        Paramaters
        ----------
        shape: torch.Tensor
            2D diffraction shape (x,y)
        sampling: torch.Tensor
            2D reciprocal space sampling (x,y)
        fraction: float
            `Fraction` is the cutoff relative to the Nyquist limit (2/3 by default, matching the standard multislice anti-aliasing rule). 
        softness: float    
            `Softness` is the roll-off width, also relative to Nyquist; setting it to 0 gives a hard cutoff. The mask is 1 inside the cutoff, smoothly drops to 0 outside.
        
        Returns
        -------
        mask: torch.Tensor
            Mask in k-space using soft cosine roll-off
        """
        ku, kv = DiffractionTomography._make_planar_reciprocal_grids(shape, sampling)
        k_radial = torch.sqrt(ku**2 + kv**2)
        dk = torch.tensor(sampling, dtype=torch.float32)
        n_pix = torch.tensor(shape, dtype=torch.int32)
        k_nyquist = float(torch.min(n_pix * dk / 2.0))
        cutoff = float(fraction) * k_nyquist

        if softness <= 0.0:
            return (k_radial <= cutoff).to(torch.float64)

        width = max(float(softness) * k_nyquist, torch.finfo(float).eps)
        mask = 0.5 * (1.0 - torch.cos(torch.pi * torch.clip((cutoff + width - k_radial) / width, 0.0, 1.0)))
        return mask

    def make_probe_aperture(
        self,
        probe_k_max: float,
        dp_shape: tuple[int, int] | torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Build a top-hat aperture probe in reciprocal space.

        With `normalize=True` (default), the returned probe satisfies
        `sum(|Psi|**2) = 1`. Vacuum propagation preserves this norm.

        Parameters
        ----------
        probe_k_max: float
            Max k-vector magnitude (A^-1)
        dp_shape: torch.Tensor
            3D diffraction shape

        Returns
        -------
        Psi0: torch.Tensor
            2D probe aperture
        """
        if dp_shape is None:
            dp_shape = self.diffraction_shape[1:]
            # dp_shape = self.diffraction_shape[:2]
        dp_shape = (int(dp_shape[0]), int(dp_shape[1]))

        ku, kv = self._make_planar_reciprocal_grids(dp_shape, self.dataset.sampling[3:5])
        k_radial = torch.sqrt(ku**2 + kv**2)
        dk = float(self.dataset.sampling[3])
        aper = torch.clip((probe_k_max - k_radial) / dk + 0.5, 0.0, 1.0)
        Psi0 = aper.to(torch.complex128)
        if normalize:
            norm = float(torch.sqrt(torch.sum(torch.abs(Psi0) ** 2)))
            if norm > 0.0:
                Psi0 = Psi0 / norm
        return Psi0

    @staticmethod
    def _make_planar_reciprocal_grids(
        shape: tuple[int, int] | list[int] | torch.Tensor,
        sampling: tuple[float, float] | list[float] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Makes 2D reciprocal grids

        Parameters
        ----------
        shape: torch.Tensor
            2D reciprocal space shape (x,y)
        sampling: torch.Tensor
            2D reciprocal space sampling

        Returns
        -------
        ku: torch.Tensor
            Reciprocal basis vector in fast-scan direction
        kv: torch.Tensor
            Reciprocal basis vector in slow-scan direction
        """
        shape = tuple(int(n) for n in shape)
        dk = torch.tensor(sampling, dtype=float)
        if len(shape) != 2:
            raise ValueError(f"shape must have length 2, got {shape}")
        if dk.shape != (2,):
            raise ValueError(f"sampling must have length 2, got shape={dk.shape}")

        ku = torch.fft.fftfreq(shape[0], d=1 / (shape[0] * dk[0]))[:, None]
        kv = torch.fft.fftfreq(shape[1], d=1 / (shape[1] * dk[1]))[None, :]
        # ku = torch.fft.fftfreq(shape[0], d=1 / (shape[0] * dk[0]))[:, None]
        # kv = torch.fft.fftfreq(shape[1], d=1 / (shape[1] * dk[1]))[None, :]
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
        device: str | int = 'cpu',
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = torch.device(device)

        """
        Determines which reciprocal space voxel (kz, ky, kx) is sampled for each diffraction pattern on the detector.

        Parameters
        ----------
        shape_2d: torch.Tensor[int,int]
            2D diffraction shape (ky, kx)
        sampling_2d: torch.Tensor (float, float)
            2D diffraction space sampling (dky, dkx)
        u_proj: torch.Tensor[float, float, float]]
            Unit vector in fast-scan direction (sample frame)
        v_proj: torch.Tensor[float, float, float]
            Unit vector in slow-scan direction (sample frame)
        sampling_3d: torch.Tensor[float, float, float]
            Sampling for 3d structure factor volume
        shape_3d: torch.Tensor[int, int, int]
            Shape of 3d structure factor volume
        device: str or int
            cpu or cuda:int

        Returns
        -------
        coords: torch.Tensor
            Locations of 2D slices of structure factor corresponding to specified detector pixel(s)
            (kx, ky, kz) --> units of pixels
        ku2: torch.Tensor
            Detector coordinates (fast-scan direction)
        kv2: torch.Tensor
            Detector coordinates (slow-scan direction)
        """
        # converts to kz, ky and dkx, dky
        shape2d = torch.asarray(shape_2d, device = device)
        sampling2d = torch.asarray(sampling_2d, device = device)

        ku, kv = cls._make_planar_reciprocal_grids(torch.flip(shape2d, dims = [0]), torch.flip(sampling2d, dims = [0])) # from zyx to xyz
        ku2, kv2 = torch.broadcast_tensors(ku, kv)
        ku2 = ku2.to(device)
        kv2 = kv2.to(device)
        u_proj = torch.asarray(u_proj, dtype=torch.float32, device=device)
        v_proj = torch.asarray(v_proj, dtype=torch.float32, device=device)
        k_xyz = ku2[..., None] * u_proj[None, None, :] + kv2[..., None] * v_proj[None, None, :] # A^-1, xyz

        dk = torch.asarray(sampling_3d, dtype=torch.float32) # in z,y,x
        shape_k = torch.asarray(shape_3d, dtype=torch.int32)
        coords = torch.stack(
            [
                torch.remainder(k_xyz[..., 0] / dk[2], shape_k[2]),
                torch.remainder(k_xyz[..., 1] / dk[1], shape_k[1]),
                torch.remainder(k_xyz[..., 2] / dk[0], shape_k[0]),
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
        device: str | int = 'cpu',
    ) -> torch.Tensor:
        """Generate (equally spaced) index-space samples along a ray spanning the slab z-extent.

        The slab is treated as a thin film of finite z-thickness whose (x, y)
        footprint is the material region. The ray's t-range is fixed by the
        z-axis only — `t` covers `z = 0 .. shape[0] - 1` — so tilted rays that
        leave the (x, y) footprint still produce slices. Callers are
        expected to treat those out-of-footprint slices as vacuum.
        
        Parameters
        ----------
        position: torch.Tensor
            (x,y,z) position on 6D volume (where the propagation occurs)
            Should be in the form (x,y,z) and in Angstroms
        direction: torch.Tensor
            Direction of vector propagating through the volume (based on tilt)
            Should be in the form (x,y,z)
        shape: torch.Tensor
            3D real shape from 6D volume
        sampling: torch.Tensor
            3D real space sampling (units: A/px)
        prop_distance: float
            Distance between top (minimum z) and bottom (maximum z) of volume (units: A)

        Returns
        -------
        ray_coords: torch.Tensor
            Equispaced coordinates along ray specified by starting position, direction, and z extent of 6D volume
        """
        device = torch.device(device)

        r0 = torch.asarray(position, dtype=torch.float32) # still xyz
        w_proj = torch.asarray(direction, dtype=torch.float32, device = device) # still xyz
        shape = torch.flip(torch.asarray(shape, dtype=torch.int32, device = device), dims = [0]) # now xyz
        sampling = torch.flip(torch.asarray(sampling, dtype=torch.float32, device = device), dims = [0]) # now xyz

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

        # t_min = -torch.inf
        # t_max = torch.inf
        # for axis in range(3):
        if torch.isclose(dr[2], torch.tensor(0.0)):
            # beam parallel to slab - cannot traverse z
            return torch.empty((0, 3), dtype=torch.float32)

        # (p0_z - r0_z)/d_z for top and bottom of volume
        t0 = (0.0 - r0[2]) / dr[2] 
        t1 = ((shape[2] - 1) - r0[2]) / dr[2]
        t_min, t_max = (t0, t1) if t0 <= t1 else (t1, t0)

        # Sample the full line segment through the slab, including both negative
        # and positive steps from the itorchut position when they remain in bounds.
        eps = 1e-9
        n_min = int(torch.ceil(t_min - eps))
        n_max = int(torch.floor(t_max + eps))
        if n_max <= n_min:
        # if n_max < n_min:
            return torch.empty((0, 3), dtype=torch.float32)

        steps = torch.arange(n_min, n_max + 1, dtype=torch.float32, device=device)[:, None]
        # print("steps test:", r0[None, :] + steps * dr[None, :])
        return r0[None, :] + steps * dr[None, :]

    # @staticmethod
    def _trilinear_real_weights(self,position: torch.Tensor) -> list[tuple[tuple[int, int, int], float]]:
        """Return real-space trilinear neighbors and weights for one sample position.
        
        Parameters
        ----------
        position: torch.Tensor
            3D coordinate of position in volume 
            Given as (x,y,z)
        
        Returns
        -------
        weights: list[tuple[tuple[int, int, int], float]]
            Intensity weights and real-space trilinear neighbors from position as xyz
        """
        device = torch.device(self.device)
        base = torch.asarray(torch.floor(position), dtype = torch.int32, device = device)
        pos = torch.asarray(position, device = device)
        frac = pos - base
        weights: torch.Tensor[tuple[tuple[int, int, int], float]] = []
        for dx in range(2):
            wx = frac[0].item() if dx else 1.0 - frac[0].item()
            # wx = frac[0] if dx else 1.0 - frac[0]
            for dy in range(2):
                wy = frac[1].item() if dy else 1.0 - frac[1].item()
                for dz in range(2):
                    wz = frac[2].item() if dz else 1.0 - frac[2].item()
                    weights.append(((int(base[2] + dz), int(base[1] + dy), int(base[0] + dx)), float(wx * wy * wz)))
                    # returns as z,y,x
                    # weights.append(((base[0] + dx, base[1] + dy, base[2] + dz), (wx * wy * wz).to(torch.float32)))
        return weights

    def get_ray_coords(
        self,
        Psi0,
        position,
        u_proj = (1,0,0),
        v_proj = (0,1,0),
    ):
        """
        Returns coordinates along a ray for a given position and sample tilt.

        Parameters
        ----------
        Psi0: torch.Tensor
            Probe aperture
        position: torch.Tensor
            3D position in the real space volume as xyz
        u_proj: torch.Tensor
            Unit vector in fast-scan direction
        v_proj
            Unit vector in slow-scan direction

        Returns 
        -------
        ray_coords_dict: dict{torch.Tensor}
            Returns coordinates along rays ('ray_coords', xyz), structure factor slice coordinates ('k_slice_coords'), and unit vectors along fast-scan ('u_proj'), slow_scan ('v_proj'), and beam ('w_proj') directions
        """
        device = torch.device(self.device)
        Psi0 = torch.as_tensor(Psi0, device=device)
        position = torch.as_tensor(position, device=device)

        # projection vectors
        u_proj = torch.tensor(u_proj,dtype=torch.float32, device = device)
        v_proj = torch.tensor(v_proj,dtype=torch.float32, device = device)
        u_proj /= torch.linalg.norm(u_proj)
        v_proj /= torch.linalg.norm(v_proj)
        w_proj = torch.linalg.cross(u_proj,v_proj)
        if torch.isclose(torch.linalg.norm(w_proj), torch.tensor(0.0)):
            raise ValueError("u_proj and v_proj must not be parallel")
        w_proj /= torch.linalg.norm(w_proj)

        # generate real space coordinates passing through volume
        if not hasattr(self, "prop_distance"):
            self.make_prop().to(device)

        position = torch.asarray(position, dtype=torch.float32).squeeze()
        if position.shape != (3,):
            raise ValueError(
                f"position must have shape (3,), got {position.shape}"
            )
        Psi0 = torch.tensor(Psi0)
        if Psi0.ndim != 2:
            raise ValueError(f"Psi0 must be 2D, got ndim={Psi0.ndim}")

        self.k_slice_coords, _, _ = self._make_projected_k_slice_coords(
            shape_2d=Psi0.shape,
            sampling_2d=self.dataset.sampling[3:5],
            u_proj=u_proj,
            v_proj=v_proj,
            sampling_3d=self.dataset.sampling[3:],
            shape_3d=self.diffraction_shape,
        )

        if not hasattr(self, "prop") or self.prop.shape != Psi0.shape:
            self.make_prop(prop_distance=self.prop_distance, shape=Psi0.shape)

        rays = self._generate_slab_ray_coordinates(
            position=position,
            direction=w_proj,
            shape=self.array.shape[:3],
            sampling=self.dataset.sampling[:3],
            prop_distance=self.prop_distance,
        )
        return {'ray_coords': rays, 
                'k_slice_coords': self.k_slice_coords,
                'u_proj': u_proj,
                'v_proj': v_proj,
                'w_proj': w_proj,
        }

    def forward_prop(
        self,
        Psi0,
        position,
        u_proj = (1,0,0),
        v_proj = (0,1,0),
        phase_only: bool = True,
    ):
        """
        Returns exit wave propagated through volume at specified position and tilt

        Parameters
        ----------
        Psi0: torch.Tensor
            Probe aperture
        positions: torch.Tensor
            3D position in the real space volume (xyz)
        u_proj: torch.Tensor
            Unit vector in fast-scan direction
        v_proj
            Unit vector in slow-scan direction

        Returns
        -------
        Psi: torch.Tensor
            Exit wave from Psi0 propagating through the volume at specified position and tilt
        """
        device = torch.device(self.device)
        Psi0 = torch.as_tensor(Psi0, device=device)
        position = torch.as_tensor(position, device=device)

        r_all = self.get_ray_coords(Psi0, position, u_proj, v_proj)['ray_coords']
        # multislice propagation through volume
        material_mask = self._get_material_mask().to(device)
        material_slice_cache = getattr(self, "_material_slice_cache", None)
        Nz, Ny, Nx = self.real_shape
        Psi = Psi0.clone()

        for ind, r in enumerate(r_all):
            # trilinear interpolation of structure factor slice from 6D volume for
            # this ray position. Vacuum cells (SF = delta at the origin) only
            # contribute `weight` to the central pixel, so we short-circuit
            # them instead of calling `ndi.map_coordinates`. Material slices
            # may be precomputed once per `simulate_4dstem` tilt and looked up
            # from `self._material_slice_cache`.
            SF = torch.zeros(Psi0.shape, dtype=self.array.dtype, device = device)
            weight_sum = 0.0
            for (iz,iy,ix), weight in self._trilinear_real_weights(r):
                if weight == 0.0:
                    continue
                if not (0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz):
                    continue
                weight_sum += weight
                if material_mask[iz, iy, ix]:
                # if material_mask[ix, iy, iz]:
                    if material_slice_cache is not None:
                        SF += weight * material_slice_cache[iz, iy, ix]
                        # SF += weight * material_slice_cache[ix, iy, iz]
                    else:
                        SF += weight * self._sample_complex_volume_trilinear(
                            self.array[iz, iy, ix],
                            # self.array[ix, iy, iz],
                            self.k_slice_coords,
                        ).to(SF.dtype)
                else:
                    SF[0,0] += weight

            if weight_sum > 0.0:
                SF /= weight_sum
                # Convention: the 2D transmission function `T_2d` satisfies
                # `T_2d[0, 0] = num_pixels` for vacuum, which corresponds to
                # `t_2d = 1` in real space. The 6D SF is stored normalized
                # (`SF[..., 0, 0, 0] = 1` for vacuum), so we scale by
                # `num_pixels` here to recover that convention.
                num_pixels = Psi.numel()          # total number of elements (Nx*Ny)
                T_2d = num_pixels * SF
                t_real = torch.fft.ifft2(T_2d)
                if phase_only:
                    # Enforce a unitary phase-grating transmission: `|t| = 1`
                    # everywhere, so probe norm is preserved per slice.
                    # Vacuum still falls out as identity (angle(1) = 0).
                    t_real = (torch.exp(1j * torch.angle(t_real))).to(device)
                Psi = torch.fft.fft2(torch.fft.ifft2(Psi) * t_real)
            # else: ray is outside the slab footprint at this slice — vacuum,
            # so skip transmission and only apply Fresnel propagation below.

            # Propagate. The band-limit anti-aliasing is folded into `self.prop`,
            # so it is applied as part of propagation and only at that step.
            if ind < len(r_all) - 1:
                Psi *= self.prop

        return Psi
    
    def forward_prop_from_points(
        self,
        Psi0,
        ray_coords,
        phase_only: bool = True,
    ):
        """
        Returns exit wave propagated through volume at specified tilt for precalculated ray coordinates

        Parameters
        ----------
        Psi0: torch.Tensor
            Probe aperture
        ray_coords: torch.Tensor
            Tensor of 3D positions in the real space volume (xyz)
        u_proj: torch.Tensor
            Unit vector in fast-scan direction
        v_proj
            Unit vector in slow-scan direction

        Returns
        -------
        Psi: torch.Tensor
            Exit wave from Psi0 propagating through the volume at specified position and tilt
        """
        device = torch.device(self.device)
        Psi0 = torch.as_tensor(Psi0, device=device)

        r_all = ray_coords

        # multislice propagation through volume
        material_mask = self._get_material_mask().to(device)
        material_slice_cache = getattr(self, "_material_slice_cache", None)
        Nz, Ny, Nx = self.real_shape
        Psi = Psi0.clone()

        for ind, r in enumerate(r_all):
            # trilinear interpolation of structure factor slice from 6D volume for
            # this ray position. Vacuum cells (SF = delta at the origin) only
            # contribute `weight` to the central pixel, so we short-circuit
            # them instead of calling `ndi.map_coordinates`. Material slices
            # may be precomputed once per `simulate_4dstem` tilt and looked up
            # from `self._material_slice_cache`.
            SF = torch.zeros(Psi0.shape, dtype=self.array.dtype, device = device)
            weight_sum = 0.0
            for (iz, iy, ix), weight in self._trilinear_real_weights(r):
                if weight == 0.0:
                    continue
                if not (0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz):
                    continue
                weight_sum += weight
                if material_mask[iz, iy, ix]:
                    if material_slice_cache is not None:
                        SF += weight * material_slice_cache[iz, iy, ix]
                    else:
                        SF += weight * self._sample_complex_volume_trilinear(
                            self.array[iz, iy, ix],
                            self.k_slice_coords,
                        ).to(SF.dtype)
                else:
                    SF[0,0] += weight

            if weight_sum > 0.0:
                SF /= weight_sum
                # Convention: the 2D transmission function `T_2d` satisfies
                # `T_2d[0, 0] = num_pixels` for vacuum, which corresponds to
                # `t_2d = 1` in real space. The 6D SF is stored normalized
                # (`SF[..., 0, 0, 0] = 1` for vacuum), so we scale by
                # `num_pixels` here to recover that convention.
                num_pixels = Psi.numel()          # total number of elements (Nx*Ny)
                T_2d = num_pixels * SF
                t_real = torch.fft.ifft2(T_2d)
                if phase_only:
                    # Enforce a unitary phase-grating transmission: `|t| = 1`
                    # everywhere, so probe norm is preserved per slice.
                    # Vacuum still falls out as identity (angle(1) = 0).
                    t_real = (torch.exp(1j * torch.angle(t_real))).to(device)
                Psi = torch.fft.fft2(torch.fft.ifft2(Psi) * t_real)
            # else: ray is outside the slab footprint at this slice — vacuum,
            # so skip transmission and only apply Fresnel propagation below.

            # Propagate. The band-limit anti-aliasing is folded into `self.prop`,
            # so it is applied as part of propagation and only at that step.
            if ind < len(r_all) - 1:
                Psi *= self.prop

        return Psi
    
    @staticmethod
    def _resolve_zxz_deg(
        tilt_x_deg: float | None,
        zxz_deg: torch.Tensor | tuple | list | None,
    ) -> torch.Tensor:
        """Normalize tilt/zxz inputs to a length-3 ZXZ vector in degrees."""
        if zxz_deg is not None:
            zxz_arr = torch.asarray(zxz_deg, dtype=torch.float32)
            if zxz_arr.shape != (3,):
                raise ValueError(f"zxz_deg must have shape (3,), got {zxz_arr.shape}")
            return zxz_arr
        return torch.tensor([0.0, float(tilt_x_deg or 0.0), 0.0], dtype=torch.float32)

    def projection_axes(
        self,
        tilt_x_deg: float | None = 0.0,
        zxz_deg: torch.Tensor | tuple | list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (u_proj, v_proj, w_proj) in sample-frame coordinates.

        `u_proj` is the fast-scan direction, `v_proj` the slow-scan direction,
        and `w_proj` the beam direction. The sample is rotated by the given
        ZXZ Euler angles relative to the (lab, beam-down-z) frame; the lab
        axes (1,0,0), (0,1,0), (0,0,1) are then expressed in the sample frame.
        """
        zxz_arr = self._resolve_zxz_deg(tilt_x_deg, zxz_deg)
        rot = Rotation.from_euler("zxz", zxz_arr, degrees=True)
        # Lab→sample mapping: apply the inverse rotation to lab basis vectors.
        u_proj = torch.tensor(rot.inv().apply([1.0, 0.0, 0.0]))
        v_proj = torch.tensor(rot.inv().apply([0.0, 1.0, 0.0]))
        w_proj = torch.linalg.cross(u_proj, v_proj)
        w_norm = torch.linalg.norm(w_proj)
        if w_norm == 0:
            raise ValueError("u_proj and v_proj must not be parallel")
        w_proj /= w_norm
        return u_proj, v_proj, w_proj

    def setup_dataloader(
            self,
            dataset: Dataset | DatasetModelType,
            batch_size: int = 1024,
            num_workers: int = 32,
            val_fraction: float = 0.15,
    ):
        pin_mem = self.device == 'cuda'
        generator = torch.Generator()
        generator.manual_seed(42)
        dataset_size = len(dataset)
        print((1-val_fraction)*dataset_size)

        if val_fraction > 0.0:
            train_size = int((1-val_fraction)*dataset_size)
            train_dataset, val_dataset = random_split(dataset, [train_size, dataset_size-train_size], generator=generator)
            # train_dataset, val_dataset = random_split(dataset, [(1-val_fraction)*dataset_size, val_fraction*dataset_size], generator=generator)
        else:
            train_dataset = dataset
            val_dataset = None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            num_workers = num_workers,
            pin_memory=pin_mem,
            drop_last=False,
            persistent_workers = num_workers > 0
            )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size = batch_size * 4, # less memory than training
            num_workers = num_workers,
            pin_memory=pin_mem,
            drop_last=False,
            persistent_workers = num_workers > 0
            )
        return train_dataloader, val_dataloader
        
    #  put this under DiffractionToomography?
    def reconstruct_pix(
        self,
        probe_k_max: float = 0.10,
        tilt_x_deg: float | None = 0.0,
        zxz_deg: torch.Tensor | tuple | list | None = None,
        scan_step: float | tuple[float, float] = 1.0,
        scan_shape: tuple[int, int] = (11, 11),
        scan_origin: torch.Tensor | tuple | list | None = None,
        num_iters: int = 10,
        batch_size: int = 1024,
        num_workers: int = 32,
        reset: bool = False,
        lr_init: float = 1e-3,
        optimizer_params: dict | None = None,
        scheduler_params: dict | None = None,
        obj_constraints: dict | ObjConstraintsType | None = None,
        dset_constraints: dict | DatasetConstraintsType | None = None,
        val_fraction: float = 0.15,
        loss_func_kwargs: dict = {},
        reset_dset: DatasetModelType | None = None,
        show_metrics: bool = False,
        show_every: int = 1,
    ):
        self.obj_model.to(self.device) #NEED TO MAKE OBJECT FOR THIS TO NOT BE NONE

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction 

        if reset or not hasattr(self,'sf_learned'):
            sf_learned = torch.ones(self.array.shape, dtype = torch.complex128, device = self.device, requires_grad = True)
        
        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
            self.optimizer = torch.optim.AdamW([sf_learned],lr = lr_init, **optimizer_params)

        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_params)

        if obj_constraints is not None:
            if isinstance(obj_constraints, dict):
                obj_constraints = ObjConstraintParams.parse_dict(obj_constraints)

        if dset_constraints is not None:
            if isinstance(dset_constraints, dict):
                dset_constraints = DatasetConstraintParams.parse_dict(dset_constraints)

            self.dset_constraints = dset_constraints

        if not hasattr(self, "dataloader") or reset_dset is not None:
            self.dataloader, self.val_dataloader =self.setup_dataloader(
                    self.dataset,
                    batch_size,
                    num_workers,
                    val_fraction,
                    )
            # print("dataloader lengths:",len(self.dataloader), len(self.val_dataloader))
            idxs, data_vals = enumerate(self.dataloader)
            print(data_vals)
        Psi0 = self.make_probe_aperture(
            probe_k_max=probe_k_max,
            dp_shape= self.diffraction_shape,
            normalize=True,
        )

        loss_fxn = torch.nn.MSELoss()

        u, v, w = self.projection_axes(tilt_x_deg=tilt_x_deg, zxz_deg=zxz_deg)

        for iter in tqdm.tqdm(range(num_iters)):
            total_loss = torch.tensor(0.0, device = self.device)
            self.obj_model.train()
            for batch_idx, batch in enumerate(self.dataloader):
                print("keys for batching:", batch.keys())
                all_coords = self.get_ray_coords(
                    Psi0,
                    (0,0,0),
                )
                self.all_densities_pred = self.forward_prop(
                    Psi0,
                    (0,0,0),
                    u_proj=u,
                    v_proj=v
                )
                all_densities_target = batch["target_value"].to(self.device, non_blocking=True).float()
                
                batch_loss = loss_fxn(all_densities_pred, all_densities_target)
                batch_loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += batch_loss.detach()
            print("dataloader lengths:",len(self.dataloader), len(self.val_dataloader))
            total_loss = total_loss.item() / len(self.dataloader)

            if self.scheduler is not None:
                self.scheduler.step(loss = total_loss)

            if self.val_dataloader is not None:
                self.dataset.eval()
                self.obj_model.eval()

                with torch.no_grad():
                    val_loss = torch.tensor(0.0, device=self.device)
                    
                    for batch in self.val_dataloader:
                        all_coords = self.get_ray_coords(
                            Psi0,
                            (0,0,0),
                        )
                        all_densities_pred = self.forward_prop(
                            Psi0,
                            (0,0,0),
                        )
                        all_densities_target = batch["target_value"].to(self.device, non_blocking=True).float()

                        batch_val_loss_fxn = loss_fxn
                        batch_val_loss = batch_val_loss_fxn(all_densities_pred, all_densities_target)

                        val_loss += batch_val_loss.detach()

                    avg_val_loss = val_loss.item() / len(self.val_dataloader)

            if show_metrics:
                metrics = torch.tensor([total_loss], device = self.device)
                msg = f"Iter {iter}: Train Loss = {total_loss:.6f}"
                if avg_val_loss:
                    msg += f", Val Loss = {avg_val_loss:.6f}"
                print(msg)

        return {'reconstructed_sf': self.sf_learned.detach().cpu(),
        'training_losses': total_loss,
        'validation_losses': avg_val_loss,}
