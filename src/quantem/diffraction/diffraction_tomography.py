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

    Storage axis order is `[x, y, z, kx, ky, kz]`: the first three axes are
    real space and the last three reciprocal space, with z (and kz) on the
    last axis of each triplet. The beam propagates along +z, so the ray
    t-range spans real-space axis 2. Position and direction arguments are
    likewise ordered (x, y, z).
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
        device: str | int = "cpu",
    ) -> Self:
        """
        Create a DiffractionTomography instance from a 6D torch tensor.
        """
        metadata = cls._merge_beam_metadata(
            metadata=None,
            energy=energy,
            wavelength=wavelength,
        )
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        dataset = Dataset6d.from_array(
            array=array,
            name=name if name is not None else "Diffraction tomography dataset",
            origin=origin,
            sampling=sampling,
            units=units if units is not None else list(cls.axis_labels),
            signal_units=signal_units,
            metadata=metadata,
        )
        return cls.from_dataset(
            dataset, obj_model=obj_model, energy=energy, wavelength=wavelength, device=device
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset | Dataset6d,
        obj_model: ObjectPixelated | None = None,
        energy: float | None = None,
        wavelength: float | None = None,
        device: str | int = "cpu",
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

        return cls(dataset=dataset1, device=device, obj_model=obj_model, _token=cls._token)

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

        Storage convention: diffraction axes are (kx, ky, kz), so kx varies
        along axis 0 and kz along axis 2 (matching the real-space (x, y, z)
        ordering and the original numpy implementation).

        Parameters
        ----------
        shape: torch.Tensor
            3D diffraction shape (kx, ky, kz)
        sampling: torch.Tensor
            3D diffraction space sampling (dkx, dky, dkz)

        Returns
        -------
        (kx, ky, kz): tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Reciprocal space grid
        """
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
        kx, ky, kz = self._make_reciprocal_grids(
            self.diffraction_shape,
            self.dataset.sampling[3:],
        )
        self.kx, self.ky, self.kz = kx, ky, kz

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
        vol = volume.to(torch.complex64)
        real = vol.real[None, None, ...]
        imag = vol.imag[None, None, ...]

        # `coords` rows index the volume's storage axes directly:
        # coords[0] -> axis 0, coords[1] -> axis 1, coords[2] -> axis 2
        # (matching scipy.ndimage.map_coordinates in the original numpy code).
        # grid_sample's last grid dim is ordered (W, H, D) = (axis 2, 1, 0),
        # so the stack order below is reversed relative to the coords rows.
        N0, N1, N2 = volume.shape
        grid = torch.stack(
            (
                2.0 * (coords[2] / (N2 - 1.0)) - 1.0,
                2.0 * (coords[1] / (N1 - 1.0)) - 1.0,
                2.0 * (coords[0] / (N0 - 1.0)) - 1.0,
            ),
            dim=-1,
        )[None, None, ...]

        map_real = F.grid_sample(real, grid, mode='bilinear', padding_mode='zeros',align_corners=True)
        map_imag = F.grid_sample(imag, grid, mode='bilinear', padding_mode='zeros',align_corners=True)

        return (map_real + 1j * map_imag).squeeze()

    @staticmethod
    def _sample_complex_volume_trilinear_batch(
        volumes: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Sample a batch of complex 3D volumes at shared coordinates.

        `volumes` is (K, N0, N1, N2); `coords` follows the same convention as
        `_sample_complex_volume_trilinear` (row i indexes storage axis i).
        Returns (K, H, W) — identical to sampling each volume separately, but
        in a single grid_sample call.
        """
        vol = volumes.to(torch.complex64)
        K = vol.shape[0]
        N0, N1, N2 = vol.shape[1:]
        real = vol.real[:, None]  # (K, 1, N0, N1, N2)
        imag = vol.imag[:, None]

        grid = torch.stack(
            (
                2.0 * (coords[2] / (N2 - 1.0)) - 1.0,
                2.0 * (coords[1] / (N1 - 1.0)) - 1.0,
                2.0 * (coords[0] / (N0 - 1.0)) - 1.0,
            ),
            dim=-1,
        )[None, None, ...].expand(K, 1, *coords.shape[1:], 3)

        map_real = F.grid_sample(real, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        map_imag = F.grid_sample(imag, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return (map_real + 1j * map_imag).reshape(K, *coords.shape[1:])

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
            3D reciprocal space volume, storage axes (kx, ky, kz)
        sampling: torch.Tensor
            3D reciprocal space sampling (dkx, dky, dkz)
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
        kx, ky, kz = cls._make_reciprocal_grids(shape, dk)
        k_grid = torch.stack(torch.broadcast_tensors(kx, ky, kz), axis=-1)

        rot = Rotation.from_euler("zxz", zxz)
        k_source = rot.inv().apply(k_grid.reshape(-1, 3)).reshape(*volume.shape, 3)
        coords = torch.stack(
            [
                torch.remainder(k_source[..., 0] / dk[0], shape[0]),
                torch.remainder(k_source[..., 1] / dk[1], shape[1]),
                torch.remainder(k_source[..., 2] / dk[2], shape[2]),
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

    def _delta_volume(self, like: torch.Tensor) -> torch.Tensor:
        """Constant one-hot 3D delta at [0, 0, 0], matching `like`."""
        key = ("vol", tuple(like.shape), like.dtype, str(like.device))
        cache = getattr(self, "_delta_cache", None)
        if cache is None:
            cache = self._delta_cache = {}
        if key not in cache:
            d = torch.zeros(like.shape, dtype=like.dtype, device=like.device)
            d[0, 0, 0] = 1.0
            cache[key] = d
        return cache[key]

    def _delta_slice(self, like: torch.Tensor) -> torch.Tensor:
        """Constant one-hot 2D delta at [0, 0], matching `like`."""
        key = ("slice", tuple(like.shape), like.dtype, str(like.device))
        cache = getattr(self, "_delta_cache", None)
        if cache is None:
            cache = self._delta_cache = {}
        if key not in cache:
            d = torch.zeros(like.shape, dtype=like.dtype, device=like.device)
            d[0, 0] = 1.0
            cache[key] = d
        return cache[key]

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
            # z is storage axis 2 of real space; slice spacing along the beam.
            prop_distance = self.dataset.sampling[2]
        self.prop_distance = prop_distance
        if shape is None:
            # detector plane = (kx, ky) = first two diffraction axes
            shape = self.diffraction_shape[:2]
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
            # detector plane = (kx, ky) = first two diffraction axes
            dp_shape = self.diffraction_shape[:2]
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
        # detector plane axes = (ku, kv) along (fast, slow) scan directions
        shape2d = torch.asarray(shape_2d, device = device)
        sampling2d = torch.asarray(sampling_2d, device = device)

        ku, kv = cls._make_planar_reciprocal_grids(shape2d, sampling2d)
        ku2, kv2 = torch.broadcast_tensors(ku, kv)
        ku2 = ku2.to(device)
        kv2 = kv2.to(device)
        u_proj = torch.asarray(u_proj, dtype=torch.float32, device=device)
        v_proj = torch.asarray(v_proj, dtype=torch.float32, device=device)
        k_xyz = ku2[..., None] * u_proj[None, None, :] + kv2[..., None] * v_proj[None, None, :] # A^-1, xyz

        # coords row i indexes storage axis i of the SF cell: (kx, ky, kz)
        dk = torch.asarray(sampling_3d, dtype=torch.float32)
        shape_k = torch.asarray(shape_3d, dtype=torch.int32)
        coords = torch.stack(
            [
                torch.remainder(k_xyz[..., 0] / dk[0], shape_k[0]),
                torch.remainder(k_xyz[..., 1] / dk[1], shape_k[1]),
                torch.remainder(k_xyz[..., 2] / dk[2], shape_k[2]),
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

        # positions, directions, shape, and sampling are all in storage order
        # (x, y, z): z is axis 2, and the ray t-range spans the slab z-extent.
        r0 = torch.asarray(position, dtype=torch.float32)
        w_proj = torch.asarray(direction, dtype=torch.float32, device = device)
        shape = torch.asarray(shape, dtype=torch.int32, device = device)
        sampling = torch.asarray(sampling, dtype=torch.float32, device = device)

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

        if torch.isclose(dr[2], torch.tensor(0.0)):
            # beam parallel to slab - cannot traverse z
            return torch.empty((0, 3), dtype=torch.float32)

        # (p0_z - r0_z)/d_z for top and bottom of volume
        t0 = (0.0 - r0[2]) / dr[2]
        t1 = ((shape[2] - 1) - r0[2]) / dr[2]
        t_min, t_max = (t0, t1) if t0 <= t1 else (t1, t0)

        # Sample the full line segment through the slab, including both negative
        # and positive steps from the input position when they remain in bounds.
        eps = 1e-9
        n_min = int(torch.ceil(t_min - eps))
        n_max = int(torch.floor(t_max + eps))
        if n_max < n_min:
            return torch.empty((0, 3), dtype=torch.float32)

        steps = torch.arange(n_min, n_max + 1, dtype=torch.float32, device=device)[:, None]
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
            Intensity weights and real-space trilinear neighbor indices in
            storage order (ix, iy, iz), suitable for `array[ix, iy, iz]`.
        """
        device = torch.device(self.device)
        base = torch.asarray(torch.floor(position), dtype = torch.int32, device = device)
        pos = torch.asarray(position, device = device)
        frac = pos - base
        weights: list[tuple[tuple[int, int, int], float]] = []
        for dx in range(2):
            wx = frac[0].item() if dx else 1.0 - frac[0].item()
            for dy in range(2):
                wy = frac[1].item() if dy else 1.0 - frac[1].item()
                for dz in range(2):
                    wz = frac[2].item() if dz else 1.0 - frac[2].item()
                    weights.append(((int(base[0] + dx), int(base[1] + dy), int(base[2] + dz)), float(wx * wy * wz)))
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
        Nx, Ny, Nz = self.real_shape
        Psi = Psi0.clone()

        for ind, r in enumerate(r_all):
            # trilinear interpolation of structure factor slice from 6D volume for
            # this ray position. Vacuum cells (SF = delta at the origin) only
            # contribute `weight` to the central pixel, so we short-circuit
            # them instead of calling `ndi.map_coordinates`. Material slices
            # may be precomputed once per `simulate_4dstem` tilt and looked up
            # from `self._material_slice_cache`.
            # Cells decompose as `baseline * delta + deviation`: the vacuum
            # baseline (SF[0, 0, 0]) lands analytically on the central pixel
            # so vacuum transmission is exactly the identity; only the Bragg
            # deviation is interpolated. The slice cache stores deviation
            # slices for the same reason.
            SF = torch.zeros(Psi0.shape, dtype=self.array.dtype, device = device)
            weight_sum = 0.0
            for (ix, iy, iz), weight in self._trilinear_real_weights(r):
                if weight == 0.0:
                    continue
                if not (0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz):
                    continue
                weight_sum += weight
                # exact vacuum baseline for every in-bounds cell
                SF[0, 0] += weight
                if material_mask[ix, iy, iz]:
                    if material_slice_cache is not None:
                        SF += weight * material_slice_cache[ix, iy, iz]
                    else:
                        cell = self.array[ix, iy, iz]
                        deviation = cell.clone()
                        deviation[0, 0, 0] = 0.0
                        SF += weight * self._sample_complex_volume_trilinear(
                            deviation,
                            self.k_slice_coords,
                        ).to(SF.dtype)

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
        volume: torch.Tensor | None = None,
        k_slice_coords: torch.Tensor | None = None,
    ):
        """
        Returns exit wave propagated through volume at specified tilt for precalculated ray coordinates

        Parameters
        ----------
        Psi0: torch.Tensor
            Probe aperture
        ray_coords: torch.Tensor
            Tensor of 3D positions in the real space volume (xyz)
        phase_only: bool
            Enforce unitary phase-grating transmission per slice
        volume: torch.Tensor | None
            Optional learnable 6D structure-factor tensor. When given, every
            in-bounds trilinear neighbor is sampled from `volume` (no
            material-mask short-circuit and no slice cache), keeping the
            computation differentiable end-to-end for reconstruction.
            When None, uses `self.array` with the vacuum short-circuit.
        k_slice_coords: torch.Tensor | None
            Precomputed 2D slice coordinates for this tilt. Falls back to
            `self.k_slice_coords` (set by a prior `get_ray_coords` call).

        Returns
        -------
        Psi: torch.Tensor
            Exit wave from Psi0 propagating through the volume at specified position and tilt
        """
        device = torch.device(self.device)
        Psi0 = torch.as_tensor(Psi0, device=device)

        if k_slice_coords is None:
            k_slice_coords = self.k_slice_coords

        r_all = ray_coords

        # multislice propagation through volume
        learnable = volume is not None
        if not learnable:
            material_mask = self._get_material_mask().to(device)
            material_slice_cache = getattr(self, "_material_slice_cache", None)
        Nx, Ny, Nz = self.real_shape
        Psi = Psi0.clone()

        for ind, r in enumerate(r_all):
            # trilinear interpolation of structure factor slice from 6D volume for
            # this ray position. Vacuum cells (SF = delta at the origin) only
            # contribute `weight` to the central pixel, so we short-circuit
            # them instead of sampling — except on the learnable path, where
            # every in-bounds cell is sampled so gradients reach all voxels.
            # Every cell is decomposed as `baseline * delta + deviation`: the
            # vacuum baseline (SF[0, 0, 0]) is placed analytically at the
            # central pixel — never interpolated — so vacuum transmission is
            # exactly the identity, while only the Bragg deviation is sampled
            # with trilinear interpolation. Interpolating the full cell would
            # leak the baseline delta into neighboring detector pixels for
            # tilted slices (asymmetrically, due to the grid_sample seam).
            weight_sum = 0.0
            if learnable:
                SF = torch.zeros(Psi0.shape, dtype=volume.dtype, device=device)
                cells: list[torch.Tensor] = []
                wts: list[float] = []
                for (ix, iy, iz), weight in self._trilinear_real_weights(r):
                    if weight == 0.0:
                        continue
                    if not (0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz):
                        continue
                    weight_sum += weight
                    cells.append(volume[ix, iy, iz])
                    wts.append(weight)
                if cells:
                    # Batch all trilinear neighbors through one grid_sample
                    # call — identical math to sampling each cell separately.
                    stack = torch.stack(cells)                      # (K, N0, N1, N2)
                    baselines = stack[:, 0, 0, 0]                   # (K,)
                    delta3 = self._delta_volume(stack[0])
                    deviations = stack - baselines[:, None, None, None] * delta3[None]
                    w_t = torch.tensor(wts, dtype=torch.float64, device=device)
                    sampled = self._sample_complex_volume_trilinear_batch(
                        deviations,
                        k_slice_coords,
                    ).to(SF.dtype)                                  # (K, H, W)
                    SF = (w_t[:, None, None].to(SF.dtype) * sampled).sum(dim=0)
                    SF = SF + (w_t.to(baselines.dtype) * baselines).sum() * self._delta_slice(SF)
            else:
                SF = torch.zeros(Psi0.shape, dtype=self.array.dtype, device=device)
                for (ix, iy, iz), weight in self._trilinear_real_weights(r):
                    if weight == 0.0:
                        continue
                    if not (0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz):
                        continue
                    weight_sum += weight
                    # exact vacuum baseline for every in-bounds cell
                    SF[0, 0] += weight
                    if material_mask[ix, iy, iz]:
                        if material_slice_cache is not None:
                            SF += weight * material_slice_cache[ix, iy, iz]
                        else:
                            cell = self.array[ix, iy, iz]
                            deviation = cell.clone()
                            deviation[0, 0, 0] = 0.0
                            SF += weight * self._sample_complex_volume_trilinear(
                                deviation,
                                k_slice_coords,
                            ).to(SF.dtype)

            if weight_sum > 0.0:
                SF = SF / weight_sum
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
                Psi = Psi * self.prop

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

    def _scan_position_indices(
        self,
        scan_shape: tuple[int, int],
        scan_step: float | tuple[float, float],
        scan_origin: torch.Tensor | tuple | list | None,
        u_proj: torch.Tensor,
        v_proj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Index-space (x, y, z) probe positions for a raster scan.

        Shared by `simulate_4dstem` and `reconstruct_pix` so the forward
        simulation and the reconstruction agree exactly on where each probe
        lands. The scan grid is centered on `scan_origin` (defaults to the
        (x, y) center of the volume at z = 0), with rows along `v_proj`
        (slow) and columns along `u_proj` (fast).

        Returns
        -------
        positions: torch.Tensor
            (n_slow, n_fast, 3) float tensor of (x, y, z) index positions.
        scan_origin_xyz: torch.Tensor
            (3,) physical-space scan origin in storage (x, y, z) order,
            recorded in metadata by `simulate_4dstem`.
        """
        device = torch.device(self.device)
        u_proj = torch.asarray(u_proj, dtype=torch.float32, device=device)
        v_proj = torch.asarray(v_proj, dtype=torch.float32, device=device)

        scan_step_arr = torch.atleast_1d(
            torch.asarray(scan_step, dtype=torch.float32, device=device)
        )
        if scan_step_arr.shape == (1,):
            scan_step_arr = torch.tensor(
                [float(scan_step_arr[0])] * 2, dtype=torch.float32, device=device
            )
        if scan_step_arr.shape != (2,):
            raise ValueError(
                f"scan_step must be a scalar or shape (2,), got shape={scan_step_arr.shape}"
            )

        n_slow, n_fast = int(scan_shape[0]), int(scan_shape[1])
        if n_slow <= 0 or n_fast <= 0:
            raise ValueError(f"scan_shape must be positive, got {scan_shape}")

        sampling3 = torch.asarray(
            self.dataset.sampling[:3], dtype=torch.float32, device=device
        )
        real_shape = torch.asarray(
            self.array.shape[:3], dtype=torch.float32, device=device
        )
        # scan_origin in physical (x, y, z): defaults to the (x, y) center of
        # the volume at the beam entrance plane z = 0.
        if scan_origin is None:
            scan_origin_xyz = (real_shape[:2] - 1) / 2 * sampling3[:2]
            scan_origin_xyz = torch.cat(
                (scan_origin_xyz, torch.zeros(1, dtype=torch.float32, device=device))
            )
        else:
            scan_origin_xyz = torch.asarray(
                scan_origin, dtype=torch.float32, device=device
            )
            if scan_origin_xyz.shape == (2,):
                scan_origin_xyz = torch.cat(
                    (scan_origin_xyz, torch.zeros(1, dtype=torch.float32, device=device))
                )
            if scan_origin_xyz.shape != (3,):
                raise ValueError(
                    f"scan_origin must have shape (2,) or (3,), got {scan_origin_xyz.shape}"
                )

        slow_centered = (
            torch.arange(n_slow, dtype=torch.float32, device=device) - (n_slow - 1) / 2.0
        )
        fast_centered = (
            torch.arange(n_fast, dtype=torch.float32, device=device) - (n_fast - 1) / 2.0
        )
        positions = torch.empty((n_slow, n_fast, 3), dtype=torch.float32, device=device)
        for j in range(n_slow):
            for i in range(n_fast):
                offset_xyz = (
                    slow_centered[j] * scan_step_arr[0] * v_proj
                    + fast_centered[i] * scan_step_arr[1] * u_proj
                )
                positions[j, i] = (scan_origin_xyz + offset_xyz) / sampling3
        return positions, scan_origin_xyz

    def setup_dataloader(
            self,
            dataset: Dataset | DatasetModelType,
            batch_size: int = 1024,
            val_fraction: float = 0.0,
    ):
        pin_mem = self.device == 'cuda'
        generator = torch.Generator()
        generator.manual_seed(42)
        dataset_size = len(dataset)

        if val_fraction > 0.0:
            train_size = int((1-val_fraction)*dataset_size)
            train_dataset, val_dataset = random_split(dataset, [train_size, dataset_size-train_size], generator=generator)
        else:
            train_dataset = dataset
            val_dataset = None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size = batch_size,
            pin_memory=pin_mem,
            drop_last=False,
            )
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size = batch_size * 4, # less memory than training
                pin_memory=pin_mem,
                drop_last=False,
                )
        else:
            val_dataloader = None
        return train_dataloader, val_dataloader
        
    def _precompute_tilt_packs(
        self,
        rays_per_probe: list[torch.Tensor],
        targets: torch.Tensor,
        kcoords: torch.Tensor,
    ) -> list[dict]:
        """Precompute vectorized trilinear geometry for one tilt.

        Probes at a fixed tilt share the ray direction but can start at
        different z (the slow-scan axis tips out of plane at nonzero tilt),
        so step counts may differ by one. Probes are grouped by step count;
        each group packs the trilinear neighbor indices and weights for every
        (probe, step) into dense tensors so the multislice can run batched
        over the group.

        Weights are computed with the same float32-fraction-then-float64-
        product recipe as `_trilinear_real_weights`, so the fast path matches
        the per-sample path to the last ulp of each weight.
        """
        device = torch.device(self.device)
        Nx, Ny, Nz = (int(n) for n in self.real_shape)
        tgt_flat = targets.reshape(-1, *targets.shape[-2:])

        by_steps: dict[int, list[int]] = {}
        for i, r in enumerate(rays_per_probe):
            if len(r) == 0:
                continue
            by_steps.setdefault(len(r), []).append(i)

        packs: list[dict] = []
        for S, keep in sorted(by_steps.items()):
            rays = torch.stack([rays_per_probe[i].to(device) for i in keep])  # (P, S, 3) f32
            P = rays.shape[0]

            base32 = torch.floor(rays)                      # f32, matches slow path
            frac32 = rays - base32                          # f32 subtraction, as in slow path
            frac64 = frac32.to(torch.float64)
            base = base32.to(torch.int64)                   # (P, S, 3)

            # 8 trilinear corners in the same (dx, dy, dz) order as the slow path
            corner = torch.tensor(
                [[dx, dy, dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
                dtype=torch.int64, device=device,
            )                                               # (8, 3)
            nbr = base[:, :, None, :] + corner[None, None]  # (P, S, 8, 3)

            # per-axis weights: frac if corner==1 else 1-frac, in float64
            w_axis = torch.where(
                corner[None, None].bool(),
                frac64[:, :, None, :],
                1.0 - frac64[:, :, None, :],
            )                                               # (P, S, 8, 3)
            wval = (w_axis[..., 0] * w_axis[..., 1]) * w_axis[..., 2]  # (P, S, 8) f64

            in_bounds = (
                (nbr[..., 0] >= 0) & (nbr[..., 0] < Nx)
                & (nbr[..., 1] >= 0) & (nbr[..., 1] < Ny)
                & (nbr[..., 2] >= 0) & (nbr[..., 2] < Nz)
            )
            wval = wval * in_bounds.to(torch.float64)
            flat = nbr[..., 0] * (Ny * Nz) + nbr[..., 1] * Nz + nbr[..., 2]
            widx = torch.where(in_bounds, flat, torch.zeros_like(flat))   # (P, S, 8)
            wsum = wval.sum(dim=-1)                                        # (P, S)

            packs.append({
                "kc": kcoords.to(device),
                "targets": tgt_flat[keep].to(device),
                "widx": widx,
                "wval": wval,
                "wsum": wsum,
                "n_probes": P,
                "n_steps": S,
            })
        return packs

    def _forward_tilt_batch(
        self,
        Psi0: torch.Tensor,
        volume: torch.Tensor,
        pack: dict,
        phase_only: bool = True,
    ) -> torch.Tensor:
        """Vectorized multislice for all probes of one tilt.

        Numerically equivalent to calling `forward_prop_from_points(volume=...)`
        per probe: the same deviation slices, vacuum-baseline placement,
        per-step normalization, phase-only transmission, and propagator are
        applied — just batched over probes, with all cell slices sampled in a
        single grid_sample call.
        """
        device = torch.device(self.device)
        H, W = Psi0.shape
        P, S = pack["n_probes"], pack["n_steps"]
        widx, wval, wsum = pack["widx"], pack["wval"], pack["wsum"]

        vol_flat = volume.reshape(-1, *volume.shape[3:])            # (Nc, D0, D1, D2)
        baselines = vol_flat[:, 0, 0, 0]                            # (Nc,) complex
        delta3 = self._delta_volume(vol_flat[0])
        deviations = vol_flat - baselines[:, None, None, None] * delta3[None]

        # One batched grid_sample for every cell's deviation slice at this tilt.
        slices = self._sample_complex_volume_trilinear_batch(
            deviations, pack["kc"],
        ).to(volume.dtype)                                          # (Nc, H, W)
        slices_flat = slices.reshape(-1, H * W)

        delta2 = self._delta_slice(slices[0])
        num_pixels = H * W

        Psi = Psi0[None].expand(P, H, W)
        for s in range(S):
            idx = widx[:, s]                                        # (P, 8)
            w_c = wval[:, s].to(volume.dtype)                       # (P, 8) complex
            SF = (w_c[..., None] * slices_flat[idx]).sum(dim=1).reshape(P, H, W)
            SF = SF + (w_c * baselines[idx]).sum(dim=1)[:, None, None] * delta2[None]

            # Steps with no in-bounds neighbors are pure vacuum: substituting
            # the exact vacuum delta (baseline 1 at the central pixel) yields
            # t = 1 — identical to the per-sample path skipping transmission.
            ws = wsum[:, s]                                          # (P,) f64
            empty = (ws == 0).to(torch.float64)
            SF = SF + empty.to(volume.dtype)[:, None, None] * delta2[None]
            ws_eff = (ws + empty).to(volume.dtype)

            T_2d = num_pixels * SF / ws_eff[:, None, None]
            t_real = torch.fft.ifft2(T_2d)
            if phase_only:
                t_real = torch.exp(1j * torch.angle(t_real))
            Psi = torch.fft.fft2(torch.fft.ifft2(Psi) * t_real)
            if s < S - 1:
                Psi = Psi * self.prop

        return Psi

    def _apply_shrink(self, tau: float) -> None:
        """In-place complex soft-threshold of the non-origin k-voxels.

        Proximal step for an L1 penalty on the structure-factor deviation:
        each complex value keeps its phase and loses `tau` of magnitude
        (clamped at zero). The vacuum baseline pixel [..., 0, 0, 0] is
        never shrunk.
        """
        with torch.no_grad():
            sf = self.sf_learned
            mag = sf.abs()
            scale = torch.clamp(1.0 - tau / mag.clamp_min(1e-30), min=0.0)
            scale[..., 0, 0, 0] = 1.0
            sf.mul_(scale.to(sf.dtype))

    def reconstruct(
        self,
        measurements: Sequence[Dataset4dstem],
        tilt_x_deg: Sequence[float] | None = None,
        zxz_deg: Sequence[torch.Tensor | tuple | list] | None = None,
        scan_step: float | tuple[float, float] = 1.0,
        scan_shape: tuple[int, int] | None = None,
        scan_origin: torch.Tensor | tuple | list | None = None,
        probe_k_max: float | None = None,
        num_iters: int = 10,
        batch_size: int = 16,
        reset: bool = True,
        lr_init: float = 1e-3,
        optimizer_params: dict | None = None,
        phase_only: bool = True,
        init_noise: float = 0.0,
        shrink: float = 0.0,
        loss_type: str = "amplitude",
        obj_constraints: dict | ObjConstraintsType | None = None,
        show_metrics: bool = True,
        seed: int = 42,
        fast: bool = True,
    ):
        """Pixelated (voxel-grid) reconstruction of the 6D structure factor.

        Optimizes a complex 6D tensor `sf_learned` (initialized from
        `self.array`, typically vacuum) so that the differentiable multislice
        forward model reproduces the measured 4D-STEM diffraction patterns.

        Parameters
        ----------
        measurements: Sequence[Dataset4dstem]
            One 4D-STEM dataset per sample orientation. Geometry (ZXZ Euler
            angles, probe aperture) is read from each dataset's metadata when
            present (as written by `SimDiffractionTomography.simulate_4dstem`),
            and can be overridden with `tilt_x_deg` / `zxz_deg` / `probe_k_max`.
        tilt_x_deg: Sequence[float] | None
            Per-measurement X-only tilt (Z1 = Z3 = 0) override.
        zxz_deg: Sequence | None
            Per-measurement full ZXZ Euler angle override (degrees).
        scan_step, scan_shape, scan_origin:
            Scan geometry, identical conventions to `simulate_4dstem`.
            `scan_shape` defaults to each measurement's scan grid.
        num_iters, batch_size, lr_init, optimizer_params:
            AdamW optimization settings. Batching is over individual
            diffraction patterns (tilt, row, col).
        reset: bool
            Re-initialize `sf_learned` from `self.array` (default True).
        init_noise: float
            Standard deviation of seeded complex Gaussian noise added to the
            non-origin k-voxels of the initial volume. Breaking the exact
            vacuum symmetry gives the optimizer's moment estimates real
            statistics to calibrate on and speeds up early convergence.
        shrink: float
            Per-step complex soft-threshold applied to the non-origin
            k-voxels (proximal L1 / sparsity regularization). The true
            structure-factor deviation is sparse — a few Bragg peaks — so a
            small threshold suppresses the diffuse backprojection fog that
            otherwise accumulates along the tilt rotation axis. Scaled by
            the current learning rate.
        loss_type: str
            'amplitude' (default): MSE on sqrt-intensities — the standard
            ptychographic choice. Intensity loss has a vanishing first-order
            gradient at dark detector pixels (no reference wave to interfere
            with), which stalls recovery of Bragg peaks from a vacuum start;
            the amplitude loss keeps a finite gradient there. 'intensity':
            MSE on raw intensities.
        obj_constraints:
            Placeholder for further regularization (real-space smoothing) —
            parsed but not yet applied.

        Returns
        -------
        dict with 'reconstructed_sf' (complex 6D tensor on cpu) and 'losses'
        (per-iteration mean training loss).
        """
        device = torch.device(self.device)
        n_meas = len(measurements)
        if n_meas == 0:
            raise ValueError("measurements must contain at least one Dataset4dstem")

        if obj_constraints is not None and isinstance(obj_constraints, dict):
            obj_constraints = ObjConstraintParams.parse_dict(obj_constraints)
        self.obj_constraints = obj_constraints

        # --- resolve per-measurement geometry -------------------------------
        zxz_list: list[torch.Tensor] = []
        for m_idx, meas in enumerate(measurements):
            if zxz_deg is not None:
                zxz = torch.asarray(zxz_deg[m_idx], dtype=torch.float32)
            elif tilt_x_deg is not None:
                zxz = torch.tensor([0.0, float(tilt_x_deg[m_idx]), 0.0])
            elif meas.metadata.get("zxz_deg") is not None:
                zxz = torch.asarray(meas.metadata["zxz_deg"], dtype=torch.float32)
            else:
                zxz = torch.zeros(3)
            zxz_list.append(zxz)

        if probe_k_max is None:
            probe_k_max = float(
                measurements[0].metadata.get("probe_k_max", 0.10) or 0.10
            )

        dp_shape = (int(measurements[0].array.shape[2]), int(measurements[0].array.shape[3]))
        Psi0 = self.make_probe_aperture(
            probe_k_max=probe_k_max,
            dp_shape=dp_shape,
            normalize=True,
        ).to(device)
        self.make_prop(shape=dp_shape)

        # --- precompute rays, k-slice coords, targets per (tilt, row, col) ---
        sample_rays: list[torch.Tensor] = []
        sample_kcoords_idx: list[int] = []
        sample_targets: list[torch.Tensor] = []
        kcoords_per_meas: list[torch.Tensor] = []
        tilt_packs: list[dict] = []

        for m_idx, meas in enumerate(measurements):
            u, v, w = self.projection_axes(zxz_deg=zxz_list[m_idx])
            u = u.to(torch.float32)
            v = v.to(torch.float32)
            w = w.to(torch.float32)

            kcoords, _, _ = self._make_projected_k_slice_coords(
                shape_2d=dp_shape,
                sampling_2d=self.dataset.sampling[3:5],
                u_proj=u,
                v_proj=v,
                sampling_3d=self.dataset.sampling[3:],
                shape_3d=self.diffraction_shape,
            )
            kcoords_per_meas.append(kcoords)

            meas_scan_shape = (
                tuple(int(n) for n in scan_shape)
                if scan_shape is not None
                else (int(meas.array.shape[0]), int(meas.array.shape[1]))
            )
            positions, _ = self._scan_position_indices(
                scan_shape=meas_scan_shape,
                scan_step=scan_step,
                scan_origin=scan_origin,
                u_proj=u,
                v_proj=v,
            )

            targets = torch.as_tensor(meas.array, dtype=torch.float64, device=device)
            rays_this_meas: list[torch.Tensor] = []
            for j in range(meas_scan_shape[0]):
                for i in range(meas_scan_shape[1]):
                    rays = self._generate_slab_ray_coordinates(
                        position=positions[j, i],
                        direction=w,
                        shape=self.array.shape[:3],
                        sampling=self.dataset.sampling[:3],
                        prop_distance=self.prop_distance,
                    )
                    rays_this_meas.append(rays)
                    if len(rays) == 0:
                        continue
                    sample_rays.append(rays)
                    sample_kcoords_idx.append(m_idx)
                    sample_targets.append(targets[j, i])

            if fast:
                tilt_packs.extend(
                    self._precompute_tilt_packs(rays_this_meas, targets, kcoords)
                )

        n_samples = len(sample_rays)
        if n_samples == 0:
            raise ValueError("no valid probe positions found for reconstruction")

        use_fast = fast and sum(p["n_probes"] for p in tilt_packs) == n_samples

        # --- learnable volume ------------------------------------------------
        if reset or getattr(self, "sf_learned", None) is None:
            sf_init = self.array.detach().clone().to(device=device, dtype=torch.complex128)
            if init_noise > 0.0:
                gen = torch.Generator(device="cpu").manual_seed(seed)
                noise = init_noise * (
                    torch.randn(sf_init.shape, generator=gen, dtype=torch.float64)
                    + 1j * torch.randn(sf_init.shape, generator=gen, dtype=torch.float64)
                ).to(device=device, dtype=torch.complex128)
                noise[..., 0, 0, 0] = 0.0  # keep the vacuum baseline exact
                sf_init = sf_init + noise
            self.sf_learned = sf_init.requires_grad_(True)

        # Default weight_decay=0: AdamW's default decay would pull the vacuum
        # baseline (SF[..., 0, 0, 0] = 1) toward zero, which breaks the
        # physics. Shrinkage regularization belongs on the *deviation* from
        # vacuum, applied explicitly via obj_constraints (future work).
        # eps default: Adam's step is lr * m / (sqrt(v) + eps). The gradients
        # of this problem are ~1e-10, so sqrt(v) ~ 1e-10 << the usual
        # eps = 1e-8 — the default epsilon would dominate the denominator and
        # throttle every step by ~100x (the "slow first 20 iterations").
        opt_params = {"weight_decay": 0.0, "eps": 1e-30}
        opt_params.update(optimizer_params or {})
        self.optimizer = torch.optim.AdamW([self.sf_learned], lr=lr_init, **opt_params)
        loss_fxn = torch.nn.MSELoss()

        # --- optimization loop -----------------------------------------------
        generator = torch.Generator()
        generator.manual_seed(seed)
        losses: list[float] = []

        if use_fast:
            # Vectorized path: one forward/backward per tilt (all probes of a
            # tilt batched together; every cell's slice sampled in a single
            # grid_sample). Gradients accumulate across tilts and the
            # optimizer steps once ~batch_size diffraction patterns have
            # contributed, so the update granularity matches the slow path.
            n_packs = len(tilt_packs)
            probes_per_pack = [p["n_probes"] for p in tilt_packs]

            for it in tqdm.tqdm(range(num_iters), desc="reconstruct_pix", unit="iter"):
                order = torch.randperm(n_packs, generator=generator).tolist()
                total_loss = 0.0
                self.optimizer.zero_grad()

                # group shuffled tilts into optimizer steps of >= batch_size DPs
                groups: list[list[int]] = [[]]
                acc = 0
                for m in order:
                    groups[-1].append(m)
                    acc += probes_per_pack[m]
                    if acc >= batch_size:
                        groups.append([])
                        acc = 0
                if not groups[-1]:
                    groups.pop()

                for group in groups:
                    group_dps = sum(probes_per_pack[m] for m in group)
                    self.optimizer.zero_grad()
                    for m in group:
                        Psi = self._forward_tilt_batch(
                            Psi0, self.sf_learned, tilt_packs[m], phase_only=phase_only,
                        )
                        dp_pred = torch.abs(Psi) ** 2
                        # sum of per-DP mean-squared errors, scaled so the
                        # accumulated gradient equals the group-mean loss
                        if loss_type == "amplitude":
                            resid = torch.sqrt(dp_pred + 1e-30) - torch.sqrt(tilt_packs[m]["targets"])
                        else:
                            resid = dp_pred - tilt_packs[m]["targets"]
                        tilt_loss = (resid ** 2).mean(dim=(1, 2)).sum()
                        (tilt_loss / group_dps).backward()
                        total_loss += tilt_loss.item()
                    self.optimizer.step()
                    if shrink > 0.0:
                        self._apply_shrink(shrink * lr_init)

                mean_loss = total_loss / n_samples
                losses.append(mean_loss)
                if show_metrics:
                    print(f"Iter {it}: train loss = {mean_loss:.3e}")
        else:
            for it in tqdm.tqdm(range(num_iters), desc="reconstruct_pix", unit="iter"):
                perm = torch.randperm(n_samples, generator=generator)
                total_loss = 0.0
                for start in range(0, n_samples, batch_size):
                    batch_idx = perm[start : start + batch_size]
                    self.optimizer.zero_grad()
                    batch_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
                    for s in batch_idx.tolist():
                        Psi = self.forward_prop_from_points(
                            Psi0,
                            sample_rays[s],
                            phase_only=phase_only,
                            volume=self.sf_learned,
                            k_slice_coords=kcoords_per_meas[sample_kcoords_idx[s]],
                        )
                        dp_pred = torch.abs(Psi) ** 2
                        if loss_type == "amplitude":
                            batch_loss = batch_loss + loss_fxn(
                                torch.sqrt(dp_pred + 1e-30),
                                torch.sqrt(sample_targets[s]),
                            )
                        else:
                            batch_loss = batch_loss + loss_fxn(dp_pred, sample_targets[s])
                    batch_loss = batch_loss / len(batch_idx)
                    batch_loss.backward()
                    self.optimizer.step()
                    if shrink > 0.0:
                        self._apply_shrink(shrink * lr_init)
                    total_loss += batch_loss.item() * len(batch_idx)

                mean_loss = total_loss / n_samples
                losses.append(mean_loss)
                if show_metrics:
                    print(f"Iter {it}: train loss = {mean_loss:.3e}")

        self.recon_losses = losses
        return {
            "reconstructed_sf": self.sf_learned.detach().cpu(),
            "losses": losses,
        }

    def reconstruct_pix(self, *args, **kwargs):
        """Deprecated alias for :meth:`reconstruct`."""
        return self.reconstruct(*args, **kwargs)
