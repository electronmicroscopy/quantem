from itertools import permutations, product
from typing import Self

from numpy.typing import NDArray
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F

from quantem.core.datastructures.dataset import Dataset
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.dataset6d import Dataset6d
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import electron_wavelength_angstrom, tqdmnd
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
        # device: str | int = "cpu",
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
        volume: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        # separates volume into real and imaginary for coordinate mapping
        # pads volume for trilinear interpolation in grid_sample
        vol = volume.to(torch.complex64)
        real = vol.real[None, None, ...]
        imag = vol.imag[None, None, ...]

        # puts grid in order (x,y,z) mapped between values (-1, 1) for grid_sample
        Nz, Ny, Nx = volume.shape
        z = coords[0]
        y = coords[1]
        x = coords[2]

        gx = 2.0 * (x / (Nx-1.0))-1.0
        gy = 2.0 * (y / (Ny-1.0))-1.0
        gz = 2.0 * (z / (Nz-1.0))-1.0

        grid = torch.stack((gx,gy,gz), dim = -1)[None, None, ...]
        map_real = F.grid_sample(real, grid, mode='bilinear', padding_mode='zeros',align_corners=True)
        map_imag = F.grid_sample(imag, grid, mode='bilinear', padding_mode='zeros',align_corners=True)

        return map_real + 1j * map_imag

    #     volume: torch.Tensor,
    #     coords: torch.Tensor,
    #     mode: str = "grid-wrap",
    #     device: str | int = "cpu",
    # ) -> torch.Tensor:
    #     """Sample a complex 3D volume with trilinear interpolation.
        
    #     Defaults to `mode='grid-wrap'` for FFT-periodic data: a length-N array
    #     repeats with period N (samples 0 and N are identified). SciPy's older
    #     `mode='wrap'` uses period N-1 instead, which leaks the central peak
    #     asymmetrically when sampling tilted slices through the SF volume.
    #     """
    #     volume_np = volume.to(torch.complex128).detach().cpu().numpy()
    #     coords_np = coords.detach().cpu().numpy()
        
    #     # for i in range(coords_np.shape[0]):
    #     #     coords_np[i] = np.remainder(coords_np[i], volume_np.shape[i])

    #     real_mapped = ndi.map_coordinates(
    #         volume_np.real,
    #         coords_np,
    #         order=1,
    #         mode=mode,
    #     )

    #     imag_mapped = ndi.map_coordinates(
    #         volume_np.imag,
    #         coords_np,
    #         order=1,
    #         mode=mode
    #     )
    #     return (torch.from_numpy(real_mapped) + 1j * torch.from_numpy(imag_mapped)).to(device)

    @classmethod
    def _rotate_complex_volume_zxz(
        cls,
        volume: torch.Tensor,
        sampling: tuple[float, float, float] | list[float] | torch.Tensor,
        zxz: torch.Tensor,
    ) -> torch.Tensor:
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

        return cls._sample_complex_volume_trilinear(volume, coords)

    @staticmethod
    def _make_vacuum_sf(
        diffraction_shape: tuple[int, int, int],
        dtype=torch.complex128,
    ) -> torch.Tensor:
        """Structure factor of a vacuum cell — identity transmission in k-space."""
        sf = torch.zeros(diffraction_shape, dtype=dtype)
        sf[0, 0, 0] = 1.0
        return sf
    
    @classmethod
    def _make_au_structure_factor(
        cls,
        diffraction_shape: tuple[int, int, int],
        diffraction_sampling: tuple[float, float, float] | torch.Tensor,
        a_Au: float = 4.08,
        hkl_amplitudes: torch.Tensor | None = None,
        phase_scale: float = 0.10,
        zxz_deg: torch.Tensor | tuple | list | None = None,
        dtype=torch.complex128,
    ) -> torch.Tensor:
        """Approximate gold structure factor with phase-grating Bragg amplitudes.

        The 6D SF is stored so that the transmission function recovered via
        `t = ifft2(num_pixels * SF_slice)` approximates `exp(i * phi)`, with
        `phi` a small real "projected potential". The central value
        `SF[0, 0, 0] = 1` is the vacuum baseline, and Bragg peaks carry small
        purely-imaginary amplitudes (the linearized expansion of `exp(i*phi)`).
        `phase_scale` controls the per-peak phase amplitude; smaller values
        keep `|t|` closer to 1 over many slices.

        Per-particle rotation is applied by rotating each Bragg (h, k, l)
        vector before placement (not by resampling the SF volume), so two
        particles with different ZXZ Euler angles end up with identical total
        scattering energy.
        """

        sf = cls._make_vacuum_sf(diffraction_shape, dtype=dtype)
        if hkl_amplitudes is None:
            hkl_amplitudes = torch.tensor(
                (
                    (1, 1, 1, 0.10),
                    (2, 0, 0, 0.06),
                    (2, 2, 0, 0.03),
                    (3, 1, 1, 0.05),
                    (2, 2, 2, 0.04),
                )
            )
        dk = torch.tensor(diffraction_sampling, dtype=torch.float32)
        # Phase-grating convention: amplitudes are purely imaginary so that
        # `t ≈ 1 + i*phi` linearizes a unitary `exp(i*phi)` transmission.
        amp_scale = 1j * phase_scale

        rot = (
            Rotation.from_euler("zxz", torch.tensor(zxz_deg, dtype=torch.float32), degrees=True)
            if zxz_deg is not None
            else None
        )

        for row in hkl_amplitudes:
            hkl = torch.tensor(row[:3], dtype=torch.int32)
            amp = amp_scale * row[3].to(torch.float32)
            # symmetry equivalent vectors --> sorting for unique combinations/permutations
            vec_set = sorted(
                {
                    tuple(torch.multiply(torch.tensor(sign), torch.tensor(perm)))
                    for perm in set(permutations(hkl.tolist()))
                    for sign in product((-1, 1), repeat=3)
                }
            )
            for vec in vec_set:
                # peak position
                k_peak = torch.tensor(vec, dtype=torch.float32) / a_Au
                if rot is not None:
                    k_peak = rot.apply(k_peak)

                # trilinear interpolation
                grid = k_peak / dk
                base = torch.floor(grid).to(torch.int32)
                frac = grid - base
                for dx in range(2):
                    wx = frac[0] if dx else 1.0 - frac[0]
                    ix = (base[0] + dx) % diffraction_shape[0]
                    for dy in range(2):
                        wy = frac[1] if dy else 1.0 - frac[1]
                        iy = (base[1] + dy) % diffraction_shape[1]
                        for dz in range(2):
                            wz = frac[2] if dz else 1.0 - frac[2]
                            iz = (base[2] + dz) % diffraction_shape[2]
                            sf[ix, iy, iz] += amp * wx * wy * wz
        return sf
    
    @classmethod
    def from_test(
        cls,
        name = 'test_data',
        origin : torch.Tensor | tuple | list | float | int | None = None,
        sampling : tuple[float, float, float, float, float, float] = (10,10,10,0.05,0.05,0.05),
        units = ('A','A','A','A^-1','A^-1','A^-1'),
        signal_units: str = "SF",
        energy: float | None = None,
        wavelength: float | None = None,
        real_shape: tuple[int, int, int] = (20, 20, 10),
        diffraction_shape: tuple[int, int, int] = (41, 41, 41),
        particle_centers: torch.Tensor | list | tuple = (
            (5, 5, 3),
            (15, 15, 3),
            (5, 15, 7),
            (15, 5, 7),
        ),
        particle_radius: float = 3.0,
        particle_zxz_deg: torch.Tensor | list | tuple = (
            (0.0, 0.0, 0.0),
            (45.0, 54.7, 15.0),
            (0.0, 45.0, -10.0),
            (12.0, 23.0, 54.0),
            ),
    ) -> Self:
        """
        Create test data for development: four hard-sphere Au nanoparticles in a vacuum slab.

        Each particle has the same approximate Au structure factor rotated by a
        distinct set of ZXZ Euler angles. Cells outside the particles are vacuum
        (`SF[0, 0, 0] = 1`, the identity transmission in k-space).
        """

        diff_sampling = torch.tensor(sampling[3:], dtype=torch.float32)

        # 6D array initialized to vacuum (SF[..., 0, 0, 0] = 1 everywhere).
        tensor = torch.zeros((*real_shape, *diffraction_shape), dtype=torch.complex128)
        tensor[..., 0, 0, 0] = 1.0

        # Build each particle's SF by rotating its (h, k, l) vectors directly,
        # so all particles end up with identical total scattering energy.
        ix = torch.arange(real_shape[0])[:, None, None]
        iy = torch.arange(real_shape[1])[None, :, None]
        iz = torch.arange(real_shape[2])[None, None, :]
        for center, zxz_deg in zip(particle_centers, particle_zxz_deg):
            center = torch.tensor(center, dtype=torch.float32)
            r2 = (ix - center[0]) ** 2 + (iy - center[1]) ** 2 + (iz - center[2]) ** 2
            mask = r2 < particle_radius**2
            if not torch.any(mask):
                continue

            #put structure factor as values mask values for each rotation
            sf_particle = cls._make_au_structure_factor(
                diffraction_shape=diffraction_shape,
                diffraction_sampling=diff_sampling,
                zxz_deg=zxz_deg,
            )
            tensor[mask] = sf_particle
        # Output
        dataset = Dataset6d.from_array(
            array=tensor.detach().cpu().numpy(),
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device(device)
        if prop_distance is None:
            prop_distance = self.dataset.sampling[2]
        self.prop_distance = prop_distance
        if shape is None:
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
        """Circular band-limit mask in k-space with a soft cosine roll-off.

        `fraction` is the cutoff relative to the Nyquist limit (2/3 by default,
        matching the standard multislice anti-aliasing rule). `softness` is the
        roll-off width, also relative to Nyquist; setting it to 0 gives a hard
        cutoff. The mask is 1 inside the cutoff, smoothly drops to 0 outside.
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
        dp_shape: tuple[int, int] | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Build a top-hat aperture probe in reciprocal space.

        With `normalize=True` (default), the returned probe satisfies
        `sum(|Psi|**2) = 1`. Vacuum propagation preserves this norm.
        """
        if dp_shape is None:
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
        shape = tuple(int(n) for n in shape)
        dk = torch.tensor(sampling, dtype=float)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ku, kv = cls._make_planar_reciprocal_grids(shape_2d, sampling_2d)
        ku2, kv2 = torch.broadcast_tensors(ku, kv)
        ku2 = ku2.to(device)
        kv2 = kv2.to(device)
        u_proj = torch.asarray(u_proj, dtype=torch.float32, device=device)
        v_proj = torch.asarray(v_proj, dtype=torch.float32, device=device)
        k_xyz = ku2[..., None] * u_proj[None, None, :] + kv2[..., None] * v_proj[None, None, :]

        dk = torch.asarray(sampling_3d, dtype=torch.float32)
        shape_k = torch.asarray(shape_3d, dtype=torch.int32)
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
        """Generate (equally spaced) index-space samples along a ray spanning the slab z-extent.

        The slab is treated as a thin film of finite z-thickness whose (x, y)
        footprint is the material region. The ray's t-range is fixed by the
        z-axis only — `t` covers `z = 0 .. shape[2] - 1` — so tilted rays that
        leave the (x, y) footprint still produce slices. Callers are
        expected to treat those out-of-footprint slices as vacuum."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # t_min = -torch.inf
        # t_max = torch.inf
        # for axis in range(3):
        if torch.isclose(dr[2], torch.tensor(0.0)):
            # beam parallel to slab - cannot traverse z
            return torch.empty((0, 3), dtype=torch.float32)

        t0 = (0.0 - r0[2]) / dr[2]
        t1 = ((shape[2] - 1) - r0[2]) / dr[2]
        t_min, t_max = (t0, t1) if t0 <= t1 else (t1, t0)

        # if t_max < t_min:
        #     return torch.empty((0, 3), dtype=float)

        # Sample the full line segment through the slab, including both negative
        # and positive steps from the itorchut position when they remain in bounds.
        eps = 1e-9
        n_min = int(torch.ceil(t_min - eps))
        n_max = int(torch.floor(t_max + eps))
        if n_max < n_min:
            return torch.empty((0, 3), dtype=torch.float32)

        steps = torch.arange(n_min, n_max + 1, dtype=torch.float32, device=device)[:, None]
        return r0[None, :] + steps * dr[None, :]

    @staticmethod
    def _trilinear_real_weights(position: torch.Tensor) -> list[tuple[tuple[int, int, int], float]]:
        """Return real-space trilinear neighbors and weights for one sample position."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base = torch.asarray(torch.floor(position), dtype = torch.int32, device = device)
        pos = torch.asarray(position, device = device)
        frac = pos - base
        weights: torch.Tensor[tuple[tuple[int, int, int], float]] = []
        for dx in range(2):
            wx = frac[0] if dx else 1.0 - frac[0]
            for dy in range(2):
                wy = frac[1] if dy else 1.0 - frac[1]
                for dz in range(2):
                    wz = frac[2] if dz else 1.0 - frac[2]
                    weights.append(((int(base[0] + dx), int(base[1] + dy), int(base[2] + dz)), float(wx * wy * wz)))
                    # weights.append(((base[0] + dx, base[1] + dy, base[2] + dz), (wx * wy * wz).to(torch.float32)))
        return weights

    def forward_prop(
        self,
        Psi0,
        position,
        u_proj = (1,0,0),
        v_proj = (0,1,0),
        phase_only: bool = True,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            SF = torch.zeros(Psi0.shape, dtype=self.array.dtype, device = device)
            weight_sum = 0.0
            for (ix, iy, iz), weight in self._trilinear_real_weights(r):
                if weight == 0.0:
                    continue
                if not (0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz):
                    continue
                weight_sum += weight
                if material_mask[ix, iy, iz]:
                    if material_slice_cache is not None:
                        SF += weight * material_slice_cache[ix, iy, iz]
                    else:
                        SF += weight * self._sample_complex_volume_trilinear(
                            self.array[ix, iy, iz],
                            k_slice_coords,
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

    def simulate_4dstem(
        self,
        *,
        tilt_x_deg: float | None = 0.0,
        zxz_deg: torch.Tensor | tuple | list | None = None,
        detector_rotation_deg: float = 0.0,
        scan_step: float | tuple[float, float] = 1.0,
        scan_shape: tuple[int, int] = (11, 11),
        scan_origin: torch.Tensor | tuple | list | None = None,
        probe_k_max: float = 0.10,
        dp_shape: tuple[int, int] | None = None,
        phase_only: bool = True,
        progress: bool = True,
        progress_desc: str | None = None,
        progress_leave: bool = True,
        progress_bar=None,
        name: str = "Simulated 4D-STEM",
        signal_units: str = "intensity",
    ) -> Dataset4dstem:
        """Simulate a 4D-STEM acquisition by forward-propagating a probe at each scan position.

        The sample is rotated by the given ZXZ Euler angles (or, equivalently, by
        an X-only tilt when `zxz_deg` is omitted). The scan plane is the lab xy
        plane, and the beam is the lab z axis; both are expressed in the
        sample-fixed 6D coordinate system before forward propagation.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        zxz_arr = self._resolve_zxz_deg(tilt_x_deg, zxz_deg) #formats rotation angles
        u_proj, v_proj, w_proj = self.projection_axes(zxz_deg=zxz_arr) #applies rotation matrix to basis vectors
        u_proj = u_proj.to(device)
        v_proj = v_proj.to(device)
        w_proj = w_proj.to(device)


        #verifies scan step available for each direction in xy
        scan_step_arr = torch.atleast_1d(torch.asarray(scan_step, dtype=torch.float32, device = device))
        if scan_step_arr.shape == (1,):
            scan_step_arr = torch.tensor([float(scan_step_arr[0])] * 2)
        if scan_step_arr.shape != (2,):
            raise ValueError(
                f"scan_step must be a scalar or shape (2,), got shape={scan_step_arr.shape}"
            )

        n_slow, n_fast = int(scan_shape[0]), int(scan_shape[1])
        if n_slow <= 0 or n_fast <= 0:
            raise ValueError(f"scan_shape must be positive, got {scan_shape}")

        #real space sampling and shape
        sampling3 = torch.asarray(self.dataset.sampling[:3], dtype=torch.float32, device = device)
        real_shape = torch.asarray(self.array.shape[:3], dtype=torch.float32, device = device)
        if scan_origin is None:
            scan_origin_arr = (real_shape - 1) / 2.0 * sampling3
        else:
            scan_origin_arr = torch.asarray(scan_origin, dtype=torch.float32, device = device)
            if scan_origin_arr.shape != (3,):
                raise ValueError(
                    f"scan_origin must have shape (3,), got {scan_origin_arr.shape}"
                )
        #setting output diffraction shape
        diff_shape = self.diffraction_shape
        if dp_shape is None:
            dp_shape_out = (int(diff_shape[0]), int(diff_shape[1]))
        else:
            dp_shape_out = (int(dp_shape[0]), int(dp_shape[1]))

        # Probe aperture in reciprocal space, normalized to sum |Psi|^2 = 1.
        Psi0 = self.make_probe_aperture(
            probe_k_max=probe_k_max,
            dp_shape=dp_shape_out,
            normalize=True,
        ).to(device)

        # Initialize Fresnel propagator for the chosen DP shape.
        self.make_prop(shape=dp_shape_out)

        # Precompute the 2D SF slice for every material cell once per tilt:
        # (u_proj, v_proj) are fixed across all scan positions, so the slice
        # only depends on the cell, not the probe position.
        k_slice_coords, _, _ = self._make_projected_k_slice_coords(
            shape_2d=dp_shape_out,
            sampling_2d=self.dataset.sampling[3:5],
            u_proj=u_proj,
            v_proj=v_proj,
            sampling_3d=self.dataset.sampling[3:],
            shape_3d=self.diffraction_shape,
        )
        material_mask = self._get_material_mask()
        Nx, Ny, Nz = self.real_shape
        slice_cache = torch.zeros(
            (Nx, Ny, Nz, dp_shape_out[0], dp_shape_out[1]),
            dtype=self.array.dtype,
            device = device,
        )
        material_idx = torch.argwhere(material_mask)
        for ix, iy, iz in material_idx:
            slice_cache[ix, iy, iz] = self._sample_complex_volume_trilinear(
                self.array[ix, iy, iz],
                k_slice_coords,
            )
        # Forward_prop will pick up this attribute through getattr.
        self._material_slice_cache = slice_cache

        #centers coordinates in fast and slow scan directions --> 0 at center with pos/neg values
        out = torch.empty((n_slow, n_fast, dp_shape_out[0], dp_shape_out[1]), dtype=torch.float64)
        slow_centered = torch.arange(n_slow, dtype=torch.float32) - (n_slow - 1) / 2.0
        fast_centered = torch.arange(n_fast, dtype=torch.float32) - (n_fast - 1) / 2.0

        rotate_detector = not torch.isclose(torch.tensor(detector_rotation_deg), torch.tensor(0.0))
        if progress_desc is None:
            zxz_arr = self._resolve_zxz_deg(tilt_x_deg, zxz_deg)
            progress_desc = (
                f"4D-STEM ZXZ=({zxz_arr[0]:+.1f}, {zxz_arr[1]:+.1f}, {zxz_arr[2]:+.1f}) deg"
            )
        n_total = n_slow * n_fast
        if progress_bar is not None:
            # Caller supplied a persistent tqdm bar — reset and relabel it.
            progress_bar.reset(total=n_total)
            progress_bar.set_description(progress_desc)
            iterator = product(range(n_slow), range(n_fast))
            external_bar = progress_bar
        elif progress:
            iterator = tqdmnd(
                range(n_slow),
                range(n_fast),
                desc=progress_desc,
                unit="probe",
                disable=False,
                leave=progress_leave,
            )
            external_bar = None
        else:
            iterator = product(range(n_slow), range(n_fast))
            external_bar = None
        
        for j, i in iterator:
            sv = slow_centered[j]
            su = fast_centered[i]
            offset_phys = sv * scan_step_arr[0] * v_proj + su * scan_step_arr[1] * u_proj
            position_index = (scan_origin_arr + offset_phys) / sampling3
            Psi = self.forward_prop(
                Psi0,
                position_index,
                u_proj=u_proj,
                v_proj=v_proj,
                phase_only=phase_only,
            )
            # need to fix the rotation below
            dp = torch.abs(Psi) ** 2
            if rotate_detector:
                dp_shifted = torch.fft.fftshift(dp)
                dp_rotated = torch.asarray(
                ndi.rotate(
                    dp_shifted,
                    detector_rotation_deg,
                    reshape=False,
                    order=1,
                    mode="constant",
                    cval=0.0,
                )
                )
                dp = torch.fft.ifftshift(dp_rotated)
            out[j, i] = dp
            if external_bar is not None:
                external_bar.update(1)
        if external_bar is not None:
            external_bar.refresh()

        # Drop the per-tilt slice cache so a subsequent forward_prop call at a
        # different orientation does not reuse stale data.
        self._material_slice_cache = None

        sampling4 = (
            float(scan_step_arr[0]),
            float(scan_step_arr[1]),
            float(self.dataset.sampling[3]),
            float(self.dataset.sampling[4]),
        )
        units4 = ("A", "A", "A^-1", "A^-1")
        dp = Dataset4dstem.from_array(
            array=out,
            name=name,
            origin=(0.0, 0.0, 0.0, 0.0),
            sampling=sampling4,
            units=units4,
            signal_units=signal_units,
        )
        dp.metadata.update(
            {
                "zxz_deg": tuple(float(v) for v in zxz_arr),
                "tilt_x_deg": (
                    float(zxz_arr[1])
                    if torch.isclose(zxz_arr[0], torch.tensor(0.0)) and torch.isclose(zxz_arr[2], torch.tensor(0.0))
                    else None
                ),
                "detector_rotation_deg": float(detector_rotation_deg),
                "scan_origin_sample": tuple(float(v) for v in scan_origin_arr),
                "u_proj_sample": tuple(float(v) for v in u_proj),
                "v_proj_sample": tuple(float(v) for v in v_proj),
                "w_proj_sample": tuple(float(v) for v in w_proj),
                "probe_k_max": float(probe_k_max),
                "r_to_q_rotation_cw_deg": float(detector_rotation_deg),
            }
        )
        return dp