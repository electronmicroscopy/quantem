from itertools import permutations, product
from typing import Self

import numpy as np
from numpy.typing import NDArray
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation

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
        mode: str = "grid-wrap",
    ) -> NDArray:
        """Sample a complex 3D volume with trilinear interpolation.

        Defaults to `mode='grid-wrap'` for FFT-periodic data: a length-N array
        repeats with period N (samples 0 and N are identified). SciPy's older
        `mode='wrap'` uses period N-1 instead, which leaks the central peak
        asymmetrically when sampling tilted slices through the SF volume.
        """
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

        return cls._sample_complex_volume_trilinear(volume, coords, mode="grid-wrap")

    @staticmethod
    def _make_vacuum_sf(
        diffraction_shape: tuple[int, int, int],
        dtype=np.complex128,
    ) -> NDArray:
        """Structure factor of a vacuum cell — identity transmission in k-space."""
        sf = np.zeros(diffraction_shape, dtype=dtype)
        sf[0, 0, 0] = 1.0
        return sf

    @classmethod
    def _make_au_structure_factor(
        cls,
        diffraction_shape: tuple[int, int, int],
        diffraction_sampling: tuple[float, float, float] | NDArray,
        a_Au: float = 4.08,
        hkl_amplitudes: NDArray | None = None,
        phase_scale: float = 0.10,
        zxz_deg: NDArray | tuple | list | None = None,
        dtype=np.complex128,
    ) -> NDArray:
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
            hkl_amplitudes = np.array(
                (
                    (1, 1, 1, 0.10),
                    (2, 0, 0, 0.06),
                    (2, 2, 0, 0.03),
                    (3, 1, 1, 0.05),
                    (2, 2, 2, 0.04),
                )
            )
        dk = np.asarray(diffraction_sampling, dtype=float)
        # Phase-grating convention: amplitudes are purely imaginary so that
        # `t ≈ 1 + i*phi` linearizes a unitary `exp(i*phi)` transmission.
        amp_scale = 1j * float(phase_scale)

        rot = (
            Rotation.from_euler("zxz", np.asarray(zxz_deg, dtype=float), degrees=True)
            if zxz_deg is not None
            else None
        )

        for row in hkl_amplitudes:
            hkl = np.asarray(row[:3], dtype=int)
            amp = amp_scale * float(row[3])
            vec_set = sorted(
                {
                    tuple(np.multiply(sign, perm))
                    for perm in set(permutations(hkl.tolist()))
                    for sign in product((-1, 1), repeat=3)
                }
            )
            for vec in vec_set:
                k_peak = np.asarray(vec, dtype=float) / a_Au
                if rot is not None:
                    k_peak = rot.apply(k_peak)
                grid = k_peak / dk
                base = np.floor(grid).astype(int)
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
        name: str = "test_data",
        origin: NDArray | tuple | list | float | int | None = None,
        sampling: tuple[float, float, float, float, float, float] = (10, 10, 10, 0.05, 0.05, 0.05),
        units: tuple[str, ...] = ("A", "A", "A", "A^-1", "A^-1", "A^-1"),
        signal_units: str = "SF",
        energy: float | None = None,
        wavelength: float | None = None,
        real_shape: tuple[int, int, int] = (20, 20, 10),
        diffraction_shape: tuple[int, int, int] = (41, 41, 41),
        particle_centers: NDArray | list | tuple = (
            (5, 5, 3),
            (15, 15, 3),
            (5, 15, 7),
            (15, 5, 7),
        ),
        particle_radius: float = 3.0,
        particle_zxz_deg: NDArray | list | tuple = (
            (0.0, 0.0, 0.0),
            (45.0, 54.7, 15.0),
            (0.0, 45.0, -10.0),
            (12.0, 23.0, 54.0),
        ),
    ) -> Self:
        """Create a test dataset: four hard-sphere Au nanoparticles in a vacuum slab.

        Each particle has the same approximate Au structure factor rotated by a
        distinct set of ZXZ Euler angles. Cells outside the particles are vacuum
        (`SF[0, 0, 0] = 1`, the identity transmission in k-space).
        """
        if len(particle_centers) != len(particle_zxz_deg):
            raise ValueError(
                "particle_centers and particle_zxz_deg must have the same length, "
                f"got {len(particle_centers)} and {len(particle_zxz_deg)}"
            )

        diff_sampling = np.asarray(sampling[3:], dtype=float)

        # 6D array initialized to vacuum (SF[..., 0, 0, 0] = 1 everywhere).
        array = np.zeros((*real_shape, *diffraction_shape), dtype=np.complex128)
        array[..., 0, 0, 0] = 1.0

        # Build each particle's SF by rotating its (h, k, l) vectors directly,
        # so all particles end up with identical total scattering energy.
        ix = np.arange(real_shape[0])[:, None, None]
        iy = np.arange(real_shape[1])[None, :, None]
        iz = np.arange(real_shape[2])[None, None, :]
        for center, zxz_deg in zip(particle_centers, particle_zxz_deg):
            center = np.asarray(center, dtype=float)
            r2 = (ix - center[0]) ** 2 + (iy - center[1]) ** 2 + (iz - center[2]) ** 2
            mask = r2 < particle_radius**2
            if not np.any(mask):
                continue

            sf_particle = cls._make_au_structure_factor(
                diffraction_shape=diffraction_shape,
                diffraction_sampling=diff_sampling,
                zxz_deg=zxz_deg,
            )
            array[mask] = sf_particle

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
        # Invalidate the material-vs-vacuum cell cache used by forward_prop.
        self._material_mask_cache: NDArray | None = None

    def _get_material_mask(self, tol: float = 1e-9) -> NDArray:
        """Bool array marking which (ix, iy, iz) real-space cells contain material.

        A vacuum cell stores `SF[0, 0, 0] = 1` and zeros elsewhere, so the
        absolute-sum over the diffraction axes equals 1. Material cells exceed
        this baseline. The result is cached and invalidated when `dataset` is
        reassigned.
        """
        if getattr(self, "_material_mask_cache", None) is None:
            abs_sum = np.abs(self.array).sum(axis=(3, 4, 5))
            self._material_mask_cache = abs_sum > 1.0 + tol
        return self._material_mask_cache

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
        prop_distance=None,
        shape: tuple[int, int] | list[int] | NDArray | None = None,
        antialias_fraction: float = 0.9,
        antialias_softness: float = 0.05,
    ):
        """Build the Fresnel propagator with anti-aliasing folded in.

        The anti-aliasing band-limit (default cutoff = 0.9 * k_Nyquist) is
        applied as part of the propagator only, so it is multiplied into the
        wave each propagation step and never separately. There is no
        per-slice mask on the transmission step.
        """
        if prop_distance is None:
            prop_distance = self.dataset.sampling[2]
        self.prop_distance = prop_distance
        if shape is None:
            shape = self.diffraction_shape[:2]
        ku, kv = self._make_planar_reciprocal_grids(shape, self.dataset.sampling[3:5])
        bare_prop = np.exp(-1j * np.pi * self.wavelength * (ku**2 + kv**2) * prop_distance)
        self.antialias_mask = self._make_antialias_mask(
            shape=shape,
            sampling=self.dataset.sampling[3:5],
            fraction=antialias_fraction,
            softness=antialias_softness,
        )
        self.prop = bare_prop * self.antialias_mask

    @staticmethod
    def _make_antialias_mask(
        shape: tuple[int, int] | list[int] | NDArray,
        sampling: tuple[float, float] | list[float] | NDArray,
        fraction: float = 2.0 / 3.0,
        softness: float = 0.05,
    ) -> NDArray:
        """Circular band-limit mask in k-space with a soft cosine roll-off.

        `fraction` is the cutoff relative to the Nyquist limit (2/3 by default,
        matching the standard multislice anti-aliasing rule). `softness` is the
        roll-off width, also relative to Nyquist; setting it to 0 gives a hard
        cutoff. The mask is 1 inside the cutoff, smoothly drops to 0 outside.
        """
        ku, kv = DiffractionTomography._make_planar_reciprocal_grids(shape, sampling)
        k_radial = np.sqrt(ku**2 + kv**2)
        dk = np.asarray(sampling, dtype=float)
        n_pix = np.asarray(shape, dtype=int)
        k_nyquist = float(np.min(n_pix * dk / 2.0))
        cutoff = float(fraction) * k_nyquist

        if softness <= 0.0:
            return (k_radial <= cutoff).astype(np.float64)

        width = max(float(softness) * k_nyquist, np.finfo(float).eps)
        mask = 0.5 * (1.0 - np.cos(np.pi * np.clip((cutoff + width - k_radial) / width, 0.0, 1.0)))
        return mask

    def make_probe_aperture(
        self,
        probe_k_max: float,
        dp_shape: tuple[int, int] | None = None,
        normalize: bool = True,
    ) -> NDArray:
        """Build a top-hat aperture probe in reciprocal space.

        With `normalize=True` (default), the returned probe satisfies
        `sum(|Psi|**2) = 1`. Vacuum propagation preserves this norm.
        """
        if dp_shape is None:
            dp_shape = self.diffraction_shape[:2]
        dp_shape = (int(dp_shape[0]), int(dp_shape[1]))

        ku, kv = self._make_planar_reciprocal_grids(dp_shape, self.dataset.sampling[3:5])
        k_radial = np.sqrt(ku**2 + kv**2)
        dk = float(self.dataset.sampling[3])
        aper = np.clip((probe_k_max - k_radial) / dk + 0.5, 0.0, 1.0)
        Psi0 = aper.astype(np.complex128)
        if normalize:
            norm = float(np.sqrt(np.sum(np.abs(Psi0) ** 2)))
            if norm > 0.0:
                Psi0 = Psi0 / norm
        return Psi0

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
        """Generate index-space samples along a ray spanning the slab z-extent.

        The slab is treated as a thin film of finite z-thickness whose (x, y)
        footprint is the material region. The ray's t-range is fixed by the
        z-axis only — `t` covers `z = 0 .. shape[2] - 1` — so tilted rays that
        leave the (x, y) footprint still produce slices. Callers are
        expected to treat those out-of-footprint slices as vacuum.
        """
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

        if np.isclose(dr[2], 0.0):
            # Beam parallel to the slab — cannot traverse z.
            return np.empty((0, 3), dtype=float)

        t0 = (0.0 - r0[2]) / dr[2]
        t1 = ((shape[2] - 1) - r0[2]) / dr[2]
        t_min, t_max = (t0, t1) if t0 <= t1 else (t1, t0)

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
        u_proj=(1, 0, 0),
        v_proj=(0, 1, 0),
        phase_only: bool = True,
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
        material_mask = self._get_material_mask()
        material_slice_cache = getattr(self, "_material_slice_cache", None)
        Nx, Ny, Nz = self.real_shape
        Psi = Psi0.copy()
        for ind, r in enumerate(r_all):
            # Trilinear interpolation of the 2D structure-factor slice for
            # this ray position. Vacuum cells (SF = delta at the origin) only
            # contribute `weight` to the central pixel, so we short-circuit
            # them instead of calling `ndi.map_coordinates`. Material slices
            # may be precomputed once per `simulate_4dstem` tilt and looked up
            # from `self._material_slice_cache`.
            SF = np.zeros(Psi0.shape, dtype=self.array.dtype)
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
                            mode="grid-wrap",
                        )
                else:
                    SF[0, 0] += weight

            if weight_sum > 0.0:
                SF /= weight_sum
                # Convention: the 2D transmission function `T_2d` satisfies
                # `T_2d[0, 0] = num_pixels` for vacuum, which corresponds to
                # `t_2d = 1` in real space. The 6D SF is stored normalized
                # (`SF[..., 0, 0, 0] = 1` for vacuum), so we scale by
                # `num_pixels` here to recover that convention.
                num_pixels = Psi.size
                T_2d = num_pixels * SF
                t_real = np.fft.ifft2(T_2d)
                if phase_only:
                    # Enforce a unitary phase-grating transmission: `|t| = 1`
                    # everywhere, so probe norm is preserved per slice.
                    # Vacuum still falls out as identity (angle(1) = 0).
                    t_real = np.exp(1j * np.angle(t_real))
                Psi = np.fft.fft2(np.fft.ifft2(Psi) * t_real)
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
        zxz_deg: NDArray | tuple | list | None,
    ) -> NDArray:
        """Normalize tilt/zxz inputs to a length-3 ZXZ vector in degrees."""
        if zxz_deg is not None:
            zxz_arr = np.asarray(zxz_deg, dtype=float)
            if zxz_arr.shape != (3,):
                raise ValueError(f"zxz_deg must have shape (3,), got {zxz_arr.shape}")
            return zxz_arr
        return np.array([0.0, float(tilt_x_deg or 0.0), 0.0], dtype=float)

    def projection_axes(
        self,
        tilt_x_deg: float | None = 0.0,
        zxz_deg: NDArray | tuple | list | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Return (u_proj, v_proj, w_proj) in sample-frame coordinates.

        `u_proj` is the fast-scan direction, `v_proj` the slow-scan direction,
        and `w_proj` the beam direction. The sample is rotated by the given
        ZXZ Euler angles relative to the (lab, beam-down-z) frame; the lab
        axes (1,0,0), (0,1,0), (0,0,1) are then expressed in the sample frame.
        """
        zxz_arr = self._resolve_zxz_deg(tilt_x_deg, zxz_deg)
        rot = Rotation.from_euler("zxz", zxz_arr, degrees=True)
        # Lab→sample mapping: apply the inverse rotation to lab basis vectors.
        u_proj = rot.inv().apply(np.array([1.0, 0.0, 0.0]))
        v_proj = rot.inv().apply(np.array([0.0, 1.0, 0.0]))
        w_proj = np.cross(u_proj, v_proj)
        w_norm = np.linalg.norm(w_proj)
        if w_norm == 0:
            raise ValueError("u_proj and v_proj must not be parallel")
        w_proj /= w_norm
        return u_proj, v_proj, w_proj

    def simulate_4dstem(
        self,
        *,
        tilt_x_deg: float | None = 0.0,
        zxz_deg: NDArray | tuple | list | None = None,
        detector_rotation_deg: float = 0.0,
        scan_step: float | tuple[float, float] = 1.0,
        scan_shape: tuple[int, int] = (11, 11),
        scan_origin: NDArray | tuple | list | None = None,
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
        zxz_arr = self._resolve_zxz_deg(tilt_x_deg, zxz_deg)
        u_proj, v_proj, w_proj = self.projection_axes(zxz_deg=zxz_arr)

        scan_step_arr = np.atleast_1d(np.asarray(scan_step, dtype=float))
        if scan_step_arr.size == 1:
            scan_step_arr = np.array([float(scan_step_arr[0])] * 2)
        if scan_step_arr.shape != (2,):
            raise ValueError(
                f"scan_step must be a scalar or shape (2,), got shape={scan_step_arr.shape}"
            )

        n_slow, n_fast = int(scan_shape[0]), int(scan_shape[1])
        if n_slow <= 0 or n_fast <= 0:
            raise ValueError(f"scan_shape must be positive, got {scan_shape}")

        sampling3 = np.asarray(self.dataset.sampling[:3], dtype=float)
        real_shape = np.asarray(self.array.shape[:3], dtype=float)
        if scan_origin is None:
            scan_origin_arr = (real_shape - 1) / 2.0 * sampling3
        else:
            scan_origin_arr = np.asarray(scan_origin, dtype=float)
            if scan_origin_arr.shape != (3,):
                raise ValueError(
                    f"scan_origin must have shape (3,), got {scan_origin_arr.shape}"
                )

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
        )

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
        slice_cache = np.zeros(
            (Nx, Ny, Nz, dp_shape_out[0], dp_shape_out[1]),
            dtype=self.array.dtype,
        )
        material_idx = np.argwhere(material_mask)
        for ix, iy, iz in material_idx:
            slice_cache[ix, iy, iz] = self._sample_complex_volume_trilinear(
                self.array[ix, iy, iz],
                k_slice_coords,
                mode="grid-wrap",
            )
        # Forward_prop will pick up this attribute through getattr.
        self._material_slice_cache = slice_cache

        out = np.empty((n_slow, n_fast, dp_shape_out[0], dp_shape_out[1]), dtype=np.float64)
        slow_centered = np.arange(n_slow, dtype=float) - (n_slow - 1) / 2.0
        fast_centered = np.arange(n_fast, dtype=float) - (n_fast - 1) / 2.0

        rotate_detector = not np.isclose(detector_rotation_deg, 0.0)

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
            dp = np.abs(Psi) ** 2
            if rotate_detector:
                dp_shifted = np.fft.fftshift(dp)
                dp_rotated = ndi.rotate(
                    dp_shifted,
                    detector_rotation_deg,
                    reshape=False,
                    order=1,
                    mode="constant",
                    cval=0.0,
                )
                dp = np.fft.ifftshift(dp_rotated)
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
                    if np.isclose(zxz_arr[0], 0.0) and np.isclose(zxz_arr[2], 0.0)
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
