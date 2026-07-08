from itertools import permutations, product
from typing import Self, Any, Literal, Sequence

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
from quantem.diffraction.full_diffraction_tomography import DiffractionTomography
from quantem.tomography.object_models import (
    ObjectModelType,
    ObjectPixelated,
    ObjConstraintsType,
    ObjConstraintParams,
)


class SimDiffractionTomography(DiffractionTomography):
    """
    Container for simulating diffraction tomography data.  
    """

    _token = object()
    default_energy = 300e3
    axis_labels = ("x", "y", "z", "kx", "ky", "kz")
    
    def __init__(
            self,
            dataset: Dataset6d,
            obj_model: ObjectPixelated | None = None,
            _token: object | None = None,
            device: int | str = 'cpu',
    ):
        super().__init__(dataset=dataset, device=device, obj_model=obj_model, _token=_token)

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
        device: str | int = 'cpu'
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
        return cls.from_dataset(dataset, energy=energy, wavelength=wavelength, device=device)


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

        Parameters 
        ----------
        tilt_x_deg: float
            Defines rotation angle around x-axis (out of plane)
        zxz_deg: torch.Tensor
            Defines rotation angle around all Euler angles (zxz)
        detector_rotation_deg: float
            Defines rotation angle of detector
        scan_step: tuple(float, float)
            yx scan sampling in real space
        scan_shape: tuple(int, int)
            yx scan shape centered around scan_origin
        scan_origin: torch.Tensor
            yx location of center of scan area
        probe_k_max: float
            Max k-vector magnitude (A^-1)
        dp_shape: tuple(int, int)
            Diffraction pattern shape
        
        Returns
        -------
        dp: Dataset4dstem
            Returns 4D STEM dataset

        """
        device = torch.device(self.device)

        zxz_arr = self._resolve_zxz_deg(tilt_x_deg, zxz_deg) #formats rotation angles
        u_proj, v_proj, w_proj = self.projection_axes(zxz_deg=zxz_arr) #applies rotation matrix to basis vectors
        u_proj = u_proj.to(device=device, dtype=torch.float32)
        v_proj = v_proj.to(device=device, dtype=torch.float32)
        w_proj = w_proj.to(device=device, dtype=torch.float32)

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

        # Scan positions shared with reconstruct_pix so forward simulation and
        # reconstruction agree exactly on probe placement.
        positions, scan_origin_arr = self._scan_position_indices(
            scan_shape=(n_slow, n_fast),
            scan_step=scan_step,
            scan_origin=scan_origin,
            u_proj=u_proj,
            v_proj=v_proj,
        )
        #setting output diffraction shape
        diff_shape = self.diffraction_shape
        if dp_shape is None:
            # detector plane = (kx, ky) = first two diffraction axes
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
            # Cache the DEVIATION slice only: the vacuum baseline is placed
            # analytically at the central pixel inside forward_prop.
            deviation = self.array[ix, iy, iz].clone()
            deviation[0, 0, 0] = 0.0
            slice_cache[ix, iy, iz] = self._sample_complex_volume_trilinear(
                deviation,
                k_slice_coords,
            )
        # Forward_prop will pick up this attribute through getattr.
        self._material_slice_cache = slice_cache

        out = torch.empty((n_slow, n_fast, dp_shape_out[0], dp_shape_out[1]), dtype=torch.float64)

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
            Psi = self.forward_prop(
                Psi0,
                positions[j, i],
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
            float(self.dataset.sampling[4]),
            float(self.dataset.sampling[5]),
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