"""Diffraction tomography.

The object is represented compactly instead of as an explicit 6D structure-factor volume:

  * ``basis``   -- ``num_structures`` shared complex 3D k-space structure factors,
    shape ``[N_kz, N_ky, N_kx, N_weights]``, spherically band-limited.
  * ``weights`` -- per real-space voxel mixing coefficients,
    shape ``[N_z, N_y, N_x, N_weights]``.
  * ``angles``  -- per real-space voxel SO(3) orientation (R9+SVD param),
    one rotation per voxel (``T = N_z * N_y * N_x``).

Each voxel's 3D structure factor is ``sum_i weight_i * Rotate(basis_i)``; the
vacuum baseline enters as an explicit delta weighted by the material complement,
``(1 - sum_i weight_i) * delta(k)``, so a voxel blends linearly from pure vacuum
(weight 0) to pure material transmission (weight 1) and the basis DC is a
learned, data-constrained quantity.

Indexing convention (new): real space is ``[z, y, x]`` (beam along +z = axis 0),
reciprocal space is ``[kz, ky, kx]``.  Any 2D quantity (detector plane, scan
raster, scan origin, scan step) is ``[row, col]`` == ``[y, x]`` to avoid xy/yx
ambiguity.

Forward pass (per ray, multislice):
  1. march the ray along z; sample it at each slice,
  2. at each sample point take the surrounding 2x2x2 voxel cluster and build
     each of the 8 voxels' 2D transmission plane separately (its own weights and
     orientation composed with the tilt),
  3. combine the 8 planes with the trilinear weights (interpolation happens in
     *transmission* space, never on the angles),
  4. add the vacuum delta ``(1 - w_eff)`` at the origin, ``w_eff`` = the
     trilinearly interpolated total material weight of the cluster,
  5. apply as a phase grating and Fresnel-propagate to the next slice.
"""

from __future__ import annotations

import math
from itertools import permutations, product

import numpy as np
from numpy.typing import NDArray
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation
import tqdm
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from quantem.core.ml.models.so3params import SO3ParamR9SVD
from quantem.core.utils.utils import electron_wavelength_angstrom


class DiffractionTomography:
    """6D diffraction tomography (see the module docstring).

    The object is a set of shared 3D k-space bases, a per-voxel weight over
    those bases, and a per-voxel SO(3) orientation, in place of an explicit 6D
    structure-factor volume. Call :meth:`init_parameters` to create the
    learnable tensors, :meth:`make_probe` / :meth:`make_propagator` to set up
    the wave optics, then :meth:`simulate` (forward) or :meth:`reconstruct`
    (inverse).
    """

    _token = object()
    default_energy = 300e3
    axis_labels = ("z", "y", "x", "kz", "ky", "kx")

    def __init__(
        self,
        real_shape: tuple[int, int, int],
        k_shape: tuple[int, int, int],
        real_sampling: tuple[float, float, float],
        k_sampling: tuple[float, float, float],
        num_structures: int = 1,
        energy: float = 3.0e5,
        probe_k_max: float = 0.10,
        basis: torch.Tensor | None = None,
        learn_basis: bool = True,
        angles=None,
        learn_angles: bool = True,
        noise: float = 1e-6,
        antialias_fraction: float = 0.9,
        antialias_softness: float = 0.05,
        device: str = "cpu",
        seed: int = 0,
    ):
        """
        Parameters
        ----------
        real_shape, k_shape : tuple[int, int, int]
            Real-space voxel grid ``(N_z, N_y, N_x)`` (beam along +z) and the
            reciprocal / basis grid ``(N_kz, N_ky, N_kx)``.
        real_sampling, k_sampling : tuple[float, float, float]
            Real-space ``(dz, dy, dx)`` [A/px] (dz = slice thickness) and
            reciprocal ``(dkz, dky, dkx)`` [1/A] sampling.
        num_structures : int, default 1
            Number of shared 3D structure-factor bases (and per-voxel weights).
        energy : float, default 3e5
            Beam energy in eV (sets the electron wavelength).
        probe_k_max : float, default 0.10
            Probe aperture radius [1/A].
        basis : torch.Tensor, optional
            Seed for the k-space bases; if None, small complex noise.
        learn_basis, learn_angles : bool, default True
            Whether the basis / per-voxel orientations are optimised.
        angles : optional
            Initial per-voxel orientation: a single ``(3, 3)`` rotation, a
            ``(n_voxels, 3, 3)`` stack, or None (random). Pass the known
            orientation with ``learn_angles=False`` to isolate basis recovery.
        noise : float, default 1e-6
            Init amplitude for the noise basis / weights.
        antialias_fraction, antialias_softness : float
            Fresnel-propagator circular anti-alias envelope.
        device : str, default "cpu"
        seed : int, default 0
            Seed for parameter initialisation and the reset generator.

        Notes
        -----
        The constructor fully sets the model up -- probe aperture, propagator and
        learnable parameters. Call :meth:`make_probe`, :meth:`make_propagator` or
        :meth:`init_parameters` again only to reconfigure.
        """
        self.real_shape = tuple(int(n) for n in real_shape)
        self.k_shape = tuple(int(n) for n in k_shape)
        self.real_sampling = tuple(float(s) for s in real_sampling)
        self.k_sampling = tuple(float(s) for s in k_sampling)
        self.num_structures = int(num_structures)
        self.energy = float(energy)
        self.wavelength = float(electron_wavelength_angstrom(self.energy))
        self.device = torch.device(device)
        self.seed = int(seed)

        Nz, Ny, Nx = self.real_shape
        self.n_voxels = Nz * Ny * Nx
        # detector plane = in-plane part of the basis: (N_ky, N_kx) == [row, col]
        self.det_shape = (self.k_shape[1], self.k_shape[2])

        self._build_reciprocal_grids()
        self._build_sphere_mask()
        self.make_probe(probe_k_max)
        self.make_propagator(antialias_fraction, antialias_softness)
        self.init_parameters(basis=basis, learn_basis=learn_basis, noise=noise,
                             angles=angles, learn_angles=learn_angles)

    def _build_reciprocal_grids(self) -> None:
        Nkz, Nky, Nkx = self.k_shape
        dkz, dky, dkx = self.k_sampling
        dev = self.device
        # fftfreq-style centered indices * dk -> A^-1, storage order [kz, ky, kx]
        self.kz = (torch.fft.fftfreq(Nkz, d=1.0 / (Nkz * dkz)).to(dev))[:, None, None]
        self.ky = (torch.fft.fftfreq(Nky, d=1.0 / (Nky * dky)).to(dev))[None, :, None]
        self.kx = (torch.fft.fftfreq(Nkx, d=1.0 / (Nkx * dkx)).to(dev))[None, None, :]
        # detector 2D grids (row=ky, col=kx), fftfreq order to match the basis
        self.det_kv = (torch.fft.fftfreq(Nky, d=1.0 / (Nky * dky)).to(dev))  # row
        self.det_ku = (torch.fft.fftfreq(Nkx, d=1.0 / (Nkx * dkx)).to(dev))  # col

    def _build_sphere_mask(self) -> None:
        """Boolean mask of the basis support.

        The radius is the smaller of the inscribed sphere and the DATA-VISIBLE
        radius set by the propagator's anti-alias envelope: beyond the envelope
        cutoff neither the measurements nor the predictions carry intensity, so
        basis voxels there are invisible to the loss and would only accumulate
        never-corrected fog at the shell.
        """
        Nkz, Nky, Nkx = self.k_shape
        dev = self.device
        # index-space radii from the (fft) origin, in *pixels*
        iz = torch.fft.fftfreq(Nkz, d=1.0 / Nkz).to(dev)[:, None, None]
        iy = torch.fft.fftfreq(Nky, d=1.0 / Nky).to(dev)[None, :, None]
        ix = torch.fft.fftfreq(Nkx, d=1.0 / Nkx).to(dev)[None, None, :]
        r_pix = torch.sqrt(iz ** 2 + iy ** 2 + ix ** 2)
        r_max = (min(self.k_shape) - 1) / 2.0        # inscribed radius
        vis = getattr(self, "_visible_k_max", None)  # A^-1, set by make_propagator
        if vis is not None:
            r_max = min(r_max, vis / min(self.k_sampling))
        self.sphere_mask = (r_pix <= r_max)          # [Nkz, Nky, Nkx] bool
        self.sphere_radius_pix = float(r_max)

    def init_parameters(
        self,
        basis: torch.Tensor | None = None,
        learn_basis: bool = True,
        noise: float = 1e-6,
        angles=None,
        learn_angles: bool = True,
    ) -> None:
        """Create the learnable ``basis``, ``weights`` and ``angles`` params.

        Parameters
        ----------
        basis : torch.Tensor, optional
            Seeds the k-space bases; if None, small complex noise with the
            origin (transmission DC) starting at 1.
        learn_basis : bool, default True
            Optimise the basis (False freezes it).
        noise : float, default 1e-6
            Init amplitude for the noise basis / weights.
        angles : optional
            Initial per-voxel orientation: a ``(3, 3)`` rotation (broadcast to
            all voxels), a ``(n_voxels, 3, 3)`` stack, or None (random SO(3)).
        learn_angles : bool, default True
            Optimise the orientations. Set False (with ``angles`` = the known
            ground-truth orientation) to isolate whether the basis alone can
            recover the Bragg peaks.
        """
        Nz, Ny, Nx = self.real_shape
        Nkz, Nky, Nkx = self.k_shape
        Nw = self.num_structures
        dev = self.device
        gen = torch.Generator(device="cpu").manual_seed(self.seed)

        if basis is None:
            b = (
                torch.randn(Nkz, Nky, Nkx, Nw, generator=gen, dtype=torch.float64)
                + 1j * torch.randn(Nkz, Nky, Nkx, Nw, generator=gen, dtype=torch.float64)
            ) * noise
        else:
            b = torch.as_tensor(basis, dtype=torch.complex128).clone()
            if b.ndim == 3:
                b = b[..., None]
            assert b.shape == (Nkz, Nky, Nkx, Nw), f"basis shape {b.shape}"
        b = b.to(dev)
        b[~self.sphere_mask] = 0.0            # spherical support
        b[0, 0, 0, :] = 1.0                    # origin pinned to 1 at init
        self.basis = b.clone().requires_grad_(learn_basis)
        self.learn_basis = learn_basis

        # per-voxel weights (unconstrained real), small so init ~ vacuum
        self.weights = (
            noise * torch.randn(Nz, Ny, Nx, Nw, generator=gen, dtype=torch.float64)
        ).to(dev).requires_grad_(True)

        # per-voxel SO(3) orientation (R9+SVD)
        if angles is None:
            torch.manual_seed(self.seed)
            self.angles = SO3ParamR9SVD(self.n_voxels, init="random").to(dev)
        else:
            R = torch.as_tensor(angles, dtype=torch.float32)
            if R.ndim == 2:
                R = R[None].expand(self.n_voxels, 3, 3).contiguous()
            assert R.shape == (self.n_voxels, 3, 3), f"angles shape {R.shape}"
            self.angles = SO3ParamR9SVD.from_matrix(R).to(dev)
        self.learn_angles = learn_angles
        if not learn_angles:
            for p in self.angles.parameters():
                p.requires_grad_(False)

    def parameters(self) -> list[torch.Tensor]:
        ps = [self.weights]
        if self.learn_basis:
            ps = [self.basis, *ps]
        if self.learn_angles:
            ps = [*ps, *self.angles.parameters()]
        return ps

    def rotation_matrices(self) -> torch.Tensor:
        """(N_z, N_y, N_x, 3, 3) per-voxel rotation matrices (body->lab).

        The R9+SVD projection can fail when a voxel's M drifts toward repeated
        singular values (LAPACK non-convergence); a tiny in-place jitter breaks
        the degeneracy and the projection is retried.
        """
        try:
            R = self.angles.as_matrix()             # (T, 3, 3)
        except Exception:
            with torch.no_grad():
                self.angles.M.add_(1e-4 * torch.randn_like(self.angles.M))
            R = self.angles.as_matrix()
        return R.reshape(*self.real_shape, 3, 3)

    def masked_basis(self) -> torch.Tensor:
        """Basis with the spherical support re-imposed (differentiable).

        The basis is deliberately *not* normalised. The vacuum delta in the
        transmission assembly carries the material complement ``(1 - w_eff)``,
        so the weight/basis split is physical, not a gauge: scaling the basis
        up and the weights down changes the vacuum mix and therefore the fit.

        Returns
        -------
        torch.Tensor
            ``[N_kz, N_ky, N_kx, N_weights]`` basis with values outside the
            inscribed sphere zeroed.
        """
        return self.basis * self.sphere_mask[..., None]

    def lab_structure_factor(
        self,
        voxel: int | tuple[int, int, int] = 0,
        keep_origin: bool = True,
    ) -> torch.Tensor:
        """One voxel's structure factor resampled on the lab k grid.

    @staticmethod
    def _sample_complex_volume_trilinear(
        volume: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Samples (6D) volume at specified grid points ((-1,-1): top left corner, (1,1): bottom right corner)
        
        Parameters
        ----------
        voxel : int or (int, int, int)
            Flat index or ``[z, y, x]`` index of the voxel.
        keep_origin : bool, default True
            With False the origin (vacuum baseline) is removed BEFORE the
            rotation. The origin is ~100x brighter than the Bragg peaks, so
            resampling bleeds it into the surrounding voxel cluster; remove it
            here rather than zeroing the center of the result when comparing
            or displaying off-origin content.

        Returns
        -------
        torch.Tensor
            ``[N_kz, N_ky, N_kx]`` complex lab-frame structure factor, in the
            same (unshifted) frequency ordering as ``basis``.
        """
        if isinstance(voxel, (tuple, list)):
            voxel = int(np.ravel_multi_index(tuple(int(v) for v in voxel), self.real_shape))
        R = self.rotation_matrices().reshape(-1, 3, 3)[voxel].detach()
        W = self.weights.reshape(-1, self.num_structures)[voxel].detach()
        body = (self.masked_basis().detach() * W).sum(-1)
        if not keep_origin:
            body = body.clone()
            body[0, 0, 0] = 0.0
        sf = torch.fft.fftshift(body, dim=(0, 1, 2))
        dev = sf.device
        ctr = torch.tensor([n // 2 for n in self.k_shape], dtype=torch.float32, device=dev)
        axes = [torch.arange(n, dtype=torch.float32, device=dev) - n // 2 for n in self.k_shape]
        QZ, QY, QX = torch.meshgrid(*axes, indexing="ij")
        q = torch.stack([QZ, QY, QX], -1).reshape(-1, 3)
        p = q @ R.to(torch.float32)                     # rows are R^T q: lab -> body frame
        size = torch.tensor([float(n) for n in self.k_shape], device=dev)
        gn = 2.0 * (p + ctr) / (size - 1.0) - 1.0
        grid = gn.flip(-1).reshape(1, *self.k_shape, 3)  # grid_sample wants (x, y, z)
        vol = torch.stack([sf.real, sf.imag], 0)[None].to(torch.float32)
        out = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        return torch.fft.ifftshift(torch.complex(out[0], out[1]), dim=(0, 1, 2))

    @classmethod
    def from_test(
        cls,
        real_shape: tuple[int, int, int] = (6, 12, 12),
        k_shape: tuple[int, int, int] = (31, 31, 31),
        real_sampling: tuple[float, float, float] = (10.0, 10.0, 10.0),
        k_sampling: tuple[float, float, float] = (0.05, 0.05, 0.05),
        particle_centers=((1, 3, 4), (2, 7, 8), (3, 9, 2), (4, 4, 9)),
        particle_zxz_deg=((0.0, 0.0, 0.0), (45.0, 54.7, 15.0),
                          (0.0, 45.0, -10.0), (12.0, 23.0, 54.0)),
        particle_radius: float = 1.6,
        energy: float = 3.0e5,
        probe_k_max: float = 0.10,
        seed: int = 0,
        **kwargs,
    ) -> "DiffractionTomography":
        """Create ground-truth test data: hard-sphere Au particles in vacuum.

        Each particle shares the same (single) Au structure factor and differs
        only by its ZXZ orientation. The returned model carries the true
        weights (1 inside the spheres), per-voxel orientations, and the frozen
        Au basis, ready for :meth:`simulate`.

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
        real_shape : tuple[int, int, int]
            Real-space voxel grid ``(N_z, N_y, N_x)``.
        k_shape, real_sampling, k_sampling, energy, probe_k_max, seed
            As in the constructor.
        particle_centers : sequence of (z, y, x)
            Sphere centers in voxel indices.
        particle_zxz_deg : sequence of (a, b, c)
            Per-particle intrinsic ZXZ Euler angles in degrees: ``a`` (about
            the beam) and ``b`` (polar, about x) select which zone axis lies
            along the beam at zero tilt; ``c`` is the in-plane rotation of the
            zero-tilt diffraction pattern.
        particle_radius : float, default 1.6
            Hard-sphere radius in voxels.

        Notes
        -----
        The particle mask and grain index are kept on the returned model as
        ``particle_mask`` (bool ``[N_z, N_y, N_x]``) and ``grain_id`` (long,
        -1 for vacuum), for use in accuracy metrics.
        """
        from scipy.spatial.transform import Rotation

        Nz, Ny, Nx = (int(n) for n in real_shape)
        au = cls.make_au_basis(k_shape, k_sampling)[..., None]
        zz, yy, xx = torch.meshgrid(torch.arange(Nz), torch.arange(Ny),
                                    torch.arange(Nx), indexing="ij")
        weights = torch.zeros(Nz, Ny, Nx, 1, dtype=torch.float64)
        R_all = torch.eye(3).reshape(1, 3, 3).repeat(Nz * Ny * Nx, 1, 1)
        grain_id = -torch.ones(Nz, Ny, Nx, dtype=torch.long)
        for g, ((pz, py, px), euler) in enumerate(zip(particle_centers, particle_zxz_deg)):
            mask = (torch.sqrt((zz - pz) ** 2 + (yy - py) ** 2 + (xx - px) ** 2)
                    <= particle_radius)
            weights[mask] = 1.0
            grain_id[mask] = g
            # scipy matrices act on (x, y, z) components; this class stores
            # vectors as [z, y, x]. Conjugate by the axis reversal (M[::-1,
            # ::-1]) and transpose so the stored body->lab rotation gives
            # pattern(k) = basis(Rz(a) Rx(b) Rz(g) k): a and b select the zone
            # axis along the beam, g spins the zero-tilt pattern in plane.
            M = Rotation.from_euler("ZXZ", euler, degrees=True).as_matrix()
            R = torch.tensor(M.T[::-1, ::-1].copy(), dtype=torch.float32)
            R_all[mask.flatten()] = R

        gt = cls(real_shape=real_shape, k_shape=k_shape, real_sampling=real_sampling,
                 k_sampling=k_sampling, num_structures=1, energy=energy,
                 probe_k_max=probe_k_max, basis=au, learn_basis=False,
                 angles=R_all, learn_angles=False, seed=seed, **kwargs)
        with torch.no_grad():
            gt.weights.copy_(weights)
        gt.particle_mask = grain_id >= 0
        gt.grain_id = grain_id
        return gt

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

    def ray_samples(self, origin_zyx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Index-space sample points along a ray through the slab z-extent.

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

    def _transmission_plane(self, point_zyx: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                            basis: torch.Tensor, R_all: torch.Tensor, W_all: torch.Tensor) -> torch.Tensor:
        """Assemble the 2D transmission SF at one continuous ray point.

        The 2x2x2 trilinear cluster is handled in one batched ``grid_sample``:
        each in-bounds voxel's orientation is composed with the tilt to give its
        sampling plane through the bases, the resulting per-voxel transmission
        planes are combined with the trilinear weights (in transmission space,
        never on the angles), and the vacuum delta ``(1 - w_eff)`` is added at
        the origin. Out-of-slab neighbours contribute vacuum.

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
        point_zyx : torch.Tensor
            ``(3,)`` continuous ray position in index space ``[z, y, x]``.
        u, v : torch.Tensor
            ``(3,)`` lab-frame detector column/row axes for this tilt.
        basis : torch.Tensor
            ``[N_kz, N_ky, N_kx, N_weights]`` complex bases.
        R_all, W_all : torch.Tensor
            Per-voxel rotations ``[N_z, N_y, N_x, 3, 3]`` and weights
            ``[N_z, N_y, N_x, N_weights]``.

        Returns
        -------
        torch.Tensor
            ``(det_row, det_col)`` complex transmission structure factor.
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

    def _transmission_planes_batch(self, points: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                                   basis: torch.Tensor, R_all: torch.Tensor, W_all: torch.Tensor) -> torch.Tensor:
        """Transmission SF for a batch of ray points (one per probe), at one slice.

        Parameters
        ----------
        points : torch.Tensor
            ``(P, 3)`` continuous ray positions in index space ``[z, y, x]``.
        u, v : torch.Tensor
            ``(3,)`` lab-frame detector column/row axes for this tilt.
        basis, R_all, W_all : torch.Tensor
            As in :meth:`_transmission_plane`.

        Returns
        -------
        torch.Tensor
            ``(P, det_row, det_col)`` complex transmission structure factors.
        """
        Nz, Ny, Nx = self.real_shape
        Nkz, Nky, Nkx = self.k_shape
        Nw = basis.shape[-1]
        dev = self.device
        P = points.shape[0]
        base = torch.floor(points).to(torch.int64)             # (P,3)
        frac = points - base.to(points.dtype)                  # (P,3)

        kv, ku = torch.meshgrid(self.det_kv, self.det_ku, indexing="ij")  # (R,C)
        det_row, det_col = kv.shape
        dk = torch.tensor(self.k_sampling, dtype=torch.float32, device=dev)
        lo = torch.tensor([0, 0, 0], device=dev)
        hi = torch.tensor([Nz - 1, Ny - 1, Nx - 1], device=dev)
        basis_cs = torch.fft.fftshift(basis, dim=(0, 1, 2))    # centered sampling space
        bre_full = basis_cs.real.permute(3, 0, 1, 2)           # (Nw,Nkz,Nky,Nkx)
        bim_full = basis_cs.imag.permute(3, 0, 1, 2)

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

        All ``P x 8`` cluster corners go through a single grid build and one
        ``grid_sample`` per real/imag part (the per-corner Python loop cost --
        and its 8x autograd graph fan-out -- dominated the profile). Returns
        ``(P, det_row, det_col)`` WITHOUT the vacuum delta (the caller adds
        ``(1 - w_eff)`` at the origin once per superslice group).
        """
        Nkz, Nky, Nkx = self.k_shape
        Nw = basis.shape[-1]
        P = vidx.shape[0]
        V = P * 8
        vflat = vidx.reshape(V)
        R = R_all.reshape(-1, 3, 3)[vflat]                              # (V, 3, 3)
        wv = W_all.reshape(-1, Nw)[vflat].to(torch.complex64)           # (V, Nw)
        u_lab, v_lab = geo["u"], geo["v"]
        if u_lab.ndim == 1:                                             # one shared tilt
            u_b = torch.einsum("vij,i->vj", R, u_lab)                   # (V, 3) = R^T u
            v_b = torch.einsum("vij,i->vj", R, v_lab)
        else:                                                            # per-ray axes (tilt batch)
            u_b = torch.einsum("vij,vi->vj", R, u_lab)
            v_b = torch.einsum("vij,vi->vj", R, v_lab)
        ku, kv = geo["ku"], geo["kv"]                                   # (Rr, Cc)
        dk = torch.tensor(self.k_sampling, dtype=torch.float32, device=self.device)
        kxyz = (ku[None, ..., None] * u_b[:, None, None, :]
                + kv[None, ..., None] * v_b[:, None, None, :])          # (V, Rr, Cc, 3)
        # sample in CENTERED (fftshift) space: fractional coordinates near the
        # frequency-zero planes then interpolate their true neighbors, and the
        # wrap seam moves beyond the spherical support where samples are zero.
        # (Unshifted remainder-wrap sampling attenuated every rotated read with
        # a coordinate component in (-1, 0) -- an axis-aligned seam cross.)
        ctr = torch.tensor([Nkz // 2, Nky // 2, Nkx // 2],
                           dtype=torch.float32, device=self.device)
        c = kxyz / dk + ctr
        grid = torch.stack((
            2.0 * c[..., 2] / (Nkx - 1.0) - 1.0,
            2.0 * c[..., 1] / (Nky - 1.0) - 1.0,
            2.0 * c[..., 0] / (Nkz - 1.0) - 1.0,
        ), dim=-1)[:, None].to(torch.float32)                           # (V, 1, Rr, Cc, 3)
        basis_c = torch.fft.fftshift(basis, dim=(0, 1, 2))
        bre = basis_c.real.permute(3, 0, 1, 2)[None].expand(V, -1, -1, -1, -1).to(torch.float32)
        bim = basis_c.imag.permute(3, 0, 1, 2)[None].expand(V, -1, -1, -1, -1).to(torch.float32)
        sre = F.grid_sample(bre, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sim = F.grid_sample(bim, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled = (sre + 1j * sim).squeeze(2)                           # (V, Nw, Rr, Cc)
        t_vox = (wv[:, :, None, None] * sampled).sum(1)                 # (V, Rr, Cc)
        det_r, det_c = self.det_shape
        return (tw.reshape(V)[:, None, None] * t_vox).reshape(P, 8, det_r, det_c).sum(1)

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

        Returns ``(1, 2*Nw, N_kz, N_ky, N_kx)`` float32: the fftshifted basis
        with real parts in the first ``Nw`` channels and imaginary in the last
        ``Nw``. Hoisting this out of the per-slice/per-tilt sampling loop
        computes the shift + permute + cast ONCE per forward instead of once
        per (slice, tilt), collapsing that many redundant autograd nodes.
        """
        basis_c = torch.fft.fftshift(basis, dim=(0, 1, 2))
        re = basis_c.real.permute(3, 0, 1, 2)                          # (Nw,kz,ky,kx)
        im = basis_c.imag.permute(3, 0, 1, 2)
        return torch.cat([re, im], 0)[None].to(torch.float32)

    def _transmission_planes_unique(self, uv: torch.Tensor, T: torch.Tensor,
                                    geo: dict, vol: torch.Tensor, Nw: int,
                                    R_all: torch.Tensor, W_all: torch.Tensor) -> torch.Tensor:
        """Slice SF for all probes via the unique-voxel factorization.

        PRISM-adapted reuse: at one tilt every ray-corner touching voxel ``v``
        needs the SAME plane (the basis sampled at ``R_v^T k``), so the
        ``n_unique`` struck voxels are sampled once and each probe's slice SF
        is assembled by the constant trilinear matrix ``T`` (``(P, n_unique)``,
        from the geometry cache): ``SF = T @ (w_v * plane_v)``. Identical
        algebra to the per-corner path, reordered; the sampling work and the
        autograd graph shrink by the corner-sharing factor (~10-60x for
        one-probe-per-voxel scans).

        ``vol`` is the channel-packed basis from :meth:`_basis_sample_vol`; all
        ``V`` unique voxels are sampled from that single volume in one
        ``grid_sample`` (real and imaginary as packed channels). Returns
        ``(P, det_r, det_c)`` WITHOUT the vacuum delta.
        """
        det_r, det_c = self.det_shape
        P = T.shape[0]
        if uv.numel() == 0:
            return torch.zeros(P, det_r, det_c, dtype=torch.complex64,
                               device=self.device)
        Nkz, Nky, Nkx = self.k_shape
        V = uv.numel()
        R = R_all.reshape(-1, 3, 3)[uv]                                 # (V,3,3)
        wv = W_all.reshape(-1, Nw)[uv].to(torch.complex64)             # (V,Nw)
        u_b = torch.einsum("vij,i->vj", R, geo["u"])                    # R^T u
        v_b = torch.einsum("vij,i->vj", R, geo["v"])
        ku, kv = geo["ku"], geo["kv"]
        dk = torch.tensor(self.k_sampling, dtype=torch.float32, device=self.device)
        kxyz = (ku[None, ..., None] * u_b[:, None, None, :]
                + kv[None, ..., None] * v_b[:, None, None, :])          # (V,r,c,3)
        ctr = torch.tensor([Nkz // 2, Nky // 2, Nkx // 2],
                           dtype=torch.float32, device=self.device)
        c = kxyz / dk + ctr
        grid = torch.stack((
            2.0 * c[..., 2] / (Nkx - 1.0) - 1.0,
            2.0 * c[..., 1] / (Nky - 1.0) - 1.0,
            2.0 * c[..., 0] / (Nkz - 1.0) - 1.0,
        ), dim=-1)[None].to(torch.float32)                              # (1,V,r,c,3)
        # one grid_sample reads the single volume; V rotations along output
        # depth, real/imag along channels
        out = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=True)[0]                     # (2Nw,V,r,c)
        sampled = torch.complex(out[:Nw], out[Nw:]).permute(1, 0, 2, 3)  # (V,Nw,r,c)
        t_vox = (wv[:, :, None, None] * sampled).sum(1)                 # (V,r,c)
        return (T.to(torch.complex64) @ t_vox.reshape(V, -1)).reshape(P, det_r, det_c)

    def forward_tilt(self, origins: torch.Tensor, tilt_x_deg: float,
                     basis: torch.Tensor, R_all: torch.Tensor, W_all: torch.Tensor,
                     phase_only: bool = True, superslice: int = 1) -> torch.Tensor:
        """Multislice exit waves for *all* probes at one tilt (batched over probes).

        Probes at a fixed tilt share the beam direction, so their rays span the
        same z-steps; the wave is carried as ``(P, det, det)`` and every FFT,
        propagation and transmission build is batched over ``P``. The sampling
        geometry is cached across calls (:meth:`_tilt_geometry`) and all
        cluster corners of a slice go through one fused ``grid_sample``.

        Parameters
        ----------
        origins : torch.Tensor
            ``(P, 3)`` probe positions in index space ``[z, y, x]``.
        tilt_x_deg : float
            X-axis tilt in degrees.
        superslice : int, default 1
            Group this many consecutive slices into one transmission event:
            within a group the sampled k-space planes are summed (projection
            approximation, ``prod exp(i V_j) ~ exp(i sum V_j)``) and a single
            ifft/phase/fft chain is applied, with one propagation over the
            group thickness. ``1`` is the exact per-slice multislice.

        Returns
        -------
        torch.Tensor
            ``(P, det_row, det_col)`` complex exit waves.
        """
        geo = self._tilt_geometry(origins, tilt_x_deg)
        slices = geo["slices"]
        Ns = len(slices)
        P = origins.shape[0]
        Psi = self.Psi0[None].expand(P, -1, -1).clone()
        num_pix = self.Psi0.numel()
        wsum = W_all.reshape(-1, W_all.shape[-1]).sum(-1)      # (n_voxels,)
        Nw = basis.shape[-1]
        vol = self._basis_sample_vol(basis)                   # once per forward
        groups = [list(range(s, min(s + max(1, superslice), Ns)))
                  for s in range(0, Ns, max(1, superslice))]
        for gi, grp in enumerate(groups):
            SF, Wg = None, None
            for s in grp:
                vidx, tw = slices[s]
                uv, T = geo["uslices"][s]
                sf_s = self._transmission_planes_unique(uv, T, geo, vol, Nw, R_all, W_all)
                w_s = (tw.real.to(wsum.dtype) * wsum[vidx]).sum(-1)   # (P,)
                SF = sf_s if SF is None else SF + sf_s
                Wg = w_s if Wg is None else Wg + w_s
            # vacuum delta carries the material complement, once per group
            SF[:, 0, 0] = SF[:, 0, 0] + (1.0 - Wg).to(SF.dtype)
            t_real = torch.fft.ifft2(num_pix * SF)
            if phase_only:
                t_real = torch.exp(1j * torch.angle(t_real))
            Psi = torch.fft.fft2(torch.fft.ifft2(Psi) * t_real)
            if gi < len(groups) - 1:
                Psi = Psi * self._prop_power(len(grp))
        return Psi * self.antialias_mask

    def _tilt_group_geometry(self, origins: torch.Tensor, tilts) -> dict:
        """Concatenated geometry for a GROUP of tilts (see ``tilt_batch``).

        Stacks each tilt's cached slice geometry along the ray axis and builds
        per-corner lab axes, so the whole group runs through one fused
        transmission call and one autograd graph per iteration chunk.
        """
        key = ("grp", origins.shape[0], hash(origins.cpu().numpy().tobytes()),
               tuple(round(float(t), 6) for t in tilts))
        cache = getattr(self, "_geo_cache", None)
        if cache is None:
            cache = self._geo_cache = {}
        if key in cache:
            return cache[key]
        geos = [self._tilt_geometry(origins, float(t)) for t in tilts]
        P = origins.shape[0]
        Ns = len(geos[0]["slices"])
        slices = []
        for s in range(Ns):
            vidx = torch.cat([g["slices"][s][0] for g in geos])          # (T*P, 8)
            tw = torch.cat([g["slices"][s][1] for g in geos])
            slices.append((vidx, tw))
        u_all = torch.cat([g["u"][None].expand(P * 8, 3) for g in geos])  # (T*P*8, 3)
        v_all = torch.cat([g["v"][None].expand(P * 8, 3) for g in geos])
        geo = {"u": u_all, "v": v_all, "slices": slices,
               "tilt_geos": geos, "ku": geos[0]["ku"], "kv": geos[0]["kv"]}
        cache[key] = geo
        return geo

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

        Returns
        -------
        torch.Tensor
            ``(T, P, det_row, det_col)`` complex exit waves.
        """
        geo = self._tilt_group_geometry(origins, tilts)
        slices = geo["slices"]
        Ns = len(slices)
        P = origins.shape[0]
        T = len(tilts)
        Psi = self.Psi0[None].expand(T * P, -1, -1).clone()
        num_pix = self.Psi0.numel()
        wsum = W_all.reshape(-1, W_all.shape[-1]).sum(-1)      # (n_voxels,)
        Nw = basis.shape[-1]
        vol = self._basis_sample_vol(basis)                   # once per forward
        groups = [list(range(s, min(s + max(1, superslice), Ns)))
                  for s in range(0, Ns, max(1, superslice))]
        for gi, grp in enumerate(groups):
            SF, Wg = None, None
            for s in grp:
                vidx, tw = slices[s]
                # per-tilt unique-voxel assembly, concatenated over the group
                sf_s = torch.cat([
                    self._transmission_planes_unique(
                        g["uslices"][s][0], g["uslices"][s][1], g,
                        vol, Nw, R_all, W_all)
                    for g in geo["tilt_geos"]])
                w_s = (tw.real.to(wsum.dtype) * wsum[vidx]).sum(-1)   # (T*P,)
                SF = sf_s if SF is None else SF + sf_s
                Wg = w_s if Wg is None else Wg + w_s
            SF[:, 0, 0] = SF[:, 0, 0] + (1.0 - Wg).to(SF.dtype)
            t_real = torch.fft.ifft2(num_pix * SF)
            if phase_only:
                t_real = torch.exp(1j * torch.angle(t_real))
            Psi = torch.fft.fft2(torch.fft.ifft2(Psi) * t_real)
            if gi < len(groups) - 1:
                Psi = Psi * self._prop_power(len(grp))
        return (Psi * self.antialias_mask).reshape(T, P, *self.det_shape)

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
        Nz, Ny, Nx = self.real_shape
        n_row, n_col = int(scan_shape[0]), int(scan_shape[1])
        if not isinstance(scan_step, (tuple, list)):
            scan_step = (float(scan_step), float(scan_step))
        if scan_origin is None:
            scan_origin = ((Ny - 1) / 2.0, (Nx - 1) / 2.0)   # [row=y, col=x] center
        z_c = (Nz - 1) / 2.0
        rows = (torch.arange(n_row, dtype=torch.float32) - (n_row - 1) / 2.0) * scan_step[0] + scan_origin[0]
        cols = (torch.arange(n_col, dtype=torch.float32) - (n_col - 1) / 2.0) * scan_step[1] + scan_origin[1]
        pos = torch.empty(n_row, n_col, 3, dtype=torch.float32)
        for j in range(n_row):
            for i in range(n_col):
                pos[j, i] = torch.tensor([z_c, rows[j], cols[i]])
        return pos

    def simulate(
        self,
        Psi0,
        position,
        u_proj = (1,0,0),
        v_proj = (0,1,0),
    ):
        """
        Returns coordinates along a ray for a given position and sample tilt.

        Every argument is optional: parameters default to the model's own
        (current basis / weights / orientations), and the scan geometry
        defaults to the one recorded by the last :meth:`reconstruct` call (or
        must be given for a fresh simulation). So the predicted patterns of a
        reconstruction are simply ``recon.simulate()``.

        Parameters
        ----------
        basis, weights, R_all : torch.Tensor, optional
            Object parameters; default to the model's own.
        tilts_deg, scan_shape, scan_step, scan_origin : optional
            Tilt series and scan raster; default to the geometry stored by the
            last reconstruction.
        phase_only : bool, default True
            Unitary phase-grating transmission.
        progress : bool, default True
            Show a tqdm bar over the tilt series.
        """
        if basis is None:
            basis = self.masked_basis().detach()
        if weights is None:
            weights = self.weights.detach()
        if R_all is None:
            R_all = self.rotation_matrices().detach()
        geo = getattr(self, "_scan_geometry", {})
        tilts_deg = geo.get("tilts_deg") if tilts_deg is None else tilts_deg
        scan_shape = geo.get("scan_shape") if scan_shape is None else scan_shape
        scan_step = geo.get("scan_step", 1.0) if scan_step is None else scan_step
        scan_origin = geo.get("scan_origin") if scan_origin is None else scan_origin
        if tilts_deg is None or scan_shape is None:
            raise ValueError(
                "No scan geometry: pass tilts_deg + scan_shape, or run reconstruct() first."
            )
        basis = basis * self.sphere_mask[..., None]          # enforce spherical support
        pos = self.scan_positions(scan_shape, scan_step, scan_origin)
        n_row, n_col = pos.shape[:2]
        origins = pos.reshape(-1, 3)                          # (P, 3)
        dp = torch.empty(len(tilts_deg), n_row, n_col, *self.det_shape, dtype=torch.float64)
        with torch.no_grad():
            for ti in tqdm(range(len(tilts_deg)), disable=not progress, desc="simulate", unit="tilt"):
                Psi = self.forward_tilt(origins, float(tilts_deg[ti]), basis, R_all, weights,
                                        phase_only=phase_only, superslice=superslice)
                dp[ti] = (Psi.abs() ** 2).to(torch.float64).reshape(n_row, n_col, *self.det_shape)
        return dp

    @staticmethod
    def _cubic_ops() -> np.ndarray:
        """The 24 proper rotations of the cubic point group, (24, 3, 3)."""
        ops = []
        for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
            for sx in (1, -1):
                for sy in (1, -1):
                    for sz in (1, -1):
                        M = np.zeros((3, 3))
                        M[0, perm[0]], M[1, perm[1]], M[2, perm[2]] = sx, sy, sz
                        if np.linalg.det(M) > 0.5:
                            ops.append(M)
        return np.stack(ops)

    def _cubic_grid_ops(self):
        """Cubic point-group operations as (permute, flip) on the k grid.

        Each of the 24 proper cubic rotations is a signed axis permutation, so
        on a (cubic) k grid it acts by permuting the three axes and flipping
        the negated ones. In centered (fftshifted) space a flip maps k -> -k
        exactly for the odd grid sizes used here. Cached: pure geometry.
        """
        cached = getattr(self, "_cubic_grid_ops_cache", None)
        if cached is not None:
            return cached
        ops = []
        for M in self._cubic_ops():
            Mt = M.T
            perm = tuple(int(np.argmax(np.abs(Mt[o]))) for o in range(3))
            flip = tuple(a for a in range(3) if Mt[a, perm[a]] < 0)
            ops.append((perm, flip))
        self._cubic_grid_ops_cache = ops
        return ops

    def symmetrize_basis(self) -> None:
        """Project the basis onto cubic point-group symmetry, in place.

        Averages the basis over the 24 proper cubic rotations (grid axis
        permutations + flips), enforcing that every symmetry-equivalent
        reflection carries the same amplitude. For a cubic crystal aligned to
        the grid's cardinal directions this is the material's exact symmetry:
        each Bragg peak is reinforced by its whole orbit, so partial data (a
        few grains/tilts sampling a few of the equivalents) determines them
        all, and every voxel's orientation search sees a full, clean set of
        peaks to fall into. Combined with :meth:`reconstruct`'s Friedel
        projection this is the full ``m-3m`` point group.

        Enforcing it from a random start pins the basis to the cardinal-cubic
        gauge throughout (the free global rotation makes cardinal alignment
        always an available gauge), so no separate alignment step is needed;
        the per-voxel rotations express each grain relative to that reference.
        """
        ops = self._cubic_grid_ops()
        with torch.no_grad():
            Bc = torch.fft.fftshift(self.basis, dim=(0, 1, 2))
            acc = torch.zeros_like(Bc)
            for perm, flip in ops:
                T = Bc.permute(perm[0], perm[1], perm[2], 3)
                if flip:
                    T = torch.flip(T, dims=flip)
                acc = acc + T
            self.basis.copy_(torch.fft.ifftshift(acc / len(ops), dim=(0, 1, 2)))

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

    def _reset_optimizer_state(self, opt, param, rows) -> None:
        """Zero Adam moments for selected voxel rows of a parameter.

        Both per-voxel params flatten to ``n_voxels`` on their leading dims:
        weights are ``[Nz,Ny,Nx,Nw]`` and angles ``[T,3,3]`` with ``T=n_voxels``.
        """
        st = opt.state.get(param, None)
        if not st:
            return
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in st:
                st[key].reshape(self.n_voxels, -1)[rows] = 0.0

    def _snapshot(self) -> dict:
        return {"basis": self.basis.detach().clone(),
                "weights": self.weights.detach().clone(),
                "M": self.angles.M.detach().clone()}

    def _restore(self, snap: dict) -> None:
        with torch.no_grad():
            self.basis.copy_(snap["basis"])
            self.weights.copy_(snap["weights"])
            self.angles.M.copy_(snap["M"])

    def _sanitize(self, opt, gen) -> int:
        """Repair non-finite parameters (and their Adam state) in place.

        The SVD backward of the R9 rotation parameterisation is singular when a
        voxel's M has (near-)repeated singular values; one such step writes NaN
        into M and Adam's moments, which then poisons every later iteration.
        Non-finite M rows get a fresh random rotation, non-finite weights the
        current mean, non-finite basis entries zero; the corresponding Adam
        moments are zeroed. Returns the number of repaired voxels.
        """
        n_fix = 0
        with torch.no_grad():
            badM = ~torch.isfinite(self.angles.M).reshape(self.n_voxels, -1).all(dim=1)
            if bool(badM.any()):
                idx = torch.nonzero(badM).flatten()
                n_fix = int(idx.numel())
                for v in idx.tolist():
                    self.angles.M[v] = (torch.eye(3)
                                        + 0.1 * torch.randn(3, 3, generator=gen)).to(self.device)
                self._reset_optimizer_state(opt, self.angles.M, idx)
            badW = ~torch.isfinite(self.weights).reshape(self.n_voxels, -1).all(dim=1)
            if bool(badW.any()):
                idx = torch.nonzero(badW).flatten()
                n_fix += int(idx.numel())
                w_ok = self.weights[torch.isfinite(self.weights)]
                fill = w_ok.mean() if w_ok.numel() else 0.0
                self.weights.reshape(self.n_voxels, self.num_structures)[idx] = fill
                self._reset_optimizer_state(opt, self.weights, idx)
            if not bool(torch.isfinite(self.basis.real).all() & torch.isfinite(self.basis.imag).all()):
                bad = ~(torch.isfinite(self.basis.real) & torch.isfinite(self.basis.imag))
                self.basis[bad] = 0.0
                st = opt.state.get(self.basis, None)
                if st:
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key in st:
                            st[key].zero_()
                n_fix += 1
        return n_fix

    def _make_optimizer(self, lr: float, lr_weights: float, lr_angles: float):
        """Adam with per-parameter-group learning rates (eps=1e-30 because the
        phase-object gradients are ~1e-10 and the default eps would throttle
        every step ~100x)."""
        groups = [{"params": [self.weights], "lr": lr_weights}]
        if self.learn_basis:
            groups.append({"params": [self.basis], "lr": lr})
        if self.learn_angles:
            groups.append({"params": list(self.angles.parameters()), "lr": lr_angles})
        return torch.optim.Adam(groups, eps=1e-30)


    def set_learnable(self, basis: bool | None = None, angles: bool | None = None) -> None:
        """Freeze / unfreeze the basis or the per-voxel orientations in place.

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
            val_fraction: float = 0.0,
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
            pin_memory=pin_mem,
            drop_last=False,
            )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size = batch_size * 4, # less memory than training
            pin_memory=pin_mem,
            drop_last=False,
            )
        return train_dataloader, val_dataloader
        
    #  put this under DiffractionTomography?
    def reconstruct_pix(
        self,
        measurements: torch.Tensor,
        tilts_deg,
        scan_shape: tuple[int, int],
        scan_step: float | tuple[float, float] = 1.0,
        n_rand: int = 8,
        n_jitter: int = 6,
        jitter_deg: tuple = (2.0, 8.0, 20.0),
        neighbors: int = 18,
        w_min: float = 0.45,
        w_material: float = 0.05,
        accept: float = 0.95,
        order: str = "error",
        progress: bool = False,
        seed: int = 0,
    ) -> dict:
        """One voxel-at-a-time local-search sweep (no gradients, no GT).

        Visits voxels one at a time (worst residual first with
        ``order='error'``, else random order) and, for each, trials the
        rotation and weight of its ``neighbors`` nearest voxels, the
        incumbent, ``n_rand`` random rotations, ``n_jitter`` small
        perturbations of the incumbent at each angular scale in ``jitter_deg``
        (local refinement -- a nearly-right voxel walks to the exact
        orientation without an explicit grid search; the coarse scale also
        hops between adjacent basins), and weight-only moves. The
        judge is the exact forward amplitude loss over ONLY the rays that
        intercept the voxel with trilinear weight above ``w_min`` -- on that
        restricted set the voxel's contribution dominates, so single-voxel
        evidence is decisive (a full-pattern loss dilutes it). A trial is
        adopted only when it beats the incumbent by the ``accept`` margin;
        without a real margin, sweeps churn and re-absorb freed grains.

        Alternated with short :meth:`reconstruct` stages, sweeps typically
        reach a stable fixed point (no accepted moves) in ~3 rounds. Which
        fixed point is reached depends on the preceding optimization
        trajectory; the data loss ranks trajectories, so a few restarts with
        the best kept by loss is the robust protocol.

        Per voxel visit, everything is batched: one fused sampling of the
        base slice planes over its rays, one grid_sample for all trial
        rotations x tilts, one FFT chain over (trials x rays).

        Returns
        -------
        dict
            ``n_changed``, ``n_visited``.
        """
        rng = np.random.default_rng(seed)
        dpt = measurements if torch.is_tensor(measurements) else torch.as_tensor(measurements)
        meas_amp = dpt.to(self.device).clamp_min(0).sqrt()
        vox_rays, pos = self._local_ray_map(tilts_deg, scan_shape, scan_step, w_min)
        Nz, Ny, Nx = self.real_shape
        n_col = scan_shape[1]
        geo_by_tilt = [self._tilt_geometry(pos, float(t)) for t in tilts_deg]
        Ns = len(geo_by_tilt[0]["slices"])
        num_pix = self.Psi0.numel()

        def ray_geometry(rays):
            parts = [[] for _ in range(Ns)]
            tw_parts = [[] for _ in range(Ns)]
            t_idx = []
            for ti, p in rays:
                g = geo_by_tilt[ti]
                for s in range(Ns):
                    vidx, tw = g["slices"][s]
                    parts[s].append(vidx[p])
                    tw_parts[s].append(tw[p])
                t_idx.append(ti)
            return ([torch.stack(ps) for ps in parts],
                    [torch.stack(ps) for ps in tw_parts], np.array(t_idx))

        def neighbor_ids(v):
            z, y, x = v // (Ny * Nx), (v // Nx) % Ny, v % Nx
            out = []
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if (dz, dy, dx) == (0, 0, 0):
                            continue
                        md = abs(dz) + abs(dy) + abs(dx)
                        if (neighbors == 6 and md != 1) or (neighbors == 18 and md > 2):
                            continue
                        zz, yy, xx = z + dz, y + dy, x + dx
                        if 0 <= zz < Nz and 0 <= yy < Ny and 0 <= xx < Nx:
                            out.append((zz * Ny + yy) * Nx + xx)
            return out

        if order == "error":
            # visit the worst-fit voxels first: forward residual per ray,
            # averaged over each voxel's intercepting rays
            with torch.no_grad():
                B0, R0, W0 = self.masked_basis().detach(), \
                    self.rotation_matrices().detach(), self.weights.detach()
                ray_err = {}
                for ti, t in enumerate(tilts_deg):
                    psi = self.forward_tilt(pos, float(t), B0, R0, W0)
                    amp = (psi.abs() ** 2).clamp_min(0).sqrt()
                    m = meas_amp[ti].reshape(-1, *self.det_shape)
                    e = ((amp - m) ** 2).mean(dim=(-1, -2))
                    for p in range(pos.shape[0]):
                        ray_err[(ti, p)] = float(e[p])
            vox_err = np.array([np.mean([ray_err[r] for r in rays]) if rays else -1.0
                                for rays in vox_rays])
            visit = np.argsort(vox_err)[::-1]
            # jitter only the misfit voxels: it is an escape/refine mechanism,
            # so a voxel that already fits its rays gains nothing from it. Gate
            # on above-median residual (among voxels with rays) -- this keeps
            # the confusion-case rescue at a fraction of the cost.
            withrays = vox_err[vox_err >= 0]
            err_gate = float(np.median(withrays)) if withrays.size else np.inf
            jitter_ok = vox_err >= err_gate
        else:
            visit = rng.permutation(self.n_voxels)
            jitter_ok = np.ones(self.n_voxels, dtype=bool)

        # skip vacuum voxels: a near-zero-weight voxel contributes ~0 to any of
        # its rays regardless of orientation, so no trial ever beats its
        # incumbent -- sweeping it is pure cost. Keep a voxel only if its
        # strongest basis weight exceeds w_material * (max over voxels), i.e.
        # a RELATIVE threshold (testing > 0 alone would still sweep the many
        # tiny fog weights). Relative, not absolute, because under phase_only
        # the weight scale is a soft gauge -- the learned weights settle at
        # whatever magnitude (e.g. ~0.03, not ~1) trades off against the basis,
        # so a fixed absolute cut would exclude everything. Fall back to the
        # worst ~64 by error before the weight field forms.
        wmax0 = self.weights.detach().abs().reshape(self.n_voxels, -1).max(-1).values
        material = (wmax0 > w_material * wmax0.max().clamp_min(1e-30)).cpu().numpy()
        keep = material[visit]
        if keep.sum() < 64:
            keep[:64] = True                          # early-phase fallback
        visit = visit[keep]

        n_changed = n_visited = 0
        _iter = tqdm(visit, disable=not progress, desc="local_search", unit="vox", leave=False)
        with torch.no_grad():
            for v in _iter:
                rays = vox_rays[v]
                if not rays:
                    continue
                n_visited += 1
                R_all = self.rotation_matrices().detach()
                W_all = self.weights.detach()
                wsum_flat = W_all.reshape(-1, W_all.shape[-1]).sum(-1)
                R_flat = R_all.reshape(-1, 3, 3)
                basis = self.masked_basis().detach()
                basis_sum_c = torch.fft.fftshift(basis.sum(-1), dim=(0, 1, 2))
                w_v = float(wsum_flat[v])
                mat = wsum_flat.abs() > 0.25 * wsum_flat.abs().max()
                w_mean = float(wsum_flat[mat].mean()) if mat.any() else 1.0

                R_v = R_flat[v].cpu().numpy()
                trials = [(R_v, w_v)]
                trials += [(R_flat[u].cpu().numpy(), float(wsum_flat[u]))
                           for u in neighbor_ids(v)]
                bank = self._uniform_rotations(n_rand, seed + int(v))
                trials += [(B, w_mean if w_v < 0.2 * w_mean else w_v) for B in bank]
                # multi-scale jitter of the incumbent (lab-frame perturbation):
                # refine a nearly-right voxel and hop adjacent basins without
                # an explicit search -- only for misfit voxels (jitter_ok), a
                # well-fit voxel gains nothing and would just pay the cost
                if n_jitter and jitter_ok[v]:
                    for si, sig in enumerate(jitter_deg):
                        for P in self._rot_perturbations(sig, n_jitter, seed + 991 * (si + 1) + int(v)):
                            trials.append((P @ R_v, w_v))
                trials += [(R_v, w_mean), (R_v, 0.0)]
                K = len(trials)

                vidx_s, tw_s, t_idx = ray_geometry(rays)
                nray = len(rays)
                tilts_used = sorted(set(t_idx))
                tmap = {ti: j for j, ti in enumerate(tilts_used)}
                U = torch.stack([geo_by_tilt[ti]["u"] for ti in tilts_used])
                V = torch.stack([geo_by_tilt[ti]["v"] for ti in tilts_used])
                ku, kv = geo_by_tilt[0]["ku"], geo_by_tilt[0]["kv"]
                planes_R = self._orientation_planes(basis_sum_c,
                                                    [T[0] for T in trials], U, V, ku, kv)
                ray_t = torch.tensor([tmap[ti] for ti in t_idx])
                trial_planes = planes_R[:, ray_t]               # (K, nray, det, det)

                base_planes, base_wsums, v_tw = [], [], []
                geo_stub = {"u": U[ray_t].repeat_interleave(8, 0),
                            "v": V[ray_t].repeat_interleave(8, 0),
                            "ku": ku, "kv": kv}
                for s in range(Ns):
                    pl = self._transmission_planes_fused(vidx_s[s], tw_s[s],
                                                         geo_stub, basis, R_all, W_all)
                    base_planes.append(pl)
                    base_wsums.append((tw_s[s].real.to(wsum_flat.dtype)
                                       * wsum_flat[vidx_s[s]]).sum(-1))
                    hit = (vidx_s[s] == v) & (tw_s[s].real > 0)
                    v_tw.append(torch.where(hit, tw_s[s].real, 0.0).sum(-1))

                Psi = self.Psi0[None, None].expand(K, nray, -1, -1).reshape(
                    K * nray, *self.det_shape).clone()
                inc = trial_planes[0]
                w_ts = torch.tensor([w for _, w in trials],
                                    dtype=torch.float64)[:, None, None, None]
                for s in range(Ns):
                    tww = v_tw[s][None, :, None, None]
                    SF = (base_planes[s][None] - tww * w_v * inc[None]
                          + tww * w_ts * trial_planes)
                    dc = (base_wsums[s][None, :] - v_tw[s][None, :] * w_v
                          + v_tw[s][None, :] * w_ts[:, :, 0, 0])
                    SF = SF.reshape(K * nray, *self.det_shape).clone()
                    SF[:, 0, 0] = SF[:, 0, 0] + (1.0 - dc.reshape(-1)).to(SF.dtype)
                    t = torch.fft.ifft2(num_pix * SF)
                    t = torch.exp(1j * torch.angle(t))
                    Psi = torch.fft.fft2(torch.fft.ifft2(Psi) * t)
                    if s < Ns - 1:
                        Psi = Psi * self._prop_power(1)
                Psi = Psi * self.antialias_mask
                amp = (Psi.abs() ** 2).clamp_min(0).sqrt().reshape(
                    K, nray, *self.det_shape)
                m = torch.stack([meas_amp[ti, p // n_col, p % n_col]
                                 for ti, p in rays])
                losses = ((amp - m[None]) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()

                best = int(np.argmin(losses))
                if best != 0 and losses[best] < accept * losses[0]:
                    R_b, w_b = trials[best]
                    self.angles.M[v] = torch.as_tensor(R_b, dtype=torch.float32,
                                                       device=self.device)
                    ns = self.weights.shape[-1]
                    self.weights.reshape(-1, ns)[v] = w_b / ns
                    n_changed += 1
        return {"n_changed": n_changed, "n_visited": n_visited}

    def plot_loss(self, figsize: tuple[float, float] = (5.5, 3.4)):
        """Semilog plot of the most recent reconstruction's loss history.

        Parameters
        ----------
        figsize : tuple[float, float], default (5.5, 3.4)
            Figure size in inches.

        Returns
        -------
        fig, ax
            The matplotlib figure and axis.
        """
        import matplotlib.pyplot as plt

        losses = getattr(self, "losses", None)
        if not losses:
            raise RuntimeError("No loss history -- run reconstruct() first.")
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        it = np.arange(len(losses))
        ax.semilogy(it, losses, "-", color="C0", label="loss")
        ax.axhline(self.best_loss, color="C0", lw=1.0, ls="--", label="best")
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean amplitude MSE", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax.xaxis.get_major_locator().set_params(integer=True)

        return {'reconstructed_sf': self.sf_learned.detach().cpu(),
        'training_losses': total_loss,
        'validation_losses': avg_val_loss,}
