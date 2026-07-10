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

        The model stores structure factors in each voxel's body frame; this
        returns the weighted sum of structures rotated by the voxel's
        orientation (trilinear, the same sampling the forward model uses for
        its tilt planes). Reconstructions with different orientation gauges
        can then be compared on a common grid.

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

    @classmethod
    def make_au_basis(
        cls,
        k_shape: tuple[int, int, int],
        k_sampling: tuple[float, float, float],
        a_Au: float = 4.08,
        phase_scale: float = 0.10,
        hkl_amplitudes=((1, 1, 1, 0.10), (2, 0, 0, 0.06), (2, 2, 0, 0.03),
                        (3, 1, 1, 0.05), (2, 2, 2, 0.04)),
        dtype=torch.complex128,
    ) -> torch.Tensor:
        """Canonical (unrotated) Au structure factor in ``[kz, ky, kx]`` order.

        Bragg peaks carry small purely-imaginary amplitudes (linearized
        ``exp(i*phi)`` phase grating); the origin is the vacuum baseline (1).
        The per-voxel SO(3) orientation supplies the actual grain rotation.
        """
        Nkz, Nky, Nkx = (int(n) for n in k_shape)
        dkz, dky, dkx = (float(s) for s in k_sampling)
        sf = torch.zeros(Nkz, Nky, Nkx, dtype=dtype)
        sf[0, 0, 0] = 1.0
        amp_scale = 1j * phase_scale
        for row in hkl_amplitudes:
            h, k, l, a = row
            amp = amp_scale * float(a)
            vec_set = sorted({
                tuple(s * p for s, p in zip(sign, perm))
                for perm in set(permutations((int(h), int(k), int(l))))
                for sign in product((-1, 1), repeat=3)
            })
            for vec in vec_set:
                # peak in (kx, ky, kz) physical -> nearest [kz, ky, kx] voxel.
                # Each Bragg reflection is a single delta spike: spreading it
                # over a trilinear cluster distorts the diffracted disks (the
                # transmission peak becomes a multi-pixel blob) and makes the
                # structure factor read as several bright voxels per peak.
                kx, ky, kz = (v / a_Au for v in vec)
                iz = int(round(kz / dkz)) % Nkz
                iy = int(round(ky / dky)) % Nky
                ix = int(round(kx / dkx)) % Nkx
                sf[iz, iy, ix] += amp
        return sf

    def make_probe(self, probe_k_max: float, normalize: bool = True) -> torch.Tensor:
        """Aperture probe with a sub-pixel anti-aliased edge, [row, col] = [ky, kx].

        The edge ramps linearly over one k-pixel (partial pixel coverage), so
        the direct and diffracted disks render as circles on the coarse
        detector grid instead of chunky hard-threshold polygons.
        """
        self.probe_k_max = float(probe_k_max)
        kv, ku = torch.meshgrid(self.det_kv, self.det_ku, indexing="ij")
        k_rad = torch.sqrt(kv ** 2 + ku ** 2)
        dk_pix = min(self.k_sampling[1], self.k_sampling[2])
        aperture = torch.clamp((probe_k_max - k_rad) / dk_pix + 0.5, 0.0, 1.0).to(torch.complex128)
        if normalize:
            aperture = aperture / torch.sqrt((aperture.abs() ** 2).sum())
        self.Psi0 = aperture.to(self.device)
        return self.Psi0

    def make_propagator(self, antialias_fraction: float = 0.9, antialias_softness: float = 0.05):
        """Fresnel propagator exp(-i pi lambda dz |k|^2) with a soft circular
        anti-alias mask folded in.  dz = real-space z sampling."""
        dz = self.real_sampling[0]
        kv, ku = torch.meshgrid(self.det_kv, self.det_ku, indexing="ij")
        k2 = kv ** 2 + ku ** 2
        bare = torch.exp(-1j * torch.pi * self.wavelength * k2 * dz)
        # soft circular anti-alias mask
        k_rad = torch.sqrt(k2)
        k_nyq = min(self.det_kv.abs().max().item(), self.det_ku.abs().max().item())
        cutoff = antialias_fraction * k_nyq
        width = max(antialias_softness * k_nyq, 1e-12)
        mask = 0.5 * (1.0 - torch.cos(torch.pi * torch.clip((cutoff + width - k_rad) / width, 0.0, 1.0)))
        self.antialias_mask = mask.to(self.device)          # circular k envelope
        self._bare_prop = bare.to(self.device)              # unit-modulus Fresnel factor
        self._prop_pow_cache = {}
        self.prop = (bare * self.antialias_mask)
        self.prop_distance = dz
        self._visible_k_max = float(cutoff + width)         # data carries no intensity beyond
        self._build_sphere_mask()                            # clip basis support to it
        return self.prop

    def tilt_axes(self, tilt_x_deg: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lab-frame (u, v, w) unit vectors for an X-axis tilt, in [z, y, x] comps.

        u = detector col (x), v = detector row (y), w = beam (nominally +z).
        Tilt about x rotates the (z, y) plane.
        """
        th = math.radians(float(tilt_x_deg))
        c, s = math.cos(th), math.sin(th)
        dev = self.device
        w = torch.tensor([c, s, 0.0], dtype=torch.float32, device=dev)   # beam
        u = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=dev)  # col = x
        v = torch.tensor([-s, c, 0.0], dtype=torch.float32, device=dev)   # row = y
        return u, v, w

    def _ray_voxel_weights(self, origin_zyx: torch.Tensor, tilt_x_deg: float) -> torch.Tensor:
        """Total trilinear weight this ray deposits in each voxel, (n_voxels,).

        Geometry only (no basis sampling) -- used to distribute per-DP residual
        onto voxels for the bad-voxel error metric. Fixed across optimization.
        """
        Nz, Ny, Nx = self.real_shape
        _, _, w = self.tilt_axes(tilt_x_deg)
        pts = self.ray_samples(origin_zyx, w)
        wv = torch.zeros(self.n_voxels, dtype=torch.float64, device=self.device)
        for s in range(pts.shape[0]):
            p = pts[s]
            base = torch.floor(p).to(torch.int64)
            frac = p - base.to(p.dtype)
            for dz in range(2):
                iz = int(base[0]) + dz; wz = float(frac[0]) if dz else 1.0 - float(frac[0])
                for dy in range(2):
                    iy = int(base[1]) + dy; wy = float(frac[1]) if dy else 1.0 - float(frac[1])
                    for dx in range(2):
                        ix = int(base[2]) + dx; wx = float(frac[2]) if dx else 1.0 - float(frac[2])
                        tw = wz * wy * wx
                        if tw > 0 and (0 <= iz < Nz) and (0 <= iy < Ny) and (0 <= ix < Nx):
                            wv[(iz * Ny + iy) * Nx + ix] += tw
        return wv

    def ray_samples(self, origin_zyx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Index-space sample points along a ray through the slab z-extent.

        origin_zyx: (3,) probe position in index space [z, y, x].
        Returns (Ns, 3) points, one per traversed z-layer (beam steps by 1 in z).
        """
        Nz = self.real_shape[0]
        dev = self.device
        r0 = origin_zyx.to(dev).to(torch.float32)
        # dr scaled so the z-component advances by exactly one voxel per step
        wz = w[0]
        if torch.abs(wz) < 1e-8:
            return r0[None, :]
        dr = w / wz                                   # step with dz = 1
        # span z = 0 .. Nz-1
        t0 = (0.0 - r0[0]) / dr[0]
        t1 = ((Nz - 1) - r0[0]) / dr[0]
        t_lo, t_hi = (t0, t1) if t0 <= t1 else (t1, t0)
        n_lo = int(math.ceil(t_lo.item() - 1e-6))
        n_hi = int(math.floor(t_hi.item() + 1e-6))
        steps = torch.arange(n_lo, n_hi + 1, dtype=torch.float32, device=dev)[:, None]
        return r0[None, :] + steps * dr[None, :]

    def _transmission_plane(self, point_zyx: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                            basis: torch.Tensor, R_all: torch.Tensor, W_all: torch.Tensor) -> torch.Tensor:
        """Assemble the 2D transmission SF at one continuous ray point.

        The 2x2x2 trilinear cluster is handled in one batched ``grid_sample``:
        each in-bounds voxel's orientation is composed with the tilt to give its
        sampling plane through the bases, the resulting per-voxel transmission
        planes are combined with the trilinear weights (in transmission space,
        never on the angles), and the vacuum delta ``(1 - w_eff)`` is added at
        the origin. Out-of-slab neighbours contribute vacuum.

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
        Nz, Ny, Nx = self.real_shape
        Nkz, Nky, Nkx = self.k_shape
        Nw = basis.shape[-1]
        dev = self.device
        p = point_zyx.to(dev)
        base = torch.floor(p).to(torch.int64)
        frac = p - base.to(p.dtype)                          # (3,)

        kv, ku = torch.meshgrid(self.det_kv, self.det_ku, indexing="ij")  # (R,C)
        det_row, det_col = kv.shape
        dk = torch.tensor(self.k_sampling, dtype=torch.float32, device=dev)

        # 2x2x2 cluster corners, trilinear weights, in-bounds mask
        offs = torch.tensor([[dz, dy, dx] for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)],
                            device=dev)                        # (8,3)
        idxs = base[None, :] + offs                            # (8,3) [iz,iy,ix]
        fw = torch.stack([torch.where(offs[:, a] == 1, frac[a], 1.0 - frac[a]) for a in range(3)], dim=1)
        tw8 = fw.prod(dim=1)                                   # (8,)
        inb = ((idxs[:, 0] >= 0) & (idxs[:, 0] < Nz) & (idxs[:, 1] >= 0) & (idxs[:, 1] < Ny)
               & (idxs[:, 2] >= 0) & (idxs[:, 2] < Nx))
        valid = inb & (tw8 > 0)

        acc = torch.zeros(det_row, det_col, dtype=torch.complex64, device=dev)
        w_eff = torch.zeros((), dtype=torch.float64, device=dev)
        if bool(valid.any()):
            vsel = idxs[valid]                                 # (nv,3)
            nv = vsel.shape[0]
            tw = tw8[valid].to(torch.complex64)                # (nv,)
            vidx = (vsel[:, 0] * Ny + vsel[:, 1]) * Nx + vsel[:, 2]
            R = R_all.reshape(-1, 3, 3)[vidx]                  # (nv,3,3) body->lab
            wv = W_all.reshape(-1, Nw)[vidx].to(torch.complex64)  # (nv,Nw)
            w_eff = (tw8[valid].to(torch.float64) * W_all.reshape(-1, Nw)[vidx].sum(-1)).sum()
            # rotate the detector (u,v) axes into each voxel's body frame: R^T @ axis
            u_b = torch.einsum("vij,i->vj", R, u)              # (nv,3) [kz,ky,kx]
            v_b = torch.einsum("vij,i->vj", R, v)
            # plane points (nv,R,C,3) in A^-1; basis is fftfreq-ordered so wrap
            # k/dk into [0, N) then normalise to [-1, 1] (grid last dim reversed)
            kxyz = (ku[None, ..., None] * u_b[:, None, None, :]
                    + kv[None, ..., None] * v_b[:, None, None, :])   # (nv,R,C,3)
            # centered (fftshift) sampling space -- see _transmission_planes_fused
            ctr = torch.tensor([Nkz // 2, Nky // 2, Nkx // 2], dtype=torch.float32, device=dev)
            c = kxyz / dk + ctr                                # (nv,R,C,3) [cz,cy,cx]
            grid = torch.stack((
                2.0 * c[..., 2] / (Nkx - 1.0) - 1.0,
                2.0 * c[..., 1] / (Nky - 1.0) - 1.0,
                2.0 * c[..., 0] / (Nkz - 1.0) - 1.0,
            ), dim=-1)[:, None].to(torch.float32)              # (nv,1,R,C,3)
            basis_c = torch.fft.fftshift(basis, dim=(0, 1, 2))
            bre = basis_c.real.permute(3, 0, 1, 2)[None].expand(nv, -1, -1, -1, -1).to(torch.float32)
            bim = basis_c.imag.permute(3, 0, 1, 2)[None].expand(nv, -1, -1, -1, -1).to(torch.float32)
            sre = F.grid_sample(bre, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            sim = F.grid_sample(bim, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            sampled = (sre + 1j * sim).squeeze(2)              # (nv,Nw,R,C)
            t_vox = (wv[:, :, None, None] * sampled).sum(1)    # (nv,R,C)
            acc = (tw[:, None, None] * t_vox).sum(0)           # (R,C)
        # vacuum delta carries the material complement: weight 0 -> pure
        # vacuum, weight 1 -> pure basis transmission (its DC included)
        acc[0, 0] = acc[0, 0] + (1.0 - w_eff).to(acc.dtype)
        return acc

    def forward_ray(self, origin_zyx: torch.Tensor, tilt_x_deg: float,
                    basis: torch.Tensor, R_all: torch.Tensor, W_all: torch.Tensor,
                    phase_only: bool = True) -> torch.Tensor:
        """Multislice exit wave for one probe at one tilt."""
        u, v, w = self.tilt_axes(tilt_x_deg)
        pts = self.ray_samples(origin_zyx, w)                 # (Ns,3)
        Psi = self.Psi0.clone()
        num_pix = Psi.numel()
        Ns = pts.shape[0]
        for s in range(Ns):
            SF = self._transmission_plane(pts[s], u, v, basis, R_all, W_all)
            T2d = num_pix * SF
            t_real = torch.fft.ifft2(T2d)
            if phase_only:
                t_real = torch.exp(1j * torch.angle(t_real))
            Psi = torch.fft.fft2(torch.fft.ifft2(Psi) * t_real)
            if s < Ns - 1:
                Psi = Psi * self.prop
        # Band-limit the exit wave with the same circular k-envelope. The last
        # slice is not propagated, so its phase-grating harmonics beyond the
        # antialias cutoff would otherwise leak into the detector corners.
        Psi = Psi * self.antialias_mask
        return Psi

    def _transmission_planes_batch(self, points: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                                   basis: torch.Tensor, R_all: torch.Tensor, W_all: torch.Tensor) -> torch.Tensor:
        """Transmission SF for a batch of ray points (one per probe), at one slice.

        Same construction as :meth:`_transmission_plane` but vectorised over the
        ``P`` probe points: loops the 8 cluster corners (cheap), each a single
        ``grid_sample`` batched over all probes.

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

        acc = torch.zeros(P, det_row, det_col, dtype=torch.complex64, device=dev)
        w_eff = torch.zeros(P, dtype=torch.float64, device=dev)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    off = torch.tensor([dz, dy, dx], device=dev)
                    idx = base + off                            # (P,3)
                    tw = (torch.where(off[0] == 1, frac[:, 0], 1.0 - frac[:, 0])
                          * torch.where(off[1] == 1, frac[:, 1], 1.0 - frac[:, 1])
                          * torch.where(off[2] == 1, frac[:, 2], 1.0 - frac[:, 2]))    # (P,)
                    inb = ((idx >= lo) & (idx <= hi)).all(dim=1)   # (P,)
                    tw = (tw * inb).to(torch.complex64)          # zero out-of-slab corners
                    idxc = torch.minimum(torch.maximum(idx, lo), hi)   # clamp for the gather
                    vidx = (idxc[:, 0] * Ny + idxc[:, 1]) * Nx + idxc[:, 2]
                    R = R_all.reshape(-1, 3, 3)[vidx]            # (P,3,3)
                    wv = W_all.reshape(-1, Nw)[vidx].to(torch.complex64)   # (P,Nw)
                    u_b = torch.einsum("pij,i->pj", R, u)        # (P,3) = R^T u
                    v_b = torch.einsum("pij,i->pj", R, v)
                    kxyz = (ku[None, ..., None] * u_b[:, None, None, :]
                            + kv[None, ..., None] * v_b[:, None, None, :])   # (P,R,C,3)
                    # centered (fftshift) sampling space -- see _transmission_planes_fused
                    ctr = torch.tensor([Nkz // 2, Nky // 2, Nkx // 2], dtype=torch.float32, device=dev)
                    c = kxyz / dk + ctr
                    grid = torch.stack((
                        2.0 * c[..., 2] / (Nkx - 1.0) - 1.0,
                        2.0 * c[..., 1] / (Nky - 1.0) - 1.0,
                        2.0 * c[..., 0] / (Nkz - 1.0) - 1.0,
                    ), dim=-1)[:, None].to(torch.float32)        # (P,1,R,C,3)
                    bre = bre_full[None].expand(P, -1, -1, -1, -1).to(torch.float32)
                    bim = bim_full[None].expand(P, -1, -1, -1, -1).to(torch.float32)
                    sre = F.grid_sample(bre, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                    sim = F.grid_sample(bim, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                    sampled = (sre + 1j * sim).squeeze(2)        # (P,Nw,R,C)
                    t_vox = (wv[:, :, None, None] * sampled).sum(1)   # (P,R,C)
                    acc = acc + tw[:, None, None] * t_vox
                    w_eff = w_eff + tw.real.to(w_eff.dtype) * W_all.reshape(-1, Nw)[vidx].sum(-1)
        # vacuum delta carries the material complement per probe
        acc[:, 0, 0] = acc[:, 0, 0] + (1.0 - w_eff).to(acc.dtype)
        return acc

    def _tilt_geometry(self, origins: torch.Tensor, tilt_x_deg: float) -> dict:
        """Fixed sampling geometry for one tilt, cached across iterations.

        For every slice along the rays: the 2x2x2 cluster corners' flat voxel
        indices ``vidx (P, 8)`` (clamped in-bounds) and trilinear weights
        ``tw (P, 8)`` (zeroed for out-of-slab corners). Pure geometry -- it
        depends only on the scan/tilt, never on the parameters -- so computing
        it once removes the largest per-iteration tensor-op overhead.
        """
        key = (round(float(tilt_x_deg), 6), origins.shape[0],
               hash(origins.cpu().numpy().tobytes()))
        cache = getattr(self, "_geo_cache", None)
        if cache is None:
            cache = self._geo_cache = {}
        if key in cache:
            return cache[key]
        Nz, Ny, Nx = self.real_shape
        dev = self.device
        u, v, w = self.tilt_axes(tilt_x_deg)
        pts = torch.stack([self.ray_samples(o, w) for o in origins])   # (P, Ns, 3)
        P, Ns = pts.shape[0], pts.shape[1]
        offs = torch.tensor([[dz, dy, dx] for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)],
                            device=dev)                                 # (8, 3)
        lo = torch.zeros(3, dtype=torch.long, device=dev)
        hi = torch.tensor([Nz - 1, Ny - 1, Nx - 1], device=dev)
        slices = []
        for s in range(Ns):
            p = pts[:, s]
            base = torch.floor(p).to(torch.int64)                      # (P, 3)
            frac = (p - base.to(p.dtype))                               # (P, 3)
            idx = base[:, None, :] + offs[None, :, :]                   # (P, 8, 3)
            fw = torch.where(offs[None, :, :] == 1, frac[:, None, :], 1.0 - frac[:, None, :])
            tw = fw.prod(dim=-1)                                        # (P, 8)
            inb = ((idx >= lo) & (idx <= hi)).all(dim=-1)               # (P, 8)
            tw = tw * inb
            idxc = torch.minimum(torch.maximum(idx, lo), hi)
            vidx = (idxc[..., 0] * Ny + idxc[..., 1]) * Nx + idxc[..., 2]  # (P, 8)
            slices.append((vidx, tw.to(torch.complex64)))
        kv, ku = torch.meshgrid(self.det_kv, self.det_ku, indexing="ij")
        geo = {"u": u, "v": v, "slices": slices, "ku": ku, "kv": kv}
        cache[key] = geo
        return geo

    def _shrink_beam_zone(self, boost: float = 5.0) -> torch.Tensor:
        """L1-threshold scale for the direct-beam overlap zone of k space.

        A transmission harmonic within twice the probe aperture radius of the
        origin diffracts a disk that overlaps the direct beam, so the data
        barely distinguishes it from a beam-amplitude wiggle -- and every tilt
        shares this zone, making it the cheapest place for the fit to park
        compensation fog (a bright segment along the tilt axle). Boosting the
        shrink threshold there during a FINAL polish stage clears the fog and
        forces its explanatory load back onto the Bragg peaks; boosting from
        the start stalls convergence (the zone is scaffolding for the beam
        residual early on). Instrument geometry only.

        Returns
        -------
        torch.Tensor
            ``[N_kz, N_ky, N_kx, 1]`` threshold scale: ``boost`` inside
            ``|k| < 2 * probe_k_max``, 1 outside.
        """
        cached = getattr(self, "_beam_zone", None)
        if cached is not None and cached[0] == boost:
            return cached[1]
        ks = [torch.fft.fftfreq(n, d=1.0 / (n * s), device=self.device)
              for n, s in zip(self.k_shape, self.k_sampling)]
        KZ, KY, KX = torch.meshgrid(*ks, indexing="ij")
        k_rad = torch.sqrt(KZ**2 + KY**2 + KX**2)
        scale = torch.where(k_rad < 2.0 * self.probe_k_max, boost, 1.0)[..., None]
        self._beam_zone = (boost, scale)
        return scale

    def _prop_power(self, g: int) -> torch.Tensor:
        """Fresnel propagator over ``g`` slice thicknesses (antialias applied once)."""
        cache = getattr(self, "_prop_pow_cache", None)
        if cache is None:
            cache = self._prop_pow_cache = {}
        if g not in cache:
            if not hasattr(self, "_bare_prop"):
                self.make_propagator()          # object predates the refactor
            cache[g] = (self._bare_prop ** g) * self.antialias_mask
        return cache[g]

    def _transmission_planes_fused(self, vidx: torch.Tensor, tw: torch.Tensor,
                                   geo: dict, basis: torch.Tensor,
                                   R_all: torch.Tensor, W_all: torch.Tensor) -> torch.Tensor:
        """Transmission SF deviation for all probes at one slice, one fused call.

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
        groups = [list(range(s, min(s + max(1, superslice), Ns)))
                  for s in range(0, Ns, max(1, superslice))]
        for gi, grp in enumerate(groups):
            SF, Wg = None, None
            for s in grp:
                vidx, tw = slices[s]
                sf_s = self._transmission_planes_fused(vidx, tw, geo, basis, R_all, W_all)
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
               "ku": geos[0]["ku"], "kv": geos[0]["kv"]}
        cache[key] = geo
        return geo

    def forward_tilts(self, origins: torch.Tensor, tilts, basis: torch.Tensor,
                      R_all: torch.Tensor, W_all: torch.Tensor,
                      phase_only: bool = True, superslice: int = 1) -> torch.Tensor:
        """Exit waves for a GROUP of tilts in one batched pass.

        Identical physics to calling :meth:`forward_tilt` per tilt; all tilts'
        probes travel together as one ``(T*P, det, det)`` wave, so one autograd
        graph (and one backward) covers the whole group.

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
        groups = [list(range(s, min(s + max(1, superslice), Ns)))
                  for s in range(0, Ns, max(1, superslice))]
        for gi, grp in enumerate(groups):
            SF, Wg = None, None
            for s in grp:
                vidx, tw = slices[s]
                sf_s = self._transmission_planes_fused(vidx, tw, geo, basis, R_all, W_all)
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

    def scan_positions(
        self,
        scan_shape: tuple[int, int],                 # (n_row, n_col)
        scan_step: float | tuple[float, float] = 1.0,  # voxels, [row, col]
        scan_origin: tuple[float, float] | None = None,  # [row, col] index; default center
    ) -> torch.Tensor:
        """Lab-frame probe positions, (n_row, n_col, 3) in index space [z, y, x].

        The raster lives in the fixed lab (row=y, col=x) plane at mid-thickness
        z; the sample tilt only changes the beam direction, not the raster.
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
        basis: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        R_all: torch.Tensor | None = None,
        tilts_deg=None,
        scan_shape: tuple[int, int] | None = None,
        scan_step: float | tuple[float, float] | None = None,
        scan_origin=None,
        phase_only: bool = True,
        superslice: int = 1,
        progress: bool = True,
    ) -> torch.Tensor:
        """Forward tilt series -> (n_tilt, n_row, n_col, det_row, det_col) intensities.

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

    @classmethod
    def _miso_deg(cls, dR: np.ndarray) -> float:
        """Cubic-symmetry-reduced misorientation angle (deg) of a relative rotation."""
        if not hasattr(cls, "_ops_cache"):
            cls._ops_cache = cls._cubic_ops()
        tr = np.einsum("sij,ji->s", cls._ops_cache, dR)
        c = np.clip((tr.max() - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(c)))

    def _spot_geometry(self, cands: np.ndarray, g: np.ndarray, tilts_deg, sigma_e: float):
        """Per-tilt predicted-spot geometry with smooth excitation weighting.

        For every candidate rotation and template reflection: a Gaussian
        excitation-error weight exp(-(g.w)^2 / 2 sigma_e^2) and a sub-pixel
        (bilinear) detector position, so the indexing score varies smoothly
        with orientation (a binary on/off tolerance would quantize it at
        ~sigma_e/|g|, blinding sub-degree refinement).
        """
        det_r, det_c = self.det_shape
        dkr, dkc = self.k_sampling[1], self.k_sampling[2]
        cr, cc = det_r // 2, det_c // 2
        geo = []
        for tilt in tilts_deg:
            u, v, w = (t.cpu().numpy() for t in self.tilt_axes(float(tilt)))
            g_lab = np.einsum("cij,mj->cmi", cands, g)
            exc = np.exp(-0.5 * (g_lab @ w) ** 2 / sigma_e ** 2)
            col = (g_lab @ u) / dkc + cc
            row = (g_lab @ v) / dkr + cr
            inb = (col >= 0) & (col <= det_c - 1) & (row >= 0) & (row <= det_r - 1)
            exc = exc * inb
            c0 = np.clip(np.floor(col).astype(int), 0, det_c - 2)
            r0 = np.clip(np.floor(row).astype(int), 0, det_r - 2)
            fc = np.clip(col - c0, 0.0, 1.0)
            fr = np.clip(row - r0, 0.0, 1.0)
            idx = np.stack([r0 * det_c + c0, r0 * det_c + c0 + 1,
                            (r0 + 1) * det_c + c0, (r0 + 1) * det_c + c0 + 1], axis=-1)
            wbl = np.stack([(1 - fr) * (1 - fc), (1 - fr) * fc,
                            fr * (1 - fc), fr * fc], axis=-1)
            geo.append((idx, wbl, exc, exc.sum(axis=1)))
        return geo

    @staticmethod
    def _uniform_rotations(n: int, seed: int) -> np.ndarray:
        """(n, 3, 3) uniform SO(3) samples (Shoemake quaternions)."""
        rng = np.random.default_rng(seed)
        u = rng.random((n, 3))
        q = np.stack([np.sqrt(1 - u[:, 0]) * np.sin(2 * np.pi * u[:, 1]),
                      np.sqrt(1 - u[:, 0]) * np.cos(2 * np.pi * u[:, 1]),
                      np.sqrt(u[:, 0]) * np.sin(2 * np.pi * u[:, 2]),
                      np.sqrt(u[:, 0]) * np.cos(2 * np.pi * u[:, 2])], axis=1)
        x, y, z, w = q.T
        return np.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ], axis=-1).reshape(n, 3, 3)

    @staticmethod
    def _rot_perturbations(sigma_deg: float, n: int, seed: int) -> np.ndarray:
        """(n, 3, 3) small random rotations with angle scale sigma_deg."""
        rng = np.random.default_rng(seed)
        ax = rng.normal(size=(n, 3))
        ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        th = np.radians(sigma_deg) * rng.normal(size=n)
        K = np.zeros((n, 3, 3))
        K[:, 0, 1], K[:, 0, 2] = -ax[:, 2], ax[:, 1]
        K[:, 1, 0], K[:, 1, 2] = ax[:, 2], -ax[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -ax[:, 1], ax[:, 0]
        I = np.eye(3)[None]
        return I + np.sin(th)[:, None, None] * K + (1 - np.cos(th))[:, None, None] * (K @ K)

    def index_orientations(
        self,
        measurements: torch.Tensor,
        tilts_deg,
        scan_shape: tuple[int, int] | None = None,
        scan_step: float | tuple[float, float] = 1.0,
        scan_origin=None,
        template: np.ndarray | torch.Tensor | None = None,
        template_weights: np.ndarray | None = None,
        n_template: int = 40,
        sigma_e: float | None = None,
        n_cand: int = 4000,
        cluster_deg: float = 6.0,
        peel_rounds: int = 2,
        icm_lambda: float = 0.15,
        seed: int | None = None,
    ) -> dict:
        """Seed the per-voxel orientations by spot indexing the measurements.

        EBSD-style two-stage indexing:

        1. **Modes**: every probe column (fixed scan position, all tilts) is
           indexed independently -- a column passes through only 1-2 grains, so
           its tilt-aggregated score is clean. Column winners are found by a
           coarse SO(3) bank + hierarchical refinement (with per-column spot
           peeling to catch a second grain), then clustered by cubic-reduced
           misorientation into candidate grain orientations ("modes"), each
           re-refined on the sum of its member columns.
        2. **Assignment**: each voxel is classified among the few modes using
           the ray->voxel weight matrix with explaining-away normalization (a
           pattern dominated by one grain is not counted as evidence for the
           others), followed by a few Potts/ICM sweeps (grains are contiguous).

        The reflection ``template`` defaults to the local maxima of the model's
        own current basis, so a short joint reconstruction to bootstrap the
        basis makes the procedure fully self-contained.

        Returns
        -------
        dict
            ``angles``: (n_voxels, 3, 3) seeded rotations (torch);
            ``modes``: (n_modes, 3, 3) grain orientations, strongest first;
            ``labels``: (n_voxels,) mode index per voxel.
        """
        from scipy.ndimage import maximum_filter

        seed = self.seed + 3 if seed is None else int(seed)
        dp = measurements.detach().cpu().numpy()
        n_tilt = dp.shape[0]
        if scan_shape is None:
            scan_shape = (dp.shape[1], dp.shape[2])
        n_row, n_col = scan_shape
        P = n_row * n_col
        det_r, det_c = self.det_shape
        dkr, dkc = self.k_sampling[1], self.k_sampling[2]
        if sigma_e is None:
            sigma_e = 0.6 * min(dkr, dkc)

        # --- reflection template ------------------------------------------
        if template is None:
            B = np.fft.fftshift(np.abs(self.masked_basis().detach().cpu().numpy()).sum(-1))
            czi, cyi, cxi = (s // 2 for s in self.k_shape)
            B[czi, cyi, cxi] = 0.0
            peaks = (B == maximum_filter(B, size=3)) & (B > 0.05 * B.max())
            zz, yy, xx = np.where(peaks)
            vals = B[zz, yy, xx]
            order = np.argsort(-vals)[:n_template]
            dkz = self.k_sampling[0]
            g = np.stack([(zz - czi) * dkz, (yy - cyi) * dkr, (xx - cxi) * dkc], axis=1)[order]
            gw = vals[order]
        else:
            g = np.asarray(template, dtype=np.float64)
            gw = (np.ones(len(g)) if template_weights is None
                  else np.asarray(template_weights, dtype=np.float64))

        # --- per-column indexing ------------------------------------------
        base = self._uniform_rotations(n_cand, seed)
        base_geo = self._spot_geometry(base, g, tilts_deg, sigma_e)
        cols = dp.transpose(1, 2, 0, 3, 4).reshape(P, n_tilt, det_r, det_c)
        cols = np.fft.fftshift(cols, axes=(-2, -1)).reshape(P, n_tilt, -1)

        def col_scores(col_work, geo):
            s = 0.0
            for ti, (idx, wbl, exc, n_eff) in enumerate(geo):
                vals = col_work[ti][idx.reshape(-1)].reshape(*idx.shape)
                spot = (vals * wbl).sum(axis=-1)
                s = s + (spot * exc * gw[None, :]).sum(axis=1) / np.maximum(n_eff, 1e-9)
            return s

        winners, strengths = [], []
        for p in range(P):
            work = cols[p].copy()
            for r in range(peel_rounds):
                s = col_scores(work, base_geo)
                R = base[int(s.argmax())]
                for sigma in (6.0, 2.0, 0.7):
                    pert = self._rot_perturbations(sigma, 120, seed + 13 * p + r + int(sigma * 10))
                    bank = np.einsum("ij,pjk->pik", R, pert)
                    geo = self._spot_geometry(bank, g, tilts_deg, sigma_e)
                    R = bank[int(col_scores(work, geo).argmax())]
                strength = float(col_scores(work, self._spot_geometry(R[None], g, tilts_deg, sigma_e))[0])
                winners.append(R)
                strengths.append(strength)
                geo1 = self._spot_geometry(R[None], g, tilts_deg, sigma_e)
                for ti, (idx, wbl, exc, _) in enumerate(geo1):
                    img = work[ti].reshape(det_r, det_c)
                    for mi in np.where(exc[0] > 0.3)[0]:
                        r0, c0 = divmod(int(idx[0, mi, 0]), det_c)
                        img[max(0, r0 - 2):r0 + 4, max(0, c0 - 2):c0 + 4] = 0.0

        # --- cluster winners into modes ------------------------------------
        order = np.argsort(-np.asarray(strengths))
        modes, mode_strength = [], []
        for i in order:
            R, s = winners[i], strengths[i]
            if s <= 0:
                continue
            placed = False
            for mi, Rm in enumerate(modes):
                if self._miso_deg(Rm.T @ R) < cluster_deg:
                    mode_strength[mi] += s
                    placed = True
                    break
            if not placed:
                modes.append(R)
                mode_strength.append(s)
        order = np.argsort(-np.asarray(mode_strength))
        modes = np.stack([modes[i] for i in order])
        mode_strength = np.asarray(mode_strength)[order]
        keep = mode_strength > 0.15 * mode_strength[0]
        modes, mode_strength = modes[keep], mode_strength[keep]

        # re-refine each mode on the sum of its member columns
        refined = []
        for mi, Rm in enumerate(modes):
            members = sorted({i // peel_rounds for i, R in enumerate(winners)
                              if strengths[i] > 0 and self._miso_deg(Rm.T @ R) < cluster_deg})
            stack = cols[members or range(P)].sum(axis=0)
            R = Rm
            for sigma in (2.0, 0.7, 0.25):
                pert = self._rot_perturbations(sigma, 150, seed + 29 * mi + int(sigma * 10))
                bank = np.einsum("ij,pjk->pik", R, pert)
                geo = self._spot_geometry(bank, g, tilts_deg, sigma_e)
                R = bank[int(col_scores(stack, geo).argmax())]
            refined.append(R)
        modes = np.stack(refined)

        # --- per-voxel assignment ------------------------------------------
        pos = self.scan_positions(scan_shape, scan_step, scan_origin)
        jobs = [(ti, j, i) for ti in range(n_tilt) for j in range(n_row) for i in range(n_col)]
        Wmat = torch.stack([self._ray_voxel_weights(pos[j, i], float(tilts_deg[ti]))
                            for (ti, j, i) in jobs]).cpu().numpy()
        dps_flat = np.stack([
            np.fft.fftshift(dp[ti].reshape(P, det_r, det_c), axes=(-2, -1)).reshape(P, -1)
            for ti in range(n_tilt)
        ]).reshape(n_tilt * P, -1)
        geo = self._spot_geometry(modes, g, tilts_deg, sigma_e)
        s_dp = np.zeros((n_tilt * P, len(modes)))
        for ti, (idx, wbl, exc, n_eff) in enumerate(geo):
            dps = dps_flat[ti * P:(ti + 1) * P]
            vals = dps[:, idx.reshape(-1)].reshape(P, *idx.shape)
            spot = (vals * wbl[None]).sum(axis=-1)
            s = (spot * (exc * gw[None, :])[None]).sum(axis=2)
            s = np.where(n_eff[None, :] >= 1.5, s / np.maximum(n_eff[None, :], 1e-9), 0.0)
            s_dp[ti * P:(ti + 1) * P] = s
        # explaining-away: a DP dominated by one grain is weak evidence for others
        tot = s_dp.sum(axis=1, keepdims=True)
        s_rel = s_dp / (tot + 1e-30) * np.sqrt(np.maximum(tot, 0.0))
        vox_score = Wmat.T @ s_rel

        # Potts / ICM smoothing (grains are contiguous)
        Nz, Ny, Nx = self.real_shape
        norm = vox_score / (vox_score.max() + 1e-30)
        labels = norm.argmax(axis=1).reshape(Nz, Ny, Nx)
        for _ in range(3):
            for z in range(Nz):
                for y in range(Ny):
                    for x in range(Nx):
                        votes = np.zeros(len(modes))
                        for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                           (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                            z2, y2, x2 = z + dz, y + dy, x + dx
                            if 0 <= z2 < Nz and 0 <= y2 < Ny and 0 <= x2 < Nx:
                                votes[labels[z2, y2, x2]] += 1
                        labels[z, y, x] = int(np.argmax(
                            norm[(z * Ny + y) * Nx + x] + icm_lambda * votes / 6.0))
        labels = labels.flatten()
        R_seed = torch.tensor(modes[labels], dtype=torch.float32)
        return {"angles": R_seed, "modes": modes, "labels": labels}

    def init_from_spots(
        self,
        measurements: torch.Tensor,
        tilts_deg,
        scan_shape: tuple[int, int] | None = None,
        scan_step: float | tuple[float, float] = 1.0,
        scan_origin=None,
        spot_threshold: float = 0.15,
        ray_weight_min: float = 0.2,
        basis_scale: float = 0.01,
        beam_radius: float | None = None,
    ) -> None:
        """Directly initialise the weights and basis from the measured spots.

        Uses the model's *current* per-voxel orientations (typically seeded by
        :meth:`index_orientations`), so together the two methods give a full
        direct (non-iterative) estimate of every parameter:

        * **weights** <- the virtual dark-field map: each pattern's off-beam
          (diffracted) energy distributed onto the voxels its ray samples.
        * **basis** <- tomographic spot backprojection: every off-beam spot
          pixel lies at a known in-plane ``k_lab``; for each voxel the ray
          samples, ``R_v^T k_lab`` is a body-frame position, and the spot
          amplitudes accumulate there (correct orientations reinforce the
          Bragg spikes, misassignments average into weak fog).

        Gradient refinement should then keep the basis frozen (or nearly so):
        the joint amplitude-loss landscape prefers k-space fog that fits the
        data over sharp spikes, and free basis refinement erases the spike
        structure even from a good initialisation.
        """
        dp = measurements.detach().cpu().numpy()
        n_tilt = dp.shape[0]
        if scan_shape is None:
            scan_shape = (dp.shape[1], dp.shape[2])
        n_row, n_col = scan_shape
        P = n_row * n_col
        det_r, det_c = self.det_shape
        dkr, dkc = self.k_sampling[1], self.k_sampling[2]
        Nkz, Nky, Nkx = self.k_shape
        cr, cc = det_r // 2, det_c // 2
        if beam_radius is None:
            beam = np.fft.fftshift(np.abs(self.Psi0.cpu().numpy()) > 0)
            rr_b = np.sqrt((np.arange(det_r)[:, None] - cr) ** 2
                           + (np.arange(det_c)[None, :] - cc) ** 2)
            beam_radius = float(rr_b[beam].max()) + 2.0
        rr = np.sqrt((np.arange(det_r)[:, None] - cr) ** 2
                     + (np.arange(det_c)[None, :] - cc) ** 2)
        off_beam = rr > beam_radius

        pos = self.scan_positions(scan_shape, scan_step, scan_origin)
        jobs = [(ti, j, i) for ti in range(n_tilt) for j in range(n_row) for i in range(n_col)]
        Wmat = torch.stack([self._ray_voxel_weights(pos[j, i], float(tilts_deg[ti]))
                            for (ti, j, i) in jobs]).cpu().numpy()
        R_all = self.rotation_matrices().detach().cpu().numpy().reshape(-1, 3, 3)

        # weights: virtual dark field
        dps_all = np.fft.fftshift(dp.reshape(-1, det_r, det_c), axes=(-2, -1))
        vdf = Wmat.T @ (dps_all * off_beam).sum(axis=(-2, -1))
        vdf = vdf / max(vdf.max(), 1e-30)
        W0 = np.repeat(vdf.reshape(*self.real_shape, 1), self.num_structures, axis=-1)

        # basis: spot backprojection through the per-voxel orientations
        acc = np.zeros(self.k_shape)
        cnt = np.zeros(self.k_shape)
        for ti in range(n_tilt):
            u, v, w = (t.cpu().numpy() for t in self.tilt_axes(float(tilts_deg[ti])))
            dps = np.fft.fftshift(dp[ti].reshape(P, det_r, det_c), axes=(-2, -1))
            for p in range(P):
                img = dps[p] * off_beam
                thr = spot_threshold * img.max()
                if thr <= 0:
                    continue
                rows, cols = np.where(img > thr)
                amps = np.sqrt(img[rows, cols])
                k_lab = (((cols - cc)[:, None] * dkc) * u[None, :]
                         + ((rows - cr)[:, None] * dkr) * v[None, :])
                wrow = Wmat[ti * P + p]
                for vi in np.where(wrow > ray_weight_min)[0]:
                    k_body = k_lab @ R_all[vi]
                    idx = np.rint(k_body / np.array(self.k_sampling)).astype(int) \
                        + np.array([Nkz // 2, Nky // 2, Nkx // 2])
                    ok = np.all((idx >= 0) & (idx < np.array(self.k_shape)), axis=1)
                    for (iz, iy, ix), a in zip(idx[ok], amps[ok] * wrow[vi]):
                        acc[iz, iy, ix] += a
                        cnt[iz, iy, ix] += wrow[vi]
        bp = np.where(cnt > 0, acc / np.maximum(cnt, 1e-9), 0.0)
        bp = np.fft.ifftshift(bp) / max(bp.max(), 1e-30) * basis_scale

        with torch.no_grad():
            self.weights.copy_(torch.tensor(W0, dtype=self.weights.dtype))
            b = torch.tensor(1j * bp, dtype=self.basis.dtype)[..., None]
            b = b.expand(*self.k_shape, self.num_structures).contiguous()
            b[0, 0, 0, :] = 1.0
            b = b * self.sphere_mask[..., None]
            b[0, 0, 0, :] = 1.0
            self.basis.copy_(b)

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

        Used for curriculum (staged) optimization: e.g. explore jointly, then
        freeze the angles and refine only the basis + weights, then release.
        Takes effect on the next :meth:`reconstruct` call. Weights are always
        learnable.
        """
        if basis is not None:
            self.learn_basis = bool(basis)
            self.basis.requires_grad_(self.learn_basis)
        if angles is not None:
            self.learn_angles = bool(angles)
            for p in self.angles.parameters():
                p.requires_grad_(self.learn_angles)

    def reconstruct(
        self,
        measurements: torch.Tensor,                 # (n_tilt, n_row, n_col, det, det) intensities
        tilts_deg,
        scan_shape: tuple[int, int],
        scan_step: float | tuple[float, float] = 1.0,
        scan_origin=None,
        num_iters: int = 100,
        lr: float = 5e-3,
        lr_weights: float | None = None,
        lr_angles: float | None = None,
        phase_only: bool = True,
        superslice: int = 1,
        tilt_batch: int = 1,
        nonneg_weights: bool = False,
        shrink_weights: float = 0.0,
        smooth_weights: float = 0.0,
        shrink_basis: float = 0.0,
        smooth_basis: float = 0.0,
        shrink_beam_zone: float = 1.0,
        basis_topk: int | None = None,
        friedel_basis: bool = False,
        angle_smooth: float = 0.0,
        reset_every: int = 10,
        reset_protect_vacuum: bool = True,
        reset_fraction: float = 0.1,
        reset_neighbor: float = 0.5,
        reset_taper: bool = True,
        reset_modes: torch.Tensor | None = None,
        progress: bool = True,
        print_every: int = 0,
    ) -> dict:
        """Fit basis + weights + angles to the measured tilt series (Adam).

        Full-batch gradient (bounded memory, one tilt at a time) + amplitude
        (sqrt-intensity) loss, with a **bad-voxel reset** every ``reset_every``
        iters: the worst ``reset_fraction`` of voxels (by residual error carried
        along their rays) get an orientation + weight jump so stuck voxels can
        escape wrong orientations. Each flipped voxel either **adopts a random
        real-space neighbor's orientation and weight** (probability
        ``reset_neighbor`` -- grains are contiguous, so neighbor adoption grows
        correctly-oriented regions) or takes a fresh random orientation with a
        material-scale weight. With ``reset_taper`` the number flipped decreases
        linearly to zero over the run (explore early, refine late). The
        lowest-loss state seen is snapshotted and returned.

        Set ``progress=False`` to hide the bar; ``print_every>0`` also prints
        the loss every N iters.
        """
        # record the geometry so simulate() can reproduce the predicted patterns
        # with no arguments after reconstruction
        self._scan_geometry = {"tilts_deg": list(tilts_deg), "scan_shape": tuple(scan_shape),
                               "scan_step": scan_step, "scan_origin": scan_origin}
        pos = self.scan_positions(scan_shape, scan_step, scan_origin)
        n_row, n_col = pos.shape[:2]
        if reset_modes is not None:
            reset_modes = torch.as_tensor(np.asarray(reset_modes), dtype=torch.float32,
                                          device=self.device)
        meas_amp = measurements.to(self.device).clamp_min(0).sqrt()
        jobs = [(ti, j, i) for ti in range(len(tilts_deg)) for j in range(n_row) for i in range(n_col)]
        n_dp = len(jobs)
        # eps=1e-30: the phase-object gradients are ~1e-10, so the default
        # eps=1e-8 would swamp sqrt(v) and throttle every Adam step ~100x.
        lr_weights = lr if lr_weights is None else lr_weights
        lr_angles = lr if lr_angles is None else lr_angles
        opt = self._make_optimizer(lr, lr_weights, lr_angles)
        group_names = ["weights"] + (["basis"] if self.learn_basis else []) \
            + (["angles"] if self.learn_angles else [])
        lrs: list[list[float]] = []
        gen = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        losses = []
        # fixed geometry: per-DP -> per-voxel deposited trilinear weight, used to
        # attribute each DP's residual to the voxels its ray strikes (reset metric)
        Wmat = torch.stack([
            self._ray_voxel_weights(pos[j, i], float(tilts_deg[ti])) for (ti, j, i) in jobs
        ])                                                            # (n_dp, n_voxels)
        res_per_dp = torch.zeros(n_dp, dtype=torch.float64, device=self.device)
        best = {"loss": float("inf"), "snap": None}

        origins = pos.reshape(-1, 3)                          # (P, 3)
        P = origins.shape[0]
        n_tilt = len(tilts_deg)
        pbar = tqdm(range(num_iters), disable=not progress, desc="reconstruct", unit="it")
        for it in pbar:
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for t0 in range(0, n_tilt, max(1, tilt_batch)):
                # tilts are processed in groups of tilt_batch: one autograd
                # graph (and one backward) per group. tilt_batch trades memory
                # for speed; gradients still accumulate into one full-batch
                # optimizer step per iteration.
                t_grp = list(range(t0, min(t0 + max(1, tilt_batch), n_tilt)))
                basis = self.masked_basis()
                R_all = self.rotation_matrices()
                Psi = self.forward_tilts(origins, [float(tilts_deg[ti]) for ti in t_grp],
                                         basis, R_all, self.weights,
                                         phase_only=phase_only,
                                         superslice=superslice)                       # (T,P,det,det)
                tgt = meas_amp[t_grp[0]:t_grp[-1] + 1].reshape(len(t_grp), P, *self.det_shape)
                tl = ((Psi.abs() - tgt) ** 2).mean(dim=(2, 3))                        # (T,P)
                (tl.sum() / n_dp).backward()
                res_per_dp[t_grp[0] * P:(t_grp[-1] + 1) * P] = tl.detach().reshape(-1)
                total += float(tl.sum())
            if angle_smooth > 0.0 and self.learn_angles:
                # angle-coherence prior: grains are contiguous, so neighboring
                # material voxels should share an orientation. Penalize the
                # weight-coupled Frobenius misfit of neighbor rotations -- this
                # makes the shared-basis factorization identifiable (without it
                # the data can be fit by giving every voxel its own orientation
                # of a smeared basis, and the true spike structure never forms).
                R_all = self.rotation_matrices().reshape(*self.real_shape, 3, 3)
                wmag = self.weights.detach().abs().sum(-1)
                wmax = wmag.max().clamp_min(1e-30)
                pen = 0.0
                for axis in (0, 1, 2):
                    r0 = R_all.narrow(axis, 0, R_all.shape[axis] - 1)
                    r1 = R_all.narrow(axis, 1, R_all.shape[axis] - 1)
                    w0 = wmag.narrow(axis, 0, wmag.shape[axis] - 1)
                    w1 = wmag.narrow(axis, 1, wmag.shape[axis] - 1)
                    wpair = (w0 * w1) / (wmax * wmax)
                    pen = pen + (wpair * ((r0 - r1) ** 2).sum(dim=(-2, -1))).mean()
                (angle_smooth * pen).backward()

            mean_loss = total / n_dp
            losses.append(mean_loss)
            lrs.append([float(g["lr"]) for g in opt.param_groups])
            if np.isfinite(mean_loss) and mean_loss < best["loss"]:
                best = {"loss": mean_loss, "snap": self._snapshot()}

            opt.step()
            self._sanitize(opt, gen)     # repair NaN/Inf params (degenerate SVD backward)
            if nonneg_weights:
                with torch.no_grad():
                    self.weights.clamp_(min=0.0)   # proximal projection: weights >= 0
            if shrink_weights > 0.0:
                # proximal L1 (soft-threshold) on the weight field: the material
                # is sparse (few voxels), so fog weights are pushed to exactly
                # zero while real ones survive. tau scales with the weight lr,
                # mirroring the explicit-6D model's shrink.
                with torch.no_grad():
                    tau = shrink_weights * lr_weights
                    self.weights.copy_(torch.sign(self.weights)
                                       * torch.clamp(self.weights.abs() - tau, min=0.0))
            if basis_topk and self.learn_basis:
                # iterative hard thresholding: keep only the K largest-magnitude
                # off-origin basis voxels (K annealed 10x -> 1x over the run).
                # Unlike a fixed L1 threshold this is self-scaling -- it always
                # preserves the strongest signal however small -- while forcing
                # the structure factor toward a few sharp spots. Zeroed entries
                # keep their Adam state and may revive if the data demands it.
                with torch.no_grad():
                    frac = it / max(num_iters - 1, 1)
                    K = int(round(basis_topk * 10.0 ** (1.0 - frac)))
                    for s in range(self.num_structures):
                        keep_origin = self.basis[0, 0, 0, s].clone()
                        mag = self.basis[..., s].abs().flatten()
                        if K < mag.numel():
                            thr = torch.topk(mag, K).values.min()
                            mask = self.basis[..., s].abs() >= thr
                            self.basis[..., s] *= mask
                        self.basis[0, 0, 0, s] = keep_origin
            if smooth_weights > 0.0:
                # real-space coherence on the WEIGHT field only (angles are
                # discontinuous at grain boundaries and must not be smoothed):
                # separable 3-tap Gaussian into the 6 neighbors each step.
                with torch.no_grad():
                    wgt = float(np.exp(-1.0 / (2.0 * smooth_weights ** 2)))
                    norm = 1.0 + 2.0 * wgt
                    W = self.weights
                    for axis in range(3):
                        n = W.shape[axis]
                        if n < 2:
                            continue
                        idx_p = torch.arange(-1, n - 1, device=W.device).clamp(min=0)
                        idx_n = torch.arange(1, n + 1, device=W.device).clamp(max=n - 1)
                        W.copy_((wgt * W.index_select(axis, idx_p) + W
                                 + wgt * W.index_select(axis, idx_n)) / norm)
            if friedel_basis and self.learn_basis:
                # the basis is the transmission's structure factor: for the
                # phase grating of a real potential the off-origin content is
                # anti-Hermitian, F(-k) = -conj(F(k)) (peaks purely imaginary,
                # as in make_au_basis). Every tilt plane contains both members
                # of a Friedel pair, so the data never constrains the Hermitian
                # component -- project it away. The origin (vacuum baseline,
                # real) is preserved separately.
                with torch.no_grad():
                    flip = torch.roll(torch.flip(self.basis, dims=(0, 1, 2)),
                                      shifts=(1, 1, 1), dims=(0, 1, 2))
                    keep = self.basis[0, 0, 0, :].clone()
                    self.basis.copy_(0.5 * (self.basis - flip.conj()))
                    self.basis[0, 0, 0, :] = keep
            if smooth_basis > 0.0 and self.learn_basis:
                # reciprocal-space coherence: a gentle 3-tap Gaussian each step
                # pools intensity split across neighboring k voxels into one
                # spike, so the shrinkage threshold sees a strong peak instead
                # of fragments, while incoherent fog averages down. Frequency
                # neighbors wrap across index 0 (unshifted storage); the origin
                # is held out entirely -- its vacuum amplitude is ~10x the
                # peaks and would bleed into the surrounding cluster.
                with torch.no_grad():
                    wgt = float(np.exp(-1.0 / (2.0 * smooth_basis ** 2)))
                    norm = 1.0 + 2.0 * wgt
                    Bv = self.basis
                    keep = Bv[0, 0, 0, :].clone()
                    Bv[0, 0, 0, :] = 0.0
                    for axis in range(3):
                        n = Bv.shape[axis]
                        idx_p = torch.arange(-1, n - 1, device=Bv.device) % n
                        idx_n = torch.arange(1, n + 1, device=Bv.device) % n
                        Bv.copy_((wgt * Bv.index_select(axis, idx_p) + Bv
                                  + wgt * Bv.index_select(axis, idx_n)) / norm)
                    Bv[0, 0, 0, :] = keep
            if shrink_basis > 0.0 and self.learn_basis:
                # proximal L1 on the basis' off-origin content: a structure
                # factor is a few sharp Bragg spots, so soft-thresholding kills
                # the k-space fog that otherwise fits the data with rings
                # instead of spots (the same cure the explicit-6D model needed).
                with torch.no_grad():
                    tau = shrink_basis * lr
                    if shrink_beam_zone != 1.0:
                        tau = tau * self._shrink_beam_zone(shrink_beam_zone)
                    mag = self.basis.abs()
                    keep = self.basis[0, 0, 0, :].clone()
                    self.basis.copy_(self.basis / mag.clamp_min(1e-30)
                                     * torch.clamp(mag - tau, min=0.0))
                    self.basis[0, 0, 0, :] = keep          # origin (vacuum) untouched

            n_res = 0
            # the reset escapes stuck orientations; skip it when angles are frozen
            if self.learn_angles and reset_every and (it + 1) % reset_every == 0 and it < num_iters - 1:
                # the worst voxels (by residual error along their rays) get an
                # orientation + weight jump: adopt a random real-space neighbor
                # (grain growth) or take a fresh random orientation. The number
                # flipped tapers to zero over the run (explore early, refine late).
                taper = (1.0 - it / num_iters) if reset_taper else 1.0
                n_res = int(round(reset_fraction * self.n_voxels * taper))
                if n_res > 0:
                    err_vox = Wmat.t() @ res_per_dp
                    if reset_protect_vacuum:
                        # settled vacuum voxels (near-zero weight) are doing
                        # their job -- strongly de-prioritize flipping them, so
                        # the jumps concentrate on misfit MATERIAL voxels.
                        wmag = self.weights.detach().abs().sum(-1).flatten()
                        vac = wmag < 0.15 * wmag.max().clamp_min(1e-30)
                        err_vox = err_vox * torch.where(vac, 0.2, 1.0)
                    bad = torch.topk(err_vox, n_res).indices
                    Nz, Ny, Nx = self.real_shape
                    with torch.no_grad():
                        Wf = self.weights.reshape(self.n_voxels, self.num_structures)
                        w_mean = self.weights.mean()
                        for v in bad.tolist():
                            if float(torch.rand(1, generator=gen)) < reset_neighbor:
                                # adopt a random in-bounds 6-neighbor's orientation + weight
                                iz, iy, ix = v // (Ny * Nx), (v // Nx) % Ny, v % Nx
                                nbrs = [(iz + dz, iy + dy, ix + dx)
                                        for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                                           (0, -1, 0), (0, 0, 1), (0, 0, -1))
                                        if 0 <= iz + dz < Nz and 0 <= iy + dy < Ny and 0 <= ix + dx < Nx]
                                jz, jy, jx = nbrs[int(torch.randint(len(nbrs), (1,), generator=gen))]
                                nb = (jz * Ny + jy) * Nx + jx
                                self.angles.M[v] = self.angles.M[nb] \
                                    + 0.02 * torch.randn(3, 3, generator=gen).to(self.device)
                                Wf[v] = Wf[nb]
                            elif reset_modes is not None:
                                # jump to a random discovered orientation mode:
                                # the indexing stage's mode list is a far better
                                # proposal distribution than uniform SO(3)
                                mi = int(torch.randint(reset_modes.shape[0], (1,), generator=gen))
                                self.angles.M[v] = reset_modes[mi].to(self.device) \
                                    + 0.03 * torch.randn(3, 3, generator=gen).to(self.device)
                                Wf[v] = w_mean
                            else:
                                self.angles.M[v] = (torch.eye(3)
                                                    + 0.1 * torch.randn(3, 3, generator=gen)).to(self.device)
                                Wf[v] = w_mean
                        self._reset_optimizer_state(opt, self.angles.M, bad)
                        self._reset_optimizer_state(opt, self.weights, bad)
            if print_every and (it % print_every == 0 or it == num_iters - 1):
                print(f"  it {it:4d}  loss {mean_loss:.4e}  best {best['loss']:.4e}"
                      + (f"  reset {n_res}" if n_res else ""), flush=True)
            pbar.set_postfix(loss=f"{mean_loss:.3e}", best=f"{best['loss']:.3e}")

        if best["snap"] is not None:
            self._restore(best["snap"])              # return the best-ever state
        self.losses = losses
        self.best_loss = best["loss"]
        self.lrs = lrs
        self.lr_group_names = group_names
        return {"losses": losses, "lrs": lrs, "best_loss": best["loss"],
                "basis": self.masked_basis().detach(),
                "weights": self.weights.detach(),
                "rotations": self.rotation_matrices().detach()}

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

        lrs = getattr(self, "lrs", None)
        if lrs:
            arr = np.asarray(lrs)                       # (n_it, n_groups)
            names = getattr(self, "lr_group_names", [f"group {i}" for i in range(arr.shape[1])])
            axr = ax.twinx()
            if np.allclose(arr, arr[:, :1]):            # all groups share one lr
                axr.semilogy(it, arr[:, 0], "--", color="C1", label="lr")
            else:
                for gi, nm in enumerate(names):
                    axr.semilogy(it, arr[:, gi], "--", label=f"lr ({nm})")
            axr.set_ylabel("learning rate", color="C1")
            axr.tick_params(axis="y", labelcolor="C1")
            axr.legend(loc="upper right", fontsize=8)
        ax.set_title("reconstruction loss + learning rate")
        ax.legend(loc="lower left", fontsize=8)
        plt.show()
        return fig, ax
