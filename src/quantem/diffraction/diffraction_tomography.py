"""Factorized diffraction tomography.

Replaces the explicit 6D structure-factor volume with a low-rank factorization:

  * ``basis``   -- ``num_structures`` shared complex 3D k-space structure factors,
    shape ``[N_kz, N_ky, N_kx, N_weights]``, spherically band-limited.
  * ``weights`` -- per real-space voxel mixing coefficients,
    shape ``[N_z, N_y, N_x, N_weights]``.
  * ``angles``  -- per real-space voxel SO(3) orientation (R9+SVD param),
    one rotation per voxel (``T = N_z * N_y * N_x``).

Each voxel's 3D structure factor is ``sum_i weight_i * Rotate(basis_i)`` with an
implicit vacuum baseline (the DC / origin pixel is pinned to 1).

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
  4. pin the origin (0,0) = 1.0 (vacuum baseline; keeps the DC beam),
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
    """Factorized 6D diffraction tomography (see the module docstring).

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
        """Boolean mask, True inside the inscribed sphere of the k cube."""
        Nkz, Nky, Nkx = self.k_shape
        dev = self.device
        # index-space radii from the (fft) origin, in *pixels*
        iz = torch.fft.fftfreq(Nkz, d=1.0 / Nkz).to(dev)[:, None, None]
        iy = torch.fft.fftfreq(Nky, d=1.0 / Nky).to(dev)[None, :, None]
        ix = torch.fft.fftfreq(Nkx, d=1.0 / Nkx).to(dev)[None, None, :]
        r_pix = torch.sqrt(iz ** 2 + iy ** 2 + ix ** 2)
        r_max = (min(self.k_shape) - 1) / 2.0        # inscribed radius, e.g. 20 for 41
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
            Seeds the k-space bases; if None, small complex noise (origin
            pinned to 1).
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
        """(N_z, N_y, N_x, 3, 3) per-voxel rotation matrices (body->lab)."""
        R = self.angles.as_matrix()                 # (T, 3, 3)
        return R.reshape(*self.real_shape, 3, 3)

    def masked_basis(self) -> torch.Tensor:
        """Basis with the spherical support re-imposed (differentiable).

        The basis is deliberately *not* normalised here. Forcing unit norm
        inside the forward makes the basis learn a unit direction on a
        high-dimensional sphere, which converges far slower than additive
        growth. The scale degeneracy is instead fixed as a gauge projection on
        the result (:meth:`normalize_gauge`, applied at the end of
        :meth:`reconstruct`), which leaves the forward model unchanged.

        Returns
        -------
        torch.Tensor
            ``[N_kz, N_ky, N_kx, N_weights]`` basis with values outside the
            inscribed sphere zeroed.
        """
        return self.basis * self.sphere_mask[..., None]

    def normalize_gauge(self) -> None:
        """Fix the weight/basis scale degeneracy in place, physically.

        Each basis is L2-normalised over its off-origin (Bragg) content so that
        ``sum |basis|**2 == 1`` there, and the scale is absorbed into the
        corresponding weights. Every ``weight * basis`` product is preserved, so
        the forward model and the data fit are unchanged -- only the gauge
        (physical, unit-norm bases; weights carrying the amplitude) changes. The
        origin pixel is overwritten during transmission assembly and is excluded
        from the norm.
        """
        with torch.no_grad():
            sq_all = (self.basis.abs() ** 2).reshape(-1, self.num_structures).sum(0)
            sq_origin = self.basis[0, 0, 0, :].abs() ** 2
            bn = torch.sqrt((sq_all - sq_origin).clamp_min(1e-24))       # (Nw,)
            self.basis /= bn
            self.weights *= bn

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
                # peak in (kx, ky, kz) physical -> place at [kz, ky, kx] index
                kx, ky, kz = (v / a_Au for v in vec)
                gz, gy, gx = kz / dkz, ky / dky, kx / dkx
                for dz in range(2):
                    bz = int(math.floor(gz)); wz = (gz - bz) if dz else (1.0 - (gz - bz)); iz = (bz + dz) % Nkz
                    for dy in range(2):
                        by = int(math.floor(gy)); wy = (gy - by) if dy else (1.0 - (gy - by)); iy = (by + dy) % Nky
                        for dx in range(2):
                            bx = int(math.floor(gx)); wx = (gx - bx) if dx else (1.0 - (gx - bx)); ix = (bx + dx) % Nkx
                            sf[iz, iy, ix] += amp * wz * wy * wx
        return sf

    def make_probe(self, probe_k_max: float, normalize: bool = True) -> torch.Tensor:
        """Top-hat aperture probe, 2D detector plane [row, col] = [ky, kx]."""
        kv, ku = torch.meshgrid(self.det_kv, self.det_ku, indexing="ij")
        aperture = (torch.sqrt(kv ** 2 + ku ** 2) <= probe_k_max).to(torch.complex128)
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
        self.prop = (bare * self.antialias_mask)
        self.prop_distance = dz
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
        never on the angles), and the origin is pinned to 1 (vacuum baseline).
        Out-of-slab neighbours contribute vacuum.

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
        Nk = torch.tensor([Nkz, Nky, Nkx], dtype=torch.float32, device=dev)

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
        if bool(valid.any()):
            vsel = idxs[valid]                                 # (nv,3)
            nv = vsel.shape[0]
            tw = tw8[valid].to(torch.complex64)                # (nv,)
            vidx = (vsel[:, 0] * Ny + vsel[:, 1]) * Nx + vsel[:, 2]
            R = R_all.reshape(-1, 3, 3)[vidx]                  # (nv,3,3) body->lab
            wv = W_all.reshape(-1, Nw)[vidx].to(torch.complex64)  # (nv,Nw)
            # rotate the detector (u,v) axes into each voxel's body frame: R^T @ axis
            u_b = torch.einsum("vij,i->vj", R, u)              # (nv,3) [kz,ky,kx]
            v_b = torch.einsum("vij,i->vj", R, v)
            # plane points (nv,R,C,3) in A^-1; basis is fftfreq-ordered so wrap
            # k/dk into [0, N) then normalise to [-1, 1] (grid last dim reversed)
            kxyz = (ku[None, ..., None] * u_b[:, None, None, :]
                    + kv[None, ..., None] * v_b[:, None, None, :])   # (nv,R,C,3)
            c = torch.remainder(kxyz / dk, Nk)                 # (nv,R,C,3) [cz,cy,cx]
            grid = torch.stack((
                2.0 * c[..., 2] / (Nkx - 1.0) - 1.0,
                2.0 * c[..., 1] / (Nky - 1.0) - 1.0,
                2.0 * c[..., 0] / (Nkz - 1.0) - 1.0,
            ), dim=-1)[:, None].to(torch.float32)              # (nv,1,R,C,3)
            bre = basis.real.permute(3, 0, 1, 2)[None].expand(nv, -1, -1, -1, -1).to(torch.float32)
            bim = basis.imag.permute(3, 0, 1, 2)[None].expand(nv, -1, -1, -1, -1).to(torch.float32)
            sre = F.grid_sample(bre, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            sim = F.grid_sample(bim, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            sampled = (sre + 1j * sim).squeeze(2)              # (nv,Nw,R,C)
            t_vox = (wv[:, :, None, None] * sampled).sum(1)    # (nv,R,C)
            acc = (tw[:, None, None] * t_vox).sum(0)           # (R,C)
        acc[0, 0] = 1.0                                        # pin the vacuum DC
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
        Nk = torch.tensor([Nkz, Nky, Nkx], dtype=torch.float32, device=dev)
        lo = torch.tensor([0, 0, 0], device=dev)
        hi = torch.tensor([Nz - 1, Ny - 1, Nx - 1], device=dev)
        bre_full = basis.real.permute(3, 0, 1, 2)              # (Nw,Nkz,Nky,Nkx)
        bim_full = basis.imag.permute(3, 0, 1, 2)

        acc = torch.zeros(P, det_row, det_col, dtype=torch.complex64, device=dev)
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
                    c = torch.remainder(kxyz / dk, Nk)
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
        acc[:, 0, 0] = 1.0                                       # pin the vacuum DC per probe
        return acc

    def forward_tilt(self, origins: torch.Tensor, tilt_x_deg: float,
                     basis: torch.Tensor, R_all: torch.Tensor, W_all: torch.Tensor,
                     phase_only: bool = True) -> torch.Tensor:
        """Multislice exit waves for *all* probes at one tilt (batched over probes).

        Probes at a fixed tilt share the beam direction, so their rays span the
        same z-steps; the wave is carried as ``(P, det, det)`` and every FFT,
        propagation and transmission build is batched over ``P``.

        Parameters
        ----------
        origins : torch.Tensor
            ``(P, 3)`` probe positions in index space ``[z, y, x]``.
        tilt_x_deg : float
            X-axis tilt in degrees.

        Returns
        -------
        torch.Tensor
            ``(P, det_row, det_col)`` complex exit waves.
        """
        u, v, w = self.tilt_axes(tilt_x_deg)
        pts = torch.stack([self.ray_samples(o, w) for o in origins])   # (P, Ns, 3)
        P, Ns = pts.shape[0], pts.shape[1]
        Psi = self.Psi0[None].expand(P, -1, -1).clone()
        num_pix = self.Psi0.numel()
        for s in range(Ns):
            SF = self._transmission_planes_batch(pts[:, s], u, v, basis, R_all, W_all)  # (P,det,det)
            t_real = torch.fft.ifft2(num_pix * SF)
            if phase_only:
                t_real = torch.exp(1j * torch.angle(t_real))
            Psi = torch.fft.fft2(torch.fft.ifft2(Psi) * t_real)
            if s < Ns - 1:
                Psi = Psi * self.prop
        return Psi * self.antialias_mask

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
        basis: torch.Tensor,
        weights: torch.Tensor,
        R_all: torch.Tensor,
        tilts_deg,
        scan_shape: tuple[int, int],
        scan_step: float | tuple[float, float] = 1.0,
        scan_origin=None,
        phase_only: bool = True,
    ) -> torch.Tensor:
        """Forward tilt series -> (n_tilt, n_row, n_col, det_row, det_col) intensities."""
        basis = basis * self.sphere_mask[..., None]          # enforce spherical support
        pos = self.scan_positions(scan_shape, scan_step, scan_origin)
        n_row, n_col = pos.shape[:2]
        origins = pos.reshape(-1, 3)                          # (P, 3)
        dp = torch.empty(len(tilts_deg), n_row, n_col, *self.det_shape, dtype=torch.float64)
        for ti, tilt in enumerate(tilts_deg):
            Psi = self.forward_tilt(origins, float(tilt), basis, R_all, weights, phase_only=phase_only)
            dp[ti] = (Psi.abs() ** 2).to(torch.float64).reshape(n_row, n_col, *self.det_shape)
        return dp

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

    def _make_optimizer(self, lr: float, lr_weights: float):
        """Adam with a higher lr on the weights: with the basis L2-normalised,
        a material weight must travel from ~0 to the basis amplitude while the
        basis and rotations are already unit-scale, so it needs a faster rate."""
        rest = ([self.basis] if self.learn_basis else []) \
            + (list(self.angles.parameters()) if self.learn_angles else [])
        return torch.optim.Adam(
            [{"params": [self.weights], "lr": lr_weights},
             {"params": rest, "lr": lr}],
            eps=1e-30,
        )

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
        phase_only: bool = True,
        reset_every: int = 10,
        reset_fraction: float = 0.1,
        progress: bool = True,
        print_every: int = 0,
    ) -> dict:
        """Fit basis + weights + angles to the measured tilt series (Adam).

        Full-batch gradient (bounded memory, one tilt at a time) + amplitude
        (sqrt-intensity) loss, with a **bad-voxel reset** every ``reset_every``
        iters: the worst ``reset_fraction`` of voxels (by residual error carried
        along their rays) get a fresh random orientation *and* a material-scale
        weight -- the aggressive "orientation + weight jumps" that let stuck
        voxels escape wrong orientations so the sparse Bragg solution can emerge.
        The lowest-loss state seen is snapshotted and returned.

        Set ``progress=False`` to hide the bar; ``print_every>0`` also prints
        the loss every N iters.
        """
        pos = self.scan_positions(scan_shape, scan_step, scan_origin)
        n_row, n_col = pos.shape[:2]
        meas_amp = measurements.to(self.device).clamp_min(0).sqrt()
        jobs = [(ti, j, i) for ti in range(len(tilts_deg)) for j in range(n_row) for i in range(n_col)]
        n_dp = len(jobs)
        # eps=1e-30: the phase-object gradients are ~1e-10, so the default
        # eps=1e-8 would swamp sqrt(v) and throttle every Adam step ~100x.
        lr_weights = lr if lr_weights is None else lr_weights
        opt = self._make_optimizer(lr, lr_weights)
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
            for ti in range(n_tilt):
                # one tilt at a time -> independent graph -> bounded memory, while
                # all P probes of the tilt are batched through forward_tilt.
                basis = self.masked_basis()
                R_all = self.rotation_matrices()
                Psi = self.forward_tilt(origins, float(tilts_deg[ti]), basis, R_all,
                                        self.weights, phase_only=phase_only)          # (P,det,det)
                tl = ((Psi.abs() - meas_amp[ti].reshape(P, *self.det_shape)) ** 2).mean(dim=(1, 2))
                (tl.sum() / n_dp).backward()
                res_per_dp[ti * P:(ti + 1) * P] = tl.detach()
                total += float(tl.sum())
            mean_loss = total / n_dp
            losses.append(mean_loss)
            if mean_loss < best["loss"]:
                best = {"loss": mean_loss, "snap": self._snapshot()}

            opt.step()

            n_res = 0
            # the reset escapes stuck orientations; skip it when angles are frozen
            if self.learn_angles and reset_every and (it + 1) % reset_every == 0 and it < num_iters - 1:
                # the worst voxels (by residual error along their rays) get a fresh
                # random orientation AND a material-scale weight -- the aggressive
                # jumps that let stuck voxels find the right orientation.
                err_vox = Wmat.t() @ res_per_dp
                n_res = max(1, int(round(reset_fraction * self.n_voxels)))
                bad = torch.topk(err_vox, n_res).indices
                with torch.no_grad():
                    newM = torch.eye(3).reshape(1, 3, 3).repeat(n_res, 1, 1) \
                        + 0.1 * torch.randn(n_res, 3, 3, generator=gen)
                    self.angles.M[bad] = newM.to(self.device)
                    self.weights.reshape(self.n_voxels, self.num_structures)[bad] = self.weights.mean()
                    self._reset_optimizer_state(opt, self.angles.M, bad)
                    self._reset_optimizer_state(opt, self.weights, bad)
            if print_every and (it % print_every == 0 or it == num_iters - 1):
                print(f"  it {it:4d}  loss {mean_loss:.4e}  best {best['loss']:.4e}"
                      + (f"  reset {n_res}" if n_res else ""), flush=True)
            pbar.set_postfix(loss=f"{mean_loss:.3e}", best=f"{best['loss']:.3e}")

        if best["snap"] is not None:
            self._restore(best["snap"])              # return the best-ever state
        self.normalize_gauge()                        # physical unit-norm bases (fit unchanged)
        return {"losses": losses, "best_loss": best["loss"],
                "basis": self.masked_basis().detach(),
                "weights": self.weights.detach(),
                "rotations": self.rotation_matrices().detach()}
