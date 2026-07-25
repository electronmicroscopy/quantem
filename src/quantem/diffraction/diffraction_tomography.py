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

from quantem.core.ml.inr import HSiren
from quantem.core.ml.models.so3params import SO3ParamR9SVD
from quantem.core.utils.utils import electron_wavelength_angstrom


class _PeakBasis(torch.nn.Module):
    """Parametric Bragg-peak basis: amplitudes on a learnable reciprocal lattice.

    The whole structure factor is ``sum_hkl i * a(hkl) * splat(G_hkl)`` with
    ``G_hkl = hkl @ B_rec`` -- a few dozen parameters (9 lattice + one
    amplitude per symmetry shell) instead of ``N_k^3`` pixels or ~50k network
    weights. Peaks land at continuous positions via trilinear splatting (the
    same placement the test data uses), so there is **no spectral leakage**:
    a real-space field parameterization leaks across all of k-space unless
    the box is an exact multiple of the unit cell (measured: an ideal crystal
    u-field on a 20 A box holding 4.9 cells already produces 51% off-peak
    mass), which is what limits the network's structure-factor fidelity.

    Purely imaginary amplitudes shared over each ``sorted(|h|,|k|,|l|)``
    shell build in both Friedel (``F(-k) = -conj F(k)``) and the cubic point
    group exactly, for a centrosymmetric crystal aligned to the grid axes.
    """

    def __init__(self, hkl: torch.Tensor, shell_idx: torch.Tensor, n_shells: int,
                 B_rec: torch.Tensor, n_struct: int, learn_cell: bool = True):
        super().__init__()
        self.register_buffer("hkl", hkl.to(torch.float32))
        self.register_buffer("shell_idx", shell_idx.to(torch.long))
        self.B_rec = torch.nn.Parameter(B_rec.to(torch.float32),
                                        requires_grad=bool(learn_cell))
        self.amp = torch.nn.Parameter(torch.full((n_shells, n_struct), 1e-3))
        self.origin = torch.nn.Parameter(torch.ones(n_struct))
        # free (off-lattice) peaks, born from residuals: fixed positions
        # [A^-1, (z,y,x)], learnable amplitudes; empty by default
        self.register_buffer("g_free", torch.zeros(0, 3))
        self.amp_free = torch.nn.Parameter(torch.zeros(0, n_struct))

    def untie_shells(self) -> None:
        """Release the shell tying: one learnable amplitude per reflection.

        Amplitudes are copied to every member, so the generated basis is
        unchanged at the moment of release; the data can then differentiate
        reflections the symmetry tied together, and unsupported ones can
        decay to zero and be pruned (the 'death' half of adaptive peaks).
        """
        with torch.no_grad():
            per_peak = self.amp[self.shell_idx].detach().clone()
        self.amp = torch.nn.Parameter(per_peak)
        self.shell_idx = torch.arange(self.hkl.shape[0], device=self.hkl.device)

    def prune(self, min_frac: float = 0.02) -> int:
        """Drop reflections below ``min_frac`` of the maximum amplitude.

        Requires untied shells (each reflection owns its amplitude). Free
        peaks are pruned by the same rule. Returns the number removed.
        """
        assert self.amp.shape[0] == self.hkl.shape[0], "untie_shells() first"
        with torch.no_grad():
            ref = torch.cat([self.amp.abs().amax(1), self.amp_free.abs().amax(1)])
            tau = min_frac * ref.max().clamp_min(1e-30)
            keep = self.amp.abs().amax(1) >= tau
            keep_f = self.amp_free.abs().amax(1) >= tau
            n_drop = int((~keep).sum() + (~keep_f).sum())
            self.hkl = self.hkl[keep]
            self.shell_idx = torch.arange(int(keep.sum()), device=self.hkl.device)
            self.amp = torch.nn.Parameter(self.amp[keep].detach().clone())
            self.g_free = self.g_free[keep_f]
            self.amp_free = torch.nn.Parameter(self.amp_free[keep_f].detach().clone())
        return n_drop

    def add_free_peaks(self, g_new: torch.Tensor, amp0: float = 1e-3) -> None:
        """Append off-lattice peaks at fixed positions ``g_new`` ((M, 3) A^-1)."""
        with torch.no_grad():
            n_struct = self.amp.shape[1]
            self.g_free = torch.cat([self.g_free, g_new.to(self.g_free)])
            grown = torch.cat([self.amp_free.detach(),
                               torch.full((g_new.shape[0], n_struct), amp0,
                                          device=self.amp_free.device)])
        self.amp_free = torch.nn.Parameter(grown)

    def volume(self, k_shape, k_sampling, device) -> torch.Tensor:
        """Splat the peaks onto the centered k grid -> (Nkz, Nky, Nkx, Nw)."""
        Nkz, Nky, Nkx = k_shape
        Nw = self.amp.shape[1]
        dk = torch.tensor(k_sampling, dtype=torch.float32, device=device)
        ctr = torch.tensor([Nkz // 2, Nky // 2, Nkx // 2], dtype=torch.float32,
                           device=device)
        g = self.hkl @ self.B_rec                           # (P, 3) lattice peaks
        a_pk = self.amp[self.shell_idx]                     # (P, Nw)
        if self.g_free.shape[0]:
            g = torch.cat([g, self.g_free])
            a_pk = torch.cat([a_pk, self.amp_free])
        c = g / dk + ctr                                    # voxel coords
        base = torch.floor(c)
        frac = c - base
        base = base.to(torch.long)
        vol = torch.zeros(Nkz * Nky * Nkx, Nw, dtype=torch.float32, device=device)
        for oz in (0, 1):
            for oy in (0, 1):
                for ox in (0, 1):
                    w = ((frac[:, 0] if oz else 1 - frac[:, 0])
                         * (frac[:, 1] if oy else 1 - frac[:, 1])
                         * (frac[:, 2] if ox else 1 - frac[:, 2]))
                    iz, iy, ix = base[:, 0] + oz, base[:, 1] + oy, base[:, 2] + ox
                    ok = ((iz >= 0) & (iz < Nkz) & (iy >= 0) & (iy < Nky)
                          & (ix >= 0) & (ix < Nkx))
                    flat = (iz[ok] * Nky + iy[ok]) * Nkx + ix[ok]
                    vol.index_add_(0, flat, a_pk[ok] * w[ok, None])
        return vol.reshape(Nkz, Nky, Nkx, Nw)


class _LatticeMotifINR(torch.nn.Module):
    """Lattice-periodic INR: a motif network over one (learnable) unit cell.

    The field is ``u(x) = net(cos 2 pi f, sin 2 pi f)`` with fractional
    coordinates ``f = x A^-1`` -- exactly periodic under the Bravais lattice
    whose ROW vectors are ``A`` (Angstroms, body frame), for ANY ``A``. Peak
    positions in k become continuous functions of the cell instead of being
    quantized to the k grid, and content between reciprocal-lattice points is
    structurally impossible (the fog an unconstrained INR paints there).
    ``A`` can be refined by gradient (``learn_cell``); the crystal system is
    then *diagnosed* from the converged metric tensor rather than assumed.
    """

    def __init__(self, net: torch.nn.Module, cell, learn_cell: bool = False):
        super().__init__()
        self.net = net
        A = torch.as_tensor(cell, dtype=torch.float32)
        if A.ndim == 0:
            A = torch.eye(3) * A
        self.A = torch.nn.Parameter(A, requires_grad=bool(learn_cell))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        f = xyz @ torch.linalg.inv(self.A)          # fractional coords
        th = 2.0 * math.pi * f
        return self.net(torch.cat([torch.cos(th), torch.sin(th)], dim=-1))


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
                # peak in (kx, ky, kz) physical -> trilinear splat onto the
                # [kz, ky, kx] grid. Splatting keeps every reflection's
                # CENTROID at its exact reciprocal position (a peak reads as
                # a 2x2x2 cluster instead of one bright voxel), whereas
                # nearest-voxel snapping moved each reflection independently
                # by up to half a cell -- on coarse grids (dk = 0.10) that
                # made the test "crystal" internally inconsistent (111 and
                # 200 implied different cells), silently breaking any
                # lattice-based validation or indexing.
                kx, ky, kz = (v / a_Au for v in vec)
                cz, cy, cx = kz / dkz, ky / dky, kx / dkx
                z0, y0, x0 = math.floor(cz), math.floor(cy), math.floor(cx)
                fz, fy, fx = cz - z0, cy - y0, cx - x0
                for oz, wz in ((0, 1.0 - fz), (1, fz)):
                    for oy, wy in ((0, 1.0 - fy), (1, fy)):
                        for ox, wx in ((0, 1.0 - fx), (1, fx)):
                            w = wz * wy * wx
                            if w == 0.0:
                                continue
                            sf[(z0 + oz) % Nkz, (y0 + oy) % Nky,
                               (x0 + ox) % Nkx] += amp * w
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
        # single-precision wave optics: the learnable basis/weights stay double
        # (precise Adam accumulation) but the FFT chain and its backward run in
        # complex64. The dominant sampling path was already float32, so the
        # forward is effectively single precision -- this just stops the FFTs
        # from up-casting the whole wave to complex128.
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
        self.antialias_mask = mask.to(self.device)   # circular k envelope
        self._bare_prop = bare.to(self.device)       # unit-modulus Fresnel factor
        self._prop_pow_cache = {}
        self.prop = (self._bare_prop * self.antialias_mask)
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
        # unique-voxel assembly (PRISM-style factorization): each distinct
        # struck voxel's plane is sampled ONCE per tilt and every probe's
        # slice SF is assembled by the constant trilinear matrix T -- the
        # same algebra as the per-corner path, reordered, so probes stop
        # re-sampling the voxels they share.
        uslices = []
        for vidx, tw in slices:
            twr = tw.real
            m = (twr > 0).reshape(-1)
            uv = torch.unique(vidx.reshape(-1)[m])
            lookup = torch.full((self.n_voxels,), 0, dtype=torch.long, device=dev)
            lookup[uv] = torch.arange(uv.numel(), device=dev)
            T = torch.zeros(P, max(int(uv.numel()), 1), device=dev)
            T.scatter_add_(1, lookup[vidx], twr * (twr > 0))
            uslices.append((uv, T))
        kv, ku = torch.meshgrid(self.det_kv, self.det_ku, indexing="ij")
        geo = {"u": u, "v": v, "slices": slices, "uslices": uslices,
               "ku": ku, "kv": kv}
        cache[key] = geo
        return geo

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

    def _basis_sample_vol(self, basis: torch.Tensor) -> torch.Tensor:
        """Centered, channel-packed basis volume for grid_sample, once per forward.

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

    def measured_support(self, tilts_deg=None, w_rel: float = 0.15,
                         halfwidth: float = 1.5) -> torch.Tensor:
        """Body-frame k support actually constrained by the tilt series.

        The data samples the basis only along each tilt's central plane,
        pulled back through each material voxel's orientation; everywhere
        else the loss is blind. A pixel basis under ``shrink_basis`` decays
        to zero there, but an INR freely extrapolates smooth content into the
        unmeasured complement. This mask -- pure acquisition geometry plus
        the converged orientations, no ground truth -- marks the union of
        sampled planes; multiply the basis by it before display or
        comparison to remove the unconstrained content.

        Parameters
        ----------
        tilts_deg : sequence, optional
            Defaults to the tilts recorded by the last :meth:`reconstruct`.
        w_rel : float, default 0.15
            Voxels with total ``|weight|`` above ``w_rel`` times the max
            contribute their orientation (vacuum voxels' orientations are
            meaningless).
        halfwidth : float, default 1.5
            Plane half-thickness in k voxels (trilinear sampling reach).

        Returns
        -------
        torch.Tensor
            ``[N_kz, N_ky, N_kx]`` bool, unshifted frequency order (matches
            ``basis``).
        """
        if tilts_deg is None:
            tilts_deg = self._scan_geometry["tilts_deg"]
        with torch.no_grad():
            R = self.rotation_matrices().reshape(-1, 3, 3)
            wmag = self.weights.detach().abs().sum(-1).reshape(-1)
            mat = wmag > w_rel * wmag.max().clamp_min(1e-30)
            if bool(mat.any()):
                R = R[mat]
            n_lab = torch.stack([self.tilt_axes(float(t))[2] for t in tilts_deg])
            nb = torch.einsum("vij,ti->vtj", R.to(torch.float32), n_lab)  # R^T n
            nb = torch.unique(torch.round(nb.reshape(-1, 3) * 200.0) / 200.0, dim=0)
            dev = self.device
            axes = [(torch.arange(n, device=dev, dtype=torch.float32) - n // 2) * dk
                    for n, dk in zip(self.k_shape, self.k_sampling)]
            KZ, KY, KX = torch.meshgrid(*axes, indexing="ij")
            K = torch.stack([KZ, KY, KX], -1).reshape(-1, 3)
            tau = halfwidth * min(self.k_sampling)
            sup = torch.zeros(K.shape[0], dtype=torch.bool, device=dev)
            for c0 in range(0, nb.shape[0], 256):
                sup |= ((K @ nb[c0:c0 + 256].T).abs() < tau).any(-1)
            return torch.fft.ifftshift(sup.reshape(self.k_shape), dim=(0, 1, 2))

    def consolidate_grains(self, tol_deg: float = 10.0, w_rel: float = 0.15,
                           min_size: int = 2) -> dict:
        """Project the orientation field onto piecewise-constant grains.

        A polycrystal's orientation field is not smooth but *piecewise
        constant*: every voxel of a grain shares one rotation. The per-voxel
        parameterization does not know this, so each voxel converges to its
        own slightly different orientation and the per-grain structure factor
        blurs. This groups material voxels into spatially connected clusters
        of similar orientation (misorientation below ``tol_deg``, reduced by
        cubic symmetry) and replaces every member's rotation with the cluster
        mean -- a projection onto the physical model that both sharpens the
        recovered structure factors and collapses ~3 x n_voxels orientation
        degrees of freedom to ~3 per grain.

        Members are symmetry-aligned to their cluster seed before averaging
        (the equivalence class of a rotation is ``R S^T`` over the 24 proper
        cubic operations, so raw averaging of equivalent-but-different
        matrices would cancel), and the mean is projected back onto SO(3).

        Parameters
        ----------
        tol_deg : float, default 10.0
            Maximum symmetry-reduced misorientation between neighbors of the
            same grain.
        w_rel : float, default 0.15
            Material threshold as a fraction of the maximum voxel weight
            (vacuum voxels have no meaningful orientation).
        min_size : int, default 2
            Clusters smaller than this keep their own orientations.

        Returns
        -------
        dict
            ``n_grains``, ``sizes`` and ``spread_deg`` (mean pre-projection
            misorientation to the cluster mean, i.e. how much was collapsed).
        """
        Nz, Ny, Nx = self.real_shape
        R = self.rotation_matrices().detach().cpu().numpy().reshape(-1, 3, 3)
        wmag = self.weights.detach().abs().sum(-1).reshape(-1).cpu().numpy()
        material = wmag > w_rel * max(wmag.max(), 1e-30)
        ops = self._cubic_ops()

        lab = -np.ones(self.n_voxels, dtype=int)
        clusters: list[list[int]] = []
        for v0 in np.nonzero(material)[0]:
            if lab[v0] >= 0:
                continue
            comp, stack = [], [int(v0)]
            lab[v0] = len(clusters)
            while stack:
                v = stack.pop()
                comp.append(v)
                iz, iy, ix = v // (Ny * Nx), (v // Nx) % Ny, v % Nx
                for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                   (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    jz, jy, jx = iz + dz, iy + dy, ix + dx
                    if not (0 <= jz < Nz and 0 <= jy < Ny and 0 <= jx < Nx):
                        continue
                    u = (jz * Ny + jy) * Nx + jx
                    if not material[u] or lab[u] >= 0:
                        continue
                    if self._miso_deg(R[v].T @ R[u]) < tol_deg:
                        lab[u] = lab[v0]
                        stack.append(u)
            clusters.append(comp)

        sizes, spreads = [], []
        with torch.no_grad():
            for comp in clusters:
                if len(comp) < min_size:
                    continue
                seed = R[comp[0]]
                aligned = []
                for v in comp:
                    # pick the symmetry copy R_v S^T closest to the seed
                    best, best_tr = None, -np.inf
                    for S in ops:
                        cand = R[v] @ S.T
                        tr = float(np.einsum("ij,ij->", cand, seed))
                        if tr > best_tr:
                            best, best_tr = cand, tr
                    aligned.append(best)
                A = np.mean(aligned, axis=0)
                U, _, Vt = np.linalg.svd(A)                 # nearest rotation
                Rm = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
                spreads.append(float(np.mean([self._miso_deg(Rm.T @ a) for a in aligned])))
                sizes.append(len(comp))
                Rt = torch.as_tensor(Rm, dtype=self.angles.M.dtype, device=self.device)
                for v in comp:
                    self.angles.M[v] = Rt
        return {"n_grains": len(sizes), "sizes": sizes,
                "spread_deg": float(np.mean(spreads)) if spreads else 0.0}

    @classmethod
    def _miso_deg(cls, dR: np.ndarray) -> float:
        """Cubic-symmetry-reduced misorientation angle (deg) of a relative rotation."""
        if not hasattr(cls, "_ops_cache"):
            cls._ops_cache = cls._cubic_ops()
        tr = np.einsum("sij,ji->s", cls._ops_cache, dR)
        c = np.clip((tr.max() - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(c)))

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
        snap = {"basis": self.basis.detach().clone(),
                "weights": self.weights.detach().clone(),
                "M": self.angles.M.detach().clone()}
        if getattr(self, "basis_inr", None) is not None:
            snap["inr"] = {k: v.detach().clone()
                           for k, v in self.basis_inr.state_dict().items()}
        return snap

    def _restore(self, snap: dict) -> None:
        with torch.no_grad():
            self.basis.copy_(snap["basis"])
            self.weights.copy_(snap["weights"])
            self.angles.M.copy_(snap["M"])
            if "inr" in snap and getattr(self, "basis_inr", None) is not None:
                self.basis_inr.load_state_dict(snap["inr"])

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

    def _make_optimizer(self, lr: float, lr_weights: float, lr_angles: float,
                        inr_lr: float | None = None):
        """Adam with per-parameter-group learning rates (eps=1e-30 because the
        phase-object gradients are ~1e-10 and the default eps would throttle
        every step ~100x). With ``inr_lr`` the basis group is the INR network's
        parameters instead of the explicit basis tensor."""
        groups = [{"params": [self.weights], "lr": lr_weights, "name": "weights"}]
        if self.learn_basis:
            if inr_lr is not None:
                # the peak model's lattice moves every reflection at once, so
                # it needs a far smaller step than the amplitudes (at equal lr
                # it runs away); it is only in the optimizer at all when the
                # caller left it learnable.
                cell = [p for n, p in self.basis_inr.named_parameters()
                        if n == "B_rec" and p.requires_grad]
                rest = [p for n, p in self.basis_inr.named_parameters()
                        if n != "B_rec"]
                groups.append({"params": rest, "lr": inr_lr, "name": "inr"})
                if cell:
                    groups.append({"params": cell, "lr": inr_lr * 1e-3,
                                   "name": "cell"})
            else:
                groups.append({"params": [self.basis], "lr": lr, "name": "basis"})
        if self.learn_angles:
            groups.append({"params": list(self.angles.parameters()),
                           "lr": lr_angles, "name": "angles"})
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

    def _inr_setup(self, width: int, depth: int, omega: float,
                   space: str = "r", periodic: bool = False,
                   cell=None, learn_cell: bool = False) -> None:
        """Create the basis INR (an HSiren MLP) and cache its input coords.

        ``space="r"``: the network represents the body-frame *real-space*
        phase field ``u(r)`` (one real output per structure) on the k-shaped
        grid; the basis is ``i * FFT(u)``. A crystal potential is a smooth,
        band-limited sum of a few sinusoids -- exactly a SIREN's natural
        output -- whereas the k-space spikes it transforms to are a SIREN's
        worst case. A real ``u`` also makes the basis exactly anti-Hermitian
        (Friedel) by construction.

        ``space="k"``: the network maps centered k-grid coordinates to
        ``2 * num_structures`` outputs (real/imaginary per structure)
        directly.

        ``space="lattice"``: like ``"r"``, but the field is a motif network
        over one unit cell (:class:`_LatticeMotifINR`) -- exactly periodic
        under the Bravais lattice ``cell`` ((3, 3) rows in Angstroms, or a
        scalar for a cubic cell). Content away from the reciprocal lattice
        is structurally impossible, and with ``learn_cell=True`` the cell
        refines by gradient. Estimate the cell from the data (diffraction
        ring radii), never from ground truth.

        Inputs are grid coordinates normalized to ``[-1, 1]^3``; with
        ``periodic=True`` (r-space only) they are lifted to torus features
        ``(cos, sin)`` of one box period, making ``u`` exactly periodic on
        the supercell the DFT already assumes -- the true function class of a
        band-limited grid field, with no crystallographic assumption. Seeded
        from ``self.seed`` inside a forked RNG so runs stay bit-reproducible
        without disturbing the global generator.
        """
        assert space in ("r", "k", "lattice", "peaks")
        self._inr_space = space
        if space == "peaks":
            assert cell is not None, "space='peaks' needs cell= (Angstroms)"
            A = torch.as_tensor(cell, dtype=torch.float32)
            B_rec = torch.eye(3) / A if A.ndim == 0 else torch.linalg.inv(A).T
            kmax = self.sphere_radius_pix * min(self.k_sampling)
            hmax = int(np.ceil(kmax * float(A if A.ndim == 0 else A.diagonal().max()))) + 1
            hkls, keys = [], []
            for h in range(-hmax, hmax + 1):
                for k in range(-hmax, hmax + 1):
                    for l in range(-hmax, hmax + 1):
                        if (h, k, l) == (0, 0, 0):
                            continue
                        G = torch.tensor([float(h), float(k), float(l)]) @ B_rec
                        if float(G.norm()) <= kmax:
                            hkls.append((h, k, l))
                            keys.append(tuple(sorted((abs(h), abs(k), abs(l)))))
            uniq = sorted(set(keys))
            shell_idx = torch.tensor([uniq.index(k) for k in keys])
            self.basis_inr = _PeakBasis(torch.tensor(hkls), shell_idx, len(uniq),
                                        B_rec, self.num_structures,
                                        learn_cell).to(self.device)
            self._peak_shells = uniq
            return
        self._inr_periodic = bool(periodic) and space == "r"
        n_out = self.num_structures if space in ("r", "lattice") else 2 * self.num_structures
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + 11)
            net = HSiren(
                in_features=6 if (self._inr_periodic or space == "lattice") else 3,
                out_features=n_out,
                hidden_layers=depth,
                hidden_features=width,
                first_omega_0=omega,
                hidden_omega_0=omega,
                final_activation="identity",
            ).to(self.device)
        if space == "lattice":
            assert cell is not None, "space='lattice' needs cell= (Angstroms)"
            self.basis_inr = _LatticeMotifINR(net, cell, learn_cell).to(self.device)
            # physical body-frame positions of the u-field grid (Angstroms):
            # the box the DFT assumes has period L_i = 1/dk_i
            axes = [(torch.arange(n, device=self.device, dtype=torch.float32) - n // 2)
                    * (1.0 / dk / n)
                    for n, dk in zip(self.k_shape, self.k_sampling)]
            ZZ, YY, XX = torch.meshgrid(*axes, indexing="ij")
            self._inr_coords = torch.stack((ZZ, YY, XX), dim=-1).reshape(-1, 3)
            return
        self.basis_inr = net
        axes = [torch.linspace(-1.0, 1.0, n, device=self.device) for n in self.k_shape]
        ZZ, YY, XX = torch.meshgrid(*axes, indexing="ij")
        c = torch.stack((ZZ, YY, XX), dim=-1).reshape(-1, 3).to(torch.float32)
        if self._inr_periodic:
            # one box period spans the grid's N samples, i.e. N/(N-1) of the
            # linspace range -- use the DFT's own angular step so index 0 and
            # index N (one past the last sample) coincide on the torus.
            w = torch.tensor([2.0 * math.pi * (n - 1) / (2.0 * n) for n in self.k_shape],
                             device=self.device)
            th = c * w
            c = torch.cat([torch.cos(th), torch.sin(th)], dim=-1)
        self._inr_coords = c

    def _inr_basis(self, friedel: bool = True, cubic: bool = False) -> torch.Tensor:
        """Generate the basis volume from the INR, differentiably.

        Evaluates the network on the full centered k grid, reassembles the
        complex volume, shifts to storage (unshifted) frequency order and
        applies the spherical support plus the optional Friedel / cubic
        projections as differentiable ops -- under the INR these symmetries
        are enforced on the *generated* volume each evaluation instead of as
        proximal steps on a parameter tensor. Returns ``[N_kz, N_ky, N_kx,
        N_w]`` complex, same layout/dtype as :meth:`masked_basis`.
        """
        Nw = self.num_structures
        if getattr(self, "_inr_space", "k") == "peaks":
            # amplitudes splatted straight onto the reciprocal lattice: no
            # spectral leakage, Friedel + cubic symmetry exact by construction
            im = self.basis_inr.volume(self.k_shape, self.k_sampling, self.device)
            Bc = torch.complex(torch.zeros_like(im), im)
            B = torch.fft.ifftshift(Bc, dim=(0, 1, 2)).to(self.basis.dtype)
            B = B * self.sphere_mask[..., None]
            B = B.clone()
            B[0, 0, 0, :] = self.basis_inr.origin.to(B.dtype)
            return B
        if getattr(self, "_inr_space", "k") in ("r", "lattice"):
            # real field u(r) -> basis = i * FFT(u): anti-Hermitian (Friedel)
            # by construction, and the network fits the smooth band-limited
            # potential instead of its spiky transform. norm="forward" keeps
            # the basis at the field's amplitude scale. ("lattice": u is
            # motif(frac coords) -- exactly lattice-periodic.)
            u = self.basis_inr(self._inr_coords).reshape(*self.k_shape, Nw)
            u = u.permute(3, 0, 1, 2).to(torch.float64)
            Bk = 1j * torch.fft.fftn(u, dim=(1, 2, 3), norm="forward")
            Bc = torch.fft.fftshift(Bk, dim=(1, 2, 3)).permute(1, 2, 3, 0)
        else:
            out = self.basis_inr(self._inr_coords).reshape(*self.k_shape, 2, Nw)
            Bc = torch.complex(out[..., 0, :], out[..., 1, :])      # centered order
        if cubic:
            acc = torch.zeros_like(Bc)
            for perm, flip in self._cubic_grid_ops():
                T = Bc.permute(perm[0], perm[1], perm[2], 3)
                if flip:
                    T = torch.flip(T, dims=flip)
                acc = acc + T
            Bc = acc / 24.0
        B = torch.fft.ifftshift(Bc, dim=(0, 1, 2)).to(self.basis.dtype)
        B = B * self.sphere_mask[..., None]
        if friedel and getattr(self, "_inr_space", "k") == "k":
            # r-space nets are anti-Hermitian by construction; project only
            # the direct k-space parameterization.
            # transmission SF off-origin is anti-Hermitian, F(-k) = -conj(F(k));
            # keep the network's own (real-valued-free) origin sample.
            flipc = torch.roll(torch.flip(B, dims=(0, 1, 2)),
                               shifts=(1, 1, 1), dims=(0, 1, 2)).conj()
            proj = 0.5 * (B - flipc)
            mask = torch.zeros(self.k_shape, dtype=torch.bool, device=self.device)
            mask[0, 0, 0] = True
            B = torch.where(mask[..., None], B, proj)
        return B

    @staticmethod
    def _l1_weighting(tau, mag: torch.Tensor, reweight: float):
        """Per-element L1 threshold, optionally iteratively reweighted.

        With ``reweight > 0`` the threshold is scaled by ``eps / (|x| + eps)``
        with ``eps = reweight * max|x|`` (Candes-Wakin-Boyd reweighting,
        normalized so the largest threshold is ``tau``): strong Bragg peaks
        are barely touched while low-level fog gets the full threshold. Plain
        L1 has to shave the peaks to reach the fog, which is why a single
        global threshold never sparsified the shared/network basis cleanly.
        """
        if reweight <= 0.0:
            return tau
        eps = reweight * mag.max().clamp_min(1e-30)
        return tau * (eps / (mag + eps))

    def fit_peak_basis(self, a_min: float = 3.0, a_max: float = 6.0,
                       n_scan: int = 60, scan_iters: int = 150,
                       iters: int = 1500, lr: float = 3e-3, clean: float = 0.1,
                       freeze_cell: bool = True, progress: bool = False) -> dict:
        """Fit the parametric peak basis (:class:`_PeakBasis`) to the current basis.

        Scans the cubic cell edge over ``[a_min, a_max]``, distilling briefly
        at each candidate (no wave optics -- this is a supervised fit to the
        basis the reconstruction already recovered, so it uses no ground
        truth), keeps the best, then refines lattice + amplitudes together.
        Installs the result as ``self.basis_inr`` so ``reconstruct(inr=True)``
        continues in the peak parameterization.

        The scan target is *cleaned* first: everything below ``clean`` times
        the maximum magnitude is zeroed. Fitting the raw basis instead
        systematically prefers a too-large cell, because a denser reciprocal
        lattice can also absorb the diffuse fog; against a cleaned target the
        extra predicted reflections have to fit zero, which they cannot do
        while also fitting the real peaks.

        Returns ``{"a": cell edge, "loss": final L2, "scan": [(a, loss), ...]}``.
        """
        target = self.masked_basis().detach().clone()
        origin = target[0, 0, 0, :].clone()
        tgt_clean = target.clone()
        tgt_clean[0, 0, 0, :] = 0.0            # fit the Bragg content only: the
        mag = tgt_clean.abs()                  # vacuum origin is ~100x the peaks
        tgt_clean[mag < clean * mag.max()] = 0.0
        scan = []
        for a in np.linspace(a_min, a_max, n_scan):
            self._inr_setup(0, 0, 0.0, space="peaks", cell=float(a), learn_cell=False)
            scan.append((float(a), self.distill_inr(target=tgt_clean, iters=scan_iters,
                                                    lr=lr, skip_origin=True)))
        a_best = min(scan, key=lambda t: t[1])[0]
        self._inr_setup(0, 0, 0.0, space="peaks", cell=a_best, learn_cell=True)
        loss = self.distill_inr(target=tgt_clean, iters=iters, lr=lr,
                                skip_origin=True, progress=progress)
        with torch.no_grad():                  # carry the learned vacuum baseline
            self.basis_inr.origin.copy_(origin.real.to(self.basis_inr.origin.dtype))
            self.basis.copy_(self._inr_basis().detach())
        # The lattice is frozen for the wave-optics phase by default: at the
        # optimizer's normal step size it runs away (measured 4.09 -> 2.83 A),
        # because every amplitude can chase a moving peak. With
        # ``freeze_cell=False`` it stays learnable and :meth:`_make_optimizer`
        # gives it a 1000x smaller step, so the data itself refines the cell --
        # worth it when the basis-fit estimate is off by a percent or more,
        # which displaces the outer reflections by a fraction of a k voxel.
        self.basis_inr.B_rec.requires_grad_(not freeze_cell)
        a_fit = float(1.0 / self.basis_inr.B_rec[0].norm())
        return {"a": a_fit, "a_scan": a_best, "loss": loss, "scan": scan}

    def adapt_peaks(self, reference: torch.Tensor | None = None,
                    prune: float = 0.02, birth: int = 0,
                    birth_frac: float = 0.1) -> dict:
        """Gaussian-splatting-style update of the peak list: release, prune, birth.

        Answers "how do we know the number of peaks is correct?" by not
        assuming it: shell tying is released (each reflection owns its
        amplitude), reflections the data does not support decay and are
        **pruned**, and if a ``reference`` volume (typically the pixel-phase
        basis) holds significant off-comb residual mass, free peaks are
        **born** at its strongest residual maxima -- catching superstructure
        reflections or second phases the assumed lattice cannot express.

        Parameters
        ----------
        reference : torch.Tensor, optional
            ``[N_kz, N_ky, N_kx, N_w]`` volume to search for missing peaks
            (unshifted order). Default: the current ``self.basis`` mirror.
        prune : float, default 0.02
            Drop reflections below this fraction of the max amplitude.
        birth : int, default 0
            Maximum number of free peaks to add (0 disables birth).
        birth_frac : float, default 0.1
            Only residual maxima above this fraction of the reference's
            strongest peak spawn a free peak.

        Returns
        -------
        dict
            ``n_peaks``, ``n_pruned``, ``n_born``.
        """
        assert isinstance(self.basis_inr, _PeakBasis), "peak basis required"
        pb = self.basis_inr
        if pb.amp.shape[0] != pb.hkl.shape[0]:
            pb.untie_shells()
        n_pruned = pb.prune(prune)
        n_born = 0
        if birth > 0:
            ref = (self.basis if reference is None else reference).detach()
            with torch.no_grad():
                res = (ref - self._inr_basis().detach()).abs().sum(-1)
                res[0, 0, 0] = 0.0
                resc = torch.fft.fftshift(res, dim=(0, 1, 2))
                # threshold against the strongest OFF-ORIGIN content (the
                # vacuum origin is ~100x the peaks and would mask everything)
                ref_off = ref.abs().sum(-1).clone()
                ref_off[0, 0, 0] = 0.0
                thr = birth_frac * ref_off.amax()
                dk = torch.tensor(self.k_sampling, device=self.device)
                ctr = torch.tensor([n // 2 for n in self.k_shape], device=self.device)
                born = []
                for _ in range(birth):
                    idx = torch.nonzero(resc == resc.max(), as_tuple=False)[0]
                    if resc[tuple(idx)] < thr:
                        break
                    born.append((idx.to(dk) - ctr) * dk)
                    # clear a 3-voxel neighborhood so the next max is distinct
                    sl = tuple(slice(max(0, int(i) - 1), int(i) + 2) for i in idx)
                    resc[sl] = 0.0
                if born:
                    pb.add_free_peaks(torch.stack(born))
                    n_born = len(born)
        # optimizer groups reference dead parameter objects after the rebuild
        self._opt = None
        return {"n_peaks": int(pb.hkl.shape[0] + pb.g_free.shape[0]),
                "n_pruned": n_pruned, "n_born": n_born}

    def distill_inr(self, target: torch.Tensor | None = None, iters: int = 800,
                    lr: float = 1e-3, skip_origin: bool = False,
                    progress: bool = False) -> float:
        """Fit the basis INR to a target volume (default: the current basis).

        The bridge of the **pixel-bootstrap** recipe for fine k grids: a short
        pixel ``reconstruct`` discovers the grains (its proximal-shrink basis
        sharpens immediately, so the orientation sweep has a clean target),
        this distills that basis into the network, and an ``inr=True``
        ``reconstruct`` then refines far past the pixel model's quality
        ceiling. Supervised on the grid, no wave optics -- seconds to a
        minute. Creates the default network via :meth:`_inr_setup` if none
        exists. Returns the final L2.

        Example (full-resolution multi-grain recipe)::

            recon.reconstruct(dp, tilts, ..., num_iters=60,
                              shrink_basis=0.3, smooth_basis=0.35,
                              cubic_symmetry=True, search_every=10)
            recon.distill_inr()
            recon.reconstruct(dp, tilts, ..., num_iters=150, inr=True,
                              shrink_basis=0.3, cubic_symmetry=True,
                              search_every=10)
        """
        if getattr(self, "basis_inr", None) is None:
            self._inr_setup(width=128, depth=3, omega=30.0, space="r", periodic=True)
        tgt = (self.masked_basis() if target is None else target).detach().clone()
        # the peak model's lattice is far more sensitive than its amplitudes
        # (moving G by a fraction of a voxel changes every peak), so it gets a
        # 10x smaller step
        cell_p = [p for n, p in self.basis_inr.named_parameters() if n == "B_rec"]
        rest = [p for n, p in self.basis_inr.named_parameters() if n != "B_rec"]
        groups = [{"params": rest, "lr": lr}]
        if cell_p:
            groups.append({"params": cell_p, "lr": lr * 0.1})
        opt = torch.optim.Adam(groups)
        bar = tqdm(range(iters), disable=not progress, desc="distill", unit="it")
        loss = torch.tensor(0.0)
        for _ in bar:
            opt.zero_grad(set_to_none=True)
            diff = self._inr_basis() - tgt
            if skip_origin:
                # the vacuum baseline is ~100x the Bragg peaks; including it
                # makes one voxel dominate the fit
                diff = diff.clone()
                diff[0, 0, 0, :] = 0.0
            loss = diff.abs().pow(2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            self.basis.copy_(self._inr_basis().detach())
        return float(loss)

    def reconstruct(
        self,
        measurements: torch.Tensor,                 # (n_tilt, n_row, n_col, det, det) intensities
        tilts_deg,
        scan_shape: tuple[int, int],
        scan_step: float | tuple[float, float] = 1.0,
        scan_origin=None,
        num_iters: int = 100,
        lr: float = 5e-3,
        superslice: int = 1,
        tilt_batch: int = 1,
        shrink_basis: float = 0.0,
        smooth_basis: float = 0.0,
        smooth_weights: float = 0.0,
        shrink_weights: float = 0.0,
        reweight_l1: float = 0.0,
        friedel_basis: bool = True,
        cubic_symmetry: bool = False,
        search_every: int = 0,
        reset_every: int = 10,
        consolidate_every: int = 0,
        phase_only: bool = True,
        beam_weight: float = 1.0,
        inr: bool = False,
        inr_lr: float = 1e-3,
        inr_prox_every: int = 0,
        inr_prox_frac: float = 0.05,
        tilt_sample: int = 0,
        resume: bool = False,
        progress: bool = True,
        print_every: int = 0,
    ) -> dict:
        """Fit basis + weights + angles to the measured tilt series (Adam).

        Full-batch amplitude (sqrt-intensity) loss, one tilt group at a time.
        The lowest-loss state is snapshotted and returned. Priors and search,
        all optional and applied each step:

        * ``shrink_basis`` -- proximal L1 on the off-origin basis (sharpens
          Bragg spots, kills k-space fog).
        * ``smooth_basis`` -- gentle reciprocal-space Gaussian (pools split
          intensity into single peaks).
        * ``smooth_weights`` -- real-space Gaussian on the weight field only
          (angles are discontinuous at grain boundaries and never smoothed).
        * ``friedel_basis`` -- project the off-origin basis onto the
          anti-Hermitian symmetry of a real potential's phase grating
          (always-true physics; on by default).
        * ``cubic_symmetry`` -- project the basis onto the cubic point group
          (see :meth:`symmetrize_basis`). Reinforces each Bragg peak with its
          whole orbit; a big win when many grains sample k-space, but it can
          hurt a single sparse grain (it symmetrizes an under-determined
          basis), so leave it off for one/few grains.
        * ``search_every`` -- every N iters run a :meth:`local_search`
          orientation sweep (exact-forward, evidence-based). Needed to discover
          the orientations of *many* grains; unnecessary (and slower) for one.
        * ``reset_every`` -- every N iters give the worst-fit voxels a cheap
          stochastic orientation/weight jump (neighbor adoption or random).
          The cheap complement to ``search_every`` -- enough on its own for
          one/few grains.
        * ``shrink_weights`` -- proximal L1 on the weight field, thresholded
          RELATIVE to the current maximum weight (the weight scale is a soft
          gauge under ``phase_only``). A particle fills a few percent of the
          volume, so this clears the low-level weight that otherwise bleeds
          into vacuum.
        * ``reweight_l1`` -- turn ``shrink_basis`` / ``shrink_weights`` into
          *iteratively reweighted* L1 (see :meth:`_l1_weighting`): the
          threshold is scaled down where the current magnitude is large, so
          Bragg peaks and material voxels keep their amplitude while fog and
          vacuum get the full threshold. The value is the reweighting floor
          ``eps`` as a fraction of the maximum magnitude (0.05 is a good
          start; 0 = plain L1).
        * ``consolidate_every`` -- every N iters project the orientation
          field onto piecewise-constant grains (:meth:`consolidate_grains`).
          A grain's voxels physically share one rotation; averaging over the
          grain cuts per-voxel orientation noise by ~sqrt(n_voxels), which
          matters most on fine k grids where a 1 deg error already shifts a
          high-order peak by a third of a voxel.
        * ``inr_prox_every`` / ``inr_prox_frac`` (INR only) -- every N iters
          soft-threshold the generated basis at ``inr_prox_frac`` of its
          maximum and re-distill the network onto the sparsified volume.
          A gradient L1 penalty cannot sparsify a smooth shared network the
          way a proximal step sparsifies free pixels; this projects the
          network itself onto the sparse set, which is how the pixel model
          keeps k-space fog out.

        ``inr=True`` swaps the explicit basis tensor for an implicit neural
        representation: an HSiren MLP (coordinates -> complex structure
        factor) is evaluated on the k grid each iteration and the generated
        volume flows through the same forward model. Weights and angles stay
        explicit per-voxel parameters, so the reset / search machinery is
        unchanged. ``friedel_basis`` / ``cubic_symmetry`` are enforced on the
        generated volume (differentiably); ``shrink_basis`` becomes an L1
        penalty on the off-origin volume; ``smooth_basis`` is ignored (the
        network is the smoothness prior). ``inr_lr`` is the network's Adam
        learning rate (``lr`` still drives weights/angles). The network
        persists on ``self.basis_inr`` and ``self.basis`` mirrors the
        generated volume, so viewers, ``local_search`` and ``simulate`` work
        as usual. Call :meth:`_inr_setup` first to choose a custom
        width/depth/omega; otherwise a default network is created.

        ``beam_weight`` scales the direct-beam pixels' contribution to the
        loss. The unscattered disk carries most of the intensity but almost
        no orientation information (it reports the projected weight), so at
        1.0 the fit is dominated by pixels that cannot distinguish grains;
        ~0.1 rebalances it toward the Bragg disks. Keep it fixed across
        resumed calls -- it changes the objective, so losses (and keep-best)
        are only comparable at one setting.

        ``tilt_sample=k`` turns each iteration into a stochastic step over
        ``k`` random tilts instead of all of them (the wave-optics forward is
        ~80% of an iteration, so this buys nearly ``n_tilt/k`` per step).
        Every 10th iteration remains full-batch: it anchors keep-best (a
        subset loss is not comparable) and the loss log. Between anchors the
        logged loss is the noisy subset mean. Most useful with ``inr=True``,
        whose shared parameterization averages the minibatch noise; the
        per-voxel reset metric simply reuses each tilt's latest residual.

        ``resume=True`` continues a previous ``reconstruct`` call: the Adam
        momentum and the accumulated loss history carry over (so ``plot_loss``
        shows the whole run and keep-best never rolls back below where the
        last call ended). The basis/weights/angles always carry over regardless
        of ``resume``; only the optimizer state and loss history need it. Use it
        to add iterations incrementally (run N, inspect, run N more) -- the same
        as one long call, but you can look in between.

        Set ``progress=False`` to hide the bar; ``print_every>0`` prints the
        loss every N iters.
        """
        # record the geometry so simulate() can reproduce the predicted patterns
        # with no arguments after reconstruction
        self._scan_geometry = {"tilts_deg": list(tilts_deg), "scan_shape": tuple(scan_shape),
                               "scan_step": scan_step, "scan_origin": scan_origin}
        pos = self.scan_positions(scan_shape, scan_step, scan_origin)
        n_row, n_col = pos.shape[:2]
        meas_amp = measurements.to(self.device).clamp_min(0).sqrt()
        jobs = [(ti, j, i) for ti in range(len(tilts_deg)) for j in range(n_row) for i in range(n_col)]
        n_dp = len(jobs)
        if inr and getattr(self, "basis_inr", None) is None:
            self._inr_setup(width=128, depth=3, omega=30.0, space="r", periodic=True)
        # eps=1e-30: the phase-object gradients are ~1e-10, so the default
        # eps=1e-8 would swamp sqrt(v) and throttle every Adam step ~100x.
        # resume=True reuses the Adam state (momentum) from the previous call
        # so a second reconstruct() continues smoothly instead of restarting
        # the optimizer; the params always carry over regardless.
        if (resume and getattr(self, "_opt", None) is not None
                and getattr(self, "_opt_inr", None) == inr):
            opt = self._opt
        else:
            opt = self._make_optimizer(lr, lr, lr, inr_lr=inr_lr if inr else None)
        self._opt = opt
        self._opt_inr = inr
        group_names = ["weights"] + (["inr" if inr else "basis"] if self.learn_basis else []) \
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
        # on resume, carry the best-loss forward so keep-best never rolls the
        # continued run back below where the previous call left off
        best = {"loss": self.best_loss if resume and hasattr(self, "best_loss")
                else float("inf"), "snap": None}

        origins = pos.reshape(-1, 3)                          # (P, 3)
        P = origins.shape[0]
        n_tilt = len(tilts_deg)
        # detector loss weighting: down-weight the unscattered disk (it holds
        # most of the intensity but no orientation information). Normalized to
        # unit mean so the loss stays on the same scale as the unweighted one.
        lw = None
        if beam_weight != 1.0:
            kv, ku = torch.meshgrid(self.det_kv, self.det_ku, indexing="ij")
            disk = torch.sqrt(kv ** 2 + ku ** 2) <= self.probe_k_max * 1.2
            lw = torch.where(disk, torch.as_tensor(float(beam_weight)),
                             torch.as_tensor(1.0)).to(self.device)
            lw = (lw / lw.mean()).to(torch.float64)
        pbar = tqdm(range(num_iters), disable=not progress, desc="reconstruct", unit="it")
        for it in pbar:
            opt.zero_grad(set_to_none=True)
            total = 0.0
            basis_leaf = None
            if inr:
                # cosine-decay the network lr to a 5% floor over the call: the
                # shared parameterization thrashes late in the run at a fixed
                # step size (every basis element moves with every step), while
                # the per-voxel params keep their constant lr.
                fac = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * it / max(1, num_iters)))
                for g in opt.param_groups:
                    if g.get("name") == "inr":
                        g["lr"] = inr_lr * fac
            if inr:
                # one network evaluation per iteration: the generated volume is
                # detached into a leaf that every tilt group backprops into, and
                # the accumulated leaf gradient is relayed through the network
                # once at the end (otherwise each group's backward would need
                # its own network graph). self.basis mirrors the volume so the
                # sweep/reset machinery and the viewers see the current state.
                gen_basis = self._inr_basis(friedel_basis, cubic_symmetry)
                with torch.no_grad():
                    self.basis.copy_(gen_basis.detach())
                basis_leaf = gen_basis.detach().requires_grad_(True)
            # tilt selection: full batch, or (tilt_sample) a random subset with
            # a full-batch anchor every 10th iteration for keep-best/logging.
            full_iter = (not tilt_sample or tilt_sample >= n_tilt
                         or it % 10 == 0 or it == num_iters - 1)
            if full_iter:
                t_sel, grp_sz, n_eff = list(range(n_tilt)), max(1, tilt_batch), n_dp
            else:
                t_sel = sorted(torch.randperm(n_tilt, generator=gen)[:tilt_sample].tolist())
                # single-tilt groups keep the geometry cache to n_tilt entries
                # (random subsets as batch keys would grow it combinatorially)
                grp_sz, n_eff = 1, len(t_sel) * P
            for c0 in range(0, len(t_sel), grp_sz):
                # tilts are processed in groups: one autograd graph (and one
                # backward) per group. tilt_batch trades memory for speed;
                # gradients still accumulate into one optimizer step.
                t_grp = t_sel[c0:c0 + grp_sz]
                basis = basis_leaf if inr else self.masked_basis()
                R_all = self.rotation_matrices()
                Psi = self.forward_tilts(origins, [float(tilts_deg[ti]) for ti in t_grp],
                                         basis, R_all, self.weights,
                                         phase_only=phase_only,
                                         superslice=superslice)                       # (T,P,det,det)
                tgt = meas_amp[t_grp[0]:t_grp[-1] + 1].reshape(len(t_grp), P, *self.det_shape) \
                    if t_grp == list(range(t_grp[0], t_grp[-1] + 1)) else \
                    meas_amp[t_grp].reshape(len(t_grp), P, *self.det_shape)
                err = (Psi.abs() - tgt) ** 2
                tl = (err if lw is None else err * lw).mean(dim=(2, 3))               # (T,P)
                (tl.sum() / n_eff).backward()
                for j, ti in enumerate(t_grp):
                    res_per_dp[ti * P:(ti + 1) * P] = tl[j].detach()
                total += float(tl.sum())
            if inr:
                if shrink_basis > 0.0 and basis_leaf.grad is not None:
                    # proximal L1 has no meaning for network weights; apply the
                    # sparsity prior as a penalty routed through the same leaf,
                    # auto-scaled to the data gradient (a fixed coefficient is
                    # either invisible or crushes the signal): tau = a fraction
                    # of the strong data gradients, so data-defended peaks
                    # out-pull the penalty while unconstrained fog decays.
                    # smoothed |.| -- exact complex abs has NaN grad at the
                    # (masked, exactly zero) off-sphere entries.
                    gmag = basis_leaf.grad.abs()
                    tau = shrink_basis * 0.1 * torch.quantile(
                        gmag[gmag > 0].to(torch.float32), 0.9).to(gmag.dtype)
                    off = (basis_leaf.real ** 2 + basis_leaf.imag ** 2 + 1e-30).sqrt()
                    tau = self._l1_weighting(tau, off.detach(), reweight_l1)
                    pen = (tau * off).sum() - (tau * off)[0, 0, 0, :].sum() \
                        if torch.is_tensor(tau) else tau * (off.sum() - off[0, 0, 0, :].sum())
                    pen.backward()
                if basis_leaf.grad is not None:
                    gen_basis.backward(basis_leaf.grad)

            mean_loss = total / n_eff
            losses.append(mean_loss)
            lrs.append([float(g["lr"]) for g in opt.param_groups])
            # subset losses are not comparable to the full objective, so only
            # full-batch iterations can claim the keep-best snapshot
            if full_iter and np.isfinite(mean_loss) and mean_loss < best["loss"]:
                best = {"loss": mean_loss, "snap": self._snapshot()}

            opt.step()
            self._sanitize(opt, gen)     # repair NaN/Inf params (degenerate SVD backward)
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
            if shrink_weights > 0.0:
                # proximal L1 on the weight field. The threshold is relative
                # to the current max weight: under phase_only the weight scale
                # is a soft gauge (learned weights sit ~0.03, not GT's 1.0), so
                # an absolute threshold either does nothing or erases the field.
                with torch.no_grad():
                    W = self.weights
                    mag = W.detach().abs()
                    tau = shrink_weights * lr * mag.max().clamp_min(1e-30)
                    tau = self._l1_weighting(tau, mag, reweight_l1)
                    W.copy_(torch.sign(W) * torch.clamp(W.abs() - tau, min=0.0))
            if cubic_symmetry and self.learn_basis and not inr:
                # project the basis onto cubic point-group symmetry each step:
                # every Bragg peak is forced equal to its whole 24-fold orbit,
                # so a peak seen by any grain/tilt is filled in for all
                # equivalents and the rotation search sees a full clean target.
                # Pins the basis to the cardinal-cubic gauge from the start.
                self.symmetrize_basis()
            if friedel_basis and self.learn_basis and not inr:
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
            if smooth_basis > 0.0 and self.learn_basis and not inr:
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
            if shrink_basis > 0.0 and self.learn_basis and not inr:
                # proximal L1 on the basis' off-origin content: a structure
                # factor is a few sharp Bragg spots, so soft-thresholding kills
                # the k-space fog that otherwise fits the data with rings
                # instead of spots (the same cure the explicit-6D model needed).
                with torch.no_grad():
                    mag = self.basis.abs()
                    tau = self._l1_weighting(shrink_basis * lr, mag, reweight_l1)
                    keep = self.basis[0, 0, 0, :].clone()
                    self.basis.copy_(self.basis / mag.clamp_min(1e-30)
                                     * torch.clamp(mag - tau, min=0.0))
                    self.basis[0, 0, 0, :] = keep          # origin (vacuum) untouched

            n_res = 0
            # cheap stochastic escape (complements search_every): every
            # reset_every iters the worst voxels (by residual error along their
            # rays) get an orientation + weight jump -- adopt a random real-space
            # neighbor (grain growth) or a fresh random orientation. The count
            # tapers to zero over the run (explore early, refine late), and
            # settled vacuum voxels are de-prioritized so jumps hit misfit
            # material. Skipped when angles are frozen.
            if self.learn_angles and reset_every and (it + 1) % reset_every == 0 and it < num_iters - 1:
                taper = 1.0 - it / num_iters
                n_res = int(round(0.1 * self.n_voxels * taper))
                if n_res > 0:
                    err_vox = Wmat.t() @ res_per_dp
                    wmag = self.weights.detach().abs().sum(-1).flatten()
                    vac = wmag < 0.15 * wmag.max().clamp_min(1e-30)
                    err_vox = err_vox * torch.where(vac, 0.2, 1.0)
                    bad = torch.topk(err_vox, n_res).indices
                    Nz, Ny, Nx = self.real_shape
                    with torch.no_grad():
                        Wf = self.weights.reshape(self.n_voxels, self.num_structures)
                        w_mean = self.weights.mean()
                        for v in bad.tolist():
                            if float(torch.rand(1, generator=gen)) < 0.5:
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
                            else:
                                self.angles.M[v] = (torch.eye(3)
                                                    + 0.1 * torch.randn(3, 3, generator=gen)).to(self.device)
                                Wf[v] = w_mean
                        self._reset_optimizer_state(opt, self.angles.M, bad)
                        self._reset_optimizer_state(opt, self.weights, bad)
            if (inr and inr_prox_every and self.learn_basis
                    and (it + 1) % inr_prox_every == 0 and it < num_iters - 1):
                # project the NETWORK onto the sparse set: soft-threshold the
                # generated volume, then re-fit the net to it. The Adam state
                # is dropped (the parameters have moved discontinuously).
                with torch.no_grad():
                    B = self._inr_basis(friedel_basis, cubic_symmetry).detach()
                    keep = B[0, 0, 0, :].clone()
                    mag = B.abs()
                    tau = inr_prox_frac * mag.max().clamp_min(1e-30)
                    B = B / mag.clamp_min(1e-30) * torch.clamp(mag - tau, min=0.0)
                    B[0, 0, 0, :] = keep
                self.distill_inr(target=B, iters=150, lr=inr_lr)
                for grp in opt.param_groups:      # net params moved discontinuously
                    if grp.get("name") == "inr":
                        for prm in grp["params"]:
                            opt.state.pop(prm, None)
            if (search_every and self.learn_angles and (it + 1) % search_every == 0
                    and it < num_iters - 1):
                # exact-forward orientation sweep: each voxel keeps the best of
                # its neighbors / jittered-incumbent / random rotations judged
                # on only the rays that strike it. Evidence-based, so it finds
                # grains the blind reset cannot; the changed voxels' stale Adam
                # moments are cleared. Each sweep transiently raises the loss
                # (the basis/weights must re-adapt to the new orientations)
                # before falling below the previous best -- that "spike" IS the
                # grain discovery, so it runs to the end; the lowest-loss state
                # over the whole run is snapshotted and returned.
                self.local_search(measurements, tilts_deg, scan_shape, scan_step,
                                  accept=0.95, order="error", seed=it, progress=progress)
                allv = torch.arange(self.n_voxels, device=self.device)
                self._reset_optimizer_state(opt, self.angles.M, allv)
                self._reset_optimizer_state(opt, self.weights, allv)
            if (consolidate_every and self.learn_angles
                    and (it + 1) % consolidate_every == 0 and it < num_iters - 1):
                # piecewise-constant projection: every grain's voxels share
                # one rotation (see consolidate_grains)
                self.consolidate_grains()
                allv = torch.arange(self.n_voxels, device=self.device)
                self._reset_optimizer_state(opt, self.angles.M, allv)
            if print_every and (it % print_every == 0 or it == num_iters - 1):
                print(f"  it {it:4d}  loss {mean_loss:.4e}  best {best['loss']:.4e}"
                      + (f"  reset {n_res}" if n_res else ""), flush=True)
            pbar.set_postfix(loss=f"{mean_loss:.3e}", best=f"{best['loss']:.3e}")

        if best["snap"] is not None:
            self._restore(best["snap"])              # return the best-ever state
        # accumulate the loss / lr history across resumed calls so plot_loss
        # shows the whole run, not just the latest call
        if resume and getattr(self, "losses", None):
            self.losses = self.losses + losses
            self.lrs = (self.lrs + lrs) if getattr(self, "lrs", None) else lrs
        else:
            self.losses = losses
            self.lrs = lrs
        self.best_loss = best["loss"]
        self.lr_group_names = group_names
        return {"losses": self.losses, "lrs": self.lrs, "best_loss": best["loss"],
                "basis": self.masked_basis().detach(),
                "weights": self.weights.detach(),
                "rotations": self.rotation_matrices().detach()}

    def _local_ray_map(self, tilts_deg, scan_shape, scan_step=1.0, w_min=0.45):
        """voxel -> list of (tilt_index, probe_index) rays that intercept it.

        A ray counts for a voxel when any slice sample gives the voxel a
        trilinear weight above ``w_min``. 0.45 (not 0.5): even-N_z slabs put
        ray samples at half-integer z, where the two co-dominant corners each
        carry exactly 0.5.
        """
        pos = self.scan_positions(scan_shape, scan_step).reshape(-1, 3)
        vox_rays = [set() for _ in range(self.n_voxels)]
        for ti, t in enumerate(tilts_deg):
            geo = self._tilt_geometry(pos, float(t))
            for vidx, tw in geo["slices"]:
                for p, v in zip(*torch.nonzero(tw.real > w_min, as_tuple=True)):
                    vox_rays[int(vidx[p, v])].add((ti, int(p)))
        return [sorted(s) for s in vox_rays], pos

    def _orientation_planes(self, basis_sum_c, R_bank, U, V, ku, kv):
        """Centered-basis plane samples for (n_R x n_tilt) rotation/axis pairs."""
        Nkz, Nky, Nkx = self.k_shape
        nR, nT = len(R_bank), U.shape[0]
        R = torch.as_tensor(np.asarray(R_bank), dtype=torch.float32, device=self.device)
        u_b = torch.einsum("rij,ti->rtj", R, U)
        v_b = torch.einsum("rij,ti->rtj", R, V)
        dk = torch.tensor(self.k_sampling, dtype=torch.float32, device=self.device)
        kxyz = (ku[None, None, ..., None] * u_b[:, :, None, None, :]
                + kv[None, None, ..., None] * v_b[:, :, None, None, :])
        ctr = torch.tensor([Nkz // 2, Nky // 2, Nkx // 2], dtype=torch.float32,
                           device=self.device)
        c = kxyz / dk + ctr
        grid = torch.stack((
            2.0 * c[..., 2] / (Nkx - 1.0) - 1.0,
            2.0 * c[..., 1] / (Nky - 1.0) - 1.0,
            2.0 * c[..., 0] / (Nkz - 1.0) - 1.0,
        ), dim=-1).reshape(nR * nT, 1, *ku.shape, 3).to(torch.float32)
        bre = basis_sum_c.real[None, None].expand(nR * nT, 1, -1, -1, -1).to(torch.float32)
        bim = basis_sum_c.imag[None, None].expand(nR * nT, 1, -1, -1, -1).to(torch.float32)
        sre = F.grid_sample(bre, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=True)
        sim = F.grid_sample(bim, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=True)
        return (sre + 1j * sim)[:, 0, 0].reshape(nR, nT, *ku.shape)

    def local_search(
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
