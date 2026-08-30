"""Orientation mapping of crystalline 4D-STEM data.

OrientationMap matches measured Bragg peaks (a quantem Vector) against a
library of simulated kinematical patterns from a Crystal, using sparse polar
correlation (Ophus et al., Microsc. Microanal. 28, 390 (2022)) implemented as
batched torch operations.

The method:

1. Sample zone axes over the symmetry-reduced fundamental wedge (or the
   hemisphere), build a polar-coordinate reference library P(zone, shell,
   gamma) where shells are the reciprocal-lattice radii of the crystal.
2. Convert measured peaks at each probe position into the same sparse polar
   representation X(shell, gamma).
3. Correlate over in-plane angle gamma by FFT, over all zones at once, using
   one batched matrix multiplication per gamma frequency. The mirror channel
   (conjugate FFT) tests inversion-related orientations at no library cost.
4. Optionally refine the best zone axes on a finer local grid.

Orientations are unit quaternions; see quantem.diffraction.rotations.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from quantem.core.datastructures.vector import Vector
from quantem.core.io.serialize import AutoSerialize
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.rotations import (
    misorientation_angle_deg,
    qconj,
    qmult,
    qnormalize,
    qrotate,
    quat_from_axis_angle,
    quat_from_zone_axis,
    sample_zone_axes,
)


def fibonacci_hemisphere(n_points: int, dtype=torch.float64) -> torch.Tensor:
    """Spherical Fibonacci sampling of the upper hemisphere, (N, 3)."""
    i = torch.arange(n_points, dtype=dtype) + 0.5
    z = i / n_points  # (0, 1): upper hemisphere
    phi = i * (np.pi * (3 - np.sqrt(5)))
    r = torch.sqrt(1 - z**2)
    return torch.stack((r * torch.cos(phi), r * torch.sin(phi), z), dim=-1)


class OrientationMap(AutoSerialize):
    """Match crystal orientations to Bragg peaks at every probe position.

    Workflow::

        om = OrientationMap.from_vectors(peaks, crystal, energy_ev=200e3)
        om.build_plan(angle_step_zone_axis_deg=2.0, angle_step_in_plane_deg=2.0)
        om.match_orientations(num_matches=1)
        om.plot_orientation()

    The object is both the engine and the result: after `match`, `quats`
    holds (R, C, M, 4) orientation quaternions, `corr` the correlation
    scores, and `mirror` the inversion flags.
    """

    _token = object()

    def __init__(
        self,
        peaks: Vector,
        crystal: Crystal,
        energy_ev: float,
        _token: object | None = None,
    ):
        if _token is not self._token:
            raise RuntimeError("Use OrientationMap.from_vectors() to construct.")
        self.peaks = peaks
        self.crystal = crystal
        self.energy_ev = float(energy_ev)
        self.wavelength = electron_wavelength_angstrom(energy_ev)

        # plan state
        self.zone_axes: torch.Tensor | None = None
        self.zone_quats: torch.Tensor | None = None
        self.plan_fft: torch.Tensor | None = None
        self.shell_radii: torch.Tensor | None = None

        # results
        self.quats: torch.Tensor | None = None
        self.corr: torch.Tensor | None = None
        self.corr_second: torch.Tensor | None = None
        self.reliability: torch.Tensor | None = None
        self.mirror: torch.Tensor | None = None

    @classmethod
    def from_vectors(
        cls,
        peaks: Vector,
        crystal: Crystal,
        energy_ev: float = 300e3,
    ) -> "OrientationMap":
        """Create from detected Bragg peaks.

        Parameters
        ----------
        peaks : Vector
            Ragged peak table over scan positions with fields including
            ('qx', 'qy', 'intensity') in calibrated 1/Angstrom units.
        crystal : Crystal
            Candidate crystal with structure factors already calculated.
        energy_ev : float, default=300e3
            Beam energy in eV.
        """
        if crystal.g_vec is None:
            raise RuntimeError("Run crystal.calculate_structure_factors() first.")
        return cls(peaks, crystal, energy_ev, _token=cls._token)

    # ------------------------------------------------------------------
    # orientation plan
    # ------------------------------------------------------------------

    def build_plan(
        self,
        angle_step_zone_axis_deg: float = 2.0,
        angle_step_in_plane_deg: float = 2.0,
        corr_kernel_size: float = 0.05,
        sigma_excitation: float = 0.04,
        power_radial: float = 1.0,
        power_intensity: float = 0.25,
        tol_shell_distance: float = 0.01,
        detector_q_max: float | tuple[float, float] | str | None = "auto",
        device: str | torch.device = "cpu",
        verbose: bool = True,
    ) -> "OrientationMap":
        """Build the polar correlation library over the fundamental wedge.

        Parameters
        ----------
        angle_step_zone_axis_deg : float, default=2.0
            Angular step between sampled zone axes.
        angle_step_in_plane_deg : float, default=2.0
            Angular step of the in-plane (gamma) axis; the number of gamma
            samples is round(360 / step).
        corr_kernel_size : float, default=0.05
            Correlation kernel size delta (1/Angstroms): azimuthal extent of
            each reference peak and radial tolerance for shell assignment.
        sigma_excitation : float, default=0.04
            Excitation error envelope of the library (1/Angstroms). Keep this
            about 2x the physical excitation tolerance: orientations halfway
            between sampled zones shift s_g by ~ (step/2) * k, and a wider
            envelope keeps their library intensities from collapsing.
        power_radial, power_intensity : float
            Weighting prefactor q^power_radial * |V_g|^power_intensity for
            library peaks. power_intensity=0 matches on positions only
            (best for strongly dynamical data).
        tol_shell_distance : float, default=0.01
            Reciprocal lattice radii closer than this merge into one shell.
        detector_q_max : float | tuple | "auto" | None, default="auto"
            Half-width of the square detector (1/Angstroms), scalar or
            (row_max, col_max). Library reflections beyond the detector edge
            cannot be measured, and which ones fall off depends on the
            in-plane rotation; the correlation is normalized by the masked
            template norm at every in-plane angle, so orientations with
            strong reflections outside the detector are not penalized.
            "auto" measures the detector footprint from the peaks themselves
            (largest |q| along each detector axis, undoing any
            detector-to-scan rotation recorded on the peaks). None disables
            the correction.
        device : str | torch.device, default="cpu"
            Device for the library and the correlation compute.
        verbose : bool, default=True
            Print the symmetry actually used for matching (including any
            pseudo-symmetry reduction) and the plan size.
        """
        crystal = self.crystal
        self.device = torch.device(device)
        self.corr_kernel_size = float(corr_kernel_size)
        self.sigma_excitation = float(sigma_excitation)
        self.power_radial = float(power_radial)
        self.power_intensity = float(power_intensity)

        # zone axis sampling over the matching (pseudo-symmetry-reduced) wedge
        wedge = crystal.zone_axis_wedge()
        if wedge is None:
            n_zones = int(np.ceil(2 * np.pi / np.deg2rad(angle_step_zone_axis_deg) ** 2))
            za = fibonacci_hemisphere(n_zones)
        else:
            za, _ = sample_zone_axes(wedge, angle_step_zone_axis_deg)
        self.zone_axes = za
        self.zone_quats = quat_from_zone_axis(za)
        self.zone_step_deg = float(angle_step_zone_axis_deg)

        # symmetry-complete neighbor sets for sub-grid zone refinement: a
        # zone on the wedge boundary only has in-wedge grid neighbors on one
        # side, and a one-sided correlation centroid would drag it inward;
        # the symmetry images of the grid across the boundary restore the
        # missing side.
        from quantem.diffraction.rotations import quat_to_matrix

        Rs = quat_to_matrix(crystal.sym_quats_matching)
        images = torch.einsum("sij,zj->szi", Rs, za)
        images = torch.cat([images, -images], dim=0).reshape(-1, 3)  # (S2*Z, 3)
        img_zone = torch.arange(za.shape[0]).repeat(
            2 * crystal.sym_quats_matching.shape[0]
        )
        # deduplicate coincident image positions (keep one per position/zone)
        key = torch.cat(
            [torch.round(images / 1e-6) * 1e-6, img_zone[:, None].to(images.dtype)],
            dim=1,
        )
        _, first = np.unique(key.numpy(), axis=0, return_index=True)
        images = images[torch.as_tensor(np.sort(first))]
        img_zone = img_zone[torch.as_tensor(np.sort(first))]

        cos_lim = np.cos(np.deg2rad(1.6 * self.zone_step_deg))
        nbr_idx_list, nbr_pos_list = [], []
        dots = images @ za.T  # (M, Z)
        for i in range(za.shape[0]):
            sel = torch.nonzero(dots[:, i] > cos_lim).squeeze(1)
            nbr_idx_list.append(img_zone[sel])
            nbr_pos_list.append(images[sel])
        K = max(len(v) for v in nbr_idx_list)
        Z = za.shape[0]
        self.zone_nbr_idx = torch.zeros((Z, K), dtype=torch.long)
        self.zone_nbr_pos = torch.zeros((Z, K, 3), dtype=torch.float64)
        self.zone_nbr_valid = torch.zeros((Z, K), dtype=torch.bool)
        for i, (idx, pos) in enumerate(zip(nbr_idx_list, nbr_pos_list)):
            k = len(idx)
            self.zone_nbr_idx[i, :k] = idx
            self.zone_nbr_pos[i, :k] = pos
            self.zone_nbr_valid[i, :k] = True

        # radial shells from unique reciprocal lattice vector lengths
        g_len = crystal.g_len
        radii = torch.unique(torch.round(g_len / tol_shell_distance) * tol_shell_distance)
        self.shell_radii = radii
        self.num_gamma = int(round(360 / angle_step_in_plane_deg))
        self.gamma = torch.linspace(
            0, 2 * np.pi, self.num_gamma + 1, dtype=torch.float64
        )[:-1]

        plan = self._build_reference(self.zone_quats)
        # store conj(fft) along gamma so matching is a single complex matmul
        self.plan_fft = torch.conj(torch.fft.fft(plan, dim=-1)).to(self.device)

        # square-detector aperture correction: the masked template norm at
        # every in-plane shift is the circular correlation of the squared
        # plan with the polar detector mask. The mask lives in the DETECTOR
        # frame: any detector-to-scan rotation recorded on the peaks rotates
        # the square aperture in the calibrated (qx, qy) frame.
        rot_deg = float(self.peaks.metadata.get("rotation_ccw_deg", 0.0) or 0.0)
        if isinstance(detector_q_max, str) and detector_q_max == "auto":
            flat = self.peaks.select_fields("qx", "qy", "intensity").flatten()
            if flat.shape[0] == 0:
                detector_q_max = None
            else:
                th_b = np.deg2rad(-rot_deg)
                rb = np.array(
                    [[np.cos(th_b), -np.sin(th_b)], [np.sin(th_b), np.cos(th_b)]]
                )
                det_rc = flat[:, :2] @ rb.T
                detector_q_max = (
                    float(np.abs(det_rc[:, 0]).max()) + self.corr_kernel_size,
                    float(np.abs(det_rc[:, 1]).max()) + self.corr_kernel_size,
                )
        if detector_q_max is not None:
            if np.isscalar(detector_q_max):
                qx_max = qy_max = float(detector_q_max)
            else:
                qx_max, qy_max = (float(v) for v in detector_q_max)
            r = self.shell_radii[:, None]
            g = self.gamma[None, :] - np.deg2rad(rot_deg)
            mask = (
                (torch.abs(r * torch.cos(g)) <= qx_max)
                & (torch.abs(r * torch.sin(g)) <= qy_max)
            ).to(torch.float64)
            self.detector_mask = mask  # (S, G)
            plan_sq_fft = torch.conj(torch.fft.fft(plan**2, dim=-1))
            mask_fft = torch.fft.fft(mask, dim=-1)
            # norm^2 per (zone, shift), direct and mirrored channels
            n2 = torch.fft.ifft(
                torch.einsum("zsg,sg->zg", plan_sq_fft, mask_fft), dim=-1
            ).real.clamp_min(0)
            n2_m = torch.fft.ifft(
                torch.einsum("zsg,sg->zg", plan_sq_fft, torch.conj(mask_fft)), dim=-1
            ).real.clamp_min(0)
            # fraction of template weight on the detector; used to suppress
            # zones that are mostly unmeasurable at a given rotation
            full = (plan**2).sum(dim=(1, 2))[:, None].clamp_min(1e-12)
            self.plan_norm_shift = torch.stack(
                [torch.sqrt(n2), torch.sqrt(n2_m)]
            ).to(self.device)  # (2, Z, G)
            self.plan_frac_shift = torch.stack([n2 / full, n2_m / full]).to(self.device)
        else:
            self.detector_mask = None
            self.plan_norm_shift = None
            self.plan_frac_shift = None
        if verbose:
            print(crystal.symmetry_summary())
            print(
                "  orientation plan %d zone axes x %d in-plane angles, "
                "%d radial shells"
                % (
                    self.zone_axes.shape[0],
                    self.gamma.shape[0],
                    self.shell_radii.shape[0],
                )
            )
        return self

    def _deposit_polar(
        self,
        qr: torch.Tensor,
        qphi: torch.Tensor,
        amp: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Deposit peaks into a polar image with the shared correlation kernel.

        Every peak spreads as a Gaussian of width delta in both the radial
        direction (across shells) and arc length (along gamma). The library
        and the experimental patterns use this same kernel, so the normalized
        correlation of a pattern with itself is exactly 1.

        Parameters
        ----------
        qr, qphi, amp : torch.Tensor
            Peak radii, azimuths, amplitudes, flat (K,).
        out : torch.Tensor
            (S, G) accumulator, modified in place.
        """
        radii = self.shell_radii.to(qr.dtype)
        delta = self.corr_kernel_size
        G = self.num_gamma

        dr = qr[:, None] - radii[None, :]  # (K, S)
        k_idx, s_idx = torch.nonzero(dr.abs() < 3 * delta, as_tuple=True)
        if k_idx.numel() == 0:
            return out
        w_r = torch.exp(-(dr[k_idx, s_idx] ** 2) / (2 * delta**2)) * amp[k_idx]

        gamma = self.gamma.to(qr.dtype)
        dg = qphi[k_idx, None] - gamma[None, :]
        dg = (dg + np.pi) % (2 * np.pi) - np.pi
        arc = dg * qr[k_idx, None]
        w = w_r[:, None] * torch.exp(-(arc**2) / (2 * delta**2))
        out.index_add_(0, s_idx, w)
        return out

    def _build_reference(self, zone_quats: torch.Tensor) -> torch.Tensor:
        """Polar reference library (Z, S, G) for the given zone-axis quats."""
        crystal = self.crystal
        lam = self.wavelength
        g = crystal.g_vec  # (N, 3)
        delta = self.corr_kernel_size

        gr = qrotate(zone_quats[:, None, :], g[None, :, :])  # (Z, N, 3)
        gz = gr[..., 2]
        g2 = (gr**2).sum(-1)
        s_g = (2 * gz - lam * g2) / (2 - 2 * lam * gz)
        amp = torch.exp(-(s_g**2) / (2 * self.sigma_excitation**2))
        amp = amp * (s_g.abs() < delta * 4)

        weight = (
            crystal.g_len**self.power_radial
            * crystal.struct_factors_int**self.power_intensity
        )
        vals = amp * weight[None, :]  # (Z, N)
        qr = torch.hypot(gr[..., 0], gr[..., 1])
        qphi = torch.atan2(gr[..., 1], gr[..., 0])

        Z = zone_quats.shape[0]
        plan = torch.zeros((Z, self.shell_radii.shape[0], self.num_gamma), dtype=torch.float64)
        for z in range(Z):
            keep = vals[z] > 1e-8
            self._deposit_polar(qr[z, keep], qphi[z, keep], vals[z, keep], plan[z])

        norm = torch.linalg.norm(plan.reshape(Z, -1), dim=1).clamp_min(1e-12)
        return plan / norm[:, None, None]

    # ------------------------------------------------------------------
    # experimental polar images
    # ------------------------------------------------------------------

    def _polar_image(
        self, qx: torch.Tensor, qy: torch.Tensor, intensity: torch.Tensor
    ) -> torch.Tensor:
        """Sparse polar image (S, G) of one measured pattern."""
        qr = torch.hypot(qx, qy)
        qphi = torch.atan2(qy, qx)
        amp = intensity.clamp_min(0) ** (self.power_intensity) * qr**self.power_radial
        out = torch.zeros(
            (self.shell_radii.shape[0], self.num_gamma), dtype=torch.float64
        )
        return self._deposit_polar(qr, qphi, amp, out)

    # ------------------------------------------------------------------
    # matching
    # ------------------------------------------------------------------

    def match_orientations(
        self,
        num_matches: int = 1,
        include_mirror: bool = True,
        min_number_peaks: int = 3,
        min_angle_between_matches_deg: float = 15.0,
        subpixel_gamma: bool = True,
        subpixel_zone: bool = True,
        min_detector_fraction: float = 0.3,
        batch_size: int = 128,
        progress_bar: bool = True,
    ) -> "OrientationMap":
        """Match all probe positions against the orientation plan.

        Patterns are processed in batches: the polar images are stacked, and
        the correlation over all zones and in-plane angles reduces to one
        complex matrix product per gamma frequency plus a batched inverse FFT.

        Correlation scores are normalized to [0, 1]: the library slices are
        unit vectors and the experimental polar image is divided by its own
        norm, so `corr` is a cosine similarity comparable across patterns,
        crystals, and datasets. After the best match, the highest correlation
        among zone axes at least `min_angle_between_matches_deg` away is
        stored in `corr_second`; `reliability = corr - corr_second` is the
        primary confidence metric.

        Parameters
        ----------
        num_matches : int, default=1
            Number of orientations to return per probe position; matches
            after the first suppress zones within
            `min_angle_between_matches_deg` of earlier matches.
        include_mirror : bool, default=True
            Also correlate against the in-plane mirrored pattern, testing
            inversion-related (opposite hemisphere) zone axes at no library
            cost. Exact in the flat-Ewald / Friedel limit.
        min_number_peaks : int, default=3
            Skip positions with fewer detected peaks.
        min_angle_between_matches_deg : float, default=15.0
            Exclusion radius (degrees, zone-axis distance) around earlier
            matches, both for later matches and for the second-best score
            used in `reliability`.
        subpixel_gamma : bool, default=True
            Parabolic sub-bin refinement of the in-plane angle.
        subpixel_zone : bool, default=True
            Sub-grid refinement of the zone axis: the correlation-weighted
            centroid of the best zone and its grid neighbors. Removes the
            zone-axis quantization of the plan (the in-plane angle is
            already continuous through subpixel_gamma).
        batch_size : int, default=128
            Number of patterns correlated at once.
        """
        if self.plan_fft is None:
            raise RuntimeError("Run build_plan() first.")
        peaks = self.peaks
        shape = peaks.shape
        R, C = shape[0], shape[1]
        M = num_matches
        device = self.device
        G = self.num_gamma
        Z = self.zone_axes.shape[0]

        quats = torch.zeros((R, C, M, 4), dtype=torch.float64)
        quats[..., 0] = 1.0
        corr_out = torch.zeros((R, C, M), dtype=torch.float64)
        corr_second = torch.zeros((R, C), dtype=torch.float64)
        mirror_out = torch.zeros((R, C, M), dtype=torch.bool)

        fields = peaks.fields
        ix = [fields.index(f) for f in ("qx", "qy", "intensity")]

        # zone-pair angular distances, for the exclusion ball around matches
        za = self.zone_axes.to(device)
        zone_ang = torch.rad2deg(torch.acos((za @ za.T).clamp(-1, 1)))  # (Z, Z)

        plan_fft = self.plan_fft  # (Z, S, G) complex
        valid_rc = [
            (rx, ry)
            for rx, ry in np.ndindex(R, C)
            if peaks[rx, ry].array.shape[0] >= min_number_peaks
        ]
        batches = [
            valid_rc[i : i + batch_size] for i in range(0, len(valid_rc), batch_size)
        ]
        if progress_bar:
            batches = tqdm(batches, desc=f"matching {self.crystal.name}")

        gamma_grid = self.gamma
        for batch in batches:
            ims = []
            for rx, ry in batch:
                data = peaks[rx, ry].array
                qx = torch.as_tensor(data[:, ix[0]], dtype=torch.float64)
                qy = torch.as_tensor(data[:, ix[1]], dtype=torch.float64)
                ii = torch.as_tensor(data[:, ix[2]], dtype=torch.float64)
                ims.append(self._polar_image(qx, qy, ii))
            im_stack = torch.stack(ims).to(device)
            norms = torch.linalg.norm(im_stack.reshape(len(ims), -1), dim=1).clamp_min(
                1e-12
            )
            im_fft = torch.fft.fft(im_stack, dim=-1)  # (B, S, G)

            # contract shells: (B, Z, G) per channel
            cc = torch.einsum("zsg,bsg->bzg", plan_fft, im_fft)
            channels = [cc]
            if include_mirror:
                channels.append(torch.einsum("zsg,bsg->bzg", plan_fft, torch.conj(im_fft)))
            corr = torch.fft.ifft(torch.stack(channels, dim=1), dim=-1).real
            # normalize: library slices are unit vectors, so dividing by the
            # experimental norm makes corr a cosine similarity in [0, 1]
            corr = corr / norms[:, None, None, None]
            if self.plan_norm_shift is not None:
                # square-detector correction: renormalize by the on-detector
                # template norm at each in-plane shift, and suppress
                # rotations where most of the template is unmeasurable
                n_ch = corr.shape[1]
                corr = corr / self.plan_norm_shift[None, :n_ch].clamp_min(1e-3)
                corr = corr.masked_fill(
                    self.plan_frac_shift[None, :n_ch] < min_detector_fraction, 0.0
                )
            # corr: (B, ch, Z, G)
            B = corr.shape[0]

            for m in range(M):
                if m > 0:
                    # suppress zones near earlier matches, per pattern
                    for b in range(B):
                        for mm in range(m):
                            rx, ry = batch[b]
                            # zone index of previous match not stored; use angle
                            # to previous zone axis
                            zprev = self._zprev[b][mm]
                            corr[b, :, zone_ang[zprev] < min_angle_between_matches_deg, :] = -torch.inf
                flat_idx = corr.reshape(B, -1).argmax(dim=1)
                n_ch = corr.shape[1]
                ch_i = flat_idx // (Z * G)
                z_i = (flat_idx // G) % Z
                g_i = flat_idx % G
                c_val = corr.reshape(B, -1).gather(1, flat_idx[:, None]).squeeze(1)

                gamma = gamma_grid[g_i.cpu()].clone()
                if subpixel_gamma:
                    b_ar = torch.arange(B, device=corr.device)
                    c1 = c_val
                    c0 = corr[b_ar, ch_i, z_i, (g_i - 1) % G]
                    c2 = corr[b_ar, ch_i, z_i, (g_i + 1) % G]
                    denom = 4 * c1 - 2 * c0 - 2 * c2
                    dg = torch.where(
                        denom.abs() > 1e-12,
                        (c2 - c0) / denom,
                        torch.zeros_like(denom),
                    ) * (2 * np.pi / G)
                    gamma = gamma + dg.double().cpu()

                is_mirror = ch_i.cpu() == 1
                q_zone = self.zone_quats[z_i.cpu()]

                if subpixel_zone:
                    # sub-grid zone axis: correlation-weighted centroid over
                    # the symmetry-complete neighborhood of the best zone
                    # (see build_plan; images across the wedge boundary keep
                    # the centroid unbiased for boundary zones)
                    b_ar = torch.arange(B, device=corr.device)
                    corr_z = corr[b_ar, ch_i].amax(dim=-1).cpu()  # (B, Z)
                    zi_cpu = z_i.cpu()
                    n_idx = self.zone_nbr_idx[zi_cpu]  # (B, K)
                    n_pos = self.zone_nbr_pos[zi_cpu]  # (B, K, 3)
                    n_ok = self.zone_nbr_valid[zi_cpu]  # (B, K)
                    c_n = corr_z.gather(1, n_idx)  # (B, K)
                    c_floor = corr_z.gather(1, zi_cpu[:, None]) * 0.7
                    wgt = (c_n - c_floor).clamp_min(0) * n_ok
                    za_ref = (wgt[:, :, None] * n_pos).sum(dim=1)
                    za_ref = za_ref / torch.linalg.norm(
                        za_ref, dim=-1, keepdim=True
                    ).clamp_min(1e-12)
                    za_old = self.zone_axes[z_i.cpu()]
                    axis = torch.cross(za_ref, za_old, dim=-1)
                    sin_t = torch.linalg.norm(axis, dim=-1)
                    ang_t = torch.atan2(sin_t, (za_ref * za_old).sum(-1))
                    ok_t = sin_t > 1e-12
                    dq = torch.zeros((B, 4), dtype=torch.float64)
                    dq[:, 0] = 1.0
                    if bool(ok_t.any()):
                        dq[ok_t] = quat_from_axis_angle(
                            axis[ok_t] / sin_t[ok_t, None], ang_t[ok_t]
                        )
                    # rotate za_ref -> za_old in the crystal frame: R' = R S
                    q_zone = qmult(q_zone, dq)

                q_flip = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
                q_zone = torch.where(
                    is_mirror[:, None], qmult(q_flip, q_zone), q_zone
                )
                gamma = torch.where(is_mirror, -gamma - np.pi, gamma)
                half = gamma / 2
                zeros = torch.zeros_like(half)
                q_spin = torch.stack(
                    (torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1
                )
                q = qmult(q_spin, q_zone)

                for b, (rx, ry) in enumerate(batch):
                    if torch.isfinite(c_val[b]):
                        quats[rx, ry, m] = q[b]
                        corr_out[rx, ry, m] = c_val[b].double().cpu()
                        mirror_out[rx, ry, m] = bool(is_mirror[b])

                if m == 0:
                    # second-best score outside the exclusion ball around the
                    # best zone axis -> reliability = corr - corr_second
                    far = zone_ang[z_i] >= min_angle_between_matches_deg  # (B, Z)
                    c2 = (
                        corr.masked_fill(~far[:, None, :, None], -torch.inf)
                        .reshape(B, -1)
                        .amax(dim=1)
                    )
                    for b, (rx, ry) in enumerate(batch):
                        if torch.isfinite(c2[b]):
                            corr_second[rx, ry] = c2[b].double().cpu()

                if M > 1:
                    if m == 0:
                        self._zprev = [[] for _ in range(B)]
                    for b in range(B):
                        self._zprev[b].append(int(z_i[b]))

        self.quats = quats
        self.corr = corr_out
        self.corr_second = corr_second
        self.reliability = corr_out[..., 0] - corr_second
        self.mirror = mirror_out
        return self

    # ------------------------------------------------------------------
    # sub-grid refinement
    # ------------------------------------------------------------------

    def refine_orientations(
        self,
        num_iterations: int = 5,
        pair_distance: float | None = None,
        sigma_excitation: float | None = None,
        min_pairs: int = 3,
        refine_tilt: bool = False,
        refine_zone: bool = True,
        zone_search_deg: float = 1.5,
        sigma_envelope: float | None = None,
        zone_max_total_deg: float | None = None,
        batched: bool = True,
        neighbor_rescue: bool = True,
        rescue_threshold_deg: float = 2.0,
        progress_bar: bool = True,
    ) -> "OrientationMap":
        """Refine matched orientations by least squares on paired peak positions.

        For each probe position and match, the simulated pattern is paired to
        the measured peaks (nearest neighbor within `pair_distance`), and the
        small rotation minimizing the weighted in-plane residuals is solved in
        closed form and applied; repeated for `num_iterations` rounds with
        re-pairing. This removes the in-plane quantization of the orientation
        plan (typically to well below 0.1 degrees).

        By default only the in-plane rotation is refined. Zero-layer peak
        positions carry almost no information about out-of-plane tilt (the
        tilt terms in the position residual are proportional to g_z, which is
        near zero for excited reflections), so fitting the full rotation from
        positions is ill-conditioned and amplifies detection noise into
        spurious tilts. Tilt is constrained by the diffracted *intensities*
        (which reflections are excited) and belongs to the dynamical
        refinement pass.

        Parameters
        ----------
        num_iterations : int, default=5
            Pairing + rotation solve rounds.
        pair_distance : float | None
            Maximum pairing distance (1/Angstroms); defaults to the plan's
            corr_kernel_size.
        sigma_excitation : float | None
            Excitation error envelope used for simulation; defaults to the
            plan's value.
        min_pairs : int, default=3
            Skip positions with fewer paired peaks.
        refine_tilt : bool, default=False
            Also solve the two tilt components from peak positions. Only
            meaningful for noise-free simulated data.
        refine_zone : bool, default=True
            Refine the zone-axis tilt from the intensity envelope (the Laue
            circle): the tilt that concentrates the measured intensity on
            the Ewald sphere, searched over +/- zone_search_deg with
            parabolic sub-stepping. Removes the zone-axis quantization of
            the orientation plan.
        zone_search_deg : float, default=1.5
            Half-range of the envelope tilt search, in degrees.
        zone_max_total_deg : float | None
            Trust region: cap on the cumulative envelope tilt applied to
            each orientation, relative to its matched start. The coarse
            match is grid-accurate to about half the zone-axis step, so tilt
            corrections beyond that scale are noise walking the orientation
            out of its basin. Defaults to 0.375 * the plan's zone step.
        sigma_envelope : float | None
            Excitation-error width of the envelope objective; defaults to
            half the plan's sigma_excitation (the plan value is widened for
            grid robustness).
        neighbor_rescue : bool, default=True
            Second pass over positions whose best match disagrees with every
            neighbor by more than rescue_threshold_deg: re-refine from each
            distinct neighbor orientation and keep the highest-scoring
            result (score = total paired measured intensity). Repairs
            isolated wrong local optima such as near-degenerate variants.
        rescue_threshold_deg : float, default=2.0
            Minimum-neighbor misorientation that triggers the rescue pass.
        """
        assert self.quats is not None
        delta = pair_distance if pair_distance is not None else self.corr_kernel_size
        sigma = (
            sigma_excitation if sigma_excitation is not None else self.sigma_excitation
        )
        peaks = self.peaks
        R, C, M = self.quats.shape[:3]
        fields = peaks.fields
        ix = [fields.index(f) for f in ("qx", "qy", "intensity")]
        g_all = self.crystal.g_vec
        lam = self.wavelength
        sigma_env = sigma_envelope if sigma_envelope is not None else sigma / 2
        f_all = self.crystal.struct_factors_int.to(torch.float64)
        tg = torch.deg2rad(
            torch.linspace(-zone_search_deg, zone_search_deg, 17, dtype=torch.float64)
        )
        eye3 = torch.eye(3, dtype=torch.float64)
        tilt_cap = np.deg2rad(
            zone_max_total_deg
            if zone_max_total_deg is not None
            else 0.375 * self.zone_step_deg
        )

        def refine_single(q, q_exp, w_exp):
            """Refine one orientation; return (q, pairing score)."""
            score = 0.0
            tilt_total = torch.zeros(2, dtype=torch.float64)
            for _ in range(num_iterations):
                g = qrotate(q, g_all)
                gz, g2 = g[:, 2], (g**2).sum(dim=1)
                s_g = (2 * gz - lam * g2) / (2 - 2 * lam * gz)
                sel = torch.abs(s_g) < 2 * sigma
                g_sel = g[sel]
                if g_sel.shape[0] == 0:
                    return q, score
                d = torch.cdist(g_sel[:, :2], q_exp)
                d_min, j_min = d.min(dim=1)
                pair = d_min < delta
                if int(pair.sum()) < min_pairs:
                    return q, score
                gp = g_sel[pair]
                tgt = q_exp[j_min[pair]]
                w = w_exp[j_min[pair]] * (1 - d_min[pair] / delta)
                score = float(w.sum())
                # solve min sum w | tgt - (g + omega x g)_xy |^2 for omega
                r = tgt - gp[:, :2]  # (P, 2)
                if refine_tilt:
                    A = torch.zeros((gp.shape[0], 2, 3), dtype=torch.float64)
                    A[:, 0, 1] = gp[:, 2]
                    A[:, 0, 2] = -gp[:, 1]
                    A[:, 1, 0] = -gp[:, 2]
                    A[:, 1, 2] = gp[:, 0]
                    Aw = A * w[:, None, None]
                    AtA = torch.einsum("pki,pkj->ij", Aw, A)
                    Atr = torch.einsum("pki,pk->i", Aw, r)
                    omega = torch.linalg.solve(AtA + 1e-12 * eye3, Atr)
                else:
                    # in-plane only: residual model r = omega_z * (-g_y, g_x)
                    a = torch.stack((-gp[:, 1], gp[:, 0]), dim=1)  # (P, 2)
                    num = (w[:, None] * a * r).sum()
                    den = (w[:, None] * a * a).sum().clamp_min(1e-12)
                    omega = torch.tensor(
                        [0.0, 0.0, float(num / den)], dtype=torch.float64
                    )
                angle = torch.linalg.norm(omega)
                if angle > 1e-10:
                    dq = quat_from_axis_angle(omega / angle, angle)
                    q = qmult(dq, q)

                if refine_zone:
                    # continuous zone-axis tilt from the intensity envelope:
                    # a small lab-frame tilt (wx, wy) shifts every excitation
                    # error by s(w) = s0 + wx*gy - wy*gx; maximize the
                    # normalized cosine between the measured intensities and
                    # the predicted |F|^2 * envelope over a grid with
                    # parabolic sub-stepping (the Laue-circle fit -- peak
                    # positions carry no tilt information, the excitation
                    # pattern does)
                    s0 = s_g[sel][pair]
                    a1 = gp[:, 1]
                    a2 = -gp[:, 0]
                    f_p = f_all[sel][pair]
                    S = (
                        s0[:, None, None]
                        + tg[None, :, None] * a1[:, None, None]
                        + tg[None, None, :] * a2[:, None, None]
                    )
                    pred = f_p[:, None, None] * torch.exp(
                        -(S**2) / (2 * sigma_env**2)
                    )
                    E = (w[:, None, None] * pred).sum(dim=0) / (
                        (pred**2).sum(dim=0).sqrt().clamp_min(1e-12)
                    )
                    ij = int(E.argmax())
                    i0, j0 = ij // 17, ij % 17
                    wx, wy = float(tg[i0]), float(tg[j0])
                    step = float(tg[1] - tg[0])
                    if 0 < i0 < 16:
                        c0, c1, c2 = (
                            float(E[i0 - 1, j0]),
                            float(E[i0, j0]),
                            float(E[i0 + 1, j0]),
                        )
                        den = 2 * c1 - c0 - c2
                        if abs(den) > 1e-12:
                            wx += 0.5 * (c2 - c0) / den * step
                    if 0 < j0 < 16:
                        c0, c1, c2 = (
                            float(E[i0, j0 - 1]),
                            float(E[i0, j0]),
                            float(E[i0, j0 + 1]),
                        )
                        den = 2 * c1 - c0 - c2
                        if abs(den) > 1e-12:
                            wy += 0.5 * (c2 - c0) / den * step
                    # trust region on the cumulative tilt from the start
                    prop = tilt_total + torch.tensor(
                        [wx, wy], dtype=torch.float64
                    )
                    over = float(torch.linalg.norm(prop)) - tilt_cap
                    if over > 0:
                        prop = prop * tilt_cap / float(torch.linalg.norm(prop))
                    step = prop - tilt_total
                    tilt_total = prop
                    tilt = torch.tensor(
                        [float(step[0]), float(step[1]), 0.0],
                        dtype=torch.float64,
                    )
                    t_ang = torch.linalg.norm(tilt)
                    if t_ang > 1e-10:
                        dq = quat_from_axis_angle(tilt / t_ang, t_ang)
                        q = qmult(dq, q)
            return q, score

        def get_exp(rx, ry):
            data = peaks[rx, ry].array
            if data.shape[0] < min_pairs:
                return None, None
            q_exp = torch.as_tensor(data[:, ix[:2]], dtype=torch.float64)
            w_exp = torch.as_tensor(data[:, ix[2]], dtype=torch.float64).clamp_min(0)
            w_exp = w_exp / w_exp.max().clamp_min(1e-12)
            return q_exp, w_exp

        scores = torch.zeros((R, C), dtype=torch.float64)
        if batched and not refine_tilt:
            self._refine_batched(
                scores,
                delta=delta,
                sigma=sigma,
                sigma_env=sigma_env,
                tg=tg,
                tilt_cap=tilt_cap,
                num_iterations=num_iterations,
                min_pairs=min_pairs,
                refine_zone=refine_zone,
                progress_bar=progress_bar,
            )
        else:
            iterator = list(np.ndindex(R, C))
            if progress_bar:
                iterator = tqdm(iterator, desc="refining orientations")
            for rx, ry in iterator:
                q_exp, w_exp = get_exp(rx, ry)
                if q_exp is None:
                    continue
                for m in range(M):
                    if self.corr[rx, ry, m] <= 0:
                        continue
                    q, sc = refine_single(self.quats[rx, ry, m], q_exp, w_exp)
                    self.quats[rx, ry, m] = q
                    if m == 0:
                        scores[rx, ry] = sc

        if neighbor_rescue:
            # a wrong local optimum (e.g. a near-degenerate variant) shows as
            # a discontinuity: retry those positions from each distinct
            # neighbor orientation and keep the best-scoring result
            q0 = self.quats[..., 0, :]
            miso_min = torch.full((R, C), torch.inf, dtype=torch.float64)
            for dr, dc in ((0, 1), (1, 0)):
                a = q0[: R - dr, : C - dc]
                b = q0[dr:, dc:]
                mm = misorientation_angle_deg(
                    a.reshape(-1, 4), b.reshape(-1, 4), self.crystal.sym_quats
                ).reshape(R - dr, C - dc)
                miso_min[: R - dr, : C - dc] = torch.minimum(
                    miso_min[: R - dr, : C - dc], mm
                )
                miso_min[dr:, dc:] = torch.minimum(miso_min[dr:, dc:], mm)
            retry = torch.nonzero(miso_min > rescue_threshold_deg)
            it2 = retry.tolist()
            if progress_bar and len(it2):
                it2 = tqdm(it2, desc="neighbor rescue")
            n_rescued = 0
            for rx, ry in it2:
                q_exp, w_exp = get_exp(rx, ry)
                if q_exp is None:
                    continue
                best_q = self.quats[rx, ry, 0]
                best_s = float(scores[rx, ry])
                cands = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rx + dr, ry + dc
                        if (dr == 0 and dc == 0) or not (
                            0 <= nr < R and 0 <= nc < C
                        ):
                            continue
                        qn = self.quats[nr, nc, 0]
                        if all(
                            float(
                                misorientation_angle_deg(
                                    qn, c, self.crystal.sym_quats
                                )
                            )
                            > 0.5
                            for c in cands
                        ):
                            cands.append(qn)
                for qc in cands:
                    q, sc = refine_single(qc.clone(), q_exp, w_exp)
                    if sc > best_s * 1.02:
                        best_q, best_s = q, sc
                if best_s > float(scores[rx, ry]):
                    n_rescued += 1
                self.quats[rx, ry, 0] = best_q
                scores[rx, ry] = best_s
        return self

    # ------------------------------------------------------------------
    # forward simulation of a match
    # ------------------------------------------------------------------

    def _refine_batched(
        self,
        scores: torch.Tensor,
        delta: float,
        sigma: float,
        sigma_env: float,
        tg: torch.Tensor,
        tilt_cap: float,
        num_iterations: int,
        min_pairs: int,
        refine_zone: bool,
        progress_bar: bool,
        chunk: int = 64,
    ) -> None:
        """Chunk-vectorized in-plane + envelope refinement (all positions)."""
        from quantem.diffraction.rotations import quat_to_matrix

        peaks = self.peaks
        R, C, M = self.quats.shape[:3]
        fields = peaks.fields
        ix = [fields.index(f) for f in ("qx", "qy", "intensity")]
        g_all = self.crystal.g_vec  # (G, 3)
        f_all = self.crystal.struct_factors_int.to(torch.float64)
        lam = self.wavelength
        n_tg = tg.shape[0]

        # flatten measured peaks once, padded per position
        cells = [peaks[r, c].array for r, c in np.ndindex(R, C)]
        counts = np.array([c.shape[0] for c in cells])
        Pmax = max(1, counts.max())
        N = R * C
        q_exp = torch.full((N, Pmax, 2), 1e6, dtype=torch.float64)
        w_exp = torch.zeros((N, Pmax), dtype=torch.float64)
        for i, arr in enumerate(cells):
            n = arr.shape[0]
            if n == 0:
                continue
            q_exp[i, :n] = torch.as_tensor(arr[:, ix[:2]], dtype=torch.float64)
            wi = torch.as_tensor(arr[:, ix[2]], dtype=torch.float64).clamp_min(0)
            w_exp[i, :n] = wi / wi.max().clamp_min(1e-12)

        quats = self.quats.reshape(N, M, 4)
        corr = self.corr.reshape(N, M)
        valid_pos = torch.as_tensor(counts >= min_pairs)

        chunks = range(0, N, chunk)
        if progress_bar:
            chunks = tqdm(chunks, desc="refining orientations (batched)")
        for i0 in chunks:
            i1 = min(i0 + chunk, N)
            B = i1 - i0
            qe = q_exp[i0:i1]  # (B, P, 2)
            we = w_exp[i0:i1]  # (B, P)
            for m in range(M):
                act = valid_pos[i0:i1] & (corr[i0:i1, m] > 0)
                if not bool(act.any()):
                    continue
                q = quats[i0:i1, m].clone()  # (B, 4)
                tilt_total = torch.zeros((B, 2), dtype=torch.float64)
                sc = torch.zeros(B, dtype=torch.float64)
                for _ in range(num_iterations):
                    Rm = quat_to_matrix(q)  # (B, 3, 3)
                    g = torch.einsum("bij,gj->bgi", Rm, g_all)  # (B, G, 3)
                    gz, g2 = g[..., 2], (g**2).sum(dim=-1)
                    s_g = (2 * gz - lam * g2) / (2 - 2 * lam * gz)
                    sel = torch.abs(s_g) < 2 * sigma  # (B, G)
                    d = torch.cdist(g[..., :2], qe)  # (B, G, P)
                    d_min, j_min = d.min(dim=-1)  # (B, G)
                    pair = sel & (d_min < delta)
                    w_g = torch.gather(we, 1, j_min) * (1 - d_min / delta).clamp_min(0)
                    w_g = w_g * pair  # (B, G)
                    n_pair = pair.sum(dim=1)
                    ok = act & (n_pair >= min_pairs)
                    if not bool(ok.any()):
                        break
                    sc = torch.where(ok, w_g.sum(dim=1), sc)
                    tgt = torch.gather(
                        qe, 1, j_min[..., None].expand(-1, -1, 2)
                    )  # (B, G, 2)
                    r_vec = tgt - g[..., :2]
                    # in-plane closed form
                    a_vec = torch.stack((-g[..., 1], g[..., 0]), dim=-1)
                    num = (w_g[..., None] * a_vec * r_vec).sum(dim=(1, 2))
                    den = (w_g[..., None] * a_vec * a_vec).sum(dim=(1, 2))
                    wz = torch.where(ok, num / den.clamp_min(1e-12), torch.zeros_like(num))
                    half = wz / 2
                    dq = torch.stack(
                        (
                            torch.cos(half),
                            torch.zeros_like(half),
                            torch.zeros_like(half),
                            torch.sin(half),
                        ),
                        dim=-1,
                    )
                    q = torch.where(ok[:, None], qmult(dq, q), q)

                    if refine_zone:
                        # sparse over paired reflections only
                        idx_b, idx_g = torch.nonzero(pair, as_tuple=True)
                        s0f = s_g[idx_b, idx_g]
                        gyf = g[idx_b, idx_g, 1]
                        gxf = g[idx_b, idx_g, 0]
                        ff = f_all[idx_g]
                        wf = w_g[idx_b, idx_g]
                        S = (
                            s0f[:, None, None]
                            + tg[None, :, None] * gyf[:, None, None]
                            - tg[None, None, :] * gxf[:, None, None]
                        )  # (Np, T, T)
                        pred = ff[:, None, None] * torch.exp(
                            -(S**2) / (2 * sigma_env**2)
                        )
                        E_num = torch.zeros(
                            (B, n_tg, n_tg), dtype=torch.float64
                        ).index_add_(0, idx_b, wf[:, None, None] * pred)
                        E_den = torch.zeros(
                            (B, n_tg, n_tg), dtype=torch.float64
                        ).index_add_(0, idx_b, pred**2)
                        E = E_num / E_den.sqrt().clamp_min(1e-12)  # (B, T, T)
                        flat_ij = E.reshape(B, -1).argmax(dim=1)
                        i_b, j_b = flat_ij // n_tg, flat_ij % n_tg
                        step = float(tg[1] - tg[0])
                        wx = tg[i_b].clone()
                        wy = tg[j_b].clone()
                        # parabolic sub-stepping where interior
                        b_ar = torch.arange(B)
                        for axis, idx, wv in ((0, i_b, wx), (1, j_b, wy)):
                            interior = (idx > 0) & (idx < n_tg - 1)
                            if not bool(interior.any()):
                                continue
                            if axis == 0:
                                c0 = E[b_ar, (idx - 1).clamp(0), j_b]
                                c1 = E[b_ar, idx, j_b]
                                c2 = E[b_ar, (idx + 1).clamp(max=n_tg - 1), j_b]
                            else:
                                c0 = E[b_ar, i_b, (idx - 1).clamp(0)]
                                c1 = E[b_ar, i_b, idx]
                                c2 = E[b_ar, i_b, (idx + 1).clamp(max=n_tg - 1)]
                            den2 = 2 * c1 - c0 - c2
                            shift = torch.where(
                                interior & (den2.abs() > 1e-12),
                                0.5 * (c2 - c0) / den2 * step,
                                torch.zeros_like(c1),
                            )
                            wv += shift
                        prop = tilt_total + torch.stack((wx, wy), dim=-1)
                        norm = torch.linalg.norm(prop, dim=-1)
                        scale_f = torch.where(
                            norm > tilt_cap, tilt_cap / norm.clamp_min(1e-12),
                            torch.ones_like(norm),
                        )
                        prop = prop * scale_f[:, None]
                        step_t = torch.where(
                            ok[:, None], prop - tilt_total, torch.zeros_like(prop)
                        )
                        tilt_total = torch.where(ok[:, None], prop, tilt_total)
                        t_ang = torch.linalg.norm(step_t, dim=-1)
                        axis_v = torch.zeros((B, 3), dtype=torch.float64)
                        nz = t_ang > 1e-10
                        if bool(nz.any()):
                            axis_v[nz, 0] = step_t[nz, 0] / t_ang[nz]
                            axis_v[nz, 1] = step_t[nz, 1] / t_ang[nz]
                            dq_t = quat_from_axis_angle(axis_v[nz], t_ang[nz])
                            q_nz = q[nz]
                            q[nz] = qmult(dq_t, q_nz)
                quats[i0:i1, m] = torch.where(act[:, None], q, quats[i0:i1, m])
                if m == 0:
                    scores.reshape(-1)[i0:i1] = torch.where(
                        act, sc, scores.reshape(-1)[i0:i1]
                    )
        self.quats = quats.reshape(R, C, M, 4)

    def generate_pattern(self, rx: int, ry: int, match: int = 0, **kwargs):
        """Simulated pattern for the matched orientation at (rx, ry)."""
        assert self.quats is not None
        return self.crystal.generate_pattern(
            self.quats[rx, ry, match],
            energy_ev=self.energy_ev,
            sigma_excitation=self.sigma_excitation,
            **kwargs,
        )

    def match_residual(
        self,
        other: "OrientationMap",
        delete_radius: float = 0.04,
        min_number_peaks: int = 3,
        min_corr_other: float = 0.0,
        progress_bar: bool = True,
    ) -> "OrientationMap":
        """Re-match this crystal on the peaks another crystal cannot explain.

        For overlapping patterns (e.g. a thin lath on a matrix), the direct
        match of the minority phase is poisoned by the majority phase's
        peaks. Here the majority candidate's simulated pattern is used to
        delete its measured peaks at each position, and this crystal is
        re-matched against the residual peaks only. Where the residual match
        beats this map's stored second match, it replaces it (match index 1),
        so the joint phase fit sees one clean candidate per phase.

        Parameters
        ----------
        other : OrientationMap
            The matched map of the (locally dominant) other crystal.
        delete_radius : float, default=0.04
            Measured peaks within this distance (1/Angstroms) of one of the
            other crystal's simulated peaks are removed.
        min_corr_other : float, default=0.0
            Skip positions where the other crystal's correlation is below
            this (nothing trustworthy to delete).
        """
        assert self.quats is not None and other.quats is not None
        peaks = self.peaks
        R, C = peaks.shape[0], peaks.shape[1]
        fields = peaks.fields
        ix = [fields.index(f) for f in ("qx", "qy", "intensity")]

        residual = Vector.from_shape(
            (R, C), fields=["qx", "qy", "intensity"],
            units=["A^-1", "A^-1", "counts"], name="residual_peaks",
        )
        cells = []
        for rx, ry in np.ndindex(R, C):
            data = peaks[rx, ry].array
            if data.shape[0] < min_number_peaks or other.corr[rx, ry, 0] <= min_corr_other:
                cells.append(np.zeros((0, 3)))
                continue
            sim = other.generate_pattern(rx, ry)
            sq = torch.stack((sim["qx"], sim["qy"]), dim=1)
            if sq.shape[0] == 0:
                cells.append(data[:, ix])
                continue
            qxy = torch.as_tensor(data[:, ix[:2]], dtype=torch.float64)
            d_min = torch.cdist(qxy, sq).min(dim=1).values
            keep = (d_min > delete_radius).numpy()
            cells.append(data[keep][:, ix])
        nested = [cells[r * C : (r + 1) * C] for r in range(R)]
        residual = Vector.from_data(
            nested, fields=["qx", "qy", "intensity"],
            units=["A^-1", "A^-1", "counts"], name="residual_peaks",
        )

        om_res = OrientationMap.from_vectors(residual, self.crystal, self.energy_ev)
        for attr in (
            "device", "corr_kernel_size", "sigma_excitation", "power_radial",
            "power_intensity", "zone_axes", "zone_quats", "zone_step_deg",
            "zone_nbr_idx", "zone_nbr_pos", "zone_nbr_valid",
            "plan_fft", "shell_radii", "num_gamma", "gamma", "detector_mask",
            "plan_norm_shift", "plan_frac_shift",
        ):
            setattr(om_res, attr, getattr(self, attr))
        om_res.match_orientations(
            num_matches=1,
            min_number_peaks=min_number_peaks,
            progress_bar=progress_bar,
        )
        om_res.refine_orientations(progress_bar=progress_bar)

        # replace the stored second match where the residual match is better
        if self.quats.shape[2] < 2:
            pad_q = torch.zeros((R, C, 1, 4), dtype=torch.float64)
            pad_q[..., 0] = 1.0
            self.quats = torch.cat([self.quats, pad_q], dim=2)
            self.corr = torch.cat(
                [self.corr, torch.zeros((R, C, 1), dtype=torch.float64)], dim=2
            )
            self.mirror = torch.cat(
                [self.mirror, torch.zeros((R, C, 1), dtype=torch.bool)], dim=2
            )
        better = om_res.corr[..., 0] > self.corr[..., 1]
        self.quats[..., 1, :] = torch.where(
            better[..., None], om_res.quats[..., 0, :], self.quats[..., 1, :]
        )
        self.corr[..., 1] = torch.where(better, om_res.corr[..., 0], self.corr[..., 1])
        self.mirror[..., 1] = torch.where(
            better, om_res.mirror[..., 0], self.mirror[..., 1]
        )
        return self

    def cluster_orientations(
        self,
        mask: np.ndarray | None = None,
        threshold_deg: float = 5.0,
        min_cluster_size: int = 10,
        match: int = 0,
    ) -> dict:
        """Greedy clustering of the matched orientations into variants.

        Positions are visited in order of decreasing correlation; each seed
        collects every unassigned position within `threshold_deg`
        (symmetry-reduced misorientation) into a cluster. Follows the variant
        analysis of MacLaren et al., J. Microscopy 295, 131 (2024).

        Parameters
        ----------
        mask : np.ndarray | None
            Boolean or weight mask of positions to include (e.g. phase mask).
        threshold_deg : float, default=5.0
            Misorientation radius of a cluster.
        min_cluster_size : int, default=10
            Smaller clusters are discarded (labels stay -1).

        Returns
        -------
        dict
            'labels' (R, C) int tensor, -1 = unassigned; 'mean_quats'
            (K, 4) cluster mean orientations; 'sizes' (K,) member counts.
        """
        assert self.quats is not None
        R, C = self.quats.shape[:2]
        q = self.quats[..., match, :].reshape(-1, 4)
        corr = self.corr[..., match].reshape(-1)
        ok = corr > 0
        if mask is not None:
            ok &= torch.as_tensor(np.asarray(mask, dtype=float).reshape(-1)) > 0.5

        labels = torch.full((R * C,), -1, dtype=torch.long)
        sym = self.crystal.sym_quats
        unassigned = ok.clone()
        means, sizes = [], []
        k = 0
        while unassigned.any():
            seed = int(torch.where(unassigned, corr, torch.full_like(corr, -1)).argmax())
            miso = misorientation_angle_deg(q[seed][None], q, sym)
            members = unassigned & (miso < threshold_deg)
            unassigned &= ~members
            if int(members.sum()) < min_cluster_size:
                continue
            labels[members] = k
            # symmetry-align members to the seed, then average
            qm = q[members]
            dq = qmult(qconj(q[seed])[None], qm)
            dq_sym = qmult(dq[:, None, :], sym)
            best = dq_sym[..., 0].abs().argmax(dim=1)
            dq_best = dq_sym[torch.arange(qm.shape[0]), best]
            sign = torch.where(dq_best[:, :1] < 0, -1.0, 1.0)
            q_aligned = qmult(q[seed][None], dq_best * sign)
            means.append(qnormalize(q_aligned.mean(dim=0)))
            sizes.append(int(members.sum()))
            k += 1
        # order clusters by size, largest first
        if means:
            order = torch.argsort(torch.tensor(sizes), descending=True)
            relabel = torch.full((len(sizes),), -1, dtype=torch.long)
            relabel[order] = torch.arange(len(sizes))
            labels = torch.where(labels >= 0, relabel[labels.clamp_min(0)], labels)
            means = [means[int(i)] for i in order]
            sizes = [sizes[int(i)] for i in order]
        return {
            "labels": labels.reshape(R, C),
            "mean_quats": torch.stack(means) if means else torch.zeros((0, 4)),
            "sizes": torch.tensor(sizes),
        }

    def calculate_strain(
        self,
        match: int = 0,
        pair_distance: float | None = None,
        min_pairs: int = 5,
        mask: np.ndarray | None = None,
        ds_sampling: float | None = None,
        ds_units: str | None = None,
        progress_bar: bool = True,
    ):
        """Per-position strain from measured vs simulated peak positions.

        At each probe position the refined orientation's simulated pattern is
        paired to the measured peaks and the in-plane deformation A
        minimizing sum w |A q_sim - q_meas|^2 is solved in closed form. The
        strain is referenced to the crystal's ideal lattice, so unlike
        lattice-vector strain mapping it is absolute, not relative to a
        reference region.

        Returns
        -------
        StrainMap
            The columns of A enter as per-position reciprocal lattice
            vectors with the identity as the fixed reference, so all
            StrainMap machinery applies: `plot_strain(rotation_angle=...)`
            for user-chosen u/v directions, `rotate_strain`, masking, and
            scale bars. `num_pairs` (R, C) is attached as an attribute.
        """
        from quantem.diffraction.strain import StrainMap

        assert self.quats is not None
        delta = pair_distance if pair_distance is not None else self.corr_kernel_size
        peaks = self.peaks
        R, C = peaks.shape[0], peaks.shape[1]
        fields = peaks.fields
        ix = [fields.index(f) for f in ("qx", "qy", "intensity")]

        A_map = np.full((R, C, 2, 2), np.nan)
        num_pairs = np.zeros((R, C), dtype=int)

        iterator = list(np.ndindex(R, C))
        if progress_bar:
            iterator = tqdm(iterator, desc="strain mapping")
        for rx, ry in iterator:
            if mask is not None and not mask[rx, ry]:
                continue
            if self.corr[rx, ry, match] <= 0:
                continue
            data = peaks[rx, ry].array
            if data.shape[0] < min_pairs:
                continue
            q_exp = torch.as_tensor(data[:, ix[:2]], dtype=torch.float64)
            w_exp = torch.as_tensor(data[:, ix[2]], dtype=torch.float64).clamp_min(0)
            sim = self.generate_pattern(rx, ry, match=match)
            sq = torch.stack((sim["qx"], sim["qy"]), dim=1)
            if sq.shape[0] == 0:
                continue
            d = torch.cdist(sq, q_exp)
            d_min, j_min = d.min(dim=1)
            pair = d_min < delta
            n = int(pair.sum())
            if n < min_pairs:
                continue
            qs = sq[pair]
            qm = q_exp[j_min[pair]]
            w = w_exp[j_min[pair]] * (1 - d_min[pair] / delta)
            # A = (sum w qm qs^T) (sum w qs qs^T)^-1
            M1 = torch.einsum("p,pi,pj->ij", w, qm, qs)
            M2 = torch.einsum("p,pi,pj->ij", w, qs, qs)
            A = M1 @ torch.linalg.inv(M2 + 1e-12 * torch.eye(2, dtype=torch.float64))
            A_map[rx, ry] = A.numpy()
            num_pairs[rx, ry] = n

        # columns of A are the measured images of the reciprocal unit basis;
        # StrainMap's reciprocal-space branch (U_ref @ inv(U) = F^T) then
        # yields the real-space strain with the shared sign conventions
        sm = StrainMap(
            u_array=A_map[..., :, 0],
            v_array=A_map[..., :, 1],
            ds_shape=(R, C),
            real_space=False,
            u_ref=np.array([1.0, 0.0]),
            v_ref=np.array([0.0, 1.0]),
            mask=None if mask is None else np.asarray(mask, dtype=float),
            ds_sampling=ds_sampling,
            ds_units=ds_units,
        )
        sm.num_pairs = num_pairs
        return sm

    def in_plane_angle_deg(
        self, match: int = 0, mod_deg: float | None = None
    ) -> torch.Tensor:
        """In-plane angle of the crystal a-axis at every position (degrees).

        The angle of the projected crystal [100] Cartesian axis, measured
        from the scan column axis toward the row axis. `mod_deg` wraps the
        angle by the crystal's in-plane symmetry (60 for hexagonal basal, 90
        for cubic <100> zones); None returns the full range.
        """
        from quantem.diffraction.rotations import quat_to_matrix

        assert self.quats is not None
        R = quat_to_matrix(self.quats[..., match, :])
        a_lab = R[..., :, 0]  # crystal x-axis in the lab frame
        ang = torch.rad2deg(torch.atan2(a_lab[..., 0], a_lab[..., 1]))
        if mod_deg is not None:
            ang = ang % mod_deg
        return ang

    def plot_orientation(self, direction: str = "z", match: int = 0, **kwargs):
        """IPF-colored orientation map; see orientation_visualization."""
        from quantem.diffraction.orientation_visualization import plot_orientation_map

        return plot_orientation_map(self, direction=direction, match=match, **kwargs)

    def plot_pole_figure(self, pole=(0, 0, 1), match: int = 0, **kwargs):
        """Stereographic pole figure; see orientation_visualization."""
        from quantem.diffraction.orientation_visualization import plot_pole_figure

        return plot_pole_figure(self, pole=pole, match=match, **kwargs)

    def misorientation_map(self, reference: torch.Tensor | None = None) -> torch.Tensor:
        """Misorientation angle (deg) of match 0 to a reference orientation."""
        assert self.quats is not None
        q = self.quats[..., 0, :]
        if reference is None:
            reference = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)
        return misorientation_angle_deg(
            reference, q, self.crystal.sym_quats_matching
        )
