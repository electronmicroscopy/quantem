import matplotlib.pyplot as plt
import torch
import numpy as np
from numpy.typing import NDArray
from scipy.special import spherical_jn

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.polar_transform import (
    as_dataset4dstem,
    find_origin_angular_grid,
    polar_transform,
)

class PairAngleDistributionFunction(AutoSerialize):
    """
    Compute the pair-angle distribution function from diffraction data:
    a single 2D pattern (Dataset2d), a 3D stack of patterns, or a 4D-STEM
    dataset. Non-4D inputs are wrapped as (1, 1, H, W) / (N, 1, H, W).

    Run the following pipeline:
    padf = PairAngleDistributionFunction(ds)
    1. find_origin_angular_grid(ds)
    2. polar_transform(ds, origin_array=self.origin)
    3. rescale_intensity(self.polar, rho=1, fq=1)
    4. compute_avg_angular_correlation(rescaled intensity)
    5. subtract_angular_median(self.ang_corr)
    6. extract_Bl_matrices(self.ang_corr_corrected)
    7. transform_to_real_space(self.Bl_mats[0], self.Bl_mats[1])
    8. reconstruct_PADF(self.real_Bl_mats, self.Bl_mats[1], Na=1); the overall scale is arbitrary

    polar_ds can instead be a polar dataset, or a list of them with the same
    q axis, in which case steps 1-2 are skipped and the angular correlation is
    averaged over the list (useful for more patterns than fit in memory).
    fq may be a function of q (1/A), evaluated on the polar q axis self.q.
    The real-space grid r = r_min, r_min + r_step, ... includes r_max.
    """
    def __init__(self,
                 ds=None,
                 origin: NDArray | None = None,
                 polar_ds: Polar4dstem | list[Polar4dstem] | None = None,
                 ang_corr: torch.Tensor | None = None,
                 Bl_mats = None,
                 rho: float = 1.0,
                 fq=None,
                 l_max: int = 40,
                 r_min: float = 0.0,
                 r_max: float = 20.0,
                 r_step: float = 0.02,
                 l_taper: bool = False,
                 ):
        super().__init__()

        # accept a single 2D pattern, a 3D stack, or a 4D-STEM dataset
        self.ds = as_dataset4dstem(ds) if ds is not None else None
        if polar_ds is None:
            self.origin = origin if origin is not None else find_origin_angular_grid(self.ds)
            polar_ds = polar_transform(self.ds, origin_array=self.origin)
        else:
            self.origin = origin
        polars = list(polar_ds) if isinstance(polar_ds, (list, tuple)) else [polar_ds]
        self.polar = polars[0]
        # radial q calibration of the polar dataset: bin i sits at q_min + i * dq
        dq = float(np.asarray(self.polar.sampling)[3])
        q_min = float(np.asarray(self.polar.origin)[3])
        self.q = q_min + np.arange(self.polar.shape[-1]) * dq
        if callable(fq):
            fq = fq(self.q)

        # TODO: look up rho and fq from sample identity when not provided
        if ang_corr is None:
            ang_corr = sum(
                self.compute_avg_angular_correlation(self.rescale_intensity(p, rho=rho, fq=fq)) for p in polars
            ) / len(polars)
        self.ang_corr = ang_corr
        self.ang_corr_corrected = self.subtract_angular_median(self.ang_corr)
        self.Bl_mats = Bl_mats if Bl_mats is not None else self.extract_Bl_matrices(self.ang_corr_corrected, l_max=l_max)
        self.real_Bl_mats = self.transform_to_real_space(self.Bl_mats[0], self.Bl_mats[1], dq=dq, q_min=q_min,
                                                         r_min=r_min, r_max=r_max + 0.5 * r_step, r_step=r_step)
        self.padf = self.reconstruct_PADF(self.real_Bl_mats, self.Bl_mats[1], 1, l_taper=l_taper)
    
    @staticmethod
    def rescale_intensity(data: Polar4dstem | Dataset4dstem | None = None,
                          rho: float | int = 1.0,
                          fq: NDArray | torch.Tensor | float | int | None = None):
        """
        Equation 7: Take the raw intensity and divide by the following factors
        - Mean number density (rho)
        - Mean atomic scattering factor squared (f(q)^2)
        - Number of atoms in the beam (N_a)
        - phi_0 (dependent on experimental parameters)

        fq may be a scalar or a per-radial-bin array f(q) of length Nq
        (broadcast along the last axis of the polar data). Dividing by
        f(q)^2 removes the atom-shape damping of the diffraction rings;
        leaving it in biases the real-space peak positions of the PADF.
        Floor small f(q) values before passing to avoid amplifying the
        high-q noise floor.

        Finally the intensity is normalized to unit mean. This is the
        analogue of pypadf's scalar 1/beamnorm^2 correlation scaling: it
        absorbs the unmeasurable overall factors (phi_0, N_a, detector
        gain) so the PADF magnitude is independent of the raw intensity
        scale.

        Handles dtype conversion to float64
        """

        intensity = data.tensor.to(dtype=torch.float64)
        if fq is None:
            fq2 = 1.0
        elif np.ndim(fq) == 0:
            fq2 = float(fq) ** 2
        else:
            fq_t = torch.as_tensor(np.asarray(fq), dtype=torch.float64)
            if fq_t.shape[-1] != intensity.shape[-1]:
                raise ValueError(
                    f"fq has length {fq_t.shape[-1]} but the polar data has "
                    f"{intensity.shape[-1]} radial bins."
                )
            fq2 = fq_t ** 2  # broadcasts along the radial (last) axis
        rescaled_intensity = intensity / (rho * fq2)
        rescaled_intensity /= rescaled_intensity.mean()
        return rescaled_intensity
    
    @staticmethod
    def compute_avg_angular_correlation(data):
        """
        Equation 8: angular cross correlation implemented via Fourier correlation theorem
        """

        scan_row, scan_col, phi, r = data.shape
        g_all = data.reshape(scan_row * scan_col, phi, r) # Flatten grid of dps to one dimension, makes for loop simpler
        N_alpha = g_all.shape[0] # Number of dps we have = number of configurations (N_alpha)

        dphi = 2.0 * torch.pi / phi # A small change in angle

        G = torch.fft.rfft(g_all, dim=1) # Fourier transform along phi axis
        P = torch.einsum('afi,afj->fij', G.conj(), G) # Outer product of r between G and its conjugate, summed over all N_alpha
        C_sum = torch.fft.irfft(P, n=phi, dim=0)

        C_avg = (C_sum / N_alpha) * dphi
        return C_avg

    def subtract_angular_mean(self, C_avg: torch.Tensor):
        """
        Subtract the theta-average from each (q, q') ring of the angular
        correlation, removing the uncorrelated background.

        Matches the subtraction in pypadf's maskcorr.py. We do not apply their |sin(theta)| 
        factor here because it changes the function being expanded and mixes Legendre orders 
        before the order-dependent Bessel transforms. Instead, we sin-weight the Legendre 
        fit in extract_Bl_matrices() to match the d(cos theta) measure of the Legendre 
        polynomials.
        """
        return C_avg - C_avg.mean(dim=0, keepdim=True)

    @staticmethod
    def subtract_angular_median(C_avg: torch.Tensor):
        """
        Subtract the theta-median from each (q, q') ring of the angular
        correlation. Unlike the mean, the median is not pulled up by the
        self-correlation peak at theta = 0. The offset only changes B_0, so
        the PADF (l >= 2) is the same as with subtract_angular_mean.
        """
        return C_avg - C_avg.median(dim=0, keepdim=True).values

    def extract_Bl_matrices(self, C_avg, l_max=40, sv_cutoff=0.05, sin_weight=True,
                            theta_exclude=0.0):
        """
        sin_weight : bool
            Weight each angle in the least-squares fit by |sin(theta)|. C is
            sampled evenly in theta, but Legendre polynomials are orthogonal in
            cos(theta), and near 0 and 180 deg many samples crowd into a small
            range of cos(theta). Since |sin(theta)| d(theta) = |d cos(theta)|,
            this weight counts each sample by the cos(theta) range it covers.
        theta_exclude : float
            In degrees, correlation bins within theta_exclude of 0 and 360 deg get zero
            weight in the fit. Use to exclude self correlations and keeps them out of B_l.
        """
        # Equation 9, 10, 11
        # for each l:
        #     build the linear system relating C to B_l(q, q')
        #     solve it by SVD
        # even l only: a single kinematic pattern is Friedel symmetric
        # 5% SVD cutoff
        Nphi, Nq, Nqp = C_avg.shape
        dphi = torch.linspace(0, 2 * torch.pi, Nphi + 1, dtype=torch.float64)[:-1] # Ranges from [0, 2pi)
        l_values = torch.arange(0, l_max, 2) # Even only
        Nl = len(l_values)
        cos_dphi = torch.cos(dphi)

        C_flat = C_avg.reshape(Nphi, Nq * Nqp)

        # Build matrix
        L = l_values.reshape(1, Nl)
        X = cos_dphi.reshape(Nphi, 1)
        leg_matrix = torch.special.legendre_polynomial_p(X, L.to(torch.float64))  # (Nphi, Nl) by broadcasting

        # per-angle fit weight: sqrt|sin| (sin_weight), zero within theta_exclude of 0/360 deg
        w = torch.sqrt(torch.abs(torch.sin(dphi))) if sin_weight else torch.ones_like(dphi)
        if theta_exclude > 0:
            te = np.deg2rad(theta_exclude)
            w[(dphi < te) | (dphi > 2 * torch.pi - te)] = 0
        leg_matrix = leg_matrix * w[:, None]
        C_flat = C_flat * w[:, None]

        U, S, Vh = torch.linalg.svd(leg_matrix, full_matrices=False)
        S_max = S.max()
        S_inv = torch.where(S > sv_cutoff * S_max, 1.0 / S, torch.zeros_like(S))

        tmp = U.T @ C_flat
        tmp = S_inv.unsqueeze(1) * tmp
        B_flat = Vh.T @ tmp

        return B_flat.reshape(Nl, Nq, Nqp), l_values

    def transform_to_real_space(self, Bl_mats, l_values, dq=0.01, q_min=0.0,
                                r_min=0.0, r_max=20.0, r_step=0.02):
        """
        Transform B_l(q, q') to B_l(r, r') by applying the spherical Bessel
        transform (eq 8) along each q axis via direct quadrature.

        dq : radial q step of the polar dataset (e.g. angstrom^-1 / bin).
        q_min : q value of the first radial bin (nonzero if the polar
            transform used radial_min > 0).
        r_min, r_max, r_step : real-space grid, in the reciprocal units of q.

        The transform kernel is nondimensionalized by q_max (the quadrature
        uses (q / q_max)^2 d(q / q_max) per axis), so the output B_l(r, r')
        is dimensionless rather than carrying units of q^6. Combined with
        the unit-mean intensity normalization in rescale_intensity, this
        keeps PADF values of order 1.
        """
        # Equation 12, 13
        # apply bessel transform twice for each l
        # → shape = (l, r, r')

        # Step one is to define q
        q = q_min + torch.arange(0, Bl_mats.shape[1], dtype=torch.float64) * dq
        q_max = float(q[-1])
        r = torch.arange(r_min, r_max, r_step, dtype=torch.float64)
        self.r = r.numpy()
        real_Bl = torch.zeros((Bl_mats.shape[0], r.shape[0], r.shape[0]), dtype=torch.float64)

        arg = 2 * torch.pi * torch.outer(r, q)  # Shape len(r) x len(q)
        for l in range(len(l_values)):
            Bl = Bl_mats[l].to(dtype=torch.float64) # The corresponding q x q' matrix
            jl = torch.from_numpy(spherical_jn(l_values[l], arg)).to(dtype=torch.float64)

            # Representing DSBT (eq 12) as a transformation matrix "sbessel",
            # with q scaled by q_max so the quadrature is dimensionless
            sbessel = 4 * torch.pi * jl * (q**2) * dq / q_max**3
            # Applying it twice (once along each axis)
            real_Bl[l] = sbessel @ Bl @ sbessel.T * (-1)**l_values[l]
        
        return real_Bl

    def reconstruct_PADF(self, real_Bl, l_values, Na, per_theta=True, l_taper=False):
        """
        Reconstruct the pair-angle distribution function from B_l(r, r')
        For each theta:
        - Sum Pl(cos theta) times B_l() over all l
        - Multiply by n_alpha * 2 pi

        Theta goes from 0 to pi in 180 steps. Na scales the result; with the
        unit-mean intensity normalization the overall scale is arbitrary.

        per_theta : bool
            The Legendre sum is the real-space correlation C(r, r', cos theta)
            (pypadf paper eq. 10). If True, multiply by |sin theta| to return a
            quantity proportional to Theta(r, r', theta) (pypadf paper eq. 11).
        l_taper : bool
            Weight each order by order_taper(l, l_max), which suppresses the
            ringing in theta from truncating the Legendre sum at l_max.
        """
        padf = torch.zeros((real_Bl.shape[1], real_Bl.shape[2], 180), dtype=torch.float64)
        l_values = np.asarray(l_values).astype(int)
        weights = order_taper(l_values, l_values[-1] + 2) if l_taper else np.ones(len(l_values))
        theta = np.linspace(0, np.pi, 180)
        self.theta = theta
        self.theta_deg = np.degrees(theta)
        cos_theta = torch.cos(torch.as_tensor(theta, dtype=torch.float64))
        chosen_l_values = l_values[1:]

        # Sum over all l
        for l in range(len(chosen_l_values)): # skipping l = 0
            Pl = torch.special.legendre_polynomial_p(cos_theta, int(chosen_l_values[l]))
            Bl = real_Bl[l + 1] * weights[l + 1] # skipping l = 0
            padf += Bl[:, :, np.newaxis] * Pl[np.newaxis, np.newaxis, :]

        padf *= 2 * torch.pi * Na
        if per_theta:
            padf *= torch.as_tensor(np.abs(np.sin(theta)), dtype=torch.float64)
        return padf

    def plot_g2_g3(
        self,
        r_min: float | None = None,
        r_max: float | None = None,
        r_display_power: int = 1,
        r_max_display: float | None = None,
        r_search_min: float = 0.5,
        r_marks=None,
        markers=None,
        title: str | None = None,
        figsize: tuple[float, float] = (4.8, 4.4),
        returnfig: bool = False,
    ):
        """
        Stacked 2-body / 3-body correlation summary of the PADF, after
        atomode's G3 explorer. The two panels share the radial axis.

        Top panel: 3-body correlation map - Theta(r, r', theta) integrated
        over r in [r_min, r_max] (symmetrized over both radial axes),
        plotted as angle theta vs r'.

        Bottom panel: 2-body correlation profile - the |sin(theta)|-weighted
        RMS of the r = r' diagonal over the interior angular range
        (15-165 deg),
            sqrt( sum_theta Theta(r, r, theta)^2 sin(theta)
                  / sum_theta sin(theta) ).
        This stands in for the isotropic pair correlation (whose true l = 0
        component is removed by the correlation mean subtraction; a plain
        sin-weighted integral of Theta cancels to ~zero by Legendre
        orthogonality). It peaks at the radial shells where the PADF has
        angular structure and is used to choose the near-neighbor shell;
        the integration shell [r_min, r_max] is shaded.

        Both panels are weighted by r^(2 * r_display_power) for display, and
        the map's color limits are taken from the interior angular range
        (15-165 deg) so the collinear theta = 0 / 180 bands saturate rather
        than compressing the color scale.

        Parameters
        ----------
        r_min, r_max : float or None
            Integration shell bounds in the r units of the PADF (Angstroms
            for calibrated data). If None, the first peak of g2 is selected
            automatically: from the last point below 5% of the peak height
            up to the first local minimum past the peak (atomode's default
            shell heuristic).
        r_display_power : int
            Display weighting exponent; each panel is multiplied by
            r^(2 * r_display_power). 0 plots the raw values.
        r_max_display : float or None
            Upper limit of the radial axes; defaults to r.max().
        r_search_min : float
            Ignore r below this value when auto-locating the g2 peak
            (excludes the r ~ 0 reconstruction artifacts).
        r_marks : sequence of float or None
            Radii (e.g. the 1st/2nd/3rd neighbor shell distances) marked
            with dotted vertical lines on both panels.
        markers : sequence of (r, theta_deg) or None
            Expected 3-body maxima overlaid on the map as open circles -
            e.g. the (shell distance, arm-arm angle) targets from a known
            structure. A third element per tuple, if present, scales the
            marker size (relative multiplicity).
        title : str or None
            Figure title; a shell summary is used if None.
        returnfig : bool
            Return (fig, (ax_g2, ax_g3)) instead of calling plt.show().
        """
        r = np.asarray(self.r)
        theta = np.asarray(self.theta)  # radians
        theta_deg = np.asarray(self.theta_deg)
        arr = self.padf.numpy() if hasattr(self.padf, "numpy") else np.asarray(self.padf)

        # pair profile: sin(theta)-weighted RMS of the diagonal over the
        # interior angles (a plain integral cancels by Legendre orthogonality
        # since the l = 0 term is removed by the mean subtraction)
        diag = np.einsum("iik->ik", arr)
        interior_t = (theta_deg > 15) & (theta_deg < 165)
        w_t = np.abs(np.sin(theta[interior_t]))
        g2 = np.sqrt((diag[:, interior_t] ** 2 * w_t[None, :]).sum(axis=1) / w_t.sum())

        # auto-select the first g2 peak if no shell was given
        if r_min is None or r_max is None:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
            kernel /= kernel.sum()
            smooth = np.convolve(np.where(r < r_search_min, 0.0, g2), kernel, mode="same")
            i_pk = None
            for idx in range(1, smooth.size - 1):
                if smooth[idx] > 0 and smooth[idx] >= smooth[idx - 1] and smooth[idx] > smooth[idx + 1]:
                    i_pk = idx
                    break
            if i_pk is None:
                i_pk = int(np.argmax(smooth))
            below = np.flatnonzero(smooth[:i_pk] < 0.05 * smooth[i_pk])
            i_lo = int(below[-1]) if below.size else int(np.searchsorted(r, r_search_min))
            i_hi = min(smooth.size - 1, 2 * i_pk - i_lo)
            for idx in range(i_pk + 1, smooth.size - 1):
                if smooth[idx] <= smooth[idx - 1] and smooth[idx] <= smooth[idx + 1]:
                    i_hi = idx
                    break
            if r_min is None:
                r_min = float(r[i_lo])
            if r_max is None:
                r_max = float(r[i_hi])

        shell = (r >= r_min) & (r <= r_max)
        if not np.any(shell):
            shell[int(np.argmin(np.abs(r - 0.5 * (r_min + r_max))))] = True

        # g3 map: integrate over the shell, symmetrized over both r axes
        g3_map = 0.5 * (arr[shell].sum(axis=0) + arr[:, shell].sum(axis=1))  # (Nr', Ntheta)

        # display weighting and color limits
        if r_max_display is None:
            r_max_display = float(r[-1])
        weight = r ** (2 * r_display_power)
        g2_disp = g2 * weight
        g3_disp = (g3_map * weight[:, None]).T  # (Ntheta, Nr')
        rs = r <= r_max_display
        interior = (theta_deg > 15) & (theta_deg < 165)
        vmax = np.quantile(np.abs(g3_disp[np.ix_(interior, rs)]), 0.999)

        fig, (ax_g3, ax_g2) = plt.subplots(
            2, 1, sharex=True, figsize=figsize,
            gridspec_kw={"height_ratios": [2, 1]},
            constrained_layout=True,
        )
        im = ax_g3.pcolormesh(
            r[rs], theta_deg, g3_disp[:, rs],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
        )
        ax_g3.set_ylabel("$\\theta$ (deg)")
        ax_g3.set_ylim(0, 180)
        ax_g3.set_yticks(np.arange(0, 181, 30))
        fig.colorbar(im, ax=[ax_g3, ax_g2], label="3-body correlation", pad=0.02)

        if markers is not None:
            for mk in markers:
                r_mk, t_mk = mk[0], mk[1]
                size = 12.0 * (mk[2] ** 0.5) if len(mk) > 2 else 10.0
                if r_mk <= r_max_display:
                    ax_g3.plot(r_mk, t_mk, "o", ms=max(size, 5.0), mfc="none",
                               mec="k", mew=1.0)

        ax_g2.plot(r[rs], g2_disp[rs], "k-", lw=1)
        ax_g2.axvspan(r_min, r_max, color="C1", alpha=0.25)
        ax_g2.axhline(0, color="0.85", lw=0.8, zorder=0)

        if r_marks is not None:
            for r_mk in r_marks:
                for ax in (ax_g3, ax_g2):
                    ax.axvline(r_mk, color="k", ls=":", lw=0.8)
        ax_g2.set_ylabel("2-body corr.")
        ax_g2.set_xlabel("r, r' (Å)")

        ax_g3.set_title(
            title if title is not None
            else f"shell r = {r_min:.2f} - {r_max:.2f}",
            fontsize=10,
        )

        if returnfig:
            return fig, (ax_g3, ax_g2)
        plt.show()

    def simple_plot(
        self,
        padf=None,
        r_display_power: int = 1,
        r_max_display: float | None = None,
        r_fixed=None,
        theta_marks=None,
        r_marks=None,
        title=None,
        axsize: tuple[float, float] = (4.5, 3.5),
        returnfig: bool = False,
    ):
        """
        Maps of the PADF against theta (y-axis, 0 to 180 deg) and r (x-axis),
        one panel per volume in a single row: the r = r' diagonal
        Theta(r, r, theta), or Theta(r_fixed, r', theta) when r_fixed is given.
        The display is weighted by (r r')^r_display_power.

        padf : (Nr, Nr, Ntheta) array, or a list of them, or None
            Volumes on the same (r, theta) grid, e.g. a model PADF; defaults to self.padf.
        r_fixed : float, list of float (one per panel) or None
            Fixed first radius (Angstroms) for each panel.
        theta_marks, r_marks : sequence of float, list of them (one per panel) or None
            Angles (degrees) and radii (Angstroms) marked with dotted lines.
        title : str or list of str (one per panel)
        Each panel has symmetric color limits from its largest |value| at
        15-165 deg and r <= r_max_display.
        """
        vols = [self.padf] if padf is None else (list(padf) if isinstance(padf, (list, tuple)) else [padf])
        n = len(vols)
        titles = _per_panel(title, n)
        r_fixeds = list(r_fixed) if isinstance(r_fixed, (list, tuple)) else [r_fixed] * n
        theta_markss = _per_panel(theta_marks, n, nested=True)
        r_markss = _per_panel(r_marks, n, nested=True)
        r = np.asarray(self.r)
        theta_deg = np.asarray(self.theta_deg)
        rs = r <= (r.max() if r_max_display is None else r_max_display)
        interior = (theta_deg > 15) & (theta_deg < 165)

        fig, axs = plt.subplots(1, n, figsize=(axsize[0] * n, axsize[1]), squeeze=False, constrained_layout=True)
        for ax, vol, t, r0, t_marks, rm_marks in zip(axs[0], vols, titles, r_fixeds, theta_markss, r_markss):
            vol = vol.numpy() if isinstance(vol, torch.Tensor) else np.asarray(vol)
            if r0 is None:
                m = np.einsum("iik->ik", vol)[rs] * (r[rs] ** (2 * r_display_power))[:, None]
                xlabel, label = "r = r' (Å)", rf"$\Theta(r, r, \theta) \cdot r^{{{2 * r_display_power}}}$"
            else:
                i0 = int(np.argmin(np.abs(r - r0)))
                m = vol[i0][rs] * ((r[i0] * r[rs]) ** r_display_power)[:, None]
                xlabel, label = f"r' (Å), r = {r[i0]:.2f} Å", rf"$\Theta(r, r', \theta) \cdot (r r')^{{{r_display_power}}}$"
            vmax = np.abs(m[:, interior]).max()
            im = ax.pcolormesh(r[rs], theta_deg, m.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               shading="auto")
            for th in t_marks or []:
                ax.axhline(th, color="k", ls=":", lw=1)
            for rm in rm_marks or []:
                ax.axvline(rm, color="k", ls=":", lw=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\theta$ (deg)")
            ax.set_ylim(0, 180)
            ax.set_yticks(np.arange(0, 181, 30))
            fig.colorbar(im, ax=ax, label=label)
            if t is not None:
                ax.set_title(t)
        if returnfig:
            return fig, axs[0]
        plt.show()

    def plot_angular_correlation(
        self,
        q: float,
        figsize: tuple[float, float] = (5, 3.5),
        returnfig: bool = False,
    ):
        """
        Median-subtracted angular correlation C(dphi, q, q') at the radial bin
        nearest q (1/Angstroms), against q' (x-axis) and dphi (y-axis).
        """
        C = self.ang_corr_corrected.numpy()
        q_axis = self.q
        iq = int(np.argmin(np.abs(q_axis - q)))
        dphi = np.arange(C.shape[0]) * 360.0 / C.shape[0]
        M = C[:, iq, :]
        vmax = np.quantile(np.abs(M), 0.995)
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        im = ax.pcolormesh(q_axis, dphi, M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_xlabel("q' (1/Å)")
        ax.set_ylabel(r"$\Delta\phi$ (deg)")
        ax.set_ylim(0, 360)
        ax.set_yticks(np.arange(0, 361, 60))
        ax.set_title(f"q = {q_axis[iq]:.3f} 1/Å")
        fig.colorbar(im, ax=ax, label=r"$C(\Delta\phi, q, q')$")
        if returnfig:
            return fig, ax
        plt.show()


# Band-limit a model g3(r, r', theta) to what a PADF measurement can resolve
# (pypadf blfilter.py approach): expand in P_l(cos theta) for even l < l_max,
# take each order to the measured q points with the order-l spherical Bessel
# transform, and transform back.
def bandlimit_volume(
    G: NDArray,
    r: NDArray,
    theta_deg: NDArray,
    q: NDArray,
    l_max: int = 40,
) -> NDArray:
    """
    Parameters
    ----------
    G : (Nr, Nr, Ntheta) array
        Volume on a uniform r grid, theta in [0, 180] degrees.
    r : (Nr,) array
        Radial grid (Angstroms), uniformly spaced.
    theta_deg : (Ntheta,) array
        Angles in degrees.
    q : (Nq,) array
        Measured q values (1/Angstroms, no 2 pi), uniformly spaced.
    l_max : int
        Even orders l = 0, 2, ..., < l_max are kept (same as extract_Bl_matrices).

    Returns
    -------
    (Nr, Nr, Ntheta) array, the band-limited volume.
    """
    G = torch.as_tensor(np.asarray(G), dtype=torch.float64)
    theta = torch.deg2rad(torch.as_tensor(np.asarray(theta_deg), dtype=torch.float64))
    l_values = torch.arange(0, l_max, 2, dtype=torch.float64)

    # Legendre projection over theta with the sin(theta) d(theta) measure
    w = torch.abs(torch.sin(theta)) * torch.gradient(theta)[0]
    P = torch.special.legendre_polynomial_p(torch.cos(theta)[None, :], l_values[:, None])  # (Nl, Ntheta)
    Gl = torch.einsum("ijt,lt->lij", G, P * w[None, :]) * (2 * l_values[:, None, None] + 1) / 2

    out = torch.zeros_like(G)
    for k, l in enumerate(l_values.int().tolist()):
        out += _bandlimit_order(Gl[k], l, r, q)[:, :, None] * P[k][None, None, :]
    return out.numpy()


def order_taper(orders, order_max):
    """cos^2 taper of harmonic orders, 1 at order 0 and 0 at order_max."""
    return np.cos(0.5 * np.pi * np.asarray(orders, dtype=np.float64) / order_max) ** 2


def _bandlimit_order(Gl: torch.Tensor, l: int, r, q) -> torch.Tensor:
    """Order-l spherical Bessel transform of Gl(r, r') to the q points and back."""
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    jl = torch.from_numpy(spherical_jn(l, 2 * np.pi * np.outer(q, r)))  # (Nq, Nr), not in torch
    fwd = 4 * np.pi * jl * torch.from_numpy(r**2 * (r[1] - r[0]))[None, :]  # r -> q
    inv = 4 * np.pi * jl.T * torch.from_numpy(q**2 * (q[1] - q[0]))[None, :]  # q -> r
    return inv @ (fwd @ Gl @ fwd.T) @ inv.T


def _real_sph_harm_power(cos_t, sin_t, phi, l_values, weights, n_r, index):
    """
    sum_m A_lm(r) conj(A_lm(r')) for each l in l_values, where
    A_lm(r) = sum_v weights[v] Y_lm(v) binned to r by index (Nvec,) -> [0, n_r].
    Orthonormal associated Legendre functions by the standard stable
    recurrence in l at fixed m; m < 0 enters through |A_l,-m| = |A_lm|.
    """
    l_max = int(l_values.max())
    C = torch.zeros((len(l_values), n_r, n_r), dtype=torch.float64)
    row = {int(l): k for k, l in enumerate(l_values.tolist())}
    p_mm = torch.full_like(cos_t, (1 / (4 * np.pi)) ** 0.5)
    for m in range(l_max + 1):
        if m > 0:
            p_mm = -((2 * m + 1) / (2 * m)) ** 0.5 * sin_t * p_mm
        e = torch.polar(torch.ones_like(phi), m * phi) * weights
        p_prev, p_l = None, p_mm
        for l in range(m, l_max + 1):
            if l == m + 1:
                p_prev, p_l = p_l, (2 * m + 3) ** 0.5 * cos_t * p_l
            elif l > m + 1:
                a = ((4 * l * l - 1) / (l * l - m * m)) ** 0.5
                b = (((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1)) ** 0.5
                p_prev, p_l = p_l, a * (cos_t * p_l - b * p_prev)
            if l in row:
                A = torch.zeros(n_r + 1, dtype=torch.complex128).index_add_(0, index, p_l * e)[:n_r]
                C[row[l]] += (1.0 if m == 0 else 2.0) * torch.real(A.conj()[:, None] * A[None, :])
    return C


def padf_from_clusters(
    clusters,
    r: NDArray,
    theta_deg: NDArray,
    q: NDArray | None = None,
    l_max: int = 40,
    l_taper: bool = False,
) -> NDArray:
    """
    PADF of atomic models: the angular correlation of all interatomic vectors
    within each cluster, summed over clusters. Like the diffraction PADF it
    holds two-, three- and four-body terms (Martin et al. IUCrJ 2017), not
    only triplets sharing an atom.

    Uses the addition theorem: for each even l < l_max,
        C_l(r, r') = 2 pi sum_m A_lm(r) conj(A_lm(r')),
        A_lm(r) = sum over vectors at |d| = r of Y_lm(d / |d|), per unit volume,
    then Theta = |sin theta| sum_{l >= 2} C_l(r, r') P_l(cos theta), matching
    reconstruct_PADF (l = 0 is dropped, as the PADF's mean subtraction does).

    Parameters
    ----------
    clusters : iterable of (N, 3) arrays
        Atom positions (Angstroms), one array per cluster.
    r : (Nr,) array
        Uniform radial grid (Angstroms); vectors are binned with linear weights.
    theta_deg : (Ntheta,) array
        Angles in degrees.
    q : (Nq,) array or None
        Measured q values (1/Angstroms, no 2 pi). If given, each C_l is
        band-limited to them, as bandlimit_volume does.
    l_max : int
        Even orders l = 0, 2, ..., < l_max.
    l_taper : bool
        Weight each order by order_taper(l, l_max), as reconstruct_PADF does.

    Returns
    -------
    (Nr, Nr, Ntheta) array.
    """
    r_t = torch.as_tensor(np.asarray(r), dtype=torch.float64)
    dr = float(r_t[1] - r_t[0])
    n_r = r_t.numel()
    l_values = torch.arange(2, l_max, 2)  # l = 0 is dropped, as the PADF's median subtraction does
    shell = 4 * np.pi * torch.clamp(r_t, min=0.5 * dr) ** 2 * dr
    Cl = torch.zeros((len(l_values), n_r, n_r), dtype=torch.float64)
    for pos in clusters:
        pos = torch.as_tensor(np.asarray(pos), dtype=torch.float64)
        d = (pos[None, :, :] - pos[:, None, :]).reshape(-1, 3)
        dist = torch.linalg.norm(d, dim=1)
        keep = (dist > 0) & (dist >= float(r_t[0])) & (dist < float(r_t[-1]))
        d, dist = d[keep], dist[keep]
        cos_t = torch.clamp(d[:, 2] / dist, -1, 1)
        sin_t = torch.sqrt(1 - cos_t**2)
        phi = torch.atan2(d[:, 1], d[:, 0])
        # linear binning onto the r grid: each vector split between bins i0 and i0 + 1
        x = (dist - float(r_t[0])) / dr
        i0 = torch.floor(x).long()
        f = x - i0
        index = torch.cat([i0, i0 + 1])
        weights = torch.cat([1 - f, f]) / torch.cat([shell, shell[-1:]])[index]  # per unit volume
        cos_t, sin_t, phi = (torch.cat([t, t]) for t in (cos_t, sin_t, phi))
        Cl += _real_sph_harm_power(cos_t, sin_t, phi, l_values, weights, n_r, index)
    Cl *= 2 * np.pi

    theta = torch.deg2rad(torch.as_tensor(np.asarray(theta_deg), dtype=torch.float64))
    out = torch.zeros((n_r, n_r, theta.numel()), dtype=torch.float64)
    for k, l in enumerate(l_values.tolist()):
        C = Cl[k] if q is None else _bandlimit_order(Cl[k], l, r, q)
        if l_taper:
            C = C * float(order_taper(l, int(l_values[-1]) + 2))  # same as reconstruct_PADF
        out += C[:, :, None] * torch.special.legendre_polynomial_p(torch.cos(theta), l)[None, None, :]
    return (out * torch.abs(torch.sin(theta))[None, None, :]).numpy()


def _per_panel(value, n, nested=False):
    """One entry per panel: a list with one item per panel is used as is,
    anything else is repeated. With nested=True only a list of lists (or
    Nones) counts as per panel, so a flat list of marks is shared."""
    if isinstance(value, (list, tuple)) and len(value) == n:
        if not nested or all(v is None or isinstance(v, (list, tuple, np.ndarray)) for v in value):
            return list(value)
    return [value] * n
