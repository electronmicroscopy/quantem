"""
Pair vector correlation (PVC) analysis of diffraction patterns.

A diffraction pattern samples the 3D intensity I(q) of the illuminated volume,
the Fourier transform of its pair-vector distribution (the Patterson function),
on the Ewald sphere. PVC measures the mean and the second moment of the circular
harmonics of many patterns,

    M_n(k)     = < I_n(k) >
    S_n(k, k') = < conj(I_n(k)) I_n(k') >,

where I_n(k) is the coefficient of exp(i n phi) on the detector ring of radius k,
and maps them to real-space pair-vector distributions and correlations with one
of three rotation models:

    "sphere"  every 3D orientation is equally likely (liquids, amorphous solids,
              randomly oriented nanocrystals): the 3D pair vector correlation
              Theta(r, r', theta), which is the pair-angle distribution function.
    "beam"    the sample only rotates about the beam (2D materials, fiber texture
              along the beam): the correlation of the projected pair vectors,
              Theta(r, r', dphi), over 0 to 360 deg.
    "none"    a single orientation: the mean pattern gives the directional
              Patterson function of the projection.

Every second moment splits into the part carried by the mean pattern,
conj(M_n(k)) M_n(k'), which is order shared by all patterns, and the fluctuations
about it. Anything identical in every pattern (the bright-field disk and its
pixelation, ellipticity or miscentering, a static detector pattern) sits in the
mean part, so the fluctuation part removes it exactly. For randomly oriented
samples the mean pattern is isotropic and both parts carry the same structure,
which is why the sphere model uses the fluctuation part by default. The moments add over patterns and every solve is linear, so averaging
the outputs of subsets of patterns equals solving their summed moments.

The radial detector coordinate is |q| (1/A, no 2 pi). On the Ewald sphere the
scattering vector makes the polar angle beta = 90 deg + asin(lambda |q| / 2)
with the beam. With the spherical harmonic coefficients I_lm(q) of the 3D
intensity, a sample in orientation R gives

    I_n(k) = sum_l Y_ln(beta(k), 0) sum_m D^l_nm(R) I_lm(|q|),

so for the sphere model, exactly,

    S_n(k, k') = sum_l 4 pi / (2l + 1) Y_ln(beta, 0) Y_ln(beta', 0) B_l(q, q'),

with B_l the rotational invariants of the 3D intensity, solved by least squares
for each radius pair. No sin(theta) weights are needed, and the uncorrelated
background (n = 0, l = 0) never enters. The output is a density per unit
volume squared and per unit cos(theta), so it carries no |sin(theta)| factor.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import jv, spherical_jn

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffraction.polar_transform import (
    as_dataset4dstem,
    find_origin_angular_grid,
    polar_transform,
)


class PairVectorCorrelation:
    """
    Pair vector correlation (PVC) from many diffraction patterns.

    Parameters
    ----------
    data : Dataset4dstem, Polar4dstem, or a list / generator of them
        Diffraction patterns. 2D patterns and 3D stacks are wrapped as 4D-STEM
        data. A list or generator is accumulated one item at a time, for more
        patterns than fit in memory; all items need the same radial axis.
    energy : float or None
        Beam energy (eV), which sets the Ewald sphere curvature. None treats the
        Ewald sphere as flat.
    origin : (row, col) or None
        Pixel of q = 0. By default it is read from the dataset calibration, or
        found with find_origin_angular_grid when the calibration has no offset.
    ellipse_params : (a, b, theta_deg) or None
        Elliptical distortion, corrected in the polar transform.
    n_phi : int
        Angular bins of the polar transform; harmonics up to n_phi // 2 - 1 are kept.
    fq : callable, array or None
        Mean atomic scattering factor f(q), or a function of |q| (1/A). Patterns
        are divided by f(q)^2.
    k_min, k_max : float
        Radial range (1/A). Set k_min just outside the bright-field disk.
    taper : (float, float)
        Widths (1/A) of the cosine ramps of the radial window at k_min and k_max.
        The window applies in the real-space transforms only; each radius pair is
        solved on its own, so the window never mixes radii.
    batch_size : int
        Patterns per chunk when accumulating the moments.

    The moments are normalized by the mean intensity where the radial window is 1.
    A mean pattern with a strong n = 1 harmonic near k_min warns: kinematic
    patterns are centrosymmetric, so it points to a miscentered pattern or a
    bright-field disk edge leaking into the window.
    """

    def __init__(
        self,
        data,
        energy: float | None = None,
        origin=None,
        ellipse_params=None,
        n_phi: int = 180,
        fq=None,
        k_min: float = 0.0,
        k_max: float | None = None,
        taper: tuple[float, float] = (0.1, 0.25),
        batch_size: int = 512,
    ):
        self.energy = energy
        self.wavelength = electron_wavelength_angstrom(energy) if energy else 0.0
        items = list(data) if isinstance(data, (list, tuple)) else data
        if isinstance(data, (Dataset4dstem, Polar4dstem)) or hasattr(data, "array"):
            items = [data]

        sum_M = sum_S = None
        self.n_patterns = 0
        for item in items:
            polar = self._to_polar(item, origin, ellipse_params, n_phi)
            if sum_M is None:
                self._set_geometry(polar, fq, k_min, k_max, taper)
            elif polar.shape[-1] != self.q.size:
                raise ValueError("all datasets need the same radial axis")
            patterns = polar.tensor.reshape(-1, polar.shape[-2], polar.shape[-1]).to(torch.float64)
            for start in range(0, patterns.shape[0], batch_size):
                F = self._harmonics(patterns[start : start + batch_size])
                M = F.sum(dim=0)
                S = torch.einsum("bnk,bnj->nkj", F.conj(), F)
                sum_M = M if sum_M is None else sum_M + M
                sum_S = S if sum_S is None else sum_S + S
                self.n_patterns += F.shape[0]

        M = sum_M / self.n_patterns
        S = sum_S / self.n_patterns
        scale = float(
            M[0].real[self.window == 1].mean()
        )  # flat part of the window, clear of the bright-field edge
        self.mean_harmonics = (M / scale).numpy()  # (n, k)
        self.second_moment = (S / scale**2).numpy()  # (n, k, k')
        self._check_centering()

    # ------------------------------------------------------------------ moments

    @staticmethod
    def _to_polar(item, origin, ellipse_params, n_phi):
        if isinstance(item, Polar4dstem):
            return item
        ds = as_dataset4dstem(item)
        if origin is None:
            offset = np.asarray(ds.origin, dtype=float)[-2:]
            sampling = np.asarray(ds.sampling, dtype=float)[-2:]
            origin = -offset / sampling if np.any(offset != 0) else find_origin_angular_grid(ds)
        return polar_transform(
            ds, origin_array=origin, ellipse_params=ellipse_params, num_annular_bins=n_phi
        )

    def _set_geometry(self, polar, fq, k_min, k_max, taper):
        n_q = polar.shape[-1]
        self.n_phi = polar.shape[-2]
        self.n_max = self.n_phi // 2 - 1
        self.q = float(np.asarray(polar.origin)[3]) + np.arange(n_q) * float(
            np.asarray(polar.sampling)[3]
        )
        half_angle = np.arcsin(np.clip(self.wavelength * self.q / 2, 0, 1))
        self.beta = np.pi / 2 + half_angle  # polar angle of q from the beam axis
        self.q_perp = self.q * np.cos(half_angle)  # component of q perpendicular to the beam
        self.dq = np.gradient(self.q)
        self.dq_perp = np.gradient(self.q_perp)
        k_max = self.q[-1] if k_max is None else k_max
        self.window = _radial_window(self.q, k_min, k_max, taper)
        fq = fq(self.q) if callable(fq) else fq
        f2 = (
            np.ones(n_q)
            if fq is None
            else np.broadcast_to(np.asarray(fq, dtype=np.float64), (n_q,)) ** 2
        )
        self._f2 = torch.from_numpy(np.ascontiguousarray(f2))

    def _harmonics(self, patterns):
        """Circular harmonics (B, n_max + 1, Nk) of a stack of polar patterns (B, n_phi, Nk)."""
        F = torch.fft.rfft(patterns / self._f2[None, None, :], dim=1) / self.n_phi
        return F[:, : self.n_max + 1, :]

    def _check_centering(self, threshold: float = 0.05):
        """Warn when the mean pattern has an n = 1 harmonic near k_min larger than
        threshold times the mean intensity in the window (the moments are normalized
        so that this mean is 1)."""
        band = (
            (self.window > 0) & (self.window < 1) & (self.q < self.q[self.window > 0].min() + 0.5)
        )
        if not band.any():
            return
        self.n1_near_k_min = float(np.abs(self.mean_harmonics[1, band]).max())
        if self.n1_near_k_min > threshold:
            warnings.warn(
                f"The mean pattern has an n = 1 harmonic of {self.n1_near_k_min:.0%} of the mean intensity near k_min. "
                "Kinematic patterns are centrosymmetric, so check the origin and ellipticity, "
                "or raise k_min so the bright-field disk edge stays out of the window."
            )

    def _part(self, part: str):
        M = self.mean_harmonics
        mean = np.conj(M)[:, :, None] * M[:, None, :]
        if part == "total":
            return self.second_moment
        if part == "mean":
            return mean
        if part == "fluctuation":
            return self.second_moment - mean
        raise ValueError(f"part must be 'total', 'mean' or 'fluctuation', got {part!r}")

    def mean_fraction(self) -> float:
        """
        Share of the second moment (harmonics n >= 1, radii in the window)
        carried by the mean pattern: about 0 for liquids, amorphous solids and
        randomly oriented domains, and 1 for a single orientation.
        """
        w = np.outer(self.window, self.window)[None]
        mean = np.abs(self._part("mean")[1:]) * w
        return float(mean.sum() / (np.abs(self.second_moment[1:]) * w).sum())

    # ------------------------------------------------------------------ solvers

    def correlation(
        self,
        model: str = "sphere",
        part: str | None = None,
        l_max: int = 40,
        n_max: int | None = None,
        r_max: float = 8.0,
        r_step: float = 0.05,
        order_taper: bool = True,
    ):
        """
        Pair vector correlation Theta(r, r', theta) in real space.

        model : "sphere" (3D, theta from 0 to 180 deg) or "beam" (projected,
            dphi from 0 to 360 deg); see the module docstring.
        part : "total", "mean" or "fluctuation" second moment; by default
            "fluctuation" for the sphere model and "total" for the beam model,
            which is often applied to single patterns.
        l_max : largest (even) spherical harmonic order, for the sphere model.
        n_max : largest circular harmonic order, for the beam model.
        r_max, r_step : real-space grid (A), including r_max.
        order_taper : weight the orders by a cos^2 taper to zero just past the
            largest order, which suppresses ringing in theta.

        Returns theta_deg (Ntheta,) and Theta (Nr, Nr, Ntheta); also stored on
        the object (r, theta_deg, Theta, and the order coefficients G of the
        orders l_values or n_values) with the settings for model_correlation.
        """
        part = ("fluctuation" if model == "sphere" else "total") if part is None else part
        self.r = np.arange(0, r_max + 0.5 * r_step, r_step)
        self.model, self.part, self.order_taper = model, part, order_taper
        S = self._part(part)
        if model == "sphere":
            self.l_values = np.arange(2, l_max + 1, 2)
            self.B = _solve_sphere(S.real, self.beta, self.l_values)
            G = self._real_space_sphere(self.B)
            self.G = G.numpy()
            self.theta_deg = np.arange(0, 181, 1.0)
            self.Theta = _legendre_sum(G, self.l_values, self.theta_deg, order_taper)
        elif model == "beam":
            self.n_values = np.arange(1, (self.n_max if n_max is None else n_max) + 1)
            G = self._real_space_beam(S[self.n_values])
            self.G = G.numpy()
            self.theta_deg = np.arange(0, 361, 1.0)
            self.Theta = _harmonic_sum(G, self.n_values, self.theta_deg, order_taper)
        else:
            raise ValueError(f"model must be 'sphere' or 'beam', got {model!r}")
        return self.theta_deg, self.Theta

    def _kernel_sphere(self, ell, r=None):
        """(Nr, Nk) spherical Bessel transform q -> r with the radial window."""
        r = self.r if r is None else r
        w = self.q**2 * self.dq * self.window
        return torch.from_numpy(spherical_jn(ell, 2 * np.pi * np.outer(r, self.q)) * w[None, :])

    def _kernel_beam(self, n, r=None):
        """(Nr, Nk) Hankel transform of order n, q_perp -> r, with the radial window."""
        r = self.r if r is None else r
        w = self.q_perp * self.dq_perp * self.window
        return torch.from_numpy(jv(n, 2 * np.pi * np.outer(r, self.q_perp)) * w[None, :])

    def _real_space_sphere(self, B):
        G = torch.zeros((len(self.l_values), self.r.size, self.r.size), dtype=torch.float64)
        for k, ell in enumerate(self.l_values):
            K = self._kernel_sphere(int(ell))
            G[k] = 2 * np.pi * (4 * np.pi) ** 3 * K @ torch.from_numpy(B[k]) @ K.T
        return G

    def _real_space_beam(self, S):
        G = torch.zeros((len(self.n_values), self.r.size, self.r.size), dtype=torch.complex128)
        for k, n in enumerate(self.n_values):
            K = self._kernel_beam(int(n)).to(torch.complex128)
            G[k] = (2 * np.pi) ** 2 * K @ torch.from_numpy(np.ascontiguousarray(S[k])) @ K.T
        return G

    def patterson(
        self,
        r_max: float = 8.0,
        r_step: float = 0.05,
        include_isotropic: bool = False,
        order_taper: bool = True,
    ):
        """
        Directional Patterson function P(row, col) of the projection, from the
        mean pattern (the "none" rotation model), on a square grid of half-width
        r_max (A). By default the isotropic n = 0 part is left out, so the map
        shows the directional pair vectors.

        Returns the (N, N) image and its pixel size (A).
        """
        r = np.arange(0, np.sqrt(2) * r_max + r_step, r_step)
        n_values = np.arange(0 if include_isotropic else 1, self.n_max + 1)
        taper = _order_taper(n_values, self.n_max + 1) if order_taper else np.ones(n_values.size)
        P_n = np.stack(
            [
                2
                * np.pi
                * (1j**n)
                * (self._kernel_beam(int(n), r).numpy() @ self.mean_harmonics[n])
                for n in n_values
            ]
        )  # (Nn, Nr)
        x = np.arange(-r_max, r_max + 0.5 * r_step, r_step)
        row, col = np.meshgrid(x, x, indexing="ij")
        rho, phi = np.hypot(row, col), np.arctan2(row, col)
        image = np.zeros(rho.shape)
        for t, n, p in zip(taper, n_values, P_n):
            p_rho = np.interp(rho, r, p.real) + 1j * np.interp(rho, r, p.imag)
            image += t * (p_rho * np.exp(1j * n * phi)).real * (1 if n == 0 else 2)
        return image, r_step

    # ------------------------------------------------------------------ model

    def model_correlation(self, clusters):
        """
        Pair vector correlation of atomic models, processed like the last
        correlation() call: same model, grid, q range, radial window and order
        taper. The correlation of all interatomic vectors within each cluster is
        summed over clusters, so it holds two-, three- and four-body terms, as
        the diffraction correlation does. For the beam model the positions are
        projected along the beam (the last coordinate).

        clusters : iterable of (N, 3) arrays of positions (A), (row, col, beam).
        Returns Theta on the stored (r, theta_deg) grid.
        """
        r_model = np.arange(0, self.r[-1] + 2.0, self.r[1] - self.r[0])
        if self.model == "sphere":
            H = _cluster_power_sphere(clusters, r_model, self.l_values)  # sum_m conj(P_lm) P_lm
            G = torch.zeros((len(self.l_values), self.r.size, self.r.size), dtype=torch.float64)
            dr = r_model[1] - r_model[0]
            for k, ell in enumerate(self.l_values):
                Kr = torch.from_numpy(
                    spherical_jn(int(ell), 2 * np.pi * np.outer(self.q, r_model))
                    * (r_model**2 * dr)[None, :]
                )
                B = 4 * np.pi * Kr @ H[k] @ Kr.T  # band-limited to the measured q
                K = self._kernel_sphere(int(ell))
                G[k] = 2 * np.pi * (4 * np.pi) ** 3 * K @ B @ K.T
            return _legendre_sum(G, self.l_values, self.theta_deg, self.order_taper)
        H = _cluster_power_beam(clusters, r_model, self.n_values)
        G = torch.zeros((len(self.n_values), self.r.size, self.r.size), dtype=torch.complex128)
        dr = r_model[1] - r_model[0]
        for k, n in enumerate(self.n_values):
            Kr = torch.from_numpy(
                jv(int(n), 2 * np.pi * np.outer(self.q_perp, r_model)) * (r_model * dr)[None, :]
            )
            Kr = Kr.to(torch.complex128)
            S = (2 * np.pi) ** 2 * Kr @ H[k] @ Kr.T
            K = self._kernel_beam(int(n)).to(torch.complex128)
            G[k] = (2 * np.pi) ** 2 * K @ S @ K.T
        return _harmonic_sum(G, self.n_values, self.theta_deg, self.order_taper)

    # ------------------------------------------------------------------ plotting

    def plot_correlation(
        self,
        Theta=None,
        r_fixed=None,
        theta_marks=None,
        r_marks=None,
        title=None,
        r_display_power: float | None = None,
        r_max_display: float | None = None,
        subtract_median: bool | None = None,
        axsize: tuple[float, float] = (4.5, 3.5),
        returnfig: bool = False,
    ):
        """
        Maps of Theta against angle (y-axis) and r (x-axis), one panel per volume
        in a single row: the r = r' diagonal, or Theta(r_fixed, r', theta) when
        r_fixed is given (one value or one per panel). The display is weighted by
        (r r')^r_display_power (default 1 for the sphere model, 0.5 for beam).
        subtract_median (default on for beam) subtracts the median over the
        interior angles at each r: the self-correlation bands at 0 and 180 deg
        otherwise offset every interior angle. Color limits use 15-165 deg and
        r <= r_max_display.

        Theta : (Nr, Nr, Ntheta) array, a list of them, or None for self.Theta.
        theta_marks, r_marks : angles (deg) and radii (A) marked with dotted
            lines, shared or one list per panel.
        """
        vols = (
            [self.Theta]
            if Theta is None
            else (list(Theta) if isinstance(Theta, (list, tuple)) else [Theta])
        )
        n = len(vols)
        beam = self.model == "beam"
        power = (0.5 if beam else 1.0) if r_display_power is None else r_display_power
        subtract = beam if subtract_median is None else subtract_median
        titles = _per_panel(title, n)
        r_fixeds = list(r_fixed) if isinstance(r_fixed, (list, tuple)) else [r_fixed] * n
        theta_markss = _per_panel(theta_marks, n, nested=True)
        r_markss = _per_panel(r_marks, n, nested=True)
        r, theta = self.r, self.theta_deg
        rs = r <= (r[-1] if r_max_display is None else r_max_display)
        interior = (theta % 180 > 15) & (theta % 180 < 165)
        theta_max = theta[-1]

        fig, axs = plt.subplots(
            1, n, figsize=(axsize[0] * n, axsize[1]), squeeze=False, constrained_layout=True
        )
        for ax, vol, t, r0, t_marks, rm_marks in zip(
            axs[0], vols, titles, r_fixeds, theta_markss, r_markss
        ):
            vol = vol.numpy() if isinstance(vol, torch.Tensor) else np.asarray(vol)
            if r0 is None:
                m = np.einsum("iik->ik", vol)[rs] * (r[rs] ** (2 * power))[:, None]
                xlabel = "r = r' (Å)"
            else:
                i0 = int(np.argmin(np.abs(r - r0)))
                m = vol[i0][rs] * ((r[i0] * r[rs]) ** power)[:, None]
                xlabel = f"r' (Å), r = {r[i0]:.2f} Å"
            if subtract:
                m = m - np.median(m[:, interior], axis=1, keepdims=True)
            vmax = np.abs(m[:, interior]).max()
            im = ax.pcolormesh(
                r[rs], theta, m.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
            )
            for th in t_marks or []:
                ax.axhline(th, color="k", ls=":", lw=1)
            for rm in rm_marks or []:
                ax.axvline(rm, color="k", ls=":", lw=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\Delta\phi$ (deg)" if beam else r"$\theta$ (deg)")
            ax.set_ylim(0, theta_max)
            ax.set_yticks(np.arange(0, theta_max + 1, 30 if theta_max <= 180 else 60))
            fig.colorbar(im, ax=ax, label=rf"$\Theta \cdot (r r')^{{{power:g}}}$")
            if t is not None:
                ax.set_title(t)
        if returnfig:
            return fig, axs[0]
        plt.show()


# ---------------------------------------------------------------------- helpers


def _radial_window(q, k_min, k_max, taper):
    """1 inside [k_min + taper[0], k_max - taper[1]], cosine ramps to 0 at k_min and k_max."""
    lo, hi = taper
    w = np.ones_like(q)
    w[(q < k_min) | (q > k_max)] = 0
    ramp = (q >= k_min) & (q < k_min + lo)
    w[ramp] = np.sin(0.5 * np.pi * (q[ramp] - k_min) / lo) ** 2
    ramp = (q > k_max - hi) & (q <= k_max)
    w[ramp] = np.cos(0.5 * np.pi * (q[ramp] - (k_max - hi)) / hi) ** 2
    return w


def _order_taper(orders, order_max):
    """cos^2 taper of harmonic orders, 1 at order 0 and 0 at order_max."""
    return np.cos(0.5 * np.pi * np.asarray(orders, dtype=np.float64) / order_max) ** 2


def _ylm_azimuth_zero(l_max, cos_beta):
    """Orthonormal Y_lm(beta, 0) for 0 <= m <= l <= l_max, as an (l_max + 1, l_max + 1, N) array [l, m]."""
    x = torch.as_tensor(np.asarray(cos_beta), dtype=torch.float64)
    s = torch.sqrt(torch.clamp(1 - x**2, min=0))
    Y = torch.zeros((l_max + 1, l_max + 1, x.numel()), dtype=torch.float64)
    p_mm = torch.full_like(x, (1 / (4 * np.pi)) ** 0.5)
    for m in range(l_max + 1):
        if m > 0:
            p_mm = -(((2 * m + 1) / (2 * m)) ** 0.5) * s * p_mm
        Y[m, m] = p_mm
        if m == l_max:
            break
        p_prev, p_l = p_mm, (2 * m + 3) ** 0.5 * x * p_mm
        Y[m + 1, m] = p_l
        for ell in range(m + 2, l_max + 1):
            a = ((4 * ell * ell - 1) / (ell * ell - m * m)) ** 0.5
            b = (((ell - 1) ** 2 - m * m) / (4 * (ell - 1) ** 2 - 1)) ** 0.5
            p_prev, p_l = p_l, a * (x * p_l - b * p_prev)
            Y[ell, m] = p_l
    return Y


def _solve_sphere(S, beta, l_values):
    """
    Least-squares B_l(q, q') for each radius pair from the real second moment
    S (n, k, k'), using the harmonics n = 1 ... l_max (n = 0 holds the l = 0
    background and is not used).
    """
    l_max = int(l_values[-1])
    n_values = np.arange(1, min(l_max, S.shape[0] - 1) + 1)
    Y = _ylm_azimuth_zero(l_max, np.cos(beta))[torch.as_tensor(l_values)][
        :, torch.as_tensor(n_values)
    ]  # (Nl, Nn, Nk)
    c = torch.as_tensor(4 * np.pi / (2 * l_values + 1), dtype=torch.float64)
    S_t = torch.from_numpy(np.ascontiguousarray(S[n_values]))  # (Nn, Nk, Nk)
    n_k = S.shape[1]
    B = torch.zeros((len(l_values), n_k, n_k), dtype=torch.float64)
    for k in range(n_k):
        A = torch.einsum("l,lnj,ln->jnl", c, Y, Y[:, :, k])  # (Nk', Nn, Nl)
        B[:, k, :] = torch.linalg.lstsq(A, S_t[:, k, :].T[:, :, None]).solution[:, :, 0].T
    return B.numpy()


def _legendre_sum(G, l_values, theta_deg, order_taper):
    """Theta(r, r', theta) = sum_l taper_l G_l(r, r') P_l(cos theta)."""
    cos_t = torch.cos(torch.deg2rad(torch.as_tensor(theta_deg, dtype=torch.float64)))
    taper = (
        _order_taper(l_values, int(l_values[-1]) + 2) if order_taper else np.ones(len(l_values))
    )
    out = torch.zeros((G.shape[1], G.shape[2], cos_t.numel()), dtype=torch.float64)
    for k, ell in enumerate(l_values):
        out += (
            float(taper[k])
            * G[k][:, :, None]
            * torch.special.legendre_polynomial_p(cos_t, int(ell))[None, None, :]
        )
    return out.numpy()


def _harmonic_sum(G, n_values, theta_deg, order_taper):
    """Theta(r, r', dphi) = 2 Re sum_n taper_n G_n(r, r') exp(i n dphi), n >= 1."""
    phi = torch.deg2rad(torch.as_tensor(theta_deg, dtype=torch.float64))
    taper = (
        _order_taper(n_values, int(n_values[-1]) + 1) if order_taper else np.ones(len(n_values))
    )
    n_t = torch.from_numpy(n_values.astype(np.float64))
    E = torch.from_numpy(taper)[:, None] * torch.exp(
        1j * n_t[:, None] * phi[None, :]
    )  # (Nn, Ntheta)
    return (2 * torch.real(torch.einsum("nij,nt->ijt", G, E))).numpy()


def _bin_vectors(dist, r):
    """Linear binning of vector lengths onto the uniform grid r: (index, weight) for both neighbors."""
    dr = r[1] - r[0]
    x = (dist - float(r[0])) / dr
    i0 = torch.floor(x).long()
    f = x - i0
    return torch.cat([i0, i0 + 1]), torch.cat([1 - f, f])


def _cluster_power_sphere(clusters, r, l_values):
    """sum_m conj(P_lm(r)) P_lm(r') summed over clusters, P_lm per unit volume, as an (Nl, Nr, Nr) tensor."""
    r_t = torch.as_tensor(r, dtype=torch.float64)
    dr = float(r_t[1] - r_t[0])
    shell = torch.clamp(r_t, min=0.5 * dr) ** 2 * dr
    shell = torch.cat([shell, shell[-1:]])
    l_max = int(l_values[-1])
    row = {int(ell): k for k, ell in enumerate(l_values)}
    H = torch.zeros((len(l_values), r.size, r.size), dtype=torch.float64)
    for pos in clusters:
        pos = torch.as_tensor(np.asarray(pos), dtype=torch.float64)
        d = (pos[None, :, :] - pos[:, None, :]).reshape(-1, 3)
        dist = torch.linalg.norm(d, dim=1)
        keep = (dist > 0) & (dist >= float(r_t[0])) & (dist < float(r_t[-1]))
        d, dist = d[keep], dist[keep]
        index, weight = _bin_vectors(dist, r)
        weight = weight / shell[index]
        cos_t = torch.cat(
            [torch.clamp(d[:, 2] / dist, -1, 1)] * 2
        )  # polar angle from the beam axis
        sin_t = torch.sqrt(1 - cos_t**2)
        phi = torch.cat([torch.atan2(d[:, 0], d[:, 1])] * 2)
        p_mm = torch.full_like(cos_t, (1 / (4 * np.pi)) ** 0.5)
        for m in range(l_max + 1):
            if m > 0:
                p_mm = -(((2 * m + 1) / (2 * m)) ** 0.5) * sin_t * p_mm
            e = torch.polar(torch.ones_like(phi), m * phi) * weight
            p_prev, p_l = None, p_mm
            for ell in range(m, l_max + 1):
                if ell == m + 1:
                    p_prev, p_l = p_l, (2 * m + 3) ** 0.5 * cos_t * p_l
                elif ell > m + 1:
                    a = ((4 * ell * ell - 1) / (ell * ell - m * m)) ** 0.5
                    b = (((ell - 1) ** 2 - m * m) / (4 * (ell - 1) ** 2 - 1)) ** 0.5
                    p_prev, p_l = p_l, a * (cos_t * p_l - b * p_prev)
                if ell in row:
                    A = torch.zeros(r.size + 1, dtype=torch.complex128).index_add_(
                        0, index, p_l * e
                    )[: r.size]
                    H[row[ell]] += (1.0 if m == 0 else 2.0) * torch.real(
                        A.conj()[:, None] * A[None, :]
                    )
    return H


def _cluster_power_beam(clusters, r, n_values):
    """conj(P_n(r)) P_n(r') of the projected positions summed over clusters, P_n per unit area, (Nn, Nr, Nr)."""
    r_t = torch.as_tensor(r, dtype=torch.float64)
    dr = float(r_t[1] - r_t[0])
    ring = 2 * np.pi * torch.clamp(r_t, min=0.5 * dr) * dr
    ring = torch.cat([ring, ring[-1:]])
    n_t = torch.as_tensor(n_values, dtype=torch.float64)
    H = torch.zeros((len(n_values), r.size, r.size), dtype=torch.complex128)
    for pos in clusters:
        xy = torch.as_tensor(np.asarray(pos)[:, :2], dtype=torch.float64)
        d = (xy[None, :, :] - xy[:, None, :]).reshape(-1, 2)
        dist = torch.hypot(d[:, 0], d[:, 1])
        keep = (dist > 0) & (dist >= float(r_t[0])) & (dist < float(r_t[-1]))
        d, dist = d[keep], dist[keep]
        index, weight = _bin_vectors(dist, r)
        phi = torch.cat([torch.atan2(d[:, 0], d[:, 1])] * 2)
        E = (weight / ring[index])[:, None] * torch.exp(
            -1j * phi[:, None] * n_t[None, :]
        )  # (Nvec, Nn)
        A = torch.zeros((r.size + 1, len(n_values)), dtype=torch.complex128).index_add_(
            0, index, E
        )[: r.size]
        H += torch.einsum("in,jn->nij", A.conj(), A)
    return H


def _per_panel(value, n, nested=False):
    """One entry per panel: a list with one item per panel is used as is, anything
    else is repeated. With nested=True only a list of lists (or Nones) counts as
    per panel, so a flat list of marks is shared."""
    if isinstance(value, (list, tuple)) and len(value) == n:
        if not nested or all(v is None or isinstance(v, (list, tuple, np.ndarray)) for v in value):
            return list(value)
    return [value] * n
