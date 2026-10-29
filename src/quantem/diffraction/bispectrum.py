"""
Angular bispectrum (triple correlation) of polar diffraction data.

The pair angular correlation used by the PADF keeps only |I_hat(q, n)|^2 -
the angular power spectrum - discarding the relative phases between angular
harmonics. The bispectrum

    B_{q1 q2 q3}(n1, n2) = < I_hat_{q1}(n1) I_hat_{q2}(n2)
                             conj( I_hat_{q3}(n1 + n2) ) >_patterns

is the lowest-order rotation-invariant statistic that retains those phases
(the harmonic closure n3 = n1 + n2 cancels the e^{i n alpha} rotation
phases). Its 2D inverse Fourier transform is the angular triple correlation

    C3(delta1, delta2) = < mean_phi I(phi) I(phi + delta1) I(phi + delta2) >.

Working in the harmonic domain with a cutoff n_max keeps the object small:
(2 n_max + 1)^2 complex values per ring triple, instead of the
(N_phi)^2-per-triple maps of a direct triple loop.

Note: for single flat-Ewald diffraction patterns, Friedel symmetry makes
I(q, phi) pi-periodic, so odd harmonics vanish and C3 is 180-deg periodic
in each angle; only the joint reflection (d1, d2) -> (pi - d1, pi - d2)
remains degenerate - the relative fold between the two angles is broken.

Class: AngularBispectrum
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


class AngularBispectrum:
    """
    Angular bispectrum / triple correlation of polar diffraction data.

    Parameters
    ----------
    polar : Polar4dstem or torch.Tensor or ndarray
        Polar data with the angular axis second-to-last and the radial axis
        last: (..., N_phi, N_q). Leading dimensions (scan positions,
        pattern index) are flattened into an ensemble of patterns.
    n_max : int
        Maximum angular harmonic retained. Governs angular resolution
        (~180 / n_max degrees) and cost; the bispectrum is
        (2 n_max + 1)^2 per ring triple.
    subtract_mean : bool
        Subtract each ring's angular mean per pattern (removes the n = 0
        term, which otherwise dominates C3 with structureless offsets).
    batch_size : int
        Patterns per chunk when accumulating ensemble averages.
    """

    def __init__(self, polar, n_max: int = 60, subtract_mean: bool = True,
                 batch_size: int = 4096):
        tensor = polar.tensor if hasattr(polar, "tensor") else torch.as_tensor(np.asarray(polar))
        tensor = tensor.to(torch.float64)
        n_phi, n_q = tensor.shape[-2], tensor.shape[-1]
        if n_max >= n_phi // 2:
            n_max = n_phi // 2 - 1
        self.n_max = int(n_max)
        self.n_phi = n_phi
        self.n_q = n_q
        self.batch_size = int(batch_size)

        data = tensor.reshape(-1, n_phi, n_q)
        if subtract_mean:
            data = data - data.mean(dim=1, keepdim=True)
        self.n_patterns = data.shape[0]

        # angular FFT per ring; normalized so coefficients are per-radian
        # Fourier amplitudes independent of the phi sampling
        F = torch.fft.rfft(data, dim=1) / n_phi  # (Npat, Nphi//2+1, Nq)
        F = F[:, : self.n_max + 1, :]
        # signed-harmonic spectrum from Hermitian symmetry of real data:
        # index i <-> n = i - n_max, so F_signed[..., 0] is n = -n_max
        neg = torch.conj(torch.flip(F[:, 1:, :], dims=[1]))
        self.F = torch.cat([neg, F], dim=1).permute(0, 2, 1).contiguous()
        # self.F: (Npat, Nq, 2*n_max+1), harmonic n at index n + n_max
        self.n_values = np.arange(-self.n_max, self.n_max + 1)

    def bispectrum(self, q1: int, q2: int | None = None, q3: int | None = None):
        """
        Ensemble-averaged bispectrum B(n1, n2) for the ring triple
        (q1, q2, q3) (radial bin indices; q2, q3 default to q1).

        Returns
        -------
        B : ndarray, complex, shape (2 n_max + 1, 2 n_max + 1)
            B[i, j] with n1 = i - n_max, n2 = j - n_max. Entries with
            |n1 + n2| > n_max are zero (outside the harmonic cutoff).
        """
        q2 = q1 if q2 is None else q2
        q3 = q1 if q3 is None else q3
        m = self.n_max
        n = self.n_values
        idx = n[:, None] + n[None, :]              # n1 + n2
        valid = np.abs(idx) <= m
        idx_c = torch.from_numpy(np.where(valid, idx + m, 0))
        valid_t = torch.from_numpy(valid)

        B = torch.zeros((2 * m + 1, 2 * m + 1), dtype=torch.complex128)
        for start in range(0, self.n_patterns, self.batch_size):
            F1 = self.F[start:start + self.batch_size, q1]  # (b, 2m+1)
            F2 = self.F[start:start + self.batch_size, q2]
            F3 = self.F[start:start + self.batch_size, q3]
            F3g = F3[:, idx_c]                              # (b, 2m+1, 2m+1)
            B += (F1[:, :, None] * F2[:, None, :] * torch.conj(F3g)).sum(dim=0)
        B = B / self.n_patterns
        B[~valid_t] = 0
        return B.numpy()

    def triple_correlation(self, q1: int, q2: int | None = None,
                           q3: int | None = None, n_delta: int = 181):
        """
        Angular triple correlation C3(delta1, delta2) for the ring triple,
        evaluated on an n_delta x n_delta grid over [0, 360) degrees.

        Returns
        -------
        delta_deg : ndarray (n_delta,)
        C3 : ndarray (n_delta, n_delta), real
        """
        B = self.bispectrum(q1, q2, q3)
        delta = 2 * np.pi * np.arange(n_delta) / n_delta
        E = np.exp(1j * np.outer(self.n_values, delta))  # (2m+1, n_delta)
        C3 = np.real(E.T @ B @ E)
        return np.degrees(delta), C3

    def plot_triple_correlation(self, q1: int, q2: int | None = None,
                                q3: int | None = None, n_delta: int = 181,
                                delta_max: float = 180.0,
                                title: str | None = None,
                                figsize: tuple[float, float] = (4.6, 4.0),
                                returnfig: bool = False):
        """
        Plot C3(delta1, delta2) as a zero-centered map. For diffraction
        data C3 is 180-deg periodic in each angle, so the default view is
        [0, 180]^2.
        """
        delta_deg, C3 = self.triple_correlation(q1, q2, q3, n_delta=2 * n_delta)
        sel = delta_deg <= delta_max + 1e-9
        d = delta_deg[sel]
        M = C3[np.ix_(sel, sel)]
        vmax = np.quantile(np.abs(M), 0.999)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        im = ax.pcolormesh(d, d, M.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="auto")
        ax.set_xlabel("$\\Delta_1$ (deg)")
        ax.set_ylabel("$\\Delta_2$ (deg)")
        ax.set_xticks(np.arange(0, delta_max + 1, 45))
        ax.set_yticks(np.arange(0, delta_max + 1, 45))
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="3-point correlation")
        ax.set_title(title if title is not None else f"C3, ring {q1}",
                     fontsize=10)
        if returnfig:
            return fig, ax
        plt.show()
