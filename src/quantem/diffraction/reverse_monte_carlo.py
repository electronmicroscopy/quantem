"""Reverse Monte Carlo fitting of diffuse electron scattering from several zone axes.

One periodic supercell of a disordered crystal is fitted to every pattern at
once. Its amplitude ``F(q) = sum_j f_s(j)(q) exp(-2 pi i q.(r_j + u_j))``
lives on the FFT grid of the (displaced) atom positions, with the Bragg
nodes of the average lattice removed; the diffuse intensity of each pattern
is ``|F|^2`` read where the pattern's Ewald sphere (with its fitted tilt)
cuts the grid, averaged over the cubic rotations so the model is as
symmetric as the (statistically cubic) foil. Each Monte Carlo move changes
``F`` by a few phase factors, so every move is scored exactly without
recomputing the supercell. The move types are species swaps, single-atom
displacements on the fine position grid and, optionally, omega embryos (see
`ReverseMonteCarlo.run`).

Bragg peaks with their tails and the direct-beam bloom are masked by sigmoid
weights; a smooth background (constant, direct-beam Lorentzian, a wide
Gaussian about the zone-axis pole, Einstein thermal diffuse and powder rings)
and the diffuse envelope are solved in closed form.

Electrons barely separate species of neighbouring atomic number (Nb and Zr
differ by a few percent in scattering factor), so their relative arrangement
is set by the moves' randomness unless something else tells them apart.
``set_size_effect`` / ``fit_size_effect`` add the linear size effect: each
species pushes its neighbours by its misfit through harmonic springs, and the
resulting displacement field (Huang and size-effect scattering, odd about
every Bragg peak) depends on which species sits where.

Workflow: ``from_images`` -> ``set_crystal`` -> ``fit_geometry`` ->
``fit_thickness`` -> ``set_mask`` -> ``build_supercell`` -> ``fit_background``
-> ``run`` -> analysis and plots. ``set_envelope``, ``set_size_effect`` and
``fit_size_effect`` are optional. See `ReverseMonteCarlo` for details.

Limitations: cubic unit cells only; every mixed-occupancy site must share one
composition; static displacements (and ``displacement_correlations``) are
implemented for BCC site lattices only.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations, product

import numpy as np
import torch
from scipy import ndimage, optimize, sparse
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.crystal import Crystal, electron_scattering_factor


def electron_wavelength(energy_ev: float) -> float:
    """Relativistic electron wavelength in Angstroms."""
    return 12.2642598 / np.sqrt(energy_ev * (1.0 + 0.97847573e-6 * energy_ev))


def cubic_rotations() -> np.ndarray:
    """The 24 proper rotations of the cube as signed permutation matrices (24, 3, 3)."""
    ops = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            m = np.zeros((3, 3), dtype=int)
            m[range(3), perm] = signs
            if round(np.linalg.det(m)) == 1:
                ops.append(m)
    return np.stack(ops)


def _zone_frame(zone_axis) -> np.ndarray:
    """Orthonormal crystal-frame basis (e1, e2, z) with z along the zone axis."""
    z = np.asarray(zone_axis, dtype=float)
    z /= np.linalg.norm(z)
    trial = np.eye(3)[np.argmin(np.abs(z))]
    e1 = trial - (trial @ z) * z
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(z, e1), z])


def _rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _default_device() -> str:
    """cuda when present, otherwise cpu (mps only on request: long runs heat a laptop)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class ReverseMonteCarlo(AutoSerialize):
    """Reverse Monte Carlo fit of one supercell to diffraction patterns along several zone axes.

    Notes
    -----
    Workflow:

    1. ``from_images``: patterns, their zone axes, beam energy and pixel size.
    2. ``set_crystal``: average structure with the mixed-occupancy sites.
    3. ``fit_geometry``: center, detector distortion and tilt of each pattern.
    4. ``fit_thickness``: Bloch-wave thickness and tilt from the Bragg
       intensities (required for ``envelope="bloch"``; recommended otherwise
       since it refines the tilts).
    5. ``set_mask``: diffuse weights, binned data and Bragg intensities.
    6. ``build_supercell``: random supercell at the crystal's composition.
    7. ``fit_background``: diffuse scale and smooth background per pattern.
    8. ``run``: Monte Carlo sweeps.
    9. Analysis (``r_factors``, ``warren_cowley``, ``displacement_correlations``,
       ``diffuse_section``, ...) and ``plot_*`` methods.

    Optional steps after ``build_supercell``: ``set_envelope`` switches the
    diffuse envelope; ``set_size_effect`` / ``fit_size_effect`` add the linear
    size effect.

    Moves: ``run`` mixes three move types on the current supercell:

    - species swaps of two unlike atoms (composition conserved);
    - single-atom random displacements by one step of the fine position grid
      along each axis (``random_fraction``, default 0.5, when the supercell
      was built with ``displacements=True``);
    - omega embryos, three consecutive atoms of a <111> row with the last two
      collapsed toward each other (``omega_fraction``, default 0).

    Limitations: cubic unit cells only. Every mixed-occupancy site must share one
    composition. Static displacements and ``displacement_correlations`` are
    implemented for BCC site lattices only.

    Saving and loading: save without the arrays that can be rebuilt, then load with
    `quantem.core.io.load`, which calls ``_post_load`` to rebuild them::

        rmc.save(path, mode="o", skip=rmc.DERIVED_ATTRIBUTES)
        rmc = quantem.core.io.load(path)
    """

    _token = object()

    def __init__(self, images, zone_axes, sampling, energy, names, bin_factor, _token=None):
        if _token is not self._token:
            raise RuntimeError("Use ReverseMonteCarlo.from_images().")
        self.images = [np.asarray(im, dtype=np.float32) for im in images]
        self.zone_axes = [tuple(int(v) for v in z) for z in zone_axes]
        self.sampling = float(sampling)
        self.energy = float(energy)
        self.wavelength = electron_wavelength(self.energy)
        self.names = list(names)
        self.bin_factor = int(bin_factor)
        self.crystal: Crystal | None = None
        self.geometry: dict | None = None
        self.mask: dict | None = None
        self.loss_history: list[float] = []

    @classmethod
    def from_images(
        cls,
        images: Sequence,
        zone_axes: Sequence[Sequence[int]],
        energy: float = 200e3,
        bin_factor: int = 8,
        sampling: float | None = None,
        names: Sequence[str] | None = None,
    ) -> "ReverseMonteCarlo":
        """Patterns (Dataset2d or arrays) and the zone axis of each.

        Parameters
        ----------
        images : sequence of Dataset2d or ndarray
            One diffraction pattern per zone axis.
        zone_axes : sequence of (u, v, w)
            Nominal zone axis of each pattern, in the crystal's lattice
            indices; the tilt off it is fitted.
        energy : float
            Beam energy in eV.
        bin_factor : int
            Detector binning for the diffuse fit (geometry uses full resolution).
        sampling : float, optional
            Detector pixel size in 1/Angstrom. Read from the first Dataset2d
            (1/nm is converted) when not given.
        names : sequence of str, optional
            Panel titles; default the zone axes.
        """
        arrays = []
        for im in images:
            if hasattr(im, "array"):
                if sampling is None:
                    s = float(np.asarray(im.sampling)[0])
                    units = str(im.units[0]).lower()
                    sampling = s * 0.1 if "nm" in units else s
                arrays.append(np.asarray(im.array))
            else:
                arrays.append(np.asarray(im))
        if sampling is None:
            raise ValueError("sampling (1/Angstrom per pixel) is required for plain arrays.")
        if len(arrays) != len(zone_axes):
            raise ValueError("one zone axis per image")
        if names is None:
            names = ["[" + "".join(str(v) for v in z) + "]" for z in zone_axes]
        return cls(arrays, zone_axes, sampling, energy, names, bin_factor, _token=cls._token)

    def _binned(self, a: np.ndarray, reduce: str = "mean") -> np.ndarray:
        b = self.bin_factor
        ny, nx = (a.shape[0] // b) * b, (a.shape[1] // b) * b
        out = a[:ny, :nx].reshape(ny // b, b, nx // b, b).sum(axis=(1, 3))
        return out / b**2 if reduce == "mean" else out

    # rebuilt from the saved state by _post_load: pass as ``skip`` to ``save`` to keep files small
    DERIVED_ATTRIBUTES = (
        "_W_all", "_Wc", "_Wv", "_sym_index", "_needed", "_used", "_h_needed", "_cos", "_sin",
        "_Fr", "_Fi", "_fs", "_keep", "_u", "_u_all", "_y", "_w", "_y_all", "_w_all", "_img",
        "_fit", "_r", "_bg", "_scale", "_basis_all", "_pix_image", "_env_cols", "_site_lookup",
    )  # fmt: skip

    def _post_load(self) -> None:
        """AutoSerialize hook: rebuild the derived arrays skipped at save time."""
        if getattr(self, "site_x", None) is None or hasattr(self, "_W_all"):
            return
        self.device = torch.device(_default_device())
        nc = self.cells * self.grid_divisor
        self._site_lookup = np.full((nc,) * 3, -1, dtype=np.int64)
        xc = self.site_x // self.refine
        self._site_lookup[xc[:, 0], xc[:, 1], xc[:, 2]] = np.arange(len(self.site_x))
        env_p = getattr(self, "_env_p", None)
        self._setup_forward()
        if self.envelope == "fitted" and env_p is not None:
            self._env_p = list(env_p)
            self._u_all = np.concatenate(
                [self._env_cols[i] @ np.asarray(self._env_p[i]) for i in range(len(self.images))]
            )
            self._u = torch.as_tensor(
                self._u_all[self._fit], dtype=torch.float32, device=self.device
            )
            self._update_residual()

    # ------------------------------------------------------------------ crystal

    def set_crystal(
        self,
        crystal: Crystal | None = None,
        cif_file: str | None = None,
        merge: dict[str, str] | None = None,
    ) -> "ReverseMonteCarlo":
        """Crystal whose mixed-occupancy sites are fitted.

        Parameters
        ----------
        crystal, cif_file : Crystal or path
            The average structure, with fractional occupancies on shared sites.
        merge : dict, optional
            Species to relabel before fitting, e.g. ``{"Zr": "Nb"}`` folds Zr
            into Nb (their electron scattering factors differ by a few percent,
            so the patterns barely tell them apart).
        """
        from ase.data import atomic_numbers, chemical_symbols

        if crystal is None:
            if cif_file is None:
                raise ValueError("give crystal or cif_file")
            crystal = Crystal.from_cif(cif_file, verbose=False)
        self.crystal = crystal
        cell = crystal.lat_real.numpy()
        a = float(np.linalg.norm(cell[0]))
        if not np.allclose(cell, a * np.eye(3), atol=1e-3 * a):
            raise NotImplementedError("Only cubic cells are supported so far.")
        merge = merge or {}
        frac = np.mod(crystal.positions_frac.numpy(), 1.0)
        symbols = [chemical_symbols[int(z)] for z in crystal.numbers]
        symbols = [merge.get(s, s) for s in symbols]
        occ = crystal.occupancy.numpy()

        # group species by site
        sites: list[tuple[np.ndarray, dict[str, float]]] = []
        for f, s, o in zip(frac, symbols, occ):
            for site in sites:
                if np.allclose(site[0], f, atol=1e-4):
                    site[1][s] = site[1].get(s, 0.0) + float(o)
                    break
            else:
                sites.append((f, {s: float(o)}))
        mixed = [(f, comp) for f, comp in sites if len(comp) > 1]
        if not mixed:
            raise ValueError("The crystal has no mixed-occupancy site to fit.")
        species = sorted({s for _, comp in mixed for s in comp})
        comps = {tuple(sorted(comp.items())) for _, comp in mixed}
        if len(comps) != 1:
            raise NotImplementedError("All mixed sites must share one composition.")
        comp = dict(next(iter(comps)))
        total = sum(comp.values())

        # smallest grid divisor that puts every mixed site on an integer grid
        fr = np.stack([f for f, _ in mixed])
        for d in range(1, 13):
            if np.allclose(fr * d, np.round(fr * d), atol=1e-4):
                break
        else:
            raise ValueError("Mixed sites are not on a rational grid with denominator <= 12.")

        self._a_crystal = a
        self.lattice_parameter = a
        self.species = species
        self.numbers = [atomic_numbers[s] for s in species]
        self.concentrations = np.array([comp[s] / total for s in species])
        self.site_grid = np.round(fr * d).astype(int)
        self.grid_divisor = d
        print(
            f"{len(mixed)} mixed site(s) per cell, "
            + ", ".join(f"{s} {comp[s] / total:.3f}" for s in species)
            + f", a = {a:.4f} A"
        )
        return self

    def _zone_reflections(self, zone_axis, k_max: float):
        """Allowed reflections in a zone: hkl (n, 3), zone-frame coords at the CIF a (n, 2), |F|^2."""
        crystal = self.crystal
        crystal.calculate_structure_factors(k_max)
        hkl = crystal.hkl.numpy()
        inten = crystal.struct_factors_int.numpy()
        keep = (hkl @ np.asarray(zone_axis) == 0) & (inten > 1e-4 * inten.max())
        hkl, inten = hkl[keep], inten[keep]
        frame = _zone_frame(zone_axis)
        g = hkl / self._a_crystal
        return hkl, g @ frame[:2].T, inten

    # ----------------------------------------------------------------- geometry

    @staticmethod
    def _find_peaks(im, n_peaks: int = 150):
        smooth = ndimage.gaussian_filter(im, 2.0)
        prom = smooth - ndimage.gaussian_filter(im, 25.0)
        local = (prom == ndimage.maximum_filter(prom, 15)) & (prom > 0)
        r, c = np.nonzero(local)
        order = np.argsort(prom[r, c])[::-1][:n_peaks]
        r, c = r[order], c[order]
        pts = []
        for ri, ci in zip(r, c):
            r0, r1 = max(ri - 4, 0), min(ri + 5, im.shape[0])
            c0, c1 = max(ci - 4, 0), min(ci + 5, im.shape[1])
            w = np.clip(prom[r0:r1, c0:c1] - 0.3 * prom[ri, ci], 0, None)
            rr, cc = np.mgrid[r0:r1, c0:c1]
            pts.append([(w * rr).sum() / w.sum(), (w * cc).sum() / w.sum()])
        return np.asarray(pts), prom[r, c], prom

    @staticmethod
    def _halo_center(im) -> np.ndarray:
        """Center of the broad inelastic halo, which sits on the direct beam."""
        small = ndimage.median_filter(im[::4, ::4].astype(np.float64), size=9)
        b = ndimage.gaussian_filter(small, 10)
        return np.asarray(np.unravel_index(np.argmax(b), b.shape), dtype=float) * 4 + 1.5

    def fit_geometry(
        self,
        scale_range: tuple[float, float] = (0.85, 1.2),
        k_max: float = 1.6,
        centers: Sequence | None = None,
        fit_tilt: bool = True,
        verbose: bool = True,
    ) -> "ReverseMonteCarlo":
        """Index every pattern, fit its detector distortion and its tilt off the zone axis.

        The direct beam is the detected peak nearest the center of the broad
        inelastic halo (a tilted pattern can have diffracted beams brighter
        than the direct beam), or nearest ``centers[i]`` (row, col). Each
        pattern then gets its own center and 2x2 detector matrix (rotation,
        scale, ellipticity), fitted to the matched peaks. The lattice
        parameter is the mean over patterns at the nominal pixel size.

        The tilt (beam direction off the zone axis, small-angle vector in the
        zone frame) is fitted to the Bragg intensities: each reflection's
        excitation error is ``s = -(|g|^2 / 2K + tilt . g)`` and its intensity
        ``|F|^2 exp(-s^2 / 2 sigma^2)``. Zero tilt puts the Laue circle on the
        direct beam.

        Parameters
        ----------
        scale_range : (float, float), optional
            Range of the pixel-size scale factor searched in the coarse
            indexing step, relative to ``sampling``. Default (0.85, 1.2).
        k_max : float, optional
            Largest scattering vector (1/A) of the reflections used for
            indexing and tilt fitting. Default 1.6.
        centers : sequence of (row, col) or None, optional
            Approximate direct-beam position of each pattern in pixels; None
            entries (or ``centers=None``) use the halo center.
        fit_tilt : bool, optional
            Fit the tilt of each pattern. If False, the tilts are zero.
        verbose : bool, optional
            Print the fitted geometry of each pattern.

        Returns
        -------
        ReverseMonteCarlo
            self. The results are stored in ``self.geometry``, a dict of
            per-pattern lists: "centers" (row, col) px, "matrices" (2x2, px
            per 1/A at the CIF lattice parameter), "tilts" (zone-frame vector,
            radians), "a" (lattice parameter, A), "rms_px", "n_matched",
            "peaks", "bragg_hkl", "bragg_g", "bragg_px", "bragg_intensity" and
            "excitation_width" (1/A). "excitation_width" is the width of the
            Gaussian excitation-error profile fitted with the tilt; it is a
            diagnostic only and is not used later. ``self.lattice_parameter``
            is set to the mean of "a".
        """
        if self.crystal is None:
            raise RuntimeError("set_crystal first")
        pix = self.sampling
        k_wave = 1.0 / self.wavelength
        geo = dict(centers=[], matrices=[], tilts=[], a=[], rms_px=[], n_matched=[], peaks=[])
        geo.update(bragg_hkl=[], bragg_g=[], bragg_intensity=[], bragg_px=[], excitation_width=[])
        geo["_inten_kin"] = []  # kinematic |F|^2 of each pattern's reflections, for _fit_tilt
        for i, (im, zone) in enumerate(zip(self.images, self.zone_axes)):
            pts, heights, prom = self._find_peaks(im)
            guess = (
                np.asarray(centers[i], dtype=float)
                if centers is not None and centers[i] is not None
                else self._halo_center(im)
            )
            center = pts[np.argmin(np.linalg.norm(pts - guess, axis=1))]
            hkl, g2, inten = self._zone_reflections(zone, k_max)
            nz = np.linalg.norm(hkl, axis=1) > 0
            g2_nz, inten_nz = g2[nz], inten[nz]

            # coarse search: in-plane rotation and scale
            score_img = ndimage.gaussian_filter(np.clip(prom, 0, None), 3.0)
            wts = np.sqrt(inten_nz)
            best = (-np.inf, 0.0, 1.0)
            for scale in np.arange(scale_range[0], scale_range[1] + 1e-9, 0.004):
                for th in np.deg2rad(np.arange(0.0, 360.0, 0.5)):
                    p = center + (g2_nz @ _rot2(th).T) / (pix * scale)
                    ok = (
                        (p[:, 0] >= 0)
                        & (p[:, 0] < im.shape[0] - 1)
                        & (p[:, 1] >= 0)
                        & (p[:, 1] < im.shape[1] - 1)
                    )
                    if ok.sum() < 4:
                        continue
                    pi = np.round(p[ok]).astype(int)
                    s = (wts[ok] * score_img[pi[:, 0], pi[:, 1]]).sum() / wts[ok].sum()
                    if s > best[0]:
                        best = (s, th, scale)
            A = _rot2(best[1]) / (pix * best[2])
            c = center.copy()

            # refine center + 2x2 matrix on matched peaks, tightening the match
            for tol in (12.0, 8.0, 5.0, 5.0):
                p = c + g2_nz @ A.T
                dist, j = cKDTree(pts).query(p)
                ok = dist < tol
                obs = pts[j[ok]]
                X = np.column_stack([np.ones(ok.sum()), g2_nz[ok]])
                coef, *_ = np.linalg.lstsq(X, obs, rcond=None)
                c, A = coef[0], coef[1:].T
            p = c + g2_nz @ A.T
            dist, j = cKDTree(pts).query(p)
            ok = dist < 5.0
            rms = float(np.sqrt(np.mean(dist[ok] ** 2)))
            a_i = self._a_crystal / (pix * np.sqrt(abs(np.linalg.det(A))))
            sv = np.linalg.svd(A, compute_uv=False)

            # Bragg intensities at the fitted positions
            p_all = c + g2 @ A.T
            r_core = 0.03 / pix
            inten_meas = _integrate_spots(im, p_all, r_core)

            geo["centers"].append(c)
            geo["matrices"].append(A)
            geo["a"].append(float(a_i))
            geo["rms_px"].append(rms)
            geo["n_matched"].append(int(ok.sum()))
            geo["peaks"].append(pts)
            geo["bragg_hkl"].append(hkl)
            geo["bragg_g"].append(g2)  # zone frame, at the CIF lattice parameter
            geo["bragg_px"].append(p_all)
            geo["bragg_intensity"].append(inten_meas)
            geo["_inten_kin"].append(inten)
            if verbose:
                print(
                    f"{self.names[i]}: center ({c[0]:.1f}, {c[1]:.1f}), a = {a_i:.4f} A, "
                    f"anisotropy {100 * (sv[0] / sv[1] - 1):.2f}%, "
                    f"{int(ok.sum())} peaks, rms {rms:.2f} px"
                )

        self.lattice_parameter = float(np.mean(geo["a"]))
        self.geometry = geo
        for i in range(len(self.images)):
            tilt, width = (np.zeros(2), np.nan)
            if fit_tilt:
                tilt, width = self._fit_tilt(i, k_wave)
            geo["tilts"].append(tilt)
            geo["excitation_width"].append(width)
            if verbose and fit_tilt:
                ang = np.rad2deg(np.linalg.norm(tilt))
                print(f"{self.names[i]}: tilt {ang:.2f} deg off the zone axis")
        if verbose:
            print(f"lattice parameter {self.lattice_parameter:.4f} A at {pix:.6f} 1/A per pixel")
        return self

    def _fit_tilt(self, i: int, k_wave: float, max_tilt_deg: float = 3.5, prior_deg: float = 2.0):
        """Tilt vector (zone frame, radians) from the Bragg intensities of pattern i, with a
        Gaussian prior of ``prior_deg`` so patterns whose intensities barely constrain it stay
        near the zone axis."""
        geo = self.geometry
        hkl = geo["bragg_hkl"][i]
        nz = np.linalg.norm(hkl, axis=1) > 0
        g = geo["bragg_g"][i][nz] * self._a_crystal / self.lattice_parameter
        p_px = geo["bragg_px"][i][nz]
        ny, nx = self.images[i].shape
        inside = (
            (p_px[:, 0] > 20) & (p_px[:, 0] < ny - 20) & (p_px[:, 1] > 20) & (p_px[:, 1] < nx - 20)
        )
        g = g[inside]
        meas = geo["bragg_intensity"][i][nz][inside]
        kin = geo["_inten_kin"][i][nz][inside]
        ok = np.isfinite(meas)
        g, meas, kin = g[ok], np.clip(meas[ok], 0, None), kin[ok]
        y = np.sqrt(meas / meas.max())
        g2 = (g**2).sum(1) / (2 * k_wave)

        def model(x):
            tilt, log_w, log_a = x[:2], x[2], x[3]
            s = -(g2 + g @ tilt)
            return (
                np.exp(log_a) * np.sqrt(kin / kin.max()) * np.exp(-0.25 * (s / np.exp(log_w)) ** 2)
            )

        best = None
        for tx in np.linspace(-0.06, 0.06, 25):
            for ty in np.linspace(-0.06, 0.06, 25):
                for lw in (np.log(0.01), np.log(0.03)):
                    x = np.array([tx, ty, lw, 0.0])
                    r = ((model(x) - y) ** 2).sum() + ((x[:2] / np.deg2rad(prior_deg)) ** 2).sum()
                    if best is None or r < best[0]:
                        best = (r, x)
        lim = np.deg2rad(max_tilt_deg)
        lo = np.array([-lim, -lim, np.log(0.003), -5.0])
        hi = np.array([lim, lim, np.log(0.05), 5.0])
        prior = np.deg2rad(prior_deg)

        def resid(x):
            return np.concatenate([model(x) - y, x[:2] / prior])

        sol = optimize.least_squares(
            resid, np.clip(best[1], lo + 1e-9, hi - 1e-9), bounds=(lo, hi)
        )
        return sol.x[:2], float(np.exp(sol.x[2]))

    def _orientation_quat(self, i: int, tilt: np.ndarray) -> torch.Tensor:
        """Quaternion rotating crystal vectors into the lab frame of pattern i at a tilt.

        The zone axis is turned toward ``(tilt_x, tilt_y, 1)``, which puts the Bloch excitation
        errors on the same Laue circle as the diffuse model's Ewald sphere."""
        from quantem.diffraction.rotations import quat_from_matrix

        n = np.array([tilt[0], tilt[1], 1.0])
        n /= np.linalg.norm(n)
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z, n)
        s_ang, c_ang = np.linalg.norm(axis), n[2]
        if s_ang < 1e-12:
            rot = np.eye(3)
        else:
            k = axis / s_ang
            kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            rot = np.eye(3) + s_ang * kx + (1 - c_ang) * kx @ kx
        u = rot @ _zone_frame(self.zone_axes[i])
        return quat_from_matrix(torch.as_tensor(u, dtype=torch.float64))

    def _measured_bragg(self, i: int):
        """Measured integrated intensities of pattern i: hkl (n, 3) with the direct beam first."""
        hkl = np.asarray(self.geometry["bragg_hkl"][i])
        meas = np.asarray(self.geometry["bragg_intensity"][i], dtype=float)
        nz = np.linalg.norm(hkl, axis=1) > 0
        hkl, meas = hkl[nz], meas[nz]
        i000 = _integrate_spots(
            self.images[i], self.geometry["centers"][i][None], 0.03 / self.sampling
        )[0]
        hkl = np.vstack([[0, 0, 0], hkl])
        meas = np.concatenate([[i000], meas])
        ok = np.isfinite(meas)
        return hkl[ok].astype(int), np.clip(meas[ok], 0, None)

    def fit_thickness(
        self,
        thickness: tuple[float, float] = (20.0, 1000.0),
        step: float = 10.0,
        tilt_range_deg: float = 0.6,
        tilt_step_deg: float = 0.1,
        k_max: float = 1.6,
        depth_samples: int = 24,
        verbose: bool = True,
    ) -> dict:
        """Bloch-wave thickness and tilt of every pattern from its Bragg intensities.

        For each pattern, a grid of tilts about the current one and of thicknesses is scored by
        comparing the Bloch exit intensities of the zone reflections with the measured
        integrated intensities (each set normalized to unit sum, compared as square roots).
        Absorptive (Weickenmeier-Kohl) structure factors of the average crystal are used. The
        best thickness and tilt are stored, with each beam's intensity averaged over depth,
        ``(1/t) int_0^t |phi_g(z)|^2 dz``: the beams that generate diffuse scattering inside
        the foil, used by ``envelope="bloch"``.

        Parameters
        ----------
        thickness : (float, float), optional
            Thickness range searched, in A. Default (20, 1000).
        step : float, optional
            Thickness step, in A. Default 10.
        tilt_range_deg : float, optional
            Half width of the tilt search about the current tilt, along each
            zone-frame axis, in degrees. Default 0.6.
        tilt_step_deg : float, optional
            Tilt search step, in degrees. Default 0.1.
        k_max : float, optional
            Largest scattering vector (1/A) of the Bloch-wave beams. Default
            1.6. Stored in ``geometry["thickness_k_max"]`` for
            ``plot_thickness``.
        depth_samples : int, optional
            Depths at which the beam intensities are averaged. Default 24.
        verbose : bool, optional
            Print the result for each pattern.

        Returns
        -------
        dict
            ``{name: {"thickness": A, "tilt_deg": degrees off the zone axis}}``
            per pattern. ``self.geometry`` is updated in place: "tilts",
            "thickness", "bloch_g", "bloch_p", "bloch_score" and
            "thickness_k_max".
        """
        if self.geometry is None:
            raise RuntimeError("fit_geometry first")
        from quantem.diffraction import bloch

        crystal = self.crystal
        crystal.calculate_structure_factors(2 * k_max)
        crystal.calculate_dynamical_structure_factors(self.energy, k_max=2 * k_max)
        t_grid = np.arange(thickness[0], thickness[1] + 1e-9, step)
        geo = self.geometry
        geo.setdefault("thickness", [np.nan] * len(self.images))
        geo.setdefault("bloch_g", [None] * len(self.images))
        geo.setdefault("bloch_p", [None] * len(self.images))
        geo.setdefault("bloch_score", [None] * len(self.images))
        geo["thickness_k_max"] = float(k_max)
        results = {}
        for i in range(len(self.images)):
            hkl_m, meas = self._measured_bragg(i)
            a_meas = np.sqrt(meas / meas.sum())
            keys = {tuple(h): k for k, h in enumerate(hkl_m)}
            tilt0 = np.asarray(geo["tilts"][i], dtype=float)
            offs = np.deg2rad(np.arange(-tilt_range_deg, tilt_range_deg + 1e-9, tilt_step_deg))
            best = (np.inf, None, None, None)
            curves = {}
            for dx in offs:
                for dy in offs:
                    tilt = tilt0 + np.array([dx, dy])
                    out = bloch.dynamical_pattern(
                        crystal, self._orientation_quat(i, tilt), t_grid, self.energy, k_max=k_max
                    )
                    calc = np.zeros((len(t_grid), len(hkl_m)))
                    calc[:, keys[(0, 0, 0)]] = out["intensity_000"].numpy()
                    for col, h in enumerate(out["hkl"].numpy().astype(int)):
                        k = keys.get(tuple(h))
                        if k is not None:
                            calc[:, k] = out["intensity"][:, col].numpy()
                    a_calc = np.sqrt(calc / calc.sum(1, keepdims=True))
                    score = ((a_calc - a_meas[None]) ** 2).sum(1)
                    curves[(dx, dy)] = score
                    j = int(np.argmin(score))
                    if score[j] < best[0]:
                        best = (score[j], tilt, t_grid[j], (dx, dy))
            score, tilt, t_best, key = best
            # depth-averaged beam intensities at the best thickness and tilt
            z = (np.arange(depth_samples) + 0.5) / depth_samples * t_best
            out = bloch.dynamical_pattern(
                crystal, self._orientation_quat(i, tilt), z, self.energy, k_max=k_max
            )
            frame = _zone_frame(self.zone_axes[i])
            hkl_b = np.vstack([[0, 0, 0], out["hkl"].numpy()])
            g_b = (
                (hkl_b / self._a_crystal) @ frame[:2].T * self._a_crystal / self.lattice_parameter
            )
            p_b = np.concatenate(
                [[out["intensity_000"].mean().item()], out["intensity"].mean(0).numpy()]
            )
            geo["tilts"][i] = tilt
            geo["thickness"][i] = float(t_best)
            geo["bloch_g"][i] = g_b
            geo["bloch_p"][i] = p_b
            geo["bloch_score"][i] = dict(thickness=t_grid, score=curves[key], best=float(score))
            results[self.names[i]] = dict(
                thickness=float(t_best), tilt_deg=float(np.rad2deg(np.linalg.norm(tilt)))
            )
            if verbose:
                print(
                    f"{self.names[i]}: thickness {t_best / 10:.0f} nm, tilt "
                    f"{np.rad2deg(np.linalg.norm(tilt)):.2f} deg, misfit {score:.3f}"
                )
        return results

    def plot_thickness(self, **kwargs):
        """Bloch thickness fit per pattern: misfit against thickness (top) and measured against
        calculated Bragg intensities at the best fit, square-root scale (bottom). Uses the
        ``k_max`` given to ``fit_thickness``."""
        import matplotlib.pyplot as plt

        from quantem.diffraction import bloch

        n = len(self.images)
        fig, axs = plt.subplots(2, n, figsize=kwargs.pop("figsize", (4.2 * n, 7.5)))
        for i in range(n):
            sc = self.geometry["bloch_score"][i]
            ax = axs[0, i]
            ax.plot(sc["thickness"] / 10, sc["score"], "k-")
            t_best = self.geometry["thickness"][i]
            ax.axvline(t_best / 10, color="tab:red", lw=1)
            ax.set_xlabel("thickness (nm)")
            ax.set_ylabel("misfit")
            ax.set_title(f"{self.names[i]}: {t_best / 10:.0f} nm")
            hkl_m, meas = self._measured_bragg(i)
            out = bloch.dynamical_pattern(
                self.crystal,
                self._orientation_quat(i, self.geometry["tilts"][i]),
                [t_best],
                self.energy,
                k_max=self.geometry.get("thickness_k_max", 1.6),
            )
            lookup = {
                tuple(h): v
                for h, v in zip(out["hkl"].numpy().astype(int), out["intensity"][0].numpy())
            }
            lookup[(0, 0, 0)] = float(out["intensity_000"][0])
            calc = np.array([lookup.get(tuple(h), 0.0) for h in hkl_m])
            ax = axs[1, i]
            x, y = np.sqrt(calc / calc.sum()), np.sqrt(meas / meas.sum())
            ax.plot(x, y, "o", ms=4, color="tab:blue")
            lim = 1.05 * max(x.max(), y.max())
            ax.plot([0, lim], [0, lim], "k--", lw=0.8)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal")
            ax.set_xlabel("Bloch sqrt(I)")
            ax.set_ylabel("measured sqrt(I)")
        fig.tight_layout()
        return fig, axs

    def bragg_positions(self, i: int, k_max: float = 3.0) -> np.ndarray:
        """Detector positions (row, col) of every zone reflection, direct beam included."""
        _, g2, _ = self._zone_reflections(self.zone_axes[i], k_max)
        return self.geometry["centers"][i] + g2 @ self.geometry["matrices"][i].T

    def _q_zone(self, i: int, rows, cols) -> np.ndarray:
        """Pixel coordinates -> in-plane scattering vector (..., 2) in the zone frame, 1/A."""
        p = np.stack([rows, cols], axis=-1) - self.geometry["centers"][i]
        g = p @ np.linalg.inv(self.geometry["matrices"][i]).T
        return g * self._a_crystal / self.lattice_parameter

    def _q_crystal(self, i: int, q2: np.ndarray) -> np.ndarray:
        """In-plane zone-frame q (n, 2) -> crystal-frame q (n, 3) on the tilted Ewald sphere."""
        k_wave = 1.0 / self.wavelength
        tilt = self.geometry["tilts"][i]
        qz = -((q2**2).sum(1) / (2 * k_wave) + q2 @ tilt)
        return np.column_stack([q2, qz]) @ _zone_frame(self.zone_axes[i])

    # --------------------------------------------------------------------- mask

    def set_mask(
        self,
        bragg_radius: float = 0.12,
        softness: float = 0.01,
        q_max: float = 1.2,
        center_radius: float = 0.25,
        edge_px: int = 8,
    ) -> "ReverseMonteCarlo":
        """Diffuse-scattering weight, binned data and Bragg intensities.

        The weight is ``sigmoid((d - bragg_radius) / softness)``, with ``d``
        the distance (1/A) to the nearest reflection, the direct beam
        included: 0 on every Bragg peak, 1 between them. Pixels beyond
        ``q_max`` or within ``edge_px`` of the detector edge are dropped, and
        a second sigmoid removes the direct beam's bloom out to
        ``center_radius``. Each ``bin_factor`` square is reduced to its
        weighted mean.

        The mask has to cover the peaks' tails (detector point spread and
        near-peak scattering), which are not modelled: here they fall to a few
        percent of the local diffuse level by 0.12 1/A.

        Each reflection's integrated intensity weights the diffuse envelope.

        Parameters
        ----------
        bragg_radius : float, optional
            Distance (1/A) from each reflection at which the weight reaches
            0.5. Default 0.12.
        softness : float, optional
            Width (1/A) of the sigmoid edges. Default 0.01.
        q_max : float, optional
            Largest scattering vector (1/A) fitted. Default 1.2.
        center_radius : float, optional
            Radius (1/A) of the direct-beam bloom removed. Default 0.25.
        edge_px : int, optional
            Detector-edge border (unbinned pixels) given zero weight. 0 keeps
            the whole detector. Default 8.

        Returns
        -------
        ReverseMonteCarlo
            self. The results are stored in ``self.mask``: per-pattern lists
            "y" (weighted mean of each binned pixel, normalized), "w" (binned
            weight, 0 to 1), "k" (zone-frame q of each binned pixel, 1/A),
            "data" (binned pattern, normalized), "scale" (normalization),
            "bragg_k" and "bragg_intensity" (reflections fully on the
            detector), plus the parameters above.
        """
        if self.geometry is None:
            raise RuntimeError("fit_geometry first")
        b = self.bin_factor
        out = dict(
            bragg_radius=bragg_radius, softness=softness, q_max=q_max, center_radius=center_radius
        )
        out.update(y=[], w=[], k=[], data=[], scale=[], bragg_k=[], bragg_intensity=[])
        for i, im in enumerate(self.images):
            ny, nx = (im.shape[0] // b) * b, (im.shape[1] // b) * b
            rows, cols = np.mgrid[0:ny, 0:nx].astype(np.float64)
            k = self._q_zone(i, rows, cols)
            bragg = self.bragg_positions(i)
            g_q = self._q_zone(i, bragg[:, 0], bragg[:, 1])
            d, j = cKDTree(g_q).query(k.reshape(-1, 2))
            d = d.reshape(ny, nx)
            j = j.reshape(ny, nx)
            q = np.linalg.norm(k, axis=-1)
            w = (
                _sigmoid((d - bragg_radius) / softness)
                * _sigmoid((q - center_radius) / softness)
                * (q < q_max)
            )
            if edge_px > 0:
                w[:edge_px] = w[-edge_px:] = 0
                w[:, :edge_px] = w[:, -edge_px:] = 0
            y = im[:ny, :nx].astype(np.float64)

            # integrated Bragg intensities over the local ring median
            n_g = len(g_q)
            core = d < bragg_radius
            ring = (d >= bragg_radius) & (d < 1.6 * bragg_radius)
            ring_med = np.zeros(n_g)
            jr, yr = j[ring], y[ring]
            order = np.argsort(jr, kind="stable")
            jr, yr = jr[order], yr[order]
            starts = np.searchsorted(jr, np.arange(n_g))
            ends = np.searchsorted(jr, np.arange(n_g), side="right")
            for g in np.nonzero(ends > starts)[0]:
                ring_med[g] = np.median(yr[starts[g] : ends[g]])
            core_sig = np.where(core, np.clip(y - ring_med[j], 0, None), 0.0)
            inten = np.bincount(j[core], weights=core_sig[core], minlength=n_g)
            n_core = np.bincount(j[core], minlength=n_g)
            seen = n_core > 0.5 * n_core.max()  # whole spot on the detector

            wb = self._binned(w, "sum")
            yb = np.where(wb > 0, self._binned(w * y, "sum") / np.maximum(wb, 1e-12), 0.0)
            rb, cb = np.mgrid[0 : ny // b, 0 : nx // b].astype(np.float64) * b + (b - 1) / 2
            norm = (wb * yb).sum() / wb.sum()
            out["y"].append(yb / norm)
            out["w"].append(wb / b**2)
            out["k"].append(self._q_zone(i, rb, cb))
            out["data"].append(self._binned(y) / norm)
            out["scale"].append(norm)
            out["bragg_k"].append(g_q[seen])
            out["bragg_intensity"].append(inten[seen] / norm)
        self.mask = out
        return self

    # ---------------------------------------------------------------- supercell

    def build_supercell(
        self,
        cells: int = 16,
        seed: int | None = 0,
        displacements: bool = True,
        displacement_grid: int = 24,
        max_displacement: float = 0.3,
        omega_amplitudes: Sequence[int] = (1, 2),
        symmetrize: bool = True,
        debye_waller: float = 0.5,
        envelope: str = "measured",
        resolution: float = 0.75,
        shared_scale: bool = True,
        max_beams: int = 40,
        device: str | None = None,
    ) -> "ReverseMonteCarlo":
        """Random supercell of ``cells^3`` unit cells at the crystal's composition.

        Every species on the mixed sites is kept (use ``merge`` in
        ``set_crystal`` to fold any together). The supercell amplitude is
        ``F(q) = sum_j f_s(j)(q) exp(-2 pi i q.(r_j + u_j))`` with the
        Bragg nodes of the average lattice removed.

        Parameters
        ----------
        cells : int
            Unit cells along each cube edge. The diffuse model is sampled
            every ``1 / (cells a)`` in reciprocal space.
        seed : int or None
            Seed of the random generator (stored as ``self.rng``) used for the
            initial species arrangement and for every later Monte Carlo move.
            None seeds from fresh OS entropy. Default 0.
        displacements : bool
            Allow static displacements. Positions live on a grid
            ``a / displacement_grid`` fine, so every move is still scored
            exactly. Implemented for BCC site lattices only: the default
            True raises NotImplementedError for any other site lattice, so
            pass False there.
        displacement_grid : int
            Steps per lattice parameter of the displacement grid (a multiple
            of 24): 24 gives 0.15 A steps, 48 gives 0.076 A for a = 3.66 A.
        max_displacement : float
            Largest displacement component (A) reached by random moves.
        omega_amplitudes : sequence of int
            Allowed displacements in units of a/24 along each axis: 2 is the
            ideal omega collapse (a/12, 0.53 A along <111> for a = 3.66 A), 1
            a half collapse.
        symmetrize : bool
            Average every pattern over the 24 cubic rotations of the supercell.
        debye_waller : float
            Isotropic B (A^2) damping the diffuse intensity.
        envelope : {"measured", "fitted", "bloch", "kinematic"}
            "measured" redistributes the diffuse intensity over the Bragg
            beams of each pattern, ``sum_g P_g fbar^2(q - g) / fbar^2(q)``
            with P_g the integrated Bragg intensities (exact for occupational
            disorder, whose diffuse intensity is periodic in the reciprocal
            lattice; approximate for displacements). "fitted" starts there
            and re-solves the non-negative P_g of each pattern with its
            background: diffuse scattering is generated by the beams' depth
            averaged intensities, which dynamical diffraction makes differ
            from their exit intensities. "bloch" takes the depth-averaged
            Bloch-wave beam intensities at each pattern's fitted thickness and
            tilt (requires ``fit_thickness``); use it with ``shared_scale``
            so one diffuse scale covers all patterns. "kinematic" keeps only
            the direct beam.
        resolution : float
            Gaussian sigma, in supercell reciprocal-grid steps, with which
            each pixel reads the diffuse grid (27 nearest points). A finite
            supercell's intensity is speckle; reading it through a kernel of
            about one step damps the speckle. 0 reads the 8 nearest points
            trilinearly, about half as many grid points in total (faster).
        shared_scale : bool
            One diffuse scale for every pattern; the envelope carries each
            pattern's absolute Bragg intensities. Ignored with
            ``envelope="fitted"``, whose beam weights are solved per pattern.
        max_beams : int
            Strongest Bragg beams of each pattern in the diffuse envelope.
        device : str, optional
            torch device. Default cuda when available, otherwise cpu; mps is
            used only when requested explicitly.

        Returns
        -------
        ReverseMonteCarlo
            self.
        """
        if self.mask is None:
            raise RuntimeError("set_mask first")
        self._check_envelope(envelope)
        rng = np.random.default_rng(seed)
        self.rng = rng
        d = self.grid_divisor
        if displacements and not (
            d == 2
            and len(self.site_grid) == 2
            and np.array_equal(np.sort(self.site_grid.sum(1)), [0, 3])
        ):
            raise NotImplementedError("Displacements are implemented for BCC sites only.")
        if displacement_grid % 24:
            raise ValueError("displacement_grid must be a multiple of 24.")
        m = displacement_grid // d if displacements else 1
        unit = m * d // 24  # fine-grid steps per a/24
        self._omega_vectors = np.array(
            [k * unit * np.array(v) for k in omega_amplitudes for v in product((1, -1), repeat=3)],
            dtype=np.int64,
        )

        n = cells * d * m
        idx = np.stack(np.meshgrid(*(np.arange(cells),) * 3, indexing="ij"), -1).reshape(-1, 3)
        x0 = ((idx[:, None, :] * d + self.site_grid[None]) * m).reshape(-1, 3)
        n_sites = len(x0)
        counts = np.round(self.concentrations * n_sites).astype(int)
        counts[-1] = n_sites - counts[:-1].sum()
        spec = np.repeat(np.arange(len(counts)), counts)
        rng.shuffle(spec)
        self.cells, self.refine, self.grid_size = cells, m, n
        self.site_x = x0
        self.species_index = spec
        self.displacement = np.zeros((n_sites, 3), dtype=np.int64)  # fine-grid steps
        nc = cells * d
        self._site_lookup = np.full((nc,) * 3, -1, dtype=np.int64)
        xc = x0 // m
        self._site_lookup[xc[:, 0], xc[:, 1], xc[:, 2]] = np.arange(n_sites)
        self.displacements = bool(displacements)
        self._max_steps = int(np.floor(max_displacement / (self._a_crystal / (d * m)) + 1e-9))
        self.size_eta = np.zeros(len(self.species))
        self.debye_waller = float(debye_waller)
        self.symmetrize = bool(symmetrize)
        self.envelope = envelope
        self.resolution = float(resolution)
        self.shared_scale = bool(shared_scale)
        self.max_beams = int(max_beams)
        self.device = torch.device(device or _default_device())
        self._setup_forward()
        print(
            f"{n_sites} sites ("
            + ", ".join(f"{s} {c}" for s, c in zip(self.species, counts))
            + f"), {n}^3 grid, {len(self._needed)} grid points read, "
            f"{int(self._fit.sum())} fitted pixels, {self.device}"
        )
        return self

    def _positions(self, sites: np.ndarray, disp: np.ndarray | None = None) -> np.ndarray:
        disp = self.displacement[sites] if disp is None else disp
        return np.mod(self.site_x[sites] + disp, self.grid_size)

    # metallic (12-fold coordination) radii, A
    _RADII = {
        "V": 1.34,
        "Nb": 1.46,
        "Zr": 1.60,
        "Ti": 1.47,
        "Mo": 1.39,
        "Ta": 1.46,
        "Hf": 1.59,
        "W": 1.39,
        "Cr": 1.28,
        "Fe": 1.26,
        "Al": 1.43,
    }

    def _pixel_grid(self, i: int, q2: np.ndarray):
        """Grid indices and weights (n, 8) trilinear, or (n, 27) Gaussian of ``resolution``
        steps, for in-plane q of pattern i."""
        n = self.grid_size
        h = self._q_crystal(i, q2) * self.lattice_parameter * self.cells
        if self.resolution > 0:
            stencil = np.array(list(product((-1, 0, 1), repeat=3)))
            h0 = np.round(h).astype(np.int64)
            f = h - h0
            wts = np.exp(
                -0.5 * ((f[:, None, :] - stencil[None]) ** 2).sum(-1) / self.resolution**2
            )
            wts /= wts.sum(1, keepdims=True)
        else:
            stencil = np.array(list(product((0, 1), repeat=3)))
            h0 = np.floor(h).astype(np.int64)
            f = h - h0
            wts = np.prod(np.where(stencil[None], f[:, None, :], 1 - f[:, None, :]), axis=-1)
        ijk = np.mod(h0[:, None, :] + stencil[None], n)
        return (ijk[..., 0] * n + ijk[..., 1]) * n + ijk[..., 2], wts

    def _setup_forward(self):
        """Ewald-sphere sampling of the supercell grid for every binned pixel."""
        n = self.grid_size
        dev = self.device
        cols_w, vals_w, u_all, basis_all, pix_image = [], [], [], [], []
        self._env_cols = [None] * len(self.images)
        self._env_p = [None] * len(self.images)
        for i, kk in enumerate(self.mask["k"]):
            q2 = kk.reshape(-1, 2)
            cols, wts = self._pixel_grid(i, q2)
            cols_w.append(cols)
            vals_w.append(wts)
            u, tds = self._envelope(i, q2)
            u_all.append(u)
            qm = np.linalg.norm(q2, axis=1)
            rings = self._powder_rings(qm)
            basis_all.append(np.column_stack([np.ones_like(qm), qm, q2, tds, rings]))
            pix_image.append(np.full(len(q2), i))
        cols = np.concatenate(cols_w)
        vals = np.concatenate(vals_w)
        # only pixels inside q_max are modelled: the fit never reads the rest, and every grid
        # point read costs time in each move
        inside = np.concatenate(
            [
                np.linalg.norm(kk.reshape(-1, 2), axis=1) < self.mask["q_max"]
                for kk in self.mask["k"]
            ]
        )
        used, inv = np.unique(cols[inside], return_inverse=True)
        cols_used = np.zeros(cols.shape, dtype=np.int64)
        cols_used[inside] = inv.reshape(-1, cols.shape[1])
        vals = np.where(inside[:, None], vals, 0.0)
        n_pix = len(cols)
        self._W_all = sparse.csr_matrix(
            (vals.ravel(), (np.repeat(np.arange(n_pix), cols.shape[1]), cols_used.ravel())),
            shape=(n_pix, len(used)),
        )
        self._used = used
        # symmetry: model reads mean_k I(S_k h) at each used h
        ops = cubic_rotations() if self.symmetrize else np.eye(3, dtype=int)[None]
        hh = np.stack(np.unravel_index(used, (n,) * 3), -1)
        sym_flat = np.stack(
            [(lambda s: (s[:, 0] * n + s[:, 1]) * n + s[:, 2])(np.mod(hh @ op.T, n)) for op in ops]
        )
        needed, inv_s = np.unique(sym_flat, return_inverse=True)
        self._needed = needed
        self._sym_index = torch.as_tensor(inv_s.reshape(sym_flat.shape), device=dev)
        hn = np.stack(np.unravel_index(needed, (n,) * 3), -1)
        fs, keep = self._grid_factors(hn)
        self._fs = torch.as_tensor(fs, dtype=torch.float32, device=dev)
        self._keep = torch.as_tensor(keep, dtype=torch.float32, device=dev)
        self._u_all = np.concatenate(u_all)
        self._basis_all = np.concatenate(basis_all)
        self._pix_image = np.concatenate(pix_image)
        y = np.concatenate([a.ravel() for a in self.mask["y"]])
        w = np.concatenate([a.ravel() for a in self.mask["w"]])
        self._y_all, self._w_all = y, w
        fit = w > 1e-3
        self._fit = fit
        self._Wc = torch.as_tensor(cols_used[fit], device=dev)
        self._Wv = torch.as_tensor(vals[fit], dtype=torch.float32, device=dev)
        self._y = torch.as_tensor(y[fit], dtype=torch.float32, device=dev)
        self._w = torch.as_tensor(w[fit], dtype=torch.float32, device=dev)
        self._img = torch.as_tensor(self._pix_image[fit], device=dev)
        self._u = torch.as_tensor(self._u_all[fit], dtype=torch.float32, device=dev)
        self._h_needed = torch.as_tensor(hn.T.copy(), dtype=torch.int32, device=dev)  # (3, n)
        ang = 2 * np.pi * np.arange(n) / n
        self._cos = torch.as_tensor(np.cos(ang), dtype=torch.float32, device=dev)
        self._sin = torch.as_tensor(np.sin(ang), dtype=torch.float32, device=dev)
        self._recompute_F()
        if getattr(self, "background_sigmas", None) is None or len(self.background_sigmas) != len(
            self.images
        ):
            # direct-beam Lorentzian half width, wide Gaussian sigma and its center, 1/A; the
            # wide Gaussian's center floats because a tilted crystal centers its smooth
            # background on the zone-axis pole rather than the direct beam
            self.background_sigmas = [(0.1, 0.8, 0.0, 0.0) for _ in self.images]
        self._sigma_bounds = (np.array([0.01, 0.3, -1.0, -1.0]), np.array([1.0, 3.0, 1.0, 1.0]))
        self.coefficients = getattr(self, "coefficients", None)
        if self.coefficients is not None:
            n_coef = 1 + self._bg_basis(0, np.arange(1)).shape[1]
            if self.coefficients.shape[1] != n_coef:  # background terms changed: pad with zeros
                c = np.zeros((len(self.images), n_coef))
                k = min(n_coef, self.coefficients.shape[1])
                c[:, :k] = self.coefficients[:, :k]
                self.coefficients = c
            self._update_residual()

    _ENVELOPES = ("measured", "fitted", "bloch", "kinematic")

    def _check_envelope(self, envelope: str) -> None:
        """Raise if ``envelope`` is unknown, or is "bloch" before ``fit_thickness``."""
        if envelope not in self._ENVELOPES:
            raise ValueError(f"unknown envelope {envelope!r}; use one of {self._ENVELOPES}")
        if envelope == "bloch":
            bloch_g = (self.geometry or {}).get("bloch_g")
            if bloch_g is None or any(g is None for g in bloch_g):
                raise RuntimeError("fit_thickness before envelope='bloch'")

    def set_envelope(self, envelope: str) -> "ReverseMonteCarlo":
        """Switch the diffuse envelope and refit the scale and background.

        Parameters
        ----------
        envelope : {"measured", "fitted", "bloch", "kinematic"}
            See ``build_supercell``. "bloch" requires ``fit_thickness``.

        Returns
        -------
        ReverseMonteCarlo
            self.
        """
        self._check_envelope(envelope)
        self.envelope = envelope
        self._setup_forward()
        self._solve_linear(self._model_diffuse(), refit_sigmas=True)
        self._update_residual()
        return self

    def _grid_factors(self, h: np.ndarray, n: int | None = None):
        """Scattering factors (K, n) of every species at grid points h (n, 3), and a 0/1 weight
        removing the Bragg nodes of the average lattice."""
        n = self.grid_size if n is None else n
        hm = np.where(h > n // 2, h - n, h)
        q = np.linalg.norm(hm, axis=1) / (self.lattice_parameter * self.cells)
        fs = electron_scattering_factor(
            torch.tensor(self.numbers), torch.as_tensor(q, dtype=torch.float64)
        ).numpy()
        eta = getattr(self, "size_eta", None)
        if eta is not None and np.any(eta != 0):
            fs = (
                fs
                + np.asarray(eta)[:, None]
                * self._size_chi(hm / (self.lattice_parameter * self.cells))[None]
            )
        on_node = np.all(np.mod(hm, self.cells) == 0, axis=1)
        hkl = hm[on_node] // self.cells
        frac = self.site_grid / self.grid_divisor
        f_avg = np.exp(-2j * np.pi * hkl @ frac.T).sum(1)
        keep = np.ones(len(h))
        keep[np.nonzero(on_node)[0][np.abs(f_avg) > 1e-6]] = 0.0
        return fs, keep

    def _size_chi(self, q: np.ndarray, k_ratio: float = 0.5, chunk: int = 200_000) -> np.ndarray:
        """First-order size-effect factor chi(q) (A) at crystal-frame q (n, 3), 1/A.

        Each atom of species s pushes its 8 nearest and 6 next-nearest neighbours with Kanzaki
        forces ``k_n eta_s |r_n| / 2`` along the bond; the lattice relaxes harmonically with the
        same springs, u(q) = D(q)^-1 Phi(q) sum_s eta_s A_s(q). To first order in u the
        amplitude gains ``-2 pi i fbar q.u``, i.e. each species' scattering factor becomes
        ``f_s + eta_s chi(q)`` with ``chi = -2 pi i fbar q.D^-1 Phi`` (real). It is odd about
        every Bragg node and grows as 1/|q - g| toward it: Huang and size-effect scattering.
        """
        a = self.lattice_parameter
        r_n = (
            np.array(
                [list(v) for v in product((0.5, -0.5), repeat=3)]
                + [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
            )
            * a
        )  # A
        length = np.linalg.norm(r_n, axis=1)
        e_n = r_n / length[:, None]
        k_n = np.where(np.arange(14) < 8, 1.0, k_ratio)
        ee = e_n[:, :, None] * e_n[:, None, :]
        out = np.zeros(len(q))
        for c0 in range(0, len(q), chunk):
            qc = q[c0 : c0 + chunk]
            theta = 2 * np.pi * qc @ r_n.T  # (n, 14)
            D = np.einsum("nk,kab->nab", k_n * (1 - np.cos(theta)), ee)
            # Phi = sum_n k |r| / 2 e_n exp(-i theta) = -i sum_n k |r| / 2 e_n sin(theta)
            psi = np.einsum("nk,ka->na", (k_n * length / 2) * np.sin(theta), e_n)
            on_node = np.abs(np.linalg.det(D)) < 1e-12
            D[on_node] = np.eye(3)
            x = np.linalg.solve(D, psi[..., None])[..., 0]  # Phi = -i psi -> D^-1 Phi = -i x
            fbar, _ = self._fbar(np.linalg.norm(qc, axis=1))
            chi = -2 * np.pi * fbar * (qc * x).sum(1)  # -2 pi i fbar q.(-i x)
            chi[on_node] = 0.0
            out[c0 : c0 + chunk] = chi
        return out

    def set_size_effect(self, eta: dict[str, float] | str | None = "radii") -> "ReverseMonteCarlo":
        """Set the linear size effect: species mismatch ``eta_s`` (dimensionless).

        Only differences between species matter (a common shift only moves the
        Bragg peaks). The forward model is rebuilt; call ``fit_background``
        afterwards to refit the scale and background.

        Parameters
        ----------
        eta : "radii", dict or None, optional
            "radii" (default) takes ``(r_s - r_mean) / r_mean`` from metallic
            radii (e.g. V 1.34, Nb 1.46, Zr 1.60 A; ASE covalent radii for
            species without a tabulated metallic radius), with ``r_mean``
            the composition-weighted mean. A dict ``{species: eta}`` sets
            them directly (missing species get 0). None switches the size
            effect off.

        Returns
        -------
        ReverseMonteCarlo
            self. The mismatches are stored in ``self.size_eta``, in the
            order of ``self.species``.
        """
        if eta is None:
            self.size_eta = np.zeros(len(self.species))
        elif isinstance(eta, str):
            from ase.data import atomic_numbers, covalent_radii

            r = np.array(
                [self._RADII.get(sp, covalent_radii[atomic_numbers[sp]]) for sp in self.species]
            )
            r_mean = float((self.concentrations * r).sum())
            self.size_eta = (r - r_mean) / r_mean
        else:
            self.size_eta = np.array([float(eta.get(sp, 0.0)) for sp in self.species])
        self._setup_forward()
        return self

    def _species_amplitudes(self) -> np.ndarray:
        """Lattice sums A_s(h) = sum_{j in s} exp(-2 pi i h.x_j / N) on the needed points (K, n)."""
        n = self.grid_size
        hn = np.stack(np.unravel_index(self._needed, (n,) * 3), -1)
        upper = hn[:, 2] > n // 2
        hr = np.where(upper[:, None], np.mod(-hn, n), hn)
        flat = (hr[:, 0] * n + hr[:, 1]) * (n // 2 + 1) + hr[:, 2]
        out = np.zeros((len(self.species), len(self._needed)), dtype=np.complex64)
        for s_, A in self._species_fft(real=True):
            v = A.ravel()[flat]
            out[s_] = np.where(upper, np.conj(v), v)
        return out

    def fit_size_effect(self, step: float = 0.005, verbose: bool = True) -> dict:
        """Fit the species size mismatches eta_s to the diffuse scattering of the current supercell.

        Nelder-Mead from no size effect; the background and scale are
        re-solved at each step, and one eta is fixed by ``sum c_s eta_s = 0``.
        If the fit does not lower the loss, the size effect stays off.

        Parameters
        ----------
        step : float, optional
            Size of the initial Nelder-Mead simplex in eta. Default 0.005.
        verbose : bool, optional
            Print the fitted mismatches and the loss change.

        Returns
        -------
        dict
            "eta": ``{species: eta_s}``; "loss": (loss without size effect,
            loss after the fit and a background refit).
        """
        dev = self.device
        A = torch.as_tensor(self._species_amplitudes(), device=dev)
        n = self.grid_size
        hn = np.stack(np.unravel_index(self._needed, (n,) * 3), -1)
        hm = np.where(hn > n // 2, hn - n, hn)
        chi = torch.as_tensor(
            self._size_chi(hm / (self.lattice_parameter * self.cells)),
            dtype=torch.float32,
            device=dev,
        )
        self.size_eta = np.zeros(len(self.species))
        fs0, _ = self._grid_factors(hn)
        fs0 = torch.as_tensor(fs0, dtype=torch.float32, device=dev)
        c = self.concentrations

        def full_eta(x):
            e = np.concatenate([x, [0.0]])
            return e - (c * e).sum()

        def loss(x):
            eta = torch.as_tensor(full_eta(x), dtype=torch.float32, device=dev)
            F = ((fs0 + eta[:, None] * chi[None]) * A).sum(0)
            self._Fr, self._Fi = F.real.contiguous(), F.imag.contiguous()
            self._solve_linear(self._model_diffuse())
            return self._update_residual()

        # start from no size effect with a small simplex: at radius-sized mismatches the
        # first-order displacements are no longer small and the loss is far from quadratic
        x0 = np.zeros(len(c) - 1)
        l0 = loss(x0)
        simplex = np.vstack([x0, x0 + step * np.eye(len(x0))])
        sol = optimize.minimize(
            loss,
            x0,
            method="Nelder-Mead",
            options=dict(xatol=1e-5, fatol=1e-4, initial_simplex=simplex),
        )
        if sol.fun > l0:
            sol.x = x0
        self.size_eta = full_eta(sol.x)
        self._setup_forward()
        self._solve_linear(self._model_diffuse(), refit_sigmas=True)
        l1 = self._update_residual()
        out = dict(eta={sp: float(e) for sp, e in zip(self.species, self.size_eta)}, loss=(l0, l1))
        if verbose:
            print(
                "size mismatch eta: "
                + ", ".join(f"{k} {v:+.4f}" for k, v in out["eta"].items())
                + f"; loss {l0:.2f} -> {l1:.2f}"
            )
        return out

    def _read(self, i_used: torch.Tensor) -> torch.Tensor:
        """Grid values on the used points (..., n_used) -> kernel-weighted values on fitted pixels."""
        return (i_used[..., self._Wc] * self._Wv).sum(-1)

    def _fbar(self, q: np.ndarray):
        fe = electron_scattering_factor(
            torch.tensor(self.numbers), torch.as_tensor(q, dtype=torch.float64)
        ).numpy()
        c = self.concentrations[:, None]
        return (c * fe).sum(0), (c * fe**2).sum(0)

    def _envelope(self, i: int, q2: np.ndarray):
        """Diffuse envelope (beam redistribution x Debye-Waller) and Einstein thermal diffuse.

        Also stores the per-beam envelope columns ``fbar^2(q - g) / fbar^2(q) x DW(q)`` and the
        beam weights, which ``envelope="fitted"`` re-solves.
        """
        if self.envelope == "bloch":
            g = (self.geometry.get("bloch_g") or [None] * len(self.images))[i]
            if g is None:
                raise RuntimeError("fit_thickness before envelope='bloch'")
            p = self.geometry["bloch_p"][i]
            order = np.argsort(p)[::-1][: getattr(self, "max_beams", 40)]
            g, p = g[order], p[order]
        elif self.envelope in ("measured", "fitted"):
            g = self.mask["bragg_k"][i]
            p = self.mask["bragg_intensity"][i]
            order = np.argsort(p)[::-1][: getattr(self, "max_beams", 40)]
            g, p = g[order], p[order]
            keep = p > 0.002 * p.max()
            g, p = g[keep], p[keep]
        elif self.envelope == "kinematic":
            g, p = np.zeros((1, 2)), np.array([self.mask["bragg_intensity"][i].sum()])
        else:
            raise ValueError(f"unknown envelope {self.envelope!r}")
        q0 = np.linalg.norm(q2, axis=1)
        fbar0, _ = self._fbar(q0)
        dw0 = np.exp(-0.5 * self.debye_waller * q0**2)
        cols = np.zeros((len(q2), len(g)))
        tds = np.zeros(len(q2))
        for k, (gi, pi) in enumerate(zip(g, p)):
            qm = np.linalg.norm(q2 - gi, axis=1)
            fbar, f2 = self._fbar(qm)
            dw = np.exp(-0.5 * self.debye_waller * qm**2)
            cols[:, k] = fbar**2 / fbar0**2 * dw0
            tds += pi * f2 * (1 - dw)
        self._env_cols[i] = cols
        self._env_p[i] = p.astype(float)
        return cols @ p, tds

    def _bg_lower(self, n: int) -> np.ndarray:
        """Lower bounds of the background amplitudes: the constant may be negative, the others
        are non-negative."""
        lo = np.zeros(n)
        lo[0] = -np.inf
        return lo

    def _powder_rings(self, q: np.ndarray, width: float = 0.025, k_max: float = 1.5) -> np.ndarray:
        """Powder rings of the average crystal about the direct beam, ``sum_g |F_g|^2 / g^2``
        broadened by ``width`` (1/A): misoriented grains or a damaged surface layer."""
        self.crystal.calculate_structure_factors(k_max)
        g = self.crystal.g_len.numpy() * self._a_crystal / self.lattice_parameter
        f2 = self.crystal.struct_factors_int.numpy()
        keep = g > 1e-6
        g, f2 = g[keep], f2[keep]
        out = np.zeros_like(q)
        for gi, fi in zip(g, f2):
            out += fi / gi**2 * np.exp(-0.5 * ((q - gi) / width) ** 2)
        return out / out.max()

    def _phases(self, pos: np.ndarray):
        """cos and sin of 2 pi h.x / N for fine-grid positions (B, 3) on every needed point."""
        p = torch.as_tensor(pos, dtype=torch.int32, device=self.device)
        h = self._h_needed
        idx = (p[:, 0:1] * h[0] + p[:, 1:2] * h[1] + p[:, 2:3] * h[2]) % self.grid_size
        idx = idx.long()
        return self._cos[idx], self._sin[idx]

    def _species_fft(self, coarsen: int = 1, real: bool = False):
        """Per-species FFT of the occupancy, one at a time. ``coarsen`` rounds positions onto a
        grid that many times coarser; ``real`` returns the half spectrum (rfftn)."""
        from scipy import fft as sfft

        n = self.grid_size // coarsen
        pos = self._positions(np.arange(len(self.site_x)))
        pos = np.mod((pos + coarsen // 2) // coarsen, n)
        for s in range(len(self.species)):
            occ = np.zeros((n,) * 3, dtype=np.float32)
            sel = self.species_index == s
            np.add.at(occ, (pos[sel, 0], pos[sel, 1], pos[sel, 2]), 1.0)
            yield s, (sfft.rfftn if real else sfft.fftn)(occ, workers=-1)

    def _recompute_F(self):
        n = self.grid_size
        hn = np.stack(np.unravel_index(self._needed, (n,) * 3), -1)
        upper = hn[:, 2] > n // 2  # read from the conjugate half
        hr = np.where(upper[:, None], np.mod(-hn, n), hn)
        flat = (hr[:, 0] * n + hr[:, 1]) * (n // 2 + 1) + hr[:, 2]
        F = np.zeros(len(self._needed), dtype=np.complex128)
        fs = self._fs.cpu().numpy()
        for s, A in self._species_fft(real=True):
            v = A.ravel()[flat]
            F += fs[s] * np.where(upper, np.conj(v), v)
        self._Fr = torch.as_tensor(F.real, dtype=torch.float32, device=self.device)
        self._Fi = torch.as_tensor(F.imag, dtype=torch.float32, device=self.device)

    def _diffuse_used(self, Fr=None, Fi=None) -> torch.Tensor:
        """Symmetrized diffuse intensity per site on the used grid points."""
        Fr = self._Fr if Fr is None else Fr
        Fi = self._Fi if Fi is None else Fi
        inten = (Fr**2 + Fi**2) * self._keep / len(self.site_x)
        return inten[..., self._sym_index].mean(dim=-2)

    def diffuse_grid(self, max_size: int = 400) -> np.ndarray:
        """Symmetrized diffuse intensity per site on the whole grid (Bragg nodes removed).

        Grids above ``max_size`` per edge are evaluated with positions rounded onto a coarser
        grid (the same reciprocal sampling over a smaller q range), which slightly blurs the
        displacement scattering."""
        coarsen = 1
        while self.grid_size // coarsen > max_size and self.refine % (2 * coarsen) == 0:
            coarsen *= 2
        n = self.grid_size // coarsen
        hh = np.stack(np.meshgrid(*(np.arange(n),) * 3, indexing="ij"), -1).reshape(-1, 3)
        fs, keep = self._grid_factors(hh, n)
        F = np.zeros(n**3, dtype=np.complex64)
        for s, A in self._species_fft(coarsen):
            F += (fs[s] * A.ravel()).astype(np.complex64)
        S = (np.abs(F) ** 2 * keep).reshape((n,) * 3).astype(np.float32) / len(self.site_x)
        if not self.symmetrize:
            return S
        og = np.ogrid[0:n, 0:n, 0:n]
        out = np.zeros_like(S)
        ops = cubic_rotations()
        for op in ops:
            perm = np.argmax(np.abs(op), axis=1)
            sign = op[np.arange(3), perm]
            out += S[tuple(np.mod(sign[r] * og[perm[r]], n) for r in range(3))]
        return out / len(ops)

    # --------------------------------------------------------------- background

    def _bg_basis(self, i: int, sel: np.ndarray) -> np.ndarray:
        q = self._basis_all[sel, 1]
        q2 = self._basis_all[sel, 2:4]
        s1, s2, cy, cx = self.background_sigmas[i]
        return np.column_stack(
            [
                np.ones_like(q),
                1.0 / (1.0 + (q / s1) ** 2),
                np.exp(-0.5 * ((q2 - [cy, cx]) ** 2).sum(1) / s2**2),
                self._basis_all[sel, 4:],
            ]
        )

    def _solve_linear(self, diffuse_fit: np.ndarray, refit_sigmas: bool = False):
        """Scale, background amplitudes (and optionally widths) per image, weighted least squares."""
        y = self._y_all[self._fit]
        w = self._w_all[self._fit]
        img = self._pix_image[self._fit]
        sel_all = np.nonzero(self._fit)[0]
        fitted = self.envelope == "fitted"
        if fitted:
            read = self._read(self._diffuse_used()).cpu().numpy().astype(np.float64)
        coefs, loss = [], 0.0
        for i in range(len(self.images)):
            m = img == i
            sel = sel_all[m]
            sw = np.sqrt(w[m])
            if fitted:
                rows = sel - np.nonzero(self._pix_image == i)[0][0]
                Xd = read[m][:, None] * self._env_cols[i][rows]
            else:
                Xd = diffuse_fit[m][:, None]
            nd = Xd.shape[1]

            def solve(par):
                par = np.clip(par, *self._sigma_bounds)
                self.background_sigmas[i] = tuple(float(v) for v in par)
                X_bg = self._bg_basis(i, sel)
                X = np.column_stack([Xd, X_bg])
                lo = np.concatenate([np.zeros(nd), self._bg_lower(X_bg.shape[1])])
                x = _lsq_bounded(X, y[m], sw, lo)
                return x, float(((X @ x - y[m]) ** 2 * w[m]).sum())

            if refit_sigmas:
                p0 = np.asarray(self.background_sigmas[i], dtype=float)

                def unpack(x):
                    return np.concatenate([np.exp(x[:2]), x[2:]])

                r = optimize.minimize(
                    lambda x: solve(unpack(x))[1],
                    np.concatenate([np.log(p0[:2]), p0[2:]]),
                    method="Nelder-Mead",
                    options=dict(xatol=1e-3, fatol=1e-6, maxiter=300),
                )
                solve(unpack(r.x))
            c, l_i = solve(self.background_sigmas[i])
            if fitted:
                self._env_p[i] = c[:nd]
                c = np.concatenate([[1.0], c[nd:]])
            coefs.append(c)
            loss += l_i
        self.coefficients = np.stack(coefs)
        if fitted:
            self._u_all = np.concatenate(
                [self._env_cols[i] @ self._env_p[i] for i in range(len(self.images))]
            )
            self._u = torch.as_tensor(
                self._u_all[self._fit], dtype=torch.float32, device=self.device
            )
            return loss
        if not self.shared_scale:
            return loss

        # one diffuse scale for all patterns, backgrounds per pattern
        blocks = [self._bg_basis(i, sel_all[img == i]) for i in range(len(self.images))]
        nb = blocks[0].shape[1]
        X = np.zeros((len(y), 1 + nb * len(blocks)))
        X[:, 0] = diffuse_fit
        lo = np.zeros(X.shape[1])
        for i, bl in enumerate(blocks):
            X[img == i, 1 + nb * i : 1 + nb * (i + 1)] = bl
            lo[1 + nb * i : 1 + nb * (i + 1)] = self._bg_lower(nb)
        sw = np.sqrt(w)
        x = _lsq_bounded(X, y, sw, lo)
        if x[0] <= 0:
            # a random supercell explains nothing yet: start the scale from the background residual
            r = y - X[:, 1:] @ x[1:]
            x[0] = max((w * r * diffuse_fit).sum() / max((w * diffuse_fit**2).sum(), 1e-30), 0)
        for i in range(len(blocks)):
            self.coefficients[i, 0] = x[0]
            self.coefficients[i, 1:] = x[1 + nb * i : 1 + nb * (i + 1)]
        return float(((X @ x - y) ** 2 * w).sum())

    def fit_background(self) -> float:
        """Fit the diffuse scale and, per pattern, a constant, a direct-beam Lorentzian, a wide
        Gaussian with a free center, Einstein thermal diffuse and powder rings of the
        average crystal."""
        loss = self._solve_linear(self._model_diffuse(), refit_sigmas=True)
        self._update_residual()
        print(
            "background widths (1/A): "
            + ", ".join(
                f"{n} {s[0]:.3f}/{s[1]:.3f} at ({s[2]:+.2f}, {s[3]:+.2f})"
                for n, s in zip(self.names, self.background_sigmas)
            )
        )
        return loss

    def _model_diffuse(self) -> np.ndarray:
        """u(q) * (W I_sym) on the fitted pixels (before scale)."""
        return (self._u * self._read(self._diffuse_used())).cpu().numpy().astype(np.float64)

    def _update_residual(self):
        dev = self.device
        img = self._pix_image[self._fit]
        sel = np.nonzero(self._fit)[0]
        bg = np.zeros(len(sel))
        for i in range(len(self.images)):
            m = img == i
            bg[m] = self._bg_basis(i, sel[m]) @ self.coefficients[i, 1:]
        self._bg = torch.as_tensor(bg, dtype=torch.float32, device=dev)
        c = torch.as_tensor(self.coefficients[:, 0], dtype=torch.float32, device=dev)
        self._scale = c[self._img]
        model = self._scale * self._u * self._read(self._diffuse_used()) + self._bg
        self._r = self._y - model
        return float((self._w * self._r**2).sum())

    # ---------------------------------------------------------------------- RMC

    def run(
        self,
        n_sweeps: int = 20,
        batch: int = 32,
        temperature: float = 0.05,
        random_fraction: float = 0.5,
        omega_fraction: float = 0.0,
        max_static_b: float | None = 0.5,
        refit_every: int = 2,
        progress: bool = True,
    ) -> "ReverseMonteCarlo":
        """Species swaps and displacements until the diffuse fit converges.

        Each batch proposes ``batch`` moves on distinct sites against the
        current supercell and scores each exactly. With probability
        ``random_fraction`` the batch moves single atoms by one grid step
        (-1, 0 or +1 along each axis, up to ``max_displacement``), with no
        assumed pattern, so any displacement correlation has to come from the
        data. With probability ``omega_fraction`` it proposes omega embryos
        instead: three consecutive atoms of a <111> row, the first fixed and
        the next two collapsed toward each other, (0, +v, -v), or cleared
        where one already stands. Otherwise it swaps two unlike atoms
        (composition conserved). ``max_static_b`` (A^2) caps the
        static Debye-Waller factor of all displacements, 8 pi^2 <u^2> / 3, so
        they stay consistent with how slowly the Bragg intensities fall off
        (a displacement field this strong would damp them; the diffuse scale
        alone does not fix how many atoms are displaced). Metropolis acceptance
        at ``temperature`` (a fraction of the median score change, falling
        linearly to 0); the accepted moves are applied together, capped at a
        number that adapts so the joint step never raises the loss. A sweep
        is one proposal per site. Scale and background are refit every
        ``refit_every`` sweeps.

        Parameters
        ----------
        n_sweeps : int, optional
            Number of sweeps. Default 20.
        batch : int, optional
            Moves proposed per batch. Default 32.
        temperature : float, optional
            Initial Metropolis temperature as a fraction of the median
            absolute score change of the first batch. Default 0.05.
        random_fraction : float, optional
            Probability that a batch proposes single-atom displacements.
            Ignored (0) if the supercell has no displacements. Default 0.5.
        omega_fraction : float, optional
            Probability that a batch proposes omega embryos. Ignored (0) if
            the supercell has no displacements. Default 0.
        max_static_b : float or None, optional
            Cap on the static Debye-Waller B (A^2) of all displacements; None
            for no cap. Default 0.5.
        refit_every : int, optional
            Sweeps between refits of the scale and background. Default 2.
        progress : bool, optional
            Show a progress bar.

        Returns
        -------
        ReverseMonteCarlo
            self. The weighted loss after each sweep is appended to
            ``self.loss_history``.
        """
        if self.coefficients is None:
            self.fit_background()
        n_sites = len(self.site_x)
        loss = self._update_residual()
        if not self.loss_history:
            self.loss_history.append(loss)
        cap = max(batch // 8, 1)
        t0 = None
        n_batches = max(n_sites // batch, 1)
        p_omega = omega_fraction if self.displacements else 0.0
        p_random = random_fraction if self.displacements else 0.0
        budget = (
            3 * max_static_b / (8 * np.pi**2) * n_sites if max_static_b is not None else np.inf
        )
        keep = self._keep / n_sites
        sweeps = tqdm(range(n_sweeps), desc="RMC sweeps", disable=not progress)
        su = None
        for sweep in sweeps:
            accepted = {"swap": 0, "random": 0, "omega": 0}
            for _ in range(n_batches):
                if su is None:
                    su = self._scale * self._u
                spec = self.species_index
                roll = self.rng.random()
                if roll < p_random:
                    kind = "random"
                    sites, new = self._random_proposals(batch)
                elif roll < p_random + p_omega:
                    kind = "omega"
                    sites, new = self._omega_proposals(batch)
                else:
                    kind = "swap"
                    j = self.rng.choice(n_sites, 2 * batch, replace=False)
                    j1, j2 = j[:batch], j[batch:]
                    ok = spec[j1] != spec[j2]
                    sites = np.stack([j1[ok], j2[ok]], 1)
                    new = None
                if len(sites) == 0:
                    continue
                if kind == "swap":
                    c1, s1 = self._phases(self._positions(sites[:, 0]))
                    c2, s2 = self._phases(self._positions(sites[:, 1]))
                    df = (
                        self._fs[torch.as_tensor(spec[sites[:, 1]], device=self.device)]
                        - self._fs[torch.as_tensor(spec[sites[:, 0]], device=self.device)]
                    )
                    dFr = df * (c1 - c2)
                    dFi = df * (s2 - s1)
                else:
                    dFr = torch.zeros((len(sites), len(self._needed)), device=self.device)
                    dFi = torch.zeros_like(dFr)
                    for k in range(sites.shape[1]):
                        j = sites[:, k]
                        changed = np.any(new[:, k] != self.displacement[j], axis=1)
                        if not changed.any():
                            continue
                        co, so = self._phases(self._positions(j))
                        cn, sn = self._phases(self._positions(j, new[:, k]))
                        f = self._fs[torch.as_tensor(spec[j], device=self.device)]
                        f = (
                            f
                            * torch.as_tensor(changed, dtype=torch.float32, device=self.device)[
                                :, None
                            ]
                        )
                        dFr += f * (cn - co)
                        dFi += f * (so - sn)
                d_int = (2 * (self._Fr * dFr + self._Fi * dFi) + dFr**2 + dFi**2) * keep
                d_used = d_int[:, self._sym_index].mean(dim=1)
                dm = su * self._read(d_used)
                dL = (self._w * (dm**2 - 2 * self._r * dm)).sum(-1)
                dl = dL.cpu().numpy()
                nb = len(dl)
                if not t0:
                    t0 = float(np.median(np.abs(dl)))
                temp = temperature * t0 * (1 - sweep / n_sweeps)
                if temp > 0:
                    acc = (dl < 0) | (self.rng.random(nb) < np.exp(-np.clip(dl / temp, 0, 50)))
                else:
                    acc = dl < 0
                pick = np.nonzero(acc)[0]
                pick = pick[np.argsort(dl[pick])][:cap]
                if kind != "swap" and max_static_b is not None:
                    du2 = (self._u2(new) - self._u2(self.displacement[sites])).sum(1)
                    total = self._u2(self.displacement).sum()
                    chosen = []
                    for b in pick:
                        if du2[b] <= 0 or total + du2[b] <= budget:
                            chosen.append(b)
                            total += du2[b]
                    pick = np.asarray(chosen, dtype=int)
                if len(pick) == 0:
                    continue
                pk = torch.as_tensor(pick, device=self.device)
                Fr_new = self._Fr + dFr[pk].sum(0)
                Fi_new = self._Fi + dFi[pk].sum(0)
                r_new = self._y - su * self._read(self._diffuse_used(Fr_new, Fi_new)) - self._bg
                loss_new = float((self._w * r_new**2).sum())
                spec_new, disp_new = spec.copy(), self.displacement.copy()
                if kind == "swap":
                    a, b = sites[pick, 0], sites[pick, 1]
                    spec_new[a], spec_new[b] = spec[b], spec[a]
                else:
                    disp_new[sites[pick]] = new[pick]
                if loss_new > loss + max(temp, 0.0) * len(pick) and len(pick) > 1:
                    cap = max(cap // 2, 1)
                    continue
                self._Fr, self._Fi, self._r = Fr_new, Fi_new, r_new
                self.species_index[:] = spec_new
                self.displacement[:] = disp_new
                loss = loss_new
                accepted[kind] += len(pick)
                cap = min(int(cap * 1.25) + 1, batch)
            if (sweep + 1) % refit_every == 0:
                self._recompute_F()
                self._solve_linear(self._model_diffuse())
                loss = self._update_residual()
                su = None
            self.loss_history.append(loss)
            sweeps.set_postfix(loss=f"{loss:.4g}", **accepted)
        self._recompute_F()
        self._solve_linear(self._model_diffuse())
        self.loss_history[-1] = self._update_residual()
        return self

    def _random_proposals(self, batch: int):
        """Single atoms moved by -1, 0 or +1 grid steps along each axis (not all zero), within
        ``max_displacement``: sites (B, 1) and new displacement vectors (B, 1, 3)."""
        n_sites = len(self.site_x)
        j = self.rng.choice(n_sites, batch, replace=False)
        delta = self.rng.integers(-1, 2, (batch, 3))
        zero = ~np.any(delta, axis=1)
        delta[zero, self.rng.integers(0, 3, zero.sum())] = self.rng.choice((-1, 1), zero.sum())
        new = self.displacement[j] + delta
        ok = np.all(np.abs(new) <= self._max_steps, axis=1)
        return j[ok][:, None], new[ok][:, None, :]

    def _omega_proposals(self, batch: int):
        """Disjoint omega embryos along <111> rows: sites (B, 3) and their new displacement
        vectors (B, 3, 3), the (0, +v, -v) collapse."""
        n_sites = len(self.site_x)
        nc = self._site_lookup.shape[0]
        j0 = self.rng.choice(n_sites, batch, replace=False)
        vec = self._omega_vectors[self.rng.integers(0, len(self._omega_vectors), batch)]
        step = np.sign(vec)  # nearest neighbour along that <111>, in site units
        xc = self.site_x[j0] // self.refine
        sites = np.stack(
            [self._site_lookup[tuple(np.mod(xc + t * step, nc).T)] for t in range(3)], 1
        )
        pattern = np.stack([np.zeros_like(vec), vec, -vec], 1)  # (B, 3, 3)
        current = self.displacement[sites]
        standing = np.all(current == pattern, axis=(1, 2))
        new = np.where(standing[:, None, None], 0, pattern)
        keep = _disjoint_rows(sites)
        return sites[keep], new[keep]

    # ---------------------------------------------------------------- analysis

    def to_atoms(self, lattice_parameter: float | None = None):
        """The fitted supercell as an ``ase.Atoms`` object, one atom per site.

        The mixed sites carry the species and static displacements of the fit; any ordered
        sites of the average crystal are repeated over the supercell without displacements.
        Atoms are grouped by species.
        Species are labelled as fitted, so with ``set_crystal(merge={"Zr": "Nb"})`` the Nb
        sites stand for both Nb and Zr.

        Parameters
        ----------
        lattice_parameter : float, optional
            Edge of the cubic unit cell in A. Default None: the lattice parameter of the
            crystal passed to :meth:`set_crystal`. The fractional coordinates do not depend
            on it; the lattice parameter fitted by :meth:`fit_geometry` also absorbs any error
            in the detector pixel size.

        Returns
        -------
        ase.Atoms
            Periodic cubic supercell of ``cells`` unit cells along each axis.
        """
        from ase import Atoms
        from ase.data import chemical_symbols

        if getattr(self, "site_x", None) is None:
            raise RuntimeError("build_supercell first")
        a = self._a_crystal if lattice_parameter is None else float(lattice_parameter)
        if not np.isfinite(a) or a <= 0:
            raise ValueError("lattice_parameter must be positive")
        frac = [self._positions(np.arange(len(self.site_x))) / self.grid_size]
        symbols = [self.species[k] for k in self.species_index]

        # ordered sites of the average crystal (any site that is not a fitted mixed site)
        mixed = self.site_grid / self.grid_divisor
        pos0 = np.mod(self.crystal.positions_frac.numpy(), 1.0)
        numbers = self.crystal.numbers.numpy()
        ordered = {}
        for f, z in zip(pos0, numbers):
            if np.any(np.all(np.abs(np.mod(f - mixed + 0.5, 1.0) - 0.5) < 1e-4, axis=1)):
                continue
            key = tuple(np.round(f, 6))
            ordered.setdefault(key, chemical_symbols[int(z)])
        if ordered:
            idx = np.stack(np.meshgrid(*(np.arange(self.cells),) * 3, indexing="ij"), -1).reshape(
                -1, 3
            )
            f0 = np.array(list(ordered))
            frac.append(((idx[:, None, :] + f0[None]) / self.cells).reshape(-1, 3))
            symbols += list(ordered.values()) * len(idx)
        # atoms grouped by species (stable within each), so the formula reads e.g. Nb12480V3520
        order = np.argsort(np.array(symbols), kind="stable")
        return Atoms(
            [symbols[i] for i in order],
            scaled_positions=np.mod(np.concatenate(frac), 1.0)[order],
            cell=np.eye(3) * a * self.cells,
            pbc=True,
        )

    def to_cif(self, path, lattice_parameter: float | None = None):
        """Write the fitted supercell (see :meth:`to_atoms`) to a CIF file with ASE.

        Parameters
        ----------
        path : str or Path
            Output file.
        lattice_parameter : float, optional
            Edge of the cubic unit cell in A. Default None: the crystal's lattice parameter.

        Returns
        -------
        pathlib.Path
            The file written.
        """
        from pathlib import Path

        from ase.io import write

        path = Path(path)
        write(path, self.to_atoms(lattice_parameter), format="cif")
        return path

    def model_images(self, diffuse_only: bool = False) -> list[np.ndarray]:
        """Model on the binned grid of every pattern (data units): scaled diffuse + background,
        or the scaled diffuse term alone. The diffuse term is zero beyond ``q_max``."""
        i_used = self._diffuse_used().cpu().numpy().astype(np.float64)
        diffuse = self._u_all * (self._W_all @ i_used)
        out = []
        for i, k in enumerate(self.mask["k"]):
            m = self._pix_image == i
            sel = np.nonzero(m)[0]
            c = self.coefficients[i]
            img = c[0] * diffuse[m]
            if not diffuse_only:
                img = img + self._bg_basis(i, sel) @ c[1:]
            out.append(img.reshape(k.shape[:2]))
        return out

    def r_factors(self) -> dict:
        """Weighted R of the diffuse fit per pattern.

        ``R = sqrt(sum w (y - model)^2 / sum w (y - background)^2)``: the
        fraction of the diffuse signal (experiment minus fitted background)
        the supercell leaves unexplained.

        Returns
        -------
        dict
            ``{pattern name: R}``.
        """
        full = self.model_images()
        bg = self.background_images()
        out = {}
        for i, name in enumerate(self.names):
            w, y = self.mask["w"][i], self.mask["y"][i]
            out[name] = float(
                np.sqrt((w * (y - full[i]) ** 2).sum() / max((w * (y - bg[i]) ** 2).sum(), 1e-30))
            )
        return out

    def background_images(self) -> list[np.ndarray]:
        """Fitted smooth background per pattern."""
        full = self.model_images()
        diffuse = self.model_images(diffuse_only=True)
        return [f - d for f, d in zip(full, diffuse)]

    def warren_cowley(self, n_shells: int | None = 6, max_radius: float | None = None) -> dict:
        """Warren-Cowley alpha for every species pair over the first neighbour shells.

        ``alpha[shell, s, t] = 1 - P(t | s) / c_t`` for unlike pairs and
        ``(P(s | s) - c_s) / (1 - c_s)`` for like pairs, with ``P(t | s)`` the
        fraction of the shell around an ``s`` atom occupied by ``t``.
        Negative: unlike neighbours preferred; positive: like.

        Parameters
        ----------
        n_shells : int, optional
            Largest number of neighbour shells. Default 6; None keeps every shell within
            `max_radius`.
        max_radius : float, optional
            Largest neighbour distance in A. Default None (no limit beyond half the
            supercell).

        Returns
        -------
        dict
            "radius": (n_shells,) shell radii in A; "alpha": (n_shells, K, K)
            Warren-Cowley parameters; "species": the K species in index order.
        """
        radii, prob = self._pair_probabilities(n_shells=n_shells, max_radius=max_radius)
        c = self.concentrations
        K = len(self.species)
        alpha = np.zeros_like(prob)
        for s in range(K):
            for t in range(K):
                p = prob[:, s, t]
                alpha[:, s, t] = (p - c[t]) / (1 - c[t]) if s == t else 1 - p / c[t]
        return dict(radius=radii, alpha=alpha, species=list(self.species))

    def _pair_probabilities(
        self, n_shells: int | None = None, max_radius: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shell radii (A) and ``P(t | s)`` (n_shells, K, K): the fraction of the sites in each
        neighbour shell of an ``s`` atom that hold a ``t`` atom, averaged over every ``s`` atom
        of the supercell (periodic boundaries). Shells are kept up to `n_shells` and up to
        `max_radius` (A), whichever ends first, and never beyond half the supercell."""
        d = self.grid_divisor
        n = self.cells * d
        xs = self.site_x // self.refine
        K = len(self.species)
        site = np.zeros((n,) * 3)
        site[xs[:, 0], xs[:, 1], xs[:, 2]] = 1
        fsite = np.fft.fftn(site)
        focc = []
        for s in range(K):
            occ = np.zeros((n,) * 3)
            sel = self.species_index == s
            occ[xs[sel, 0], xs[sel, 1], xs[sel, 2]] = 1
            focc.append(np.fft.fftn(occ))
        ss = np.real(np.fft.ifftn(fsite * np.conj(fsite)))
        v = np.stack(np.meshgrid(*(np.fft.fftfreq(n, 1 / n),) * 3, indexing="ij"), -1)
        r = np.linalg.norm(v, axis=-1) / d * self.lattice_parameter
        half_box = self.cells * self.lattice_parameter / 2
        r_max = half_box if max_radius is None else min(max_radius, half_box)
        valid = (ss > 0.5) & (r <= r_max + 1e-6)
        radii = np.unique(np.round(r[valid], 4))[1:]
        if n_shells is not None:
            radii = radii[:n_shells]
        shell = np.searchsorted(radii, np.round(r, 4))
        inside = valid & (shell < len(radii))
        inside[inside] = radii[shell[inside]] == np.round(r[inside], 4)
        index = shell[inside]

        def shell_sums(a):
            return np.bincount(index, weights=a[inside], minlength=len(radii))

        prob = np.zeros((len(radii), K, K))
        for s in range(K):
            cs_site = shell_sums(np.real(np.fft.ifftn(focc[s] * np.conj(fsite))))
            for t in range(K):
                cst = shell_sums(np.real(np.fft.ifftn(focc[s] * np.conj(focc[t]))))
                prob[:, s, t] = cst / cs_site
        return radii, prob

    def shell_correlations(
        self, n_shells: int | None = None, max_radius: float | None = None
    ) -> dict:
        """Pair correlation of every species pair per neighbour shell, relative to a random
        alloy.

        ``ratio[shell, s, t] = P(t | s) / c_t``, the number of ``s``-``t`` pairs in the shell
        divided by the number expected for a random arrangement at the same composition. It
        is symmetric in ``s`` and ``t``, equals 1 for a random alloy, is above 1 for pairs that
        are favoured and below 1 for pairs that are avoided. ``c_t`` is the composition of
        the supercell itself. For an unlike pair, ``ratio = 1 - alpha`` with the Warren-Cowley
        parameter of :meth:`warren_cowley`, up to the rounding of the composition to whole
        atoms.

        Parameters
        ----------
        n_shells : int, optional
            Largest number of neighbour shells. Default None (every shell within
            `max_radius`).
        max_radius : float, optional
            Largest neighbour distance in A. Default None: five lattice parameters, far
            enough for the short-range order to decay. Never beyond half the supercell.

        Returns
        -------
        dict
            "radius": (n_shells,) shell radii in A; "shell": bond-vector labels in units of a
            (e.g. "1/2<111>"), or None for site lattices other than BCC; "ratio":
            (n_shells, K, K) pair correlations relative to random; "species": the K species
            in index order.
        """
        if max_radius is None:
            max_radius = 5 * self.lattice_parameter
        radii, prob = self._pair_probabilities(n_shells=n_shells, max_radius=max_radius)
        # the supercell's own composition, so that the ratio is exactly symmetric
        counts = np.bincount(self.species_index, minlength=len(self.species))
        ratio = prob / (counts / counts.sum())[None, None, :]
        shell = None
        if self.grid_divisor == 2:
            reach = int(np.ceil(radii[-1] / (self.lattice_parameter / 2))) + 1
            offsets = self._shells(len(radii), reach=reach)
            dist = [np.linalg.norm(o[0]) * self.lattice_parameter / 2 for o in offsets]
            if len(offsets) == len(radii) and np.allclose(dist, radii, atol=1e-3):
                shell = [self._shell_label(o[0]) for o in offsets]
        return dict(radius=radii, shell=shell, ratio=ratio, species=list(self.species))

    def _u2(self, disp: np.ndarray) -> np.ndarray:
        """Squared displacement (A^2) of each displacement vector (..., 3)."""
        step = self.lattice_parameter / (self.grid_divisor * self.refine)
        return (disp**2).sum(-1) * step**2

    def static_b(self) -> float:
        """Static Debye-Waller B (A^2) of the displacements, 8 pi^2 <u^2> / 3."""
        return float(8 * np.pi**2 * self._u2(self.displacement).mean() / 3)

    def _omega_like(self) -> np.ndarray:
        """Atoms displaced along a <111> by at least a half omega collapse."""
        a = np.abs(self.displacement)
        unit = self.refine * self.grid_divisor // 24
        return (a[:, 0] >= unit) & (a[:, 0] == a[:, 1]) & (a[:, 1] == a[:, 2])

    def displacement_summary(self) -> dict:
        """Static Debye-Waller B, mean displacement and omega fraction per species."""
        step = self.lattice_parameter / (self.grid_divisor * self.refine)
        mag = np.linalg.norm(self.displacement, axis=1) * step
        omega = self._omega_like()
        out = dict(
            static_b=self.static_b(),
            mean_displacement={
                s: float(mag[self.species_index == k].mean()) for k, s in enumerate(self.species)
            },
            omega_fraction={
                s: float(omega[self.species_index == k].mean()) for k, s in enumerate(self.species)
            },
        )
        return out

    def diffuse_section(
        self,
        normal=(0, 0, 1),
        extent: float = 2.0,
        smooth: bool = True,
        grid: np.ndarray | None = None,
    ):
        """Symmetrized supercell diffuse intensity on a reciprocal-lattice plane, in Laue units.

        Parameters
        ----------
        normal : (h, k, l)
            Plane normal; the plane passes through the origin.
        extent : float
            Half width in units of the cubic reciprocal lattice vector 1/a.
        smooth : bool
            Blur by the fit's ``resolution`` kernel.
        grid : ndarray, optional
            Output of ``diffuse_grid()``, to reuse.

        Returns
        -------
        image, (u, v) in-plane axes (crystal frame, unit vectors), distance of each pixel from
        the nearest reciprocal-lattice node (1/a units)
        """
        grid = self.diffuse_grid() if grid is None else grid
        if smooth and self.resolution > 0:
            grid = ndimage.gaussian_filter(grid, self.resolution, mode="wrap")
        frame = _zone_frame(normal)
        u, v = frame[0], frame[1]
        steps = np.arange(-extent * self.cells, extent * self.cells + 1)
        su, sv = np.meshgrid(steps, steps, indexing="ij")
        pts = su[..., None] * u + sv[..., None] * v  # grid units (cells per 1/a)
        img = ndimage.map_coordinates(
            grid, np.moveaxis(pts, -1, 0).reshape(3, -1), order=1, mode="grid-wrap"
        ).reshape(su.shape)
        q = np.linalg.norm(pts, axis=-1) / (self.cells * self.lattice_parameter)
        fbar, f2 = self._fbar(q.ravel())
        laue = (f2 - fbar**2).reshape(q.shape)
        frac = pts / self.cells
        node_dist = np.linalg.norm(frac - np.round(frac), axis=-1)  # 1/a units
        return img / laue, (u, v), node_dist

    # ----------------------------------------------------------------- plotting

    def plot_images(
        self, quantiles: tuple[float, float] = (0.86, 0.98), cmap: str = "turbo_black", **kwargs
    ):
        """Binned patterns on a linear scale. Once a mask is set, the scale spans min to max of the
        diffuse region of each pattern; before that, ``quantiles`` of the whole pattern."""
        from quantem.core.visualization import show_2d

        arrays, norms = [], []
        for i, im in enumerate(self.images):
            b = self._binned(im)
            if self.mask is not None:
                q = np.linalg.norm(self.mask["k"][i], axis=-1)
                sel = (self.mask["w"][i] > 0.5) & (q < self.mask["q_max"])
                vals = b[sel]
                norms.append(dict(interval_type="manual", vmin=vals.min(), vmax=vals.max()))
            else:
                lo, hi = np.quantile(b, quantiles)
                norms.append(dict(interval_type="manual", vmin=lo, vmax=hi))
            arrays.append(b)
        return show_2d(
            arrays,
            title=self.names,
            norm=norms,
            cmap=cmap,
            axsize=kwargs.pop("axsize", (6, 4)),
            **kwargs,
        )

    def plot_geometry(self, power: float = 0.3, q_view: float = 1.4, **kwargs):
        """Each pattern with its fitted reflections (red), direct beam (cyan) and Laue circle (yellow)."""
        import matplotlib.patches as mpatches

        from quantem.core.visualization import show_2d

        fig, axs = show_2d(
            self.images,
            title=self.names,
            norm={
                "stretch_type": "power",
                "power": power,
                "lower_quantile": 0.05,
                "upper_quantile": 0.999,
            },
            axsize=kwargs.pop("axsize", (5, 5)),
            **kwargs,
        )
        axs = np.atleast_1d(axs).ravel()
        r = 0.03 / self.sampling
        k_wave = 1.0 / self.wavelength
        for i, ax in enumerate(axs):
            p = self.bragg_positions(i)
            c = self.geometry["centers"][i]
            for pr, pc in p:
                ax.add_patch(mpatches.Circle((pc, pr), r, fill=False, color="tab:red", lw=1.0))
            ax.add_patch(mpatches.Circle((c[1], c[0]), 1.5 * r, fill=False, color="cyan", lw=1.5))
            tilt = self.geometry["tilts"][i]
            if np.linalg.norm(tilt) > 0:
                th = np.linspace(0, 2 * np.pi, 361)
                circ = -k_wave * tilt + k_wave * np.linalg.norm(tilt) * np.column_stack(
                    [np.cos(th), np.sin(th)]
                )
                px = (
                    c
                    + (circ * self.lattice_parameter / self._a_crystal)
                    @ self.geometry["matrices"][i].T
                )
                ax.plot(px[:, 1], px[:, 0], "--", color="yellow", lw=1.0)
            half = q_view / self.sampling
            ax.set_xlim(c[1] - half, c[1] + half)
            ax.set_ylim(c[0] + half, c[0] - half)
        return fig, axs

    def plot_mask(self, **kwargs):
        """Binned diffuse weight of every pattern."""
        from quantem.core.visualization import show_2d

        return show_2d(
            self.mask["w"],
            title=self.names,
            cmap="gray",
            axsize=kwargs.pop("axsize", (4, 2.7)),
            **kwargs,
        )

    def plot_fit(
        self,
        columns: Sequence[str] = ("experiment", "background", "difference", "model", "residual"),
        diffuse_only: bool = False,
        sigma: float = 0.7,
        quantiles: tuple[float, float] = (0.01, 0.99),
        cmap: str = "turbo_black",
        **kwargs,
    ):
        """Experiment, fitted background, experiment - background, supercell diffuse model and
        residual, one row per zone, cropped to ``q_max``.

        Experiment and background share a linear scale spanning ``quantiles`` of the
        experiment inside the diffuse mask (weight > 0.5); experiment - background and the
        model share one spanning ``quantiles`` of the difference; the residual uses a diverging
        map at half that range (white = no difference). Masked pixels are black in the last
        three columns. The experiment is blurred by ``sigma`` binned pixels. ``diffuse_only``
        keeps the last three columns.
        """
        import matplotlib

        from quantem.core.visualization import show_2d

        if diffuse_only:
            columns = ("difference", "model", "residual")
        full = self.model_images()
        diffuse = self.model_images(diffuse_only=True)
        cmap_obj = matplotlib.colormaps[cmap].with_extremes(bad="black")
        diverging = matplotlib.colormaps["RdBu_r"].with_extremes(bad="black")
        titles_of = {
            "experiment": "experiment",
            "background": "background",
            "difference": "experiment - background",
            "model": "model",
            "residual": "residual",
        }
        rows, titles, norms, cmaps = [], [], [], []
        for i in range(len(self.images)):
            q = np.linalg.norm(self.mask["k"][i], axis=-1)
            inside = q < self.mask["q_max"]
            w = self.mask["w"][i]
            sel = (w > 0.5) & inside
            rr, cc = np.nonzero(inside)
            crop = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
            background = full[i] - diffuse[i]
            exp = self.mask["data"][i]
            if sigma:
                exp = ndimage.gaussian_filter(exp, sigma)
            diff = self.mask["y"][i] - background
            if sigma:
                diff = _nan_blur(np.where(w > 0.2, diff, np.nan), sigma)
            diff = np.where(sel, diff, np.nan)
            panels = {
                "experiment": np.where(inside, exp, np.nan),
                "background": np.where(inside, background, np.nan),
                "difference": diff,
                "model": np.where(sel, diffuse[i], np.nan),
                "residual": diff - np.where(sel, diffuse[i], np.nan),
            }
            lo, hi = np.quantile(exp[sel], quantiles)
            dlo, dhi = np.nanquantile(diff, quantiles)
            half = 0.5 * (dhi - dlo)
            scales = {
                "experiment": (lo, hi),
                "background": (lo, hi),
                "difference": (dlo, dhi),
                "model": (dlo, dhi),
                "residual": (-half, half),
            }
            rows.append([panels[c][crop] for c in columns])
            titles.append([f"{self.names[i]} {titles_of[c]}" for c in columns])
            norms.append(
                [
                    dict(interval_type="manual", vmin=scales[c][0], vmax=scales[c][1])
                    for c in columns
                ]
            )
            cmaps.append([diverging if c == "residual" else cmap_obj for c in columns])
        return show_2d(
            rows,
            title=titles,
            norm=norms,
            cmap=cmaps,
            axsize=kwargs.pop("axsize", (3.4, 3.4)),
            **kwargs,
        )

    def plot_warren_cowley(self, max_radius: float | None = None):
        """Warren-Cowley alpha against neighbour distance (0 = random, < 0 unlike neighbours
        preferred, > 0 like); one curve for a binary site, one per species pair otherwise.

        Parameters
        ----------
        max_radius : float, optional
            Largest neighbour distance in A. Default None: five lattice parameters.

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt

        max_radius = self._default_shell_radius(max_radius)
        sro = self.warren_cowley(n_shells=None, max_radius=max_radius)
        sc = self.shell_correlations(max_radius=max_radius)
        labelled = self._direction_shells(sc["radius"], sc["shell"])
        K = len(self.species)
        fig, ax = plt.subplots(figsize=(8.0, 3.6))
        ax.axhline(0, color="0.6", lw=0.8)
        for radius, _ in labelled:
            ax.axvline(radius, color="0.9", lw=0.6, zorder=0)
        if K == 2:  # a binary site has one alpha for every pair
            ax.plot(sro["radius"], sro["alpha"][:, 0, 0], "o-", ms=3, lw=1.2, color="tab:blue")
            ax.set_ylabel(f"Warren-Cowley alpha, {self.species[0]}-{self.species[1]}")
        else:
            for s_ in range(K):
                for t in range(s_, K):
                    ax.plot(
                        sro["radius"],
                        sro["alpha"][:, s_, t],
                        "o-" if s_ == t else "s--",
                        ms=3,
                        lw=1.2,
                        label=f"{self.species[s_]}-{self.species[t]}",
                    )
            ax.legend(fontsize=8)
            ax.set_ylabel("Warren-Cowley alpha")
        ax.set_xlabel("neighbour distance (A)")
        ax.set_xlim(0, sro["radius"][-1] * 1.02)
        fig.tight_layout()
        self._label_direction_shells(fig, ax, labelled, sro["radius"][-1] * 1.02)
        return fig, ax

    def _default_shell_radius(self, max_radius: float | None) -> float:
        """Largest neighbour distance of the shell plots: `max_radius`, or five lattice parameters."""
        return 5 * self.lattice_parameter if max_radius is None else max_radius

    @staticmethod
    def _direction_shells(radii, labels) -> list[tuple[float, str]]:
        """(radius, label) of the shells along <100>, <110> and <111>, from the bond-vector
        labels of :meth:`shell_correlations`; empty when there are no labels."""
        if labels is None:
            return []
        out = []
        for radius, label in zip(radii, labels):
            digits = [int(c) for c in label.split("<")[1].rstrip(">")]
            if (
                digits[1:] == [0, 0]
                or (digits[0] == digits[1] and digits[2] == 0)
                or digits[0] == digits[1] == digits[2]
            ):
                out.append((radius, label))
        return out

    @staticmethod
    def _label_direction_shells(fig, ax, labelled, span: float) -> None:
        """Write shell labels above `ax`, stacked in rows so that none overlap, and make room
        for them at the top of the figure. Call after ``tight_layout``."""
        last: list[float] = []
        for radius, label in labelled:
            width = 0.012 * span * len(label)
            row = next((k for k, x in enumerate(last) if radius - width / 2 > x), len(last))
            if row == len(last):
                last.append(0.0)
            last[row] = radius + width / 2
            ax.annotate(
                label,
                (radius, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 3 + 11 * row),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                annotation_clip=False,
            )
        if last:
            height = fig.get_size_inches()[1]
            fig.subplots_adjust(top=fig.subplotpars.top - (0.03 + 0.15 * len(last)) / height)

    def plot_shell_correlations(
        self,
        max_radius: float | None = None,
        panel_size: tuple[float, float] = (8.0, 2.2),
    ):
        """Pair correlation of every species pair against neighbour distance, relative to a
        random alloy (1 = random), one panel per pair stacked on shared distance and ratio axes. The
        shells along <100>, <110> and <111> are labelled above the top panel (BCC only).

        Parameters
        ----------
        max_radius : float, optional
            Largest neighbour distance in A. Default None: five lattice parameters.
        panel_size : tuple of float, optional
            Width and height of each panel in inches. Default (8.0, 2.2).

        Returns
        -------
        fig, axs
            The figure and the array of panels, top to bottom.
        """
        import matplotlib.pyplot as plt

        sc = self.shell_correlations(max_radius=self._default_shell_radius(max_radius))
        K = len(self.species)
        pairs = [(s, t) for s in range(K) for t in range(s, K)]
        fig, axs = plt.subplots(
            len(pairs),
            1,
            sharex=True,
            sharey=True,
            figsize=(panel_size[0], panel_size[1] * len(pairs)),
        )
        axs = np.atleast_1d(axs)
        labelled = self._direction_shells(sc["radius"], sc["shell"])
        for ax, (s, t) in zip(axs, pairs):
            ax.axhline(1, color="0.6", lw=0.8)
            for radius, _ in labelled:
                ax.axvline(radius, color="0.9", lw=0.6, zorder=0)
            ax.plot(sc["radius"], sc["ratio"][:, s, t], "o-", ms=3, lw=1.2, color="tab:blue")
            ax.set_ylabel(f"{self.species[s]}-{self.species[t]}\n/ random")
        axs[-1].set_xlabel("neighbour distance (A)")
        axs[-1].set_xlim(0, sc["radius"][-1] * 1.02)
        fig.tight_layout()
        self._label_direction_shells(fig, axs[0], labelled, sc["radius"][-1] * 1.02)
        return fig, axs

    def plot_diffuse_sections(self, extent: float = 2.0, normals=((0, 0, 1), (1, -1, 0))):
        """Symmetrized diffuse intensity of the supercell in Laue units (1 = random alloy) on
        reciprocal-lattice planes through the origin. Maxima at special points name the order
        (100: B2-type, 1/2 1/2 1/2: D0_3, 2/3 2/3 2/3: omega). The color scale is set away
        from the reciprocal-lattice nodes."""
        import matplotlib.pyplot as plt

        grid = self.diffuse_grid()
        secs = [self.diffuse_section(n, extent, grid=grid) for n in normals]
        between = np.concatenate([sec[d > 0.2] for sec, _, d in secs])
        vmax = np.quantile(between, 0.995)
        fig, axs = plt.subplots(1, len(normals), figsize=(5.2 * len(normals), 4.4))
        axs = np.atleast_1d(axs)
        ext = [-extent, extent, -extent, extent]
        for ax, n, (sec, (u, v), _) in zip(axs, normals, secs):
            im = ax.imshow(
                sec.T, origin="lower", extent=ext, cmap="turbo_black", vmin=0, vmax=vmax
            )
            ax.set_title("(" + "".join(f"{int(x)}" for x in n) + ") section, Laue units")
            ax.set_xlabel("[" + " ".join(f"{x:.2f}" for x in u) + "] (1/a)")
            ax.set_ylabel("[" + " ".join(f"{x:.2f}" for x in v) + "] (1/a)")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        return fig, axs

    def _shells(self, n_shells: int, reach: int = 4):
        """Neighbour offsets of the BCC site lattice (site units, a / 2) grouped by distance,
        complete for every shell within `reach` site units."""
        r = np.arange(-reach, reach + 1)
        v = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
        bcc = np.all(v % 2 == 0, axis=1) | np.all(v % 2 == 1, axis=1)
        v = v[bcc & np.any(v != 0, axis=1)]
        d2 = (v**2).sum(1)
        return [v[d2 == d] for d in np.unique(d2[d2 <= reach**2])[:n_shells]]

    @staticmethod
    def _shell_label(offset) -> str:
        """Bond vector of a BCC neighbour shell in units of a, e.g. "1/2<111>" or "<100>", from
        one offset in site units (a / 2)."""
        v = np.sort(np.abs(np.asarray(offset)))[::-1]
        if np.all(v % 2 == 0):
            return "<" + "".join(str(int(x)) for x in v // 2) + ">"
        return "1/2<" + "".join(str(int(x)) for x in v) + ">"

    def displacement_correlations(self, n_shells: int = 6) -> dict:
        """Displacement short-range order per neighbour shell.

        ``longitudinal``: <(u_i.r)(u_j.r)> / <(u.r)^2> with r the bond direction;
        ``transverse``: the same for the components normal to the bond. Omega embryos give a
        strong negative longitudinal correlation on the nearest-neighbour <111> bond (the
        collapsing pair moves together).

        Parameters
        ----------
        n_shells : int, optional
            Number of neighbour shells. Default 6.

        Returns
        -------
        dict
            "radius": (n_shells,) shell radii in A; "shell": bond-vector
            labels (e.g. "1/2<111>", "<100>"); "longitudinal" and
            "transverse": lists of correlations, dimensionless.

        Raises
        ------
        NotImplementedError
            For site lattices other than BCC.
        """
        if self.grid_divisor != 2:
            raise NotImplementedError("Displacement correlations are implemented for BCC sites.")
        step = self.lattice_parameter / (self.grid_divisor * self.refine)
        u = self.displacement * step
        nc = self._site_lookup.shape[0]
        xc = self.site_x // self.refine
        out = dict(radius=[], shell=[], longitudinal=[], transverse=[])
        for offs in self._shells(n_shells):
            out["shell"].append(self._shell_label(offs[0]))
            rhat = offs / np.linalg.norm(offs, axis=1, keepdims=True)
            nb = np.stack([self._site_lookup[tuple(np.mod(xc + o, nc).T)] for o in offs], 1)
            ui = u[:, None, :]
            uj = u[nb]
            li = (ui * rhat[None]).sum(-1)
            lj = (uj * rhat[None]).sum(-1)
            ti = ui - li[..., None] * rhat[None]
            tj = uj - lj[..., None] * rhat[None]
            ll = (li**2).mean()
            tt = (ti**2).sum(-1).mean()
            out["radius"].append(np.linalg.norm(offs[0]) * self.lattice_parameter / 2)
            out["longitudinal"].append(float((li * lj).mean() / ll) if ll > 0 else 0.0)
            out["transverse"].append(float((ti * tj).sum(-1).mean() / tt) if tt > 0 else 0.0)
        out["radius"] = np.asarray(out["radius"])
        return out

    def plot_displacement_correlations(self, n_shells: int = 8):
        """Displacement short-range order: longitudinal and transverse displacement
        correlations for each neighbour shell, labelled by the shell's bond vector in units of
        a. BCC has no shell between a (second neighbours) and a sqrt(2) (third)."""
        import matplotlib.pyplot as plt

        c = self.displacement_correlations(n_shells)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot(c["radius"], c["longitudinal"], "o-", ms=5, label="longitudinal (along bond)")
        ax.plot(c["radius"], c["transverse"], "s--", ms=5, label="transverse")
        ax.set_xticks(c["radius"])
        ax.set_xticklabels(c["shell"], rotation=45, fontsize=8)
        ax.set_xlabel("neighbour shell (bond vector / a)")
        ax.set_ylabel("displacement correlation")
        top = ax.secondary_xaxis("top")
        top.set_xticks(c["radius"])
        top.set_xticklabels([f"{r:.2f}" for r in c["radius"]], fontsize=7)
        top.set_xlabel("distance (A)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig, ax

    def displacement_distributions(self) -> dict:
        """Probability of each projected displacement u.n (A) per species, pooled over the
        symmetry-equivalent directions n of <100>, <110> and <111> (both senses)."""
        step = self.lattice_parameter / (self.grid_divisor * self.refine)
        families = {
            "<100>": np.eye(3),
            "<110>": np.array(
                [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]]
            ),
            "<111>": np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]),
        }
        out = {}
        for name, dirs in families.items():
            n = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
            proj = self.displacement @ n.T * step  # (n_sites, n_dirs)
            proj = np.concatenate([proj, -proj], axis=1)
            out[name] = {}
            for k, sp in enumerate(self.species):
                v = np.round(proj[self.species_index == k].ravel(), 4)
                vals, counts = np.unique(v, return_counts=True)
                out[name][sp] = (vals, counts / counts.sum())
        return out

    def plot_displacements(self, **kwargs):
        """Probability distribution of the static displacement of each species projected on
        <100>, <110> and <111> (pooled over equivalent directions and both senses), on a log
        scale. Displacements live on a grid, so each curve connects the grid values."""
        import matplotlib.pyplot as plt

        dist = self.displacement_distributions()
        fig, axs = plt.subplots(1, 3, figsize=kwargs.pop("figsize", (14, 3.8)), sharey=True)
        colors = plt.get_cmap("tab10")(np.arange(len(self.species)))
        for ax, (name, per_species) in zip(axs, dist.items()):
            for k, sp in enumerate(self.species):
                vals, prob = per_species[sp]
                ax.plot(vals, prob, "o-", ms=4, lw=1.2, color=colors[k], label=sp)
            ax.set_yscale("log")
            ax.set_xlabel(f"u . n, n along {name} (A)")
            ax.set_title(name)
        axs[0].set_ylabel("probability")
        axs[0].legend()
        summary = self.displacement_summary()
        fig.suptitle(
            f"static B = {summary['static_b']:.3f} A^2; mean |u| "
            + ", ".join(f"{k} {v:.3f} A" for k, v in summary["mean_displacement"].items())
        )
        fig.tight_layout()
        return fig, axs

    def plot_loss(self):
        """Weighted loss after each sweep (``loss_history``) on a log scale.

        Returns
        -------
        tuple
            ``(fig, ax)``.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(self.loss_history, "k.-")
        ax.set_xlabel("sweep")
        ax.set_ylabel("weighted loss")
        ax.set_yscale("log")
        fig.tight_layout()
        return fig, ax


def _lsq_bounded(X: np.ndarray, y: np.ndarray, sw: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Weighted least squares with lower bounds, columns normalized for conditioning."""
    Xw = X * sw[:, None]
    norm = np.linalg.norm(Xw, axis=0)
    norm[norm == 0] = 1.0
    res = optimize.lsq_linear(Xw / norm, y * sw, bounds=(lo * norm, np.inf))
    return res.x / norm


def _disjoint_rows(sites: np.ndarray) -> list[int]:
    """Rows of ``sites`` sharing no site with an earlier kept row."""
    seen, keep = set(), []
    for b, row in enumerate(sites):
        r = row.tolist()
        if seen.isdisjoint(r):
            seen.update(r)
            keep.append(b)
    return keep


def _integrate_spots(im: np.ndarray, positions: np.ndarray, radius: float) -> np.ndarray:
    """Integrated intensity inside ``radius`` px of each position, over the median of the ring
    out to 1.6 radius; NaN for spots off the detector."""
    out = np.full(len(positions), np.nan)
    r_out = int(np.ceil(1.6 * radius)) + 1
    yy, xx = np.mgrid[-r_out : r_out + 1, -r_out : r_out + 1]
    for n, (pr, pc) in enumerate(positions):
        r0, c0 = int(round(pr)), int(round(pc))
        if (
            r0 - r_out < 0
            or c0 - r_out < 0
            or r0 + r_out >= im.shape[0]
            or c0 + r_out >= im.shape[1]
        ):
            continue
        win = im[r0 - r_out : r0 + r_out + 1, c0 - r_out : c0 + r_out + 1]
        dist = np.hypot(yy + r0 - pr, xx + c0 - pc)
        bg = np.median(win[(dist >= radius) & (dist < 1.6 * radius)])
        out[n] = (win[dist < radius] - bg).sum()
    return out


def _nan_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that ignores NaNs."""
    ok = np.isfinite(a)
    num = ndimage.gaussian_filter(np.where(ok, a, 0.0), sigma)
    den = ndimage.gaussian_filter(ok.astype(float), sigma)
    return np.where(ok, num / np.maximum(den, 1e-12), np.nan)
