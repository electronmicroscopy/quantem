"""Crystal structures and kinematical diffraction for orientation mapping.

A Crystal wraps an ase.Atoms object and computes the reciprocal lattice,
kinematical structure factors, symmetry operators (via spglib), and simulated
diffraction patterns for arbitrary orientations. All numerical state is stored
as torch tensors (float64) so downstream matching and refinement can run on
GPU and differentiate through the calculation.

Conventions
-----------
- Real lattice vectors are rows of `lat_real` (Angstroms).
- Reciprocal lattice vectors are rows of `lat_recip` (1/Angstroms, no 2*pi).
- Structure factors follow F_hkl = (1/V) * sum_n f_n * exp(-2*pi*i * hkl.p_n),
  so intensities have units of scattering amplitude per unit volume.
- Orientations are unit quaternions rotating crystal Cartesian vectors into
  the lab frame (see quantem.diffraction.rotations).
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols

from quantem.core.utils.utils import electron_wavelength_angstrom
from quantem.diffraction.rotations import qrotate, symmetry_quaternions

# Zone-axis fundamental wedge corners (Cartesian) for each Laue class, with
# display labels for the IPF legend. Hexagonal / trigonal labels use 4-index
# Miller-Bravais direction symbols. Any Laue class not listed falls back to
# hemisphere sampling, which is always sufficient (all Laue classes contain
# inversion) but redundant.
_SQRT3_2 = np.sqrt(3) / 2
LAUE_WEDGES: dict[str, list[list[float]]] = {
    "m-3m": [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
    "m-3": [[0, 0, 1], [1, 0, 0], [1, 1, 1]],
    "6/mmm": [[0, 0, 1], [_SQRT3_2, 0.5, 0], [1, 0, 0]],
    "6/m": [[0, 0, 1], [1, 0, 0], [0.5, _SQRT3_2, 0]],
    "-3m": [[0, 0, 1], [1, 0, 0], [0.5, _SQRT3_2, 0]],
    "4/mmm": [[0, 0, 1], [1, 0, 0], [1, 1, 0]],
    "4/m": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    "mmm": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
}
LAUE_WEDGE_LABELS: dict[str, list[str]] = {
    "m-3m": ["[001]", "[011]", "[111]"],
    "m-3": ["[001]", "[100]", "[111]"],
    "6/mmm": ["[0001]", "[10$\\bar{1}$0]", "[2$\\bar{1}\\bar{1}$0]"],
    "6/m": ["[0001]", "[2$\\bar{1}\\bar{1}$0]", "[11$\\bar{2}$0]"],
    "-3m": ["[0001]", "[2$\\bar{1}\\bar{1}$0]", "[11$\\bar{2}$0]"],
    "4/mmm": ["[001]", "[100]", "[110]"],
    "4/m": ["[001]", "[100]", "[010]"],
    "mmm": ["[001]", "[100]", "[010]"],
}
# plain-text (unicode combining-overline) forms for terminal printing
_B = "\u0305"  # combining overline, applies to the preceding character
LAUE_WEDGE_LABELS_TEXT: dict[str, list[str]] = {
    "m-3m": ["[001]", "[011]", "[111]"],
    "m-3": ["[001]", "[100]", "[111]"],
    "6/mmm": ["[0001]", f"[101{_B}0]", f"[21{_B}1{_B}0]"],
    "6/m": ["[0001]", f"[21{_B}1{_B}0]", f"[112{_B}0]"],
    "-3m": ["[0001]", f"[21{_B}1{_B}0]", f"[112{_B}0]"],
    "4/mmm": ["[001]", "[100]", "[110]"],
    "4/m": ["[001]", "[100]", "[010]"],
    "mmm": ["[001]", "[100]", "[010]"],
}


def miller_to_miller_bravais(uvw: np.ndarray) -> np.ndarray:
    """Convert 3-index [u'v'w'] direction indices to 4-index [u v t w].

    u = (2u' - v') / 3, v = (2v' - u') / 3, t = -(u + v), w = w', cleared to
    the smallest integer form.
    """
    uvw = np.atleast_2d(np.asarray(uvw, dtype=float))
    u = (2 * uvw[:, 0] - uvw[:, 1]) / 3
    v = (2 * uvw[:, 1] - uvw[:, 0]) / 3
    out = np.stack([u, v, -(u + v), uvw[:, 2]], axis=1)
    # clear fractions and common factors
    out = out * 3
    gcd = np.gcd.reduce(np.abs(np.round(out)).astype(int), axis=1)
    gcd[gcd == 0] = 1
    out = out / gcd[:, None]
    return out.astype(int).squeeze()


def miller_bravais_to_miller(uvtw: np.ndarray) -> np.ndarray:
    """Convert 4-index [u v t w] direction indices to 3-index [u'v'w'].

    u' = 2u + v, v' = 2v + u, w' = w (t is redundant: t = -(u + v)).
    """
    uvtw = np.atleast_2d(np.asarray(uvtw, dtype=float))
    out = np.stack(
        [2 * uvtw[:, 0] + uvtw[:, 1], 2 * uvtw[:, 1] + uvtw[:, 0], uvtw[:, 3]], axis=1
    )
    gcd = np.gcd.reduce(np.abs(np.round(out)).astype(int), axis=1)
    gcd[gcd == 0] = 1
    return (out / gcd[:, None]).astype(int).squeeze()

# point group -> Laue class
_LAUE_CLASS = {
    "1": "-1", "-1": "-1",
    "2": "2/m", "m": "2/m", "2/m": "2/m",
    "222": "mmm", "mm2": "mmm", "mmm": "mmm",
    "4": "4/m", "-4": "4/m", "4/m": "4/m",
    "422": "4/mmm", "4mm": "4/mmm", "-42m": "4/mmm", "4/mmm": "4/mmm",
    "3": "-3", "-3": "-3",
    "32": "-3m", "3m": "-3m", "-3m": "-3m",
    "6": "6/m", "-6": "6/m", "6/m": "6/m",
    "622": "6/mmm", "6mm": "6/mmm", "-6m2": "6/mmm", "6/mmm": "6/mmm",
    "23": "m-3", "m-3": "m-3",
    "432": "m-3m", "-43m": "m-3m", "m-3m": "m-3m",
}


def _load_lobato_params() -> dict[str, np.ndarray]:
    with resources.files("quantem.diffraction").joinpath("data/lobato.json").open() as f:
        raw = json.load(f)
    return {sym: np.array(p) for sym, p in raw.items()}


_LOBATO: dict[str, np.ndarray] | None = None


def electron_scattering_factor(numbers: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Lobato & Van Dyck (2014) electron scattering factors.

    Parameters
    ----------
    numbers : torch.Tensor
        Atomic numbers (N,).
    g : torch.Tensor
        Scattering vector magnitudes (M,) in 1/Angstroms.

    Returns
    -------
    torch.Tensor
        f_e(g) of shape (N, M) in Angstroms.
    """
    global _LOBATO
    if _LOBATO is None:
        _LOBATO = _load_lobato_params()
    g2 = (g**2)[None, :, None]  # (1, M, 5)
    a = torch.stack(
        [
            torch.as_tensor(_LOBATO[chemical_symbols[int(z)]][0], dtype=g.dtype, device=g.device)
            for z in numbers
        ]
    )[:, None, :]  # (N, 1, 5)
    b = torch.stack(
        [
            torch.as_tensor(_LOBATO[chemical_symbols[int(z)]][1], dtype=g.dtype, device=g.device)
            for z in numbers
        ]
    )[:, None, :]
    return (a * (2.0 + b * g2) / (1.0 + b * g2) ** 2).sum(dim=-1)


class Crystal:
    """A crystal structure with kinematical diffraction methods.

    Build with `from_ase` or `from_cif`, then call
    `calculate_structure_factors` before generating patterns or orientation
    plans.
    """

    def __init__(
        self,
        atoms: Atoms,
        name: str | None = None,
        symprec: float = 1e-4,
        pseudo_symmetry_tol: float | None = None,
        verbose: bool = True,
    ):
        self.atoms = atoms
        self.name = name if name is not None else atoms.get_chemical_formula()
        self._pseudo_symmetry_tol = pseudo_symmetry_tol

        self.lat_real = torch.as_tensor(atoms.cell[:], dtype=torch.float64)
        self.positions_frac = torch.as_tensor(
            atoms.get_scaled_positions(), dtype=torch.float64
        )
        self.numbers = torch.as_tensor(atoms.numbers, dtype=torch.long)
        occupancy = atoms.arrays.get("occupancy", np.ones(len(atoms)))
        self.occupancy = torch.as_tensor(np.asarray(occupancy, dtype=float))

        self._setup_symmetry(symprec, pseudo_symmetry_tol)
        if verbose:
            print(self.symmetry_summary())

        # populated by calculate_structure_factors
        self.k_max: float | None = None
        self.hkl: torch.Tensor | None = None
        self.g_vec: torch.Tensor | None = None
        self.g_len: torch.Tensor | None = None
        self.struct_factors: torch.Tensor | None = None
        self.struct_factors_int: torch.Tensor | None = None

    @classmethod
    def from_ase(cls, atoms: Atoms, name: str | None = None, **kwargs) -> "Crystal":
        return cls(atoms, name=name, **kwargs)

    @classmethod
    def from_cif(cls, file_path: str | Path, name: str | None = None, **kwargs) -> "Crystal":
        from ase.io import read

        atoms = read(file_path)
        assert isinstance(atoms, Atoms)
        return cls(atoms, name=name, **kwargs)

    @property
    def volume(self) -> float:
        return float(torch.abs(torch.linalg.det(self.lat_real)))

    @property
    def lat_recip(self) -> torch.Tensor:
        """Reciprocal lattice vectors as rows, no 2*pi factor."""
        return torch.linalg.inv(self.lat_real).T

    def _setup_symmetry(self, symprec: float, pseudo_symmetry_tol: float | None) -> None:
        """Detect the true symmetry group, and optionally a pseudo-symmetry group.

        The true group (at `symprec`) is stored for reporting and refinement.
        When `pseudo_symmetry_tol` is set, the symmetry is re-detected at that
        looser tolerance: nearly-degenerate cells (e.g. an orthorhombic cell
        with a = 4.000, b = 4.001, c = 4.002 Angstroms) are idealized to their
        higher-symmetry parent, and *matching* uses that group --- orientations
        that no experiment could distinguish are never sampled separately.
        """
        import spglib

        cell = (
            self.lat_real.numpy(),
            self.positions_frac.numpy(),
            self.numbers.numpy(),
        )
        dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        self.spacegroup: str = f"{dataset.international} ({dataset.number})"
        pg = spglib.get_pointgroup(dataset.rotations)[0].strip()
        self.pointgroup: str = pg
        self.laue_group: str = _LAUE_CLASS.get(pg, "-1")
        self.sym_quats = symmetry_quaternions(dataset.rotations, self.lat_real.numpy())

        if pseudo_symmetry_tol is not None and pseudo_symmetry_tol > symprec:
            ds_pseudo = spglib.get_symmetry_dataset(cell, symprec=pseudo_symmetry_tol)
            pg_pseudo = spglib.get_pointgroup(ds_pseudo.rotations)[0].strip()
            self.pointgroup_matching: str = pg_pseudo
            self.laue_group_matching: str = _LAUE_CLASS.get(pg_pseudo, "-1")
            self.sym_quats_matching = symmetry_quaternions(
                ds_pseudo.rotations, self.lat_real.numpy()
            )
        else:
            self.pointgroup_matching = pg
            self.laue_group_matching = self.laue_group
            self.sym_quats_matching = self.sym_quats

    def zone_axis_wedge(self) -> torch.Tensor | None:
        """Fundamental zone-axis wedge corners (3, 3) Cartesian, or None.

        None means the Laue class has no simple 3-corner wedge and the
        orientation plan should sample the full hemisphere.
        """
        corners = LAUE_WEDGES.get(self.laue_group)
        if corners is None:
            return None
        c = torch.tensor(corners, dtype=torch.float64)
        return c / torch.linalg.norm(c, dim=-1, keepdim=True)

    def zone_axis_wedge_labels(self) -> list[str] | None:
        """Direction labels of the wedge corners (4-index for hex/trigonal)."""
        return LAUE_WEDGE_LABELS.get(self.laue_group)

    def symmetry_summary(self) -> str:
        """Human-readable symmetry report, including any pseudo-symmetry."""
        import re

        # subscript the space group screw/glide digits: P6_3/mmc -> P6[sub3]/mmc
        subs = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        sg = re.sub(r"_(\d)", lambda m: m.group(1).translate(subs), self.spacegroup)
        lines = [
            f"{self.name}",
            f"  space group      {sg}",
            f"  point group      {self.pointgroup}   (Laue class {self.laue_group})",
        ]
        if self.pointgroup_matching != self.pointgroup:
            lines += [
                f"  pseudo-symmetry  {self.pointgroup_matching} "
                f"(Laue class {self.laue_group_matching}) "
                "-- used for orientation matching",
            ]
        elif self._pseudo_symmetry_tol is not None:
            lines += [
                "  pseudo-symmetry  none found at tol = "
                f"{self._pseudo_symmetry_tol:g} A",
            ]
        else:
            lines += ["  pseudo-symmetry  not checked (set pseudo_symmetry_tol)"]
        # matching line reflects the symmetry actually used, after any
        # pseudo-symmetry reduction
        labels = LAUE_WEDGE_LABELS_TEXT.get(self.laue_group_matching)
        wedge_txt = (
            f"zone axis wedge {labels[0]}, {labels[1]}, {labels[2]}"
            if labels is not None
            else "full hemisphere"
        )
        lines += [
            f"  matching         {self.sym_quats_matching.shape[0]} proper "
            f"rotations, {wedge_txt}"
        ]
        return "\n".join(lines)

    def calculate_structure_factors(
        self,
        k_max: float = 1.5,
        tol_structure_factor: float = 1e-4,
        thermal_sigma: float | dict[str, float] | None = None,
    ) -> "Crystal":
        """Kinematical structure factors for all reflections with |g| <= k_max.

        Parameters
        ----------
        k_max : float, default=1.5
            Maximum scattering vector magnitude, 1/Angstroms.
        tol_structure_factor : float, default=1e-4
            Discard reflections with |F| below this threshold.
        thermal_sigma : float | dict[str, float] | None
            RMS thermal displacement (Angstroms), scalar or per-element,
            applied as a Debye-Waller factor.

        Returns
        -------
        Crystal
            self, for chaining.
        """
        self.k_max = float(k_max)
        recip = self.lat_recip

        # index range: project k_max onto each reciprocal cell direction
        k_len = torch.linalg.norm(recip, dim=1)
        n_max = torch.ceil(k_max / k_len * 2).to(torch.long)
        ranges = [torch.arange(-int(n), int(n) + 1) for n in n_max]
        hkl = torch.cartesian_prod(*ranges).to(torch.float64)
        g_vec = hkl @ recip
        g_len = torch.linalg.norm(g_vec, dim=1)
        keep = (g_len <= k_max) & (g_len > 0)
        hkl, g_vec, g_len = hkl[keep], g_vec[keep], g_len[keep]

        f_e = electron_scattering_factor(self.numbers, g_len)  # (N_atoms, N_g)

        if thermal_sigma is not None:
            if isinstance(thermal_sigma, dict):
                sigma = torch.tensor(
                    [thermal_sigma[chemical_symbols[int(z)]] for z in self.numbers],
                    dtype=torch.float64,
                )
            else:
                sigma = torch.full((len(self.numbers),), float(thermal_sigma))
            dwf = torch.exp(-0.5 * (2 * np.pi * sigma[:, None] * g_len[None, :]) ** 2)
            f_e = f_e * dwf

        phase = torch.exp(-2j * np.pi * (self.positions_frac @ hkl.T))  # (N_atoms, N_g)
        F = (f_e * self.occupancy[:, None] * phase).sum(dim=0) / self.volume

        keep = torch.abs(F) > tol_structure_factor
        self.hkl = hkl[keep].to(torch.long)
        self.g_vec = g_vec[keep]
        self.g_len = g_len[keep]
        self.struct_factors = F[keep]
        self.struct_factors_int = torch.abs(F[keep]) ** 2
        return self

    def calculate_dynamical_structure_factors(
        self,
        energy_ev: float,
        thermal_sigma: float | dict[str, float] = 0.05,
        k_max: float | None = None,
        include_core: bool = True,
        include_phonon: bool = True,
    ) -> "Crystal":
        """Absorptive structure factors for Bloch wave calculations.

        Uses the Weickenmeier-Kohl parameterization (Acta Cryst. A47, 590
        (1991)): the elastic part is Debye-Waller damped, and the imaginary
        (absorptive) part includes core-loss and phonon/TDS contributions.
        The returned factors are relativistically corrected and already carry
        the 1/pi convention of the Bloch structure matrix, i.e. they are the
        U_g of De Graef ch. 5 after division by the unit cell volume.

        All reflections up to k_max are kept, including kinematically
        forbidden ones (their U_g can be nonzero through absorption and they
        are required as coupling vectors g - h).

        Parameters
        ----------
        energy_ev : float
            Beam energy in eV.
        thermal_sigma : float | dict[str, float], default=0.05
            RMS thermal displacement (Angstroms), scalar or per-element.
        k_max : float | None
            Maximum |g| of stored factors; defaults to the kinematical k_max.
            For Bloch calculations with beams out to k, this should be 2k so
            every coupling vector is covered.
        """
        from quantem.diffraction.wk_scattering_factors import compute_WK_factor

        if k_max is None:
            if self.k_max is None:
                raise RuntimeError("Provide k_max or run calculate_structure_factors.")
            k_max = self.k_max
        recip = self.lat_recip
        k_len = torch.linalg.norm(recip, dim=1)
        n_max = torch.ceil(k_max / k_len * 2).to(torch.long)
        ranges = [torch.arange(-int(n), int(n) + 1) for n in n_max]
        hkl = torch.cartesian_prod(*ranges).to(torch.float64)
        g_vec = hkl @ recip
        g_len = torch.linalg.norm(g_vec, dim=1)
        keep = g_len <= k_max
        hkl, g_len = hkl[keep], g_len[keep]

        g_np = g_len.numpy()
        if isinstance(thermal_sigma, dict):
            sigma_per_atom = np.array(
                [thermal_sigma[chemical_symbols[int(z)]] for z in self.numbers]
            )
        else:
            sigma_per_atom = np.full(len(self.numbers), float(thermal_sigma))

        # one WK evaluation per unique (Z, sigma) pair
        f_atoms = np.zeros((len(self.numbers), g_np.size), dtype=np.complex128)
        cache: dict[tuple[int, float], np.ndarray] = {}
        for i, (z, sig) in enumerate(zip(self.numbers.tolist(), sigma_per_atom)):
            key = (int(z), float(sig))
            if key not in cache:
                cache[key] = compute_WK_factor(
                    g_np,
                    int(z),
                    energy_ev,
                    thermal_sigma=float(sig),
                    include_core=include_core,
                    include_phonon=include_phonon,
                )
            f_atoms[i] = cache[key]

        phase = np.exp(-2j * np.pi * (self.positions_frac.numpy() @ hkl.numpy().T))
        occ = self.occupancy.numpy()[:, None]
        U = (f_atoms * occ * phase).sum(axis=0) / self.volume

        self.hkl_dyn = hkl.to(torch.long)
        self.g_len_dyn = g_len
        self.U_dyn = torch.as_tensor(U, dtype=torch.complex128)
        self.dyn_energy_ev = float(energy_ev)
        return self

    def generate_pattern(
        self,
        orientation: torch.Tensor,
        energy_ev: float = 300e3,
        sigma_excitation: float = 0.02,
        tol_excitation_mult: float = 3.0,
        k_max: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Kinematical diffraction pattern for one orientation.

        Parameters
        ----------
        orientation : torch.Tensor
            Unit quaternion (4,) rotating crystal vectors into the lab frame.
        energy_ev : float, default=300e3
            Beam energy in eV.
        sigma_excitation : float, default=0.02
            Excitation error tolerance (1/Angstroms) in the shape-factor
            envelope exp(-s_g^2 / 2 sigma^2).
        tol_excitation_mult : float, default=3.0
            Include reflections with |s_g| below this multiple of sigma.
        k_max : float | None
            Optionally trim the pattern below the structure-factor k_max.

        Returns
        -------
        dict with 'qx', 'qy', 'intensity', 'hkl', 's_g' tensors.
        """
        if self.g_vec is None:
            raise RuntimeError("Run calculate_structure_factors first.")
        lam = electron_wavelength_angstrom(energy_ev)
        g = qrotate(orientation, self.g_vec)
        gz, g2 = g[:, 2], (g**2).sum(dim=1)
        s_g = (2 * gz - lam * g2) / (2 - 2 * lam * gz)
        keep = torch.abs(s_g) < sigma_excitation * tol_excitation_mult
        if k_max is not None:
            keep &= self.g_len <= k_max
        intensity = self.struct_factors_int[keep] * torch.exp(
            -(s_g[keep] ** 2) / (2 * sigma_excitation**2)
        )
        return {
            "qx": g[keep, 0],
            "qy": g[keep, 1],
            "intensity": intensity,
            "hkl": self.hkl[keep],
            "s_g": s_g[keep],
        }

    def __repr__(self) -> str:
        return (
            f"Crystal({self.name}, {len(self.numbers)} atoms, "
            f"spacegroup {self.spacegroup}, pointgroup {self.pointgroup})"
        )
