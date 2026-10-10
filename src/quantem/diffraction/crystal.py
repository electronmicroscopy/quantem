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
import warnings
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.data import chemical_symbols

from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.defaults import SIGMA_EXCITATION
from quantem.diffraction.rotations import qrotate, symmetry_quaternions

# unicode combining overline, applies to the preceding character
_B = "\u0305"

_EXCITATION_MODELS = ("gaussian", "slab")


@contextmanager
def _spglib_raises():
    """Within the block, spglib raises SpglibError on failure.

    spglib 2.7 reports failures by returning None and emitting a
    DeprecationWarning on every call, success or not, unless
    ``spglib.error.OLD_ERROR_HANDLING`` is False, which spglib 2.8 makes the
    default. Every spglib call here is wrapped in try/except, so the flag is
    switched for the block only and restored afterwards, leaving other
    spglib users in the process unaffected. Older spglib versions without
    the flag are left as they are.
    """
    import spglib

    err = getattr(spglib, "error", None)
    if err is None or not hasattr(err, "OLD_ERROR_HANDLING"):
        yield
        return
    old = err.OLD_ERROR_HANDLING
    err.OLD_ERROR_HANDLING = False
    try:
        yield
    finally:
        err.OLD_ERROR_HANDLING = old


def direction_indices(
    lat_real: torch.Tensor | np.ndarray, d, max_multiple: int = 12, atol: float = 2e-3
) -> np.ndarray | None:
    """Smallest integer [uvw] along a Cartesian direction.

    Parameters
    ----------
    lat_real : torch.Tensor | np.ndarray
        Real-space lattice vectors as rows (3, 3), Angstroms.
    d : array-like
        Cartesian direction (3,) in the crystal frame; need not be
        normalized.
    max_multiple : int, default=12
        Largest multiplier tried to make the indices integer.
    atol : float, default=2e-3
        Allowed deviation of the scaled indices from integers. Loosen it to
        index the axes of a pseudo-symmetry, which are lattice directions of
        an ideal parent but only nearly so in the real cell.

    Returns
    -------
    np.ndarray | None
        Integer [uvw] (3,) with no common factor, or None if `d` is not a
        lattice direction with indices up to `max_multiple`.
    """
    A_T_inv = np.linalg.inv(np.asarray(lat_real, dtype=float).T)
    v = A_T_inv @ np.asarray(d, dtype=float)
    v = v / np.abs(v).max()
    for m in range(1, max_multiple + 1):
        w = v * m
        if np.allclose(w, np.round(w), atol=atol):
            ints = np.round(w).astype(int)
            g = np.gcd.reduce(np.abs(ints))
            return ints // max(g, 1)
    return None


def format_direction(uvw, hexagonal: bool = False, mathtext: bool = True) -> str:
    """Direction label such as [011] or [10-10], with overlines on negatives.

    Parameters
    ----------
    uvw : array-like | None
        Integer 3-index direction [uvw].
    hexagonal : bool, default=False
        Write the 4-index [UVTW] symbol instead, see
        :func:`miller_to_miller_bravais`.
    mathtext : bool, default=True
        Overlines as matplotlib mathtext (``$\\bar{1}$``) for figures; False
        uses unicode combining overlines for plain text.

    Returns
    -------
    str
        The label, or an empty string for None.
    """
    if uvw is None:
        return ""
    ks = miller_to_miller_bravais(uvw) if hexagonal else np.asarray(uvw)
    ks = np.atleast_1d(ks)
    if mathtext:
        body = "".join(str(k) if k >= 0 else "$\\bar{%d}$" % -k for k in ks)
    else:
        body = "".join(str(k) if k >= 0 else "%d%s" % (-k, _B) for k in ks)
    return "[" + body + "]"


def miller_to_miller_bravais(uvw: np.ndarray) -> np.ndarray:
    """Convert 3-index [u'v'w'] direction indices to 4-index [u v t w].

    u = (2u' - v') / 3, v = (2v' - u') / 3, t = -(u + v), w = w', cleared to
    the smallest integer form.

    Parameters
    ----------
    uvw : array-like
        Integer 3-index directions, (3,) or (N, 3).

    Returns
    -------
    np.ndarray
        Integer [u v t w], (4,) or (N, 4).
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
    return np.rint(out).astype(int).squeeze()


def miller_bravais_to_miller(uvtw: np.ndarray) -> np.ndarray:
    """Convert 4-index [u v t w] direction indices to 3-index [u'v'w'].

    u' = 2u + v, v' = 2v + u, w' = w (t is redundant: t = -(u + v)),
    cleared to the smallest integer form.

    Parameters
    ----------
    uvtw : array-like
        Integer 4-index directions, (4,) or (N, 4).

    Returns
    -------
    np.ndarray
        Integer [u'v'w'], (3,) or (N, 3).
    """
    uvtw = np.atleast_2d(np.asarray(uvtw, dtype=float))
    out = np.stack([2 * uvtw[:, 0] + uvtw[:, 1], 2 * uvtw[:, 1] + uvtw[:, 0], uvtw[:, 3]], axis=1)
    gcd = np.gcd.reduce(np.abs(np.round(out)).astype(int), axis=1)
    gcd[gcd == 0] = 1
    return np.rint(out / gcd[:, None]).astype(int).squeeze()


# point group -> Laue class
_LAUE_CLASS = {
    "1": "-1",
    "-1": "-1",
    "2": "2/m",
    "m": "2/m",
    "2/m": "2/m",
    "222": "mmm",
    "mm2": "mmm",
    "mmm": "mmm",
    "4": "4/m",
    "-4": "4/m",
    "4/m": "4/m",
    "422": "4/mmm",
    "4mm": "4/mmm",
    "-42m": "4/mmm",
    "4/mmm": "4/mmm",
    "3": "-3",
    "-3": "-3",
    "32": "-3m",
    "3m": "-3m",
    "-3m": "-3m",
    "6": "6/m",
    "-6": "6/m",
    "6/m": "6/m",
    "622": "6/mmm",
    "6mm": "6/mmm",
    "-6m2": "6/mmm",
    "6/mmm": "6/mmm",
    "23": "m-3",
    "m-3": "m-3",
    "432": "m-3m",
    "-43m": "m-3m",
    "m-3m": "m-3m",
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


def _expand_partial_occupancy(atoms: Atoms) -> Atoms:
    """One atom per species on every shared site, with its fractional occupancy.

    ASE's CIF reader keeps the majority species of a mixed site and records
    the full composition in ``atoms.info['occupancy']``, keyed by the site
    index held in ``atoms.arrays['spacegroup_kinds']``. Structures with only
    fully occupied sites are returned unchanged.

    Parameters
    ----------
    atoms : Atoms
        As returned by ``ase.io.read`` on a CIF.

    Returns
    -------
    Atoms
        The expanded structure, with ``arrays['occupancy']`` set.
    """
    occ = atoms.info.get("occupancy")
    kinds = atoms.arrays.get("spacegroup_kinds")
    if not occ or kinds is None:
        return atoms
    sites = [occ.get(str(int(k))) for k in kinds]
    if all(
        s is None or (len(s) == 1 and abs(float(next(iter(s.values()))) - 1.0) < 1e-9)
        for s in sites
    ):
        return atoms

    frac = atoms.get_scaled_positions(wrap=False)
    symbols, positions, occupancy = [], [], []
    for i, site in enumerate(sites):
        if not site:
            symbols.append(atoms[i].symbol)
            positions.append(frac[i])
            occupancy.append(1.0)
            continue
        for element, fraction in site.items():
            symbols.append(element)
            positions.append(frac[i])
            occupancy.append(float(fraction))
    out = Atoms(symbols=symbols, scaled_positions=positions, cell=atoms.cell, pbc=atoms.pbc)
    out.set_array("occupancy", np.asarray(occupancy, dtype=float))
    out.info.update({k: v for k, v in atoms.info.items() if k != "occupancy"})
    return out


class Crystal(AutoSerialize):
    """A crystal structure with kinematical diffraction methods.

    Build with `from_ase` or `from_cif`, then call
    `calculate_structure_factors` before generating patterns or orientation
    plans.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure. Fractional site occupancies are read from
        ``atoms.arrays['occupancy']`` when present (see :meth:`from_cif`).
    name : str | None
        Display name; defaults to the chemical formula.
    symprec : float, default=1e-4
        spglib tolerance (Angstroms) for the cell's own symmetry.
    pseudo_symmetry_tol : float | None, default=0.01
        Dimensionless distance tolerance for the symmetry used in
        orientation matching: a fraction of the shortest lattice vector
        within which atoms and lattice vectors are allowed to deviate from a
        higher-symmetry parent (a 4 A cell with an atom at
        (0.5, 0.5, 0.50001) is body centered at any tolerance above 1e-5).
        Cells within it are matched with the parent group, so variants no
        experiment can separate are never sampled as distinct orientations;
        the library builders warn when the matching group differs from the
        cell's own. None matches with the exact symmetry.
    pseudo_symmetry_intensity_tol : float, default=0.05
        Largest intensity difference allowed between reflections that a
        candidate pseudo-symmetry would make equivalent, as a fraction of
        the strongest reflection's kinematical intensity |F|^2. Each extra
        rotation is applied to every reflection within 2.0 1/A and each
        intensity compared with its image's; if any pair differs by more
        than this fraction, the orientations the rotation relates are
        distinguishable and it is rejected. 0.05 merges only orientations
        whose patterns differ by reflections at 5% of the strongest; 0.4
        also merges orientations told apart only by a reflection at 40%,
        appropriate when that reflection is known to be weak or absent in
        the data (stacking disorder, cation mixing). Candidates are the
        relaxed-position group, the lattice's own holohedry, and the
        holohedries of the parent lattices generated by the strong
        reflections, so superstructure twin variants are tested as well.
        The printout names the reflection pair that decides each candidate.
    verbose : bool, default=True
        Print :meth:`symmetry_summary` after the symmetry analysis.
    """

    def __init__(
        self,
        atoms: Atoms,
        name: str | None = None,
        symprec: float = 1e-4,
        pseudo_symmetry_tol: float | None = 0.01,
        pseudo_symmetry_intensity_tol: float = 0.05,
        verbose: bool = True,
    ):
        self.atoms = atoms
        self.name = name if name is not None else atoms.get_chemical_formula()
        self._pseudo_symmetry_tol = pseudo_symmetry_tol
        self._pseudo_symmetry_intensity_tol = float(pseudo_symmetry_intensity_tol)
        self.pseudo_symmetry_report: dict = {}
        self._wedge_cache: torch.Tensor | None | str = "unset"

        self.lat_real = torch.as_tensor(atoms.cell[:], dtype=torch.float64)
        self.positions_frac = torch.as_tensor(atoms.get_scaled_positions(), dtype=torch.float64)
        self.numbers = torch.as_tensor(atoms.numbers, dtype=torch.long)
        occupancy = atoms.arrays.get("occupancy", np.ones(len(atoms)))
        self.occupancy = torch.as_tensor(np.asarray(occupancy, dtype=float))

        with _spglib_raises():
            self._setup_symmetry(symprec, pseudo_symmetry_tol, pseudo_symmetry_intensity_tol)
        # the summary states any pseudo-symmetry adopted; when it has been
        # shown, the orientation plan does not warn about it again
        self._summary_shown = bool(verbose)
        if verbose:
            print(self.symmetry_summary())

        # populated by calculate_structure_factors
        self.k_max: float | None = None
        self.hkl: torch.Tensor | None = None
        self.g_vec: torch.Tensor | None = None
        self.g_len: torch.Tensor | None = None
        self.struct_factors: torch.Tensor | None = None
        self.struct_factors_int: torch.Tensor | None = None

        # populated by calculate_dynamical_structure_factors
        self.hkl_dyn: torch.Tensor | None = None
        self.g_len_dyn: torch.Tensor | None = None
        self.U_dyn: torch.Tensor | None = None
        self.dyn_energy_ev: float | None = None
        self.dyn_k_max: float | None = None

    @classmethod
    def from_ase(cls, atoms: Atoms, name: str | None = None, **kwargs) -> "Crystal":
        """Build a Crystal from an ase.Atoms object.

        Parameters
        ----------
        atoms : ase.Atoms
            The structure, e.g. from ``ase.build.bulk``.
        name : str, optional
            Display name; defaults to the chemical formula.
        **kwargs
            Passed to the Crystal constructor, e.g. `pseudo_symmetry_tol` or
            `verbose`.

        Returns
        -------
        Crystal
        """
        return cls(atoms, name=name, **kwargs)

    @classmethod
    def from_cif(cls, file_path: str | Path, name: str | None = None, **kwargs) -> "Crystal":
        """Build a Crystal from a CIF file, keeping fractional site occupancies.

        ASE reads a mixed-occupancy site as a single atom of the majority
        species, which silently deletes every minority element: a layered
        oxide with Sb sharing a site with Fe loads with no Sb at all, and Sb
        is by far its strongest scatterer. ASE does record the occupancies it
        discarded, so each shared site is expanded here back into one atom
        per species, each carrying its fraction in
        ``atoms.arrays['occupancy']``, which the structure-factor sum uses.

        Parameters
        ----------
        file_path : str or Path
            Path to the CIF file.
        name : str, optional
            Display name; defaults to the chemical formula.
        **kwargs
            Passed to the Crystal constructor, e.g. `pseudo_symmetry_tol`.

        Returns
        -------
        Crystal
        """
        from ase.io import read

        atoms = read(file_path)
        assert isinstance(atoms, Atoms)
        return cls(_expand_partial_occupancy(atoms), name=name, **kwargs)

    @property
    def volume(self) -> float:
        """Unit cell volume, cubic Angstroms."""
        return float(torch.abs(torch.linalg.det(self.lat_real)))

    @property
    def lat_recip(self) -> torch.Tensor:
        """Reciprocal lattice vectors as rows, no 2*pi factor."""
        return torch.linalg.inv(self.lat_real).T

    def _quick_intensities(self, k_max: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Kinematical |F|^2 of every reflection with |g| <= k_max (hkl, I),
        for the pseudo-symmetry intensity check; no thermal factors."""
        recip = self.lat_recip
        k_len = torch.linalg.norm(recip, dim=1)
        n_max = torch.ceil(k_max / k_len * 2).to(torch.long)
        ranges = [torch.arange(-int(n), int(n) + 1) for n in n_max]
        hkl = torch.cartesian_prod(*ranges).to(torch.float64)
        g_vec = hkl @ recip
        g_len = torch.linalg.norm(g_vec, dim=1)
        keep = (g_len <= k_max) & (g_len > 0)
        hkl, g_len = hkl[keep], g_len[keep]
        f_e = electron_scattering_factor(self.numbers, g_len)
        phase = torch.exp(-2j * np.pi * (self.positions_frac @ hkl.T))
        F = (f_e * self.occupancy[:, None] * phase).sum(dim=0) / self.volume
        return hkl.to(torch.long), torch.abs(F) ** 2

    def _setup_symmetry(
        self, symprec: float, pseudo_symmetry_tol: float | None, intensity_tol: float
    ) -> None:
        """Detect the true symmetry group, and optionally a pseudo-symmetry group.

        The true group (at `symprec`) is stored for reporting and refinement.
        The pseudo-symmetry group is detected at a distance tolerance of
        `pseudo_symmetry_tol` times the shortest lattice vector and kept
        only if its extra operations relate reflections of equal kinematical
        intensity to within `intensity_tol` of the strongest reflection:
        two orientations are merged only when no experiment could tell
        their patterns apart, in position or in intensity. Matching uses
        that group, so nearly-degenerate cells are idealized to their
        higher-symmetry parent.
        """
        import spglib

        cell = (
            self.lat_real.numpy(),
            self.positions_frac.numpy(),
            self.numbers.numpy(),
        )
        try:
            dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
        except Exception:
            dataset = None
        if dataset is None:
            # no symmetry found at all (e.g. overlapping atoms): carry on in P1
            warnings.warn(
                f"{self.name}: spglib found no symmetry at symprec={symprec:g}; "
                "using P1. Check the structure for overlapping atoms.",
                stacklevel=3,
            )
            rotations = np.eye(3, dtype=np.intc)[None]
            self.spacegroup: str = "P1 (1)"
        else:
            rotations = dataset.rotations
            self.spacegroup = f"{dataset.international} ({dataset.number})"
        pg = spglib.get_pointgroup(rotations)[0].strip()
        self.pointgroup: str = pg
        self.laue_group: str = _LAUE_CLASS.get(pg, "-1")
        self.sym_quats = symmetry_quaternions(rotations, self.lat_real.numpy())

        self.pointgroup_matching = pg
        self.laue_group_matching = self.laue_group
        self.sym_quats_matching = self.sym_quats
        if pseudo_symmetry_tol is None:
            return
        a_min = float(torch.linalg.norm(self.lat_real, dim=1).min())
        symprec_pseudo = float(pseudo_symmetry_tol) * a_min
        self.pseudo_symmetry_report = {"distance_A": symprec_pseudo}
        if symprec_pseudo <= symprec:
            return

        # Two independent routes to a higher matching symmetry.
        #
        # "relaxed positions": the group spglib finds when every atom is
        # allowed to move by symprec_pseudo, which catches a cell that is a
        # slightly distorted child of a higher-symmetry parent.
        #
        # "lattice": the point group of the lattice alone, ignoring the
        # basis. A structure whose symmetry is broken only by weakly
        # scattering atoms (lithium and oxygen against a transition metal)
        # or by a faint superstructure sits exactly here: no relaxation of
        # the positions recovers the parent, because the atoms are already
        # where they belong, but the diffraction still has the symmetry of
        # the heavy sublattice. Both candidate sets are filtered by the same
        # intensity test, so an operation is adopted only when it leaves the
        # kinematical pattern unchanged.
        #
        # "parent lattice": the lattice generated by the strong reflections
        # alone. A superstructure (cation ordering on a rocksalt or layered
        # frame) has a larger cell than its parent, and the parent's
        # symmetries map the superstructure onto a twin of itself rather than
        # onto itself, so neither route above can see them. When the
        # superstructure reflections are weak the twins give the same
        # pattern, and these are exactly the variants matching must merge.
        lat = self.lat_real.numpy()
        candidates: list[tuple[str, np.ndarray, np.ndarray]] = []
        try:
            ds_relaxed = spglib.get_symmetry_dataset(cell, symprec=symprec_pseudo)
        except Exception:
            ds_relaxed = None
        if ds_relaxed is not None:
            candidates.append(("relaxed positions", ds_relaxed.rotations, lat))
        lattice_cell = (lat, np.zeros((1, 3)), np.ones(1, dtype=int))
        try:
            ds_lattice = spglib.get_symmetry_dataset(lattice_cell, symprec=symprec_pseudo)
        except Exception:
            ds_lattice = None
        if ds_lattice is not None:
            candidates.append(("lattice", ds_lattice.rotations, lat))
        candidates += self._parent_lattice_candidates(pseudo_symmetry_tol, intensity_tol)

        best = None
        closest = None  # the rejected candidate that came nearest to passing
        for route, rotations, lattice in candidates:
            quats = symmetry_quaternions(rotations, lattice)
            if quats.shape[0] <= self.sym_quats.shape[0]:
                continue
            pg_cand = spglib.get_pointgroup(rotations)[0].strip()
            accepted, worst = self._intensity_preserving_subgroup(quats, intensity_tol)
            if accepted is None:
                if closest is None or worst < closest[0]:
                    closest = (worst, pg_cand, route, self._breaking_reflection)
                continue
            if best is None or accepted.shape[0] > best[1].shape[0]:
                best = (route, accepted, pg_cand, quats.shape[0])

        if best is None:
            if closest is not None:
                worst, pg_cand, route, pair = closest
                self.pseudo_symmetry_report.update(
                    candidate=pg_cand,
                    intensity_mismatch=worst,
                    broken_by=pair,
                    route=route,
                    rejected=True,
                )
            return
        route, accepted, pg_cand, n_cand = best
        # the largest difference among the rotations actually adopted
        _, worst = self._intensity_preserving_subgroup(accepted, 1.0)
        self.pseudo_symmetry_report.update(
            candidate=pg_cand,
            intensity_mismatch=worst,
            broken_by=self._breaking_reflection,
            route=route,
            rejected=False,
        )
        self.sym_quats_matching = accepted
        # name the accepted group by the candidate symbol when every one of
        # its rotations survived the intensity test, otherwise by its size
        if accepted.shape[0] == n_cand:
            self.pointgroup_matching = pg_cand
            self.laue_group_matching = _LAUE_CLASS.get(pg_cand, self.laue_group)
        else:
            self.pointgroup_matching = f"{accepted.shape[0]} rotations"
            self.laue_group_matching = self.laue_group

    def _parent_lattice_candidates(
        self, pseudo_symmetry_tol: float, intensity_tol: float
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Holohedries of the lattices generated by the strong reflections.

        For each intensity cut, the parent translations are the fractions t
        of the cell with h.t integer for every reflection h stronger than
        the cut; they form the real-space lattice dual to the strong
        reflections. Its holohedry is a candidate group. Cuts run from very
        low (the cell's own lattice once centring is removed) up to the
        intensity tolerance, since reflections weaker than that are allowed
        to break the pseudo-symmetry anyway.

        Returns
        -------
        list of tuple
            ``(route, rotations, lattice)``: integer rotations in the basis of
            ``lattice``, whose rows are the parent vectors in the crystal's
            own Cartesian frame.
        """
        import itertools

        import spglib

        hkl, inten = self._quick_intensities()
        if inten.numel() == 0 or float(inten.max()) <= 0:
            return []
        inten = inten / inten.max()
        lat = self.lat_real.numpy()
        vol = abs(float(np.linalg.det(lat)))
        # denominators 1, 2, 3, 4, 6, 12 cover the supercells met in practice
        n_grid = 12
        grid = np.array(list(itertools.product(range(n_grid), repeat=3)), dtype=float) / n_grid

        out: list[tuple[str, np.ndarray, np.ndarray]] = []
        seen: set[int] = set()
        cuts = sorted({0.02, 0.05, 0.1, 0.2, 0.3, 0.5, float(intensity_tol)})
        for cut in cuts:
            if cut > 1.0:
                continue
            strong = hkl[inten > cut].numpy().astype(float)
            if strong.shape[0] < 3 or np.linalg.matrix_rank(strong) < 3:
                continue
            phase = strong @ grid.T
            t = grid[np.all(np.abs(phase - np.round(phase)) < 1e-6, axis=0)]
            if t.shape[0] < 2:
                continue  # the cell is its own parent: nothing new
            try:
                parent = spglib.standardize_cell(
                    (lat, t, np.ones(t.shape[0], dtype=int)),
                    to_primitive=True,
                    no_idealize=True,
                    symprec=1e-5,
                )
            except Exception:
                parent = None
            if parent is None:
                continue
            L = np.asarray(parent[0], dtype=float)
            ratio = int(round(vol / abs(float(np.linalg.det(L)))))
            if ratio in seen:
                continue
            seen.add(ratio)
            a_min = float(np.linalg.norm(L, axis=1).min())
            try:
                ds = spglib.get_symmetry_dataset(
                    (L, np.zeros((1, 3)), np.ones(1, dtype=int)),
                    symprec=float(pseudo_symmetry_tol) * a_min,
                )
            except Exception:
                ds = None
            if ds is not None:
                out.append((f"parent lattice, {ratio}x smaller cell", ds.rotations, L))
        return out

    def _intensity_preserving_subgroup(
        self, quats: torch.Tensor, intensity_tol: float
    ) -> tuple[torch.Tensor | None, float]:
        """Largest subgroup of `quats` that leaves the kinematical intensities
        invariant, or None when nothing beyond the true symmetry survives.

        Every candidate operation is applied to the reflection list and the
        intensity of each reflection compared with the intensity of its
        image, relative to the strongest reflection. Operations that pass
        are kept; the survivors are then closed under composition (dropping
        the worst offender until they are), because a set of operations that
        is not a group cannot be used to fold orientations.

        Returns the accepted quaternions and the worst mismatch among the
        operations that were tested.
        """
        from quantem.diffraction.rotations import quat_to_matrix

        hkl, inten = self._quick_intensities()
        lut = {tuple(h): i for i, h in enumerate(hkl.tolist())}
        g = hkl.to(torch.float64) @ self.lat_recip
        i_max = float(inten.max())
        Rs = quat_to_matrix(quats)
        Rs_true = quat_to_matrix(self.sym_quats)

        # a pseudo-symmetry group comes from a lattice that is only nearly
        # ideal, so its rotations and their products agree to the distortion
        # (~1e-2 here, ~1e-6 for a hexagonal cell given to 5 decimals).
        # Distinct crystallographic rotations are at least 60 degrees apart,
        # with matrix entries differing by ~0.5, so 0.05 is unambiguous.
        match_tol = 0.05

        def is_true(R):
            return any(float((R - Rt).abs().max()) < match_tol for Rt in Rs_true)

        mismatch = torch.zeros(quats.shape[0], dtype=torch.float64)
        worst = 0.0
        self._breaking_reflection = None
        # a pseudo-symmetry holds only approximately in the metric too, so an
        # image lands near, not on, the reflection it maps onto: snap it to
        # the nearest one within the same fractional tolerance allowed for
        # the atom positions, and treat anything farther as absent
        snap = max(float(self._pseudo_symmetry_tol or 0.0), 1e-3)
        g_len = torch.linalg.norm(g, dim=1)
        for i, R in enumerate(Rs):
            if is_true(R):
                continue
            g_img = g @ R.T
            hkl_img = torch.round(g_img @ self.lat_real.T).to(torch.long)
            idx = torch.tensor([lut.get(tuple(h), -1) for h in hkl_img.tolist()])
            ok = idx >= 0
            if not bool(ok.any()):
                continue
            near = torch.linalg.norm(g_img - g[idx.clamp(min=0)], dim=1) <= snap * g_len + 1e-9
            i_img = torch.where(near, inten[idx.clamp(min=0)], torch.zeros_like(inten))
            diff = (inten[ok] - i_img[ok]).abs()
            m = float(diff.max()) / i_max
            mismatch[i] = m
            if m > worst:
                # the reflection pair responsible, reported so the user can
                # judge whether the data actually resolve it
                j = int(torch.argmax(diff))
                src = torch.nonzero(ok).squeeze(1)[j]
                self._breaking_reflection = (
                    tuple(int(v) for v in hkl[src].tolist()),
                    float(inten[src]) / i_max,
                    tuple(int(v) for v in hkl[idx[src]].tolist()),
                    float(i_img[src]) / i_max,
                )
            worst = max(worst, m)

        keep = mismatch <= intensity_tol
        # close under composition: a product of kept operations must also be
        # kept, or the set is not a group
        for _ in range(quats.shape[0]):
            idx = torch.nonzero(keep).squeeze(1)
            if idx.numel() <= self.sym_quats.shape[0]:
                return None, worst
            R_keep = Rs[idx]
            prod = torch.einsum("aij,bjk->abik", R_keep, R_keep).reshape(-1, 3, 3)
            d = (prod[:, None] - R_keep[None]).abs().amax(dim=(-1, -2))
            closed = bool((d.min(dim=1).values < match_tol).all())
            if closed:
                return quats[idx], worst
            drop = idx[int(torch.argmax(mismatch[idx]))]
            keep[drop] = False
        return None, worst

    def projected_rotation_order(
        self,
        zone_axis,
        k_max: float | None = None,
        tol_zone: float = 0.02,
        intensity_tol: float = 0.05,
        snap_deg: float = 4.0,
        max_index: int = 3,
    ):
        """Apparent rotational symmetry of the zero-layer pattern, per zone axis.

        A zone-layer pattern can be more symmetric about the beam than the
        crystal is, and where it is, the in-plane orientation cannot be
        indexed. Body-centered cubic along <111> is the standard case: the
        zero-layer net of {110} reflections is hexagonal, so the pattern
        repeats every 60 degrees while the crystal repeats every 120, and the
        two orientations 60 degrees apart give the same peak positions and the
        same kinematical intensities. Only the higher-order Laue zones or the
        dynamical intensities separate them.

        Returned is the largest n in (6, 4, 3, 2, 1) for which rotating the
        zero-layer reflections by 360/n about the zone axis reproduces the
        set, in position and in kinematical intensity. Fold an in-plane angle
        or color by 360/n to get a map that is continuous across the
        ambiguity, and use `n` against the crystal's own rotational order
        about the same axis to see where indexing is degenerate.

        Parameters
        ----------
        zone_axis : array-like
            Cartesian zone axis (3,), or a stack of them (..., 3); need not
            be normalized.
        k_max : float | None
            Only reflections within this scattering vector are tested;
            defaults to the crystal's own k_max.
        tol_zone : float, default=0.02
            Half-thickness of the zero layer (1/Angstroms): reflections with
            |g . zone_axis| below this count as zero layer.
        intensity_tol : float, default=0.05
            A reflection and its image must agree in |F|^2 to within this
            fraction of the strongest zero-layer reflection.
        snap_deg : float, default=4.0
            Zone axes within this angle of a low-index lattice direction are
            evaluated at that direction. The extra symmetry is exact only on
            the pole and decays away from it, but a beam a degree or two off
            still produces a pattern whose positions carry it, which is
            where a measured orientation normally sits; testing the exact
            tilted axis would report no symmetry at all and miss the
            ambiguity the indexing actually suffers. Set to 0 to test the
            axis as given.
        max_index : int, default=3
            Largest |u|, |v|, |w| considered when snapping.

        Returns
        -------
        int | np.ndarray
            The order n, scalar for a single zone axis.
        """
        if self.g_vec is None:
            raise RuntimeError("Run calculate_structure_factors() first.")
        axes = torch.as_tensor(np.asarray(zone_axis, dtype=float), dtype=torch.float64)
        single = axes.ndim == 1
        axes = axes.reshape(-1, 3)
        axes = axes / torch.linalg.norm(axes, dim=1, keepdim=True).clamp_min(1e-12)

        g = self.g_vec
        inten = self.struct_factors_int.to(torch.float64)
        if k_max is not None:
            sel = self.g_len <= float(k_max)
            g, inten = g[sel], inten[sel]

        if snap_deg > 0:
            rng = torch.arange(-max_index, max_index + 1, dtype=torch.float64)
            uvw = torch.cartesian_prod(rng, rng, rng)
            uvw = uvw[uvw.abs().sum(dim=1) > 0]
            cart = uvw @ self.lat_real
            cart = cart / torch.linalg.norm(cart, dim=1, keepdim=True).clamp_min(1e-12)
            dots = torch.abs(axes @ cart.T)
            best = dots.max(dim=1)
            near = best.values > np.cos(np.deg2rad(snap_deg))
            snapped = cart[best.indices]
            # keep the original sense so the returned axis still points along
            # the beam, and only replace the ones close enough to snap
            sign = torch.sign(torch.einsum("ni,ni->n", snapped, axes)).unsqueeze(1)
            axes = torch.where(near.unsqueeze(1), snapped * sign, axes)

        out = np.ones(axes.shape[0], dtype=int)
        eye = torch.eye(3, dtype=torch.float64)
        for i, u in enumerate(axes):
            zol = torch.abs(g @ u) <= tol_zone
            gz, iz = g[zol], inten[zol]
            if gz.shape[0] < 3:
                continue
            i_max = abs(float(iz.max())) or 1.0
            ux = torch.tensor(
                [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]],
                dtype=torch.float64,
            )
            for n in (6, 4, 3, 2):
                th = 2 * np.pi / n
                R = eye + np.sin(th) * ux + (1 - np.cos(th)) * (ux @ ux)  # Rodrigues
                d = torch.cdist(gz @ R.T, gz)
                dmin, j = d.min(dim=1)
                if float(dmin.max()) > tol_zone:
                    continue
                if float((iz - iz[j]).abs().max()) / i_max <= intensity_tol:
                    out[i] = n
                    break
        return int(out[0]) if single else out

    def zone_axis_wedge(self) -> torch.Tensor | None:
        """Fundamental zone-axis wedge corners (3, 3) Cartesian, or None.

        Built from the symmetry operations actually used for matching (the
        pseudo-symmetry group when one was found), so the wedge is right
        for every crystal setting. None means the Laue class (-1 or 2/m)
        has no 3-corner wedge and libraries sample the full hemisphere.
        """
        if isinstance(self._wedge_cache, str):
            from quantem.diffraction.rotations import fundamental_zone_axis_wedge

            self._wedge_cache = fundamental_zone_axis_wedge(self.sym_quats_matching)
        return self._wedge_cache

    @property
    def hexagonal_matching(self) -> bool:
        """Whether directions are written with 4-index symbols.

        Directions are always indexed in the crystal's own cell, so this
        follows that cell's Laue class, not the matching group's: a
        monoclinic superstructure matched with a trigonal parent group still
        has a monoclinic cell, and Miller-Bravais indices would be wrong.
        """
        return self.laue_group in ("6/m", "6/mmm", "-3", "-3m")

    def zone_axis_wedge_labels(self, mathtext: bool = True) -> list[str] | None:
        """Direction labels of the wedge corners (4-index for hexagonal and
        trigonal crystals), indexed from the corner directions themselves."""
        corners = self.zone_axis_wedge()
        if corners is None:
            return None
        loose = max(2.0 * float(self._pseudo_symmetry_tol or 0.0), 0.02)
        labels = []
        for c in corners:
            uvw = direction_indices(self.lat_real, c.numpy())
            prefix = ""
            if uvw is None:
                # an axis of the pseudo-symmetry parent, a lattice direction
                # only to within the distortion of the real cell
                uvw = direction_indices(self.lat_real, c.numpy(), atol=loose)
                prefix = "~"
            if uvw is not None:
                # v and -v are the same zone axis: name it with the first
                # nonzero index of the printed symbol positive, [100] rather
                # than [-100] (for 4-index symbols, [U V T W] with U = 2u - v,
                # V = 2v - u, T = -(u + v), up to a common factor)
                uvw = np.asarray(uvw)
                u, v, w = uvw
                shown = (
                    np.array([2 * u - v, 2 * v - u, -(u + v), w])
                    if self.hexagonal_matching
                    else uvw
                )
                nz = np.flatnonzero(shown)
                if nz.size and shown[nz[0]] < 0:
                    uvw = -uvw
            labels.append(
                prefix
                + format_direction(uvw, hexagonal=self.hexagonal_matching, mathtext=mathtext)
                if uvw is not None
                else "(irrational)"
            )
        return labels

    def matching_symmetry_warning(self) -> str | None:
        """Message when the matching (pseudo) symmetry differs from the
        cell's own symmetry, or None when they agree."""
        if self.pointgroup_matching == self.pointgroup:
            return None
        n_extra = self.sym_quats_matching.shape[0] // max(self.sym_quats.shape[0], 1)
        # a partially accepted group has no Laue class of its own, and keeps
        # the cell's: name it only when it differs
        laue = (
            f"Laue class {self.laue_group_matching}, "
            if self.laue_group_matching != self.laue_group
            else ""
        )
        return (
            f"{self.name}: orientation libraries are built with the "
            f"pseudo-symmetry point group {self.pointgroup_matching} ({laue}"
            f"found at pseudo_symmetry_tol = "
            f"{self._pseudo_symmetry_tol:g} of the shortest lattice vector, "
            f"intensities matching within {self.pseudo_symmetry_report.get('intensity_mismatch', 0.0):.2f} "
            "of the strongest reflection), "
            f"while the cell's own symmetry "
            f"is {self.pointgroup} (Laue class {self.laue_group}). Orientations "
            f"related by the extra operations give the same library entry, so "
            f"the {n_extra} variants they generate are reported as one and the "
            f"distortion between them is not resolved. To match with the exact "
            f"symmetry, build the Crystal with pseudo_symmetry_tol=None (or a "
            f"tolerance below the distortion)."
        )

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
            laue = (
                f"(Laue class {self.laue_group_matching}) "
                if self.laue_group_matching != self.laue_group
                else ""
            )
            lines += [
                f"  pseudo-symmetry  {self.pointgroup_matching} {laue}"
                "-- used for orientation matching",
            ]
            rep = self.pseudo_symmetry_report
            if rep.get("route"):
                lines += [f"                   from the {rep['route']}"]
            if rep.get("intensity_mismatch", 0.0) > 0 and rep.get("broken_by") is not None:
                h0, i0, h1, i1 = rep["broken_by"]
                lines += [
                    "                   accepted at intensity tol %.2f: largest difference "
                    "(%s) at %.2f against (%s) at %.2f"
                    % (
                        self._pseudo_symmetry_intensity_tol,
                        " ".join(map(str, h0)),
                        i0,
                        " ".join(map(str, h1)),
                        i1,
                    )
                ]
        elif self._pseudo_symmetry_tol is not None:
            rep = self.pseudo_symmetry_report
            if rep.get("rejected"):
                lines += [
                    f"  pseudo-symmetry  {rep['candidate']} from the {rep.get('route', 'lattice')}, "
                    f"rejected: intensities differ by "
                    f"{rep['intensity_mismatch']:.2f} (tol {self._pseudo_symmetry_intensity_tol:.2f})",
                ]
                if rep.get("broken_by") is not None:
                    h0, i0, h1, i1 = rep["broken_by"]
                    lines += [
                        "                   broken by (%s) at %.2f against (%s) at %.2f of the "
                        "strongest reflection"
                        % (" ".join(map(str, h0)), i0, " ".join(map(str, h1)), i1)
                    ]
            else:
                lines += [
                    f"  pseudo-symmetry  none found at tol = {self._pseudo_symmetry_tol:g} "
                    f"({rep.get('distance_A', 0.0):.3f} A)",
                ]
        else:
            lines += ["  pseudo-symmetry  not checked (set pseudo_symmetry_tol)"]
        # matching line reflects the symmetry actually used, after any
        # pseudo-symmetry reduction
        labels = self.zone_axis_wedge_labels(mathtext=False)
        wedge_txt = (
            f"zone axis wedge {labels[0]}, {labels[1]}, {labels[2]}"
            if labels is not None
            else "full hemisphere"
        )
        lines += [
            f"  matching         {self.sym_quats_matching.shape[0]} proper rotations, {wedge_txt}"
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
            For Bloch calculations with beams out to k, the couplings reach
            2k, but the factors fall off fast and 1.5k is enough.
        include_core : bool, default=True
            Include the core-loss (inner-shell ionization) absorptive part.
        include_phonon : bool, default=True
            Include the phonon (thermal diffuse scattering) absorptive part.

        Returns
        -------
        Crystal
            self, for chaining. Sets ``hkl_dyn`` (N, 3) and ``g_len_dyn`` (N,)
            for every reflection with |g| <= k_max including (000),
            ``U_dyn`` (N,) complex128 in 1/Angstroms^2, and the
            ``dyn_energy_ev`` and ``dyn_k_max`` they were computed for.

        Raises
        ------
        RuntimeError
            If `k_max` is None and :meth:`calculate_structure_factors` has
            not been run.
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
        self.dyn_k_max = float(k_max)
        return self

    def direction_vector(self, direction) -> torch.Tensor:
        """Unit Cartesian vector (3,) of a lattice direction in the crystal frame.

        Parameters
        ----------
        direction : sequence of float
            [uvw] in this cell, or [UVTW] for a hexagonal or trigonal cell.
        """
        d = np.asarray(direction, dtype=float).ravel()
        if d.size == 4:
            U, V, T, W = d
            d = np.array([U - T, V - T, W])
        elif d.size != 3:
            raise ValueError(f"a direction has 3 or 4 indices, got {direction}")
        v = d @ self.lat_real.numpy()
        return torch.as_tensor(v / np.linalg.norm(v), dtype=torch.float64)

    def generate_pattern(
        self,
        orientation: torch.Tensor,
        energy_ev: float = 300e3,
        sigma_excitation: float = SIGMA_EXCITATION,
        tol_excitation_mult: float = 3.0,
        k_max: float | None = None,
        precession_deg: float = 0.0,
        semiconv_mrad: float = 0.0,
        excitation_model: str = "gaussian",
        thickness_A: float | None = None,
        foil_normal=None,
    ) -> dict[str, torch.Tensor]:
        """Kinematical diffraction pattern for one orientation.

        The intensity of each reflection is |F_g|^2 times a Gaussian
        excitation envelope of width sigma_excitation, averaged exactly over
        the illumination when a precession angle or a convergence
        semiangle is given (quantem.diffraction.illumination): the
        precession ring sweeps the excitation error of reflection g by
        +- a_g = r |g_xy| / |K - g_z| about its central value c_g, and the
        averaged envelope is the Bessel transform G(c_g, a_g, b_g; sigma).

        Parameters
        ----------
        orientation : torch.Tensor
            Unit quaternion (4,) rotating crystal vectors into the lab frame.
        energy_ev : float, default=300e3
            Beam energy in eV.
        sigma_excitation : float, default=SIGMA_EXCITATION
            Excitation error tolerance (1/Angstroms) in the shape-factor
            envelope exp(-s_g^2 / 2 sigma^2); the default is
            :data:`quantem.diffraction.defaults.SIGMA_EXCITATION`.
        tol_excitation_mult : float, default=3.0
            Include reflections with |s_g| below this multiple of sigma.
        k_max : float | None
            Optionally trim the pattern below the structure-factor k_max.
        precession_deg, semiconv_mrad : float
            Precession semi-angle (degrees) and convergence semiangle
            (mrad) of the illumination the intensities are averaged over.
        excitation_model : {"gaussian", "slab"}
            "gaussian" is the empirical envelope of width sigma_excitation
            used by the orientation library. "slab" is the finite-thickness
            first Born rocking curve, (pi |U_g| z / k0)^2 sinc(s_g z)^2
            with U_g = gamma_rel F_g / pi, averaged over the illumination
            the same way; it needs thickness_A and is the kinematical limit
            of the Bloch wave calculation for thin crystals.
        thickness_A : float | None
            Thickness for the slab model (Angstroms).
        foil_normal : sequence of float, optional
            Plate normal of the specimen as a direction in this crystal,
            [uvw] or [UVTW], e.g. (0, 0, 0, 1) for a 2D material lying in its
            basal plane. Every reflection is then a rod along that normal: it
            is excited by its distance along the rod to the Ewald sphere, and
            its spot sits where the rod meets the sphere rather than at the
            projection of g. For a tilted flake, whose rods are long, the
            spots shift by up to s_g tan(tilt). None (default) takes the
            normal along the beam, the usual geometry.

        Returns
        -------
        dict
            'qx', 'qy' (1/Angstroms), 'intensity', 'hkl', 's_g' (the central
            excitation error, along the rod when `foil_normal` is given), 'a'
            and 'b' (ring and disk sweep amplitudes), one entry per excited
            reflection.

        Raises
        ------
        RuntimeError
            If :meth:`calculate_structure_factors` has not been run.
        ValueError
            If `excitation_model` is not "gaussian" or "slab", or the slab
            model is asked for without `thickness_A`.
        """
        if self.g_vec is None:
            raise RuntimeError("Run calculate_structure_factors first.")
        if excitation_model not in _EXCITATION_MODELS:
            raise ValueError(
                f"excitation_model must be one of {_EXCITATION_MODELS}, got {excitation_model!r}"
            )
        from quantem.diffraction.illumination import (
            averaged_gaussian_intensity_envelope,
            excitation_coefficients,
            relrod_factor,
            slab_envelope,
        )

        g = qrotate(orientation, self.g_vec)
        n_lab = None
        if foil_normal is not None:
            n_lab = qrotate(orientation, self.direction_vector(foil_normal)[None])[0]
        if excitation_model == "slab":
            if thickness_A is None:
                raise ValueError("the slab excitation model needs thickness_A")
            c, a, b = excitation_coefficients(g, energy_ev, precession_deg, semiconv_mrad)
            if n_lab is not None:
                f = relrod_factor(g.numpy(), n_lab.numpy(), energy_ev, precession_deg)
                c, a, b = c * f, a * np.abs(f), b * np.abs(f)
            # the sinc^2 tails are algebraic: keep everything whose main
            # lobe (width 1/z) plus illumination sweep is within the tolerance
            width = tol_excitation_mult / float(thickness_A)
            c_t = torch.as_tensor(c, dtype=torch.float64)
            a_t = torch.as_tensor(a, dtype=torch.float64)
            b_t = torch.as_tensor(b, dtype=torch.float64)
            keep = torch.abs(c_t) < a_t + b_t + width
            if k_max is not None:
                keep &= self.g_len <= k_max
            env = slab_envelope(
                c[keep.numpy()], a[keep.numpy()], b[keep.numpy()], float(thickness_A)
            )
            from quantem.core.utils.utils import electron_wavelength_angstrom

            lam = electron_wavelength_angstrom(energy_ev)
            gamma_rel = 1.0 + float(energy_ev) / 510998.95
            u_abs = torch.abs(self.struct_factors[keep]) * (gamma_rel / np.pi)
            intensity = (np.pi * u_abs * float(thickness_A) * lam) ** 2 * torch.as_tensor(
                env, dtype=torch.float64
            )
        else:
            env, c, a, b = averaged_gaussian_intensity_envelope(
                g,
                energy_ev,
                sigma_excitation,
                precession_deg,
                semiconv_mrad,
                foil_normal_lab=None if n_lab is None else n_lab.numpy(),
            )
            c_t = torch.as_tensor(c, dtype=torch.float64)
            a_t = torch.as_tensor(a, dtype=torch.float64)
            b_t = torch.as_tensor(b, dtype=torch.float64)
            # the full illumination support enters the selection, not only
            # the central excitation error
            keep = torch.abs(c_t) < a_t + b_t + sigma_excitation * tol_excitation_mult
            if k_max is not None:
                keep &= self.g_len <= k_max
            intensity = (
                self.struct_factors_int[keep] * torch.as_tensor(env, dtype=torch.float64)[keep]
            )
        qxy = g[keep, :2]
        if n_lab is not None:
            # the spot is where the rod meets the sphere: g + t n, t = -c
            qxy = qxy - c_t[keep, None] * n_lab[None, :2]
        return {
            "qx": qxy[:, 0],
            "qy": qxy[:, 1],
            "intensity": intensity,
            "hkl": self.hkl[keep],
            "s_g": c_t[keep],
            "a": a_t[keep],
            "b": b_t[keep],
        }

    def __repr__(self) -> str:
        return (
            f"Crystal({self.name}, {len(self.numbers)} atoms, "
            f"spacegroup {self.spacegroup}, pointgroup {self.pointgroup})"
        )
