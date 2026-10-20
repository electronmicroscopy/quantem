"""3D atomic model analysis: calibration, neighbor finding, structure classification.

:class:`AtomicModel` holds the sites of a 3D atomic model (for example traced
from an atomic electron tomography reconstruction) in a
:class:`~quantem.core.datastructures.Vector` and provides the analysis
pipeline:

1. :meth:`AtomicModel.compute_pdf` - radial distribution function and first
   nearest-neighbor (NN) peak fit.
2. :meth:`AtomicModel.calibrate` - set the physical size of one voxel from
   the measured NN distance of a reference crystal.
3. :meth:`AtomicModel.find_neighbors` - neighbor lists, coordination and bond
   lengths.
4. :meth:`AtomicModel.match_templates` - fast polyhedral template matching
   against ``fcc``, ``hcp``, ``bcc``, ``diamond`` ... environments.
5. :meth:`AtomicModel.classify` / :meth:`AtomicModel.segment_grains` /
   :meth:`AtomicModel.compute_strain` - per-site structure, grain (sector)
   labels via orientation clustering, and local strain.

Every per-site result is stored as a named *channel* (a field of the sites
``Vector``) so it can be plotted with :meth:`AtomicModel.plot` or explored
interactively with :meth:`AtomicModel.show`.

Coordinate convention
---------------------
Fields ``x, y, z`` are the positions along array axes 0, 1, 2 of the source
volume, stored in their native (typically voxel) units.  ``sampling`` converts
them to physical units; :attr:`AtomicModel.positions` returns calibrated
coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from quantem.atoms import measurements as meas
from quantem.atoms.matching import TemplateMatch, match_template
from quantem.atoms.pdf import (
    find_neighbors,
    fit_first_peak,
    nn_distance_from_lattice,
    radial_distribution,
)
from quantem.atoms.templates import PolyhedralTemplate, get_template
from quantem.atoms.visualization import PLOT_REGISTRY
from quantem.core.datastructures import Vector
from quantem.core.io.serialize import AutoSerialize

__all__ = ["AtomicModel"]

_POSITION_FIELDS = ("x", "y", "z")


class AtomicModel(AutoSerialize):
    """A 3D atomic model with per-site measurement channels.

    Use the ``from_*`` constructors rather than ``__init__``.

    Parameters
    ----------
    sites : Vector
        0-D ``Vector`` whose cell holds one row per site with at least the
        fields ``x, y, z``.
    sampling : ndarray
        ``(3,)`` physical size of one native coordinate unit along each axis.
    units : str
        Physical length unit after calibration (e.g. ``"A"``).
    name : str
        Model name.
    metadata : dict
        Free-form metadata.
    """

    _token = object()

    def __init__(
        self,
        sites: Vector,
        sampling: NDArray,
        units: str,
        name: str,
        metadata: dict[str, Any] | None = None,
        _token: object | None = None,
    ) -> None:
        if _token is not self._token:
            raise RuntimeError("Use AtomicModel.from_array() or another from_* constructor.")
        self._sites = sites
        self._sampling = np.asarray(sampling, dtype=float).reshape(3)
        self._units = str(units)
        self._name = str(name)
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._pdf: dict[str, Any] | None = None
        self._nn_fit: dict[str, Any] | None = None
        self._neighbor_distances: NDArray | None = None
        self._neighbor_indices: NDArray | None = None
        self._matches: dict[str, TemplateMatch] = {}
        self._template_specs: dict[str, dict[str, Any]] = {}
        self._structure_names: list[str] = []
        self._categories: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_array(
        cls,
        xyz: NDArray,
        sampling: float | Sequence[float] = 1.0,
        units: str = "voxels",
        name: str | None = None,
        channels: dict[str, NDArray] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AtomicModel":
        """Create a model from an ``(N, 3)`` coordinate array.

        Parameters
        ----------
        xyz : ndarray
            ``(N, 3)`` positions along axes 0, 1, 2.  A ``(3, N)`` array is
            transposed automatically.
        sampling : float or sequence of float
            Physical size of one coordinate unit (scalar or per axis).
        units : str
            Physical length unit (``"voxels"`` if uncalibrated).
        name : str, optional
            Model name.
        channels : dict, optional
            Extra per-site arrays stored as channels, e.g. ``{"species": ...}``.
        metadata : dict, optional
            Free-form metadata.
        """
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or 3 not in xyz.shape:
            raise ValueError(f"xyz must be (N, 3) or (3, N), got {xyz.shape}")
        if xyz.shape[1] != 3:
            xyz = xyz.T
        sites = Vector.from_shape(
            shape=(),
            fields=list(_POSITION_FIELDS),
            units=[units] * 3,
            name="sites",
        )
        sites[...] = np.ascontiguousarray(xyz)
        sampling_arr = np.broadcast_to(np.asarray(sampling, dtype=float), (3,)).copy()
        model = cls(
            sites=sites,
            sampling=sampling_arr,
            units=units,
            name=name or "atomic model",
            metadata=metadata,
            _token=cls._token,
        )
        for key, values in (channels or {}).items():
            model.set_channel(key, values)
        return model

    @classmethod
    def from_xyz(
        cls,
        path: str | Path,
        sampling: float | Sequence[float] = 1.0,
        units: str = "A",
        name: str | None = None,
    ) -> "AtomicModel":
        """Load an ``.xyz`` text file (``element x y z`` rows after a 2-line header)."""
        path = Path(path)
        lines = path.read_text().strip().splitlines()
        try:
            count = int(lines[0].split()[0])
            body = lines[2 : 2 + count]
        except (ValueError, IndexError):
            body = lines
        symbols, coords = [], []
        for line in body:
            parts = line.split()
            if len(parts) < 4:
                continue
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        xyz = np.asarray(coords)
        names = sorted(set(symbols))
        species = np.array([names.index(s) for s in symbols], dtype=float)
        model = cls.from_array(xyz, sampling=sampling, units=units, name=name or path.stem)
        model.set_channel("species", species, categories=names)
        return model

    @classmethod
    def from_atoms(cls, atoms: Any, name: str | None = None) -> "AtomicModel":
        """Create a model from a :class:`quantem.tomography.Atoms` tracing result.

        Positions are taken in voxel units together with the traced intensity
        and Gaussian width, and ``sampling`` is copied from the source volume.
        """
        sites = atoms.sites.array
        sampling = np.asarray(atoms._sampling, dtype=float)
        xyz = (sites[:, :3] - np.asarray(atoms._origin)[None, :]) / sampling[None, :]
        model = cls.from_array(
            xyz,
            sampling=sampling,
            units=str(atoms._units[0]),
            name=name or f"{atoms._source.name} atoms",
        )
        model.set_channel("intensity", sites[:, 3])
        model.set_channel("sigma", sites[:, 4] / float(sampling.mean()))
        return model

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        """Model name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def metadata(self) -> dict[str, Any]:
        """Free-form metadata dictionary."""
        return self._metadata

    @property
    def sites(self) -> Vector:
        """Per-site table (0-D ``Vector``); fields ``x, y, z`` plus channels."""
        return self._sites

    @property
    def num_sites(self) -> int:
        """Number of atomic sites."""
        return int(self._sites.array.shape[0])

    @property
    def sampling(self) -> NDArray:
        """``(3,)`` physical size of one native coordinate unit per axis."""
        return self._sampling

    @property
    def units(self) -> str:
        """Physical length unit of :attr:`positions`."""
        return self._units

    @property
    def positions_native(self) -> NDArray:
        """``(N, 3)`` positions in native (uncalibrated) units."""
        return np.array(self._sites.select_fields(*_POSITION_FIELDS).array, dtype=float)

    @positions_native.setter
    def positions_native(self, value: NDArray) -> None:
        value = np.asarray(value, dtype=float)
        if value.shape != (self.num_sites, 3):
            raise ValueError(f"positions must have shape {(self.num_sites, 3)}")
        self._sites.select_fields(*_POSITION_FIELDS)[...] = value
        self._invalidate()

    @property
    def positions(self) -> NDArray:
        """``(N, 3)`` calibrated positions (native * sampling)."""
        return self.positions_native * self._sampling[None, :]

    @property
    def center(self) -> NDArray:
        """``(3,)`` mean calibrated position."""
        return self.positions.mean(0)

    @property
    def channels(self) -> list[str]:
        """Names of all per-site channels (fields other than ``x, y, z``)."""
        return [f for f in self._sites.fields if f not in _POSITION_FIELDS]

    @property
    def categories(self) -> dict[str, list[str]]:
        """Label names for categorical channels, e.g. ``{"structure": ["fcc", "hcp"]}``."""
        return self._categories

    @property
    def structure_names(self) -> list[str]:
        """Template names indexed by the ``structure`` channel value."""
        return list(self._structure_names)

    @property
    def templates(self) -> dict[str, PolyhedralTemplate]:
        """Templates used in the last :meth:`match_templates` call."""
        return {
            name: PolyhedralTemplate(
                name=name,
                vectors=np.asarray(spec["vectors"]),
                shells=tuple(float(x) for x in spec["shells"]),
                shell_counts=tuple(int(x) for x in spec["shell_counts"]),
                symmetry=np.asarray(spec["symmetry"]),
            )
            for name, spec in self._template_specs.items()
        }

    @property
    def matches(self) -> dict[str, TemplateMatch]:
        """Raw per-template matching results (see :class:`TemplateMatch`)."""
        return self._matches

    @property
    def pdf(self) -> dict[str, Any] | None:
        """Result of :meth:`compute_pdf` (native units), or ``None``."""
        return self._pdf

    @property
    def nn_fit(self) -> dict[str, Any] | None:
        """First-peak fit from :meth:`compute_pdf` (native units), or ``None``."""
        return self._nn_fit

    @property
    def nn_distance(self) -> float:
        """Mean nearest-neighbor distance in native units (requires :meth:`compute_pdf`)."""
        if self._nn_fit is None:
            self.compute_pdf()
        assert self._nn_fit is not None
        return float(self._nn_fit["r_nn"])

    @property
    def bond_length(self) -> float:
        """Mean nearest-neighbor distance in calibrated units."""
        return self.nn_distance * float(self._sampling.mean())

    @property
    def neighbor_indices(self) -> NDArray:
        """``(N, K)`` neighbor indices sorted by distance (``-1`` = missing)."""
        if self._neighbor_indices is None:
            self.find_neighbors()
        assert self._neighbor_indices is not None
        return self._neighbor_indices

    @property
    def neighbor_distances(self) -> NDArray:
        """``(N, K)`` neighbor distances in native units."""
        if self._neighbor_distances is None:
            self.find_neighbors()
        assert self._neighbor_distances is not None
        return self._neighbor_distances

    def _invalidate(self) -> None:
        self._pdf = None
        self._nn_fit = None
        self._neighbor_distances = None
        self._neighbor_indices = None
        self._matches = {}

    # ------------------------------------------------------------------ #
    # Channels
    # ------------------------------------------------------------------ #
    def get_channel(self, name: str) -> NDArray:
        """Return a per-site channel as a ``(N,)`` array."""
        if name in _POSITION_FIELDS:
            return self.positions[:, _POSITION_FIELDS.index(name)]
        if name not in self._sites.fields:
            raise KeyError(f"Unknown channel {name!r}; available: {self.channels}")
        return np.array(self._sites.select_fields(name).array[:, 0], dtype=float)

    def set_channel(
        self,
        name: str,
        values: NDArray,
        units: str = "none",
        categories: Sequence[str] | None = None,
    ) -> None:
        """Add or overwrite a per-site channel.

        Parameters
        ----------
        name : str
            Channel name.
        values : ndarray
            ``(N,)`` values (cast to float; use integer codes for categories).
        units : str
            Units label.
        categories : sequence of str, optional
            Names for integer codes ``0, 1, ...``; marks the channel categorical.
        """
        if name in _POSITION_FIELDS:
            raise ValueError("Use positions_native to modify coordinates.")
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape[0] != self.num_sites:
            raise ValueError(f"values must have length {self.num_sites}, got {values.shape[0]}")
        if name in self._sites.fields:
            self._sites.select_fields(name)[...] = values[:, None]
        else:
            self._sites.add_fields(name, values[:, None], units)
        if categories is not None:
            self._categories[name] = [str(c) for c in categories]
        elif name in self._categories:
            del self._categories[name]

    def remove_channel(self, name: str) -> None:
        """Delete a channel."""
        self._sites.remove_fields(name)
        self._categories.pop(name, None)

    def __getitem__(self, name: str) -> NDArray:
        return self.get_channel(name)

    # ------------------------------------------------------------------ #
    # Pair distribution function and calibration
    # ------------------------------------------------------------------ #
    def compute_pdf(
        self,
        r_max: float | None = None,
        dr: float | None = None,
        sigma: float | None = None,
        fit_radius: float = 1.25,
        cutoff_sigma: float = 2.0,
        r_min: float | None = None,
    ) -> dict[str, Any]:
        """Compute the radial distribution function and fit the first NN peak.

        All radii are in native units.

        Parameters
        ----------
        r_max : float, optional
            Maximum radius; default 4x an initial NN estimate.
        dr : float, optional
            Bin width; default ``r_max / 600``.
        sigma : float, optional
            Smoothing width; default ``2 * dr``.
        fit_radius, cutoff_sigma, r_min
            See :func:`quantem.atoms.pdf.fit_first_peak`.

        Returns
        -------
        dict
            RDF arrays (``r``, ``g``, ``g_smooth``, ``counts``) plus the fit.
        """
        xyz = self.positions_native
        if r_max is None:
            d1, _ = find_neighbors(xyz, 1)
            r_max = 4.0 * float(np.median(d1))
        if dr is None:
            dr = r_max / 600.0
        pdf = radial_distribution(xyz, r_max=r_max, dr=dr, sigma=sigma)
        fit = fit_first_peak(
            pdf["r"],
            pdf["g_smooth"],
            fit_radius=fit_radius,
            cutoff_sigma=cutoff_sigma,
            r_min=r_min,
        )
        self._pdf = pdf
        self._nn_fit = fit
        return {**pdf, **{k: v for k, v in fit.items()}}

    def calibrate(
        self,
        structure: str | None = None,
        lattice_constant: float | None = None,
        nn_distance: float | None = None,
        units: str = "A",
    ) -> float:
        """Set ``sampling`` so the measured NN distance matches a reference.

        Provide either ``structure`` + ``lattice_constant`` or ``nn_distance``.

        Parameters
        ----------
        structure : str, optional
            Reference crystal (``"fcc"``, ``"bcc"``, ``"hcp"`` ...).
        lattice_constant : float, optional
            Lattice constant of the reference crystal in ``units``.
        nn_distance : float, optional
            Target NN distance in ``units``.
        units : str
            Physical unit of the reference.

        Returns
        -------
        float
            The isotropic scale (physical units per native unit).
        """
        if nn_distance is None:
            if structure is None or lattice_constant is None:
                raise ValueError("Give structure and lattice_constant, or nn_distance.")
            nn_distance = nn_distance_from_lattice(structure, lattice_constant)
        scale = float(nn_distance) / self.nn_distance
        self._sampling = np.full(3, scale)
        self._units = units
        return scale

    # ------------------------------------------------------------------ #
    # Neighbors and bonds
    # ------------------------------------------------------------------ #
    def find_neighbors(self, num_neighbors: int = 24, cutoff: float | None = None) -> None:
        """Build neighbor lists and per-site bond statistics.

        Adds channels ``num_neighbors`` (first-shell coordination),
        ``bond_mean`` and ``bond_std`` (calibrated units).

        Parameters
        ----------
        num_neighbors : int
            Neighbors stored per site; must exceed the largest template.
        cutoff : float, optional
            First-shell radial cutoff in native units.  Default: upper cutoff
            from the RDF first-peak fit.
        """
        dist, idx = find_neighbors(self.positions_native, num_neighbors)
        self._neighbor_distances = dist
        self._neighbor_indices = idx
        if cutoff is None:
            cutoff = self.first_shell_cutoff
        first = dist <= cutoff
        scale = float(self._sampling.mean())
        d_first = np.where(first, dist, np.nan) * scale
        import warnings

        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            self.set_channel("num_neighbors", first.sum(1))
            self.set_channel("bond_mean", np.nanmean(d_first, axis=1), self._units)
            self.set_channel("bond_std", np.nanstd(d_first, axis=1), self._units)

    @property
    def first_shell_cutoff(self) -> float:
        """Upper radial cutoff of the first shell (native units) from the RDF fit."""
        if self._nn_fit is None:
            self.compute_pdf()
        assert self._nn_fit is not None
        return float(self._nn_fit["cutoff"][1])

    def neighbor_vectors(self, normalize: bool = True) -> tuple[NDArray, NDArray]:
        """Neighbor displacement vectors.

        Parameters
        ----------
        normalize : bool
            Divide by the NN distance so bonds have length ~1.

        Returns
        -------
        dxyz, dist : ndarray
            ``(N, K, 3)`` vectors and ``(N, K)`` lengths (native units, or NN
            units when normalized).  Missing neighbors are ``inf``.
        """
        idx = self.neighbor_indices
        dist = self.neighbor_distances.copy()
        xyz = self.positions_native
        safe = np.where(idx >= 0, idx, 0)
        dxyz = xyz[safe] - xyz[:, None, :]
        dxyz[idx < 0] = np.inf
        if normalize:
            r_nn = self.nn_distance
            return dxyz / r_nn, dist / r_nn
        return dxyz, dist

    def bond_angles(self) -> NDArray:
        """All first-shell bond angles per site, ``(N, K(K-1)/2)`` degrees with ``nan`` padding."""
        dxyz, dist = self.neighbor_vectors(normalize=False)
        valid = dist <= self.first_shell_cutoff
        dxyz = np.where(np.isfinite(dxyz), dxyz, 0.0)
        return meas.bond_angles(dxyz, valid)

    # ------------------------------------------------------------------ #
    # Template matching and classification
    # ------------------------------------------------------------------ #
    def match_templates(
        self,
        templates: Sequence[str | PolyhedralTemplate] = ("fcc", "hcp"),
        score_radius: float = 0.5,
        cutoff_factor: float = 1.15,
        angle_tolerance: float = 30.0,
        num_refine: int = 2,
        chunk_size: int = 512,
        device: str | None = None,
        progress: bool = True,
    ) -> dict[str, TemplateMatch]:
        """Match polyhedral templates to every site.

        Adds channels ``score_<name>``, ``rmsd_<name>`` and ``matched_<name>``
        for each template, then calls :meth:`classify` with default settings.

        Parameters
        ----------
        templates : sequence of str or PolyhedralTemplate
            Template names (see :data:`quantem.atoms.TEMPLATE_NAMES`) or objects.
        score_radius : float
            Matching radius in NN units; see :func:`quantem.atoms.matching.match_template`.
        cutoff_factor : float
            Neighbors farther than ``cutoff_factor * template.max_radius`` (NN
            units) are ignored for that template.
        angle_tolerance, num_refine, chunk_size, device, progress
            Forwarded to :func:`quantem.atoms.matching.match_template`.

        Returns
        -------
        dict
            ``{name: TemplateMatch}``.
        """
        dxyz, dist = self.neighbor_vectors(normalize=True)
        dxyz = np.where(np.isfinite(dxyz), dxyz, 1e3)
        self._matches = {}
        self._template_specs = {}
        for item in templates:
            template = get_template(item) if isinstance(item, str) else item
            name = template.name
            valid = dist <= cutoff_factor * template.max_radius
            if valid.shape[1] < template.num_neighbors:
                raise ValueError(
                    f"Template {name!r} has {template.num_neighbors} neighbors but only "
                    f"{valid.shape[1]} are stored; call find_neighbors(num_neighbors=...)."
                )
            result = match_template(
                dxyz,
                valid,
                template,
                score_radius=score_radius,
                angle_tolerance=angle_tolerance,
                num_refine=num_refine,
                chunk_size=chunk_size,
                device=device,
                progress=progress,
            )
            self._matches[name] = result
            self._template_specs[name] = {
                "vectors": np.asarray(template.vectors),
                "shells": list(template.shells),
                "shell_counts": list(template.shell_counts),
                "symmetry": np.asarray(template.symmetry),
            }
            self.set_channel(f"score_{name}", result["score"])
            self.set_channel(f"rmsd_{name}", result["rmsd"])
            self.set_channel(f"matched_{name}", result["num_matched"])
        self.classify()
        return self._matches

    def classify(
        self, threshold: float = 0.5, smooth: bool = False, use_strained: bool = False
    ) -> NDArray:
        """Assign each site to its best-scoring template.

        Adds channels ``structure`` (categorical: template index, ``-1`` for
        unclassified), ``score_max`` and, when exactly two templates were
        matched, ``score_diff`` (first minus second).

        Parameters
        ----------
        threshold : float
            Minimum score for a site to be classified.
        smooth : bool
            Average each site's scores with its first-shell neighbors before
            deciding (more robust for noisy models).
        use_strained : bool
            Use the affine-fit scores (``score_strained``) instead.

        Returns
        -------
        ndarray
            ``(N,)`` structure codes.
        """
        if not self._matches:
            raise RuntimeError("Call match_templates() first.")
        names = list(self._matches)
        key = "score_strained" if use_strained else "score"
        scores = np.stack([self._matches[n][key] for n in names], axis=1)
        if smooth:
            idx = self.neighbor_indices
            first = self.neighbor_distances <= self.first_shell_cutoff
            safe = np.where(idx >= 0, idx, 0)
            nb = scores[safe] * first[..., None]
            scores = (scores + nb.sum(1)) / (1.0 + first.sum(1))[:, None]
        best = scores.argmax(1)
        score_max = scores.max(1)
        structure = np.where(score_max >= threshold, best, -1)
        self._structure_names = names
        self.set_channel("structure", structure, categories=names)
        self.set_channel("score_max", score_max)
        if len(names) == 2:
            self.set_channel("score_diff", scores[:, 0] - scores[:, 1])
        return structure

    def rotations(self, template: str | None = None) -> NDArray:
        """``(N, 3, 3)`` fitted orientations (lab <- crystal) for a template.

        With ``template=None`` the rotation from each site's classified
        structure is returned (identity for unclassified sites).
        """
        if template is not None:
            return self._matches[template]["rotation"]
        structure = self.get_channel("structure").astype(int)
        out = np.tile(np.eye(3), (self.num_sites, 1, 1))
        for i, name in enumerate(self._structure_names):
            sel = structure == i
            out[sel] = self._matches[name]["rotation"][sel]
        return out

    def segment_grains(
        self,
        structure: str = "fcc",
        angle_threshold: float = 5.0,
        min_size: int = 20,
        min_score: float | None = None,
        fill_iterations: int = 3,
        fill_angle: float | None = None,
    ) -> NDArray:
        """Cluster sites of one structure into grains by local orientation.

        Neighboring sites of the given structure whose disorientation is below
        ``angle_threshold`` are connected, and connected components become
        grains.  Sites of the structure left without a grain (typically sites
        between two grains, or sites in components smaller than ``min_size``)
        are then assigned by majority vote of their first-shell neighbors,
        provided their disorientation to that grain is below ``fill_angle``;
        the vote is repeated ``fill_iterations`` times so gaps close inward.

        Adds channels ``grain`` (``-1`` = none), ``misorientation`` (largest
        disorientation to any first-shell neighbor of the same structure) and
        ``boundary`` (categorical: ``interior``, ``grain_boundary`` for sites
        with a first-shell neighbor in another grain, ``twin`` for sites of
        the second matched template such as ``hcp``, and ``surface`` for sites
        with fewer than 9 first-shell neighbors).

        Parameters
        ----------
        structure : str
            Template name to segment (e.g. ``"fcc"``).
        angle_threshold : float
            Maximum disorientation (degrees) between connected sites.
        min_size : int
            Components with fewer sites are dissolved and re-filled.
        min_score : float, optional
            Only sites with ``score_<structure>`` above this take part;
            default: sites classified as ``structure``.
        fill_iterations : int
            Number of majority-vote passes over unassigned sites (0 disables).
        fill_angle : float, optional
            Maximum disorientation for a vote to count; default
            ``3 * angle_threshold``.

        Returns
        -------
        ndarray
            ``(N,)`` grain labels sorted by decreasing size.
        """
        if structure not in self._matches:
            raise KeyError(f"No match for {structure!r}; run match_templates first.")
        template = self.templates[structure]
        rot = self._matches[structure]["rotation"]
        idx = self.neighbor_indices
        first = self.neighbor_distances <= self.first_shell_cutoff
        if min_score is None:
            member = self.get_channel("structure").astype(int) == self._structure_names.index(
                structure
            )
        else:
            member = self._matches[structure]["score"] >= min_score
        ang = meas.misorientation(rot, idx, template.symmetry)
        safe = np.where(idx >= 0, idx, 0)
        same = member[:, None] & member[safe] & first & (idx >= 0)
        edge = same & (ang < angle_threshold)
        labels = meas.segment_grains(idx, edge, min_size=min_size)
        labels[~member] = -1
        if fill_angle is None:
            fill_angle = 3.0 * angle_threshold
        for _ in range(int(fill_iterations)):
            labels = meas.fill_labels(labels, idx, same & (ang < fill_angle), member)
        # relabel by decreasing size
        if labels.max() >= 0:
            sizes = np.bincount(labels[labels >= 0])
            order = np.argsort(-sizes, kind="stable")
            rank = np.empty_like(order)
            rank[order] = np.arange(order.size)
            labels = np.where(labels >= 0, rank[np.clip(labels, 0, None)], -1)
        worst = np.where(same, ang, -1.0).max(axis=1).clip(0.0)
        num_grains = int(labels.max()) + 1 if labels.size else 0
        self.set_channel("grain", labels, categories=[str(i) for i in range(num_grains)])
        self.set_channel("misorientation", worst, "deg")
        # boundary classification
        structure_id = self.get_channel("structure").astype(int)
        nb_labels = np.where(idx >= 0, labels[np.where(idx >= 0, idx, 0)], -2)
        other_grain = (
            first & (nb_labels >= 0) & (nb_labels != labels[:, None]) & (labels[:, None] >= 0)
        )
        boundary = np.zeros(self.num_sites, dtype=int)
        boundary[other_grain.any(axis=1)] = 1
        names = self._structure_names
        twin_ids = [i for i, n in enumerate(names) if n != structure]
        if twin_ids:
            boundary[np.isin(structure_id, twin_ids)] = 2
        boundary[first.sum(axis=1) < 9] = 3
        self.set_channel(
            "boundary", boundary, categories=["interior", "grain_boundary", "twin", "surface"]
        )
        return labels

    def compute_strain(
        self, template: str | None = None, frame: str = "lab"
    ) -> dict[str, NDArray]:
        """Local strain from the affine template fit.

        Adds channels ``strain_xx, strain_yy, strain_zz, strain_xy, strain_xz,
        strain_yz, strain_dilation, strain_equivalent``.  Strain is relative to
        the mean NN distance of the model.

        Parameters
        ----------
        template : str, optional
            Template whose fit to use; default: each site's classified structure.
        frame : {"lab", "crystal"}
            Frame of the strain tensor.
        """
        if template is not None:
            f = self._matches[template]["deformation"]
            r = self._matches[template]["rotation"]
        else:
            structure = self.get_channel("structure").astype(int)
            f = np.tile(np.eye(3), (self.num_sites, 1, 1))
            r = f.copy()
            for i, name in enumerate(self._structure_names):
                sel = structure == i
                f[sel] = self._matches[name]["deformation"][sel]
                r[sel] = self._matches[name]["rotation"][sel]
        strain = meas.strain_from_deformation(f, r, frame=frame)
        for key in ("e_xx", "e_yy", "e_zz", "e_xy", "e_xz", "e_yz"):
            self.set_channel("strain_" + key[2:], strain[key])
        self.set_channel("strain_dilation", strain["dilation"])
        self.set_channel("strain_equivalent", strain["equivalent"])
        return strain

    # ------------------------------------------------------------------ #
    # Other measurements
    # ------------------------------------------------------------------ #
    def surface_distance(self) -> NDArray:
        """Distance of each site to the convex hull (calibrated units); channel ``surface_distance``."""
        d = meas.convex_hull_distance(self.positions)
        self.set_channel("surface_distance", d, self._units)
        return d

    def sample_volume(self, volume: Any, radius: float = 1.5, name: str = "intensity") -> NDArray:
        """Mean reconstruction intensity around each site; stored as a channel.

        Parameters
        ----------
        volume : ndarray or Dataset3d
            Source volume indexed like the native coordinates.
        radius : float
            Sphere radius in voxels.
        name : str
            Channel name.
        """
        arr = getattr(volume, "array", volume)
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        values = meas.sample_volume(np.asarray(arr), self.positions_native, radius=radius)
        self.set_channel(name, values)
        return values

    def classify_species(
        self,
        channel: str = "intensity",
        num_species: int = 2,
        names: Sequence[str] | None = None,
        method: str = "gmm",
        min_posterior: float = 0.8,
        mask: NDArray | None = None,
    ) -> NDArray:
        """Assign species from a per-site channel such as the traced intensity.

        With ``method="gmm"`` a one-dimensional Gaussian mixture with
        ``num_species`` components is fit by expectation maximization and each
        site takes the component with the largest posterior probability; sites
        whose largest posterior is below ``min_posterior`` are left unassigned
        (code ``-1``, label ``unassigned``) so that ambiguous candidates stay in
        the model without being counted as either species.  ``method="kmeans"``
        assigns every site to the nearest cluster center.

        Adds channels ``species`` (categorical) and, for the mixture, ``species_posterior``.
        The fitted means, widths and fractions are stored in
        ``metadata["species_model"]``.

        Parameters
        ----------
        channel : str
            Channel to split (``"intensity"`` from tracing or volume sampling).
        num_species : int
            Number of species.
        names : sequence of str, optional
            Species names in order of increasing channel value.
        method : {"gmm", "kmeans"}
            Classifier.
        min_posterior : float
            Minimum posterior probability for an assignment (``"gmm"`` only).
        mask : ndarray, optional
            ``(N,)`` boolean; only these sites take part in the fit and receive
            a species, all others are unassigned.  Use it to exclude surface
            sites, whose intensities are reduced by the missing neighbors.

        Returns
        -------
        ndarray
            ``(N,)`` species codes, ``-1`` for unassigned sites.
        """
        values = self.get_channel(channel).astype(float)
        use = np.ones(self.num_sites, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        fit_values = np.where(use, values, np.nan)
        if names is None:
            names = [f"species_{i}" for i in range(num_species)]
        if method == "kmeans":
            labels, centers = meas.kmeans_1d(fit_values, num_species)
            posterior = np.ones(self.num_sites)
            model_info = {"method": "kmeans", "centers": centers.tolist()}
        elif method == "gmm":
            means, sigmas, weights, resp = meas.gaussian_mixture_1d(fit_values, num_species)
            labels = resp.argmax(1)
            posterior = resp.max(1)
            labels = np.where(posterior >= min_posterior, labels, -1)
            model_info = {
                "method": "gmm",
                "means": means.tolist(),
                "sigmas": sigmas.tolist(),
                "weights": weights.tolist(),
                "min_posterior": float(min_posterior),
            }
        else:
            raise ValueError("method must be 'gmm' or 'kmeans'")
        labels = np.where(use, labels, -1)
        posterior = np.where(use, posterior, 0.0)
        self.set_channel("species", labels, categories=list(names))
        self.set_channel("species_posterior", posterior)
        self._metadata["species_model"] = model_info
        return labels

    # ------------------------------------------------------------------ #
    # Multiply twinned particle geometry
    # ------------------------------------------------------------------ #
    def twin_plane_normals(self, structure: str = "hcp") -> NDArray:
        """``(N, 3)`` twin-plane normal at every site from the ``hcp`` template fit.

        The hexagonal ``c`` axis of the fitted HCP template is the normal of
        the close-packed plane, which for a site on a coherent twin boundary
        is the twin plane.  Values are only meaningful at sites classified as
        ``structure``.
        """
        return self._matches[structure]["rotation"][:, :, 2].copy()

    def fit_icosahedral_centers(
        self,
        num_centers: int = 2,
        min_score: float = 0.6,
        max_residual: float = 2.0,
        axial_tolerance: float = 0.35,
        num_iterations: int = 10,
    ) -> dict[str, Any]:
        """Locate the centers of two icosahedra that share a 5-fold axis.

        All twin planes of one Mackay icosahedron pass through its center, so
        the center is the least-squares intersection of the planes carried by
        its twin (``hcp``) sites.  Planes that contain the shared axis pass
        through both centers and are excluded; the remaining sites are
        assigned to the center whose planes they fit best, outliers beyond
        ``max_residual`` are dropped, and the fit is iterated.  The result
        is stored in ``metadata["icosahedral_centers"]``.

        Parameters
        ----------
        num_centers : int
            Only ``2`` is supported at present.
        min_score : float
            Minimum ``score_hcp`` of the twin sites used.
        max_residual : float
            Plane-distance cutoff (calibrated units) for keeping a site.
        axial_tolerance : float
            Planes whose normal has ``|n . axis|`` below this are treated as
            containing the axis and excluded.
        num_iterations : int
            Outer iterations of the assignment.

        Returns
        -------
        dict
            ``centers`` (2, 3) relative to the model center, ``axis`` unit
            vector from center 0 to center 1, ``separation`` in calibrated
            units, ``rms`` plane residual per center, ``num_sites`` per
            center, and ``assignment`` (``(N,)`` with ``-1`` = unused).
        """
        if num_centers != 2:
            raise NotImplementedError("Only two centers are supported.")
        if "hcp" not in self._matches:
            raise RuntimeError("Match the 'hcp' template first (match_templates(['fcc', 'hcp'])).")
        xyz = self.positions - self.center[None, :]
        normals = self.twin_plane_normals("hcp")
        first_shell = (self.neighbor_distances <= self.first_shell_cutoff).sum(1)
        twin = self.get_channel("structure").astype(int) == self._structure_names.index("hcp")
        twin &= (self._matches["hcp"]["score"] >= min_score) & (first_shell >= 12)
        center0, res = meas.fit_plane_intersection(xyz[twin], normals[twin])
        # the shared axis is the direction most perpendicular to the well-fitting planes
        good = np.abs(res) < 1.5 * max_residual
        cov = np.einsum("ni,nj->ij", normals[twin][good], normals[twin][good])
        axis = np.linalg.eigh(cov)[1][:, 0]
        c1, c2 = center0 - 0.5 * axis, center0 + 0.5 * axis
        assign = np.full(self.num_sites, -1)
        for _ in range(num_iterations):
            non_axial = np.abs(normals @ axis) > axial_tolerance
            usable = twin & non_axial
            r1 = np.abs(np.einsum("ni,ni->n", normals, xyz - c1))
            r2 = np.abs(np.einsum("ni,ni->n", normals, xyz - c2))
            a1 = usable & (r1 <= r2) & (r1 < max_residual)
            a2 = usable & (r2 < r1) & (r2 < max_residual)
            if a1.sum() < 10 or a2.sum() < 10:
                h = (xyz - center0) @ axis
                a1 = usable & (h < 0)
                a2 = usable & (h >= 0)
            c1n, _ = meas.fit_plane_intersection(xyz[a1], normals[a1])
            c2n, _ = meas.fit_plane_intersection(xyz[a2], normals[a2])
            shift = np.linalg.norm(c1n - c1) + np.linalg.norm(c2n - c2)
            c1, c2 = c1n, c2n
            axis = (c2 - c1) / np.linalg.norm(c2 - c1)
            if shift < 1e-4:
                break
        assign[a1] = 0
        assign[a2] = 1
        rms = [
            float(np.sqrt(np.mean(np.einsum("ni,ni->n", normals[a], xyz[a] - c) ** 2)))
            for a, c in ((a1, c1), (a2, c2))
        ]
        result = {
            "centers": np.stack([c1, c2]),
            "axis": axis,
            "separation": float(np.linalg.norm(c2 - c1)),
            "rms": np.array(rms),
            "num_sites": np.array([int(a1.sum()), int(a2.sum())]),
            "assignment": assign,
        }
        self._metadata["icosahedral_centers"] = {
            "centers": result["centers"].tolist(),
            "axis": axis.tolist(),
            "separation": result["separation"],
        }
        return result

    def layer_positions(
        self,
        normal: str | NDArray,
        bin_width: float | None = None,
        sigma: float | None = None,
        min_fraction: float = 0.25,
    ) -> NDArray:
        """Positions of the atomic layers perpendicular to ``normal``.

        Sites are projected onto ``normal`` (measured from the model center),
        the projected density is histogrammed and smoothed, and its peaks are
        returned.  The result can be passed as ``positions`` to
        ``plot("slices", ...)`` to step through the model one layer at a time.

        Parameters
        ----------
        normal : str or array
            Layer normal (``"x"``, ``"y"``, ``"z"`` or a 3-vector).
        bin_width : float, optional
            Histogram bin (default: bond length / 40).
        sigma : float, optional
            Smoothing (default: bond length / 12).
        min_fraction : float
            Peaks below this fraction of the highest peak are ignored.

        Returns
        -------
        ndarray
            Layer offsets along ``normal`` relative to the model center.
        """
        from quantem.atoms.visualization import view_matrix

        n = view_matrix(normal)[2]
        h = (self.positions - self.center[None, :]) @ n
        bond = self.bond_length
        pos, _, _ = meas.layer_positions(
            h,
            bin_width=bond / 40.0 if bin_width is None else bin_width,
            sigma=bond / 12.0 if sigma is None else sigma,
            min_fraction=min_fraction,
        )
        return pos

    def explode_grains(
        self,
        distance: float | None = None,
        origin: str | NDArray | None = None,
        include_shared: bool = True,
        keep_other: bool = False,
        structure: str = "fcc",
    ) -> "AtomicModel":
        """Return a copy with every grain displaced away from its origin.

        Each grain (sector) moves rigidly by ``distance`` along the direction
        from ``origin`` to its centroid, so the grains separate and their
        boundaries become visible.  Twin sites and other boundary sites that
        touch several grains are copied into every adjacent grain, displaced
        with it, and labelled ``shared`` in the new categorical ``grain``
        channel; sites classified as neither ``structure`` nor a twin are
        labelled ``other`` and kept in place when ``keep_other`` is True.

        Parameters
        ----------
        distance : float, optional
            Displacement per grain (calibrated units); default 2 bond lengths.
        origin : {"center", "icosahedral"} or array, optional
            Point the grains move away from: the model center (default), the
            nearest of the fitted icosahedral centers (after
            :meth:`fit_icosahedral_centers`), or an explicit ``(3,)`` point.
        include_shared : bool
            Copy boundary sites into each adjacent grain.
        keep_other : bool
            Keep unclassified sites (``other``) at their original positions.
        structure : str
            Template name of the grains (``"fcc"``).

        Returns
        -------
        AtomicModel
            New model with channels ``grain`` (categorical), ``source_index``
            (index in this model) and copies of all other channels.
        """
        if "grain" not in self.channels:
            raise RuntimeError("Call segment_grains() first.")
        if distance is None:
            distance = 2.0 * self.bond_length
        xyz = self.positions
        grain = self.get_channel("grain").astype(int)
        struct = self.get_channel("structure").astype(int)
        num_grains = int(grain.max()) + 1
        centroids = np.array([xyz[grain == g].mean(0) for g in range(num_grains)])
        if origin is None or origin == "center":
            origins = np.tile(self.center, (num_grains, 1))
        elif isinstance(origin, str) and origin == "icosahedral":
            info = self._metadata.get("icosahedral_centers")
            if info is None:
                raise RuntimeError(
                    "Call fit_icosahedral_centers() first for origin='icosahedral'."
                )
            cents = np.asarray(info["centers"]) + self.center[None, :]
            nearest = np.argmin(
                np.linalg.norm(centroids[:, None, :] - cents[None, :, :], axis=2), axis=1
            )
            origins = cents[nearest]
        else:
            origins = np.tile(np.asarray(origin, dtype=float).reshape(3), (num_grains, 1))
        direction = centroids - origins
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        shift = distance * direction / np.where(norm > 1e-9, norm, 1.0)

        table = np.array(self._sites.array, dtype=float)
        pos_cols = [self._sites.fields.index(f) for f in _POSITION_FIELDS]
        scale = self._sampling[None, :]
        rows, labels, source = [], [], []
        # grain members
        member = grain >= 0
        idx = np.where(member)[0]
        t = table[idx].copy()
        t[:, pos_cols] += shift[grain[idx]] / scale
        rows.append(t)
        labels.append(grain[idx])
        source.append(idx)
        # shared boundary sites: copied into each adjacent grain
        twin_id = [i for i, n in enumerate(self._structure_names) if n != structure]
        boundary = (~member) & (
            np.isin(struct, twin_id) | (struct == self._structure_names.index(structure))
        )
        if include_shared and boundary.any():
            nb_idx = self.neighbor_indices
            first = self.neighbor_distances <= self.first_shell_cutoff
            nb_grain = np.where(first & (nb_idx >= 0), grain[np.where(nb_idx >= 0, nb_idx, 0)], -1)
            for i in np.where(boundary)[0]:
                adjacent = np.unique(nb_grain[i][nb_grain[i] >= 0])
                for g in adjacent:
                    r = table[i].copy()
                    r[pos_cols] += shift[g] / scale[0]
                    rows.append(r[None, :])
                    labels.append(np.array([num_grains]))
                    source.append(np.array([i]))
        other = (~member) & ~boundary
        if keep_other and other.any():
            idx = np.where(other)[0]
            rows.append(table[idx])
            labels.append(np.full(idx.size, num_grains + 1))
            source.append(idx)
        new_table = np.vstack(rows)
        sites = Vector.from_shape(
            shape=(), fields=list(self._sites.fields), units=list(self._sites.units), name="sites"
        )
        sites[...] = np.ascontiguousarray(new_table)
        out = AtomicModel(
            sites=sites,
            sampling=self._sampling.copy(),
            units=self._units,
            name=f"{self._name} (exploded)",
            metadata={"exploded_distance": float(distance)},
            _token=self._token,
        )
        out._categories = {k: v for k, v in self._categories.items() if k != "grain"}
        out._structure_names = list(self._structure_names)
        out._nn_fit = None if self._nn_fit is None else dict(self._nn_fit)
        out.set_channel(
            "grain",
            np.concatenate(labels),
            categories=[str(g) for g in range(num_grains)] + ["shared", "other"],
        )
        out.set_channel("source_index", np.concatenate(source))
        return out

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def rotate(self, rotation: NDArray, about_center: bool = True) -> None:
        """Rotate all positions in place with a ``(3, 3)`` matrix (``x' = R x``)."""
        rotation = np.asarray(rotation, dtype=float)
        xyz = self.positions_native
        c = xyz.mean(0) if about_center else np.zeros(3)
        self.positions_native = (xyz - c) @ rotation.T + c

    def merge_close_sites(
        self,
        min_distance: float | None = None,
        mode: str = "merge",
        weight_channel: str | None = None,
    ) -> int:
        """Merge or remove sites closer than ``min_distance`` (in place).

        Sites are grouped into clusters by connecting every pair closer than
        ``min_distance``.  With ``mode="merge"`` each cluster is replaced by
        one site at its (weighted) mean position, with all channels averaged;
        with ``mode="remove"`` only the site with the largest weight (or the
        first site) of each cluster is kept.  Neighbor lists, template matches
        and the RDF are cleared afterwards.

        Parameters
        ----------
        min_distance : float, optional
            Distance threshold in native units.  Default: half the NN distance
            from the RDF fit.
        mode : {"merge", "remove"}
            How to resolve each cluster.
        weight_channel : str, optional
            Channel used as weights (e.g. ``"intensity"``); equal weights if
            omitted.

        Returns
        -------
        int
            Number of sites removed.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        if min_distance is None:
            min_distance = 0.5 * self.nn_distance
        xyz = self.positions_native
        n = xyz.shape[0]
        pairs = cKDTree(xyz).query_pairs(float(min_distance), output_type="ndarray")
        if pairs.shape[0] == 0:
            return 0
        graph = coo_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, labels = connected_components(graph, directed=False)
        table = np.array(self._sites.array, dtype=float)
        weights = (
            np.ones(n)
            if weight_channel is None
            else self.get_channel(weight_channel).astype(float)
        )
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1e-12)
        num_clusters = int(labels.max()) + 1
        if mode == "merge":
            sums = np.zeros((num_clusters, table.shape[1]))
            np.add.at(sums, labels, table * weights[:, None])
            wsum = np.bincount(labels, weights=weights, minlength=num_clusters)
            new_table = sums / wsum[:, None]
        elif mode == "remove":
            order = np.lexsort((-weights, labels))
            first = np.ones(n, dtype=bool)
            first[1:] = labels[order][1:] != labels[order][:-1]
            new_table = table[order][first]
        else:
            raise ValueError("mode must be 'merge' or 'remove'")
        sites = Vector.from_shape(
            shape=(), fields=list(self._sites.fields), units=list(self._sites.units), name="sites"
        )
        sites[...] = np.ascontiguousarray(new_table)
        self._sites = sites
        self._invalidate()
        return int(n - new_table.shape[0])

    def select(self, mask: NDArray) -> "AtomicModel":
        """Return a new model containing only the sites where ``mask`` is True."""
        mask = np.asarray(mask, dtype=bool)
        table = self._sites.array[mask]
        sites = Vector.from_shape(
            shape=(), fields=list(self._sites.fields), units=list(self._sites.units), name="sites"
        )
        sites[...] = np.ascontiguousarray(table)
        model = AtomicModel(
            sites=sites,
            sampling=self._sampling.copy(),
            units=self._units,
            name=self._name,
            metadata=dict(self._metadata),
            _token=self._token,
        )
        model._categories = dict(self._categories)
        model._structure_names = list(self._structure_names)
        return model

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #
    def plot(self, kind: str = "slab", show_docstring: bool = False, **kwargs):
        """Static matplotlib plots; see :mod:`quantem.atoms.visualization`.

        Parameters
        ----------
        kind : str
            One of ``"pdf"``, ``"histogram"``, ``"slab"``, ``"slices"``,
            ``"template"``.
        show_docstring : bool
            Print the plot function's docstring instead of plotting.
        **kwargs
            Forwarded to the plot function.
        """
        if kind not in PLOT_REGISTRY:
            raise ValueError(f"Unknown plot kind {kind!r}; choose from {list(PLOT_REGISTRY)}")
        fn = PLOT_REGISTRY[kind]
        if show_docstring:
            print(fn.__doc__)
            return None
        return fn(self, **kwargs)

    def show(self, **kwargs):
        """Open the interactive 3D viewer ``quantem.widget.ShowAtoms3D``.

        Requires the ``quantem.widget`` package.  Keyword arguments are
        forwarded to the viewer (``channel``, ``cmap``, ``canvas_size``,
        ``marker_size``, ``dark_background``, ``title``).
        """
        try:
            from quantem.widget import ShowAtoms3D
        except ImportError as exc:
            raise ImportError(
                "AtomicModel.show() requires the quantem.widget package: "
                "pip install quantem.widget"
            ) from exc
        return ShowAtoms3D(self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"AtomicModel(name={self._name!r}, num_sites={self.num_sites}, "
            f"units={self._units!r}, channels={self.channels})"
        )
