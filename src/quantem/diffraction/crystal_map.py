"""Orientation and phase mapping over one or more candidate crystals.

CrystalMap is the standard entry point for ACOM. It owns one
:class:`~quantem.diffraction.orientation.OrientationMap` per candidate
crystal, fans the matching and refinement stages out over all of them, and
holds the :class:`~quantem.diffraction.phase.PhaseMap` that decides which
crystal sits at each probe position::

    cm = CrystalMap.from_vectors(peaks, [Cu_metal, Cu2O], energy_ev=300e3)
    cm.build_plan(**plan_params)
    cm.match_orientations(**match_params)
    cm.refine_orientations(**refine_params)
    cm.fit()
    cm.plot_phase()
    cm.plot_orientation()

The individual maps stay available for anything asymmetric or per-crystal --
``cm["Cu metal"]``, ``cm[0]`` or ``cm.orientation_maps`` -- and every method
on OrientationMap works there exactly as before.
"""

from __future__ import annotations

import numpy as np

from quantem.core.io.serialize import AutoSerialize
from quantem.diffraction.crystal import Crystal
from quantem.diffraction.orientation import OrientationMap
from quantem.diffraction.phase import SHADE_GAMMA, PhaseMap


def _common_k_max(crystals, k_max: float | None) -> float:
    """The one k_max every crystal is simulated to, or an error saying why not."""
    if k_max is not None:
        return float(k_max)
    have = {xtl.name: xtl.k_max for xtl in crystals}
    missing = [n for n, k in have.items() if k is None]
    if missing:
        raise ValueError(
            f"no structure factors for {missing}: pass k_max= to CrystalMap.from_vectors, "
            "which computes them for every crystal"
        )
    if len({round(float(k), 9) for k in have.values()}) > 1:
        raise ValueError(
            f"crystals are simulated to different k_max {have}; pass k_max= to "
            "CrystalMap.from_vectors to set one range for all of them"
        )
    return float(next(iter(have.values())))


class CrystalMap(AutoSerialize):
    """Per-position crystal orientation and phase over a scan.

    Parameters
    ----------
    orientation_maps : list of OrientationMap
        One matched (or unmatched) map per candidate crystal.

    Attributes
    ----------
    orientation_maps : list of OrientationMap
        The per-crystal maps, in the order they were given.
    names : list of str
        Crystal names, used for indexing and plot labels.
    phases : PhaseMap or None
        The phase decision, populated by :meth:`fit`.
    dynamical : dict or None
        The last :meth:`refine_dynamical` result.
    """

    _token = object()

    def __init__(
        self, orientation_maps: list[OrientationMap], _token=None, k_max: float | None = None
    ):
        """Private constructor; use :meth:`from_vectors` or :meth:`from_orientation_maps`.

        Parameters
        ----------
        orientation_maps : list of OrientationMap
            One per crystal, all sharing a scan shape. Names must be unique,
            because crystals are indexed by name.
        _token : object
            Guard against direct construction.
        k_max : float, optional
            Scattering-vector limit shared by the crystals; taken from them
            when None.

        Raises
        ------
        RuntimeError
            If called without the class token.
        ValueError
            If no crystals are given, names repeat, or scan shapes differ.
        """
        if _token is not self._token:
            raise RuntimeError(
                "Use CrystalMap.from_vectors() or CrystalMap.from_orientation_maps()."
            )
        if len(orientation_maps) == 0:
            raise ValueError("CrystalMap needs at least one crystal.")
        names = [om.crystal.name for om in orientation_maps]
        if len(set(names)) != len(names):
            raise ValueError(
                f"crystal names must be unique for indexing by name, got {names}. "
                "Set Crystal(name=...) to tell them apart."
            )
        shapes = {tuple(om.peaks.shape[:2]) for om in orientation_maps}
        if len(shapes) != 1:
            raise ValueError(f"all crystals must share one scan shape, got {shapes}")
        self.orientation_maps = orientation_maps
        self.phases: PhaseMap | None = None
        self.dynamical: dict | None = None
        self.metadata: dict = {}
        self.k_max = _common_k_max([om.crystal for om in orientation_maps], k_max)

    def __attrs_post_init__(self):
        """After loading: the phase map shares this map's orientation maps.

        A saved file holds the phase map's copies of the orientation maps
        separately, and without this they would load as independent objects:
        anything computed on one (orientations written back by a refinement,
        structure factors attached to a crystal) would be missed by the other.
        """
        if self.phases is not None:
            self.phases.orientation_maps = self.orientation_maps

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_vectors(
        cls,
        peaks,
        crystals: Crystal | list[Crystal],
        energy_ev: float = 300e3,
        precession_deg: float = 0.0,
        semiconv_mrad: float = 0.0,
        k_max: float | None = None,
        foil_normal=None,
    ) -> "CrystalMap":
        """Build one OrientationMap per crystal from a shared peak table.

        Parameters
        ----------
        peaks : Vector
            Calibrated Bragg peaks, shared by every crystal.
        crystals : Crystal or list of Crystal
            Candidate phases. A single Crystal is accepted, so single-phase
            work uses the same entry point.
        energy_ev, precession_deg, semiconv_mrad
            Passed to :meth:`OrientationMap.from_vectors`.
        k_max : float, optional
            Largest scattering vector (1/Angstroms) in the simulated patterns,
            applied to every crystal: their structure factors are computed
            here, so it is set once. Match it to the detector; reflections
            beyond it cannot be paired. None keeps the structure factors the
            crystals already have, which must then share one k_max, since two
            phases simulated to different ranges are not compared fairly.
        foil_normal : sequence or dict, optional
            Plate normal of a 2D material or thin flake, as a direction in each
            crystal, [uvw] or [UVTW]; a dict keyed by crystal name sets it per
            crystal. Reflections are then rods along it, which places the
            spots of a tilted flake where the rods meet the Ewald sphere (see
            :meth:`OrientationMap.from_vectors`). None (default) is the usual
            geometry.

        Raises
        ------
        ValueError
            If `k_max` is None and the crystals have no structure factors, or
            have them to different ranges.
        """
        xtls = list(crystals) if isinstance(crystals, (list, tuple)) else [crystals]
        if k_max is not None:
            for xtl in xtls:
                xtl.calculate_structure_factors(k_max=float(k_max))
        else:
            _common_k_max(xtls, None)  # say what is wrong before anything is built
        oms = [
            OrientationMap.from_vectors(
                peaks,
                xtl,
                energy_ev=energy_ev,
                precession_deg=precession_deg,
                semiconv_mrad=semiconv_mrad,
                foil_normal=(
                    foil_normal.get(xtl.name) if isinstance(foil_normal, dict) else foil_normal
                ),
            )
            for xtl in xtls
        ]
        return cls(oms, _token=cls._token, k_max=k_max)

    @classmethod
    def from_orientation_maps(cls, orientation_maps: list[OrientationMap]) -> "CrystalMap":
        """Wrap maps that were built and matched by hand.

        Parameters
        ----------
        orientation_maps : list of OrientationMap
            One per crystal, all sharing a scan shape and with unique
            crystal names. The scattering-vector limit is taken from the
            crystals, which must share one.

        Returns
        -------
        CrystalMap

        Raises
        ------
        ValueError
            If the list is empty, names repeat, scan shapes differ, or the
            crystals have no common k_max.
        """
        return cls(list(orientation_maps), _token=cls._token)

    # ------------------------------------------------------------------
    # access
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Crystal names, in the order the maps were given."""
        return [om.crystal.name for om in self.orientation_maps]

    @property
    def peaks(self):
        """The shared peak list; every crystal was matched against these."""
        return self.orientation_maps[0].peaks

    @property
    def shape(self) -> tuple[int, int]:
        """Scan shape ``(rows, cols)`` in probe positions."""
        return tuple(self.orientation_maps[0].peaks.shape[:2])

    def __len__(self) -> int:
        """Number of crystals."""
        return len(self.orientation_maps)

    def __iter__(self):
        """Iterate over the per-crystal OrientationMaps."""
        return iter(self.orientation_maps)

    def __getitem__(self, key) -> OrientationMap:
        """`cm[0]` or `cm["Cu metal"]` -> the OrientationMap of that crystal."""
        if isinstance(key, str):
            try:
                return self.orientation_maps[self.names.index(key)]
            except ValueError:
                raise KeyError(f"no crystal named {key!r}; have {self.names}") from None
        return self.orientation_maps[key]

    def __repr__(self) -> str:
        """Scan shape, how far the analysis has run, and the crystal names."""
        R, C = self.shape
        stage = "unmatched"
        if self.orientation_maps[0].quats is not None:
            stage = "matched"
            if "refine" in self.orientation_maps[0].metadata:
                stage = "refined"
        if self.phases is not None and self.phases.phase_index is not None:
            stage += ", phase fit"
        return "CrystalMap(%d x %d, %s, [%s])" % (R, C, stage, ", ".join(self.names))

    # ------------------------------------------------------------------
    # staged workflow, fanned out over the crystals
    # ------------------------------------------------------------------

    def build_plan(self, overrides: dict | None = None, **kwargs) -> "CrystalMap":
        """Build the correlation plan for every crystal.

        Parameters
        ----------
        overrides : dict, optional
            Per-crystal keyword overrides, keyed by crystal name, e.g.
            ``overrides={"Cu2O": dict(angle_step_zone_axis_deg=2.0)}``.
        **kwargs
            Passed to :meth:`OrientationMap.build_plan` for every crystal.
        """
        return self._fanout("build_plan", overrides, **kwargs)

    def match_orientations(self, overrides: dict | None = None, **kwargs) -> "CrystalMap":
        """Match every crystal against the measured peaks.

        Each crystal is correlated against its own plan, so the scores are
        comparable: the library slices are unit vectors and the measured
        polar image is divided by its norm, making the correlation a cosine
        similarity in [0, 1]. With `num_matches` above 1 each further match
        is fitted to what the earlier ones leave unexplained, so a probe
        straddling two grains indexes both. Every match is still scored
        against the pattern as measured, so comparing the matches tells the
        two cases apart: two grains score alike, while a spurious second
        match on a single grain scores well below the first.

        Parameters
        ----------
        overrides : dict, optional
            Per-crystal keyword overrides, keyed by crystal name.
        **kwargs
            Passed to :meth:`OrientationMap.match_orientations` for every
            crystal. The ones usually set are `num_matches`,
            `min_number_peaks` and `positions`, the last restricting the
            match to a few probe positions for a staged test run.

        Returns
        -------
        CrystalMap
            Self, so stages chain.

        Notes
        -----
        The correlation saturates on sparse patterns: a position carrying
        the direct beam and two noise peaks scores about as well as a real
        grain, because some library orientation almost always has a
        reflection at that radius and angle. Judge a match by
        :meth:`signal_confidence` or the peak count, not by the correlation
        alone.
        """
        return self._fanout("match_orientations", overrides, **kwargs)

    def refine_orientations(
        self,
        overrides: dict | None = None,
        competitive_margin: float | None = 0.1,
        **kwargs,
    ) -> "CrystalMap":
        """Refine every crystal off the library grid.

        Least squares on the paired peak positions removes the quantization
        of the plan: the in-plane rotation comes from the pairing, the zone
        axis tilt from the intensity envelope. Positions that disagree with
        a neighbour are then retried from the candidates around them, and
        ties go to the orientation the neighbours share.

        With several crystals, each is refined only where it is in the
        running: where its library correlation is within
        `competitive_margin` of the best crystal's. Elsewhere its
        orientations are noise -- the crystal is not there -- and refining
        them, and retrying them against their equally random neighbours, is
        most of the work in a two-phase map for no change in the phase
        decision.

        Parameters
        ----------
        overrides : dict, optional
            Per-crystal keyword overrides, keyed by crystal name.
        competitive_margin : float or None, default=0.1
            Correlation margin behind the best crystal within which a crystal
            is still refined. None refines every crystal everywhere.
        **kwargs
            Passed to :meth:`OrientationMap.refine_orientations` for every
            crystal, commonly `num_iterations` and `zone_search_deg`. An
            explicit `positions` is used as given.

        Returns
        -------
        CrystalMap
            Self, so stages chain.
        """
        oms = self.orientation_maps
        if competitive_margin is None or len(oms) < 2 or "positions" in kwargs:
            return self._fanout("refine_orientations", overrides, **kwargs)
        overrides = self._check_overrides(overrides)
        corr = np.stack([om.corr[..., 0].numpy() for om in oms])
        best = corr.max(axis=0)
        for i, om in enumerate(oms):
            kw = {**kwargs, **overrides.get(om.crystal.name, {})}
            kw.setdefault("positions", corr[i] >= best - competitive_margin)
            om.refine_orientations(**kw)
        return self

    def _check_overrides(self, overrides: dict | None) -> dict:
        """Per-crystal overrides, checked against the crystal names.

        Parameters
        ----------
        overrides : dict or None
            Keyword arguments per crystal name.

        Returns
        -------
        dict
            `overrides`, or an empty dict for None.

        Raises
        ------
        KeyError
            If an override names a crystal not in this map.
        """
        overrides = overrides or {}
        unknown = set(overrides) - set(self.names)
        if unknown:
            raise KeyError(f"overrides name unknown crystals {sorted(unknown)}; have {self.names}")
        return overrides

    def _fanout(self, method: str, overrides: dict | None, **kwargs) -> "CrystalMap":
        """Call ``method`` on every OrientationMap, with per-crystal overrides.

        Parameters
        ----------
        method : str
            Name of the OrientationMap method to call.
        overrides : dict or None
            Keyword arguments per crystal name, merged over ``kwargs``.
        **kwargs
            Arguments common to every crystal.

        Returns
        -------
        CrystalMap
            Self, so stages chain.

        Raises
        ------
        KeyError
            If an override names a crystal not in this map.
        """
        overrides = self._check_overrides(overrides)
        for om in self.orientation_maps:
            kw = {**kwargs, **overrides.get(om.crystal.name, {})}
            getattr(om, method)(**kw)
        return self

    def fit(self, **kwargs) -> "CrystalMap":
        """Decide the phase at every position; see :meth:`PhaseMap.fit`.

        With a single crystal there is nothing to choose between, but the fit
        still runs: it applies the null hypothesis, so positions with no
        diffracted signal come out unindexed and the maps below fade them.
        """
        self.phases = PhaseMap.from_orientation_maps(self.orientation_maps)
        self.phases.fit(**kwargs)
        return self

    def refine_dynamical(
        self,
        mask=None,
        k_max: float | None = None,
        k_max_coupling: float | None = None,
        **kwargs,
    ) -> "CrystalMap":
        """Dynamical refinement of orientation, thickness, strain and phase.

        Bloch-wave intensities, averaged over the precession ring, are fit to
        the measured peaks of every candidate at every position of `mask`,
        and the candidate with the lowest cost decides the phase there; the
        rest of the scan keeps its decision. See
        :func:`~quantem.diffraction.bloch.refine_dynamical` for the model and
        every argument. The refined orientations are written back, so calls
        can be staged: thickness and tilt first with
        ``refine_deformation=False``, then the in-plane strain from those
        orientations with a narrower tilt search.

        Every crystal is given absorptive structure factors out to
        `k_max_coupling`, which the couplings between beams need; they are
        computed here when missing or too short.

        Parameters
        ----------
        mask : np.ndarray or list of tuple, optional
            Positions to refine, an (R, C) boolean mask or (row, col) list.
            None refines every matched position, which takes hours.
        k_max : float, optional
            Largest |g| (1/Angstroms) of the beams in the Bloch calculation.
            None takes the k_max of the kinematical simulation. Cutting it
            low saves time but drops beams that carry real dynamical
            coupling.
        k_max_coupling : float, optional
            Largest |g| (1/Angstroms) of the structure factors coupling the
            beams. The couplings g - h reach twice `k_max`, but the factors
            fall off fast; None uses 1.5 `k_max`. It sets accuracy, not run
            time, which the number of beams sets.
        **kwargs
            Passed to :func:`~quantem.diffraction.bloch.refine_dynamical`.
            `require_phase_weight` defaults to False here, so a crystal the
            kinematical fit rejected still competes.

        Returns
        -------
        CrystalMap
            Self, with the result in :attr:`dynamical`.
        """
        from quantem.diffraction import bloch

        pm = self._require_fit("refine_dynamical()")
        k_max = float(k_max if k_max is not None else self._k_max_or_crystals())
        k_c = float(k_max_coupling if k_max_coupling is not None else 1.5 * k_max)
        energy_ev = self.orientation_maps[0].energy_ev
        for om in self.orientation_maps:
            xtl = om.crystal
            if (
                getattr(xtl, "U_dyn", None) is None
                or getattr(xtl, "dyn_k_max", 0.0) < k_c - 1e-9
                or abs(getattr(xtl, "dyn_energy_ev", energy_ev) - energy_ev) > 1.0
            ):
                xtl.calculate_dynamical_structure_factors(energy_ev, k_max=k_c)
        kwargs["k_max"] = k_max
        kwargs.setdefault("require_phase_weight", False)
        self.dynamical = bloch.refine_dynamical(pm, mask=mask, **kwargs)
        pm.apply_dynamical(self.dynamical)
        return self

    def plot_dynamical(self, phase=None, strain: bool = False, crop: bool = True, **kwargs):
        """Maps of the last :meth:`refine_dynamical`.

        Thickness, tilt correction, the cost gain of the tilt search and the
        final cost, or with `strain` the six crystal-frame strain components.

        Parameters
        ----------
        phase : int or str, optional
            Show only positions this crystal won. None shows all of them.
        strain : bool, default=False
            Plot the strain components instead.
        crop : bool, default=True
            Crop to the refined positions.
        **kwargs
            Passed to :func:`~quantem.diffraction.bloch.plot_dynamical_maps`
            or :func:`~quantem.diffraction.bloch.plot_strain_crystal_frame`.

        Returns
        -------
        tuple
            ``(fig, axs)``.
        """
        from quantem.diffraction import bloch

        result = getattr(self, "dynamical", None)
        if result is None:
            raise ValueError("run refine_dynamical() before plot_dynamical().")
        i = None if phase is None else self._phase_indices(phase)[0]
        maps = bloch.dynamical_maps(result, self.phases, crystal_index=i)
        m = maps["mask"].numpy()
        sl = (slice(None), slice(None))
        if crop and m.any():
            rows, cols = np.nonzero(m)
            sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        kwargs.setdefault("scalebar", self.orientation_maps[0].scan_scalebar)
        if strain:
            comps = {k: v.numpy()[sl] for k, v in maps["strain"].items()}
            return bloch.plot_strain_crystal_frame(comps, mask=m[sl], **kwargs)
        cropped = dict(maps)
        for k in ("thickness", "tilt_deg", "gain", "cost"):
            cropped[k] = maps[k][sl]
        return bloch.plot_dynamical_maps(cropped, **kwargs)

    def example_positions(
        self,
        phase=None,
        num: int = 4,
        ambiguous: bool = False,
        min_distance: float = 16.0,
        min_signal: float = 0.3,
    ) -> list[tuple[int, int]]:
        """Well-separated probe positions for inspecting the phase decision.

        The clearest examples of a crystal are where it won by the largest
        margin (the phase reliability); the ambiguous ones are where the two
        best crystals scored closest. With a single crystal there is no
        margin, and positions are ranked by its correlation instead. Only positions that diffract are
        considered, and each pick is at least `min_distance` from the others,
        so the examples come from different parts of the scan.

        Parameters
        ----------
        phase : int or str, optional
            Crystal the positions must have been assigned to. None allows
            any crystal.
        num : int, default=4
            Number of positions.
        ambiguous : bool, default=False
            Pick the closest decisions instead of the clearest.
        min_distance : float, default=16.0
            Smallest separation between picks, in probe positions.
        min_signal : float, default=0.3
            Smallest :meth:`signal_confidence` a position needs.

        Returns
        -------
        list of tuple of int
            ``(row, col)`` positions, clearest (or closest) first.
        """
        pm = self._require_fit("example_positions()")
        ph = self.phase_index
        rel = np.asarray(pm.reliability, dtype=float)
        if not np.isfinite(rel[ph >= 0]).any():
            # one crystal: no runner-up to compare against, so rank by how
            # well the crystal itself matches
            corr = np.stack([om.corr[..., 0].numpy() for om in self.orientation_maps])
            rel = np.where(
                ph >= 0, np.take_along_axis(corr, np.clip(ph, 0, None)[None], 0)[0], np.nan
            )
        ok = (ph >= 0) & np.isfinite(rel) & (self.signal_confidence() >= min_signal)
        if phase is not None:
            ok &= ph == self._phase_indices(phase)[0]
        rc = np.argwhere(ok)
        order = np.argsort(rel[ok] if ambiguous else -rel[ok], kind="stable")
        picks: list[tuple[int, int]] = []
        for r, c in rc[order]:
            if all((r - a) ** 2 + (c - b) ** 2 >= min_distance**2 for a, b in picks):
                picks.append((int(r), int(c)))
                if len(picks) == num:
                    break
        return picks

    # ------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------

    def _require_fit(self, what: str) -> PhaseMap:
        """Return the fitted PhaseMap, or explain which call needs it first.

        Parameters
        ----------
        what : str
            Name of the caller, used in the error message.

        Returns
        -------
        PhaseMap

        Raises
        ------
        ValueError
            If :meth:`fit` has not been run.
        """
        if self.phases is None or self.phases.phase_index is None:
            raise ValueError(f"run fit() before {what}.")
        return self.phases

    @property
    def phase_index(self) -> np.ndarray:
        """Winning crystal at each probe position.

        Returns
        -------
        np.ndarray
            ``(scan_row, scan_col)`` index into :attr:`names`, or -1 where
            the null hypothesis in :meth:`fit` found too little diffracted
            signal to name a crystal.
        """
        return self._require_fit("phase_index").phase_index.numpy()

    def signal_confidence(self, signal_range="auto", gamma: float = 1.0) -> np.ndarray:
        """Confidence in [0, 1] that a crystal is present, from the data alone.

        The measured intensity beyond the direct beam, scaled to [0, 1].
        Vacuum and amorphous support diffract nothing, so they score zero
        however well some orientation happens to correlate -- which the
        correlation itself cannot tell you, since it saturates on sparse
        patterns.

        Parameters
        ----------
        signal_range : tuple or "auto", default="auto"
            Diffracted intensity mapped to 0 ... 1. "auto" spans zero to half
            the median over the indexed positions.
        gamma : float, default=1.0
            Exponent applied to the result. The default is the raw confidence,
            which is what a threshold should be taken on.

        Returns
        -------
        np.ndarray
            ``(scan_row, scan_col)`` confidence in [0, 1].
        """
        return self._require_fit("signal_confidence()").signal_confidence(signal_range, gamma)

    def mask(self, phase=None, signal_range="auto", gamma: float = SHADE_GAMMA) -> np.ndarray:
        """Display mask in [0, 1] for one crystal, or for all indexed positions.

        The phase decision times the diffracted-signal confidence: positions
        of another crystal, and positions with nothing there, are zero.

        Parameters
        ----------
        phase : int or str, optional
            Crystal index or name. None (default) keeps every indexed
            position, whichever crystal won.
        signal_range : tuple or "auto"
            Passed to :meth:`signal_confidence`. Set it to override the
            automatic brightness range, e.g. (0, 500).
        gamma : float, default=:data:`~quantem.diffraction.phase.SHADE_GAMMA`
            Brightness exponent, the same one :meth:`plot_phase` shades with,
            so a masked orientation map and the phase map agree. Diffracted
            intensity is strongly skewed, so the default lifts the faint
            positions; 1.0 gives the linear scale. Zero stays zero, so vacuum
            is black at any value.

        Returns
        -------
        np.ndarray
            ``(scan_row, scan_col)`` mask in [0, 1].

        Raises
        ------
        KeyError
            If `phase` names no crystal in this map.
        """
        conf = self.signal_confidence(signal_range, gamma)
        if phase is None:
            return conf
        i = self._phase_indices(phase)[0]
        return (self.phase_index == i) * conf

    def phase_fractions(self) -> dict[str, float]:
        """Fraction of the scan won by each crystal, plus the unindexed share.

        These are area fractions of the phase decision, counted over every
        probe position. The per-position model weights are
        `phases.crystal_weights`.

        Returns
        -------
        dict[str, float]
            "unindexed" and one entry per crystal name; the values sum to 1.
        """
        ph = self.phase_index
        out = {"unindexed": float((ph == -1).mean())}
        for i, n in enumerate(self.names):
            out[n] = float((ph == i).mean())
        return out

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------

    def plot_phase(self, **kwargs):
        """Map of which crystal won at each probe position.

        Color gives the crystal, brightness gives the evidence. Positions
        the null hypothesis left unindexed in :meth:`fit` -- vacuum,
        amorphous support, anything that diffracts nothing -- are black.

        Parameters
        ----------
        shade_by : {"signal", "reliability", "none"}, default="signal"
            What the brightness means. "signal" fades by the measured
            diffracted intensity, so the map shows where crystals are.
            "reliability" uses the cost gap to the best model without the
            winning crystal, which answers which phase rather than whether
            there is one.
        shade_range : tuple or "auto", default="auto"
            Values mapped to black ... full color.
        shade_gamma : float, default=0.5
            Exponent applied to the brightness. Diffracted intensity is
            strongly skewed, so below 1 lifts the faint positions and 1.0 is
            the linear scale. Vacuum stays black at any value.
        majority_filter : int, default=0
            Radius in probe positions of a majority filter on the phase
            decision, for display only. 1 replaces each position by the most
            common phase in its 3x3 neighbourhood, dropping isolated single
            positions without moving a real boundary.
        phase_colors : np.ndarray, optional
            One RGB color per crystal.
        scalebar : dict, "auto" or None, default="auto"
            Real-space scale bar; "auto" takes the scan step carried by the
            peaks.
        **kwargs
            Further arguments of :meth:`PhaseMap.plot_phase`.

        Returns
        -------
        tuple
            ``(fig, ax)``.

        Raises
        ------
        ValueError
            If :meth:`fit` has not been run.
        """
        return self._require_fit("plot_phase()").plot_phase(**kwargs)

    def plot_orientation(
        self,
        direction=("z", "r"),
        phase=None,
        mask=None,
        signal_range="auto",
        shade_gamma: float = SHADE_GAMMA,
        **kwargs,
    ):
        """Inverse pole figure maps of every crystal, masked by the phase decision.

        Parameters
        ----------
        direction : str or sequence of str, default=("z", "r")
            Out-of-plane, in-plane, or both.
        phase : int or str, optional
            Restrict to one crystal. None (default) plots all of them.
        mask : np.ndarray, optional
            Overrides the automatic phase-and-signal mask entirely.
        signal_range : tuple or "auto", default="auto"
            Diffracted intensity mapped to black ... full color. Set it to
            override the automatic range, e.g. (0, 500).
        shade_gamma : float, default=:data:`~quantem.diffraction.phase.SHADE_GAMMA`
            Brightness exponent of that mask, the same one :meth:`plot_phase`
            shades with. Below 1 lifts the faint positions, 1.0 is the linear
            scale. Both are ignored when `mask` is given.
        **kwargs
            Passed to :meth:`OrientationMap.plot_orientation`, e.g. `smooth`,
            and `saturation_power` and `chroma` for the color wedge.

        Returns
        -------
        tuple or list of tuple
            A single ``(fig, ax)`` when `phase` names one crystal and one
            direction is given; otherwise one ``(fig, ax)`` per crystal and
            direction, the same rule as :meth:`plot_pole_figure`.
        """
        dirs = [direction] if isinstance(direction, str) else list(direction)
        out = []
        for i in self._phase_indices(phase):
            m = mask if mask is not None else self.mask(i, signal_range, shade_gamma)
            for d in dirs:
                out.append(
                    self.orientation_maps[i].plot_orientation(direction=d, mask=m, **kwargs)
                )
        return out[0] if phase is not None and len(out) == 1 else out

    def plot_pole_figure(
        self,
        pole=(0, 0, 1),
        phase=None,
        mask=None,
        signal_range="auto",
        shade_gamma: float = SHADE_GAMMA,
        **kwargs,
    ):
        """Stereographic pole figure of each crystal, masked by the phase decision.

        Parameters
        ----------
        pole : tuple of int, default=(0, 0, 1)
            Crystal direction plotted, in Miller indices [uvw] or [uvtw] of
            each crystal.
        phase : int or str, optional
            Restrict to one crystal. None (default) plots all of them.
        mask : np.ndarray, optional
            Overrides the automatic phase-and-signal mask entirely.
        signal_range : tuple or "auto", default="auto"
            Diffracted intensity mapped to black ... full color.
        shade_gamma : float, default=:data:`~quantem.diffraction.phase.SHADE_GAMMA`
            Brightness exponent of that mask. Ignored when `mask` is given.
        **kwargs
            Passed to :meth:`OrientationMap.plot_pole_figure`, e.g.
            `color_by`, `int_range` and `overlay`.

        Returns
        -------
        tuple or list of tuple
            A single ``(fig, ax)`` when `phase` names one crystal; otherwise
            one ``(fig, ax)`` per crystal, the same rule as
            :meth:`plot_orientation`.
        """
        out = []
        for i in self._phase_indices(phase):
            m = mask if mask is not None else self.mask(i, signal_range, shade_gamma)
            out.append(self.orientation_maps[i].plot_pole_figure(pole=pole, mask=m, **kwargs))
        return out[0] if phase is not None and len(out) == 1 else out

    def plot_matches(self, positions, phase=None, **kwargs):
        """Matched patterns at a few probe positions, over the measured peaks.

        One panel per candidate, with the measured peaks as gray disks and
        the simulated pattern as colored markers, both sized by intensity.
        Passing a `dataset` puts the recorded pattern behind them instead;
        the pixel size and the fitted origins then come from the peaks, which
        carry them from the dataset through the calibration, so neither needs
        passing. A position matched by nothing is drawn with its peaks alone
        and labelled "no match".

        Parameters
        ----------
        positions : list of tuple of int
            ``(row, col)`` probe positions, one panel row each.
        phase : int or str, optional
            Restrict to one crystal. None (default) shows all of them.
        matches : tuple of int, default=(0, 1)
            Which matches of each crystal to draw; indices a crystal does not
            hold are skipped, so the default draws one panel per crystal after
            `num_matches=1`. With `num_matches` of 2, (0, 1) shows the best
            and the residual match side by side, which is how a probe
            straddling two grains shows itself.
        dataset : Dataset4dstem, optional
            Show the recorded diffraction pattern behind the overlay.
        norm : dict or str, optional
            Passed to `show_2d`, which draws that pattern, e.g.
            {"power": 0.5, "upper_quantile": 0.98}.
        measured_scale, measured_power : float, optional
            Size and intensity compression of the gray measured peaks.
        q_max_plot : float, optional
            Half-width of every panel, 1/Angstroms. The default fits it to
            the peaks plotted.
        q_max_quantile : float, default=0.98
            Quantile of the measured peak radii setting that automatic limit.
            A few stray high-angle detections would otherwise set the scale
            for every panel and leave the pattern surrounded by empty space.
            Lower it to crop in further; 1.0 encloses every peak.
        transpose_plots : bool, default=False
            Panel layout only: rows are positions unless this is True.
        **kwargs
            Further arguments of
            :func:`~quantem.diffraction.orientation_visualization.plot_pattern_matches`.

        Returns
        -------
        tuple
            ``(fig, axs)``.
        """
        from quantem.diffraction.orientation_visualization import plot_pattern_matches

        oms = [self.orientation_maps[i] for i in self._phase_indices(phase)]
        md = self.peaks.metadata or {}
        if kwargs.get("dataset") is not None:
            if md.get("pixel_size") is not None:
                kwargs.setdefault("pixel_size", float(md["pixel_size"]))
            if md.get("origins") is not None:
                kwargs.setdefault("origins", np.asarray(md["origins"]))
        return plot_pattern_matches(oms, positions=positions, **kwargs)

    def plot_ring_comparison(self, k_min: float = 0.1, k_max: float | None = None, **kwargs):
        """Measured radial peak distribution against the rings of every crystal.

        One panel per candidate: the red fill is the histogram of every
        calibrated peak, the black lines are that crystal's ring positions.
        Run it before matching. With the scale fixed by a standard, a ring
        that sits beside the measured peaks means either the reference
        lattice parameter is wrong for this specimen or the calibration did
        not transfer -- and matching cannot recover from either.

        Parameters
        ----------
        k_min : float, default=0.1
            Smallest scattering vector shown, 1/Angstroms.
        k_max : float, optional
            Largest scattering vector shown; defaults to the map's own k_max.
        k_broadening : float, optional
            Broaden the rings into a simulated profile; None (default) draws
            sharp lines.
        **kwargs
            Further arguments of
            :func:`~quantem.diffraction.calibration.plot_ring_comparison`.

        Returns
        -------
        tuple
            ``(fig, axs)``.
        """
        from quantem.diffraction import calibration

        return calibration.plot_ring_comparison(
            self.peaks,
            [om.crystal for om in self.orientation_maps],
            k_min=k_min,
            k_max=k_max if k_max is not None else self._k_max_or_crystals(),
            **kwargs,
        )

    def plot_calibration(self, k_min: float = 0.05, k_max: float | None = None, **kwargs):
        """Calibrated peaks against the rings of every crystal in the map.

        One column per crystal: the radial histogram of every peak against
        that crystal's rings, and every peak as azimuth against scattering
        vector with the same rings. Run it after calibrating, whichever
        phase the calibration was fit to: every candidate should line up,
        and one that does not has the wrong lattice parameter for this
        specimen, which matching cannot recover from.

        Parameters
        ----------
        k_min : float, default=0.05
            Smallest scattering vector shown, 1/Angstroms.
        k_max : float, optional
            Largest scattering vector shown; defaults to the map's own k_max.
        **kwargs
            Further arguments of
            :func:`~quantem.diffraction.calibration.plot_calibration`, e.g.
            `zone_axis` to show only the rings of one zone, `k_broadening`
            and `marker_size`.

        Returns
        -------
        tuple
            ``(fig, axs)``, axs of shape (2, number of crystals).
        """
        from quantem.diffraction import calibration

        return calibration.plot_calibration(
            self.peaks,
            [om.crystal for om in self.orientation_maps],
            k_min=k_min,
            k_max=k_max if k_max is not None else self._k_max_or_crystals(),
            **kwargs,
        )

    def plot_bragg_rings(self, **kwargs):
        """Bragg vector map of every peak with the rings of every crystal.

        Parameters
        ----------
        **kwargs
            Arguments of
            :func:`~quantem.diffraction.calibration.plot_bragg_rings`, e.g.
            `n_rings`, `q_max` and `zone_axis`.

        Returns
        -------
        tuple
            ``(fig, ax)``.
        """
        from quantem.diffraction import calibration

        return calibration.plot_bragg_rings(
            self.peaks, [om.crystal for om in self.orientation_maps], **kwargs
        )

    def _k_max_or_crystals(self) -> float:
        """Scattering-vector limit, falling back to the crystals\' own values.

        Maps saved before ``k_max`` lived on the CrystalMap carry it only on
        their crystals, so this keeps those files loadable.

        Returns
        -------
        float
        """
        k = getattr(self, "k_max", None)
        if k is None:
            k = max(float(om.crystal.k_max or 1.5) for om in self.orientation_maps)
        return float(k)

    def plot_correlation(self, **kwargs):
        """Correlation and reliability of every crystal, one panel each.

        The top row is the best correlation of each crystal, the bottom row
        its reliability. Read them with care: the correlation is a cosine
        similarity, so it saturates on patterns carrying only a few peaks
        and stays high on the substrate, and the reliability compares
        crystals rather than testing whether one is there at all. Use
        :meth:`plot_phase` or :meth:`signal_confidence` for that.

        Parameters
        ----------
        mask : bool, default=False
            If True, multiply every panel by :meth:`signal_confidence`, so
            positions with no diffracted signal go to zero.
        shared_scale : bool, default=True
            Put every crystal on one scale, so the panels can be compared
            directly. Each row keeps its own range, since correlation and
            reliability are different quantities. False lets each panel
            autoscale, which shows the structure within a weak crystal at
            the cost of comparability. Passing `norm` overrides both.
        **kwargs
            Passed to :func:`~quantem.core.visualization.show_2d`.

        Returns
        -------
        tuple
            ``(fig, axs)``.
        """
        from quantem.core.visualization import show_2d

        mask = kwargs.pop("mask", False)
        shared_scale = kwargs.pop("shared_scale", True)
        corr = [om.corr[..., 0].numpy() for om in self.orientation_maps]
        rel = [om.reliability.numpy() for om in self.orientation_maps]
        if mask:
            conf = self.signal_confidence()
            corr = [c * conf for c in corr]
            rel = [r * conf for r in rel]
        if shared_scale and "norm" not in kwargs:
            # one scale per row, so the crystals are directly comparable;
            # correlation and reliability keep their own ranges
            kwargs["norm"] = [
                [
                    {
                        "interval_type": "manual",
                        "vmin": float(min(np.nanmin(a) for a in row)),
                        "vmax": float(max(np.nanmax(a) for a in row)),
                    }
                ]
                * len(row)
                for row in (corr, rel)
            ]
        kwargs.setdefault("cbar", True)
        # panels shaped like the scan, so a wide map leaves no gap between rows
        R, C = self.shape
        kwargs.setdefault("axsize", (4.5, 4.5 * R / C))
        kwargs.setdefault(
            "title",
            [
                [f"{n} correlation" for n in self.names],
                [f"{n} reliability" for n in self.names],
            ],
        )
        sb = self.orientation_maps[0].scan_scalebar
        if sb is not None:
            kwargs.setdefault("scalebar", [[sb] + [False] * (len(corr) - 1), [False] * len(corr)])
        return show_2d([corr, rel], **kwargs)

    def _phase_indices(self, phase) -> list[int]:
        """Resolve a crystal selector to a list of indices.

        Parameters
        ----------
        phase : int, str or None
            One crystal by index or name, or None for all of them.

        Returns
        -------
        list of int

        Raises
        ------
        KeyError
            If `phase` names no crystal in this map, by name or index.
        """
        if phase is None:
            return list(range(len(self.orientation_maps)))
        if isinstance(phase, str):
            if phase not in self.names:
                raise KeyError(f"no crystal named {phase!r}; have {self.names}")
            return [self.names.index(phase)]
        i = int(phase)
        if not -len(self.names) <= i < len(self.names):
            raise KeyError(f"no crystal {i}; have {len(self.names)}")
        return [i % len(self.names)]

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def save(self, path, mode: str = "w", include_plan: bool = False, **kwargs):
        """Save the whole analysis to one file.

        Load it back with :func:`quantem.core.io.serialize.load`.

        Parameters
        ----------
        path : str or Path
            Target path; a ".zip" extension writes one zip file, anything
            else a directory.
        mode : {"w", "o"}, default="w"
            "w" refuses to replace an existing file, "o" overwrites it.
        include_plan : bool, default=False
            The correlation plan dominates the file size and is rebuilt in
            seconds by :meth:`build_plan`, so it is dropped by default. Pass
            True to keep it and reload a map ready to match again.
        **kwargs
            Passed to :meth:`AutoSerialize.save`, e.g. `compression_level`.
        """
        if include_plan:
            return AutoSerialize.save(self, path, mode=mode, **kwargs)
        stash = [(om, om.plan_fft) for om in self.orientation_maps]
        try:
            for om, _ in stash:
                om.plan_fft = None
            return AutoSerialize.save(self, path, mode=mode, **kwargs)
        finally:
            for om, plan in stash:
                om.plan_fft = plan
