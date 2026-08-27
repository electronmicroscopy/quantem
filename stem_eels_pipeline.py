"""
stem_eels_pipeline.py

Generic loader + analysis pipeline for monochromated STEM-EELS data collected on
TEAM I, built on `quantem` with **no hyperspy dependency**. Handles two
acquisition styles automatically:

1. "Single-pass" datasets, like the one in OMIEC_Lamella_EELS_O_10.ipynb: the
   STEM SI.dm4 file (plus its LL/HL raw sidecars) already contains a single,
   already-summed spectrum image. These are read directly with
   `quantem.io.read_3d_spectroscopy`, exactly as in that notebook.

2. "Multi-pass" in-situ datasets, like InSitu (8) / InSitu (9): DigitalMicrograph's
   in-situ / multi-pass scanning mode records every individual pass as its own
   frame in the raw sidecar file (see Loading_in_situ_DM_files.ipynb, which does
   this today via ncempy + hyperspy). This module detects that case from the
   DM4 tags, loads the full frame stack with `ncempy` only (no hyperspy), asks
   which passes to keep, sums/averages them, and hands the combined array to
   quantem for the same general analysis used in OMIEC_Lamella_EELS_O_10.ipynb
   (ZLP correction, thickness mapping, background subtraction, energy-window
   mapping, PCA).

`quantem.io.read_3d_spectroscopy` (as used in OMIEC_Lamella_EELS_O_10.ipynb)
only reads directly from a DM4/raw pair on disk. There is no separate
"array-based" constructor in quantem for "I already have a calibrated numpy
array in memory, wrap it as a 3D spectroscopy dataset" -- but there doesn't
need to be one: `read_3d_spectroscopy` itself just builds a
`Dataset3deels`/`Dataset3dxeds` via `.from_array(array=..., sampling=...,
origin=..., units=...)` (see
quantem/core/io/file_readers.py:read_3d_spectroscopy), and `from_array()` is
public API on every quantem Dataset subclass. `array_to_spectroscopy3d()`
below does exactly that -- verified against quantem's actual source in this
checkout (quantem/core/datastructures/dataset.py,
quantem/core/datastructures/dataset3d.py,
quantem/spectroscopy/dataset3dspectroscopy.py,
quantem/spectroscopy/dataset3deels.py).

`_get_dm4_calibration()` no longer guesses DM4 tag path strings. It instead
reuses ncempy's own already-parsed per-object calibration arrays
(`fileDM.scale`/`.origin`/`.scaleUnit`/`.dataShape`) -- the same values
`ncempy.io.dm.fileDM.getDataset()`/`dmReader()` use internally to build the
energy-loss axis for spectra -- so it tracks whatever tag-path spellings
ncempy itself already handles, instead of hardcoding one guess. See the
function's docstring for the indexing details.

Everything else -- multipass detection, raw multi-frame loading, pass
selection/combination, and the analysis call sequence -- is adapted directly
from your two existing notebooks and should run as-is wherever quantem +
ncempy are already set up (e.g. on mallard).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover
    gaussian_filter = None


# --------------------------------------------------------------------------- #
# Soft imports for the heavy, environment-specific dependencies
# --------------------------------------------------------------------------- #


def _import_quantem():
    try:
        import quantem as em
    except ImportError as e:
        raise ImportError(
            "quantem is not installed/importable in this environment. This "
            "pipeline is meant to run wherever quantem + ncempy are set up "
            "(e.g. your mallard VS Code environment), not in a generic sandbox."
        ) from e
    return em


def _import_ncempy_dm():
    try:
        import ncempy.io as nio
    except ImportError as e:
        raise ImportError(
            "ncempy is not installed/importable in this environment. It is "
            "required to read the DM4 tags / raw sidecars (see "
            "Loading_in_situ_DM_files.ipynb, which uses the same package)."
        ) from e
    return nio


# --------------------------------------------------------------------------- #
# DM4 tag inspection helpers
# --------------------------------------------------------------------------- #


def list_dm4_objects(dm4_path: Union[str, Path]) -> List[Tuple[int, str]]:
    """List the (0-indexed) objects stored in a DM4 file, e.g.

        [(0, 'Thumbnail'), (1, 'Survey Image'), (2, 'STEM SI_ADF Image'),
         (3, 'STEM SI_EELS LL SI'), (4, 'STEM SI_EELS HL SI')]

    Same idea as cell 4 of Loading_in_situ_DM_files.ipynb, generalized to any
    number of objects.
    """
    nio = _import_ncempy_dm()
    dm0 = nio.dm.fileDM(str(dm4_path))
    names = []
    for i in range(dm0.numObjects):
        name = dm0.allTags.get(f".ImageList.{i + 1}.Name", f"<object {i}: no Name tag>")
        names.append((i, name))
    return names


def inspect_dm4_tags(dm4_path: Union[str, Path], contains: Optional[str] = None) -> Dict[str, Any]:
    """Dump (optionally filtered) DM4 tags for quick, ad-hoc debugging in a
    notebook, e.g.:

        inspect_dm4_tags(dm4_path, contains="In-situ")
        inspect_dm4_tags(dm4_path, contains="Calibrat")
    """
    nio = _import_ncempy_dm()
    dm0 = nio.dm.fileDM(str(dm4_path))
    if contains is None:
        return dict(dm0.allTags)
    contains_l = contains.lower()
    return {k: v for k, v in dm0.allTags.items() if contains_l in k.lower()}


def _get_object_tag(dm0, obj_idx: int, *tag_path_suffixes: str, default=None):
    """Look up a tag for a 0-indexed DM4 object, trying multiple plausible tag
    path spellings in order (DM's exact tag names shift a bit across
    versions/acquisition modes)."""
    tag_idx = obj_idx + 1
    for suffix in tag_path_suffixes:
        key = f".ImageList.{tag_idx}.{suffix}"
        if key in dm0.allTags:
            return dm0.allTags[key]
    return default


def _dtype_from_bytes(nbytes) -> np.dtype:
    mapping = {1: np.uint8, 2: np.uint16, 4: np.float32, 8: np.float64}
    if nbytes is None:
        # every in-situ raw sidecar we've seen from this instrument is float32
        return np.dtype(np.float32)
    return np.dtype(mapping.get(int(nbytes), np.float32))


@dataclass
class DM4ObjectInfo:
    index: int
    name: str
    n_frames: int
    dims: Tuple[int, ...]  # DM order: fastest-varying axis first (dim0, dim1, [dim2])
    dtype: np.dtype
    is_multipass: bool


def detect_multipass(
    dm4_path: Union[str, Path], object_hint: Sequence[str] = ("EELS", "ADF")
) -> Dict[str, DM4ObjectInfo]:
    """
    THE "IS THIS MULTI-PASS?" CHECK.

    Inspects every DM4 object whose name matches `object_hint` and reports
    whether it was recorded in DM's in-situ / multi-pass mode, i.e. whether the
    raw sidecar stores one frame per pass rather than a single already-summed
    frame. Mirrors the tag lookup in Loading_in_situ_DM_files.ipynb
    (`.ImageList.N.ImageTags.In-situ.Recorded.# Frames`), generalized across
    every matching object instead of one hardcoded index, and with a fallback
    tag spelling in case your DM version differs.

    Returns a dict keyed by object name -> DM4ObjectInfo.
    """
    nio = _import_ncempy_dm()
    dm0 = nio.dm.fileDM(str(dm4_path))

    results: Dict[str, DM4ObjectInfo] = {}
    for i in range(dm0.numObjects):
        name = dm0.allTags.get(f".ImageList.{i + 1}.Name", "")
        if object_hint and not any(h.lower() in name.lower() for h in object_hint):
            continue

        n_frames = _get_object_tag(
            dm0,
            i,
            "ImageTags.In-situ.Recorded.# Frames",
            "ImageTags.In-situ.Number of frames",  # fallback spelling
            default=1,
        )
        dtype_bytes = _get_object_tag(
            dm0, i, "ImageTags.In-situ.Raw File Format Info.Data Size (bytes)"
        )
        dims: List[int] = []
        d = 1
        while True:
            val = _get_object_tag(dm0, i, f"ImageTags.In-situ.Raw File Format Info.Dimensions.{d}")
            if val is None:
                break
            dims.append(int(val))
            d += 1

        n_frames = int(n_frames) if n_frames else 1
        results[name] = DM4ObjectInfo(
            index=i,
            name=name,
            n_frames=n_frames,
            dims=tuple(dims),
            dtype=_dtype_from_bytes(dtype_bytes),
            is_multipass=n_frames > 1,
        )
    return results


# --------------------------------------------------------------------------- #
# Locating the DM4 + raw sidecar files for one acquisition folder
# --------------------------------------------------------------------------- #


def find_stem_si_files(folder: Union[str, Path]) -> Dict[str, Optional[Path]]:
    """
    Locate the DM4 header + raw sidecars for a STEM SI acquisition folder like
    'InSitu (8)', following the naming convention seen in your TEAM I exports:

        STEM SI.dm4
        STEM SI_ADF Image.raw
        STEM SI_EELS HL SI.raw
        STEM SI_EELS LL SI.raw
        ADF ImagePostAcquisition.dm4   (separate post-acquisition survey image;
                                         ignored here, not part of the SI stack)
    """
    folder = Path(folder)
    dm4_candidates = sorted(folder.glob("*SI.dm4")) or sorted(folder.glob("*.dm4"))
    dm4_candidates = [
        p for p in dm4_candidates if "postacq" not in p.name.lower()
    ] or dm4_candidates
    if not dm4_candidates:
        raise FileNotFoundError(f"No .dm4 file found in {folder}")
    dm4_path = dm4_candidates[0]
    stem = dm4_path.stem  # e.g. "STEM SI"

    def _sidecar(suffix: str) -> Optional[Path]:
        cands = list(folder.glob(f"{stem}{suffix}"))
        return cands[0] if cands else None

    return {
        "dm4": dm4_path,
        "adf_raw": _sidecar("_ADF Image.raw"),
        "eels_hl_raw": _sidecar("_EELS HL SI.raw"),
        "eels_ll_raw": _sidecar("_EELS LL SI.raw"),
    }


def describe_folder(folder: Union[str, Path]) -> Dict[str, DM4ObjectInfo]:
    """Quick, read-only summary of an acquisition folder: which files were
    found and whether it's multi-pass. Handy to run before committing to a
    full load."""
    files = find_stem_si_files(folder)
    print(f"Folder: {folder}")
    for k, v in files.items():
        print(f"  {k}: {v.name if v else '(not found)'}")
    info = detect_multipass(files["dm4"])
    for name, o in info.items():
        tag = "MULTI-PASS" if o.is_multipass else "single-pass"
        print(f"  [{tag}] {name}: frames={o.n_frames}, dims={o.dims}, dtype={o.dtype}")
    return info


# --------------------------------------------------------------------------- #
# Multi-pass raw stack loading (ncempy only -- no hyperspy)
# --------------------------------------------------------------------------- #


def load_raw_stack(
    raw_path: Union[str, Path], n_frames: int, dims: Tuple[int, ...], dtype=np.float32
) -> np.ndarray:
    """
    Load a DM in-situ raw sidecar as a (n_frames, ...) stack, without
    hyperspy -- adapted from cells 6-14 of Loading_in_situ_DM_files.ipynb.

    `dims` is (dim0, dim1[, dim2]) exactly as read from the DM4
    'Raw File Format Info.Dimensions.N' tags (fastest-varying axis first, DM
    convention). The returned array has the frame axis first and the DM
    dimensions reversed (slowest-varying first) -- e.g. for a spectrum image
    stack with dims=(nx, ny, n_energy) you get back shape
    (n_frames, ny, nx, n_energy).
    """
    raw_path = str(raw_path)
    with open(raw_path, "rb") as f:
        arr = np.fromfile(f, dtype=dtype, count=-1)
    shape = (n_frames,) + tuple(reversed(dims))
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(
            f"{raw_path}: read {arr.size} elements but expected {expected} for "
            f"shape {shape} (n_frames={n_frames}, dims={dims}, dtype={dtype}). "
            f"Re-check these against inspect_dm4_tags(dm4_path, contains='In-situ')."
        )
    return arr.reshape(shape)


# --------------------------------------------------------------------------- #
# Pass selection ("ask user which passes to take for analysis")
# --------------------------------------------------------------------------- #


def _parse_pass_spec(spec: Union[str, Sequence[int]], n_frames: int) -> List[int]:
    """
    Parse a pass-selection spec into a sorted list of 0-indexed frame numbers.
    Pass numbers themselves are 1-indexed (matching how passes are counted on
    the microscope / in your acquisition log), e.g.:

        'all'            -> every recorded pass
        '5'              -> just pass 5
        '1-15'           -> passes 1 through 15 inclusive
        '1,3,5'          -> specific passes
        '1-5,10,20-25'   -> mix of ranges and singles
        [1, 2, 3]        -> a list/tuple of 1-indexed pass numbers directly
    """
    if isinstance(spec, str):
        spec = spec.strip()
        if spec.lower() == "all":
            return list(range(n_frames))
        chosen = set()
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                a, b = chunk.split("-")
                chosen.update(range(int(a), int(b) + 1))
            else:
                chosen.add(int(chunk))
        passes_1idx = sorted(chosen)
    else:
        passes_1idx = sorted(int(p) for p in spec)

    bad = [p for p in passes_1idx if p < 1 or p > n_frames]
    if bad:
        raise ValueError(f"Pass number(s) {bad} out of range 1..{n_frames}")
    return [p - 1 for p in passes_1idx]


def select_passes(
    n_frames: int,
    mode: str = "ask",
    passes: Optional[Union[str, Sequence[int]]] = None,
    prompt_label: str = "passes",
) -> List[int]:
    """
    Decide which of the `n_frames` recorded passes to use for analysis.

    mode="ask"    -> interactively prompt (input()) for a pass spec. This is
                      the default so a plain call from a notebook always asks.
    mode="all"    -> use every pass.
    mode="manual" -> use `passes` (parsed the same way as the interactive
                      prompt) without asking -- for scripted/batch use.
    """
    if mode == "all":
        return list(range(n_frames))
    if mode == "manual":
        if passes is None:
            raise ValueError("mode='manual' requires the `passes` argument")
        return _parse_pass_spec(passes, n_frames)
    if mode == "ask":
        while True:
            raw = input(
                f"This dataset has {n_frames} {prompt_label} (1-indexed). "
                f"Which would you like to use for analysis?\n"
                f"  Enter 'all', a single number, a range like '1-15', or a "
                f"comma list like '1-10,20,25-30': "
            ).strip()
            if not raw:
                print("  -> please enter something, e.g. 'all'")
                continue
            try:
                chosen = _parse_pass_spec(raw, n_frames)
                print(f"  -> using {len(chosen)}/{n_frames} passes")
                return chosen
            except ValueError as e:
                print(f"  -> {e}; try again")
    raise ValueError(f"Unknown mode {mode!r}")


def combine_passes(
    stack: np.ndarray, indices: Sequence[int], method: str = "sum", pass_axis: int = 0
) -> np.ndarray:
    """
    Collapse the selected passes down to a single frame.

    method='sum' (default) is the physically correct combination for raw EELS
    counts -- it keeps downstream SNR/thickness calculations meaningful, the
    same way DM itself would combine passes during acquisition. method='mean'
    just rescales to a per-pass average.
    """
    sub = np.take(stack, indices, axis=pass_axis)
    if method == "sum":
        return sub.sum(axis=pass_axis)
    elif method == "mean":
        return sub.mean(axis=pass_axis)
    raise ValueError("method must be 'sum' or 'mean'")


# --------------------------------------------------------------------------- #
# DM4 calibration extraction (via ncempy's parsed arrays -- see module docstring)
# --------------------------------------------------------------------------- #


def _get_dm4_calibration(
    dm4_path: Union[str, Path], obj_idx: int, n_dims: int, energy_rank: Optional[int] = None
) -> Tuple[List[float], List[float], List[str]]:
    """
    Read (origin, scale, units) for each dimension of a DM4 object, 0-indexed
    with the spectral/energy axis last (matching how load_raw_stack orders
    the returned array).

    IMPORTANT CAVEAT, found by running this against a real multi-pass in-situ
    acquisition (TEAM I "InSitu (9)"): ncempy's per-object calibration arrays
    (`fileDM.scale` / `.origin` / `.scaleUnit` / `.dataShape`) do **not**
    always align 1:1, in file order, with `.ImageList` object index. That
    file had 5 named ImageList objects but 8 low-level calibrated blocks
    (extra, unnamed blocks interleaved -- likely per-object preview/ancillary
    data) -- so `dm0.dataShape[obj_idx]` does not reliably tell you how many
    calibrated dims *this* named object has, and ncempy's own public
    `fileDM.getDataset()` (which indexes the same underlying arrays) hit the
    same problem: it could retrieve only one of the two EELS objects'
    calibration before raising an internal IndexError.

    What *is* structurally guaranteed by the DM4 format is order: a DM4 file
    always emits one object's tags contiguously before the next, in
    ascending ImageList order. So the Nth calibrated block whose unit is
    'eV', in on-disk tag order, must belong to the Nth energy-bearing object
    in ascending ImageList order -- this holds even when unrelated
    non-energy blocks are interleaved. `energy_rank` (0-indexed) selects
    that Nth block directly; pass it when the caller already knows this
    object's rank among the acquisition's energy-bearing objects (see
    `_load_multi_pass`, which always resolves EELS LL before EELS HL and
    whose LL object always has a lower ImageList index than HL for this
    acquisition mode).

    The spatial (pixel-size) calibration is comparatively low-stakes here --
    every channel of one STEM-SI acquisition shares the same scan step size
    -- so it's taken as the statistical mode of every recorded non-'eV'
    scale value, which robustly ignores one-off outliers (e.g. an unrelated
    single survey image at a different pixel size).
    """
    nio = _import_ncempy_dm()
    dm0 = nio.dm.fileDM(str(dm4_path))

    if obj_idx < 0 or obj_idx >= dm0.numObjects:
        raise ValueError(
            f"DM4 object index {obj_idx} out of range for {dm4_path} "
            f"({dm0.numObjects} objects found)."
        )

    scales_all = [float(s) for s in dm0.scale]
    origins_all = [float(o) for o in dm0.origin]
    units_all = [str(u) for u in dm0.scaleUnit]

    ev_positions = [i for i, u in enumerate(units_all) if u.strip().lower() == "ev"]
    spatial_positions = [i for i, u in enumerate(units_all) if u.strip().lower() != "ev"]
    if not spatial_positions:
        raise ValueError(f"No spatial (non-eV) calibration entries found in {dm4_path}.")

    from collections import Counter

    spatial_scale = Counter(scales_all[i] for i in spatial_positions).most_common(1)[0][0]
    spatial_unit = next(units_all[i] for i in spatial_positions if scales_all[i] == spatial_scale)

    origins = [0.0] * (n_dims - 1)
    scales = [spatial_scale] * (n_dims - 1)
    units = [spatial_unit] * (n_dims - 1)

    if n_dims >= 1:
        if energy_rank is None:
            raise ValueError(
                "_get_dm4_calibration() requires an explicit `energy_rank` to "
                "locate the energy axis -- see its docstring for why this "
                "can't be inferred from `obj_idx` alone."
            )
        if energy_rank < 0 or energy_rank >= len(ev_positions):
            raise ValueError(
                f"Requested energy_rank={energy_rank} but only "
                f"{len(ev_positions)} 'eV'-unit calibration block(s) were "
                f"found in {dm4_path}. Run inspect_dm4_tags(dm4_path, "
                f"contains='Calibrat') to double-check."
            )
        k = ev_positions[energy_rank]
        energy_scale = scales_all[k]
        # DM stores the origin as a pixel offset, not a calibrated value;
        # ncempy computes the calibrated origin as -pixelOrigin * pixelSize
        # (see ncempy.io.dm.dmReader()).
        energy_origin = -origins_all[k] * energy_scale
        origins.append(energy_origin)
        scales.append(energy_scale)
        units.append(units_all[k])

    return origins, scales, units


def _energy_axis_from_dm4(
    dm4_path: Union[str, Path],
    obj_idx: int,
    n_channels: int,
    n_dims: int = 3,
    energy_rank: Optional[int] = None,
) -> np.ndarray:
    origins, scales, units = _get_dm4_calibration(
        dm4_path, obj_idx, n_dims, energy_rank=energy_rank
    )
    origin, scale = origins[-1], scales[-1]
    # matches the coordinate-axis construction in ncempy.io.dm.dmReader()
    return np.linspace(0, scale * (n_channels - 1), n_channels) + origin


def _pixel_size_nm_from_dm4(
    dm4_path: Union[str, Path], obj_idx: int, n_dims: int = 3, energy_rank: Optional[int] = None
) -> Optional[float]:
    origins, scales, units = _get_dm4_calibration(
        dm4_path, obj_idx, n_dims, energy_rank=energy_rank
    )
    # first spatial dimension; convert to nm if given in a different (but
    # recognizable) unit
    scale, unit = scales[0], (units[0] or "").lower()
    if unit in ("nm", ""):
        return scale
    if unit in ("um", "µm", "micron", "microns"):
        return scale * 1000.0
    warnings.warn(f"Unrecognized spatial unit {unit!r} in DM4 calibration; returning raw scale.")
    return scale


# --------------------------------------------------------------------------- #
# In-memory array -> quantem Dataset3deels (see module docstring)
# --------------------------------------------------------------------------- #


def array_to_spectroscopy3d(
    data: np.ndarray,
    energy_axis: np.ndarray,
    pixel_size_nm: Optional[float] = None,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Wrap an in-memory (ny, nx, n_energy) array (e.g. the result of
    combine_passes()) as a quantem 3D EELS dataset, so it can be passed to
    run_general_eels_analysis() / the same methods used in
    OMIEC_Lamella_EELS_O_10.ipynb (.show_mean_spectrum(), .energy_axis,
    .apply_zlp_correction(), etc.).

    This is exactly the class `quantem.core.io.read_3d_spectroscopy(...,
    data_type="EELS")` returns (see
    quantem/core/io/file_readers.py:read_3d_spectroscopy), just built via
    `Dataset3deels.from_array()` directly from an array instead of from a
    file on disk -- there's no separate "array-based" constructor in
    quantem; `from_array()` on any Dataset subclass already is that
    constructor.

    Dataset3dspectroscopy has no stored `energy_axis` array -- it's a
    property computed on the fly from `origin[2]`/`sampling[2]` (see
    Dataset3dspectroscopy.energy_axis in
    quantem/spectroscopy/dataset3dspectroscopy.py), so `energy_axis` here
    must be uniformly spaced: only its first value and spacing are kept.
    """
    em = _import_quantem()

    energy_axis = np.asarray(energy_axis, dtype=float)
    if energy_axis.ndim != 1 or len(energy_axis) < 2:
        raise ValueError("energy_axis must be a 1D array with at least 2 points")
    if data.ndim != 3 or data.shape[2] != len(energy_axis):
        raise ValueError(
            f"data must be (ny, nx, n_energy) with n_energy == len(energy_axis); "
            f"got data.shape={data.shape}, len(energy_axis)={len(energy_axis)}"
        )
    energy_scale = float(energy_axis[1] - energy_axis[0])

    spatial_scale = float(pixel_size_nm) if pixel_size_nm is not None else 1.0
    spatial_unit = "nm" if pixel_size_nm is not None else "pixels"

    dataset = em.spectroscopy.Dataset3deels.from_array(
        array=data,
        name=name if name is not None else "EELS dataset",
        origin=[0.0, 0.0, float(energy_axis[0])],
        sampling=[spatial_scale, spatial_scale, energy_scale],
        units=[spatial_unit, spatial_unit, "eV"],
    )
    if metadata:
        # Dataset3d.from_array() has no metadata param; attach it the same
        # way quantem's own read_abtem() reader does (file_readers.py).
        dataset._metadata = dict(metadata)
    return dataset


# --------------------------------------------------------------------------- #
# High-level orchestration
# --------------------------------------------------------------------------- #


@dataclass
class StemEelsRaw:
    """Container returned by load_stem_eels_folder()."""

    folder: Path
    dm4_path: Path
    is_multipass: bool
    n_passes: int
    eels_ll: Any
    eels_hl: Any
    adf: Optional[np.ndarray]
    energy_axis_ll: Optional[np.ndarray]
    energy_axis_hl: Optional[np.ndarray]
    pixel_size_nm: Optional[float]
    passes_used: Optional[List[int]]  # 1-indexed pass numbers actually used
    combine_method: Optional[str]


def _load_single_pass(
    dm4_path: Path, eels_infos: Optional[Dict[str, DM4ObjectInfo]] = None
) -> StemEelsRaw:
    em = _import_quantem()
    from rsciio.digitalmicrograph import file_reader

    eels_ll = em.io.read_3d_spectroscopy(
        str(dm4_path), file_type="digitalmicrograph", data_type="EELS"
    )

    hl_dataset_index = 3  # default seen in OMIEC_Lamella_EELS_O_10.ipynb cell 4
    if eels_infos:
        hl = next(
            (o for n, o in eels_infos.items() if "hl" in n.lower() or "high" in n.lower()),
            None,
        )
        if hl is not None:
            hl_dataset_index = hl.index

    eels_hl = em.io.read_3d_spectroscopy(
        str(dm4_path),
        file_type="digitalmicrograph",
        data_type="EELS",
        dataset_index=hl_dataset_index,
    )
    all_data = file_reader(str(dm4_path))
    adf = all_data[1]["data"] if len(all_data) > 1 else None

    return StemEelsRaw(
        folder=dm4_path.parent,
        dm4_path=dm4_path,
        is_multipass=False,
        n_passes=1,
        eels_ll=eels_ll,
        eels_hl=eels_hl,
        adf=adf,
        energy_axis_ll=getattr(eels_ll, "energy_axis", None),
        energy_axis_hl=getattr(eels_hl, "energy_axis", None),
        pixel_size_nm=None,
        passes_used=None,
        combine_method=None,
    )


@dataclass
class MultipassRawStacks:
    """
    Per-pass raw arrays and calibration for a multi-pass in-situ
    acquisition, loaded but NOT combined across passes.

    This is the one shared per-pass access path for anything that needs to
    look at individual passes -- _load_multi_pass() (which sums the selected
    passes for the general analysis pipeline), inspect_single_pass(), and
    any future per-pass function (e.g. damage detection) all build on this
    instead of each re-reading the raw sidecars their own way.

    ll_stack / hl_stack : (n_frames, n_energy, ny, nx) ndarray
        Energy axis is at position 1, not last -- load_raw_stack() reshapes
        each frame as tuple(reversed(dims)), which for these EELS SI raw
        sidecars puts energy right after the frame axis. Verified
        empirically: a real EELS spectrum (varying by >10x across channels)
        only appears when averaging over the *other* two axes.
    adf_stack : (n_frames, ny, nx) ndarray, or None if no ADF raw sidecar
        with a per-pass frame count was found.
    """

    folder: Path
    dm4_path: Path
    files: Dict[str, Optional[Path]]
    obj_info: Dict[str, DM4ObjectInfo]
    ll_info: DM4ObjectInfo
    hl_info: DM4ObjectInfo
    adf_info: Optional[DM4ObjectInfo]
    n_passes: int
    ll_stack: np.ndarray
    hl_stack: np.ndarray
    adf_stack: Optional[np.ndarray]
    ll_energy_axis: np.ndarray
    hl_energy_axis: np.ndarray
    pixel_size_nm: Optional[float]


def load_multipass_raw_stacks(folder: Union[str, Path]) -> MultipassRawStacks:
    """
    Load every recorded pass of a multi-pass in-situ acquisition's EELS
    LL/HL and ADF raw stacks, with calibration, but without combining across
    passes. See MultipassRawStacks for why this exists as a shared function.
    """
    folder = Path(folder)
    files = find_stem_si_files(folder)
    obj_info = detect_multipass(files["dm4"])

    ll_info = next(
        o
        for n, o in obj_info.items()
        if "eels" in n.lower() and ("ll" in n.lower() or "low" in n.lower())
    )
    hl_info = next(
        o
        for n, o in obj_info.items()
        if "eels" in n.lower() and ("hl" in n.lower() or "high" in n.lower())
    )
    n_passes = max(ll_info.n_frames, hl_info.n_frames)
    adf_info = next(
        (
            o
            for n, o in obj_info.items()
            if "adf" in n.lower() and "postacq" not in n.lower() and o.n_frames == n_passes
        ),
        None,
    )

    print(f"  EELS LL: dims={ll_info.dims}, dtype={ll_info.dtype}, frames={ll_info.n_frames}")
    print(f"  EELS HL: dims={hl_info.dims}, dtype={hl_info.dtype}, frames={hl_info.n_frames}")

    ll_stack = load_raw_stack(files["eels_ll_raw"], ll_info.n_frames, ll_info.dims, ll_info.dtype)
    hl_stack = load_raw_stack(files["eels_hl_raw"], hl_info.n_frames, hl_info.dims, hl_info.dtype)
    adf_stack = None
    if adf_info is not None and files["adf_raw"] is not None:
        adf_stack = load_raw_stack(
            files["adf_raw"], adf_info.n_frames, adf_info.dims, adf_info.dtype
        )

    # energy_rank: LL is always written to the DM4 ImageList before HL for
    # this acquisition mode (ll_info.index < hl_info.index), and DM4 objects'
    # tags -- including their calibration blocks -- always appear in that
    # same ascending order on disk. See _get_dm4_calibration()'s docstring
    # for why this ordinal rank is used instead of obj_idx-based lookup.
    assert ll_info.index < hl_info.index, (
        "expected EELS LL to precede EELS HL in the DM4 ImageList; "
        "_energy_axis_from_dm4()'s energy_rank assumption below does not hold "
        "for this file -- re-check with inspect_dm4_tags()."
    )
    ll_energy_axis = _energy_axis_from_dm4(
        files["dm4"], ll_info.index, ll_info.dims[-1], energy_rank=0
    )
    hl_energy_axis = _energy_axis_from_dm4(
        files["dm4"], hl_info.index, hl_info.dims[-1], energy_rank=1
    )
    pixel_size_nm = _pixel_size_nm_from_dm4(files["dm4"], ll_info.index, energy_rank=0)

    return MultipassRawStacks(
        folder=folder,
        dm4_path=files["dm4"],
        files=files,
        obj_info=obj_info,
        ll_info=ll_info,
        hl_info=hl_info,
        adf_info=adf_info,
        n_passes=n_passes,
        ll_stack=ll_stack,
        hl_stack=hl_stack,
        adf_stack=adf_stack,
        ll_energy_axis=ll_energy_axis,
        hl_energy_axis=hl_energy_axis,
        pixel_size_nm=pixel_size_nm,
    )


def _load_multi_pass(
    folder: Path,
    pass_mode: str,
    passes: Optional[Union[str, Sequence[int]]],
    combine_method: str,
) -> StemEelsRaw:
    raw = load_multipass_raw_stacks(folder)

    chosen = select_passes(raw.n_passes, mode=pass_mode, passes=passes)
    print(
        f"  Using {len(chosen)}/{raw.n_passes} passes "
        f"(pass numbers {', '.join(str(p + 1) for p in chosen)}), "
        f"combine method='{combine_method}'"
    )

    # combine_passes() sums over the frame axis (axis 0), leaving energy at
    # axis 0 of the result -- move it to the end to match the (ny, nx,
    # n_energy) layout array_to_spectroscopy3d()/quantem require.
    ll_combined = np.moveaxis(combine_passes(raw.ll_stack, chosen, method=combine_method), 0, -1)
    hl_combined = np.moveaxis(combine_passes(raw.hl_stack, chosen, method=combine_method), 0, -1)
    adf_combined = (
        combine_passes(raw.adf_stack, chosen, method="mean") if raw.adf_stack is not None else None
    )
    # ADF is averaged (not summed) by default since it's an image you want to
    # look at, not a counting signal -- switch to method="sum" if you'd rather
    # see the cumulative dose/drift pattern across the selected passes.

    eels_ll = array_to_spectroscopy3d(
        ll_combined, raw.ll_energy_axis, raw.pixel_size_nm, name="EELS_LL"
    )
    eels_hl = array_to_spectroscopy3d(
        hl_combined, raw.hl_energy_axis, raw.pixel_size_nm, name="EELS_HL"
    )

    return StemEelsRaw(
        folder=Path(folder),
        dm4_path=raw.dm4_path,
        is_multipass=True,
        n_passes=raw.n_passes,
        eels_ll=eels_ll,
        eels_hl=eels_hl,
        adf=adf_combined,
        energy_axis_ll=raw.ll_energy_axis,
        energy_axis_hl=raw.hl_energy_axis,
        pixel_size_nm=raw.pixel_size_nm,
        passes_used=[p + 1 for p in chosen],
        combine_method=combine_method,
    )


def load_stem_eels_folder(
    folder: Union[str, Path],
    pass_mode: str = "ask",
    passes: Optional[Union[str, Sequence[int]]] = None,
    combine_method: str = "sum",
) -> StemEelsRaw:
    """
    THE MAIN ENTRY POINT.

    Given a TEAM-I acquisition folder (e.g. 'InSitu (8)'):
      1. Checks whether it's multi-pass (detect_multipass()).
      2. If not: reads it directly with quantem.io.read_3d_spectroscopy, as in
         OMIEC_Lamella_EELS_O_10.ipynb.
      3. If yes: loads the full per-pass frame stack with ncempy (no
         hyperspy), asks which passes to use (select_passes(), default
         interactive), sums them (combine_passes()), and wraps the result as
         a quantem dataset (array_to_spectroscopy3d()).

    pass_mode/passes/combine_method are forwarded to select_passes() /
    combine_passes() and are only used if the dataset turns out multi-pass.
    Pass mode="ask" (the default) will call input() -- fine in a notebook,
    but pass mode="manual", passes=... (or mode="all") for non-interactive/
    scripted use.

    Returns a StemEelsRaw with .eels_ll / .eels_hl ready for
    run_general_eels_analysis().
    """
    folder = Path(folder)
    files = find_stem_si_files(folder)
    dm4_path = files["dm4"]

    obj_info = detect_multipass(dm4_path)
    print(
        f"[{folder.name}] DM4 objects inspected: "
        + ", ".join(f"{n} (frames={o.n_frames})" for n, o in obj_info.items())
    )

    eels_infos = {n: o for n, o in obj_info.items() if "eels" in n.lower()}
    if not eels_infos:
        raise RuntimeError(
            f"Could not find an EELS object in {dm4_path.name}'s tags. Run "
            f"inspect_dm4_tags({dm4_path!r}) to look at the raw tags and adjust "
            f"detect_multipass()'s object_hint if the naming differs."
        )
    is_multipass = any(o.is_multipass for o in eels_infos.values())
    n_passes = max(o.n_frames for o in eels_infos.values())

    if not is_multipass:
        print(
            "Detected: single-pass (already-summed) acquisition -> reading "
            "directly with quantem.io.read_3d_spectroscopy, as in "
            "OMIEC_Lamella_EELS_O_10.ipynb."
        )
        return _load_single_pass(dm4_path, eels_infos)

    print(f"Detected: MULTI-PASS in-situ acquisition ({n_passes} passes recorded).")
    return _load_multi_pass(
        folder,
        pass_mode=pass_mode,
        passes=passes,
        combine_method=combine_method,
    )


# --------------------------------------------------------------------------- #
# Single-pass inspection (built on load_multipass_raw_stacks())
# --------------------------------------------------------------------------- #


def inspect_single_pass(
    folder: Union[str, Path],
    pass_number: Optional[int] = None,
    good_passes: Union[str, Sequence[int]] = "all",
    show: bool = True,
) -> Dict[str, Any]:
    """
    Inspect one specific pass (1-indexed, matching passes_used elsewhere in
    this module) from a multi-pass in-situ acquisition on its own -- without
    summing it into the rest of the scan -- against two references: pass 1
    (the first/reference pass) and the aggregate over `good_passes`.

    Builds on load_multipass_raw_stacks(), the same per-pass access path
    _load_multi_pass() uses (and available to any other per-pass function,
    e.g. damage detection), so the raw sidecars only get parsed once however
    many pass-level functions you call, not once per function.

    Parameters
    ----------
    folder : str or Path
        Acquisition folder, e.g. 'InSitu (9)'.
    pass_number : int or None, default None
        1-indexed pass number to inspect. If None (the default), resolves to
        `n_passes` -- the true last pass in the sequence (1-indexed, so for a
        37-pass acquisition this is pass 37, not 36) -- so calling this with
        no pass_number gives a quick look at the most likely-damaged pass
        (the raw last scan) against the reference pass 1, with no need to
        already know which passes are damaged.
    good_passes : str or sequence of int, default "all"
        Which passes make up the "overall mean" comparison spectra -- same
        spec format as select_passes()/passes=... elsewhere in this module
        ('all', '1-15', '1,3,5', '1-10,20,25-30', or a list of 1-indexed
        pass numbers).
    show : bool, default True
        Plot the ADF image (pass 1 vs. inspected pass, side by side) and the
        LL/HL spectrum comparisons (pass 1, inspected pass, and the overall
        mean, all on the same axes).

    Returns
    -------
    dict with keys:
        adf_image_i : (ny, nx) ndarray, or None if no ADF raw sidecar was
            found -- the real-space ADF image for the inspected pass alone.
        adf_image_1 : same, for pass 1 (the reference pass); None under the
            same condition as adf_image_i.
        ll_spectrum_i, hl_spectrum_i : (n_energy,) ndarray -- the inspected
            pass's spatially-averaged LL/HL spectra.
        ll_spectrum_1, hl_spectrum_1 : (n_energy,) ndarray -- pass 1's
            spatially-averaged LL/HL spectra.
        ll_mean_overall, hl_mean_overall : (n_energy,) ndarray -- LL/HL
            spectra averaged over `good_passes` (spatially and across those
            passes).
        ll_energy_axis, hl_energy_axis : (n_energy,) ndarray.
        pass_number : the (1-indexed) pass inspected.
        good_passes_used : 1-indexed pass numbers that went into the
            "overall mean" comparison.
    """
    raw = load_multipass_raw_stacks(folder)

    if pass_number is None:
        pass_number = raw.n_passes
    if pass_number < 1 or pass_number > raw.n_passes:
        raise ValueError(f"pass_number={pass_number} out of range 1..{raw.n_passes}")
    i = pass_number - 1
    good_indices = _parse_pass_spec(good_passes, raw.n_passes)

    # ll_stack/hl_stack are (n_frames, n_energy, ny, nx) -- see
    # MultipassRawStacks for why energy is axis 1, not last.
    ll_spectrum_i = raw.ll_stack[i].mean(axis=(1, 2))
    hl_spectrum_i = raw.hl_stack[i].mean(axis=(1, 2))
    ll_spectrum_1 = raw.ll_stack[0].mean(axis=(1, 2))
    hl_spectrum_1 = raw.hl_stack[0].mean(axis=(1, 2))
    ll_mean_overall = raw.ll_stack[good_indices].mean(axis=(0, 2, 3))
    hl_mean_overall = raw.hl_stack[good_indices].mean(axis=(0, 2, 3))
    if raw.adf_stack is not None:
        adf_image_i = raw.adf_stack[i]
        adf_image_1 = raw.adf_stack[0]
    else:
        adf_image_i = None
        adf_image_1 = None

    if show:
        if adf_image_i is not None:
            fig, (ax1, axi) = plt.subplots(1, 2, figsize=(8, 4))
            im1 = ax1.imshow(adf_image_1, cmap="gray")
            ax1.set_title("ADF -- pass 1 (reference)")
            plt.colorbar(im1, ax=ax1)
            imi = axi.imshow(adf_image_i, cmap="gray")
            axi.set_title(f"ADF -- pass {pass_number}")
            plt.colorbar(imi, ax=axi)
            plt.tight_layout()
            plt.show()
        else:
            print(f"No ADF raw stack found for {folder} -- skipping ADF image.")

        fig, (ax_ll, ax_hl) = plt.subplots(1, 2, figsize=(12, 4))
        ax_ll.plot(
            raw.ll_energy_axis, ll_spectrum_1, label="pass 1 (reference)", color="tab:green"
        )
        ax_ll.plot(
            raw.ll_energy_axis, ll_spectrum_i, label=f"pass {pass_number}", color="tab:blue"
        )
        ax_ll.plot(
            raw.ll_energy_axis,
            ll_mean_overall,
            label=f"mean ({good_passes})",
            color="tab:orange",
            ls="--",
        )
        ax_ll.set_xlabel("Energy (eV)")
        ax_ll.set_ylabel("Intensity")
        ax_ll.set_title("LL spectrum")
        ax_ll.legend()

        ax_hl.plot(
            raw.hl_energy_axis, hl_spectrum_1, label="pass 1 (reference)", color="tab:green"
        )
        ax_hl.plot(
            raw.hl_energy_axis, hl_spectrum_i, label=f"pass {pass_number}", color="tab:blue"
        )
        ax_hl.plot(
            raw.hl_energy_axis,
            hl_mean_overall,
            label=f"mean ({good_passes})",
            color="tab:orange",
            ls="--",
        )
        ax_hl.set_xlabel("Energy (eV)")
        ax_hl.set_ylabel("Intensity")
        ax_hl.set_title("HL spectrum")
        ax_hl.legend()

        plt.tight_layout()
        plt.show()

    return {
        "adf_image_i": adf_image_i,
        "adf_image_1": adf_image_1,
        "ll_spectrum_i": ll_spectrum_i,
        "hl_spectrum_i": hl_spectrum_i,
        "ll_spectrum_1": ll_spectrum_1,
        "hl_spectrum_1": hl_spectrum_1,
        "ll_mean_overall": ll_mean_overall,
        "hl_mean_overall": hl_mean_overall,
        "ll_energy_axis": raw.ll_energy_axis,
        "hl_energy_axis": raw.hl_energy_axis,
        "pass_number": pass_number,
        "good_passes_used": [p + 1 for p in good_indices],
    }


# --------------------------------------------------------------------------- #
# General STEM-EELS analysis (mirrors OMIEC_Lamella_EELS_O_10.ipynb)
# --------------------------------------------------------------------------- #


def run_general_eels_analysis(
    eels_ll,
    eels_hl,
    adf: Optional[np.ndarray] = None,
    target_edge: float = 532.0,
    zlp_window: Tuple[float, float] = (-5, 5),
    background_pre_edge_range: Optional[Tuple[float, float]] = None,
    energy_windows: Sequence[Tuple[float, float]] = ((529.5, 532), (534, 536), (537, 543)),
    do_pca: bool = True,
    show: bool = True,
) -> Dict[str, Any]:
    """
    Runs the same general STEM-EELS workflow demonstrated in
    OMIEC_Lamella_EELS_O_10.ipynb: thickness mapping from the low-loss data,
    ZLP alignment (measured on LL, applied to both LL and HL), background
    subtraction around `target_edge` on the high-loss data, energy-window
    chemical maps, and PCA.

    Defaults (target_edge=532 eV, three O K-edge-ish windows) match the O
    K-edge used in your prior "Oxygen/ZLP-lock" runs, which is also what
    InSitu (8)/(9) were collected for -- override for a different edge.

    Every quantem call below has been checked against quantem's real source
    (dataset3deels.py, dataset3dspectroscopy.py, spectroscopy_visualzitions.py,
    core/visualization/visualization.py) -- see call-site comments. Note
    there is no `thickness_window` parameter: calculate_thickness_log_ratio()
    (dataset3deels.py:877) always integrates the *entire* energy axis for the
    low-loss "total" intensity; quantem provides no way to restrict that
    window, so this pipeline can't expose one either.

    Returns a dict with the intermediate/final results (thickness_map,
    zlp-corrected eels_ll/eels_hl, background-subtracted eels_hl,
    energy_window_maps, pca) so you can keep exploring them in a notebook.
    """
    em = _import_quantem()
    results: Dict[str, Any] = {}

    if show:
        eels_ll.show_mean_spectrum()
        eels_hl.show_mean_spectrum()

    # ---- thickness map (low-loss) ----
    # calculate_thickness_log_ratio() (dataset3deels.py:877) takes a single
    # scalar eV half-width around the auto-detected ZLP peak -- not an
    # index-range dict -- and always integrates the *entire* energy axis for
    # the "total" intensity (no way to restrict that window), so the old
    # `thickness_window` parameter can't be honored and has been removed.
    zlp_half_width = (zlp_window[1] - zlp_window[0]) / 2.0

    # calculate_thickness_log_ratio() doesn't expose the per-pixel ZLP fit
    # center (mu0) it computes internally. measure_zlp_offset()
    # (dataset3deels.py:542) is the same Gaussian-fit routine factored into
    # its own method -- call it here (same fit window) purely as a
    # sanity-check diagnostic; it also plots the fitted ZLP-center map.
    zlp_fit_centers = eels_ll.measure_zlp_offset(fit_window=zlp_half_width, fit_zlp=True)
    print(
        f"ZLP fit check: zlp_half_width={zlp_half_width:.3f} eV | "
        f"detected ZLP center -- median={np.median(zlp_fit_centers):.3f} eV, "
        f"range=[{zlp_fit_centers.min():.3f}, {zlp_fit_centers.max():.3f}] eV"
    )

    thickness_map = eels_ll.calculate_thickness_log_ratio(zlp_window=zlp_half_width, plot=show)
    results["thickness_map"] = thickness_map
    print(
        f"Thickness map: min={thickness_map.min():.3f}, max={thickness_map.max():.3f}, "
        f"mean={thickness_map.mean():.3f} t/λ"
    )

    # ---- ZLP correction ----
    # Verified against apply_zlp_correction()'s real signature in
    # dataset3deels.py:712 -- every kwarg here exists, and the real
    # implementation's return-value branching matches the unpacks below.
    eels_ll, zlp_shifts = eels_ll.apply_zlp_correction(
        measure_offset=True,
        fit_to_plane=True,
        return_3d_dataset=True,
        return_shifts=True,
    )
    eels_hl = eels_hl.apply_zlp_correction(
        zlp_shifts_array=zlp_shifts,
        measure_offset=False,
        return_3d_dataset=True,
    )
    if show:
        eels_ll.show_mean_spectrum()
        eels_hl.show_mean_spectrum()

    # ---- background subtraction around target_edge ----
    # Verified against subtract_background_limited_preedge()'s real signature
    # in dataset3deels.py:145 -- every kwarg here exists and matches.
    if background_pre_edge_range is not None:
        eels_hl_bgsub = eels_hl.subtract_background_limited_preedge(
            target_edge=target_edge,
            pre_edge_range=background_pre_edge_range,
            method="linear",
            show=show,
        )
    else:
        eels_hl_bgsub = eels_hl.subtract_background_limited_preedge(
            target_edge=target_edge,
            method="polynomial",
            polynomial_degree=2,
            show=show,
        )
    if show:
        eels_hl_bgsub.show_mean_spectrum()

    # ---- energy-window chemical maps ----
    energy_window_maps: Dict[Tuple[float, float], Any] = {}
    for lo, hi in energy_windows:
        fig, _, emap = eels_hl_bgsub.show_energy_window_map(
            energy_window=[lo, hi], cmap="hot", show=show
        )
        if show:
            plt.suptitle(f"[{lo}, {hi}] eV")
        energy_window_maps[(lo, hi)] = emap
    results["energy_window_maps"] = energy_window_maps

    if adf is not None and energy_window_maps and show and gaussian_filter is not None:
        last_map = list(energy_window_maps.values())[-1]
        em.visualization.show_2d(
            [adf, gaussian_filter(last_map, 1)],
            title=["ADF", f"{energy_windows[-1]} eV map"],
            cmap="magma",
        )

    # ---- PCA ----
    if do_pca:
        # perform_pca()'s real default is return_results=False, which
        # silently returns None despite its `-> dict` type hint
        # (dataset3dspectroscopy.py:380) -- must pass return_results=True to
        # actually get the results back.
        results["pca"] = eels_hl_bgsub.perform_pca(return_results=True, plot_results=show)
        print("✓ PCA complete.")

    results.update(
        {
            "eels_ll": eels_ll,
            "eels_hl": eels_hl,
            "eels_hl_background_subtracted": eels_hl_bgsub,
        }
    )
    print("\n✓ Analysis complete!")
    return results
