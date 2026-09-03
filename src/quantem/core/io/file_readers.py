import importlib
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from quantem.core.datastructures import Dataset as Dataset
from quantem.core.datastructures import Dataset2d as Dataset2d
from quantem.core.datastructures import Dataset3d as Dataset3d
from quantem.core.datastructures import Dataset4dstem as Dataset4dstem
from quantem.core.io.serialize import AutoSerialize
from quantem.spectroscopy import (
    Dataset3deels as Dataset3deels,
)
from quantem.spectroscopy import Dataset3dspectroscopy as Dataset3dspectroscopy
from quantem.spectroscopy import (
    Dataset3dxeds as Dataset3dxeds,
)


def _print_available_datasets(data_list):
    print("Available datasets:")
    for index, entry in enumerate(data_list):
        array = entry["data"]
        print(f"  Dataset {index}: shape {array.shape}, ndim={array.ndim}")


def read_4dstem(
    file_path: str | PathLike,
    file_type: str | None = None,
    dataset_index: int | None = None,
    hot_pixel_filter: bool = False,
    **kwargs,
) -> Dataset4dstem:
    """
    File reader for 4D-STEM data

    Parameters
    ----------
    file_path: str | PathLike
        Path to data
    file_type: str
        The type of file reader needed. See rosettasciio for supported formats
        https://hyperspy.org/rosettasciio/supported_formats/index.html
    dataset_index: int, optional
        Index of the dataset to load if file contains multiple datasets.
        If None, automatically selects the first 4D dataset found.
    hot_pixel_filter: bool, optional
        If True, detect and replace hot detector pixels immediately after
        loading using `quantem.core.utils.filter.filter_hot_pixels` with its
        default parameters. For custom thresholds, call `filter_hot_pixels`
        directly on the array.
    **kwargs: dict
        Additional keyword arguments to pass to the file reader.

    Other Parameters
    ----------------
    name : str | None, optional
        A descriptive name for the dataset. If None, defaults to "4D-STEM dataset"
    origin : NDArray | tuple | list | float | int | None, optional
        The origin coordinates for each dimension in calibrated units. If None, defaults to zeros
    sampling : NDArray | tuple | list | float | int | None, optional
        The sampling rate/spacing for each dimension. If None, defaults to ones
    units : list[str] | tuple | list | None, optional
        Units for each dimension. If None, defaults to ["pixels"] * 4
    signal_units : str, optional
        Units for the array values, by default "arb. units"

    Returns
    --------
    Dataset4dstem

    Examples
    --------
    Load a raw Arina 4D-STEM master file:

    >>> from quantem.core.io import read_4dstem
    >>> ds = read_4dstem(
    ...     '/path/to/gold_013_master.h5',
    ...     file_type='arina',
    ... )
    >>> ds.array.shape
    (256, 256, 192, 192)

    Enable the hot pixel filter to repair stuck detector pixels on load:

    >>> ds = read_4dstem(
    ...     '/path/to/gold_013_master.h5',
    ...     file_type='arina',
    ...     hot_pixel_filter=True,
    ... )
    """
    if file_type is None:
        file_type = Path(file_path).suffix.lower().lstrip(".")

    sampling_override = kwargs.pop("sampling", None)
    origin_override = kwargs.pop("origin", None)
    units_override = kwargs.pop("units", None)
    name_override = kwargs.pop("name", None)

    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader
    data_list = file_reader(file_path, **kwargs)

    # If specific index provided, use it
    if dataset_index is not None:
        imported_data = data_list[dataset_index]
        if imported_data["data"].ndim != 4:
            raise ValueError(
                f"Dataset at index {dataset_index} has {imported_data['data'].ndim} dimensions, "
                f"expected 4D. Shape: {imported_data['data'].shape}"
            )
    else:
        # Automatically find first 4D dataset
        four_d_datasets = [(i, d) for i, d in enumerate(data_list) if d["data"].ndim == 4]
        _print_available_datasets(data_list)

        if len(four_d_datasets) == 0:
            print(f"No 4D datasets found in {file_path}.")
            raise ValueError("No 4D dataset found in file")

        dataset_index, imported_data = four_d_datasets[0]

        print(
            f"Using first 4D dataset at index {dataset_index} with shape {imported_data['data'].shape}"
        )

    imported_axes = imported_data["axes"]

    sampling = (
        sampling_override
        if sampling_override is not None
        else [ax.get("scale", 1) for ax in imported_axes]
    )
    origin = (
        origin_override
        if origin_override is not None
        else [ax.get("offset", 0) for ax in imported_axes]
    )
    units = (
        units_override
        if units_override is not None
        else ["pixels" if ax["units"] == "1" else ax["units"] for ax in imported_axes]
    )

    array = imported_data["data"]
    if hot_pixel_filter:
        from quantem.core.utils.filter import filter_hot_pixels

        array = filter_hot_pixels(array)

    dataset = Dataset4dstem.from_array(
        array=array,
        sampling=sampling,
        origin=origin,
        units=units,
        name=name_override,
    )

    return dataset


def read_3d_spectroscopy(
    file_path: str, file_type: str, data_type: str, dataset_index: int | None = None
) -> Dataset3dspectroscopy:
    """
    File reader for 3D spectroscopy data

    Parameters
    ----------
    file_path: str
        Path to data
    file_type: str
        The type of file reader needed. See rosettasciio for supported formats
        https://hyperspy.org/rosettasciio/supported_formats/index.html
    data_type: str
        type of spectroscopy data 'EELS' or 'XEDS'
    Returns
    --------
    Dataset3dspectroscopy
    """
    data_type_normalized = str(data_type).upper()

    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader  # type: ignore
    data_list = file_reader(file_path)

    # If specific index provided, use it
    if dataset_index is not None:
        imported_data = data_list[dataset_index]
        if imported_data["data"].ndim != 3:
            raise ValueError(
                f"Dataset at index {dataset_index} has {imported_data['data'].ndim} dimensions, "
                f"expected 3D. Shape: {imported_data['data'].shape}"
            )
    else:
        # Automatically find first 3D dataset
        three_d_datasets = [(i, d) for i, d in enumerate(data_list) if d["data"].ndim == 3]
        _print_available_datasets(data_list)

        if len(three_d_datasets) == 0:
            print(f"No 3D datasets found in {file_path}.")
            raise ValueError("No 3D dataset found in file")

        dataset_index, imported_data = three_d_datasets[0]

        dataset_indices = [entry[0] for entry in three_d_datasets]
        print(
            f"Using first 3D dataset at index {dataset_index} with shape {imported_data['data'].shape}. "
            f"3D dataset indices: {', '.join(map(str, dataset_indices))}"
        )

    imported_axes = imported_data["axes"]
    # axis_order = (0, 1, 2) if file_type == "digitalmicrograph" else (2, 0, 1)
    axis_order = (1, 2, 0) if file_type == "digitalmicrograph" else (0, 1, 2)
    array = (
        imported_data["data"].transpose(axis_order)
        if file_type == "digitalmicrograph"
        else imported_data["data"]
    )
    ordered_axes = [imported_axes[idx] for idx in axis_order]
    sampling = [ax.get("scale", 1) for ax in ordered_axes]
    origin = [ax.get("offset", 0) for ax in ordered_axes]
    units = [
        "pixels" if ax.get("units", "1") == "1" else ax.get("units", "pixels")
        for ax in ordered_axes
    ]

    for i, unit in enumerate(units):
        if unit == "eV" and data_type_normalized == "XEDS":
            sampling[i] = sampling[i] / 1000
            origin[i] = origin[i] / 1000
            units[i] = "keV"

    if data_type_normalized == "EELS":
        dataset_cls = Dataset3deels
    elif data_type_normalized == "XEDS":
        dataset_cls = Dataset3dxeds
    else:
        raise ValueError(f"`data_type` must be `XEDS` or `EELS` not `{data_type}`")

    dataset = dataset_cls.from_array(
        array=array,
        sampling=sampling,
        origin=origin,
        units=units,
    )

    return dataset


# --------------------------------------------------------------------------- #
# Multi-pass STEM-EELS (DigitalMicrograph in-situ) reading
#
# DigitalMicrograph's in-situ/multi-pass scanning mode records every
# individual scan pass as its own frame in a raw sidecar file next to the
# DM4 header, instead of a single already-summed spectrum image. rsciio's
# `digitalmicrograph.file_reader()` only returns the latter, so reading a
# multi-pass acquisition needs lower-level access to the DM4 tag tree (via
# ncempy) to detect the per-pass frame stacks and read them directly. A
# regular, already-summed acquisition is still read the normal way, via
# `read_3d_spectroscopy()` above -- `read_stem_eels_folder()` picks between
# the two automatically.
# --------------------------------------------------------------------------- #


@dataclass
class DM4ObjectInfo:
    """One DM4 ImageList object's multipass metadata, from `detect_multipass()`."""

    index: int
    name: str
    n_frames: int
    dims: tuple[int, ...]  # DM order: fastest-varying axis first
    dtype: np.dtype
    is_multipass: bool


def list_dm4_objects(dm4_path: str | PathLike) -> list[tuple[int, str]]:
    """List the (0-indexed) objects stored in a DM4 file, e.g.
    `[(0, 'Thumbnail'), (2, 'STEM SI_ADF Image'), (3, 'STEM SI_EELS LL SI')]`.
    """
    import ncempy.io as nio

    dm0 = nio.dm.fileDM(str(dm4_path))
    return [
        (i, dm0.allTags.get(f".ImageList.{i + 1}.Name", f"<object {i}: no Name tag>"))
        for i in range(dm0.numObjects)
    ]


def inspect_dm4_tags(dm4_path: str | PathLike, contains: str | None = None) -> dict[str, Any]:
    """Dump (optionally filtered) DM4 tags for ad-hoc debugging, e.g.
    `inspect_dm4_tags(dm4_path, contains="In-situ")`.
    """
    import ncempy.io as nio

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
        return np.dtype(np.float32)
    return np.dtype(mapping.get(int(nbytes), np.float32))


def detect_multipass(
    dm4_path: str | PathLike, object_hint: Sequence[str] = ("EELS", "ADF")
) -> dict[str, DM4ObjectInfo]:
    """
    Inspect every DM4 object whose name matches `object_hint` and report
    whether it was recorded in DM's in-situ/multi-pass mode -- i.e. whether
    its raw sidecar stores one frame per pass rather than a single
    already-summed frame. Returns a dict keyed by object name.
    """
    import ncempy.io as nio

    dm0 = nio.dm.fileDM(str(dm4_path))

    results: dict[str, DM4ObjectInfo] = {}
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
        dims: list[int] = []
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


def find_stem_si_files(folder: str | PathLike) -> dict[str, Path | None]:
    """
    Locate the DM4 header + raw sidecars for a STEM SI acquisition folder,
    following the naming convention:

        STEM SI.dm4
        STEM SI_ADF Image.raw
        STEM SI_EELS HL SI.raw
        STEM SI_EELS LL SI.raw
    """
    folder = Path(folder)
    dm4_candidates = sorted(folder.glob("*SI.dm4")) or sorted(folder.glob("*.dm4"))
    # "Picker of ..." and "... PostAcq ..." files are auxiliary DM4s (drift
    # reference / post-acquisition survey images) that also match "*SI.dm4"
    # and can sort before the real acquisition file -- exclude them when a
    # non-auxiliary candidate exists.
    filtered = [
        p
        for p in dm4_candidates
        if "postacq" not in p.name.lower() and "picker" not in p.name.lower()
    ]
    dm4_candidates = filtered or dm4_candidates
    if not dm4_candidates:
        raise FileNotFoundError(f"No .dm4 file found in {folder}")
    dm4_path = dm4_candidates[0]
    stem = dm4_path.stem

    def _sidecar(suffix: str) -> Path | None:
        cands = list(folder.glob(f"{stem}{suffix}"))
        return cands[0] if cands else None

    return {
        "dm4": dm4_path,
        "adf_raw": _sidecar("_ADF Image.raw"),
        "eels_hl_raw": _sidecar("_EELS HL SI.raw"),
        "eels_ll_raw": _sidecar("_EELS LL SI.raw"),
    }


def describe_folder(folder: str | PathLike) -> dict[str, DM4ObjectInfo]:
    """Read-only summary of a STEM SI acquisition folder: which files were
    found and whether it's multi-pass. Handy to run before a full load."""
    files = find_stem_si_files(folder)
    print(f"Folder: {folder}")
    for k, v in files.items():
        print(f"  {k}: {v.name if v else '(not found)'}")
    info = detect_multipass(files["dm4"])
    for name, o in info.items():
        tag = "MULTI-PASS" if o.is_multipass else "single-pass"
        print(f"  [{tag}] {name}: frames={o.n_frames}, dims={o.dims}, dtype={o.dtype}")
    return info


def load_raw_stack(
    raw_path: str | PathLike, n_frames: int, dims: tuple[int, ...], dtype=np.float32
) -> np.ndarray:
    """
    Load a DM in-situ raw sidecar as a (n_frames, ...) stack.

    `dims` is (dim0, dim1[, dim2]) exactly as read from the DM4
    'Raw File Format Info.Dimensions.N' tags (fastest-varying axis first, DM
    convention). The returned array has the frame axis first and the DM
    dimensions reversed (slowest-varying first) -- e.g. for a spectrum image
    stack with dims=(nx, ny, n_energy) you get back shape
    (n_frames, ny, nx, n_energy).
    """
    with open(str(raw_path), "rb") as f:
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


def _parse_pass_spec(spec: str | Sequence[int], n_frames: int) -> list[int]:
    """
    Parse a pass-selection spec into a sorted list of 0-indexed frame numbers.
    Pass numbers themselves are 1-indexed, e.g.:

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
    mode: str = "manual",
    passes: str | Sequence[int] | None = None,
    prompt_label: str = "passes",
) -> list[int]:
    """
    Decide which of the `n_frames` recorded passes to use for analysis.

    mode="manual" -> use `passes` (parsed the same way as the interactive
                      prompt) without asking -- the default, for scripted/
                      batch use. Raises ValueError if `passes` isn't given.
    mode="all"    -> use every pass.
    mode="ask"    -> interactively prompt (input()) for a pass spec.
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
    counts -- it keeps downstream SNR/thickness calculations meaningful.
    method='mean' just rescales to a per-pass average.
    """
    sub = np.take(stack, indices, axis=pass_axis)
    if method == "sum":
        return sub.sum(axis=pass_axis)
    elif method == "mean":
        return sub.mean(axis=pass_axis)
    raise ValueError("method must be 'sum' or 'mean'")


def _get_dm4_calibration(
    dm4_path: str | PathLike, obj_idx: int, n_dims: int, energy_rank: int | None = None
) -> tuple[list[float], list[float], list[str]]:
    """
    Read (origin, scale, units) for each dimension of a DM4 object, 0-indexed
    with the spectral/energy axis last.

    ncempy's per-object calibration arrays (`fileDM.scale`/`.origin`/
    `.scaleUnit`/`.dataShape`) don't always align 1:1, in file order, with
    `.ImageList` object index -- extra unnamed calibrated blocks can be
    interleaved between named objects. What *is* guaranteed by the DM4
    format is order: the Nth calibrated block whose unit is 'eV', in on-disk
    tag order, belongs to the Nth energy-bearing object in ascending
    ImageList order. `energy_rank` (0-indexed) selects that Nth block
    directly -- pass it when the caller already knows this object's rank
    among the acquisition's energy-bearing objects.

    The spatial (pixel-size) calibration is taken as the statistical mode of
    every recorded non-'eV' scale value, which robustly ignores one-off
    outliers (e.g. an unrelated single survey image at a different pixel
    size).
    """
    import ncempy.io as nio

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
        # ncempy computes the calibrated origin as -pixelOrigin * pixelSize.
        energy_origin = -origins_all[k] * energy_scale
        origins.append(energy_origin)
        scales.append(energy_scale)
        units.append(units_all[k])

    return origins, scales, units


def _energy_axis_from_dm4(
    dm4_path: str | PathLike,
    obj_idx: int,
    n_channels: int,
    n_dims: int = 3,
    energy_rank: int | None = None,
) -> np.ndarray:
    origins, scales, _units = _get_dm4_calibration(
        dm4_path, obj_idx, n_dims, energy_rank=energy_rank
    )
    origin, scale = origins[-1], scales[-1]
    return np.linspace(0, scale * (n_channels - 1), n_channels) + origin


def _pixel_size_nm_from_dm4(
    dm4_path: str | PathLike, obj_idx: int, n_dims: int = 3, energy_rank: int | None = None
) -> float | None:
    _origins, scales, units = _get_dm4_calibration(
        dm4_path, obj_idx, n_dims, energy_rank=energy_rank
    )
    scale, unit = scales[0], (units[0] or "").lower()
    if unit in ("nm", ""):
        return scale
    if unit in ("um", "µm", "micron", "microns"):
        return scale * 1000.0
    warnings.warn(f"Unrecognized spatial unit {unit!r} in DM4 calibration; returning raw scale.")
    return scale


def array_to_spectroscopy3d(
    data: np.ndarray,
    energy_axis: np.ndarray,
    pixel_size_nm: float | None = None,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Dataset3deels:
    """
    Wrap an in-memory (ny, nx, n_energy) array (e.g. the result of
    `combine_passes()`) as a `Dataset3deels`, the same class
    `read_3d_spectroscopy(..., data_type="EELS")` returns -- built via
    `Dataset3deels.from_array()` directly from an array instead of a file on
    disk.

    `energy_axis` must be uniformly spaced: only its first value and spacing
    are kept, since `Dataset3dspectroscopy.energy_axis` is a property
    computed from `origin[2]`/`sampling[2]`, not a stored array.
    """
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

    dataset = Dataset3deels.from_array(
        array=data,
        name=name if name is not None else "EELS dataset",
        origin=[0.0, 0.0, float(energy_axis[0])],
        sampling=[spatial_scale, spatial_scale, energy_scale],
        units=[spatial_unit, spatial_unit, "eV"],
    )
    if metadata:
        dataset._metadata = dict(metadata)
    return dataset


class StemEelsRaw(AutoSerialize):
    """
    Container returned by `read_stem_eels_folder()`.

    An `AutoSerialize` subclass (not a plain dataclass) so a loaded
    multipass/single-pass result -- both `Dataset3deels` objects plus the
    provenance of how they were assembled -- can be checkpointed and
    reloaded as one unit via `.save()` / `quantem.io.load()`.

    Attributes
    ----------
    folder : Path
        Acquisition folder this was loaded from.
    dm4_path : Path
        The DM4 header file used.
    is_multipass : bool
        Whether the acquisition was a DigitalMicrograph in-situ/multi-pass
        scan.
    n_passes : int
        Number of passes recorded (1 for single-pass acquisitions).
    eels_ll, eels_hl : Dataset3deels
        The low-loss and high-loss spectrum images.
    adf : NDArray | None
        Real-space ADF image, if one was found.
    energy_axis_ll, energy_axis_hl : NDArray | None
        Energy axes for `eels_ll` / `eels_hl`.
    pixel_size_nm : float | None
        Spatial pixel size, if resolved from the DM4 calibration.
    passes_used : list[int] | None
        1-indexed pass numbers combined into the result (multipass only).
    combine_method : str | None
        How `passes_used` were combined ("sum" or "mean"; multipass only).
    """

    def __init__(
        self,
        folder: Path,
        dm4_path: Path,
        is_multipass: bool,
        n_passes: int,
        eels_ll: Dataset3deels,
        eels_hl: Dataset3deels,
        adf: np.ndarray | None,
        energy_axis_ll: np.ndarray | None,
        energy_axis_hl: np.ndarray | None,
        pixel_size_nm: float | None,
        passes_used: list[int] | None,
        combine_method: str | None,
    ):
        self.folder = folder
        self.dm4_path = dm4_path
        self.is_multipass = is_multipass
        self.n_passes = n_passes
        self.eels_ll = eels_ll
        self.eels_hl = eels_hl
        self.adf = adf
        self.energy_axis_ll = energy_axis_ll
        self.energy_axis_hl = energy_axis_hl
        self.pixel_size_nm = pixel_size_nm
        self.passes_used = passes_used
        self.combine_method = combine_method


def _load_single_pass(
    dm4_path: Path, eels_infos: dict[str, DM4ObjectInfo] | None = None
) -> StemEelsRaw:
    from rsciio.digitalmicrograph import file_reader

    all_data = file_reader(str(dm4_path))

    # NOTE: a DM4 object's ncempy ImageList index (DM4ObjectInfo.index) is
    # NOT the same as its position in rsciio's returned data_list -- rsciio
    # only returns a subset of ImageList objects (e.g. it drops non-image
    # entries), renumbered from 0. rsciio does carry the original DM4 object
    # name in each entry's metadata.General.title, so datasets are matched
    # by that name instead of reusing the ncempy index directly.
    def _rsciio_index_by_title(name: str) -> int | None:
        for i, d in enumerate(all_data):
            if d.get("metadata", {}).get("General", {}).get("title") == name:
                return i
        return None

    ll_info = next(
        (o for n, o in (eels_infos or {}).items() if "ll" in n.lower() or "low" in n.lower()),
        None,
    )
    hl_info = next(
        (o for n, o in (eels_infos or {}).items() if "hl" in n.lower() or "high" in n.lower()),
        None,
    )

    ll_dataset_index = _rsciio_index_by_title(ll_info.name) if ll_info is not None else None
    eels_ll = read_3d_spectroscopy(
        str(dm4_path),
        file_type="digitalmicrograph",
        data_type="EELS",
        dataset_index=ll_dataset_index,
    )

    hl_dataset_index = _rsciio_index_by_title(hl_info.name) if hl_info is not None else None
    eels_hl = read_3d_spectroscopy(
        str(dm4_path),
        file_type="digitalmicrograph",
        data_type="EELS",
        dataset_index=hl_dataset_index,
    )

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

    ll_stack / hl_stack : (n_frames, n_energy, ny, nx) ndarray
        Energy axis is at position 1, not last -- `load_raw_stack()`
        reshapes each frame as tuple(reversed(dims)), which for these EELS
        SI raw sidecars puts energy right after the frame axis.
    adf_stack : (n_frames, ny, nx) ndarray, or None if no ADF raw sidecar
        with a per-pass frame count was found.
    """

    folder: Path
    dm4_path: Path
    files: dict[str, Path | None]
    obj_info: dict[str, DM4ObjectInfo]
    ll_info: DM4ObjectInfo
    hl_info: DM4ObjectInfo
    adf_info: DM4ObjectInfo | None
    n_passes: int
    ll_stack: np.ndarray
    hl_stack: np.ndarray
    adf_stack: np.ndarray | None
    ll_energy_axis: np.ndarray
    hl_energy_axis: np.ndarray
    pixel_size_nm: float | None


def load_multipass_raw_stacks(folder: str | PathLike) -> MultipassRawStacks:
    """
    Load every recorded pass of a multi-pass in-situ acquisition's EELS
    LL/HL and ADF raw stacks, with calibration, but without combining across
    passes. Shared by `read_stem_eels_folder()` and any future per-pass
    analysis (e.g. damage detection).
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

    ll_stack = load_raw_stack(files["eels_ll_raw"], ll_info.n_frames, ll_info.dims, ll_info.dtype)
    hl_stack = load_raw_stack(files["eels_hl_raw"], hl_info.n_frames, hl_info.dims, hl_info.dtype)
    adf_stack = None
    if adf_info is not None and files["adf_raw"] is not None:
        adf_stack = load_raw_stack(
            files["adf_raw"], adf_info.n_frames, adf_info.dims, adf_info.dtype
        )

    # LL always precedes HL in the DM4 ImageList for this acquisition mode,
    # and DM4 objects' tags always appear in that same ascending order on
    # disk -- see _get_dm4_calibration()'s docstring for why this ordinal
    # rank is used instead of obj_idx-based lookup.
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
    passes: str | Sequence[int] | None,
    combine_method: str,
) -> StemEelsRaw:
    raw = load_multipass_raw_stacks(folder)
    chosen = select_passes(raw.n_passes, mode=pass_mode, passes=passes)

    # combine_passes() sums over the frame axis (axis 0), leaving energy at
    # axis 0 of the result -- move it to the end to match the (ny, nx,
    # n_energy) layout array_to_spectroscopy3d() requires.
    ll_combined = np.moveaxis(combine_passes(raw.ll_stack, chosen, method=combine_method), 0, -1)
    hl_combined = np.moveaxis(combine_passes(raw.hl_stack, chosen, method=combine_method), 0, -1)
    adf_combined = (
        combine_passes(raw.adf_stack, chosen, method="mean") if raw.adf_stack is not None else None
    )
    # ADF is averaged (not summed) by default since it's an image you want to
    # look at, not a counting signal -- pass combine_method="sum" if you'd
    # rather see the cumulative dose/drift pattern across selected passes.

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


def read_stem_eels_folder(
    folder: str | PathLike,
    pass_mode: str = "manual",
    passes: str | Sequence[int] | None = None,
    combine_method: str = "sum",
) -> StemEelsRaw:
    """
    Read a STEM-EELS acquisition folder (DM4 header + raw sidecars),
    handling both acquisition styles automatically:

    1. Single-pass: the STEM SI.dm4 file already contains one summed
       spectrum image -- read directly via `read_3d_spectroscopy()`.
    2. Multi-pass (DigitalMicrograph in-situ/multi-pass scanning): every
       individual pass is its own frame in the raw sidecar file. This loads
       the full frame stack, selects passes (`select_passes()`), combines
       them (`combine_passes()`), and wraps the result the same way as (1).

    Parameters
    ----------
    folder : str | PathLike
        Acquisition folder containing the DM4 header + raw sidecars.
    pass_mode, passes, combine_method
        Forwarded to `select_passes()`/`combine_passes()`; only used if the
        dataset turns out multi-pass. Default `pass_mode="manual"` requires
        `passes=...` (or use `pass_mode="all"`) and never prompts; pass
        `pass_mode="ask"` for an interactive `input()` prompt instead.

    Returns
    -------
    StemEelsRaw
        `.eels_ll` / `.eels_hl` are ready-to-use `Dataset3deels` instances.
    """
    folder = Path(folder)
    files = find_stem_si_files(folder)
    dm4_path = files["dm4"]
    assert dm4_path is not None

    obj_info = detect_multipass(dm4_path)
    eels_infos = {n: o for n, o in obj_info.items() if "eels" in n.lower()}
    if not eels_infos:
        raise RuntimeError(
            f"Could not find an EELS object in {dm4_path.name}'s tags. Run "
            f"inspect_dm4_tags({dm4_path!r}) to look at the raw tags and adjust "
            f"detect_multipass()'s object_hint if the naming differs."
        )
    is_multipass = any(o.is_multipass for o in eels_infos.values())

    if not is_multipass:
        return _load_single_pass(dm4_path, eels_infos)

    return _load_multi_pass(
        folder, pass_mode=pass_mode, passes=passes, combine_method=combine_method
    )


def read_2d(
    file_path: str | PathLike,
    file_type: str | None = None,
) -> Dataset2d:
    """
    File reader for images

    Parameters
    ----------
    file_path: str | PathLike
        Path to data
    file_type: str
        The type of file reader needed. See rosettasciio for supported formats
        https://hyperspy.org/rosettasciio/supported_formats/index.html

    Returns
    --------
    Dataset
    """
    if file_type is None:
        file_type = Path(file_path).suffix.lower().lstrip(".")

    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader
    imported_data = file_reader(file_path)[0]

    dataset = Dataset2d.from_array(
        array=imported_data["data"],
        sampling=[
            imported_data["axes"][0]["scale"],
            imported_data["axes"][1]["scale"],
        ],
        origin=[
            imported_data["axes"][0]["offset"],
            imported_data["axes"][1]["offset"],
        ],
        units=[
            imported_data["axes"][0]["units"],
            imported_data["axes"][1]["units"],
        ],
    )
    dataset.file_path = file_path

    return dataset


def read_emdfile_to_4dstem(
    file_path: str | PathLike,
    data_keys: list[str] | None = None,
    calibration_keys: list[str] | None = None,
) -> Dataset4dstem:
    """
    File reader for legacy `emdFile` / `py4DSTEM` files.

    Parameters
    ----------
    file_path: str | PathLike
        Path to data

    Returns
    --------
    Dataset4dstem
    """
    with h5py.File(file_path, "r") as file:
        # Access the data directly
        data_keys = ["datacube_root", "datacube", "data"] if data_keys is None else data_keys
        print("keys: ", data_keys)
        try:
            data: Any = file
            for key in data_keys:
                data = data[key]
        except KeyError:
            raise KeyError(f"Could not find key {data_keys} in {file_path}")

        # Access calibration values directly
        calibration_keys = (
            ["datacube_root", "metadatabundle", "calibration"]
            if calibration_keys is None
            else calibration_keys
        )
        try:
            calibration = file
            for key in calibration_keys:
                calibration = calibration[key]
        except KeyError:
            raise KeyError(f"Could not find calibration key {calibration_keys} in {file_path}")
        r_pixel_size = calibration["R_pixel_size"][()]
        q_pixel_size = calibration["Q_pixel_size"][()]
        r_pixel_units = calibration["R_pixel_units"][()]
        q_pixel_units = calibration["Q_pixel_units"][()]

        dataset = Dataset4dstem.from_array(
            array=data,
            sampling=[r_pixel_size, r_pixel_size, q_pixel_size, q_pixel_size],
            units=[r_pixel_units, r_pixel_units, q_pixel_units, q_pixel_units],
        )
    dataset.file_path = file_path

    return dataset


def read_abtem(url: str | PathLike):
    """
    Read canonical abTEM Zarr file(s) into quantem Dataset(s).

    Returns
    -------
    Dataset or list[Dataset]
    """

    def _open_zarr(url):
        import zarr

        if url.endswith(".zip"):
            store = zarr.storage.ZipStore(url, mode="r")  # type: ignore
            return zarr.open(store=store, mode="r")
        return zarr.open(url, mode="r")

    def _validate_canonical_format(root):
        if "metadata0" in root.attrs:
            return

        if "kwargs0" in root.attrs:
            raise ValueError(
                "Legacy abTEM Zarr format detected.\n\n"
                "quantem supports only canonical abTEM Zarr format.\n"
                "Re-save using abtem>=1.1.0:\n\n"
                "    measurement = abtem.from_zarr(<legacy_path>)\n"
                "    measurement.to_zarr(<new_path>)"
            )

        raise ValueError("Unrecognized Zarr format.")

    def _iter_metadata_indices(root):
        i = 0
        while f"metadata{i}" in root.attrs:
            yield i
            i += 1

    def _decode_types(obj) -> Any:
        if isinstance(obj, dict):
            if obj.get("_type") == "tuple":
                return tuple(_decode_types(v) for v in obj["_value"])
            return {k: _decode_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_decode_types(v) for v in obj]
        return obj

    def _normalize_unit(unit):
        if unit is None:
            return "pixels"

        unit = unit.strip()

        UNIT_MAP = {
            "Å": "A",
            "Ångström": "A",
            "Angstrom": "A",
            "1/Å": "A^-1",
            "Å^-1": "A^-1",
            "1/A": "A^-1",
        }

        return UNIT_MAP.get(unit, unit)

    def _convert_axes(axes_dict):
        sampling = []
        origin = []
        units = []

        for key in sorted(axes_dict, key=lambda x: int(x.split("_")[1])):
            axis = axes_dict[key]

            sampling.append(axis.get("sampling", 1.0))
            units.append(_normalize_unit(axis.get("units", None)))
            origin.append(0.0)  # deliberate design choice

        return tuple(origin), tuple(sampling), tuple(units)

    def _read_single_dataset(root, index):
        metadata = _decode_types(root.attrs[f"metadata{index}"]).copy()

        axes_dict = metadata.pop("axes")
        dataset_type = metadata.pop("type")
        metadata.pop("data_origin", None)

        origin, sampling, units = _convert_axes(axes_dict)

        array = root[f"array{index}"]
        signal_units = metadata.get("units", "arb. units")

        dataset = Dataset.from_array(
            array=array,
            name=dataset_type,
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units,
        )

        dataset._metadata = metadata
        return dataset

    root = _open_zarr(url)
    _validate_canonical_format(root)

    datasets = [_read_single_dataset(root, i) for i in _iter_metadata_indices(root)]

    return datasets[0] if len(datasets) == 1 else datasets
