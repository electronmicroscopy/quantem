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


def _find_split_eels_files(folder: Path) -> dict[str, Path | None] | None:
    """Detect the split-file dual-EELS layout (see `find_stem_si_files()`).
    Returns None unless both an `EELS LL SI.dm4` and an `EELS HL SI.dm4` exist."""

    def _channel(tag: str) -> Path | None:
        # "(12) Spectrum of EELS LL SI.dm4" etc. are extracted 1D spectra / pickers
        # that also end in "EELS LL SI.dm4" -- not the spectrum image itself.
        cands = [
            p
            for p in sorted(folder.glob(f"*EELS {tag} SI.dm4"))
            if not any(k in p.name.lower() for k in ("spectrum of", "picker", "postacq"))
        ]
        exact = [p for p in cands if p.name == f"EELS {tag} SI.dm4"]
        return (exact or cands or [None])[0]

    ll_path, hl_path = _channel("LL"), _channel("HL")
    if ll_path is None or hl_path is None:
        return None
    adf_path = folder / "ADF Image.dm4"
    return {
        "dm4": hl_path,
        "adf_raw": None,
        "eels_hl_raw": None,
        "eels_ll_raw": None,
        "dm4_ll": ll_path,
        "dm4_hl": hl_path,
        "dm4_adf": adf_path if adf_path.exists() else None,
    }


def find_stem_si_files(folder: str | PathLike) -> dict[str, Path | None]:
    """
    Locate the DM4 header + raw sidecars for a STEM SI acquisition folder,
    following the naming convention:

        STEM SI.dm4
        STEM SI_ADF Image.raw
        STEM SI_EELS HL SI.raw
        STEM SI_EELS LL SI.raw

    Also recognizes the "split-file" layout, where DigitalMicrograph saved
    each channel of a (single-pass) dual-EELS acquisition as its own DM4:

        EELS LL SI.dm4
        EELS HL SI.dm4
        ADF Image.dm4        (optional, co-registered with the scan)

    In that case the returned dict additionally carries `"dm4_ll"`,
    `"dm4_hl"` and `"dm4_adf"`, and `"dm4"` points at the HL file.
    """
    folder = Path(folder)
    split = _find_split_eels_files(folder)
    if split is not None:
        return split
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
    if files.get("dm4_ll") is not None:
        info = {**detect_multipass(files["dm4_ll"]), **info}
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


def estimate_pass_shifts(
    images: np.ndarray,
    passes: Sequence[int] | None = None,
    *,
    reference: str = "mean",
    upsample_factor: int = 10,
    max_shift_px: float | None = None,
    normalization: str | None = None,
) -> np.ndarray:
    """
    Estimate per-pass (dy, dx) rigid shifts, in pixels, via FFT-based
    subpixel cross-correlation (`skimage.registration.phase_cross_correlation`)
    against a single fixed reference image built from `images`.

    Meant to correct the drift-during-summation artifact multi-pass
    acquisitions are prone to: sample/stage drift between passes shifts
    each pass's raster a little relative to the last, and summing
    unregistered passes (as `combine_passes()` does on its own) smears any
    sharp feature into a diagonal streak. Estimating (and then applying,
    via `apply_pass_shifts()`) a per-pass shift before summing corrects
    for that. `read_stem_eels_folder(..., align_passes=True)` runs this
    automatically; call it directly only for a custom alignment workflow.

    Parameters
    ----------
    images : (n_frames, ny, nx) ndarray
        One 2D image per pass to register on -- pass the ADF stack
        (`MultipassRawStacks.adf_stack`) when available, since its much
        higher per-pass SNR than a single EELS pass makes the
        cross-correlation far more reliable; fall back to an
        energy-summed EELS stack (`stack.sum(axis=1)`, from either
        `.ll_stack` or `.hl_stack`) only if no ADF sidecar was recorded.
    passes : sequence of int, optional
        0-indexed subset of `images`' frame axis to estimate shifts for
        (e.g. the same `chosen` list `select_passes()` returned) and to
        build the reference from. Defaults to every frame in `images`.
    reference : {"mean", "first"}, optional
        "mean" (default): register every selected pass against the
        (unregistered) mean of all selected passes -- a smoother,
        higher-SNR reference, though itself already blurred by drift if
        the drift is large relative to the field of view.
        "first": register every pass against the first selected pass --
        avoids a blurred reference, but feeds that one frame's own noise
        directly into every shift estimate.
    upsample_factor : int, optional
        Subpixel precision passed to `phase_cross_correlation` -- shifts
        are resolved to `1/upsample_factor` of a pixel. Default 10 (0.1 px).
    max_shift_px : float, optional
        If given, any estimated shift with magnitude
        `sqrt(dy**2 + dx**2)` greater than this is treated as a failed
        registration (clamped to zero, i.e. that pass is left unshifted)
        and reported in one `UserWarning` -- phase correlation can return
        a wild spurious shift on a low-contrast or near-featureless pass;
        this guards against actually applying one. `None` (default)
        disables the check.
    normalization : {"phase", None}, optional
        Forwarded to `phase_cross_correlation`. Default `None` here
        (skimage's own default is `"phase"`) -- phase-only normalization
        whitens the cross-power spectrum, which amplifies noise and can
        fail badly (near-zero correlation, a near-random shift estimate)
        on images that are smooth/low-contrast relative to their noise --
        common for STEM ADF/EELS frames (e.g. a lamella with a broad
        thickness gradient and comparatively little fine texture).
        Verified empirically on synthetic drifted ADF-like frames: default
        `"phase"` normalization returned shift estimates uncorrelated with
        the injected drift and a near-1.0 (fully decorrelated) match
        error, while `normalization=None` recovered the injected drift
        accurately. Pass `"phase"` explicitly if your data has enough
        broadband texture (e.g. clear atomic-column contrast) for it to
        help instead of hurt.

    Returns
    -------
    ndarray, shape (len(passes), 2)
        `[dy, dx]` per selected pass, in the same order as `passes`, in
        pixels -- already the correction to *apply* to that pass (i.e.
        `scipy.ndimage.shift(pass_image, shifts[i])` registers it onto the
        reference; this is `phase_cross_correlation`'s own convention, the
        negative of the pass's raw displacement from the reference).
        `apply_pass_shifts()` consumes this directly, unchanged.
    """
    from skimage.registration import phase_cross_correlation

    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError(f"images must be (n_frames, ny, nx), got shape {images.shape}")

    idx = list(range(images.shape[0])) if passes is None else list(passes)
    if not idx:
        raise ValueError("passes must be non-empty")

    selected = images[idx].astype(float)

    if reference == "mean":
        ref_image = selected.mean(axis=0)
    elif reference == "first":
        ref_image = selected[0]
    else:
        raise ValueError(f"reference must be 'mean' or 'first', got {reference!r}")

    shifts = np.zeros((len(idx), 2), dtype=float)
    clamped = []
    for i, frame in enumerate(selected):
        shift, _error, _diffphase = phase_cross_correlation(
            ref_image, frame, upsample_factor=upsample_factor, normalization=normalization
        )
        if max_shift_px is not None and float(np.hypot(*shift)) > max_shift_px:
            clamped.append((idx[i], float(np.hypot(*shift))))
            shift = np.zeros(2)
        shifts[i] = shift

    if clamped:
        details = ", ".join(f"pass {p + 1}: {d:.1f}px" for p, d in clamped)
        warnings.warn(
            f"estimate_pass_shifts: {len(clamped)} pass(es) exceeded "
            f"max_shift_px={max_shift_px} and were left unshifted (likely a "
            f"failed/spurious registration, e.g. a low-contrast pass): {details}",
            UserWarning,
        )

    return shifts


def apply_pass_shifts(
    stack: np.ndarray,
    shifts: np.ndarray,
    passes: Sequence[int] | None = None,
    *,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """
    Apply per-pass rigid shifts (from `estimate_pass_shifts()`) to the
    trailing two (spatial) axes of `stack`, one frame at a time, leaving
    any axes in between (e.g. an EELS stack's energy axis) untouched.

    Parameters
    ----------
    stack : ndarray
        `(n_frames, ny, nx)` (e.g. an ADF stack) or `(n_frames, n_energy,
        ny, nx)` (e.g. `MultipassRawStacks.ll_stack` / `.hl_stack`) --
        anything whose first axis is the pass/frame axis and last two axes
        are spatial.
    shifts : (len(passes), 2) ndarray
        `[dy, dx]` per pass, in the same order as `passes`, as returned by
        `estimate_pass_shifts()`.
    passes : sequence of int, optional
        0-indexed frame indices `shifts` corresponds to. Defaults to
        `range(stack.shape[0])`, which requires `len(shifts) ==
        stack.shape[0]`. Frames of `stack` not listed in `passes` are
        copied through unshifted.
    order : int, optional
        Spline interpolation order for `scipy.ndimage.shift`. Default 1
        (bilinear) -- a reasonable default for STEM images; use 0
        (nearest) to avoid introducing any new interpolated values (e.g.
        before an analysis sensitive to exact per-pixel counts).
    mode, cval : optional
        Forwarded to `scipy.ndimage.shift` for the newly-exposed border
        region after shifting. Default `mode="constant", cval=0.0`, i.e.
        the shifted-in border reads as zero, matching a raw EELS/ADF
        stack's natural "no signal" value.

    Returns
    -------
    ndarray
        A new array, same shape as `stack`, with each pass in `passes`
        shifted by `shifts[i]` (as returned by `estimate_pass_shifts()`,
        unchanged -- that function already returns the correction to
        apply, not the raw displacement). Cast through float for the
        interpolation; cast back to `stack`'s original dtype (rounding) if
        it was integer.
    """
    from scipy.ndimage import shift as ndi_shift

    stack = np.asarray(stack)
    idx = list(range(stack.shape[0])) if passes is None else list(passes)
    shifts = np.asarray(shifts, dtype=float)
    if shifts.shape != (len(idx), 2):
        raise ValueError(
            f"shifts must have shape ({len(idx)}, 2) to match passes, got {shifts.shape}"
        )

    out = stack.astype(float, copy=True)
    for i, p in enumerate(idx):
        dy, dx = shifts[i]
        frame_shift = [0.0] * (stack.ndim - 1)
        frame_shift[-2] = dy
        frame_shift[-1] = dx
        out[p] = ndi_shift(out[p], shift=frame_shift, order=order, mode=mode, cval=cval)

    if np.issubdtype(stack.dtype, np.integer):
        out = np.round(out).astype(stack.dtype)

    return out


def plot_pass_shifts(
    pass_shifts_px: np.ndarray,
    passes_used: Sequence[int] | None = None,
    title: str = "Estimated per-pass drift",
):
    """
    Quick diagnostic: plot the per-pass `[dy, dx]` shifts from
    `estimate_pass_shifts()` (or `StemEelsRaw.pass_shifts_px`) against pass
    number. A roughly linear/monotonic trend across passes is the classic
    signature of steady thermal/stage drift during acquisition, as opposed
    to a single outlier pass (drift-unrelated damage, a dropped frame,
    etc.) -- useful both to confirm drift is the cause of a streaky summed
    map before turning on `align_passes=True`, and to sanity-check the
    fitted shifts afterward.

    Parameters
    ----------
    pass_shifts_px : (n, 2) ndarray
        `[dy, dx]` per pass, in pixels.
    passes_used : sequence of int, optional
        0-indexed pass numbers `pass_shifts_px` corresponds to (e.g.
        `StemEelsRaw.passes_used`, 1-indexed there -- subtract 1, or just
        omit this and let the x-axis default to `1..n`). Only used to
        label the x-axis; defaults to `1, 2, ..., n`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    pass_shifts_px = np.asarray(pass_shifts_px, dtype=float)
    x = (
        np.asarray(passes_used, dtype=float) + 1
        if passes_used is not None
        else np.arange(1, len(pass_shifts_px) + 1)
    )

    fig, (ax_xy, ax_mag) = plt.subplots(1, 2, figsize=(11, 4))
    ax_xy.plot(x, pass_shifts_px[:, 1], "o-", label="dx (col)")
    ax_xy.plot(x, pass_shifts_px[:, 0], "o-", label="dy (row)")
    ax_xy.set_xlabel("Pass number")
    ax_xy.set_ylabel("Shift (px)")
    ax_xy.set_title("Shift components vs. pass")
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)

    magnitude = np.hypot(pass_shifts_px[:, 0], pass_shifts_px[:, 1])
    ax_mag.plot(x, magnitude, "o-", color="k")
    ax_mag.set_xlabel("Pass number")
    ax_mag.set_ylabel("|shift| (px)")
    ax_mag.set_title("Shift magnitude vs. pass")
    ax_mag.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


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
    pass_shifts_px : NDArray | None
        `[dy, dx]` per entry of `passes_used`, in pixels, if
        `read_stem_eels_folder(..., align_passes=True)` was used to
        register passes against drift before combining them (multipass
        only). `None` if alignment wasn't requested (the default) or this
        is a single-pass acquisition. See `estimate_pass_shifts()` /
        `plot_pass_shifts()`. When this came from `remove_drift_frames()`,
        covers only the *kept* passes, in the same order as `passes_used` --
        pass directly to `crop_alignment_border()`.
    dropped_passes : list[int] | None
        1-indexed passes `remove_drift_frames()` dropped before combining
        (same numbering convention as `passes_used`). `None` unless this
        came from `remove_drift_frames()`.
    drop_reason : str | None
        Human-readable reason for `dropped_passes`, from
        `suggest_drift_frames_to_drop()` (or `"manual drop_list"`). `None`
        unless this came from `remove_drift_frames()`.
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
        pass_shifts_px: np.ndarray | None = None,
        dropped_passes: list[int] | None = None,
        drop_reason: str | None = None,
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
        self.dropped_passes = dropped_passes
        self.drop_reason = drop_reason
        self.pass_shifts_px = pass_shifts_px


def _load_split_files(files: dict[str, Path | None]) -> StemEelsRaw:
    """Load the split-file dual-EELS layout (`EELS LL SI.dm4` + `EELS HL SI.dm4`,
    optional `ADF Image.dm4`) -- see `find_stem_si_files()`. Single-pass only."""
    from rsciio.digitalmicrograph import file_reader

    ll_path, hl_path, adf_path = files["dm4_ll"], files["dm4_hl"], files.get("dm4_adf")
    assert ll_path is not None and hl_path is not None
    eels_ll = read_3d_spectroscopy(str(ll_path), file_type="digitalmicrograph", data_type="EELS")
    eels_hl = read_3d_spectroscopy(str(hl_path), file_type="digitalmicrograph", data_type="EELS")
    if tuple(eels_ll.shape[:2]) != tuple(eels_hl.shape[:2]):
        raise ValueError(
            f"LL and HL scan shapes differ in {ll_path.parent}: "
            f"{tuple(eels_ll.shape[:2])} vs {tuple(eels_hl.shape[:2])}"
        )

    adf = None
    if adf_path is not None:
        adf_2d = [d["data"] for d in file_reader(str(adf_path)) if np.ndim(d["data"]) == 2]
        if adf_2d:
            adf = np.asarray(adf_2d[0])

    pixel_size_nm = None
    unit = str(eels_ll.units[0] or "").lower()
    if unit == "nm":
        pixel_size_nm = float(eels_ll.sampling[0])
    elif unit in ("um", "µm", "micron", "microns"):
        pixel_size_nm = float(eels_ll.sampling[0]) * 1000.0

    return StemEelsRaw(
        folder=hl_path.parent,
        dm4_path=hl_path,
        is_multipass=False,
        n_passes=1,
        eels_ll=eels_ll,
        eels_hl=eels_hl,
        adf=adf,
        energy_axis_ll=getattr(eels_ll, "energy_axis", None),
        energy_axis_hl=getattr(eels_hl, "energy_axis", None),
        pixel_size_nm=pixel_size_nm,
        passes_used=None,
        combine_method=None,
    )


def _load_single_pass(
    dm4_path: Path,
    eels_infos: dict[str, DM4ObjectInfo] | None = None,
    obj_info: dict[str, DM4ObjectInfo] | None = None,
) -> StemEelsRaw:
    from rsciio.digitalmicrograph import file_reader

    all_data = file_reader(str(dm4_path))

    # NOTE: a DM4 object's ncempy ImageList index (DM4ObjectInfo.index) is
    # NOT the same as its position in rsciio's returned data_list -- rsciio
    # only returns a subset of ImageList objects (e.g. it drops non-image
    # entries), renumbered from 0. rsciio does carry the original DM4 object
    # name in each entry's metadata.General.title, so datasets are matched
    # by that name instead of reusing the ncempy index directly. Confirmed
    # on real data (05-15-2026_STEM_ELLS_Mono_pg3T2, 90 meV dispersion
    # acquisitions): rsciio's data_list can also DROP an object entirely
    # (there the ADF survey image wasn't retrievable via its own ncempy
    # getDataset() index either), which is exactly why ADF below falls back
    # to the previous fixed-index behavior rather than erroring when its
    # name isn't found in `all_data`.
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

    # Match the ADF object by name too (same "adf", exclude "postacq"
    # convention `load_multipass_raw_stacks()` uses below), instead of
    # assuming it always sits at rsciio data_list position 1 -- that
    # assumption silently mis-assigned an EELS cube's own data as `.adf`
    # whenever a file's object ordering didn't happen to match (confirmed
    # on 05-15-2026_STEM_ELLS_Mono_pg3T2's 90 meV dispersion acquisitions:
    # `.adf` came out as a (3232, 59, 62) EELS-shaped array instead of a 2D
    # image). Falls back to the old `all_data[1]` behavior only if no
    # ADF-named object is found in the tags or it can't be matched by
    # title in `all_data`, so files that happened to work under the old
    # assumption keep working.
    #
    # Some single-pass DM4 files carry TWO distinct "adf"-named objects:
    # a separate, lower-magnification "ADF Image (SI Survey)" context image
    # (own field of view, NOT co-registered with the LL/HL scan -- same
    # pixel count as the scan purely by coincidence, if at all) and a
    # "STEM SI_ADF Image" acquired as part of the same SI raster as the
    # LL/HL channels (same "STEM SI_" prefix as "STEM SI_EELS LL/HL SI"),
    # i.e. the genuinely co-registered one -- confirmed on real data
    # (05-15-2026_STEM_ELLS_Mono_pg3T2 / the original InSitu session alike):
    # "STEM SI_ADF Image" matches the scan's own (ny, nx), "ADF Image (SI
    # Survey)" does not. Prefer the "STEM SI_"-prefixed one when both exist;
    # the plain "adf" match remains the fallback for files (e.g. most of
    # the pg3T2 set) that only ever have the survey-style name.
    _adf_candidates = [
        o for n, o in (obj_info or {}).items() if "adf" in n.lower() and "postacq" not in n.lower()
    ]
    adf_info = next((o for o in _adf_candidates if "stem si" in o.name.lower()), None)
    if adf_info is None:
        adf_info = next(iter(_adf_candidates), None)
    adf = None
    if adf_info is not None:
        adf_dataset_index = _rsciio_index_by_title(adf_info.name)
        if adf_dataset_index is not None:
            adf = all_data[adf_dataset_index]["data"]
    if adf is None and len(all_data) > 1:
        adf = all_data[1]["data"]

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
    align_passes: bool = False,
    alignment_stack: str = "adf",
    alignment_reference: str = "mean",
    alignment_upsample_factor: int = 10,
    alignment_max_shift_px: float | None = None,
    alignment_normalization: str | None = None,
) -> StemEelsRaw:
    raw = load_multipass_raw_stacks(folder)
    chosen = select_passes(raw.n_passes, mode=pass_mode, passes=passes)

    ll_stack, hl_stack, adf_stack = raw.ll_stack, raw.hl_stack, raw.adf_stack
    pass_shifts_px = None
    if align_passes:
        if alignment_stack == "adf" and raw.adf_stack is None:
            warnings.warn(
                "_load_multi_pass: align_passes=True with alignment_stack='adf' but "
                "no ADF stack was found for this acquisition -- falling back to "
                "alignment_stack='hl' (energy-summed high-loss image per pass).",
                UserWarning,
            )
            alignment_stack = "hl"

        if alignment_stack == "adf":
            registration_images = raw.adf_stack
        elif alignment_stack == "ll":
            registration_images = raw.ll_stack.sum(axis=1)
        elif alignment_stack == "hl":
            registration_images = raw.hl_stack.sum(axis=1)
        else:
            raise ValueError(
                f"alignment_stack must be 'adf', 'll', or 'hl', got {alignment_stack!r}"
            )

        pass_shifts_px = estimate_pass_shifts(
            registration_images,
            chosen,
            reference=alignment_reference,
            upsample_factor=alignment_upsample_factor,
            max_shift_px=alignment_max_shift_px,
            normalization=alignment_normalization,
        )
        ll_stack = apply_pass_shifts(raw.ll_stack, pass_shifts_px, chosen)
        hl_stack = apply_pass_shifts(raw.hl_stack, pass_shifts_px, chosen)
        if raw.adf_stack is not None:
            adf_stack = apply_pass_shifts(raw.adf_stack, pass_shifts_px, chosen)

    # combine_passes() sums over the frame axis (axis 0), leaving energy at
    # axis 0 of the result -- move it to the end to match the (ny, nx,
    # n_energy) layout array_to_spectroscopy3d() requires.
    ll_combined = np.moveaxis(combine_passes(ll_stack, chosen, method=combine_method), 0, -1)
    hl_combined = np.moveaxis(combine_passes(hl_stack, chosen, method=combine_method), 0, -1)
    adf_combined = (
        combine_passes(adf_stack, chosen, method="mean") if adf_stack is not None else None
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
        pass_shifts_px=pass_shifts_px,
    )


def read_stem_eels_folder(
    folder: str | PathLike,
    pass_mode: str = "manual",
    passes: str | Sequence[int] | None = None,
    combine_method: str = "sum",
    align_passes: bool = False,
    alignment_stack: str = "adf",
    alignment_reference: str = "mean",
    alignment_upsample_factor: int = 10,
    alignment_max_shift_px: float | None = None,
    alignment_normalization: str | None = None,
) -> StemEelsRaw:
    """
    Read a STEM-EELS acquisition folder (DM4 header + raw sidecars),
    handling both acquisition styles automatically:

    1. Single-pass: the STEM SI.dm4 file already contains one summed
       spectrum image -- read directly via `read_3d_spectroscopy()`.
    2. Multi-pass (DigitalMicrograph in-situ/multi-pass scanning): every
       individual pass is its own frame in the raw sidecar file. This loads
       the full frame stack, selects passes (`select_passes()`), optionally
       registers them against drift (`align_passes`), combines them
       (`combine_passes()`), and wraps the result the same way as (1).

    Parameters
    ----------
    folder : str | PathLike
        Acquisition folder containing the DM4 header + raw sidecars.
    pass_mode, passes, combine_method
        Forwarded to `select_passes()`/`combine_passes()`; only used if the
        dataset turns out multi-pass. Default `pass_mode="manual"` requires
        `passes=...` (or use `pass_mode="all"`) and never prompts; pass
        `pass_mode="ask"` for an interactive `input()` prompt instead.
    align_passes : bool, optional
        If True (multi-pass only), estimate a per-pass rigid (dy, dx)
        shift via cross-correlation (`estimate_pass_shifts()`) and apply it
        (`apply_pass_shifts()`) to LL/HL/ADF before combining passes --
        corrects the diagonal-streak artifact that summing unregistered
        passes produces when the sample/stage drifts during acquisition
        (e.g. in-situ heating series). Default False, matching prior
        behavior (passes summed/meaned as recorded, no registration). The
        resulting shifts are returned as `StemEelsRaw.pass_shifts_px` --
        pass it to `plot_pass_shifts()` to sanity-check them (a roughly
        linear trend across passes is the expected drift signature).
    alignment_stack : {"adf", "ll", "hl"}, optional
        Which per-pass image stack to estimate shifts from when
        `align_passes=True`. Default `"adf"` -- much higher per-pass SNR
        than a single EELS pass, so the registration is far more reliable;
        falls back to `"hl"` (with a warning) if no ADF sidecar was found.
        `"ll"`/`"hl"` register on that stack's own per-pass energy-summed
        image instead.
    alignment_reference : {"mean", "first"}, optional
        Forwarded to `estimate_pass_shifts()`. Default `"mean"`.
    alignment_upsample_factor : int, optional
        Forwarded to `estimate_pass_shifts()` (subpixel precision, as
        `1/upsample_factor` px). Default 10.
    alignment_max_shift_px : float, optional
        Forwarded to `estimate_pass_shifts()` -- shifts larger than this
        are treated as a failed registration and left unshifted (with a
        warning) instead of applied. Default `None` (no check).
    alignment_normalization : {"phase", None}, optional
        Forwarded to `estimate_pass_shifts()`. Default `None` -- more
        robust than skimage's own `"phase"` default for typical STEM
        ADF/EELS frames; see `estimate_pass_shifts()`'s docstring for why.

    Returns
    -------
    StemEelsRaw
        `.eels_ll` / `.eels_hl` are ready-to-use `Dataset3deels` instances.
    """
    folder = Path(folder)
    files = find_stem_si_files(folder)
    if files.get("dm4_ll") is not None:
        return _load_split_files(files)
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
        return _load_single_pass(dm4_path, eels_infos, obj_info)

    return _load_multi_pass(
        folder,
        pass_mode=pass_mode,
        passes=passes,
        combine_method=combine_method,
        align_passes=align_passes,
        alignment_stack=alignment_stack,
        alignment_reference=alignment_reference,
        alignment_upsample_factor=alignment_upsample_factor,
        alignment_max_shift_px=alignment_max_shift_px,
        alignment_normalization=alignment_normalization,
    )


def _load_alignment_stack(
    folder: str | PathLike, alignment_stack: str = "adf"
) -> tuple[np.ndarray, int]:
    """
    Load just the ONE per-pass image stack needed to estimate drift
    (default: ADF), without loading the much larger LL/HL EELS raw stacks --
    the DM4-tag-only object lookup (`detect_multipass()`) is cheap; only the
    single requested raw sidecar is actually read from disk. Falls back to
    an energy-summed EELS stack (with a warning) if `alignment_stack="adf"`
    has no recorded ADF sidecar.

    Returns
    -------
    stack : (n_frames, ny, nx) ndarray
    n_passes : int
        Total recorded passes (from the EELS objects' frame counts), even
        when the returned stack itself is the (possibly smaller) ADF one.
    """
    folder = Path(folder)
    files = find_stem_si_files(folder)
    if files.get("dm4_ll") is not None:
        # split-file layout (EELS LL SI.dm4 + EELS HL SI.dm4) is always single-pass
        raise ValueError(
            f"_load_alignment_stack: {folder} is single-pass (n_passes=1) -- "
            "nothing to estimate drift over."
        )
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
    if n_passes <= 1:
        # Genuinely single-pass acquisitions don't have the per-pass raw
        # sidecars (STEM SI_*.raw) this function reads at all -- fail here,
        # before touching any of them, rather than a confusing
        # FileNotFoundError from inside the alignment_stack fallback below.
        raise ValueError(
            f"_load_alignment_stack: {folder} is single-pass (n_passes={n_passes}) -- "
            "nothing to estimate drift over."
        )

    if alignment_stack == "adf":
        adf_info = next(
            (
                o
                for n, o in obj_info.items()
                if "adf" in n.lower() and "postacq" not in n.lower() and o.n_frames == n_passes
            ),
            None,
        )
        if adf_info is None or files["adf_raw"] is None:
            warnings.warn(
                f"_load_alignment_stack: no ADF raw sidecar found for {folder} -- "
                "falling back to alignment_stack='hl' (energy-summed high-loss image "
                "per pass; more expensive to load).",
                UserWarning,
            )
            alignment_stack = "hl"
        else:
            return (
                load_raw_stack(files["adf_raw"], adf_info.n_frames, adf_info.dims, adf_info.dtype),
                n_passes,
            )

    if alignment_stack == "hl":
        stack = load_raw_stack(files["eels_hl_raw"], hl_info.n_frames, hl_info.dims, hl_info.dtype)
    elif alignment_stack == "ll":
        stack = load_raw_stack(files["eels_ll_raw"], ll_info.n_frames, ll_info.dims, ll_info.dtype)
    else:
        raise ValueError(f"alignment_stack must be 'adf', 'll', or 'hl', got {alignment_stack!r}")
    # load_raw_stack() returns (n_frames, n_energy, ny, nx) for EELS sidecars --
    # energy-sum to a single 2D image per pass, same as _load_multi_pass() does
    # for alignment_stack="ll"/"hl".
    return stack.sum(axis=1), n_passes


def plot_pass_drift(
    folder: str | PathLike,
    *,
    alignment_stack: str = "adf",
    reference: str = "mean",
    upsample_factor: int = 10,
    max_shift_px: float | None = None,
    normalization: str | None = None,
    title: str | None = None,
    show: bool = True,
):
    """
    Cheap per-pass drift screening for a multi-pass acquisition folder --
    estimates and plots per-pass `(dy, dx)` shifts WITHOUT loading, aligning,
    or combining the (much larger) LL/HL EELS stacks: only the requested
    `alignment_stack` (`"adf"` by default, a single 2D image per pass) is
    read from disk. Meant to run across many acquisitions as a fast triage
    step -- decide which datasets have real drift worth correcting (see
    `suggest_drift_frames_to_drop()`) before paying for a full
    `align_passes=True` / `remove_drift_frames()` reload of any of them.

    Uses the same estimator `align_passes=True` uses internally
    (`estimate_pass_shifts()`); see its docstring for what each keyword
    parameter here does.

    Parameters
    ----------
    folder : str | PathLike
        Acquisition folder. Raises `ValueError` if it turns out single-pass
        (nothing to estimate drift over).
    title : str, optional
        Plot title. Defaults to `"{folder.name} -- estimated per-pass drift
        ({alignment_stack})"`.
    show : bool, optional
        If True (default), also build and return the `plot_pass_shifts()`
        figure (dx and dy as two separate lines vs. pass number, plus
        magnitude). Set False to only compute `shifts`.

    Returns
    -------
    shifts : (n_passes, 2) ndarray
        `[dy, dx]` per pass, in pixels, 0-indexed pass order (pass 1 is
        `shifts[0]`).
    fig : matplotlib.figure.Figure | None
        `None` if `show=False`.
    """
    folder = Path(folder)
    stack, n_passes = _load_alignment_stack(folder, alignment_stack)
    if n_passes <= 1:
        raise ValueError(
            f"plot_pass_drift: {folder} is single-pass (n_passes={n_passes}) -- "
            "nothing to estimate drift over."
        )
    shifts = estimate_pass_shifts(
        stack,
        reference=reference,
        upsample_factor=upsample_factor,
        max_shift_px=max_shift_px,
        normalization=normalization,
    )
    fig = None
    if show:
        fig = plot_pass_shifts(
            shifts,
            title=title or f"{folder.name} -- estimated per-pass drift ({alignment_stack})",
        )
    return shifts, fig


@dataclass
class DriftFrameSuggestion:
    """Result of `suggest_drift_frames_to_drop()`. All pass indices are
    0-indexed into the `shifts` array passed in (pass 1 is index 0)."""

    drop: list[int]
    keep: list[int]
    leading_run: tuple[int, int] | None
    trailing_run: tuple[int, int] | None
    mid_outliers: list[int]
    mid_outliers_dropped: list[int]
    plateau_magnitude_px: float
    threshold_px: float
    reason: str


def _plot_drift_drop_suggestion(
    shifts: np.ndarray, suggestion: "DriftFrameSuggestion", labels: np.ndarray, title: str
):
    import matplotlib.pyplot as plt

    x = labels.astype(float)
    magnitude = np.hypot(shifts[:, 0], shifts[:, 1])

    fig, (ax_xy, ax_mag) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax in (ax_xy, ax_mag):
        if suggestion.leading_run is not None:
            a, b = suggestion.leading_run
            ax.axvspan(
                x[a] - 0.5,
                x[b] + 0.5,
                color="#C0392B",
                alpha=0.15,
                label="suggested drop (leading)" if ax is ax_xy else None,
            )
        if suggestion.trailing_run is not None:
            a, b = suggestion.trailing_run
            ax.axvspan(
                x[a] - 0.5,
                x[b] + 0.5,
                color="#8E44AD",
                alpha=0.15,
                label="suggested drop (trailing)" if ax is ax_xy else None,
            )

    ax_xy.plot(x, shifts[:, 1], "o-", ms=3, color="#1F4E79", label="dx (col)")
    ax_xy.plot(x, shifts[:, 0], "o-", ms=3, color="#C0392B", label="dy (row)")
    ax_xy.set_xlabel("Pass number")
    ax_xy.set_ylabel("Shift (px)")
    ax_xy.set_title("Shift components vs. pass")
    ax_xy.legend(fontsize=8)
    ax_xy.grid(True, alpha=0.3)

    ax_mag.plot(x, magnitude, "o-", ms=3, color="k", label="|shift|")
    ax_mag.axhline(
        suggestion.threshold_px,
        color="#E67E22",
        ls="--",
        lw=1.2,
        label=f"threshold {suggestion.threshold_px:.2f} px",
    )
    ax_mag.axhline(
        suggestion.plateau_magnitude_px,
        color="0.5",
        ls=":",
        lw=1.2,
        label=f"plateau {suggestion.plateau_magnitude_px:.2f} px",
    )
    mo_kept = [i for i in suggestion.mid_outliers if i not in set(suggestion.mid_outliers_dropped)]
    if mo_kept:
        ax_mag.plot(
            x[mo_kept],
            magnitude[mo_kept],
            "x",
            ms=9,
            mew=2,
            color="#E74C3C",
            label="mid-sequence outlier (NOT dropped)",
        )
    if suggestion.mid_outliers_dropped:
        mo_d = suggestion.mid_outliers_dropped
        ax_mag.plot(
            x[mo_d],
            magnitude[mo_d],
            "x",
            ms=9,
            mew=2,
            color="#8E44AD",
            label="mid-sequence outlier (dropped, also_drop_mid_outliers=True)",
        )
    ax_mag.set_xlabel("Pass number")
    ax_mag.set_ylabel("|shift| (px)")
    ax_mag.set_title("Shift magnitude vs. pass")
    ax_mag.legend(fontsize=8)
    ax_mag.grid(True, alpha=0.3)

    fig.suptitle(f"{title}\n{suggestion.reason}", fontsize=10)
    fig.tight_layout()
    return fig


def suggest_drift_frames_to_drop(
    shifts: np.ndarray,
    *,
    threshold_multiplier: float = 3.0,
    min_threshold_px: float = 0.5,
    plateau_fraction: float = 1.0 / 3.0,
    min_keep: int = 2,
    also_drop_mid_outliers: bool = False,
    guard_passes: int = 0,
    passes_used: Sequence[int] | None = None,
    title: str = "Suggested drift-frame drop",
    show: bool = True,
) -> tuple["DriftFrameSuggestion", Any]:
    """
    Flag which passes of a multi-pass acquisition to drop before combining,
    from their per-pass `(dy, dx)` drift alone -- matches the settling-
    in / late-series-creep pattern (large drift concentrated in a
    CONTIGUOUS run at the start and/or end of the sequence), not scattered
    mid-sequence outliers, which are reported separately (`.mid_outliers`)
    and, by default, never silently folded into `.drop`.

    Why mid-sequence outliers are treated differently by default: an
    isolated single-pass spike can be a genuine transient (real, brief
    specimen jump) as easily as it can be the same physical settling event
    glitching a second time -- distinguishing those needs a human look at
    the plot, not a threshold. `2_HL_InSitu5` is a concrete example of the
    latter: pass 23 there sits at 13.0 px, sandwiched between passes 22 and
    24 at ~0.2-0.3 px, in the same (dy>0, dx<0) direction as the leading
    run's passes -- almost certainly one more settling glitch, not
    chemistry. Once you've looked and agree, pass
    `also_drop_mid_outliers=True` to fold it into `.drop` too; the default
    stays `False` so nothing is dropped without that look.

    Method
    ------
    1. `plateau_magnitude_px` = median `|shift|` over the middle
       `plateau_fraction` of the sequence (default: the middle third).
       Using the middle of the sequence, not all of it, keeps this estimate
       of the acquisition's ordinary/stable drift robust even when a
       leading or trailing run is itself contaminated with large shifts.
    2. `threshold_px` = `max(threshold_multiplier * plateau_magnitude_px,
       min_threshold_px)`. A pass is "flagged" if `|shift| > threshold_px`.
       `min_threshold_px` guards against a near-zero plateau making the
       multiplier alone hypersensitive to ordinary registration noise.
    3. `leading_run` = the longest prefix of flagged passes starting at
       pass 0 (or `None`); `trailing_run` = the longest suffix of flagged
       passes ending at the last pass (or `None`). `.drop` is their union.
       Any other flagged pass is a mid-sequence outlier: listed in
       `.mid_outliers`, excluded from `.drop`.
    4. If the suggested drop would leave fewer than `min_keep` passes,
       nothing is dropped automatically -- `.drop` is empty and `.reason`
       says why (fail loudly rather than silently dropping nearly
       everything).

    Parameters
    ----------
    shifts : (n, 2) ndarray
        `[dy, dx]` per pass, 0-indexed order (as returned by
        `estimate_pass_shifts()` / `plot_pass_drift()`, or
        `StemEelsRaw.pass_shifts_px`).
    threshold_multiplier, min_threshold_px, plateau_fraction, min_keep
        Tunable; see Method above. Defaults were chosen to be conservative
        against the pattern already confirmed on `2_HL_InSitu5` (peak
        15.5 px vs. a ~0.3-0.4 px plateau, i.e. ~40-50x the plateau) --
        `3x` won't over-flag a much milder settling run on other datasets,
        but check the plot either way.
    also_drop_mid_outliers : bool, optional
        Default `False`: mid-sequence outliers are reported
        (`.mid_outliers`) but never dropped automatically. `True`: fold
        them into `.drop`/`.keep` too (recorded separately in
        `.mid_outliers_dropped` so the plot/reason can still show which
        passes were leading/trailing-run drops vs. opted-in mid-sequence
        drops). Look at the plot before turning this on.
    guard_passes : int, optional
        Default `0` (no change from the plain threshold boundary): extend
        each detected `leading_run`/`trailing_run` by this many additional
        passes past where the magnitude first drops below `threshold_px`.
        The pass immediately after a violent settling run can register a
        small rigid shift (this function only fits one shift per whole
        pass) while the specimen was still relaxing *during* that pass --
        a plain threshold cut can accept it as "stable" too early. Set to
        1-2 if the plot shows the pass right at a run's boundary still
        looks like part of the same event even though its own magnitude
        fell under threshold (e.g. `2_HL_InSitu5`'s pass 22, immediately
        after its 21-pass leading run). Extended passes are recorded the
        same as any other leading/trailing drop (not as
        `.mid_outliers`/`.mid_outliers_dropped`, since they were never
        flagged by the threshold in the first place -- `.reason` notes the
        extension explicitly either way).
    passes_used : sequence of int, optional
        1-indexed pass numbers `shifts` corresponds to (e.g.
        `StemEelsRaw.passes_used`), for readable `.reason` text and plot
        labels only. Defaults to `1..n`.
    show : bool, optional
        If True (default), also build a figure: dx/dy and `|shift|` vs.
        pass, with the suggested drop region(s) shaded and mid-sequence
        outliers marked with an `x` -- meant to be visually checked, not
        trusted blindly.

    Returns
    -------
    suggestion : DriftFrameSuggestion
    fig : matplotlib.figure.Figure | None
        `None` if `show=False`.
    """
    shifts = np.asarray(shifts, dtype=float)
    n = len(shifts)
    if n == 0:
        raise ValueError("suggest_drift_frames_to_drop: shifts is empty")
    labels = np.asarray(passes_used, dtype=int) if passes_used is not None else np.arange(1, n + 1)

    magnitude = np.hypot(shifts[:, 0], shifts[:, 1])

    half_window = max(1, int(round(n * plateau_fraction / 2)))
    mid = n // 2
    lo, hi = max(0, mid - half_window), min(n, mid + half_window)
    if hi <= lo:
        lo, hi = 0, n
    plateau_magnitude = float(np.median(magnitude[lo:hi]))
    threshold = max(threshold_multiplier * plateau_magnitude, min_threshold_px)
    flagged = magnitude > threshold

    leading_run = None
    i = 0
    while i < n and flagged[i]:
        i += 1
    if i > 0:
        leading_run = (0, i - 1)

    trailing_run = None
    j = n - 1
    while j >= 0 and flagged[j]:
        j -= 1
    if j < n - 1:
        trailing_run = (j + 1, n - 1)

    guard_extended = False
    if guard_passes > 0:
        if leading_run is not None:
            new_end = min(leading_run[1] + guard_passes, n - 1)
            if trailing_run is not None:
                new_end = min(new_end, trailing_run[0] - 1)
            if new_end > leading_run[1]:
                leading_run = (leading_run[0], new_end)
                guard_extended = True
        if trailing_run is not None:
            new_start = max(trailing_run[0] - guard_passes, 0)
            if leading_run is not None:
                new_start = max(new_start, leading_run[1] + 1)
            if new_start < trailing_run[0]:
                trailing_run = (new_start, trailing_run[1])
                guard_extended = True

    drop_set: set[int] = set()
    if leading_run is not None:
        drop_set.update(range(leading_run[0], leading_run[1] + 1))
    if trailing_run is not None:
        drop_set.update(range(trailing_run[0], trailing_run[1] + 1))
    keep_idx = sorted(set(range(n)) - drop_set)

    if len(keep_idx) < min_keep:
        reason = (
            f"the leading/trailing high-drift run(s) would leave only {len(keep_idx)} "
            f"pass(es) (< min_keep={min_keep}) -- refusing to auto-drop; threshold was "
            f"{threshold:.2f} px ({threshold_multiplier:g}x the {plateau_magnitude:.2f} px "
            "plateau estimate). Check the plot and pick a drop_list by hand, or loosen "
            "threshold_multiplier/plateau_fraction/min_keep."
        )
        leading_run = trailing_run = None
        drop_idx: list[int] = []
        mid_outliers = sorted(int(k) for k in np.flatnonzero(flagged))
        mid_outliers_dropped: list[int] = []
        keep_idx = list(range(n))
    else:
        mid_outliers = sorted(i for i in range(n) if flagged[i] and i not in drop_set)
        mid_outliers_dropped = list(mid_outliers) if also_drop_mid_outliers else []
        drop_set = drop_set | set(mid_outliers_dropped)
        drop_idx = sorted(drop_set)
        keep_idx = sorted(set(range(n)) - drop_set)
        parts = []
        if leading_run is not None:
            a, b = leading_run
            parts.append(
                f"leading: passes {int(labels[a])}-{int(labels[b])} "
                f"({b - a + 1} passes, peak {magnitude[a : b + 1].max():.1f} px)"
            )
        if trailing_run is not None:
            a, b = trailing_run
            parts.append(
                f"trailing: passes {int(labels[a])}-{int(labels[b])} "
                f"({b - a + 1} passes, peak {magnitude[a : b + 1].max():.1f} px)"
            )
        if not parts:
            reason = (
                f"no leading/trailing high-drift run found (threshold {threshold:.2f} px, "
                f"{threshold_multiplier:g}x the {plateau_magnitude:.2f} px plateau estimate); "
                "nothing suggested to drop."
            )
        else:
            reason = (
                f"{' and '.join(parts)} vs. {plateau_magnitude:.2f} px plateau "
                f"(threshold {threshold:.2f} px)"
            )
        if guard_extended:
            reason += f"; extended by guard_passes={guard_passes}"
        if mid_outliers_dropped:
            reason += (
                f"; also dropped {len(mid_outliers_dropped)} mid-sequence outlier(s) "
                f"(also_drop_mid_outliers=True): passes "
                f"{[int(labels[i]) for i in mid_outliers_dropped]}"
            )
        elif mid_outliers:
            reason += (
                f"; {len(mid_outliers)} mid-sequence outlier(s) NOT dropped (pass "
                f"also_drop_mid_outliers=True to include them): passes "
                f"{[int(labels[i]) for i in mid_outliers]}"
            )

    suggestion = DriftFrameSuggestion(
        drop=drop_idx,
        keep=keep_idx,
        leading_run=leading_run,
        trailing_run=trailing_run,
        mid_outliers=mid_outliers,
        mid_outliers_dropped=mid_outliers_dropped,
        plateau_magnitude_px=plateau_magnitude,
        threshold_px=threshold,
        reason=reason,
    )

    fig = None
    if show:
        fig = _plot_drift_drop_suggestion(shifts, suggestion, labels, title=title)
    return suggestion, fig


def remove_drift_frames(
    folder_or_raw: "str | PathLike | MultipassRawStacks",
    drop_list: Sequence[int] | None = None,
    *,
    pass_mode: str = "manual",
    passes: str | Sequence[int] | None = "all",
    combine_method: str = "sum",
    alignment_stack: str = "adf",
    alignment_reference: str = "mean",
    alignment_upsample_factor: int = 10,
    alignment_max_shift_px: float | None = None,
    alignment_normalization: str | None = None,
    drop_threshold_multiplier: float = 3.0,
    drop_min_threshold_px: float = 0.5,
    drop_plateau_fraction: float = 1.0 / 3.0,
    drop_min_keep: int = 2,
    drop_also_mid_outliers: bool = False,
    drop_guard_passes: int = 0,
) -> "StemEelsRaw":
    """
    Multi-pass combine step that fixes the coverage-dilution problem of the
    plain `align_passes=True` path: **align -> DROP the leading/trailing
    high-drift passes -> THEN sum**, instead of summing every aligned pass
    including the ones that mostly contribute zero-padded border (see
    `crop_alignment_border()`'s docstring for why that border matters even
    after dropping).

    Parameters
    ----------
    folder_or_raw : str | PathLike | MultipassRawStacks
        An acquisition folder (loaded internally via
        `load_multipass_raw_stacks()`), or an already-loaded
        `MultipassRawStacks` (e.g. to reuse one load across several
        `drop_list` trials without re-reading the raw sidecars each time).
    drop_list : sequence of int, optional
        1-indexed passes to drop, overriding automatic detection entirely.
        `None` (default): `suggest_drift_frames_to_drop()` decides, using
        the `drop_*` parameters below. Its mid-sequence outliers (if any)
        are reported via a `UserWarning` but NOT dropped automatically --
        pass them in `drop_list` yourself if you want them out too.
    pass_mode, passes, combine_method
        Forwarded to `select_passes()`/`combine_passes()`, same meaning as
        in `read_stem_eels_folder()`. `passes="all"` (default) considers
        every recorded pass as a drop/keep candidate.
    alignment_stack, alignment_reference, alignment_upsample_factor,
    alignment_max_shift_px, alignment_normalization
        Forwarded to `estimate_pass_shifts()`, same meaning as in
        `read_stem_eels_folder(..., align_passes=True)`.
    drop_threshold_multiplier, drop_min_threshold_px, drop_plateau_fraction,
    drop_min_keep, drop_also_mid_outliers, drop_guard_passes
        Forwarded to `suggest_drift_frames_to_drop()`'s equivalents (see
        its docstring) when `drop_list` is `None`; ignored otherwise.

    Returns
    -------
    StemEelsRaw
        Same shape of result as `read_stem_eels_folder()`, with two extra
        attributes: `.dropped_passes` (1-indexed, matching `.passes_used`'s
        convention) and `.drop_reason`. `.pass_shifts_px` covers only the
        KEPT passes, in the same order as `.passes_used` -- feed both
        straight to `crop_alignment_border()`.
    """
    if isinstance(folder_or_raw, MultipassRawStacks):
        raw = folder_or_raw
        folder = raw.folder
    else:
        folder = Path(folder_or_raw)
        raw = load_multipass_raw_stacks(folder)

    chosen = select_passes(raw.n_passes, mode=pass_mode, passes=passes)
    if len(chosen) <= 1:
        raise ValueError(
            f"remove_drift_frames: {folder} has only {len(chosen)} selected pass(es) -- "
            "nothing to align or drop."
        )

    # --- 1. align (same estimator align_passes=True uses) ---
    stack_kind = alignment_stack
    if stack_kind == "adf" and raw.adf_stack is None:
        warnings.warn(
            "remove_drift_frames: alignment_stack='adf' but no ADF stack was found for "
            f"{folder} -- falling back to alignment_stack='hl'.",
            UserWarning,
        )
        stack_kind = "hl"
    if stack_kind == "adf":
        registration_images = raw.adf_stack
    elif stack_kind == "ll":
        registration_images = raw.ll_stack.sum(axis=1)
    elif stack_kind == "hl":
        registration_images = raw.hl_stack.sum(axis=1)
    else:
        raise ValueError(f"alignment_stack must be 'adf', 'll', or 'hl', got {alignment_stack!r}")

    shifts = estimate_pass_shifts(
        registration_images,
        chosen,
        reference=alignment_reference,
        upsample_factor=alignment_upsample_factor,
        max_shift_px=alignment_max_shift_px,
        normalization=alignment_normalization,
    )

    # --- 2. drop the flagged high-drift passes, BEFORE summing ---
    if drop_list is None:
        suggestion, _ = suggest_drift_frames_to_drop(
            shifts,
            threshold_multiplier=drop_threshold_multiplier,
            min_threshold_px=drop_min_threshold_px,
            plateau_fraction=drop_plateau_fraction,
            min_keep=drop_min_keep,
            also_drop_mid_outliers=drop_also_mid_outliers,
            guard_passes=drop_guard_passes,
            passes_used=[p + 1 for p in chosen],
            show=False,
        )
        drop_local = suggestion.drop  # indices into `chosen`/`shifts`
        reason = suggestion.reason
        still_kept_outliers = [
            i for i in suggestion.mid_outliers if i not in set(suggestion.mid_outliers_dropped)
        ]
        if still_kept_outliers:
            warnings.warn(
                f"remove_drift_frames: {len(still_kept_outliers)} mid-sequence outlier "
                "pass(es) exceeded the drift threshold but are not part of the leading/"
                "trailing run and were NOT dropped automatically: "
                f"{[chosen[i] + 1 for i in still_kept_outliers]} (1-indexed). Pass "
                "them in drop_list explicitly, or set drop_also_mid_outliers=True, "
                "if you want them removed too.",
                UserWarning,
            )
    else:
        drop_set_1idx = {int(d) for d in drop_list}
        drop_local = [i for i, p in enumerate(chosen) if (p + 1) in drop_set_1idx]
        reason = "manual drop_list"

    keep_local = [i for i in range(len(chosen)) if i not in set(drop_local)]
    if len(keep_local) < 2:
        raise ValueError(
            f"remove_drift_frames: only {len(keep_local)} pass(es) would remain after "
            f"dropping {len(drop_local)} -- refusing to combine. reason: {reason}"
        )
    kept_chosen = [chosen[i] for i in keep_local]  # 0-indexed frame numbers
    kept_shifts = shifts[keep_local]
    dropped_chosen = [chosen[i] for i in drop_local]

    # --- align + combine the KEPT passes only (load/shift only their data,
    # not the dropped passes') ---
    ll_stack = apply_pass_shifts(raw.ll_stack[kept_chosen], kept_shifts)
    hl_stack = apply_pass_shifts(raw.hl_stack[kept_chosen], kept_shifts)
    adf_stack = (
        apply_pass_shifts(raw.adf_stack[kept_chosen], kept_shifts)
        if raw.adf_stack is not None
        else None
    )

    idx_all = list(range(len(kept_chosen)))
    ll_combined = np.moveaxis(combine_passes(ll_stack, idx_all, method=combine_method), 0, -1)
    hl_combined = np.moveaxis(combine_passes(hl_stack, idx_all, method=combine_method), 0, -1)
    adf_combined = (
        combine_passes(adf_stack, idx_all, method="mean") if adf_stack is not None else None
    )

    eels_ll = array_to_spectroscopy3d(
        ll_combined, raw.ll_energy_axis, raw.pixel_size_nm, name="EELS_LL"
    )
    eels_hl = array_to_spectroscopy3d(
        hl_combined, raw.hl_energy_axis, raw.pixel_size_nm, name="EELS_HL"
    )

    print(
        f"remove_drift_frames: dropped {len(dropped_chosen)}/{len(chosen)} passes "
        f"({reason}); kept {len(kept_chosen)} for combining."
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
        passes_used=[p + 1 for p in kept_chosen],
        combine_method=combine_method,
        pass_shifts_px=kept_shifts,
        dropped_passes=[p + 1 for p in dropped_chosen],
        drop_reason=reason,
    )


def _compute_crop_margins(
    shifts: np.ndarray,
    ny: int,
    nx: int,
    extra_margin_px: int = 0,
) -> tuple[int, int, int, int]:
    """
    Pure computation behind `crop_alignment_border`: given the non-empty
    per-pass `[dy, dx]` shifts actually applied (same sign convention as
    `apply_pass_shifts`) and the field-of-view size `(ny, nx)`, compute the
    `(top, bottom, left, right)` pixel margins to crop so every surviving
    pass's zero-filled/under-sampled border is excluded.

    A pass shifted by `+dy` exposes a zero border at the TOP
    (low-row-index) side, `-dy` at the bottom, and likewise `+dx` left /
    `-dx` right (matches `scipy.ndimage.shift`'s convention, as used by
    `apply_pass_shifts`).

    Takes/returns only plain arrays and numbers -- no `StemEelsRaw`
    dependency -- so it can be unit-tested directly against synthetic
    shift arrays; see `tests/core/io/test_pass_drift.py`.

    Parameters
    ----------
    shifts : (n, 2) ndarray
        Non-empty `[dy, dx]` per pass. The "no shifts at all" precondition
        is `crop_alignment_border`'s concern (it depends on where `shifts`
        came from -- `combined_data.pass_shifts_px` vs. an explicit
        `shifts_used` -- which this function doesn't know about), not
        re-checked here.
    ny, nx : int
        Field-of-view size, in pixels, of the array being cropped.
    extra_margin_px : int, optional
        Additional pixels to add on every side beyond the computed
        coverage-safe margin. Default 0.

    Returns
    -------
    (top, bottom, left, right) : tuple of int

    Raises
    ------
    ValueError
        If the computed margins would consume the entire `(ny, nx)` field
        (no pixels would survive the crop).
    """
    shifts = np.asarray(shifts, dtype=float)
    dy, dx = shifts[:, 0], shifts[:, 1]
    top = int(np.ceil(max(0.0, float(dy.max())))) + extra_margin_px
    bottom = int(np.ceil(max(0.0, float(-dy.min())))) + extra_margin_px
    left = int(np.ceil(max(0.0, float(dx.max())))) + extra_margin_px
    right = int(np.ceil(max(0.0, float(-dx.min())))) + extra_margin_px

    if top + bottom >= ny or left + right >= nx:
        raise ValueError(
            f"crop_alignment_border: computed margins (top={top}, bottom={bottom}, "
            f"left={left}, right={right}) would leave no pixels on a {ny}x{nx} field -- "
            "the shifts are too large relative to the field of view for a safe crop."
        )
    return top, bottom, left, right


def crop_alignment_border(
    combined_data: "StemEelsRaw",
    shifts_used: np.ndarray | None = None,
    *,
    extra_margin_px: int = 0,
    modify_in_place: bool = False,
) -> "StemEelsRaw":
    """
    Crop away the zero-filled/under-sampled field-of-view border left by
    aligning + summing shifted passes, and **update the cropped datasets'
    spatial metadata** (`origin`) to match.

    Why this matters: after `apply_pass_shifts()` shifts each pass by its
    own `(dy, dx)`, the newly-exposed border of that pass reads as 0
    (`mode="constant", cval=0.0`) -- summing shifted passes together means
    that border region is under-sampled (fewer, or zero, real passes
    contribute there) relative to the interior, which every downstream
    per-pixel/map/mean-spectrum computation would otherwise silently
    include. A **naive fix that only slices the array**
    (`array[top:-bottom, left:-right]`) leaves the OLD `origin` in place --
    every per-pixel coordinate downstream code derives from
    `origin + pixel_index * sampling` (e.g. `Dataset3dspectroscopy`'s own
    axis properties, spatial-coherence/ADF-correlation checks, anything
    that overlays a map on the ADF image) would then be silently offset by
    exactly the cropped margin. This function crops via `Dataset.crop()`,
    which updates `origin` by `crop_start * sampling` on the cropped axes
    (`sampling`, the per-pixel physical size, is unchanged by cropping) --
    see the verification in this module's tests / the alignment-check
    notebook for a before/after comparison confirming this.

    Crops to exactly the region every SURVIVING, shifted pass actually
    covers -- computed per-side from the signed shifts (not a fixed or
    symmetric margin): a pass shifted by `+dy` exposes a zero border at the
    TOP (low-row-index) side, `-dy` at the bottom, and likewise `+dx`
    left / `-dx` right (verified empirically against
    `scipy.ndimage.shift`'s convention).

    Parameters
    ----------
    combined_data : StemEelsRaw
        The result of `remove_drift_frames()` (or
        `read_stem_eels_folder(..., align_passes=True)`) -- must carry real
        per-pass shifts.
    shifts_used : (n, 2) ndarray, optional
        `[dy, dx]` per pass actually summed, 0-indexed order. Defaults to
        `combined_data.pass_shifts_px`.
    extra_margin_px : int, optional
        Additional pixels to crop on every side beyond the computed
        coverage-safe margin (e.g. to also clear residual interpolation
        softening right at the coverage boundary, from `apply_pass_shifts`'
        default bilinear `order=1`). Default 0.
    modify_in_place : bool, optional
        Forwarded to `Dataset.crop()` for `eels_ll`/`eels_hl`. Default
        `False`: returns a new `StemEelsRaw`, `combined_data` untouched.

    Returns
    -------
    StemEelsRaw
        New object (or `combined_data` itself, mutated, if
        `modify_in_place=True`) with `eels_ll`/`eels_hl` cropped and their
        `.origin` updated to match, and `.adf` cropped identically (a plain
        array -- no metadata of its own to update).
    """
    shifts = (
        np.asarray(shifts_used, dtype=float)
        if shifts_used is not None
        else combined_data.pass_shifts_px
    )
    if shifts is None or len(shifts) == 0:
        raise ValueError(
            "crop_alignment_border: no pass_shifts_px on this StemEelsRaw and no "
            "shifts_used given -- was this combined with align_passes=True / "
            "remove_drift_frames()?"
        )
    shifts = np.asarray(shifts, dtype=float)

    ny, nx = combined_data.eels_hl.shape[0], combined_data.eels_hl.shape[1]
    top, bottom, left, right = _compute_crop_margins(shifts, ny, nx, extra_margin_px)

    dy, dx = shifts[:, 0], shifts[:, 1]
    crop_widths = ((top, ny - bottom), (left, nx - right))
    if modify_in_place:
        combined_data.eels_ll.crop(crop_widths, axes=(0, 1), modify_in_place=True)
        combined_data.eels_hl.crop(crop_widths, axes=(0, 1), modify_in_place=True)
        eels_ll, eels_hl = combined_data.eels_ll, combined_data.eels_hl
    else:
        eels_ll = combined_data.eels_ll.crop(crop_widths, axes=(0, 1))
        eels_hl = combined_data.eels_hl.crop(crop_widths, axes=(0, 1))

    adf = combined_data.adf
    if adf is not None:
        adf = adf[top : ny - bottom, left : nx - right]

    print(
        f"crop_alignment_border: cropped to rows [{top}:{ny - bottom}] (of {ny}), "
        f"cols [{left}:{nx - right}] (of {nx}) -- margins top={top} bottom={bottom} "
        f"left={left} right={right} px, from shifts dy in "
        f"[{dy.min():.2f}, {dy.max():.2f}], dx in [{dx.min():.2f}, {dx.max():.2f}]"
    )

    if modify_in_place:
        combined_data.adf = adf
        return combined_data

    return StemEelsRaw(
        folder=combined_data.folder,
        dm4_path=combined_data.dm4_path,
        is_multipass=combined_data.is_multipass,
        n_passes=combined_data.n_passes,
        eels_ll=eels_ll,
        eels_hl=eels_hl,
        adf=adf,
        energy_axis_ll=combined_data.energy_axis_ll,
        energy_axis_hl=combined_data.energy_axis_hl,
        pixel_size_nm=combined_data.pixel_size_nm,
        passes_used=combined_data.passes_used,
        combine_method=combined_data.combine_method,
        pass_shifts_px=combined_data.pass_shifts_px,
        dropped_passes=getattr(combined_data, "dropped_passes", None),
        drop_reason=getattr(combined_data, "drop_reason", None),
    )


def crop_unacquired_rows(raw: StemEelsRaw) -> StemEelsRaw:
    """
    Crop away the scan rows a spectrum-image acquisition never reached.

    When an acquisition is stopped part way through, DigitalMicrograph saves the
    full requested raster with the unvisited pixels left at exactly zero (whole
    trailing rows, plus the tail of the row it stopped in). Those all-zero spectra
    have no ZLP and derail per-pixel ZLP alignment / thickness mapping, so keep
    only the leading block of rows in which every pixel has LL signal. Returns
    `raw` itself, untouched, when every row is complete (or none is).
    """
    total = np.asarray(raw.eels_ll.array).sum(axis=-1)
    ny = total.shape[0]
    complete = (total > 0).all(axis=1)
    if complete.all() or not complete.any():
        return raw
    start = int(np.argmax(complete))
    incomplete_after = np.flatnonzero(~complete[start:])
    stop = start + int(incomplete_after[0]) if len(incomplete_after) else ny
    crop_widths = ((start, stop), (0, total.shape[1]))
    print(
        f"crop_unacquired_rows: kept rows [{start}:{stop}] of {ny} -- the rest were "
        f"never (fully) acquired ({int((total <= 0).sum())} all-zero pixels)"
    )
    return StemEelsRaw(
        folder=raw.folder,
        dm4_path=raw.dm4_path,
        is_multipass=raw.is_multipass,
        n_passes=raw.n_passes,
        eels_ll=raw.eels_ll.crop(crop_widths, axes=(0, 1)),
        eels_hl=raw.eels_hl.crop(crop_widths, axes=(0, 1)),
        adf=raw.adf[start:stop] if raw.adf is not None else None,
        energy_axis_ll=raw.energy_axis_ll,
        energy_axis_hl=raw.energy_axis_hl,
        pixel_size_nm=raw.pixel_size_nm,
        passes_used=raw.passes_used,
        combine_method=raw.combine_method,
        pass_shifts_px=raw.pass_shifts_px,
        dropped_passes=getattr(raw, "dropped_passes", None),
        drop_reason=getattr(raw, "drop_reason", None),
    )


def suggest_pass_range_for_analysis(
    drift_suggestion: "DriftFrameSuggestion",
    dose_block_results: Sequence[dict],
    *,
    metric_key: str = "ratio",
    trend_corr_threshold: float = 0.6,
    trend_p_threshold: float = 0.05,
    trend_relative_range_threshold: float = 0.05,
) -> dict:
    """
    Combine the drift screen (position/settling) and a dose series
    (chemistry-or-thickness vs. dose) into ONE recommendation: which passes
    to actually sum for the most reliable signal.

    WHAT
    ----
    Starts from `drift_suggestion.keep` (passes surviving the settling-
    in / late-series-creep filter, see `suggest_drift_frames_to_drop()`).
    Then tests `dose_block_results` (one dict per dose block, in
    acquisition order, each carrying at least `metric_key`) for a genuine
    monotonic trend with dose, via a Spearman rank correlation between
    block order and the metric.

    WHY / WHAT WE LEARN
    --------------------
    Two independent reasons a multi-pass acquisition's later passes can be
    unusable: (1) drift/settling smears real space early on -- a POSITION
    problem, fixed by dropping/aligning (`remove_drift_frames()`); (2)
    radiolysis (ionization damage: bonding/chemistry changes with dose,
    thickness can stay flat) or knock-on damage (displacement damage: mass
    is actually removed, thickness drops) change the MATERIAL itself as
    dose accumulates -- no amount of alignment fixes that; the only fix is
    not summing the damaged passes. This function's job is to say, in one
    place, given both checks: which pass range is actually safe to sum.
    Run it twice -- once with `metric_key` set to a chemistry/shape ratio
    (e.g. `pre_edge_white_line_ratio`'s `"ratio"`, the "Radiolysis
    (ionization) damage" signal) and once with `metric_key="t_mean"` (the
    "Knock-on/mass-loss damage" signal) -- they can show different, or no,
    trends independently; do not average them into one verdict.

    CAVEAT on `metric_key="t_mean"` -- direction matters, read `"direction"`
    before calling it "knock-on damage". Knock-on/displacement damage
    removes mass, so its signature is thickness DECREASING with dose. A
    thickness INCREASE with dose is a different, common artifact instead:
    beam-induced contamination buildup -- residual hydrocarbons or water
    vapor in the vacuum system, cracked and deposited by the beam onto the
    irradiated area (ice or amorphous carbon). This function reports which
    direction it saw (`result["direction"]`) precisely so "damage_detected"
    is never read as "knock-on" without checking that it's actually the
    thinning direction; a real, significant, but SMALL rising trend (see
    the `2_HL_InSitu5` case in this module's own tests: Spearman rho=0.94,
    p=0.005, but only 2.3% of the median, correctly not flagged by the
    `trend_relative_range_threshold` gate) is more consistent with mild
    contamination than with knock-on -- and either way, "not flagged as
    damage" is not the same claim as "no contamination is present," only
    that it isn't large enough here to warrant cutting passes over.

    HOW
    ---
    A trend is flagged "real" only if ALL of:
    - Spearman `|rho| >= trend_corr_threshold` (default 0.6: a fairly
      consistent monotonic direction, not scatter) -- rank correlation is
      used (not Pearson) so the test doesn't assume the trend is linear,
      only monotonic;
    - `p-value <= trend_p_threshold` (default 0.05);
    - the metric's total peak-to-peak range is
      `>= trend_relative_range_threshold` (default 5%) of its own median --
      guards against a statistically "significant" but physically
      negligible wobble in a metric with very low block-to-block noise.
    If flagged, the recommended cutoff is the first block whose metric
    value leaves the range spanned by the first two blocks (assumed
    representative of the undamaged/early state) -- i.e. where the series
    visibly leaves its early plateau -- and every pass in later blocks is
    dropped from the recommendation. If not flagged, every
    `drift_suggestion.keep` pass is kept.

    THIS IS A SCREENING HEURISTIC, NOT A PROOF.
    Its thresholds are defaults, not physical constants -- tune them and
    always look at the plotted dose series yourself before trusting the
    recommendation blindly, the same way `suggest_drift_frames_to_drop()`'s
    suggestion is meant to be checked against its shaded-region plot, not
    applied automatically. With only a handful of dose blocks (typical: 4-8)
    the Spearman test has limited power -- a real but small trend can fail
    to reach `trend_p_threshold` simply from too few points, not because it
    isn't there.

    Parameters
    ----------
    drift_suggestion : DriftFrameSuggestion
        From `suggest_drift_frames_to_drop()`.
    dose_block_results : sequence of dict
        One dict per dose block, in acquisition order, each with at least
        `metric_key` (float). Block order is taken from list order, not
        from any pass-number field.
    metric_key : str, optional
        Which key in each `dose_block_results` dict to test for a trend.
        Default `"ratio"`.
    trend_corr_threshold, trend_p_threshold, trend_relative_range_threshold
        Tunable; see HOW above.

    Returns
    -------
    dict
        `"recommended_passes"`: the subset of `drift_suggestion.keep`
        (0-indexed, same convention) surviving both checks.
        `"damage_detected"`: bool. `"cutoff_block"`: int index into
        `dose_block_results`, or `None`. `"direction"`: `"increasing"` /
        `"decreasing"` / `"flat"` (sign of the metric's first-to-last
        change; see the CAVEAT above -- for `metric_key="t_mean"`,
        `"decreasing"` is the knock-on/mass-loss signature,
        `"increasing"` is more consistent with contamination buildup,
        reported regardless of `damage_detected` so the direction is never
        silently lost even when the trend wasn't large enough to flag).
        `"spearman_rho"`, `"spearman_p"`: the test statistics.
        `"reason"`: human-readable summary.
    """
    from scipy.stats import spearmanr

    values = np.array([float(b[metric_key]) for b in dose_block_results], dtype=float)
    n_blocks = len(values)
    if n_blocks < 3:
        return dict(
            recommended_passes=list(drift_suggestion.keep),
            damage_detected=False,
            cutoff_block=None,
            direction="flat",
            spearman_rho=float("nan"),
            spearman_p=float("nan"),
            reason=f"only {n_blocks} dose block(s) -- too few to test a trend; "
            "keeping every drift-surviving pass.",
        )

    order = np.arange(n_blocks)
    rho, p = spearmanr(order, values)
    rel_range = (
        (values.max() - values.min()) / abs(np.median(values)) if np.median(values) else 0.0
    )
    direction = (
        "increasing"
        if values[-1] > values[0]
        else ("decreasing" if values[-1] < values[0] else "flat")
    )

    damage_detected = (
        abs(rho) >= trend_corr_threshold
        and p <= trend_p_threshold
        and rel_range >= trend_relative_range_threshold
    )

    if not damage_detected:
        # Distinguish WHY it wasn't flagged: the Spearman test itself can fail to reach
        # significance (rho/p gate), or it can pass that gate but still not clear the
        # magnitude gate (rel_range) -- these are different claims, and collapsing both
        # into "no significant trend" is wrong for the second case: rho/p there ARE
        # significant, the trend is just too small to act on (see the docstring's
        # 2_HL_InSitu5 example: rho=0.94, p=0.005, range=2.3% -- a real but tiny trend).
        corr_significant = abs(rho) >= trend_corr_threshold and p <= trend_p_threshold
        if corr_significant:
            reason = (
                f"a statistically real monotonic trend in '{metric_key}' was detected across "
                f"{n_blocks} dose blocks (Spearman rho={rho:.2f}, p={p:.3f}, direction="
                f"{direction}), but its magnitude (range={rel_range * 100:.1f}% of median) is "
                f"below trend_relative_range_threshold ({trend_relative_range_threshold * 100:.0f}%) "
                "-- too small to warrant cutting passes over, not statistically absent. Keeping "
                "every drift-surviving pass."
            )
        else:
            reason = (
                f"no significant monotonic trend detected in '{metric_key}' across {n_blocks} "
                f"dose blocks (Spearman rho={rho:.2f}, p={p:.3f}, range={rel_range * 100:.1f}% "
                f"of median, direction={direction}) -- keeping every drift-surviving pass."
            )
        return dict(
            recommended_passes=list(drift_suggestion.keep),
            damage_detected=False,
            cutoff_block=None,
            direction=direction,
            spearman_rho=float(rho),
            spearman_p=float(p),
            reason=reason,
        )

    # early-plateau reference = the range spanned by the first two blocks
    early = values[: min(2, n_blocks)]
    lo, hi = early.min(), early.max()
    pad = max(hi - lo, 1e-9)
    cutoff_block = None
    for i in range(2, n_blocks):
        if values[i] < lo - pad or values[i] > hi + pad:
            cutoff_block = i
            break

    reason = (
        f"significant monotonic trend in '{metric_key}' (Spearman rho={rho:.2f}, "
        f"p={p:.3f}, range={rel_range * 100:.1f}% of median, direction={direction}) "
        f"across {n_blocks} dose blocks -- consistent with progressive damage. "
    )
    if metric_key == "t_mean":
        reason += (
            "decreasing -> knock-on/mass-loss signature. "
            if direction == "decreasing"
            else "increasing -> more consistent with contamination buildup (ice/carbon) "
            "than knock-on damage, which removes mass rather than adding it. "
        )
    if cutoff_block is None:
        reason += "trend detected but never clearly leaves the early plateau -- flagged, not cut."
        recommended_passes = list(drift_suggestion.keep)
    else:
        reason += f"recommend cutting at dose block {cutoff_block} (0-indexed)."
        n_drift_blocks_worth = len(drift_suggestion.keep)
        cut_at = int(round(n_drift_blocks_worth * cutoff_block / n_blocks))
        recommended_passes = list(drift_suggestion.keep[:cut_at])

    return dict(
        recommended_passes=recommended_passes,
        damage_detected=True,
        cutoff_block=cutoff_block,
        direction=direction,
        spearman_rho=float(rho),
        spearman_p=float(p),
        reason=reason,
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
