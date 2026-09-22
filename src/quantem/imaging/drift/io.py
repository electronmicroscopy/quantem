"""Read Velox EMD images, spectrum images, and scan metadata for drift correction."""

import json
import math
from pathlib import Path

import h5py
import numpy as np


def _decode_velox_json(dataset: h5py.Dataset) -> dict:
    """Decode one JSON record stored as bytes or a padded uint8 array."""
    value = dataset[0] if dataset.dtype.kind in {"O", "S", "U"} else dataset[:]
    if isinstance(value, str):
        raw = value.encode()
    elif isinstance(value, bytes):
        raw = value
    else:
        raw = bytes(np.asarray(value, dtype=np.uint8).reshape(-1))
    raw = raw.replace(b"\x00", b"")
    for candidate in (raw, _deinterleave_velox_metadata(raw)):
        try:
            return json.loads(candidate.decode("utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not decode Velox JSON dataset {dataset.name}")


def _read_emd_energy_windows(
    path: str | Path,
    energy_windows: dict[str, tuple[float, float]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Count selected energy windows directly from compressed EDS streams."""
    windows: dict[str, tuple[float, float]] = {}
    for name, limits in energy_windows.items():
        if len(limits) != 2:
            raise ValueError(f"Energy window {name!r} must contain (low, high)")
        low, high = (float(limits[0]), float(limits[1]))
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise ValueError(f"Invalid energy window {name!r}: {limits!r}")
        windows[str(name)] = (low, high)
    if not windows:
        return {}, np.empty(0, dtype=np.float32)

    maps: dict[str, np.ndarray] | None = None
    reference_shape: tuple[int, int] | None = None
    reference_axis: np.ndarray | None = None
    with h5py.File(path, "r") as handle:
        stream_group = handle.get("Data/SpectrumStream")
        if stream_group is None or not stream_group:
            raise ValueError(f"{path} contains no Velox SpectrumStream data")
        for stream in stream_group.values():
            settings = _decode_velox_json(stream["AcquisitionSettings"])
            raster = settings["RasterScanDefinition"]
            shape = (int(raster["Height"]), int(raster["Width"]))
            channels = int(settings["bincount"])
            if reference_shape is None:
                reference_shape = shape
                maps = {
                    name: np.zeros(shape[0] * shape[1], dtype=np.uint32)
                    for name in windows
                }
            elif shape != reference_shape:
                raise ValueError("EDS detector streams have inconsistent scan shapes")

            metadata = _decode_velox_json(stream["Metadata"])
            detector_name = metadata["BinaryResult"]["Detector"]
            detector = next(
                (
                    item
                    for item in metadata["Detectors"].values()
                    if item.get("DetectorName") == detector_name
                ),
                None,
            )
            if detector is None:
                raise ValueError(f"Missing calibration for EDS detector {detector_name}")
            scale = np.float32(float(detector["Dispersion"]) / 1000.0)
            offset = np.float32(float(detector["OffsetEnergy"]) / 1000.0)
            axis = np.arange(channels, dtype=np.float32) * scale + offset
            if reference_axis is None:
                reference_axis = axis
            elif not np.array_equal(axis, reference_axis):
                raise ValueError("EDS detector streams have inconsistent energy axes")

            encoded = np.asarray(stream["Data"][:, 0], dtype=np.uint16)
            gates = encoded == np.uint16(65535)
            pixel_index = np.cumsum(gates, dtype=np.int32)
            pixel_count = shape[0] * shape[1]
            assert maps is not None
            for name, (low, high) in windows.items():
                selected_channels = (axis >= low) & (axis <= high)
                lookup = np.zeros(65536, dtype=bool)
                lookup[:channels] = selected_channels
                selected_events = (~gates) & lookup[encoded]
                counts = np.bincount(
                    pixel_index[selected_events] % pixel_count,
                    minlength=pixel_count,
                )
                maps[name] += counts[:pixel_count].astype(np.uint32, copy=False)

    assert maps is not None and reference_shape is not None and reference_axis is not None
    return (
        {
            name: values.reshape(reference_shape).astype(np.float32)
            for name, values in maps.items()
        },
        reference_axis,
    )


def _axis_calibration(
    axes: list[dict],
    ndim: int,
) -> tuple[list[float], list[float], list[str]]:
    """Return origin, sampling, and units in array-axis order."""
    indexed = sorted(
        (
            (int(axis.get("index_in_array", fallback)), axis)
            for fallback, axis in enumerate(axes[:ndim])
        ),
        key=lambda item: item[0],
    )
    ordered = [axis for _, axis in indexed]
    origin = [float(axis.get("offset", 0.0)) for axis in ordered]
    sampling = [float(axis.get("scale", 1.0)) for axis in ordered]
    units = [(axis.get("units") or "pixels") for axis in ordered]
    return origin, sampling, units


def _deinterleave_velox_metadata(raw: bytes) -> bytes:
    """Undo Velox's repeated-byte metadata encoding for image stacks."""
    for stride in range(2, 17):
        candidate = raw[::stride]
        if candidate.startswith(b"{"):
            return candidate
    return raw


def _read_velox_metadata_record(file_path: str | Path) -> dict:
    """Decode the first Velox image metadata record without loading pixels."""
    with h5py.File(file_path, "r") as handle:
        image_group = handle.get("Data/Image")
        if image_group is None:
            return {}
        for image_name in image_group:
            raw = bytes(image_group[image_name]["Metadata"][:]).replace(b"\x00", b"")
            for candidate in (raw, _deinterleave_velox_metadata(raw)):
                try:
                    return json.loads(candidate.decode("utf-8", errors="ignore"))
                except json.JSONDecodeError:
                    continue
    return {}


def read_emd_metadata(file_path: str | Path) -> dict:
    """Read normalized Velox acquisition metadata without loading image data.

    Parameters
    ----------
    file_path : str or Path
        Velox EMD file.

    Returns
    -------
    dict
        Normalized fields: ``scan_rotation_deg``, ``magnification``,
        ``stage_xy_m``, ``pixel_size_nm``, ``scan_shape``, ``fov_m``,
        ``acquisition_timestamp``, and ``acquisition_context``. Missing Velox
        fields are returned as ``None``. ``original_metadata`` retains the
        decoded metadata tree for specialized consumers.

    Notes
    -----
    Multi-stream EMD files can contain slightly different stage readouts on
    individual image streams. Uses the first Velox image metadata record,
    matching QuantEM Live's pairing policy.

    Examples
    --------
    >>> metadata = read_emd_metadata("scan_0.emd")
    >>> metadata["scan_rotation_deg"]
    0.0
    """
    metadata = _read_velox_metadata_record(file_path)
    with h5py.File(file_path, "r") as handle:
        acquisition_context = (
            "spectrum_image"
            if handle.get("Data/SpectrumImage") is not None
            or handle.get("Data/SpectrumStream") is not None
            else "image"
        )
    rotation = metadata.get("Scan", {}).get("ScanRotation")
    magnification = metadata.get("Optics", {}).get("NominalMagnification")
    position = metadata.get("Stage", {}).get("Position", {}) or {}
    stage_xy_m = (
        (float(position["x"]), float(position["y"]))
        if "x" in position and "y" in position
        else None
    )
    pixel_size_m = metadata.get("BinaryResult", {}).get("PixelSize", {}).get("width")
    scan_size = metadata.get("Scan", {}).get("ScanSize", {}) or {}
    scan_width = scan_size.get("width")
    scan_height = scan_size.get("height", scan_width)
    scan_shape = (
        (int(scan_height), int(scan_width))
        if scan_height and scan_width
        else None
    )
    timestamp = (
        metadata.get("Acquisition", {})
        .get("AcquisitionStartDatetime", {})
        .get("DateTime")
    )
    return {
        "path": str(file_path),
        "scan_rotation_deg": (
            None if rotation is None else math.degrees(float(rotation))
        ),
        "magnification": (
            None if magnification is None else float(magnification)
        ),
        "stage_xy_m": stage_xy_m,
        "pixel_size_nm": (
            None if pixel_size_m is None else float(pixel_size_m) * 1e9
        ),
        "scan_shape": scan_shape,
        "fov_m": (
            float(pixel_size_m) * float(scan_width)
            if pixel_size_m and scan_width
            else None
        ),
        "acquisition_timestamp": (
            int(timestamp) if timestamp and str(timestamp).isdigit() else None
        ),
        "acquisition_context": acquisition_context,
        "original_metadata": metadata,
    }


def _image_dataset(stream, path, metadata):
    from quantem.core.datastructures.dataset2d import Dataset2d

    origin, sampling, units = _axis_calibration(stream.get("axes", []), 2)
    title = stream.get("metadata", {}).get("General", {}).get("title")
    image = Dataset2d.from_array(
        np.ascontiguousarray(stream["data"], dtype=np.float32),
        name=title or Path(path).stem,
        origin=origin,
        sampling=sampling,
        units=units,
    )
    image.file_path = path
    image.metadata.update(metadata)
    return image


def _spectrum_dataset(stream, path, metadata):
    """Build a calibrated ``(scan_row, scan_col, energy)`` Dataset3d."""
    from quantem.core.datastructures.dataset3d import Dataset3d

    array = np.asarray(stream["data"])
    axes = [
        axis
        for _, axis in sorted(
            (
                (int(axis.get("index_in_array", fallback)), axis)
                for fallback, axis in enumerate(stream.get("axes", [])[:3])
            ),
            key=lambda item: item[0],
        )
    ]
    energy_axis = next(
        (
            index
            for index, axis in enumerate(axes)
            if (axis.get("units") or "").lower() in {"ev", "kev"}
            or "energy" in (axis.get("name") or "").lower()
        ),
        2,
    )
    if energy_axis != 2:
        array = np.moveaxis(array, energy_axis, 2)
        axes.append(axes.pop(energy_axis))
    origin = [float(axis.get("offset", 0.0)) for axis in axes]
    sampling = [float(axis.get("scale", 1.0)) for axis in axes]
    units = [(axis.get("units") or "pixels") for axis in axes]
    title = stream.get("metadata", {}).get("General", {}).get("title")
    spectrum = Dataset3d.from_array(
        np.ascontiguousarray(array, dtype=np.float32),
        name=title or Path(path).stem,
        origin=origin,
        sampling=sampling,
        units=units,
        signal_units="counts",
    )
    spectrum.file_path = str(path)
    spectrum.metadata.update(metadata)
    return spectrum


def read_emd(path: str | Path):
    """Read the HAADF image and acquisition geometry from a Velox EMD file.

    Drift correction needs one calibrated scan image plus its recorded scan
    direction. RosettaSciIO selects the image stream. ``read_emd_metadata``
    supplies rotation, stage position, magnification, and acquisition time.

    Parameters
    ----------
    path : str or Path
        Velox EMD acquisition.

    Returns
    -------
    Dataset2d
        Calibrated HAADF image carrying the normalized acquisition metadata.

    Examples
    --------
    >>> image = read_emd("scan_0.emd")
    >>> image.metadata["scan_rotation_deg"]
    0.0
    """
    from rsciio.emd import file_reader

    streams = file_reader(str(path), select_type="images")
    stream = next(
        (
            item
            for item in streams
            if item["data"].ndim == 2
            and item.get("metadata", {}).get("General", {}).get("title") == "HAADF"
        ),
        next((item for item in streams if item["data"].ndim == 2), None),
    )
    if stream is None:
        raise ValueError(f"{path} contains no two-dimensional image stream")

    return _image_dataset(stream, path, read_emd_metadata(path))


def read_emd_eds(
    path: str | Path,
    *,
    load_spectrum: bool = False,
    energy_windows: dict[str, tuple[float, float]] | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Read images from a Velox EDS/EELS spectrum-image EMD.

    Encapsulates the raw rsciio stream traversal so callers never write a
    ``for ds in file_reader(...)`` loop: the simultaneously-acquired HAADF
    survey and any Velox pre-quantified 2-D element maps are separated here
    and returned by name. The full spectrum is opt-in because expanding it can
    require tens of gigabytes while drift correction needs only the HAADF.
    The scan angle comes from the EMD metadata (never typed).

    Parameters
    ----------
    path : str or Path
        Velox spectrum-image EMD file.
    load_spectrum : bool, optional
        Load the full ``(row, col, energy)`` spectrum. The default ``False``
        loads only the HAADF and stored 2-D elemental maps.
    energy_windows : dict, optional
        Named ``(low_keV, high_keV)`` intervals to count directly from the
        compressed Velox SpectrumStream. This avoids expanding the full
        spectrum cube. Window endpoints are inclusive.
    verbose : bool, optional
        Print a compact summary of the loaded images.

    Returns
    -------
    dict[str, object]
        Calibrated HAADF image, optional spectrum and energy axis, stored
        element maps, and the acquisition geometry needed for correction.

    Examples
    --------
    >>> acquisition = read_emd_eds("spectrum_image.emd")
    >>> acquisition["haadf"].shape
    (2048, 2048)
    """
    from rsciio.emd import file_reader

    streams = list(
        file_reader(
            str(path),
            select_type=None if load_spectrum else "images",
        )
    )
    spectrum_stream = next(
        (stream for stream in streams if np.asarray(stream["data"]).ndim == 3),
        None,
    )
    if load_spectrum and spectrum_stream is None:
        raise ValueError(f"{path} contains no 3-D spectrum")
    haadf_stream = next(
        (
            stream
            for stream in streams
            if np.asarray(stream["data"]).ndim == 2
            and stream.get("metadata", {}).get("General", {}).get("title")
            == "HAADF"
        ),
        None,
    )
    element_maps = {
        stream.get("metadata", {}).get("General", {}).get("title", "map"):
        np.ascontiguousarray(stream["data"], dtype=np.float32)
        for stream in streams
        if np.asarray(stream["data"]).ndim == 2 and stream is not haadf_stream
    }
    metadata = read_emd_metadata(path)
    scan_rot = metadata["scan_rotation_deg"]
    if scan_rot is None:
        scan_rot = 0.0
    px_nm = metadata["pixel_size_nm"]
    px_nm = float("nan") if px_nm is None else float(px_nm)
    if haadf_stream is None:
        haadf = None
    else:
        haadf = _image_dataset(
            haadf_stream,
            path,
            metadata | {
                "scan_rotation_deg": float(scan_rot),
                "pixel_size_nm": px_nm,
            },
        )
    spectrum = (
        None
        if spectrum_stream is None
        else _spectrum_dataset(spectrum_stream, path, metadata)
    )
    window_maps: dict[str, np.ndarray] = {}
    if energy_windows is not None:
        window_maps, window_axis = _read_emd_energy_windows(path, energy_windows)
    else:
        window_axis = None
    if spectrum is None:
        energy_axis = window_axis
    else:
        scale = float(spectrum.sampling[-1])
        offset = float(spectrum.origin[-1])
        to_kev = 1e-3 if str(spectrum.units[-1]).lower() == "ev" else 1.0
        energy_axis = (
            offset + np.arange(spectrum.shape[-1], dtype=np.float64) * scale
        ) * to_kev
    image = haadf if haadf is not None else next(iter(element_maps.values()), None)
    if image is None:
        raise ValueError(f"{path} contains no 2-D HAADF or elemental maps")
    shape = tuple(image.shape)
    if verbose:
        spectrum_shape = "not loaded" if spectrum is None else str(spectrum.shape)
        print(
            f"HAADF {None if haadf is None else haadf.shape}  "
            f"elements {list(element_maps)}  spectrum {spectrum_shape}  "
            f"scan {scan_rot:.1f}°"
        )
    return {
        "haadf": haadf,
        "spectrum": spectrum,
        "cube": None if spectrum is None else spectrum.array,
        "energy_axis_keV": energy_axis,
        "element_maps": element_maps,
        "window_maps": window_maps,
        "scan_rotation_deg": scan_rot,
        "pixel_size_nm": px_nm,
        "shape": shape,
        "metadata": metadata,
        "path": str(path),
    }


def scan_pairs(
    folder: str | Path,
    *,
    max_rotation_tolerance_deg: float = 5.0,
):
    """Pair orthogonal Velox scans acquired from the same specimen area.

    Stage position identifies the shared field of view; scan rotation identifies
    the orthogonal acquisition. Shape, pixel calibration, field of view, and
    nominal magnification reject incompatible acquisitions when those metadata
    are present. Only unique mutual matches are paired. The returned inventory
    includes every file and a reason for every acquisition that is not included.

    Parameters
    ----------
    folder : str or Path
        Session folder containing Velox EMD files.
    max_rotation_tolerance_deg : float, optional
        Angular tolerance around 0 and ±90 degrees. Default is 5 degrees.

    Returns
    -------
    pandas.DataFrame
        Acquisition metadata and pair assignments. ``pair_order=0`` marks the
        scan closest to 0 degrees and ``pair_order=1`` its orthogonal partner.

    Examples
    --------
    >>> pairs = scan_pairs("~/data/session")
    >>> pairs[pairs.pair_order == 0][["file", "partner"]]
    """
    import pandas as pd

    folder = Path(folder).expanduser()
    records = []
    for path in sorted(folder.iterdir()):
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() == ".npy":
            records.append({"file": path.name, "shape": tuple(np.load(path, mmap_mode="r").shape)})
            continue
        if path.suffix.lower() != ".emd":
            continue

        metadata = read_emd_metadata(path)
        shape = metadata["scan_shape"]
        pixel_size_nm = metadata["pixel_size_nm"]
        stage = metadata["stage_xy_m"]
        records.append(
            {
                "file": path.name,
                "shape": shape,
                "pixel_size_nm": pixel_size_nm,
                "fov_nm": None if metadata["fov_m"] is None else metadata["fov_m"] * 1e9,
                "magnification": metadata["magnification"],
                "rotation_deg": metadata["scan_rotation_deg"],
                "stage_x_m": None if stage is None else stage[0],
                "stage_y_m": None if stage is None else stage[1],
                "fov_m": metadata["fov_m"],
                "acquired": metadata["acquisition_timestamp"],
                "acquisition_context": metadata.get("acquisition_context", "image"),
            }
        )

    table = pd.DataFrame(records)
    if table.empty:
        return table
    for column in (
        "pixel_size_nm",
        "fov_nm",
        "magnification",
        "rotation_deg",
        "stage_x_m",
        "stage_y_m",
        "fov_m",
        "acquired",
        "acquisition_context",
    ):
        if column not in table:
            table[column] = None

    table = table.reset_index(drop=True)
    table["pair"] = ""
    table["partner"] = ""
    table["pair_order"] = pd.array([pd.NA] * len(table), dtype="Int64")
    table["partner_rotation_deg"] = np.nan
    table["relative_partner_rotation_deg"] = np.nan
    table["stage_distance_nm"] = np.nan
    table["pair_tolerance_nm"] = np.nan
    table["pair_status"] = "not_included"
    table["pair_reason"] = ""

    tolerance = float(max_rotation_tolerance_deg)
    zero_indices = [
        index
        for index, angle in table.rotation_deg.items()
        if pd.notna(angle)
        and abs(float(angle)) < tolerance
        and table.at[index, "acquisition_context"] != "spectrum_image"
    ]
    ninety_indices = [
        index
        for index, angle in table.rotation_deg.items()
        if pd.notna(angle) and abs(abs(float(angle)) - 90.0) < tolerance
        and table.at[index, "acquisition_context"] != "spectrum_image"
    ]
    candidates_by_zero = {zero: [] for zero in zero_indices}
    zeros_by_ninety = {ninety: [] for ninety in ninety_indices}
    for zero in zero_indices:
        for ninety in ninety_indices:
            incompatible = False
            if table.at[zero, "shape"] and table.at[ninety, "shape"]:
                incompatible = tuple(table.at[zero, "shape"]) != tuple(
                    table.at[ninety, "shape"]
                )
            for column, relative_tolerance in (
                ("pixel_size_nm", 0.02),
                ("fov_m", 0.02),
                ("magnification", 0.02),
            ):
                first, second = table.loc[[zero, ninety], column]
                if pd.notna(first) and pd.notna(second):
                    scale = max(abs(float(first)), abs(float(second)), 1e-30)
                    incompatible |= abs(float(first) - float(second)) > (
                        relative_tolerance * scale
                    )
            if incompatible:
                continue
            stage_values = table.loc[
                [zero, ninety], ["stage_x_m", "stage_y_m"]
            ].to_numpy(dtype=float)
            if np.isfinite(stage_values).all():
                distance = float(np.linalg.norm(stage_values[0] - stage_values[1]))
                fovs = [
                    float(value)
                    for value in table.loc[[zero, ninety], "fov_m"]
                    if pd.notna(value) and float(value) > 0
                ]
                pair_tolerance = max(0.25 * min(fovs), 10e-9) if fovs else 10e-9
                if distance > pair_tolerance:
                    continue
            else:
                distance, pair_tolerance = float("inf"), float("nan")
            candidate = (distance, ninety, pair_tolerance)
            candidates_by_zero[zero].append(candidate)
            zeros_by_ninety[ninety].append((distance, zero, pair_tolerance))

    for index, angle in table.rotation_deg.items():
        if table.at[index, "acquisition_context"] == "spectrum_image":
            table.at[index, "pair_reason"] = (
                "Spectrum-image acquisition belongs in the EDS/EELS reference workflow."
            )
        elif pd.isna(angle):
            table.at[index, "pair_reason"] = "Missing scan-rotation metadata."
        elif index not in zero_indices and index not in ninety_indices:
            table.at[index, "pair_reason"] = (
                f"Scan rotation is not within {tolerance:g}° of 0° or ±90°."
            )

    pair_count = 0
    for zero in zero_indices:
        candidates = candidates_by_zero[zero]
        if not candidates:
            table.at[zero, "pair_reason"] = (
                "No orthogonal scan has compatible shape, calibration, field of view, and stage position."
            )
            continue
        if len(candidates) > 1:
            table.at[zero, "pair_reason"] = (
                f"Ambiguous: {len(candidates)} compatible ±90° scans match this 0° acquisition."
            )
            for _, ninety, _ in candidates:
                table.at[ninety, "pair_reason"] = (
                    "Ambiguous: this ±90° scan is one of multiple candidates for the same 0° acquisition."
                )
            continue

        distance, ninety, pair_tolerance = candidates[0]
        reverse_candidates = zeros_by_ninety[ninety]
        if len(reverse_candidates) != 1:
            table.at[zero, "pair_reason"] = (
                "Ambiguous: the compatible ±90° scan also matches multiple 0° acquisitions."
            )
            table.at[ninety, "pair_reason"] = (
                f"Ambiguous: {len(reverse_candidates)} compatible 0° scans match this ±90° acquisition."
            )
            continue

        pair_count += 1
        pair_name = f"P{pair_count:02d}"
        rotations = table.loc[[zero, ninety], "rotation_deg"].astype(float).to_numpy()
        table.loc[[zero, ninety], "pair"] = pair_name
        table.loc[[zero, ninety], "pair_status"] = "confident"
        table.loc[[zero, ninety], "pair_reason"] = ""
        table.at[zero, "partner"] = table.at[ninety, "file"]
        table.at[ninety, "partner"] = table.at[zero, "file"]
        table.at[zero, "pair_order"] = 0
        table.at[ninety, "pair_order"] = 1
        table.at[zero, "partner_rotation_deg"] = rotations[1]
        table.at[ninety, "partner_rotation_deg"] = rotations[0]
        table.at[zero, "relative_partner_rotation_deg"] = (
            rotations[1] - rotations[0] + 180.0
        ) % 360.0 - 180.0
        table.at[ninety, "relative_partner_rotation_deg"] = (
            rotations[0] - rotations[1] + 180.0
        ) % 360.0 - 180.0
        if np.isfinite(distance):
            table.loc[[zero, ninety], "stage_distance_nm"] = distance * 1e9
            table.loc[[zero, ninety], "pair_tolerance_nm"] = pair_tolerance * 1e9

    for ninety in ninety_indices:
        if not table.at[ninety, "pair"] and not table.at[ninety, "pair_reason"]:
            table.at[ninety, "pair_reason"] = (
                "No 0° scan has compatible shape, calibration, field of view, and stage position."
            )

    return table.sort_values("acquired", na_position="last").reset_index(drop=True)


def _relative_difference(first: float | None, second: float | None) -> float:
    """Return relative difference, treating missing metadata as compatible."""
    if first is None or second is None:
        return 0.0
    scale = max(abs(float(first)), abs(float(second)), 1e-30)
    return abs(float(first) - float(second)) / scale


def _same_specimen_area(first: dict, second: dict) -> bool:
    """Match scan grid, calibration, field of view, and stage position."""
    if (
        first["scan_shape"] is not None
        and second["scan_shape"] is not None
        and tuple(first["scan_shape"]) != tuple(second["scan_shape"])
    ):
        return False
    if _relative_difference(first["pixel_size_nm"], second["pixel_size_nm"]) > 0.02:
        return False
    if _relative_difference(first["fov_m"], second["fov_m"]) > 0.02:
        return False
    if _relative_difference(first["magnification"], second["magnification"]) > 0.02:
        return False
    first_stage = first["stage_xy_m"]
    second_stage = second["stage_xy_m"]
    if first_stage is None or second_stage is None:
        return False
    distance = float(np.linalg.norm(np.subtract(first_stage, second_stage)))
    fields = [
        float(value)
        for value in (first["fov_m"], second["fov_m"])
        if value is not None and float(value) > 0
    ]
    tolerance = max(0.25 * min(fields), 10e-9) if fields else 10e-9
    return distance <= tolerance


def pair_spectrum_image_references(
    folder: str | Path,
    *,
    rotation_tolerance_degrees: float = 5.0,
) -> list[dict[str, object]]:
    """Match spectrum images to orthogonal HAADF references from metadata.

    Filenames and acquisition order are never used. Each spectrum image must
    have one compatible near-zero reference and one compatible near-90-degree
    reference from the same specimen area, scan grid, and calibration. The
    assignment must also be mutual: a reference pair cannot silently match
    multiple spectrum images. Ambiguous and incomplete matches are returned
    with a reason rather than guessed.

    Parameters
    ----------
    folder : str or Path
        Folder containing Velox EMD acquisitions.
    rotation_tolerance_degrees : float, default 5.0
        Allowed deviation from 0 and 90 degrees.

    Returns
    -------
    list[dict[str, object]]
        One record per spectrum image with paths, status, and reason.

    Examples
    --------
    >>> matches = pair_spectrum_image_references("~/data/session")
    >>> ready = [match for match in matches if match["status"] == "ready"]
    """
    folder = Path(folder).expanduser()
    records = [
        read_emd_metadata(path)
        for path in sorted(folder.glob("*.emd"))
        if not path.name.startswith("._")
    ]
    references = [
        record for record in records if record["acquisition_context"] == "image"
    ]
    spectrum_images = [
        record
        for record in records
        if record["acquisition_context"] == "spectrum_image"
    ]
    tolerance = float(rotation_tolerance_degrees)
    matches = []
    for spectrum_image in spectrum_images:
        candidates = [
            reference
            for reference in references
            if _same_specimen_area(spectrum_image, reference)
        ]
        zero = [
            item
            for item in candidates
            if item["scan_rotation_deg"] is not None
            and abs(float(item["scan_rotation_deg"])) <= tolerance
        ]
        orthogonal = [
            item
            for item in candidates
            if item["scan_rotation_deg"] is not None
            and abs(abs(float(item["scan_rotation_deg"])) - 90.0) <= tolerance
        ]
        if len(zero) == 1 and len(orthogonal) == 1:
            status = "ready"
            reason = ""
        elif len(zero) > 1 or len(orthogonal) > 1:
            status = "ambiguous"
            reason = (
                f"Found {len(zero)} near-zero and {len(orthogonal)} "
                "near-90-degree compatible references."
            )
        else:
            status = "unpaired"
            reason = (
                f"Found {len(zero)} near-zero and {len(orthogonal)} "
                "near-90-degree compatible references."
            )
        matches.append(
            {
                "spectrum_image": Path(spectrum_image["path"]),
                "reference_zero": Path(zero[0]["path"]) if len(zero) == 1 else None,
                "reference_orthogonal": (
                    Path(orthogonal[0]["path"]) if len(orthogonal) == 1 else None
                ),
                "status": status,
                "reason": reason,
            }
        )
    pair_users: dict[tuple[Path, Path], list[int]] = {}
    for index, match in enumerate(matches):
        if match["status"] != "ready":
            continue
        pair = (match["reference_zero"], match["reference_orthogonal"])
        pair_users.setdefault(pair, []).append(index)
    for indices in pair_users.values():
        if len(indices) == 1:
            continue
        reason = (
            "Ambiguous: the same compatible reference pair matches "
            f"{len(indices)} spectrum images."
        )
        for index in indices:
            matches[index]["status"] = "ambiguous"
            matches[index]["reason"] = reason
            matches[index]["reference_zero"] = None
            matches[index]["reference_orthogonal"] = None
    return matches
