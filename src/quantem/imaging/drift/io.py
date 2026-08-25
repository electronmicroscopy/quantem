"""Velox EMD input and metadata pairing for drift correction."""

import json
import math
from pathlib import Path

import h5py
import numpy as np

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d


def _deinterleave_velox_metadata(raw: bytes) -> bytes:
    """Undo repeated-byte encodings found in some Velox metadata records."""
    for stride in range(2, 17):
        candidate = raw[::stride]
        if candidate.startswith(b"{"):
            return candidate
    return raw


def _read_velox_metadata_record(file_path: str | Path) -> dict:
    """Decode the first image metadata record without loading image pixels."""
    with h5py.File(file_path, "r") as handle:
        image_group = handle.get("Data/Image")
        if image_group is None:
            return {}
        for image_name in image_group:
            metadata = image_group[image_name].get("Metadata")
            if metadata is None:
                continue
            raw = bytes(metadata[:]).replace(b"\x00", b"")
            for candidate in (raw, _deinterleave_velox_metadata(raw)):
                try:
                    return json.loads(candidate.decode("utf-8", errors="ignore"))
                except json.JSONDecodeError:
                    continue
    return {}


def read_emd_metadata(file_path: str | Path) -> dict[str, object]:
    """Read normalized Velox acquisition metadata without loading pixels.

    Parameters
    ----------
    file_path : str or Path
        Velox EMD file.

    Returns
    -------
    dict[str, object]
        Scan rotation, stage position, calibration, field of view, acquisition
        time, acquisition context, and the decoded original metadata.

    Examples
    --------
    >>> metadata = read_emd_metadata("scan.emd")
    >>> metadata["scan_rotation_deg"]
    0.0
    """
    file_path = Path(file_path)
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
    stage_position = (
        (float(position["x"]), float(position["y"]))
        if "x" in position and "y" in position
        else None
    )
    pixel_size_m = metadata.get("BinaryResult", {}).get("PixelSize", {}).get("width")
    scan_size = metadata.get("Scan", {}).get("ScanSize", {}) or {}
    scan_columns = scan_size.get("width")
    scan_rows = scan_size.get("height", scan_columns)
    scan_shape = (
        (int(scan_rows), int(scan_columns))
        if scan_rows and scan_columns
        else None
    )
    timestamp = metadata.get("Acquisition", {}).get(
        "AcquisitionStartDatetime", {}
    ).get("DateTime")
    return {
        "path": str(file_path),
        "scan_rotation_deg": None if rotation is None else math.degrees(float(rotation)),
        "magnification": None if magnification is None else float(magnification),
        "stage_position_m": stage_position,
        "pixel_size_nm": None if pixel_size_m is None else float(pixel_size_m) * 1e9,
        "scan_shape": scan_shape,
        "field_of_view_m": (
            float(pixel_size_m) * float(scan_columns)
            if pixel_size_m and scan_columns
            else None
        ),
        "acquisition_timestamp": (
            int(timestamp) if timestamp and str(timestamp).isdigit() else None
        ),
        "acquisition_context": acquisition_context,
        "original_metadata": metadata,
    }


def _axis_calibration(axes: list[dict], ndim: int) -> tuple[list[float], list[float], list[str]]:
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


def _image_dataset(stream: dict, path: str | Path, metadata: dict) -> Dataset2d:
    """Build a calibrated image dataset from one RosettaSciIO stream."""
    origin, sampling, units = _axis_calibration(stream.get("axes", []), 2)
    title = stream.get("metadata", {}).get("General", {}).get("title")
    image = Dataset2d.from_array(
        np.ascontiguousarray(stream["data"], dtype=np.float32),
        name=title or Path(path).stem,
        origin=origin,
        sampling=sampling,
        units=units,
    )
    image.file_path = str(path)
    image.metadata.update(metadata)
    return image


def read_emd(path: str | Path) -> Dataset2d:
    """Read the calibrated HAADF image and recorded scan geometry.

    Parameters
    ----------
    path : str or Path
        Velox EMD acquisition.

    Returns
    -------
    Dataset2d
        HAADF image carrying normalized acquisition metadata.

    Examples
    --------
    >>> image = read_emd("scan.emd")
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
        raise ValueError(f"{path} contains no two-dimensional image stream.")
    return _image_dataset(stream, path, read_emd_metadata(path))


def _spectrum_dataset(stream: dict, path: str | Path, metadata: dict) -> Dataset3d:
    """Build a scan-row, scan-column, energy Dataset3d."""
    array = np.asarray(stream["data"])
    axes = stream.get("axes", [])
    indexed_axes = [
        axis
        for _, axis in sorted(
            (
                (int(axis.get("index_in_array", fallback)), axis)
                for fallback, axis in enumerate(axes[:3])
            ),
            key=lambda item: item[0],
        )
    ]
    energy_axis = next(
        (
            index
            for index, axis in enumerate(indexed_axes)
            if (axis.get("units") or "").lower() in {"ev", "kev"}
            or "energy" in (axis.get("name") or "").lower()
        ),
        2,
    )
    if energy_axis != 2:
        array = np.moveaxis(array, energy_axis, 2)
        axis = indexed_axes.pop(energy_axis)
        indexed_axes.append(axis)
    origin = [float(axis.get("offset", 0.0)) for axis in indexed_axes]
    sampling = [float(axis.get("scale", 1.0)) for axis in indexed_axes]
    units = [(axis.get("units") or "pixels") for axis in indexed_axes]
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


def read_emd_eds(
    path: str | Path,
    *,
    load_spectrum: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    """Read one Velox EDS or EELS spectrum-image acquisition.

    The full spectrum is opt-in because it can be large. When loaded, it is a
    calibrated :class:`Dataset3d` with ``(scan_row, scan_col, energy)`` axes.
    Energy calibration and units are retained exactly as recorded.

    Parameters
    ----------
    path : str or Path
        Velox spectrum-image EMD file.
    load_spectrum : bool, default False
        Load the full spectrum in addition to the HAADF and element maps.
    verbose : bool, default True
        Print a compact inventory.

    Returns
    -------
    dict[str, object]
        HAADF image, optional spectrum, element maps, and acquisition metadata.

    Examples
    --------
    >>> acquisition = read_emd_eds("spectrum_image.emd", load_spectrum=True)
    >>> acquisition["spectrum"].units[-1]
    'keV'
    """
    from rsciio.emd import file_reader

    metadata = read_emd_metadata(path)
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
        raise ValueError(f"{path} contains no three-dimensional spectrum.")
    haadf_stream = next(
        (
            stream
            for stream in streams
            if np.asarray(stream["data"]).ndim == 2
            and stream.get("metadata", {}).get("General", {}).get("title") == "HAADF"
        ),
        None,
    )
    element_maps = {
        stream.get("metadata", {}).get("General", {}).get("title", "map"):
        np.ascontiguousarray(stream["data"], dtype=np.float32)
        for stream in streams
        if np.asarray(stream["data"]).ndim == 2 and stream is not haadf_stream
    }
    haadf = (
        None
        if haadf_stream is None
        else _image_dataset(haadf_stream, path, metadata)
    )
    spectrum = (
        None
        if spectrum_stream is None
        else _spectrum_dataset(spectrum_stream, path, metadata)
    )
    if haadf is None and not element_maps:
        raise ValueError(f"{path} contains no HAADF image or element maps.")
    if verbose:
        print(
            f"HAADF {None if haadf is None else haadf.shape}  "
            f"elements {list(element_maps)}  "
            f"spectrum {None if spectrum is None else spectrum.shape}  "
            f"scan {metadata['scan_rotation_deg']} deg"
        )
    return {
        "haadf": haadf,
        "spectrum": spectrum,
        "element_maps": element_maps,
        "metadata": metadata,
        "path": str(path),
    }


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
    if _relative_difference(first["field_of_view_m"], second["field_of_view_m"]) > 0.02:
        return False
    if _relative_difference(first["magnification"], second["magnification"]) > 0.02:
        return False
    first_stage = first["stage_position_m"]
    second_stage = second["stage_position_m"]
    if first_stage is None or second_stage is None:
        return False
    distance = float(np.linalg.norm(np.subtract(first_stage, second_stage)))
    fields = [
        float(value)
        for value in (first["field_of_view_m"], second["field_of_view_m"])
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
