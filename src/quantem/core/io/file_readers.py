import importlib
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from quantem.core.datastructures import Dataset as Dataset
from quantem.core.datastructures import Dataset2d as Dataset2d
from quantem.core.datastructures import Dataset3d as Dataset3d
from quantem.core.datastructures import Dataset4dstem as Dataset4dstem
from quantem.spectroscopy import (
    Dataset3deels as Dataset3deels,
)
from quantem.spectroscopy import Dataset3dspectroscopy as Dataset3dspectroscopy
from quantem.spectroscopy import (
    Dataset3dxeds as Dataset3dxeds,
)


def _resolve_rsciio_plugin(file_path: str | PathLike, file_type: str | None = None) -> str:
    """
    Resolve the RosettaSciIO plugin module used to read a file.

    Parameters
    ----------
    file_path : str | PathLike
        Path to the file. Its extension is used when ``file_type`` is None.
    file_type : str, optional
        RosettaSciIO plugin name (e.g. "digitalmicrograph", "quantumdetector")
        or a file extension (e.g. "dm4", "mib"). Case-insensitive.

    Returns
    -------
    str
        Module name of the plugin, e.g. "rsciio.digitalmicrograph".

    Raises
    ------
    ValueError
        If no plugin matches, or if an extension is listed by more than one
        plugin (e.g. ".h5"); pass the plugin name as ``file_type`` in that case.
    """
    import rsciio

    key = file_type if file_type is not None else Path(file_path).suffix.lstrip(".")
    key = str(key).lower().lstrip(".")
    if not key:
        raise ValueError(
            f"Cannot infer the file type of '{file_path}'; pass file_type= "
            "(a RosettaSciIO plugin name such as 'digitalmicrograph')."
        )

    plugins = rsciio.IO_PLUGINS
    by_name = sorted({p["api"] for p in plugins if p["api"].lower() == f"rsciio.{key}"})
    if by_name:
        return by_name[0]

    by_ext = sorted(
        {p["api"] for p in plugins if key in (ext.lower() for ext in p["file_extensions"])}
    )
    if len(by_ext) == 1:
        return by_ext[0]
    if len(by_ext) > 1:
        names = ", ".join(f"'{api.removeprefix('rsciio.')}'" for api in by_ext)
        raise ValueError(
            f"File extension '{key}' is used by several RosettaSciIO plugins ({names}). "
            "Pass one of them as file_type=."
        )
    raise ValueError(f"No RosettaSciIO reader for file type '{key}'.")


def _rsciio_reader(file_path: str | PathLike, file_type: str | None = None):
    """
    Return ``(plugin, file_reader)`` for a file, see `_resolve_rsciio_plugin`.

    An ImportError raised here means the plugin exists but one of its optional
    dependencies is missing.
    """
    plugin = _resolve_rsciio_plugin(file_path, file_type)
    return plugin, importlib.import_module(plugin).file_reader


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
    scan_length: int | None = None,
    scan_axis: int = 0,
    transpose_scan_axes: bool = False,
    **kwargs,
) -> Dataset4dstem:
    """
    File reader for 4D-STEM data.

    Parameters
    ----------
    file_path : str | PathLike
        Path to data.
    file_type : str, optional
        RosettaSciIO plugin name (e.g. "arina", "digitalmicrograph") or file
        extension. If None, the extension of `file_path` is used. Extensions
        shared by several plugins (e.g. "h5") require the plugin name. See
        https://hyperspy.org/rosettasciio/supported_formats/index.html
    dataset_index : int, optional
        Index of the dataset to load if file contains multiple datasets.
        If None, automatically selects the first 4D dataset found.
        If no 4D dataset is found but a 3D stack exists, a 3D dataset can be
        interpreted as 4D if `scan_length` is provided.
    hot_pixel_filter : bool, default False
        If True, detect and replace hot detector pixels immediately after
        loading using `quantem.core.utils.filter.filter_hot_pixels` with its
        default parameters. For custom thresholds, call `filter_hot_pixels`
        directly on the array.
    scan_length : int, optional
        For 3D datasets shaped (n_frames, ny, nx) (after possibly moving the
        scan axis to the front), interpret the data as a raster scan with shape
        (scan_y, scan_x, ny, nx), where scan_y = n_frames // scan_length and
        scan_x = scan_length. Required if you want to treat a 3D stack as 4D.
    scan_axis : int, default 0
        Which axis of a 3D dataset is the scan/time axis before reshaping.
        Must be 0 or 1. The specified axis is moved to axis 0 before the
        (scan_y, scan_x) reshape.
    transpose_scan_axes : bool, default False
        Only used when interpreting a 3D dataset as 4D via `scan_length`.
        If True, transpose the scan axes after reshaping so that
        (scan_y, scan_x) -> (scan_x, scan_y). This effectively swaps the
        interpretation of scan rows and columns in the final 4D array.
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
    -------
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

    def _reshape_3d_to_4d(
        imported_data: dict,
        *,
        dataset_index_local: int,
        scan_length_local: int,
        scan_axis_local: int,
        transpose_scan_axes_local: bool,
    ) -> dict:
        data = imported_data["data"]
        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D data to reshape, got ndim={data.ndim} with shape {data.shape}"
            )

        # Move scan axis to front so it becomes the frame axis
        if scan_axis_local != 0:
            data = np.moveaxis(data, scan_axis_local, 0)

        n_frames, ny, nx = data.shape

        if scan_length_local <= 0:
            raise ValueError(f"scan_length must be positive, got {scan_length_local}")
        if n_frames % scan_length_local != 0:
            raise ValueError(
                f"scan_length={scan_length_local} is not compatible with n_frames={n_frames}; "
                f"n_frames % scan_length = {n_frames % scan_length_local}"
            )

        scan_y = n_frames // scan_length_local
        scan_x = scan_length_local

        data_4d = data.reshape(scan_y, scan_x, ny, nx)

        if transpose_scan_axes_local:
            data_4d = np.transpose(data_4d, (1, 0, 2, 3))
            scan_y, scan_x = scan_x, scan_y

        old_axes = imported_data.get("axes", None)
        if old_axes is None or len(old_axes) != 3:
            raise ValueError(
                f"Expected 3 axes for 3D data when reshaping to 4D; got axes={old_axes}"
            )

        ax_scan_y = {
            "scale": 1.0,
            "offset": 0.0,
            "units": "pixels",
            "name": "scan_y",
        }
        ax_scan_x = {
            "scale": 1.0,
            "offset": 0.0,
            "units": "pixels",
            "name": "scan_x",
        }

        # Detector calibrations come from the two axes that are not the scan axis.
        ax_qy, ax_qx = (dict(ax) for i, ax in enumerate(old_axes) if i != scan_axis_local)

        imported_data_4d = imported_data.copy()
        imported_data_4d["data"] = data_4d
        imported_data_4d["axes"] = [ax_scan_y, ax_scan_x, ax_qy, ax_qx]

        original_shape = imported_data["data"].shape
        new_shape = data_4d.shape
        print(
            f"Using 3D dataset {dataset_index_local} with shape {original_shape} "
            f"interpreted as 4D with shape={new_shape} "
            f"(scan_axis={scan_axis_local}, scan_length={scan_length_local}, "
            f"transpose_scan_axes={transpose_scan_axes_local})."
        )

        return imported_data_4d

    if scan_axis not in (0, 1):
        raise ValueError(f"scan_axis must be 0 or 1, got {scan_axis}")

    sampling_override = kwargs.pop("sampling", None)
    origin_override = kwargs.pop("origin", None)
    units_override = kwargs.pop("units", None)
    name_override = kwargs.pop("name", None)

    plugin, file_reader = _rsciio_reader(file_path, file_type)
    data_list = file_reader(file_path, **kwargs)

    if not data_list:
        raise ValueError(f"No datasets returned by {plugin} for '{file_path}'")

    # Case 1: dataset_index specified explicitly
    if dataset_index is not None:
        imported_data = data_list[dataset_index]
        ndim = imported_data["data"].ndim

        if ndim == 4:
            # Use 4D as-is
            pass
        elif ndim == 3:
            if scan_length is None:
                raise ValueError(
                    f"Dataset at index {dataset_index} is 3D (shape={imported_data['data'].shape}). "
                    "To interpret it as 4D-STEM, please provide scan_length."
                )
            imported_data = _reshape_3d_to_4d(
                imported_data,
                dataset_index_local=dataset_index,
                scan_length_local=scan_length,
                scan_axis_local=scan_axis,
                transpose_scan_axes_local=transpose_scan_axes,
            )
        else:
            raise ValueError(
                f"Dataset at index {dataset_index} has ndim={ndim}, "
                f"expected 4D or 3D. Shape: {imported_data['data'].shape}"
            )

    else:
        # Case 2: auto-select dataset
        four_d_datasets = [(i, d) for i, d in enumerate(data_list) if d["data"].ndim == 4]
        _print_available_datasets(data_list)

        if four_d_datasets:
            dataset_index, imported_data = four_d_datasets[0]
            if len(data_list) > 1:
                print(
                    f"File contains {len(data_list)} dataset(s). Using 4D dataset "
                    f"{dataset_index} with shape {imported_data['data'].shape}"
                )
        else:
            three_d_datasets = [(i, d) for i, d in enumerate(data_list) if d["data"].ndim == 3]

            if not three_d_datasets:
                print(f"No 4D datasets found in {file_path}.")
                raise ValueError("No 4D or 3D dataset found in file")

            if scan_length is None:
                print(f"No 4D datasets found in {file_path}.")
                raise ValueError(
                    "File contains only 3D datasets. To interpret one as 4D-STEM, "
                    "please specify scan_length so that n_frames % scan_length == 0."
                )

            # Choose first 3D dataset compatible with scan_length along scan_axis
            candidates: list[tuple[int, dict]] = []
            for i, d in three_d_datasets:
                shape = d["data"].shape
                n_frames_axis = shape[scan_axis]
                if n_frames_axis % scan_length == 0:
                    candidates.append((i, d))

            if not candidates:
                print(f"3D datasets in {file_path}:")
                for i, d in three_d_datasets:
                    print(f"  Dataset {i}: shape {d['data'].shape}")
                raise ValueError(
                    f"No 3D dataset has length along scan_axis={scan_axis} "
                    f"divisible by scan_length={scan_length}."
                )

            dataset_index, imported_data = candidates[0]
            if len(candidates) > 1:
                print(
                    f"Multiple 3D datasets compatible with scan_length={scan_length} "
                    f"along scan_axis={scan_axis}. Using dataset {dataset_index} "
                    f"with shape {imported_data['data'].shape}"
                )

            imported_data = _reshape_3d_to_4d(
                imported_data,
                dataset_index_local=dataset_index,
                scan_length_local=scan_length,
                scan_axis_local=scan_axis,
                transpose_scan_axes_local=transpose_scan_axes,
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

    plugin, file_reader = _rsciio_reader(file_path, file_type)
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
    # DigitalMicrograph spectrum images are reordered so that axis 0 moves last.
    is_dm = plugin == "rsciio.digitalmicrograph"
    axis_order = (1, 2, 0) if is_dm else (0, 1, 2)
    array = imported_data["data"].transpose(axis_order) if is_dm else imported_data["data"]
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
    _, file_reader = _rsciio_reader(file_path, file_type)
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
