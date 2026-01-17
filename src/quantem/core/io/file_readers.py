import importlib
import json
from os import PathLike
from pathlib import Path

import h5py
import numpy as np

from quantem.core.datastructures import Dataset as Dataset
from quantem.core.datastructures import Dataset2d as Dataset2d
from quantem.core.datastructures import Dataset3d as Dataset3d
from quantem.core.datastructures import Dataset4dstem as Dataset4dstem
from quantem.core.datastructures import Dataset5dstem as Dataset5dstem


def read_4dstem(
    file_path: str | PathLike,
    file_type: str | None = None,
    dataset_index: int | None = None,
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
    **kwargs: dict
        Additional keyword arguments to pass to the Dataset4dstem constructor.

    Returns
    --------
    Dataset4dstem
    """
    if file_type is None:
        file_type = Path(file_path).suffix.lower().lstrip(".")

    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader
    data_list = file_reader(file_path)

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

        if len(four_d_datasets) == 0:
            print(f"No 4D datasets found in {file_path}. Available datasets:")
            for i, d in enumerate(data_list):
                print(f"  Dataset {i}: shape {d['data'].shape}, ndim={d['data'].ndim}")
            raise ValueError("No 4D dataset found in file")

        dataset_index, imported_data = four_d_datasets[0]

        if len(data_list) > 1:
            print(
                f"File contains {len(data_list)} dataset(s). Using dataset {dataset_index} with shape {imported_data['data'].shape}"
            )

    imported_axes = imported_data["axes"]

    sampling = kwargs.pop(
        "sampling",
        [ax["scale"] for ax in imported_axes],
    )
    origin = kwargs.pop(
        "origin",
        [ax["offset"] for ax in imported_axes],
    )
    units = kwargs.pop(
        "units",
        ["pixels" if ax["units"] == "1" else ax["units"] for ax in imported_axes],
    )

    dataset = Dataset4dstem.from_array(
        array=imported_data["data"],
        sampling=sampling,
        origin=origin,
        units=units,
        **kwargs,
    )

    return dataset


def read_5dstem(
    file_path: str | PathLike,
    file_type: str | None = None,
    stack_type: str = "auto",
    **kwargs,
) -> Dataset5dstem:
    """
    File reader for 5D-STEM data.

    Supports:
    - Nion Swift h5 files (auto-detected from 'properties' attribute)
    - rosettasciio formats with 5D data

    Parameters
    ----------
    file_path : str | PathLike
        Path to data
    file_type : str | None, optional
        The type of file reader needed. If None, auto-detect.
    stack_type : str, optional
        Stack type ("sequence", "tilt", etc.) or "auto" to detect from metadata.
    **kwargs : dict
        Additional keyword arguments to pass to Dataset5dstem constructor.

    Returns
    -------
    Dataset5dstem
    """
    file_path = Path(file_path)

    # Try Nion Swift h5 format first
    if file_path.suffix.lower() in [".h5", ".hdf5"]:
        try:
            with h5py.File(file_path, "r") as f:
                if "data" in f and "properties" in f["data"].attrs:
                    # Nion Swift format detected
                    return _read_nion_swift_5dstem(file_path, stack_type, **kwargs)
        except Exception:
            pass  # Fall through to rsciio

    # Fall back to rosettasciio
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip(".")

    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader
    data_list = file_reader(file_path)

    # Find first 5D dataset
    five_d_datasets = [(i, d) for i, d in enumerate(data_list) if d["data"].ndim == 5]

    if len(five_d_datasets) == 0:
        print(f"No 5D datasets found in {file_path}. Available datasets:")
        for i, d in enumerate(data_list):
            print(f"  Dataset {i}: shape {d['data'].shape}, ndim={d['data'].ndim}")
        raise ValueError("No 5D dataset found in file")

    dataset_index, imported_data = five_d_datasets[0]

    if len(data_list) > 1:
        print(
            f"File contains {len(data_list)} dataset(s). Using dataset {dataset_index} "
            f"with shape {imported_data['data'].shape}"
        )

    imported_axes = imported_data["axes"]

    sampling = kwargs.pop(
        "sampling",
        [ax["scale"] for ax in imported_axes],
    )
    origin = kwargs.pop(
        "origin",
        [ax["offset"] for ax in imported_axes],
    )
    units = kwargs.pop(
        "units",
        ["pixels" if ax["units"] == "1" else ax["units"] for ax in imported_axes],
    )

    # Determine stack type
    if stack_type == "auto":
        stack_type = "generic"

    dataset = Dataset5dstem.from_array(
        array=imported_data["data"],
        sampling=sampling,
        origin=origin,
        units=units,
        stack_type=stack_type,
        **kwargs,
    )

    return dataset


def _read_nion_swift_5dstem(
    file_path: str | PathLike,
    stack_type: str = "auto",
    **kwargs,
) -> Dataset5dstem:
    """
    Read Nion Swift 5D-STEM h5 file.

    Nion Swift stores data with:
    - f['data'] containing the array
    - f['data'].attrs['properties'] containing JSON metadata

    Parameters
    ----------
    file_path : str | PathLike
        Path to Nion Swift h5 file
    stack_type : str, optional
        Stack type or "auto" to detect from metadata

    Returns
    -------
    Dataset5dstem
    """
    with h5py.File(file_path, "r") as f:
        data = f["data"][:]
        props = json.loads(f["data"].attrs["properties"])

    if data.ndim != 5:
        raise ValueError(f"Expected 5D data, got {data.ndim}D with shape {data.shape}")

    # Extract calibrations
    cals = props.get("dimensional_calibrations", [])
    if len(cals) == 5:
        origin = np.array([c.get("offset", 0.0) for c in cals])
        sampling = np.array([c.get("scale", 1.0) for c in cals])
        units = [c.get("units", "") or "pixels" for c in cals]
    else:
        origin = np.zeros(5)
        sampling = np.ones(5)
        units = ["pixels"] * 5

    # Determine stack type from metadata
    if stack_type == "auto":
        if props.get("is_sequence", False):
            stack_type = "time"
        else:
            stack_type = "generic"

    # Get intensity calibration
    intensity_cal = props.get("intensity_calibration", {})
    signal_units = intensity_cal.get("units", "arb. units") or "arb. units"

    dataset = Dataset5dstem.from_array(
        array=data,
        origin=origin,
        sampling=sampling,
        units=units,
        signal_units=signal_units,
        stack_type=stack_type,
        **kwargs,
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
    if file_type is None:
        file_type = Path(file_path).suffix.lower().lstrip(".")

    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader  # type: ignore
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
            data = file
            for key in data_keys:
                data = data[key]  # type: ignore
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
                calibration = calibration[key]  # type: ignore
        except KeyError:
            raise KeyError(f"Could not find calibration key {calibration_keys} in {file_path}")
        r_pixel_size = calibration["R_pixel_size"][()]  # type: ignore
        q_pixel_size = calibration["Q_pixel_size"][()]  # type: ignore
        r_pixel_units = calibration["R_pixel_units"][()]  # type: ignore
        q_pixel_units = calibration["Q_pixel_units"][()]  # type: ignore

        dataset = Dataset4dstem.from_array(
            array=data,
            sampling=[r_pixel_size, r_pixel_size, q_pixel_size, q_pixel_size],
            units=[r_pixel_units, r_pixel_units, q_pixel_units, q_pixel_units],
        )
    dataset.file_path = file_path

    return dataset
