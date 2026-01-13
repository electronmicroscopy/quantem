import importlib
from os import PathLike
from pathlib import Path

import h5py

from quantem.core.datastructures import Dataset as Dataset
from quantem.core.datastructures import Dataset2d as Dataset2d
from quantem.core.datastructures import Dataset3d as Dataset3d
from quantem.core.datastructures import Dataset4dstem as Dataset4dstem


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


def _find_4d_dataset(group: h5py.Group, path: list[str] | None = None) -> tuple[list[str], h5py.Dataset] | None:
    """Recursively search for a 4D dataset in an HDF5 group."""
    if path is None:
        path = []
    for key in group.keys():
        item = group[key]
        current_path = path + [key]
        if isinstance(item, h5py.Dataset):
            if item.ndim == 4:
                return current_path, item
        elif isinstance(item, h5py.Group):
            result = _find_4d_dataset(item, current_path)
            if result is not None:
                return result
    return None


def _find_calibration(group: h5py.Group, path: list[str] | None = None) -> tuple[list[str], h5py.Group] | None:
    """Recursively search for a calibration group containing R_pixel_size and Q_pixel_size."""
    if path is None:
        path = []
    for key in group.keys():
        item = group[key]
        current_path = path + [key]
        if isinstance(item, h5py.Group):
            # Check if this group has calibration keys
            if "R_pixel_size" in item and "Q_pixel_size" in item:
                return current_path, item
            # Recurse into subgroups
            result = _find_calibration(item, current_path)
            if result is not None:
                return result
    return None


def read_emdfile_to_4dstem(
    file_path: str | PathLike,
    data_keys: list[str] | None = None,
    calibration_keys: list[str] | None = None,
) -> Dataset4dstem:
    """
    File reader for legacy `emdFile` / `py4DSTEM` files.

    If data_keys and calibration_keys are not provided, the function will
    automatically search for a 4D dataset and calibration metadata.

    Parameters
    ----------
    file_path: str | PathLike
        Path to data
    data_keys: list[str], optional
        List of keys to navigate to the data. If None, auto-detects.
    calibration_keys: list[str], optional
        List of keys to navigate to calibration. If None, auto-detects.

    Returns
    --------
    Dataset4dstem
    """
    with h5py.File(file_path, "r") as file:
        # Auto-detect or use provided data keys
        if data_keys is None:
            result = _find_4d_dataset(file)
            if result is None:
                raise KeyError(f"Could not find any 4D dataset in {file_path}")
            data_keys, data = result
        else:
            try:
                data = file
                for key in data_keys:
                    data = data[key]  # type: ignore
            except KeyError:
                raise KeyError(f"Could not find key {data_keys} in {file_path}")

        # Auto-detect or use provided calibration keys
        if calibration_keys is None:
            result = _find_calibration(file)
            if result is None:
                # No calibration found, use defaults
                r_pixel_size = 1.0
                q_pixel_size = 1.0
                r_pixel_units = "pixels"
                q_pixel_units = "pixels"
            else:
                calibration_keys, calibration = result
                r_pixel_size = calibration["R_pixel_size"][()]  # type: ignore
                q_pixel_size = calibration["Q_pixel_size"][()]  # type: ignore
                r_pixel_units = calibration.get("R_pixel_units", [()])
                if hasattr(r_pixel_units, "__getitem__"):
                    r_pixel_units = r_pixel_units[()]
                q_pixel_units = calibration.get("Q_pixel_units", [()])
                if hasattr(q_pixel_units, "__getitem__"):
                    q_pixel_units = q_pixel_units[()]
        else:
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

        # Decode bytes to string if needed
        if isinstance(r_pixel_units, bytes):
            r_pixel_units = r_pixel_units.decode("utf-8")
        if isinstance(q_pixel_units, bytes):
            q_pixel_units = q_pixel_units.decode("utf-8")

        dataset = Dataset4dstem.from_array(
            array=data,
            sampling=[r_pixel_size, r_pixel_size, q_pixel_size, q_pixel_size],
            units=[r_pixel_units, r_pixel_units, q_pixel_units, q_pixel_units],
        )
    dataset.file_path = file_path

    return dataset
