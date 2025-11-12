import importlib
from pathlib import Path

import h5py

from quantem.core.datastructures import Dataset as Dataset
from quantem.core.datastructures import Dataset2d as Dataset2d
from quantem.core.datastructures import Dataset3d as Dataset3d
from quantem.core.datastructures import Dataset4dstem as Dataset4dstem
from quantem.spectroscopy import (
    Dataset3deds as Dataset3deds,
)
from quantem.spectroscopy import (
    Dataset3deels as Dataset3deels,
)
from quantem.spectroscopy import Dataset3dspectroscopy as Dataset3dspectroscopy


def read_4dstem(
    file_path: str,
    file_type: str,
    dataset_index: int | None = None,
    **kwargs,
) -> Dataset4dstem:
    """
    File reader for 4D-STEM data

    Parameters
    ----------
    file_path: str
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


def read_3d_spectroscopy(file_path: str, file_type: str, data_type: str, dataset_index: int | None = None) -> Dataset3dspectroscopy:
    """
    File reader for 3D spectroscopy data data

    Parameters
    ----------
    file_path: str
        Path to data
    file_type: str
        The type of file reader needed. See rosettasciio for supported formats
        https://hyperspy.org/rosettasciio/supported_formats/index.html
    data_type: str
        type of spectroscopy data 'EELS' or 'EDS'
    Returns
    --------
    Dataset3dspectroscopy
    """
    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader  # type: ignore
    data_list = file_reader(file_path)
    
    # If specific index provided, use it
    if dataset_index is not None:
        imported_data = data_list[dataset_index]
        if imported_data["data"].ndim != 3:
            raise ValueError(
                f"Dataset at index {dataset_index} has {imported_data['data'].ndim} dimensions, "
                f"expected 4D. Shape: {imported_data['data'].shape}"
            )
    else:
        # Automatically find first 3D dataset
        three_d_datasets = [(i, d) for i, d in enumerate(data_list) if d["data"].ndim == 3]

        if len(three_d_datasets) == 0:
            print(f"No 3D datasets found in {file_path}. Available datasets:")
            for i, d in enumerate(data_list):
                print(f"  Dataset {i}: shape {d['data'].shape}, ndim={d['data'].ndim}")
            raise ValueError("No 3D dataset found in file")

        dataset_index, imported_data = three_d_datasets[0]

        if len(data_list) > 1:
            print(
                f"File contains {len(data_list)} dataset(s). Using dataset {dataset_index} with shape {imported_data['data'].shape}"
            )

    imported_axes = imported_data["axes"]
    # imported_data[0], 
    if data_type == "EELS":
        dataset = Dataset3deels.from_array(
            array=imported_data["data"].transpose((2, 0, 1)),
            sampling=[
                imported_data["axes"][2]["scale"],
                imported_data["axes"][0]["scale"],
                imported_data["axes"][1]["scale"],
            ],
            origin=[
                imported_data["axes"][2]["offset"],
                imported_data["axes"][0]["offset"],
                imported_data["axes"][1]["offset"],
            ],
            units=[
                imported_data["axes"][2]["units"],
                imported_data["axes"][0]["units"],
                imported_data["axes"][1]["units"],
            ],
        )
    elif data_type == "EDS":
        dataset = Dataset3deds.from_array(
            array=imported_data["data"].transpose((2, 0, 1)),
            sampling=[
                imported_data["axes"][2]["scale"],
                imported_data["axes"][0]["scale"],
                imported_data["axes"][1]["scale"],
            ],
            origin=[
                imported_data["axes"][2]["offset"],
                imported_data["axes"][0]["offset"],
                imported_data["axes"][1]["offset"],
            ],
            units=[
                imported_data["axes"][2]["units"],
                imported_data["axes"][0]["units"],
                imported_data["axes"][1]["units"],
            ],
        )
    else:
        raise ValueError(f"`data_type` must be `EDS` or `EELS` not `{data_type}`")

    return dataset


def read_2d(
    file_path: str,
    file_type: str | None = None,
) -> Dataset2d:
    """
    File reader for images

    Parameters
    ----------
    file_path: str
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
    file_path: str, data_keys: list[str] | None = None, calibration_keys: list[str] | None = None
) -> Dataset4dstem:
    """
    File reader for legacy `emdFile` / `py4DSTEM` files.

    Parameters
    ----------
    file_path: str
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
