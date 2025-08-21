import importlib
from pathlib import Path

import h5py

from quantem.core.datastructures import Dataset as Dataset
from quantem.core.datastructures import Dataset2d as Dataset2d
from quantem.core.datastructures import Dataset3d as Dataset3d
from quantem.core.datastructures import Dataset4dstem as Dataset4dstem


def read_4dstem(
    file_path: str,
    file_type: str | None = None,
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
    **kwargs: dict
        Additional keyword arguments to pass to the Dataset4dstem constructor.

    Returns
    --------
    Dataset4dstem
    """
    if file_type is None:
        file_type = Path(file_path).suffix.lower().lstrip(".")

    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader  # type: ignore
    imported_data = file_reader(file_path)[0]
    imported_axes = imported_data["axes"]
    sampling = kwargs.pop(
        "sampling",
        [
            imported_axes[0]["scale"],
            imported_axes[1]["scale"],
            imported_axes[2]["scale"],
            imported_axes[3]["scale"],
        ],
    )
    origin = kwargs.pop(
        "origin",
        [
            imported_axes[0]["offset"],
            imported_axes[1]["offset"],
            imported_axes[2]["offset"],
            imported_axes[3]["offset"],
        ],
    )
    units = kwargs.pop(
        "units",
        [
            "pixels" if imported_axes[0]["units"] == "1" else imported_axes[0]["units"],
            "pixels" if imported_axes[1]["units"] == "1" else imported_axes[1]["units"],
            "pixels" if imported_axes[2]["units"] == "1" else imported_axes[2]["units"],
            "pixels" if imported_axes[3]["units"] == "1" else imported_axes[3]["units"],
        ],
    )
    dataset = Dataset4dstem.from_array(
        array=imported_data["data"],
        sampling=sampling,
        origin=origin,
        units=units,
        **kwargs,
    )
    dataset.file_path = file_path

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
