import importlib
from pathlib import Path

import h5py

from quantem.core.datastructures import Dataset as Dataset
from quantem.core.datastructures import Dataset2d as Dataset2d
from quantem.core.datastructures import Dataset3d as Dataset3d
from quantem.core.datastructures import Dataset4dstem as Dataset4dstem
from quantem.spectroscopy.dataset3dspectroscopy import (
    Dataset3dspectroscopy as Dataset3dspectroscopy,
)


def read_4dstem(
    file_path: str,
    file_type: str,
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

    Returns
    --------
    Dataset4dstem
    """
    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader  # type: ignore
    imported_data = file_reader(file_path)[0]
    dataset = Dataset4dstem.from_array(
        array=imported_data["data"],
        sampling=[
            imported_data["axes"][0]["scale"],
            imported_data["axes"][1]["scale"],
            imported_data["axes"][2]["scale"],
            imported_data["axes"][3]["scale"],
        ],
        origin=[
            imported_data["axes"][0]["offset"],
            imported_data["axes"][1]["offset"],
            imported_data["axes"][2]["offset"],
            imported_data["axes"][3]["offset"],
        ],
        units=[
            imported_data["axes"][0]["units"],
            imported_data["axes"][1]["units"],
            imported_data["axes"][2]["units"],
            imported_data["axes"][3]["units"],
        ],
    )

    return dataset


def read_3d_spectroscopy(
    file_path: str,
    file_type: str,
) -> Dataset3dspectroscopy:
    """
    File reader for 3D spectroscopy data data

    Parameters
    ----------
    file_path: str
        Path to data
    file_type: str
        The type of file reader needed. See rosettasciio for supported formats
        https://hyperspy.org/rosettasciio/supported_formats/index.html

    Returns
    --------
    Dataset3dspectroscopy
    """
    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader  # type: ignore
    imported_data = file_reader(file_path)[0]
    dataset = Dataset3dspectroscopy.from_array(
        array=imported_data["data"],
        sampling=[
            imported_data["axes"][0]["scale"],
            imported_data["axes"][1]["scale"],
            imported_data["axes"][2]["scale"],
        ],
        origin=[
            imported_data["axes"][0]["offset"],
            imported_data["axes"][1]["offset"],
            imported_data["axes"][2]["offset"],
        ],
        units=[
            imported_data["axes"][0]["units"],
            imported_data["axes"][1]["units"],
            imported_data["axes"][2]["units"],
        ],
    )

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

    return dataset


def read_emdfile_to_4dstem(file_path: str) -> Dataset4dstem:
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
        data = file["datacube_root"]["datacube"]["data"]  # type: ignore

        # Access calibration values directly
        calibration = file["datacube_root"]["metadatabundle"]["calibration"]  # type: ignore
        r_pixel_size = calibration["R_pixel_size"][()]  # type: ignore
        q_pixel_size = calibration["Q_pixel_size"][()]  # type: ignore
        r_pixel_units = calibration["R_pixel_units"][()]  # type: ignore
        q_pixel_units = calibration["Q_pixel_units"][()]  # type: ignore

        dataset = Dataset4dstem.from_array(
            array=data,
            sampling=[r_pixel_size, r_pixel_size, q_pixel_size, q_pixel_size],
            units=[r_pixel_units, r_pixel_units, q_pixel_units, q_pixel_units],
        )

    return dataset
