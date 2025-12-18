import importlib
from os import PathLike
from pathlib import Path

import h5py
import numpy as np

from quantem.core.datastructures import Dataset as Dataset
from quantem.core.datastructures import Dataset2d as Dataset2d
from quantem.core.datastructures import Dataset3d as Dataset3d
from quantem.core.datastructures import Dataset4dstem as Dataset4dstem


def read_4dstem(
    file_path: str | PathLike,
    file_type: str | None = None,
    dataset_index: int | None = None,
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
        The type of file reader needed. See RosettaSciIO for supported formats:
        https://hyperspy.org/rosettasciio/supported_formats/index.html
    dataset_index : int, optional
        Index of the dataset to load if file contains multiple datasets.
        If None, automatically selects the first 4D dataset found.
        If no 4D dataset is found but a 3D stack exists, a 3D dataset can be
        interpreted as 4D if `scan_length` is provided.
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

    **kwargs : dict
        Additional keyword arguments to pass to the Dataset4dstem constructor.

    Returns
    -------
    Dataset4dstem
    """

    def _reshape_3d_to_4d(
        imported_data: dict,
        *,
        dataset_index_local: int | None,
        scan_length_local: int,
        scan_axis_local: int,
        transpose_scan_axes_local: bool,
    ) -> dict:
        data = imported_data["data"]
        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D data to reshape, got ndim={data.ndim} "
                f"with shape {data.shape}"
            )

        if scan_axis_local not in (0, 1):
            raise ValueError(f"scan_axis must be 0 or 1, got {scan_axis_local}")

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
                "Expected 3 axes for 3D data when reshaping to 4D; "
                f"got axes={old_axes}"
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

        ax_qy = dict(old_axes[1])
        ax_qx = dict(old_axes[2])

        imported_data_4d = imported_data.copy()
        imported_data_4d["data"] = data_4d
        imported_data_4d["axes"] = [ax_scan_y, ax_scan_x, ax_qy, ax_qx]

        original_shape = imported_data["data"].shape
        new_shape = data_4d.shape
        if dataset_index_local is not None:
            print(
                f"Using 3D dataset {dataset_index_local} with shape {original_shape} "
                f"interpreted as 4D with shape={new_shape} "
                f"(scan_axis={scan_axis_local}, scan_length={scan_length_local}, "
                f"transpose_scan_axes={transpose_scan_axes_local})."
            )
        else:
            print(
                f"Using 3D dataset with shape {original_shape} "
                f"interpreted as 4D with shape={new_shape} "
                f"(scan_axis={scan_axis_local}, scan_length={scan_length_local}, "
                f"transpose_scan_axes={transpose_scan_axes_local})."
            )

        return imported_data_4d

    if file_type is None:
        file_type = Path(file_path).suffix.lower().lstrip(".")

    file_reader = importlib.import_module(f"rsciio.{file_type}").file_reader
    data_list = file_reader(file_path)

    if not data_list:
        raise ValueError(f"No datasets returned by rsciio.{file_type} for '{file_path}'")

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
                print(f"No 4D datasets found in {file_path}. Available datasets:")
                for i, d in enumerate(data_list):
                    print(f"  Dataset {i}: shape {d['data'].shape}, ndim={d['data'].ndim}")
                raise ValueError("No 4D or 3D dataset found in file")

            if scan_length is None:
                print(f"No 4D datasets found in {file_path}. Available datasets:")
                for i, d in enumerate(data_list):
                    print(f"  Dataset {i}: shape {d['data'].shape}, ndim={d['data'].ndim}")
                raise ValueError(
                    "File contains only 3D datasets. To interpret one as 4D-STEM, "
                    "please specify scan_length so that n_frames % scan_length == 0."
                )

            # Choose first 3D dataset compatible with scan_length along scan_axis
            candidates: list[tuple[int, dict]] = []
            for i, d in three_d_datasets:
                shape = d["data"].shape
                if scan_axis < 0 or scan_axis > 2:
                    raise ValueError(f"scan_axis must be in [0, 2] for 3D data, got {scan_axis}")
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
