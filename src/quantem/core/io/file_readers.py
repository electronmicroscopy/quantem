import importlib
from os import PathLike
from pathlib import Path
from typing import Any

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
        Additional keyword arguments to pass to the file reader.

    Other Parameters
    ----------------
    name : str | None, optional
        A descriptive name for the dataset. If None, defaults to "4D-STEM dataset"
    origin : NDArray | tuple | list | float | int | None, optional
        The origin coordinates for each dimension. If None, defaults to zeros
    sampling : NDArray | tuple | list | float | int | None, optional
        The sampling rate/spacing for each dimension. If None, defaults to ones
    units : list[str] | tuple | list | None, optional
        Units for each dimension. If None, defaults to ["pixels"] * 4
    signal_units : str, optional
        Units for the array values, by default "arb. units"

    Returns
    --------
    Dataset4dstem
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

    dataset = Dataset4dstem.from_array(
        array=imported_data["data"],
        sampling=sampling,
        origin=origin,
        units=units,
        name=name_override,
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
