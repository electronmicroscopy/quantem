"""Optional MAPED input preparation through QuantEM's existing readers."""

from pathlib import Path

import h5py
import numpy as np
import torch

from quantem.core.datastructures import Dataset4dstem
from quantem.core.io import read_4dstem


def load(
    file_path: str | Path,
    file_type: str | None = None,
    *,
    apply_mask: bool = False,
    **kwargs,
) -> Dataset4dstem:
    """Load one CPU acquisition and retain its file name for MAPED tilt labels.

    Uses ``read_4dstem`` without changing its behavior. Masking is opt-in;
    ordinary ``Dataset4dstem.from_file`` inputs also work directly with MAPED.
    Accelerated acquisitions can be loaded separately in the calling notebook.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Acquisition file.
    file_type : str, optional
        Reader name accepted by ``read_4dstem``, such as ``"arina"``.
    apply_mask : bool, optional
        Replace Arina detector-mask pixels with zero in every pattern. Requires
        native detector sampling. The input file is unchanged. Default False.
    **kwargs
        Options passed to ``read_4dstem``, including calibration and binning.

    Returns
    -------
    Dataset4dstem
        Native CPU dataset with source and optional correction metadata.

    Examples
    --------
    >>> from quantem.diffraction.maped import MAPED, io
    >>> tilts = [io.load(file, "arina") for file in files]
    >>> merged = MAPED.from_datasets(tilts).run(show=False)
    """
    if apply_mask and (
        (file_type or "").lower() not in {"arina", "rsciio.arina"}
        or kwargs.get("rebin_diffraction", 1) != 1
    ):
        raise ValueError(
            "apply_mask requires file_type='arina' and rebin_diffraction=1; "
            "load the native detector dimensions before applying the mask."
        )
    dataset = read_4dstem(file_path, file_type, **kwargs)
    dataset.file_path = file_path
    dataset.metadata.update(source_path=str(file_path), representation="dense")
    if apply_mask:
        with h5py.File(file_path, "r") as source:
            mask = source["entry/instrument/detector/detectorSpecific/pixel_mask"][()] != 0
        if mask.shape != dataset.shape[-2:]:
            raise ValueError(
                f"Detector mask shape {mask.shape} differs from {dataset.shape[-2:]}; "
                "load the native detector dimensions before applying the mask."
            )
        dataset.array[..., mask] = 0
        dataset.metadata["hot_pixel_correction"] = {
            "method": "zero",
            "applied": True,
            "pixel_count": int(mask.sum()),
            "coordinates_row_column": np.argwhere(mask).tolist(),
        }
    return dataset


def summaries(sources, device: torch.device):
    """Compute mean diffraction patterns and scan images in bounded memory."""
    patterns, images = [], []
    for source in sources:
        if not isinstance(source, Dataset4dstem) and device.type != "cpu":
            # The supplied dataset owns cached reductions and its representation.
            patterns.append(source.mean(dim=(0, 1)))
            images.append(source.mean(dim=(2, 3)))
            continue
        rows, columns, detector_rows, detector_columns = source.shape
        pattern_t = torch.zeros((detector_rows, detector_columns), dtype=torch.float64)
        image_t = torch.empty((rows, columns), dtype=torch.float32)
        for row in range(0, rows, 16):
            stop = min(row + 16, rows)
            counts_t = read_region(source, (row, stop, 0, columns)).to(torch.float64)
            pattern_t += counts_t.sum(dim=(0, 1))
            image_t[row:stop] = counts_t.mean(dim=(2, 3))
        patterns.append((pattern_t / (rows * columns)).to(torch.float32))
        images.append(image_t)
    return patterns, images


def read_region(source, region: tuple[int, int, int, int]) -> torch.Tensor:
    """Read native CPU counts or decode an acquisition on its resident device."""
    if isinstance(source, Dataset4dstem):
        row0, row1, col0, col1 = region
        values = source.array if source.array is not None else source.tensor
        return torch.as_tensor(values[row0:row1, col0:col1])
    return source.read(scan_region=region)
