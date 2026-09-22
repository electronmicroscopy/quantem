"""Drift accepts encoded acquisitions without materializing full inputs."""

import os

import numpy as np
import pytest
import torch

from quantem.imaging.drift import DriftCorrection
from quantem.imaging.drift.fourdstem import corrected_4dstem_views
from tests.imaging.test_drift4d import _add_known_raw_drift


def test_ans_alignment_and_correction_match_dense(tmp_path):
    """Load ANS, integrate, correct a known field, and inspect native regions."""
    backend = os.environ.get("QEM_TEST_BACKEND")
    if backend not in {"cuda", "mps"}:
        pytest.skip("Set QEM_TEST_BACKEND to cuda or mps on that accelerator.")
    from quantem.gpu.io import FourDSTEMData, load
    from quantem.core.io import load as load_correction

    rng = np.random.default_rng(15)
    values = rng.integers(0, 200, (16, 16, 4, 6), dtype=np.uint16)
    sources = []
    for index, data in enumerate((values, np.rot90(values, -1).copy())):
        path = tmp_path / f"scan_{index}.npy"
        np.save(path, data)
        sources.append(load(path, backend=backend, verbose=False))
        sources[-1].metadata["entry/instrument/detector/count_time"] = 0.00005
    try:
        original_read = FourDSTEMData.read
        calls = []

        def bounded_read(self, *, scan_region=None, detector_region=None,
                         detector_bin=1):
            assert (scan_region is not None or detector_region is not None
                    or detector_bin > 1)
            calls.append((scan_region, detector_region))
            return original_read(self, scan_region=scan_region,
                                 detector_region=detector_region,
                                 detector_bin=detector_bin)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(FourDSTEMData, "read", bounded_read)
            encoded = DriftCorrection.from_4dstem(*sources, device=backend)
            dense = DriftCorrection.from_4dstem(
                torch.from_numpy(values).to(backend),
                torch.from_numpy(np.rot90(values, -1).copy()).to(backend),
                device=backend,
            )
            for actual, expected in zip(encoded.imgs, dense.imgs, strict=True):
                np.testing.assert_array_equal(actual.array, expected.array)
            assert encoded._datasets[0] is sources[0]
            for drift in (encoded, dense):
                drift.preprocess(num_knots=1, show_combined=False,
                                 show_scans=False, show_knots=False, verbose=False)
                _add_known_raw_drift(drift, 0, np.linspace(-0.5, 0.5, 16), 0.25)
            saved = tmp_path / "drift.zip"
            encoded.save(saved, mode="o")
            reopened = load_correction(saved)
            np.testing.assert_array_equal(reopened.imgs[0].array, encoded.imgs[0].array)
            actual = encoded.apply_correction(image_index=0, chunk_size=5,
                                             verbose=False)
            expected = dense.apply_correction(image_index=0, chunk_size=5,
                                             verbose=False)
            assert isinstance(actual, torch.Tensor)
            assert actual.device.type == backend
            np.testing.assert_allclose(actual.cpu().numpy(), expected.cpu().numpy(),
                                       rtol=0, atol=1e-5)
            regions = {"center": (8, 8)}
            actual = encoded.regional_diffraction_patterns(regions, radius_px=2)
            expected = dense.regional_diffraction_patterns(regions, radius_px=2)
            np.testing.assert_allclose(actual["patterns"], expected["patterns"],
                                       rtol=0, atol=1e-5)
            actual_views = corrected_4dstem_views(encoded, det_bin=2)
            expected_views = corrected_4dstem_views(dense, det_bin=2)
            for actual, expected in zip(actual_views, expected_views, strict=True):
                np.testing.assert_array_equal(actual, expected)
            actual = encoded.corrected_4dstem(chunk_size=5, verbose=False)
            expected = dense.corrected_4dstem(chunk_size=5, verbose=False)
            np.testing.assert_allclose(
                actual.corrected_4dstem.cpu().numpy(),
                expected.corrected_4dstem.cpu().numpy(), rtol=0, atol=1e-5,
            )
            # The correction must not close the caller-owned encoded source.
            np.testing.assert_array_equal(
                sources[0].read(scan_region=(0, 1, 0, 1)).cpu().numpy(),
                values[:1, :1],
            )
            assert calls
    finally:
        for source in sources:
            source.close()
