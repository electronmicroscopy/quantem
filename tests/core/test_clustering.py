"""Torch DBSCAN against unambiguous synthetic ground truth."""

import numpy as np

from quantem.core.utils.clustering import cluster_vector, dbscan, filter_rows


def test_dbscan_blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10], [30, 30]], dtype=float)
    pts = np.concatenate(
        [c + rng.normal(0, 0.5, (200, 2)) for c in centers]
        + [rng.uniform(-5, 40, (40, 2))]  # sparse background
    )
    labels = dbscan(pts, eps=1.5, min_samples=10)
    # four clusters, sorted by size; blob members agree internally
    assert labels.max() + 1 == 4
    for k in range(4):
        blob = labels[200 * k : 200 * (k + 1)]
        vals, cnt = np.unique(blob[blob >= 0], return_counts=True)
        assert cnt.max() > 190  # one dominant label per blob
    assert (labels[800:] == -1).mean() > 0.8  # background mostly noise


def test_cluster_vector_and_filter():
    from quantem.core.datastructures.vector import Vector

    rng = np.random.default_rng(1)
    R, C = 4, 5
    nested = []
    for r in range(R):
        row = []
        for c in range(C):
            # one tight cluster at (1,2) plus background
            n = 30 if (r, c) == (1, 2) else 3
            q = (
                rng.normal(0, 0.01, (n, 2)) + [0.5, -0.3]
                if (r, c) == (1, 2)
                else rng.uniform(-1, 1, (n, 2))
            )
            inten = rng.uniform(1, 2, (n, 1))
            row.append(np.concatenate([q, inten], axis=1))
        nested.append(row)
    vec = Vector.from_data(
        nested, fields=["qx", "qy", "intensity"], units=["A^-1"] * 3, name="t"
    )
    labeled, labels = cluster_vector(
        vec, fields=("qx", "qy"), eps=0.05, min_samples=10
    )
    assert "cluster" in labeled.fields
    assert labels.max() == 0  # exactly one cluster found
    got = labeled[1, 2].array
    assert (got[:, -1] == 0).sum() >= 28

    kept = filter_rows(vec, labels == 0)
    assert kept.total_rows == int((labels == 0).sum())
    assert kept.shape[:2] == vec.shape[:2]
