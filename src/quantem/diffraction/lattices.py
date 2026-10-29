"""
Lattice test-image generation for PADF testing.

Generates 2D "atom" images for lattices with different symmetries and
near-neighbor spacings:
    - triangular  (hexagonal Bravais lattice, 6-fold coordination)
    - square
    - honeycomb   (graphene-like, 2-atom basis; called "hexagonal" informally)
    - decagonal   (10-fold quasilattice via the de Bruijn pentagrid method)

plus kinematic diffraction patterns of random clusters cut from 3D structures
(random_clusters, cluster_patterns).

The 2D lattice positions are computed in pixels with (row, col) ordering; bond
lengths and atom widths are specified in Angstroms and converted with
pixel_size_ang. The cluster functions work in Angstroms.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

GOLDEN = (1 + np.sqrt(5)) / 2


def gaussian_form_factor(q, atom_sigma):
    """
    Scattering factor f(q) of the Gaussian atoms drawn by render_atoms:
    a real-space Gaussian of width sigma transforms to
    f(q) = exp(-2 pi^2 sigma^2 q^2).

    Divide diffraction intensities by f(q)^2 (e.g. via the PADF module's
    `fq` argument) to undo the atom-shape damping of the ring intensities;
    floor the returned values before dividing so the high-q noise floor is
    not amplified.
    """
    q = np.asarray(q, dtype=float)
    return np.exp(-2 * np.pi**2 * atom_sigma**2 * q**2)


def lattice_vectors(lattice: str, bond_length: float):
    """
    Return (lat, basis) for a periodic lattice with the requested bond length.

    lat : (2, 2) array, rows are the lattice vectors in Angstroms.
    basis : (n_atoms, 2) array of fractional coordinates in the cell.

    The triangular and square lattices have lattice parameter equal to the
    bond length. The honeycomb cell is the triangular cell scaled by sqrt(3)
    because its bond length is a / sqrt(3).
    """
    if lattice == "boron_nitride":
        # honeycomb geometry with two species; weights looked up separately
        lattice = "honeycomb"
    if lattice == "triangular":
        lat = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
        ]) * bond_length
        basis = np.array([[0.0, 0.0]])
    elif lattice == "square":
        lat = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ]) * bond_length
        basis = np.array([[0.0, 0.0]])
    elif lattice == "honeycomb":
        lat = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
        ]) * (bond_length * np.sqrt(3))
        basis = np.array([
            [1 / 3, 1 / 3],
            [2 / 3, 2 / 3],
        ])
    else:
        raise ValueError(f"Unknown lattice '{lattice}'")
    return lat, basis


def tile_lattice(lat_px, basis, im_shape, pad_px=2.0):
    """
    Tile a periodic lattice so it fully covers a rectangular image.

    Parameters
    ----------
    lat_px : (2, 2) array
        Lattice vectors in pixels (rows are vectors, (row, col) components).
    basis : (n, 2) array
        Fractional coordinates of the atoms in the cell.
    im_shape : tuple
        Image shape (rows, cols) in pixels.
    pad_px : float
        Extra margin (pixels) kept around the image so atoms straddling the
        border still render correctly.

    Returns
    -------
    points_px : (m, 2) array of atom positions in pixels.

    Notes
    -----
    The (a, b) index range needed to cover the image is found by expressing
    the four image corners in lattice coordinates: solve lat_px.T @ (a, b) =
    corner for each corner. Because the image is convex, the extreme lattice
    coordinates occur at the corners, so tiling the bounding index box and
    cropping afterwards is guaranteed to cover the image.
    """
    n_row, n_col = im_shape
    corners = np.array([
        [-pad_px, -pad_px],
        [-pad_px, n_col + pad_px],
        [n_row + pad_px, -pad_px],
        [n_row + pad_px, n_col + pad_px],
    ])
    # corners and lat_px are both in PIXELS - mixing units here breaks the
    # tiling range. lat_px.T is 2x2 and invertible, so solve exactly.
    ab = np.linalg.solve(lat_px.T, corners.T)  # (2, 4) lattice coords
    # widen by 1 for the basis offsets (fractional coords in [0, 1))
    a_lo, a_hi = int(np.floor(ab[0].min())) - 1, int(np.ceil(ab[0].max())) + 1
    b_lo, b_hi = int(np.floor(ab[1].min())) - 1, int(np.ceil(ab[1].max())) + 1

    a, b = np.meshgrid(
        np.arange(a_lo, a_hi + 1),
        np.arange(b_lo, b_hi + 1),  # b gets its own range - a and b differ
        indexing="ij",
    )
    cells = np.stack([a.ravel(), b.ravel()], axis=1).astype(float)  # (m, 2)
    # combine every cell with every basis atom
    frac = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 2)
    points_px = frac @ lat_px

    keep = (
        (points_px[:, 0] >= -pad_px)
        & (points_px[:, 0] <= n_row + pad_px)
        & (points_px[:, 1] >= -pad_px)
        & (points_px[:, 1] <= n_col + pad_px)
    )
    return points_px[keep]


def decagonal_points(radius, edge_length, gamma=None, rng=None):
    """
    Vertices of a decagonal (Penrose-like) quasilattice via the de Bruijn
    pentagrid dual method, out to the given radius (both in Angstroms).

    Five families of parallel grid lines with normals e_j at angles 2*pi*j/5
    and offsets gamma_j are intersected pairwise; each intersection maps to a
    rhombic tile of the dual tiling whose four vertices are sums of integer
    multiples of the e_j. Edges of the tiling all have length `edge_length`.
    NOTE: the SHORTEST pair distance in the vertex set is the thin-rhomb
    short diagonal, edge_length / GOLDEN (= 0.618 * edge_length) - scale
    edge_length by GOLDEN if the shortest distance should equal a target
    bond length (make_images does this).

    gamma : (5,) grid offsets. Random (generalized Penrose) by default,
        which avoids the degenerate >2-line intersections of the singular
        sum(gamma) = 0 case.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if gamma is None:
        gamma = rng.uniform(0.05, 0.95, 5)
    angles = 2 * np.pi * np.arange(5) / 5
    e = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (5, 2)

    r_lat = radius / edge_length  # work in units of the edge length
    nmax = int(np.ceil(r_lat)) + 2
    ns = np.arange(-nmax, nmax + 1)

    all_verts = []
    for j in range(5):
        for k in range(j + 1, 5):
            A = np.array([e[j], e[k]])  # (2, 2)
            nj, nk = np.meshgrid(ns, ns, indexing="ij")
            nj, nk = nj.ravel(), nk.ravel()
            rhs = np.stack([nj + gamma[j], nk + gamma[k]])  # (2, m)
            X = np.linalg.solve(A, rhs)  # (2, m) intersection points
            near = np.hypot(X[0], X[1]) <= r_lat + 1.5
            X, nj_s, nk_s = X[:, near], nj[near], nk[near]
            proj = e @ X - gamma[:, None]  # (5, m)
            K = np.ceil(proj)
            for da in (0.0, 1.0):
                for db in (0.0, 1.0):
                    Kv = K.copy()
                    # on the two defining grids the projection is exactly
                    # integer; set those indices explicitly (ceil is
                    # numerically unstable there)
                    Kv[j] = nj_s + da
                    Kv[k] = nk_s + db
                    all_verts.append(Kv.T @ e)  # (m, 2)

    verts = np.vstack(all_verts)
    verts = np.unique(np.round(verts, 6), axis=0)
    verts = verts[np.hypot(verts[:, 0], verts[:, 1]) <= r_lat]
    return verts * edge_length


# per-basis-atom scattering weights (relative); default is 1 for every atom.
# boron_nitride: honeycomb with B (Z=5) on one sublattice, N (Z=7) on the
# other - the projection loses its inversion center, so dynamical scattering
# can violate Friedel symmetry (unlike every other lattice here).
BASIS_WEIGHTS = {
    "boron_nitride": (5.0 / 7.0, 1.0),
}

SHELL_MULTS = {
    "triangular": (1.0, np.sqrt(3), 2.0),
    "square": (1.0, np.sqrt(2), 2.0),
    "honeycomb": (1.0, np.sqrt(3), 2.0),
    "boron_nitride": (1.0, np.sqrt(3), 2.0),
    # decagonal: shortest-distance pairs (thin-rhomb diagonals) are rare;
    # the strong shells are tile edges (phi d), the thick-rhomb short
    # diagonal (2 sin36 phi d = 1.902 d), and phi^2 d
    "decagonal": (GOLDEN, 2 * np.sin(np.radians(36)) * GOLDEN, GOLDEN**2),
}


def bond_angle_targets(lattice, bond_length, tol_frac=0.03):
    """
    Expected (r', theta) maxima of the 3-body (PADF) correlation when one
    arm is fixed on a 1st-shell bond: for every central atom, one arm goes
    to a 1st-shell neighbor and the other to a 1st / 2nd / 3rd shell
    neighbor; collect the shell distance r' and the angle between the arms.

        1st:  O--O      2nd:  O----O      3rd:  O------O
               |               |                 |
               O               O                 O

    Computed numerically by enumerating a patch of the actual lattice
    (shell distances from SHELL_MULTS; for the decagonal quasilattice the
    "1st shell" arm is the tile edge at GOLDEN * bond_length, its dominant
    pair distance).

    Returns
    -------
    targets : list of (r_shell, angles_deg, weights)
        One entry per shell. angles_deg are the distinct arm-arm angles in
        [0, 180] (degrees); weights are the relative multiplicities of each
        angle (normalized to the most frequent).
    """
    shells = [m * bond_length for m in SHELL_MULTS[lattice]]
    span = shells[-1] * 2.5
    if lattice == "decagonal":
        pts = decagonal_points(span + 2 * bond_length, bond_length * GOLDEN,
                               rng=np.random.default_rng(0))
    else:
        lat, basis = lattice_vectors(lattice, bond_length)
        n_cell = int(np.ceil(span / bond_length)) + 2
        a, b = np.meshgrid(np.arange(-n_cell, n_cell + 1),
                           np.arange(-n_cell, n_cell + 1), indexing="ij")
        cells = np.stack([a.ravel(), b.ravel()], axis=1).astype(float)
        frac = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 2)
        pts = frac @ lat
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= span]

    tol = tol_frac * bond_length + 1e-9
    radii = np.hypot(pts[:, 0], pts[:, 1])
    # central atoms far enough from the patch edge to see all shells
    centers = pts[radii <= span - shells[-1] - tol]

    targets = []
    diff_all = pts[None, :, :] - centers[:, None, :]  # (Nc, Np, 2)
    dist_all = np.hypot(diff_all[..., 0], diff_all[..., 1])
    arm_mask = np.abs(dist_all - shells[0]) < tol
    for r_shell in shells:
        partner_mask = np.abs(dist_all - r_shell) < tol
        angles = []
        for i in range(centers.shape[0]):
            arms = diff_all[i][arm_mask[i]]
            partners = diff_all[i][partner_mask[i]]
            if arms.size == 0 or partners.size == 0:
                continue
            cosang = (arms @ partners.T) / (
                np.hypot(arms[:, 0], arms[:, 1])[:, None]
                * np.hypot(partners[:, 0], partners[:, 1])[None, :]
            )
            ang = np.degrees(np.arccos(np.clip(cosang, -1, 1))).ravel()
            # drop the self-pair (same vector for both arms -> angle 0)
            if abs(r_shell - shells[0]) < tol:
                ang = ang[ang > 0.5]
            angles.append(ang)
        ang = np.concatenate(angles)
        # cluster to distinct angles (1 degree bins)
        vals, counts = np.unique(np.round(ang).astype(int), return_counts=True)
        keep = counts > 0.05 * counts.max()
        vals, counts = vals[keep], counts[keep]
        # merge adjacent-degree bins
        merged_vals, merged_wts = [], []
        for v, c in zip(vals, counts):
            if merged_vals and v - merged_vals[-1][-1] <= 1:
                merged_vals[-1].append(v)
                merged_wts[-1] += c
            else:
                merged_vals.append([v])
                merged_wts.append(c)
        angs = np.array([float(np.mean(g)) for g in merged_vals])
        wts = np.array(merged_wts, dtype=float)
        targets.append((float(r_shell), angs, wts / wts.max()))
    return targets


def render_atoms(points_px, im_shape, atom_sigma_px, weights=None):
    """
    Render atoms as Gaussians of width atom_sigma_px (pixels): bilinear
    deposition of one weight per atom onto a padded grid, then a Gaussian
    blur, then crop. Atoms slightly outside the image contribute their
    in-image tails, as they would in a real micrograph.

    weights : per-atom deposit weights (e.g. relative scattering strengths
        of different species); defaults to 1 for every atom.
    """
    n_row, n_col = im_shape
    p = int(np.ceil(4 * atom_sigma_px + 2))  # padding so border atoms blur in
    im = np.zeros((n_row + 2 * p, n_col + 2 * p))

    pts = points_px + p
    w = np.ones(pts.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    keep = (
        (pts[:, 0] >= 0) & (pts[:, 0] < im.shape[0] - 1)
        & (pts[:, 1] >= 0) & (pts[:, 1] < im.shape[1] - 1)
    )
    pts, w = pts[keep], w[keep]
    r0 = np.floor(pts[:, 0]).astype(int)
    c0 = np.floor(pts[:, 1]).astype(int)
    dr = pts[:, 0] - r0
    dc = pts[:, 1] - c0
    np.add.at(im, (r0, c0), w * (1 - dr) * (1 - dc))
    np.add.at(im, (r0, c0 + 1), w * (1 - dr) * dc)
    np.add.at(im, (r0 + 1, c0), w * dr * (1 - dc))
    np.add.at(im, (r0 + 1, c0 + 1), w * dr * dc)

    im = gaussian_filter(im, atom_sigma_px)
    return im[p:-p, p:-p]


def make_images(
    im_shape=(256, 256),
    pixel_size_ang=0.2,
    atom_sigma_ang=0.4,
    bond_length_ang=(2.0, 2.5, 3.0),
    lattices=("triangular", "square", "honeycomb", "decagonal"),
    rotation_deg=None,
    seed=0,
    plot_images=True,
):
    """
    Generate lattice test images for every (lattice, bond length) pair.

    Parameters
    ----------
    im_shape : tuple
        Image shape (rows, cols) in pixels.
    pixel_size_ang : float
        Pixel size in Angstroms.
    atom_sigma_ang : float
        Gaussian atom width (sigma) in Angstroms.
    bond_length_ang : sequence of float
        Near-neighbor bond lengths to generate. For every lattice type
        (including decagonal) this is the SHORTEST pair distance in the
        generated points.
    lattices : sequence of str
        Any of "triangular", "square", "honeycomb", "decagonal".
    rotation_deg : float or None
        In-plane lattice rotation. None draws a random angle per image so no
        lattice aligns with the pixel grid (avoids FFT axis artifacts).
    seed : int
        Seed for the rotation angles and quasilattice offsets.
    plot_images : bool
        Show a grid of the generated images.

    Returns
    -------
    data : list of (rows, cols) float arrays
    info : list of dicts with keys
        lattice, bond_length_ang, rotation_deg, pixel_size_ang, num_atoms,
        positions_ang (the (row, col) atom coordinates in Angstroms that
        were rendered into the image, e.g. for building an ase.Atoms
        ground truth structure)
    """
    rng = np.random.default_rng(seed)
    n_row, n_col = im_shape
    atom_sigma_px = atom_sigma_ang / pixel_size_ang
    center_px = np.array([(n_row - 1) / 2, (n_col - 1) / 2])

    data = []
    info = []
    for lattice in lattices:
        for bond in bond_length_ang:
            theta = (
                rng.uniform(0, 360) if rotation_deg is None else float(rotation_deg)
            )
            t = np.deg2rad(theta)
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

            if lattice == "decagonal":
                # Tile edges are scaled by the golden ratio so the SHORTEST
                # pair distance in the vertex set (the thin-rhomb diagonal,
                # edge / GOLDEN) equals the requested bond length. Pair
                # distances are then bond, GOLDEN * bond (tile edges), ...
                radius = 0.5 * np.hypot(n_row, n_col) * pixel_size_ang + 2 * bond
                pts_ang = decagonal_points(radius, bond * GOLDEN, rng=rng)
                pts_px = (pts_ang @ R.T) / pixel_size_ang + center_px
                pad = 2.0
                keep = (
                    (pts_px[:, 0] >= -pad) & (pts_px[:, 0] <= n_row + pad)
                    & (pts_px[:, 1] >= -pad) & (pts_px[:, 1] <= n_col + pad)
                )
                pts_px = pts_px[keep]
                weights = None
            else:
                lat, basis = lattice_vectors(lattice, bond)
                lat_px = (lat @ R.T) / pixel_size_ang
                basis_weights = BASIS_WEIGHTS.get(lattice)
                if basis_weights is None:
                    pts_px = tile_lattice(lat_px, basis, im_shape)
                    weights = None
                else:
                    # tile each basis atom separately so it keeps its weight
                    pts_list, wt_list = [], []
                    for b_frac, b_w in zip(basis, basis_weights):
                        pts_b = tile_lattice(lat_px, b_frac[None, :], im_shape)
                        pts_list.append(pts_b)
                        wt_list.append(np.full(pts_b.shape[0], b_w))
                    pts_px = np.vstack(pts_list)
                    weights = np.concatenate(wt_list)

            data.append(render_atoms(pts_px, im_shape, atom_sigma_px, weights=weights))
            info.append({
                "lattice": lattice,
                "bond_length_ang": bond,
                "rotation_deg": theta,
                "pixel_size_ang": pixel_size_ang,
                "atom_sigma_ang": atom_sigma_ang,
                "num_atoms": pts_px.shape[0],
                "basis_weights": BASIS_WEIGHTS.get(lattice),
                "positions_ang": pts_px * pixel_size_ang,
            })

    if plot_images:
        n_bond = len(bond_length_ang)
        n_lat = len(lattices)
        fig, axs = plt.subplots(
            n_lat, n_bond,
            figsize=(3 * n_bond, 3 * n_lat),
            squeeze=False,
        )
        for i, (im, meta) in enumerate(zip(data, info)):
            ax = axs[i // n_bond, i % n_bond]
            ax.imshow(im, cmap="gray")
            ax.set_title(
                f"{meta['lattice']}, d = {meta['bond_length_ang']:.2f} $\\AA$",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        plt.show()

    return data, info


def cluster_patterns(atoms, n_patterns, n_px=256, q_max=2.5, r_cluster=None,
                     voltage=200e3, seed=0, random_orientation=True, blur_px=0.0):
    """
    Kinematic diffraction patterns of spherical clusters cut from a periodic
    structure, each with a random centre and a random 3D orientation
    (random_clusters, same seed).

    Projected atoms are rendered as sampled Gaussians (sigma 1.5 px, which is
    band-limited well inside Nyquist, so crystal reflections do not alias) on
    a grid at 2x Nyquist, Fourier transformed and cropped to q_max. The
    Gaussian is divided out and replaced by the Weickenmeier-Kohl electron
    scattering factor of each element, so patterns follow f(q). blur_px is the sigma (pixels) of a
    Gaussian blur of each intensity pattern, which approximates the partial
    coherence of the illumination.

    Returns
    -------
    Dataset4dstem of shape (n_patterns, 1, n_px, n_px), calibrated in 1/A,
    with q = 0 at pixel (n_px // 2, n_px // 2).
    """
    from quantem.core.datastructures import Dataset4dstem
    from quantem.diffraction.wk_scattering_factors import compute_WK_factor

    pix = 1.0 / (4.0 * q_max)
    n_big = 2 * n_px
    dq = 1.0 / (n_big * pix)
    sigma, half = 1.5, 6  # Gaussian atom width and patch half-width (px)
    r_cluster = max_cluster_radius(atoms) if r_cluster is None else min(r_cluster, max_cluster_radius(atoms))
    if r_cluster + (half + 1) * pix > 0.5 * n_big * pix:
        raise ValueError(
            f"r_cluster = {r_cluster:.1f} A does not fit the render grid of half-width "
            f"{0.5 * n_big * pix:.1f} A = n_px / (4 q_max); increase n_px or reduce r_cluster."
        )
    qx = (torch.arange(n_px, dtype=torch.float64) - n_px // 2) * dq
    QX, QY = torch.meshgrid(qx, qx, indexing="ij")
    Q = torch.hypot(QX, QY)
    gaussian = torch.exp(-2 * np.pi**2 * (sigma * pix) ** 2 * Q**2)
    crop = slice(n_big // 2 - n_px // 2, n_big // 2 + n_px // 2)
    species = np.unique(atoms.numbers)
    filters = {
        z: torch.from_numpy(compute_WK_factor(Q.numpy().ravel(), int(z), voltage).real).reshape(Q.shape) / gaussian
        for z in species
    }

    offsets = torch.arange(-half, half + 1)
    n_pad = n_big + 2 * half + 2
    if blur_px > 0:  # real-space Gaussian kernel, positive, so no ringing from the bright central beam
        x = torch.arange(-int(np.ceil(4 * blur_px)), int(np.ceil(4 * blur_px)) + 1, dtype=torch.float64)
        kernel = torch.exp(-(x**2) / (2 * blur_px**2))
        kernel = (kernel / kernel.sum())[None, None, :]
    patterns = torch.empty((n_patterns, n_px, n_px), dtype=torch.float32)
    for k, (pos, z_atoms) in enumerate(random_clusters(atoms, n_patterns, r_cluster, seed, random_orientation)):
        xy = torch.as_tensor(pos[:, :2] / pix + n_big / 2)
        psi = 0
        for z in species:
            p = xy[torch.as_tensor(z_atoms == z)]
            i0 = torch.round(p).long()
            rows = i0[:, 0, None] + offsets[None, :]  # (Natoms, patch)
            cols = i0[:, 1, None] + offsets[None, :]
            wr = torch.exp(-((rows - p[:, 0, None]) ** 2) / (2 * sigma**2))
            wc = torch.exp(-((cols - p[:, 1, None]) ** 2) / (2 * sigma**2))
            w = (wr[:, :, None] * wc[:, None, :]).reshape(-1) / (2 * np.pi * sigma**2)
            idx = ((rows[:, :, None] + half + 1) * n_pad + cols[:, None, :] + half + 1).reshape(-1)
            im = torch.zeros(n_pad * n_pad, dtype=torch.float64).index_add_(0, idx, w)
            im = im.reshape(n_pad, n_pad)[half + 1:half + 1 + n_big, half + 1:half + 1 + n_big]
            psi = psi + torch.fft.fftshift(torch.fft.fft2(im))[crop, crop] * filters[z]
        intensity = torch.abs(psi) ** 2
        if blur_px > 0:
            pad = kernel.shape[-1] // 2
            intensity = torch.nn.functional.conv1d(intensity[:, None, :], kernel, padding=pad)[:, 0, :]
            intensity = torch.nn.functional.conv1d(intensity.T[:, None, :], kernel, padding=pad)[:, 0, :].T
        patterns[k] = intensity
    return Dataset4dstem.from_array(
        patterns.numpy()[:, None],
        origin=(0, 0, -(n_px // 2) * dq, -(n_px // 2) * dq),
        sampling=(1, 1, dq, dq),
        units=["pixels", "pixels", "A^-1", "A^-1"],
        signal_units="intensity",
    )


def max_cluster_radius(atoms):
    """Half the smallest distance between opposite faces of the periodic cell:
    the largest sphere that contains no periodic image twice."""
    cell = np.asarray(atoms.cell)
    volume = abs(np.linalg.det(cell))
    heights = [volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)]
    return 0.5 * min(heights)


def random_clusters(atoms, n_clusters, r_cluster=None, seed=0, random_orientation=True):
    """
    Yield (positions, atomic numbers) of spherical clusters cut from a periodic
    structure, each with a random centre and rotated to a random orientation
    (the beam is along z). With random_orientation=False the cell axes are
    kept, so the beam is along the cell's z axis. The same seed gives the
    clusters of cluster_patterns. r_cluster is capped at max_cluster_radius,
    so no atom appears twice; any cell shape works.
    """
    cell = torch.as_tensor(np.asarray(atoms.cell), dtype=torch.float64)  # rows are the cell vectors
    frac0 = torch.as_tensor(atoms.get_scaled_positions(wrap=True), dtype=torch.float64)
    r_max = max_cluster_radius(atoms)
    r_cluster = r_max if r_cluster is None else min(r_cluster, r_max)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(n_clusters):
        # uniform random rotation from a normalized Gaussian quaternion
        w, x, y, z = torch.nn.functional.normalize(torch.randn(4, generator=gen, dtype=torch.float64), dim=0)
        rot = torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
        ])
        # minimum image about a random centre, in fractional coordinates: within r_max of
        # the centre every fractional offset is at most 1/2, so rounding picks the right image
        frac = frac0 - torch.rand(3, generator=gen, dtype=torch.float64)
        d = (frac - torch.round(frac)) @ cell
        keep = torch.linalg.norm(d, dim=1) <= r_cluster
        pos = d[keep] @ rot.T if random_orientation else d[keep]
        yield pos.numpy(), atoms.numbers[keep.numpy()]
