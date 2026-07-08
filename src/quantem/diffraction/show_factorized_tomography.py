"""Interactive 3-panel viewer for the factorized :class:`DiffractionTomography`.

Panels (left to right):

* real-space **weights** -- a z-slice of the per-voxel weight for one basis (or
  the sum over bases), or the per-voxel **orientation** rendered as RGB
  (brightness modulated by weight so vacuum voxels stay dark).
* the selected **basis** structure factor -- a slice along kz/ky/kx (or the sum)
  of ``|basis|``, with a power-law display.
* the selected basis's **SF spots** -- a rotatable 3D scatter of its local
  maxima (Bragg peaks), marker area scaled by intensity.

A single ``basis`` slider drives both the middle and right panels.
"""

from __future__ import annotations

import numpy as np
import torch


def show_factorized_tomography(dt, spot_floor: float = 0.1, power: float = 0.5):
    """Build the interactive viewer for a factorized model.

    Parameters
    ----------
    dt : DiffractionTomography
        A model whose parameters have been initialised (and usually
        reconstructed). Reads ``weights``, ``masked_basis()`` and
        ``rotation_matrices()``.
    spot_floor : float, default 0.1
        Initial local-maxima threshold for the right panel, as a fraction of the
        per-basis maximum (origin excluded).
    power : float, default 0.5
        Initial display power for the middle panel.

    Returns
    -------
    ipywidgets.Widget
        A displayable box of controls plus the linked figure.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from scipy.ndimage import maximum_filter

    W = dt.weights.detach().abs().cpu().numpy()                       # [Nz,Ny,Nx,Nw]
    B = np.fft.fftshift(dt.masked_basis().detach().abs().cpu().numpy(), axes=(0, 1, 2))
    R = dt.rotation_matrices().detach().cpu().numpy()                 # [Nz,Ny,Nx,3,3]
    Nz, Ny, Nx, Nw = W.shape
    Nkz, Nky, Nkx = B.shape[:3]
    cz, cy, cx = Nkz // 2, Nky // 2, Nkx // 2

    def maxima(vol, floor_frac):
        v = vol.copy()
        v[cz, cy, cx] = 0.0                                          # drop the origin
        vmax = float(v.max())
        if vmax <= 0:
            return np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        peaks = (v == maximum_filter(v, size=3)) & (v > floor_frac * vmax)
        zz, yy, xx = np.where(peaks)
        return xx, yy, zz, v[zz, yy, xx]

    basis = widgets.IntSlider(value=0, min=0, max=Nw - 1, step=1, description="basis",
                              disabled=(Nw == 1))
    left = widgets.Dropdown(options=(["sum"] + [f"basis {i}" for i in range(Nw)] + ["orientation"]),
                            value="sum", description="left")
    zsl = widgets.IntSlider(value=Nz // 2, min=0, max=Nz - 1, step=1, description="z (real)")
    kax = widgets.Dropdown(options=["kz", "ky", "kx"], value="kz", description="k-axis")
    kmode = widgets.Dropdown(options=["slice", "sum"], value="slice", description="mid")
    ksl = widgets.IntSlider(value=cz, min=0, max=Nkz - 1, step=1, description="k-slice")
    pw = widgets.FloatSlider(value=power, min=0.1, max=1.0, step=0.05, description="power")
    flr = widgets.FloatSlider(value=spot_floor, min=0.01, max=1.0, step=0.01, description="spot floor")
    azim = widgets.IntSlider(value=30, min=-180, max=180, step=5, description="azim")
    elev = widgets.IntSlider(value=20, min=-90, max=90, step=5, description="elev")

    def render(basis, left, zsl, kax, kmode, ksl, pw, flr, azim, elev):
        fig = plt.figure(figsize=(12, 4), constrained_layout=True)
        a1 = fig.add_subplot(1, 3, 1)
        a2 = fig.add_subplot(1, 3, 2)
        a3 = fig.add_subplot(1, 3, 3, projection="3d")

        # LEFT: weights (sum / one basis) or orientation RGB at this z-slice
        if left == "orientation":
            beam = R[zsl, :, :, :, 0]                                # rotated beam axis, [Ny,Nx,3]
            rgb = np.abs(beam) / (np.abs(beam).max() + 1e-9)
            wnorm = W[zsl].sum(-1)
            rgb = rgb * (wnorm / (wnorm.max() + 1e-12))[..., None]   # dark where vacuum
            a1.imshow(np.clip(rgb, 0, 1), origin="lower")
            a1.set_title(f"orientation (|R·ẑ| RGB)  z={zsl}", fontsize=9)
        else:
            wv = W.sum(-1) if left == "sum" else W[..., int(left.split()[1])]
            a1.imshow(wv[zsl], origin="lower", cmap="magma")
            a1.set_title(f"weights [{left}]  z={zsl}", fontsize=9)

        # MIDDLE: basis |SF| slice or sum along the chosen k axis
        b = B[..., basis]
        ax = {"kz": 0, "ky": 1, "kx": 2}[kax]
        if kmode == "sum":
            img = b.sum(axis=ax)
        else:
            idx = [slice(None)] * 3
            idx[ax] = min(ksl, b.shape[ax] - 1)
            img = b[tuple(idx)]
        a2.imshow(img ** pw, cmap="inferno")
        a2.set_title(f"|basis {basis}|  {kax} {kmode}", fontsize=9)

        # RIGHT: 3D scatter of this basis's local maxima
        xx, yy, zz, ii = maxima(B[..., basis], flr)
        if len(ii):
            s = 6 + 240 * (ii / ii.max())
            a3.scatter(xx, yy, zz, s=s, c=ii, cmap="inferno", depthshade=True)
        a3.set_xlim(0, Nkx); a3.set_ylim(0, Nky); a3.set_zlim(0, Nkz)
        a3.set_xlabel("kx"); a3.set_ylabel("ky"); a3.set_zlabel("kz")
        a3.view_init(elev=elev, azim=azim)
        a3.set_title(f"SF spots  basis {basis}  ({len(ii)} shown)", fontsize=9)
        plt.show()

    out = widgets.interactive_output(render, {
        "basis": basis, "left": left, "zsl": zsl, "kax": kax, "kmode": kmode,
        "ksl": ksl, "pw": pw, "flr": flr, "azim": azim, "elev": elev,
    })
    controls = widgets.VBox([
        widgets.HBox([basis, left, zsl]),
        widgets.HBox([kax, kmode, ksl, pw]),
        widgets.HBox([flr, azim, elev]),
    ])
    return widgets.VBox([controls, out])
