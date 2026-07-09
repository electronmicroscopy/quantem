"""Interactive 3-panel viewer for :class:`DiffractionTomography`.

Panels (left to right):

* real-space **weights** -- a z-slice of the per-voxel weight for one structure
  (or the sum), or the per-voxel **orientation** rendered as RGB (brightness
  modulated by weight so vacuum voxels stay dark).
* the selected **structure factor** -- a slice along kz/ky/kx (or the sum) of
  ``|SF|``, with power-law display and optional ``|k|`` / ``|k|**2`` radial
  weighting to suppress the low-frequency envelope near the origin.
* the structure factor's **Bragg spots** -- a rotatable 3D scatter of its local
  maxima, marker area scaled by intensity.

A single ``structure`` slider drives the middle and right panels.

Passing the first viewer's return value as ``controls=`` to a second call
reuses the same control widgets, so the two viewers (e.g. ground truth and
reconstruction) stay in lockstep.

Objects from the deprecated explicit-6D model are dispatched to the legacy
anywidget viewer (``show_diffraction_tomography_6d``).
"""

from __future__ import annotations

import numpy as np
import torch


def show_diffraction_tomography(dt, controls=None, title: str = "",
                                spot_floor: float = 0.1, power: float = 0.5):
    """Build the interactive viewer for a diffraction tomography model.

    Parameters
    ----------
    dt : DiffractionTomography
        A model whose parameters have been initialised (and usually
        reconstructed). Reads ``weights``, ``masked_basis()`` and
        ``rotation_matrices()``. Objects from the deprecated explicit-6D
        model are forwarded to the legacy viewer.
    controls : ipywidgets.Widget, optional
        The return value of a previous call. The same control widgets are
        reused, so the two viewers stay linked (shared structure index,
        slices, view settings).
    title : str, optional
        Heading shown above the panels.
    spot_floor : float, default 0.1
        Initial local-maxima threshold for the right panel, as a fraction of
        the per-structure maximum (origin excluded).
    power : float, default 0.5
        Initial display power for the middle panel.

    Returns
    -------
    ipywidgets.Widget
        A displayable box of controls plus the linked figure; pass it as
        ``controls=`` to a second call to link viewers.
    """
    # legacy explicit-6D objects -> the old anywidget viewer
    if not hasattr(dt, "masked_basis"):
        from quantem.diffraction.show_diffraction_tomography_6d import (
            show_diffraction_tomography as _show_6d,
        )
        return _show_6d(dt, controls=controls, title=title)

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from scipy.ndimage import maximum_filter

    W = dt.weights.detach().abs().cpu().numpy()                       # [Nz,Ny,Nx,Ns]
    B = np.fft.fftshift(dt.masked_basis().detach().abs().cpu().numpy(), axes=(0, 1, 2))
    R = dt.rotation_matrices().detach().cpu().numpy()                 # [Nz,Ny,Nx,3,3]
    Nz, Ny, Nx, Ns = W.shape
    Nkz, Nky, Nkx = B.shape[:3]
    cz, cy, cx = Nkz // 2, Nky // 2, Nkx // 2
    # fftshifted radial |k| (pixels) for the optional radial weighting
    ZZ, YY, XX = np.meshgrid(np.arange(Nkz) - cz, np.arange(Nky) - cy,
                             np.arange(Nkx) - cx, indexing="ij")
    krad3 = np.sqrt(ZZ ** 2 + YY ** 2 + XX ** 2)

    def maxima(vol, floor_frac):
        v = vol.copy()
        v[cz, cy, cx] = 0.0                                          # drop the origin
        vmax = float(v.max())
        if vmax <= 0:
            return np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        peaks = (v == maximum_filter(v, size=3)) & (v > floor_frac * vmax)
        zz, yy, xx = np.where(peaks)
        return xx, yy, zz, v[zz, yy, xx]

    if controls is not None and hasattr(controls, "ctrl"):
        c = controls.ctrl                                            # shared widgets
    else:
        c = dict(
            structure=widgets.IntSlider(value=0, min=0, max=Ns - 1, step=1,
                                        description="structure", disabled=(Ns == 1)),
            left=widgets.Dropdown(options=(["sum"] + [f"structure {i}" for i in range(Ns)]
                                           + ["orientation"]),
                                  value="sum", description="left"),
            zsl=widgets.IntSlider(value=Nz // 2, min=0, max=Nz - 1, step=1, description="z (real)"),
            kax=widgets.Dropdown(options=["kz", "ky", "kx"], value="kz", description="k-axis"),
            kmode=widgets.Dropdown(options=["slice", "sum"], value="slice", description="mid"),
            ksl=widgets.IntSlider(value=cz, min=0, max=Nkz - 1, step=1, description="k-slice"),
            pw=widgets.FloatSlider(value=power, min=0.1, max=1.0, step=0.05, description="power"),
            kpow=widgets.Dropdown(options=[("none", 0), ("×k", 1), ("×k²", 2)], value=2,
                                  description="radial"),
            flr=widgets.FloatSlider(value=spot_floor, min=0.01, max=1.0, step=0.01,
                                    description="spot floor"),
            azim=widgets.IntSlider(value=30, min=-180, max=180, step=5, description="azim"),
            elev=widgets.IntSlider(value=20, min=-90, max=90, step=5, description="elev"),
        )
        for s in c.values():
            s.style = {"description_width": "70px"}
            s.layout = widgets.Layout(width="360px")

    def render(structure, left, zsl, kax, kmode, ksl, pw, kpow, flr, azim, elev):
        si = min(structure, Ns - 1)
        zsl = min(zsl, Nz - 1)
        fig = plt.figure(figsize=(12, 4), constrained_layout=True)
        a1 = fig.add_subplot(1, 3, 1)
        a2 = fig.add_subplot(1, 3, 2)
        a3 = fig.add_subplot(1, 3, 3, projection="3d")

        # LEFT: weights (sum / one structure) or orientation RGB at this z-slice
        if left == "orientation":
            beam = R[zsl, :, :, :, 0]                                # rotated beam axis
            rgb = np.abs(beam) / (np.abs(beam).max() + 1e-9)
            wnorm = W[zsl].sum(-1)
            rgb = rgb * (wnorm / (wnorm.max() + 1e-12))[..., None]   # dark where vacuum
            a1.imshow(np.clip(rgb, 0, 1), origin="lower")
            a1.set_title(f"orientation (|R·ẑ| RGB)  z={zsl}", fontsize=9)
        else:
            wv = W.sum(-1) if left == "sum" else W[..., min(int(left.split()[1]), Ns - 1)]
            a1.imshow(wv[zsl], origin="lower", cmap="magma")
            a1.set_title(f"weights [{left}]  z={zsl}", fontsize=9)

        # MIDDLE: |SF| (radially weighted) slice or sum along the chosen k axis
        b = B[..., si] * (krad3 ** kpow)
        ax = {"kz": 0, "ky": 1, "kx": 2}[kax]
        if kmode == "sum":
            img = b.sum(axis=ax)
        else:
            idx = [slice(None)] * 3
            idx[ax] = min(ksl, b.shape[ax] - 1)
            img = b[tuple(idx)]
        a2.imshow(img ** pw, cmap="inferno")
        a2.set_title(f"|SF|  structure {si}  {kax} {kmode}", fontsize=9)

        # RIGHT: 3D scatter of the structure factor's local maxima
        xx, yy, zz, ii = maxima(B[..., si] * (krad3 ** kpow), flr)
        if len(ii):
            s = 6 + 240 * (ii / ii.max())
            a3.scatter(xx, yy, zz, s=s, c=ii, cmap="inferno", depthshade=True)
        a3.set_xlim(0, Nkx); a3.set_ylim(0, Nky); a3.set_zlim(0, Nkz)
        a3.set_xlabel("kx"); a3.set_ylabel("ky"); a3.set_zlabel("kz")
        a3.view_init(elev=elev, azim=azim)
        a3.set_title(f"Bragg spots  ({len(ii)} shown)", fontsize=9)
        plt.show()

    out = widgets.interactive_output(render, {
        "structure": c["structure"], "left": c["left"], "zsl": c["zsl"], "kax": c["kax"],
        "kmode": c["kmode"], "ksl": c["ksl"], "pw": c["pw"], "kpow": c["kpow"],
        "flr": c["flr"], "azim": c["azim"], "elev": c["elev"],
    })
    col = widgets.Layout(width="390px")
    left_ctrl = widgets.VBox([widgets.HTML("<b>weights (left)</b>"), c["left"], c["zsl"]], layout=col)
    mid_ctrl = widgets.VBox([widgets.HTML("<b>structure factor (middle)</b>"), c["structure"],
                             c["kax"], c["kmode"], c["ksl"], c["pw"], c["kpow"]], layout=col)
    right_ctrl = widgets.VBox([widgets.HTML("<b>Bragg spots (right)</b>"), c["flr"],
                               c["azim"], c["elev"]], layout=col)
    rows = []
    if title:
        rows.append(widgets.HTML(f"<h3 style='margin:2px 0'>{title}</h3>"))
    if controls is None:
        rows.append(widgets.HBox([left_ctrl, mid_ctrl, right_ctrl]))
    rows.append(out)
    box = widgets.VBox(rows)
    box.ctrl = c
    return box
