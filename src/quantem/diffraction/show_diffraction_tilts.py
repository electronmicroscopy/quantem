"""Interactive tilt-series diffraction-pattern viewer (factorized tomography).

One reusable widget for stepping through a 4D-STEM tilt series and comparing
several stacks (e.g. measured vs reconstructed) on shared controls:

* a **tilt** slider to flip through orientations,
* **probe step** to subsample the scan positions (so full-size datasets are
  viewable),
* an **origin radius** + **radial weight** (``|k|`` / ``|k|**2``) so the direct
  beam is removed / suppressed and the weak Bragg spots set the scale,
* a **histogram** panel with a draggable display **range** (the intensity scale
  is taken from the off-origin pixels only),
* a **power** law for display.

All controls are shared across the stacks shown.
"""

from __future__ import annotations

import numpy as np
import torch


def show_diffraction_tilts(series, tilts, labels=None, probe_step=1,
                           origin_radius=2, kpow=0, power=0.5):
    """Build the tilt-series viewer.

    Parameters
    ----------
    series : array or sequence of arrays
        One or more diffraction stacks, each ``(n_tilt, n_row, n_col, det_row,
        det_col)`` of intensities (torch or numpy). Multiple stacks are shown
        side by side on shared controls (e.g. measured vs reconstructed).
    tilts : sequence
        Tilt angles (deg) for the axis labels; length ``n_tilt``.
    labels : sequence of str, optional
        Per-stack titles.
    probe_step : int, default 1
        Initial scan-position subsampling stride.
    origin_radius : int, default 2
        Initial radius (detector pixels) of the central region excluded when
        setting the intensity scale (the direct beam).
    kpow : int, default 0
        Initial radial weighting ``|k|**kpow`` applied for display (0/1/2).
        Off by default -- multiplying the image by ``|k|**2`` distorts the
        (few-pixel) probe disk and Bragg spots; the off-origin display range
        already handles the direct beam.
    power : float, default 0.5
        Initial display power law.

    Returns
    -------
    ipywidgets.Widget
        A displayable box of controls plus the linked figure.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    if not isinstance(series, (list, tuple)):
        series = [series]
    S = [(s.detach().cpu().numpy() if hasattr(s, "detach") else np.asarray(s)) for s in series]
    n_series = len(S)
    n_tilt, n_row, n_col, det_r, det_c = S[0].shape
    if labels is None:
        labels = [f"series {i}" for i in range(n_series)]
    tilts = list(tilts)

    # fftshifted radial coordinate (detector pixels) for |k| weighting + origin mask
    cr, cc = det_r // 2, det_c // 2
    rr = np.sqrt((np.arange(det_r)[:, None] - cr) ** 2 + (np.arange(det_c)[None, :] - cc) ** 2)
    Ssh = [np.fft.fftshift(s, axes=(-2, -1)) for s in S]           # beam at (cr, cc)

    def weighted(sh, kp):
        return sh * (rr ** kp)[None, None, None] if kp else sh

    def tile(frame, step):                                         # frame: (n_row,n_col,det,det)
        d = frame[::step, ::step]
        nr, nc = d.shape[:2]
        return d.transpose(0, 2, 1, 3).reshape(nr * det_r, nc * det_c)

    tilt = widgets.IntSlider(value=n_tilt // 2, min=0, max=n_tilt - 1, step=1, description="tilt")
    pstep = widgets.IntSlider(value=probe_step, min=1, max=max(1, min(n_row, n_col)), step=1,
                              description="probe step")
    kpw = widgets.Dropdown(options=[("none", 0), ("×k", 1), ("×k²", 2)], value=kpow, description="radial")
    pw = widgets.FloatSlider(value=power, min=0.15, max=1.0, step=0.05, description="power")
    orad = widgets.IntSlider(value=origin_radius, min=0, max=min(cr, cc), step=1, description="origin r")
    vrange = widgets.FloatRangeSlider(value=[0.0, 1.0], min=0.0, max=1.0, step=0.01,
                                      description="range", readout_format=".2f")

    def render(tilt, pstep, kpw, pw, orad, vrange):
        mask = rr > orad                                           # off-origin pixels
        wser = [weighted(sh, kpw) for sh in Ssh]
        # scale from off-origin pixels only, pooled over stacks (subsampled by probe step)
        offvals = np.concatenate([
            w[:, ::pstep, ::pstep][..., mask].ravel() for w in wser
        ])
        offvals = np.clip(offvals, 0.0, None) ** pw
        hi = float(np.quantile(offvals, 0.999)) or 1.0
        vmin, vmax = vrange[0] * hi, vrange[1] * hi

        fig = plt.figure(figsize=(3.6 * (n_series + 1), 3.8), constrained_layout=True)
        axh = fig.add_subplot(1, n_series + 1, 1)
        axh.hist(offvals, bins=80, color="0.6")
        axh.axvline(vmin, color="C0", lw=1.5)
        axh.axvline(vmax, color="C1", lw=1.5)
        axh.set_yscale("log")
        axh.set_title("off-origin histogram\n(display scale)", fontsize=8)
        axh.set_xlabel(f"(|k|^{kpw} · I)^{pw:.2f}", fontsize=8)
        for i, w in enumerate(wser):
            ax = fig.add_subplot(1, n_series + 1, 2 + i)
            img = tile(w[tilt], pstep) ** pw
            ax.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_title(f"{labels[i]}   tilt {tilts[tilt]:+g}°", fontsize=9)
            ax.axis("off")
        plt.show()

    out = widgets.interactive_output(render, {
        "tilt": tilt, "pstep": pstep, "kpw": kpw, "pw": pw, "orad": orad, "vrange": vrange,
    })
    for s in (tilt, pstep, kpw, pw, orad, vrange):
        s.style = {"description_width": "80px"}
        s.layout = widgets.Layout(width="330px")
    controls = widgets.VBox([
        widgets.HBox([tilt, pstep, kpw]),
        widgets.HBox([pw, orad, vrange]),
    ])
    return widgets.VBox([controls, out])
