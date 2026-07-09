"""Interactive tilt-series diffraction-pattern viewer (factorized tomography).

One reusable widget for stepping through a 4D-STEM tilt series and comparing
several stacks (e.g. measured vs reconstructed) on shared controls:

* a **tilt** slider to flip through orientations,
* **probe step** to subsample the scan positions (so full-size datasets are
  viewable),
* automatic **direct-beam exclusion**: the central disk (detected from the mean
  pattern, plus a margin) never sets the display scale, so the weak diffracted
  disks fill the color range by default,
* a **histogram** panel of the diffracted intensities with a draggable display
  **range**, a **power** law, and an optional ``|k|``/``|k|**2`` radial weight.

All controls are shared across the stacks shown.
"""

from __future__ import annotations

import numpy as np
import torch


def show_diffraction_tilts(series, tilts, labels=None, probe_step=1,
                           kpow=0, power=0.5):
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
    kpow : int, default 0
        Initial radial weighting ``|k|**kpow`` applied for display (0/1/2).
        Off by default -- multiplying the image by ``|k|**2`` distorts the
        (few-pixel) disks; the automatic beam exclusion already handles the
        direct beam.
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

    # fftshifted radial coordinate (detector pixels) for |k| weighting
    cr, cc = det_r // 2, det_c // 2
    rr = np.sqrt((np.arange(det_r)[:, None] - cr) ** 2 + (np.arange(det_c)[None, :] - cc) ** 2)
    Ssh = [np.fft.fftshift(s, axes=(-2, -1)) for s in S]           # beam at (cr, cc)

    # auto-detect the direct-beam disk so it never sets the display scale: the
    # central bright blob of the mean pattern (plus its Fresnel skirt) is
    # excluded via its radius + a 2-pixel margin.
    mean_dp = np.mean([s.mean(axis=(0, 1, 2)) for s in Ssh], axis=0)
    beam_pix = mean_dp > 0.5 * mean_dp[cr, cc]
    beam_r = float(rr[beam_pix].max()) + 2.0
    off_beam = rr > beam_r

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
    vrange = widgets.FloatRangeSlider(value=[0.0, 1.0], min=0.0, max=1.0, step=0.01,
                                      description="range", readout_format=".2f")

    def render(tilt, pstep, kpw, pw, vrange):
        wser = [weighted(sh, kpw) for sh in Ssh]
        # display scale from the diffracted (off-beam) pixels only, pooled over
        # all stacks and tilts, so the direct beam saturates and the scattered
        # disks fill the color range by default.
        offvals = np.concatenate([
            w[:, ::pstep, ::pstep][..., off_beam].ravel() for w in wser
        ])
        offvals = np.clip(offvals, 0.0, None) ** pw
        hi = float(offvals.max()) or 1.0
        vmin, vmax = vrange[0] * hi, vrange[1] * hi

        fig = plt.figure(figsize=(3.6 * (n_series + 1), 3.8), constrained_layout=True)
        axh = fig.add_subplot(1, n_series + 1, 1)
        axh.hist(offvals[offvals > 0], bins=80, color="0.6")
        axh.axvline(vmin, color="C0", lw=1.5)
        axh.axvline(vmax, color="C1", lw=1.5)
        axh.set_yscale("log")
        axh.set_title(f"diffracted intensities\n(beam disk r<{beam_r:.0f}px excluded)", fontsize=8)
        axh.set_xlabel(f"(|k|^{kpw} · I)^{pw:.2f}", fontsize=8)
        for i, w in enumerate(wser):
            ax = fig.add_subplot(1, n_series + 1, 2 + i)
            img = tile(w[tilt], pstep) ** pw
            ax.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_title(f"{labels[i]}   tilt {tilts[tilt]:+g}°", fontsize=9)
            ax.axis("off")
        plt.show()

    out = widgets.interactive_output(render, {
        "tilt": tilt, "pstep": pstep, "kpw": kpw, "pw": pw, "vrange": vrange,
    })
    for s in (tilt, pstep, kpw, pw, vrange):
        s.style = {"description_width": "80px"}
        s.layout = widgets.Layout(width="330px")
    controls = widgets.VBox([
        widgets.HBox([tilt, pstep, kpw]),
        widgets.HBox([pw, vrange]),
    ])
    return widgets.VBox([controls, out])
