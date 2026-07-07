"""Interactive 6D diffraction-tomography viewer (anywidget).

Three linked panels:

- **left**: real-space map of per-voxel deviation intensity (sum of
  |SF|^2 over all k-voxels except the vacuum origin pixel), with a z
  slider, a draggable voxel selector, and a histogram range selector.
- **middle**: fftshifted |SF| of the selected voxel, summed over a chosen
  k axis (default axis 2) or the central slice through the origin, with a
  power-law display (default 0.5) and a histogram range selector.
- **right**: rotatable 3D scatter of the local maxima of the selected
  voxel's fftshifted |SF| (strictly brighter than all 26 neighbors, edge
  voxels excluded), marker area scaled with intensity (capped), with a
  histogram range selector.

Two viewers can be locked together (same voxel, sum axis, view mode,
power, and 3D rotation) by passing the first as ``controls=`` to the
second::

    controls = show_diffraction_tomography(sim, title="ground truth")
    show_diffraction_tomography(recon, controls=controls, title="reconstruction")
"""

from __future__ import annotations

import numpy as np
import torch
import anywidget
import traitlets

__all__ = ["show_diffraction_tomography", "DiffractionTomographyViewer"]


_LINKED_TRAITS = (
    "sel_x",
    "sel_y",
    "sel_z",
    "sum_axis",
    "view_mode",
    "power",
    "rot_theta",
    "rot_phi",
)


def _to_numpy_6d(obj) -> np.ndarray:
    """Extract a 6D complex numpy array from the supported input types."""
    # DiffractionTomography-like: prefer the reconstructed volume when present
    vol = getattr(obj, "sf_learned", None)
    if vol is None:
        vol = getattr(obj, "array", obj)
    if isinstance(vol, torch.Tensor):
        vol = vol.detach().cpu().numpy()
    vol = np.asarray(vol)
    if vol.ndim != 6:
        raise ValueError(f"expected a 6D volume, got ndim={vol.ndim}")
    return vol


class DiffractionTomographyViewer(anywidget.AnyWidget):
    """anywidget viewer for a 6D structure-factor volume."""

    # --- linked control state -------------------------------------------
    sel_x = traitlets.Int(0).tag(sync=True)
    sel_y = traitlets.Int(0).tag(sync=True)
    sel_z = traitlets.Int(0).tag(sync=True)
    sum_axis = traitlets.Int(2).tag(sync=True)          # k axis to sum over
    view_mode = traitlets.Unicode("sum").tag(sync=True)  # 'sum' | 'slice'
    power = traitlets.Float(0.5).tag(sync=True)
    rot_theta = traitlets.Float(-60.0).tag(sync=True)   # deg, azimuth
    rot_phi = traitlets.Float(20.0).tag(sync=True)      # deg, elevation

    # --- per-widget display state ---------------------------------------
    title = traitlets.Unicode("").tag(sync=True)
    left_range = traitlets.List(traitlets.Float(), default_value=[0.0, 1.0]).tag(sync=True)
    mid_range = traitlets.List(traitlets.Float(), default_value=[0.0, 1.0]).tag(sync=True)
    pts_range = traitlets.List(traitlets.Float(), default_value=[0.0, 1.0]).tag(sync=True)

    # --- data (python -> js) ---------------------------------------------
    real_shape = traitlets.List(traitlets.Int()).tag(sync=True)
    real_map = traitlets.Bytes().tag(sync=True)          # f32 (Nx*Ny*Nz)
    k_shape = traitlets.List(traitlets.Int()).tag(sync=True)
    k_vol = traitlets.Bytes().tag(sync=True)             # f32 fftshifted |SF|
    max_pos = traitlets.Bytes().tag(sync=True)           # f32 (M*3), centered
    max_int = traitlets.Bytes().tag(sync=True)           # f32 (M,)

    def __init__(self, volume6d, title: str = "", **kwargs):
        self._vol = _to_numpy_6d(volume6d)
        nx, ny, nz = self._vol.shape[:3]

        # real-space deviation intensity: sum |SF|^2 excluding the origin pixel
        amp2 = np.abs(self._vol) ** 2
        dev_map = amp2.sum(axis=(3, 4, 5)) - amp2[..., 0, 0, 0]
        self._dev_map = dev_map.astype(np.float32)

        # default selection: strongest deviation voxel
        ix, iy, iz = np.unravel_index(int(dev_map.argmax()), dev_map.shape)

        super().__init__(
            title=title,
            sel_x=int(ix),
            sel_y=int(iy),
            sel_z=int(iz),
            real_shape=[int(nx), int(ny), int(nz)],
            k_shape=[int(n) for n in self._vol.shape[3:]],
            real_map=self._dev_map.tobytes(),
            **kwargs,
        )
        self._push_voxel_data()
        self.observe(self._on_selection, names=["sel_x", "sel_y", "sel_z"])

    # ------------------------------------------------------------------
    def _on_selection(self, change=None):
        self._push_voxel_data()

    def _push_voxel_data(self):
        nx, ny, nz = self._vol.shape[:3]
        ix = int(np.clip(self.sel_x, 0, nx - 1))
        iy = int(np.clip(self.sel_y, 0, ny - 1))
        iz = int(np.clip(self.sel_z, 0, nz - 1))

        kv = np.abs(np.fft.fftshift(self._vol[ix, iy, iz])).astype(np.float32)
        pos, val = self._local_maxima(kv)

        with self.hold_trait_notifications():
            self.k_vol = kv.tobytes()
            self.max_pos = pos.astype(np.float32).tobytes()
            self.max_int = val.astype(np.float32).tobytes()

    @staticmethod
    def _local_maxima(kv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Voxels strictly brighter than all 26 neighbors (edges excluded)."""
        core = kv[1:-1, 1:-1, 1:-1]
        is_max = np.ones(core.shape, dtype=bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nb = kv[
                        1 + dx : kv.shape[0] - 1 + dx,
                        1 + dy : kv.shape[1] - 1 + dy,
                        1 + dz : kv.shape[2] - 1 + dz,
                    ]
                    is_max &= core > nb
        idx = np.argwhere(is_max) + 1                    # back to full-volume indices
        vals = kv[idx[:, 0], idx[:, 1], idx[:, 2]]
        keep = vals > 0
        idx, vals = idx[keep], vals[keep]
        center = (np.asarray(kv.shape, dtype=np.float64) // 2).astype(np.float64)
        return idx.astype(np.float64) - center[None, :], vals

    _esm = r"""
function decodeF32(view) {
  if (view == null) return new Float32Array(0);
  const b = view.buffer !== undefined ? view : new DataView(view);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

// inferno-ish colormap from anchor points
const CMAP = [
  [0.001, 0.000, 0.014], [0.166, 0.040, 0.348], [0.397, 0.083, 0.433],
  [0.610, 0.166, 0.393], [0.797, 0.280, 0.290], [0.930, 0.444, 0.168],
  [0.985, 0.652, 0.096], [0.949, 0.877, 0.318], [0.988, 0.998, 0.645],
];
function cmap(t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (CMAP.length - 1);
  const i = Math.min(CMAP.length - 2, Math.floor(x));
  const f = x - i;
  const c0 = CMAP[i], c1 = CMAP[i + 1];
  return [
    Math.round(255 * (c0[0] + f * (c1[0] - c0[0]))),
    Math.round(255 * (c0[1] + f * (c1[1] - c0[1]))),
    Math.round(255 * (c0[2] + f * (c1[2] - c0[2]))),
  ];
}

function el(tag, style, parent) {
  const e = document.createElement(tag);
  if (style) Object.assign(e.style, style);
  if (parent) parent.appendChild(e);
  return e;
}

// histogram strip with two draggable range handles; onChange([lo, hi]) with
// fractions of the data range
function histPanel(parent, width, onChange) {
  const H = 44;
  const canvas = el("canvas", { display: "block", marginTop: "2px" }, parent);
  canvas.width = width; canvas.height = H;
  const ctx = canvas.getContext("2d");
  let data = new Float32Array(0);
  let range = [0, 1];
  let drag = null;

  function draw() {
    ctx.clearRect(0, 0, width, H);
    ctx.fillStyle = "#f2f2f2";
    ctx.fillRect(0, 0, width, H);
    if (data.length) {
      const nb = 64;
      let mn = Infinity, mx = -Infinity;
      for (const v of data) { if (v < mn) mn = v; if (v > mx) mx = v; }
      if (mx <= mn) mx = mn + 1;
      const bins = new Float32Array(nb);
      for (const v of data) {
        const b = Math.min(nb - 1, Math.floor((v - mn) / (mx - mn) * nb));
        bins[b] += 1;
      }
      let bmax = 0;
      for (const b of bins) bmax = Math.max(bmax, b);
      ctx.fillStyle = "#9aa7bd";
      for (let i = 0; i < nb; i++) {
        const h = bins[i] > 0 ? Math.max(2, (H - 10) * Math.log1p(bins[i]) / Math.log1p(bmax)) : 0;
        ctx.fillRect(i * width / nb, H - 8 - h, width / nb - 1, h);
      }
    }
    // selected range shading + handles
    const x0 = range[0] * width, x1 = range[1] * width;
    ctx.fillStyle = "rgba(70,130,220,0.15)";
    ctx.fillRect(x0, 0, x1 - x0, H - 8);
    ctx.fillStyle = "#3b6fd4";
    for (const x of [x0, x1]) ctx.fillRect(x - 2, 0, 4, H - 8);
    ctx.fillStyle = "#666";
    ctx.fillRect(0, H - 6, width, 1);
  }

  canvas.addEventListener("pointerdown", (ev) => {
    const r = canvas.getBoundingClientRect();
    const fx = (ev.clientX - r.left) / width;
    drag = Math.abs(fx - range[0]) < Math.abs(fx - range[1]) ? 0 : 1;
    canvas.setPointerCapture(ev.pointerId);
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (drag === null) return;
    const r = canvas.getBoundingClientRect();
    let fx = Math.max(0, Math.min(1, (ev.clientX - r.left) / width));
    range[drag] = fx;
    if (range[0] > range[1]) { range = [range[1], range[0]]; drag = 1 - drag; }
    draw();
    onChange([range[0], range[1]]);
  });
  canvas.addEventListener("pointerup", () => { drag = null; });

  return {
    setData(d) { data = d; draw(); },
    setRange(r) { range = [r[0], r[1]]; draw(); },
  };
}

function render({ model, el: root }) {
  root.style.fontFamily = "sans-serif";
  root.style.fontSize = "12px";

  const PW = 300;                 // panel width
  const title = el("div", { fontWeight: "bold", margin: "2px 0 6px 0" }, root);
  const row = el("div", { display: "flex", gap: "14px", alignItems: "flex-start" }, root);

  // ---------- shared state decoded from the model ----------
  let realShape = model.get("real_shape");
  let kShape = model.get("k_shape");
  let realMap = decodeF32(model.get("real_map"));
  let kVol = decodeF32(model.get("k_vol"));
  let maxPos = decodeF32(model.get("max_pos"));
  let maxInt = decodeF32(model.get("max_int"));

  // =====================  LEFT: real space  =====================
  const pL = el("div", {}, row);
  el("div", {}, pL).textContent = "real space: deviation intensity";
  const zRow = el("div", { margin: "2px 0" }, pL);
  const zLabel = el("span", { marginRight: "6px" }, zRow);
  const zSlider = el("input", { verticalAlign: "middle", width: "200px" }, zRow);
  zSlider.type = "range"; zSlider.min = 0; zSlider.step = 1;
  const canL = el("canvas", { border: "1px solid #ccc", cursor: "crosshair" }, pL);
  canL.width = PW; canL.height = PW;
  const ctxL = canL.getContext("2d");
  const histL = histPanel(pL, PW, (r) => { model.set("left_range", r); model.save_changes(); });
  const voxLabel = el("div", { color: "#555", marginTop: "2px" }, pL);

  function drawLeft() {
    const [nx, ny, nz] = realShape;
    const z = model.get("sel_z");
    zSlider.max = nz - 1; zSlider.value = z;
    zLabel.textContent = "z = " + z;
    let mn = Infinity, mx = -Infinity;
    for (const v of realMap) { if (v < mn) mn = v; if (v > mx) mx = v; }
    if (mx <= mn) mx = mn + 1;
    const lr = model.get("left_range");
    const lo = mn + lr[0] * (mx - mn), hi = mn + lr[1] * (mx - mn);
    const img = ctxL.createImageData(PW, PW);
    const sx = PW / nx, sy = PW / ny;
    for (let py = 0; py < PW; py++) {
      const iy = Math.min(ny - 1, Math.floor(py / sy));
      for (let px = 0; px < PW; px++) {
        const ix = Math.min(nx - 1, Math.floor(px / sx));
        const v = realMap[(ix * ny + iy) * nz + z];
        const t = (v - lo) / Math.max(hi - lo, 1e-30);
        const c = cmap(t);
        const o = 4 * (py * PW + px);
        img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
      }
    }
    ctxL.putImageData(img, 0, 0);
    // selection marker (x horizontal, y vertical to match array indexing [x, y])
    const mxp = (model.get("sel_x") + 0.5) * sx, myp = (model.get("sel_y") + 0.5) * sy;
    ctxL.strokeStyle = "#00e5ff"; ctxL.lineWidth = 2;
    ctxL.strokeRect(mxp - sx / 2, myp - sy / 2, sx, sy);
    voxLabel.textContent =
      "voxel (" + model.get("sel_x") + ", " + model.get("sel_y") + ", " + model.get("sel_z") + ")";
    histL.setData(realMap);
    histL.setRange(lr);
  }

  function pickVoxel(ev) {
    const r = canL.getBoundingClientRect();
    const [nx, ny] = realShape;
    const ix = Math.max(0, Math.min(nx - 1, Math.floor((ev.clientX - r.left) / (PW / nx))));
    const iy = Math.max(0, Math.min(ny - 1, Math.floor((ev.clientY - r.top) / (PW / ny))));
    if (ix !== model.get("sel_x") || iy !== model.get("sel_y")) {
      model.set("sel_x", ix); model.set("sel_y", iy); model.save_changes();
    }
  }
  let dragL = false;
  canL.addEventListener("pointerdown", (ev) => { dragL = true; canL.setPointerCapture(ev.pointerId); pickVoxel(ev); });
  canL.addEventListener("pointermove", (ev) => { if (dragL) pickVoxel(ev); });
  canL.addEventListener("pointerup", () => { dragL = false; });
  zSlider.addEventListener("input", () => {
    model.set("sel_z", parseInt(zSlider.value)); model.save_changes();
  });

  // =====================  MIDDLE: k space  =====================
  const pM = el("div", {}, row);
  el("div", {}, pM).textContent = "k space of selected voxel (fftshifted)";
  const ctrlM = el("div", { margin: "2px 0" }, pM);
  const axBtns = [];
  for (const ax of [0, 1, 2]) {
    const b = el("button", { marginRight: "4px" }, ctrlM);
    b.textContent = "k" + ax;
    b.addEventListener("click", () => { model.set("sum_axis", ax); model.save_changes(); });
    axBtns.push(b);
  }
  const modeBtn = el("button", { marginLeft: "8px" }, ctrlM);
  modeBtn.addEventListener("click", () => {
    model.set("view_mode", model.get("view_mode") === "sum" ? "slice" : "sum");
    model.save_changes();
  });
  const powRow = el("div", { margin: "2px 0" }, pM);
  const powLabel = el("span", { marginRight: "6px" }, powRow);
  const powSlider = el("input", { verticalAlign: "middle", width: "170px" }, powRow);
  powSlider.type = "range"; powSlider.min = 0.1; powSlider.max = 1.0; powSlider.step = 0.05;
  powSlider.addEventListener("input", () => {
    model.set("power", parseFloat(powSlider.value)); model.save_changes();
  });
  const canM = el("canvas", { border: "1px solid #ccc" }, pM);
  canM.width = PW; canM.height = PW;
  const ctxM = canM.getContext("2d");
  const histM = histPanel(pM, PW, (r) => { model.set("mid_range", r); model.save_changes(); });

  function midImage() {
    // sum over (or slice at the center of) the chosen k axis, then power
    const [K0, K1, K2] = kShape;
    const ax = model.get("sum_axis");
    const mode = model.get("view_mode");
    const dims = [[K1, K2], [K0, K2], [K0, K1]][ax];
    const out = new Float32Array(dims[0] * dims[1]);
    const c = [Math.floor(K0 / 2), Math.floor(K1 / 2), Math.floor(K2 / 2)];
    const p = model.get("power");
    for (let a = 0; a < dims[0]; a++) {
      for (let b = 0; b < dims[1]; b++) {
        let s = 0;
        if (mode === "sum") {
          if (ax === 0) { for (let k = 0; k < K0; k++) s += kVol[(k * K1 + a) * K2 + b]; }
          else if (ax === 1) { for (let k = 0; k < K1; k++) s += kVol[(a * K1 + k) * K2 + b]; }
          else { for (let k = 0; k < K2; k++) s += kVol[(a * K1 + b) * K2 + k]; }
        } else {
          if (ax === 0) s = kVol[(c[0] * K1 + a) * K2 + b];
          else if (ax === 1) s = kVol[(a * K1 + c[1]) * K2 + b];
          else s = kVol[(a * K1 + b) * K2 + c[2]];
        }
        out[a * dims[1] + b] = Math.pow(s, p);
      }
    }
    return { out, dims };
  }

  function drawMid() {
    const ax = model.get("sum_axis");
    const mode = model.get("view_mode");
    axBtns.forEach((b, i) => { b.style.fontWeight = i === ax ? "bold" : "normal"; });
    modeBtn.textContent = mode === "sum" ? "mode: sum" : "mode: center slice";
    powLabel.textContent = "power = " + model.get("power").toFixed(2);
    powSlider.value = model.get("power");

    const { out, dims } = midImage();
    let mn = Infinity, mx = -Infinity;
    for (const v of out) { if (v < mn) mn = v; if (v > mx) mx = v; }
    if (mx <= mn) mx = mn + 1;
    const mr = model.get("mid_range");
    const lo = mn + mr[0] * (mx - mn), hi = mn + mr[1] * (mx - mn);
    const img = ctxM.createImageData(PW, PW);
    for (let py = 0; py < PW; py++) {
      const b = Math.min(dims[1] - 1, Math.floor(py / (PW / dims[1])));
      for (let px = 0; px < PW; px++) {
        const a = Math.min(dims[0] - 1, Math.floor(px / (PW / dims[0])));
        const t = (out[a * dims[1] + b] - lo) / Math.max(hi - lo, 1e-30);
        const c2 = cmap(t);
        const o = 4 * (py * PW + px);
        img.data[o] = c2[0]; img.data[o + 1] = c2[1]; img.data[o + 2] = c2[2]; img.data[o + 3] = 255;
      }
    }
    ctxM.putImageData(img, 0, 0);
    histM.setData(out);
    histM.setRange(mr);
  }

  // =====================  RIGHT: 3D maxima  =====================
  const pR = el("div", {}, row);
  el("div", {}, pR).textContent = "k-space local maxima (drag to rotate)";
  const cntLabel = el("div", { margin: "2px 0", color: "#555" }, pR);
  const canR = el("canvas", { border: "1px solid #ccc", cursor: "grab" }, pR);
  canR.width = PW; canR.height = PW;
  const ctxR = canR.getContext("2d");
  const histR = histPanel(pR, PW, (r) => { model.set("pts_range", r); model.save_changes(); });

  function drawRight() {
    ctxR.fillStyle = "#111";
    ctxR.fillRect(0, 0, PW, PW);
    const M = maxInt.length;
    cntLabel.textContent = M + " maxima";
    const th = model.get("rot_theta") * Math.PI / 180;
    const ph = model.get("rot_phi") * Math.PI / 180;
    const ct = Math.cos(th), st = Math.sin(th), cp = Math.cos(ph), sp = Math.sin(ph);
    // rotate about z by theta, then about the screen-x axis by phi
    const [K0, K1, K2] = kShape;
    const scale = (PW / 2 - 12) / (0.5 * Math.max(K0, K1, K2) * 1.15);

    let mn = Infinity, mx = -Infinity;
    for (const v of maxInt) { if (v < mn) mn = v; if (v > mx) mx = v; }
    if (mx <= mn) mx = mn + 1;
    const pr = model.get("pts_range");
    const lo = mn + pr[0] * (mx - mn), hi = mn + pr[1] * (mx - mn);

    // axes
    const axes = [[K0 / 2, 0, 0, "#e55"], [0, K1 / 2, 0, "#5c5"], [0, 0, K2 / 2, "#59f"]];
    for (const [x, y, z, col] of axes) {
      const rx = ct * x - st * y, ry0 = st * x + ct * y;
      const ry = cp * ry0 - sp * z;
      ctxR.strokeStyle = col; ctxR.lineWidth = 1;
      ctxR.beginPath();
      ctxR.moveTo(PW / 2, PW / 2);
      ctxR.lineTo(PW / 2 + rx * scale, PW / 2 - ry * scale);
      ctxR.stroke();
    }

    // depth-sorted points
    const pts = [];
    for (let i = 0; i < M; i++) {
      const v = maxInt[i];
      if (v < lo) continue;
      const x = maxPos[3 * i], y = maxPos[3 * i + 1], z = maxPos[3 * i + 2];
      const rx = ct * x - st * y, ry0 = st * x + ct * y;
      const ry = cp * ry0 - sp * z;
      const rz = sp * ry0 + cp * z;
      const tnorm = Math.min(1, (v - lo) / Math.max(hi - lo, 1e-30));
      const area = Math.max(4, 500 * tnorm);              // capped marker area
      pts.push([rx, ry, rz, Math.sqrt(area / Math.PI), tnorm]);
    }
    pts.sort((a, b) => a[2] - b[2]);
    for (const [rx, ry, , rad, t] of pts) {
      const c = cmap(0.25 + 0.75 * t);
      ctxR.fillStyle = "rgba(" + c[0] + "," + c[1] + "," + c[2] + ",0.85)";
      ctxR.beginPath();
      ctxR.arc(PW / 2 + rx * scale, PW / 2 - ry * scale, rad, 0, 2 * Math.PI);
      ctxR.fill();
    }
    histR.setData(maxInt);
    histR.setRange(pr);
  }

  let dragR = null;
  canR.addEventListener("pointerdown", (ev) => {
    dragR = [ev.clientX, ev.clientY, model.get("rot_theta"), model.get("rot_phi")];
    canR.setPointerCapture(ev.pointerId);
  });
  canR.addEventListener("pointermove", (ev) => {
    if (!dragR) return;
    const dth = (ev.clientX - dragR[0]) * 0.5;
    const dph = (ev.clientY - dragR[1]) * 0.5;
    model.set("rot_theta", dragR[2] + dth);
    model.set("rot_phi", Math.max(-89, Math.min(89, dragR[3] + dph)));
    model.save_changes();
  });
  canR.addEventListener("pointerup", () => { dragR = null; });

  // ---------- model wiring ----------
  function refreshData() {
    realShape = model.get("real_shape");
    kShape = model.get("k_shape");
    realMap = decodeF32(model.get("real_map"));
    kVol = decodeF32(model.get("k_vol"));
    maxPos = decodeF32(model.get("max_pos"));
    maxInt = decodeF32(model.get("max_int"));
  }
  function drawAll() {
    title.textContent = model.get("title");
    drawLeft(); drawMid(); drawRight();
  }

  model.on("change:k_vol change:max_pos change:max_int change:real_map", () => {
    refreshData(); drawAll();
  });
  model.on(
    "change:sel_x change:sel_y change:sel_z change:left_range", () => { drawLeft(); },
  );
  model.on(
    "change:sum_axis change:view_mode change:power change:mid_range", () => { drawMid(); },
  );
  model.on(
    "change:rot_theta change:rot_phi change:pts_range", () => { drawRight(); },
  );
  model.on("change:title", () => { title.textContent = model.get("title"); });

  drawAll();
}

export default { render };
"""


def show_diffraction_tomography(
    obj,
    controls: DiffractionTomographyViewer | None = None,
    title: str = "",
):
    """Display the 6D diffraction-tomography viewer for `obj`.

    Parameters
    ----------
    obj:
        A `DiffractionTomography` / `SimDiffractionTomography` instance
        (the reconstructed `sf_learned` volume is used when present,
        otherwise `array`), or a raw 6D numpy array / torch tensor.
    controls:
        A previously created viewer. When given, the voxel selection, k-sum
        axis, view mode, power, and 3D rotation of this viewer are linked to
        it, so the two stay in lockstep for side-by-side comparison.
    title:
        Optional label shown above the panels.

    Returns
    -------
    DiffractionTomographyViewer — pass as `controls=` to another call to
    lock the two viewers together.
    """
    viewer = DiffractionTomographyViewer(obj, title=title)
    if controls is not None:
        for name in _LINKED_TRAITS:
            traitlets.link((controls, name), (viewer, name))
        # adopt the controlling viewer's current state
        for name in _LINKED_TRAITS:
            setattr(viewer, name, getattr(controls, name))
    try:
        from IPython.display import display

        display(viewer)
    except ImportError:
        pass
    return viewer
