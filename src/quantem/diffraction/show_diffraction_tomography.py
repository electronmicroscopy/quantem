"""Interactive 3-panel viewer for :class:`DiffractionTomography` (anywidget).

Panels (left to right):

* real-space **weights** (or per-voxel **orientation** as RGB, brightness
  modulated by weight) at a chosen z slice,
* the selected **structure factor** -- a kz/ky/kx slice or sum of ``|SF|``,
  with power-law display and optional ``|k|``/``|k|^2`` radial weighting that
  suppresses the low-frequency envelope so the Bragg peaks stand out,
* the structure factor's **Bragg spots** -- local maxima as a rotatable 3D
  scatter (drag to rotate), marker area scaled by intensity.

All volume data is embedded in the widget and every interaction is rendered
client-side in JS, so the viewers stay interactive in a saved executed
notebook (no live kernel required). Passing the first viewer as ``controls=``
to a second call links the two (shared slices, structure index, power,
rotation, ...); display ranges stay independent.

Objects from the deprecated explicit-6D model are dispatched to the legacy
anywidget viewer (``show_diffraction_tomography_6d``).
"""

from __future__ import annotations

import anywidget
import numpy as np
import torch
import traitlets

_LINKED_TRAITS = (
    "left_mode", "sel_z", "sel_y", "sel_x", "mid_mode", "structure", "sum_axis",
    "view_mode", "slice_idx", "power", "kpow", "pts_power", "pts_scale",
    "pts_floor", "rot_theta", "rot_phi",
)


class CompactTomographyViewer(anywidget.AnyWidget):
    """3-panel client-side viewer for the compact (basis/weights/angles) model."""

    # interaction state (linkable across viewers)
    title = traitlets.Unicode("").tag(sync=True)
    left_mode = traitlets.Unicode("weights").tag(sync=True)   # 'weights' | 'orientation'
    sel_z = traitlets.Int(0).tag(sync=True)
    sel_y = traitlets.Int(0).tag(sync=True)
    sel_x = traitlets.Int(0).tag(sync=True)
    mid_mode = traitlets.Unicode("voxel").tag(sync=True)      # 'voxel' | 'basis'
    structure = traitlets.Int(0).tag(sync=True)
    sum_axis = traitlets.Int(0).tag(sync=True)                # 0=kz 1=ky 2=kx
    view_mode = traitlets.Unicode("slice").tag(sync=True)     # 'slice' | 'sum'
    slice_idx = traitlets.Int(-1).tag(sync=True)              # -1 -> center
    power = traitlets.Float(0.5).tag(sync=True)
    kpow = traitlets.Int(2).tag(sync=True)                    # |k|^kpow display weight
    left_range = traitlets.List(traitlets.Float(), default_value=[0.0, 1.0]).tag(sync=True)
    mid_range = traitlets.List(traitlets.Float(), default_value=[0.0, 1.0]).tag(sync=True)
    pts_range = traitlets.List(traitlets.Float(), default_value=[0.0, 1.0]).tag(sync=True)
    pts_power = traitlets.Float(0.25).tag(sync=True)
    pts_scale = traitlets.Float(1.0).tag(sync=True)
    pts_floor = traitlets.Float(0.1).tag(sync=True)           # fraction of max
    rot_theta = traitlets.Float(-60.0).tag(sync=True)
    rot_phi = traitlets.Float(20.0).tag(sync=True)

    # embedded data (client-side rendering only, no python round trips)
    real_shape = traitlets.List(traitlets.Int()).tag(sync=True)     # [Nz, Ny, Nx]
    n_struct = traitlets.Int(1).tag(sync=True)
    w_map = traitlets.Unicode("").tag(sync=True)                    # b64 f32 (Nz*Ny*Nx)
    orient_rgb = traitlets.Unicode("").tag(sync=True)               # b64 f32 (Nz*Ny*Nx*3)
    k_shape = traitlets.List(traitlets.Int()).tag(sync=True)        # [Nkz, Nky, Nkx]
    k_vol = traitlets.Unicode("").tag(sync=True)                    # b64 f32 (Ns*Nkz*Nky*Nkx), fftshifted
    rot_flat = traitlets.Unicode("").tag(sync=True)                 # b64 f32 (n_voxels*9), body->lab

    def __init__(self, dt, **kwargs):
        W = dt.weights.detach().abs().cpu().numpy()                 # [Nz,Ny,Nx,Ns]
        R = dt.rotation_matrices().detach().cpu().numpy()           # [Nz,Ny,Nx,3,3]
        B = np.abs(dt.masked_basis().detach().cpu().numpy())        # [Nkz,Nky,Nkx,Ns]
        Nz, Ny, Nx, Ns = W.shape
        w_sum = W.sum(-1)
        # orientation as |R . z_hat| RGB, dimmed by the (normalized) weight so
        # vacuum voxels stay dark
        beam = np.abs(R[..., :, 0])                                  # [Nz,Ny,Nx,3]
        beam = beam / max(beam.max(), 1e-12)
        rgb = beam * (w_sum / max(w_sum.max(), 1e-30))[..., None]
        kv = np.fft.fftshift(B, axes=(0, 1, 2)).transpose(3, 0, 1, 2)  # (Ns,Kz,Ky,Kx)

        import base64
        b64 = lambda a: base64.b64encode(a.astype(np.float32).copy().tobytes()).decode("ascii")
        zi, yi, xi = np.unravel_index(int(w_sum.argmax()), w_sum.shape)
        super().__init__(
            real_shape=[Nz, Ny, Nx],
            n_struct=Ns,
            w_map=b64(w_sum),
            orient_rgb=b64(rgb),
            k_shape=list(B.shape[:3]),
            k_vol=b64(kv),
            rot_flat=b64(R.reshape(-1, 9)),
            sel_z=int(zi),
            sel_y=int(yi),
            sel_x=int(xi),
            **kwargs,
        )

    _esm = r"""
function decodeF32(view) {
  if (view == null) return new Float32Array(0);
  if (typeof view === "string") {                    // base64 (JSON-safe, survives saved state)
    if (!view.length) return new Float32Array(0);
    const bin = atob(view);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
  }
  const b = view.buffer !== undefined ? view : new DataView(view);
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}

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

function styleButton(b, active) {
  Object.assign(b.style, {
    padding: "3px 10px", marginRight: "4px",
    border: active ? "1px solid #3b6fd4" : "1px solid #c5cbd6",
    borderRadius: "4px",
    background: active ? "#e3ecfb" : "#fafbfc",
    color: active ? "#1d4ed8" : "#333",
    fontWeight: active ? "600" : "400",
    fontSize: "11px", cursor: "pointer",
  });
}

function ctrlRow(parent) {
  return el("div", {
    display: "flex", alignItems: "center", gap: "6px",
    margin: "5px 0 0 0", fontSize: "11px", color: "#444",
  }, parent);
}

function slider(parent, min, max, step, width) {
  const s = el("input", { verticalAlign: "middle", width: width || "150px" }, parent);
  s.type = "range"; s.min = min; s.max = max; s.step = step;
  return s;
}

function histPanel(parent, width, label, onChange) {
  const wrap = el("div", { marginTop: "5px" }, parent);
  el("div", { fontSize: "10px", color: "#777" }, wrap).textContent = label;
  const H = 40;
  const canvas = el("canvas", { display: "block", borderRadius: "3px" }, wrap);
  canvas.width = width; canvas.height = H;
  const ctx = canvas.getContext("2d");
  let data = new Float32Array(0);
  let range = [0, 1];
  let drag = null;

  function draw() {
    ctx.clearRect(0, 0, width, H);
    ctx.fillStyle = "#f4f5f7";
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
        const h = bins[i] > 0 ? Math.max(2, (H - 8) * Math.log1p(bins[i]) / Math.log1p(bmax)) : 0;
        ctx.fillRect(i * width / nb, H - 6 - h, width / nb - 1, h);
      }
    }
    const x0 = range[0] * width, x1 = range[1] * width;
    ctx.fillStyle = "rgba(70,130,220,0.15)";
    ctx.fillRect(x0, 0, x1 - x0, H - 6);
    ctx.fillStyle = "#3b6fd4";
    for (const x of [x0, x1]) ctx.fillRect(x - 2, 0, 4, H - 6);
    ctx.fillStyle = "#999";
    ctx.fillRect(0, H - 5, width, 1);
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
  root.style.fontFamily =
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif";
  root.style.fontSize = "12px";

  const PW = 300;
  const title = el("div", {
    fontWeight: "600", fontSize: "13px", margin: "2px 0 8px 2px", color: "#222",
  }, root);
  const row = el("div", { display: "flex", gap: "18px", alignItems: "flex-start" }, root);

  function makePanel(labelText) {
    const p = el("div", {
      background: "#fff", border: "1px solid #e2e5ea", borderRadius: "6px",
      padding: "8px",
    }, row);
    el("div", {
      fontWeight: "600", fontSize: "11.5px", color: "#333", marginBottom: "6px",
    }, p).textContent = labelText;
    return p;
  }

  // ---------- data ----------
  let realShape = model.get("real_shape");        // [Nz, Ny, Nx]
  let kShape = model.get("k_shape");              // [Kz, Ky, Kx]
  let nStruct = model.get("n_struct");
  let wMap = decodeF32(model.get("w_map"));
  let orientRGB = decodeF32(model.get("orient_rgb"));
  let kVolAll = decodeF32(model.get("k_vol"));    // Ns * Kz*Ky*Kx
  let rotFlat = decodeF32(model.get("rot_flat")); // n_voxels * 9, body->lab

  // radial |k| in pixels (centered), shared by mid/right weighting
  function radial3() {
    const [Kz, Ky, Kx] = kShape;
    const r = new Float32Array(Kz * Ky * Kx);
    const cz = Math.floor(Kz / 2), cy = Math.floor(Ky / 2), cx = Math.floor(Kx / 2);
    let i = 0;
    for (let z = 0; z < Kz; z++)
      for (let y = 0; y < Ky; y++)
        for (let x = 0; x < Kx; x++, i++)
          r[i] = Math.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2);
    return r;
  }
  let kRad = radial3();

  function selVoxIdx() {
    const [Nz, Ny, Nx] = realShape;
    const z = Math.min(model.get("sel_z"), Nz - 1);
    const y = Math.min(model.get("sel_y"), Ny - 1);
    const x = Math.min(model.get("sel_x"), Nx - 1);
    return (z * Ny + y) * Nx + x;
  }

  // trilinear sample of the (fftshifted) structure volume at R_v^T k for every
  // output voxel: the selected voxel's structure factor, rendered client-side
  function rotatedVol(s, vidx) {
    const [Kz, Ky, Kx] = kShape;
    const n = Kz * Ky * Kx;
    const base = s * n;
    const R = rotFlat.subarray(9 * vidx, 9 * vidx + 9);   // row-major body->lab
    const cz = Math.floor(Kz / 2), cy = Math.floor(Ky / 2), cx = Math.floor(Kx / 2);
    const out = new Float32Array(n);
    let i = 0;
    for (let z = 0; z < Kz; z++) {
      const dz = z - cz;
      for (let y = 0; y < Ky; y++) {
        const dy = y - cy;
        for (let x = 0; x < Kx; x++, i++) {
          const dx = x - cx;
          // body = R^T [dz, dy, dx]
          const bz = R[0] * dz + R[3] * dy + R[6] * dx + cz;
          const by = R[1] * dz + R[4] * dy + R[7] * dx + cy;
          const bx = R[2] * dz + R[5] * dy + R[8] * dx + cx;
          const z0 = Math.floor(bz), y0 = Math.floor(by), x0 = Math.floor(bx);
          if (z0 < 0 || z0 >= Kz - 1 || y0 < 0 || y0 >= Ky - 1 || x0 < 0 || x0 >= Kx - 1) continue;
          const fz = bz - z0, fy = by - y0, fx = bx - x0;
          const i000 = (z0 * Ky + y0) * Kx + x0 + base;
          const v =
            (1 - fz) * ((1 - fy) * ((1 - fx) * kVolAll[i000] + fx * kVolAll[i000 + 1])
                        + fy * ((1 - fx) * kVolAll[i000 + Kx] + fx * kVolAll[i000 + Kx + 1]))
            + fz * ((1 - fy) * ((1 - fx) * kVolAll[i000 + Ky * Kx] + fx * kVolAll[i000 + Ky * Kx + 1])
                    + fy * ((1 - fx) * kVolAll[i000 + Ky * Kx + Kx] + fx * kVolAll[i000 + Ky * Kx + Kx + 1]));
          out[i] = v;
        }
      }
    }
    return out;
  }

  // current display volume: raw basis or the selected voxel's rotated SF,
  // with the radial display weight applied
  let wVol = null, wVolKey = "";
  function weightedVol() {
    const s = Math.min(model.get("structure"), nStruct - 1);
    const kp = model.get("kpow");
    const mm = model.get("mid_mode");
    const vidx = mm === "voxel" ? selVoxIdx() : -1;
    const key = s + "_" + kp + "_" + mm + "_" + vidx;
    if (wVolKey === key && wVol) return wVol;
    const [Kz, Ky, Kx] = kShape;
    const n = Kz * Ky * Kx;
    const base = s * n;
    const src = vidx >= 0 ? rotatedVol(s, vidx) : null;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const w = kp === 0 ? 1 : (kp === 1 ? kRad[i] : kRad[i] * kRad[i]);
      out[i] = (src ? src[i] : kVolAll[base + i]) * w;
    }
    wVolKey = key; wVol = out;
    return out;
  }

  function originIdx() {
    const [Kz, Ky, Kx] = kShape;
    return (Math.floor(Kz / 2) * Ky + Math.floor(Ky / 2)) * Kx + Math.floor(Kx / 2);
  }

  // =====================  LEFT: real space  =====================
  const pL = makePanel("real space — weights / orientation (click to pick voxel)");
  const canL = el("canvas", {
    border: "1px solid #d6dae1", borderRadius: "3px", display: "block", cursor: "crosshair",
  }, pL);
  canL.width = PW; canL.height = PW;
  const ctxL = canL.getContext("2d");
  const modeRow = ctrlRow(pL);
  const wBtn = el("button", {}, modeRow); wBtn.textContent = "weights";
  const oBtn = el("button", {}, modeRow); oBtn.textContent = "orientation";
  wBtn.addEventListener("click", () => { model.set("left_mode", "weights"); model.save_changes(); });
  oBtn.addEventListener("click", () => { model.set("left_mode", "orientation"); model.save_changes(); });
  const zRow = ctrlRow(pL);
  const zLabel = el("span", { minWidth: "44px" }, zRow);
  const zSlider = slider(zRow, 0, 1, 1, "200px");
  zSlider.addEventListener("input", () => {
    model.set("sel_z", parseInt(zSlider.value)); model.save_changes();
  });
  const histL = histPanel(pL, PW, "display range", (r) => {
    model.set("left_range", r); model.save_changes();
  });

  function pickVoxel(ev) {
    const r = canL.getBoundingClientRect();
    const [Nz, Ny, Nx] = realShape;
    const ix = Math.max(0, Math.min(Nx - 1, Math.floor((ev.clientX - r.left) / (PW / Nx))));
    const iy = Math.max(0, Math.min(Ny - 1, Math.floor((ev.clientY - r.top) / (PW / Ny))));
    if (ix !== model.get("sel_x") || iy !== model.get("sel_y")) {
      model.set("sel_x", ix); model.set("sel_y", iy); model.save_changes();
    }
  }
  let dragL = false;
  canL.addEventListener("pointerdown", (ev) => { dragL = true; canL.setPointerCapture(ev.pointerId); pickVoxel(ev); });
  canL.addEventListener("pointermove", (ev) => { if (dragL) pickVoxel(ev); });
  canL.addEventListener("pointerup", () => { dragL = false; });

  function drawLeft() {
    const [Nz, Ny, Nx] = realShape;
    const z = Math.min(model.get("sel_z"), Nz - 1);
    zSlider.max = Nz - 1; zSlider.value = z;
    zLabel.textContent = "z = " + z;
    const mode = model.get("left_mode");
    styleButton(wBtn, mode === "weights");
    styleButton(oBtn, mode === "orientation");
    const img = ctxL.createImageData(PW, PW);
    const sx = PW / Nx, sy = PW / Ny;
    if (mode === "orientation") {
      for (let py = 0; py < PW; py++) {
        const iy = Math.min(Ny - 1, Math.floor(py / sy));
        for (let px = 0; px < PW; px++) {
          const ix = Math.min(Nx - 1, Math.floor(px / sx));
          const o3 = 3 * ((z * Ny + iy) * Nx + ix);
          const o = 4 * (py * PW + px);
          img.data[o] = Math.round(255 * Math.min(1, orientRGB[o3]));
          img.data[o + 1] = Math.round(255 * Math.min(1, orientRGB[o3 + 1]));
          img.data[o + 2] = Math.round(255 * Math.min(1, orientRGB[o3 + 2]));
          img.data[o + 3] = 255;
        }
      }
      histL.setData(new Float32Array(0));
    } else {
      let mn = Infinity, mx = -Infinity;
      for (const v of wMap) { if (v < mn) mn = v; if (v > mx) mx = v; }
      if (mx <= mn) mx = mn + 1;
      const lr = model.get("left_range");
      const lo = mn + lr[0] * (mx - mn), hi = mn + lr[1] * (mx - mn);
      for (let py = 0; py < PW; py++) {
        const iy = Math.min(Ny - 1, Math.floor(py / sy));
        for (let px = 0; px < PW; px++) {
          const ix = Math.min(Nx - 1, Math.floor(px / sx));
          const v = wMap[(z * Ny + iy) * Nx + ix];
          const t = (v - lo) / Math.max(hi - lo, 1e-30);
          const c = cmap(t);
          const o = 4 * (py * PW + px);
          img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
        }
      }
      histL.setData(wMap);
      histL.setRange(model.get("left_range"));
    }
    ctxL.putImageData(img, 0, 0);
    // selection marker
    const msx = PW / Nx, msy = PW / Ny;
    const mx = (model.get("sel_x") + 0.5) * msx, my = (model.get("sel_y") + 0.5) * msy;
    ctxL.strokeStyle = "#00e5ff"; ctxL.lineWidth = 2;
    ctxL.strokeRect(mx - msx / 2, my - msy / 2, msx, msy);
  }

  // =====================  MIDDLE: structure factor  =====================
  const pM = makePanel("|structure factor| (fftshifted)");
  const canM = el("canvas", {
    border: "1px solid #d6dae1", borderRadius: "3px", display: "block",
  }, pM);
  canM.width = PW; canM.height = PW;
  const ctxM = canM.getContext("2d");
  const mmRow = ctrlRow(pM);
  const vBtn = el("button", {}, mmRow); vBtn.textContent = "voxel SF";
  const bBtn = el("button", {}, mmRow); bBtn.textContent = "basis";
  const voxLabel = el("span", { color: "#777", marginLeft: "6px" }, mmRow);
  vBtn.addEventListener("click", () => { model.set("mid_mode", "voxel"); model.save_changes(); });
  bBtn.addEventListener("click", () => { model.set("mid_mode", "basis"); model.save_changes(); });
  const sRow = ctrlRow(pM);
  const sLabel = el("span", { minWidth: "84px" }, sRow);
  const sSlider = slider(sRow, 0, Math.max(0, nStruct - 1), 1, "170px");
  sSlider.addEventListener("input", () => {
    model.set("structure", parseInt(sSlider.value)); model.save_changes();
  });
  const axRow = ctrlRow(pM);
  const axBtns = [];
  ["kz", "ky", "kx"].forEach((nm, ax) => {
    const b = el("button", {}, axRow);
    b.textContent = nm;
    b.addEventListener("click", () => { model.set("sum_axis", ax); model.save_changes(); });
    axBtns.push(b);
  });
  const modeBtn = el("button", { marginLeft: "6px" }, axRow);
  modeBtn.addEventListener("click", () => {
    model.set("view_mode", model.get("view_mode") === "sum" ? "slice" : "sum");
    model.save_changes();
  });
  const kpRow = ctrlRow(pM);
  el("span", {}, kpRow).textContent = "radial:";
  const kpBtns = [];
  ["none", "×k", "×k²"].forEach((nm, kp) => {
    const b = el("button", {}, kpRow);
    b.textContent = nm;
    b.addEventListener("click", () => { model.set("kpow", kp); model.save_changes(); });
    kpBtns.push(b);
  });
  const powRow = ctrlRow(pM);
  const powLabel = el("span", { minWidth: "84px" }, powRow);
  const powSlider = slider(powRow, 0.05, 1.0, 0.05, "170px");
  powSlider.addEventListener("input", () => {
    model.set("power", parseFloat(powSlider.value)); model.save_changes();
  });
  const slcRow = ctrlRow(pM);
  const slcLabel = el("span", { minWidth: "84px" }, slcRow);
  const slcSlider = slider(slcRow, 0, 1, 1, "170px");
  slcSlider.addEventListener("input", () => {
    model.set("slice_idx", parseInt(slcSlider.value)); model.save_changes();
  });
  const histM = histPanel(pM, PW, "display range (after power)", (r) => {
    model.set("mid_range", r); model.save_changes();
  });

  function sliceIndex(ax) {
    const K = kShape[ax];
    let s = model.get("slice_idx");
    if (s < 0 || s >= K) s = Math.floor(K / 2);
    return s;
  }

  function drawMid() {
    const [Kz, Ky, Kx] = kShape;
    const ax = model.get("sum_axis");
    const mode = model.get("view_mode");
    axBtns.forEach((b, i) => styleButton(b, i === ax));
    kpBtns.forEach((b, i) => styleButton(b, i === model.get("kpow")));
    const mm = model.get("mid_mode");
    styleButton(vBtn, mm === "voxel");
    styleButton(bBtn, mm === "basis");
    if (mm === "voxel") {
      const [Nz2, Ny2, Nx2] = realShape;
      const z = Math.min(model.get("sel_z"), Nz2 - 1);
      const y = Math.min(model.get("sel_y"), Ny2 - 1);
      const x = Math.min(model.get("sel_x"), Nx2 - 1);
      const wv = wMap[(z * Ny2 + y) * Nx2 + x];
      voxLabel.textContent = "voxel (" + z + ", " + y + ", " + x + ")  w = " + wv.toPrecision(3);
    } else {
      voxLabel.textContent = "(shared basis)";
    }
    styleButton(modeBtn, mode === "slice");
    modeBtn.textContent = mode === "sum" ? "sum" : "slice";
    sLabel.textContent = "structure = " + Math.min(model.get("structure"), nStruct - 1);
    sSlider.value = Math.min(model.get("structure"), nStruct - 1);
    sSlider.disabled = nStruct < 2;
    powLabel.textContent = "power = " + model.get("power").toFixed(2);
    powSlider.value = model.get("power");
    const K = kShape[ax];
    const si = sliceIndex(ax);
    slcSlider.max = K - 1; slcSlider.value = si;
    slcSlider.disabled = (mode !== "slice");
    slcLabel.textContent = mode === "slice"
      ? ["kz", "ky", "kx"][ax] + " slice = " + (si - Math.floor(K / 2))
      : "(slice off)";
    slcLabel.style.color = mode === "slice" ? "#444" : "#bbb";

    const vol = weightedVol();
    const dims = [[Ky, Kx], [Kz, Kx], [Kz, Ky]][ax];
    const p = model.get("power");
    const out = new Float32Array(dims[0] * dims[1]);
    for (let a = 0; a < dims[0]; a++) {
      for (let b = 0; b < dims[1]; b++) {
        let s = 0;
        if (mode === "sum") {
          if (ax === 0) { for (let k = 0; k < Kz; k++) s += vol[(k * Ky + a) * Kx + b]; }
          else if (ax === 1) { for (let k = 0; k < Ky; k++) s += vol[(a * Ky + k) * Kx + b]; }
          else { for (let k = 0; k < Kx; k++) s += vol[(a * Ky + b) * Kx + k]; }
        } else {
          if (ax === 0) s = vol[(si * Ky + a) * Kx + b];
          else if (ax === 1) s = vol[(a * Ky + si) * Kx + b];
          else s = vol[(a * Ky + b) * Kx + si];
        }
        out[a * dims[1] + b] = Math.pow(s, p);
      }
    }

    // range excludes the (already-zeroed) origin; in slice mode the range is
    // over the whole power-scaled volume so it stays fixed while scrubbing
    let mn = Infinity, mx = -Infinity;
    const midVals = [];
    const oIdx = originIdx();
    if (mode === "slice") {
      for (let i = 0; i < vol.length; i++) {
        if (i === oIdx) continue;                     // origin (=1) never sets the scale
        const v = Math.pow(vol[i], p);
        midVals.push(v);
        if (v < mn) mn = v; if (v > mx) mx = v;
      }
    } else {
      const [KzS, KyS, KxS] = kShape;
      const axS = model.get("sum_axis");
      const oab = [[Math.floor(KyS/2), Math.floor(KxS/2)], [Math.floor(KzS/2), Math.floor(KxS/2)],
                   [Math.floor(KzS/2), Math.floor(KyS/2)]][axS];
      const oFlat = oab[0] * dims[1] + oab[1];
      for (let i = 0; i < out.length; i++) {
        if (i === oFlat) continue;
        const v = out[i];
        midVals.push(v); if (v < mn) mn = v; if (v > mx) mx = v;
      }
    }
    if (!isFinite(mn)) { mn = 0; mx = 1; }
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
    histM.setData(Float32Array.from(midVals));
    histM.setRange(mr);
  }

  // =====================  RIGHT: 3D Bragg maxima  =====================
  const pR = makePanel("Bragg spots — drag to rotate");
  const canR = el("canvas", {
    border: "1px solid #d6dae1", borderRadius: "3px", cursor: "grab", display: "block",
    background: "#ffffff",
  }, pR);
  canR.width = PW; canR.height = PW;
  const ctxR = canR.getContext("2d");
  const cntRow = ctrlRow(pR);
  const cntLabel = el("span", { color: "#777" }, cntRow);
  const pPowRow = ctrlRow(pR);
  const pPowLabel = el("span", { minWidth: "104px" }, pPowRow);
  const pPowSlider = slider(pPowRow, 0.05, 1.0, 0.05, "150px");
  pPowSlider.addEventListener("input", () => {
    model.set("pts_power", parseFloat(pPowSlider.value)); model.save_changes();
  });
  const pSclRow = ctrlRow(pR);
  const pSclLabel = el("span", { minWidth: "104px" }, pSclRow);
  const pSclSlider = slider(pSclRow, 0.1, 10.0, 0.1, "150px");
  pSclSlider.addEventListener("input", () => {
    model.set("pts_scale", parseFloat(pSclSlider.value)); model.save_changes();
  });
  const pFlrRow = ctrlRow(pR);
  const pFlrLabel = el("span", { minWidth: "104px" }, pFlrRow);
  const pFlrSlider = slider(pFlrRow, -3, 0, 0.05, "150px");
  pFlrSlider.addEventListener("input", () => {
    model.set("pts_floor", Math.pow(10, parseFloat(pFlrSlider.value))); model.save_changes();
  });
  const histR = histPanel(pR, PW, "intensity range", (r) => {
    model.set("pts_range", r); model.save_changes();
  });

  // local maxima of the weighted volume (26-neighborhood), cached per key
  let maxima = null, maximaKey = "";
  function findMaxima() {
    const s = Math.min(model.get("structure"), nStruct - 1);
    const kp = model.get("kpow");
    const key = s + "_" + kp;
    if (maximaKey === key && maxima) return maxima;
    const [Kz, Ky, Kx] = kShape;
    const vol = weightedVol();
    const pos = [], val = [];
    const cz = Math.floor(Kz / 2), cy = Math.floor(Ky / 2), cx = Math.floor(Kx / 2);
    for (let z = 1; z < Kz - 1; z++) {
      for (let y = 1; y < Ky - 1; y++) {
        for (let x = 1; x < Kx - 1; x++) {
          const v = vol[(z * Ky + y) * Kx + x];
          if (v <= 0) continue;
          let isMax = true;
          for (let dz = -1; dz <= 1 && isMax; dz++)
            for (let dy = -1; dy <= 1 && isMax; dy++)
              for (let dx = -1; dx <= 1; dx++) {
                if (!dz && !dy && !dx) continue;
                if (vol[((z + dz) * Ky + (y + dy)) * Kx + (x + dx)] > v) { isMax = false; break; }
              }
          if (isMax) {
            if (z === cz && y === cy && x === cx) continue;   // origin: not a Bragg spot
            pos.push([x - cx, y - cy, z - cz]);   // (kx, ky, kz) centered
            val.push(v);
          }
        }
      }
    }
    maxima = { pos, val };
    maximaKey = key;
    return maxima;
  }

  function drawRight() {
    ctxR.fillStyle = "#ffffff";
    ctxR.fillRect(0, 0, PW, PW);
    const { pos, val } = findMaxima();
    const M = val.length;
    const pPow = model.get("pts_power");
    const pScl = model.get("pts_scale");
    const pFlr = model.get("pts_floor");
    pPowLabel.textContent = "size power = " + pPow.toFixed(2);
    pPowSlider.value = pPow;
    pSclLabel.textContent = "size scale = " + pScl.toFixed(1);
    pSclSlider.value = pScl;
    pFlrLabel.textContent = "min = " + (100 * pFlr).toFixed(2) + "% max";
    pFlrSlider.value = Math.log10(Math.max(pFlr, 1e-4));

    const th = model.get("rot_theta") * Math.PI / 180;
    const ph = model.get("rot_phi") * Math.PI / 180;
    const ct = Math.cos(th), st = Math.sin(th), cp = Math.cos(ph), sp = Math.sin(ph);
    const [Kz, Ky, Kx] = kShape;
    const scale = (PW / 2 - 12) / (0.5 * Math.max(Kz, Ky, Kx) * 1.15);

    // exclude the 3x3x3 origin cluster from the intensity range
    let mn = Infinity, mx = -Infinity;
    const rangeVals = [];
    for (let i = 0; i < M; i++) {
      const p = pos[i];
      if (Math.abs(p[0]) <= 1 && Math.abs(p[1]) <= 1 && Math.abs(p[2]) <= 1) continue;
      rangeVals.push(val[i]);
      if (val[i] < mn) mn = val[i]; if (val[i] > mx) mx = val[i];
    }
    if (!isFinite(mn)) { mn = 0; mx = 1; }
    if (mx <= mn) mx = mn + 1;
    const pr = model.get("pts_range");
    const lo = mn + pr[0] * (mx - mn), hi = mn + pr[1] * (mx - mn);
    const floor = mn + pFlr * (mx - mn);

    const axes = [[Kx / 2, 0, 0, "#d33"], [0, Ky / 2, 0, "#2a2"], [0, 0, Kz / 2, "#36c"]];
    for (const [x, y, z, col] of axes) {
      const rx = ct * x - st * y, ry0 = st * x + ct * y;
      const ry = cp * ry0 - sp * z;
      ctxR.strokeStyle = col; ctxR.lineWidth = 1.2;
      ctxR.beginPath();
      ctxR.moveTo(PW / 2, PW / 2);
      ctxR.lineTo(PW / 2 + rx * scale, PW / 2 - ry * scale);
      ctxR.stroke();
    }

    const pts = [];
    let shown = 0;
    for (let i = 0; i < M; i++) {
      const v = val[i];
      if (v < lo || v < floor) continue;
      shown++;
      const [x, y, z] = pos[i];
      const rx = ct * x - st * y, ry0 = st * x + ct * y;
      const ry = cp * ry0 - sp * z;
      const rz = sp * ry0 + cp * z;
      const tlin = Math.min(1, (v - lo) / Math.max(hi - lo, 1e-30));
      const tnorm = Math.pow(tlin, pPow);
      const area = Math.min(500, Math.max(4, 150 * pScl * tnorm));
      pts.push([rx, ry, rz, Math.sqrt(area / Math.PI), tnorm]);
    }
    cntLabel.textContent = shown + " / " + M + " maxima (origin cluster excluded)";
    pts.sort((a, b) => a[2] - b[2]);
    for (const [rx, ry, , rad, t] of pts) {
      const c = cmap(0.15 + 0.75 * t);
      ctxR.fillStyle = "rgba(" + c[0] + "," + c[1] + "," + c[2] + ",0.85)";
      ctxR.strokeStyle = "rgba(40,40,40,0.6)";
      ctxR.lineWidth = 0.75;
      ctxR.beginPath();
      ctxR.arc(PW / 2 + rx * scale, PW / 2 - ry * scale, rad, 0, 2 * Math.PI);
      ctxR.fill();
      ctxR.stroke();
    }
    histR.setData(Float32Array.from(rangeVals));
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

  // ---------- wiring ----------
  function refreshData() {
    realShape = model.get("real_shape");
    kShape = model.get("k_shape");
    nStruct = model.get("n_struct");
    wMap = decodeF32(model.get("w_map"));
    orientRGB = decodeF32(model.get("orient_rgb"));
    kVolAll = decodeF32(model.get("k_vol"));
    rotFlat = decodeF32(model.get("rot_flat"));
    kRad = radial3();
    wVolKey = ""; maximaKey = "";
  }
  function drawAll() {
    title.textContent = model.get("title");
    drawLeft(); drawMid(); drawRight();
  }

  model.on("change:w_map change:orient_rgb change:k_vol", () => { refreshData(); drawAll(); });
  model.on("change:left_mode change:left_range", () => { drawLeft(); });
  model.on("change:sel_z change:sel_y change:sel_x", () => { drawLeft(); drawMid(); drawRight(); });
  model.on("change:structure change:kpow change:mid_mode", () => { drawMid(); drawRight(); });
  model.on(
    "change:sum_axis change:view_mode change:slice_idx change:power change:mid_range",
    () => { drawMid(); },
  );
  model.on(
    "change:rot_theta change:rot_phi change:pts_range change:pts_power change:pts_scale change:pts_floor",
    () => { drawRight(); },
  );
  model.on("change:title", () => { title.textContent = model.get("title"); });

  drawAll();
}

export default { render };
"""


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
    controls : CompactTomographyViewer, optional
        A previously created viewer; the interaction state (slices, structure
        index, view mode, power, radial weight, 3D rotation) is linked so the
        two viewers move together. Display ranges stay independent.
    title : str, optional
        Heading shown above the panels.
    spot_floor : float, default 0.1
        Initial local-maxima cutoff (fraction of the off-origin maximum).
    power : float, default 0.5
        Initial display power for the middle panel.

    Returns
    -------
    CompactTomographyViewer
        The displayable widget; pass as ``controls=`` to a later call to link.
    """
    # legacy explicit-6D objects -> the old anywidget viewer
    if not hasattr(dt, "masked_basis"):
        from quantem.diffraction.show_diffraction_tomography_6d import (
            show_diffraction_tomography as _show_6d,
        )
        return _show_6d(dt, controls=controls, title=title)

    viewer = CompactTomographyViewer(dt, title=title, pts_floor=spot_floor, power=power)
    if controls is not None:
        for name in _LINKED_TRAITS:
            traitlets.link((controls, name), (viewer, name))
    return viewer
