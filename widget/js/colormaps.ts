// ============================================================================
// Color palettes (LUT control points)
// ============================================================================

const COLORMAP_POINTS: Record<string, number[][]> = {
  inferno: [
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [252, 193, 57], [252, 255, 164],
  ],
  viridis: [
    [68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141],
    [42, 120, 142], [33, 145, 140], [34, 168, 132], [68, 191, 112],
    [122, 209, 81], [189, 223, 38], [253, 231, 37],
  ],
  plasma: [
    [13, 8, 135], [75, 3, 161], [126, 3, 168], [168, 34, 150],
    [203, 70, 121], [229, 107, 93], [248, 148, 65], [253, 195, 40], [240, 249, 33],
  ],
  magma: [
    [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
    [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191],
  ],
  hot: [
    [0, 0, 0], [87, 0, 0], [173, 0, 0], [255, 0, 0],
    [255, 87, 0], [255, 173, 0], [255, 255, 0], [255, 255, 128], [255, 255, 255],
  ],
  gray: [[0, 0, 0], [255, 255, 255]],
  hsv: [
    [255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255],
    [0, 0, 255], [255, 0, 255], [255, 0, 0],
  ],
  turbo: [
    [48, 18, 59], [69, 55, 161], [66, 107, 230], [30, 162, 230],
    [29, 212, 169], [79, 241, 89], [175, 240, 32], [244, 195, 12],
    [248, 118, 11], [207, 46, 3], [122, 4, 2],
  ],
  RdBu: [
    [103, 0, 31], [178, 24, 43], [214, 96, 77], [244, 165, 130],
    [253, 219, 199], [247, 247, 247], [209, 229, 240], [146, 197, 222],
    [67, 147, 195], [33, 102, 172], [5, 48, 97],
  ],
  // cividis: perceptually uniform, colorblind-safe.
  cividis: [
    [0, 32, 76], [0, 42, 102], [13, 64, 117], [42, 80, 125],
    [70, 97, 125], [99, 113, 124], [127, 130, 121], [156, 148, 117],
    [187, 167, 105], [221, 188, 80], [253, 215, 21],
  ],
  // seismic: divergent blue-white-red, saturated.
  seismic: [
    [0, 0, 76], [0, 0, 153], [0, 0, 255], [124, 124, 255],
    [255, 255, 255], [255, 124, 124], [255, 0, 0], [153, 0, 0], [76, 0, 0],
  ],
  // RdBu_r: reverse of RdBu (blue → red, blue at low values).
  RdBu_r: [
    [5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222],
    [209, 229, 240], [247, 247, 247], [253, 219, 199], [244, 165, 130],
    [214, 96, 77], [178, 24, 43], [103, 0, 31],
  ],
  // twilight: cyclic, dark-light-dark.
  twilight: [
    [225, 216, 226], [184, 192, 224], [136, 158, 213], [101, 124, 197],
    [80, 92, 174], [69, 64, 135], [57, 36, 87], [40, 17, 47],
    [57, 36, 87], [69, 64, 135], [80, 92, 174], [101, 124, 197],
    [136, 158, 213], [184, 192, 224], [225, 216, 226],
  ],
  // twilight_shifted: same as twilight but phase-shifted to start mid-cycle.
  twilight_shifted: [
    [40, 17, 47], [57, 36, 87], [69, 64, 135], [80, 92, 174],
    [101, 124, 197], [136, 158, 213], [184, 192, 224], [225, 216, 226],
    [184, 192, 224], [136, 158, 213], [101, 124, 197], [80, 92, 174],
    [69, 64, 135], [57, 36, 87], [40, 17, 47],
  ],
};

export const COLORMAP_NAMES = Object.keys(COLORMAP_POINTS);

function createColormapLUT(points: number[][]): Uint8Array {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (points.length - 1);
    const idx = Math.floor(t);
    const frac = t - idx;
    const p0 = points[Math.min(idx, points.length - 1)];
    const p1 = points[Math.min(idx + 1, points.length - 1)];
    lut[i * 3] = Math.round(p0[0] + frac * (p1[0] - p0[0]));
    lut[i * 3 + 1] = Math.round(p0[1] + frac * (p1[1] - p0[1]));
    lut[i * 3 + 2] = Math.round(p0[2] + frac * (p1[2] - p0[2]));
  }
  return lut;
}

export const COLORMAPS: Record<string, Uint8Array> = Object.fromEntries(
  Object.entries(COLORMAP_POINTS).map(([name, points]) => [name, createColormapLUT(points)])
);

// ============================================================================
// CPU colormap (Float32 -> RGBA via 256-entry LUT)
// ============================================================================

/** Apply colormap LUT to float data, writing into an RGBA Uint8ClampedArray. */
export function applyColormap(
  data: Float32Array,
  rgba: Uint8ClampedArray,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
): void {
  const range = vmax > vmin ? vmax - vmin : 1;
  const uniformData = !(vmax > vmin);
  for (let i = 0; i < data.length; i++) {
    const clipped = Math.max(vmin, Math.min(vmax, data[i]));
    const v = uniformData ? 128 : Math.min(255, Math.floor(((clipped - vmin) / range) * 255));
    const j = i * 4;
    const lutIdx = v * 3;
    rgba[j] = lut[lutIdx];
    rgba[j + 1] = lut[lutIdx + 1];
    rgba[j + 2] = lut[lutIdx + 2];
    rgba[j + 3] = 255;
  }
}

/** Create an offscreen canvas with colormapped data. Returns null if context unavailable. */
export function renderToOffscreen(
  data: Float32Array,
  width: number,
  height: number,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
): HTMLCanvasElement | null {
  const offscreen = document.createElement("canvas");
  offscreen.width = width;
  offscreen.height = height;
  const ctx = offscreen.getContext("2d");
  if (!ctx) return null;
  const imgData = ctx.createImageData(width, height);
  applyColormap(data, imgData.data, lut, vmin, vmax);
  ctx.putImageData(imgData, 0, 0);
  return offscreen;
}

/** Render colormapped data to a reusable offscreen canvas + ImageData (avoids per-frame allocation). */
export function renderToOffscreenReuse(
  data: Float32Array,
  lut: Uint8Array,
  vmin: number,
  vmax: number,
  offscreen: HTMLCanvasElement,
  imgData: ImageData,
): void {
  applyColormap(data, imgData.data, lut, vmin, vmax);
  offscreen.getContext("2d")!.putImageData(imgData, 0, 0);
}

// ============================================================================
// WebGPU-accelerated colormap engine
// ============================================================================

// 2D dispatch (16×16 workgroups) to stay within WebGPU's 65535 workgroup limit.
// 1D dispatch with wg=256 needs ceil(4096*4096/256)=65536 — exceeds the limit by 1.
// ============================================================================
// WebGPU colormap engine (compute shader, ~300x faster than CPU loop on 4K data)
// ============================================================================

const COLORMAP_SHADER = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  vmin: f32,
  vmax: f32,
  log_scale: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  var val = data[idx];
  if (params.log_scale == 1u) {
    val = log(1.0 + max(val, 0.0));
  }
  let range = max(params.vmax - params.vmin, 1e-30);
  let clipped = clamp(val, params.vmin, params.vmax);
  let t = (clipped - params.vmin) / range;
  let lutIdx = min(u32(t * 255.0), 255u);
  let rgb = lut[lutIdx];
  // Simplified: LUT is already packed as R|(G<<8)|(B<<16), just add alpha
  rgba[idx] = rgb | 0xFF000000u;
}
`;

const SCALED_COLORMAP_SHADER = /* wgsl */ `
struct Params {
  src_width: u32,
  src_height: u32,
  out_width: u32,
  out_height: u32,
  vmin: f32,
  vmax: f32,
  log_scale: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.out_width || gid.y >= params.out_height) { return; }
  let src_x = min(u32((f32(gid.x) + 0.5) * f32(params.src_width) / f32(params.out_width)), params.src_width - 1u);
  let src_y = min(u32((f32(gid.y) + 0.5) * f32(params.src_height) / f32(params.out_height)), params.src_height - 1u);
  let src_idx = src_y * params.src_width + src_x;
  let out_idx = gid.y * params.out_width + gid.x;
  var val = data[src_idx];
  if (params.log_scale == 1u) {
    val = log(1.0 + max(val, 0.0));
  }
  let range = max(params.vmax - params.vmin, 1e-30);
  let clipped = clamp(val, params.vmin, params.vmax);
  let t = (clipped - params.vmin) / range;
  let lutIdx = min(u32(t * 255.0), 255u);
  let rgb = lut[lutIdx];
  rgba[out_idx] = rgb | 0xFF000000u;
}
`;

const SHARED_GRID_COLORMAP_SHADER = /* wgsl */ `
struct Params {
  src_width: u32,
  src_height: u32,
  src_panel_width: u32,
  out_width: u32,
  out_height: u32,
  panel_count: u32,
  cols: u32,
  rows: u32,
  log_scale: u32,
  bg_rgb: u32,
  shared_source: u32,
  _pad0: u32,
  vmin: f32,
  vmax: f32,
  gap: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;

fn pack_rgb(rgb: u32) -> u32 {
  return rgb | 0xFF000000u;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.out_width || gid.y >= params.out_height) { return; }
  let out_idx = gid.y * params.out_width + gid.x;
  let bg = pack_rgb(params.bg_rgb);
  if (params.cols == 0u || params.rows == 0u || params.panel_count == 0u || params.src_width == 0u || params.src_height == 0u) {
    rgba[out_idx] = bg;
    return;
  }

  let src_panel_w = max(1u, min(params.src_panel_width, params.src_width));
  let gap = params.gap;
  let panel_w = (f32(params.out_width) - gap * f32(params.cols - 1u)) / f32(params.cols);
  let panel_h = (f32(params.out_height) - gap * f32(params.rows - 1u)) / f32(params.rows);
  let stride_x = panel_w + gap;
  let stride_y = panel_h + gap;
  let px = f32(gid.x) + 0.5;
  let py = f32(gid.y) + 0.5;
  let col = u32(floor(px / stride_x));
  let row = u32(floor(py / stride_y));
  if (col >= params.cols || row >= params.rows) {
    rgba[out_idx] = bg;
    return;
  }

  let local_x = px - f32(col) * stride_x;
  let local_y = py - f32(row) * stride_y;
  let panel_idx = row * params.cols + col;
  if (panel_idx >= params.panel_count || local_x < 0.0 || local_y < 0.0 || local_x >= panel_w || local_y >= panel_h) {
    rgba[out_idx] = bg;
    return;
  }

  let src_panel_idx = select(panel_idx, 0u, params.shared_source == 1u);
  let src_local_x = min(u32(local_x * f32(src_panel_w) / panel_w), src_panel_w - 1u);
  let src_x = min(src_panel_idx * src_panel_w + src_local_x, params.src_width - 1u);
  let src_y = min(u32(local_y * f32(params.src_height) / panel_h), params.src_height - 1u);
  let src_idx = src_y * params.src_width + src_x;
  var val = data[src_idx];
  if (params.log_scale == 1u) {
    val = log(1.0 + max(val, 0.0));
  }
  let range = max(params.vmax - params.vmin, 1e-30);
  let clipped = clamp(val, params.vmin, params.vmax);
  let t = (clipped - params.vmin) / range;
  let lutIdx = min(u32(t * 255.0), 255u);
  let rgb = lut[lutIdx];
  rgba[out_idx] = rgb | 0xFF000000u;
}
`;

const DIRECT_GRID_COLORMAP_SHADER = /* wgsl */ `
struct Params {
  src_width: u32,
  src_height: u32,
  src_panel_width: u32,
  out_width: u32,
  out_height: u32,
  panel_count: u32,
  cols: u32,
  rows: u32,
  log_scale: u32,
  bg_rgb: u32,
  shared_source: u32,
  _pad0: u32,
  vmin: f32,
  vmax: f32,
  gap: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
  let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  return out;
}

fn unpack_rgb(rgb: u32) -> vec4f {
  let r = f32(rgb & 0xFFu) / 255.0;
  let g = f32((rgb >> 8u) & 0xFFu) / 255.0;
  let b = f32((rgb >> 16u) & 0xFFu) / 255.0;
  return vec4f(r, g, b, 1.0);
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  if (params.cols == 0u || params.rows == 0u || params.panel_count == 0u || params.src_width == 0u || params.src_height == 0u) {
    return unpack_rgb(params.bg_rgb);
  }

  let out_x = min(u32(in.uv.x * f32(params.out_width)), params.out_width - 1u);
  let out_y = min(u32(in.uv.y * f32(params.out_height)), params.out_height - 1u);
  let src_panel_w = max(1u, min(params.src_panel_width, params.src_width));
  let gap = params.gap;
  let panel_w = (f32(params.out_width) - gap * f32(params.cols - 1u)) / f32(params.cols);
  let panel_h = (f32(params.out_height) - gap * f32(params.rows - 1u)) / f32(params.rows);
  let stride_x = panel_w + gap;
  let stride_y = panel_h + gap;
  let px = f32(out_x) + 0.5;
  let py = f32(out_y) + 0.5;
  let col = u32(floor(px / stride_x));
  let row = u32(floor(py / stride_y));
  if (col >= params.cols || row >= params.rows) {
    return unpack_rgb(params.bg_rgb);
  }

  let local_x = px - f32(col) * stride_x;
  let local_y = py - f32(row) * stride_y;
  let panel_idx = row * params.cols + col;
  if (panel_idx >= params.panel_count || local_x < 0.0 || local_y < 0.0 || local_x >= panel_w || local_y >= panel_h) {
    return unpack_rgb(params.bg_rgb);
  }

  let src_panel_idx = select(panel_idx, 0u, params.shared_source == 1u);
  let src_local_x = min(u32(local_x * f32(src_panel_w) / panel_w), src_panel_w - 1u);
  let src_x = min(src_panel_idx * src_panel_w + src_local_x, params.src_width - 1u);
  let src_y = min(u32(local_y * f32(params.src_height) / panel_h), params.src_height - 1u);
  let src_idx = src_y * params.src_width + src_x;
  var val = data[src_idx];
  if (params.log_scale == 1u) {
    val = log(1.0 + max(val, 0.0));
  }
  let range = max(params.vmax - params.vmin, 1e-30);
  let clipped = clamp(val, params.vmin, params.vmax);
  let t = (clipped - params.vmin) / range;
  let lut_idx = min(u32(t * 255.0), 255u);
  return unpack_rgb(lut[lut_idx]);
}
`;

const DIRECT_GRID_RANGES_COLORMAP_SHADER = /* wgsl */ `
struct Params {
  src_width: u32,
  src_height: u32,
  src_panel_width: u32,
  out_width: u32,
  out_height: u32,
  panel_count: u32,
  cols: u32,
  rows: u32,
  _unused_log_scale: u32,
  bg_rgb: u32,
  shared_source: u32,
  _pad0: u32,
  _unused_vmin: f32,
  _unused_vmax: f32,
  gap: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read> panel_ranges: array<vec4f>;

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
  let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  return out;
}

fn unpack_rgb(rgb: u32) -> vec4f {
  let r = f32(rgb & 0xFFu) / 255.0;
  let g = f32((rgb >> 8u) & 0xFFu) / 255.0;
  let b = f32((rgb >> 16u) & 0xFFu) / 255.0;
  return vec4f(r, g, b, 1.0);
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  if (params.cols == 0u || params.rows == 0u || params.panel_count == 0u || params.src_width == 0u || params.src_height == 0u) {
    return unpack_rgb(params.bg_rgb);
  }

  let out_x = min(u32(in.uv.x * f32(params.out_width)), params.out_width - 1u);
  let out_y = min(u32(in.uv.y * f32(params.out_height)), params.out_height - 1u);
  let src_panel_w = max(1u, min(params.src_panel_width, params.src_width));
  let gap = params.gap;
  let panel_w = (f32(params.out_width) - gap * f32(params.cols - 1u)) / f32(params.cols);
  let panel_h = (f32(params.out_height) - gap * f32(params.rows - 1u)) / f32(params.rows);
  let stride_x = panel_w + gap;
  let stride_y = panel_h + gap;
  let px = f32(out_x) + 0.5;
  let py = f32(out_y) + 0.5;
  let col = u32(floor(px / stride_x));
  let row = u32(floor(py / stride_y));
  if (col >= params.cols || row >= params.rows) {
    return unpack_rgb(params.bg_rgb);
  }

  let local_x = px - f32(col) * stride_x;
  let local_y = py - f32(row) * stride_y;
  let panel_idx = row * params.cols + col;
  if (panel_idx >= params.panel_count || local_x < 0.0 || local_y < 0.0 || local_x >= panel_w || local_y >= panel_h) {
    return unpack_rgb(params.bg_rgb);
  }

  let src_panel_idx = select(panel_idx, 0u, params.shared_source == 1u);
  let src_local_x = min(u32(local_x * f32(src_panel_w) / panel_w), src_panel_w - 1u);
  let src_x = min(src_panel_idx * src_panel_w + src_local_x, params.src_width - 1u);
  let src_y = min(u32(local_y * f32(params.src_height) / panel_h), params.src_height - 1u);
  let src_idx = src_y * params.src_width + src_x;
  let panel_range = panel_ranges[panel_idx];
  var val = data[src_idx];
  if (panel_range.z > 0.5) {
    val = log(1.0 + max(val, 0.0));
  }
  let vmin = panel_range.x;
  let vmax = panel_range.y;
  let range = max(vmax - vmin, 1e-30);
  let clipped = clamp(val, vmin, vmax);
  let t = (clipped - vmin) / range;
  let lut_idx = min(u32(t * 255.0), 255u);
  return unpack_rgb(lut[lut_idx]);
}
`;

const DIRECT_SLOT_COLORMAP_SHADER = /* wgsl */ `
struct Params {
  src_width: u32,
  src_height: u32,
  src_x0: u32,
  src_region_width: u32,
  out_height: u32,
  out_width: u32,
  _unused_cols: u32,
  _unused_rows: u32,
  log_scale: u32,
  bg_rgb: u32,
  zoom: f32,
  _pad0: u32,
  vmin: f32,
  vmax: f32,
  pan_x: f32,
  pan_y: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
  let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  return out;
}

fn unpack_rgb(rgb: u32) -> vec4f {
  let r = f32(rgb & 0xFFu) / 255.0;
  let g = f32((rgb >> 8u) & 0xFFu) / 255.0;
  let b = f32((rgb >> 16u) & 0xFFu) / 255.0;
  return vec4f(r, g, b, 1.0);
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  if (params.src_width == 0u || params.src_height == 0u) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  let region_w = max(1u, min(params.src_region_width, params.src_width));
  let region_x0 = min(params.src_x0, params.src_width - 1u);
  let out_w = f32(max(1u, params.out_width));
  let out_h = f32(max(1u, params.out_height));
  let local_x = in.uv.x * out_w;
  let local_y = in.uv.y * out_h;
  let image_x = (local_x - params.pan_x) / max(params.zoom, 1e-6);
  let image_y = (local_y - params.pan_y) / max(params.zoom, 1e-6);
  if (image_x < 0.0 || image_y < 0.0 || image_x >= out_w || image_y >= out_h) {
    return unpack_rgb(params.bg_rgb);
  }
  let src_local_x = min(u32(image_x * f32(region_w) / out_w), region_w - 1u);
  let src_x = min(region_x0 + src_local_x, params.src_width - 1u);
  let src_y = min(u32(image_y * f32(params.src_height) / out_h), params.src_height - 1u);
  let src_idx = src_y * params.src_width + src_x;
  var val = data[src_idx];
  if (params.log_scale == 1u) {
    val = log(1.0 + max(val, 0.0));
  }
  let range = max(params.vmax - params.vmin, 1e-30);
  let clipped = clamp(val, params.vmin, params.vmax);
  let t = (clipped - params.vmin) / range;
  let lut_idx = min(u32(t * 255.0), 255u);
  return unpack_rgb(lut[lut_idx]);
}
`;

// Fullscreen-quad blit shader: reads RGBA u32 buffer, renders to canvas texture
const BLIT_SHADER = /* wgsl */ `
struct BlitParams { width: u32, height: u32 };
@group(0) @binding(0) var<uniform> params: BlitParams;
@group(0) @binding(1) var<storage, read> rgba: array<u32>;

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  // Fullscreen triangle (3 vertices, covers entire clip space)
  var out: VSOut;
  let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
  let y = f32(i32(vi >> 1u)) * 4.0 - 1.0;
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  return out;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let px = min(u32(in.uv.x * f32(params.width)), params.width - 1u);
  let py = min(u32(in.uv.y * f32(params.height)), params.height - 1u);
  let idx = py * params.width + px;
  let packed = rgba[idx];
  let r = f32(packed & 0xFFu) / 255.0;
  let g = f32((packed >> 8u) & 0xFFu) / 255.0;
  let b = f32((packed >> 16u) & 0xFFu) / 255.0;
  return vec4f(r, g, b, 1.0);
}
`;

// Volume-resident orthogonal slice + colormap in ONE compute pass. The whole 3D
// volume lives in a GPU storage buffer (uploaded once); per scrub only a tiny
// uniform (axis + slice index + vmin/vmax) changes, so there is NO per-frame CPU
// slice extraction and NO per-frame volume re-upload. axis: 0=XY(z fixed),
// 1=XZ(y fixed), 2=YZ(x fixed). Order matches the CPU path: log THEN flip.
const VOLUME_SLICE_SHADER = /* wgsl */ `
struct VParams {
  nx: u32, ny: u32, nz: u32, axis: u32,
  index: u32, outW: u32, outH: u32, logScale: u32,
  flip: u32, viewMode: u32, canvasW: u32, canvasH: u32,
  vmin: f32, vmax: f32, zoom: f32, panX: f32,
  panY: f32, _p0: f32, _p1: f32, _p2: f32,
};
@group(0) @binding(0) var<uniform> p: VParams;
@group(0) @binding(1) var<storage, read> vol: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;

fn sampleSlice(p_axis: u32, p_index: u32, nx: u32, ny: u32, sliceX: u32, sliceY: u32) -> f32 {
  var sx: u32; var sy: u32; var sz: u32;
  if (p_axis == 0u) { sx = sliceX; sy = sliceY; sz = p_index; }          // XY
  else if (p_axis == 1u) { sx = sliceX; sy = p_index; sz = sliceY; }     // XZ
  else { sx = p_index; sy = sliceX; sz = sliceY; }                       // YZ
  return vol[sz * ny * nx + sy * nx + sx];
}

fn signedLog1p(v: f32) -> f32 {
  if (v >= 0.0) { return log(1.0 + v); }
  return -log(1.0 - v);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.outW || gid.y >= p.outH) { return; }
  // Full slice dims for this axis (source resolution).
  var fullW: u32; var fullH: u32;
  if (p.axis == 0u) { fullW = p.nx; fullH = p.ny; }        // XY
  else if (p.axis == 1u) { fullW = p.nx; fullH = p.nz; }   // XZ
  else { fullW = p.ny; fullH = p.nz; }                     // YZ
  var x0: u32; var y0: u32; var x1: u32; var y1: u32;
  if (p.viewMode == 0u) {
    // AREA AVERAGE downsample: this output pixel covers the source block
    // [x0,x1) x [y0,y1). Average every covered source value, so no source pixels
    // are silently skipped when a 4k slice is displayed in a smaller panel. When
    // outW==fullW the block is 1x1 = exact native pixel.
    x0 = (gid.x * fullW) / p.outW;
    y0 = (gid.y * fullH) / p.outH;
    x1 = max(x0 + 1u, ((gid.x + 1u) * fullW) / p.outW);
    y1 = max(y0 + 1u, ((gid.y + 1u) * fullH) / p.outH);
  } else {
    let cw = max(f32(p.canvasW), 1.0);
    let ch = max(f32(p.canvasH), 1.0);
    let z = max(p.zoom, 1e-6);
    let cx = cw * 0.5;
    let cy = ch * 0.5;
    let sx0 = (((f32(gid.x) - cx - p.panX) / z) + cx) * f32(fullW) / cw;
    let sy0 = (((f32(gid.y) - cy - p.panY) / z) + cy) * f32(fullH) / ch;
    let sx1 = (((f32(gid.x + 1u) - cx - p.panX) / z) + cx) * f32(fullW) / cw;
    let sy1 = (((f32(gid.y + 1u) - cy - p.panY) / z) + cy) * f32(fullH) / ch;
    let loX = min(sx0, sx1);
    let hiX = max(sx0, sx1);
    let loY = min(sy0, sy1);
    let hiY = max(sy0, sy1);
    if (hiX <= 0.0 || hiY <= 0.0 || loX >= f32(fullW) || loY >= f32(fullH)) {
      rgba[gid.y * p.outW + gid.x] = 0xFF000000u;
      return;
    }
    x0 = u32(clamp(floor(loX), 0.0, f32(fullW - 1u)));
    y0 = u32(clamp(floor(loY), 0.0, f32(fullH - 1u)));
    x1 = max(x0 + 1u, u32(clamp(ceil(hiX), 1.0, f32(fullW))));
    y1 = max(y0 + 1u, u32(clamp(ceil(hiY), 1.0, f32(fullH))));
  }
  var sum = 0.0; var cnt = 0.0;
  var yy = y0;
  loop {
    if (yy >= y1) { break; }
    var xx = x0;
    loop {
      if (xx >= x1) { break; }
      sum = sum + sampleSlice(p.axis, p.index, p.nx, p.ny, min(xx, fullW - 1u), min(yy, fullH - 1u));
      cnt = cnt + 1.0;
      xx = xx + 1u;
    }
    yy = yy + 1u;
  }
  var val = sum / max(cnt, 1.0);
  if (p.logScale == 1u) { val = signedLog1p(val); }
  if (p.flip == 1u) { val = -val; }
  let range = max(p.vmax - p.vmin, 1e-30);
  let t = clamp((val - p.vmin) / range, 0.0, 1.0);
  let li = min(u32(t * 255.0), 255u);
  rgba[gid.y * p.outW + gid.x] = lut[li] | 0xFF000000u;
}
`;

const VOLUME_TEXTURE_SLICE_SHADER = /* wgsl */ `
struct VParams {
  nx: u32, ny: u32, nz: u32, axis: u32,
  index: u32, outW: u32, outH: u32, logScale: u32,
  flip: u32, viewMode: u32, canvasW: u32, canvasH: u32,
  vmin: f32, vmax: f32, zoom: f32, panX: f32,
  panY: f32, _p0: f32, _p1: f32, _p2: f32,
};
@group(0) @binding(0) var<uniform> p: VParams;
@group(0) @binding(1) var volTex: texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;

fn sampleSlice(p_axis: u32, p_index: u32, sliceX: u32, sliceY: u32) -> f32 {
  var sx: u32; var sy: u32; var sz: u32;
  if (p_axis == 0u) { sx = sliceX; sy = sliceY; sz = p_index; }
  else if (p_axis == 1u) { sx = sliceX; sy = p_index; sz = sliceY; }
  else { sx = p_index; sy = sliceX; sz = sliceY; }
  return textureLoad(volTex, vec2<i32>(i32(sx), i32(sy)), i32(sz), 0).r;
}

fn signedLog1p(v: f32) -> f32 {
  if (v >= 0.0) { return log(1.0 + v); }
  return -log(1.0 - v);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.outW || gid.y >= p.outH) { return; }
  var fullW: u32; var fullH: u32;
  if (p.axis == 0u) { fullW = p.nx; fullH = p.ny; }
  else if (p.axis == 1u) { fullW = p.nx; fullH = p.nz; }
  else { fullW = p.ny; fullH = p.nz; }
  var x0: u32; var y0: u32; var x1: u32; var y1: u32;
  if (p.viewMode == 0u) {
    x0 = (gid.x * fullW) / p.outW;
    y0 = (gid.y * fullH) / p.outH;
    x1 = max(x0 + 1u, ((gid.x + 1u) * fullW) / p.outW);
    y1 = max(y0 + 1u, ((gid.y + 1u) * fullH) / p.outH);
  } else {
    let cw = max(f32(p.canvasW), 1.0);
    let ch = max(f32(p.canvasH), 1.0);
    let z = max(p.zoom, 1e-6);
    let cx = cw * 0.5;
    let cy = ch * 0.5;
    let sx0 = (((f32(gid.x) - cx - p.panX) / z) + cx) * f32(fullW) / cw;
    let sy0 = (((f32(gid.y) - cy - p.panY) / z) + cy) * f32(fullH) / ch;
    let sx1 = (((f32(gid.x + 1u) - cx - p.panX) / z) + cx) * f32(fullW) / cw;
    let sy1 = (((f32(gid.y + 1u) - cy - p.panY) / z) + cy) * f32(fullH) / ch;
    let loX = min(sx0, sx1);
    let hiX = max(sx0, sx1);
    let loY = min(sy0, sy1);
    let hiY = max(sy0, sy1);
    if (hiX <= 0.0 || hiY <= 0.0 || loX >= f32(fullW) || loY >= f32(fullH)) {
      rgba[gid.y * p.outW + gid.x] = 0xFF000000u;
      return;
    }
    x0 = u32(clamp(floor(loX), 0.0, f32(fullW - 1u)));
    y0 = u32(clamp(floor(loY), 0.0, f32(fullH - 1u)));
    x1 = max(x0 + 1u, u32(clamp(ceil(hiX), 1.0, f32(fullW))));
    y1 = max(y0 + 1u, u32(clamp(ceil(hiY), 1.0, f32(fullH))));
  }
  var sum = 0.0; var cnt = 0.0;
  var yy = y0;
  loop {
    if (yy >= y1) { break; }
    var xx = x0;
    loop {
      if (xx >= x1) { break; }
      sum = sum + sampleSlice(p.axis, p.index, min(xx, fullW - 1u), min(yy, fullH - 1u));
      cnt = cnt + 1.0;
      xx = xx + 1u;
    }
    yy = yy + 1u;
  }
  var val = sum / max(cnt, 1.0);
  if (p.logScale == 1u) { val = signedLog1p(val); }
  if (p.flip == 1u) { val = -val; }
  let range = max(p.vmax - p.vmin, 1e-30);
  let t = clamp((val - p.vmin) / range, 0.0, 1.0);
  let li = min(u32(t * 255.0), 255u);
  rgba[gid.y * p.outW + gid.x] = lut[li] | 0xFF000000u;
}
`;

const VOLUME_PARAMS_BYTES = 96;

interface VolumeSliceView {
  zoom: number;
  panX: number;
  panY: number;
  canvasW: number;
  canvasH: number;
}

// Tiny per-pass GPU buffers (e.g. 32B region uniforms) that must live until
// the GPU has consumed them. We push them here when recorded into an encoder
// and destroy them once the caller has submitted the work.
const paramsBufQueue: GPUBuffer[] = [];
function flushParamsBufQueue(): void {
  for (const b of paramsBufQueue) b.destroy();
  paramsBufQueue.length = 0;
}

/**
 * GPU-accelerated colormap engine. Holds persistent data buffers on GPU;
 * histogram slider changes only update a small uniform — no data re-upload.
 */
type GPUSlot = {
  dataBuffer: GPUBuffer;
  rgbaBuffer: GPUBuffer;
  readBuffer: GPUBuffer;
  paramsBuffer: GPUBuffer;
  blitParamsBuffer: GPUBuffer;
  histBinsBuffer: GPUBuffer;
  histReadBuffer: GPUBuffer;
  // Lazily allocated per-slot 16-byte buffer holding { vmin, vmax, _p0, _p1 }.
  // Populated by computeRange* on GPU and consumed directly by the range-aware
  // colormap shader (no CPU readback between passes).
  rangeBuffer: GPUBuffer | null;
  directGridBindGroup: GPUBindGroup | null;
  directSlotBindGroup: GPUBindGroup | null;
  directRegionParamsBuffers: (GPUBuffer | null)[];
  directRegionBindGroups: (GPUBindGroup | null)[];
  sharedGridBindGroup: GPUBindGroup | null;
  sharedGridBlitBindGroup: GPUBindGroup | null;
  count: number;
  rgbaCapacity: number;
  width: number;
  height: number;
  directOnly: boolean;
};

export class GPUColormapEngine {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline | null = null;
  private scaledPipeline: GPUComputePipeline | null = null;
  private sharedGridPipeline: GPUComputePipeline | null = null;
  private directGridPipeline: GPURenderPipeline | null = null;
  private directGridRangesPipeline: GPURenderPipeline | null = null;
  private directSlotPipeline: GPURenderPipeline | null = null;
  private blitPipeline: GPURenderPipeline | null = null;
  // Per-image GPU state: persistent buffers (data, rgba, read, params, histogram)
  private slots: GPUSlot[] = [];
  private lutBuffer: GPUBuffer | null = null;
  private currentLutName: string = "";
  private directGridParams = new ArrayBuffer(64);
  private directGridParamsU32 = new Uint32Array(this.directGridParams);
  private directGridParamsF32 = new Float32Array(this.directGridParams);
  private directGridRangesBuffer: GPUBuffer | null = null;
  private directGridRangesCapacity = 0;
  // Volume-resident slice pipeline (Show3DSlices): volume uploaded once, slice +
  // colormap done on GPU per scrub - no per-frame CPU extract / re-upload.
  private volumePipeline: GPUComputePipeline | null = null;
  private volumeTexturePipeline: GPUComputePipeline | null = null;
  private volumeBuffer: GPUBuffer | null = null;
  private volumeTexture: GPUTexture | null = null;
  private volTextureView: GPUTextureView | null = null;
  private volUseTexture = false;
  private volTextureWidth = 0;
  private volNx = 0;
  private volNy = 0;
  private volNz = 0;
  private volCount = 0;
  private volParamsBuffer: GPUBuffer | null = null;
  private volRgbaBuffer: GPUBuffer | null = null;
  private volRgbaCapacity = 0;
  private volParams = new ArrayBuffer(VOLUME_PARAMS_BYTES);
  private volParamsU32 = new Uint32Array(this.volParams);
  private volParamsF32 = new Float32Array(this.volParams);
  private volBlitCanvas: OffscreenCanvas | null = null;
  private volBlitContext: GPUCanvasContext | null = null;
  private volBlitFormat: GPUTextureFormat | null = null;
  private volBlitWidth = 0;
  private volBlitHeight = 0;
  private volBlitParamsBuffer: GPUBuffer | null = null;
  private volBlitParams = new Uint32Array(2);
  private volBlitBindGroup: GPUBindGroup | null = null;
  private volComputeBindGroup: GPUBindGroup | null = null;
  private volTextureBindGroup: GPUBindGroup | null = null;

  constructor(device: GPUDevice) { this.device = device; }

  private destroySlot(slot: GPUSlot): void {
    slot.dataBuffer.destroy();
    slot.rgbaBuffer.destroy();
    slot.readBuffer.destroy();
    slot.paramsBuffer.destroy();
    slot.blitParamsBuffer.destroy();
    slot.histBinsBuffer.destroy();
    slot.histReadBuffer.destroy();
    slot.rangeBuffer?.destroy();
    for (const buf of slot.directRegionParamsBuffers) buf?.destroy();
  }

  private ensurePipeline(): void {
    if (this.pipeline) return;
    const module = this.device.createShaderModule({ code: COLORMAP_SHADER });
    this.pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  private ensureScaledPipeline(): void {
    if (this.scaledPipeline) return;
    const module = this.device.createShaderModule({ code: SCALED_COLORMAP_SHADER });
    this.scaledPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  private ensureSharedGridPipeline(): void {
    if (this.sharedGridPipeline) return;
    const module = this.device.createShaderModule({ code: SHARED_GRID_COLORMAP_SHADER });
    this.sharedGridPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  private ensureDirectGridPipeline(format: GPUTextureFormat): void {
    if (this.directGridPipeline) return;
    const module = this.device.createShaderModule({ code: DIRECT_GRID_COLORMAP_SHADER });
    this.directGridPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: {
        module,
        entryPoint: "fs",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  private ensureDirectGridRangesPipeline(format: GPUTextureFormat): void {
    if (this.directGridRangesPipeline) return;
    const module = this.device.createShaderModule({ code: DIRECT_GRID_RANGES_COLORMAP_SHADER });
    this.directGridRangesPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: {
        module,
        entryPoint: "fs",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  private ensureDirectSlotPipeline(format: GPUTextureFormat): void {
    if (this.directSlotPipeline) return;
    const module = this.device.createShaderModule({ code: DIRECT_SLOT_COLORMAP_SHADER });
    this.directSlotPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: {
        module,
        entryPoint: "fs",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  /** Upload LUT to GPU (only when colormap name changes). */
  uploadLUT(lutName: string, lut: Uint8Array): void {
    if (this.currentLutName === lutName && this.lutBuffer) return;
    this.ensurePipeline();
    if (this.lutBuffer) {
      this.lutBuffer.destroy();
      this.volComputeBindGroup = null;
      this.volTextureBindGroup = null;
      for (const slot of this.slots) {
        if (!slot) continue;
        slot.directGridBindGroup = null;
        slot.directSlotBindGroup = null;
        slot.directRegionBindGroups = slot.directRegionBindGroups.map(() => null);
        slot.sharedGridBindGroup = null;
      }
    }
    // Pack RGB triplets into u32 for GPU (R in low bits)
    const packed = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      packed[i] = lut[i * 3] | (lut[i * 3 + 1] << 8) | (lut[i * 3 + 2] << 16);
    }
    this.lutBuffer = this.device.createBuffer({
      size: packed.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.lutBuffer, 0, packed);
    this.currentLutName = lutName;
  }


  /** Upload float32 image data for slot `idx`. Only call when data changes. */
  uploadData(idx: number, data: Float32Array, width?: number, height?: number, rgbaCapacityHint?: number, directOnly: boolean = false): void {
    this.ensurePipeline();
    while (this.slots.length <= idx) this.slots.push(null as never);
    // Validate dimensions — if width*height doesn't match data length, derive from sqrt
    // (catches stale closure values like width=1 from mount effects)
    const validDims = width && height && width > 1 && height > 1 && width * height === data.length;
    const w = validDims ? width : Math.round(Math.sqrt(data.length));
    const h = validDims ? height : Math.round(data.length / w);
    const byteSize = data.byteLength;
    const rgbaCapacity = directOnly ? 1 : Math.max(1, Math.round(rgbaCapacityHint ?? data.length));
    const rgbaSize = rgbaCapacity * 4;
    const existing = this.slots[idx];
    if (existing && existing.directOnly === directOnly && existing.count === data.length && existing.width === w && existing.height === h && existing.rgbaCapacity >= rgbaCapacity) {
      this.device.queue.writeBuffer(existing.dataBuffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
      return;
    }
    if (existing) {
      this.destroySlot(existing);
    }
    const dataBuffer = this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(dataBuffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    const rgbaBuffer = this.device.createBuffer({
      size: rgbaSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // Persistent read buffer — reused on every applySlots call (no create/destroy overhead)
    const readBuffer = this.device.createBuffer({
      size: rgbaSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    // Persistent params buffer — reused (just writeBuffer on each call).
    // Size 64 covers the 24-byte colormap/histogram structs, 32-byte scaled
    // colormap structs, and 64-byte direct grid colormap struct.
    const paramsBuffer = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const blitParamsBuffer = this.device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Persistent histogram buffers (256 bins × 4 bytes = 1KB each)
    const histBinsBuffer = this.device.createBuffer({
      size: 256 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const histReadBuffer = this.device.createBuffer({
      size: 256 * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.slots[idx] = {
      dataBuffer,
      rgbaBuffer,
      readBuffer,
      paramsBuffer,
      blitParamsBuffer,
      histBinsBuffer,
      histReadBuffer,
      rangeBuffer: null,
      directGridBindGroup: null,
      directSlotBindGroup: null,
      directRegionParamsBuffers: [],
      directRegionBindGroups: [],
      sharedGridBindGroup: null,
      sharedGridBlitBindGroup: null,
      count: data.length,
      rgbaCapacity,
      width: w,
      height: h,
      directOnly,
    };
  }

  // Params buffer: 24 bytes = { width: u32, height: u32, vmin: f32, vmax: f32, log_scale: u32, _pad: u32 }
  private _writeParams(buf: ArrayBuffer, width: number, height: number, vmin: number, vmax: number, logScale: boolean): void {
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    u[0] = width;
    u[1] = height;
    f[2] = vmin;
    f[3] = vmax;
    u[4] = logScale ? 1 : 0;
    u[5] = 0; // pad
  }

  /**
   * Apply colormap to specific slot indices with per-image vmin/vmax.
   * Uses persistent per-slot read buffers (no create/destroy overhead).
   * Log scale is applied on GPU per pixel.
   */
  async applySlots(
    indices: number[],
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean = false,
  ): Promise<{ idx: number; rgba: Uint8ClampedArray }[]> {
    if (!this.pipeline || !this.lutBuffer || indices.length === 0) return [];

    const activeSlots: { idx: number; slot: GPUSlot; count: number }[] = [];
    const encoder = this.device.createCommandEncoder();
    const params = new ArrayBuffer(24);

    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      const slot = this.slots[i];
      if (!slot || slot.directOnly || slot.rgbaCapacity < slot.count) continue;
      const range = ranges[k] || { vmin: 0, vmax: 1 };

      // Reuse persistent paramsBuffer — just write new values
      this._writeParams(params, slot.width, slot.height, range.vmin, range.vmax, logScale);
      this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.lutBuffer } },
          { binding: 3, resource: { buffer: slot.rgbaBuffer } },
        ],
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
      pass.end();

      // Copy to persistent read buffer
      encoder.copyBufferToBuffer(slot.rgbaBuffer, 0, slot.readBuffer, 0, slot.count * 4);
      activeSlots.push({ idx: i, slot, count: slot.count });
    }
    this.device.queue.submit([encoder.finish()]);
    await Promise.all(activeSlots.map(s => s.slot.readBuffer.mapAsync(GPUMapMode.READ)));

    const results: { idx: number; rgba: Uint8ClampedArray }[] = [];
    for (const s of activeSlots) {
      const mapped = s.slot.readBuffer.getMappedRange();
      const rgba = new Uint8ClampedArray(s.count * 4);
      rgba.set(new Uint8ClampedArray(mapped));
      s.slot.readBuffer.unmap();
      results.push({ idx: s.idx, rgba });
    }

    // applySlots is for callers that need raw RGBA arrays (not rendering to canvas)
    // For rendering, use renderSlots which avoids the intermediate copy
    return results;
  }

  /** Apply colormap to ALL slots with shared vmin/vmax. */
  async apply(vmin: number, vmax: number, logScale: boolean = false): Promise<Uint8ClampedArray[]> {
    const indices = this.slots.map((_, i) => i).filter(i => this.slots[i]);
    const ranges = indices.map(() => ({ vmin, vmax }));
    const results = await this.applySlots(indices, ranges, logScale);
    // Return in slot order
    const out: Uint8ClampedArray[] = [];
    for (const r of results) out[r.idx] = r.rgba;
    return out.filter(x => x);
  }

  /** Apply colormap with per-image vmin/vmax. */
  async applyPerImage(ranges: { vmin: number; vmax: number }[], logScale: boolean = false): Promise<Uint8ClampedArray[]> {
    const indices = this.slots.map((_, i) => i).filter(i => this.slots[i]);
    const perSlotRanges = indices.map(i => ranges[i] || { vmin: 0, vmax: 1 });
    const results = await this.applySlots(indices, perSlotRanges, logScale);
    const out: Uint8ClampedArray[] = [];
    for (const r of results) out[r.idx] = r.rgba;
    return out.filter(x => x);
  }

  /** Apply colormap to a SINGLE slot (fast path for slider drag). */
  async applySingle(idx: number, vmin: number, vmax: number, logScale: boolean = false): Promise<Uint8ClampedArray | null> {
    const results = await this.applySlots([idx], [{ vmin, vmax }], logScale);
    return results.length > 0 ? results[0].rgba : null;
  }

  /**
   * GPU colormap → offscreen canvas in one pass (zero intermediate allocation).
   * Writes from GPU mapped memory directly into ImageData, then putImageData.
   * Eliminates the 768MB temp Uint8ClampedArray that applySlots allocates.
   */
  async renderSlots(
    indices: number[],
    ranges: { vmin: number; vmax: number }[],
    offscreens: (HTMLCanvasElement | null)[],
    imgDatas: (ImageData | null)[],
    logScale: boolean = false,
  ): Promise<number> {
    if (!this.pipeline || !this.lutBuffer || indices.length === 0) return 0;

    const activeSlots: { k: number; idx: number; slot: GPUSlot }[] = [];
    const encoder = this.device.createCommandEncoder();
    const params = new ArrayBuffer(24);

    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      const slot = this.slots[i];
      if (!slot || slot.directOnly || slot.rgbaCapacity < slot.count || !offscreens[k] || !imgDatas[k]) continue;
      const range = ranges[k] || { vmin: 0, vmax: 1 };

      this._writeParams(params, slot.width, slot.height, range.vmin, range.vmax, logScale);
      this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.lutBuffer } },
          { binding: 3, resource: { buffer: slot.rgbaBuffer } },
        ],
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
      pass.end();
      encoder.copyBufferToBuffer(slot.rgbaBuffer, 0, slot.readBuffer, 0, slot.count * 4);
      activeSlots.push({ k, idx: i, slot });
    }
    this.device.queue.submit([encoder.finish()]);
    await Promise.all(activeSlots.map(s => s.slot.readBuffer.mapAsync(GPUMapMode.READ)));

    // Write directly from GPU mapped memory → ImageData → offscreen canvas
    let rendered = 0;
    for (const s of activeSlots) {
      const mapped = s.slot.readBuffer.getMappedRange();
      const imgData = imgDatas[s.k]!;
      imgData.data.set(new Uint8ClampedArray(mapped));
      s.slot.readBuffer.unmap();
      offscreens[s.k]!.getContext("2d")!.putImageData(imgData, 0, 0);
      rendered++;
    }
    return rendered;
  }

  private ensureBlitPipeline(format: GPUTextureFormat): void {
    if (this.blitPipeline) return;
    const module = this.device.createShaderModule({ code: BLIT_SHADER });
    this.blitPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: {
        module, entryPoint: "fs",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  /**
   * Zero-copy GPU render: compute colormap + blit directly to WebGPU canvas textures.
   * No mapAsync, no CPU copy, no putImageData. Target: <16ms for 60fps.
   *
   * Each canvas must have a 'webgpu' context (not '2d'). Call configureCanvas() first.
   * Returns the number of images rendered.
   */
  renderSlotsZeroCopy(
    indices: number[],
    ranges: { vmin: number; vmax: number }[],
    contexts: (GPUCanvasContext | null)[],
    logScale: boolean = false,
  ): number {
    if (!this.pipeline || !this.lutBuffer || indices.length === 0) return 0;

    // Get texture format from first valid context
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ensureBlitPipeline(fmt);
    if (!this.blitPipeline) return 0;

    const encoder = this.device.createCommandEncoder();
    const params = new ArrayBuffer(24);
    let rendered = 0;
    const tempBuffers: GPUBuffer[] = [];

    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      const slot = this.slots[i];
      const ctx = contexts[k];
      if (!slot || slot.directOnly || slot.rgbaCapacity < slot.count || !ctx) continue;
      const range = ranges[k] || { vmin: 0, vmax: 1 };

      // 1. Compute colormap (same as renderSlots)
      this._writeParams(params, slot.width, slot.height, range.vmin, range.vmax, logScale);
      this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

      const computeGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.lutBuffer } },
          { binding: 3, resource: { buffer: slot.rgbaBuffer } },
        ],
      });
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(this.pipeline);
      computePass.setBindGroup(0, computeGroup);
      computePass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
      computePass.end();

      // 2. Blit RGBA buffer → canvas texture (zero-copy render pass)
      const blitParamsBuffer = this.device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(blitParamsBuffer, 0, new Uint32Array([slot.width, slot.height]));

      const blitGroup = this.device.createBindGroup({
        layout: this.blitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: blitParamsBuffer } },
          { binding: 1, resource: { buffer: slot.rgbaBuffer } },
        ],
      });

      const texture = ctx.getCurrentTexture();
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: texture.createView(),
          loadOp: "clear" as GPULoadOp,
          storeOp: "store" as GPUStoreOp,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      renderPass.setPipeline(this.blitPipeline);
      renderPass.setBindGroup(0, blitGroup);
      renderPass.draw(3); // fullscreen triangle
      renderPass.end();
      rendered++;

      // The 8-byte uniform is finished referencing once the encoder is closed;
      // we destroy after submit to avoid a per-frame leak (was previously accumulating).
      tempBuffers.push(blitParamsBuffer);
    }

    this.device.queue.submit([encoder.finish()]);
    for (const b of tempBuffers) b.destroy();
    return rendered;
  }

  /**
   * GPU colormap → OffscreenCanvas → ImageBitmap (zero mapAsync).
   * Compute shader writes RGBA, render pass blits to OffscreenCanvas texture,
   * transferToImageBitmap() returns ImageBitmap for drawImage on 2D canvas.
   * Eliminates the 35ms JS memcpy for 12×4K images.
   */
  renderSlotsToImageBitmap(
    indices: number[],
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean = false,
  ): ImageBitmap[] | null {
    if (!this.pipeline || !this.lutBuffer || indices.length === 0) return null;
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ensureBlitPipeline(fmt);
    if (!this.blitPipeline) return null;

    const encoder = this.device.createCommandEncoder();
    const params = new ArrayBuffer(24);
    const canvases: OffscreenCanvas[] = [];
    const tempBuffers: GPUBuffer[] = [];

    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      const slot = this.slots[i];
      if (!slot || slot.directOnly || slot.rgbaCapacity < slot.count) { canvases.push(null as never); continue; }
      const range = ranges[k] || { vmin: 0, vmax: 1 };

      // Compute colormap
      this._writeParams(params, slot.width, slot.height, range.vmin, range.vmax, logScale);
      this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

      const computeGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.lutBuffer } },
          { binding: 3, resource: { buffer: slot.rgbaBuffer } },
        ],
      });
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(this.pipeline);
      computePass.setBindGroup(0, computeGroup);
      computePass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
      computePass.end();

      // Blit to OffscreenCanvas
      const oc = new OffscreenCanvas(slot.width, slot.height);
      const ctx = oc.getContext("webgpu") as GPUCanvasContext;
      ctx.configure({ device: this.device, format: fmt, alphaMode: "opaque" });

      const blitParamsBuffer = this.device.createBuffer({
        size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(blitParamsBuffer, 0, new Uint32Array([slot.width, slot.height]));
      tempBuffers.push(blitParamsBuffer);

      const blitGroup = this.device.createBindGroup({
        layout: this.blitPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: blitParamsBuffer } },
          { binding: 1, resource: { buffer: slot.rgbaBuffer } },
        ],
      });

      const texture = ctx.getCurrentTexture();
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: texture.createView(),
          loadOp: "clear" as GPULoadOp,
          storeOp: "store" as GPUStoreOp,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      renderPass.setPipeline(this.blitPipeline!);
      renderPass.setBindGroup(0, blitGroup);
      renderPass.draw(3);
      renderPass.end();
      canvases.push(oc);
    }

    this.device.queue.submit([encoder.finish()]);
    for (const b of tempBuffers) b.destroy();

    // transferToImageBitmap after GPU finishes (synchronous, no mapAsync)
    const bitmaps: ImageBitmap[] = [];
    for (const oc of canvases) {
      if (oc) bitmaps.push(oc.transferToImageBitmap());
      else bitmaps.push(null as never);
    }
    return bitmaps;
  }

  private ensureVolumePipeline(): void {
    if (this.volumePipeline) return;
    const module = this.device.createShaderModule({ code: VOLUME_SLICE_SHADER });
    this.volumePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    if (!this.volParamsBuffer) {
      this.volParamsBuffer = this.device.createBuffer({
        size: VOLUME_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
  }

  private ensureVolumeTexturePipeline(): void {
    if (this.volumeTexturePipeline) return;
    const module = this.device.createShaderModule({ code: VOLUME_TEXTURE_SLICE_SHADER });
    this.volumeTexturePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    if (!this.volParamsBuffer) {
      this.volParamsBuffer = this.device.createBuffer({
        size: VOLUME_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
  }

  /**
   * Upload a 3D volume (nz, ny, nx) row-major float32 into GPU memory once.
   * 4k and row-padded large stacks use a 2D texture array to avoid
   * storage-buffer binding limits. The texture path keeps original float32
   * values; unaligned rows are padded only in the upload stride and never
   * sampled. Other shapes use the storage-buffer path.
   */
  uploadVolume(vol: Float32Array, nx: number, ny: number, nz: number): boolean {
    const rowBytes = nx * 4;
    const paddedRowBytes = Math.ceil(rowBytes / 256) * 256;
    const textureWidth = paddedRowBytes / 4;
    const canTexture = textureWidth <= this.device.limits.maxTextureDimension2D &&
      ny <= this.device.limits.maxTextureDimension2D &&
      nz <= this.device.limits.maxTextureArrayLayers;
    if (canTexture) {
      try {
        this.ensureVolumeTexturePipeline();
        const needsTexture = !this.volumeTexture || this.volCount !== vol.length ||
          this.volNx !== nx || this.volNy !== ny || this.volNz !== nz ||
          this.volTextureWidth !== textureWidth;
        if (needsTexture) {
          this.volumeTexture?.destroy();
          this.volumeTexture = this.device.createTexture({
            size: { width: textureWidth, height: ny, depthOrArrayLayers: nz },
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
          });
          this.volTextureView = this.volumeTexture.createView({ dimension: "2d-array" });
          this.volTextureBindGroup = null;
          this.volCount = vol.length;
        }
        const texture = this.volumeTexture;
        if (!texture) return false;
        if (paddedRowBytes === rowBytes) {
          this.device.queue.writeTexture(
            { texture },
            vol.buffer as ArrayBuffer,
            { offset: vol.byteOffset, bytesPerRow: rowBytes, rowsPerImage: ny },
            { width: nx, height: ny, depthOrArrayLayers: nz },
          );
        } else {
          const layer = new Float32Array(textureWidth * ny);
          const sliceStride = nx * ny;
          for (let z = 0; z < nz; z++) {
            const srcZ = z * sliceStride;
            for (let y = 0; y < ny; y++) {
              const src = srcZ + y * nx;
              layer.set(vol.subarray(src, src + nx), y * textureWidth);
            }
            this.device.queue.writeTexture(
              { texture, origin: { x: 0, y: 0, z } },
              layer.buffer,
              { bytesPerRow: paddedRowBytes, rowsPerImage: ny },
              { width: nx, height: ny, depthOrArrayLayers: 1 },
            );
          }
        }
        this.volUseTexture = true;
        this.volumeBuffer?.destroy();
        this.volumeBuffer = null;
        this.volComputeBindGroup = null;
        this.volTextureWidth = textureWidth;
        this.volNx = nx; this.volNy = ny; this.volNz = nz;
        return true;
      } catch {
        this.volumeTexture?.destroy();
        this.volumeTexture = null;
        this.volTextureView = null;
        this.volTextureBindGroup = null;
        this.volUseTexture = false;
        this.volTextureWidth = 0;
      }
    }
    this.ensureVolumePipeline();
    this.volumeTexture?.destroy();
    this.volumeTexture = null;
    this.volTextureView = null;
    this.volTextureBindGroup = null;
    this.volTextureWidth = 0;
    const maxBind = this.device.limits.maxStorageBufferBindingSize;
    if (vol.byteLength > maxBind) return false;
    if (!this.volumeBuffer || this.volCount !== vol.length) {
      this.volumeBuffer?.destroy();
      this.volumeBuffer = this.device.createBuffer({
        size: vol.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.volCount = vol.length;
      this.volComputeBindGroup = null;
    }
    this.device.queue.writeBuffer(this.volumeBuffer, 0, vol.buffer as ArrayBuffer, vol.byteOffset, vol.byteLength);
    this.volUseTexture = false;
    this.volNx = nx; this.volNy = ny; this.volNz = nz;
    return true;
  }

  /**
   * Slice the resident volume along `axis` (0=XY, 1=XZ, 2=YZ) at `index`,
   * colormap with the current LUT + vmin/vmax (logScale/flip applied in-shader to
   * match the CPU path), and blit to an ImageBitmap. Returns null if the volume
   * isn't uploaded or the LUT/pipeline isn't ready (caller falls back to CPU).
   */
  renderVolumeSliceToImageBitmap(
    axis: number, index: number,
    range: { vmin: number; vmax: number },
    logScale: boolean, flip: boolean,
    maxOut?: number,
    view?: VolumeSliceView,
  ): ImageBitmap | null {
    const texturePipeline = this.volUseTexture ? this.volumeTexturePipeline : null;
    const textureView = this.volUseTexture ? this.volTextureView : null;
    const useTexture = texturePipeline != null && textureView != null;
    const bufferPipeline = this.volumePipeline;
    const useBuffer = !useTexture && this.volumeBuffer != null && bufferPipeline != null;
    if ((!useTexture && !useBuffer) || !this.lutBuffer || !this.volParamsBuffer) return null;
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ensureBlitPipeline(fmt);
    if (!this.blitPipeline) return null;
    const nx = this.volNx, ny = this.volNy, nz = this.volNz;
    const fullW = axis === 0 ? nx : axis === 1 ? nx : ny;
    const fullH = axis === 0 ? ny : nz;
    // If maxOut is supplied, cap the output raster while still sampling from
    // the full-resolution source. Callers that need native-pixel zoom leave it
    // undefined, so outW/outH stay at the full slice dimensions.
    const cap = maxOut && maxOut > 0 ? maxOut : Math.max(fullW, fullH);
    const scale = Math.min(1, cap / Math.max(fullW, fullH));
    const outW = view ? Math.max(1, Math.round(view.canvasW)) : Math.max(1, Math.round(fullW * scale));
    const outH = view ? Math.max(1, Math.round(view.canvasH)) : Math.max(1, Math.round(fullH * scale));
    const idx = Math.max(0, Math.min((axis === 2 ? nx : axis === 1 ? ny : nz) - 1, Math.round(index)));
    const rgbaCount = outW * outH;
    if (!this.volRgbaBuffer || this.volRgbaCapacity < rgbaCount) {
      this.volRgbaBuffer?.destroy();
      this.volRgbaBuffer = this.device.createBuffer({
        size: rgbaCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.volRgbaCapacity = rgbaCount;
      this.volBlitBindGroup = null;
      this.volComputeBindGroup = null;
      this.volTextureBindGroup = null;
    }
    const paramsBuffer = this.volParamsBuffer;
    const lutBuffer = this.lutBuffer;
    const rgbaBuffer = this.volRgbaBuffer;
    if (!paramsBuffer || !lutBuffer || !rgbaBuffer) return null;
    // VParams: u32 control block + float contrast / viewport block.
    const vp = this.volParams;
    const u = this.volParamsU32; const f = this.volParamsF32;
    u[0] = nx; u[1] = ny; u[2] = nz; u[3] = axis;
    u[4] = idx; u[5] = outW; u[6] = outH; u[7] = logScale ? 1 : 0;
    u[8] = flip ? 1 : 0;
    u[9] = view ? 1 : 0;
    u[10] = view ? Math.max(1, Math.round(view.canvasW)) : outW;
    u[11] = view ? Math.max(1, Math.round(view.canvasH)) : outH;
    f[12] = range.vmin; f[13] = range.vmax;
    f[14] = view ? Math.max(1e-6, view.zoom) : 1;
    f[15] = view ? view.panX : 0;
    f[16] = view ? view.panY : 0;
    this.device.queue.writeBuffer(paramsBuffer, 0, vp);
    const encoder = this.device.createCommandEncoder();
    if (useTexture) {
      if (!this.volTextureBindGroup) {
        this.volTextureBindGroup = this.device.createBindGroup({
          layout: texturePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: textureView },
            { binding: 2, resource: { buffer: lutBuffer } },
            { binding: 3, resource: { buffer: rgbaBuffer } },
          ],
        });
      }
    } else if (!this.volComputeBindGroup && bufferPipeline && this.volumeBuffer) {
      this.volComputeBindGroup = this.device.createBindGroup({
        layout: bufferPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: this.volumeBuffer } },
          { binding: 2, resource: { buffer: lutBuffer } },
          { binding: 3, resource: { buffer: rgbaBuffer } },
        ],
      });
    }
    const cpass = encoder.beginComputePass();
    if (useTexture) {
      cpass.setPipeline(texturePipeline);
      cpass.setBindGroup(0, this.volTextureBindGroup!);
    } else {
      if (!bufferPipeline || !this.volComputeBindGroup) { cpass.end(); return null; }
      cpass.setPipeline(bufferPipeline);
      cpass.setBindGroup(0, this.volComputeBindGroup);
    }
    cpass.dispatchWorkgroups(Math.ceil(outW / 16), Math.ceil(outH / 16));
    cpass.end();
    const sizeChanged = !this.volBlitCanvas || this.volBlitWidth !== outW || this.volBlitHeight !== outH;
    const formatChanged = this.volBlitFormat !== fmt;
    if (sizeChanged || formatChanged || !this.volBlitContext) {
      this.volBlitCanvas = new OffscreenCanvas(outW, outH);
      this.volBlitContext = this.volBlitCanvas.getContext("webgpu") as GPUCanvasContext | null;
      if (!this.volBlitContext) return null;
      this.volBlitContext.configure({ device: this.device, format: fmt, alphaMode: "opaque" });
      this.volBlitWidth = outW;
      this.volBlitHeight = outH;
      this.volBlitFormat = fmt;
    }
    if (!this.volBlitParamsBuffer) {
      this.volBlitParamsBuffer = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    this.volBlitParams[0] = outW;
    this.volBlitParams[1] = outH;
    this.device.queue.writeBuffer(this.volBlitParamsBuffer, 0, this.volBlitParams);
    if (!this.volBlitBindGroup) {
      this.volBlitBindGroup = this.device.createBindGroup({
        layout: this.blitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.volBlitParamsBuffer } },
          { binding: 1, resource: { buffer: rgbaBuffer } },
        ],
      });
    }
    const blitContext = this.volBlitContext;
    const blitCanvas = this.volBlitCanvas;
    if (!blitContext || !blitCanvas) return null;
    const rpass = encoder.beginRenderPass({
      colorAttachments: [{ view: blitContext.getCurrentTexture().createView(), loadOp: "clear" as GPULoadOp, storeOp: "store" as GPUStoreOp, clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    rpass.setPipeline(this.blitPipeline);
    rpass.setBindGroup(0, this.volBlitBindGroup);
    rpass.draw(3);
    rpass.end();
    this.device.queue.submit([encoder.finish()]);
    return blitCanvas.transferToImageBitmap();
  }

  /**
   * GPU colormap one slot, then blit the full-resolution RGBA buffer into a
   * smaller OffscreenCanvas. The fragment shader samples the source buffer by
   * UV, so values stay full-precision through the colormap step while playback
   * avoids creating a 4096x4096 ImageBitmap when the visible canvas is smaller.
   */
  renderSlotScaledToImageBitmap(
    idx: number,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    outW: number,
    outH: number,
  ): ImageBitmap | null {
    if (!this.pipeline || !this.lutBuffer) return null;
    const slot = this.slots[idx];
    if (!slot || slot.directOnly) return null;
    const w = Math.max(1, Math.round(outW));
    const h = Math.max(1, Math.round(outH));
    if (w * h > slot.rgbaCapacity) {
      if (slot.rgbaCapacity < slot.count) return null;
      const bitmaps = this.renderSlotsToImageBitmap([idx], [range], logScale);
      return bitmaps?.[0] ?? null;
    }
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ensureScaledPipeline();
    this.ensureBlitPipeline(fmt);
    if (!this.scaledPipeline || !this.blitPipeline) return null;

    const encoder = this.device.createCommandEncoder();
    const params = new ArrayBuffer(32);

    const pu = new Uint32Array(params);
    const pf = new Float32Array(params);
    pu[0] = slot.width;
    pu[1] = slot.height;
    pu[2] = w;
    pu[3] = h;
    pf[4] = range.vmin;
    pf[5] = range.vmax;
    pu[6] = logScale ? 1 : 0;
    pu[7] = 0;
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

    const computeGroup = this.device.createBindGroup({
      layout: this.scaledPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: this.lutBuffer } },
        { binding: 3, resource: { buffer: slot.rgbaBuffer } },
      ],
    });
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this.scaledPipeline);
    computePass.setBindGroup(0, computeGroup);
    computePass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
    computePass.end();

    const oc = new OffscreenCanvas(w, h);
    const ctx = oc.getContext("webgpu") as GPUCanvasContext | null;
    if (!ctx) return null;
    ctx.configure({ device: this.device, format: fmt, alphaMode: "opaque" });

    const blitParamsBuffer = this.device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(blitParamsBuffer, 0, new Uint32Array([w, h]));

    const blitGroup = this.device.createBindGroup({
      layout: this.blitPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: blitParamsBuffer } },
        { binding: 1, resource: { buffer: slot.rgbaBuffer } },
      ],
    });

    const texture = ctx.getCurrentTexture();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    renderPass.setPipeline(this.blitPipeline);
    renderPass.setBindGroup(0, blitGroup);
    renderPass.draw(3);
    renderPass.end();

    this.device.queue.submit([encoder.finish()]);
    blitParamsBuffer.destroy();
    return oc.transferToImageBitmap();
  }

  renderSharedGridToCanvas(
    idx: number,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    ctx: GPUCanvasContext,
  opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth?: number;
      sharedSource?: boolean;
    },
  ): boolean {
    if (!this.lutBuffer) return false;
    const slot = this.slots[idx];
    if (!slot || slot.directOnly) return false;
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    if (outW * outH > slot.rgbaCapacity) return false;
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ensureSharedGridPipeline();
    this.ensureBlitPipeline(fmt);
    if (!this.sharedGridPipeline || !this.blitPipeline) return false;

    const encoder = this.device.createCommandEncoder();
    const params = this.directGridParams;
    const pu = this.directGridParamsU32;
    const pf = this.directGridParamsF32;
    pu[0] = slot.width;
    pu[1] = slot.height;
    pu[2] = Math.max(1, Math.min(slot.width, Math.round(opts.sourcePanelWidth ?? slot.width)));
    pu[3] = outW;
    pu[4] = outH;
    pu[5] = Math.max(1, Math.round(opts.panelCount));
    pu[6] = Math.max(1, Math.round(opts.cols));
    pu[7] = Math.max(1, Math.round(opts.rows));
    pu[8] = logScale ? 1 : 0;
    pu[9] = opts.bgRgb & 0xFFFFFF;
    pu[10] = opts.sharedSource ? 1 : 0;
    pu[11] = 0;
    pf[12] = range.vmin;
    pf[13] = range.vmax;
    pf[14] = Math.max(0, opts.gap);
    pf[15] = 0;
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

    let computeGroup = slot.sharedGridBindGroup;
    if (!computeGroup) {
      computeGroup = this.device.createBindGroup({
        layout: this.sharedGridPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.lutBuffer } },
          { binding: 3, resource: { buffer: slot.rgbaBuffer } },
        ],
      });
      slot.sharedGridBindGroup = computeGroup;
    }
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this.sharedGridPipeline);
    computePass.setBindGroup(0, computeGroup);
    computePass.dispatchWorkgroups(Math.ceil(outW / 16), Math.ceil(outH / 16));
    computePass.end();

    this.device.queue.writeBuffer(slot.blitParamsBuffer, 0, new Uint32Array([outW, outH]));

    let blitGroup = slot.sharedGridBlitBindGroup;
    if (!blitGroup) {
      blitGroup = this.device.createBindGroup({
        layout: this.blitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.blitParamsBuffer } },
          { binding: 1, resource: { buffer: slot.rgbaBuffer } },
        ],
      });
      slot.sharedGridBlitBindGroup = blitGroup;
    }

    const texture = ctx.getCurrentTexture();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    renderPass.setPipeline(this.blitPipeline);
    renderPass.setBindGroup(0, blitGroup);
    renderPass.draw(3);
    renderPass.end();

    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  renderSharedGridDirectToCanvas(
    idx: number,
    range: { vmin: number; vmax: number },
    logScale: boolean,
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth?: number;
      sharedSource?: boolean;
    },
  ): boolean {
    if (!this.lutBuffer) return false;
    const slot = this.slots[idx];
    if (!slot) return false;
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ensureDirectGridPipeline(fmt);
    const pipeline = this.directGridPipeline;
    if (!pipeline) return false;

    const params = new ArrayBuffer(64);
    const pu = new Uint32Array(params);
    const pf = new Float32Array(params);
    pu[0] = slot.width;
    pu[1] = slot.height;
    pu[2] = Math.max(1, Math.min(slot.width, Math.round(opts.sourcePanelWidth ?? slot.width)));
    pu[3] = outW;
    pu[4] = outH;
    pu[5] = Math.max(1, Math.round(opts.panelCount));
    pu[6] = Math.max(1, Math.round(opts.cols));
    pu[7] = Math.max(1, Math.round(opts.rows));
    pu[8] = logScale ? 1 : 0;
    pu[9] = opts.bgRgb & 0xFFFFFF;
    pu[10] = opts.sharedSource ? 1 : 0;
    pu[11] = 0;
    pf[12] = range.vmin;
    pf[13] = range.vmax;
    pf[14] = Math.max(0, opts.gap);
    pf[15] = 0;
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

    let bindGroup = slot.directGridBindGroup;
    if (!bindGroup) {
      bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.lutBuffer } },
        ],
      });
      slot.directGridBindGroup = bindGroup;
    }

    const texture = ctx.getCurrentTexture();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  renderPanelSlotsDirectToCanvas(
    indices: number[],
    range: { vmin: number; vmax: number } | { vmin: number; vmax: number }[],
    logScale: boolean | boolean[],
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      transforms?: { zoom: number; panX: number; panY: number }[];
    },
  ): boolean {
    if (!this.lutBuffer || indices.length === 0) return false;
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const n = Math.max(1, Math.min(indices.length, Math.round(opts.panelCount)));
    const cols = Math.max(1, Math.round(opts.cols));
    const rows = Math.max(1, Math.round(opts.rows));
    const gap = Math.max(0, opts.gap);
    const panelW = (outW - gap * (cols - 1)) / cols;
    const panelH = (outH - gap * (rows - 1)) / rows;
    if (panelW <= 0 || panelH <= 0) return false;

    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ensureDirectSlotPipeline(fmt);
    const pipeline = this.directSlotPipeline;
    if (!pipeline) return false;

    const params = this.directGridParams;
    const pu = this.directGridParamsU32;
    const pf = this.directGridParamsF32;
    for (let panel = 0; panel < n; panel++) {
      const slot = this.slots[indices[panel]];
      if (!slot) return false;
      const panelRange = Array.isArray(range) ? (range[panel] ?? range[0]) : range;
      const panelLogScale = Array.isArray(logScale) ? !!logScale[panel] : logScale;
      pu[0] = slot.width;
      pu[1] = slot.height;
      pu[2] = 0;
      pu[3] = slot.width;
      pu[4] = Math.max(1, Math.round(panelH));
      pu[5] = Math.max(1, Math.round(panelW));
      pu[6] = 1;
      pu[7] = 1;
      pu[8] = panelLogScale ? 1 : 0;
      pu[9] = opts.bgRgb & 0xFFFFFF;
      const transform = opts.transforms?.[panel];
      pf[10] = Math.max(1e-6, transform?.zoom ?? 1);
      pu[11] = 0;
      pf[12] = panelRange.vmin;
      pf[13] = panelRange.vmax;
      pf[14] = transform?.panX ?? 0;
      pf[15] = transform?.panY ?? 0;
      this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);
      if (!slot.directSlotBindGroup) {
        slot.directSlotBindGroup = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: slot.paramsBuffer } },
            { binding: 1, resource: { buffer: slot.dataBuffer } },
            { binding: 2, resource: { buffer: this.lutBuffer } },
          ],
        });
      }
    }

    const texture = ctx.getCurrentTexture();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: {
          r: ((opts.bgRgb & 0xFF) / 255),
          g: (((opts.bgRgb >> 8) & 0xFF) / 255),
          b: (((opts.bgRgb >> 16) & 0xFF) / 255),
          a: 1,
        },
      }],
    });
    pass.setPipeline(pipeline);
    for (let panel = 0; panel < n; panel++) {
      const slot = this.slots[indices[panel]];
      if (!slot?.directSlotBindGroup) continue;
      const col = panel % cols;
      const row = Math.floor(panel / cols);
      const x = col * (panelW + gap);
      const y = row * (panelH + gap);
      const sx = Math.max(0, Math.floor(x));
      const sy = Math.max(0, Math.floor(y));
      const sw = Math.max(1, Math.ceil(panelW));
      const sh = Math.max(1, Math.ceil(panelH));
      pass.setViewport(x, y, panelW, panelH, 0, 1);
      pass.setScissorRect(sx, sy, Math.min(sw, outW - sx), Math.min(sh, outH - sy));
      pass.setBindGroup(0, slot.directSlotBindGroup);
      pass.draw(3);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  renderCombinedGridRangesDirectToCanvas(
    slotIdx: number,
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean | boolean[],
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth: number;
      sharedSource?: boolean;
    },
  ): boolean {
    if (!this.lutBuffer) return false;
    const slot = this.slots[slotIdx];
    if (!slot || ranges.length === 0) return false;
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const n = Math.max(1, Math.round(opts.panelCount));
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ensureDirectGridRangesPipeline(fmt);
    const pipeline = this.directGridRangesPipeline;
    if (!pipeline) return false;

    const neededRangeBytes = Math.max(1, n) * 16;
    if (!this.directGridRangesBuffer || this.directGridRangesCapacity < neededRangeBytes) {
      this.directGridRangesBuffer?.destroy();
      this.directGridRangesBuffer = this.device.createBuffer({
        size: neededRangeBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.directGridRangesCapacity = neededRangeBytes;
    }
    const packedRanges = new Float32Array(n * 4);
    for (let panel = 0; panel < n; panel++) {
      const panelRange = ranges[panel] ?? ranges[0];
      packedRanges[panel * 4] = panelRange.vmin;
      packedRanges[panel * 4 + 1] = panelRange.vmax;
      packedRanges[panel * 4 + 2] = Array.isArray(logScale) ? (logScale[panel] ? 1 : 0) : (logScale ? 1 : 0);
      packedRanges[panel * 4 + 3] = 0;
    }
    this.device.queue.writeBuffer(this.directGridRangesBuffer, 0, packedRanges);

    const params = this.directGridParams;
    const pu = this.directGridParamsU32;
    const pf = this.directGridParamsF32;
    pu[0] = slot.width;
    pu[1] = slot.height;
    pu[2] = Math.max(1, Math.min(slot.width, Math.round(opts.sourcePanelWidth)));
    pu[3] = outW;
    pu[4] = outH;
    pu[5] = n;
    pu[6] = Math.max(1, Math.round(opts.cols));
    pu[7] = Math.max(1, Math.round(opts.rows));
    pu[8] = 0;
    pu[9] = opts.bgRgb & 0xFFFFFF;
    pu[10] = opts.sharedSource ? 1 : 0;
    pu[11] = 0;
    pf[12] = 0;
    pf[13] = 1;
    pf[14] = Math.max(0, opts.gap);
    pf[15] = 0;
    this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.paramsBuffer } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: this.lutBuffer } },
        { binding: 3, resource: { buffer: this.directGridRangesBuffer } },
      ],
    });

    const texture = ctx.getCurrentTexture();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  renderCombinedPanelRegionsDirectToCanvas(
    slotIdx: number,
    range: { vmin: number; vmax: number } | { vmin: number; vmax: number }[],
    logScale: boolean | boolean[],
    ctx: GPUCanvasContext,
    opts: {
      width: number;
      height: number;
      panelCount: number;
      cols: number;
      rows: number;
      gap: number;
      bgRgb: number;
      sourcePanelWidth: number;
    },
  ): boolean {
    if (!this.lutBuffer) return false;
    const slot = this.slots[slotIdx];
    if (!slot) return false;
    const outW = Math.max(1, Math.round(opts.width));
    const outH = Math.max(1, Math.round(opts.height));
    const n = Math.max(1, Math.round(opts.panelCount));
    const cols = Math.max(1, Math.round(opts.cols));
    const rows = Math.max(1, Math.round(opts.rows));
    const gap = Math.max(0, opts.gap);
    const panelW = (outW - gap * (cols - 1)) / cols;
    const panelH = (outH - gap * (rows - 1)) / rows;
    if (panelW <= 0 || panelH <= 0) return false;

    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ensureDirectSlotPipeline(fmt);
    const pipeline = this.directSlotPipeline;
    if (!pipeline) return false;

    const params = this.directGridParams;
    const pu = this.directGridParamsU32;
    const pf = this.directGridParamsF32;
    const sourcePanelW = Math.max(1, Math.min(slot.width, Math.round(opts.sourcePanelWidth)));
    while (slot.directRegionParamsBuffers.length < n) {
      slot.directRegionParamsBuffers.push(null);
      slot.directRegionBindGroups.push(null);
    }
    for (let panel = 0; panel < n; panel++) {
      let paramsBuffer = slot.directRegionParamsBuffers[panel];
      if (!paramsBuffer) {
        paramsBuffer = this.device.createBuffer({
          size: 64,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        slot.directRegionParamsBuffers[panel] = paramsBuffer;
      }
      const panelRange = Array.isArray(range) ? (range[panel] ?? range[0]) : range;
      const panelLogScale = Array.isArray(logScale) ? !!logScale[panel] : logScale;
      const srcX0 = Math.min(panel * sourcePanelW, Math.max(0, slot.width - 1));
      pu[0] = slot.width;
      pu[1] = slot.height;
      pu[2] = srcX0;
      pu[3] = Math.max(1, Math.min(sourcePanelW, slot.width - srcX0));
      pu[4] = Math.max(1, Math.round(panelH));
      pu[5] = 1;
      pu[6] = 1;
      pu[7] = 1;
      pu[8] = panelLogScale ? 1 : 0;
      pu[9] = opts.bgRgb & 0xFFFFFF;
      pu[10] = 1;
      pu[11] = 0;
      pf[12] = panelRange.vmin;
      pf[13] = panelRange.vmax;
      pf[14] = 0;
      pf[15] = 0;
      this.device.queue.writeBuffer(paramsBuffer, 0, params);
      if (!slot.directRegionBindGroups[panel]) {
        slot.directRegionBindGroups[panel] = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: slot.dataBuffer } },
            { binding: 2, resource: { buffer: this.lutBuffer } },
          ],
        });
      }
    }

    const texture = ctx.getCurrentTexture();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear" as GPULoadOp,
        storeOp: "store" as GPUStoreOp,
        clearValue: {
          r: ((opts.bgRgb & 0xFF) / 255),
          g: (((opts.bgRgb >> 8) & 0xFF) / 255),
          b: (((opts.bgRgb >> 16) & 0xFF) / 255),
          a: 1,
        },
      }],
    });
    pass.setPipeline(pipeline);
    for (let panel = 0; panel < n; panel++) {
      const bindGroup = slot.directRegionBindGroups[panel];
      if (!bindGroup) continue;
      const col = panel % cols;
      const row = Math.floor(panel / cols);
      const x = col * (panelW + gap);
      const y = row * (panelH + gap);
      const sx = Math.max(0, Math.floor(x));
      const sy = Math.max(0, Math.floor(y));
      const sw = Math.max(1, Math.min(Math.ceil(panelW), outW - sx));
      const sh = Math.max(1, Math.min(Math.ceil(panelH), outH - sy));
      pass.setViewport(x, y, panelW, panelH, 0, 1);
      pass.setScissorRect(sx, sy, sw, sh);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  /**
   * Configure a canvas for WebGPU zero-copy rendering.
   * Returns the GPUCanvasContext, or null if WebGPU canvas is not supported.
   */
  configureCanvas(canvas: HTMLCanvasElement, width: number, height: number): GPUCanvasContext | null {
    try {
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("webgpu") as GPUCanvasContext | null;
      if (!ctx) return null;
      ctx.configure({
        device: this.device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        alphaMode: "opaque",
      });
      return ctx;
    } catch {
      return null;
    }
  }

  /** Release all GPU resources. */
  destroy(): void {
    for (const slot of this.slots) {
      if (slot) this.destroySlot(slot);
    }
    this.slots = [];
    this.lutBuffer?.destroy();
    this.lutBuffer = null;
    this.directGridRangesBuffer?.destroy();
    this.directGridRangesBuffer = null;
    this.directGridRangesCapacity = 0;
    this.currentLutName = "";
    for (const v of this.panelRgbaBuffers.values()) { v.rgba.destroy(); v.range.destroy(); }
    this.panelRgbaBuffers.clear();
    this.volumeBuffer?.destroy(); this.volumeBuffer = null;
    this.volumeTexture?.destroy(); this.volumeTexture = null; this.volTextureView = null;
    this.volParamsBuffer?.destroy(); this.volParamsBuffer = null;
    this.volRgbaBuffer?.destroy(); this.volRgbaBuffer = null;
    this.volBlitParamsBuffer?.destroy(); this.volBlitParamsBuffer = null;
    this.volBlitCanvas = null; this.volBlitContext = null; this.volBlitFormat = null;
    this.volBlitBindGroup = null; this.volComputeBindGroup = null; this.volTextureBindGroup = null; this.volBlitWidth = 0; this.volBlitHeight = 0;
    this.volUseTexture = false;
    this.volCount = 0; this.volRgbaCapacity = 0; this.volTextureWidth = 0;
  }

  /** Number of uploaded image slots. */
  get slotCount(): number { return this.slots.filter(s => s).length; }

  /** Resolve once all GPU work submitted so far has completed. */
  async waitForSubmittedWork(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
  }

  // ── GPU min/max reduction ──

  private rangePipeline: GPUComputePipeline | null = null;
  private RANGE_WG_SIZE = 256;

  private ensureRangePipeline(): void {
    if (this.rangePipeline) return;
    // Two-pass parallel reduction: each workgroup reduces a chunk to one min/max pair.
    // Output: array of [min, max] pairs (one per workgroup). JS reduces the partials.
    const code = /* wgsl */ `
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> count: u32;

var<workgroup> sMin: array<f32, 256>;
var<workgroup> sMax: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let i = gid.x;
  if (i < count) {
    sMin[lid.x] = data[i];
    sMax[lid.x] = data[i];
  } else {
    sMin[lid.x] = 3.4028235e+38;
    sMax[lid.x] = -3.4028235e+38;
  }
  workgroupBarrier();

  // Tree reduction in shared memory
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid.x < s) {
      sMin[lid.x] = min(sMin[lid.x], sMin[lid.x + s]);
      sMax[lid.x] = max(sMax[lid.x], sMax[lid.x + s]);
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    out[wid.x * 2u] = sMin[0];
    out[wid.x * 2u + 1u] = sMax[0];
  }
}
`;
    const module = this.device.createShaderModule({ code });
    this.rangePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "reduce" },
    });
  }

  /**
   * Batch-compute min/max for multiple slots on GPU.
   * Returns { min, max } per slot. One GPU submission for all slots.
   */
  async computeRangeBatch(indices: number[]): Promise<{ min: number; max: number }[]> {
    this.ensureRangePipeline();
    if (!this.rangePipeline || indices.length === 0) return [];
    const WG = this.RANGE_WG_SIZE;

    const encoder = this.device.createCommandEncoder();
    const jobs: { idx: number; nGroups: number; outBuf: GPUBuffer; readBuf: GPUBuffer; countBuf: GPUBuffer }[] = [];

    for (const i of indices) {
      const slot = this.slots[i];
      if (!slot) continue;
      const N = slot.count;
      const nGroups = Math.ceil(N / WG);
      const outSize = nGroups * 2 * 4; // 2 floats (min, max) per workgroup
      const outBuf = this.device.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readBuf = this.device.createBuffer({ size: outSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const countBuf = this.device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(countBuf, 0, new Uint32Array([N]));

      const bg = this.device.createBindGroup({
        layout: this.rangePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.dataBuffer } },
          { binding: 1, resource: { buffer: outBuf } },
          { binding: 2, resource: { buffer: countBuf } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.rangePipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(nGroups);
      pass.end();
      encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outSize);
      jobs.push({ idx: i, nGroups, outBuf, readBuf, countBuf });
    }

    this.device.queue.submit([encoder.finish()]);
    await Promise.all(jobs.map(j => j.readBuf.mapAsync(GPUMapMode.READ)));

    const results: { min: number; max: number }[] = [];
    for (const j of jobs) {
      const partials = new Float32Array(j.readBuf.getMappedRange().slice(0));
      j.readBuf.unmap();
      j.outBuf.destroy(); j.readBuf.destroy(); j.countBuf.destroy();
      // JS reduces partials: ~65K elements for 16M data = trivial
      let dmin = Infinity, dmax = -Infinity;
      for (let k = 0; k < j.nGroups; k++) {
        if (partials[k * 2] < dmin) dmin = partials[k * 2];
        if (partials[k * 2 + 1] > dmax) dmax = partials[k * 2 + 1];
      }
      results.push({ min: dmin, max: dmax });
    }
    return results;
  }

  // ── GPU region min/max → range-aware colormap (no CPU readback) ──
  //
  // Used by Show3D per-panel contrast: each panel is a sub-region of one full
  // frame buffer. We avoid the JS slab-extract + findDataRange loop entirely
  // by reducing on GPU and feeding the result straight into the colormap pass
  // via a small storage buffer (no mapAsync between the two passes).

  private rangeRegionPipeline: GPUComputePipeline | null = null;
  private colormapRangePipeline: GPUComputePipeline | null = null;
  // Per-panel scratch state for `renderPerPanelGpu` when N panels share ONE
  // GPU slot (full frame). Each entry holds the panel-sized rgba output
  // buffer and the 16-byte range buffer. Keyed by panel index.
  private panelRgbaBuffers: Map<number, { rgba: GPUBuffer; range: GPUBuffer; size: number }> = new Map();

  private ensurePanelScratch(panel: number, panelPixels: number): { rgba: GPUBuffer; range: GPUBuffer } {
    const want = panelPixels * 4;
    const existing = this.panelRgbaBuffers.get(panel);
    if (existing && existing.size === want) return existing;
    if (existing) { existing.rgba.destroy(); existing.range.destroy(); }
    const rgba = this.device.createBuffer({
      size: want,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const range = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const entry = { rgba, range, size: want };
    this.panelRgbaBuffers.set(panel, entry);
    return entry;
  }

  private ensureRangeRegionPipeline(): void {
    if (this.rangeRegionPipeline) return;
    // Single-workgroup grid-stride reduction over a rectangular region of a
    // larger frame buffer. region = (x_offset, y_offset, width, height).
    // fullWidth is the stride of the underlying data buffer.
    const code = /* wgsl */ `
struct RangeOut { vmin: f32, vmax: f32, _p0: f32, _p1: f32 };
struct RegionParams { region: vec4u, fullWidth: u32, _pad0: u32, _pad1: u32, _pad2: u32 };

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<uniform> params: RegionParams;
@group(0) @binding(2) var<storage, read_write> out: RangeOut;

var<workgroup> sMin: array<f32, 256>;
var<workgroup> sMax: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_index) lid: u32) {
  var lmin = 3.4028235e+38;
  var lmax = -3.4028235e+38;
  let rw = params.region.z;
  let rh = params.region.w;
  let n = rw * rh;
  var i = lid;
  loop {
    if (i >= n) { break; }
    let r = i / rw;
    let c = i - r * rw;
    let v = data[(params.region.y + r) * params.fullWidth + params.region.x + c];
    if (v < lmin) { lmin = v; }
    if (v > lmax) { lmax = v; }
    i = i + 256u;
  }
  sMin[lid] = lmin;
  sMax[lid] = lmax;
  workgroupBarrier();
  var s = 128u;
  loop {
    if (s == 0u) { break; }
    if (lid < s) {
      sMin[lid] = min(sMin[lid], sMin[lid + s]);
      sMax[lid] = max(sMax[lid], sMax[lid + s]);
    }
    workgroupBarrier();
    s = s >> 1u;
  }
  if (lid == 0u) {
    out.vmin = sMin[0];
    out.vmax = sMax[0];
    out._p0 = 0.0;
    out._p1 = 0.0;
  }
}
`;
    const module = this.device.createShaderModule({ code });
    this.rangeRegionPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "reduce" },
    });
  }

  private ensureColormapRangePipeline(): void {
    if (this.colormapRangePipeline) return;
    // Same as COLORMAP_SHADER but reads vmin/vmax from a storage buffer
    // (filled by computeRangeRegion) and applies the user slider percentages
    // on GPU so we never round-trip back through JS for those scalars.
    // Also accepts a region (offset + size into the full data buffer) and a
    // stride so the colormap output is the panel sub-image, sourced from
    // the full frame in-place — no slab extraction in JS.
    const code = /* wgsl */ `
struct Params {
  width: u32,        // output (panel) width
  height: u32,       // output (panel) height
  vmin_pct: f32,
  vmax_pct: f32,
  log_scale: u32,
  src_x: u32,        // region offset x in source data
  src_y: u32,        // region offset y in source data
  src_stride: u32,   // row stride of source data
};
struct RangeOut { vmin: f32, vmax: f32, _p0: f32, _p1: f32 };

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<u32>;
@group(0) @binding(3) var<storage, read_write> rgba: array<u32>;
@group(0) @binding(4) var<storage, read> range_in: RangeOut;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let out_idx = gid.y * params.width + gid.x;
  let src_idx = (params.src_y + gid.y) * params.src_stride + (params.src_x + gid.x);
  var val = data[src_idx];
  if (params.log_scale == 1u) {
    val = log(1.0 + max(val, 0.0));
  }
  let span = range_in.vmax - range_in.vmin;
  let vmin = range_in.vmin + span * (params.vmin_pct / 100.0);
  let vmax = range_in.vmin + span * (params.vmax_pct / 100.0);
  let range = max(vmax - vmin, 1e-30);
  let clipped = clamp(val, vmin, vmax);
  let t = (clipped - vmin) / range;
  let lutIdx = min(u32(t * 255.0), 255u);
  let rgb = lut[lutIdx];
  rgba[out_idx] = rgb | 0xFF000000u;
}
`;
    const module = this.device.createShaderModule({ code });
    this.colormapRangePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  private ensureSlotRangeBuffer(slot: GPUSlot): GPUBuffer {
    if (!slot.rangeBuffer) {
      slot.rangeBuffer = this.device.createBuffer({
        // 4 floats: vmin, vmax, _p0, _p1
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    }
    return slot.rangeBuffer;
  }

  /**
   * Reduce a rectangular region of slot `idx`'s data buffer to (vmin, vmax)
   * on GPU and stash the result in `slot.rangeBuffer`. Caller chains a
   * `renderSlotsWithGpuRange` pass that reads it directly — no CPU sync.
   *
   * `region` is { x, y, width, height } in pixels into the slot's full frame
   * (which has stride `slot.width`). Omit `region` to scan the whole slot.
   *
   * Records into the supplied encoder so callers can fuse multiple panels
   * into a single submit.
   */
  recordComputeRangeRegion(
    encoder: GPUCommandEncoder,
    idx: number,
    region?: { x: number; y: number; width: number; height: number },
  ): boolean {
    this.ensureRangeRegionPipeline();
    const slot = this.slots[idx];
    if (!slot || !this.rangeRegionPipeline) return false;
    const r = region ?? { x: 0, y: 0, width: slot.width, height: slot.height };
    const rangeBuf = this.ensureSlotRangeBuffer(slot);

    // Region params: 32 bytes = vec4u + 4xu32 (we only use first u32 of the tail)
    const paramsBuf = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(
      paramsBuf, 0,
      new Uint32Array([r.x, r.y, r.width, r.height, slot.width, 0, 0, 0]),
    );

    const bg = this.device.createBindGroup({
      layout: this.rangeRegionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: slot.dataBuffer } },
        { binding: 1, resource: { buffer: paramsBuf } },
        { binding: 2, resource: { buffer: rangeBuf } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.rangeRegionPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
    pass.end();
    // paramsBuf can be destroyed once the encoder is submitted; defer to caller.
    // Stash on the slot's rangeBuffer-adjacent state via the returned descriptor.
    // Simpler: rely on JS GC for the small (32B) buffer. Mark it for destroy.
    paramsBufQueue.push(paramsBuf);
    return true;
  }

  /**
   * Convenience wrapper: standalone submit of a single region reduction.
   * For batched per-panel work prefer `recordComputeRangeRegion` + your own
   * encoder so all panels share one submit.
   */
  computeRangeRegion(
    idx: number,
    region?: { x: number; y: number; width: number; height: number },
  ): void {
    const encoder = this.device.createCommandEncoder();
    if (!this.recordComputeRangeRegion(encoder, idx, region)) return;
    this.device.queue.submit([encoder.finish()]);
    flushParamsBufQueue();
  }

  /**
   * Range-aware colormap → ImageBitmap, reading vmin/vmax from each slot's
   * `rangeBuffer` (populated by `recordComputeRangeRegion`). Slider scaling
   * is applied on GPU.
   *
   * `vminPct`/`vmaxPct` parallel `indices`; pass `[0,100]` for raw range.
   *
   * Returns one ImageBitmap per index (null entries for missing slots).
   */
  renderSlotsWithGpuRange(
    indices: number[],
    vminPct: number[],
    vmaxPct: number[],
    logScale: boolean = false,
  ): ImageBitmap[] | null {
    this.ensureColormapRangePipeline();
    if (!this.colormapRangePipeline || !this.lutBuffer || indices.length === 0) return null;
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ensureBlitPipeline(fmt);
    if (!this.blitPipeline) return null;

    const encoder = this.device.createCommandEncoder();
    const params = new ArrayBuffer(32);
    const canvases: (OffscreenCanvas | null)[] = [];
    const tempBuffers: GPUBuffer[] = [];

    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      const slot = this.slots[i];
      if (!slot || slot.directOnly || slot.rgbaCapacity < slot.count || !slot.rangeBuffer) { canvases.push(null); continue; }
      const lowPct = vminPct[k] ?? 0;
      const highPct = vmaxPct[k] ?? 100;

      // Whole-slot colormap: region = full slot, stride = width
      const pu = new Uint32Array(params);
      const pf = new Float32Array(params);
      pu[0] = slot.width; pu[1] = slot.height;
      pf[2] = lowPct; pf[3] = highPct;
      pu[4] = logScale ? 1 : 0;
      pu[5] = 0; pu[6] = 0; pu[7] = slot.width;
      this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

      const computeGroup = this.device.createBindGroup({
        layout: this.colormapRangePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.lutBuffer } },
          { binding: 3, resource: { buffer: slot.rgbaBuffer } },
          { binding: 4, resource: { buffer: slot.rangeBuffer } },
        ],
      });
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(this.colormapRangePipeline);
      computePass.setBindGroup(0, computeGroup);
      computePass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
      computePass.end();

      const oc = new OffscreenCanvas(slot.width, slot.height);
      const ctx = oc.getContext("webgpu") as GPUCanvasContext;
      ctx.configure({ device: this.device, format: fmt, alphaMode: "opaque" });

      const blitParamsBuffer = this.device.createBuffer({
        size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(blitParamsBuffer, 0, new Uint32Array([slot.width, slot.height]));
      tempBuffers.push(blitParamsBuffer);

      const blitGroup = this.device.createBindGroup({
        layout: this.blitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: blitParamsBuffer } },
          { binding: 1, resource: { buffer: slot.rgbaBuffer } },
        ],
      });

      const texture = ctx.getCurrentTexture();
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: texture.createView(),
          loadOp: "clear" as GPULoadOp,
          storeOp: "store" as GPUStoreOp,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      renderPass.setPipeline(this.blitPipeline);
      renderPass.setBindGroup(0, blitGroup);
      renderPass.draw(3);
      renderPass.end();
      canvases.push(oc);
    }

    this.device.queue.submit([encoder.finish()]);
    for (const b of tempBuffers) b.destroy();
    flushParamsBufQueue();

    const bitmaps: ImageBitmap[] = [];
    for (const oc of canvases) {
      if (oc) bitmaps.push(oc.transferToImageBitmap());
      else bitmaps.push(null as never);
    }
    return bitmaps;
  }

  /**
   * Render panel sub-regions with explicit per-panel ranges. Used when
   * Show3D contrast is unlinked and each histogram owns its own clip state.
   */
  renderPerPanelGpuExplicit(
    slotIdx: number,
    regions: { x: number; y: number; width: number; height: number }[],
    ranges: { vmin: number; vmax: number }[],
    logScale: boolean | boolean[] = false,
  ): ImageBitmap[] | null {
    this.ensureColormapRangePipeline();
    if (!this.colormapRangePipeline || !this.lutBuffer) return null;
    const slot = this.slots[slotIdx];
    if (!slot || regions.length === 0) return null;
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ensureBlitPipeline(fmt);
    if (!this.blitPipeline) return null;

    const encoder = this.device.createCommandEncoder();
    const cmParams = new ArrayBuffer(32);
    const canvases: (OffscreenCanvas | null)[] = [];
    const tempBuffers: GPUBuffer[] = [];

    for (let k = 0; k < regions.length; k++) {
      const r = regions[k];
      const panelRange = ranges[k] ?? ranges[0];
      if (!r || !panelRange) { canvases.push(null); continue; }
      const scratch = this.ensurePanelScratch(k, r.width * r.height);
      this.device.queue.writeBuffer(
        scratch.range,
        0,
        new Float32Array([panelRange.vmin, panelRange.vmax, 0, 0]),
      );

      const pu = new Uint32Array(cmParams);
      const pf = new Float32Array(cmParams);
      pu[0] = r.width; pu[1] = r.height;
      pf[2] = 0; pf[3] = 100;
      pu[4] = Array.isArray(logScale) ? (logScale[k] ? 1 : 0) : (logScale ? 1 : 0);
      pu[5] = r.x; pu[6] = r.y; pu[7] = slot.width;
      const cmParamsBuf = this.device.createBuffer({
        size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(cmParamsBuf, 0, cmParams);
      tempBuffers.push(cmParamsBuf);

      const cmGroup = this.device.createBindGroup({
        layout: this.colormapRangePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: cmParamsBuf } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.lutBuffer } },
          { binding: 3, resource: { buffer: scratch.rgba } },
          { binding: 4, resource: { buffer: scratch.range } },
        ],
      });
      const cmPass = encoder.beginComputePass();
      cmPass.setPipeline(this.colormapRangePipeline);
      cmPass.setBindGroup(0, cmGroup);
      cmPass.dispatchWorkgroups(Math.ceil(r.width / 16), Math.ceil(r.height / 16));
      cmPass.end();

      const oc = new OffscreenCanvas(r.width, r.height);
      const ctx = oc.getContext("webgpu") as GPUCanvasContext;
      ctx.configure({ device: this.device, format: fmt, alphaMode: "opaque" });
      const blitParamsBuffer = this.device.createBuffer({
        size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(blitParamsBuffer, 0, new Uint32Array([r.width, r.height]));
      tempBuffers.push(blitParamsBuffer);

      const blitGroup = this.device.createBindGroup({
        layout: this.blitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: blitParamsBuffer } },
          { binding: 1, resource: { buffer: scratch.rgba } },
        ],
      });
      const texture = ctx.getCurrentTexture();
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: texture.createView(),
          loadOp: "clear" as GPULoadOp,
          storeOp: "store" as GPUStoreOp,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      renderPass.setPipeline(this.blitPipeline);
      renderPass.setBindGroup(0, blitGroup);
      renderPass.draw(3);
      renderPass.end();
      canvases.push(oc);
    }

    this.device.queue.submit([encoder.finish()]);
    for (const b of tempBuffers) b.destroy();

    const bitmaps: ImageBitmap[] = [];
    for (const oc of canvases) bitmaps.push(oc ? oc.transferToImageBitmap() : null as never);
    return bitmaps;
  }

  /**
   * Fused per-panel pipeline for Show3D: ONE GPU slot holds the full frame;
   * each panel reads a sub-region. In ONE submit, for each panel:
   *   1. Region-reduce slot.dataBuffer over the panel region → vmin/vmax
   *      into a per-panel 16-byte range buffer
   *   2. Colormap (reading range buffer + slider pcts on GPU) → per-panel
   *      rgba buffer (sized to the panel sub-image)
   *   3. Blit per-panel rgba → OffscreenCanvas texture
   * Then synchronously transferToImageBitmap per panel. Zero CPU round-trips
   * for vmin/vmax — replaces the JS slab-extract + findDataRange loop.
   *
   * `slotIdx` is the GPU slot holding the full frame.
   * `regions[k]` is the sub-rect of panel k inside the full frame.
   * `vminPct/vmaxPct[k]` are the user contrast slider percentages [0,100].
   */
  renderPerPanelGpu(
    slotIdx: number,
    regions: { x: number; y: number; width: number; height: number }[],
    vminPct: number[],
    vmaxPct: number[],
    logScale: boolean = false,
  ): ImageBitmap[] | null {
    this.ensureRangeRegionPipeline();
    this.ensureColormapRangePipeline();
    if (!this.rangeRegionPipeline || !this.colormapRangePipeline || !this.lutBuffer) return null;
    const slot = this.slots[slotIdx];
    if (!slot || regions.length === 0) return null;
    const fmt = navigator.gpu.getPreferredCanvasFormat();
    this.ensureBlitPipeline(fmt);
    if (!this.blitPipeline) return null;

    const encoder = this.device.createCommandEncoder();
    const cmParams = new ArrayBuffer(32);
    const canvases: (OffscreenCanvas | null)[] = [];
    const tempBuffers: GPUBuffer[] = [];

    for (let k = 0; k < regions.length; k++) {
      const r = regions[k];
      if (!r) { canvases.push(null); continue; }
      const scratch = this.ensurePanelScratch(k, r.width * r.height);

      // --- 1. Region reduce → scratch.range ---
      const rgParams = this.device.createBuffer({
        size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(
        rgParams, 0,
        new Uint32Array([r.x, r.y, r.width, r.height, slot.width, 0, 0, 0]),
      );
      tempBuffers.push(rgParams);
      const rgGroup = this.device.createBindGroup({
        layout: this.rangeRegionPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.dataBuffer } },
          { binding: 1, resource: { buffer: rgParams } },
          { binding: 2, resource: { buffer: scratch.range } },
        ],
      });
      const rgPass = encoder.beginComputePass();
      rgPass.setPipeline(this.rangeRegionPipeline);
      rgPass.setBindGroup(0, rgGroup);
      rgPass.dispatchWorkgroups(1);
      rgPass.end();

      // --- 2. Colormap reading scratch.range + slider pcts ---
      // Output is the panel sub-image (size r.width × r.height) sourced from
      // slot.dataBuffer at offset (r.x, r.y) with stride slot.width.
      const lowPct = vminPct[k] ?? 0;
      const highPct = vmaxPct[k] ?? 100;
      const pu = new Uint32Array(cmParams);
      const pf = new Float32Array(cmParams);
      pu[0] = r.width; pu[1] = r.height;
      pf[2] = lowPct; pf[3] = highPct;
      pu[4] = logScale ? 1 : 0;
      pu[5] = r.x; pu[6] = r.y; pu[7] = slot.width;
      const cmParamsBuf = this.device.createBuffer({
        size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(cmParamsBuf, 0, cmParams);
      tempBuffers.push(cmParamsBuf);

      const cmGroup = this.device.createBindGroup({
        layout: this.colormapRangePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: cmParamsBuf } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: this.lutBuffer } },
          { binding: 3, resource: { buffer: scratch.rgba } },
          { binding: 4, resource: { buffer: scratch.range } },
        ],
      });
      const cmPass = encoder.beginComputePass();
      cmPass.setPipeline(this.colormapRangePipeline);
      cmPass.setBindGroup(0, cmGroup);
      cmPass.dispatchWorkgroups(Math.ceil(r.width / 16), Math.ceil(r.height / 16));
      cmPass.end();

      // --- 3. Blit scratch.rgba → OffscreenCanvas texture ---
      const oc = new OffscreenCanvas(r.width, r.height);
      const ctx = oc.getContext("webgpu") as GPUCanvasContext;
      ctx.configure({ device: this.device, format: fmt, alphaMode: "opaque" });

      const blitParamsBuffer = this.device.createBuffer({
        size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(blitParamsBuffer, 0, new Uint32Array([r.width, r.height]));
      tempBuffers.push(blitParamsBuffer);

      const blitGroup = this.device.createBindGroup({
        layout: this.blitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: blitParamsBuffer } },
          { binding: 1, resource: { buffer: scratch.rgba } },
        ],
      });
      const texture = ctx.getCurrentTexture();
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: texture.createView(),
          loadOp: "clear" as GPULoadOp,
          storeOp: "store" as GPUStoreOp,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      renderPass.setPipeline(this.blitPipeline);
      renderPass.setBindGroup(0, blitGroup);
      renderPass.draw(3);
      renderPass.end();
      canvases.push(oc);
    }

    this.device.queue.submit([encoder.finish()]);
    for (const b of tempBuffers) b.destroy();

    const bitmaps: ImageBitmap[] = [];
    for (const oc of canvases) {
      if (oc) bitmaps.push(oc.transferToImageBitmap());
      else bitmaps.push(null as never);
    }
    return bitmaps;
  }

  // ── GPU histogram ──

  private histPipeline: GPUComputePipeline | null = null;
  private histClearPipeline: GPUComputePipeline | null = null;

  private ensureHistPipeline(): void {
    if (this.histPipeline) return;
    const code = /* wgsl */ `
struct HistParams {
  width: u32,
  height: u32,
  dmin: f32,
  dmax: f32,
  log_scale: u32,
  _pad: u32,
};
@group(0) @binding(0) var<uniform> params: HistParams;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> bins: array<atomic<u32>>;

@compute @workgroup_size(16, 16)
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  var val = data[idx];
  if (params.log_scale == 1u) { val = log(1.0 + max(val, 0.0)); }
  let range = max(params.dmax - params.dmin, 1e-30);
  let t = clamp((val - params.dmin) / range, 0.0, 1.0);
  let bin = min(u32(t * 256.0), 255u);
  atomicAdd(&bins[bin], 1u);
}

@compute @workgroup_size(256)
fn clear_bins(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < 256u) { atomicStore(&bins[gid.x], 0u); }
}
`;
    const module = this.device.createShaderModule({ code });
    this.histPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "histogram" },
    });
    this.histClearPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "clear_bins" },
    });
  }

  /**
   * Batch-compute 256-bin histograms for multiple slots in ONE GPU submission.
   * Uses persistent per-slot histogram buffers (zero create/destroy overhead).
   * Returns normalized bins per image.
   */
  async computeHistogramBatch(
    indices: number[],
    ranges: { min: number; max: number }[],
    logScale: boolean = false,
  ): Promise<number[][]> {
    this.ensureHistPipeline();
    if (!this.histPipeline || !this.histClearPipeline || indices.length === 0) return [];

    const encoder = this.device.createCommandEncoder();
    const activeSlots: { k: number; slot: GPUSlot }[] = [];
    const params = new ArrayBuffer(24);

    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      const slot = this.slots[i];
      if (!slot) continue;
      const r = ranges[k] || { min: 0, max: 1 };
      if (r.min === r.max) continue;

      // Reuse persistent paramsBuffer for histogram (same layout as colormap params)
      const pu = new Uint32Array(params);
      const pf = new Float32Array(params);
      pu[0] = slot.width; pu[1] = slot.height;
      pf[2] = r.min; pf[3] = r.max;
      pu[4] = logScale ? 1 : 0; pu[5] = 0;
      this.device.queue.writeBuffer(slot.paramsBuffer, 0, params);

      // Clear bins (persistent buffer)
      const clearGroup = this.device.createBindGroup({
        layout: this.histClearPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 2, resource: { buffer: slot.histBinsBuffer } },
        ],
      });
      const clearPass = encoder.beginComputePass();
      clearPass.setPipeline(this.histClearPipeline!);
      clearPass.setBindGroup(0, clearGroup);
      clearPass.dispatchWorkgroups(1);
      clearPass.end();

      // Histogram (persistent buffer)
      const histGroup = this.device.createBindGroup({
        layout: this.histPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
          { binding: 2, resource: { buffer: slot.histBinsBuffer } },
        ],
      });
      const histPass = encoder.beginComputePass();
      histPass.setPipeline(this.histPipeline!);
      histPass.setBindGroup(0, histGroup);
      histPass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
      histPass.end();

      encoder.copyBufferToBuffer(slot.histBinsBuffer, 0, slot.histReadBuffer, 0, 256 * 4);
      activeSlots.push({ k, slot });
    }

    this.device.queue.submit([encoder.finish()]);
    await Promise.all(activeSlots.map(s => s.slot.histReadBuffer.mapAsync(GPUMapMode.READ)));

    const results: number[][] = [];
    for (const s of activeSlots) {
      const rawBins = new Uint32Array(s.slot.histReadBuffer.getMappedRange().slice(0));
      s.slot.histReadBuffer.unmap();

      let maxCount = 0;
      for (let j = 0; j < 256; j++) if (rawBins[j] > maxCount) maxCount = rawBins[j];
      const norm = new Array(256);
      for (let j = 0; j < 256; j++) norm[j] = maxCount > 0 ? rawBins[j] / maxCount : 0;
      results.push(norm);
    }
    return results;
  }

  /**
   * Compute a 256-bin histogram for slot `idx` on GPU, given known data range.
   * Returns normalized bins (0–1) matching `computeHistogramFromBytes`.
   */
  async computeHistogramWithRange(
    idx: number, dmin: number, dmax: number, logScale: boolean = false,
  ): Promise<number[]> {
    this.ensureHistPipeline();
    const slot = this.slots[idx];
    if (!slot || !this.histPipeline || !this.histClearPipeline) return new Array(256).fill(0);
    if (dmin === dmax) return new Array(256).fill(0);

    const binsBuffer = this.device.createBuffer({
      size: 256 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readBuffer = this.device.createBuffer({
      size: 256 * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const paramsBuf = this.device.createBuffer({
      size: 24,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const params = new ArrayBuffer(24);
    const pu = new Uint32Array(params);
    const pf = new Float32Array(params);
    pu[0] = slot.width; pu[1] = slot.height;
    pf[2] = dmin; pf[3] = dmax;
    pu[4] = logScale ? 1 : 0; pu[5] = 0;
    this.device.queue.writeBuffer(paramsBuf, 0, params);

    const encoder = this.device.createCommandEncoder();

    // Clear bins
    const clearGroup = this.device.createBindGroup({
      layout: this.histClearPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 2, resource: { buffer: binsBuffer } },
      ],
    });
    const clearPass = encoder.beginComputePass();
    clearPass.setPipeline(this.histClearPipeline);
    clearPass.setBindGroup(0, clearGroup);
    clearPass.dispatchWorkgroups(1);
    clearPass.end();

    // Histogram
    const histGroup = this.device.createBindGroup({
      layout: this.histPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
        { binding: 2, resource: { buffer: binsBuffer } },
      ],
    });
    const histPass = encoder.beginComputePass();
    histPass.setPipeline(this.histPipeline);
    histPass.setBindGroup(0, histGroup);
    histPass.dispatchWorkgroups(Math.ceil(slot.width / 16), Math.ceil(slot.height / 16));
    histPass.end();

    encoder.copyBufferToBuffer(binsBuffer, 0, readBuffer, 0, 256 * 4);
    this.device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const rawBins = new Uint32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    binsBuffer.destroy();
    readBuffer.destroy();
    paramsBuf.destroy();

    // Normalize (match CPU: divide by max count)
    let maxCount = 0;
    for (let i = 0; i < 256; i++) if (rawBins[i] > maxCount) maxCount = rawBins[i];
    const result = new Array(256);
    if (maxCount > 0) {
      for (let i = 0; i < 256; i++) result[i] = rawBins[i] / maxCount;
    } else {
      for (let i = 0; i < 256; i++) result[i] = 0;
    }
    return result;
  }
}

/** Create a GPU colormap engine. Returns null if WebGPU unavailable. */
export async function createGPUColormapEngine(): Promise<GPUColormapEngine | null> {
  try {
    const { getGPUDevice } = await import("./fft");
    const device = await getGPUDevice();
    if (!device) return null;
    return new GPUColormapEngine(device);
  } catch {
    return null;
  }
}

let gpuColormapEngine: GPUColormapEngine | null = null;

/** Get or create the singleton GPU colormap engine. Returns null if WebGPU unavailable. */
export async function getGPUColormapEngine(): Promise<GPUColormapEngine | null> {
  if (gpuColormapEngine) return gpuColormapEngine;
  gpuColormapEngine = await createGPUColormapEngine();
  return gpuColormapEngine;
}

/** Query the GPU's max buffer size in bytes. Returns 0 if WebGPU unavailable. */
export async function getGPUMaxBufferSize(): Promise<number> {
  try {
    if (!navigator.gpu) return 0;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return 0;
    return adapter.limits.maxStorageBufferBindingSize || adapter.limits.maxBufferSize || 0;
  } catch {
    return 0;
  }
}
