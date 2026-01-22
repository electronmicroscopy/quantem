/// <reference types="@webgpu/types" />
import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Switch from "@mui/material/Switch";
import Tooltip from "@mui/material/Tooltip";
import JSZip from "jszip";
import "./styles.css";

// ============================================================================
// Theme Detection - detect environment and light/dark mode
// ============================================================================
type Environment = "jupyterlab" | "vscode" | "colab" | "jupyter-classic" | "unknown";
type Theme = "light" | "dark";

interface ThemeInfo {
  environment: Environment;
  theme: Theme;
}

function detectTheme(): ThemeInfo {
  // 1. JupyterLab - has data-jp-theme-light attribute
  const jpThemeLight = document.body.dataset.jpThemeLight;
  if (jpThemeLight !== undefined) {
    return {
      environment: "jupyterlab",
      theme: jpThemeLight === "true" ? "light" : "dark",
    };
  }

  // 2. VS Code - has vscode-* classes on body or html
  const bodyClasses = document.body.className;
  const htmlClasses = document.documentElement.className;
  if (bodyClasses.includes("vscode-") || htmlClasses.includes("vscode-")) {
    const isDark = bodyClasses.includes("vscode-dark") || htmlClasses.includes("vscode-dark");
    return {
      environment: "vscode",
      theme: isDark ? "dark" : "light",
    };
  }

  // 3. Google Colab - has specific markers
  if (document.querySelector('colab-shaded-scroller') || document.body.classList.contains('colaboratory')) {
    // Colab: check computed background color
    const bg = getComputedStyle(document.body).backgroundColor;
    return {
      environment: "colab",
      theme: isColorDark(bg) ? "dark" : "light",
    };
  }

  // 4. Classic Jupyter Notebook - has #notebook element
  if (document.getElementById('notebook')) {
    const bodyBg = getComputedStyle(document.body).backgroundColor;
    return {
      environment: "jupyter-classic",
      theme: isColorDark(bodyBg) ? "dark" : "light",
    };
  }

  // 5. Fallback: check OS preference, then computed background
  const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)')?.matches;
  if (prefersDark !== undefined) {
    return {
      environment: "unknown",
      theme: prefersDark ? "dark" : "light",
    };
  }

  // Final fallback: check body background luminance
  const bg = getComputedStyle(document.body).backgroundColor;
  return {
    environment: "unknown",
    theme: isColorDark(bg) ? "dark" : "light",
  };
}

/** Check if a CSS color string is dark (luminance < 0.5) */
function isColorDark(color: string): boolean {
  // Parse rgb(r, g, b) or rgba(r, g, b, a)
  const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
  if (!match) return true; // Default to dark if can't parse
  const [, r, g, b] = match.map(Number);
  // Relative luminance formula (simplified)
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance < 0.5;
}

// ============================================================================
// Colormaps - pre-computed LUTs for image display
// ============================================================================
const COLORMAP_POINTS: Record<string, number[][]> = {
  inferno: [[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[252,193,57],[252,255,164]],
  viridis: [[68,1,84],[72,36,117],[65,68,135],[53,95,141],[42,120,142],[33,145,140],[34,168,132],[68,191,112],[122,209,81],[189,223,38],[253,231,37]],
  plasma: [[13,8,135],[75,3,161],[126,3,168],[168,34,150],[203,70,121],[229,107,93],[248,148,65],[253,195,40],[240,249,33]],
  magma: [[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],[229,80,100],[251,135,97],[254,194,135],[252,253,191]],
  hot: [[0,0,0],[87,0,0],[173,0,0],[255,0,0],[255,87,0],[255,173,0],[255,255,0],[255,255,128],[255,255,255]],
  gray: [[0,0,0],[255,255,255]],
};

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

const COLORMAPS: Record<string, Uint8Array> = Object.fromEntries(
  Object.entries(COLORMAP_POINTS).map(([name, points]) => [name, createColormapLUT(points)])
);

// ============================================================================
// FFT Utilities - CPU implementation with WebGPU acceleration
// ============================================================================
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

function nextPow2(n: number): number { return Math.pow(2, Math.ceil(Math.log2(n))); }
function isPow2(n: number): boolean { return n > 0 && (n & (n - 1)) === 0; }

function fft1dPow2(real: Float32Array, imag: Float32Array, inverse: boolean = false) {
  const n = real.length;
  if (n <= 1) return;
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) { [real[i], real[j]] = [real[j], real[i]]; [imag[i], imag[j]] = [imag[j], imag[i]]; }
    let k = n >> 1;
    while (k <= j) { j -= k; k >>= 1; }
    j += k;
  }
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (sign * 2 * Math.PI) / len;
    const wReal = Math.cos(angle), wImag = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let curReal = 1, curImag = 0;
      for (let k = 0; k < halfLen; k++) {
        const evenIdx = i + k, oddIdx = i + k + halfLen;
        const tReal = curReal * real[oddIdx] - curImag * imag[oddIdx];
        const tImag = curReal * imag[oddIdx] + curImag * real[oddIdx];
        real[oddIdx] = real[evenIdx] - tReal; imag[oddIdx] = imag[evenIdx] - tImag;
        real[evenIdx] += tReal; imag[evenIdx] += tImag;
        const newReal = curReal * wReal - curImag * wImag;
        curImag = curReal * wImag + curImag * wReal; curReal = newReal;
      }
    }
  }
  if (inverse) { for (let i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; } }
}

function fft2d(real: Float32Array, imag: Float32Array, width: number, height: number, inverse: boolean = false) {
  const paddedW = nextPow2(width), paddedH = nextPow2(height);
  const needsPadding = paddedW !== width || paddedH !== height;
  let workReal: Float32Array, workImag: Float32Array;
  if (needsPadding) {
    workReal = new Float32Array(paddedW * paddedH); workImag = new Float32Array(paddedW * paddedH);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      workReal[y * paddedW + x] = real[y * width + x]; workImag[y * paddedW + x] = imag[y * width + x];
    }
  } else { workReal = real; workImag = imag; }
  const rowReal = new Float32Array(paddedW), rowImag = new Float32Array(paddedW);
  for (let y = 0; y < paddedH; y++) {
    const offset = y * paddedW;
    for (let x = 0; x < paddedW; x++) { rowReal[x] = workReal[offset + x]; rowImag[x] = workImag[offset + x]; }
    fft1dPow2(rowReal, rowImag, inverse);
    for (let x = 0; x < paddedW; x++) { workReal[offset + x] = rowReal[x]; workImag[offset + x] = rowImag[x]; }
  }
  const colReal = new Float32Array(paddedH), colImag = new Float32Array(paddedH);
  for (let x = 0; x < paddedW; x++) {
    for (let y = 0; y < paddedH; y++) { colReal[y] = workReal[y * paddedW + x]; colImag[y] = workImag[y * paddedW + x]; }
    fft1dPow2(colReal, colImag, inverse);
    for (let y = 0; y < paddedH; y++) { workReal[y * paddedW + x] = colReal[y]; workImag[y * paddedW + x] = colImag[y]; }
  }
  if (needsPadding) {
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      real[y * width + x] = workReal[y * paddedW + x]; imag[y * width + x] = workImag[y * paddedW + x];
    }
  }
}

function fftshift(data: Float32Array, width: number, height: number): void {
  const halfW = width >> 1, halfH = height >> 1;
  const temp = new Float32Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    temp[((y + halfH) % height) * width + ((x + halfW) % width)] = data[y * width + x];
  }
  data.set(temp);
}

// ============================================================================
// WebGPU FFT - GPU-accelerated FFT when available
// ============================================================================
const FFT_SHADER = `fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }
fn twiddle(k: u32, N: u32, inverse: f32) -> vec2<f32> { let angle = inverse * 2.0 * 3.14159265359 * f32(k) / f32(N); return vec2<f32>(cos(angle), sin(angle)); }
fn bitReverse(x: u32, log2N: u32) -> u32 { var result: u32 = 0u; var val = x; for (var i: u32 = 0u; i < log2N; i = i + 1u) { result = (result << 1u) | (val & 1u); val = val >> 1u; } return result; }
struct FFTParams { N: u32, log2N: u32, stage: u32, inverse: f32, }
@group(0) @binding(0) var<uniform> params: FFTParams;
@group(0) @binding(1) var<storage, read_write> data: array<vec2<f32>>;
@compute @workgroup_size(256) fn bitReversePermute(@builtin(global_invocation_id) gid: vec3<u32>) { let idx = gid.x; if (idx >= params.N) { return; } let rev = bitReverse(idx, params.log2N); if (idx < rev) { let temp = data[idx]; data[idx] = data[rev]; data[rev] = temp; } }
@compute @workgroup_size(256) fn butterflyStage(@builtin(global_invocation_id) gid: vec3<u32>) { let idx = gid.x; if (idx >= params.N / 2u) { return; } let stage = params.stage; let halfSize = 1u << stage; let fullSize = halfSize << 1u; let group = idx / halfSize; let pos = idx % halfSize; let i = group * fullSize + pos; let j = i + halfSize; let w = twiddle(pos, fullSize, params.inverse); let u = data[i]; let t = cmul(w, data[j]); data[i] = u + t; data[j] = u - t; }
@compute @workgroup_size(256) fn normalize(@builtin(global_invocation_id) gid: vec3<u32>) { let idx = gid.x; if (idx >= params.N) { return; } let scale = 1.0 / f32(params.N); data[idx] = data[idx] * scale; }`;

const FFT_2D_SHADER = `fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }
fn twiddle(k: u32, N: u32, inverse: f32) -> vec2<f32> { let angle = inverse * 2.0 * 3.14159265359 * f32(k) / f32(N); return vec2<f32>(cos(angle), sin(angle)); }
fn bitReverse(x: u32, log2N: u32) -> u32 { var result: u32 = 0u; var val = x; for (var i: u32 = 0u; i < log2N; i = i + 1u) { result = (result << 1u) | (val & 1u); val = val >> 1u; } return result; }
struct FFT2DParams { width: u32, height: u32, log2Size: u32, stage: u32, inverse: f32, isRowWise: u32, }
@group(0) @binding(0) var<uniform> params: FFT2DParams;
@group(0) @binding(1) var<storage, read_write> data: array<vec2<f32>>;
fn getIndex(row: u32, col: u32) -> u32 { return row * params.width + col; }
@compute @workgroup_size(16, 16) fn bitReverseRows(@builtin(global_invocation_id) gid: vec3<u32>) { let row = gid.y; let col = gid.x; if (row >= params.height || col >= params.width) { return; } let rev = bitReverse(col, params.log2Size); if (col < rev) { let idx1 = getIndex(row, col); let idx2 = getIndex(row, rev); let temp = data[idx1]; data[idx1] = data[idx2]; data[idx2] = temp; } }
@compute @workgroup_size(16, 16) fn bitReverseCols(@builtin(global_invocation_id) gid: vec3<u32>) { let row = gid.y; let col = gid.x; if (row >= params.height || col >= params.width) { return; } let rev = bitReverse(row, params.log2Size); if (row < rev) { let idx1 = getIndex(row, col); let idx2 = getIndex(rev, col); let temp = data[idx1]; data[idx1] = data[idx2]; data[idx2] = temp; } }
@compute @workgroup_size(16, 16) fn butterflyRows(@builtin(global_invocation_id) gid: vec3<u32>) { let row = gid.y; let idx = gid.x; if (row >= params.height || idx >= params.width / 2u) { return; } let stage = params.stage; let halfSize = 1u << stage; let fullSize = halfSize << 1u; let group = idx / halfSize; let pos = idx % halfSize; let col_i = group * fullSize + pos; let col_j = col_i + halfSize; if (col_j >= params.width) { return; } let w = twiddle(pos, fullSize, params.inverse); let i = getIndex(row, col_i); let j = getIndex(row, col_j); let u = data[i]; let t = cmul(w, data[j]); data[i] = u + t; data[j] = u - t; }
@compute @workgroup_size(16, 16) fn butterflyCols(@builtin(global_invocation_id) gid: vec3<u32>) { let col = gid.x; let idx = gid.y; if (col >= params.width || idx >= params.height / 2u) { return; } let stage = params.stage; let halfSize = 1u << stage; let fullSize = halfSize << 1u; let group = idx / halfSize; let pos = idx % halfSize; let row_i = group * fullSize + pos; let row_j = row_i + halfSize; if (row_j >= params.height) { return; } let w = twiddle(pos, fullSize, params.inverse); let i = getIndex(row_i, col); let j = getIndex(row_j, col); let u = data[i]; let t = cmul(w, data[j]); data[i] = u + t; data[j] = u - t; }
@compute @workgroup_size(16, 16) fn normalize2D(@builtin(global_invocation_id) gid: vec3<u32>) { let row = gid.y; let col = gid.x; if (row >= params.height || col >= params.width) { return; } let idx = getIndex(row, col); let scale = 1.0 / f32(params.width * params.height); data[idx] = data[idx] * scale; }`;

class WebGPUFFT {
  private device: GPUDevice;
  private pipelines1D: { bitReverse: GPUComputePipeline; butterfly: GPUComputePipeline; normalize: GPUComputePipeline } | null = null;
  private pipelines2D: { bitReverseRows: GPUComputePipeline; bitReverseCols: GPUComputePipeline; butterflyRows: GPUComputePipeline; butterflyCols: GPUComputePipeline; normalize: GPUComputePipeline } | null = null;
  private initialized = false;
  constructor(device: GPUDevice) { this.device = device; }
  async init(): Promise<void> {
    if (this.initialized) return;
    const module1D = this.device.createShaderModule({ code: FFT_SHADER });
    this.pipelines1D = {
      bitReverse: this.device.createComputePipeline({ layout: 'auto', compute: { module: module1D, entryPoint: 'bitReversePermute' } }),
      butterfly: this.device.createComputePipeline({ layout: 'auto', compute: { module: module1D, entryPoint: 'butterflyStage' } }),
      normalize: this.device.createComputePipeline({ layout: 'auto', compute: { module: module1D, entryPoint: 'normalize' } })
    };
    const module2D = this.device.createShaderModule({ code: FFT_2D_SHADER });
    this.pipelines2D = {
      bitReverseRows: this.device.createComputePipeline({ layout: 'auto', compute: { module: module2D, entryPoint: 'bitReverseRows' } }),
      bitReverseCols: this.device.createComputePipeline({ layout: 'auto', compute: { module: module2D, entryPoint: 'bitReverseCols' } }),
      butterflyRows: this.device.createComputePipeline({ layout: 'auto', compute: { module: module2D, entryPoint: 'butterflyRows' } }),
      butterflyCols: this.device.createComputePipeline({ layout: 'auto', compute: { module: module2D, entryPoint: 'butterflyCols' } }),
      normalize: this.device.createComputePipeline({ layout: 'auto', compute: { module: module2D, entryPoint: 'normalize2D' } })
    };
    this.initialized = true;
  }
  async fft2D(realData: Float32Array, imagData: Float32Array, width: number, height: number, inverse: boolean = false): Promise<{ real: Float32Array, imag: Float32Array }> {
    await this.init();
    const paddedWidth = nextPow2(width), paddedHeight = nextPow2(height);
    const needsPadding = paddedWidth !== width || paddedHeight !== height;
    const log2Width = Math.log2(paddedWidth), log2Height = Math.log2(paddedHeight);
    const paddedSize = paddedWidth * paddedHeight, originalSize = width * height;
    let workReal: Float32Array, workImag: Float32Array;
    if (needsPadding) {
      workReal = new Float32Array(paddedSize); workImag = new Float32Array(paddedSize);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) { workReal[y * paddedWidth + x] = realData[y * width + x]; workImag[y * paddedWidth + x] = imagData[y * width + x]; }
    } else { workReal = realData; workImag = imagData; }
    const complexData = new Float32Array(paddedSize * 2);
    for (let i = 0; i < paddedSize; i++) { complexData[i * 2] = workReal[i]; complexData[i * 2 + 1] = workImag[i]; }
    const dataBuffer = this.device.createBuffer({ size: complexData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(dataBuffer, 0, complexData);
    const paramsBuffer = this.device.createBuffer({ size: 24, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const readBuffer = this.device.createBuffer({ size: complexData.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const inverseVal = inverse ? 1.0 : -1.0;
    const workgroupsX = Math.ceil(paddedWidth / 16), workgroupsY = Math.ceil(paddedHeight / 16);
    const runPass = (pipeline: GPUComputePipeline) => {
      const bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: paramsBuffer } }, { binding: 1, resource: { buffer: dataBuffer } }] });
      const encoder = this.device.createCommandEncoder(); const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(workgroupsX, workgroupsY); pass.end();
      this.device.queue.submit([encoder.finish()]);
    };
    const params = new ArrayBuffer(24); const paramsU32 = new Uint32Array(params); const paramsF32 = new Float32Array(params);
    paramsU32[0] = paddedWidth; paramsU32[1] = paddedHeight; paramsU32[2] = log2Width; paramsU32[3] = 0; paramsF32[4] = inverseVal; paramsU32[5] = 1;
    this.device.queue.writeBuffer(paramsBuffer, 0, params); runPass(this.pipelines2D!.bitReverseRows);
    for (let stage = 0; stage < log2Width; stage++) { paramsU32[3] = stage; this.device.queue.writeBuffer(paramsBuffer, 0, params); runPass(this.pipelines2D!.butterflyRows); }
    paramsU32[2] = log2Height; paramsU32[3] = 0; paramsU32[5] = 0;
    this.device.queue.writeBuffer(paramsBuffer, 0, params); runPass(this.pipelines2D!.bitReverseCols);
    for (let stage = 0; stage < log2Height; stage++) { paramsU32[3] = stage; this.device.queue.writeBuffer(paramsBuffer, 0, params); runPass(this.pipelines2D!.butterflyCols); }
    if (inverse) runPass(this.pipelines2D!.normalize);
    const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, complexData.byteLength);
    this.device.queue.submit([encoder.finish()]); await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0)); readBuffer.unmap();
    dataBuffer.destroy(); paramsBuffer.destroy(); readBuffer.destroy();
    if (needsPadding) {
      const realResult = new Float32Array(originalSize), imagResult = new Float32Array(originalSize);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) { realResult[y * width + x] = result[(y * paddedWidth + x) * 2]; imagResult[y * width + x] = result[(y * paddedWidth + x) * 2 + 1]; }
      return { real: realResult, imag: imagResult };
    }
    const realResult = new Float32Array(paddedSize), imagResult = new Float32Array(paddedSize);
    for (let i = 0; i < paddedSize; i++) { realResult[i] = result[i * 2]; imagResult[i] = result[i * 2 + 1]; }
    return { real: realResult, imag: imagResult };
  }
  destroy(): void { this.initialized = false; }
}

let gpuFFT: WebGPUFFT | null = null;
async function getWebGPUFFT(): Promise<WebGPUFFT | null> {
  if (gpuFFT) return gpuFFT;
  if (!navigator.gpu) { console.warn('WebGPU not supported, using CPU FFT'); return null; }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    gpuFFT = new WebGPUFFT(device); await gpuFFT.init();
    console.log('🚀 WebGPU FFT ready');
    return gpuFFT;
  } catch (e) { console.warn('WebGPU init failed:', e); return null; }
}

// ============================================================================
// UI Styles - component styling helpers
// ============================================================================
const typography = {
  label: { color: "#aaa", fontSize: 11 },
  labelSmall: { color: "#888", fontSize: 10 },
  value: { color: "#888", fontSize: 10, fontFamily: "monospace" },
  title: { color: "#0af", fontWeight: "bold" as const },
};

const controlPanel = {
  group: { bgcolor: "#222", px: 1.5, py: 0.5, borderRadius: 1, border: "1px solid #444", height: 32 },
  button: { color: "#888", fontSize: 10, cursor: "pointer", "&:hover": { color: "#fff" }, bgcolor: "#222", px: 1, py: 0.25, borderRadius: 0.5, border: "1px solid #444" },
  select: { minWidth: 90, bgcolor: "#333", color: "#fff", fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const container = {
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "monospace", borderRadius: 1, overflow: "visible" },
  imageBox: { bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" as const },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

const switchStyles = {
  small: { '& .MuiSwitch-thumb': { width: 12, height: 12 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
  medium: { '& .MuiSwitch-thumb': { width: 14, height: 14 }, '& .MuiSwitch-switchBase': { padding: '4px' } },
};

// ============================================================================
// Layout Constants - consistent spacing throughout
// ============================================================================
const SPACING = {
  XS: 4,    // Extra small gap
  SM: 8,    // Small gap (default between elements)
  MD: 12,   // Medium gap (between control groups)
  LG: 16,   // Large gap (between major sections)
};

const CANVAS_SIZE = 450;  // Both DP and VI canvases

// Interaction constants
const RESIZE_HIT_AREA_PX = 10;
const CIRCLE_HANDLE_ANGLE = 0.707;  // cos(45°)
const LINE_WIDTH_FRACTION = 0.015;
const LINE_WIDTH_MIN_PX = 1.5;
const LINE_WIDTH_MAX_PX = 3;

// Compact button style for Reset/Export
const compactButton = {
  fontSize: 10,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": {
    color: "#666",
    borderColor: "#444",
  },
};

// Control row style - bordered container for each row
const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  borderRadius: "2px",
  px: 1,
  py: 0.5,
  width: "fit-content",
};

/** Round to a nice value (1, 2, 5, 10, 20, 50, etc.) */
function roundToNiceValue(value: number): number {
  if (value <= 0) return 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
  const normalized = value / magnitude;
  if (normalized < 1.5) return magnitude;
  if (normalized < 3.5) return 2 * magnitude;
  if (normalized < 7.5) return 5 * magnitude;
  return 10 * magnitude;
}

/** Format scale bar label with appropriate unit */
function formatScaleLabel(value: number, unit: "Å" | "mrad" | "px"): string {
  const nice = roundToNiceValue(value);
  if (unit === "Å") {
    if (nice >= 10) return `${Math.round(nice / 10)} nm`;
    return nice >= 1 ? `${Math.round(nice)} Å` : `${nice.toFixed(2)} Å`;
  }
  if (unit === "px") {
    return nice >= 1 ? `${Math.round(nice)} px` : `${nice.toFixed(1)} px`;
  }
  if (nice >= 1000) return `${Math.round(nice / 1000)} rad`;
  return nice >= 1 ? `${Math.round(nice)} mrad` : `${nice.toFixed(2)} mrad`;
}

/** Format stat value for display (compact scientific notation for small values) */
function formatStat(value: number): string {
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 0.001 || abs >= 10000) {
    return value.toExponential(2);
  }
  if (abs < 0.01) return value.toFixed(4);
  if (abs < 1) return value.toFixed(3);
  return value.toFixed(2);
}

/**
 * Draw scale bar and zoom indicator on a high-DPI UI canvas.
 * This renders crisp text/lines independent of the image resolution.
 */
function drawScaleBarHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  zoom: number,
  pixelSize: number,
  unit: "Å" | "mrad" | "px",
  imageWidth: number,
  imageHeight: number
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Scale context for device pixel ratio
  ctx.save();
  ctx.scale(dpr, dpr);

  // CSS pixel dimensions
  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;

  // Calculate separate X/Y scale factors (canvas stretches to fill, not aspect-preserving)
  const scaleX = cssWidth / imageWidth;
  // Use X scale for horizontal measurements (scale bar is horizontal)
  const effectiveZoom = zoom * scaleX;
  
  // Fixed UI sizes in CSS pixels (always crisp)
  const targetBarPx = 60;  // Target bar length in CSS pixels
  const barThickness = 5;
  const fontSize = 16;
  const margin = 12;
  
  // Calculate what physical size the target bar represents
  const targetPhysical = (targetBarPx / effectiveZoom) * pixelSize;
  
  // Round to a nice value
  const nicePhysical = roundToNiceValue(targetPhysical);
  
  // Calculate actual bar length for the nice value (in CSS pixels)
  const barPx = (nicePhysical / pixelSize) * effectiveZoom;

  const barY = cssHeight - margin;
  const barX = cssWidth - barPx - margin;

  // Draw bar with shadow for visibility
  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  
  ctx.fillStyle = "white";
  ctx.fillRect(barX, barY, barPx, barThickness);

  // Draw label (centered above bar)
  const label = formatScaleLabel(nicePhysical, unit);
  ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
  ctx.fillStyle = "white";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText(label, barX + barPx / 2, barY - 4);

  // Draw zoom indicator (bottom left)
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${zoom.toFixed(1)}×`, margin, cssHeight - margin + barThickness);
  
  ctx.restore();
}

/**
 * Draw VI crosshair on high-DPI canvas (crisp regardless of image resolution)
 * Note: Does NOT clear canvas - should be called after drawScaleBarHiDPI
 */
function drawViPositionMarker(
  canvas: HTMLCanvasElement,
  dpr: number,
  posX: number,  // Position in image coordinates
  posY: number,
  zoom: number,
  panX: number,
  panY: number,
  imageWidth: number,
  imageHeight: number,
  isDragging: boolean
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const scaleY = cssHeight / imageHeight;

  // Convert image coordinates to CSS pixel coordinates
  const screenX = posY * zoom * scaleX + panX * scaleX;
  const screenY = posX * zoom * scaleY + panY * scaleY;

  // Simple crosshair (no circle)
  const crosshairSize = 12;
  const lineWidth = 1.5;

  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(255, 100, 100, 0.9)";
  ctx.lineWidth = lineWidth;

  // Draw crosshair lines only
  ctx.beginPath();
  ctx.moveTo(screenX - crosshairSize, screenY);
  ctx.lineTo(screenX + crosshairSize, screenY);
  ctx.moveTo(screenX, screenY - crosshairSize);
  ctx.lineTo(screenX, screenY + crosshairSize);
  ctx.stroke();

  ctx.restore();
}

/**
 * Draw VI ROI overlay on high-DPI canvas for real-space region selection
 * Note: Does NOT clear canvas - should be called after drawViPositionMarker
 */
function drawViRoiOverlayHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  roiMode: string,
  centerX: number,
  centerY: number,
  radius: number,
  roiWidth: number,
  roiHeight: number,
  zoom: number,
  panX: number,
  panY: number,
  imageWidth: number,
  imageHeight: number,
  isDragging: boolean,
  isDraggingResize: boolean,
  isHoveringResize: boolean
) {
  if (roiMode === "off") return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  const scaleX = cssWidth / imageWidth;
  const scaleY = cssHeight / imageHeight;

  // Convert image coordinates to screen coordinates (note: Y is row, X is col in image)
  const screenX = centerY * zoom * scaleX + panX * scaleX;
  const screenY = centerX * zoom * scaleY + panY * scaleY;

  const lineWidth = 2.5;
  const crosshairSize = 10;
  const handleRadius = 6;

  ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;

  // Helper to draw resize handle (purple color for VI ROI to differentiate from DP)
  const drawResizeHandle = (handleX: number, handleY: number) => {
    let handleFill: string;
    let handleStroke: string;

    if (isDraggingResize) {
      handleFill = "rgba(180, 100, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else if (isHoveringResize) {
      handleFill = "rgba(220, 150, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else {
      handleFill = "rgba(160, 80, 255, 0.8)";
      handleStroke = "rgba(255, 255, 255, 0.8)";
    }
    ctx.beginPath();
    ctx.arc(handleX, handleY, handleRadius, 0, 2 * Math.PI);
    ctx.fillStyle = handleFill;
    ctx.fill();
    ctx.strokeStyle = handleStroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };

  // Helper to draw center crosshair (purple/magenta for VI ROI)
  const drawCenterCrosshair = () => {
    ctx.strokeStyle = isDragging ? "rgba(255, 200, 0, 0.9)" : "rgba(180, 80, 255, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(screenX - crosshairSize, screenY);
    ctx.lineTo(screenX + crosshairSize, screenY);
    ctx.moveTo(screenX, screenY - crosshairSize);
    ctx.lineTo(screenX, screenY + crosshairSize);
    ctx.stroke();
  };

  // Purple/magenta color for VI ROI to differentiate from green DP detector
  const strokeColor = isDragging ? "rgba(255, 200, 0, 0.9)" : "rgba(180, 80, 255, 0.9)";
  const fillColor = isDragging ? "rgba(255, 200, 0, 0.15)" : "rgba(180, 80, 255, 0.15)";

  if (roiMode === "circle" && radius > 0) {
    const screenRadiusX = radius * zoom * scaleX;
    const screenRadiusY = radius * zoom * scaleY;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusX, screenRadiusY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    ctx.fillStyle = fillColor;
    ctx.fill();

    drawCenterCrosshair();

    // Resize handle at 45° diagonal
    const handleOffsetX = screenRadiusX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetY = screenRadiusY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetX, screenY + handleOffsetY);

  } else if (roiMode === "square" && radius > 0) {
    // Square uses radius as half-size
    const screenHalfW = radius * zoom * scaleX;
    const screenHalfH = radius * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = fillColor;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "rect" && roiWidth > 0 && roiHeight > 0) {
    const screenHalfW = (roiWidth / 2) * zoom * scaleX;
    const screenHalfH = (roiHeight / 2) * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = fillColor;
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);
  }

  ctx.restore();
}

/**
 * Draw DP crosshair on high-DPI canvas (crisp regardless of detector resolution)
 * Note: Does NOT clear canvas - should be called after drawScaleBarHiDPI
 */
function drawDpCrosshairHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  kx: number,  // Position in detector coordinates
  ky: number,
  zoom: number,
  panX: number,
  panY: number,
  detWidth: number,
  detHeight: number,
  isDragging: boolean
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  // Use separate X/Y scale factors (canvas stretches to fill container)
  const scaleX = cssWidth / detWidth;
  const scaleY = cssHeight / detHeight;

  // Convert detector coordinates to CSS pixel coordinates (no swap - kx is X, ky is Y)
  const screenX = kx * zoom * scaleX + panX * scaleX;
  const screenY = ky * zoom * scaleY + panY * scaleY;
  
  // Fixed UI sizes in CSS pixels (consistent with VI crosshair)
  const crosshairSize = 18;
  const lineWidth = 3;
  const dotRadius = 6;
  
  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  
  ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
  ctx.lineWidth = lineWidth;
  
  // Draw crosshair
  ctx.beginPath();
  ctx.moveTo(screenX - crosshairSize, screenY);
  ctx.lineTo(screenX + crosshairSize, screenY);
  ctx.moveTo(screenX, screenY - crosshairSize);
  ctx.lineTo(screenX, screenY + crosshairSize);
  ctx.stroke();
  
  // Draw center dot
  ctx.beginPath();
  ctx.arc(screenX, screenY, dotRadius, 0, 2 * Math.PI);
  ctx.stroke();
  
  ctx.restore();
}

/**
 * Draw ROI overlay (circle, square, rect, annular) on high-DPI canvas
 * Note: Does NOT clear canvas - should be called after drawScaleBarHiDPI
 */
function drawRoiOverlayHiDPI(
  canvas: HTMLCanvasElement,
  dpr: number,
  roiMode: string,
  centerX: number,
  centerY: number,
  radius: number,
  radiusInner: number,
  roiWidth: number,
  roiHeight: number,
  zoom: number,
  panX: number,
  panY: number,
  detWidth: number,
  detHeight: number,
  isDragging: boolean,
  isDraggingResize: boolean,
  isDraggingResizeInner: boolean,
  isHoveringResize: boolean,
  isHoveringResizeInner: boolean
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.save();
  ctx.scale(dpr, dpr);

  const cssWidth = canvas.width / dpr;
  const cssHeight = canvas.height / dpr;
  // Use separate X/Y scale factors (canvas stretches to fill container)
  const scaleX = cssWidth / detWidth;
  const scaleY = cssHeight / detHeight;

  // Convert detector coordinates to CSS pixel coordinates
  const screenX = centerX * zoom * scaleX + panX * scaleX;
  const screenY = centerY * zoom * scaleY + panY * scaleY;
  
  // Fixed UI sizes in CSS pixels
  const lineWidth = 2.5;
  const crosshairSizeSmall = 10;
  const handleRadius = 6;
  
  ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
  ctx.shadowBlur = 2;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 1;
  
  // Helper to draw resize handle
  const drawResizeHandle = (handleX: number, handleY: number, isInner: boolean = false) => {
    let handleFill: string;
    let handleStroke: string;
    const dragging = isInner ? isDraggingResizeInner : isDraggingResize;
    const hovering = isInner ? isHoveringResizeInner : isHoveringResize;
    
    if (dragging) {
      handleFill = "rgba(0, 200, 255, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else if (hovering) {
      handleFill = "rgba(255, 100, 100, 1)";
      handleStroke = "rgba(255, 255, 255, 1)";
    } else {
      handleFill = isInner ? "rgba(0, 220, 255, 0.8)" : "rgba(0, 255, 0, 0.8)";
      handleStroke = "rgba(255, 255, 255, 0.8)";
    }
    ctx.beginPath();
    ctx.arc(handleX, handleY, handleRadius, 0, 2 * Math.PI);
    ctx.fillStyle = handleFill;
    ctx.fill();
    ctx.strokeStyle = handleStroke;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };
  
  // Helper to draw center crosshair
  const drawCenterCrosshair = () => {
    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(screenX - crosshairSizeSmall, screenY);
    ctx.lineTo(screenX + crosshairSizeSmall, screenY);
    ctx.moveTo(screenX, screenY - crosshairSizeSmall);
    ctx.lineTo(screenX, screenY + crosshairSizeSmall);
    ctx.stroke();
  };
  
  if (roiMode === "circle" && radius > 0) {
    // Use separate X/Y radii for ellipse (handles non-square detectors)
    const screenRadiusX = radius * zoom * scaleX;
    const screenRadiusY = radius * zoom * scaleY;

    // Draw ellipse (becomes circle if scaleX === scaleY)
    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusX, screenRadiusY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Semi-transparent fill
    ctx.fillStyle = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";
    ctx.fill();

    drawCenterCrosshair();

    // Resize handle at 45° diagonal
    const handleOffsetX = screenRadiusX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetY = screenRadiusY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetX, screenY + handleOffsetY);

  } else if (roiMode === "square" && radius > 0) {
    // Square in detector space uses same half-size in both dimensions
    const screenHalfW = radius * zoom * scaleX;
    const screenHalfH = radius * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "rect" && roiWidth > 0 && roiHeight > 0) {
    const screenHalfW = (roiWidth / 2) * zoom * scaleX;
    const screenHalfH = (roiHeight / 2) * zoom * scaleY;
    const left = screenX - screenHalfW;
    const top = screenY - screenHalfH;

    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
    ctx.stroke();

    ctx.fillStyle = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";
    ctx.fill();

    drawCenterCrosshair();
    drawResizeHandle(screenX + screenHalfW, screenY + screenHalfH);

  } else if (roiMode === "annular" && radius > 0) {
    // Use separate X/Y radii for ellipses
    const screenRadiusOuterX = radius * zoom * scaleX;
    const screenRadiusOuterY = radius * zoom * scaleY;
    const screenRadiusInnerX = (radiusInner || 0) * zoom * scaleX;
    const screenRadiusInnerY = (radiusInner || 0) * zoom * scaleY;

    // Outer ellipse (green)
    ctx.strokeStyle = isDragging ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusOuterX, screenRadiusOuterY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Inner ellipse (cyan)
    ctx.strokeStyle = isDragging ? "rgba(255, 200, 0, 0.9)" : "rgba(0, 220, 255, 0.9)";
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusInnerX, screenRadiusInnerY, 0, 0, 2 * Math.PI);
    ctx.stroke();

    // Fill annular region
    ctx.fillStyle = isDragging ? "rgba(255, 255, 0, 0.12)" : "rgba(0, 255, 0, 0.12)";
    ctx.beginPath();
    ctx.ellipse(screenX, screenY, screenRadiusOuterX, screenRadiusOuterY, 0, 0, 2 * Math.PI);
    ctx.ellipse(screenX, screenY, screenRadiusInnerX, screenRadiusInnerY, 0, 0, 2 * Math.PI, true);
    ctx.fill();

    drawCenterCrosshair();

    // Outer handle at 45° diagonal
    const handleOffsetOuterX = screenRadiusOuterX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetOuterY = screenRadiusOuterY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetOuterX, screenY + handleOffsetOuterY);

    // Inner handle at 45° diagonal
    const handleOffsetInnerX = screenRadiusInnerX * CIRCLE_HANDLE_ANGLE;
    const handleOffsetInnerY = screenRadiusInnerY * CIRCLE_HANDLE_ANGLE;
    drawResizeHandle(screenX + handleOffsetInnerX, screenY + handleOffsetInnerY, true);
  }
  
  ctx.restore();
}

// ============================================================================
// Histogram Component
// ============================================================================

/**
 * Compute histogram from byte data (0-255).
 * Returns 256 bins normalized to 0-1 range.
 */
function computeHistogramFromBytes(data: Uint8Array | Float32Array | null, numBins = 256): number[] {
  if (!data || data.length === 0) {
    return new Array(numBins).fill(0);
  }

  const bins = new Array(numBins).fill(0);

  // For Float32Array, find min/max and bin accordingly
  if (data instanceof Float32Array) {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    if (!isFinite(min) || !isFinite(max) || min === max) {
      return bins;
    }
    const range = max - min;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (isFinite(v)) {
        const binIdx = Math.min(numBins - 1, Math.floor(((v - min) / range) * numBins));
        bins[binIdx]++;
      }
    }
  } else {
    // Uint8Array - values are already 0-255
    for (let i = 0; i < data.length; i++) {
      const binIdx = Math.min(numBins - 1, data[i]);
      bins[binIdx]++;
    }
  }

  // Normalize bins to 0-1
  const maxCount = Math.max(...bins);
  if (maxCount > 0) {
    for (let i = 0; i < numBins; i++) {
      bins[i] /= maxCount;
    }
  }

  return bins;
}

interface HistogramProps {
  data: Uint8Array | Float32Array | null;
  colormap: string;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
}

/**
 * Info tooltip component - small ⓘ icon with hover tooltip
 */
function InfoTooltip({ text, theme = "dark" }: { text: string; theme?: "light" | "dark" }) {
  const isDark = theme === "dark";
  return (
    <Tooltip
      title={<Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>}
      arrow
      placement="bottom"
      componentsProps={{
        tooltip: {
          sx: {
            bgcolor: isDark ? "#333" : "#fff",
            color: isDark ? "#ddd" : "#333",
            border: `1px solid ${isDark ? "#555" : "#ccc"}`,
            maxWidth: 280,
            p: 1,
          },
        },
        arrow: {
          sx: {
            color: isDark ? "#333" : "#fff",
            "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` },
          },
        },
      }}
    >
      <Typography
        component="span"
        sx={{
          fontSize: 12,
          color: isDark ? "#888" : "#666",
          cursor: "help",
          ml: 0.5,
          "&:hover": { color: isDark ? "#aaa" : "#444" },
        }}
      >
        ⓘ
      </Typography>
    </Tooltip>
  );
}

/**
 * Histogram component with integrated vmin/vmax slider and statistics.
 * Shows data distribution with colormap gradient and adjustable clipping.
 */
function Histogram({
  data,
  colormap,
  vminPct,
  vmaxPct,
  onRangeChange,
  width = 120,
  height = 40,
  theme = "dark",
  dataMin = 0,
  dataMax = 1,
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(() => computeHistogramFromBytes(data), [data]);

  // Theme-aware colors
  const colors = theme === "dark" ? {
    bg: "#1a1a1a",
    barActive: "#888",
    barInactive: "#444",
    border: "#333",
  } : {
    bg: "#f0f0f0",
    barActive: "#666",
    barInactive: "#bbb",
    border: "#ccc",
  };

  // Draw histogram (vertical gray bars)
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear with theme background
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);

    // Reduce to fewer bins for cleaner display
    const displayBins = 64;
    const binRatio = Math.floor(bins.length / displayBins);
    const reducedBins: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) {
        sum += bins[i * binRatio + j] || 0;
      }
      reducedBins.push(sum / binRatio);
    }

    // Normalize
    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;

    // Calculate which bins are in the clipped range
    const vminBin = Math.floor((vminPct / 100) * displayBins);
    const vmaxBin = Math.floor((vmaxPct / 100) * displayBins);

    // Draw histogram bars
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      const x = i * barWidth;

      // Bars inside range are highlighted, outside are dimmed
      const inRange = i >= vminBin && i <= vmaxBin;
      ctx.fillStyle = inRange ? colors.barActive : colors.barInactive;
      ctx.fillRect(x + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }

  }, [bins, colormap, vminPct, vmaxPct, width, height, colors]);

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, borderRadius: 2, border: `1px solid ${colors.border}` }}
      />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => {
          const [newMin, newMax] = v as number[];
          onRangeChange(Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1));
        }}
        min={0}
        max={100}
        size="small"
        valueLabelDisplay="auto"
        valueLabelFormat={(pct) => {
          const val = dataMin + (pct / 100) * (dataMax - dataMin);
          return val >= 1000 ? val.toExponential(1) : val.toFixed(1);
        }}
        sx={{
          width,
          py: 0,
          "& .MuiSlider-thumb": { width: 8, height: 8 },
          "& .MuiSlider-rail": { height: 2 },
          "& .MuiSlider-track": { height: 2 },
          "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
        }}
      />
    </Box>
  );
}

// ============================================================================
// Main Component
// ============================================================================
function Show4DSTEM() {
  // ─────────────────────────────────────────────────────────────────────────
  // Model State (synced with Python)
  // ─────────────────────────────────────────────────────────────────────────
  const [shapeX] = useModelState<number>("shape_x");
  const [shapeY] = useModelState<number>("shape_y");
  const [detX] = useModelState<number>("det_x");
  const [detY] = useModelState<number>("det_y");

  const [posX, setPosX] = useModelState<number>("pos_x");
  const [posY, setPosY] = useModelState<number>("pos_y");
  const [roiCenterX, setRoiCenterX] = useModelState<number>("roi_center_x");
  const [roiCenterY, setRoiCenterY] = useModelState<number>("roi_center_y");
  const [, setRoiActive] = useModelState<boolean>("roi_active");

  const [pixelSize] = useModelState<number>("pixel_size");
  const [kPixelSize] = useModelState<number>("k_pixel_size");
  const [kCalibrated] = useModelState<boolean>("k_calibrated");

  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [virtualImageBytes] = useModelState<DataView>("virtual_image_bytes");

  // ROI state
  const [roiRadius, setRoiRadius] = useModelState<number>("roi_radius");
  const [roiRadiusInner, setRoiRadiusInner] = useModelState<number>("roi_radius_inner");
  const [roiMode, setRoiMode] = useModelState<string>("roi_mode");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");

  // Global min/max for DP normalization (from Python)
  const [dpGlobalMin] = useModelState<number>("dp_global_min");
  const [dpGlobalMax] = useModelState<number>("dp_global_max");

  // VI min/max for normalization (from Python)
  const [viDataMin] = useModelState<number>("vi_data_min");
  const [viDataMax] = useModelState<number>("vi_data_max");

  // Detector calibration (for presets)
  const [bfRadius] = useModelState<number>("bf_radius");
  const [centerX] = useModelState<number>("center_x");
  const [centerY] = useModelState<number>("center_y");

  // Path animation state
  const [pathPlaying, setPathPlaying] = useModelState<boolean>("path_playing");
  const [pathIndex, setPathIndex] = useModelState<number>("path_index");
  const [pathLength] = useModelState<number>("path_length");
  const [pathIntervalMs] = useModelState<number>("path_interval_ms");
  const [pathLoop] = useModelState<boolean>("path_loop");

  // Auto-detection trigger
  const [, setAutoDetectTrigger] = useModelState<boolean>("auto_detect_trigger");

  // ─────────────────────────────────────────────────────────────────────────
  // Local State (UI-only, not synced to Python)
  // ─────────────────────────────────────────────────────────────────────────
  const [localKx, setLocalKx] = React.useState(roiCenterX);
  const [localKy, setLocalKy] = React.useState(roiCenterY);
  const [localPosX, setLocalPosX] = React.useState(posX);
  const [localPosY, setLocalPosY] = React.useState(posY);
  const [isDraggingDP, setIsDraggingDP] = React.useState(false);
  const [isDraggingVI, setIsDraggingVI] = React.useState(false);
  const [isDraggingFFT, setIsDraggingFFT] = React.useState(false);
  const [fftDragStart, setFftDragStart] = React.useState<{ x: number, y: number, panX: number, panY: number } | null>(null);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false); // For annular inner handle
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  // VI ROI drag/resize states (same pattern as DP)
  const [isDraggingViRoi, setIsDraggingViRoi] = React.useState(false);
  const [isDraggingViRoiResize, setIsDraggingViRoiResize] = React.useState(false);
  const [isHoveringViRoiResize, setIsHoveringViRoiResize] = React.useState(false);
  // Independent colormaps for DP and VI panels
  const [dpColormap, setDpColormap] = React.useState("inferno");
  const [viColormap, setViColormap] = React.useState("inferno");
  // vmin/vmax percentile clipping (0-100)
  const [dpVminPct, setDpVminPct] = React.useState(0);
  const [dpVmaxPct, setDpVmaxPct] = React.useState(100);
  const [viVminPct, setViVminPct] = React.useState(0);
  const [viVmaxPct, setViVmaxPct] = React.useState(100);
  // Scale mode: "linear" | "log" | "power"
  const [dpScaleMode, setDpScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const dpPowerExp = 0.5;
  const [viScaleMode, setViScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const viPowerExp = 0.5;

  // VI ROI state (real-space region selection for summed DP) - synced with Python
  const [viRoiMode, setViRoiMode] = useModelState<string>("vi_roi_mode");
  const [viRoiCenterX, setViRoiCenterX] = useModelState<number>("vi_roi_center_x");
  const [viRoiCenterY, setViRoiCenterY] = useModelState<number>("vi_roi_center_y");
  const [viRoiRadius, setViRoiRadius] = useModelState<number>("vi_roi_radius");
  const [viRoiWidth, setViRoiWidth] = useModelState<number>("vi_roi_width");
  const [viRoiHeight, setViRoiHeight] = useModelState<number>("vi_roi_height");
  // Local VI ROI center for smooth dragging
  const [localViRoiCenterX, setLocalViRoiCenterX] = React.useState(viRoiCenterX || 0);
  const [localViRoiCenterY, setLocalViRoiCenterY] = React.useState(viRoiCenterY || 0);
  const [summedDpBytes] = useModelState<DataView>("summed_dp_bytes");
  const [summedDpCount] = useModelState<number>("summed_dp_count");
  const [dpStats] = useModelState<number[]>("dp_stats");  // [mean, min, max, std]
  const [viStats] = useModelState<number[]>("vi_stats");  // [mean, min, max, std]
  const [hotPixelFilter, setHotPixelFilter] = useModelState<boolean>("hot_pixel_filter");
  const [showFft, setShowFft] = React.useState(false);  // Hidden by default per feedback

  // Theme detection - detect environment and light/dark mode
  const [themeInfo, setThemeInfo] = React.useState<ThemeInfo>(() => detectTheme());

  // Re-detect theme on mount and when OS preference changes
  React.useEffect(() => {
    setThemeInfo(detectTheme());

    // Listen for OS preference changes
    const mediaQuery = window.matchMedia?.('(prefers-color-scheme: dark)');
    const handleChange = () => setThemeInfo(detectTheme());
    mediaQuery?.addEventListener?.('change', handleChange);

    // Also observe body attributes for JupyterLab theme changes
    const observer = new MutationObserver(() => setThemeInfo(detectTheme()));
    observer.observe(document.body, { attributes: true, attributeFilter: ['data-jp-theme-light', 'class'] });

    return () => {
      mediaQuery?.removeEventListener?.('change', handleChange);
      observer.disconnect();
    };
  }, []);

  // Theme colors based on detected theme
  const themeColors = themeInfo.theme === "dark" ? {
    bg: "#1e1e1e",
    bgAlt: "#1a1a1a",
    text: "#e0e0e0",
    textMuted: "#888",
    border: "#3a3a3a",
    controlBg: "#252525",
    accent: "#5af",
  } : {
    bg: "#ffffff",
    bgAlt: "#f5f5f5",
    text: "#1e1e1e",
    textMuted: "#666",
    border: "#ccc",
    controlBg: "#f0f0f0",
    accent: "#0066cc",
  };

  // Compute VI canvas dimensions to respect aspect ratio of rectangular scans
  // The longer dimension gets CANVAS_SIZE, the shorter scales proportionally
  const viCanvasWidth = shapeX > shapeY ? Math.round(CANVAS_SIZE * (shapeY / shapeX)) : CANVAS_SIZE;
  const viCanvasHeight = shapeY > shapeX ? Math.round(CANVAS_SIZE * (shapeX / shapeY)) : CANVAS_SIZE;

  // Histogram data - use state to ensure re-renders (both are Float32Array now)
  const [dpHistogramData, setDpHistogramData] = React.useState<Float32Array | null>(null);
  const [viHistogramData, setViHistogramData] = React.useState<Float32Array | null>(null);

  // Parse DP frame bytes for histogram (float32 now)
  React.useEffect(() => {
    if (!frameBytes) return;
    // Parse as Float32Array since Python now sends raw float32
    const rawData = new Float32Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength / 4);
    // Apply scale transformation for histogram display
    const scaledData = new Float32Array(rawData.length);
    if (dpScaleMode === "log") {
      for (let i = 0; i < rawData.length; i++) {
        scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
      }
    } else if (dpScaleMode === "power") {
      for (let i = 0; i < rawData.length; i++) {
        scaledData[i] = Math.pow(Math.max(0, rawData[i]), dpPowerExp);
      }
    } else {
      scaledData.set(rawData);
    }
    setDpHistogramData(scaledData);
  }, [frameBytes, dpScaleMode, dpPowerExp]);

  // GPU FFT state
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);

  // Path animation timer
  React.useEffect(() => {
    if (!pathPlaying || pathLength === 0) return;

    const timer = setInterval(() => {
      setPathIndex((prev: number) => {
        const next = prev + 1;
        if (next >= pathLength) {
          if (pathLoop) {
            return 0;  // Loop back to start
          } else {
            setPathPlaying(false);  // Stop at end
            return prev;
          }
        }
        return next;
      });
    }, pathIntervalMs);

    return () => clearInterval(timer);
  }, [pathPlaying, pathLength, pathIntervalMs, pathLoop, setPathIndex, setPathPlaying]);

  // Keyboard shortcuts
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      const step = e.shiftKey ? 10 : 1;

      switch (e.key) {
        case 'ArrowUp':
          e.preventDefault();
          setPosX(Math.max(0, posX - step));
          break;
        case 'ArrowDown':
          e.preventDefault();
          setPosX(Math.min(shapeX - 1, posX + step));
          break;
        case 'ArrowLeft':
          e.preventDefault();
          setPosY(Math.max(0, posY - step));
          break;
        case 'ArrowRight':
          e.preventDefault();
          setPosY(Math.min(shapeY - 1, posY + step));
          break;
        case ' ':  // Space bar
          e.preventDefault();
          if (pathLength > 0) {
            setPathPlaying(!pathPlaying);
          }
          break;
        case 'r':  // Reset view
        case 'R':
          setDpZoom(1); setDpPanX(0); setDpPanY(0);
          setViZoom(1); setViPanX(0); setViPanY(0);
          setFftZoom(1); setFftPanX(0); setFftPanY(0);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [posX, posY, shapeX, shapeY, pathPlaying, pathLength, setPosX, setPosY, setPathPlaying]);

  // Initialize WebGPU FFT on mount
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) {
        gpuFFTRef.current = fft;
        setGpuReady(true);
      }
    });
  }, []);

  // Root element ref (theme-aware styling handled via CSS variables)
  const rootRef = React.useRef<HTMLDivElement>(null);

  // Zoom state
  const [dpZoom, setDpZoom] = React.useState(1);
  const [dpPanX, setDpPanX] = React.useState(0);
  const [dpPanY, setDpPanY] = React.useState(0);
  const [viZoom, setViZoom] = React.useState(1);
  const [viPanX, setViPanX] = React.useState(0);
  const [viPanY, setViPanY] = React.useState(0);
  const [fftZoom, setFftZoom] = React.useState(1);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const [fftScaleMode, setFftScaleMode] = React.useState<"linear" | "log" | "power">("linear");
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftAuto, setFftAuto] = React.useState(true);  // Auto: mask DC + 99.9% clipping
  const [fftVminPct, setFftVminPct] = React.useState(0);
  const [fftVmaxPct, setFftVmaxPct] = React.useState(100);
  const [fftStats, setFftStats] = React.useState<number[] | null>(null);  // [mean, min, max, std]
  const [fftHistogramData, setFftHistogramData] = React.useState<Float32Array | null>(null);
  const [fftDataMin, setFftDataMin] = React.useState(0);
  const [fftDataMax, setFftDataMax] = React.useState(1);

  // Sync local state
  React.useEffect(() => {
    if (!isDraggingDP && !isDraggingResize) { setLocalKx(roiCenterX); setLocalKy(roiCenterY); }
  }, [roiCenterX, roiCenterY, isDraggingDP, isDraggingResize]);

  React.useEffect(() => {
    if (!isDraggingVI) { setLocalPosX(posX); setLocalPosY(posY); }
  }, [posX, posY, isDraggingVI]);

  // Sync VI ROI local state
  React.useEffect(() => {
    if (!isDraggingViRoi && !isDraggingViRoiResize) {
      setLocalViRoiCenterX(viRoiCenterX || shapeX / 2);
      setLocalViRoiCenterY(viRoiCenterY || shapeY / 2);
    }
  }, [viRoiCenterX, viRoiCenterY, isDraggingViRoi, isDraggingViRoiResize, shapeX, shapeY]);

  // Canvas refs
  const dpCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const dpOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const dpUiRef = React.useRef<HTMLCanvasElement>(null);  // High-DPI UI overlay for scale bar
  const dpOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const dpImageDataRef = React.useRef<ImageData | null>(null);
  const virtualCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const virtualOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const viUiRef = React.useRef<HTMLCanvasElement>(null);  // High-DPI UI overlay for scale bar
  const viOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const viImageDataRef = React.useRef<ImageData | null>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const fftOffscreenRef = React.useRef<HTMLCanvasElement | null>(null);
  const fftImageDataRef = React.useRef<ImageData | null>(null);

  // Device pixel ratio for high-DPI UI overlays
  const DPR = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

  // ─────────────────────────────────────────────────────────────────────────
  // Effects: Canvas Rendering & Animation
  // ─────────────────────────────────────────────────────────────────────────

  // Prevent page scroll when scrolling on canvases
  // Re-run when showFft changes since FFT canvas is conditionally rendered
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const overlays = [dpOverlayRef.current, virtualOverlayRef.current, fftOverlayRef.current];
    overlays.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => overlays.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, [showFft]);

  // Store raw data for filtering/FFT
  const rawVirtualImageRef = React.useRef<Float32Array | null>(null);
  const fftWorkRealRef = React.useRef<Float32Array | null>(null);
  const fftWorkImagRef = React.useRef<Float32Array | null>(null);
  const fftMagnitudeRef = React.useRef<Float32Array | null>(null);

  // Parse virtual image bytes into Float32Array and apply scale for histogram
  React.useEffect(() => {
    if (!virtualImageBytes) return;
    // Parse as Float32Array since Python now sends raw float32
    const numFloats = virtualImageBytes.byteLength / 4;
    const rawData = new Float32Array(virtualImageBytes.buffer, virtualImageBytes.byteOffset, numFloats);

    // Store a copy for filtering/FFT (rawData is a view, we need a copy)
    let storedData = rawVirtualImageRef.current;
    if (!storedData || storedData.length !== numFloats) {
      storedData = new Float32Array(numFloats);
      rawVirtualImageRef.current = storedData;
    }
    storedData.set(rawData);

    // Apply scale transformation for histogram display
    const scaledData = new Float32Array(numFloats);
    if (viScaleMode === "log") {
      for (let i = 0; i < numFloats; i++) {
        scaledData[i] = Math.log1p(Math.max(0, rawData[i]));
      }
    } else if (viScaleMode === "power") {
      for (let i = 0; i < numFloats; i++) {
        scaledData[i] = Math.pow(Math.max(0, rawData[i]), viPowerExp);
      }
    } else {
      scaledData.set(rawData);
    }
    // Update histogram state (triggers re-render)
    setViHistogramData(scaledData);
  }, [virtualImageBytes, viScaleMode, viPowerExp]);

  // Render DP with zoom (use summed DP when VI ROI is active)
  React.useEffect(() => {
    if (!dpCanvasRef.current) return;

    // Determine which bytes to display: summed DP (if VI ROI active) or single frame
    const usesSummedDp = viRoiMode && viRoiMode !== "off" && summedDpBytes && summedDpBytes.byteLength > 0;
    const sourceBytes = usesSummedDp ? summedDpBytes : frameBytes;
    if (!sourceBytes) return;

    const canvas = dpCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const lut = COLORMAPS[dpColormap] || COLORMAPS.inferno;

    // Parse data based on source (summedDp is still uint8, frame is now float32)
    let scaled: Float32Array;
    if (usesSummedDp) {
      // Summed DP is still uint8 from Python
      const bytes = new Uint8Array(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength);
      scaled = new Float32Array(bytes.length);
      for (let i = 0; i < bytes.length; i++) {
        scaled[i] = bytes[i];
      }
    } else {
      // Frame is now float32 from Python - parse and apply scale transformation
      const rawData = new Float32Array(sourceBytes.buffer, sourceBytes.byteOffset, sourceBytes.byteLength / 4);
      scaled = new Float32Array(rawData.length);

      if (dpScaleMode === "log") {
        for (let i = 0; i < rawData.length; i++) {
          scaled[i] = Math.log1p(Math.max(0, rawData[i]));
        }
      } else if (dpScaleMode === "power") {
        for (let i = 0; i < rawData.length; i++) {
          scaled[i] = Math.pow(Math.max(0, rawData[i]), dpPowerExp);
        }
      } else {
        scaled.set(rawData);
      }
    }

    // Compute actual min/max of scaled data for normalization
    let dataMin = Infinity, dataMax = -Infinity;
    for (let i = 0; i < scaled.length; i++) {
      if (scaled[i] < dataMin) dataMin = scaled[i];
      if (scaled[i] > dataMax) dataMax = scaled[i];
    }

    // Apply vmin/vmax percentile clipping
    const dataRange = dataMax - dataMin;
    const vmin = dataMin + dataRange * dpVminPct / 100;
    const vmax = dataMin + dataRange * dpVmaxPct / 100;
    const range = vmax > vmin ? vmax - vmin : 1;

    let offscreen = dpOffscreenRef.current;
    if (!offscreen) {
      offscreen = document.createElement("canvas");
      dpOffscreenRef.current = offscreen;
    }
    const sizeChanged = offscreen.width !== detY || offscreen.height !== detX;
    if (sizeChanged) {
      offscreen.width = detY;
      offscreen.height = detX;
      dpImageDataRef.current = null;
    }
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    let imgData = dpImageDataRef.current;
    if (!imgData) {
      imgData = offCtx.createImageData(detY, detX);
      dpImageDataRef.current = imgData;
    }
    const rgba = imgData.data;

    for (let i = 0; i < scaled.length; i++) {
      // Clamp to vmin/vmax and rescale to 0-255 for colormap lookup
      const clamped = Math.max(vmin, Math.min(vmax, scaled[i]));
      const v = Math.floor(((clamped - vmin) / range) * 255);
      const j = i * 4;
      const lutIdx = Math.max(0, Math.min(255, v)) * 3;
      rgba[j] = lut[lutIdx];
      rgba[j + 1] = lut[lutIdx + 1];
      rgba[j + 2] = lut[lutIdx + 2];
      rgba[j + 3] = 255;
    }
    offCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(dpPanX, dpPanY);
    ctx.scale(dpZoom, dpZoom);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [frameBytes, summedDpBytes, viRoiMode, detX, detY, dpColormap, dpVminPct, dpVmaxPct, dpScaleMode, dpPowerExp, dpZoom, dpPanX, dpPanY]);

  // Render DP overlay - just clear (ROI shapes now drawn on high-DPI UI canvas)
  React.useEffect(() => {
    if (!dpOverlayRef.current) return;
    const canvas = dpOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // All visual overlays (crosshair, ROI shapes, scale bar) are now on dpUiRef for crisp rendering
  }, [localKx, localKy, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner, dpZoom, dpPanX, dpPanY, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, detX, detY]);

  // Render filtered virtual image
  React.useEffect(() => {
    if (!rawVirtualImageRef.current || !virtualCanvasRef.current) return;
    const canvas = virtualCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = shapeY;
    const height = shapeX;

    const renderData = (filtered: Float32Array) => {
      // Normalize and render
      // Apply scale transformation first
      let scaled = filtered;
      if (viScaleMode === "log") {
        scaled = new Float32Array(filtered.length);
        for (let i = 0; i < filtered.length; i++) {
          scaled[i] = Math.log1p(Math.max(0, filtered[i]));
        }
      } else if (viScaleMode === "power") {
        scaled = new Float32Array(filtered.length);
        for (let i = 0; i < filtered.length; i++) {
          scaled[i] = Math.pow(Math.max(0, filtered[i]), viPowerExp);
        }
      }

      // Compute actual min/max of scaled data
      let dataMin = Infinity, dataMax = -Infinity;
      for (let i = 0; i < scaled.length; i++) {
        if (scaled[i] < dataMin) dataMin = scaled[i];
        if (scaled[i] > dataMax) dataMax = scaled[i];
      }

      // Apply vmin/vmax percentile clipping
      const dataRange = dataMax - dataMin;
      const vmin = dataMin + dataRange * viVminPct / 100;
      const vmax = dataMin + dataRange * viVmaxPct / 100;
      const range = vmax > vmin ? vmax - vmin : 1;

      const lut = COLORMAPS[viColormap] || COLORMAPS.inferno;
      let offscreen = viOffscreenRef.current;
      if (!offscreen) {
        offscreen = document.createElement("canvas");
        viOffscreenRef.current = offscreen;
      }
      const sizeChanged = offscreen.width !== width || offscreen.height !== height;
      if (sizeChanged) {
        offscreen.width = width;
        offscreen.height = height;
        viImageDataRef.current = null;
      }
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) return;

      let imageData = viImageDataRef.current;
      if (!imageData) {
        imageData = offCtx.createImageData(width, height);
        viImageDataRef.current = imageData;
      }
      for (let i = 0; i < scaled.length; i++) {
        // Clamp to vmin/vmax and rescale to 0-255
        const clamped = Math.max(vmin, Math.min(vmax, scaled[i]));
        const val = Math.floor(((clamped - vmin) / range) * 255);
        imageData.data[i * 4] = lut[val * 3];
        imageData.data[i * 4 + 1] = lut[val * 3 + 1];
        imageData.data[i * 4 + 2] = lut[val * 3 + 2];
        imageData.data[i * 4 + 3] = 255;
      }
      offCtx.putImageData(imageData, 0, 0);

      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(viPanX, viPanY);
      ctx.scale(viZoom, viZoom);
      ctx.drawImage(offscreen, 0, 0);
      ctx.restore();
    };

    if (!rawVirtualImageRef.current) return;
    renderData(rawVirtualImageRef.current);
  }, [virtualImageBytes, shapeX, shapeY, viColormap, viVminPct, viVmaxPct, viScaleMode, viPowerExp, viZoom, viPanX, viPanY]);

  // Render virtual image overlay (just clear - crosshair drawn on high-DPI UI canvas)
  React.useEffect(() => {
    if (!virtualOverlayRef.current) return;
    const canvas = virtualOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Crosshair and scale bar now drawn on high-DPI UI canvas (viUiRef)
  }, [localPosX, localPosY, isDraggingVI, viZoom, viPanX, viPanY, pixelSize, shapeX, shapeY]);

  // Render FFT (WebGPU when available, CPU fallback)
  React.useEffect(() => {
    if (!rawVirtualImageRef.current || !fftCanvasRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (!showFft) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }

    const width = shapeY;
    const height = shapeX;
    const sourceData = rawVirtualImageRef.current;
    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;

    // Helper to render magnitude to canvas
    const renderMagnitude = (real: Float32Array, imag: Float32Array) => {
      // Compute magnitude (log or linear)
      let magnitude = fftMagnitudeRef.current;
      if (!magnitude || magnitude.length !== real.length) {
        magnitude = new Float32Array(real.length);
        fftMagnitudeRef.current = magnitude;
      }
      for (let i = 0; i < real.length; i++) {
        const mag = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
        if (fftScaleMode === "log") {
          magnitude[i] = Math.log1p(mag);
        } else if (fftScaleMode === "power") {
          magnitude[i] = Math.pow(mag, 0.5);  // gamma = 0.5
        } else {
          magnitude[i] = mag;
        }
      }

      // Auto mode: mask DC component + 99.9% percentile clipping
      let displayMin: number, displayMax: number;
      if (fftAuto) {
        // Mask DC (center pixel) by replacing with neighbor average
        const centerIdx = Math.floor(height / 2) * width + Math.floor(width / 2);
        const neighbors = [
          magnitude[centerIdx - 1],
          magnitude[centerIdx + 1],
          magnitude[centerIdx - width],
          magnitude[centerIdx + width]
        ];
        magnitude[centerIdx] = neighbors.reduce((a, b) => a + b, 0) / 4;

        // Apply 99.9% percentile clipping for display range
        const sorted = magnitude.slice().sort((a, b) => a - b);
        displayMin = sorted[0];
        displayMax = sorted[Math.floor(sorted.length * 0.999)];
      } else {
        // No auto: use actual min/max
        displayMin = Infinity;
        displayMax = -Infinity;
        for (let i = 0; i < magnitude.length; i++) {
          if (magnitude[i] < displayMin) displayMin = magnitude[i];
          if (magnitude[i] > displayMax) displayMax = magnitude[i];
        }
      }
      setFftDataMin(displayMin);
      setFftDataMax(displayMax);

      // Stats use same values
      const actualMin = displayMin;
      const actualMax = displayMax;
      let sum = 0;
      for (let i = 0; i < magnitude.length; i++) {
        sum += magnitude[i];
      }
      const mean = sum / magnitude.length;
      let sumSq = 0;
      for (let i = 0; i < magnitude.length; i++) {
        const diff = magnitude[i] - mean;
        sumSq += diff * diff;
      }
      const std = Math.sqrt(sumSq / magnitude.length);
      setFftStats([mean, actualMin, actualMax, std]);

      // Store histogram data (copy of magnitude for histogram component)
      setFftHistogramData(magnitude.slice());

      let offscreen = fftOffscreenRef.current;
      if (!offscreen) {
        offscreen = document.createElement("canvas");
        fftOffscreenRef.current = offscreen;
      }
      const sizeChanged = offscreen.width !== width || offscreen.height !== height;
      if (sizeChanged) {
        offscreen.width = width;
        offscreen.height = height;
        fftImageDataRef.current = null;
      }
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) return;

      let imgData = fftImageDataRef.current;
      if (!imgData) {
        imgData = offCtx.createImageData(width, height);
        fftImageDataRef.current = imgData;
      }
      const rgba = imgData.data;

      // Apply histogram slider range on top of percentile clipping
      const dataRange = displayMax - displayMin;
      const vmin = displayMin + (fftVminPct / 100) * dataRange;
      const vmax = displayMin + (fftVmaxPct / 100) * dataRange;
      const range = vmax > vmin ? vmax - vmin : 1;

      for (let i = 0; i < magnitude.length; i++) {
        const v = Math.round(((magnitude[i] - vmin) / range) * 255);
        const j = i * 4;
        const lutIdx = Math.max(0, Math.min(255, v)) * 3;
        rgba[j] = lut[lutIdx];
        rgba[j + 1] = lut[lutIdx + 1];
        rgba[j + 2] = lut[lutIdx + 2];
        rgba[j + 3] = 255;
      }
      offCtx.putImageData(imgData, 0, 0);

      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(fftPanX, fftPanY);
      ctx.scale(fftZoom, fftZoom);
      ctx.drawImage(offscreen, 0, 0);
      ctx.restore();
    };

    // Try WebGPU first, fall back to CPU
    if (gpuFFTRef.current && gpuReady) {
      // WebGPU path (async)
      let isCancelled = false;
      const runGpuFFT = async () => {
        const real = sourceData.slice();
        const imag = new Float32Array(real.length);
        
        const { real: fReal, imag: fImag } = await gpuFFTRef.current!.fft2D(real, imag, width, height, false);
        if (isCancelled) return;
        
        // Shift in CPU (TODO: move to GPU shader)
        fftshift(fReal, width, height);
        fftshift(fImag, width, height);
        
        renderMagnitude(fReal, fImag);
      };
      runGpuFFT();
      return () => { isCancelled = true; };
    } else {
      // CPU fallback (sync)
      const len = sourceData.length;
      let real = fftWorkRealRef.current;
      if (!real || real.length !== len) {
        real = new Float32Array(len);
        fftWorkRealRef.current = real;
      }
      real.set(sourceData);
      let imag = fftWorkImagRef.current;
      if (!imag || imag.length !== len) {
        imag = new Float32Array(len);
        fftWorkImagRef.current = imag;
      } else {
        imag.fill(0);
      }
      fft2d(real, imag, width, height, false);
      fftshift(real, width, height);
      fftshift(imag, width, height);
      renderMagnitude(real, imag);
    }
  }, [virtualImageBytes, shapeX, shapeY, fftColormap, fftZoom, fftPanX, fftPanY, gpuReady, showFft, fftScaleMode, fftAuto, fftVminPct, fftVmaxPct]);

  // Render FFT overlay with high-pass filter circle
  React.useEffect(() => {
    if (!fftOverlayRef.current) return;
    const canvas = fftOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, [fftZoom, fftPanX, fftPanY, showFft]);

  // ─────────────────────────────────────────────────────────────────────────
  // High-DPI Scale Bar UI Overlays
  // ─────────────────────────────────────────────────────────────────────────
  
  // DP scale bar + crosshair + ROI overlay (high-DPI)
  React.useEffect(() => {
    if (!dpUiRef.current) return;
    // Draw scale bar first (clears canvas)
    const kUnit = kCalibrated ? "mrad" : "px";
    drawScaleBarHiDPI(dpUiRef.current, DPR, dpZoom, kPixelSize || 1, kUnit, detY, detX);
    // Draw ROI overlay (circle, square, rect, annular) or point crosshair
    if (roiMode === "point") {
      drawDpCrosshairHiDPI(dpUiRef.current, DPR, localKx, localKy, dpZoom, dpPanX, dpPanY, detY, detX, isDraggingDP);
    } else {
      drawRoiOverlayHiDPI(
        dpUiRef.current, DPR, roiMode,
        localKx, localKy, roiRadius, roiRadiusInner, roiWidth, roiHeight,
        dpZoom, dpPanX, dpPanY, detY, detX,
        isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner
      );
    }
  }, [dpZoom, dpPanX, dpPanY, kPixelSize, kCalibrated, detX, detY, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, localKx, localKy, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner]);
  
  // VI scale bar + crosshair + ROI (high-DPI)
  React.useEffect(() => {
    if (!viUiRef.current) return;
    // Draw scale bar first (clears canvas)
    drawScaleBarHiDPI(viUiRef.current, DPR, viZoom, pixelSize || 1, "Å", shapeY, shapeX);
    // Draw crosshair only when ROI is off (ROI replaces the crosshair)
    if (!viRoiMode || viRoiMode === "off") {
      drawViPositionMarker(viUiRef.current, DPR, localPosX, localPosY, viZoom, viPanX, viPanY, shapeY, shapeX, isDraggingVI);
    } else {
      // Draw VI ROI instead of crosshair
      drawViRoiOverlayHiDPI(
        viUiRef.current, DPR, viRoiMode,
        localViRoiCenterX, localViRoiCenterY, viRoiRadius || 5, viRoiWidth || 10, viRoiHeight || 10,
        viZoom, viPanX, viPanY, shapeY, shapeX,
        isDraggingViRoi, isDraggingViRoiResize, isHoveringViRoiResize
      );
    }
  }, [viZoom, viPanX, viPanY, pixelSize, shapeX, shapeY, localPosX, localPosY, isDraggingVI,
      viRoiMode, localViRoiCenterX, localViRoiCenterY, viRoiRadius, viRoiWidth, viRoiHeight,
      isDraggingViRoi, isDraggingViRoiResize, isHoveringViRoiResize]);
  
  // Generic zoom handler
  const createZoomHandler = (
    setZoom: React.Dispatch<React.SetStateAction<number>>,
    setPanX: React.Dispatch<React.SetStateAction<number>>,
    setPanY: React.Dispatch<React.SetStateAction<number>>,
    zoom: number, panX: number, panY: number,
    canvasRef: React.RefObject<HTMLCanvasElement | null>
  ) => (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * zoomFactor));
    const zoomRatio = newZoom / zoom;
    setZoom(newZoom);
    setPanX(mouseX - (mouseX - panX) * zoomRatio);
    setPanY(mouseY - (mouseY - panY) * zoomRatio);
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Mouse Handlers
  // ─────────────────────────────────────────────────────────────────────────

  // Helper: check if point is near the outer resize handle
  const isNearResizeHandle = (imgX: number, imgY: number): boolean => {
    if (roiMode === "rect") {
      // For rectangle, check near bottom-right corner
      const handleX = roiCenterX + roiWidth / 2;
      const handleY = roiCenterY + roiHeight / 2;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      return dist < RESIZE_HIT_AREA_PX / dpZoom;
    }
    if ((roiMode !== "circle" && roiMode !== "square" && roiMode !== "annular") || !roiRadius) return false;
    const offset = roiMode === "square" ? roiRadius : roiRadius * CIRCLE_HANDLE_ANGLE;
    const handleX = roiCenterX + offset;
    const handleY = roiCenterY + offset;
    const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
    return dist < RESIZE_HIT_AREA_PX / dpZoom;
  };

  // Helper: check if point is near the inner resize handle (annular mode only)
  const isNearResizeHandleInner = (imgX: number, imgY: number): boolean => {
    if (roiMode !== "annular" || !roiRadiusInner) return false;
    const offset = roiRadiusInner * CIRCLE_HANDLE_ANGLE;
    const handleX = roiCenterX + offset;
    const handleY = roiCenterY + offset;
    const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
    return dist < RESIZE_HIT_AREA_PX / dpZoom;
  };

  // Helper: check if point is near VI ROI resize handle (same logic as DP)
  // Hit area is capped to avoid overlap with center for small ROIs
  const isNearViRoiResizeHandle = (imgX: number, imgY: number): boolean => {
    if (!viRoiMode || viRoiMode === "off") return false;
    if (viRoiMode === "rect") {
      const halfH = (viRoiHeight || 10) / 2;
      const halfW = (viRoiWidth || 10) / 2;
      const handleX = localViRoiCenterX + halfH;
      const handleY = localViRoiCenterY + halfW;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      const cornerDist = Math.sqrt(halfW ** 2 + halfH ** 2);
      const hitArea = Math.min(RESIZE_HIT_AREA_PX / viZoom, cornerDist * 0.5);
      return dist < hitArea;
    }
    if (viRoiMode === "circle" || viRoiMode === "square") {
      const radius = viRoiRadius || 5;
      const offset = viRoiMode === "square" ? radius : radius * CIRCLE_HANDLE_ANGLE;
      const handleX = localViRoiCenterX + offset;
      const handleY = localViRoiCenterY + offset;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      // Cap hit area to 50% of radius so center remains draggable
      const hitArea = Math.min(RESIZE_HIT_AREA_PX / viZoom, radius * 0.5);
      return dist < hitArea;
    }
    return false;
  };

  // Mouse handlers
  const handleDpMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = dpOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenX - dpPanX) / dpZoom;
    const imgY = (screenY - dpPanY) / dpZoom;

    // Check if clicking on resize handle (inner first, then outer)
    if (isNearResizeHandleInner(imgX, imgY)) {
      setIsDraggingResizeInner(true);
      return;
    }
    if (isNearResizeHandle(imgX, imgY)) {
      setIsDraggingResize(true);
      return;
    }

    setIsDraggingDP(true);
    setLocalKx(imgX); setLocalKy(imgY);
    setRoiActive(true);
    setRoiCenterX(Math.round(Math.max(0, Math.min(detY - 1, imgX))));
    setRoiCenterY(Math.round(Math.max(0, Math.min(detX - 1, imgY))));
  };

  const handleDpMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = dpOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenX - dpPanX) / dpZoom;
    const imgY = (screenY - dpPanY) / dpZoom;

    // Handle inner resize dragging (annular mode)
    if (isDraggingResizeInner) {
      const dx = Math.abs(imgX - roiCenterX);
      const dy = Math.abs(imgY - roiCenterY);
      const newRadius = Math.sqrt(dx ** 2 + dy ** 2);
      // Inner radius must be less than outer radius
      setRoiRadiusInner(Math.max(1, Math.min(roiRadius - 1, Math.round(newRadius))));
      return;
    }

    // Handle outer resize dragging - use model state center, not local values
    if (isDraggingResize) {
      const dx = Math.abs(imgX - roiCenterX);
      const dy = Math.abs(imgY - roiCenterY);
      if (roiMode === "rect") {
        // For rectangle, update width and height independently
        setRoiWidth(Math.max(2, Math.round(dx * 2)));
        setRoiHeight(Math.max(2, Math.round(dy * 2)));
      } else {
        const newRadius = roiMode === "square" ? Math.max(dx, dy) : Math.sqrt(dx ** 2 + dy ** 2);
        // For annular mode, outer radius must be greater than inner radius
        const minRadius = roiMode === "annular" ? (roiRadiusInner || 0) + 1 : 1;
        setRoiRadius(Math.max(minRadius, Math.round(newRadius)));
      }
      return;
    }

    // Check hover state for resize handles
    if (!isDraggingDP) {
      setIsHoveringResizeInner(isNearResizeHandleInner(imgX, imgY));
      setIsHoveringResize(isNearResizeHandle(imgX, imgY));
      return;
    }

    setLocalKx(imgX); setLocalKy(imgY);
    setRoiCenterX(Math.round(Math.max(0, Math.min(detY - 1, imgX))));
    setRoiCenterY(Math.round(Math.max(0, Math.min(detX - 1, imgY))));
  };

  const handleDpMouseUp = () => { setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false); };
  const handleDpMouseLeave = () => { setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false); setIsHoveringResize(false); setIsHoveringResizeInner(false); };
  const handleDpDoubleClick = () => { setDpZoom(1); setDpPanX(0); setDpPanY(0); };

  const handleViMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - viPanY) / viZoom;
    const imgY = (screenX - viPanX) / viZoom;

    // Check if VI ROI mode is active - same logic as DP
    if (viRoiMode && viRoiMode !== "off") {
      // Check if clicking on resize handle
      if (isNearViRoiResizeHandle(imgX, imgY)) {
        setIsDraggingViRoiResize(true);
        return;
      }

      // Otherwise, move ROI center to click position (same as DP)
      setIsDraggingViRoi(true);
      setLocalViRoiCenterX(imgX);
      setLocalViRoiCenterY(imgY);
      setViRoiCenterX(Math.round(Math.max(0, Math.min(shapeX - 1, imgX))));
      setViRoiCenterY(Math.round(Math.max(0, Math.min(shapeY - 1, imgY))));
      return;
    }

    // Regular position selection (when ROI is off)
    setIsDraggingVI(true);
    setLocalPosX(imgX); setLocalPosY(imgY);
    setPosX(Math.round(Math.max(0, Math.min(shapeX - 1, imgX))));
    setPosY(Math.round(Math.max(0, Math.min(shapeY - 1, imgY))));
  };

  const handleViMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - viPanY) / viZoom;
    const imgY = (screenX - viPanX) / viZoom;

    // Handle VI ROI resize dragging (same pattern as DP)
    if (isDraggingViRoiResize) {
      const dx = Math.abs(imgX - localViRoiCenterX);
      const dy = Math.abs(imgY - localViRoiCenterY);
      if (viRoiMode === "rect") {
        setViRoiWidth(Math.max(2, Math.round(dy * 2)));
        setViRoiHeight(Math.max(2, Math.round(dx * 2)));
      } else if (viRoiMode === "square") {
        const newHalfSize = Math.max(dx, dy);
        setViRoiRadius(Math.max(1, Math.round(newHalfSize)));
      } else {
        // circle
        const newRadius = Math.sqrt(dx ** 2 + dy ** 2);
        setViRoiRadius(Math.max(1, Math.round(newRadius)));
      }
      return;
    }

    // Check hover state for resize handles (same as DP)
    if (!isDraggingViRoi) {
      setIsHoveringViRoiResize(isNearViRoiResizeHandle(imgX, imgY));
      if (viRoiMode && viRoiMode !== "off") return;  // Don't update position when ROI active
    }

    // Handle VI ROI center dragging (same as DP)
    if (isDraggingViRoi) {
      setLocalViRoiCenterX(imgX);
      setLocalViRoiCenterY(imgY);
      setViRoiCenterX(Math.round(Math.max(0, Math.min(shapeX - 1, imgX))));
      setViRoiCenterY(Math.round(Math.max(0, Math.min(shapeY - 1, imgY))));
      return;
    }

    // Handle regular position dragging (when ROI is off)
    if (!isDraggingVI) return;
    setLocalPosX(imgX); setLocalPosY(imgY);
    setPosX(Math.round(Math.max(0, Math.min(shapeX - 1, imgX))));
    setPosY(Math.round(Math.max(0, Math.min(shapeY - 1, imgY))));
  };

  const handleViMouseUp = () => {
    setIsDraggingVI(false);
    setIsDraggingViRoi(false);
    setIsDraggingViRoiResize(false);
  };
  const handleViMouseLeave = () => {
    setIsDraggingVI(false);
    setIsDraggingViRoi(false);
    setIsDraggingViRoiResize(false);
    setIsHoveringViRoiResize(false);
  };
  const handleViDoubleClick = () => { setViZoom(1); setViPanX(0); setViPanY(0); };
  const handleFftDoubleClick = () => { setFftZoom(1); setFftPanX(0); setFftPanY(0); };

  // FFT drag-to-pan handlers
  const handleFftMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDraggingFFT(true);
    setFftDragStart({ x: e.clientX, y: e.clientY, panX: fftPanX, panY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDraggingFFT || !fftDragStart) return;
    const canvas = fftOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const dx = (e.clientX - fftDragStart.x) * scaleX;
    const dy = (e.clientY - fftDragStart.y) * scaleY;
    setFftPanX(fftDragStart.panX + dx);
    setFftPanY(fftDragStart.panY + dy);
  };

  const handleFftMouseUp = () => { setIsDraggingFFT(false); setFftDragStart(null); };
  const handleFftMouseLeave = () => { setIsDraggingFFT(false); setFftDragStart(null); };

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  // Export DP handler
  const handleExportDP = async () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const zip = new JSZip();
    const metadata = {
      exported_at: new Date().toISOString(),
      type: "diffraction_pattern",
      scan_position: { x: posX, y: posY },
      scan_shape: { x: shapeX, y: shapeY },
      detector_shape: { x: detX, y: detY },
      roi: { mode: roiMode, center_x: roiCenterX, center_y: roiCenterY, radius_outer: roiRadius, radius_inner: roiRadiusInner },
      display: { colormap: dpColormap, vmin_pct: dpVminPct, vmax_pct: dpVmaxPct, scale_mode: dpScaleMode },
      calibration: { bf_radius: bfRadius, center_x: centerX, center_y: centerY, k_pixel_size: kPixelSize, k_calibrated: kCalibrated },
    };
    zip.file("metadata.json", JSON.stringify(metadata, null, 2));
    const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => new Promise((resolve) => canvas.toBlob((blob) => resolve(blob!), 'image/png'));
    if (dpCanvasRef.current) zip.file("diffraction_pattern.png", await canvasToBlob(dpCanvasRef.current));
    const zipBlob = await zip.generateAsync({ type: "blob" });
    const link = document.createElement('a');
    link.download = `dp_export_${timestamp}.zip`;
    link.href = URL.createObjectURL(zipBlob);
    link.click();
    URL.revokeObjectURL(link.href);
  };

  // Export VI handler
  const handleExportVI = async () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const zip = new JSZip();
    const metadata = {
      exported_at: new Date().toISOString(),
      scan_position: { x: posX, y: posY },
      scan_shape: { x: shapeX, y: shapeY },
      detector_shape: { x: detX, y: detY },
      roi: { mode: roiMode, center_x: roiCenterX, center_y: roiCenterY, radius_outer: roiRadius, radius_inner: roiRadiusInner },
      display: { dp_colormap: dpColormap, vi_colormap: viColormap, dp_scale_mode: dpScaleMode, vi_scale_mode: viScaleMode },
      calibration: { bf_radius: bfRadius, center_x: centerX, center_y: centerY, pixel_size: pixelSize, k_pixel_size: kPixelSize },
    };
    zip.file("metadata.json", JSON.stringify(metadata, null, 2));
    const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => new Promise((resolve) => canvas.toBlob((blob) => resolve(blob!), 'image/png'));
    if (virtualCanvasRef.current) zip.file("virtual_image.png", await canvasToBlob(virtualCanvasRef.current));
    if (dpCanvasRef.current) zip.file("diffraction_pattern.png", await canvasToBlob(dpCanvasRef.current));
    if (fftCanvasRef.current) zip.file("fft.png", await canvasToBlob(fftCanvasRef.current));
    const zipBlob = await zip.generateAsync({ type: "blob" });
    const link = document.createElement('a');
    link.download = `4dstem_export_${timestamp}.zip`;
    link.href = URL.createObjectURL(zipBlob);
    link.click();
    URL.revokeObjectURL(link.href);
  };

  // Common styles for panel control groups (fills parent width = canvas width)
  const panelControlStyle = {
    display: "flex",
    alignItems: "center",
    gap: `${SPACING.SM}px`,
    width: "100%",
    boxSizing: "border-box",
  };

  // Theme-aware select style
  const themedSelect = {
    ...controlPanel.select,
    bgcolor: themeColors.controlBg,
    color: themeColors.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: themeColors.accent },
  };

  return (
    <Box ref={rootRef} className="show4dstem-root" sx={{ p: `${SPACING.LG}px`, bgcolor: themeColors.bg, color: themeColors.text }}>
      {/* HEADER */}
      <Typography variant="h6" sx={{ ...typography.title, mb: `${SPACING.SM}px` }}>
        4D-STEM Explorer
      </Typography>

      {/* MAIN CONTENT: DP | VI | FFT (three columns when FFT shown) */}
      <Stack direction="row" spacing={`${SPACING.LG}px`}>
        {/* LEFT COLUMN: DP Panel */}
        <Box sx={{ width: CANVAS_SIZE }}>
          {/* DP Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typography.label }}>
              DP at ({Math.round(localPosX)}, {Math.round(localPosY)})
              <span style={{ color: "#0f0", marginLeft: SPACING.SM }}>k: ({Math.round(localKx)}, {Math.round(localKy)})</span>
              <InfoTooltip text="Diffraction Pattern: 2D detector image I(kx,ky) at scan position (x,y). The ROI mask M(kx,ky) defines which pixels are integrated for the virtual image. Drag to move ROI center, scroll to zoom, double-click to reset." theme={themeInfo.theme} />
            </Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`}>
              <Button size="small" sx={compactButton} disabled={dpZoom === 1 && dpPanX === 0 && dpPanY === 0 && roiCenterX === centerX && roiCenterY === centerY} onClick={() => { setDpZoom(1); setDpPanX(0); setDpPanY(0); setRoiCenterX(centerX); setRoiCenterY(centerY); }}>Reset</Button>
              <Button size="small" sx={compactButton} onClick={handleExportDP}>Export</Button>
            </Stack>
          </Stack>

          {/* DP Canvas */}
          <Box sx={{ ...container.imageBox, width: CANVAS_SIZE, height: CANVAS_SIZE }}>
            <canvas ref={dpCanvasRef} width={detY} height={detX} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={dpOverlayRef} width={detY} height={detX}
              onMouseDown={handleDpMouseDown} onMouseMove={handleDpMouseMove}
              onMouseUp={handleDpMouseUp} onMouseLeave={handleDpMouseLeave}
              onWheel={createZoomHandler(setDpZoom, setDpPanX, setDpPanY, dpZoom, dpPanX, dpPanY, dpOverlayRef)}
              onDoubleClick={handleDpDoubleClick}
              style={{ position: "absolute", width: "100%", height: "100%", cursor: isHoveringResize || isDraggingResize ? "nwse-resize" : "crosshair" }}
            />
            <canvas ref={dpUiRef} width={CANVAS_SIZE * DPR} height={CANVAS_SIZE * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
          </Box>

          {/* DP Stats Bar */}
          {dpStats && dpStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, borderRadius: "2px", display: "flex", gap: 2, alignItems: "center" }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(dpStats[3])}</Box></Typography>
              <Box sx={{ display: "flex", alignItems: "center", ml: "auto" }}>
                <Typography sx={{ fontSize: 10, color: themeColors.textMuted }}>Show hot px:<InfoTooltip text="Toggle hot pixel display. When OFF, clips display range to 99.99 percentile to exclude outlier pixels. Hot pixels are defective detector elements that show abnormally high values." theme={themeInfo.theme} /></Typography>
                <Switch checked={!(hotPixelFilter ?? true)} onChange={(e) => setHotPixelFilter(!e.target.checked)} size="small" sx={switchStyles.small} />
              </Box>
            </Box>
          )}

          {/* DP Controls - two rows with histogram on right */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
            {/* Left: two rows of controls */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              {/* Row 1: Detector + slider */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Detector:</Typography>
                <Select value={roiMode || "point"} onChange={(e) => setRoiMode(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={upwardMenuProps}>
                  <MenuItem value="point">Point</MenuItem>
                  <MenuItem value="circle">Circle</MenuItem>
                  <MenuItem value="square">Square</MenuItem>
                  <MenuItem value="rect">Rect</MenuItem>
                  <MenuItem value="annular">Annular</MenuItem>
                </Select>
                {(roiMode === "circle" || roiMode === "square" || roiMode === "annular") && (
                  <>
                    <Slider
                      value={roiMode === "annular" ? [roiRadiusInner, roiRadius] : [roiRadius]}
                      onChange={(_, v) => {
                        if (roiMode === "annular") {
                          const [inner, outer] = v as number[];
                          setRoiRadiusInner(Math.min(inner, outer - 1));
                          setRoiRadius(Math.max(outer, inner + 1));
                        } else {
                          setRoiRadius(v as number);
                        }
                      }}
                      min={1}
                      max={Math.min(detX, detY) / 2}
                      size="small"
                      sx={{
                        width: roiMode === "annular" ? 100 : 70,
                        mx: 1,
                        "& .MuiSlider-thumb": { width: 14, height: 14 }
                      }}
                    />
                    <Typography sx={{ ...typography.label, fontSize: 10 }}>
                      {roiMode === "annular" ? `${Math.round(roiRadiusInner)}-${Math.round(roiRadius)}px` : `${Math.round(roiRadius)}px`}
                    </Typography>
                  </>
                )}
              </Box>
              {/* Row 2: Presets + Color + Scale */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography component="span" onClick={() => { setRoiMode("circle"); setRoiRadius(bfRadius || 10); setRoiCenterX(centerX); setRoiCenterY(centerY); }} sx={{ color: "#4f4", fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>BF</Typography>
                <Typography component="span" onClick={() => { setRoiMode("annular"); setRoiRadiusInner((bfRadius || 10) * 0.5); setRoiRadius(bfRadius || 10); setRoiCenterX(centerX); setRoiCenterY(centerY); }} sx={{ color: "#4af", fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>ABF</Typography>
                <Typography component="span" onClick={() => { setRoiMode("annular"); setRoiRadiusInner(bfRadius || 10); setRoiRadius(Math.min((bfRadius || 10) * 3, Math.min(detX, detY) / 2 - 2)); setRoiCenterX(centerX); setRoiCenterY(centerY); }} sx={{ color: "#fa4", fontSize: 11, fontWeight: "bold", cursor: "pointer", "&:hover": { textDecoration: "underline" } }}>ADF</Typography>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                <Select value={dpColormap} onChange={(e) => setDpColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={upwardMenuProps}>
                  <MenuItem value="inferno">Inferno</MenuItem>
                  <MenuItem value="viridis">Viridis</MenuItem>
                  <MenuItem value="plasma">Plasma</MenuItem>
                  <MenuItem value="magma">Magma</MenuItem>
                  <MenuItem value="hot">Hot</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                <Select value={dpScaleMode} onChange={(e) => setDpScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={upwardMenuProps}>
                  <MenuItem value="linear">Lin</MenuItem>
                  <MenuItem value="log">Log</MenuItem>
                  <MenuItem value="power">Pow</MenuItem>
                </Select>
              </Box>
            </Box>
            {/* Right: Histogram spanning both rows */}
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
              <Histogram data={dpHistogramData} colormap={dpColormap} vminPct={dpVminPct} vmaxPct={dpVmaxPct} onRangeChange={(min, max) => { setDpVminPct(min); setDpVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={dpGlobalMin} dataMax={dpGlobalMax} />
            </Box>
          </Box>
        </Box>

        {/* SECOND COLUMN: VI Panel */}
        <Box sx={{ width: viCanvasWidth }}>
          {/* VI Header */}
          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
            <Typography variant="caption" sx={{ ...typography.label }}>Virtual Image<InfoTooltip text="Virtual Image: Integrated intensity within detector ROI at each scan position. Computed as Σ(DP × mask) for each (x,y). Double-click to select position." theme={themeInfo.theme} /></Typography>
            <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
              <Typography sx={{ ...typography.label, color: themeColors.textMuted, fontSize: 10 }}>
                {shapeX}×{shapeY} | {detX}×{detY}
              </Typography>
              <Typography sx={{ ...typography.label, fontSize: 10 }}>FFT:</Typography>
              <Switch checked={showFft} onChange={(e) => setShowFft(e.target.checked)} size="small" sx={switchStyles.small} />
              <Button size="small" sx={compactButton} disabled={viZoom === 1 && viPanX === 0 && viPanY === 0} onClick={() => { setViZoom(1); setViPanX(0); setViPanY(0); }}>Reset</Button>
              <Button size="small" sx={compactButton} onClick={handleExportVI}>Export</Button>
            </Stack>
          </Stack>

          {/* VI Canvas */}
          <Box sx={{ ...container.imageBox, width: viCanvasWidth, height: viCanvasHeight }}>
            <canvas ref={virtualCanvasRef} width={shapeY} height={shapeX} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={virtualOverlayRef} width={shapeY} height={shapeX}
              onMouseDown={handleViMouseDown} onMouseMove={handleViMouseMove}
              onMouseUp={handleViMouseUp} onMouseLeave={handleViMouseLeave}
              onWheel={createZoomHandler(setViZoom, setViPanX, setViPanY, viZoom, viPanX, viPanY, virtualOverlayRef)}
              onDoubleClick={handleViDoubleClick}
              style={{ position: "absolute", width: "100%", height: "100%", cursor: "crosshair" }}
            />
            <canvas ref={viUiRef} width={viCanvasWidth * DPR} height={viCanvasHeight * DPR} style={{ position: "absolute", width: "100%", height: "100%", pointerEvents: "none" }} />
          </Box>

          {/* VI Stats Bar */}
          {viStats && viStats.length === 4 && (
            <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, borderRadius: "2px", display: "flex", gap: 2 }}>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[0])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[1])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[2])}</Box></Typography>
              <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(viStats[3])}</Box></Typography>
            </Box>
          )}

          {/* VI Controls - Two rows with histogram on right */}
          <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
            {/* Left: Two rows of controls */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
              {/* Row 1: ROI selector */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>ROI:</Typography>
                <Select value={viRoiMode || "off"} onChange={(e) => setViRoiMode(e.target.value)} size="small" sx={{ ...themedSelect, minWidth: 60, fontSize: 10 }} MenuProps={upwardMenuProps}>
                  <MenuItem value="off">Off</MenuItem>
                  <MenuItem value="circle">Circle</MenuItem>
                  <MenuItem value="square">Square</MenuItem>
                  <MenuItem value="rect">Rect</MenuItem>
                </Select>
                {viRoiMode && viRoiMode !== "off" && (
                  <>
                    {(viRoiMode === "circle" || viRoiMode === "square") && (
                      <>
                        <Slider
                          value={viRoiRadius || 5}
                          onChange={(_, v) => setViRoiRadius(v as number)}
                          min={1}
                          max={Math.min(shapeX, shapeY) / 2}
                          size="small"
                          sx={{ width: 80, mx: 1 }}
                        />
                        <Typography sx={{ ...typography.value, fontSize: 10, minWidth: 30 }}>
                          {Math.round(viRoiRadius || 5)}px
                        </Typography>
                      </>
                    )}
                    {summedDpCount > 0 && (
                      <Typography sx={{ ...typography.label, fontSize: 9, color: "#a6f" }}>
                        {summedDpCount} pos
                      </Typography>
                    )}
                  </>
                )}
              </Box>
              {/* Row 2: Color + Scale */}
              <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                <Select value={viColormap} onChange={(e) => setViColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={upwardMenuProps}>
                  <MenuItem value="inferno">Inferno</MenuItem>
                  <MenuItem value="viridis">Viridis</MenuItem>
                  <MenuItem value="plasma">Plasma</MenuItem>
                  <MenuItem value="magma">Magma</MenuItem>
                  <MenuItem value="hot">Hot</MenuItem>
                  <MenuItem value="gray">Gray</MenuItem>
                </Select>
                <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                <Select value={viScaleMode} onChange={(e) => setViScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={upwardMenuProps}>
                  <MenuItem value="linear">Lin</MenuItem>
                  <MenuItem value="log">Log</MenuItem>
                  <MenuItem value="power">Pow</MenuItem>
                </Select>
              </Box>
            </Box>
            {/* Right: Histogram spanning both rows */}
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
              <Histogram data={viHistogramData} colormap={viColormap} vminPct={viVminPct} vmaxPct={viVmaxPct} onRangeChange={(min, max) => { setViVminPct(min); setViVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={viDataMin} dataMax={viDataMax} />
            </Box>
          </Box>
        </Box>

        {/* THIRD COLUMN: FFT Panel (conditionally shown) */}
        {showFft && (
          <Box sx={{ width: viCanvasWidth }}>
            {/* FFT Header */}
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 28 }}>
              <Typography variant="caption" sx={{ ...typography.label }}>FFT<InfoTooltip text="Fast Fourier Transform: Shows spatial frequency content of the virtual image. Center = low frequencies (large features), edges = high frequencies (fine detail). Useful for detecting periodic structures and scan artifacts." theme={themeInfo.theme} /></Typography>
              <Stack direction="row" spacing={`${SPACING.SM}px`} alignItems="center">
                <Button size="small" sx={compactButton} disabled={fftZoom === 1 && fftPanX === 0 && fftPanY === 0} onClick={() => { setFftZoom(1); setFftPanX(0); setFftPanY(0); }}>Reset</Button>
              </Stack>
            </Stack>

            {/* FFT Canvas */}
            <Box sx={{ ...container.imageBox, width: viCanvasWidth, height: viCanvasHeight }}>
              <canvas ref={fftCanvasRef} width={shapeY} height={shapeX} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
              <canvas
                ref={fftOverlayRef} width={shapeY} height={shapeX}
                onMouseDown={handleFftMouseDown} onMouseMove={handleFftMouseMove}
                onMouseUp={handleFftMouseUp} onMouseLeave={handleFftMouseLeave}
                onWheel={createZoomHandler(setFftZoom, setFftPanX, setFftPanY, fftZoom, fftPanX, fftPanY, fftOverlayRef)}
                onDoubleClick={handleFftDoubleClick}
                style={{ position: "absolute", width: "100%", height: "100%", cursor: isDraggingFFT ? "grabbing" : "grab" }}
              />
            </Box>

            {/* FFT Stats Bar */}
            {fftStats && fftStats.length === 4 && (
              <Box sx={{ mt: `${SPACING.XS}px`, px: 1, py: 0.5, bgcolor: themeColors.bgAlt, borderRadius: "2px", display: "flex", gap: 2 }}>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Mean <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats[0])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Min <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats[1])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Max <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats[2])}</Box></Typography>
                <Typography sx={{ fontSize: 11, color: themeColors.textMuted }}>Std <Box component="span" sx={{ color: themeColors.accent }}>{formatStat(fftStats[3])}</Box></Typography>
              </Box>
            )}

            {/* FFT Controls - Two rows with histogram on right */}
            <Box sx={{ mt: `${SPACING.SM}px`, display: "flex", gap: `${SPACING.SM}px`, width: "100%", boxSizing: "border-box" }}>
              {/* Left: Two rows of controls */}
              <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, flex: 1, justifyContent: "center" }}>
                {/* Row 1: Scale + Clip */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Scale:</Typography>
                  <Select value={fftScaleMode} onChange={(e) => setFftScaleMode(e.target.value as "linear" | "log" | "power")} size="small" sx={{ ...themedSelect, minWidth: 50, fontSize: 10 }} MenuProps={upwardMenuProps}>
                    <MenuItem value="linear">Lin</MenuItem>
                    <MenuItem value="log">Log</MenuItem>
                    <MenuItem value="power">Pow</MenuItem>
                  </Select>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Auto:<InfoTooltip text="Auto-enhance FFT display. When ON: (1) Masks DC component at center - DC = F(0,0) = Σ(image), replaced with average of 4 neighbors. (2) Clips display to 99.9 percentile to exclude outliers. When OFF: shows raw FFT with full dynamic range." theme={themeInfo.theme} /></Typography>
                  <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} size="small" sx={switchStyles.small} />
                </Box>
                {/* Row 2: Color */}
                <Box sx={{ ...controlRow, border: `1px solid ${themeColors.border}`, bgcolor: themeColors.controlBg }}>
                  <Typography sx={{ ...typography.label, fontSize: 10 }}>Color:</Typography>
                  <Select value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...themedSelect, minWidth: 65, fontSize: 10 }} MenuProps={upwardMenuProps}>
                    <MenuItem value="inferno">Inferno</MenuItem>
                    <MenuItem value="viridis">Viridis</MenuItem>
                    <MenuItem value="plasma">Plasma</MenuItem>
                    <MenuItem value="magma">Magma</MenuItem>
                    <MenuItem value="hot">Hot</MenuItem>
                    <MenuItem value="gray">Gray</MenuItem>
                  </Select>
                </Box>
              </Box>
              {/* Right: Histogram spanning both rows */}
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "center" }}>
                {fftHistogramData && (
                  <Histogram data={fftHistogramData} colormap={fftColormap} vminPct={fftVminPct} vmaxPct={fftVmaxPct} onRangeChange={(min, max) => { setFftVminPct(min); setFftVmaxPct(max); }} width={110} height={58} theme={themeInfo.theme} dataMin={fftDataMin} dataMax={fftDataMax} />
                )}
              </Box>
            </Box>
          </Box>
        )}
      </Stack>

      {/* BOTTOM CONTROLS - Path only (FFT toggle moved to VI panel) */}
      {pathLength > 0 && (
        <Stack direction="row" spacing={`${SPACING.MD}px`} sx={{ mt: `${SPACING.LG}px` }}>
          <Box className="show4dstem-control-group" sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px` }}>
            <Typography sx={{ ...typography.label }}>Path:</Typography>
            <Typography component="span" onClick={() => { setPathPlaying(false); setPathIndex(0); }} sx={{ color: themeColors.textMuted, fontSize: 14, cursor: "pointer", "&:hover": { color: "#fff" }, px: 0.5 }} title="Stop">⏹</Typography>
            <Typography component="span" onClick={() => setPathPlaying(!pathPlaying)} sx={{ color: pathPlaying ? "#0f0" : "#888", fontSize: 14, cursor: "pointer", "&:hover": { color: "#fff" }, px: 0.5 }} title={pathPlaying ? "Pause" : "Play"}>{pathPlaying ? "⏸" : "▶"}</Typography>
            <Typography sx={{ ...typography.value, minWidth: 60 }}>{pathIndex + 1}/{pathLength}</Typography>
            <Slider value={pathIndex} onChange={(_, v) => { setPathPlaying(false); setPathIndex(v as number); }} min={0} max={Math.max(0, pathLength - 1)} size="small" sx={{ width: 100 }} />
          </Box>
        </Stack>
      )}
    </Box>
  );
}

export const render = createRender(Show4DSTEM);
