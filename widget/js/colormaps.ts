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
  let px = u32(in.uv.x * f32(params.width));
  let py = u32(in.uv.y * f32(params.height));
  let idx = py * params.width + px;
  let packed = rgba[idx];
  let r = f32(packed & 0xFFu) / 255.0;
  let g = f32((packed >> 8u) & 0xFFu) / 255.0;
  let b = f32((packed >> 16u) & 0xFFu) / 255.0;
  return vec4f(r, g, b, 1.0);
}
`;

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
  histBinsBuffer: GPUBuffer;
  histReadBuffer: GPUBuffer;
  // Lazily allocated per-slot 16-byte buffer holding { vmin, vmax, _p0, _p1 }.
  // Populated by computeRange* on GPU and consumed directly by the range-aware
  // colormap shader (no CPU readback between passes).
  rangeBuffer: GPUBuffer | null;
  count: number;
  width: number;
  height: number;
};

export class GPUColormapEngine {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline | null = null;
  private blitPipeline: GPURenderPipeline | null = null;
  // Per-image GPU state: persistent buffers (data, rgba, read, params, histogram)
  private slots: GPUSlot[] = [];
  private lutBuffer: GPUBuffer | null = null;
  private currentLutName: string = "";

  constructor(device: GPUDevice) { this.device = device; }

  private ensurePipeline(): void {
    if (this.pipeline) return;
    const module = this.device.createShaderModule({ code: COLORMAP_SHADER });
    this.pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  /** Upload LUT to GPU (only when colormap name changes). */
  uploadLUT(lutName: string, lut: Uint8Array): void {
    if (this.currentLutName === lutName && this.lutBuffer) return;
    this.ensurePipeline();
    if (this.lutBuffer) this.lutBuffer.destroy();
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
  uploadData(idx: number, data: Float32Array, width?: number, height?: number): void {
    this.ensurePipeline();
    while (this.slots.length <= idx) this.slots.push(null as never);
    if (this.slots[idx]) {
      this.slots[idx].dataBuffer.destroy();
      this.slots[idx].rgbaBuffer.destroy();
      this.slots[idx].readBuffer.destroy();
      this.slots[idx].paramsBuffer.destroy();
      this.slots[idx].histBinsBuffer.destroy();
      this.slots[idx].histReadBuffer.destroy();
      this.slots[idx].rangeBuffer?.destroy();
    }
    // Validate dimensions — if width*height doesn't match data length, derive from sqrt
    // (catches stale closure values like width=1 from mount effects)
    const validDims = width && height && width > 1 && height > 1 && width * height === data.length;
    const w = validDims ? width : Math.round(Math.sqrt(data.length));
    const h = validDims ? height : Math.round(data.length / w);
    const byteSize = data.byteLength;
    const rgbaSize = data.length * 4;
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
    // Size 32 covers both the 24-byte colormap/histogram structs and the
    // 32-byte range-aware colormap struct (extra trailing bytes are unused
    // by the smaller shaders).
    const paramsBuffer = this.device.createBuffer({
      size: 32,
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
    this.slots[idx] = { dataBuffer, rgbaBuffer, readBuffer, paramsBuffer, histBinsBuffer, histReadBuffer, rangeBuffer: null, count: data.length, width: w, height: h };
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
      if (!slot) continue;
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
      if (!slot || !offscreens[k] || !imgDatas[k]) continue;
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
      if (!slot || !ctx) continue;
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
      if (!slot) { canvases.push(null as never); continue; }
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

  /**
   * Configure a canvas for WebGPU zero-copy rendering.
   * Returns the GPUCanvasContext, or null if WebGPU canvas is not supported.
   */
  configureCanvas(canvas: HTMLCanvasElement, width: number, height: number): GPUCanvasContext | null {
    try {
      const ctx = canvas.getContext("webgpu") as GPUCanvasContext | null;
      if (!ctx) return null;
      ctx.configure({
        device: this.device,
        format: navigator.gpu.getPreferredCanvasFormat(),
        alphaMode: "opaque",
      });
      canvas.width = width;
      canvas.height = height;
      return ctx;
    } catch {
      return null;
    }
  }

  /** Release all GPU resources. */
  destroy(): void {
    for (const slot of this.slots) {
      if (slot) {
        slot.dataBuffer.destroy();
        slot.rgbaBuffer.destroy();
        slot.readBuffer.destroy();
        slot.paramsBuffer.destroy();
        slot.histBinsBuffer.destroy();
        slot.histReadBuffer.destroy();
        slot.rangeBuffer?.destroy();
      }
    }
    this.slots = [];
    this.lutBuffer?.destroy();
    this.lutBuffer = null;
    this.currentLutName = "";
    for (const v of this.panelRgbaBuffers.values()) { v.rgba.destroy(); v.range.destroy(); }
    this.panelRgbaBuffers.clear();
  }

  /** Number of uploaded image slots. */
  get slotCount(): number { return this.slots.filter(s => s).length; }

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
      if (!slot || !slot.rangeBuffer) { canvases.push(null); continue; }
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
          { binding: 0, resource: { buffer: slot.paramsBuffer } },
          { binding: 1, resource: { buffer: slot.dataBuffer } },
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
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: slot.dataBuffer } },
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

let gpuColormapEngine: GPUColormapEngine | null = null;

/** Get or create the singleton GPU colormap engine. Returns null if WebGPU unavailable. */
export async function getGPUColormapEngine(): Promise<GPUColormapEngine | null> {
  if (gpuColormapEngine) return gpuColormapEngine;
  // Reuse the GPU device from fft
  try {
    const { getGPUDevice } = await import("./fft");
    const device = await getGPUDevice();
    if (!device) return null;
    gpuColormapEngine = new GPUColormapEngine(device);
    return gpuColormapEngine;
  } catch {
    return null;
  }
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
