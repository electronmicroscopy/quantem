/// <reference types="@webgpu/types" />

/**
 * WebGPU FFT Implementation
 * 
 * Implements Cooley-Tukey radix-2 FFT using WebGPU compute shaders.
 * Supports 1D and 2D FFT with forward and inverse transforms.
 */

// WGSL Shader for FFT butterfly operations
const FFT_SHADER = /* wgsl */`
// Complex number operations
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}

// Twiddle factor: e^(-2πi * k / N) for forward, e^(2πi * k / N) for inverse
fn twiddle(k: u32, N: u32, inverse: f32) -> vec2<f32> {
  let angle = inverse * 2.0 * 3.14159265359 * f32(k) / f32(N);
  return vec2<f32>(cos(angle), sin(angle));
}

// Bit reversal for index
fn bitReverse(x: u32, log2N: u32) -> u32 {
  var result: u32 = 0u;
  var val = x;
  for (var i: u32 = 0u; i < log2N; i = i + 1u) {
    result = (result << 1u) | (val & 1u);
    val = val >> 1u;
  }
  return result;
}

struct FFTParams {
  N: u32,           // FFT size
  log2N: u32,       // log2(N)
  stage: u32,       // Current butterfly stage
  inverse: f32,     // -1.0 for forward, 1.0 for inverse
}

@group(0) @binding(0) var<uniform> params: FFTParams;
@group(0) @binding(1) var<storage, read_write> data: array<vec2<f32>>;

// Bit-reversal permutation kernel
@compute @workgroup_size(256)
fn bitReversePermute(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  
  let rev = bitReverse(idx, params.log2N);
  if (idx < rev) {
    let temp = data[idx];
    data[idx] = data[rev];
    data[rev] = temp;
  }
}

// Butterfly operation kernel for one stage
@compute @workgroup_size(256)
fn butterflyStage(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N / 2u) { return; }
  
  let stage = params.stage;
  let halfSize = 1u << stage;           // 2^stage
  let fullSize = halfSize << 1u;        // 2^(stage+1)
  
  let group = idx / halfSize;
  let pos = idx % halfSize;
  
  let i = group * fullSize + pos;
  let j = i + halfSize;
  
  let w = twiddle(pos, fullSize, params.inverse);
  
  let u = data[i];
  let t = cmul(w, data[j]);
  
  data[i] = u + t;
  data[j] = u - t;
}

// Normalization for inverse FFT
@compute @workgroup_size(256)
fn normalize(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.N) { return; }
  
  let scale = 1.0 / f32(params.N);
  data[idx] = data[idx] * scale;
}
`;

// 2D FFT Shader (row-wise and column-wise transforms)
const FFT_2D_SHADER = /* wgsl */`
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x
  );
}

fn twiddle(k: u32, N: u32, inverse: f32) -> vec2<f32> {
  let angle = inverse * 2.0 * 3.14159265359 * f32(k) / f32(N);
  return vec2<f32>(cos(angle), sin(angle));
}

fn bitReverse(x: u32, log2N: u32) -> u32 {
  var result: u32 = 0u;
  var val = x;
  for (var i: u32 = 0u; i < log2N; i = i + 1u) {
    result = (result << 1u) | (val & 1u);
    val = val >> 1u;
  }
  return result;
}

struct FFT2DParams {
  width: u32,
  height: u32,
  log2Size: u32,
  stage: u32,
  inverse: f32,
  isRowWise: u32,  // 1 for row-wise, 0 for column-wise
}

@group(0) @binding(0) var<uniform> params: FFT2DParams;
@group(0) @binding(1) var<storage, read_write> data: array<vec2<f32>>;

// Get linear index for 2D data
fn getIndex(row: u32, col: u32) -> u32 {
  return row * params.width + col;
}

// Bit-reversal for rows
@compute @workgroup_size(16, 16)
fn bitReverseRows(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= params.height || col >= params.width) { return; }
  
  let rev = bitReverse(col, params.log2Size);
  if (col < rev) {
    let idx1 = getIndex(row, col);
    let idx2 = getIndex(row, rev);
    let temp = data[idx1];
    data[idx1] = data[idx2];
    data[idx2] = temp;
  }
}

// Bit-reversal for columns
@compute @workgroup_size(16, 16)
fn bitReverseCols(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= params.height || col >= params.width) { return; }
  
  let rev = bitReverse(row, params.log2Size);
  if (row < rev) {
    let idx1 = getIndex(row, col);
    let idx2 = getIndex(rev, col);
    let temp = data[idx1];
    data[idx1] = data[idx2];
    data[idx2] = temp;
  }
}

// Butterfly for rows
@compute @workgroup_size(16, 16)
fn butterflyRows(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let idx = gid.x;
  if (row >= params.height || idx >= params.width / 2u) { return; }
  
  let stage = params.stage;
  let halfSize = 1u << stage;
  let fullSize = halfSize << 1u;
  
  let group = idx / halfSize;
  let pos = idx % halfSize;
  
  let col_i = group * fullSize + pos;
  let col_j = col_i + halfSize;
  
  if (col_j >= params.width) { return; }
  
  let w = twiddle(pos, fullSize, params.inverse);
  
  let i = getIndex(row, col_i);
  let j = getIndex(row, col_j);
  
  let u = data[i];
  let t = cmul(w, data[j]);
  
  data[i] = u + t;
  data[j] = u - t;
}

// Butterfly for columns
@compute @workgroup_size(16, 16)
fn butterflyCols(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let idx = gid.y;
  if (col >= params.width || idx >= params.height / 2u) { return; }
  
  let stage = params.stage;
  let halfSize = 1u << stage;
  let fullSize = halfSize << 1u;
  
  let group = idx / halfSize;
  let pos = idx % halfSize;
  
  let row_i = group * fullSize + pos;
  let row_j = row_i + halfSize;
  
  if (row_j >= params.height) { return; }
  
  let w = twiddle(pos, fullSize, params.inverse);
  
  let i = getIndex(row_i, col);
  let j = getIndex(row_j, col);
  
  let u = data[i];
  let t = cmul(w, data[j]);
  
  data[i] = u + t;
  data[j] = u - t;
}

// Normalization for inverse 2D FFT
@compute @workgroup_size(16, 16)
fn normalize2D(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= params.height || col >= params.width) { return; }
  
  let idx = getIndex(row, col);
  let scale = 1.0 / f32(params.width * params.height);
  data[idx] = data[idx] * scale;
}
`;

/**
 * Get next power of 2 >= n
 */
function nextPow2(n: number): number {
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

/**
 * WebGPU FFT class for 1D and 2D transforms
 */
export class WebGPUFFT {
  private device: GPUDevice;
  private pipelines1D: {
    bitReverse: GPUComputePipeline;
    butterfly: GPUComputePipeline;
    normalize: GPUComputePipeline;
  } | null = null;
  private pipelines2D: {
    bitReverseRows: GPUComputePipeline;
    bitReverseCols: GPUComputePipeline;
    butterflyRows: GPUComputePipeline;
    butterflyCols: GPUComputePipeline;
    normalize: GPUComputePipeline;
  } | null = null;
  private initialized = false;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  async init(): Promise<void> {
    if (this.initialized) return;

    // Create 1D FFT pipelines
    const module1D = this.device.createShaderModule({ code: FFT_SHADER });

    this.pipelines1D = {
      bitReverse: this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: module1D, entryPoint: 'bitReversePermute' }
      }),
      butterfly: this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: module1D, entryPoint: 'butterflyStage' }
      }),
      normalize: this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: module1D, entryPoint: 'normalize' }
      })
    };

    // Create 2D FFT pipelines
    const module2D = this.device.createShaderModule({ code: FFT_2D_SHADER });

    this.pipelines2D = {
      bitReverseRows: this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: module2D, entryPoint: 'bitReverseRows' }
      }),
      bitReverseCols: this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: module2D, entryPoint: 'bitReverseCols' }
      }),
      butterflyRows: this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: module2D, entryPoint: 'butterflyRows' }
      }),
      butterflyCols: this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: module2D, entryPoint: 'butterflyCols' }
      }),
      normalize: this.device.createComputePipeline({
        layout: 'auto',
        compute: { module: module2D, entryPoint: 'normalize2D' }
      })
    };

    this.initialized = true;
    console.log('WebGPU FFT initialized');
  }

  /**
   * Perform 2D FFT - supports ANY size via automatic zero-padding
   */
  async fft2D(
    realData: Float32Array,
    imagData: Float32Array,
    width: number,
    height: number,
    inverse: boolean = false
  ): Promise<{ real: Float32Array, imag: Float32Array }> {
    await this.init();

    // Compute padded power-of-2 dimensions
    const paddedWidth = nextPow2(width);
    const paddedHeight = nextPow2(height);
    const needsPadding = paddedWidth !== width || paddedHeight !== height;

    const log2Width = Math.log2(paddedWidth);
    const log2Height = Math.log2(paddedHeight);

    const paddedSize = paddedWidth * paddedHeight;
    const originalSize = width * height;

    // Zero-pad input if needed
    let workReal: Float32Array;
    let workImag: Float32Array;

    if (needsPadding) {
      workReal = new Float32Array(paddedSize);
      workImag = new Float32Array(paddedSize);
      // Copy original data into top-left corner
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const srcIdx = y * width + x;
          const dstIdx = y * paddedWidth + x;
          workReal[dstIdx] = realData[srcIdx];
          workImag[dstIdx] = imagData[srcIdx];
        }
      }
    } else {
      workReal = realData;
      workImag = imagData;
    }

    const size = paddedSize;

    // Interleave real and imaginary (use padded work arrays)
    const complexData = new Float32Array(size * 2);
    for (let i = 0; i < size; i++) {
      complexData[i * 2] = workReal[i];
      complexData[i * 2 + 1] = workImag[i];
    }

    // Create buffers
    const dataBuffer = this.device.createBuffer({
      size: complexData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(dataBuffer, 0, complexData);

    const paramsBuffer = this.device.createBuffer({
      size: 24, // 6 x u32/f32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const readBuffer = this.device.createBuffer({
      size: complexData.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const inverseVal = inverse ? 1.0 : -1.0;
    const workgroupsX = Math.ceil(paddedWidth / 16);
    const workgroupsY = Math.ceil(paddedHeight / 16);

    // Helper to run a pass
    const runPass = (pipeline: GPUComputePipeline) => {
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuffer } },
          { binding: 1, resource: { buffer: dataBuffer } },
        ]
      });

      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroupsX, workgroupsY);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    };

    // Row-wise FFT (use padded dimensions)
    const params = new ArrayBuffer(24);
    const paramsU32 = new Uint32Array(params);
    const paramsF32 = new Float32Array(params);
    paramsU32[0] = paddedWidth;
    paramsU32[1] = paddedHeight;
    paramsU32[2] = log2Width;
    paramsU32[3] = 0;
    paramsF32[4] = inverseVal;
    paramsU32[5] = 1;
    this.device.queue.writeBuffer(paramsBuffer, 0, params);
    runPass(this.pipelines2D!.bitReverseRows);

    for (let stage = 0; stage < log2Width; stage++) {
      paramsU32[3] = stage;
      this.device.queue.writeBuffer(paramsBuffer, 0, params);
      runPass(this.pipelines2D!.butterflyRows);
    }

    // Column-wise FFT
    paramsU32[2] = log2Height;
    paramsU32[3] = 0;
    paramsU32[5] = 0;
    this.device.queue.writeBuffer(paramsBuffer, 0, params);
    runPass(this.pipelines2D!.bitReverseCols);

    for (let stage = 0; stage < log2Height; stage++) {
      paramsU32[3] = stage;
      this.device.queue.writeBuffer(paramsBuffer, 0, params);
      runPass(this.pipelines2D!.butterflyCols);
    }

    if (inverse) {
      runPass(this.pipelines2D!.normalize);
    }

    // Read back results
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, complexData.byteLength);
    this.device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    // Cleanup GPU buffers
    dataBuffer.destroy();
    paramsBuffer.destroy();
    readBuffer.destroy();

    // Deinterleave and crop back to original size if needed
    if (needsPadding) {
      const realResult = new Float32Array(originalSize);
      const imagResult = new Float32Array(originalSize);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const srcIdx = y * paddedWidth + x;
          const dstIdx = y * width + x;
          realResult[dstIdx] = result[srcIdx * 2];
          imagResult[dstIdx] = result[srcIdx * 2 + 1];
        }
      }
      return { real: realResult, imag: imagResult };
    } else {
      const realResult = new Float32Array(size);
      const imagResult = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        realResult[i] = result[i * 2];
        imagResult[i] = result[i * 2 + 1];
      }
      return { real: realResult, imag: imagResult };
    }
  }

  destroy(): void {
    this.initialized = false;
  }
}

// Singleton instance
let gpuFFT: WebGPUFFT | null = null;
let gpuDevice: GPUDevice | null = null;
let gpuInfo = "GPU";

/**
 * Initialize WebGPU and get FFT instance
 */
export async function getWebGPUFFT(): Promise<WebGPUFFT | null> {
  if (gpuFFT) return gpuFFT;

  if (!navigator.gpu) {
    console.warn('WebGPU not supported, falling back to CPU FFT');
    return null;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.warn('No GPU adapter found');
      return null;
    }

    // Attempt to get GPU info
    try {
      // In modern browsers, we can request adapter info
      // @ts-ignore - requestAdapterInfo is not yet in all type definitions
      const info = await adapter.requestAdapterInfo?.();
      if (info) {
        // Prioritize 'description' which usually has the full name (e.g. "NVIDIA GeForce RTX 4090")
        // Fallback to vendor/device if description is missing
        gpuInfo = info.description ||
          `${info.vendor} ${info.architecture || ""} ${info.device || ""}`.trim() ||
          "Generic WebGPU Adapter";
      }
    } catch (e) {
      console.log("Could not get detailed adapter info", e);
    }

    gpuDevice = await adapter.requestDevice();
    gpuFFT = new WebGPUFFT(gpuDevice);
    await gpuFFT.init();

    console.log(`🚀 WebGPU FFT ready on ${gpuInfo}!`);
    return gpuFFT;
  } catch (e) {
    console.warn('WebGPU init failed:', e);
    return null;
  }
}

/**
 * Get current GPU info string
 */
export function getGPUInfo(): string {
  return gpuInfo;
}

export default WebGPUFFT;
