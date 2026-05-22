/// <reference types="@webgpu/types" />
/**
 * WebGPU Volume Renderer — ray-casting with slice plane indicators.
 * Standalone module following the pattern of webgpu-fft.ts.
 */

import { getGPUDevice } from "./fft";

// ============================================================================
// Types
// ============================================================================

export interface VolumeRenderParams {
  sliceX: number;  // 0..nx-1 (current slice positions for plane indicators)
  sliceY: number;  // 0..ny-1
  sliceZ: number;  // 0..nz-1
  nx: number;
  ny: number;
  nz: number;
  opacity: number;       // global opacity multiplier 0..1
  brightness: number;    // brightness adjustment 0.1..3
  showSlicePlanes: boolean;  // toggle slice plane indicators
  slicePlaneOpacity: number; // slice plane alpha 0..1 (default 0.35)
  vmin: number;  // 0..1 normalized (maps to texture's [0,1] range)
  vmax: number;  // 0..1 normalized
}

export interface CameraState {
  yaw: number;       // radians, horizontal rotation
  pitch: number;     // radians, vertical rotation (clamped ±89°)
  roll?: number;     // radians, screen-plane rotation around the view direction
  distance: number;  // camera distance from volume center
  panX: number;      // horizontal pan
  panY: number;      // vertical pan
}

export const DEFAULT_CAMERA: CameraState = {
  yaw: Math.PI / 6,     // 30°
  pitch: Math.PI / 8,   // 22.5°
  roll: 0,
  distance: 1.8,
  panX: 0,
  panY: 0,
};

// ============================================================================
// Matrix math (column-major Float32Array[16])
// ============================================================================

function mat4Identity(): Float32Array {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

function mat4Inverse(m: Float32Array): Float32Array {
  const inv = new Float32Array(16);
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-10) return mat4Identity();
  det = 1.0 / det;

  inv[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  inv[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  inv[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  inv[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  inv[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  inv[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  inv[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  inv[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  inv[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  inv[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  inv[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  inv[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  inv[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  inv[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  inv[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  inv[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return inv;
}

function lookAt(
  eyeX: number, eyeY: number, eyeZ: number,
  centerX: number, centerY: number, centerZ: number,
  upX: number, upY: number, upZ: number,
): Float32Array {
  let fx = centerX - eyeX, fy = centerY - eyeY, fz = centerZ - eyeZ;
  const fLen = Math.sqrt(fx * fx + fy * fy + fz * fz);
  fx /= fLen; fy /= fLen; fz /= fLen;

  // side = forward × up
  let sx = fy * upZ - fz * upY, sy = fz * upX - fx * upZ, sz = fx * upY - fy * upX;
  const sLen = Math.sqrt(sx * sx + sy * sy + sz * sz);
  sx /= sLen; sy /= sLen; sz /= sLen;

  // recomputed up = side × forward
  const ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;

  const m = new Float32Array(16);
  m[0] = sx;  m[1] = ux;  m[2] = -fx; m[3] = 0;
  m[4] = sy;  m[5] = uy;  m[6] = -fy; m[7] = 0;
  m[8] = sz;  m[9] = uz;  m[10] = -fz; m[11] = 0;
  m[12] = -(sx * eyeX + sy * eyeY + sz * eyeZ);
  m[13] = -(ux * eyeX + uy * eyeY + uz * eyeZ);
  m[14] = (fx * eyeX + fy * eyeY + fz * eyeZ);
  m[15] = 1;
  return m;
}

// OpenGL-style perspective (z maps to [-1, 1]). We use this unchanged from
// the WebGL version because the ray reconstruction in the fragment shader
// unprojections at z=-1 and z=1 regardless of WebGPU's [0,1] clip range.
// The fullscreen triangle sits at z=0 which is within WebGPU's [0,1] range,
// and no depth buffer is used, so there's no clipping issue.
function perspective(fov: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fov / 2);
  const rangeInv = 1.0 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * rangeInv;
  m[11] = -1;
  m[14] = 2 * far * near * rangeInv;
  return m;
}

function orthographic(left: number, right: number, bottom: number, top: number, near: number, far: number): Float32Array {
  const lr = 1 / (left - right);
  const bt = 1 / (bottom - top);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = -2 * lr;
  m[5] = -2 * bt;
  m[10] = 2 * nf;
  m[12] = (left + right) * lr;
  m[13] = (top + bottom) * bt;
  m[14] = (far + near) * nf;
  m[15] = 1;
  return m;
}

// ============================================================================
// WGSL Ray-Casting Shader
// ============================================================================

const VOLUME_SHADER = /* wgsl */`
struct Uniforms {
  invViewProj: mat4x4<f32>,
  cameraPos: vec3<f32>,
  _pad0: f32,
  aspectRatio: vec3<f32>,
  _pad1: f32,
  bgColor: vec4<f32>,
  sliceX: f32,
  sliceY: f32,
  sliceZ: f32,
  opacity: f32,
  brightness: f32,
  numSteps: u32,
  showSlicePlanes: u32,
  vmin: f32,
  vmax: f32,
  slicePlaneOpacity: f32,
  _pad4: f32,
  _pad5: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var volume: texture_3d<f32>;
@group(0) @binding(2) var volumeSampler: sampler;
@group(0) @binding(3) var colormap: texture_2d<f32>;
@group(0) @binding(4) var colormapSampler: sampler;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let x = f32((vertexIndex & 1u) << 2u) - 1.0;
  let y = f32((vertexIndex & 2u) << 1u) - 1.0;
  var out: VertexOutput;
  out.uv = vec2<f32>(x, y) * 0.5 + 0.5;
  out.position = vec4<f32>(x, y, 0.5, 1.0);
  return out;
}

fn intersectBox(origin: vec3<f32>, dir: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
  let invDir = 1.0 / dir;
  let t1 = (bmin - origin) * invDir;
  let t2 = (bmax - origin) * invDir;
  let tmin = min(t1, t2);
  let tmax = max(t1, t2);
  let tNear = max(max(tmin.x, tmin.y), tmin.z);
  let tFar = min(min(tmax.x, tmax.y), tmax.z);
  return vec2<f32>(tNear, tFar);
}

fn worldToTex(p: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec3<f32> {
  return (p - bmin) / (bmax - bmin);
}

fn intersectSlicePlane(origin: vec3<f32>, dir: vec3<f32>, axis: i32, pos: f32,
                       bmin: vec3<f32>, bmax: vec3<f32>) -> f32 {
  var worldPos: f32;
  var dirComponent: f32;
  var originComponent: f32;
  if (axis == 0) {
    worldPos = bmin.x + pos * (bmax.x - bmin.x);
    dirComponent = dir.x;
    originComponent = origin.x;
  } else if (axis == 1) {
    worldPos = bmin.y + pos * (bmax.y - bmin.y);
    dirComponent = dir.y;
    originComponent = origin.y;
  } else {
    worldPos = bmin.z + pos * (bmax.z - bmin.z);
    dirComponent = dir.z;
    originComponent = origin.z;
  }
  if (abs(dirComponent) < 1e-8) { return -1.0; }
  let t = (worldPos - originComponent) / dirComponent;
  if (t < 0.0) { return -1.0; }
  let p = origin + t * dir;
  if (axis != 0 && (p.x < bmin.x || p.x > bmax.x)) { return -1.0; }
  if (axis != 1 && (p.y < bmin.y || p.y > bmax.y)) { return -1.0; }
  if (axis != 2 && (p.z < bmin.z || p.z > bmax.z)) { return -1.0; }
  return t;
}

fn applyWindow(value: f32) -> f32 {
  let denom = max(u.vmax - u.vmin, 1e-6);
  return clamp((value - u.vmin) / denom, 0.0, 1.0);
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  // Reconstruct ray from clip space — OpenGL convention z in [-1, 1]
  let ndc = uv * 2.0 - 1.0;
  var worldNear = u.invViewProj * vec4<f32>(ndc, -1.0, 1.0);
  var worldFar = u.invViewProj * vec4<f32>(ndc, 1.0, 1.0);
  worldNear = worldNear / worldNear.w;
  worldFar = worldFar / worldFar.w;

  let rayOrigin = worldNear.xyz;
  let rayDir = normalize(worldFar.xyz - worldNear.xyz);

  let halfExt = u.aspectRatio * 0.5;
  let bmin = -halfExt;
  let bmax = halfExt;

  let tHit = intersectBox(rayOrigin, rayDir, bmin, bmax);
  let tNear = tHit.x;
  let tFar = tHit.y;
  if (tNear > tFar || tFar < 0.0) {
    return u.bgColor;
  }

  let tStart = max(tNear, 0.0);
  let stepSize = (tFar - tStart) / f32(u.numSteps);

  // Compute slice plane intersections (only if enabled)
  var tSliceXY: f32 = -1.0;
  var tSliceXZ: f32 = -1.0;
  var tSliceYZ: f32 = -1.0;
  if (u.showSlicePlanes != 0u) {
    tSliceXY = intersectSlicePlane(rayOrigin, rayDir, 2, u.sliceZ, bmin, bmax);
    tSliceXZ = intersectSlicePlane(rayOrigin, rayDir, 1, u.sliceY, bmin, bmax);
    tSliceYZ = intersectSlicePlane(rayOrigin, rayDir, 0, u.sliceX, bmin, bmax);
  }

  // Front-to-back compositing
  var accum = vec4<f32>(0.0);

  for (var i: u32 = 0u; i < 512u; i = i + 1u) {
    if (i >= u.numSteps) { break; }

    let t = tStart + (f32(i) + 0.5) * stepSize;
    let pos = rayOrigin + t * rayDir;
    let texCoord = worldToTex(pos, bmin, bmax);

    // Composite slice planes at their depth (before volume at this step)
    // XY plane (blue)
    if (tSliceXY > 0.0 && abs(t - tSliceXY) < stepSize * 0.6) {
      let slicePos = rayOrigin + tSliceXY * rayDir;
      let sliceTex = worldToTex(slicePos, bmin, bmax);
      var sliceValXY = textureSampleLevel(volume, volumeSampler, sliceTex, 0.0).r;
      sliceValXY = applyWindow(sliceValXY);
      var sliceCol = textureSampleLevel(colormap, colormapSampler, vec2<f32>(clamp(sliceValXY * u.brightness, 0.0, 1.0), 0.5), 0.0).rgb;
      sliceCol = mix(sliceCol, vec3<f32>(0.3, 0.5, 1.0), 0.25);
      let sliceAlpha = u.slicePlaneOpacity * (1.0 - accum.a);
      accum = vec4<f32>(accum.rgb + sliceCol * sliceAlpha, accum.a + sliceAlpha);
      tSliceXY = -1.0;
    }
    // XZ plane (green)
    if (tSliceXZ > 0.0 && abs(t - tSliceXZ) < stepSize * 0.6) {
      let slicePos = rayOrigin + tSliceXZ * rayDir;
      let sliceTex = worldToTex(slicePos, bmin, bmax);
      var sliceValXZ = textureSampleLevel(volume, volumeSampler, sliceTex, 0.0).r;
      sliceValXZ = applyWindow(sliceValXZ);
      var sliceCol = textureSampleLevel(colormap, colormapSampler, vec2<f32>(clamp(sliceValXZ * u.brightness, 0.0, 1.0), 0.5), 0.0).rgb;
      sliceCol = mix(sliceCol, vec3<f32>(0.3, 1.0, 0.4), 0.25);
      let sliceAlpha = u.slicePlaneOpacity * (1.0 - accum.a);
      accum = vec4<f32>(accum.rgb + sliceCol * sliceAlpha, accum.a + sliceAlpha);
      tSliceXZ = -1.0;
    }
    // YZ plane (red)
    if (tSliceYZ > 0.0 && abs(t - tSliceYZ) < stepSize * 0.6) {
      let slicePos = rayOrigin + tSliceYZ * rayDir;
      let sliceTex = worldToTex(slicePos, bmin, bmax);
      var sliceValYZ = textureSampleLevel(volume, volumeSampler, sliceTex, 0.0).r;
      sliceValYZ = applyWindow(sliceValYZ);
      var sliceCol = textureSampleLevel(colormap, colormapSampler, vec2<f32>(clamp(sliceValYZ * u.brightness, 0.0, 1.0), 0.5), 0.0).rgb;
      sliceCol = mix(sliceCol, vec3<f32>(1.0, 0.3, 0.3), 0.25);
      let sliceAlpha = u.slicePlaneOpacity * (1.0 - accum.a);
      accum = vec4<f32>(accum.rgb + sliceCol * sliceAlpha, accum.a + sliceAlpha);
      tSliceYZ = -1.0;
    }

    // Sample volume — remap from [vmin, vmax] to [0, 1]
    var intensity = textureSampleLevel(volume, volumeSampler, texCoord, 0.0).r;
    intensity = applyWindow(intensity);
    intensity = clamp(intensity * u.brightness, 0.0, 1.0);

    // Colormap lookup
    let color = textureSampleLevel(colormap, colormapSampler, vec2<f32>(intensity, 0.5), 0.0).rgb;

    // Transfer function: opacity proportional to intensity
    let alpha = intensity * u.opacity * stepSize * 10.0;

    // Front-to-back compositing (emission-absorption)
    accum = vec4<f32>(
      accum.rgb + (1.0 - accum.a) * color * alpha,
      accum.a + (1.0 - accum.a) * alpha
    );

    if (accum.a > 0.95) { break; }
  }

  // Blend with background
  return vec4<f32>(accum.rgb + u.bgColor.rgb * (1.0 - accum.a), 1.0);
}
`;

// ============================================================================
// Uniform buffer layout (WGSL uniform alignment rules)
// ============================================================================
// offset  field              type               bytes
// 0       invViewProj        mat4x4<f32>        64
// 64      cameraPos          vec3<f32>          12
// 76      _pad0              f32                 4
// 80      aspectRatio        vec3<f32>          12
// 92      _pad1              f32                 4
// 96      bgColor            vec4<f32>          16
// 112     sliceX             f32                 4
// 116     sliceY             f32                 4
// 120     sliceZ             f32                 4
// 124     opacity            f32                 4
// 128     brightness         f32                 4
// 132     numSteps           u32                 4
// 136     showSlicePlanes    u32                 4
// 140     vmin               f32                 4
// 144     vmax               f32                 4
// 148     slicePlaneOpacity  f32                 4
// 152-159 padding            2×f32               8
// total: 160 bytes (must be multiple of 16)

const UNIFORM_BUFFER_SIZE = 160;

// ============================================================================
// VolumeRenderer class
// ============================================================================

export class VolumeRenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private canvasFormat: GPUTextureFormat;
  private pipeline: GPURenderPipeline;
  private volumeTexture: GPUTexture;
  private colormapTexture: GPUTexture;
  private uniformBuffer: GPUBuffer;
  private sampler: GPUSampler;
  private bindGroupLayout: GPUBindGroupLayout;
  private bindGroup: GPUBindGroup | null = null;
  private aspectRatio: [number, number, number] = [1, 1, 1];
  private canvas: HTMLCanvasElement;
  private deviceLost: boolean = false;

  static isSupported(): boolean {
    return typeof navigator !== "undefined" && !!navigator.gpu;
  }

  static async create(canvas: HTMLCanvasElement): Promise<VolumeRenderer> {
    const device = await getGPUDevice();
    if (!device) throw new Error("WebGPU not available");
    return new VolumeRenderer(device, canvas);
  }

  private constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.device = device;
    this.canvas = canvas;

    // Mark renderer dead on device loss so render() early-returns instead of
    // crashing on a dead handle. Matches the pattern used in fft.ts.
    device.lost.then((info) => {
      this.deviceLost = true;
      console.warn("VolumeRenderer: WebGPU device lost", info?.reason, info?.message);
    });

    // Configure canvas context
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("WebGPU canvas context not available");
    this.context = context;
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format: this.canvasFormat,
      alphaMode: "opaque",
    });

    // Create sampler (shared for volume + colormap)
    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      addressModeW: "clamp-to-edge",
    });

    // Create uniform buffer
    this.uniformBuffer = device.createBuffer({
      size: UNIFORM_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create placeholder textures (will be replaced by uploadVolume/uploadColormap)
    this.volumeTexture = device.createTexture({
      dimension: "3d",
      size: [1, 1, 1],
      format: "r8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.colormapTexture = device.createTexture({
      dimension: "2d",
      size: [256, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Create bind group layout
    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "3d" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      ],
    });

    // Create render pipeline (no MSAA — render directly to canvas)
    const shaderModule = device.createShaderModule({ code: VOLUME_SHADER });
    shaderModule.getCompilationInfo().then(info => {
      for (const msg of info.messages) {
        const level = msg.type === "error" ? "error" : msg.type === "warning" ? "warn" : "info";
        console[level](`WGSL ${msg.type} [${msg.lineNum}:${msg.linePos}]: ${msg.message}`);
      }
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    this.pipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: "vs_main" },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format: this.canvasFormat }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.rebuildBindGroup();
  }

  private rebuildBindGroup(): void {
    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.volumeTexture.createView() },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: this.colormapTexture.createView() },
        { binding: 4, resource: this.sampler },
      ],
    });
  }

  uploadVolume(data: Float32Array, nx: number, ny: number, nz: number): void {
    if (this.deviceLost) return;
    // Normalize to [0,255] uint8 — R8 always supports LINEAR filtering
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const value = data[i];
      if (!Number.isFinite(value)) continue;
      if (value < min) min = value;
      if (value > max) max = value;
    }
    const normalized = new Uint8Array(data.length);
    if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
      normalized.fill(128);
    } else {
      const invRange = 255 / (max - min);
      for (let i = 0; i < data.length; i++) {
        const value = Number.isFinite(data[i]) ? data[i] : min;
        normalized[i] = Math.max(0, Math.min(255, Math.round((value - min) * invRange)));
      }
    }

    // Compute aspect ratio (longest axis = 1.0)
    const maxDim = Math.max(nx, ny, nz);
    this.aspectRatio = [nx / maxDim, ny / maxDim, nz / maxDim];

    // Destroy old texture and create new 3D texture
    this.volumeTexture.destroy();
    this.volumeTexture = this.device.createTexture({
      dimension: "3d",
      size: [nx, ny, nz],
      format: "r8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Upload data. WebGPU requires bytesPerRow aligned to 256. When nx is
    // already a multiple of 256 the row stride matches and we can skip the
    // padded copy entirely (saves a transient nx*ny*nz Uint8Array allocation
    // and one full pass).
    const bytesPerRow = Math.ceil(nx / 256) * 256;
    let payload: Uint8Array;
    if (bytesPerRow === nx) {
      payload = normalized;
    } else {
      payload = new Uint8Array(bytesPerRow * ny * nz);
      for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
          const srcOffset = (z * ny + y) * nx;
          const dstOffset = (z * ny + y) * bytesPerRow;
          payload.set(normalized.subarray(srcOffset, srcOffset + nx), dstOffset);
        }
      }
    }
    this.device.queue.writeTexture(
      { texture: this.volumeTexture },
      payload as unknown as GPUAllowSharedBufferSource,
      { bytesPerRow, rowsPerImage: ny },
      { width: nx, height: ny, depthOrArrayLayers: nz },
    );

    this.rebuildBindGroup();
  }

  uploadColormap(lut: Uint8Array): void {
    if (this.deviceLost) return;
    // Convert RGB (3 bytes per entry) to RGBA (4 bytes per entry) — WebGPU doesn't support RGB8
    const rgba = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      rgba[i * 4 + 0] = lut[i * 3 + 0];
      rgba[i * 4 + 1] = lut[i * 3 + 1];
      rgba[i * 4 + 2] = lut[i * 3 + 2];
      rgba[i * 4 + 3] = 255;
    }

    this.colormapTexture.destroy();
    this.colormapTexture = this.device.createTexture({
      dimension: "2d",
      size: [256, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture(
      { texture: this.colormapTexture },
      rgba,
      { bytesPerRow: 256 * 4 },
      { width: 256, height: 1 },
    );

    this.rebuildBindGroup();
  }

  render(params: VolumeRenderParams, camera: CameraState, bgColor: [number, number, number], dprOverride?: number, numStepsOverride?: number, zStretch: number = 1, orthographicView: boolean = false): void {
    if (this.deviceLost) return;
    const canvas = this.canvas;

    // Handle high-DPI displays (dprOverride allows reduced resolution during drag)
    const dpr = dprOverride ?? (window.devicePixelRatio || 1);
    const displayW = canvas.clientWidth;
    const displayH = canvas.clientHeight;
    if (displayW === 0 || displayH === 0) return;
    const bufferW = Math.round(displayW * dpr);
    const bufferH = Math.round(displayH * dpr);
    if (bufferW === 0 || bufferH === 0) return;
    if (canvas.width !== bufferW || canvas.height !== bufferH) {
      canvas.width = bufferW;
      canvas.height = bufferH;
      this.context.configure({
        device: this.device,
        format: this.canvasFormat,
        alphaMode: "opaque",
      });
    }

    // Camera setup
    const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
    const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
    const eyeX = camera.distance * cp * sy + camera.panX;
    const eyeY = camera.distance * sp + camera.panY;
    const eyeZ = camera.distance * cp * cy;

    const targetX = camera.panX, targetY = camera.panY, targetZ = 0;
    let fx = targetX - eyeX, fy = targetY - eyeY, fz = targetZ - eyeZ;
    const fLen = Math.sqrt(fx * fx + fy * fy + fz * fz) || 1;
    fx /= fLen; fy /= fLen; fz /= fLen;
    let sideX = fy * 0 - fz * 1, sideY = fz * 0 - fx * 0, sideZ = fx * 1 - fy * 0;
    const sLen = Math.sqrt(sideX * sideX + sideY * sideY + sideZ * sideZ) || 1;
    sideX /= sLen; sideY /= sLen; sideZ /= sLen;
    const ux = sideY * fz - sideZ * fy, uy = sideZ * fx - sideX * fz, uz = sideX * fy - sideY * fx;
    const roll = camera.roll ?? 0;
    const cr = Math.cos(roll), sr = Math.sin(roll);
    const upX = ux * cr + sideX * sr;
    const upY = uy * cr + sideY * sr;
    const upZ = uz * cr + sideZ * sr;
    const viewMatrix = lookAt(eyeX, eyeY, eyeZ, targetX, targetY, targetZ, upX, upY, upZ);
    const fov = Math.PI / 4;
    const aspect = displayW / displayH;
    const projMatrix = orthographicView
      ? (() => {
          // Match apparent scale at the volume center when toggling from
          // perspective, while keeping wheel zoom semantics via camera.distance.
          const viewH = 2 * camera.distance * Math.tan(fov / 2);
          const viewW = viewH * aspect;
          return orthographic(-viewW / 2, viewW / 2, -viewH / 2, viewH / 2, 0.01, 100.0);
        })()
      : perspective(fov, aspect, 0.01, 100.0);
    const viewProjMatrix = mat4Multiply(projMatrix, viewMatrix);
    const invViewProj = mat4Inverse(viewProjMatrix);

    // Number of steps scales with volume size
    const maxDim = Math.max(params.nx, params.ny, params.nz);
    const numSteps = numStepsOverride ?? Math.min(512, Math.max(128, maxDim * 2));

    // Write uniforms
    const uniformData = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
    const f32 = new Float32Array(uniformData);
    const u32 = new Uint32Array(uniformData);

    // invViewProj: mat4x4 at offset 0 (16 floats)
    f32.set(invViewProj, 0);
    // cameraPos: vec3 at offset 64/4=16
    f32[16] = eyeX; f32[17] = eyeY; f32[18] = eyeZ;
    // _pad0 at 19
    // aspectRatio: vec3 at offset 80/4=20
    f32[20] = this.aspectRatio[0]; f32[21] = this.aspectRatio[1]; f32[22] = this.aspectRatio[2] * Math.max(1, zStretch);
    // _pad1 at 23
    // bgColor: vec4 at offset 96/4=24
    f32[24] = bgColor[0]; f32[25] = bgColor[1]; f32[26] = bgColor[2]; f32[27] = 1.0;
    // sliceX, sliceY, sliceZ at offset 112/4=28
    // Slice-plane indicator at texel CENTER (not edge): (i + 0.5) / N
    // aligns with the volume sampler, which addresses texel centers.
    f32[28] = params.nx > 0 ? (params.sliceX + 0.5) / params.nx : 0.5;
    f32[29] = params.ny > 0 ? (params.sliceY + 0.5) / params.ny : 0.5;
    f32[30] = params.nz > 0 ? (params.sliceZ + 0.5) / params.nz : 0.5;
    // opacity at offset 124/4=31
    f32[31] = params.opacity;
    // brightness at offset 128/4=32
    f32[32] = params.brightness;
    // numSteps at offset 132/4=33
    u32[33] = numSteps;
    // showSlicePlanes at offset 136/4=34
    u32[34] = params.showSlicePlanes ? 1 : 0;
    // vmin at offset 140/4=35
    f32[35] = params.vmin;
    // vmax at offset 144/4=36
    f32[36] = params.vmax;
    // slicePlaneOpacity at offset 148/4=37
    f32[37] = params.slicePlaneOpacity ?? 0.35;

    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    // Render directly to canvas (no MSAA)
    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: textureView,
        clearValue: { r: bgColor[0], g: bgColor[1], b: bgColor[2], a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, this.bindGroup!);
    renderPass.draw(3);
    renderPass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  dispose(): void {
    if (this.deviceLost) return;
    this.volumeTexture.destroy();
    this.colormapTexture.destroy();
    this.uniformBuffer.destroy();
  }
}
