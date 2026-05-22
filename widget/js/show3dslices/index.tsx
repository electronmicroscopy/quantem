/// <reference types="@webgpu/types" />
/**
 * Show3DSlices - Orthogonal slice viewer for 3D volumetric data.
 *
 * Top/row/column slice panels with synchronized sliders and a 3D orientation view.
 * All slicing done in JS from raw float32 volume data for instant response.
 *
 * Ptycho-focused single-object workflow; tomography/comparison flows belong in
 * Show3DVolume.
 */
import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Slider from "@mui/material/Slider";
import Tooltip from "@mui/material/Tooltip";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import PauseIcon from "@mui/icons-material/Pause";
import FastForwardIcon from "@mui/icons-material/FastForward";
import FastRewindIcon from "@mui/icons-material/FastRewind";
import StopIcon from "@mui/icons-material/Stop";
import { useTheme } from "../theme";
import { VolumeRenderer, CameraState, DEFAULT_CAMERA } from "../webgpu-volume";
import { drawScaleBarHiDPI, drawFFTScaleBarHiDPI, drawColorbar } from "../figure";
import { extractFloat32, formatNumber } from "../format";
import { findDataRange, applyLogScale, percentileClip, sliderRange, computeHistogramFromBytes } from "../stats";

// ============================================================================
// Style tokens (inlined - matches Show2D/Show4DSTEM single-file convention)
// ============================================================================
const SPACING = { XS: 4, SM: 8, MD: 12, LG: 16 } as const;
const controlRow = {
  display: "flex",
  alignItems: "center",
  gap: `${SPACING.SM}px`,
  px: 1,
  py: 0.5,
  width: "fit-content",
};
const compactButton = {
  fontSize: 10,
  textTransform: "none" as const,
  letterSpacing: 0,
  py: 0.25,
  px: 1,
  minWidth: 0,
  "&.Mui-disabled": { color: "#666", borderColor: "#444" },
};
const switchStyles = {
  small: {
    "& .MuiSwitch-thumb": { width: 12, height: 12 },
    "& .MuiSwitch-switchBase": { padding: "4px" },
  },
};
const sliderStyles = {
  small: {
    py: 0,
    "& .MuiSlider-thumb": { width: 10, height: 10 },
    "& .MuiSlider-rail": { height: 2 },
    "& .MuiSlider-track": { height: 2 },
  },
};
const typographyLabel = {
  fontSize: 10,
  textTransform: "none" as const,
  letterSpacing: 0,
};
const typography = {
  label: { fontSize: 11 },
  labelSmall: { fontSize: 10 },
  value: { fontSize: 10, fontFamily: "monospace" },
  title: { fontWeight: "bold" as const },
};

// ============================================================================
// Inlined utilities (mirrors Show3D - keep widgets self-contained)
// ============================================================================
const signedLog1p = (x: number): number => x >= 0 ? Math.log1p(x) : -Math.log1p(-x);

function shouldIgnoreWidgetShortcut(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return target.closest([
    "input", "textarea", "button", "select",
    "[contenteditable='true']", "[role='button']", "[role='slider']",
    "[role='switch']", "[role='textbox']", "[role='combobox']", "[role='menuitem']",
    ".MuiSlider-root", ".MuiSelect-select",
  ].join(",")) !== null;
}

function extractXY(vol: Float32Array, nx: number, ny: number, nz: number, z: number): Float32Array {
  if (z < 0 || z >= nz) return new Float32Array(ny * nx);
  const start = z * ny * nx;
  return vol.subarray(start, start + ny * nx);
}

function extractXZ(vol: Float32Array, nx: number, ny: number, nz: number, y: number): Float32Array {
  const out = new Float32Array(nz * nx);
  if (y < 0 || y >= ny) return out;
  for (let z = 0; z < nz; z++) {
    const srcOffset = z * ny * nx + y * nx;
    for (let x = 0; x < nx; x++) out[z * nx + x] = vol[srcOffset + x];
  }
  return out;
}

function extractYZ(vol: Float32Array, nx: number, ny: number, nz: number, x: number): Float32Array {
  const out = new Float32Array(nz * ny);
  if (x < 0 || x >= nx) return out;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) out[z * ny + y] = vol[z * ny * nx + y * nx + x];
  }
  return out;
}

function reverseLut(lut: Uint8Array): Uint8Array {
  const out = new Uint8Array(lut.length);
  const n = lut.length / 3;
  for (let i = 0; i < n; i++) {
    const src = (n - 1 - i) * 3;
    const dst = i * 3;
    out[dst + 0] = lut[src + 0];
    out[dst + 1] = lut[src + 1];
    out[dst + 2] = lut[src + 2];
  }
  return out;
}

function maybeFlip(data: Float32Array, flip: boolean): Float32Array {
  if (!flip) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = -data[i];
  return out;
}

function findFFTPeak(
  mag: Float32Array, width: number, height: number,
  col: number, row: number, radius: number,
): { row: number; col: number } {
  const c0 = Math.max(0, Math.floor(col) - radius);
  const r0 = Math.max(0, Math.floor(row) - radius);
  const c1 = Math.min(width - 1, Math.floor(col) + radius);
  const r1 = Math.min(height - 1, Math.floor(row) + radius);
  let bestCol = Math.round(col), bestRow = Math.round(row), bestVal = -Infinity;
  for (let ir = r0; ir <= r1; ir++) {
    for (let ic = c0; ic <= c1; ic++) {
      const val = mag[ir * width + ic];
      if (val > bestVal) { bestVal = val; bestCol = ic; bestRow = ir; }
    }
  }
  const wc0 = Math.max(0, bestCol - 1), wc1 = Math.min(width - 1, bestCol + 1);
  const wr0 = Math.max(0, bestRow - 1), wr1 = Math.min(height - 1, bestRow + 1);
  let sumW = 0, sumWC = 0, sumWR = 0;
  for (let ir = wr0; ir <= wr1; ir++) {
    for (let ic = wc0; ic <= wc1; ic++) {
      const w = mag[ir * width + ic];
      sumW += w; sumWC += w * ic; sumWR += w * ir;
    }
  }
  if (sumW > 0) return { row: sumWR / sumW, col: sumWC / sumW };
  return { row: bestRow, col: bestCol };
}

function resolveDisplayBounds(
  dataMin: number, dataMax: number,
  traitVmin: number | null | undefined, traitVmax: number | null | undefined,
  logScale: boolean,
): { min: number; max: number } {
  return {
    min: logScale ? signedLog1p(traitVmin ?? dataMin) : (traitVmin ?? dataMin),
    max: logScale ? signedLog1p(traitVmax ?? dataMax) : (traitVmax ?? dataMax),
  };
}

// ============================================================================
// Inlined components (Histogram + InfoTooltip + KeyboardShortcuts)
// ============================================================================
function InfoTooltip({ text, theme = "dark" }: { text: React.ReactNode; theme?: "light" | "dark" }) {
  const isDark = theme === "dark";
  const content = typeof text === "string"
    ? <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>{text}</Typography>
    : text;
  return (
    <Tooltip
      title={content} arrow placement="bottom"
      componentsProps={{
        tooltip: { sx: { bgcolor: isDark ? "#333" : "#fff", color: isDark ? "#ddd" : "#333", border: `1px solid ${isDark ? "#555" : "#ccc"}`, maxWidth: 280, p: 1 } },
        arrow: { sx: { color: isDark ? "#333" : "#fff", "&::before": { border: `1px solid ${isDark ? "#555" : "#ccc"}` } } },
      }}
    >
      <Typography component="span" sx={{ fontSize: 12, color: isDark ? "#888" : "#666", cursor: "help", ml: 0.5, "&:hover": { color: isDark ? "#aaa" : "#444" } }}>
        ⓘ
      </Typography>
    </Tooltip>
  );
}

function KeyboardShortcuts({ items }: { items: [string, string][] }) {
  return (
    <Box
      component="table"
      sx={{
        borderCollapse: "collapse",
        "& td": { py: 0.25, fontSize: 11, lineHeight: 1.3, verticalAlign: "top" },
        "& td:first-of-type": { pr: 1.5, opacity: 0.7, fontFamily: "monospace", fontSize: 10, whiteSpace: "nowrap" },
      }}
    >
      <tbody>
        {items.map(([key, desc], i) => (
          <tr key={i}><td>{key}</td><td>{desc}</td></tr>
        ))}
      </tbody>
    </Box>
  );
}

interface HistogramProps {
  data: Float32Array | null;
  vminPct: number;
  vmaxPct: number;
  onRangeChange: (min: number, max: number) => void;
  width?: number;
  height?: number;
  theme?: "light" | "dark";
  dataMin?: number;
  dataMax?: number;
  pinBinsToRange?: boolean;
  ariaHidden?: boolean;
}

function Histogram({
  data, vminPct, vmaxPct, onRangeChange,
  width = 110, height = 40, theme = "dark",
  dataMin = 0, dataMax = 1, pinBinsToRange = true, ariaHidden = false,
}: HistogramProps) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const bins = React.useMemo(
    () => pinBinsToRange
      ? computeHistogramFromBytes(data, 256, dataMin, dataMax)
      : computeHistogramFromBytes(data),
    [data, dataMin, dataMax, pinBinsToRange],
  );
  const colors = React.useMemo(
    () => theme === "dark"
      ? { bg: "#1a1a1a", barActive: "#888", barInactive: "#444", border: "#333" }
      : { bg: "#f0f0f0", barActive: "#666", barInactive: "#bbb", border: "#ccc" },
    [theme],
  );
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, width, height);
    const displayBins = 64;
    const binRatio = Math.max(1, Math.floor(bins.length / displayBins));
    const reducedBins: number[] = [];
    for (let i = 0; i < displayBins; i++) {
      let sum = 0;
      for (let j = 0; j < binRatio; j++) sum += bins[i * binRatio + j] || 0;
      reducedBins.push(sum / binRatio);
    }
    const maxVal = Math.max(...reducedBins, 0.001);
    const barWidth = width / displayBins;
    const vminBin = Math.floor((vminPct / 100) * displayBins);
    const vmaxBin = Math.floor((vmaxPct / 100) * displayBins);
    for (let i = 0; i < displayBins; i++) {
      const barHeight = (reducedBins[i] / maxVal) * (height - 2);
      const x = i * barWidth;
      ctx.fillStyle = i >= vminBin && i <= vmaxBin ? colors.barActive : colors.barInactive;
      ctx.fillRect(x + 0.5, height - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
  }, [bins, vminPct, vmaxPct, width, height, colors]);
  const formatValue = (pct: number) => {
    const val = dataMin + (pct / 100) * (dataMax - dataMin);
    return val >= 1000 ? val.toExponential(1) : val.toFixed(1);
  };
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
      <canvas
        ref={canvasRef}
        style={{ width, height, border: `1px solid ${colors.border}` }}
        role={ariaHidden ? undefined : "img"}
        aria-hidden={ariaHidden ? "true" : undefined}
        aria-label={ariaHidden ? undefined : "Histogram of intensity values with min and max clip handles"}
      />
      <Slider
        value={[vminPct, vmaxPct]}
        onChange={(_, v) => {
          const [newMin, newMax] = v as number[];
          onRangeChange(Math.min(newMin, newMax - 1), Math.max(newMax, newMin + 1));
        }}
        min={0} max={100} size="small"
        valueLabelDisplay="auto" valueLabelFormat={formatValue}
        aria-label="Histogram intensity clip range"
        sx={{
          width, py: 0,
          "& .MuiSlider-thumb": { width: 8, height: 8 },
          "& .MuiSlider-rail": { height: 2 },
          "& .MuiSlider-track": { height: 2 },
          "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
        }}
      />
      <Box sx={{ display: "flex", justifyContent: "space-between", width }}>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{formatValue(vminPct)}</Typography>
        <Typography sx={{ fontSize: 8, fontFamily: "monospace", opacity: 0.6, lineHeight: 1 }}>{formatValue(vmaxPct)}</Typography>
      </Box>
    </Box>
  );
}

const controlLabel = { ...typography.label, ...typographyLabel };

const controlPanel = {
  select: { minWidth: 90, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } },
};

const container = {
  // overflowX:auto so panels stay reachable via horizontal scroll on narrow
  // viewport instead of being silently clipped past the cell edge.
  root: { p: 2, bgcolor: "transparent", color: "inherit", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", overflowX: "auto", overflowY: "visible" },
  imageBox: { bgcolor: "#000", border: "1px solid #444", overflow: "hidden", position: "relative" as const },
};

const upwardMenuProps = {
  anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
  transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
  sx: { zIndex: 9999 },
};

import { COLORMAPS, COLORMAP_NAMES, renderToOffscreen, renderToOffscreenReuse } from "../colormaps";

import { WebGPUFFT, getWebGPUFFT, fft2d, fftshift, nextPow2, computeMagnitude, autoEnhanceFFT, applyHannWindow2D } from "../fft";

// ============================================================================
// Zoom constants (matching Show3D)
// ============================================================================
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 10;

// ============================================================================
// Constants
// ============================================================================
type ZoomState = { zoom: number; panX: number; panY: number };
const DEFAULT_ZOOM: ZoomState = { zoom: 1, panX: 0, panY: 0 };
const DEFAULT_FFT_ZOOM: ZoomState = { zoom: 2, panX: 0, panY: 0 };
const CANVAS_TARGET = 480;
const AXES = ["xy", "xz", "yz"] as const;
// Show3DSlices opens in the same orientation as the main top slice panel:
// x/columns left-to-right and y/rows top-to-bottom.
const SHOW3DSLICES_DEFAULT_CAMERA: CameraState = {
  ...DEFAULT_CAMERA,
  yaw: Math.PI,
  pitch: 0,
  roll: Math.PI,
};
const VOLUME_VIEW_PRESETS = [
  { value: "xy", label: "Top", description: "top (XY) view" },
  { value: "xz", label: "Row", description: "row (XZ) view" },
  { value: "yz", label: "Col", description: "column (YZ) view" },
] as const;
const DPR = window.devicePixelRatio || 1;

// ============================================================================
// Main Component
// ============================================================================
const FFT_SNAP_RADIUS = 5;

function Show3DSlices() {
  // Theme detection
  const { themeInfo, colors: baseColors } = useTheme();
  const tc = {
    ...baseColors,
    accentGreen: themeInfo.theme === "dark" ? "#0f0" : "#1a7a1a",
    accentYellow: themeInfo.theme === "dark" ? "#ff0" : "#b08800",
  };

  const themedSelect = {
    ...controlPanel.select,
    bgcolor: tc.controlBg,
    color: tc.text,
    "& .MuiSelect-select": { py: 0.5 },
    "& .MuiOutlinedInput-notchedOutline": { borderColor: tc.border },
    "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: tc.accent },
  };

  const themedMenuProps = {
    ...upwardMenuProps,
    PaperProps: { sx: { bgcolor: tc.controlBg, color: tc.text, border: `1px solid ${tc.border}` } },
  };

  // Model state
  const [nx] = useModelState<number>("nx");
  const [ny] = useModelState<number>("ny");
  const [nz] = useModelState<number>("nz");
  const [volumeBytes] = useModelState<DataView>("volume_bytes");
  const [sliceX, setSliceX] = useModelState<number>("slice_x");
  const [sliceY, setSliceY] = useModelState<number>("slice_y");
  const [sliceZ, setSliceZ] = useModelState<number>("slice_z");
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [logScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");
  const [traitVmin] = useModelState<number | null>("vmin");
  const [traitVmax] = useModelState<number | null>("vmax");
  const [showControls] = useModelState<boolean>("show_controls");
  const [showCrosshair, setShowCrosshair] = useModelState<boolean>("show_crosshair");
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [orthographic, setOrthographic] = useModelState<boolean>("orthographic");
  const [smooth, setSmooth] = useModelState<boolean>("smooth");
  const [flip, setFlip] = useModelState<boolean>("flip");
  // No disabled_tools / hidden_tools traits in new monorepo Show3DSlices.
  const [dimLabels] = useModelState<string[]>("dim_labels");
  const [pixelSize] = useModelState<number>("pixel_size");
  // Per-axis sampling [pz, py, px] for anisotropic data; falls back to [pixelSize]*3.
  const [pixelSizeAxes] = useModelState<number[]>("pixel_size_axes");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [zStretch, setZStretch] = useModelState<number>("z_stretch");

  // No tool-parity in new monorepo. Everything visible + unlocked.
  const hideDisplay = false;
  const hideHistogram = false;
  const hidePlayback = false;
  const hideView = false;
  const hideVolume = false;
  const lockDisplay = false;
  const lockHistogram = false;
  const lockPlayback = false;
  const lockView = false;
  const lockVolume = false;

  // Initialize WebGPU FFT
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) { gpuFFTRef.current = fft; setGpuReady(true); }
    });
  }, []);

  // Canvas refs
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const uiRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);

  // FFT state
  const [fftColormap, setFftColormap] = React.useState("inferno");
  const [fftLogScale, setFftLogScale] = React.useState(false);
  const [fftAuto, setFftAuto] = React.useState(true);
  const [fftWindow, setFftWindow] = useModelState<boolean>("fft_window");
  const [fftZooms, setFftZooms] = React.useState<ZoomState[]>([DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM]);
  const [fftDragAxis, setFftDragAxis] = React.useState<number | null>(null);
  const [fftDragStart, setFftDragStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);

  // FFT d-spacing measurement
  type FftClickInfo = {
    axis: number; row: number; col: number; distPx: number;
    spatialFreq: number | null; dSpacing: number | null;
  };
  const [fftClickInfo, setFftClickInfo] = React.useState<FftClickInfo | null>(null);
  const fftClickStartRef = React.useRef<{ x: number; y: number; axis: number } | null>(null);
  const fftCanvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const fftOverlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const fftOffscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  const fftImgDataRefs = React.useRef<(ImageData | null)[]>([null, null, null]);
  const fftMagCacheRefs = React.useRef<(Float32Array | null)[]>([null, null, null]);
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const fftComputeGenerationRef = React.useRef(0);
  const [gpuReady, setGpuReady] = React.useState(false);
  // Counter to trigger FFT redraw after async compute finishes
  const [fftVersion, setFftVersion] = React.useState(0);

  // Zoom/pan per axis
  const [zooms, setZooms] = React.useState<ZoomState[]>([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]);
  const [dragAxis, setDragAxis] = React.useState<number | null>(null);
  const [dragStart, setDragStart] = React.useState<{ x: number; y: number; pX: number; pY: number } | null>(null);
  // rAF bypass: keep live zoom in ref during drag, sync to React state on mouseup.
  // Only sync ref from state when NOT dragging - otherwise an unrelated re-render
  // (playback tick, cursor update) would clobber in-flight pan values.
  const liveZoomsRef = React.useRef<ZoomState[]>([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]);
  if (dragAxis === null) liveZoomsRef.current = zooms;
  const zoomRafRef = React.useRef<number>(0);
  const liveFftZoomsRef = React.useRef<ZoomState[]>([DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM]);
  if (fftDragAxis === null) liveFftZoomsRef.current = fftZooms;
  const fftZoomRafRef = React.useRef<number>(0);

  // Canvas resize (matching Show2D pattern)
  const [canvasTarget, setCanvasTarget] = React.useState(CANVAS_TARGET);
  const [sideCanvasTarget, setSideCanvasTarget] = React.useState(CANVAS_TARGET);
  const [isResizing, setIsResizing] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number; target: "primary" | "side" } | null>(null);

  // Playback state (synced with Python)
  const [playing, setPlaying] = useModelState<boolean>("playing");
  const [playAxis, setPlayAxis] = useModelState<number>("play_axis");
  const [reverse, setReverse] = useModelState<boolean>("reverse");
  const [fps, setFps] = useModelState<number>("fps");
  const [loop, setLoop] = useModelState<boolean>("loop");
  const playIntervalRef = React.useRef<number | null>(null);
  const [boomerang, setBoomerang] = useModelState<boolean>("boomerang");
  const bounceDirRef = React.useRef<1 | -1>(1);
  const [loopStarts, setLoopStarts] = React.useState([0, 0, 0]);
  const [loopEnds, setLoopEnds] = React.useState([-1, -1, -1]);

  // 3D volume renderer state
  const volumeCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const volumeRendererRef = React.useRef<VolumeRenderer | null>(null);
  const [camera, setCamera] = React.useState<CameraState>(SHOW3DSLICES_DEFAULT_CAMERA);
  const [volumeDrag, setVolumeDrag] = React.useState<{
    button: number; x: number; y: number; yaw: number; pitch: number; panX: number; panY: number;
  } | null>(null);
  const [webgpuSupported, setWebgpuSupported] = React.useState(true);
  const [rendererReady, setRendererReady] = React.useState(0);
  const [volumeCanvasSize, setVolumeCanvasSize] = React.useState(CANVAS_TARGET);
  const [volumeResizing, setVolumeResizing] = React.useState(false);
  const volumeResizeStartRef = React.useRef<{ x: number; y: number; size: number } | null>(null);
  const [showSlicePlanes, setShowSlicePlanes] = React.useState(true);

  // Histogram state
  const [imageVminPct, setImageVminPct] = React.useState(0);
  const [imageVmaxPct, setImageVmaxPct] = React.useState(100);
  const [imageHistogramData, setImageHistogramData] = React.useState<Float32Array | null>(null);
  const [imageDataRange, setImageDataRange] = React.useState<{ min: number; max: number }>({ min: 0, max: 1 });

  // Volume opacity for the 3D context renderer.
  const [opacityA, setOpacityA] = React.useState(0.5);
  // Slice plane opacity in 3D renderer
  const [slicePlaneOpacity, setSlicePlaneOpacity] = React.useState(0.35);

  // Cached offscreen canvases for slice rendering (avoids recomputing colormap on zoom/pan)
  const sliceOffscreenRefs = React.useRef<(HTMLCanvasElement | null)[]>([null, null, null]);
  // Reusable ImageData per axis to avoid GC churn (allocated once per dimension change)
  const sliceImgDataRefs = React.useRef<(ImageData | null)[]>([null, null, null]);

  // Colorbar state
  const [showColorbar, setShowColorbar] = React.useState(false);

  // Show3DSlices always uses the compact widget layout. The old Python
  // `compact` trait is kept only as a compatibility no-op.
  const effectiveShowFft = showFft && !hideDisplay;

  // Cursor readout state
  const [cursorInfo, setCursorInfo] = React.useState<{ row: number; col: number; value: number; view: string } | null>(null);

  // Parse volume data
  const allFloats = React.useMemo(() => extractFloat32(volumeBytes), [volumeBytes]);

  // Slice dimensions: [xy: ny x nx], [xz: nz x nx], [yz: nz x ny]
  const sliceDims: [number, number][] = React.useMemo(() => [[ny, nx], [nz, nx], [nz, ny]], [nx, ny, nz]);

  // Canvas sizes. For depth panels (XZ=1, YZ=2) when nz << nxy, multiply
  // display height by z_stretch so the depth axis is readable. The internal
  // canvas pixel resolution (w x h_native) stays at scan-aligned dims; CSS
  // height stretches the rendered pixels with zero extra memory.
  // smooth=true → CSS bilinear (auto); smooth=false → nearest-neighbor (pixelated).
  // Overlay canvases (crosshair, scale bar, colorbar, FFT scale bar) use displayH
  // for their pixel buffer to avoid distortion under CSS stretch.
  const canvasSizes = React.useMemo(() => {
    return sliceDims.map(([h, w], a) => {
      const isDepth = a > 0;
      const target = isDepth ? sideCanvasTarget : canvasTarget;
      const scale = target / Math.max(w, h);
      const baseW = Math.round(w * scale);
      const baseH = Math.round(h * scale);
      const displayH = isDepth ? Math.min(target, Math.round(baseH * Math.max(1, zStretch))) : baseH;
      return { w: baseW, h: baseH, displayH, scale };
    });
  }, [sliceDims, canvasTarget, sideCanvasTarget, zStretch]);

  // Pre-allocate reusable offscreen canvases + ImageData per axis (avoids GC churn)
  React.useEffect(() => {
    for (let a = 0; a < 3; a++) {
      const [h, w] = sliceDims[a];
      // Check if existing offscreen matches dimensions
      const existing = sliceOffscreenRefs.current[a];
      if (!existing || existing.width !== w || existing.height !== h) {
        const c = document.createElement("canvas");
        c.width = w; c.height = h;
        sliceOffscreenRefs.current[a] = c;
        sliceImgDataRefs.current[a] = new ImageData(w, h);
      }
    }
  }, [sliceDims]);

  React.useEffect(() => {
    if (hideDisplay && showFft) {
      setShowFft(false);
    }
  }, [hideDisplay, showFft, setShowFft]);

  React.useEffect(() => {
    if (lockPlayback && playing) {
      setPlaying(false);
    }
  }, [lockPlayback, playing, setPlaying]);

  // Prevent page scroll on canvases
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    canvasRefs.current.forEach(c => c?.addEventListener("wheel", preventDefault, { passive: false }));
    fftCanvasRefs.current.forEach(c => c?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => {
      canvasRefs.current.forEach(c => c?.removeEventListener("wheel", preventDefault));
      fftCanvasRefs.current.forEach(c => c?.removeEventListener("wheel", preventDefault));
    };
  }, [allFloats, effectiveShowFft]);

  // log-scaled volume Float32Array, shared by 3D upload AND histogram useEffects.
  // applyLogScale allocates a fresh 32 MB buffer per call; without memoization a
  // 200³ volume with logScale=True allocates ~5 copies (3 histograms + 2 volume
  // uploads) on every toggle. Cache once per (volume, logScale) tuple.
  const volumeFloats = React.useMemo(() => {
    if (!allFloats) return null;
    return logScale ? applyLogScale(allFloats) : allFloats;
  }, [allFloats, logScale]);

  // Compute histogram from full volume (stable range across slices).
  // Read the shared `volumeFloats` memo so we don't re-allocate on logScale toggle.
  React.useEffect(() => {
    if (!volumeFloats || volumeFloats.length === 0) return;
    setImageHistogramData(volumeFloats);
    setImageDataRange(findDataRange(volumeFloats));
  }, [volumeFloats]);

  // Initial-mount Auto snap: when autoContrast is true from Python and histogram data
  // just loaded with default 0/100 slider, snap thumbs to 2/98 percentile so user sees
  // the actual range being rendered.
  React.useEffect(() => {
    if (!autoContrast || !imageHistogramData) return;
    if (imageVminPct !== 0 || imageVmaxPct !== 100) return;  // user already moved
    const { vmin: pmin, vmax: pmax } = percentileClip(imageHistogramData, 2, 98);
    const span = imageDataRange.max - imageDataRange.min;
    if (span > 0) {
      setImageVminPct(Math.max(0, Math.min(100, ((pmin - imageDataRange.min) / span) * 100)));
      setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - imageDataRange.min) / span) * 100)));
    }
  }, [autoContrast, imageHistogramData, imageDataRange]);


  // Sync boomerang direction ref with reverse state
  React.useEffect(() => {
    bounceDirRef.current = reverse ? -1 : 1;
  }, [reverse]);

  // -------------------------------------------------------------------------
  // 3D Volume Renderer - init, upload, render
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    const canvas = volumeCanvasRef.current;
    if (!canvas) return;
    if (!VolumeRenderer.isSupported()) { setWebgpuSupported(false); return; }
    let disposed = false;
    VolumeRenderer.create(canvas).then(renderer => {
      if (disposed) { renderer.dispose(); return; }
      volumeRendererRef.current = renderer;
      setRendererReady(n => n + 1);
    }).catch(() => { setWebgpuSupported(false); });
    return () => { disposed = true; volumeRendererRef.current?.dispose(); volumeRendererRef.current = null; };
  }, []);


  // Upload volume data
  React.useEffect(() => {
    const renderer = volumeRendererRef.current;
    if (!renderer || !volumeFloats || volumeFloats.length === 0) return;
    renderer.uploadVolume(volumeFloats, nx, ny, nz);
  }, [volumeFloats, nx, ny, nz, rendererReady]);

  // Upload colormap. When flip, reverse the LUT entry order so the 3D volume
  // inverts contrast the same way slice panels do (slices negate the data and
  // swap vmin/vmax, equivalent to reversing the colormap lookup). LUT is
  // 256 RGB triplets (768 bytes); reverse per-entry, not per-byte.
  React.useEffect(() => {
    const renderer = volumeRendererRef.current;
    if (!renderer) return;
    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    renderer.uploadColormap(flip ? reverseLut(lut) : lut);
  }, [cmap, rendererReady, flip]);

  // Render 3D volume
  // Map slider %s + optional traitVmin/Vmax to the texture's [0,1] normalized space.
  // The 3D texture is normalized per-volume to [0, 255] across (dataMin, dataMax),
  // so absolute traitVmin/Vmax must be converted to that normalized space before
  // being passed to the WGSL remap. Without this, slice panels honor traitVmin/Vmax
  // but the ray-cast view ignores it - giving inconsistent contrast.
  const volTexRange = React.useMemo(() => {
    const span = imageDataRange.max - imageDataRange.min;
    const hasTrait = (traitVmin != null || traitVmax != null) && span > 0;
    let baseMin: number, baseMax: number;
    if (hasTrait) {
      const { min: tMin, max: tMax } = resolveDisplayBounds(imageDataRange.min, imageDataRange.max, traitVmin, traitVmax, logScale);
      baseMin = (tMin - imageDataRange.min) / span;
      baseMax = (tMax - imageDataRange.min) / span;
    } else {
      baseMin = 0;
      baseMax = 1;
    }
    const subMin = baseMin + (baseMax - baseMin) * (imageVminPct / 100);
    const subMax = baseMin + (baseMax - baseMin) * (imageVmaxPct / 100);
    return { vmin: subMin, vmax: subMax };
  }, [traitVmin, traitVmax, imageDataRange, imageVminPct, imageVmaxPct, logScale]);
  // Keep render params in ref for direct rAF rendering (bypasses React during drag)
  const volumeRenderParamsRef = React.useRef({
    sliceX, sliceY, sliceZ, nx, ny, nz,
    opacity: opacityA, brightness: 1.0, showSlicePlanes, slicePlaneOpacity,
    vmin: volTexRange.vmin, vmax: volTexRange.vmax,
  });
  volumeRenderParamsRef.current = {
    sliceX, sliceY, sliceZ, nx, ny, nz,
    opacity: opacityA, brightness: 1.0, showSlicePlanes, slicePlaneOpacity,
    vmin: volTexRange.vmin, vmax: volTexRange.vmax,
  };
  const bgColorRef = React.useRef<[number, number, number]>([0, 0, 0]);
  React.useEffect(() => {
    const r = parseInt(tc.bg.slice(1, 3), 16) / 255;
    const g = parseInt(tc.bg.slice(3, 5), 16) / 255;
    const b = parseInt(tc.bg.slice(5, 7), 16) / 255;
    bgColorRef.current = [r, g, b];
  }, [tc.bg]);

  // Render 3D volume (non-interactive: triggered by React state changes)
  React.useEffect(() => {
    if (volumeDrag) return; // Skip during drag - rAF handles it directly
    const renderer = volumeRendererRef.current;
    if (!renderer || !volumeFloats || volumeFloats.length === 0) return;
    renderer.render(volumeRenderParamsRef.current, camera, bgColorRef.current, undefined, undefined, zStretch, orthographic);
  }, [volumeFloats, sliceX, sliceY, sliceZ, nx, ny, nz, cmap, camera, volumeCanvasSize, tc.bg, showSlicePlanes, slicePlaneOpacity, volumeDrag, rendererReady, volTexRange, opacityA, zStretch, orthographic, flip]);

  // Prevent scroll on volume canvas
  React.useEffect(() => {
    const canvas = volumeCanvasRef.current;
    if (!canvas || !webgpuSupported) return;
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    canvas.addEventListener("wheel", preventDefault, { passive: false });
    return () => canvas.removeEventListener("wheel", preventDefault);
  }, [webgpuSupported]);

  // -------------------------------------------------------------------------
  // 3D Volume mouse handlers - document-level listeners for robust drag
  // -------------------------------------------------------------------------
  const volumeRafRef = React.useRef<number>(0);
  const liveCameraRef = React.useRef<CameraState>(camera);
  // Live z_stretch ref for rAF drag path - keeps latest value without re-binding closure.
  const zStretchRef = React.useRef(zStretch);
  zStretchRef.current = zStretch;
  if (!volumeDrag) liveCameraRef.current = camera;
  const volumeDragDataRef = React.useRef<{ button: number; x: number; y: number; yaw: number; pitch: number; panX: number; panY: number } | null>(null);

  const handleVolumeMouseDown = (e: React.MouseEvent) => {
    const dragData = {
      button: e.button, x: e.clientX, y: e.clientY,
      yaw: camera.yaw, pitch: camera.pitch, panX: camera.panX, panY: camera.panY,
    };
    volumeDragDataRef.current = dragData;
    setVolumeDrag(dragData);
    e.preventDefault();
  };

  React.useEffect(() => {
    if (!volumeDrag) return;
    const onMove = (e: MouseEvent) => {
      const drag = volumeDragDataRef.current;
      if (!drag) return;
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      let next: CameraState;
      if (drag.button === 0 && !e.shiftKey) {
        next = {
          ...liveCameraRef.current,
          yaw: drag.yaw + dx * 0.005,
          pitch: Math.max(-Math.PI * 0.49, Math.min(Math.PI * 0.49, drag.pitch - dy * 0.005)),
        };
      } else {
        const sens = 0.003 * liveCameraRef.current.distance;
        next = {
          ...liveCameraRef.current,
          panX: drag.panX + dx * sens,
          panY: drag.panY - dy * sens,
        };
      }
      liveCameraRef.current = next;
      if (!volumeRafRef.current) {
        volumeRafRef.current = requestAnimationFrame(() => {
          volumeRafRef.current = 0;
          const cam = liveCameraRef.current;
          const params = volumeRenderParamsRef.current;
          const bg = bgColorRef.current;
          const rendererA = volumeRendererRef.current;
          if (rendererA) rendererA.render(params, cam, bg, undefined, undefined, zStretchRef.current, orthographic);
        });
      }
    };
    const onUp = () => {
      setCamera(liveCameraRef.current);
      setVolumeDrag(null);
      volumeDragDataRef.current = null;
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => { document.removeEventListener("mousemove", onMove); document.removeEventListener("mouseup", onUp); };
  }, [volumeDrag, orthographic]);

  const handleVolumeWheel = (e: React.WheelEvent) => {
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    const next = { ...liveCameraRef.current, distance: Math.max(0.5, Math.min(10, liveCameraRef.current.distance * factor)) };
    liveCameraRef.current = next;
    setCamera(next);
  };

  const handleVolumeDoubleClick = () => setCamera(SHOW3DSLICES_DEFAULT_CAMERA);

  const setVolumeView = (view: "xy" | "xz" | "yz") => {
    const distance = liveCameraRef.current.distance || camera.distance || SHOW3DSLICES_DEFAULT_CAMERA.distance;
    // Match the 2D slice panels rather than mathematical world-up:
    // Top: x right, row/y down. Row: x right, z down. Col: y right, z down.
    const presets: Record<"xy" | "xz" | "yz", Pick<CameraState, "yaw" | "pitch" | "roll">> = {
      xy: { yaw: Math.PI, pitch: 0, roll: Math.PI },
      xz: { yaw: 0, pitch: Math.PI * 0.49, roll: 0 },
      yz: { yaw: -Math.PI / 2, pitch: 0, roll: -Math.PI / 2 },
    };
    const next = { ...SHOW3DSLICES_DEFAULT_CAMERA, ...presets[view], distance, panX: 0, panY: 0 };
    liveCameraRef.current = next;
    setCamera(next);
  };

  const rollVolumeView = (direction: -1 | 1) => {
    const current = liveCameraRef.current;
    const next = { ...current, roll: (current.roll ?? 0) + direction * Math.PI / 2 };
    liveCameraRef.current = next;
    setCamera(next);
  };

  // -------------------------------------------------------------------------
  // 3D Volume canvas resize
  // -------------------------------------------------------------------------
  const volumeResizeRafRef = React.useRef(0);

  const handleVolumeResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation(); e.preventDefault();
    setVolumeResizing(true);
    volumeResizeStartRef.current = { x: e.clientX, y: e.clientY, size: volumeCanvasSize };
  };

  React.useEffect(() => {
    if (!volumeResizing) return;
    const onMove = (e: MouseEvent) => {
      const start = volumeResizeStartRef.current;
      if (!start) return;
      const delta = Math.max(e.clientX - start.x, e.clientY - start.y);
      const newSize = Math.max(300, Math.min(800, start.size + delta));
      // Throttle canvas resize to rAF for smooth drag
      if (!volumeResizeRafRef.current) {
        volumeResizeRafRef.current = requestAnimationFrame(() => {
          volumeResizeRafRef.current = 0;
          setVolumeCanvasSize(newSize);
        });
      }
    };
    const onUp = () => {
      if (volumeResizeRafRef.current) { cancelAnimationFrame(volumeResizeRafRef.current); volumeResizeRafRef.current = 0; }
      setVolumeResizing(false);
      volumeResizeStartRef.current = null;
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    return () => { document.removeEventListener("mousemove", onMove); document.removeEventListener("mouseup", onUp); };
  }, [volumeResizing]);

  const cameraChanged = camera.yaw !== SHOW3DSLICES_DEFAULT_CAMERA.yaw || camera.pitch !== SHOW3DSLICES_DEFAULT_CAMERA.pitch || (camera.roll ?? 0) !== (SHOW3DSLICES_DEFAULT_CAMERA.roll ?? 0) || camera.distance !== SHOW3DSLICES_DEFAULT_CAMERA.distance || camera.panX !== SHOW3DSLICES_DEFAULT_CAMERA.panX || camera.panY !== SHOW3DSLICES_DEFAULT_CAMERA.panY;

  // Reset Zoom is intentionally narrow: slice and FFT zoom/pan only. Camera,
  // contrast, colormap, and playback loop state have their own controls/state.
  const anyZoomDirty = zooms.some(z => z.zoom !== 1 || z.panX !== 0 || z.panY !== 0)
    || fftZooms.some(z => z.zoom !== DEFAULT_FFT_ZOOM.zoom || z.panX !== DEFAULT_FFT_ZOOM.panX || z.panY !== DEFAULT_FFT_ZOOM.panY);

  // -------------------------------------------------------------------------
  // Build colormapped offscreen canvases (expensive: log scale, percentile, colormap LUT)
  // Per-axis: only recompute the axis whose slice actually changed.
  // XY depends on sliceZ, XZ on sliceY, YZ on sliceX.
  // Excludes zoom/pan so dragging only triggers the cheap redraw below.
  // useLayoutEffect so offscreens are ready before the draw useLayoutEffect runs.
  // -------------------------------------------------------------------------
  const prevCacheRef = React.useRef<{
    sliceX: number; sliceY: number; sliceZ: number;
    cmap: string; logScale: boolean; autoContrast: boolean;
    imageVminPct: number; imageVmaxPct: number;
    allFloats: Float32Array | null;
    nx: number; ny: number; nz: number;
    traitVmin: number | null; traitVmax: number | null;
    flip: boolean;
  }>({ sliceX: -1, sliceY: -1, sliceZ: -1, cmap: "", logScale: false, autoContrast: false, imageVminPct: -1, imageVmaxPct: -1, allFloats: null, nx: 0, ny: 0, nz: 0, traitVmin: null, traitVmax: null, flip: false });

  React.useLayoutEffect(() => {
    if (!allFloats || allFloats.length === 0) return;

    const prev = prevCacheRef.current;
    const globalChanged = allFloats !== prev.allFloats || cmap !== prev.cmap ||
      logScale !== prev.logScale || autoContrast !== prev.autoContrast ||
      imageVminPct !== prev.imageVminPct || imageVmaxPct !== prev.imageVmaxPct ||
      traitVmin !== prev.traitVmin || traitVmax !== prev.traitVmax ||
      flip !== prev.flip ||
      nx !== prev.nx || ny !== prev.ny || nz !== prev.nz;
    const axisChanged = [
      globalChanged || sliceZ !== prev.sliceZ,  // axis 0 (XY) depends on sliceZ
      globalChanged || sliceY !== prev.sliceY,  // axis 1 (XZ) depends on sliceY
      globalChanged || sliceX !== prev.sliceX,  // axis 2 (YZ) depends on sliceX
    ];

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
    const extractors = [
      () => extractXY(allFloats, nx, ny, nz, sliceZ),
      () => extractXZ(allFloats, nx, ny, nz, sliceY),
      () => extractYZ(allFloats, nx, ny, nz, sliceX),
    ];
    for (let a = 0; a < 3; a++) {
      if (!axisChanged[a]) continue;
      const [sliceH, sliceW] = sliceDims[a];
      const processed = maybeFlip(logScale ? applyLogScale(extractors[a]()) : extractors[a](), flip);
      let vmin: number, vmax: number;
      const hasTraitRange = traitVmin != null || traitVmax != null;
      // Flip negates data, so the range must also flip (min<->max with sign).
      const { min: rawMin, max: rawMax } = resolveDisplayBounds(imageDataRange.min, imageDataRange.max, traitVmin, traitVmax, logScale);
      const rMin = flip ? -rawMax : rawMin;
      const rMax = flip ? -rawMin : rawMax;
      if (!hasTraitRange && autoContrast) {
        ({ vmin, vmax } = percentileClip(processed, 2, 98));
      } else if (imageVminPct > 0 || imageVmaxPct < 100) {
        ({ vmin, vmax } = sliderRange(rMin, rMax, imageVminPct, imageVmaxPct));
      } else {
        vmin = rMin;
        vmax = rMax;
      }
      const offscreen = sliceOffscreenRefs.current[a];
      const imgData = sliceImgDataRefs.current[a];
      if (offscreen && imgData && offscreen.width === sliceW && offscreen.height === sliceH) {
        renderToOffscreenReuse(processed, lut, vmin, vmax, offscreen, imgData);
      } else {
        sliceOffscreenRefs.current[a] = renderToOffscreen(processed, sliceW, sliceH, lut, vmin, vmax);
      }
    }
    prevCacheRef.current = { sliceX, sliceY, sliceZ, cmap, logScale, autoContrast, imageVminPct, imageVmaxPct, allFloats, nx, ny, nz, traitVmin, traitVmax, flip };
  }, [allFloats, sliceX, sliceY, sliceZ, nx, ny, nz, cmap, logScale, autoContrast, sliceDims, imageVminPct, imageVmaxPct, imageDataRange, traitVmin, traitVmax, flip]);

  // -------------------------------------------------------------------------
  // Redraw slices with zoom/pan (cheap: just drawImage from cached offscreen)
  // useLayoutEffect prevents black flash when canvas dimensions change (resize)
  // -------------------------------------------------------------------------
  React.useLayoutEffect(() => {
    for (let a = 0; a < 3; a++) {
      const canvas = canvasRefs.current[a];
      const offscreen = sliceOffscreenRefs.current[a];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      const [sliceH, sliceW] = sliceDims[a];
      const { w: cw, h: ch } = canvasSizes[a];
      ctx.imageSmoothingEnabled = smooth;
      ctx.clearRect(0, 0, cw, ch);
      const zs = zooms[a];
      if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
        ctx.save();
        const cx = cw / 2, cy = ch / 2;
        ctx.translate(cx + zs.panX, cy + zs.panY);
        ctx.scale(zs.zoom, zs.zoom);
        ctx.translate(-cx, -cy);
        ctx.drawImage(offscreen, 0, 0, sliceW, sliceH, 0, 0, cw, ch);
        ctx.restore();
      } else {
        ctx.drawImage(offscreen, 0, 0, sliceW, sliceH, 0, 0, cw, ch);
      }
    }
  }, [allFloats, sliceX, sliceY, sliceZ, nx, ny, nz, cmap, logScale, autoContrast, zooms, sliceDims, canvasSizes, imageVminPct, imageVmaxPct, smooth, flip]);

  // -------------------------------------------------------------------------
  // Render crosshair lines for the orthogonal slice intersections.
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!allFloats) return;
    const crossPositions: [number, number][] = [
      [sliceX, sliceY],
      [sliceX, sliceZ],
      [sliceY, sliceZ],
    ];
    for (let a = 0; a < 3; a++) {
      const overlay = overlayRefs.current[a];
      if (!overlay) continue;
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;
      const { w: cw, h: ch, displayH: dh, scale } = canvasSizes[a];
      const stretchY = dh / ch;
      ctx.clearRect(0, 0, cw, dh);
      if (!showCrosshair) continue;
      const zs = zooms[a];
      const [dataX, dataY] = crossPositions[a];
      const cx = cw / 2, cy = dh / 2;
      let canvasX = dataX * scale;
      let canvasY = dataY * scale * stretchY;
      if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
        canvasX = (canvasX - cx) * zs.zoom + cx + zs.panX;
        canvasY = (canvasY - cy) * zs.zoom + cy + zs.panY * stretchY;
      }
      ctx.strokeStyle = tc.accentYellow + "80";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(canvasX, 0); ctx.lineTo(canvasX, dh); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, canvasY); ctx.lineTo(cw, canvasY); ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [allFloats, sliceX, sliceY, sliceZ, zooms, showCrosshair, tc, canvasSizes]);

  // -------------------------------------------------------------------------
  // Scale bar (HiDPI UI overlay)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    for (let a = 0; a < 3; a++) {
      const uiCanvas = uiRefs.current[a];
      if (!uiCanvas) continue;
      const { w: cw, displayH: dh } = canvasSizes[a];
      uiCanvas.width = Math.round(cw * DPR);
      uiCanvas.height = Math.round(dh * DPR);
      const uiCtx = uiCanvas.getContext("2d");
      if (!uiCtx) continue;
      uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
      if (scaleBarVisible) {
        // Width-direction sampling per panel: XY → px (axes[2]), XZ → px (axes[2]),
        // YZ → py (axes[1]). Falls back to scalar pixelSize if axes triple absent.
        const widthAxis = [2, 2, 1][a];
        const axes = pixelSizeAxes && pixelSizeAxes.length === 3 ? pixelSizeAxes : null;
        const pxSize = axes ? axes[widthAxis] : (pixelSize || 0);
        const sliceW = sliceDims[a][1];
        const unit = pxSize > 0 ? "Å" : "px";
        const size = pxSize > 0 ? pxSize : 1;
        drawScaleBarHiDPI(uiCanvas, DPR, zooms[a].zoom, size, unit, sliceW);
      }

      if (showColorbar) {
        const lut = COLORMAPS[cmap] || COLORMAPS.inferno;
        const { min: rawMin, max: rawMax } = resolveDisplayBounds(
          imageDataRange.min,
          imageDataRange.max,
          traitVmin,
          traitVmax,
          logScale,
        );
        const baseMin = flip ? -rawMax : rawMin;
        const baseMax = flip ? -rawMin : rawMax;
        const { vmin, vmax } = sliderRange(baseMin, baseMax, imageVminPct, imageVmaxPct);
        const cssW = uiCanvas.width / DPR;
        const cssH = uiCanvas.height / DPR;
        uiCtx.save();
        uiCtx.scale(DPR, DPR);
        drawColorbar(uiCtx, cssW, cssH, lut, vmin, vmax, logScale);
        uiCtx.restore();
      }
    }
  }, [pixelSize, pixelSizeAxes, scaleBarVisible, zooms, canvasSizes, sliceDims, showColorbar, cmap, imageDataRange, imageVminPct, imageVmaxPct, traitVmin, traitVmax, logScale, flip, themeInfo.theme]);

  // -------------------------------------------------------------------------
  // FFT computation and caching (per-axis: only recompute changed axes)
  // -------------------------------------------------------------------------
  const prevFFTCacheRef = React.useRef<{
    sliceX: number; sliceY: number; sliceZ: number;
    allFloats: Float32Array | null;
    fftColormap: string; fftLogScale: boolean; fftAuto: boolean; fftWindow: boolean; gpuReady: boolean;
    effectiveShowFft: boolean;
  }>({ sliceX: -1, sliceY: -1, sliceZ: -1, allFloats: null, fftColormap: "", fftLogScale: false, fftAuto: false, fftWindow: false, gpuReady: false, effectiveShowFft: false });

  React.useEffect(() => {
    if (!effectiveShowFft || !allFloats || allFloats.length === 0) {
      // Release FFT caches when toggling off (each is up to 64 MB per axis).
      if (prevFFTCacheRef.current.effectiveShowFft && !effectiveShowFft) {
        for (let a = 0; a < 3; a++) {
          fftMagCacheRefs.current[a] = null;
          fftOffscreenRefs.current[a] = null;
          fftImgDataRefs.current[a] = null;
        }
        prevFFTCacheRef.current.effectiveShowFft = false;
      }
      return;
    }

    const prevFFT = prevFFTCacheRef.current;
    const globalFFTChanged = allFloats !== prevFFT.allFloats || fftColormap !== prevFFT.fftColormap ||
      fftLogScale !== prevFFT.fftLogScale || fftAuto !== prevFFT.fftAuto ||
      fftWindow !== prevFFT.fftWindow ||
      gpuReady !== prevFFT.gpuReady || !prevFFT.effectiveShowFft;
    const fftAxisChanged = [
      globalFFTChanged || sliceZ !== prevFFT.sliceZ,
      globalFFTChanged || sliceY !== prevFFT.sliceY,
      globalFFTChanged || sliceX !== prevFFT.sliceX,
    ];

    const lut = COLORMAPS[fftColormap] || COLORMAPS.inferno;
    const generation = ++fftComputeGenerationRef.current;
    let cancelled = false;

    const computeFFTsForVolume = async (
      floats: Float32Array,
      magCache: React.MutableRefObject<(Float32Array | null)[]>,
      offscreenCache: React.MutableRefObject<(HTMLCanvasElement | null)[]>,
      imgDataCache: React.MutableRefObject<(ImageData | null)[]>,
      forceAll: boolean,
    ) => {
      const extractors = [
        () => extractXY(floats, nx, ny, nz, sliceZ),
        () => extractXZ(floats, nx, ny, nz, sliceY),
        () => extractYZ(floats, nx, ny, nz, sliceX),
      ];
      const dims: [number, number][] = [[ny, nx], [nz, nx], [nz, ny]];

      for (let a = 0; a < 3; a++) {
        if (!forceAll && !fftAxisChanged[a]) continue;
        const extracted = extractors[a]();
        const data = fftWindow ? new Float32Array(extracted) : extracted;
        const [sliceH, sliceW] = dims[a];
        if (fftWindow) applyHannWindow2D(data, sliceW, sliceH);

        const pw = nextPow2(sliceW);
        const ph = nextPow2(sliceH);
        const paddedSize = pw * ph;
        let real: Float32Array, imag: Float32Array;

        if (gpuReady && gpuFFTRef.current) {
          const padReal = new Float32Array(paddedSize);
          const padImag = new Float32Array(paddedSize);
          for (let y = 0; y < sliceH; y++) for (let x = 0; x < sliceW; x++) padReal[y * pw + x] = data[y * sliceW + x];
          const result = await gpuFFTRef.current.fft2D(padReal, padImag, pw, ph, false);
          real = result.real; imag = result.imag;
        } else {
          real = new Float32Array(paddedSize);
          imag = new Float32Array(paddedSize);
          for (let y = 0; y < sliceH; y++) for (let x = 0; x < sliceW; x++) real[y * pw + x] = data[y * sliceW + x];
          fft2d(real, imag, pw, ph, false);
        }

        fftshift(real, pw, ph);
        fftshift(imag, pw, ph);

        const mag = computeMagnitude(real, imag);
        magCache.current[a] = mag;

        let displayMin: number, displayMax: number;
        if (fftAuto) {
          ({ min: displayMin, max: displayMax } = autoEnhanceFFT(mag, pw, ph));
        } else {
          ({ min: displayMin, max: displayMax } = findDataRange(mag));
        }

        const displayData = fftLogScale ? applyLogScale(mag) : mag;
        if (fftLogScale) { displayMin = Math.log1p(displayMin); displayMax = Math.log1p(displayMax); }

        // Reuse cached offscreen if dims match - saves ~4 MB ImageData alloc per axis.
        const existingOff = offscreenCache.current[a];
        const existingImg = imgDataCache.current[a];
        if (existingOff && existingImg && existingOff.width === pw && existingOff.height === ph) {
          renderToOffscreenReuse(displayData, lut, displayMin, displayMax, existingOff, existingImg);
        } else {
          const offscreen = renderToOffscreen(displayData, pw, ph, lut, displayMin, displayMax);
          if (!offscreen) continue;
          offscreenCache.current[a] = offscreen;
          const ctx = offscreen.getContext("2d");
          imgDataCache.current[a] = ctx ? ctx.getImageData(0, 0, pw, ph) : null;
        }

        // Drawing is handled by the separate cheap redraw effect below
      }
    };

    const computeAllFFTs = async () => {
      const localMagCache = { current: fftMagCacheRefs.current.map((value, axis) => fftAxisChanged[axis] ? null : value) } as React.MutableRefObject<(Float32Array | null)[]>;
      const localOffscreenCache = { current: fftOffscreenRefs.current.map((value, axis) => fftAxisChanged[axis] ? null : value) } as React.MutableRefObject<(HTMLCanvasElement | null)[]>;
      const localImgDataCache = { current: fftImgDataRefs.current.map((value, axis) => fftAxisChanged[axis] ? null : value) } as React.MutableRefObject<(ImageData | null)[]>;
      await computeFFTsForVolume(allFloats, localMagCache, localOffscreenCache, localImgDataCache, false);
      if (cancelled || generation !== fftComputeGenerationRef.current) return false;
      fftMagCacheRefs.current = localMagCache.current;
      fftOffscreenRefs.current = localOffscreenCache.current;
      fftImgDataRefs.current = localImgDataCache.current;
      prevFFTCacheRef.current = { sliceX, sliceY, sliceZ, allFloats, fftColormap, fftLogScale, fftAuto, fftWindow, gpuReady, effectiveShowFft };
      return true;
    };

    // Debounce FFT compute during slider scrubbing: defer 80 ms so a 60 Hz drag
    // collapses to ~12 Hz, freeing the main thread for image redraws.
    const debounceMs = 80;
    const timeoutId = setTimeout(() => {
      if (cancelled) return;
      computeAllFFTs().then((committed) => { if (committed) setFftVersion(v => v + 1); });
    }, debounceMs);
    return () => { cancelled = true; clearTimeout(timeoutId); };
  }, [effectiveShowFft, allFloats, sliceX, sliceY, sliceZ, nx, ny, nz, fftColormap, fftLogScale, fftAuto, fftWindow, gpuReady]);

  // Redraw cached FFT with zoom/pan (cheap -- no recomputation)
  React.useLayoutEffect(() => {
    if (!effectiveShowFft) return;
    for (let a = 0; a < 3; a++) {
      const canvas = fftCanvasRefs.current[a];
      const offscreen = fftOffscreenRefs.current[a];
      if (!canvas || !offscreen) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      const { w: cw, h: ch } = canvasSizes[a];
      const ow = offscreen.width, oh = offscreen.height;
      ctx.imageSmoothingEnabled = smooth;
      ctx.clearRect(0, 0, cw, ch);
      const zs = fftZooms[a];
      if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
        ctx.save();
        const cx = cw / 2, cy = ch / 2;
        ctx.translate(cx + zs.panX, cy + zs.panY); ctx.scale(zs.zoom, zs.zoom); ctx.translate(-cx, -cy);
        ctx.drawImage(offscreen, 0, 0, ow, oh, 0, 0, cw, ch);
        ctx.restore();
      } else {
        ctx.drawImage(offscreen, 0, 0, ow, oh, 0, 0, cw, ch);
      }
    }
  }, [effectiveShowFft, fftZooms, canvasSizes, fftVersion, smooth]);

  // Render FFT overlays (reciprocal-space scale bars + d-spacing crosshair per axis)
  React.useEffect(() => {
    if (!effectiveShowFft) return;
    const dims: [number, number][] = [[ny, nx], [nz, nx], [nz, ny]];
    for (let a = 0; a < 3; a++) {
      const overlay = fftOverlayRefs.current[a];
      if (!overlay) continue;
      const { w: cw, h: ch, displayH: dh } = canvasSizes[a];
      const stretchY = dh / ch;
      overlay.width = Math.round(cw * DPR);
      overlay.height = Math.round(dh * DPR);
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      // FFT scale bar (only when calibrated). Use width-direction sampling per
      // panel so anisotropic data shows correct |g| units. XY/XZ width -> px; YZ width -> py.
      const widthAxis = [2, 2, 1][a];
      const axes = pixelSizeAxes && pixelSizeAxes.length === 3 ? pixelSizeAxes : null;
      const realPx = axes ? axes[widthAxis] : pixelSize;
      if (realPx > 0) {
        const [, sliceW] = dims[a];
        const pw = nextPow2(sliceW);
        const fftPixelSize = 1 / (pw * realPx);
        drawFFTScaleBarHiDPI(overlay, DPR, fftZooms[a].zoom, fftPixelSize, pw, "Å⁻¹");
      }

      if (fftClickInfo && fftClickInfo.axis === a) {
        const [sliceH, sliceW] = dims[a];
        const fftW = nextPow2(sliceW);
        const fftH = nextPow2(sliceH);

        ctx.save();
        ctx.scale(DPR, DPR);
        const zs = fftZooms[a];
        const cx = cw / 2, cy = dh / 2;
        const rawX = fftClickInfo.col / fftW * cw;
        const rawY = fftClickInfo.row / fftH * dh;
        const screenX = (rawX - cx) * zs.zoom + cx + zs.panX;
        const screenY = (rawY - cy) * zs.zoom + cy + zs.panY * stretchY;

        ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
        ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
        ctx.shadowBlur = 2;
        ctx.lineWidth = 1.5;
        const r = 8;
        ctx.beginPath();
        ctx.moveTo(screenX - r, screenY); ctx.lineTo(screenX - 3, screenY);
        ctx.moveTo(screenX + 3, screenY); ctx.lineTo(screenX + r, screenY);
        ctx.moveTo(screenX, screenY - r); ctx.lineTo(screenX, screenY - 3);
        ctx.moveTo(screenX, screenY + 3); ctx.lineTo(screenX, screenY + r);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(screenX, screenY, 4, 0, Math.PI * 2);
        ctx.stroke();

        if (fftClickInfo.dSpacing != null) {
          const d = fftClickInfo.dSpacing;
          const label = d >= 10 ? `d = ${(d / 10).toFixed(2)} nm` : `d = ${d.toFixed(2)} \u00C5`;
          ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
          ctx.fillStyle = "white";
          ctx.textAlign = "left";
          ctx.textBaseline = "bottom";
          ctx.fillText(label, screenX + 10, screenY - 4);
        }
        ctx.restore();
      }
    }
  }, [effectiveShowFft, fftZooms, canvasSizes, pixelSize, pixelSizeAxes, nx, ny, nz, fftClickInfo]);

  // -------------------------------------------------------------------------
  // Playback logic (matching Show3D pattern)
  // -------------------------------------------------------------------------
  const sliceSettersRef = React.useRef<((v: number) => void)[]>([setSliceZ, setSliceY, setSliceX]);
  sliceSettersRef.current = [setSliceZ, setSliceY, setSliceX];
  const effectiveLoopEnds = React.useMemo(
    () => loopEnds.map((end, i) => {
      const max = [nz - 1, ny - 1, nx - 1][i];
      return end < 0 ? max : Math.min(end, max);
    }),
    [loopEnds, nx, ny, nz],
  );
  React.useEffect(() => {
    if (!playing) return;
    const intervalMs = 1000 / fps;

    // Factor the interval creation so visibilitychange can restart it without
    // re-running the whole effect (which would lose ref state).
    const startInterval = () => {
      if (playIntervalRef.current) return;
      if (playAxis === 3) {
        // "All" mode: advance all 3 axes simultaneously
        playIntervalRef.current = window.setInterval(() => {
          const dir = boomerang ? bounceDirRef.current : (reverse ? -1 : 1);
          // Check if any axis would go out of range
          let shouldBounce = false;
          for (let a = 0; a < 3; a++) {
            const next = sliceValuesRef.current[a] + dir;
            if (next > effectiveLoopEnds[a] || next < loopStarts[a]) { shouldBounce = true; break; }
          }
          if (boomerang && shouldBounce) {
            bounceDirRef.current = (-bounceDirRef.current) as 1 | -1;
          }
          const finalDir = boomerang ? bounceDirRef.current : dir;
          for (let a = 0; a < 3; a++) {
            const start = loopStarts[a];
            const end = effectiveLoopEnds[a];
            let next = sliceValuesRef.current[a] + finalDir;
            if (next > end) next = loop || boomerang ? start : end;
            else if (next < start) next = loop || boomerang ? end : start;
            sliceSettersRef.current[a](next);
            sliceValuesRef.current[a] = next;
          }
          if (!loop && !boomerang && shouldBounce) setPlaying(false);
        }, intervalMs);
      } else {
        // Single axis mode
        const axis = playAxis;
        const start = loopStarts[axis];
        const end = effectiveLoopEnds[axis];
        const setter = sliceSettersRef.current[axis];
        playIntervalRef.current = window.setInterval(() => {
          const prev = sliceValuesRef.current[axis];
          let next = prev;
          if (boomerang) {
            const candidate = prev + bounceDirRef.current;
            if (candidate > end) {
              bounceDirRef.current = -1;
              next = prev - 1 >= start ? prev - 1 : prev;
            } else if (candidate < start) {
              bounceDirRef.current = 1;
              next = prev + 1 <= end ? prev + 1 : prev;
            } else {
              next = candidate;
            }
          } else {
            next = prev + (reverse ? -1 : 1);
            if (reverse) {
              if (next < start) {
                if (!loop) setPlaying(false);
                next = loop ? end : start;
              }
            } else if (next > end) {
              if (!loop) setPlaying(false);
              next = loop ? start : end;
            }
          }
          setter(next);
          sliceValuesRef.current[axis] = next;
        }, intervalMs);
      }
    };

    startInterval();

    // Pause when the tab/window is hidden, auto-resume on show.
    // setInterval keeps firing on hidden tabs in Chrome (rate-limited, not zero)
    // and wastes Comm traffic, so we clear it. Track whether playback was active
    // at hide time so we restart only if the user hadn't paused in between.
    let wasPlayingBeforeHide = false;
    const onVis = () => {
      if (document.hidden) {
        if (playIntervalRef.current) {
          wasPlayingBeforeHide = true;
          clearInterval(playIntervalRef.current);
          playIntervalRef.current = null;
        }
      } else if (wasPlayingBeforeHide) {
        wasPlayingBeforeHide = false;
        startInterval();
      }
    };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      document.removeEventListener("visibilitychange", onVis);
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
      }
    };
  }, [playing, fps, reverse, boomerang, loop, playAxis, loopStarts, effectiveLoopEnds]);

  // -------------------------------------------------------------------------
  // Direct canvas draw (bypasses React state for 60fps pan during drag)
  // -------------------------------------------------------------------------
  const drawSliceDirect = (axis: number) => {
    const zs = liveZoomsRef.current[axis];
    const [sliceH, sliceW] = sliceDims[axis];
    const cs = canvasSizes[axis];
    const cw = cs.w, ch = cs.h;
    const canvas = canvasRefs.current[axis];
    const offscreen = sliceOffscreenRefs.current[axis];
    if (!canvas || !offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = smooth;
    ctx.clearRect(0, 0, cw, ch);
    if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
      ctx.save();
      const cx = cw / 2, cy = ch / 2;
      ctx.translate(cx + zs.panX, cy + zs.panY);
      ctx.scale(zs.zoom, zs.zoom);
      ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, sliceW, sliceH, 0, 0, cw, ch);
      ctx.restore();
    } else {
      ctx.drawImage(offscreen, 0, 0, sliceW, sliceH, 0, 0, cw, ch);
    }
  };

  const drawFftDirect = (axis: number) => {
    const zs = liveFftZoomsRef.current[axis];
    const cs = canvasSizes[axis];
    const cw = cs.w, ch = cs.h;
    const canvas = fftCanvasRefs.current[axis];
    const offscreen = fftOffscreenRefs.current[axis];
    if (!canvas || !offscreen) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const ow = offscreen.width, oh = offscreen.height;
    ctx.imageSmoothingEnabled = smooth;
    ctx.clearRect(0, 0, cw, ch);
    if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
      ctx.save();
      const cx = cw / 2, cy = ch / 2;
      ctx.translate(cx + zs.panX, cy + zs.panY); ctx.scale(zs.zoom, zs.zoom); ctx.translate(-cx, -cy);
      ctx.drawImage(offscreen, 0, 0, ow, oh, 0, 0, cw, ch);
      ctx.restore();
    } else {
      ctx.drawImage(offscreen, 0, 0, ow, oh, 0, 0, cw, ch);
    }
  };

  // -------------------------------------------------------------------------
  // Zoom/Pan handlers (matching Show3D)
  // -------------------------------------------------------------------------
  const handleWheel = (e: React.WheelEvent, axis: number) => {
    const canvas = canvasRefs.current[axis];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const zs = zooms[axis];
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const imgX = (mouseX - cx - zs.panX) / zs.zoom + cx;
    const imgY = (mouseY - cy - zs.panY) / zs.zoom + cy;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * factor));
    const newPanX = mouseX - (imgX - cx) * newZoom - cx;
    const newPanY = mouseY - (imgY - cy) * newZoom - cy;
    setZooms(prev => { const next = [...prev]; next[axis] = { zoom: newZoom, panX: newPanX, panY: newPanY }; return next; });
  };

  const clickJumpTimerRef = React.useRef<number | null>(null);

  const handleDoubleClick = (axis: number) => {
    if (clickJumpTimerRef.current !== null) {
      window.clearTimeout(clickJumpTimerRef.current);
      clickJumpTimerRef.current = null;
    }
    setZooms(prev => { const next = [...prev]; next[axis] = DEFAULT_ZOOM; return next; });
  };

  // Synchronous click-detection ref: synthetic events (CDP, automation) fire
  // mousedown→mouseup back-to-back before React commits setDragStart. The ref
  // is always current, so handleMouseUp can detect a stationary click even
  // when dragStart state hasn't been flushed yet.
  const clickStartRef = React.useRef<{ x: number; y: number; axis: number } | null>(null);
  const handleMouseDown = (e: React.MouseEvent, axis: number) => {
    if (clickJumpTimerRef.current !== null) {
      window.clearTimeout(clickJumpTimerRef.current);
      clickJumpTimerRef.current = null;
    }
    const zs = liveZoomsRef.current[axis];
    setDragAxis(axis);
    setDragStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
    clickStartRef.current = { x: e.clientX, y: e.clientY, axis };
  };
  React.useEffect(() => () => {
    if (clickJumpTimerRef.current !== null) window.clearTimeout(clickJumpTimerRef.current);
  }, []);

  const handleMouseMove = (e: React.MouseEvent, axis: number) => {
    if (dragAxis === axis && dragStart) {
      const canvas = canvasRefs.current?.[axis];
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const dx = (e.clientX - dragStart.x) * (canvas.width / rect.width);
      const dy = (e.clientY - dragStart.y) * (canvas.height / rect.height);
      const newZoom = { ...liveZoomsRef.current[axis], panX: dragStart.pX + dx, panY: dragStart.pY + dy };
      const next = [...liveZoomsRef.current]; next[axis] = newZoom;
      liveZoomsRef.current = next;
      if (!zoomRafRef.current) {
        zoomRafRef.current = requestAnimationFrame(() => {
          zoomRafRef.current = 0;
          drawSliceDirect(axis);
        });
      }
      return;
    }
    const cursorCanvas = canvasRefs.current?.[axis];
    if (!cursorCanvas || !allFloats || allFloats.length === 0) return;
    const rect = cursorCanvas.getBoundingClientRect();
    const canvasX = (e.clientX - rect.left) * (cursorCanvas.width / rect.width);
    const canvasY = (e.clientY - rect.top) * (cursorCanvas.height / rect.height);
    const { w: cw, h: ch, scale } = canvasSizes[axis];
    const zs = liveZoomsRef.current[axis];
    const cx = cw / 2, cy = ch / 2;
    let imgX: number, imgY: number;
    if (zs.zoom !== 1 || zs.panX !== 0 || zs.panY !== 0) {
      imgX = ((canvasX - cx - zs.panX) / zs.zoom + cx) / scale;
      imgY = ((canvasY - cy - zs.panY) / zs.zoom + cy) / scale;
    } else {
      imgX = canvasX / scale;
      imgY = canvasY / scale;
    }
    const px = Math.floor(imgX);
    const py = Math.floor(imgY);
    const [sliceH, sliceW] = sliceDims[axis];
    if (px < 0 || px >= sliceW || py < 0 || py >= sliceH) {
      setCursorInfo(null);
      return;
    }
    // 3D voxel lookup. XY: slice along Z. XZ: slice along Y. YZ: slice along X.
    let value: number;
    if (axis === 0)       value = allFloats[sliceZ * ny * nx + py * nx + px];
    else if (axis === 1)  value = allFloats[py * ny * nx + sliceY * nx + px];
    else                  value = allFloats[py * ny * nx + px * nx + sliceX];
    setCursorInfo({ row: py, col: px, value, view: ["XY", "XZ", "YZ"][axis] });
  };

  // Stationary click on a slice panel = jump-to-voxel. Convert the click's
  // canvas-pixel position into image-pixel coords (same math as handleMouseMove
  // cursor readout), then set the OTHER two slice indices. XY click → updates
  // sliceY+sliceX; XZ click → sliceZ+sliceX; YZ click → sliceZ+sliceY.
  const handleMouseUp = (e?: React.MouseEvent, axis?: number, refs?: React.RefObject<(HTMLCanvasElement | null)[]>) => {
    if (zoomRafRef.current) { cancelAnimationFrame(zoomRafRef.current); zoomRafRef.current = 0; }
    setZooms(liveZoomsRef.current);
    const click = clickStartRef.current;
    if (e && axis !== undefined && refs && click && click.axis === axis && !lockPlayback) {
      const moved = Math.abs(e.clientX - click.x) + Math.abs(e.clientY - click.y);
      if (moved < 4) {
        const canvas = refs.current?.[axis];
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const canvasX = (e.clientX - rect.left) * (canvas.width / rect.width);
          const canvasY = (e.clientY - rect.top) * (canvas.height / rect.height);
          const { w: cw, h: ch, scale } = canvasSizes[axis];
          const zs = liveZoomsRef.current[axis];
          const cx = cw / 2, cy = ch / 2;
          const imgX = ((canvasX - cx - zs.panX) / zs.zoom + cx) / scale;
          const imgY = ((canvasY - cy - zs.panY) / zs.zoom + cy) / scale;
          const px = Math.floor(imgX), py = Math.floor(imgY);
          const [sliceH, sliceW] = sliceDims[axis];
          if (px >= 0 && px < sliceW && py >= 0 && py < sliceH) {
            if (clickJumpTimerRef.current !== null) {
              window.clearTimeout(clickJumpTimerRef.current);
            }
            clickJumpTimerRef.current = window.setTimeout(() => {
              if (axis === 0) { setSliceY(py); setSliceX(px); }
              else if (axis === 1) { setSliceZ(py); setSliceX(px); }
              else { setSliceZ(py); setSliceY(px); }
              clickJumpTimerRef.current = null;
            }, 220);
          }
        }
      }
    }
    clickStartRef.current = null;
    setDragAxis(null); setDragStart(null);
  };
  // Don't kill the drag when the cursor briefly leaves the panel - users routinely
  // drag past the edge while panning. Only clear the cursor readout overlay.
  const handleMouseLeave = () => { setCursorInfo(null); };

  // Global mouseup ensures drag ends even if the user releases the mouse outside
  // any slice or FFT canvas (e.g. they drag onto the volume panel and let go).
  // Without this the dragAxis state stays pinned and the next mouseMove on ANY
  // panel pans it - very confusing.
  React.useEffect(() => {
    if (dragAxis === null && fftDragAxis === null) return;
    const onUp = () => {
      if (zoomRafRef.current) { cancelAnimationFrame(zoomRafRef.current); zoomRafRef.current = 0; }
      if (fftZoomRafRef.current) { cancelAnimationFrame(fftZoomRafRef.current); fftZoomRafRef.current = 0; }
      setZooms(liveZoomsRef.current);
      setFftZooms(liveFftZoomsRef.current);
      setDragAxis(null); setDragStart(null);
      setFftDragAxis(null); setFftDragStart(null);
      fftClickStartRef.current = null;
    };
    document.addEventListener("mouseup", onUp);
    return () => document.removeEventListener("mouseup", onUp);
  }, [dragAxis, fftDragAxis]);

  const handleResetSlices = () => {
    if (!lockView) {
      setZooms([DEFAULT_ZOOM, DEFAULT_ZOOM, DEFAULT_ZOOM]);
      setFftZooms([DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM]);
      setFftClickInfo(null);
    }
  };

  // -------------------------------------------------------------------------
  // Keyboard shortcuts
  // -------------------------------------------------------------------------
  // Arrow Left/Right  : prev/next Z slice
  // Arrow Up/Down     : prev/next Y slice  (Up = decrease, Down = increase)
  // Shift + Arrow L/R : prev/next X slice
  // Home / End        : first / last on active axis (playAxis)
  // Space             : play/pause
  // r / R             : reset slice/FFT zoom and pan
  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Keep native keyboard behavior for sliders/selects/buttons.
    if (shouldIgnoreWidgetShortcut(e.target)) return;
    const axisSetters = [setSliceZ, setSliceY, setSliceX];
    const axisValues = [sliceZ, sliceY, sliceX];
    const axisMaxes = [nz - 1, ny - 1, nx - 1];
    const activeAxis = playAxis < 3 ? playAxis : 0;
    const advance = (axis: number, delta: number) => {
      if (lockPlayback) return;
      e.preventDefault();
      axisSetters[axis](Math.max(0, Math.min(axisMaxes[axis], axisValues[axis] + delta)));
    };
    switch (e.key) {
      case " ":
        if (!lockPlayback) {
          e.preventDefault();
          setPlaying(!playing);
        }
        break;
      case "ArrowLeft":
        // ← / →  scrub the ACTIVE axis (matches the popup help + Space/Home/End
        // semantics + the play_axis dropdown). Shift+← / → still scrubs X
        // as an explicit override regardless of active axis.
        advance(e.shiftKey ? 2 : activeAxis, -1);
        break;
      case "ArrowRight":
        advance(e.shiftKey ? 2 : activeAxis, 1);
        break;
      case "ArrowUp":
        // ↑ / ↓ scrub Y (image-coords: up = smaller row index).
        advance(1, -1);
        break;
      case "ArrowDown":
        advance(1, 1);
        break;
      case "Home":
        if (!lockPlayback) {
          e.preventDefault();
          axisSetters[activeAxis](0);
        }
        break;
      case "End":
        if (!lockPlayback) {
          e.preventDefault();
          axisSetters[activeAxis](axisMaxes[activeAxis]);
        }
        break;
      case "r":
      case "R":
        // Only handle 'r' when no modifier so we don't shadow Ctrl+R / Cmd+R reload.
        if (!e.ctrlKey && !e.metaKey && !e.altKey) {
          e.preventDefault();
          handleResetSlices();
        }
        break;
    }
  };

  // -------------------------------------------------------------------------
  // FFT Zoom/Pan handlers
  // -------------------------------------------------------------------------
  const handleFftWheel = (e: React.WheelEvent, axis: number) => {
    const canvas = fftCanvasRefs.current[axis];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const zs = fftZooms[axis];
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const imgX = (mouseX - cx - zs.panX) / zs.zoom + cx;
    const imgY = (mouseY - cy - zs.panY) / zs.zoom + cy;
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * factor));
    const newPanX = mouseX - (imgX - cx) * newZoom - cx;
    const newPanY = mouseY - (imgY - cy) * newZoom - cy;
    setFftZooms(prev => { const next = [...prev]; next[axis] = { zoom: newZoom, panX: newPanX, panY: newPanY }; return next; });
  };

  const handleFftDoubleClick = (axis: number) => {
    setFftZooms(prev => { const next = [...prev]; next[axis] = DEFAULT_FFT_ZOOM; return next; });
  };

  const handleFftMouseDown = (e: React.MouseEvent, axis: number) => {
    fftClickStartRef.current = { x: e.clientX, y: e.clientY, axis };
    const zs = fftZooms[axis];
    setFftDragAxis(axis);
    setFftDragStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
  };

  const handleFftMouseMove = (e: React.MouseEvent, axis: number) => {
    if (fftDragAxis !== axis || !fftDragStart) return;
    const canvas = fftCanvasRefs.current[axis];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dx = (e.clientX - fftDragStart.x) * (canvas.width / rect.width);
    const dy = (e.clientY - fftDragStart.y) * (canvas.height / rect.height);
    const newZoom = { ...liveFftZoomsRef.current[axis], panX: fftDragStart.pX + dx, panY: fftDragStart.pY + dy };
    const next = [...liveFftZoomsRef.current]; next[axis] = newZoom;
    liveFftZoomsRef.current = next;
    if (!fftZoomRafRef.current) {
      fftZoomRafRef.current = requestAnimationFrame(() => {
        fftZoomRafRef.current = 0;
        drawFftDirect(axis);
      });
    }
  };

  const handleFftMouseUp = (e: React.MouseEvent, axis: number) => {
    // Click detection for d-spacing measurement
    if (fftClickStartRef.current && fftClickStartRef.current.axis === axis) {
      const dx = e.clientX - fftClickStartRef.current.x;
      const dy = e.clientY - fftClickStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) < 3) {
        const canvas = fftCanvasRefs.current[axis];
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const { w: cw, h: ch } = canvasSizes[axis];
          const zs = fftZooms[axis];

          // Determine FFT dimensions for this axis
          const dims: [number, number][] = [[ny, nx], [nz, nx], [nz, ny]];
          const [sliceH, sliceW] = dims[axis];
          const fftW = nextPow2(sliceW);
          const fftH = nextPow2(sliceH);

          const mouseX = (e.clientX - rect.left) * (cw / rect.width);
          const mouseY = (e.clientY - rect.top) * (ch / rect.height);
          const cx = cw / 2, cy = ch / 2;
          const imgX = (mouseX - cx - zs.panX) / zs.zoom + cx;
          const imgY = (mouseY - cy - zs.panY) / zs.zoom + cy;
          let imgCol = imgX / cw * fftW;
          let imgRow = imgY / ch * fftH;

          const cachedMag = fftMagCacheRefs.current[axis];
          if (cachedMag && imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
            const snapped = findFFTPeak(cachedMag, fftW, fftH, imgCol, imgRow, FFT_SNAP_RADIUS);
            imgCol = snapped.col;
            imgRow = snapped.row;
          }

          if (imgCol >= 0 && imgCol < fftW && imgRow >= 0 && imgRow < fftH) {
            const dcCol = imgCol - fftW / 2;
            const dcRow = imgRow - fftH / 2;
            const distPx = Math.sqrt(dcCol * dcCol + dcRow * dcRow);
            if (distPx < 1) {
              setFftClickInfo(null);
            } else {
              let spatialFreq: number | null = null;
              let dSpacing: number | null = null;
              const axes = pixelSizeAxes && pixelSizeAxes.length === 3 ? pixelSizeAxes : null;
              const rowSpacing = axes ? axes[[1, 0, 0][axis]] : pixelSize;
              const colSpacing = axes ? axes[[2, 2, 1][axis]] : pixelSize;
              if (rowSpacing > 0 && colSpacing > 0) {
                const paddedW = fftW;
                const paddedH = fftH;
                const freqC = dcCol / paddedW / colSpacing;
                const freqR = dcRow / paddedH / rowSpacing;
                spatialFreq = Math.sqrt(freqC * freqC + freqR * freqR);
                dSpacing = spatialFreq > 0 ? 1 / spatialFreq : null;
              }
              setFftClickInfo({ axis, row: imgRow, col: imgCol, distPx, spatialFreq, dSpacing });
            }
          }
        }
      }
    }
    fftClickStartRef.current = null;
    if (fftZoomRafRef.current) { cancelAnimationFrame(fftZoomRafRef.current); fftZoomRafRef.current = 0; }
    setFftZooms(liveFftZoomsRef.current);
    setFftDragAxis(null);
    setFftDragStart(null);
  };

  const handleFftResetAxis = (a: number) => {
    setFftZooms(prev => { const next = [...prev]; next[a] = DEFAULT_FFT_ZOOM; return next; });
    if (fftClickInfo && fftClickInfo.axis === a) setFftClickInfo(null);
  };

  const fftNeedsResetAxis = (a: number) => {
    const z = fftZooms[a];
    return z.zoom !== DEFAULT_FFT_ZOOM.zoom || z.panX !== DEFAULT_FFT_ZOOM.panX || z.panY !== DEFAULT_FFT_ZOOM.panY;
  };

  // -------------------------------------------------------------------------
  // Canvas resize (matching Show2D)
  // -------------------------------------------------------------------------
  const handleResizeStart = (e: React.MouseEvent, axis: number = 0) => {
    e.stopPropagation();
    e.preventDefault();
    const target = axis > 0 ? "side" : "primary";
    setIsResizing(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: target === "side" ? sideCanvasTarget : canvasTarget, target });
  };

  React.useEffect(() => {
    if (!isResizing) return;
    let rafId = 0;
    let latestSize = resizeStart ? resizeStart.size : canvasTarget;
    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      latestSize = Math.max(300, resizeStart.size + delta);
      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          rafId = 0;
          if (resizeStart.target === "side") setSideCanvasTarget(latestSize);
          else setCanvasTarget(latestSize);
        });
      }
    };
    const handleMouseUp = () => {
      if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
      if (resizeStart?.target === "side") setSideCanvasTarget(latestSize);
      else setCanvasTarget(latestSize);
      setIsResizing(false);
      setResizeStart(null);
    };
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing, resizeStart, canvasTarget]);

  // -------------------------------------------------------------------------
  // Labels and setters
  // -------------------------------------------------------------------------
  // Default mirrors Python's dim_labels default ["Z", "Y", "X"]: axis 0 is the slice
  // dim. Fallback fires only when the trait is briefly undefined (initial mount race).
  const dl = dimLabels || ["Z", "Y", "X"];
  const sliceValues = [sliceZ, sliceY, sliceX];
  // Mirror of slice values for playback intervals to read between renders.
  // The interval's `sliceValuesRef.current[a] = next` writes are load-bearing
  // at high fps (>~20): React batches setSliceZ/Y/X so two ticks can fire
  // before the next render reassigns this ref to the new [sliceZ,sliceY,sliceX].
  // Without the mutation the second tick reads the stale value and computes the
  // same `next`, freezing playback.
  const sliceValuesRef = React.useRef(sliceValues);
  sliceValuesRef.current = sliceValues;
  const sliceMaxes = [nz - 1, ny - 1, nx - 1];
  const sliderValues = sliceValues;
  const sliceSetters = [
    (_: Event, v: number | number[]) => setSliceZ(v as number),
    (_: Event, v: number | number[]) => setSliceY(v as number),
    (_: Event, v: number | number[]) => setSliceX(v as number),
  ];
  const loopSliderValues = (axis: number) => {
    return [loopStarts[axis], sliceValues[axis], effectiveLoopEnds[axis]];
  };
  const handleLoopSliderChange = (axis: number, vals: number[]) => {
    setLoopStarts(prev => { const next = [...prev]; next[axis] = vals[0]; return next; });
    [setSliceZ, setSliceY, setSliceX][axis](vals[1]);
    setLoopEnds(prev => { const next = [...prev]; next[axis] = vals[2]; return next; });
  };
  // Over-clip detection: user dragged hist thumbs past data peak → image goes black.
  // Compute effective vmin/vmax in data units, compare against 1st/99th percentile of histogram.
  // If vmin > 99% of data OR vmax < 1% of data, no visible content.
  const imageClipBounds = React.useMemo(() => {
    if (!imageHistogramData || imageHistogramData.length === 0) return null;
    return percentileClip(imageHistogramData, 1, 99);
  }, [imageHistogramData]);
  const isOverClipped = React.useMemo(() => {
    if (!imageClipBounds) return false;
    const span = imageDataRange.max - imageDataRange.min;
    if (span <= 0) return false;
    const vmin = imageDataRange.min + (imageVminPct / 100) * span;
    const vmax = imageDataRange.min + (imageVmaxPct / 100) * span;
    return vmin >= imageClipBounds.vmax || vmax <= imageClipBounds.vmin;
  }, [imageClipBounds, imageDataRange, imageVminPct, imageVmaxPct]);

  // Thin-Z layout: depth axis much smaller than lateral. Stack YZ/XZ panels vertically beside XY.
  const thinZ = nz < Math.min(nx, ny) / 4;
  const thinZGridTemplate = thinZ
    ? `"a0 a1" "a0 a2" / ${canvasSizes[0].w}px ${Math.max(canvasSizes[1].w, canvasSizes[2].w)}px`
    : `"a0 a1 a2" / ${canvasSizes[0].w}px ${canvasSizes[1].w}px ${canvasSizes[2].w}px`;
  const panelTotalW = (canvasSizes[0]?.w ?? CANVAS_TARGET) + (thinZ
    ? Math.max(canvasSizes[1]?.w ?? 0, canvasSizes[2]?.w ?? 0)
    : ((canvasSizes[1]?.w ?? 0) + (canvasSizes[2]?.w ?? 0) + SPACING.SM)) + SPACING.SM;
  const primaryPanelW = canvasSizes[0]?.w ?? CANVAS_TARGET;
  const controlRowHeight = 28;
  const denseControlRow = {
    ...controlRow,
    minHeight: controlRowHeight,
    py: 0.25,
    boxSizing: "border-box" as const,
  };
  const panelControlRow = {
    ...denseControlRow,
    border: `1px solid ${tc.border}`,
    bgcolor: tc.controlBg,
    boxSizing: "border-box" as const,
  };
  const inlineVolumeControlRow = {
    ...denseControlRow,
    px: 0,
    py: 0,
    minHeight: 22,
    width: volumeCanvasSize,
    maxWidth: volumeCanvasSize,
    flexWrap: "nowrap" as const,
    overflow: "hidden",
  };
  const denseSelect = {
    ...themedSelect,
    height: 22,
    fontSize: 10,
    "& .MuiSelect-select": { py: 0.25, px: 1 },
  };

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <Box className="show3dslices-root" tabIndex={0} onKeyDown={handleKeyDown} sx={{ ...container.root, bgcolor: tc.bg, color: tc.text, outline: "none", "&:focus": { outline: "2px solid #0af", outlineOffset: 2 }, "& canvas": { display: "block" } }}>
      {/* 3D volume on the LEFT, slice toolbar + projected slice panels on the RIGHT.
          Side-by-side layout keeps the whole widget within a 13" laptop viewport. */}
      <Box sx={{ display: "flex", flexDirection: "row", alignItems: "flex-start", gap: `${SPACING.SM}px` }}>
      {/* 3D Volume Renderer (left column) */}
      {!hideVolume && (
      <Box sx={{ mb: 0, flexShrink: 0 }}>
        {/* Title row */}
        <Typography variant="caption" sx={{ ...typography.label, color: tc.accent, mb: `${SPACING.XS}px`, display: "block", height: 16, lineHeight: "16px", overflow: "hidden" }}>
          {title || "Volume 3D"}<InfoTooltip text={<Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
            <Typography sx={{ fontSize: 11, fontWeight: "bold" }}>Controls</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>FFT shows the power spectrum below each slice.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Auto uses percentile-based contrast (2nd-98th percentile). FFT Auto masks DC + clips to 99.9th.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Colorbar displays a colorbar overlay on each slice canvas.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Loop repeats playback. Drag end markers on slider for loop range.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Bounce alternates forward and reverse playback.</Typography>
            <Typography sx={{ fontSize: 11, lineHeight: 1.4 }}>Planes shows or hides slice planes in the 3D volume view.</Typography>
            <Typography sx={{ fontSize: 11, fontWeight: "bold", mt: 0.5 }}>Keyboard</Typography>
            <KeyboardShortcuts items={[["Space", "Play / Pause"], ["← / →", "Active axis -/+"], ["↑ / ↓", "Y slice -/+"], ["Shift+← / →", "X slice -/+"], ["Home / End", "First / Last on active axis"], ["R", "Reset zoom"], ["Click panel", "Jump to voxel"], ["Scroll", "Zoom"], ["Dbl-click", "Reset view"]]} />
          </Box>} theme={themeInfo.theme} />
          {/* ControlCustomizer dropped in new monorepo */}
        </Typography>
        {webgpuSupported ? (
          <Stack direction="row" spacing={`${SPACING.SM}px`}>
            {/* Volume A */}
            <Box>
              <Box sx={{ ...inlineVolumeControlRow, mb: `${SPACING.XS}px` }}>
                <Typography sx={{ ...controlLabel }}>Planes</Typography>
                <Switch checked={showSlicePlanes} onChange={(e) => setShowSlicePlanes(e.target.checked)} disabled={lockVolume} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle slice planes in 3D volume" }} />
                <Typography sx={{ ...controlLabel }}>Ortho</Typography>
                <Switch checked={orthographic} onChange={(e) => setOrthographic(e.target.checked)} disabled={lockVolume} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle orthographic 3D projection" }} />
                {showSlicePlanes && (
                  <>
                    <Typography sx={{ ...controlLabel }}>Opacity</Typography>
                    <Slider value={slicePlaneOpacity} min={0.05} max={1} step={0.05} onChange={(_, v) => setSlicePlaneOpacity(v as number)} disabled={lockVolume} size="small" sx={{ ...sliderStyles.small, width: 50 }} aria-label="Slice plane opacity" valueLabelDisplay="auto" />
                  </>
                )}
                <Typography sx={{ ...controlLabel }}>Vol Strength</Typography>
                <Slider value={opacityA} min={0} max={1} step={0.05} onChange={(_, v) => setOpacityA(v as number)} disabled={lockVolume} size="small" sx={{ ...sliderStyles.small, width: 50 }} aria-label="Volume strength" valueLabelDisplay="auto" />
              </Box>
              <Box
                sx={{
                  ...container.imageBox,
                  border: `1px solid ${tc.border}`,
                  width: volumeCanvasSize,
                  height: volumeCanvasSize,
                  cursor: lockVolume ? "default" : (volumeDrag ? "grabbing" : "grab"),
                }}
                onMouseDown={(e) => { if (!lockVolume) handleVolumeMouseDown(e); }}
                onWheel={(e) => { if (!lockVolume) handleVolumeWheel(e); }}
                onDoubleClick={() => { if (!lockVolume && !lockView) handleVolumeDoubleClick(); }}
                onContextMenu={(e) => e.preventDefault()}
              >
                <canvas
                  ref={volumeCanvasRef}
                  style={{ width: volumeCanvasSize, height: volumeCanvasSize, display: "block" }}
                  role="img"
                  aria-label={`3D volume rendering${title ? `: ${title}` : ""} (${nx} by ${ny} by ${nz} voxels). Drag to rotate, wheel to zoom.`}
                />
                {cameraChanged && (
                  <Button
                    size="small"
                    sx={{ ...compactButton, position: "absolute", top: 4, right: 4, minWidth: 0, px: 0.75, bgcolor: "rgba(255,255,255,0.75)", "&:hover": { bgcolor: "rgba(255,255,255,0.9)" } }}
                    onClick={(e) => { e.stopPropagation(); if (!lockVolume) setCamera(SHOW3DSLICES_DEFAULT_CAMERA); }}
                    disabled={lockVolume}
                    aria-label="Reset 3D camera view"
                    title="Reset 3D camera view"
                  >
                    Reset View
                  </Button>
                )}
                <Box
                  onMouseDown={(e) => { if (!lockVolume) handleVolumeResizeStart(e); }}
                  sx={{
                    position: "absolute", bottom: 2, right: 2, width: 12, height: 12,
                    cursor: lockVolume ? "default" : "nwse-resize", opacity: lockVolume ? 0.2 : 0.4,
                    background: `linear-gradient(135deg, transparent 50%, ${tc.textMuted} 50%)`,
                    "&:hover": { opacity: 1 },
                  }}
                />
              </Box>
              <Box sx={{ ...inlineVolumeControlRow, mt: 0 }}>
                <Typography sx={{ ...controlLabel }} title="Align the 3D camera to a slice plane.">View</Typography>
                {VOLUME_VIEW_PRESETS.map(({ value, label, description }) => (
                  <Button
                    key={value}
                    size="small"
                    sx={{ ...compactButton, minWidth: label === "Top" ? 28 : 30, px: 0.5 }}
                    disabled={lockVolume}
                    onClick={() => setVolumeView(value)}
                    aria-label={`Set 3D view to ${description}`}
                    title={`Set 3D view to ${description}`}
                  >
                    {label}
                  </Button>
                ))}
                <Button
                  size="small"
                  sx={{ ...compactButton, minWidth: 28, px: 0.5, fontSize: 13 }}
                  disabled={lockVolume}
                  onClick={() => rollVolumeView(1)}
                  aria-label="Roll 3D camera view counterclockwise 90 degrees"
                  title="Roll view counterclockwise 90 degrees"
                >
                  ↺90
                </Button>
                <Button
                  size="small"
                  sx={{ ...compactButton, minWidth: 28, px: 0.5, fontSize: 13 }}
                  disabled={lockVolume}
                  onClick={() => rollVolumeView(-1)}
                  aria-label="Roll 3D camera view clockwise 90 degrees"
                  title="Roll view clockwise 90 degrees"
                >
                  ↻90
                </Button>
              </Box>
            </Box>
          </Stack>
        ) : (
          <Box sx={{
            ...container.imageBox, width: volumeCanvasSize, height: 80,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Typography sx={{ ...typography.label, color: tc.textMuted, px: 2, textAlign: "center" }}>
              WebGPU not available. 3D volume rendering requires a WebGPU capable browser.
            </Typography>
          </Box>
        )}
      </Box>
      )}
      {/* Right column: slice toolbar + projected slice panels (grouped so they
          sit beside the 3D volume rather than below it). */}
      <Box sx={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0 }}>
      {/* Slice toolbar: compact row above the side column. */}
      <Box sx={{ display: "flex", alignItems: "center", gap: `${SPACING.SM}px`, mt: 0, mb: 0, minHeight: 18, justifyContent: "flex-end", width: panelTotalW, maxWidth: panelTotalW, boxSizing: "border-box" }}>
        {!hideDisplay && (
          <>
            <Typography sx={{ ...controlLabel }}>FFT</Typography>
            <Switch checked={showFft} onChange={(e) => { if (!lockDisplay) setShowFft(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle FFT power spectrum panels" }} />
          </>
        )}
        {(!hideView || !hideDisplay) && (
          <Button
            size="small"
            sx={compactButton}
            disabled={lockView || !anyZoomDirty}
            onClick={() => { if (!lockView) handleResetSlices(); }}
            title="Reset slice and FFT zoom/pan only"
            aria-label="Reset slice and FFT zoom/pan"
          >
            Reset Zoom
          </Button>
        )}
      </Box>
      {(() => {
        const panels = AXES.map((_, a) => {
          const { w: cw, h: ch, displayH: dh } = canvasSizes[a];
          // In thin-Z stacked layout, hide headers for axes 1+2 (Y, X depth panels) so
          // they butt up against each other with zero whitespace. Colored borders + slider
          // labels still identify axes.
          return (
            <Box key={a} sx={{ minWidth: cw, gridArea: `a${a}` }}>
              {/* Canvas with plane-colored border. dh = displayH (stretched for depth panels). */}
              <Box
                sx={{ ...container.imageBox, width: cw, height: dh, cursor: "grab", borderColor: ["#4d80ff", "#4dff66", "#ff4d4d"][a] }}
                onMouseDown={(e) => { if (!lockView) handleMouseDown(e, a); }}
                onMouseMove={(e) => handleMouseMove(e, a)}
                onMouseUp={(e) => handleMouseUp(e, a, canvasRefs)}
                onMouseLeave={handleMouseLeave}
                onWheel={(e) => { if (!lockView) handleWheel(e, a); }}
                onDoubleClick={() => { if (!lockView) handleDoubleClick(a); }}
              >
                <canvas
                  ref={(el) => { canvasRefs.current[a] = el; }}
                  width={cw}
                  height={ch}
                  style={{ width: cw, height: dh, imageRendering: smooth ? "auto" : "pixelated" }}
                  role="img"
                  aria-label={`${["XY", "XZ", "YZ"][a]} slice ${sliceValues[a] + 1} of ${sliceMaxes[a] + 1} along ${dl[a]} axis${title ? `: ${title}` : ""} (${cw} by ${ch} pixels)`}
                />
                <canvas
                  ref={(el) => { overlayRefs.current[a] = el; }}
                  width={cw}
                  height={dh}
                  style={{ position: "absolute", top: 0, left: 0, width: cw, height: dh, pointerEvents: "none" }}
                  aria-hidden="true"
                />
                <canvas
                  ref={(el) => { uiRefs.current[a] = el; }}
                  width={Math.round(cw * DPR)}
                  height={Math.round(dh * DPR)}
                  style={{ position: "absolute", top: 0, left: 0, width: cw, height: dh, pointerEvents: "none" }}
                  aria-hidden="true"
                />
                {/* Cursor readout overlay */}
                {cursorInfo && cursorInfo.view === ["XY", "XZ", "YZ"][a] && (
                  <Box sx={{ position: "absolute", top: 3, right: 3, bgcolor: "rgba(0,0,0,0.35)", px: 0.5, py: 0.15, pointerEvents: "none", minWidth: 100, textAlign: "right" }}>
                    <Typography sx={{ fontSize: 9, fontFamily: "monospace", color: "rgba(255,255,255,0.7)", whiteSpace: "nowrap", lineHeight: 1.2 }}>
                      ({cursorInfo.row}, {cursorInfo.col}) {formatNumber(cursorInfo.value)}
                    </Typography>
                  </Box>
                )}
                {/* Over-clip warning: image is mostly black because histogram thumbs sit outside data range */}
                {isOverClipped && a === 0 && (
                  <Box sx={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", bgcolor: "rgba(255, 180, 0, 0.85)", color: "#000", px: 1, py: 0.5, fontSize: 11, fontWeight: "bold", borderRadius: 0.5, textAlign: "center", lineHeight: 1.3, pointerEvents: "none", maxWidth: cw - 20 }}>
                    No data visible<br/>
                    <span style={{ fontSize: 9, fontWeight: "normal" }}>Adjust contrast range or enable Auto</span>
                  </Box>
                )}
                {/* Resize handle */}
                <Box
                  onMouseDown={(e) => { if (!lockView) handleResizeStart(e, a); }}
                  sx={{
                    position: "absolute", bottom: 2, right: 2, width: 12, height: 12,
                    cursor: lockView ? "default" : "nwse-resize", opacity: lockView ? 0.2 : 0.4,
                    background: `linear-gradient(135deg, transparent 50%, ${tc.textMuted} 50%)`,
                    "&:hover": { opacity: 1 },
                  }}
                />
              </Box>
              {/* FFT canvas (inline, below stats) */}
              {effectiveShowFft && (
                <Box sx={{ mt: `${SPACING.SM}px` }}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: `${SPACING.XS}px`, height: 20 }}>
                    <Stack direction="row" alignItems="center" sx={{ overflow: "hidden" }}>
                      <Typography variant="caption" sx={{ ...typography.label, fontSize: 10, flexShrink: 0 }}>
                        {`FFT ${[`${dl[1]}${dl[2]}`, `${dl[0]}${dl[2]}`, `${dl[0]}${dl[1]}`][a]} ${gpuReady ? "" : " (CPU fallback)"}`}
                      </Typography>
                      {fftClickInfo && fftClickInfo.axis === a && (
                        <Typography sx={{ fontSize: 10, fontFamily: "monospace", color: tc.textMuted, ml: 1, whiteSpace: "nowrap" }}>
                          {fftClickInfo.dSpacing != null ? (
                            <>d=<Box component="span" sx={{ color: tc.accent, fontWeight: "bold" }}>{fftClickInfo.dSpacing >= 10 ? `${(fftClickInfo.dSpacing / 10).toFixed(2)} nm` : `${fftClickInfo.dSpacing.toFixed(2)} \u00C5`}</Box>{" |g|="}<Box component="span" sx={{ color: tc.accent }}>{`${fftClickInfo.spatialFreq!.toFixed(4)} \u00C5\u207B\u00B9`}</Box></>
                          ) : (
                            <>dist=<Box component="span" sx={{ color: tc.accent }}>{fftClickInfo.distPx.toFixed(1)} px</Box></>
                          )}
                        </Typography>
                      )}
                    </Stack>
                    <Button size="small" sx={compactButton} disabled={lockView || !fftNeedsResetAxis(a)} onClick={() => handleFftResetAxis(a)} aria-label={`Reset ${["XY", "XZ", "YZ"][a]} FFT zoom and pan`}>Reset</Button>
                  </Stack>
                  <Box
                    sx={{ ...container.imageBox, width: cw, height: dh, cursor: "grab", borderColor: ["#4d80ff", "#4dff66", "#ff4d4d"][a] }}
                    onMouseDown={(e) => { if (!lockView) handleFftMouseDown(e, a); }}
                    onMouseMove={(e) => { if (!lockView) handleFftMouseMove(e, a); }}
                    onMouseUp={(e) => { if (!lockView) handleFftMouseUp(e, a); }}
                    onMouseLeave={() => { if (!lockView) { fftClickStartRef.current = null; setFftDragAxis(null); setFftDragStart(null); } }}
                    onWheel={(e) => { if (!lockView) handleFftWheel(e, a); }}
                    onDoubleClick={() => { if (!lockView) handleFftDoubleClick(a); }}
                  >
                    <canvas
                      ref={(el) => { fftCanvasRefs.current[a] = el; }}
                      width={cw}
                      height={ch}
                      style={{ width: cw, height: dh, imageRendering: smooth ? "auto" : "pixelated" }}
                      role="img"
                      aria-label={`FFT power spectrum of ${["XY", "XZ", "YZ"][a]} slice (reciprocal space, ${cw} by ${ch} pixels)`}
                    />
                    <canvas
                      ref={(el) => { fftOverlayRefs.current[a] = el; }}
                      width={Math.round(cw * DPR)}
                      height={Math.round(dh * DPR)}
                      style={{ position: "absolute", top: 0, left: 0, width: cw, height: dh, pointerEvents: "none" }}
                      aria-hidden="true"
                    />
                  </Box>
                </Box>
              )}
              {!hidePlayback && (
              <Box sx={{ ...controlRow, mt: `${SPACING.SM}px`, border: `1px solid ${tc.border}`, bgcolor: tc.controlBg, width: cw, maxWidth: cw, boxSizing: "border-box" }}>
                <Typography sx={{ ...controlLabel, color: tc.textMuted, flexShrink: 0 }}>{dl[a]}</Typography>
                {loop ? (
                  <Slider
                    value={loopSliderValues(a)}
                    onChange={(_, v) => {
                      handleLoopSliderChange(a, v as number[]);
                    }}
                    disableSwap
                    min={0}
                    max={sliceMaxes[a]}
                    disabled={lockPlayback}
                    size="small"
                    valueLabelDisplay="auto"
                    sx={{
                      ...sliderStyles.small,
                      flex: 1,
                      minWidth: 40,
                      "& .MuiSlider-thumb[data-index='0']": { width: 8, height: 8, bgcolor: tc.textMuted },
                      "& .MuiSlider-thumb[data-index='1']": { width: 12, height: 12 },
                      "& .MuiSlider-thumb[data-index='2']": { width: 8, height: 8, bgcolor: tc.textMuted },
                      "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" },
                    }}
                    aria-label={`Loop range and current ${dl[a]} slice (${sliceValues[a] + 1} of ${sliceMaxes[a] + 1}, loop ${loopStarts[a] + 1} to ${effectiveLoopEnds[a] + 1})`}
                    valueLabelFormat={(v) => `${v as number}`}
                  />
                ) : (
                  <Slider
                    value={sliderValues[a]}
                    min={0}
                    max={sliceMaxes[a]}
                    onChange={sliceSetters[a]}
                    disabled={lockPlayback}
                    size="small"
                    sx={{ ...sliderStyles.small, flex: 1, minWidth: 40 }}
                    aria-label={`${dl[a]} slice ${sliceValues[a] + 1} of ${sliceMaxes[a] + 1}`}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(v) => `${v as number}`}
                  />
                )}
                <Typography sx={{ ...typography.value, color: tc.textMuted, minWidth: 28, textAlign: "right", flexShrink: 0 }}>
                  {sliceValues[a]}/{sliceMaxes[a]}
                </Typography>
              </Box>
              )}
            </Box>
          );
        });
        return thinZ ? (
          <Box sx={{ display: "flex", alignItems: "flex-start", gap: `${SPACING.SM}px`, justifyContent: "flex-start" }}>
            {panels[0]}
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px` }}>
              {panels[1]}
              {panels[2]}
            </Box>
          </Box>
        ) : (
          <Box sx={{ display: "grid", gridTemplate: thinZGridTemplate, rowGap: 0, columnGap: `${SPACING.SM}px`, justifyContent: "start" }}>
            {panels}
          </Box>
        );
      })()}
      </Box> {/* end right column (toolbar + slices) */}
      </Box> {/* end side-by-side row (3D volume + slices) */}
      {/* FFT controls row */}
      {effectiveShowFft && (
        <Box sx={{ ...panelControlRow, mt: `${SPACING.SM}px`, width: primaryPanelW, maxWidth: primaryPanelW, flexWrap: "wrap" }}>
          <Typography sx={{ ...controlLabel }}>FFT Scale</Typography>
          <Select disabled={lockDisplay} value={fftLogScale ? "log" : "linear"} onChange={(e) => setFftLogScale(e.target.value === "log")} size="small" sx={{ ...denseSelect, minWidth: 45 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT intensity scale (linear or logarithmic)" }}>
            <MenuItem value="linear">Lin</MenuItem>
            <MenuItem value="log">Log</MenuItem>
          </Select>
          <Typography sx={{ ...controlLabel }}>FFT Color</Typography>
          <Select disabled={lockDisplay} value={fftColormap} onChange={(e) => setFftColormap(String(e.target.value))} size="small" sx={{ ...denseSelect, minWidth: 60 }} MenuProps={themedMenuProps} inputProps={{ "aria-label": "FFT colormap" }}>
            {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
          </Select>
          <Typography sx={{ ...controlLabel }}>FFT Auto</Typography>
          <Switch checked={fftAuto} onChange={(e) => setFftAuto(e.target.checked)} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle automatic FFT contrast" }} />
          <Typography sx={{ ...controlLabel }} title="Apply a Hann window before zero-padding each slice FFT to reduce edge leakage.">Window</Typography>
          <Switch checked={!!fftWindow} onChange={(e) => setFftWindow(e.target.checked)} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle Hann window before FFT" }} />
        </Box>
      )}
      {/* Controls row with histogram anchored to the slice panel columns. */}
      {showControls && (!hideDisplay || !hideHistogram) && (() => {
        const histogramW = 110;
        const histogramH = controlRowHeight * 2 + SPACING.XS;
        return (
        <Box sx={{
          mt: `${SPACING.SM}px`,
          display: "flex",
          gap: `${SPACING.SM}px`,
          alignItems: "flex-start",
          width: "fit-content",
          maxWidth: panelTotalW,
          boxSizing: "border-box",
        }}>
          {!hideDisplay && (
            <Box sx={{ display: "flex", flexDirection: "column", gap: `${SPACING.XS}px`, justifyContent: "flex-start", minWidth: 0 }}>
              <Box sx={{ ...panelControlRow, width: primaryPanelW, maxWidth: primaryPanelW, flexWrap: "wrap" }}>
                <Typography sx={{ ...controlLabel }}>Color</Typography>
                <Select disabled={lockDisplay} size="small" value={cmap} onChange={(e) => setCmap(e.target.value)} MenuProps={themedMenuProps} sx={{ ...denseSelect, minWidth: 60 }} inputProps={{ "aria-label": "Image colormap" }}>
                  {COLORMAP_NAMES.map((name) => (<MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>))}
                </Select>
                <Typography sx={{ ...controlLabel }}>Colorbar</Typography>
                <Switch checked={showColorbar} onChange={(e) => { if (!lockDisplay) setShowColorbar(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle colorbar overlay" }} />
                <Typography sx={{ ...controlLabel }} title="CSS bilinear interpolation on image canvas. Off = pixelated.">Smooth</Typography>
                <Switch checked={smooth} onChange={(e) => { if (!lockDisplay) setSmooth(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle bilinear smoothing" }} />
              </Box>
              <Box sx={{ ...panelControlRow, width: primaryPanelW, maxWidth: primaryPanelW, flexWrap: "wrap" }}>
                {thinZ && (
                  <>
                    <Typography sx={{ ...controlLabel }} title="Depth-axis display height multiplier (1-30x). CSS-only stretch; data unchanged. Useful when nz << nxy (e.g. multislice ptycho).">Z stretch</Typography>
                    <Slider value={zStretch} min={1} max={30} step={0.5} onChange={(_, v) => { if (!lockDisplay) setZStretch(v as number); }} disabled={lockDisplay} size="small" valueLabelDisplay="auto" sx={{ ...sliderStyles.small, width: 80, mr: 1, "& .MuiSlider-valueLabel": { fontSize: 10, padding: "2px 4px" } }} aria-label="Depth axis display stretch multiplier" />
                  </>
                )}
                <Typography sx={{ ...controlLabel }} title="Show slice intersection guides across orthogonal panels.">Cross</Typography>
                <Switch checked={showCrosshair} onChange={(e) => { if (!lockDisplay) setShowCrosshair(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle crosshair overlay on slice panels" }} />
                <Typography sx={{ ...controlLabel }} title="Negate displayed values. Useful when phase sign is inverted.">Flip</Typography>
                <Switch checked={flip} onChange={(e) => { if (!lockDisplay) setFlip(e.target.checked); }} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Flip (negate) displayed values" }} />
                <Typography sx={{ ...controlLabel }}>Auto</Typography>
                <Switch checked={autoContrast} onChange={(e) => {
                  if (lockDisplay) return;
                  const on = e.target.checked;
                  setAutoContrast(on);
                  if (on && imageHistogramData) {
                    // ON → snap to 2/98 percentile.
                    const { vmin: pmin, vmax: pmax } = percentileClip(imageHistogramData, 2, 98);
                    const span = imageDataRange.max - imageDataRange.min;
                    if (span > 0) {
                      setImageVminPct(Math.max(0, Math.min(100, ((pmin - imageDataRange.min) / span) * 100)));
                      setImageVmaxPct(Math.max(0, Math.min(100, ((pmax - imageDataRange.min) / span) * 100)));
                    }
                  } else {
                    // OFF → reset slider(s) to full range 0/100 so user gets default contrast back.
                    setImageVminPct(0);
                    setImageVmaxPct(100);
                  }
                }} disabled={lockDisplay} size="small" sx={switchStyles.small} inputProps={{ "aria-label": "Toggle automatic percentile-based contrast" }} />
              </Box>
            </Box>
          )}
          {!hideHistogram && (
            <Box sx={{ display: "flex", flexDirection: "row", gap: `${SPACING.SM}px`, alignItems: "flex-start", justifyContent: "flex-start", opacity: lockHistogram ? 0.5 : 1, pointerEvents: lockHistogram ? "none" : "auto" }}>
              <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end", justifyContent: "flex-start" }}>
                <Histogram
                  data={imageHistogramData}
                  vminPct={imageVminPct}
                  vmaxPct={imageVmaxPct}
                  onRangeChange={(min, max) => {
                    if (!lockHistogram) {
                      // User drag overrides Auto - Auto would otherwise win and ignore slider.
                      if (autoContrast) setAutoContrast(false);
                      setImageVminPct(min);
                      setImageVmaxPct(max);
                    }
                  }}
                  width={histogramW}
                  height={histogramH}
                  theme={themeInfo.theme === "dark" ? "dark" : "light"}
                  dataMin={flip ? -imageDataRange.max : imageDataRange.min}
                  dataMax={flip ? -imageDataRange.min : imageDataRange.max}
                  pinBinsToRange={false}
                  ariaHidden
                />
              </Box>
            </Box>
          )}
        </Box>
        );
      })()}
      {/* Playback: transport + axis selector + fps + loop + bounce */}
      {!hidePlayback && (() => {
        return (
      <Box sx={{ ...panelControlRow, mt: `${SPACING.SM}px`, width: primaryPanelW, maxWidth: primaryPanelW, flexWrap: "nowrap" }}>
        <Select
          value={playAxis}
          onChange={(e) => { if (!lockPlayback) { setPlaying(false); setPlayAxis(Number(e.target.value)); } }}
          disabled={lockPlayback}
          size="small"
          sx={{ ...denseSelect, minWidth: 40 }}
          MenuProps={themedMenuProps}
          inputProps={{ "aria-label": "Playback axis (Z, Y, X, or All)" }}
        >
          <MenuItem value={0}>{dl[0]}</MenuItem>
          <MenuItem value={1}>{dl[1]}</MenuItem>
          <MenuItem value={2}>{dl[2]}</MenuItem>
          <MenuItem value={3}>All</MenuItem>
        </Select>
        <Stack direction="row" spacing={0} sx={{ flexShrink: 0 }}>
          <IconButton size="small" disabled={lockPlayback} onClick={() => { if (!lockPlayback) { setReverse(true); setPlaying(true); } }} sx={{ color: reverse && playing ? tc.accent : tc.textMuted, p: 0.25 }} aria-label="Play in reverse" title="Play reverse">
            <FastRewindIcon sx={{ fontSize: 18 }} />
          </IconButton>
          <IconButton size="small" disabled={lockPlayback} onClick={() => { if (!lockPlayback) setPlaying(!playing); }} sx={{ color: tc.accent, p: 0.25 }} aria-label={playing ? "Pause playback" : "Play"} title={playing ? "Pause (Space)" : "Play (Space)"}>
            {playing ? <PauseIcon sx={{ fontSize: 18 }} /> : <PlayArrowIcon sx={{ fontSize: 18 }} />}
          </IconButton>
          <IconButton size="small" disabled={lockPlayback} onClick={() => { if (!lockPlayback) { setReverse(false); setPlaying(true); } }} sx={{ color: !reverse && playing ? tc.accent : tc.textMuted, p: 0.25 }} aria-label="Play forward" title="Play forward">
            <FastForwardIcon sx={{ fontSize: 18 }} />
          </IconButton>
          <IconButton size="small" disabled={lockPlayback} onClick={() => {
            if (!lockPlayback) {
              setPlaying(false);
              if (playAxis === 3) {
                for (let a = 0; a < 3; a++) sliceSettersRef.current[a](loopStarts[a]);
              } else {
                sliceSettersRef.current[playAxis](loopStarts[playAxis]);
              }
            }
          }} sx={{ color: tc.textMuted, p: 0.25 }} aria-label="Stop and rewind to loop start" title="Stop">
            <StopIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Stack>
        <Typography sx={{ ...controlLabel, color: tc.textMuted, flexShrink: 0 }}>fps</Typography>
        <Slider disabled={lockPlayback} value={fps} min={1} max={60} step={1} onChange={(_, v) => setFps(v as number)} size="small" sx={{ ...sliderStyles.small, width: 35, flexShrink: 0 }} aria-label={`Playback frames per second (${Math.round(fps)})`} valueLabelDisplay="auto" />
        <Typography sx={{ ...controlLabel, color: tc.textMuted, flexShrink: 0 }}>Loop</Typography>
        <Switch size="small" checked={loop} onChange={() => { if (!lockPlayback) setLoop(!loop); }} disabled={lockPlayback} sx={{ ...switchStyles.small, flexShrink: 0 }} inputProps={{ "aria-label": "Toggle loop playback" }} />
        <Typography sx={{ ...controlLabel, color: tc.textMuted, flexShrink: 0 }}>Bounce</Typography>
        <Switch size="small" checked={boomerang} onChange={() => { if (!lockPlayback) setBoomerang(!boomerang); }} disabled={lockPlayback} sx={{ ...switchStyles.small, flexShrink: 0 }} inputProps={{ "aria-label": "Toggle bounce (ping-pong) playback" }} />
      </Box>
        );
      })()}
    </Box>
  );
}

// anywidget v0.9+ deprecates `export render` in favor of `export default { render }`.
const render = createRender(Show3DSlices);
export default { render };
