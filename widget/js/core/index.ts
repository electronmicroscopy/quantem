/**
 * Core utilities for bobleesj.widget components.
 * Re-exports all shared modules.
 */

// Colors and theming
export { COLORS, colors, cssVars } from "./colors";

// Colormaps
export {
  COLORMAP_NAMES,
  COLORMAP_POINTS,
  COLORMAPS,
  applyColormapToImage,
  applyColormapValue,
  createColormapLUT,
} from "./colormaps";

// Canvas utilities
export {
  calculateDisplayScale,
  calculateNiceScaleBar,
  drawCrosshair,
  drawROI,
  drawROICircle,
  drawScaleBar,
  extractBytes,
  formatScaleBarLabel,
} from "./canvas";

// Formatting
export {
  clamp,
  formatBytes,
  formatDuration,
  formatNumber,
} from "./format";

// Base CSS
export { baseCSS } from "./styles";

// FFT and histogram utilities
export {
  computeMagnitude,
  fftshift,
  renderFFT,
  renderHistogram,
} from "./fft-utils";

// React hooks
export {
  DEFAULT_ZOOM_PAN,
  ZOOM_LIMITS,
  usePreventScroll,
  useResize,
  useZoomPan,
  type UseResizeOptions,
  type UseResizeResult,
  type UseZoomPanOptions,
  type UseZoomPanResult,
  type ZoomPanState,
} from "./hooks";

// WebGPU hook
export { useWebGPU, type UseWebGPUResult } from "./webgpu-hook";

// Advanced canvas utilities (high-DPI, colormap rendering)
export {
  canvasToBlob,
  drawCrosshairHiDPI,
  drawScaleBarHiDPI,
  drawWithZoomPan,
  formatScaleLabel,
  renderFloat32WithColormap,
  renderWithColormap,
  roundToNiceValue,
} from "./canvas-utils";

// Export utilities
export {
  compositeCanvases,
  downloadCanvas,
  exportGalleryAsZip,
  exportWithOverlay,
  generateFilename,
} from "./export";
