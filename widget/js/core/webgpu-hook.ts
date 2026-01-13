/**
 * Shared WebGPU FFT hook for all widgets.
 * Provides consistent GPU acceleration across Show4DSTEM and Reconstruct.
 */

import * as React from "react";
import { getWebGPUFFT, WebGPUFFT } from "../webgpu-fft";

export interface UseWebGPUResult {
  gpuFFT: WebGPUFFT | null;
  gpuReady: boolean;
}

/**
 * Hook to initialize WebGPU FFT on mount.
 * Returns null if WebGPU is not available (falls back to CPU).
 */
export function useWebGPU(): UseWebGPUResult {
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;

    getWebGPUFFT().then(fft => {
      if (cancelled) return;
      if (fft) {
        gpuFFTRef.current = fft;
        setGpuReady(true);
      }
    });

    return () => { cancelled = true; };
  }, []);

  return { gpuFFT: gpuFFTRef.current, gpuReady };
}
