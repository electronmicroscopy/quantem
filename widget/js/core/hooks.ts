/**
 * Shared React hooks for widget functionality.
 * Provides reusable zoom/pan and resize logic.
 */

import * as React from "react";

// ============================================================================
// Constants
// ============================================================================
export const ZOOM_LIMITS = {
  MIN: 0.5,
  MAX: 10,
  WHEEL_IN: 1.1,
  WHEEL_OUT: 0.9,
} as const;

// ============================================================================
// Types
// ============================================================================
export interface ZoomPanState {
  zoom: number;
  panX: number;
  panY: number;
}

export const DEFAULT_ZOOM_PAN: ZoomPanState = {
  zoom: 1,
  panX: 0,
  panY: 0,
};

// ============================================================================
// useZoomPan Hook
// ============================================================================
export interface UseZoomPanOptions {
  canvasRef: React.RefObject<HTMLCanvasElement>;
  canvasWidth: number;
  canvasHeight: number;
  initialState?: ZoomPanState;
}

export interface UseZoomPanResult {
  state: ZoomPanState;
  setState: React.Dispatch<React.SetStateAction<ZoomPanState>>;
  reset: () => void;
  handleWheel: (e: React.WheelEvent) => void;
  handleMouseDown: (e: React.MouseEvent) => void;
  handleMouseMove: (e: React.MouseEvent) => void;
  handleMouseUp: () => void;
  handleDoubleClick: () => void;
  isDragging: boolean;
}

export function useZoomPan(options: UseZoomPanOptions): UseZoomPanResult {
  const { canvasRef, canvasWidth, canvasHeight, initialState = DEFAULT_ZOOM_PAN } = options;

  const [state, setState] = React.useState<ZoomPanState>(initialState);
  const [isDragging, setIsDragging] = React.useState(false);
  const [dragStart, setDragStart] = React.useState<{ x: number; y: number; panX: number; panY: number } | null>(null);

  const reset = React.useCallback(() => {
    setState(DEFAULT_ZOOM_PAN);
  }, []);

  const handleWheel = React.useCallback((e: React.WheelEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // Mouse position in canvas coordinates
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    // Canvas center
    const cx = canvasWidth / 2;
    const cy = canvasHeight / 2;

    setState(prev => {
      // Calculate position in image space
      const imageX = (mouseX - cx - prev.panX) / prev.zoom + cx;
      const imageY = (mouseY - cy - prev.panY) / prev.zoom + cy;

      // Apply zoom factor
      const zoomFactor = e.deltaY > 0 ? ZOOM_LIMITS.WHEEL_OUT : ZOOM_LIMITS.WHEEL_IN;
      const newZoom = Math.max(ZOOM_LIMITS.MIN, Math.min(ZOOM_LIMITS.MAX, prev.zoom * zoomFactor));

      // Calculate new pan to keep mouse position fixed
      const newPanX = mouseX - (imageX - cx) * newZoom - cx;
      const newPanY = mouseY - (imageY - cy) * newZoom - cy;

      return { zoom: newZoom, panX: newPanX, panY: newPanY };
    });
  }, [canvasRef, canvasWidth, canvasHeight]);

  const handleMouseDown = React.useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY, panX: state.panX, panY: state.panY });
  }, [state.panX, state.panY]);

  const handleMouseMove = React.useCallback((e: React.MouseEvent) => {
    if (!isDragging || !dragStart) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const dx = (e.clientX - dragStart.x) * scaleX;
    const dy = (e.clientY - dragStart.y) * scaleY;

    setState(prev => ({ ...prev, panX: dragStart.panX + dx, panY: dragStart.panY + dy }));
  }, [isDragging, dragStart, canvasRef]);

  const handleMouseUp = React.useCallback(() => {
    setIsDragging(false);
    setDragStart(null);
  }, []);

  const handleDoubleClick = React.useCallback(() => {
    reset();
  }, [reset]);

  return {
    state,
    setState,
    reset,
    handleWheel,
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleDoubleClick,
    isDragging,
  };
}

// ============================================================================
// useResize Hook
// ============================================================================
export interface UseResizeOptions {
  initialSize: number;
  minSize?: number;
  maxSize?: number;
}

export interface UseResizeResult {
  size: number;
  setSize: React.Dispatch<React.SetStateAction<number>>;
  isResizing: boolean;
  handleResizeStart: (e: React.MouseEvent) => void;
}

export function useResize(options: UseResizeOptions): UseResizeResult {
  const { initialSize, minSize = 80, maxSize = 600 } = options;

  const [size, setSize] = React.useState(initialSize);
  const [isResizing, setIsResizing] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number; y: number; size: number } | null>(null);

  const handleResizeStart = React.useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizing(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size });
  }, [size]);

  React.useEffect(() => {
    if (!isResizing || !resizeStart) return;

    const handleMouseMove = (e: MouseEvent) => {
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);
      const newSize = Math.max(minSize, Math.min(maxSize, resizeStart.size + delta));
      setSize(newSize);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      setResizeStart(null);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizing, resizeStart, minSize, maxSize]);

  return { size, setSize, isResizing, handleResizeStart };
}

// ============================================================================
// usePreventScroll Hook
// ============================================================================
export function usePreventScroll(refs: React.RefObject<HTMLElement | null>[]): void {
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const elements = refs.map(ref => ref.current).filter(Boolean) as HTMLElement[];

    elements.forEach(el => el.addEventListener("wheel", preventDefault, { passive: false }));

    return () => {
      elements.forEach(el => el.removeEventListener("wheel", preventDefault));
    };
  }, [refs]);
}
