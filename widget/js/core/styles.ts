/**
 * Shared CSS for widget components.
 * Base styles used by Show2D, Show3D, and other vanilla JS widgets.
 */

export const baseCSS = `
/* ============================================================================
   Base Styles - Shared across Show2D, Show3D
   ============================================================================ */

/* Root container */
.widget-root {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background-color: var(--bg, #1a1a1a);
  color: var(--text-primary, #fff);
  padding: 12px;
  border-radius: 6px;
  display: inline-block;
  min-width: 320px;

  --bg: #1a1a1a;
  --bg-panel: #222;
  --bg-input: #333;
  --bg-canvas: #000;
  --border: #444;
  --border-light: #555;
  --text-primary: #fff;
  --text-secondary: #aaa;
  --text-muted: #888;
  --text-dim: #666;
  --accent: #0af;
  --accent-green: #0f0;
  --accent-red: #f00;
}

.widget-root:focus {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

/* Title bar */
.widget-title-bar {
  margin-bottom: 8px;
}

.widget-title {
  color: var(--accent);
  font-weight: bold;
  font-size: 13px;
}

/* Canvas container */
.widget-canvas-container {
  position: relative;
  background-color: var(--bg-canvas);
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
}

.widget-canvas {
  display: block;
  image-rendering: pixelated;
  image-rendering: crisp-edges;
}

.widget-overlay {
  position: absolute;
  top: 0;
  left: 0;
  pointer-events: none;
}

/* Panels */
.widget-panel {
  background-color: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 6px;
}

.widget-panel-title {
  font-size: 10px;
  color: var(--text-muted);
  text-transform: uppercase;
  margin-bottom: 4px;
}

/* Control group */
.widget-control-group {
  display: flex;
  align-items: center;
  gap: 6px;
  background-color: var(--bg-panel);
  padding: 4px 8px;
  border-radius: 4px;
  border: 1px solid var(--border);
}

/* Buttons */
.widget-btn {
  background-color: var(--bg-input);
  border: 1px solid var(--border-light);
  color: var(--text-secondary);
  min-width: 32px;
  height: 28px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
  padding: 0 8px;
}

.widget-btn:hover {
  background-color: var(--border);
  color: var(--text-primary);
}

.widget-btn:active,
.widget-btn-active {
  background-color: var(--accent);
  color: #000;
  border-color: var(--accent);
}

.widget-btn-primary {
  background-color: var(--accent);
  color: #000;
  border-color: var(--accent);
}

.widget-btn-primary:hover {
  background-color: #0cf;
}

/* Slider */
.widget-slider {
  flex: 1;
  height: 6px;
  -webkit-appearance: none;
  appearance: none;
  background: var(--border);
  border-radius: 3px;
  cursor: pointer;
}

.widget-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  background: var(--accent);
  border-radius: 50%;
  cursor: pointer;
  border: 2px solid var(--bg);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}

.widget-slider::-moz-range-thumb {
  width: 14px;
  height: 14px;
  background: var(--accent);
  border-radius: 50%;
  cursor: pointer;
  border: 2px solid var(--bg);
}

.widget-slider:focus {
  outline: none;
}

/* Inputs */
.widget-input {
  background-color: var(--bg-input);
  border: 1px solid var(--border-light);
  color: var(--text-secondary);
  border-radius: 3px;
  padding: 4px 6px;
  font-size: 11px;
  font-family: monospace;
}

.widget-input:focus {
  outline: none;
  border-color: var(--accent);
  color: var(--text-primary);
}

.widget-input-small {
  width: 45px;
  height: 24px;
  text-align: center;
}

/* Toggles / Checkboxes */
.widget-toggle {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--text-muted);
  cursor: pointer;
  user-select: none;
}

.widget-toggle:hover {
  color: var(--text-primary);
}

.widget-toggle input[type="checkbox"] {
  width: 14px;
  height: 14px;
  accent-color: var(--accent);
  cursor: pointer;
}

/* Select */
.widget-select {
  background-color: var(--bg-input);
  border: 1px solid var(--border-light);
  color: var(--text-secondary);
  border-radius: 3px;
  padding: 4px 8px;
  font-size: 11px;
  cursor: pointer;
}

.widget-select:focus {
  outline: none;
  border-color: var(--accent);
}

/* Stats bar */
.widget-stats-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  background-color: var(--bg-panel);
  padding: 6px 12px;
  border-radius: 4px;
  border: 1px solid var(--border);
}

.widget-stat-item {
  display: flex;
  gap: 6px;
  align-items: baseline;
}

.widget-stat-label {
  font-size: 10px;
  color: var(--text-dim);
}

.widget-stat-value {
  font-size: 11px;
  font-family: monospace;
  color: var(--accent);
}

/* Labels */
.widget-label {
  color: var(--text-secondary);
  font-size: 11px;
}

.widget-label-small {
  color: var(--text-dim);
  font-size: 10px;
}

/* Layout helpers */
.widget-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.widget-col {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.widget-flex {
  flex: 1;
}

/* Monospace text */
.widget-mono {
  font-family: monospace;
}
`;
