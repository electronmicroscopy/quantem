/**
 * Shared color palette for all bobleesj.widget components.
 * Single source of truth for theming across Show2D, Show3D, Show4D, and Reconstruct.
 */

// Primary color definitions (SCREAMING_SNAKE_CASE for constants)
export const COLORS = {
  // Backgrounds
  BG: "#1a1a1a",
  BG_PANEL: "#222",
  BG_INPUT: "#333",
  BG_CANVAS: "#000",

  // Borders
  BORDER: "#444",
  BORDER_LIGHT: "#555",

  // Text
  TEXT_PRIMARY: "#fff",
  TEXT_SECONDARY: "#aaa",
  TEXT_MUTED: "#888",
  TEXT_DIM: "#666",

  // Accent colors
  ACCENT: "#0af",
  ACCENT_GREEN: "#0f0",
  ACCENT_RED: "#f00",
  ACCENT_ORANGE: "#fa0",
  ACCENT_CYAN: "#0cf",
  ACCENT_YELLOW: "#ff0",
} as const;

// Convenience alias with camelCase keys (for existing widget code)
export const colors = {
  bg: COLORS.BG,
  bgPanel: COLORS.BG_PANEL,
  bgInput: COLORS.BG_INPUT,
  bgCanvas: COLORS.BG_CANVAS,
  border: COLORS.BORDER,
  borderLight: COLORS.BORDER_LIGHT,
  textPrimary: COLORS.TEXT_PRIMARY,
  textSecondary: COLORS.TEXT_SECONDARY,
  textMuted: COLORS.TEXT_MUTED,
  textDim: COLORS.TEXT_DIM,
  accent: COLORS.ACCENT,
  accentGreen: COLORS.ACCENT_GREEN,
  accentRed: COLORS.ACCENT_RED,
  accentOrange: COLORS.ACCENT_ORANGE,
  accentCyan: COLORS.ACCENT_CYAN,
  accentYellow: COLORS.ACCENT_YELLOW,
} as const;

// CSS variable export for vanilla JS widgets
export const cssVars = `
  --bg: ${COLORS.BG};
  --bg-panel: ${COLORS.BG_PANEL};
  --bg-input: ${COLORS.BG_INPUT};
  --bg-canvas: ${COLORS.BG_CANVAS};
  --border: ${COLORS.BORDER};
  --border-light: ${COLORS.BORDER_LIGHT};
  --text-primary: ${COLORS.TEXT_PRIMARY};
  --text-secondary: ${COLORS.TEXT_SECONDARY};
  --text-muted: ${COLORS.TEXT_MUTED};
  --text-dim: ${COLORS.TEXT_DIM};
  --accent: ${COLORS.ACCENT};
  --accent-green: ${COLORS.ACCENT_GREEN};
  --accent-red: ${COLORS.ACCENT_RED};
  --accent-orange: ${COLORS.ACCENT_ORANGE};
  --accent-cyan: ${COLORS.ACCENT_CYAN};
  --accent-yellow: ${COLORS.ACCENT_YELLOW};
`;
