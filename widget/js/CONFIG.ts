/**
 * Global configuration for bobleesj.widget.
 * Layout constants and styling presets for all widgets.
 */

// Import colors from single source of truth
import { COLORS, colors } from "./core/colors";
export { COLORS, colors };

// ============================================================================
// TYPOGRAPHY
// ============================================================================
export const TYPOGRAPHY = {
    LABEL: {
        color: COLORS.TEXT_SECONDARY,
        fontSize: 11,
    },
    LABEL_SMALL: {
        color: COLORS.TEXT_MUTED,
        fontSize: 10,
    },
    VALUE: {
        color: COLORS.TEXT_MUTED,
        fontSize: 10,
        fontFamily: "monospace",
    },
    TITLE: {
        color: COLORS.ACCENT,
        fontWeight: "bold" as const,
    },
};

// ============================================================================
// CONTROL PANEL STYLES
// ============================================================================
export const CONTROL_PANEL = {
    // Standard control group (height: 32px)
    GROUP: {
        bgcolor: COLORS.BG_PANEL,
        px: 1.5,
        py: 0.5,
        borderRadius: 1,
        border: `1px solid ${COLORS.BORDER}`,
        height: 32,
    },
    // Compact button
    BUTTON: {
        color: COLORS.TEXT_MUTED,
        fontSize: 10,
        cursor: "pointer",
        "&:hover": { color: COLORS.TEXT_PRIMARY },
        bgcolor: COLORS.BG_PANEL,
        px: 1,
        py: 0.25,
        borderRadius: 0.5,
        border: `1px solid ${COLORS.BORDER}`,
    },
    // Select dropdown
    SELECT: {
        minWidth: 90,
        bgcolor: COLORS.BG_INPUT,
        color: COLORS.TEXT_PRIMARY,
        fontSize: 11,
        "& .MuiSelect-select": {
            py: 0.5,
        },
    },
};

// ============================================================================
// CONTAINER STYLES
// ============================================================================
export const CONTAINER = {
    ROOT: {
        p: 2,
        // Use transparent background to inherit from parent (light/dark mode aware)
        bgcolor: "transparent",
        // Inherit text color from parent for theme awareness
        color: "inherit",
        fontFamily: "monospace",
        borderRadius: 1,
        // CRITICAL: Allow dropdowns to overflow
        overflow: "visible",
    },
    IMAGE_BOX: {
        bgcolor: "#000",
        border: `1px solid ${COLORS.BORDER}`,
        overflow: "hidden",
        position: "relative" as const,
    },
};

// ============================================================================
// SLIDER SIZES
// ============================================================================
export const SLIDER = {
    // Width presets
    WIDTH: {
        TINY: 60,      // Very compact (e.g., ms/frame slider)
        SMALL: 80,     // Standard small slider
        MEDIUM: 100,   // Medium slider
        LARGE: 120,    // Larger slider
    },
    // Container min-widths (for label + slider + value combos)
    CONTAINER: {
        COMPACT: 120,   // Minimal container
        STANDARD: 150,  // Standard container (e.g., delay slider)
        WIDE: 180,      // Wider container
    },
};

// ============================================================================
// PANEL SIZES (for canvases and image boxes)
// ============================================================================
export const PANEL = {
    // Main image canvas sizes
    MAIN: {
        DEFAULT: 300,   // Default main canvas size
        MIN: 150,       // Minimum resizable size
        MAX: 600,       // Maximum resizable size
    },
    // Side panels (FFT, histogram, etc.)
    SIDE: {
        DEFAULT: 150,   // Default side panel size
        MIN: 80,        // Minimum resizable size  
        MAX: 250,       // Maximum resizable size
    },
    // Show4DSTEM specific
    DP: {
        DEFAULT: 400,   // Diffraction pattern panel
    },
    VIRTUAL: {
        DEFAULT: 300,   // Virtual image panel
    },
    FFT: {
        DEFAULT: 300,   // FFT panel
    },
    // Gallery mode
    GALLERY: {
        IMAGE_SIZE: 200, // Target size for gallery images
        MIN_COLS: 2,     // Minimum columns
        MAX_COLS: 4,     // Maximum columns
    },
};

// ============================================================================
// ZOOM/PAN LIMITS
// ============================================================================
export const ZOOM = {
    MIN: 0.5,
    MAX: 10,
    WHEEL_FACTOR: {
        IN: 1.1,
        OUT: 0.9,
    },
};

// ============================================================================
// ANIMATION/PLAYBACK
// ============================================================================
export const PLAYBACK = {
    MS_PER_FRAME: {
        DEFAULT: 1000,  // Default: 1 fps
        MIN: 200,       // Fastest: 5 fps
        MAX: 3000,      // Slowest: ~0.33 fps
        STEP: 100,      // Step size for slider
    },
};

// ============================================================================
// LEGACY ALIASES (for backward compatibility during migration)
// These use camelCase keys to match existing widget code
// Note: `colors` is imported from core/colors.ts and re-exported at the top
// ============================================================================
export const typography = {
    label: TYPOGRAPHY.LABEL,
    labelSmall: TYPOGRAPHY.LABEL_SMALL,
    value: TYPOGRAPHY.VALUE,
    title: TYPOGRAPHY.TITLE,
};

export const controlPanel = {
    group: CONTROL_PANEL.GROUP,
    button: CONTROL_PANEL.BUTTON,
    select: CONTROL_PANEL.SELECT,
};

export const container = {
    root: CONTAINER.ROOT,
    imageBox: CONTAINER.IMAGE_BOX,
};
