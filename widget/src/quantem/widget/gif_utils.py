"""
Shared animated-GIF export helper for Show3D.

A widget produces per-frame contrast-normalized uint8 arrays (so the GIF matches
the histogram range / log scale the operator set); these helpers colorize with
the chosen matplotlib colormap, burn in a scale bar whose format matches the
widget's figure-export scale bar exactly (js/figure.ts), downscale by a quality
factor, and assemble the animation at the widget's fps. No widget dependency:
only numpy / matplotlib / PIL.
"""
import math
import pathlib

import numpy as np

# Resolution multiplier per quality tier. GIF is a 256-colour palette format
# regardless, so quality here means spatial resolution (and therefore file size).
QUALITY_SCALE = {"high": 1.0, "medium": 0.6, "low": 0.35}


def _round_to_nice_value(value: float) -> float:
    """Port of js/figure.ts roundToNiceValue: snap to 1 / 2 / 5 / 10 x 10^n."""
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    normalized = value / magnitude
    if normalized < 1.5:
        return magnitude
    if normalized < 3.5:
        return 2 * magnitude
    if normalized < 7.5:
        return 5 * magnitude
    return 10 * magnitude


def _unit_symbol(unit: str) -> str:
    """Port of js/figure.ts unitSymbol: render the conventional glyph."""
    u = (unit or "").strip()
    lc = u.lower()
    if lc in ("micron", "microns", "um") or u in ("μm", "µm"):
        return "µm"
    if lc in ("angstrom", "angstroms", "ang", "a") or u == "Å":
        return "Å"
    if lc in ("nanometer", "nanometers", "nm"):
        return "nm"
    if lc in ("picometer", "picometers", "pm"):
        return "pm"
    if lc in ("millimeter", "millimeters", "mm"):
        return "mm"
    return u


def _format_scale_label(value: float, unit: str) -> str:
    """Port of js/figure.ts formatScaleLabel."""
    nice = _round_to_nice_value(value)
    sym = _unit_symbol(unit)
    return f"{round(nice)} {sym}" if nice >= 1 else f"{nice:.2f} {sym}"


def _draw_scalebar(img, pixel_size: float, unit: str):
    """Burn a bottom-right scale bar matching js/figure.ts exportFigure exactly.

    White bar + bold centered label with a 1px drop shadow, sized relative to the
    image: targetBarPx = max(60, w*0.15), thickness = max(4, h*0.012),
    font = max(14, h*0.04), margin = max(12, w*0.03). The bar physical length is
    snapped to a 1/2/5/10 value, identical to the on-screen and figure exports.
    """
    from PIL import ImageDraw, ImageFont
    if pixel_size <= 0:
        return img
    width, height = img.size
    target_bar_px = max(60.0, width * 0.15)
    bar_thickness = max(4, round(height * 0.012))
    # Slightly smaller than figure.ts (0.04) so the GIF label is less heavy.
    font_size = max(12, round(height * 0.032))
    margin = max(12, round(width * 0.03))
    nice_phys = _round_to_nice_value(target_bar_px * pixel_size)
    bar_px = nice_phys / pixel_size
    bar_y = height - margin
    bar_x = width - bar_px - margin
    draw = ImageDraw.Draw(img)
    # 1px drop shadow (figure.ts uses shadowOffset 1,1): black underlay then white.
    draw.rectangle([bar_x + 1, bar_y + 1, bar_x + bar_px + 1, bar_y + bar_thickness + 1], fill=(0, 0, 0))
    draw.rectangle([bar_x, bar_y, bar_x + bar_px, bar_y + bar_thickness], fill=(255, 255, 255))
    label = _format_scale_label(nice_phys, unit)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    tb = draw.textbbox((0, 0), label, font=font)
    tw = tb[2] - tb[0]
    # textAlign center, baseline bottom, 4px above the bar (matches fillText).
    tx = bar_x + bar_px / 2 - tw / 2
    ty = bar_y - 4 - (tb[3] - tb[1])
    draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0), font=font)
    draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
    return img


def colorize(normalized_uint8: np.ndarray, cmap_name: str):
    """Map a 0-255 normalized frame through a matplotlib colormap to an RGB PIL image."""
    from matplotlib import colormaps
    from PIL import Image
    cmap_fn = colormaps.get_cmap(cmap_name)
    rgb = (cmap_fn(normalized_uint8 / 255.0)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def finalize_frame(img, quality: str, pixel_size: float, unit: str):
    """Downscale by the quality factor, then draw the scale bar at output res."""
    scale = QUALITY_SCALE.get(quality, 1.0)
    if scale < 1.0:
        from PIL import Image
        width, height = img.size
        img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
    # Each output pixel spans pixel_size / scale of sample after the downscale.
    return _draw_scalebar(img, pixel_size / scale if scale > 0 else pixel_size, unit)


def write_gif(frames: list, path: str | pathlib.Path, fps: float) -> pathlib.Path:
    """Assemble RGB PIL frames into a looping GIF at the given fps."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(1, int(round(1000.0 / max(0.1, fps))))
    frames[0].save(
        str(path), save_all=True, append_images=frames[1:],
        duration=duration, loop=0, optimize=True, disposal=2,
    )
    return path
