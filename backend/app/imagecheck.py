"""Deterministic image and accessibility checks. Uses Pillow only, no AI."""
import io
import os
import re
import json
import base64
from PIL import Image

# Refuse very large images instead of slowing the server down
Image.MAX_IMAGE_PIXELS = 25_000_000

CONFIG_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "brand_pack", "image_guidelines.json"))

MAX_BYTES = 2_500_000
FORMAT_TO_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}

DEFAULT_CONFIG = {
    "palette": [{"name": "Navy", "hex": "#14213D"}, {"name": "Gold", "hex": "#D4AF37"}],
    "allow_neutrals": True, "tolerance": 70, "min_share": 0.08,
    "min_width": 600, "min_height": 315, "alt_min_length": 10, "alt_max_length": 125
}


def load_config() -> dict:
    """Read the image guidelines (palette and limits) from the brand pack."""
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_CONFIG
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_data_url(data_url: str) -> bytes:
    """Turn a browser data URL into raw image bytes, with size checks."""
    match = re.match(r"^data:image/[a-zA-Z0-9.+-]+;base64,(.+)$", data_url, re.DOTALL)
    if not match:
        raise ValueError("The image data is not valid. Please choose a PNG, JPEG or WEBP file.")
    try:
        raw = base64.b64decode(match.group(1), validate=False)
    except Exception:
        raise ValueError("The image data could not be decoded.")
    if len(raw) > MAX_BYTES:
        raise ValueError("The image is larger than 2.5 MB. Please use a smaller file.")
    return raw


def open_image(raw: bytes):
    """Open the image and return (image, mime type). Only PNG, JPEG and WEBP are allowed."""
    try:
        img = Image.open(io.BytesIO(raw))
        fmt = img.format
        img.load()
    except Exception:
        raise ValueError("This file could not be read as an image.")
    if fmt not in FORMAT_TO_MIME:
        raise ValueError("Only PNG, JPEG or WEBP images are supported.")
    return img, FORMAT_TO_MIME[fmt]


def to_rgb(img):
    """Convert to RGB, placing transparent images on a white background."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return img.convert("RGB")


def make_thumbnail(img) -> str:
    """A small JPEG copy of the image, stored so the manager can see it."""
    rgb = to_rgb(img)
    rgb.thumbnail((640, 640))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------- colour maths ----------
def hex_to_rgb(value: str) -> tuple:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple) -> str:
    return "#%02X%02X%02X" % tuple(rgb)


def distance(a: tuple, b: tuple) -> float:
    """How far apart two colours are (straight-line distance in RGB)."""
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def is_neutral(rgb: tuple) -> bool:
    """Black, white and greys are always allowed."""
    return max(rgb) - min(rgb) < 20


def luminance(rgb: tuple) -> float:
    """Relative luminance as defined by WCAG."""
    def channel(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: tuple, b: tuple) -> float:
    """WCAG contrast ratio between two colours, from 1 to 21."""
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def dominant_colours(img, n: int = 6) -> list:
    """The main colours of the image with their share of the pixels, biggest first."""
    small = to_rgb(img)
    small.thumbnail((160, 160))
    quantized = small.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    total = small.width * small.height
    colours = []
    for count, index in quantized.getcolors():
        rgb = tuple(int(v) for v in palette[index * 3: index * 3 + 3])
        colours.append({"rgb": rgb, "hex": rgb_to_hex(rgb), "share": count / total})
    colours.sort(key=lambda c: -c["share"])
    return colours


# ---------- findings ----------
def make_finding(category, severity, original, reason, source, quote,
                 suggestion="", confidence=1.0, verdict="CONTRADICTED") -> dict:
    return {
        "category": category,
        "severity": severity,
        "original_text": original,
        "verdict": verdict,
        "reason": reason,
        "source_name": source,
        "source_quote": quote,
        "suggestion": suggestion,
        "confidence": confidence,
        "status": "open"
    }


def palette_findings(colours: list, cfg: dict) -> list:
    """Flag large areas of colour that are not close to any approved brand colour."""
    palette = [{"name": p["name"], "hex": p["hex"], "rgb": hex_to_rgb(p["hex"])} for p in cfg["palette"]]
    approved = ", ".join(f'{p["name"]} {p["hex"]}' for p in palette)
    found = []
    for c in colours:
        if c["share"] < cfg["min_share"]:
            continue
        if cfg.get("allow_neutrals", True) and is_neutral(c["rgb"]):
            continue
        nearest = min(palette, key=lambda p: distance(c["rgb"], p["rgb"]))
        if distance(c["rgb"], nearest["rgb"]) <= cfg["tolerance"]:
            continue
        found.append(make_finding(
            "image_palette", "medium",
            f'{c["hex"]} ({round(c["share"] * 100)}% of the image)',
            f'This colour is not close to any approved brand colour. The nearest is {nearest["name"]}.',
            "image_guidelines.json", f"Approved colours: {approved}",
            f'Use {nearest["name"]} {nearest["hex"]} instead.'))
    return found[:3]


def contrast_finding(colours: list, cfg: dict):
    """Estimate text contrast from the two main colours. This is an estimate, not a text reading."""
    if len(colours) < 2:
        return None
    background = colours[0]
    others = [c for c in colours[1:] if c["share"] >= 0.02]
    if not others:
        return None
    foreground = others[0]
    ratio = contrast_ratio(background["rgb"], foreground["rgb"])
    if ratio >= 4.5:
        return None

    # Suggest the approved colour (or white) with the best contrast on the background
    candidates = [(p["name"], p["hex"]) for p in cfg["palette"]] + [("White", "#FFFFFF")]
    best = max(candidates, key=lambda c: contrast_ratio(background["rgb"], hex_to_rgb(c[1])))
    best_ratio = contrast_ratio(background["rgb"], hex_to_rgb(best[1]))

    return make_finding(
        "image_contrast", "high" if ratio < 3 else "medium",
        f'{foreground["hex"]} on {background["hex"]} ({ratio:.1f}:1)',
        f"The two main colours have a contrast ratio of {ratio:.1f}:1. Text needs at least 4.5:1 "
        "(3:1 for large text). This is an estimate from the two main colours, not a reading of the "
        "actual text, so check it by eye.",
        "WCAG 2.1, success criterion 1.4.3 (Contrast Minimum)",
        "Text needs a contrast ratio of at least 4.5:1, or 3:1 for large text.",
        f"Try {best[0]} {best[1]} on {background['hex']} ({best_ratio:.1f}:1).")


def alt_text_findings(alt_text: str, cfg: dict) -> list:
    """Check the alt text a screen reader would read out."""
    source = "WCAG 2.1, success criterion 1.1.1 (Non-text Content)"
    quote = "Images need a text alternative that serves the same purpose."
    alt = (alt_text or "").strip()

    if not alt:
        return [make_finding("alt_text", "high", "(no alt text)",
                             "No alt text was provided, so a screen reader user gets nothing from this image.",
                             source, quote,
                             "Describe what the image shows and why it is there, in one short sentence.")]
    findings = []
    if re.match(r"^[\w\- ]+\.(png|jpe?g|webp|gif)$", alt, re.IGNORECASE) or \
            re.match(r"^(img|dsc|image|screenshot|banner)[_\- ]?\d+", alt, re.IGNORECASE):
        findings.append(make_finding("alt_text", "medium", alt,
                                     "The alt text looks like a file name. It does not describe the image.",
                                     source, quote, "Describe what the image shows."))
    elif len(alt) < cfg["alt_min_length"]:
        findings.append(make_finding("alt_text", "medium", alt,
                                     f"The alt text is very short ({len(alt)} characters).",
                                     source, quote, "Say what the image shows in a short sentence."))
    if len(alt) > cfg["alt_max_length"]:
        findings.append(make_finding("alt_text", "low", alt[:60] + "...",
                                     f"The alt text is long ({len(alt)} characters). "
                                     f"Keep it under {cfg['alt_max_length']}.",
                                     source, quote, "Shorten it. Put long text in the page itself."))
    if re.match(r"^(image|picture|photo|graphic) of\b", alt, re.IGNORECASE):
        findings.append(make_finding("alt_text", "low", alt,
                                     "Screen readers already say it is an image, so 'image of' is repeated.",
                                     source, quote,
                                     re.sub(r"^(image|picture|photo|graphic) of\s+", "", alt, flags=re.IGNORECASE)))
    return findings


def image_findings(img, alt_text: str):
    """Run every deterministic check. Returns (findings, dominant colours)."""
    cfg = load_config()
    findings = []

    width, height = img.size
    if width < cfg["min_width"] or height < cfg["min_height"]:
        findings.append(make_finding(
            "image_size", "low", f"{width} x {height} pixels",
            f"The image is smaller than {cfg['min_width']} x {cfg['min_height']} pixels and may look blurry.",
            "image_guidelines.json", f"Minimum size: {cfg['min_width']} x {cfg['min_height']} pixels",
            "Use a larger version of the image."))

    colours = dominant_colours(img)
    findings += palette_findings(colours, cfg)
    contrast = contrast_finding(colours, cfg)
    if contrast:
        findings.append(contrast)
    findings += alt_text_findings(alt_text, cfg)
    return findings, colours