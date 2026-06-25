import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

# Resolve and cache fonts once at import time to avoid per-call filesystem searches.
_font_cache: dict[tuple[str | None, int], ImageFont.FreeTypeFont] = {}


def _get_font(font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont:
    key = (font_path, font_size)
    if key not in _font_cache:
        if font_path is None:
            try:
                path = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
                _font_cache[key] = ImageFont.truetype(path, font_size)
            except OSError:
                print("Warning: DejaVu Sans not found. Using default font, which may not support all characters.")
                _font_cache[key] = ImageFont.load_default()
        else:
            _font_cache[key] = ImageFont.truetype(font_path, font_size)
    return _font_cache[key]


def render_name(
    name: str,
    height: int = 32,
    font_path: str = None,
    background: str = "white",
) -> np.ndarray:
    """Render a string to a fixed-height grayscale NumPy array.

    Width scales with string length, with a minimum of 128px.
    Uses DejaVu Sans if no font_path is provided.

    Args:
        background: 'white' (black ink on white canvas) or 'black' (white ink
                    on black canvas). Pixel values reflect actual brightness:
                    white pixels = 1.0, black pixels = 0.0.

    Returns:
        float32 array of shape (height, width) with values in [0, 1].
        background='white': blank space = 1.0, ink = 0.0.
        background='black': blank space = 0.0, ink = 1.0.
    """
    if background not in ("white", "black"):
        raise ValueError(f"background must be 'white' or 'black', got {background!r}")

    font_size = int(height * 0.8)
    font = _get_font(font_path, font_size)

    # Measure text width using a scratch image
    scratch = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), name, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    width = max(128, text_w + 4)  # 128 is the minimum width, + 4 for some padding

    canvas_bg = 0 if background == "black" else 255
    ink = 255 if background == "black" else 0

    img = Image.new("L", (width, height), color=canvas_bg)
    draw = ImageDraw.Draw(img)

    x = (width - text_w) // 2
    y = (height - text_h) // 2 - bbox[1]
    draw.text((x, y), name, font=font, fill=ink)

    return np.array(img, dtype=np.float32) / 255.0


if __name__ == "__main__":
    for word in ["Google", "G00gle"]:
        for bg in ["white", "black"]:
            arr = render_name(word, background=bg)
            print(
                f"render_name({word!r}, background={bg!r}) -> "
                f"shape={arr.shape}  min={arr.min():.3f}  max={arr.max():.3f}"
            )
            print(arr)
