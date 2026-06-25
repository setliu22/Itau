import numpy as np


def slice_image(
    img: np.ndarray,
    slice_width: int = 4,
    stride: int | None = None,
    remove_padding: bool = False,
    pad_to_width: int | None = None,
) -> np.ndarray:
    """Slice a 2-D grayscale image into overlapping (or non-overlapping) column strips.

    Args:
        img: float32 array of shape (height, width) with values in [0, 1]
        slice_width: width of each slice in pixels
        stride: step between consecutive slice start positions; defaults to
            slice_width (non-overlapping) when None
        remove_padding: if True, strip leading and trailing columns where every
            pixel is exactly 0.0 or 1.0 (pure background, no ink) before slicing
        pad_to_width: if set, pad (or truncate) the image to exactly this many
            columns before slicing, using the corner pixel value as background.
            Setting slice_width == pad_to_width yields a single strip of the
            whole image.

    Returns:
        float32 array of shape (num_slices, height, slice_width)
    """
    if stride is None:
        stride = slice_width
    if remove_padding:
        # A column is padding if every pixel is exactly at the background extreme
        # (0.0 or 1.0) with no antialiased ink values in between.
        col_is_padding = np.all((img == 0.0) | (img == 1.0), axis=0)
        content_cols = np.where(~col_is_padding)[0]
        if content_cols.size > 0:
            img = img[:, content_cols[0] : content_cols[-1] + 1]

    if pad_to_width is not None:
        height, width = img.shape
        if width < pad_to_width:
            bg_val = float(img[0, 0])  # corner pixel is always background
            pad = np.full((height, pad_to_width - width), bg_val, dtype=np.float32)
            img = np.concatenate([img, pad], axis=1)
        elif width > pad_to_width:
            img = img[:, :pad_to_width]

    height, width = img.shape
    starts = range(0, width - slice_width + 1, stride)
    slices = []
    for x in starts:
        strip = img[:, x : x + slice_width]  # (height, slice_width)
        slices.append(strip)
    return np.stack(slices, axis=0)


if __name__ == "__main__":
    from renderer import render_name

    for word in ["Google", "G00gle"]:
        img = render_name(word)
        slices = slice_image(img)
        print(f"slice_image(render_name({word!r})) -> shape {slices.shape}")

    print()
    img = render_name("Google")
    no_trim = slice_image(img, remove_padding=False)
    trimmed = slice_image(img, remove_padding=True)
    print(f"render_name('Google') -> image width: {img.shape[1]}px")
    print(f"remove_padding=False -> {no_trim.shape[0]} slices")
    print(f"remove_padding=True  -> {trimmed.shape[0]} slices")
