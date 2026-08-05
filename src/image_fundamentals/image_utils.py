"""Pure OpenCV/NumPy image utilities: I/O, transforms, conversions, and
inspection.  Every function is stateless — it takes an image (and
parameters) and returns a new image or data, without modifying inputs.

These form the fundamental building blocks used by every later module
(processing, morphology, detection, OCR).
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.utils.exceptions import ImageError
from src.utils.logger import setup_logger

_logger = setup_logger("ImageUtils")

# A BGR/ grayscale image is an ndarray; here we alias for clarity.
Array = np.ndarray


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------

def read_image(path: str | Path, grayscale: bool = False) -> Array:
    """Read an image from disk.

    Args:
        path: Path to the image file.
        grayscale: If True, load as a single-channel grayscale image.

    Returns:
        Image as a numpy array (BGR or grayscale).

    Raises:
        ImageError: If the file is missing or corrupted.
    """
    path = Path(path)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ImageError(f"Could not read image: {path}")
    _logger.info("Loaded %s | shape=%s dtype=%s",
                 path.name, image.shape, image.dtype)
    return image


def save_image(image: Array, path: str | Path,
               jpeg_quality: int = 95) -> str:
    """Save an image to disk.

    Args:
        image: Image array to save.
        path: Destination path.  The extension determines the format.
        jpeg_quality: JPEG quality (0-100), used only for .jpg/.jpeg.

    Returns:
        Absolute path of the saved file.

    Raises:
        ImageError: If the image is empty or writing fails.
    """
    if image is None or image.size == 0:
        raise ImageError("Cannot save an empty image.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    params = []
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, max(0, min(100, jpeg_quality))]

    ok = cv2.imwrite(str(path), image, params)
    if not ok:
        raise ImageError(f"Failed to save image: {path}")
    _logger.info("Saved %s | shape=%s", path.name, image.shape)
    return str(path.resolve())


# ----------------------------------------------------------------------
# Geometric transforms
# ----------------------------------------------------------------------

def resize(image: Array,
           width: Optional[int] = None,
           height: Optional[int] = None,
           scale: Optional[float] = None,
           interpolation: int = cv2.INTER_AREA) -> Array:
    """Resize an image by new dimensions or by a scale factor.

    Provide either (width, height) or scale.  If scale is given it
    overrides width/height.

    Args:
        image: Input image.
        width: Target width in pixels.
        height: Target height in pixels.
        scale: Uniform scale factor (0 < scale).
        interpolation: OpenCV interpolation method.

    Returns:
        Resized image.
    """
    if scale is not None:
        if scale <= 0:
            raise ImageError(f"Scale must be positive, got {scale}.")
        return cv2.resize(image, None, fx=scale, fy=scale,
                          interpolation=interpolation)
    if width is None or height is None:
        raise ImageError("Provide both width and height, or a scale factor.")
    if width <= 0 or height <= 0:
        raise ImageError(f"Invalid size: {width}x{height}")
    return cv2.resize(image, (width, height), interpolation=interpolation)


def crop(image: Array, x: int, y: int, width: int, height: int) -> Array:
    """Crop a rectangular region from an image.

    Args:
        image: Input image.
        x, y: Top-left corner of the crop region.
        width, height: Size of the crop region.

    Returns:
        Cropped image.

    Raises:
        ImageError: If the region is outside the image or has no area.
    """
    h, w = image.shape[:2]
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ImageError(f"Invalid crop region: x={x} y={y} "
                         f"w={width} h={height}")
    if x + width > w or y + height > h:
        raise ImageError(
            f"Crop region ({x+width}x{y+height}) exceeds image size "
            f"({w}x{h})."
        )
    return image[y:y + height, x:x + width]


def flip(image: Array, flip_code: int = 1) -> Array:
    """Flip an image.

    Args:
        image: Input image.
        flip_code: 0 = vertical, 1 = horizontal, -1 = both axes.

    Returns:
        Flipped image.
    """
    if flip_code not in (0, 1, -1):
        raise ImageError(f"Invalid flip code: {flip_code} "
                         "(use 0, 1, or -1)")
    return cv2.flip(image, flip_code)


def rotate(image: Array, angle: float,
           center: Optional[Tuple[int, int]] = None,
           scale: float = 1.0,
           border_value: Tuple[int, int, int] = (0, 0, 0)) -> Array:
    """Rotate an image by an angle (degrees, counter-clockwise).

    Args:
        image: Input image.
        angle: Rotation angle in degrees.
        center: Rotation centre (default: image centre).
        scale: Zoom scale applied during rotation.
        border_value: Fill colour for areas outside the rotated image.

    Returns:
        Rotated image (same output size as input).
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, matrix, (w, h),
                          borderValue=border_value)


# ----------------------------------------------------------------------
# Colour-space conversions
# ----------------------------------------------------------------------

def to_grayscale(image: Array) -> Array:
    """Convert an image to single-channel grayscale.

    If the input is already single-channel, it is returned unchanged.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_rgb(image: Array) -> Array:
    """Convert a BGR image to RGB (for display libraries)."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_bgr(image: Array) -> Array:
    """Convert an image to BGR (OpenCV standard).

    Accepts grayscale (upcast to 3 channels) or RGB input.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def to_hsv(image: Array) -> Array:
    """Convert a BGR image to HSV colour space."""
    if image.ndim == 2:
        image = to_bgr(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def to_bgr_from_hsv(image: Array) -> Array:
    """Convert an HSV image back to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


# ----------------------------------------------------------------------
# Pixel inspection & metadata
# ----------------------------------------------------------------------

def image_info(image: Array) -> Dict[str, object]:
    """Return metadata about an image.

    Returns:
        Dict with keys: 'height', 'width', 'channels', 'dtype',
        'pixel_count', and 'total_bytes'.
    """
    if image is None:
        raise ImageError("image_info received None.")
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    return {
        "height": h,
        "width": w,
        "channels": channels,
        "dtype": str(image.dtype),
        "pixel_count": h * w,
        "total_bytes": image.nbytes,
    }


def pixel_value(image: Array, x: int, y: int) -> object:
    """Read the value of a single pixel.

    Args:
        image: Input image.
        x, y: Pixel coordinates (column, row).

    Returns:
        A scalar (grayscale) or tuple (multi-channel) value.

    Raises:
        ImageError: If coordinates are out of bounds.
    """
    h, w = image.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        raise ImageError(f"Pixel ({x}, {y}) out of bounds for {w}x{h}.")
    return tuple(int(v) for v in image[y, x]) if image.ndim == 3 \
        else int(image[y, x])


def image_stats(image: Array) -> Dict[str, object]:
    """Compute min/max/mean/std statistics per channel.

    Returns:
        Dict with 'min', 'max', 'mean', 'std' as numpy arrays (one entry
        per channel), and 'shape'.
    """
    if image.ndim == 2:
        pixels = image.astype(np.float32)
    else:
        pixels = image.reshape(-1, image.shape[2]).astype(np.float32)
    return {
        "shape": image.shape,
        "min": pixels.min(axis=0),
        "max": pixels.max(axis=0),
        "mean": pixels.mean(axis=0),
        "std": pixels.std(axis=0),
    }


# ----------------------------------------------------------------------
# Histograms
# ----------------------------------------------------------------------

def histogram(image: Array, channel: int = 0,
              bins: int = 256,
              ranges: Tuple[int, int] = (0, 256)) -> Tuple[Array, Array]:
    """Compute a histogram of pixel intensities.

    Args:
        image: Input image (grayscale or multi-channel).
        channel: Channel index to histogram (0-based).
        bins: Number of bins.
        ranges: Intensity range as (min, max).

    Returns:
        Tuple (hist, bin_centers) where hist is a (bins,) numpy array
        and bin_centers are the bin midpoint values.
    """
    if image.ndim == 2:
        data = image
    else:
        data = image[:, :, channel]
    hist = cv2.calcHist([data], [0], None, [bins], list(ranges))
    hist = hist.flatten()
    centers = (np.arange(bins) + 0.5) * (ranges[1] - ranges[0]) / bins \
        + ranges[0]
    return hist, centers


def histogram_image(image: Array, bins: int = 256,
                    size: Tuple[int, int] = (420, 320)) -> Array:
    """Render a histogram as a grayscale image (no matplotlib needed).

    Args:
        image: Source image.
        bins: Number of bins.
        size: Output image (width, height).

    Returns:
        A grayscale image visualising the intensity histogram.
    """
    img_w, img_h = size
    hist, _ = histogram(to_grayscale(image), bins=bins)
    max_count = float(hist.max()) if hist.max() > 0 else 1.0
    norm = (hist / max_count * (img_h - 24)).astype(np.int32)

    canvas = np.full((img_h, img_w), 255, dtype=np.uint8)
    step = (img_w - 16) / bins
    for i, height in enumerate(norm):
        x = int(8 + i * step)
        cv2.rectangle(canvas, (x, img_h - 12),
                      (max(x + 1, int(x + step)), img_h - 12 - height),
                      color=0, thickness=-1)
    return canvas