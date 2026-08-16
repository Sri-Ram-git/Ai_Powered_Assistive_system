"""Image processing operations: filtering, thresholding, edge detection,
brightness/contrast adjustment, and noise handling.

Every function is stateless: it takes an image (and parameters) and
returns a new image.  Inputs are never mutated.
"""
import cv2
import numpy as np

from src.image_fundamentals.image_utils import to_grayscale
from src.utils.exceptions import ProcessingError
from src.utils.logger import setup_logger

_logger = setup_logger("Processing")

Array = np.ndarray


# ----------------------------------------------------------------------
# Smoothing / denoising filters
# ----------------------------------------------------------------------

def blur_gaussian(image: Array, kernel_size: int = 5,
                  sigma: float = 0.0) -> Array:
    """Apply a Gaussian blur.

    Args:
        image: Input image.
        kernel_size: Odd kernel size (>= 1).
        sigma: Gaussian sigma; 0 lets OpenCV derive it from the kernel.

    Returns:
        Blurred image (same shape as input).
    """
    _check_odd_kernel(kernel_size)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def blur_median(image: Array, kernel_size: int = 5) -> Array:
    """Apply a median blur (good for salt-and-pepper noise).

    Args:
        image: Input image.
        kernel_size: Odd kernel size (>= 1).
    """
    _check_odd_kernel(kernel_size)
    return cv2.medianBlur(image, kernel_size)


def blur_bilateral(image: Array, d: int = 9,
                   sigma_color: float = 75,
                   sigma_space: float = 75) -> Array:
    """Apply a bilateral filter (denoise while preserving edges).

    Args:
        image: Input image.
        d: Diameter of the pixel neighbourhood.
        sigma_color: Filter sigma in colour space.
        sigma_space: Filter sigma in coordinate space.
    """
    if d % 2 == 0 or d < 1:
        raise ProcessingError(f"d must be a positive odd number, got {d}.")
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# ----------------------------------------------------------------------
# Thresholding
# ----------------------------------------------------------------------

def threshold(image: Array, thresh: float = 127,
              maxval: float = 255,
              method: int = cv2.THRESH_BINARY) -> Array:
    """Apply a fixed-intensity threshold.

    Args:
        image: Input image (converted to grayscale if needed).
        thresh: Threshold value.
        maxval: Value assigned to pixels passing the threshold.
        method: OpenCV thresholding type.

    Returns:
        Thresholded image.
    """
    gray = to_grayscale(image)
    _, result = cv2.threshold(gray, thresh, maxval, method)
    return result


def adaptive_threshold(image: Array, maxval: float = 255,
                       block_size: int = 11, c: float = 2,
                       method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C) -> Array:
    """Apply an adaptive (per-region) threshold.

    Args:
        image: Input image (converted to grayscale if needed).
        maxval: Value assigned to pixels passing the threshold.
        block_size: Odd size of the local neighbourhood.
        c: Constant subtracted from the local mean.
        method: OpenCV adaptive method.
    """
    _check_odd_kernel(block_size)
    gray = to_grayscale(image)
    return cv2.adaptiveThreshold(gray, maxval, method,
                                 cv2.THRESH_BINARY, block_size, c)


# ----------------------------------------------------------------------
# Edge detection
# ----------------------------------------------------------------------

def canny(image: Array, low: float = 50, high: float = 150) -> Array:
    """Detect edges with the Canny algorithm.

    Args:
        image: Input image (converted to grayscale if needed).
        low: Lower hysteresis threshold.
        high: Upper hysteresis threshold.
    """
    gray = to_grayscale(image)
    return cv2.Canny(gray, low, high)


def sobel_x(image: Array, ksize: int = 3) -> Array:
    """Vertical edges via the Sobel operator (x-direction)."""
    gray = to_grayscale(image)
    return _normalize(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize))


def sobel_y(image: Array, ksize: int = 3) -> Array:
    """Horizontal edges via the Sobel operator (y-direction)."""
    gray = to_grayscale(image)
    return _normalize(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize))


def sobel_magnitude(image: Array, ksize: int = 3) -> Array:
    """Edge magnitude combining Sobel x and y responses."""
    gray = to_grayscale(image)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    return _normalize(cv2.magnitude(gx, gy))


def laplacian(image: Array, ksize: int = 3) -> Array:
    """Detect edges with the Laplacian operator."""
    gray = to_grayscale(image)
    return _normalize(cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize))


def sharpen(image: Array, amount: float = 1.0) -> Array:
    """Sharpen an image with an unsharp-mask blend.

    Args:
        image: Input image.
        amount: Sharpening strength (0 = unchanged).
    """
    if amount < 0:
        raise ProcessingError(f"amount must be >= 0, got {amount}.")
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return sharpened


# ----------------------------------------------------------------------
# Brightness / contrast
# ----------------------------------------------------------------------

def adjust_brightness(image: Array, delta: float = 30) -> Array:
    """Shift brightness by delta (-255..255), with clipping.

    Args:
        image: Input image.
        delta: Brightness offset added to every pixel.
    """
    if not -255 <= delta <= 255:
        raise ProcessingError(f"delta must be in [-255, 255], got {delta}.")
    adjusted = image.astype(np.int16) + int(delta)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def adjust_contrast(image: Array, alpha: float = 1.5) -> Array:
    """Scale contrast by factor alpha.

    Args:
        image: Input image.
        alpha: Contrast multiplier (> 0; 1 = unchanged).
    """
    if alpha <= 0:
        raise ProcessingError(f"alpha must be > 0, got {alpha}.")
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


# ----------------------------------------------------------------------
# Noise
# ----------------------------------------------------------------------

def add_noise(image: Array, noise_type: str = "gaussian",
              amount: float = 0.02) -> Array:
    """Add synthetic noise to an image.

    Args:
        image: Input image.
        noise_type: 'gaussian' or 'salt_pepper'.
        amount: Noise intensity (gaussian: std fraction of 255;
            salt_pepper: fraction of pixels corrupted).

    Returns:
        Noisy image (float or uint8).
    """
    if not 0 <= amount <= 1:
        raise ProcessingError(f"amount must be in [0, 1], got {amount}.")

    noisy = image.astype(np.float32)
    if noise_type == "gaussian":
        std = amount * 255
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy = np.clip(noisy + noise, 0, 255)
    elif noise_type == "salt_pepper":
        mask = np.random.random(image.shape[:2]) < amount
        salt = np.zeros(image.shape[:2], dtype=np.uint8)
        salt[mask] = 255
        # Salt peaks take the max, pepper take the min
        noisy = image.astype(np.float32)
        for c in range(image.shape[2] if image.ndim == 3 else 1):
            channel = noisy[:, :, c] if image.ndim == 3 else noisy
            channel[mask] = 255
            channel[mask & (np.random.random(mask.shape) < 0.5)] = 0
    else:
        raise ProcessingError(
            f"Unknown noise_type '{noise_type}' "
            "(use 'gaussian' or 'salt_pepper')."
        )
    return noisy.astype(np.uint8)


def remove_noise(image: Array, method: str = "median",
                 kernel_size: int = 5) -> Array:
    """Remove noise from an image.

    Args:
        image: Input image.
        method: 'median', 'bilateral', or 'gaussian'.
        kernel_size: Kernel size for median/gaussian methods.
    """
    method = method.lower()
    if method == "median":
        return blur_median(image, kernel_size)
    if method == "bilateral":
        return blur_bilateral(image)
    if method == "gaussian":
        return blur_gaussian(image, kernel_size)
    raise ProcessingError(
        f"Unknown method '{method}' "
        "(use 'median', 'bilateral', or 'gaussian')."
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _check_odd_kernel(size: int) -> None:
    if size % 2 == 0 or size < 1:
        raise ProcessingError(f"Kernel size must be a positive odd "
                              f"number, got {size}.")


def _normalize(image: Array) -> Array:
    """Scale a float image to the uint8 [0, 255] range."""
    min_val, max_val = float(image.min()), float(image.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - min_val) / (max_val - min_val) * 255
    return scaled.astype(np.uint8)
