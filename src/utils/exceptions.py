class CameraError(Exception):
    """Base exception for all camera-related errors."""
    pass


class CameraNotFoundError(CameraError):
    """Raised when the requested camera is not found or cannot be opened."""
    pass


class CameraAccessError(CameraError):
    """Raised when the camera is accessed without being properly initialized."""
    pass


class InvalidResolutionError(CameraError):
    """Raised when an invalid resolution (width/height zero or negative) is provided."""
    pass


class FrameGrabError(CameraError):
    """Raised when reading a frame from the camera fails."""
    pass


class RecordingError(CameraError):
    """Raised when video recording initialization or writing fails."""
    pass


class ImageError(Exception):
    """Base exception for image processing errors."""
    pass


class ProcessingError(Exception):
    """Base exception for image processing pipeline errors."""
    pass
