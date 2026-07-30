import logging
import sys
from pathlib import Path


def setup_logger(
    name: str,
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger with console and file handlers.

    Args:
        name: Logger name (typically __name__ or module name).
        log_file: Path to the log file, relative or absolute.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
