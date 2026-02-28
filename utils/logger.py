import logging
import os
from logging.handlers import RotatingFileHandler


_LOG_DIR = "logs"
_initialized = False


def setup_logging(level: int = logging.INFO):
    """Configure root logging for the application."""
    global _initialized
    if _initialized:
        return
    os.makedirs(_LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s - %(message)s"))
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        os.path.join(_LOG_DIR, "app.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s - %(message)s")
    )
    root.addHandler(file_handler)
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; ensures logging is set up."""
    setup_logging()
    return logging.getLogger(name)
