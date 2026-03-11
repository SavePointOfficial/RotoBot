"""
Rotobot Logging — Centralized file + console debug logging.

Every module calls ``get_logger("NAME")`` once at import time.
Logs are written to ``logs/rotobot_YYYY-MM-DD.log`` in the Rotobot
directory (DEBUG level) and echoed to the console (INFO level).

The log directory and daily-rotating file are created automatically.
"""

import os
import logging
from logging.handlers import TimedRotatingFileHandler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROTOBOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROTOBOT_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "rotobot.log")

# ---------------------------------------------------------------------------
# Shared formatting
# ---------------------------------------------------------------------------
_FILE_FMT = logging.Formatter(
    "[%(asctime)s] [%(name)-8s] [%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_CONSOLE_FMT = logging.Formatter(
    "[%(name)s] %(message)s",
)

# ---------------------------------------------------------------------------
# One-time setup (runs once even if imported from multiple modules)
# ---------------------------------------------------------------------------
_root_configured = False


def _ensure_root_handlers():
    """Attach file + console handlers to the 'rotobot' root logger."""
    global _root_configured
    if _root_configured:
        return
    _root_configured = True

    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger("rotobot")
    root.setLevel(logging.DEBUG)

    # --- Daily rotating file handler (DEBUG) ---
    fh = TimedRotatingFileHandler(
        LOG_FILE,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FILE_FMT)
    root.addHandler(fh)

    # --- Console handler (INFO) ---
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_CONSOLE_FMT)
    root.addHandler(ch)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the ``rotobot`` namespace.

    Usage::

        from rotobot_logging import get_logger
        log = get_logger("ENGINE")
        log.debug("Loading SAM2 model...")
        log.info("Model loaded in %.1fs", elapsed)
    """
    _ensure_root_handlers()
    return logging.getLogger("rotobot.%s" % name)
