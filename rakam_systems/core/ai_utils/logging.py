from __future__ import annotations

"""
Lightweight logging utilities for rakam_systems.

This module is a very thin wrapper around Python's standard :mod:`logging`
library that provides:

- A single place to configure default log format and level
- A stable import path for all internal modules:

    >>> from rakam_systems.ai_utils import logging
    >>> logger = logging.getLogger(__name__)
    >>> logger.info("hello")

It intentionally mirrors the standard logging API so most existing code
can simply replace ``import logging`` with::

    from rakam_systems.ai_utils import logging

and continue to work as before.
"""

import logging as _logging
import os
from typing import Any, Optional

# Re-export common types and level constants so callers can use
# logging.INFO, logging.WARNING, etc.
Logger = _logging.Logger

DEBUG = _logging.DEBUG
INFO = _logging.INFO
WARNING = _logging.WARNING
ERROR = _logging.ERROR
CRITICAL = _logging.CRITICAL


_DEFAULT_LEVEL_NAME = os.getenv("RAKAM_LOG_LEVEL", "INFO").upper()
_DEFAULT_LEVEL = getattr(_logging, _DEFAULT_LEVEL_NAME, _logging.INFO)
_DEFAULT_FORMAT = os.getenv(
    "RAKAM_LOG_FORMAT",
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_DEFAULT_DATEFMT = os.getenv("RAKAM_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")


def basicConfig(
    *,
    level: int | str = _DEFAULT_LEVEL,
    format: str = _DEFAULT_FORMAT,
    datefmt: Optional[str] = _DEFAULT_DATEFMT,
    **kwargs: Any,
) -> None:
    """
    Configure the root logger.

    This is a thin wrapper over :func:`logging.basicConfig` that also
    accepts string log levels (\"INFO\", \"DEBUG\", ...).
    """
    if isinstance(level, str):
        level = getattr(_logging, level.upper(), _DEFAULT_LEVEL)

    _logging.basicConfig(level=level, format=format, datefmt=datefmt, **kwargs)


def _ensure_basic_config() -> None:
    """
    Ensure there is at least a minimal logging configuration.

    If user code has already configured logging (root logger has handlers),
    this function is a no-op.
    """
    root = _logging.getLogger()
    if root.handlers:
        return
    basicConfig()


def get_logger(name: str = "ai") -> Logger:
    """
    Return a configured logger instance.

    This is the preferred entry point for application code. It mirrors the
    standard :func:`logging.getLogger` but ensures a basic configuration
    exists first.
    """
    _ensure_basic_config()
    return _logging.getLogger(name)


def getLogger(name: str = "ai") -> Logger:
    """
    Alias for :func:`get_logger` to match the stdlib API.

    Allows existing code that uses ``logging.getLogger`` to keep working
    when importing this module as ``logging``.
    """
    return get_logger(name)


def setLevel(level: int | str) -> None:
    """
    Convenience helper to set the global root log level.

    Example:
        >>> from rakam_systems.ai_utils import logging
        >>> logging.setLevel("DEBUG")
    """
    if isinstance(level, str):
        level = getattr(_logging, level.upper(), _DEFAULT_LEVEL)
    _logging.getLogger().setLevel(level)


__all__ = [
    "Logger",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "basicConfig",
    "get_logger",
    "getLogger",
    "setLevel",
]

