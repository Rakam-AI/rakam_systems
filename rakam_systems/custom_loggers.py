import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name, log_file, suffix="", level=logging.INFO):
    """Function to setup as many loggers as you want"""

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # Add suffix to the log filename only if provided
    if suffix:
        base, ext = os.path.splitext(log_file)
        log_file = f"{base}_{suffix}{ext}"

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# Create default loggers without suffix
chat_logger = setup_logger("chat_logger", "logs/chat.log")
prompt_logger = setup_logger("prompt_logger", "logs/prompts.log")
