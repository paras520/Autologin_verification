from __future__ import annotations

import logging

from src.config import config


def configure_logging() -> logging.Logger:
    log_level = config.log_level
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("autologin.log", encoding="utf-8"),
            ],
        )
    else:
        root_logger.setLevel(log_level)

    return logging.getLogger("autologin.app")
