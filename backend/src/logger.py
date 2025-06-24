import logging
import logging.config

from src.config import LOGGING_CONFIG


def init_logger():
    logging.config.dictConfig(LOGGING_CONFIG)
    logging.captureWarnings(True)
