"""
    Configure logger class for logging
"""

import logging.config
from logging.handlers import RotatingFileHandler

import logging
import logging.config


def setup_logger(
        logger_name: str,
        logfile_name: str = "debug_log.log",
        console_log_level: int = logging.ERROR,
        file_log_level: int = logging.DEBUG,
        total_log_level: int = logging.DEBUG,
        only_log_to_file: bool = False,
) -> logging.Logger:
    """
    Sets up and returns a logger with the specified name and configuration.
    :param logger_name: Name of the logger to configure.
    :param logfile_name: Name of the log file.
    :param console_log_level: Log level for console output.
    :param file_log_level: Log level for file output.
    :param total_log_level: Overall log level for the logger.
    :param only_log_to_file: Only log to file.
    :return: Configured logger.
    """
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(threadName)20.20s (%(levelname)5.10s) [%(filename)5.24s] %(funcName)s %(lineno)d: %(message)s"
            },
        },
        "handlers": {
            "console": {
                "level": console_log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
            "file": {
                "level": file_log_level,
                "class": "logging.handlers.RotatingFileHandler",
                "filename": logfile_name,
                "formatter": "standard",
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 5,
            },
        },
        "loggers": {
            logger_name: {
                "handlers": (
                    ["console", "file"] if not only_log_to_file else ["file"]
                ),
                "level": total_log_level,
                "propagate": False,
            }
        },
    }

    logging.config.dictConfig(logging_config)
    return logging.getLogger(logger_name)


def close_logger(logger_name: str):
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
