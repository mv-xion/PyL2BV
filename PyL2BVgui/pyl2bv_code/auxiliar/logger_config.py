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
    log_level: int = logging.DEBUG,
    only_log_to_console: bool = False,
) -> logging.Logger:
    """
    Sets up and returns a logger with the specified name and configuration.
    :param logger_name: Name of the logger to configure.
    :param logfile_name: Name of the log file.
    :param log_level: Overall log level for the logger.
    :param only_log_to_console: Only log to console.
    :return: Configured logger.
    """
    # Base logging configuration
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
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "loggers": {
            logger_name: {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            }
        },
    }

    # Add file handler only if not `only_log_to_console`
    if not only_log_to_console:
        logging_config["handlers"]["file"] = {
            "level": log_level,
            "class": "logging.handlers.RotatingFileHandler",
            "filename": logfile_name,
            "formatter": "standard",
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 5,
        }
        logging_config["loggers"][logger_name]["handlers"].append("file")

    # Apply the logging configuration
    logging.config.dictConfig(logging_config)
    return logging.getLogger(logger_name)


def close_logger(logger_name: str):
    logger = logging.getLogger(logger_name)

    for handler in logger.handlers[:]:  # Copy the list to avoid modification issues
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    logging.shutdown()  # Ensure all logging resources are freed
