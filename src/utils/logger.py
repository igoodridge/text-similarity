import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_file: str ="logs/app.log", log_level=logging.INFO):
    """
    Set up logging to file and console with rotating file handler.

    Args:
    - name: Logger name
    - log_file (str): Path to the log file.
    - log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    
    Returns:
    - logger: Configured logger instance.
    """
    # Ensure the logs directory exists
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        # Create a rotating file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        )  # 5 MB max size, keep 3 backups
        file_handler.setLevel(log_level)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)  # Console gets detailed logs

        # Define log format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
