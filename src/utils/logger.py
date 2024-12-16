import logging
import os
from logging.handlers import RotatingFileHandler
from src.utils.config import load_config
from pathlib import Path

config = load_config()

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Set up logging to file and console with rotating file handler.

    Args:
    - name: Logger name
    - log_file: Path to the log file. If None, uses default from config.
    
    Returns:
    - logger: Configured logger instance.
    """
    # Get logging config
    log_config = config["logging"]
    
    # Determine log file path
    if log_file is None:
        # Use default log file from config based on logger name
        log_file = log_config["file"].get(name.lower(), log_config["file"]["app"])
    
    # Convert to Path and ensure it's absolute
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = Path(config["paths"]["logs"]) / log_path

    # Ensure the logs directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    
    # Set level from config
    log_level = getattr(logging, log_config["level"].upper())
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        # Create a rotating file handler
        file_handler = RotatingFileHandler(
            str(log_path),
            maxBytes=log_config.get("max_bytes", 5 * 1024 * 1024),  # 5 MB default
            backupCount=log_config.get("backup_count", 3)  # 3 backups default
        )
        file_handler.setLevel(log_level)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_config.get("console_level", "DEBUG"))

        # Create formatter from config
        formatter = logging.Formatter(
            log_config["format"],
            datefmt=log_config.get("date_format", "%Y-%m-%d %H:%M:%S")
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger