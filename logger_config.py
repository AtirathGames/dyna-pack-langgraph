import logging
import sys
import os

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger with both file and console handlers.
    
    Args:
        name: The name of the logger (usually __name__).
        
    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # dedicated log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check if handlers already exist to avoid duplicates
    if not logger.handlers:
        # File Handler - writes to trip_planner.log
        try:
            file_handler = logging.FileHandler('trip_planner.log', mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback if file writing fails
            sys.stderr.write(f"Failed to setup file handler: {e}\n")

        # Stream Handler - writes to console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)  # Keep console cleaner
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
