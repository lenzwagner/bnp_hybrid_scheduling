import logging
import sys
from datetime import datetime

def setup_logging(log_level='INFO', log_to_file=True, log_dir='logs'):
    """
    Configure logging for Branch-and-Price solver.

    Args:
        log_level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        log_to_file: If True, also write logs to file
        log_dir: Directory for log files
    """
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        import os
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'bnp_{timestamp}.log')

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        print(f"Logging to file: {log_file}")

    return root_logger

def get_logger(name):
    """Get a logger for a specific module."""
    return logging.getLogger(name)