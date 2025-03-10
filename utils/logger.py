import logging
import sys
import os
from pathlib import Path

def setup_logger(name, level=logging.INFO):
    """Set up and return a logger with the given name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create log directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{name.replace('.', '_')}.log"
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file)
    
    # Set level
    console_handler.setLevel(level)
    file_handler.setLevel(level)
    
    # Create and set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name, level=logging.INFO):
    """Get or create a logger with the given name."""
    return setup_logger(name, level)

def configure_root_logger(level=logging.INFO):
    """Configure the root logger."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('enron_rag.log')
        ]
    ) 