import os
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def detect_enron_format(data_dir):
    """Detect the format of the Enron dataset.
    
    Args:
        data_dir (str): Directory containing the Enron data
        
    Returns:
        tuple: (format_type, file_path or directory_path)
        format_type can be 'csv', 'raw_emails', or 'unknown'
    """
    data_path = Path(data_dir)
    
    # Check for CSV file (from Kaggle dataset)
    csv_files = list(data_path.glob("*.csv"))
    for csv_file in csv_files:
        try:
            # Try to open and verify it's the Enron dataset
            df = pd.read_csv(csv_file, nrows=5)
            if all(col in df.columns for col in ['sender', 'recipients', 'subject', 'body']):
                logger.info(f"Detected Enron CSV format: {csv_file}")
                return ('csv', str(csv_file))
        except Exception as e:
            logger.warning(f"Error checking CSV file {csv_file}: {e}")
    
    # Check for raw email directories
    # Typical Enron raw emails have a directory structure with user mailboxes
    maildir = data_path / "maildir"
    if maildir.exists() and maildir.is_dir():
        # Check if it has expected subdirectories
        subdirs = [d for d in maildir.iterdir() if d.is_dir()]
        if subdirs:
            logger.info(f"Detected Enron raw email format: {maildir}")
            return ('raw_emails', str(maildir))
    
    # Check if the data_dir itself might be the maildir
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if any(d.name in ['sent', 'inbox', 'deleted_items'] for d in subdirs):
        logger.info(f"Detected possible Enron maildir structure: {data_path}")
        return ('raw_emails', str(data_path))
    
    # Search for raw email files recursively
    email_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(500)  # Read just the beginning
                    if 'Message-ID:' in content or 'From:' in content:
                        email_files.append(file_path)
                        if len(email_files) >= 10:  # Found enough to confirm
                            break
            except Exception:
                pass
        if len(email_files) >= 10:
            break
    
    if email_files:
        logger.info(f"Detected raw email files: found {len(email_files)} sample emails")
        return ('raw_emails', str(data_path))
    
    logger.warning(f"Unable to determine Enron dataset format in {data_path}")
    return ('unknown', str(data_path)) 