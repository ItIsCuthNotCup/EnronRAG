import os
import json
import email
import uuid
import logging
from pathlib import Path
from email.utils import parsedate_to_datetime
from tqdm import tqdm

logger = logging.getLogger(__name__)

def parse_email_file(file_path):
    """Parse a single email file and return structured data."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Parse the email
        msg = email.message_from_string(content)
        
        # Extract basic metadata
        email_id = str(uuid.uuid4())
        sender = msg.get('From', '')
        recipients = msg.get('To', '')
        cc = msg.get('Cc', '')
        bcc = msg.get('Bcc', '')
        subject = msg.get('Subject', '')
        date_str = msg.get('Date', '')
        
        # Parse date
        try:
            if date_str:
                date = parsedate_to_datetime(date_str).isoformat()
            else:
                date = None
        except Exception:
            date = None
        
        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        part_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        body += part_body + "\n"
                    except Exception as e:
                        logger.warning(f"Error decoding email part: {e}")
        else:
            try:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            except Exception as e:
                logger.warning(f"Error decoding email body: {e}")
                body = msg.get_payload(decode=False)
        
        # Create structured email object
        email_obj = {
            "id": email_id,
            "sender": sender,
            "recipients": recipients,
            "cc": cc,
            "bcc": bcc,
            "subject": subject,
            "date": date,
            "body": body,
            "path": str(file_path)
        }
        
        return email_obj
    
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None

def ingest_emails(data_dir, output_dir):
    """Ingest all emails from the Enron dataset."""
    logger.info(f"Starting email ingestion from {data_dir}")
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    emails = []
    processed_count = 0
    error_count = 0
    
    # Walk through the directory structure
    email_files = list(data_dir.glob('**/*'))
    email_files = [f for f in email_files if f.is_file() and not f.name.startswith('.')]
    
    for file_path in tqdm(email_files, desc="Processing emails"):
        try:
            email_obj = parse_email_file(file_path)
            if email_obj:
                emails.append(email_obj)
                processed_count += 1
                
                # Save in batches to avoid memory issues
                if len(emails) >= 1000:
                    batch_file = output_dir / f"emails_batch_{processed_count//1000}.json"
                    with open(batch_file, 'w') as f:
                        json.dump(emails, f)
                    emails = []
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            error_count += 1
    
    # Save any remaining emails
    if emails:
        batch_file = output_dir / f"emails_batch_final.json"
        with open(batch_file, 'w') as f:
            json.dump(emails, f)
    
    logger.info(f"Ingestion complete. Processed: {processed_count}, Errors: {error_count}")
    return processed_count 