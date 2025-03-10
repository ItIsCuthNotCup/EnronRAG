import re
import json
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

def clean_email_body(body):
    """Clean the email body text."""
    if not body:
        return ""
    
    # Remove email forwarding markers
    body = re.sub(r'(?i)-+\s*forwarded\s+by.*?-+', ' ', body)
    
    # Remove email signatures
    body = re.sub(r'(?i)^[-_*]{2,}[\r\n].*', '', body, flags=re.MULTILINE|re.DOTALL)
    
    # Remove multiple newlines
    body = re.sub(r'\n{3,}', '\n\n', body)
    
    # Remove excessive whitespace
    body = re.sub(r'\s{2,}', ' ', body)
    
    # Strip leading/trailing whitespace
    body = body.strip()
    
    return body

def clean_header_field(field):
    """Clean header fields like subject, sender, etc."""
    if not field:
        return ""
    
    # Remove any newlines
    field = re.sub(r'[\r\n]+', ' ', field)
    
    # Remove excess whitespace
    field = re.sub(r'\s{2,}', ' ', field)
    
    return field.strip()

def preprocess_email(email_obj):
    """Preprocess a single email object."""
    if not email_obj:
        return None
    
    # Clean header fields
    email_obj['subject'] = clean_header_field(email_obj.get('subject', ''))
    email_obj['sender'] = clean_header_field(email_obj.get('sender', ''))
    email_obj['recipients'] = clean_header_field(email_obj.get('recipients', ''))
    
    # Clean body
    email_obj['body'] = clean_email_body(email_obj.get('body', ''))
    
    # Remove duplicate headers sometimes found in the body
    if email_obj['body'] and email_obj['subject']:
        email_obj['body'] = re.sub(
            rf"(?i)subject:\s*{re.escape(email_obj['subject'])}\s*\n", 
            "", 
            email_obj['body']
        )
    
    return email_obj

def preprocess_emails(input_dir, output_dir):
    """Preprocess all ingested emails."""
    logger.info(f"Starting email preprocessing from {input_dir}")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get all batch files
    batch_files = list(input_dir.glob('emails_batch_*.json'))
    
    processed_count = 0
    error_count = 0
    
    # Process each batch file
    for batch_file in tqdm(batch_files, desc="Preprocessing batches"):
        try:
            with open(batch_file, 'r') as f:
                emails = json.load(f)
            
            processed_emails = []
            for email_obj in emails:
                try:
                    processed_email = preprocess_email(email_obj)
                    if processed_email and processed_email['body']:  # Only keep emails with content
                        processed_emails.append(processed_email)
                        processed_count += 1
                except Exception as e:
                    logger.error(f"Error preprocessing email: {e}")
                    error_count += 1
            
            # Save processed batch
            output_file = output_dir / f"processed_{batch_file.name}"
            with open(output_file, 'w') as f:
                json.dump(processed_emails, f)
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_file}: {e}")
    
    logger.info(f"Preprocessing complete. Processed: {processed_count}, Errors: {error_count}")
    return processed_count 