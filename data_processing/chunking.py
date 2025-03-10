import json
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

def chunk_text(text, chunk_size, chunk_overlap):
    """Split text into chunks with overlap."""
    if not text:
        return []
    
    # Split the text into sentences for more natural chunks
    sentences = text.split('. ')
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        # Approximate token count by word count (rough estimate)
        sentence_size = len(sentence.split())
        
        if current_size + sentence_size > chunk_size and current_chunk:
            # Add period back to the end of sentences except for the last one
            chunk_text = '. '.join(current_chunk[:-1]) + '. ' + current_chunk[-1]
            chunks.append(chunk_text)
            
            # Keep overlap for next chunk
            overlap_count = min(len(current_chunk), max(1, int(chunk_overlap / 10)))
            current_chunk = current_chunk[-overlap_count:]
            current_size = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Add the last chunk if not empty
    if current_chunk:
        chunk_text = '. '.join(current_chunk[:-1]) + '. ' + current_chunk[-1] if len(current_chunk) > 1 else current_chunk[0]
        chunks.append(chunk_text)
    
    return chunks

def create_email_chunks(email_obj, chunk_size, chunk_overlap):
    """Create chunks from an email with metadata."""
    if not email_obj or not email_obj.get('body'):
        return []
    
    # Get email data
    email_id = email_obj['id']
    sender = email_obj['sender']
    subject = email_obj['subject']
    body = email_obj['body']
    date = email_obj.get('date', '')
    
    # Create chunks from the body
    body_chunks = chunk_text(body, chunk_size, chunk_overlap)
    
    # Create chunk objects with metadata
    email_chunks = []
    for i, chunk_text in enumerate(body_chunks):
        chunk = {
            "id": f"{email_id}_chunk_{i}",
            "email_id": email_id,
            "sender": sender,
            "subject": subject,
            "date": date,
            "chunk_index": i,
            "text": chunk_text,
            "total_chunks": len(body_chunks)
        }
        email_chunks.append(chunk)
    
    return email_chunks

def chunk_emails(input_dir, output_dir, chunk_size, chunk_overlap):
    """Chunk all preprocessed emails."""
    logger.info(f"Starting email chunking from {input_dir}")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get all processed batch files
    batch_files = list(input_dir.glob('processed_emails_batch_*.json'))
    
    total_emails = 0
    total_chunks = 0
    
    # Process each batch file
    for batch_file in tqdm(batch_files, desc="Chunking email batches"):
        try:
            with open(batch_file, 'r') as f:
                emails = json.load(f)
            
            all_chunks = []
            for email_obj in emails:
                try:
                    email_chunks = create_email_chunks(email_obj, chunk_size, chunk_overlap)
                    if email_chunks:
                        all_chunks.extend(email_chunks)
                        total_emails += 1
                        total_chunks += len(email_chunks)
                except Exception as e:
                    logger.error(f"Error chunking email {email_obj.get('id')}: {e}")
            
            # Save chunks batch
            if all_chunks:
                output_file = output_dir / f"chunks_{batch_file.name}"
                with open(output_file, 'w') as f:
                    json.dump(all_chunks, f)
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_file}: {e}")
    
    logger.info(f"Chunking complete. Processed {total_emails} emails into {total_chunks} chunks")
    return total_chunks 