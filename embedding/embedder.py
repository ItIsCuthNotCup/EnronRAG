import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmailEmbedder:
    def __init__(self, model_name, device="cuda", batch_size=32, instruction=None):
        """Initialize the email embedder.
        
        Args:
            model_name (str): Name or path of the sentence transformer model
            device (str): Device to use (cuda or cpu)
            batch_size (int): Batch size for embedding generation
            instruction (str): Optional instruction to prepend to each text
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.batch_size = batch_size
        self.instruction = instruction
        
        logger.info(f"Loading embedding model {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def prepare_text(self, chunk):
        """Prepare text for embedding with optional instruction."""
        if isinstance(chunk, dict):
            # Extract text from chunk dict
            text = chunk.get('text', '')
            
            # Optionally add metadata for context
            subject = chunk.get('subject', '')
            if subject:
                text = f"Subject: {subject}\n{text}"
        else:
            # Handle case where chunk is already a string
            text = chunk
        
        # Prepend instruction if provided
        if self.instruction:
            text = f"{self.instruction} {text}"
            
        return text
    
    def embed_chunks(self, chunks):
        """Generate embeddings for a list of text chunks."""
        # Prepare texts with instructions if needed
        texts = [self.prepare_text(chunk) for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+self.batch_size]
            with torch.no_grad():
                embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
            all_embeddings.append(embeddings)
        
        # Combine batches
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
            return embeddings
        return np.array([])
    
    def embed_query(self, query):
        """Generate embedding for a single query string."""
        if self.instruction:
            query = f"{self.instruction} {query}"
        
        with torch.no_grad():
            embedding = self.model.encode(query, convert_to_numpy=True)
        
        return embedding 