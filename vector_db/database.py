import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, db_path, embedding_dim=768):
        """Initialize the vector database using ChromaDB.
        
        Args:
            db_path (str): Path to store the database
            embedding_dim (int): Dimension of embeddings
        """
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.collection_name = "enron_emails"
        
        # Create database directory
        self.db_path.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Initializing ChromaDB at {db_path}")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"Found existing collection with {self.collection.count()} documents")
        except:
            logger.info(f"Creating new collection: {self.collection_name}")
            self.collection = self.chroma_client.create_collection(self.collection_name)
    
    def add_documents(self, documents, embeddings):
        """Add documents and their embeddings to the vector database.
        
        Args:
            documents (list): List of document dictionaries
            embeddings (np.ndarray): Numpy array of document embeddings
        """
        if len(documents) != embeddings.shape[0]:
            raise ValueError(f"Number of documents ({len(documents)}) must match number of embeddings ({embeddings.shape[0]})")
        
        # Generate IDs and metadata
        ids = [doc.get('id') or f"chunk_{i}" for i, doc in enumerate(documents)]
        
        # Prepare metadata for ChromaDB
        metadatas = []
        for doc in documents:
            # Create a copy of document with 'text' field removed for metadata
            metadata = {k: v for k, v in doc.items() if k != 'text' and not isinstance(v, (dict, list))}
            metadatas.append(metadata)
        
        # Add in batches to avoid memory issues
        batch_size = 1000
        for i in tqdm(range(0, len(documents), batch_size), desc="Adding to vector DB"):
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size].tolist()
            batch_texts = [doc.get('text', '') for doc in documents[i:i+batch_size]]
            batch_metadatas = metadatas[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def search(self, query_embedding, top_k=5):
        """Search for similar documents using a query embedding.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            top_k (int): Number of results to return
            
        Returns:
            list: List of (document, score) tuples
        """
        # Ensure query embedding is a list
        query_embedding_list = query_embedding.tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            doc = {
                "text": results["documents"][0][i],
                **results["metadatas"][0][i]
            }
            score = 1.0 - results["distances"][0][i]  # Convert distance to similarity score
            formatted_results.append((doc, score))
        
        return formatted_results 