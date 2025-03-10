import logging
import json
import requests
from pathlib import Path
import time

# Import configuration
from config import OLLAMA_BASE_URL, MODEL_NAME

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, embedder, vector_db, ollama_base_url=OLLAMA_BASE_URL, model_name=MODEL_NAME):
        """Initialize the RAG Engine.
        
        Args:
            embedder: The embedder instance for encoding queries
            vector_db: The vector database instance for retrieval
            ollama_base_url: Base URL for Ollama API
            model_name: Name of the LLM model in Ollama
        """
        self.embedder = embedder
        self.vector_db = vector_db
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model_name = model_name
        
        # Cache for query results to improve speed for repeated queries
        self.query_cache = {}
        
        # Test connection to Ollama
        try:
            response = requests.get(f"{self.ollama_base_url}/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {model_names}")
                else:
                    logger.info(f"Successfully connected to Ollama. Using model {self.model_name}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama at {self.ollama_base_url}: {e}")
    
    def format_context(self, documents):
        """Format retrieved documents into context for the LLM."""
        context = ""
        
        # Sort documents by relevance score
        sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)
        
        for i, (doc, score) in enumerate(sorted_docs):
            # Format email metadata
            sender = doc.get("sender", "Unknown")
            date = doc.get("date", "Unknown date")
            subject = doc.get("subject", "No subject")
            text = doc.get("text", "")
            
            # Add separator between documents
            if i > 0:
                context += "\n\n" + "-" * 40 + "\n"
            
            # Add formatted email chunk with relevance score
            context += f"EMAIL {i+1} [Relevance: {score:.2f}]:\n"
            context += f"From: {sender}\n"
            context += f"Date: {date}\n"
            context += f"Subject: {subject}\n"
            context += f"Content:\n{text}\n"
        
        return context
    
    def generate_prompt(self, query, documents):
        """Generate a prompt for the LLM including retrieved context."""
        context = self.format_context(documents)
        
        # Create a more efficient prompt that focuses on relevant information
        prompt = f"""
You are an assistant analyzing the Enron email dataset. Answer the following question based ONLY on the email excerpts provided below.
If the answer cannot be determined from the provided emails, please say so clearly.
Focus on the most relevant emails (those with higher relevance scores).

USER QUESTION: {query}

EMAIL EXCERPTS:
{context}

ANSWER:
"""
        return prompt.strip()
    
    def query_ollama(self, prompt):
        """Query the Ollama API with the given prompt."""
        try:
            url = f"{self.ollama_base_url}/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1024,
                }
            }
            
            start_time = time.time()
            response = requests.post(url, json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                logger.info(f"Ollama response time: {end_time - start_time:.2f} seconds")
                return response.json().get("response", "")
            else:
                logger.error(f"Error from Ollama API: {response.status_code} - {response.text}")
                return f"Error: Failed to get response from LLM API (Status {response.status_code})"
        
        except Exception as e:
            logger.error(f"Exception when calling Ollama API: {e}")
            return f"Error: {str(e)}"
    
    def retrieve_and_generate(self, query, top_k=5):
        """Retrieve relevant documents and generate an answer."""
        try:
            # Check cache first for faster responses
            cache_key = f"{query}_{top_k}"
            if cache_key in self.query_cache:
                logger.info(f"Using cached result for query: {query}")
                return self.query_cache[cache_key]
            
            start_time = time.time()
            
            # Step 1: Embed the query
            logger.info(f"Embedding query: {query}")
            query_embedding = self.embedder.embed_query(query)
            
            # Step 2: Retrieve relevant documents
            logger.info(f"Retrieving top {top_k} documents")
            retrieved_docs = self.vector_db.search(query_embedding, top_k=top_k)
            
            # Log retrieval time
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieval time: {retrieval_time:.2f} seconds")
            
            # Step 3: Generate prompt with context
            prompt = self.generate_prompt(query, retrieved_docs)
            
            # Step 4: Generate answer with LLM
            logger.info(f"Generating answer with {self.model_name}")
            answer = self.query_ollama(prompt)
            
            # Step 5: Return answer and sources
            sources = []
            for doc, score in retrieved_docs:
                source = {
                    "id": doc.get("id", ""),
                    "email_id": doc.get("email_id", ""),
                    "sender": doc.get("sender", ""),
                    "subject": doc.get("subject", ""),
                    "date": doc.get("date", ""),
                    "text": doc.get("text", ""),
                    "score": float(score)
                }
                sources.append(source)
            
            # Calculate total time
            total_time = time.time() - start_time
            logger.info(f"Total RAG process time: {total_time:.2f} seconds")
            
            result = {
                "answer": answer.strip(),
                "sources": sources,
                "retrieval_time": retrieval_time,
                "total_time": total_time
            }
            
            # Cache the result for future use
            self.query_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": []
            } 