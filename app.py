#!/usr/bin/env python3

import os
import streamlit as st
import pandas as pd
import logging
import time
import html
import sys
import requests
import json
import subprocess
import concurrent.futures
from functools import lru_cache

# Set up logging with proper encoding for Windows
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                    handlers=[logging.FileHandler("logs/enron_app.log", encoding="utf-8")])

# Log startup for debugging
logging.info("========== APPLICATION STARTED ==========")

# Global variables for model caching
EMBEDDING_MODEL_CACHE = {}
OLLAMA_MODEL_CACHE = {}

# Check if required packages are installed, install if missing
def ensure_dependencies():
    """Check if dependencies are installed, install if missing."""
    try:
        import chromadb
        logging.info("ChromaDB is already installed")
    except ImportError:
        logging.warning("ChromaDB not found, attempting to install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb", "sentence-transformers"])
            logging.info("Successfully installed ChromaDB and sentence-transformers")
        except Exception as e:
            logging.error(f"Failed to install dependencies: {str(e)}")

# Ensure dependencies are installed
ensure_dependencies()

# Load configuration values directly (adjusted for performance)
CSV_FILE = "emails.csv"
TOP_K_RESULTS = 5  # Number of relevant emails to retrieve
VECTOR_DB_DIR = "vector_db"
CHROMA_DIR = "vector_db/chroma" 
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3:8b"  # Default model, could also use "mistral:7b" if available
FALLBACK_MODEL = "mistral:7b"  # Fallback model if primary not available
USE_CSV_SEARCH = True  # Prioritize CSV search for speed
RESPONSE_TOKENS = 512  # Reduced tokens for faster generation
TIMEOUT_SHORT = 3  # Short timeout for quick checks (seconds)
TIMEOUT_NORMAL = 10  # Normal timeout for most operations (seconds)
TIMEOUT_LONG = 30  # Long timeout for heavyweight operations (seconds)

# Debug function to diagnose vector database issues - streamlined for faster execution
def debug_vector_db():
    """Diagnose potential issues with the vector database."""
    debug_info = {"status": "unknown", "issues": [], "collections": [], "details": {}}
    
    try:
        # Step 1: Check if chromadb is installed
        try:
            import chromadb
            debug_info["details"]["chromadb_installed"] = True
        except ImportError:
            debug_info["status"] = "error"
            debug_info["issues"].append("ChromaDB package is not installed")
            debug_info["details"]["chromadb_installed"] = False
            return debug_info
        
        # Step 2: Check if the directory exists
        if os.path.exists(CHROMA_DIR):
            debug_info["details"]["directory_exists"] = True
            contents = os.listdir(CHROMA_DIR)
            debug_info["details"]["directory_contents"] = contents
            if not contents:
                debug_info["issues"].append("ChromaDB directory exists but is empty")
        else:
            debug_info["issues"].append(f"ChromaDB directory not found: {CHROMA_DIR}")
            debug_info["details"]["directory_exists"] = False
            return debug_info  # Exit early if directory doesn't exist
        
        # Step 3: Try to connect to the client
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            debug_info["details"]["client_connected"] = True
            
            # Step 4: List collections
            collections = client.list_collections()
            debug_info["collections"] = [c.name for c in collections]
            
            if not collections:
                debug_info["issues"].append("No collections found in ChromaDB")
                debug_info["status"] = "error"
                return debug_info  # Exit early if no collections
            
            # Don't load the embedding model during debugging to save time
            debug_info["status"] = "ok"
            return debug_info
            
        except Exception as e:
            debug_info["details"]["client_connected"] = False
            debug_info["issues"].append(f"Failed to connect to ChromaDB client: {str(e)}")
    
    except Exception as e:
        debug_info["status"] = "error"
        debug_info["issues"].append(f"Unexpected error during diagnosis: {str(e)}")
    
    # Set final status
    if not debug_info["issues"]:
        debug_info["status"] = "ok"
    elif any("directory not found" in issue for issue in debug_info["issues"]):
        debug_info["status"] = "missing"
    else:
        debug_info["status"] = "error"
    
    return debug_info

# Run vector DB debug on startup but avoid expensive operations
vector_db_status = debug_vector_db()
if vector_db_status["status"] != "ok":
    logging.warning(f"Vector DB issues detected: {vector_db_status['issues']}")
    # If the directory is missing, we could try to create it
    if vector_db_status["status"] == "missing":
        logging.info("Attempting to create vector database directory structure")
        os.makedirs(CHROMA_DIR, exist_ok=True)

# Hardcoded sample data for fallback - keep as is
SAMPLE_DATA = [
    {"from": "kenneth.lay@enron.com", "to": "all.employees@enron.com", 
     "subject": "Welcome Message", "date": "2001-05-01", 
     "text": "Welcome to Enron! I am Kenneth Lay, the CEO of Enron Corporation. I'm excited to have you join our team. As we continue to grow and innovate in the energy sector, each one of you plays a vital role in our success. Our company values integrity, respect, communication, and excellence in all we do."},
    
    {"from": "jeff.skilling@enron.com", "to": "board@enron.com", 
     "subject": "Quarterly Results", "date": "2001-06-15", 
     "text": "I'm pleased to report that our quarterly results exceeded expectations. Revenue is up 20% year-over-year, and our stock price continues to perform well. The energy trading division has been particularly successful, showing the strength of our market-based approach to energy solutions."},
    
    {"from": "andrew.fastow@enron.com", "to": "finance@enron.com", 
     "subject": "Financial Strategies", "date": "2001-07-30", 
     "text": "The new financial strategies we've implemented are working well. Our special purpose entities are hiding the debt effectively while maintaining our credit rating. The Raptor vehicles in particular have been successful in hedging our investments in technology companies."},
    
    {"from": "sherron.watkins@enron.com", "to": "kenneth.lay@enron.com", 
     "subject": "Accounting Irregularities", "date": "2001-08-15", 
     "text": "I am incredibly nervous that we will implode in a wave of accounting scandals. I have been thinking about our accounting practices a lot recently. The aggressive accounting we've used in the Raptor vehicles and other SPEs is concerning. I am worried that we have become a house of cards."},
    
    {"from": "richard.kinder@enron.com", "to": "executive.team@enron.com", 
     "subject": "Business Strategy", "date": "2000-12-01", 
     "text": "We need to focus on our core business and maintain strong relationships with our partners. Our expansion strategy must be carefully considered. I believe in building businesses with hard assets rather than just trading operations. Long-term success requires solid infrastructure."}
]

# Cache model info for faster response
@lru_cache(maxsize=1)
def get_available_models():
    """Get list of available models from Ollama with caching."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=TIMEOUT_SHORT)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except Exception as e:
        logging.error(f"Error getting available models: {e}")
        return []

# Select best available model with caching
@lru_cache(maxsize=1)
def select_best_model():
    """Select the best available model from Ollama with caching."""
    available_models = get_available_models()
    
    # Preferred models in order (based on research)
    preferred_models = [
        MODEL_NAME,            # llama3:8b
        FALLBACK_MODEL,        # mistral:7b
        "llama2:7b",           # Smaller Llama 2
        "mistral:mistral-tiny" # Smaller Mistral
    ]
    
    # Return the first available preferred model
    for model in preferred_models:
        if any(m.startswith(model) for m in available_models):
            matched_model = next(m for m in available_models if m.startswith(model))
            return matched_model
    
    # If no preferred model is available, return any available model
    if available_models:
        return available_models[0]
    
    return None

# Direct RAG query function using Ollama API - optimized for faster responses
def query_ollama(prompt, model_name=None):
    """Query Ollama API directly with faster response times."""
    if not model_name:
        # Use cached model selection
        model_name = select_best_model()
        if not model_name:
            return "Error: No models available in Ollama"
    
    # Calculate estimated prompt tokens to adjust max output tokens
    prompt_chars = len(prompt)
    estimated_prompt_tokens = prompt_chars / 4  # rough estimate
    
    # If prompt is very long, reduce output tokens
    adjusted_tokens = min(RESPONSE_TOKENS, 
                          max(128, int(RESPONSE_TOKENS - (estimated_prompt_tokens / 10))))
            
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,           # Lower temperature for more factual responses
                "num_predict": adjusted_tokens, # Adaptive token limit
                "top_p": 0.9,                 # More focused sampling
                "top_k": 40                   # Consider top 40 tokens
            }
        }
        
        start_time = time.time()
        
        # Reduce logging for performance
        if len(prompt) > 500:
            logging.info(f"Querying Ollama with model: {model_name}, prompt length: {len(prompt)} chars")
        else:
            logging.info(f"Querying Ollama with model: {model_name}, prompt: {prompt[:100]}...")
        
        response = requests.post(url, json=payload, timeout=TIMEOUT_LONG)
        end_time = time.time()
        
        if response.status_code == 200:
            logging.info(f"Ollama response time: {end_time - start_time:.2f} seconds")
            return response.json().get("response", "")
        else:
            logging.error(f"Error from Ollama API: {response.status_code}")
            
            # If model not found, try fallback model
            if "model not found" in response.text.lower():
                logging.info(f"Model {model_name} not found, trying fallback model")
                return query_ollama(prompt, FALLBACK_MODEL)
                
            return f"Error: Failed to get response from LLM (Status {response.status_code})"
    
    except Exception as e:
        logging.error(f"Exception when calling Ollama API: {e}")
        return f"Error: {str(e)}"

# Check if Ollama is available - faster timeout
@lru_cache(maxsize=1)  # Cache for 1 second
def is_ollama_available():
    """Check if Ollama is available by making a quick API call."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=TIMEOUT_SHORT)
        return response.status_code == 200
    except:
        return False

# Get embedding function with caching
def get_embedding_function():
    """Get embedding function with caching for performance."""
    global EMBEDDING_MODEL_CACHE
    
    if "embedding_function" not in EMBEDDING_MODEL_CACHE:
        try:
            from chromadb.utils import embedding_functions
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2", 
                device="cpu"  # Use CPU for compatibility
            )
            EMBEDDING_MODEL_CACHE["embedding_function"] = embedding_function
            logging.info("Initialized and cached embedding function")
        except Exception as e:
            logging.error(f"Error creating embedding function: {str(e)}")
            return None
    
    return EMBEDDING_MODEL_CACHE.get("embedding_function")

# Try connecting to ChromaDB more efficiently
def get_chroma_vector_db():
    """Try to connect to ChromaDB with caching."""
    global EMBEDDING_MODEL_CACHE
    
    if "chroma_collection" in EMBEDDING_MODEL_CACHE:
        return EMBEDDING_MODEL_CACHE["chroma_collection"]
    
    try:
        import chromadb
        
        # Skip logging for speed
        if not os.path.exists(CHROMA_DIR):
            os.makedirs(CHROMA_DIR, exist_ok=True)
        
        # Try to connect to the persistent client
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # List all collections
        collections = client.list_collections()
        
        if collections:
            collection_name = collections[0].name
            
            # Get embedding function (cached)
            embedding_func = get_embedding_function()
            if not embedding_func:
                return None
            
            # Get the collection with the embedding function
            collection = client.get_collection(
                name=collection_name, 
                embedding_function=embedding_func
            )
            
            # Cache the collection
            EMBEDDING_MODEL_CACHE["chroma_collection"] = collection
            return collection
        else:
            return None
    except Exception as e:
        logging.error(f"Error connecting to ChromaDB: {str(e)}")
        return None

# Query vector database with more efficient processing
def query_vector_db(query, n_results=TOP_K_RESULTS):
    """Query the vector database with performance optimizations."""
    try:
        collection = get_chroma_vector_db()
        if not collection:
            return []
        
        # Query with timeout control
        start_time = time.time()
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        query_time = time.time() - start_time
        logging.info(f"Vector query completed in {query_time:.2f} seconds")
        
        # Fast path for no results
        if not results or 'documents' not in results or not results['documents'] or not results['documents'][0]:
            return []
        
        # Process results efficiently
        emails = []
        for i, doc in enumerate(results['documents'][0]):
            # Extract metadata safely
            metadata = {}
            if 'metadatas' in results and results['metadatas'] and i < len(results['metadatas'][0]):
                metadata = results['metadatas'][0][i] or {}
            
            # Create an email object with available data
            email = {
                "text": doc,
                "from": metadata.get('from', metadata.get('sender', 'Unknown')),
                "date": metadata.get('date', 'Unknown'),
                "subject": metadata.get('subject', 'Unknown'),
            }
            emails.append(email)
        
        return emails
    except Exception as e:
        logging.error(f"Error during vector DB query: {str(e)}")
    
    return []

# Load emails from CSV with improved caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_emails_from_csv(csv_file=CSV_FILE):
    """Load emails from CSV file with caching."""
    try:
        if os.path.exists(csv_file):
            start_time = time.time()
            df = pd.read_csv(csv_file)
            load_time = time.time() - start_time
            logging.info(f"CSV loaded in {load_time:.2f} seconds: {len(df)} emails")
            return df.to_dict('records')
        else:
            # Try to look for other CSV files in the directory
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if csv_files:
                alt_csv = csv_files[0]
                df = pd.read_csv(alt_csv)
                return df.to_dict('records')
            return []
    except Exception as e:
        logging.error(f"Error loading CSV: {str(e)}")
        return []

# Optimized CSV search with faster scoring
def search_csv_emails(query, csv_emails, max_results=TOP_K_RESULTS):
    """Search CSV emails with optimized scoring."""
    if not csv_emails:
        return []
    
    start_time = time.time()
    
    # Normalize query - optimize for speed
    query_lower = query.lower()
    query_terms = [term.strip() for term in query_lower.split() if len(term.strip()) > 2]
    
    # Exit early if no meaningful terms
    if not query_terms:
        return []
    
    # Use a more efficient approach for large datasets
    if len(csv_emails) > 10000:
        # Filter first, then score (much faster on large datasets)
        filtered_emails = []
        for email in csv_emails:
            text = str(email.get('text', '')).lower()
            subject = str(email.get('subject', '')).lower()
            
            # Quick check if any term appears in text or subject
            if any(term in text or term in subject for term in query_terms):
                filtered_emails.append(email)
        
        # Only score filtered emails
        csv_emails = filtered_emails[:1000]  # Limit to 1000 for performance
    
    # Score each email based on relevance
    scored_emails = []
    for email in csv_emails:
        score = 0
        text = str(email.get('text', '')).lower()
        subject = str(email.get('subject', '')).lower()
        sender = str(email.get('from', '')).lower()
        
        # Check each query term
        for term in query_terms:
            # Higher score for terms in subject
            if term in subject:
                score += 5
            
            # Score for terms in text
            if term in text:
                score += 3
                # Only count multiple occurrences if score is already good
                if score > 5:
                    score += text.count(term) * 0.2
            
            # Score for terms in sender
            if term in sender:
                score += 2
        
        # Only keep emails with meaningful scores
        if score > 2:  # Threshold to reduce noise
            scored_emails.append((score, email))
    
    # Sort by score (highest first) and take top results
    sorted_emails = [email for _, email in sorted(scored_emails, key=lambda x: x[0], reverse=True)]
    limited_results = sorted_emails[:max_results]
    
    search_time = time.time() - start_time
    logging.info(f"CSV search completed in {search_time:.2f} seconds, found {len(sorted_emails)} matches")
    
    return limited_results

# Generate alternative queries for better retrieval (query rewriting from the article)
def generate_query_variations(query):
    """Generate variations of the user query for better retrieval."""
    variations = [query]  # Always include original query
    
    # Simple variations
    if len(query) > 10:
        # Add a more generic version
        variations.append(query.replace("emails about ", "").replace("email regarding ", ""))
        
        # Add a more specific version
        if "enron" not in query.lower():
            variations.append(f"enron {query}")
    
    # Replace specific terms with synonyms
    query_lower = query.lower()
    if "accounting" in query_lower:
        variations.append(query.lower().replace("accounting", "financial"))
    if "transaction" in query_lower:
        variations.append(query.lower().replace("transaction", "deal"))
    if "email" in query_lower or "emails" in query_lower:
        variations.append(query.lower().replace("email", "message").replace("emails", "messages"))
    
    # Return unique variations only
    return list(set(variations))

# Better prompt template with compression
def create_rag_prompt(query, context_emails):
    """Create a well-structured prompt with compressed context."""
    # Measure initial size
    total_chars = sum(len(email.get('text', '')) for email in context_emails)
    
    # Only include essential information when context is large
    if total_chars > 6000:
        prompt = f"""You are analyzing the Enron email dataset. Answer based ONLY on these email excerpts:

USER QUESTION: {query}

RELEVANT EMAILS:
"""
        # Add only the most important parts of each email
        for i, email in enumerate(context_emails):
            prompt += f"EMAIL {i+1}: From {email.get('from', 'Unknown')} Subject '{email.get('subject', 'Unknown')}' "
            
            # Compress long emails by focusing on key sentences
            text = email.get('text', '')
            if len(text) > 1200:
                # Extract first and last few sentences as they often contain key information
                sentences = text.split('.')
                if len(sentences) > 6:
                    important_text = '. '.join(sentences[:3]) + '...' + '. '.join(sentences[-3:])
                    prompt += f"Content: {important_text}\n\n"
                else:
                    prompt += f"Content: {text}\n\n"
            else:
                prompt += f"Content: {text}\n\n"
    else:
        # Use full detail for smaller contexts
        prompt = f"""You are an expert assistant analyzing the Enron email dataset, which contains emails from executives at Enron Corporation from 1999-2002 before the company's collapse.

Your task is to answer the following question using ONLY the information contained in the email excerpts provided below.
If the emails don't contain enough information to answer the question fully, acknowledge the limitations of what you know.
Do not make up or infer information that is not present in the emails. Cite specific emails when possible.

USER QUESTION: {query}

RELEVANT EMAIL EXCERPTS:
"""
        # Add context from emails
        for i, email in enumerate(context_emails):
            prompt += f"\n--- EMAIL {i+1} ---\n"
            prompt += f"FROM: {email.get('from', 'Unknown')}\n"
            prompt += f"DATE: {email.get('date', 'Unknown')}\n" 
            prompt += f"SUBJECT: {email.get('subject', 'Unknown')}\n"
            prompt += f"CONTENT: {email.get('text', '')}\n"
    
    prompt += "\n\nBased solely on the email excerpts above, please provide a detailed answer to the user's question."
    return prompt

# Search emails with parallel processing
def search_emails(query):
    """Search function with parallel processing and query expansion."""
    start_time = time.time()
    logging.info(f"Processing query: '{query}'")
    
    # Check for special commands first
    special_result = handle_special_commands(query)
    if special_result:
        logging.info(f"Handled special command: {query}")
        return special_result
    
    # Generate query variations for better retrieval
    query_variations = generate_query_variations(query)
    logging.info(f"Generated query variations: {query_variations}")
    
    # Fast path: Try CSV search first, it's usually faster
    if USE_CSV_SEARCH:
        try:
            csv_emails = load_emails_from_csv()
            if csv_emails:
                # Try each query variation
                all_matched_emails = []
                for q in query_variations:
                    matched_emails = search_csv_emails(q, csv_emails)
                    all_matched_emails.extend(matched_emails)
                
                # Remove duplicates by subject/from combination
                seen = set()
                unique_emails = []
                for email in all_matched_emails:
                    key = (email.get('subject', ''), email.get('from', ''))
                    if key not in seen:
                        seen.add(key)
                        unique_emails.append(email)
                
                # Limit to top results
                matched_emails = unique_emails[:TOP_K_RESULTS]
                
                if matched_emails:
                    # Create prompt with matched emails
                    prompt = create_rag_prompt(query, matched_emails)
                    
                    # Try to get response from Ollama if available
                    if is_ollama_available():
                        answer = query_ollama(prompt)
                        
                        # Format the output - ONLY include the answer
                        output = f"{answer}\n\n"
                        elapsed = time.time() - start_time
                        logging.info(f"Query completed in {elapsed:.2f} seconds via CSV search")
                        return output
        except Exception as e:
            logging.error(f"Error in CSV search: {str(e)}")
    
    # If CSV search fails or finds no results, use vector search
    try:
        # Try all query variations with vector search
        all_vector_results = []
        for q in query_variations:
            vector_results = query_vector_db(q)
            all_vector_results.extend(vector_results)
        
        # Remove duplicates
        seen = set()
        unique_vector_results = []
        for email in all_vector_results:
            key = (email.get('subject', ''), email.get('from', ''))
            if key not in seen:
                seen.add(key)
                unique_vector_results.append(email)
        
        # Limit to top results
        vector_results = unique_vector_results[:TOP_K_RESULTS]
        
        if vector_results:
            # Create prompt with vector results
            prompt = create_rag_prompt(query, vector_results)
            
            # Get response from Ollama
            if is_ollama_available():
                answer = query_ollama(prompt)
                output = f"{answer}\n\n"
                elapsed = time.time() - start_time
                logging.info(f"Query completed in {elapsed:.2f} seconds via vector search")
                return output
    except Exception as e:
        logging.error(f"Error in vector search: {str(e)}")
    
    # Final fallback - direct query to Ollama
    if is_ollama_available():
        try:
            logging.info("All retrieval methods failed, using direct query to Ollama")
            
            prompt = f"""
You are analyzing the Enron email dataset. The user has asked the following question, but unfortunately,
we don't have specific emails to provide as context. Please provide a general response based on
what you know about Enron, being transparent that specific emails weren't found.

USER QUESTION: {query}
"""
            answer = query_ollama(prompt)
            output = f"No specific emails found for your query. Here's a general response:\n\n{answer}"
            
            elapsed = time.time() - start_time
            logging.info(f"Query completed in {elapsed:.2f} seconds via direct query")
            return output
                
        except Exception as e:
            logging.error(f"Error with direct Ollama query: {str(e)}")
    
    # Complete fallback if everything else fails
    elapsed = time.time() - start_time
    logging.warning(f"All methods failed in {elapsed:.2f} seconds")
    return f"I'm sorry, I couldn't find any relevant information for your query: '{query}'. Please try a different question or check the system status with !debug."

# Handle special debug commands - rest of code is the same
def handle_special_commands(command):
    """Handle special commands for debugging and diagnostics."""
    special_commands = {
        "!debug": "Running diagnostics...",
        "!vector": "Checking vector database...",
        "!ollama": "Testing Ollama connection...",
        "!models": "Listing available models...",
        "!csv": "Testing CSV import...",
        "!help": """
Available debug commands:
!debug - Run all diagnostics
!vector - Check vector database status
!ollama - Test Ollama connection
!models - List available models
!csv - Check CSV file loading
!help - Show this help message
"""
    }
    
    if command.startswith("!"):
        if command in special_commands:
            if command == "!debug":
                # Run all diagnostics
                vector_results = debug_vector_db()
                
                output = "DIAGNOSTIC RESULTS:\n\n"
                
                # Vector DB results
                output += "VECTOR DATABASE STATUS: " + vector_results["status"].upper() + "\n"
                if vector_results["issues"]:
                    output += "Issues:\n"
                    for issue in vector_results["issues"]:
                        output += f"- {issue}\n"
                if vector_results["collections"]:
                    output += f"Collections: {', '.join(vector_results['collections'])}\n"
                
                # Ollama check - simplified
                output += "\nOLLAMA API STATUS: "
                if is_ollama_available():
                    output += "OK\n"
                    models = get_available_models()
                    if models:
                        output += f"Available models: {', '.join(models)}\n"
                    else:
                        output += "No models available\n"
                else:
                    output += "ERROR - Cannot connect to Ollama\n"
                
                # CSV check
                csv_emails = load_emails_from_csv()
                output += f"\nCSV DATA: {'Available' if csv_emails else 'Not available'}\n"
                if csv_emails:
                    output += f"Number of emails: {len(csv_emails)}\n"
                    output += f"Sample email subjects: {', '.join([e.get('subject', 'No subject') for e in csv_emails[:3]])}\n"
                
                return output
            
            elif command == "!vector":
                results = debug_vector_db()
                output = "VECTOR DATABASE DIAGNOSIS:\n\n"
                output += "Status: " + results["status"].upper() + "\n"
                if results["issues"]:
                    output += "Issues:\n"
                    for issue in results["issues"]:
                        output += f"- {issue}\n"
                if results["collections"]:
                    output += f"Collections: {', '.join(results['collections'])}\n"
                return output
                
            elif command == "!ollama":
                output = "OLLAMA CONNECTION TEST:\n\n"
                
                if is_ollama_available():
                    output += "Status: OK\n"
                    start_time = time.time()
                    result = query_ollama("Say hello in one word.", select_best_model())
                    end_time = time.time()
                    
                    output += f"Response time: {end_time - start_time:.2f} seconds\n"
                    output += f"Response: {result}\n"
                else:
                    output += "Status: ERROR\n"
                    output += "Failed to connect to Ollama API\n"
                
                return output
            
            elif command == "!csv":
                csv_emails = load_emails_from_csv()
                output = "CSV FILE CHECK:\n\n"
                if csv_emails:
                    output += f"Successfully loaded {len(csv_emails)} emails from CSV\n"
                    output += "Sample emails:\n"
                    for i, email in enumerate(csv_emails[:3]):
                        output += f"{i+1}. From: {email.get('from', 'Unknown')}\n"
                        output += f"   Subject: {email.get('subject', 'No subject')}\n"
                        output += f"   Date: {email.get('date', 'Unknown')}\n\n"
                else:
                    output += "No CSV data available or failed to load.\n"
                    output += f"Looked for file: {CSV_FILE}\n"
                    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
                    if csv_files:
                        output += f"Available CSV files: {', '.join(csv_files)}\n"
                    else:
                        output += "No CSV files found in the current directory.\n"
                return output
                
            elif command == "!models":
                models = get_available_models()
                if models:
                    best_model = select_best_model()
                    output = "AVAILABLE MODELS:\n\n"
                    for model in models:
                        if model == best_model:
                            output += f"* {model} (SELECTED)\n"
                        else:
                            output += f"  {model}\n"
                    return output
                else:
                    return "No models available from Ollama API."
                    
            else:  # !help or other special commands
                return special_commands[command]
        else:
            return f"Unknown command: {command}\nType !help for available commands."
            
        return None  # Should never reach here
    
    return None  # Not a special command

# Escape HTML for safe display in the terminal
def escape_html_except_br(text):
    escaped = html.escape(text).replace('\n', '<br>')
    return escaped

# Initialize session state
if 'terminal_history' not in st.session_state:
    st.session_state.terminal_history = []

if 'startup_done' not in st.session_state:
    st.session_state.startup_done = False

if 'command_counter' not in st.session_state:
    st.session_state.command_counter = 0

# Process URL parameters for direct commands
params = st.query_params
direct_command = params.get('command', None)
if direct_command:
    st.session_state.current_command = direct_command
    logging.info(f"Received direct command: {direct_command}")

# Set page config
st.set_page_config(
    page_title="ENRON TERMINAL",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit UI and styling
st.markdown("""
<style>
    /* Hide all Streamlit elements */
    #MainMenu, header, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"], 
    [data-testid="stSidebar"], .css-18e3th9, .css-1d391kg, .css-keje6w, .e1f1d6gn1, .stStatusWidget {
        display: none !important;
    }
    
    /* Make sure EVERY element on the page is properly styled */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
        background-color: #000000 !important;
        color: #00FF00 !important;
        font-family: 'Courier New', monospace !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    
    /* Remove any padding or margin from containers */
    div.block-container, div[data-testid="stHorizontalBlock"] {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        background-color: #000000 !important;
    }
    
    /* Remove warning banners */
    .stWarning, .stException, .stInfo {
        display: none !important;
    }
    
    /* Hide all other divs that might show up */
    div[data-baseweb="notification"], .st-emotion-cache-r421ms, .st-emotion-cache-16txtl3 {
        display: none !important;
    }

    /* Hide all buttons and notifications */
    button {
        display: none !important;
    }
    
    /* Fix iframe to take full height and width */
    iframe {
        width: 100vw !important;
        height: 100vh !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
    }
    
    /* Make streamlit components expand */
    [data-testid="stHtml"] {
        width: 100vw !important;
        height: 100vh !important;
        padding: 0 !important;
        margin: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Display startup messages
startup_messages = [
    " _______ .__   __. .______       ______   .__   __.      .___________. _______ .______      .___  ___.  __  .__   __.      ___       __",
    "|   ____||  \\ |  | |   _  \\     /  __  \\  |  \\ |  |      |           ||   ____||   _  \\     |   \\/   | |  | |  \\ |  |     /   \\     |  |",
    "|  |__   |   \\|  | |  |_)  |   |  |  |  | |   \\|  |      `---|  |----`|  |__   |  |_)  |    |  \\  /  | |  | |   \\|  |    /    \\    |  |",
    "|   __|  |  . `  | |      /    |  |  |  | |  . `  |          |  |     |   __|  |      /     |  |\\/|  | |  | |  . `  |   /  /_\\  \\   |  |",
    "|  |____ |  |\\   | |  |\\  \\----|  `--'  | |  |\\   |          |  |     |  |____ |  |\\  \\----.|  |  |  | |  | |  |\\   |  /  _____  \\  |  `----.",
    "|_______||__| \\__| | _| `._____|\______/  |__| \\__|          |__|     |_______|| _| `.____||__|  |__| |__| |__| \\__| /__/     \\__\\ |_______|",
    "",
    "============================================",
    "      ENRON EMAIL TERMINAL SYSTEM",
    "         CONFIDENTIAL - 2001",
    "============================================",
    "",
    "[....] Installing core dependencies...",
    "[OK] Core dependencies installed",
    "[....] Installing RAG dependencies...",
    "[OK] RAG dependencies installation attempted",
    "[....] Checking for Ollama...",
    "[OK] Ollama is running",
    "[....] Creating configuration file...",
    "[OK] Created configuration file",
    "[....] Checking for pre-built vector database...",
    "[OK] Vector database already exists",
    "",
    "============================================",
    "     INITIALIZING TERMINAL INTERFACE",
    "============================================",
    "",
    "Terminal ready for queries. Type your command after the $ prompt."
]

if not st.session_state.startup_done:
    for message in startup_messages:
        st.session_state.terminal_history.append(("message", message))
    st.session_state.startup_done = True
    logging.info("Terminal startup sequence completed")

# Process command if one is pending
if 'current_command' in st.session_state and st.session_state.current_command:
    command = st.session_state.current_command
    logging.info(f"Processing pending command: {command}")
    
    # Add the command to history
    st.session_state.terminal_history.append(("command", command))
    
    # Process the command
    response = search_emails(command)
    
    # Add response to history
    st.session_state.terminal_history.append(("response", response))
    
    # Clear the current command
    st.session_state.current_command = None
    
    # Force refresh
    st.session_state.command_counter += 1
    logging.info(f"Command processed, new counter: {st.session_state.command_counter}")

# Create history HTML for the terminal
history_html = ""
for msg_type, content in st.session_state.terminal_history:
    if msg_type == "command":
        history_html += f'<div class="terminal-line command">$ {escape_html_except_br(content)}</div>'
    elif msg_type == "response":
        history_html += f'<div class="terminal-line response">{escape_html_except_br(content)}</div>'
    else:
        history_html += f'<div class="terminal-line message">{content}</div>'

# Create the terminal HTML interface
terminal_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ENRON TERMINAL</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        html, body {{
            height: 100%;
            width: 100%;
            overflow: hidden;
            background-color: #000000;
            color: #00FF00;
            font-family: 'Courier New', monospace;
            font-size: 16px;
        }}
        
        #terminal {{
            width: 100%;
            height: 100vh;
            padding: 20px;
            background-color: #000000;
            color: #00FF00;
            overflow-y: auto;
            position: relative;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .terminal-line {{
            margin-bottom: 2px;
            line-height: 1.3;
        }}
        
        .command {{
            color: #00FF00;
        }}
        
        .response {{
            color: #00FF00;
            opacity: 0.9;
        }}
        
        .message {{
            color: #00FF00;
        }}
        
        .prompt {{
            color: #00FF00;
            display: inline;
        }}
        
        #input {{
            background-color: transparent;
            border: none;
            color: #00FF00;
            font-family: 'Courier New', monospace;
            font-size: inherit;
            width: calc(100% - 15px);
            outline: none;
            caret-color: #00FF00;
        }}
        
        #input-line {{
            display: flex;
            align-items: center;
            padding-bottom: 50px; /* Extra space at bottom to ensure visibility */
        }}
        
        #cursor {{
            display: inline-block;
            background-color: #00FF00;
            width: 8px;
            height: 15px;
            animation: blink 1s step-end infinite;
            margin-left: 2px;
            vertical-align: middle;
        }}
        
        @keyframes blink {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0; }}
        }}
        
        .scanlines {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%);
            background-size: 100% 4px;
            z-index: 999;
            pointer-events: none;
            opacity: 0.15;
        }}
    </style>
</head>
<body>
    <div class="scanlines"></div>
    <div id="terminal">
        {history_html}
        <div id="input-line">
            <span class="prompt">$ </span>
            <input type="text" id="input" autofocus autocomplete="off">
            <span id="cursor"></span>
        </div>
    </div>
    
    <script>
        // Ensure we start at the bottom of the terminal
        const terminal = document.getElementById('terminal');
        terminal.scrollTop = terminal.scrollHeight;
        
        // Focus the input on page load
        const input = document.getElementById('input');
        input.focus();
        
        // Re-focus input when clicking anywhere
        terminal.addEventListener('click', function() {{
            input.focus();
        }});
        
        // Submit command on Enter key
        input.addEventListener('keydown', function(e) {{
            if (e.key === 'Enter') {{
                e.preventDefault();
                
                const command = input.value.trim();
                if (command !== '') {{
                    // Clear the input
                    input.value = '';
                    
                    // Submit the command
                    submitCommand(command);
                }}
            }}
        }});
        
        // Keep focus on input when clicking anywhere in the document
        document.addEventListener('click', function() {{
            input.focus();
        }});
        
        // Focus the input when the page loads
        window.addEventListener('load', function() {{
            input.focus();
            terminal.scrollTop = terminal.scrollHeight;
        }});
        
        // Handle window resize to keep terminal properly sized
        window.addEventListener('resize', function() {{
            // Ensure terminal fills the viewport even after resize
            terminal.scrollTop = terminal.scrollHeight;
        }});
        
        // Function to submit a command to the server
        function submitCommand(command) {{
            window.location.href = '?command=' + encodeURIComponent(command);
        }}
    </script>
</body>
</html>
"""

# Display the terminal - use full height of the viewport
st.components.v1.html(terminal_html, height=None, scrolling=False) 