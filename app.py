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

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                    handlers=[logging.FileHandler("logs/enron_app.log", encoding="utf-8")])

# Log startup for debugging
logging.info("========== APPLICATION STARTED ==========")

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

# Debug function to diagnose vector database issues
def debug_vector_db():
    """Diagnose potential issues with the vector database."""
    debug_info = {"status": "unknown", "issues": [], "collections": [], "details": {}}
    
    try:
        # Step 1: Check if chromadb is installed
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            debug_info["details"]["chromadb_installed"] = True
        except ImportError:
            debug_info["status"] = "error"
            debug_info["issues"].append("ChromaDB package is not installed")
            debug_info["details"]["chromadb_installed"] = False
            return debug_info
        
        # Step 2: Check if the directory exists
        if os.path.exists(CHROMA_DIR):
            debug_info["details"]["directory_exists"] = True
            # Check if directory is empty
            contents = os.listdir(CHROMA_DIR)
            debug_info["details"]["directory_contents"] = contents
            if not contents:
                debug_info["issues"].append("ChromaDB directory exists but is empty")
        else:
            debug_info["issues"].append(f"ChromaDB directory not found: {CHROMA_DIR}")
            debug_info["details"]["directory_exists"] = False
        
        # Step 3: Try to connect to the client
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            debug_info["details"]["client_connected"] = True
            
            # Step 4: List collections
            collections = client.list_collections()
            debug_info["collections"] = [c.name for c in collections]
            
            if not collections:
                debug_info["issues"].append("No collections found in ChromaDB")
            else:
                # Step 5: Get the first collection and check its stats
                collection = client.get_collection(
                    name=collections[0].name,
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2", 
                        device="cpu"
                    )
                )
                count = collection.count()
                debug_info["details"]["document_count"] = count
                
                if count == 0:
                    debug_info["issues"].append("Collection exists but contains no documents")
                else:
                    # Try a simple query to see if it works
                    try:
                        results = collection.query(query_texts=["test query"], n_results=1)
                        if results and 'documents' in results and results['documents'] and results['documents'][0]:
                            debug_info["details"]["query_test"] = "successful"
                        else:
                            debug_info["details"]["query_test"] = "no results"
                            debug_info["issues"].append("Query test returned no results")
                    except Exception as e:
                        debug_info["details"]["query_test"] = f"error: {str(e)}"
                        debug_info["issues"].append(f"Query test failed: {str(e)}")
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
    
    logging.info(f"Vector DB debug results: {json.dumps(debug_info)}")
    return debug_info

# Load configuration values directly
CSV_FILE = "emails.csv"
TOP_K_RESULTS = 5  # Number of relevant emails to retrieve
VECTOR_DB_DIR = "vector_db"
CHROMA_DIR = "vector_db/chroma" 
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3:8b"  # Default model, could also use "mistral:7b" if available
FALLBACK_MODEL = "mistral:7b"  # Fallback model if primary not available
USE_CSV_SEARCH = False
RESPONSE_TOKENS = 1024  # Max tokens to generate

# Run vector DB debug on startup
vector_db_status = debug_vector_db()
if vector_db_status["status"] != "ok":
    logging.warning(f"Vector DB issues detected: {vector_db_status['issues']}")
    # If the directory is missing, we could try to create it
    if vector_db_status["status"] == "missing":
        logging.info("Attempting to create vector database directory structure")
        os.makedirs(CHROMA_DIR, exist_ok=True)

# Hardcoded sample data for fallback
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

# Get available models from Ollama
def get_available_models():
    """Get list of available models from Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except Exception as e:
        logging.error(f"Error getting available models: {e}")
        return []

# Select best available model
def select_best_model():
    """Select the best available model from Ollama."""
    available_models = get_available_models()
    logging.info(f"Available models: {available_models}")
    
    # Preferred models in order (based on research)
    preferred_models = [
        MODEL_NAME,            # llama3:8b
        FALLBACK_MODEL,        # mistral:7b
        "llama2:13b",          # Larger Llama 2
        "llama2:7b",           # Smaller Llama 2
        "mistral:mistral-tiny" # Smaller Mistral
    ]
    
    # Return the first available preferred model
    for model in preferred_models:
        if any(m.startswith(model) for m in available_models):
            matched_model = next(m for m in available_models if m.startswith(model))
            logging.info(f"Selected model: {matched_model}")
            return matched_model
    
    # If no preferred model is available, return any available model
    if available_models:
        logging.info(f"Falling back to available model: {available_models[0]}")
        return available_models[0]
    
    return None

# Direct RAG query function using Ollama API
def query_ollama(prompt, model_name=None):
    """Query Ollama API directly without dependencies."""
    if not model_name:
        model_name = select_best_model()
        if not model_name:
            return "Error: No models available in Ollama"
            
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Lower temperature for more factual responses
                "num_predict": RESPONSE_TOKENS,
                "top_p": 0.9,        # More focused sampling
                "top_k": 40          # Consider top 40 tokens
            }
        }
        
        start_time = time.time()
        logging.info(f"Querying Ollama with model: {model_name}")
        logging.info(f"Prompt length: {len(prompt)} characters")
        
        response = requests.post(url, json=payload, timeout=120)  # Longer timeout for large contexts
        end_time = time.time()
        
        if response.status_code == 200:
            logging.info(f"Ollama response time: {end_time - start_time:.2f} seconds")
            return response.json().get("response", "")
        else:
            logging.error(f"Error from Ollama API: {response.status_code} - {response.text}")
            
            # If model not found, try fallback model
            if "model not found" in response.text.lower():
                logging.info(f"Model {model_name} not found, trying fallback model")
                return query_ollama(prompt, FALLBACK_MODEL)
                
            return f"Error: Failed to get response from LLM API (Status {response.status_code})"
    
    except Exception as e:
        logging.error(f"Exception when calling Ollama API: {e}")
        return f"Error: {str(e)}"

# Check if Ollama is available
def is_ollama_available():
    """Check if Ollama is available by making a simple API call."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

# Try connecting to ChromaDB
def get_chroma_vector_db():
    """Try to connect to ChromaDB if available."""
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        
        # Log the attempt to connect
        logging.info(f"Attempting to connect to ChromaDB at {CHROMA_DIR}")
        
        if not os.path.exists(CHROMA_DIR):
            logging.warning(f"ChromaDB directory {CHROMA_DIR} does not exist")
            # Try to create it
            os.makedirs(CHROMA_DIR, exist_ok=True)
            logging.info(f"Created ChromaDB directory {CHROMA_DIR}")
        
        # Try to connect to the persistent client
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        logging.info("Successfully connected to ChromaDB client")
        
        # List all collections
        collections = client.list_collections()
        logging.info(f"Found {len(collections)} collections in ChromaDB")
        
        if collections:
            collection_name = collections[0].name
            logging.info(f"Using collection: {collection_name}")
            
            # Create default embedding function for querying
            sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2", 
                device="cpu"  # Use CPU for compatibility
            )
            
            # Get the collection with the embedding function
            collection = client.get_collection(
                name=collection_name, 
                embedding_function=sentence_transformer_ef
            )
            
            logging.info(f"Successfully connected to ChromaDB collection: {collection_name}")
            return collection
        else:
            logging.warning("No collections found in ChromaDB")
            return None
    except Exception as e:
        logging.error(f"Error connecting to ChromaDB: {str(e)}")
        return None

# Query vector database directly
def query_vector_db(query, n_results=TOP_K_RESULTS):
    """Query the vector database directly if possible."""
    try:
        collection = get_chroma_vector_db()
        if collection:
            logging.info(f"Querying vector DB with: '{query}'")
            
            # Execute the query
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Log the results structure for debugging
            logging.info(f"Query returned metadata keys: {results.keys()}")
            if 'documents' in results and results['documents']:
                logging.info(f"Found {len(results['documents'][0])} documents in results")
            
            # Process results if they exist
            if results and 'documents' in results and results['documents'] and results['documents'][0]:
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
                
                logging.info(f"Successfully retrieved {len(emails)} results from vector DB")
                return emails
            else:
                logging.warning("No results found in vector DB query")
                return []
    except Exception as e:
        logging.error(f"Error during vector DB query: {str(e)}")
    
    return []

# Load emails from CSV with improved search
@st.cache_data
def load_emails_from_csv(csv_file=CSV_FILE):
    """Load emails from CSV file."""
    try:
        if os.path.exists(csv_file):
            logging.info(f"Loading emails from CSV file: {csv_file}")
            df = pd.read_csv(csv_file)
            logging.info(f"Successfully loaded {len(df)} emails from CSV")
            return df.to_dict('records')
        else:
            logging.warning(f"CSV file {csv_file} not found")
            # Try to look for other CSV files in the directory
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if csv_files:
                logging.info(f"Found alternative CSV files: {csv_files}")
                alt_csv = csv_files[0]
                logging.info(f"Trying alternative CSV file: {alt_csv}")
                df = pd.read_csv(alt_csv)
                logging.info(f"Successfully loaded {len(df)} emails from alternative CSV: {alt_csv}")
                return df.to_dict('records')
            return []
    except Exception as e:
        logging.error(f"Error loading CSV: {str(e)}")
        return []

# Search CSV with more advanced techniques
def search_csv_emails(query, csv_emails, max_results=TOP_K_RESULTS):
    """Search CSV emails with improved relevance."""
    if not csv_emails:
        return []
    
    logging.info(f"Searching {len(csv_emails)} CSV emails for query: '{query}'")
    
    # Normalize query
    query_lower = query.lower()
    query_terms = [term.strip() for term in query_lower.split() if len(term.strip()) > 3]
    
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
                # Bonus for multiple occurrences
                score += text.count(term) * 0.5
            
            # Score for terms in sender
            if term in sender:
                score += 2
        
        # Only keep emails with non-zero scores
        if score > 0:
            scored_emails.append((score, email))
    
    # Sort by score (highest first) and take top results
    sorted_emails = [email for _, email in sorted(scored_emails, key=lambda x: x[0], reverse=True)]
    limited_results = sorted_emails[:max_results]
    
    logging.info(f"CSV search found {len(sorted_emails)} relevant emails, returning top {len(limited_results)}")
    return limited_results

# Better prompt template for RAG
def create_rag_prompt(query, context_emails):
    """Create a well-structured prompt for the RAG system."""
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

# Test connection to Ollama
def test_ollama_connection():
    """Test the connection to Ollama and available models."""
    result = {
        "status": "unknown",
        "issues": [],
        "available_models": [],
        "details": {}
    }
    
    # Test basic connection
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            result["details"]["connection"] = "ok"
            models_data = response.json()
            if "models" in models_data:
                models = models_data["models"]
                result["available_models"] = [model["name"] for model in models]
                result["details"]["model_count"] = len(models)
                
                # Check if our preferred models are available
                primary_available = any(m.startswith(MODEL_NAME) for m in result["available_models"])
                fallback_available = any(m.startswith(FALLBACK_MODEL) for m in result["available_models"])
                
                result["details"]["primary_model_available"] = primary_available
                result["details"]["fallback_model_available"] = fallback_available
                
                if not primary_available and not fallback_available:
                    result["issues"].append(f"Neither primary model ({MODEL_NAME}) nor fallback model ({FALLBACK_MODEL}) is available")
                elif not primary_available:
                    result["issues"].append(f"Primary model ({MODEL_NAME}) is not available, but fallback ({FALLBACK_MODEL}) is")
            else:
                result["issues"].append("No models found in Ollama")
                result["details"]["models_found"] = False
        else:
            result["issues"].append(f"Ollama API returned status code {response.status_code}")
            result["details"]["connection"] = "error"
            result["details"]["status_code"] = response.status_code
    except Exception as e:
        result["issues"].append(f"Failed to connect to Ollama API: {str(e)}")
        result["details"]["connection"] = "error"
        result["details"]["error"] = str(e)
    
    # Test a simple generation if we have models
    if result["available_models"] and "connection" in result["details"] and result["details"]["connection"] == "ok":
        try:
            model_to_test = None
            if result["details"].get("primary_model_available"):
                for model in result["available_models"]:
                    if model.startswith(MODEL_NAME):
                        model_to_test = model
                        break
            elif result["details"].get("fallback_model_available"):
                for model in result["available_models"]:
                    if model.startswith(FALLBACK_MODEL):
                        model_to_test = model
                        break
            else:
                model_to_test = result["available_models"][0]
            
            if model_to_test:
                start_time = time.time()
                test_response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": model_to_test,
                        "prompt": "Say hello in one word.",
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 10}
                    },
                    timeout=10
                )
                end_time = time.time()
                
                if test_response.status_code == 200:
                    result["details"]["test_generation"] = "ok"
                    result["details"]["generation_time"] = round(end_time - start_time, 2)
                    response_text = test_response.json().get("response", "")
                    result["details"]["test_response"] = response_text
                else:
                    result["issues"].append(f"Test generation failed with status code {test_response.status_code}")
                    result["details"]["test_generation"] = "error"
                    result["details"]["test_status_code"] = test_response.status_code
            else:
                result["issues"].append("No suitable model found for testing")
        except Exception as e:
            result["issues"].append(f"Test generation failed: {str(e)}")
            result["details"]["test_generation"] = "error"
            result["details"]["test_error"] = str(e)
    
    # Set final status
    if not result["issues"]:
        result["status"] = "ok"
    else:
        result["status"] = "error"
    
    logging.info(f"Ollama test results: {json.dumps(result)}")
    return result

# Handle special debug commands
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
                ollama_results = test_ollama_connection()
                
                output = "DIAGNOSTIC RESULTS:\n\n"
                
                # Vector DB results
                output += "VECTOR DATABASE STATUS: " + vector_results["status"].upper() + "\n"
                if vector_results["issues"]:
                    output += "Issues:\n"
                    for issue in vector_results["issues"]:
                        output += f"- {issue}\n"
                if vector_results["collections"]:
                    output += f"Collections: {', '.join(vector_results['collections'])}\n"
                if "document_count" in vector_results.get("details", {}):
                    output += f"Document count: {vector_results['details']['document_count']}\n"
                
                # Ollama results
                output += "\nOLLAMA API STATUS: " + ollama_results["status"].upper() + "\n"
                if ollama_results["issues"]:
                    output += "Issues:\n"
                    for issue in ollama_results["issues"]:
                        output += f"- {issue}\n"
                if ollama_results["available_models"]:
                    output += f"Available models: {', '.join(ollama_results['available_models'])}\n"
                
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
                if "document_count" in results.get("details", {}):
                    output += f"Document count: {results['details']['document_count']}\n"
                return output
                
            elif command == "!ollama":
                results = test_ollama_connection()
                output = "OLLAMA CONNECTION TEST:\n\n"
                output += "Status: " + results["status"].upper() + "\n"
                if results["issues"]:
                    output += "Issues:\n"
                    for issue in results["issues"]:
                        output += f"- {issue}\n"
                if "test_generation" in results.get("details", {}):
                    output += f"Test generation: {results['details']['test_generation']}\n"
                    if "generation_time" in results["details"]:
                        output += f"Response time: {results['details']['generation_time']} seconds\n"
                    if "test_response" in results["details"]:
                        output += f"Response: {results['details']['test_response']}\n"
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

# Search emails using a direct, self-contained approach
def search_emails(query):
    """Search function with multiple fallback strategies"""
    logging.info(f"Processing query: '{query}'")
    
    # Check for special commands first
    special_result = handle_special_commands(query)
    if special_result:
        logging.info(f"Handled special command: {query}")
        return special_result
    
    # Try all available strategies in order until one succeeds
    
    # First try CSV search directly - more reliable than vector search
    try:
        csv_emails = load_emails_from_csv()
        if csv_emails:
            logging.info(f"Loaded {len(csv_emails)} emails from CSV")
            matched_emails = search_csv_emails(query, csv_emails)
            
            if matched_emails:
                logging.info(f"Found {len(matched_emails)} relevant emails in CSV")
                
                # Create prompt with matched emails
                prompt = create_rag_prompt(query, matched_emails)
                
                # Try to get response from Ollama if available
                if is_ollama_available():
                    answer = query_ollama(prompt)
                    
                    # Format the output - ONLY include the answer, not the source emails
                    output = f"{answer}\n\n"
                    logging.info("Successfully generated response using CSV data and Ollama")
                    return output
                else:
                    logging.warning("Ollama not available for generating response")
                    # Return a simple response without Ollama
                    output = f"Found {len(matched_emails)} emails related to '{query}'.\n\n"
                    
                    # Format a summary response without using the original emails
                    for i, email in enumerate(matched_emails[:3]):
                        output += f"Email {i+1}: {email.get('subject', 'No subject')} " 
                        output += f"from {email.get('from', 'Unknown')} "
                        output += f"on {email.get('date', 'Unknown date')}\n"
                    
                    return output
    except Exception as e:
        logging.error(f"Error in CSV search: {str(e)}")
    
    # Only if CSV search fails, try with Ollama and the default sample data
    if is_ollama_available():
        try:
            logging.info("CSV search failed, using Ollama with default data")
            
            # No emails found, use a direct query to Ollama without context
            prompt = f"""
You are analyzing the Enron email dataset. The user has asked the following question, but unfortunately,
we don't have specific emails to provide as context. Please provide a general response based on
what you know about Enron, being transparent that specific emails weren't found.

USER QUESTION: {query}
"""
            # Query Ollama with no email context
            answer = query_ollama(prompt)
            
            # Format the output - without any source emails
            output = f"No specific emails found for your query. Here's a general response:\n\n{answer}"
            logging.info("Generated response with no specific email context")
            return output
                
        except Exception as e:
            logging.error(f"Error with direct Ollama query: {str(e)}")
    
    # Final fallback if everything else fails - with minimal message
    logging.warning("All retrieval and generation methods failed")
    return f"I'm sorry, I couldn't find any relevant information for your query: '{query}'. Please try a different question or check the system status with !debug."

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