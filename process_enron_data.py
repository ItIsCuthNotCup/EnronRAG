#!/usr/bin/env python3
"""
Enron Email Dataset Processor

This script processes the Enron email dataset for use with the terminal interface.
It copies the dataset, creates embeddings, and builds a vector database for RAG.
"""

import os
import sys
import shutil
import pandas as pd
import time
import logging
import traceback
from pathlib import Path

# Import configuration
from config import (
    DATA_DIR, 
    PROCESSED_DIR, 
    VECTOR_DB_DIR, 
    CHROMA_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_BATCH_SIZE
)

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='logs/data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Source file locations to check (in order of preference)
SOURCE_DATA_PATHS = [
    # Current directory
    "emails.csv",
    # Data directory
    f"{DATA_DIR}/emails.csv",
    # Other possible locations
    "../emails.csv",
    "../../emails.csv",
    # User-specified path - Move to last priority
    r"C:\Users\Jacob\OneDrive\Desktop\Georgia Tech\Spring 2025\CSE 6242\GROUP PROJECT\emails.csv"
]

def print_step(message, step_type="INFO"):
    """Print a formatted message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    prefix = {
        "INFO": "[INFO]",
        "WAIT": "[....]",
        "OK": "[OK]  ",
        "WARN": "[WARNING]",
        "ERROR": "[ERROR]"
    }.get(step_type, "[INFO]")
    
    print(f"{prefix} {message}")
    logging.info(message)

def setup_directories():
    """Create necessary directories if they don't exist."""
    for directory in [DATA_DIR, PROCESSED_DIR, VECTOR_DB_DIR, "logs"]:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Directory {directory} created/verified")

def find_source_data():
    """Find the Enron dataset from potential locations."""
    for path in SOURCE_DATA_PATHS:
        if os.path.exists(path):
            print_step(f"Found dataset at: {path}", "OK")
            return path
    
    print_step("Enron email dataset not found in any of the expected locations", "ERROR")
    print_step("Please download the dataset or place it in one of these locations:", "ERROR")
    for path in SOURCE_DATA_PATHS:
        print(f"  - {path}")
    return None

def copy_dataset(source_path):
    """Copy the dataset from source to data directory."""
    target_file = os.path.join(DATA_DIR, "enron_emails.csv")
    
    # Check if the file already exists in our data directory
    if os.path.exists(target_file):
        file_size = os.path.getsize(target_file) / (1024 * 1024)  # Size in MB
        print_step(f"Dataset already in data directory ({file_size:.2f} MB)", "OK")
        return target_file
    
    print_step(f"Copying dataset from {source_path}", "WAIT")
    
    try:
        # Copy with progress reporting for large files
        source_size = os.path.getsize(source_path)
        source_size_mb = source_size / (1024 * 1024)
        
        print_step(f"Source file size: {source_size_mb:.2f} MB", "INFO")
        
        # Handle large file copy with progress reporting
        with open(source_path, 'rb') as src_file, open(target_file, 'wb') as dst_file:
            copied = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            start_time = time.time()
            
            while True:
                chunk = src_file.read(chunk_size)
                if not chunk:
                    break
                
                dst_file.write(chunk)
                copied += len(chunk)
                
                # Report progress every ~5%
                progress = (copied / source_size) * 100
                elapsed = time.time() - start_time
                
                if elapsed > 0:
                    speed = copied / (1024 * 1024 * elapsed)  # MB/s
                    eta = (source_size - copied) / (1024 * 1024 * speed) if speed > 0 else 0
                    
                    sys.stdout.write(f"\rProgress: {progress:.1f}% ({copied/(1024*1024):.1f} MB of {source_size_mb:.1f} MB) "
                                    f"Speed: {speed:.2f} MB/s ETA: {eta:.1f}s")
                    sys.stdout.flush()
                
        print("\n")
        print_step("Dataset copied successfully", "OK")
        return target_file
    
    except Exception as e:
        print_step(f"Error copying dataset: {str(e)}", "ERROR")
        logging.error(traceback.format_exc())
        return None

def process_dataset(csv_path):
    """Process the dataset and create vector embeddings."""
    try:
        # Create vector database directory if it doesn't exist
        os.makedirs(VECTOR_DB_DIR, exist_ok=True)
        os.makedirs(CHROMA_DIR, exist_ok=True)
        
        # Force recreation of vector database if it's empty or incomplete
        if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
            db_files = os.listdir(CHROMA_DIR)
            required_files = ['chroma.sqlite3']
            
            if all(file in db_files for file in required_files):
                db_size = sum(os.path.getsize(os.path.join(CHROMA_DIR, f)) for f in os.listdir(CHROMA_DIR) if os.path.isfile(os.path.join(CHROMA_DIR, f)))
                db_size_mb = db_size / (1024 * 1024)
                print_step(f"Vector database exists ({db_size_mb:.2f} MB)", "OK")
                print_step(f"Database location: {os.path.abspath(CHROMA_DIR)}", "INFO")
                # Create a visible marker file to show database location
                with open(os.path.join(VECTOR_DB_DIR, "DATABASE_LOCATION.txt"), "w") as f:
                    f.write(f"The Enron email vector database is located at:\n{os.path.abspath(CHROMA_DIR)}\n\nSize: {db_size_mb:.2f} MB\nCreated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                return True
            else:
                print_step("Vector database exists but appears incomplete - recreating", "WARN")
                # Remove incomplete database
                for file in os.listdir(CHROMA_DIR):
                    file_path = os.path.join(CHROMA_DIR, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print_step(f"Error cleaning up database file {file_path}: {e}", "WARN")
        
        # Import dependencies for RAG
        print_step("Importing RAG dependencies", "WAIT")
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.embeddings import HuggingFaceEmbeddings
            from langchain.vectorstores import Chroma
            from langchain.document_loaders import DataFrameLoader
        except ImportError as e:
            print_step(f"Error importing RAG dependencies: {str(e)}", "ERROR")
            print_step("Try running: pip install langchain chromadb sentence-transformers huggingface-hub", "INFO")
            return False
        
        # Load the CSV file
        print_step("Loading CSV file", "WAIT")
        
        try:
            df = pd.read_csv(csv_path)
            print_step(f"Loaded dataframe with {len(df)} rows", "OK")
        except Exception as e:
            print_step(f"Error loading CSV: {str(e)}", "ERROR")
            return False
        
        # Use more emails for better context - up to 25,000 for better performance
        min_sample_size = 1000  # Minimum number of emails for a useful RAG system
        target_sample_size = min(25000, len(df))  # Reduced target sample size for better performance
        
        if len(df) < min_sample_size:
            print_step(f"Warning: Dataset only has {len(df)} emails, which is smaller than recommended ({min_sample_size})", "WARN")
            # Use all available emails if less than minimum
            print_step(f"Using all {len(df)} available emails", "INFO")
        elif len(df) > target_sample_size:
            print_step(f"Taking a random sample of {target_sample_size} emails from {len(df)} total", "INFO")
            # Use random sampling with fixed seed for reproducibility
            df = df.sample(target_sample_size, random_state=42)
            print_step(f"Random sampling complete - using {len(df)} emails", "OK")
        else:
            print_step(f"Using all {len(df)} emails in the dataset", "INFO")
        
        # Check the DataFrame columns
        columns = df.columns.tolist()
        print_step(f"Columns found in dataset: {', '.join(columns)}", "INFO")
        
        # Determine which columns to use for document content
        content_column = None
        for potential_column in ['content', 'message', 'body', 'text', 'email']:
            if potential_column in columns:
                content_column = potential_column
                break
        
        if not content_column:
            print_step("Could not identify a content column in the dataset", "ERROR")
            print_step(f"Available columns: {columns}", "ERROR")
            return False
        
        print_step(f"Using '{content_column}' as the content column", "OK")
        
        # Create a text field combining relevant metadata
        print_step("Creating document objects", "WAIT")
        
        # Identify metadata columns
        metadata_columns = []
        for col in ['subject', 'from', 'to', 'date', 'cc', 'bcc']:
            if col in columns:
                metadata_columns.append(col)
        
        if metadata_columns:
            print_step(f"Using metadata columns: {', '.join(metadata_columns)}", "INFO")
        
        # Create documents
        try:
            loader = DataFrameLoader(df, page_content_column=content_column)
            documents = loader.load()
            print_step(f"Created {len(documents)} document objects", "OK")
        except Exception as e:
            print_step(f"Error creating documents: {str(e)}", "ERROR")
            return False
        
        # Split documents - use smaller chunks for faster processing
        print_step("Splitting documents into chunks", "WAIT")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents(documents)
            print_step(f"Created {len(chunks)} document chunks", "OK")
        except Exception as e:
            print_step(f"Error splitting documents: {str(e)}", "ERROR")
            return False
        
        # Generate embeddings
        print_step("Initializing embedding model (this may take a moment)", "WAIT")
        try:
            # Use the configured embedding model
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            print_step("Embedding model initialized", "OK")
        except Exception as e:
            print_step(f"Error initializing embeddings: {str(e)}", "ERROR")
            return False
        
        # Create and persist vector store in batches
        print_step("Creating vector database (this will take some time)", "WAIT")
        try:
            # Process chunks in batches to avoid memory issues
            total_chunks = len(chunks)
            
            # Initialize the vector store with the first batch
            print_step(f"Processing in batches of {MAX_BATCH_SIZE} chunks", "INFO")
            
            # Initialize empty vector store
            vectordb = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )
            
            # Add chunks in batches
            for batch_start in range(0, total_chunks, MAX_BATCH_SIZE):
                batch_end = min(batch_start + MAX_BATCH_SIZE, total_chunks)
                current_batch = chunks[batch_start:batch_end]
                
                print_step(f"Processing batch {batch_start//MAX_BATCH_SIZE + 1}/{(total_chunks-1)//MAX_BATCH_SIZE + 1} ({len(current_batch)} chunks)", "INFO")
                
                # Create and add batch to vector store
                vectordb.add_documents(current_batch)
                
                # Persist after each batch
                vectordb.persist()
                print_step(f"Batch {batch_start//MAX_BATCH_SIZE + 1} processed and persisted", "OK")
            
            # Verify database was created
            if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
                db_size = sum(os.path.getsize(os.path.join(CHROMA_DIR, f)) for f in os.listdir(CHROMA_DIR) if os.path.isfile(os.path.join(CHROMA_DIR, f)))
                db_size_mb = db_size / (1024 * 1024)
                print_step(f"Vector database created and persisted successfully ({db_size_mb:.2f} MB)", "OK")
                print_step(f"Database location: {os.path.abspath(CHROMA_DIR)}", "INFO")
                print_step(f"Contains embeddings for {len(chunks)} text chunks from {len(documents)} emails", "INFO")
                
                # Create a visible marker file to show database location
                with open(os.path.join(VECTOR_DB_DIR, "DATABASE_LOCATION.txt"), "w") as f:
                    f.write(f"The Enron email vector database is located at:\n{os.path.abspath(CHROMA_DIR)}\n\nSize: {db_size_mb:.2f} MB\nCreated: {time.strftime('%Y-%m-%d %H:%M:%S')}\nContains: {len(chunks)} text chunks from {len(documents)} emails")
                
                return True
            else:
                print_step("Vector database directory exists but appears empty", "ERROR")
                return False
                
        except Exception as e:
            print_step(f"Error creating vector database: {str(e)}", "ERROR")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print_step(f"Unexpected error in data processing: {str(e)}", "ERROR")
        traceback.print_exc()
        return False

def create_sample_data():
    """Create a more comprehensive sample dataset if no dataset is available."""
    print_step("Creating a realistic sample dataset for RAG testing", "WAIT")
    
    sample_file = os.path.join(DATA_DIR, "enron_emails.csv")
    
    # Check if we already have a sample file
    if os.path.exists(sample_file):
        print_step(f"Sample dataset already exists at {sample_file}", "OK")
        return sample_file
    
    # Create a larger sample with 100+ emails for better RAG performance
    print_step("Generating 100+ sample emails for proper RAG functionality", "INFO")
    
    # Base sample data with key Enron emails
    sample_data = """message_id,date,from,to,subject,content
1,2001-05-14,jeff.skilling@enron.com,all.employees@enron.com,California Energy Crisis,"To all Enron employees,

The California energy situation remains difficult. Our traders are working around the clock to manage our exposure in these volatile markets. Despite regulatory challenges, we continue to provide reliable energy services to our customers.

Regards,
Jeff Skilling
President and CEO"
2,2001-06-18,kenneth.lay@enron.com,board.directors@enron.com,Financial Update,"Board members,

The company's financial position remains strong despite market rumors. Our mark-to-market accounting practices follow all regulatory guidelines. Q2 projections look excellent, with continued growth in our energy trading business.

Ken Lay
Chairman"
3,2001-07-22,andrew.fastow@enron.com,richard.causey@enron.com,Special Purpose Entities,"Rick,

The SPE structure we've designed provides appropriate risk management benefits while keeping these vehicles off our balance sheet. Legal has reviewed all documentation and found no issues with our approach.

Let's discuss at tomorrow's meeting.
Andy"
4,2001-08-14,sherron.watkins@enron.com,kenneth.lay@enron.com,Accounting Concerns,"Mr. Lay,

I am incredibly nervous that we will implode in a wave of accounting scandals. I have concerns about some of the accounting practices we are using, particularly in the Raptor vehicles and with FAS 140 transactions. We've booked income without real economic gains.

I would appreciate a confidential meeting to discuss these issues.

Regards,
Sherron Watkins"
5,2001-09-26,jeffrey.mcmahon@enron.com,executive.committee@enron.com,Liquidity Issues,"Executive Committee,

Our available cash has been decreased by several hundred million dollars due to recent market developments. We need to address our liquidity position immediately. I recommend reducing trading exposure and accelerating asset sales.

Jeff McMahon
Treasurer"
6,2001-10-15,rebecca.mark@enron.com,jeff.skilling@enron.com,International Assets,"Jeff,

The performance of our international assets continues to underwhelm expectations. The Dabhol project in India has regulatory roadblocks, and Brazilian operations are facing currency devaluation issues. We may need to consider writing down some of these investments.

Rebecca Mark
Former Chair, Enron International"
7,2001-11-28,jeff.skilling@enron.com,kenneth.lay@enron.com,Resignation Reasons,"Ken,

As I mentioned before my departure, I believe the stock price will recover. My resignation was for personal reasons only. I maintain full confidence in Enron's business model and future prospects.

Best regards,
Jeff"
8,2001-10-30,kenneth.lay@enron.com,all.employees@enron.com,Company Direction,"Enron Employees,

Despite recent challenges, Enron remains a strong company with great assets and talented people. Our core businesses continue to perform well. We're taking decisive action to restore market confidence and maintain our industry leadership.

Ken Lay
Chairman and CEO"
9,2001-03-12,louise.kitchen@enron.com,john.lavorato@enron.com,Trading Strategies,"John,

Our trading strategies for Q2 should focus on leveraging our market information advantage in the Western markets. The California crisis provides significant opportunities if we position correctly. Let's increase our capacity bookings and structured deals.

Louise"
10,2001-08-22,david.delainey@enron.com,janet.dietrich@enron.com,Retail Energy Business,"Janet,

The retail energy business is underperforming projections. We need to either restructure or consider strategic alternatives. Acquisition costs are too high, and customer churn is affecting margins. Please prepare options for next week's meeting.

Dave"
11,2001-02-04,tim.belden@enron.com,kevin.presto@enron.com,West Power Trading,"Kevin,

The strategies we've implemented in the Western markets are working exceptionally well. Our traders have found creative ways to maximize profits through congestion management and scheduling practices. January was our most profitable month ever.

Tim Belden
Head of Trading, Western Region"
12,2001-07-12,vince.kaminski@enron.com,greg.whalley@enron.com,Risk Models,"Greg,

My team has completed the new VAR models for the trading portfolio. The results show our risk exposure is higher than previously estimated. We should consider reducing positions in certain volatile markets.

Vince Kaminski
Managing Director, Research"
13,2000-12-07,thomas.white@enron.com,kenneth.lay@enron.com,Enron Energy Services,"Ken,

EES continues its rapid growth in the retail energy sector. We've added 20 new Fortune 500 clients this quarter. Revenue recognition under mark-to-market accounting shows excellent profits, though actual cash collections are lagging projections.

Thomas White
Vice Chairman, Enron Energy Services"
14,2001-05-10,steven.kean@enron.com,richard.shapiro@enron.com,Regulatory Challenges,"Rick,

Our regulatory team is facing increased scrutiny in Washington. FERC is requesting extensive documentation on our trading activities in California. We need to coordinate our response carefully and maintain our messaging that price caps would harm the market.

Steve Kean
EVP and Chief of Staff"
15,2001-10-22,james.derrick@enron.com,andrew.fastow@enron.com,LJM Partnerships,"Andy,

The legal department has concerns about the related-party transactions with LJM partnerships. We need clearer documentation of the approval process and confirmation that these arrangements are truly arm's length. Please provide additional information.

James Derrick
General Counsel"
"""

    # Generate additional emails to reach 100+ total
    # Define common senders, recipients, and topics for variety
    senders = [
        "kenneth.lay@enron.com", "jeff.skilling@enron.com", "andrew.fastow@enron.com",
        "sherron.watkins@enron.com", "rebecca.mark@enron.com", "louise.kitchen@enron.com",
        "tim.belden@enron.com", "vince.kaminski@enron.com", "greg.whalley@enron.com",
        "richard.causey@enron.com", "david.delainey@enron.com", "john.lavorato@enron.com",
        "mark.frevert@enron.com", "ben.glisan@enron.com", "steven.kean@enron.com",
        "jeffrey.mcmahon@enron.com", "rick.buy@enron.com", "raymond.bowen@enron.com",
        "michael.brown@enron.com", "james.derrick@enron.com", "kevin.hannon@enron.com",
        "stanley.horton@enron.com", "mark.koenig@enron.com", "kenneth.rice@enron.com",
        "jeffrey.shankman@enron.com", "john.sherriff@enron.com", "jeff.dasovich@enron.com"
    ]
    
    recipients = [
        "board.directors@enron.com", "executive.committee@enron.com", "all.employees@enron.com",
        "trading.team@enron.com", "legal.department@enron.com", "finance.team@enron.com",
        "risk.management@enron.com", "international.team@enron.com", "california.desk@enron.com",
        "houston.office@enron.com", "london.office@enron.com", "regulatory.affairs@enron.com"
    ]
    
    subjects = [
        "Q{} Financial Results", "California Market Update", "Risk Management Concerns",
        "Trading Strategy for {}", "Regulatory Issues in {}", "SPE Documentation",
        "Mark-to-Market Accounting", "Liquidity Position", "Asset Sales Update",
        "Board Meeting Agenda", "Executive Committee Briefing", "Dabhol Project Status",
        "Brazil Operations", "West Power Trading", "East Power Markets",
        "Natural Gas Strategy", "LJM Partnership Structure", "Raptor Vehicles",
        "Employee Concerns", "Stock Price Volatility", "Media Response Strategy",
        "FERC Investigation", "Credit Rating Update", "Analyst Meeting Preparation",
        "Restructuring Options", "Cost Cutting Measures", "New Trading Opportunities",
        "Bandwidth Trading", "Weather Derivatives", "Energy Services Growth"
    ]
    
    content_templates = [
        "Dear {},\n\nI wanted to update you on our {} situation. The numbers are {} than expected, and we need to {}. Please review the attached {} and provide your feedback.\n\nRegards,\n{}",
        
        "Team,\n\nOur {} performance in {} has been {}. We're seeing {} in the market, which presents an opportunity to {}. Let's discuss this at the next meeting.\n\nThanks,\n{}",
        
        "{},\n\nThe {} issue we discussed needs immediate attention. I've spoken with {}, and they recommend we {}. This could impact our {} by approximately ${}M.\n\nBest,\n{}",
        
        "To the {} team,\n\nI'm writing regarding the recent {} situation. As you know, we've been working to {} for the past {} weeks. The results show {} and we should consider {} as our next step.\n\nThank you,\n{}",
        
        "{},\n\nFollowing up on our conversation about {}. I've reviewed the {} and found some concerning issues with {}. We need to {} before the end of the {}.\n\nRegards,\n{}"
    ]
    
    # Generate additional emails
    import random
    from datetime import datetime, timedelta
    
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2001, 12, 1)
    date_range = (end_date - start_date).days
    
    additional_emails = []
    for i in range(16, 105):  # Generate emails #16-104 (89 more emails)
        # Generate random date within range
        random_days = random.randint(0, date_range)
        email_date = start_date + timedelta(days=random_days)
        date_str = email_date.strftime("%Y-%m-%d")
        
        # Select random sender and recipient
        sender = random.choice(senders)
        recipient = random.choice(recipients)
        
        # Create subject with formatting
        subject_template = random.choice(subjects)
        if "{}" in subject_template:
            if "Q{}" in subject_template:
                quarter = random.choice(["1", "2", "3", "4"])
                subject = subject_template.format(quarter)
            else:
                topics = ["California", "Texas", "New York", "London", "Tokyo", 
                          "Houston", "Chicago", "Portland", "Europe", "Asia", 
                          "Q1", "Q2", "Q3", "Q4", "2001", "2002"]
                subject = subject_template.format(random.choice(topics))
        else:
            subject = subject_template
        
        # Create content with formatting
        template = random.choice(content_templates)
        
        # Fill in template placeholders with random but relevant content
        placeholders = template.count("{}")
        if placeholders >= 1:
            # First placeholder is usually a name/greeting
            name_options = ["team", recipient.split("@")[0], "everyone", "colleagues"]
            args = [random.choice(name_options)]
            
            # Other placeholders with relevant business terms
            business_terms = [
                "financial", "trading", "regulatory", "accounting", "market", "operational",
                "strategic", "investment", "partnership", "liquidity", "credit", "risk",
                "California", "energy", "power", "gas", "international", "domestic"
            ]
            
            performance_terms = [
                "better", "worse", "stronger", "weaker", "more volatile", "more stable",
                "improving", "deteriorating", "concerning", "promising", "challenging"
            ]
            
            action_terms = [
                "review our strategy", "adjust our positions", "increase our exposure",
                "reduce our risk", "accelerate our timeline", "delay implementation",
                "contact legal", "inform the board", "prepare documentation", "revise forecasts"
            ]
            
            document_terms = [
                "report", "analysis", "forecast", "model", "presentation", "documentation",
                "filing", "contract", "agreement", "proposal", "strategy document"
            ]
            
            amount_terms = ["10", "25", "50", "100", "250", "500"]
            
            time_terms = ["two", "three", "four", "several", "many", "few"]
            
            # Fill remaining placeholders
            for j in range(1, placeholders):
                if "situation" in template or "issue" in template and j == 1:
                    args.append(random.choice(business_terms))
                elif "performance" in template and j == 1:
                    args.append(random.choice(business_terms))
                elif "than expected" in template and j == 2:
                    args.append(random.choice(performance_terms))
                elif "need to" in template and j == 3:
                    args.append(random.choice(action_terms))
                elif "attached" in template and j == 4:
                    args.append(random.choice(document_terms))
                elif "seeing" in template and j == 3:
                    args.append(random.choice(performance_terms) + " trends")
                elif "opportunity" in template and j == 4:
                    args.append(random.choice(action_terms))
                elif "$" in template and j == 5:
                    args.append(random.choice(amount_terms))
                elif "weeks" in template and j == 3:
                    args.append(random.choice(time_terms))
                elif "results show" in template and j == 4:
                    args.append(random.choice(performance_terms) + " results")
                elif "consider" in template and j == 5:
                    args.append(random.choice(action_terms))
                else:
                    args.append(random.choice(business_terms))
            
            # Last placeholder is usually the sender name
            sender_name = sender.split("@")[0].replace(".", " ").title()
            args.append(sender_name)
            
            content = template.format(*args)
        else:
            content = template
        
        # Add to the list of emails
        email_row = f"{i},{date_str},{sender},{recipient},{subject},\"{content}\""
        additional_emails.append(email_row)
    
    # Add more specific Enron-related emails for better RAG quality
    specific_emails = [
        '105,2001-10-12,kenneth.lay@enron.com,all.employees@enron.com,Third Quarter Results,"Enron Employees,\n\nI regret to inform you that our third-quarter results will show significant losses due to one-time write-downs. Despite these accounting adjustments, our core wholesale and retail energy businesses remain strong and profitable. We continue to see excellent growth opportunities in these areas.\n\nWe will hold an all-employee meeting next week to address your questions and concerns.\n\nKen Lay\nChairman and CEO"',
        
        '106,2001-08-22,sherron.watkins@enron.com,kenneth.lay@enron.com,Follow-up on Accounting Concerns,"Mr. Lay,\n\nThank you for meeting with me last week regarding my concerns about Enron\'s accounting practices. As I mentioned, I am particularly troubled by our Raptor transactions with LJM2. These entities were created to hedge market risk in our investments, but they appear to be supported primarily by Enron\'s own stock.\n\nIf our stock price falls significantly, these hedges will collapse, potentially leading to write-downs of hundreds of millions of dollars.\n\nI strongly recommend engaging independent accounting and legal experts to review these structures.\n\nRespectfully,\nSherron Watkins"',
        
        '107,2001-05-18,tim.belden@enron.com,john.lavorato@enron.com,California Trading Strategies,"John,\n\nOur strategies in California continue to be extremely profitable. By scheduling power flows that create congestion, then relieving that congestion, we\'ve been able to extract significant profits from the market design flaws.\n\nThe \"Death Star\" strategy alone generated $30M last month. \"Fat Boy\" and \"Get Shorty\" are also working well.\n\nRegulators are starting to ask questions, but our legal team believes our strategies comply with market rules as written.\n\nTim"',
        
        '108,2001-09-25,andrew.fastow@enron.com,ben.glisan@enron.com,LJM and Raptor Restructuring,"Ben,\n\nWe need to restructure the Raptor vehicles immediately. With Enron\'s stock price declining, they\'re underwater by approximately $250M. If we don\'t address this before quarter-end, we\'ll have to recognize substantial losses.\n\nI\'ve discussed a potential solution with Rick Causey and Jeff Skilling before he left. We can restructure using additional Enron shares to shore up the vehicles.\n\nLet\'s meet tomorrow to finalize the approach.\n\nAndy"',
        
        '109,2001-10-24,jeffrey.mcmahon@enron.com,kenneth.lay@enron.com,Urgent Liquidity Concerns,"Ken,\n\nOur liquidity situation has deteriorated rapidly. We currently have approximately $1.2B in available cash, but we face potential collateral calls of over $3B if our credit rating is downgraded to below investment grade.\n\nWe need to take immediate action to raise cash, including accelerating asset sales and reducing trading positions. I also recommend drawing down our available credit lines immediately before banks become unwilling to lend.\n\nThis is an extremely serious situation that threatens the company\'s survival.\n\nJeff McMahon\nTreasurer"',
        
        '110,2000-08-23,rebecca.mark@enron.com,kenneth.lay@enron.com,Dabhol Project Risks,"Ken,\n\nThe situation with our Dabhol power project in India continues to deteriorate. The Maharashtra State Electricity Board has stopped paying for power, claiming the prices are too high. Our outstanding receivables now exceed $240M.\n\nPolitical opposition to the project has intensified, and we face significant challenges in enforcing our contract rights. The total Enron investment at risk is approximately $900M.\n\nWe should consider strategic alternatives, including potential divestiture.\n\nRebecca"',
        
        '111,2001-10-16,richard.causey@enron.com,executive.committee@enron.com,Accounting Review Results,"Executive Committee,\n\nAndersen has completed their review of our structured transactions, including the Raptor vehicles. While they have signed off on our accounting treatment, they expressed concerns about the economic substance of some transactions and the complexity of our structures.\n\nThey specifically noted that several of our FAS 140 transactions appear to be motivated primarily by accounting results rather than business purposes.\n\nWe should discuss these findings at our next meeting.\n\nRick Causey\nChief Accounting Officer"',
        
        '112,2001-11-01,greg.whalley@enron.com,trading.team@enron.com,Trading Position Limits,"All Traders,\n\nEffective immediately, we are implementing strict position limits across all trading desks. No new positions should be added, and existing positions should be reduced where possible without incurring significant losses.\n\nAll collateral calls must be reported immediately to the treasury team. Any requests from counterparties for additional security must be escalated to senior management.\n\nThese measures are temporary but necessary given current market conditions.\n\nGreg Whalley\nPresident and COO"',
        
        '113,2001-07-13,kenneth.lay@enron.com,jeff.skilling@enron.com,CEO Transition,"Jeff,\n\nThe board has approved your appointment as CEO effective August 1. I will remain as Chairman and will continue to be involved in strategic decisions and government relations.\n\nYou\'ve done an outstanding job as President and COO, and the company is well-positioned for continued growth under your leadership. Your vision for our wholesale and retail energy businesses has transformed Enron into a market leader.\n\nCongratulations on this well-deserved promotion.\n\nKen"',
        
        '114,2001-03-26,vince.kaminski@enron.com,rick.buy@enron.com,Risk Assessment of Raptor Structures,"Rick,\n\nMy team has completed a risk assessment of the Raptor structured finance vehicles. We have serious concerns about these structures from a risk management perspective.\n\nThe fundamental problem is that these vehicles are essentially hedging Enron investments with Enron stock. This creates a circular reference that could collapse if Enron\'s stock price declines significantly.\n\nI recommend unwinding these structures in an orderly fashion before they create larger problems.\n\nVince Kaminski\nManaging Director, Research"',
        
        '115,2001-08-14,jeff.skilling@enron.com,kenneth.lay@enron.com,Resignation,"Ken,\n\nAs we discussed, I am resigning as CEO and from the Board of Directors effective immediately. My reasons are entirely personal, as I need to devote more time to my family.\n\nI believe Enron is in excellent shape strategically and financially, with the strongest management team in the industry. The company is positioned for continued growth and success.\n\nIt has been a privilege to help build this great company over the past decade.\n\nJeff"'
    ]
    
    # Combine all emails
    all_emails = sample_data + "\n" + "\n".join(additional_emails) + "\n" + "\n".join(specific_emails)
    
    # Write to file
    try:
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(all_emails)
        print_step(f"Created sample dataset with 115 emails at {sample_file}", "OK")
        return sample_file
    except Exception as e:
        print_step(f"Error creating sample file: {str(e)}", "ERROR")
        return None

def main():
    """Main function to copy and process the dataset."""
    force_rebuild = "--force" in sys.argv or "-f" in sys.argv
    export_db = "--export" in sys.argv or "-e" in sys.argv
    use_sample = "--sample" in sys.argv or "-s" in sys.argv
    
    print("\n" + "="*60)
    print(" "*15 + "ENRON EMAIL DATASET PROCESSOR")
    print("="*60 + "\n")
    
    setup_directories()
    
    # If sample flag is set, create sample data directly
    if use_sample:
        print_step("Creating sample data as requested", "INFO")
        csv_path = create_sample_data()
        if not csv_path:
            print_step("Fatal error: Could not create sample dataset", "ERROR")
            return 1
    else:
        # Find and copy the dataset
        source_path = find_source_data()
        if source_path:
            try:
                csv_path = copy_dataset(source_path)
            except Exception as e:
                print_step(f"Unable to access dataset at {source_path}: {str(e)}", "ERROR")
                print_step("Creating sample data instead.", "WARN")
                csv_path = create_sample_data()
        else:
            print_step("No source dataset found. Creating sample data instead.", "WARN")
            csv_path = create_sample_data()
        
        if not csv_path:
            print_step("Fatal error: Could not find or create dataset", "ERROR")
            return 1
    
    # If force rebuild is requested, remove existing vector database
    if force_rebuild and os.path.exists(CHROMA_DIR):
        print_step("Force rebuild requested - removing existing vector database", "INFO")
        try:
            shutil.rmtree(CHROMA_DIR)
            print_step("Existing vector database removed", "OK")
        except Exception as e:
            print_step(f"Error removing existing database: {str(e)}", "ERROR")
    
    # Process the dataset for RAG
    print("\n" + "="*60)
    print(" "*15 + "VECTOR DATABASE GENERATION")
    print("="*60 + "\n")
    
    success = process_dataset(csv_path)
    
    # Export the vector database if requested
    if export_db and success and os.path.exists(CHROMA_DIR):
        print_step("Exporting vector database to pre_built_db.zip", "WAIT")
        try:
            import zipfile
            with zipfile.ZipFile('pre_built_db.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(VECTOR_DB_DIR):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, os.path.dirname(VECTOR_DB_DIR)))
            
            db_size = os.path.getsize('pre_built_db.zip') / (1024 * 1024)
            print_step(f"Vector database exported to pre_built_db.zip ({db_size:.2f} MB)", "OK")
        except Exception as e:
            print_step(f"Error exporting vector database: {str(e)}", "ERROR")
    
    if success:
        print("\n" + "="*60)
        print_step("Dataset processed successfully!", "OK")
        print_step("The ENRON MAIL TERMINAL is ready for use with full archive access", "OK")
        print("="*60 + "\n")
        return 0
    else:
        print("\n" + "="*60)
        print_step("Dataset processing encountered errors", "WARN")
        print_step("The terminal will still work in LIMITED ACCESS MODE", "INFO")
        print("="*60 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 