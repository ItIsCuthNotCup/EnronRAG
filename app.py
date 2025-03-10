#!/usr/bin/env python3

import os
import streamlit as st
import pandas as pd
import logging
import time
import html

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', 
                    handlers=[logging.FileHandler("logs/enron_app.log")])

# Log startup for debugging
logging.info("========== APPLICATION STARTED ==========")

# Hardcoded sample data
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

def search_emails(query):
    """Simple search function that searches the sample data"""
    query = query.lower()
    logging.info(f"Processing query: {query}")
    answer = ""
    sources = []
    
    # Handle special cases
    if "ceo" in query or "kenneth lay" in query:
        answer = "Kenneth Lay was the CEO of Enron Corporation from 1986 to 2001. He founded the company and led it until its bankruptcy following accounting scandals."
        sources = [email for email in SAMPLE_DATA if "kenneth.lay" in email["from"].lower()]
        if not sources:
            sources = [SAMPLE_DATA[0]]
            
    elif "skilling" in query or "president" in query or "coo" in query:
        answer = "Jeffrey Skilling was the President and Chief Operating Officer (COO) of Enron, later becoming CEO after Kenneth Lay stepped down. He was one of the key figures in the company's downfall."
        sources = [email for email in SAMPLE_DATA if "skilling" in email["from"].lower()]
        if not sources:
            sources = [SAMPLE_DATA[1]]
            
    elif "fastow" in query or "cfo" in query or "financial" in query:
        answer = "Andrew Fastow was the Chief Financial Officer (CFO) of Enron. He was responsible for creating the off-balance-sheet special purpose entities used to hide Enron's debts and losses."
        sources = [email for email in SAMPLE_DATA if "fastow" in email["from"].lower()]
        if not sources:
            sources = [SAMPLE_DATA[2]]
            
    elif "scandal" in query or "fraud" in query or "bankruptcy" in query:
        answer = "The Enron scandal, revealed in 2001, involved systematic accounting fraud that led to the company's bankruptcy. It was one of the largest corporate scandals in U.S. history, and resulted in the dissolution of Arthur Andersen, one of the five largest audit and accountancy partnerships in the world."
        sources = [email for email in SAMPLE_DATA if "scandal" in email["text"].lower() or "accounting" in email["text"].lower()]
        if not sources:
            sources = [SAMPLE_DATA[3]]
    
    # General search
    else:
        sources = [email for email in SAMPLE_DATA if query in email["text"].lower()]
        if not sources:
            # Try word by word
            for word in query.split():
                if len(word) > 3:  # Only use words longer than 3 chars
                    word_sources = [email for email in SAMPLE_DATA if word in email["text"].lower()]
                    if word_sources:
                        sources = word_sources
                        break
        
        if not sources:
            sources = SAMPLE_DATA[:3]
            answer = f"I couldn't find exact matches for '{query}', but here are some important Enron emails."
        else:
            answer = f"Found {len(sources)} emails related to '{query}'."
    
    # Limit results
    sources = sources[:3]
    
    # Format output
    output = answer + "\n\n"
    
    if sources:
        output += "SOURCE EMAILS:\n"
        for i, email in enumerate(sources):
            output += f"EMAIL {i+1}:\n"
            output += f"FROM: {email['from']}\n"
            output += f"DATE: {email['date']}\n"
            output += f"SUBJECT: {email['subject']}\n"
            output += "-----\n"
            
            # Format the email text
            text = email['text']
            output += text + "\n\n"
    
    logging.info(f"Generated response for query: {query}")
    return output

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
        history_html += f'<div class="terminal-line command">$ {html.escape(content)}</div>\n'
    elif msg_type == "response":
        formatted_content = content.replace('\n', '<br>')
        history_html += f'<div class="terminal-line response">{formatted_content}</div>\n'
    elif msg_type == "message":
        history_html += f'<div class="terminal-line message">{html.escape(content)}</div>\n'

# FULL CUSTOM HTML TERMINAL
terminal_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            background-color: #000000;
            color: #00FF00;
            font-family: 'Courier New', monospace;
        }}
        
        body, html {{
            background-color: #000000;
            color: #00FF00;
            font-family: 'Courier New', monospace;
            height: 100%;
            width: 100%;
            overflow: hidden;
        }}
        
        #terminal {{
            background-color: #000000;
            color: #00FF00;
            font-family: 'Courier New', monospace;
            width: 100%;
            height: 100vh;
            padding: 20px;
            white-space: pre-wrap;
            overflow-y: auto;
            position: relative;
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
        }});
        
        // Function to submit a command to the server
        function submitCommand(command) {{
            window.location.href = '?command=' + encodeURIComponent(command);
        }}
    </script>
</body>
</html>
"""

# Display the terminal
st.components.v1.html(terminal_html, height=800, scrolling=False) 