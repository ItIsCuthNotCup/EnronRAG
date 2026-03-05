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

# Hardcoded sample data used by search_emails() in simple (non-RAG) mode.
# These five emails represent key figures and events in the Enron scandal and
# allow the app to answer common queries without any external dependencies.
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

def _load_email_corpus():
    """Load emails from the best available source on disk, falling back to SAMPLE_DATA.

    Checks for a CSV in this order:
      1. data/enron_emails.csv  (built by: python process_enron_data.py --sample)
      2. emails.csv             (the full Kaggle Enron dataset in the project root)

    Returns:
        list[dict]: List of email dicts with keys from, to, subject, date, text.
    """
    csv_candidates = ["data/enron_emails.csv", "emails.csv"]
    content_col_candidates = ["content", "message", "body", "text"]

    for path in csv_candidates:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
            content_col = next((c for c in content_col_candidates if c in df.columns), None)
            if content_col is None:
                logging.warning(f"{path}: no recognised content column, skipping")
                continue
            emails = []
            for _, row in df.iterrows():
                emails.append({
                    "from":    str(row.get("from",    "")),
                    "to":      str(row.get("to",      "")),
                    "subject": str(row.get("subject", "")),
                    "date":    str(row.get("date",    "")),
                    "text":    str(row.get(content_col, "")),
                })
            logging.info(f"Loaded {len(emails)} emails from {path}")
            return emails
        except Exception as e:
            logging.warning(f"Could not load {path}: {e}")

    logging.info("No CSV found — using built-in 5-email sample")
    return SAMPLE_DATA


# Load the email corpus once at startup so every search call can use it.
_EMAIL_CORPUS = _load_email_corpus()


def search_emails(query):
    """Search the email corpus by keyword relevance and return the top results.

    Scores each email by counting how many query words appear in its subject
    and body, then returns the top 3 matches with their actual text.

    Args:
        query (str): The user's natural-language question or keyword string.

    Returns:
        str: Formatted terminal output showing matching emails.
    """
    logging.info(f"Processing query: {query}")
    query_words = [w for w in query.lower().split() if len(w) > 2]

    # Score every email by total query-word hits across subject + body + sender
    scored = []
    for email in _EMAIL_CORPUS:
        haystack = " ".join([
            email.get("subject", ""),
            email.get("text", ""),
            email.get("from", ""),
        ]).lower()
        score = sum(haystack.count(w) for w in query_words)
        if score > 0:
            scored.append((score, email))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [e for _, e in scored[:3]]

    if not top:
        output = f"No emails matched '{query}' in the archive. Showing recent emails:\n\n"
        top = _EMAIL_CORPUS[:3]
    else:
        output = f"Found {len(scored)} matching email(s). Top results:\n\n"

    for i, email in enumerate(top):
        output += f"EMAIL {i + 1}:\n"
        output += f"FROM:    {email.get('from', 'Unknown')}\n"
        output += f"TO:      {email.get('to', 'Unknown')}\n"
        output += f"DATE:    {email.get('date', 'Unknown')}\n"
        output += f"SUBJECT: {email.get('subject', '(no subject)')}\n"
        output += "-" * 40 + "\n"
        text = email.get("text", "")
        if len(text) > 800:
            text = text[:800] + "\n...[truncated]"
        output += text + "\n\n"

    logging.info(f"Returned {len(top)} results for query: {query}")
    return output

# Escape HTML for safe display in the terminal
def escape_html_except_br(text):
    """Escape HTML special characters in text, then convert newlines to <br> tags.

    This prevents XSS when user-supplied content or email bodies are injected
    into the terminal's innerHTML. Newlines are preserved as visible line breaks.

    Args:
        text (str): Raw text that may contain HTML special characters.

    Returns:
        str: HTML-safe text with newlines replaced by <br>.
    """
    escaped = html.escape(text).replace('\n', '<br>')
    return escaped

# Initialize session state
# terminal_history: list of (type, content) tuples representing every line
#   displayed in the terminal. type is one of "command", "response", "message".
#   Persists across Streamlit rerenders so the full session is visible.
if 'terminal_history' not in st.session_state:
    st.session_state.terminal_history = []

# startup_done: guards the one-time ASCII banner / boot sequence so it is
#   only appended to terminal_history on the very first render.
if 'startup_done' not in st.session_state:
    st.session_state.startup_done = False

# command_counter: incremented after each command is processed; causes
#   Streamlit to re-render the component and display the updated history.
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
        formatted_content = escape_html_except_br(content)
        history_html += f'<div class="terminal-line response">{formatted_content}</div>\n'
    elif msg_type == "message":
        history_html += f'<div class="terminal-line message">{html.escape(content)}</div>\n'

# ---------------------------------------------------------------------------
# HTML/CSS/JS TERMINAL
#
# The entire UI is a self-contained HTML document rendered inside an iframe
# via st.components.v1.html().  Streamlit itself is hidden with CSS overrides
# above; only this component is visible.
#
# Architecture:
#   - Python builds `history_html` (a string of <div> elements) from
#     st.session_state.terminal_history and embeds it as an f-string below.
#   - The <input> element captures keystrokes.  On Enter, submitCommand()
#     redirects the parent page to ?command=<value>, which triggers a full
#     Streamlit rerender with the new command in st.query_params.
#   - The scanlines <div> is a purely cosmetic CSS overlay that mimics the
#     horizontal scan lines of a CRT monitor.
# ---------------------------------------------------------------------------
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