@echo off
color 0A
title ENRON MAIL TERMINAL v1.2.1
cls

echo.
echo  _______ .__   __. .______       ______   .__   __.      .___________. _______ .______      .___  ___.  __  .__   __.      ___       __      
echo ^|   ____^|^|  \ ^|  ^| ^|   _  \     /  __  \  ^|  \ ^|  ^|      ^|           ^|^|   ____^|^|   _  \     ^|   \/   ^| ^|  ^| ^|  \ ^|  ^|     /   \     ^|  ^|     
echo ^|  ^|__   ^|   \^|  ^| ^|  ^|_)  ^|   ^|  ^|  ^|  ^| ^|   \^|  ^|      `---^|  ^|----`^|  ^|__   ^|  ^|_)  ^|    ^|  \  /  ^| ^|  ^| ^|   \^|  ^|    /  ^  \    ^|  ^|     
echo ^|   __^|  ^|  . `  ^| ^|      /    ^|  ^|  ^|  ^| ^|  . `  ^|          ^|  ^|     ^|   __^|  ^|      /     ^|  ^|\/^|  ^| ^|  ^| ^|  . `  ^|   /  /_\  \   ^|  ^|     
echo ^|  ^|____ ^|  ^|\   ^| ^|  ^|\  \----^|  `--'  ^| ^|  ^|\   ^|          ^|  ^|     ^|  ^|____ ^|  ^|\  \----.^|  ^|  ^|  ^| ^|  ^| ^|  ^|\   ^|  /  _____  \  ^|  `----.
echo ^|_______^|^|__^| \__^| ^| _^| `._____^|\______/  ^|__^| \__^|          ^|__^|     ^|_______^|^| _^| `.____^|^|__^|  ^|__^| ^|__^| ^|__^| \__^| /__/     \__\ ^|_______^|
echo.
echo ============================================
echo       ENRON EMAIL TERMINAL SYSTEM
echo          CONFIDENTIAL - 2001
echo ============================================
echo.

:: Create required directories
mkdir logs 2>nul
mkdir data 2>nul
mkdir vector_db 2>nul

:: Install core dependencies
echo [....] Installing core dependencies...
python -m pip install -q streamlit pandas requests
echo [OK] Core dependencies installed

:: Try to install optional RAG dependencies (won't fail if they don't install)
echo [....] Installing RAG dependencies...
python -m pip install -q langchain sentence-transformers chromadb --quiet
echo [OK] RAG dependencies installation attempted

:: Check for Ollama
echo [....] Checking for Ollama...
python -c "import requests; exit(0 if requests.get('http://localhost:11434/api/tags', timeout=2).status_code == 200 else 1)" 2>nul
if %ERRORLEVEL% == 0 (
    echo [OK] Ollama is running
    set USE_VECTOR_DB=True
) else (
    echo [INFO] Ollama not detected. Will use CSV search.
    set USE_VECTOR_DB=False
)

:: Create config file
echo [....] Creating configuration file...
echo # Enron Email RAG Configuration > config.py
echo CSV_FILE = "emails.csv" >> config.py
echo TOP_K_RESULTS = 5 >> config.py
echo VECTOR_DB_DIR = "vector_db" >> config.py
echo CHROMA_DIR = "vector_db/chroma" >> config.py
echo OLLAMA_BASE_URL = "http://localhost:11434" >> config.py
echo MODEL_NAME = "llama3:8b" >> config.py
if "%USE_VECTOR_DB%"=="True" (
    echo USE_CSV_SEARCH = False >> config.py
) else (
    echo USE_CSV_SEARCH = True >> config.py
)
echo [OK] Created configuration file

:: Check for pre-built vector DB
if "%USE_VECTOR_DB%"=="True" (
    echo [....] Checking for pre-built vector database...
    if exist "pre_built_db.zip" (
        if not exist "vector_db\chroma" (
            echo [....] Extracting pre-built vector database...
            python -c "import zipfile, os; os.makedirs('vector_db', exist_ok=True); zipfile.ZipFile('pre_built_db.zip').extractall('vector_db')"
            echo [OK] Extracted pre-built vector database
        ) else (
            echo [OK] Vector database already exists
        )
    )
)

:: Create sample data if needed
if not exist "emails.csv" (
    echo [....] Creating sample email data...
    echo from,to,subject,date,message_id,text > emails.csv
    echo "kenneth.lay@enron.com","all.employees@enron.com","Welcome Message","2001-05-01","KL001","Welcome to Enron! I am Kenneth Lay, the CEO of Enron Corporation. I'm excited to have you join our team. As we continue to grow and innovate in the energy sector, each one of you plays a vital role in our success. Our company values integrity, respect, communication, and excellence in all we do." >> emails.csv
    echo "jeff.skilling@enron.com","board@enron.com","Quarterly Results","2001-06-15","JS001","I'm pleased to report that our quarterly results exceeded expectations. Revenue is up 20% year-over-year, and our stock price continues to perform well. The energy trading division has been particularly successful, showing the strength of our market-based approach to energy solutions." >> emails.csv
    echo "andrew.fastow@enron.com","finance@enron.com","Financial Strategies","2001-07-30","AF001","The new financial strategies we've implemented are working well. Our special purpose entities are hiding the debt effectively while maintaining our credit rating. The Raptor vehicles in particular have been successful in hedging our investments in technology companies." >> emails.csv
    echo "sherron.watkins@enron.com","kenneth.lay@enron.com","Accounting Irregularities","2001-08-15","SW001","I am incredibly nervous that we will implode in a wave of accounting scandals. I have been thinking about our accounting practices a lot recently. The aggressive accounting we've used in the Raptor vehicles and other SPEs is concerning. I am worried that we have become a house of cards." >> emails.csv
    echo "richard.kinder@enron.com","executive.team@enron.com","Business Strategy","2000-12-01","RK001","We need to focus on our core business and maintain strong relationships with our partners. Our expansion strategy must be carefully considered. I believe in building businesses with hard assets rather than just trading operations. Long-term success requires solid infrastructure." >> emails.csv
    echo [OK] Created sample email database
)

:: Start the application
echo.
echo ============================================
echo      INITIALIZING TERMINAL INTERFACE
echo ============================================
echo.

echo [....] Starting ENRON MAIL TERMINAL...
echo [INFO] Terminal will open in your browser momentarily...

:: Use start command to run in the background
start python -m streamlit run app.py

echo.
echo [OK] ENRON MAIL TERMINAL has been launched in your browser
echo [INFO] If the browser didn't open automatically, visit: http://localhost:8501
echo.
echo Press any key to close this window. The application will continue running in your browser.
pause > nul 