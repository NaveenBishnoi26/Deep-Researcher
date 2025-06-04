"""
Configuration module for the research pipeline.
"""

import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
EXPORTS_DIR = BASE_DIR / "exports"

# Create directories if they don't exist
for directory in [LOGS_DIR, VECTORSTORE_DIR, EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API keys and settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
EXA_API_KEY = os.getenv('EXA_API_KEY')

# Model settings
BASE_MODEL = "gpt-4o-mini"  # Using GPT-4o-mini for all LLM calls
EMBEDDING_MODEL = "text-embedding-3-small"  # Smallest embedding model
MODEL_TEMPERATURE = 0.7  # Default temperature for model responses

# Vector store settings
VECTORSTORE_CONFIGS = {
    "collection_name": "research_documents",
    "auto_persist": False
}

# PDF Export settings
PDF_OPTIONS = {
    "page-size": "A4",
    "margin-top": "20mm",
    "margin-right": "20mm",
    "margin-bottom": "20mm",
    "margin-left": "20mm",
    "encoding": "UTF-8",
    "no-outline": None
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "research.log"),
        logging.StreamHandler()
    ]
) 