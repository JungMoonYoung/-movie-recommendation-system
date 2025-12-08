"""
Configuration file for MovieLens Recommendation System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# SQL directories
SQL_DIR = PROJECT_ROOT / "sql"
QUERIES_DIR = SQL_DIR / "queries"
VIEWS_DIR = SQL_DIR / "views"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"

# Database configuration
# Check if running on Streamlit Cloud
try:
    import streamlit as st
    # Try to access secrets, will raise error if secrets.toml doesn't exist
    if hasattr(st, 'secrets'):
        try:
            if 'database' in st.secrets:
                # Use Streamlit secrets for cloud deployment
                DB_CONFIG = {
                    'host': st.secrets['database']['DB_HOST'],
                    'port': int(st.secrets['database']['DB_PORT']),
                    'database': st.secrets['database']['DB_NAME'],
                    'user': st.secrets['database']['DB_USER'],
                    'password': st.secrets['database']['DB_PASSWORD']
                }
            else:
                raise KeyError("database not in secrets")
        except Exception:
            # secrets.toml doesn't exist or is invalid, use .env
            DB_CONFIG = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'database': os.getenv('DB_NAME', 'movielens_db'),
                'user': os.getenv('DB_USER', 'movielens_user'),
                'password': os.getenv('DB_PASSWORD', '')
            }
    else:
        # Use environment variables for local development
        DB_CONFIG = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'movielens_db'),
            'user': os.getenv('DB_USER', 'movielens_user'),
            'password': os.getenv('DB_PASSWORD', '')
        }
except ImportError:
    # Fallback to environment variables if Streamlit is not available
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'movielens_db'),
        'user': os.getenv('DB_USER', 'movielens_user'),
        'password': os.getenv('DB_PASSWORD', '')
    }

# MovieLens 1M Dataset URLs
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

# Recommendation parameters
DEFAULT_TOP_N = 10
MAX_TOP_N = 50
MIN_RATINGS_FOR_POPULARITY = 30
TOP_K_GENRES = 3
MIN_COMMON_USERS_FOR_SIMILARITY = 20

# Rating constraints
MIN_RATING = 0.5
MAX_RATING = 5.0
RATING_STEP = 0.5

# Train/Test split
TEST_RATIO = 0.2

# Performance thresholds (seconds)
TARGET_QUERY_TIME = 3.0
MAX_QUERY_TIME = 5.0
