"""
Configuration settings for the vector database.

This file should be customized based on the chosen database solution.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


DATABASE_TYPE = "chroma"  # Local ChromaDB

# ChromaDB configuration
CHROMA_PERSIST_DIRECTORY = os.environ.get(
    "CHROMA_PERSIST_DIRECTORY", 
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "chroma_db")
)
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "research_papers")


# Embedding model configuration
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # Dimension of the all-MiniLM-L6-v2 model

# Document processing configuration
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Characters overlap between chunks

# Database connection timeouts
CONNECTION_TIMEOUT = 10  # seconds
OPERATION_TIMEOUT = 30  # seconds

# Schema version - increment when making breaking changes
SCHEMA_VERSION = 1 