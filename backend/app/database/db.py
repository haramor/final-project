"""
Database connector for the vector database.

This module provides an abstraction over different vector database implementations.
Sarah should update this file based on the chosen database solution.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import logging
from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import pandas as pd 
import json

# Import config
from app.database.config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIRECTORY,
    SCHEMA_VERSION
)
from app.database.schema import validate_metadata, COLLECTION_SCHEMAS, ArticleMetadata

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    A unified interface for interacting with different vector databases.
    """
    
    def __init__(self):
        """
        Initialize the database connector based on the configured database type.
        """
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.collections = {}
        self._init_collections()

        logger.info(f"Initialized Chroma vector database")
    
    def _init_collections(self):
        """Initialize collections based on schema definitions"""
        try:
            os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)

            for collection_name, schema_class in COLLECTION_SCHEMAS.items():
                self.collections[collection_name] = Chroma(
                    persist_directory=CHROMA_PERSIST_DIRECTORY,
                    embedding_function=self.embedding_function,
                    collection_name=collection_name,
                    collection_metadata={
                        "description": schema_class.__doc__,
                        "schema_version": SCHEMA_VERSION
                    }
                )
                logger.info(f"Initialized collection: {collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize collections: {str(e)}")
            raise

    def add_to_collection(self, collection_name: str, documents: Union[Document, List[Document]]):
        """Add documents to a specific collection with schema validation"""
        if collection_name not in self.collections:
            raise ValueError(f"Unknown collection: {collection_name}")
        
        collection = self.collections[collection_name]
        try:
            if isinstance(documents, Document):
                documents = [documents]
            
            validated_documents = []
            for doc in documents:
                # Validate and sanitize metadata
                validated_metadata = validate_metadata(doc.metadata, collection_name)
                sanitized_metadata = {
                    key: (value if isinstance(value, (str, int, float, bool)) else str(value) if value is not None else "N/A")
                    for key, value in validated_metadata.items()
                }
                validated_doc = Document(
                    page_content=doc.page_content,
                    metadata=sanitized_metadata
                )
                validated_documents.append(validated_doc)
            
            # Batch processing
            batch_size = 1000  # Define a batch size
            total_added = 0
            for i in range(0, len(validated_documents), batch_size):
                batch = validated_documents[i:i + batch_size]
                texts = [doc.page_content for doc in batch]
                metadatas = [doc.metadata for doc in batch]
                
                collection.add_texts(
                    texts=texts,
                    metadatas=metadatas
                )
                total_added += len(texts)
                logger.info(f"Added batch {i // batch_size + 1} to {collection_name} ({len(texts)} documents)")
            
            logger.info(f"Successfully added {total_added} documents in total to {collection_name}")

        except Exception as e:
            logger.error(f"Failed to add documents to {collection_name}: {str(e)}")
            raise

    def preprocess_json_for_rag(self, json_file_path: str):
        try:
            # Load JSON data
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            # Extract abstracts and clean text
            documents = [entry.get('abstract', '').strip() for entry in data if 'abstract' in entry]
            cleaned_documents = [doc.replace('\n', ' ').replace('\r', '') for doc in documents]
            
            # Prepare data for RAG
            df = pd.DataFrame({'document': cleaned_documents})
            
            # Add metadata (if available)
            df['metadata'] = [
                {
                    'title': entry.get('title', ''),
                    'journal': entry.get('journal', ''),
                    'year': entry.get('year', ''),
                    'source': entry.get('title')  # Add source field
                }
                for entry in data if 'abstract' in entry
            ]
            
            return df
        except Exception as e:
            print(f"An error occurred while processing JSON for RAG: {e}")
            return None


    def preprocess_and_add_json(self, json_file_path: str, collection_name: str):
        """Preprocess JSON data and add it to the vector database"""
        try:
            # Preprocess JSON data
            df = self.preprocess_json_for_rag(json_file_path)
            if df is None:
                raise ValueError("Failed to preprocess JSON data.")
            
            # Generate embeddings
            embeddings = generate_embeddings(df['document'].tolist(), model_name=EMBEDDING_MODEL_NAME)
            if embeddings is None:
                raise ValueError("Failed to generate embeddings.")
            
            # Create Document objects
            documents = [
                Document(
                    page_content=document,
                    metadata=metadata
                )
                for document, metadata in zip(df['document'], df['metadata'])
            ]
            
            # Add to collection
            self.add_to_collection(collection_name, documents)
            logger.info(f"Successfully added preprocessed JSON data to collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to preprocess and add JSON data: {str(e)}")
            raise

    def search_collection(
        self, 
        collection_name: str, 
        query: str, 
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Search within a specific collection"""
        if collection_name not in self.collections:
            raise ValueError(f"Unknown collection: {collection_name}")
        
        try:
            if metadata_filter:
                valid_fields = COLLECTION_SCHEMAS[collection_name].model_fields.keys()
                invalid_fields = [f for f in metadata_filter if f not in valid_fields]
                
                if invalid_fields:
                    raise ValueError(f"Invalid filter fields: {invalid_fields}")
                
                kwargs['filter'] = metadata_filter
            
            results = self.collections[collection_name].similarity_search(query, **kwargs)
            logger.info(f"Search in collection '{collection_name}' returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error during search in collection '{collection_name}': {str(e)}")
            raise

    def search_all(self, query: str, **kwargs):
        """Search across all collections"""
        results = []
        for name, collection in self.collections.items():
            collection_results = collection.similarity_search(query, **kwargs)
            results.extend(collection_results)
        return results

    def get_sources(docs):
        """Helper function to extract unique source titles from retrieved documents."""
        return list(set(doc.metadata.get('title', 'Unknown Title') for doc in docs))

def generate_embeddings(documents: List[str], model_name: str) -> List[List[float]]:
    """
    Generate embeddings for a list of documents using the HuggingFaceEmbeddings model.

    Args:
        documents: A list of text documents to embed.
        model_name: The name of the embedding model.

    Returns:
        A list of embeddings, where each embedding is a list of floats.
    """
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        embeddings = embedding_model.embed_documents(documents)
        return embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise

# Singleton instance for the connector
_db_instance = None

def get_db() -> VectorDatabase:
    """
    Get or create the database connector instance.
    
    Returns:
        The database connector
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = VectorDatabase()
        
    return _db_instance

# Example usage:
if __name__ == "__main__":
    db = get_db()
    
    # Preprocess and add JSON data to the database
    # Ensure this part is not commented out if you need to reload data
    # print("Starting data preprocessing and loading...")
    # db.preprocess_and_add_json(
    #     "../../../pubmed_contraception_abstracts2.json",  # Adjusted path
    #     "research_papers"
    # )
    # print("Data preprocessing and loading finished.")

    print("\n=== Test Search 1: WITHOUT metadata filter ===")
    test_query_1 = "What are IUDs?"
    results_1 = db.search_collection(
        "research_papers",
        test_query_1
    )
    print(f"Query: '{test_query_1}'")
    print(f"Search results found (without filter): {len(results_1)}")
    if results_1:
        print(f"First result (without filter): {results_1[0].page_content[:200]}... Metadata: {results_1[0].metadata}")
    else:
        print("No results found for the test query without filter.")

    print("\n=== Test Search 2: WITH metadata filter ===")
    test_query_2 = "What are IUDs?"
    journal_filter = {"journal": "American Journal of Obstetrics and Gynecology"}
    results_2 = db.search_collection(
        "research_papers",
        test_query_2,
        metadata_filter=journal_filter
    )
    print(f"Query: '{test_query_2}' with filter: {journal_filter}")
    print(f"Search results found (with filter): {len(results_2)}")
    if results_2:
        print(f"First result (with filter): {results_2[0].page_content[:200]}... Metadata: {results_2[0].metadata}")
    else:
        print("No results found for the test query with the specified journal filter.")

    # Optional: Check total documents in collection if possible (Chroma specific)
    try:
        collection = db.collections.get("research_papers")
        if collection:
            print(f"\nTotal documents in 'research_papers' collection: {collection.count()}")
    except Exception as e:
        print(f"Could not get collection count: {e}")