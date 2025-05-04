"""
Database connector for the vector database.

This module provides an abstraction over different vector database implementations.
Sarah should update this file based on the chosen database solution.
"""

from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import logging
import os

# Import config
from app.database.config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIRECTORY,
    SCHEMA_VERSION
)

# Import schema
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
            # Ensure the persistence directory exists
            os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)

            # Create collections based on schema definitions
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
                # Validate metadata using schema
                validated_metadata = validate_metadata(
                    doc.metadata, 
                    collection_name
                )
                
                # Create new document with validated metadata
                validated_doc = Document(
                    page_content=doc.page_content,
                    metadata=validated_metadata
                )
                validated_documents.append(validated_doc)
            
            # Add validated documents
            texts = [doc.page_content for doc in validated_documents]
            metadatas = [doc.metadata for doc in validated_documents]
            
            collection.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            logger.info(f"Successfully added {len(texts)} documents to {collection_name}")

        except Exception as e:
            logger.error(f"Failed to add documents to {collection_name}: {str(e)}")
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
        
        if metadata_filter:
            # Just validate that the filter fields exist in the schema
            valid_fields = COLLECTION_SCHEMAS[collection_name].model_fields.keys()
            invalid_fields = [f for f in metadata_filter if f not in valid_fields]
            
            if invalid_fields:
                raise ValueError(f"Invalid filter fields: {invalid_fields}")
            
            kwargs['filter'] = metadata_filter
        
        return self.collections[collection_name].similarity_search(query, **kwargs)

    def search_all(self, query: str, **kwargs):
        """Search across all collections"""
        results = []
        for name, collection in self.collections.items():
            collection_results = collection.similarity_search(query, **kwargs)
            results.extend(collection_results)
        return results

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
    
    # Example article document using ArticleMetadata schema
    research_paper = Document(
        page_content="IUDs are a form of long-acting reversible contraception.",
        metadata={
            "doc_type": "article",
            "title": "Overview of Long-Acting Reversible Contraception",
            "url": "https://example.com/article",
            "journal": "American Journal of Obstetrics and Gynecology",
            "year": "2023",  # Schema will validate this
            "doi": "10.1234/ajog.2023.123"
        }
    )
    
    # Add document - metadata will be validated against ArticleMetadata schema
    db.add_to_collection("research_papers", research_paper)
    
    # Search with metadata filter
    results = db.search_collection(
        "research_papers",
        "What are IUDs?",
        metadata_filter={"journal": "American Journal of Obstetrics and Gynecology"}
    )
    print(f"Search results: {results}") 