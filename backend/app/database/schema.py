"""
Database schema definitions.

This module defines the schema for the vector database collections.
Sarah should update this based on the chosen database solution.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class ResearchPaperMetadata(BaseModel):
    """Metadata specific to medical journal articles."""
    type: str = "research_paper"
    title: str = Field(..., description="Title of the article")
    url: str = Field(None, description="URL of the article")
    journal: Optional[str] = Field(None, description="Journal name")
    year: Optional[str] = Field(None, description="Publication date (YYYY-MM-DD)")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")

class ArticleMetadata(BaseModel):
    """Metadata specific to medical journal articles."""
    type: str = "article"
    title: str = Field(..., description="Title of the article")
    url: str = Field(None, description="URL of the article")
    year: Optional[str] = Field(None, description="Publication date (YYYY-MM-DD)")



# Map of collection names to their respective schema classes
COLLECTION_SCHEMAS = {
    "articles": ArticleMetadata,
    "research_papers": ResearchPaperMetadata
}


def validate_metadata(metadata: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
    """
    Validate and normalize metadata for a specific collection.
    
    Args:
        metadata: The metadata to validate
        collection_name: The name of the collection
        
    Returns:
        Validated and normalized metadata
        
    Raises:
        ValueError: If the metadata is invalid for the collection
    """
    # Get the schema class for the collection
    schema_class = COLLECTION_SCHEMAS.get(collection_name)

    # Validate and normalize
    try:
        validated = schema_class(**metadata)
        return validated.dict()
    except Exception as e:
        raise ValueError(f"Invalid metadata for collection '{collection_name}': {str(e)}")



# Chroma doesn't need explicit schema, but we can define collections
CHROMA_COLLECTIONS = list(COLLECTION_SCHEMAS.keys())