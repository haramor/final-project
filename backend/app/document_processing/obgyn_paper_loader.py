"""
Script to load papers from ScienceDirect (American Journal of Obstetrics and Gynecology)
and add them to ChromaDB.
"""

import os
import logging
from typing import List, Dict, Any
from datetime import datetime
import requests
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.database.db import get_db
from app.database.schema import ResearchPaperMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScienceDirectLoader:
    """Loader for ScienceDirect papers."""
    
    def __init__(self, api_key: str):
        """
        Initialize the ScienceDirect loader.
        
        Args:
            api_key: ScienceDirect API key
        """
        self.api_key = api_key
        self.base_url = "https://api.elsevier.com/content/search/sciencedirect"
        self.headers = {
            "X-ELS-APIKey": api_key,
            "Accept": "application/json"
        }
        
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for papers on ScienceDirect.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata
        """
        params = {
            "query": query,
            "count": max_results,
            "sort": "date",
            "view": "COMPLETE"
        }
        
        try:
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params
            )
            response.raise_for_status()
            return response.json().get("search-results", {}).get("entry", [])
        except Exception as e:
            logger.error(f"Error searching ScienceDirect: {e}")
            return []
            
    def get_full_text(self, doi: str) -> str:
        """
        Get full text of a paper by DOI.
        
        Args:
            doi: Digital Object Identifier
            
        Returns:
            Full text of the paper
        """
        try:
            response = requests.get(
                f"https://api.elsevier.com/content/article/doi/{doi}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error getting full text for DOI {doi}: {e}")
            return ""
            
    def process_papers(self, papers: List[Dict[str, Any]]) -> List[Document]:
        """
        Process papers into documents for ChromaDB.
        
        Args:
            papers: List of paper metadata
            
        Returns:
            List of processed documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        documents = []
        for paper in papers:
            try:
                # Get full text
                doi = paper.get("prism:doi", "")
                full_text = self.get_full_text(doi)
                if not full_text:
                    continue
                    
                # Split text into chunks
                chunks = text_splitter.split_text(full_text)
                
                # Create metadata
                metadata = {
                    "type": "research_paper",
                    "title": paper.get("dc:title", ""),
                    "url": paper.get("prism:url", ""),
                    "journal": "American Journal of Obstetrics and Gynecology",
                    "year": paper.get("prism:coverDate", "").split("-")[0],
                    "doi": doi
                }
                
                # Validate metadata
                validated_metadata = ResearchPaperMetadata(**metadata).dict()
                
                # Create documents for each chunk
                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        **validated_metadata,
                        "chunk": i + 1,
                        "total_chunks": len(chunks)
                    }
                    documents.append(Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    ))
                    
            except Exception as e:
                logger.error(f"Error processing paper {paper.get('dc:title', '')}: {e}")
                continue
                
        return documents

    def check_existing_papers(self, db, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out papers that already exist in the database.
        
        Args:
            db: Database instance
            papers: List of paper metadata
            
        Returns:
            List of papers that don't exist in the database
        """
        new_papers = []
        for paper in papers:
            doi = paper.get("prism:doi", "")
            if not doi:
                continue
                
            # Search for existing paper with this DOI
            results = db.search_collection(
                "research_papers",
                "",  # Empty query string since we're only filtering
                metadata_filter={"doi": doi},
                k=1  # We only need to know if at least one exists
            )
            
            if not results:
                new_papers.append(paper)
            else:
                logger.info(f"Skipping existing paper with DOI: {doi}")
                
        return new_papers

def main():
    """Main function to load and process papers."""
    # Get API key from environment
    api_key = os.getenv("SCIENCEDIRECT_API_KEY")
    if not api_key:
        raise ValueError("SCIENCEDIRECT_API_KEY environment variable not set")
        
    # Initialize loader and database
    loader = ScienceDirectLoader(api_key)
    db = get_db()
    
    # Search for papers about contraceptives
    query = "contraception AND journal:American Journal of Obstetrics and Gynecology"
    papers = loader.search_papers(query, max_results=1)
    
    if not papers:
        logger.warning("No papers found")
        return
        
    # Filter out existing papers
    new_papers = loader.check_existing_papers(db, papers)
    
    if not new_papers:
        logger.info("No new papers to process")
        return
    
    # Process papers into documents
    documents = loader.process_papers(new_papers)
    
    if not documents:
        logger.warning("No documents processed")
        return
        
    # Add documents to ChromaDB
    try:
        db.add_to_collection("research_papers", documents)
        logger.info(f"Successfully added {len(documents)} document chunks to ChromaDB")
    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB: {e}")

if __name__ == "__main__":
    main() 