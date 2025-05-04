"""
Test script for the document processing pipeline.

This script tests the full document processing pipeline, from loading 
documents to indexing them in the vector database.
"""

import os
import logging
from typing import Dict, Any
from langchain_core.documents import Document

# Import the components
from .loader import load_document
from .processor import create_standard_processor, SectionExtractor, TextChunker
from .embedding import EmbeddingGenerator
from .indexer import get_indexer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_text_processing():
    """Test the text processing components."""
    # Sample medical document text
    sample_text = """
    Abstract: This study examines the efficacy of various contraceptive methods.
    
    Introduction: Contraception remains an important aspect of reproductive health.
    
    Methods: We conducted a systematic review of clinical trials.
    
    Results: IUDs showed the highest efficacy rate at 99%.
    
    Discussion: The findings suggest that long-acting reversible contraceptives are most effective.
    
    Conclusion: Healthcare providers should consider efficacy rates when counseling patients.
    
    References: Smith et al. (2020), Jones et al. (2019)
    """
    
    # Create a document
    doc = Document(page_content=sample_text, metadata={"source": "test_sample"})
    
    # Create processors
    processor, section_extractor, chunker = create_standard_processor()
    
    # Process the document
    processed_doc = processor.process_document(doc)
    logger.info(f"Processed document length: {len(processed_doc.page_content)}")
    
    # Extract sections
    sections = section_extractor.extract_sections(processed_doc.page_content)
    logger.info(f"Extracted sections: {list(sections.keys())}")
    
    # Split into chunks
    chunked_docs = chunker.split_document(processed_doc)
    logger.info(f"Split into {len(chunked_docs)} chunks")
    for i, chunk_doc in enumerate(chunked_docs):
        logger.info(f"Chunk {i+1} length: {len(chunk_doc.page_content)}")
    
    return chunked_docs

def test_embedding_generation(documents: list[Document]):
    """Test the embedding generation component."""
    embedding_generator = EmbeddingGenerator()
    
    # Generate embeddings for documents
    embedded_docs = embedding_generator.embed_documents(documents)
    
    logger.info(f"Generated embeddings for {len(embedded_docs)} documents")
    
    # Check the first embedding
    first_embedding = embedded_docs[0]["embedding"]
    logger.info(f"First embedding dimensions: {len(first_embedding)}")
    logger.info(f"First 5 values: {first_embedding[:5]}")
    
    return embedded_docs

def test_sample_contraception_text():
    """Test processing and embedding of contraception-related text."""
    # Sample text about contraception
    contraception_texts = [
        "IUDs are a form of long-acting reversible contraception that are more than 99% effective at preventing pregnancy.",
        "Combined oral contraceptive pills contain estrogen and progestin, and have a typical-use failure rate of around 7%.",
        "Condoms are barrier methods that prevent pregnancy and reduce the risk of sexually transmitted infections.",
        "Emergency contraception, such as Plan B, can be used within 72 hours after unprotected sex to prevent pregnancy.",
        "Contraceptive implants are small, flexible rods inserted under the skin of the upper arm that release progestin."
    ]
    
    # Create documents
    documents = []
    for i, text in enumerate(contraception_texts):
        doc = Document(
            page_content=text,
            metadata={
                "source": f"test_contraception_{i}",
                "type": "patient_info",
                "contraceptive_method": [method.lower() for method in ["IUD", "pill", "condom", "emergency", "implant"][i:i+1]]
            }
        )
        documents.append(doc)
    
    # Process documents
    processor, section_extractor, chunker = create_standard_processor()
    processed_docs = processor.process_documents(documents)
    
    # Generate embeddings
    embedding_generator = EmbeddingGenerator()
    embedded_docs = embedding_generator.embed_documents(processed_docs)
    
    logger.info(f"Generated embeddings for {len(embedded_docs)} contraception documents")
    
    return embedded_docs

def test_pdf_loader():
    """Test the PDF loader with a sample PDF."""
    # Lily: Update this with the path to a sample PDF
    pdf_path = "sample_docs/sample_article.pdf"
    
    # Skip if the file doesn't exist
    if not os.path.exists(pdf_path):
        logger.info(f"Skipping PDF test - file not found: {pdf_path}")
        return None
    
    # Load the PDF
    documents = load_document(pdf_path)
    logger.info(f"Loaded {len(documents)} documents from PDF")
    
    if documents:
        logger.info(f"First document preview: {documents[0].page_content[:100]}...")
        logger.info(f"Metadata: {documents[0].metadata}")
    
    return documents

def test_full_pipeline():
    """Test the full document processing pipeline."""
    logger.info("Starting full pipeline test")
    
    # Get an indexer
    indexer = get_indexer()
    
    # Create a sample document
    sample_text = "IUDs are a form of long-acting reversible contraception that are more than 99% effective at preventing pregnancy."
    doc = Document(
        page_content=sample_text,
        metadata={
            "source": "test_pipeline",
            "type": "patient_info",
            "contraceptive_method": ["IUD"]
        }
    )
    
    # Try to index the document
    try:
        doc_ids = indexer.index_document(doc, collection_name="test")
        logger.info(f"Successfully indexed document with IDs: {doc_ids}")
        
        # Try a query to see if it works
        query = "How effective are IUDs?"
        
        # Since we're just testing the pipeline, we won't actually run the query
        # Here's what it would look like: 
        # results = indexer.db.search(query, collection_name="test")
        # logger.info(f"Query results: {results}")
        
        logger.info("Full pipeline test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pipeline test: {e}")
        
    return None

if __name__ == "__main__":
    logger.info("Testing document processing pipeline...")
    
    # Test text processing
    chunked_docs = test_text_processing()
    
    # Test embedding generation
    embedded_docs = test_embedding_generation(chunked_docs)
    
    # Test with contraception-specific text
    contraception_embeds = test_sample_contraception_text()
    
    # Try loading a PDF if available
    pdf_docs = test_pdf_loader()
    
    # Finally, test the full pipeline
    test_full_pipeline()
    
    logger.info("All tests completed") 