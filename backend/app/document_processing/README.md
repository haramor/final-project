# Document Processing Pipeline

This directory contains the skeleton implementation of the document processing pipeline for the contraception research assistant. These files provide a framework that you (Lily) can build upon to create a robust system for ingesting, processing, and indexing documents.

## Files Structure

- `__init__.py`: Module initialization
- `loader.py`: Document loading from various sources (PDF, PubMed, JSON, CSV)
- `processor.py`: Text processing, chunking, and section extraction
- `embedding.py`: Generating embeddings from text
- `indexer.py`: Adding documents to the vector database
- `test_pipeline.py`: Test script to validate the pipeline functionality

## Getting Started

New idea, delete all this and just create one file that:

- Calls the Science Direct API (and others) to get all the documents we might need
- Web scrape to get articles from Mayo Clinic and Planned Parenthood
- Take documents from API calls and web scraping and put them into raw text with necessary metadata (communicate with Sarah to figure out exact metadata)
- Chunking? or does DB.add_document do that automatically
- DB.add_document() ?
- Figure out how to make this an autonomous pipeline (once a month?)

1. **Review the existing code**: Familiarize yourself with the structure and functionality of each file. The code contains many placeholder implementations that you'll need to expand.

2. **Install additional dependencies**: You may need to install specific libraries for document processing:

```bash
pip install pypdf langchain-huggingface requests beautifulsoup4
```

3. **Start with `loader.py`**: Implement the PDF loading functionality using pypdf or another PDF library of your choice. This will allow you to test the pipeline with real documents.

4. **Run tests**: Use the `test_pipeline.py` script to test your implementations as you go:

```bash
cd backend
python -m app.document_processing.test_pipeline
```

## Implementation Tasks

### Week 1: Document Loading

- [ ] Implement robust PDF text extraction in `PDFLoader.load_document()`
- [ ] Implement PubMed article fetching in `PubMedLoader.load_document()`
- [ ] Add support for table extraction from PDFs
- [ ] Handle different file encodings and formats

### Week 2: Text Processing

- [ ] Enhance the text cleaning in `CleanTextProcessor.process_text()`
- [ ] Implement medical term extraction in `MedicalTermExtractor`
- [ ] Improve section extraction for medical documents in `SectionExtractor`
- [ ] Create a specialized chunking strategy for contraception documents

### Week 3: Embedding & Indexing

- [ ] Evaluate different embedding models for medical text
- [ ] Implement caching for embeddings in `EmbeddingCache`
- [ ] Optimize the indexing process for large document collections
- [ ] Create a sample dataset of contraception documents for testing

## Tips for Implementation

### PDF Loading

Consider using [PyMuPDF](https://github.com/pymupdf/PyMuPDF) as an alternative to pypdf for better text extraction quality:

```python
# Example with PyMuPDF (fitz)
import fitz  # pip install pymupdf

def extract_text_with_pymupdf(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

### PubMed Integration

Use the NCBI E-utilities API to fetch PubMed articles:

```python
import requests
from bs4 import BeautifulSoup

def fetch_pubmed_article(pmid, api_key=None):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "rettype": "abstract"
    }
    if api_key:
        params["api_key"] = api_key

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "xml")
        # Extract article details
        # ...
        return article_text, metadata
    else:
        raise Exception(f"Failed to fetch PubMed article: {response.status_code}")
```

### Chunking Strategies

Consider different chunking strategies based on document type:

- Academic papers: Chunk by section (abstract, methods, results, etc.)
- Clinical guidelines: Chunk by recommendation sections
- Patient information: Chunk by topic or question

### Medical Term Extraction

For medical term extraction, consider using libraries like:

- ScispaCy: `pip install scispacy`
- UMLS Lexical Tools: https://lexsrv3.nlm.nih.gov/LexSysGroup/Projects/lvg/current/web/index.html

### Testing

Create a set of test documents in different formats to validate your implementation. Focus on contraception-related content to ensure domain-specific processing works correctly.

## Working with Sarah

Sarah is implementing the vector database that your document processing pipeline will feed into. Coordinate with her on:

1. **Metadata schema**: Ensure the metadata you extract matches the schema in `backend/app/database/schema.py`
2. **Batch processing**: Optimize how documents are sent to the database for indexing
3. **API design**: Agree on function signatures and data formats

Good luck with your implementation, Lily! The work you're doing will significantly improve the contraception research assistant's ability to provide accurate information to users.
