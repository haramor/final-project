# Vector Database Module

This module provides a unified interface for interacting with ChromaDB vector database, specifically designed for storing and retrieving medical articles and research papers.

## Structure

- `db.py`: Main database connector implementation using ChromaDB through Langchain
- `config.py`: Configuration settings for the database
- `schema.py`: Pydantic models for metadata validation

## Usage

### Initialize Database

```python
from database.db import get_db

# Get database instance (singleton)
db = get_db()
```

### Add Documents

```python
from langchain_core.documents import Document

# Create a document
doc = Document(
    page_content="IUDs are a form of long-acting reversible contraception.",
    metadata={
        "doc_type": "article",
        "title": "Overview of Long-Acting Reversible Contraception",
        "url": "https://example.com/article",
        "journal": "American Journal of Obstetrics and Gynecology",
        "year": "2023",
        "doi": "10.1234/ajog.2023.123"
    }
)

# Add to specific collection
db.add_to_collection("research_papers", doc)
```

### Search Documents

```python
# Search in specific collection
results = db.search_collection(
    "research_papers",
    "What are IUDs?",
    metadata_filter={"journal": "American Journal of Obstetrics and Gynecology"}
)

# Search across all collections
results = db.search_all("IUD effectiveness")
```

## Configuration

Set these environment variables in `.env`:

```bash
CHROMA_PERSIST_DIRECTORY=/path/to/your/storage
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

Or use defaults from `config.py`.

## Collections and Schemas

Collections are defined in `schema.py` using Pydantic models:

- `articles`: Medical journal articles
- `research_papers`: Academic research papers

Each collection has its own metadata schema that is validated when adding documents.

## Dependencies

```bash
pip install langchain-core langchain-community langchain-chroma chromadb sentence-transformers
```


### File Organization

```
backend/app/
├── database/
│   ├── README.md               # This file
│   ├── config.py               # Database configuration
│   ├── db_connector.py         # Main database interface (Sarah)
│   ├── schema.py               # Database schema definition (Sarah)
├── document_processing/
│   ├── loader.py               # Document loading (Lily)
│   ├── processor.py            # Text processing (Lily)
│   ├── embedding.py            # Embedding generation (Lily)
│   └── indexer.py              # Database indexing (Lily)
└── rag/
    └── rag_service.py          # RAG implementation (Both to modify)
```

## Tech Stack Decision Points


### Embedding Models (Lily to Decide)
- **all-MiniLM-L6-v2**: Current baseline model
- **SPECTER**: Specialized for scientific papers
- **BiomedCLIP**: Medical domain specific
- **PubmedBERT**: Trained on medical literature