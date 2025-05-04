# Women's Health Contraceptive Research Assistant

A tool for accessing medically accurate research on contraception using a combination of LLM and RAG (Retrieval-Augmented Generation) technologies.

## Project Overview

This application empowers users to query medical information about contraception using natural language. The system retrieves relevant information from verified medical sources and generates accurate, evidence-based responses.

### Features

- **Natural Language Interface**: Ask questions about contraception in everyday language
- **Evidence-Based Responses**: All information is drawn directly from medical literature
- **Source Transparency**: Responses include references to source documents
- **Filtering Options**: Narrow searches based on age groups, specific contraceptive methods, and side effects

## Project Structure

```
xx/
├── backend/               # Flask backend API
│   ├── app/               # Application code
│   │   ├── __init__.py    # Flask app initialization
│   │   ├── routes/        # API endpoints
│   │   │   ├── __init__.py
│   │   │   ├── pubmed.py  # PubMed API integration
│   │   │   └── rag.py     # RAG endpoint
│   │   └── rag/           # RAG implementation
│   │       ├── __init__.py
│   │       └── rag_service.py  # Core RAG functionality
│   ├── requirements.txt   # Python dependencies
│   └── run.py             # Entry point
└── frontend/              # React frontend
    ├── src/
    │   ├── App.jsx        # Main application component
    │   ├── components/    # React components
    │   │   └── SearchForm.jsx  # Search interface
    │   └── services/      # API client services
    │       └── api.js     # API integration
    └── package.json       # Node.js dependencies
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- Node.js 16+
- Ollama (for local LLM)

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Install and Run Ollama**:
   - Download and install Ollama from [ollama.ai](https://ollama.ai)
   - Start the Ollama server:
     ```bash
     ollama serve
     ```
   - Pull the LLama3 model (in a separate terminal):
     ```bash
     ollama pull llama3
     ```

3. **Run the Flask server**:
   ```bash
   python run.py
   ```
   The server will start at http://localhost:5001

### Frontend Setup

1. **Install Node.js dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Run the development server**:
   ```bash
   npm run dev
   ```
   The app will be available at http://localhost:5173

## Current Implementation Status

- ✅ Basic frontend with search interface
- ✅ Backend Flask structure
- ✅ RAG implementation with in-memory vector store
- ✅ PubMed API integration
- ✅ Placeholder data for testing
- ⏳ Persistent vector database integration (in progress)
- ⏳ Full dataset of medical literature (in progress)

## Next Steps (For Sarah and Lily)

### 1. Vector Database Setup

**Sarah**: Focus on setting up a persistent vector database to replace the current in-memory placeholder.

- **Options**:
  - ChromaDB
  - Pinecone
  - Weaviate
  - Milvus/Zilliz

- **Implementation Steps**:
  - Choose a vector database
  - Create a `database/` directory with setup scripts
  - Update `backend/app/rag/rag_service.py` to connect to the database
  - Implement functions to load and index documents

### 2. Document Processing Pipeline

**Lily**: Develop a pipeline for processing and embedding medical literature.

- **Pipeline Components**:
  - Document loading (PDFs, research papers)
  - Text extraction and cleaning
  - Chunking for optimal retrieval
  - Metadata extraction (publication date, authors, journal)
  - Embeddings generation
  - Vector database ingestion

- **Suggested Files**:
  - `backend/app/document_processing/`
    - `loader.py` - Functions to load documents
    - `processor.py` - Text extraction and chunking
    - `embedding.py` - Generate embeddings
    - `indexer.py` - Add to vector database

### 3. Integration Points

- Update `rag_service.py` to:
  - Replace the in-memory vector store with your persistent database
  - Remove the placeholder documents and connect to the real dataset
  - Enhance the prompt template based on testing results

- Modify the frontend to:
  - Better display sources with clickable links to papers
  - Improve the filtering UI based on metadata
  - Add visualization of search results

### 4. Advanced Features (When Basic RAG Works)

- Implement citation extraction for each fact in responses
- Add user-specific history and favorites
- Deploy to a cloud service for broader access
- Add authentication for data security

## Using the Current System

1. **Start both backend and frontend** (as described in setup)
2. **Open the web interface** at http://localhost:5173
3. **Ask questions** about contraception:
   - "What are the side effects of an IUD?"
   - "How effective are hormonal patches?"
   - "What contraception methods are best for women over 40?"

4. **API Testing** (if needed):
   ```bash
   curl -X POST -H "Content-Type: application/json" \
     -d '{"query": "What are the side effects of an IUD?"}' \
     http://localhost:5001/api/rag_query
   ```

## Notes for Sarah and Lily

- The current system has a fallback mode if Ollama isn't available - it will use simple pattern matching to respond
- The `rag_service.py` file has detailed comments explaining each component
- The in-memory vector store will reset whenever the server restarts
- Focus on one component at a time - get the database working first, then the document pipeline
- Keep the system modular for easier testing and maintenance

## Troubleshooting

- **ChromaDB connection issues**: ChromaDB needs specific setup for persistence - see their documentation
- **Ollama connection refused**: Make sure Ollama is running with `ollama serve`
- **Missing model**: Run `ollama pull llama3` to download the model
- **CORS errors**: The backend is configured for localhost:5173, update if using different ports

## Future Vision

This tool aims to become an open-source, reliable source of contraception information for women, combining the accessibility of AI with the reliability of evidence-based medicine. With your contributions, it can help bridge the information gap in women's health research.

Good luck, and feel free to extend and improve this foundation!