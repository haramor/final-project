# Women's Health Contraceptive Research Assistant

A tool for accessing medically accurate research on contraception using a combination of LLM and RAG (Retrieval-Augmented Generation) technologies.

## Project Overview

This application empowers users to query medical information about contraception using natural language. The system retrieves relevant information from verified medical sources and generates accurate, evidence-based responses.

Key Features:
- Natural language interface for contraceptive queries
- Evidence-based responses from medical literature
- Source transparency with references to medical papers
- Filtering options for age groups, methods, and side effects

## Project Structure

```
.
├── backend/                 # Python Flask backend
│   ├── app/
│   │   ├── database/       # Database models and operations
│   │   ├── document_processing/  # Document processing pipeline
│   │   ├── rag/           # RAG implementation
│   │   ├── routes/        # API endpoints
│   │   └── services/      # Core business logic
│   └── requirements.txt    # Python dependencies
│
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   └── services/      # API integration
│   └── package.json       # Node dependencies
│
└── pubmed_queries/        # PubMed data collection scripts
```

## Current Features

### Backend
- PubMed API integration for contraceptive research papers
- Document processing pipeline for medical papers
- RAG-based question answering system for contraceptive queries
- Database integration for storing processed documents

### Frontend
- User-friendly search interface for contraceptive information
- Article list display with filtering options
- Experience sharing component for user stories
- API integration with backend services

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- PostgreSQL (for database)

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Create a .env file with necessary configurations
   # Required variables:
   # - DATABASE_URL
   # - PUBMED_API_KEY (optional)
   ```

5. Run the backend server:
   ```bash
   python run.py
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

## Development Status

### Completed
- Basic PubMed search functionality for contraceptive research
- Document processing pipeline for medical papers
- Database schema and models
- Frontend search interface
- Basic RAG implementation for contraceptive queries

### In Progress
- Enhanced RAG capabilities for more accurate responses
- Improved document processing for better information extraction
- User experience improvements for better accessibility

### Planned
- Advanced filtering for contraceptive methods and side effects
- Fine-tuning model on medical data using LORA or other technique

## Example Queries

The system can handle questions like:
- "What are the side effects of an IUD?"
- "How effective are hormonal patches?"
- "What contraception methods are best for women over 40?"
- "What are the risks of long-term birth control use?"

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

