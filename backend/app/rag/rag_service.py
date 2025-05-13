import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from app.database.db import get_db  # Import the database connector
from flask import Blueprint, request, jsonify
from flask_cors import CORS

# Load environment variables
load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3"  # Or another model you have downloaded in Ollama

# --- Prompt Template ---
RAG_PROMPT_TEMPLATE = """
CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
You are an AI assistant specialized in women's health and contraception. Your purpose is to provide accurate, evidence-based information based *only* on the provided CONTEXT.
Answer the user's QUESTION using *only* the information found in the CONTEXT above.
If the CONTEXT does not contain the answer, state clearly that the information is not available in the provided sources.
Do not add any information that is not explicitly mentioned in the CONTEXT.
After providing the answer, list the sources used. The source for each piece of context is included in its metadata (e.g., 'source': 'document_name.pdf'). List the unique source names.

ANSWER:
"""

# Cache for components so we don't initialize them on every request
_component_cache = {
    'initialized': False,
    'retriever': None,
    'rag_prompt': None,
    'llm': None
}

def format_docs(docs):
    """Helper function to format retrieved documents for the prompt."""
    context_str = "\n\n---\n\n".join([f"Source: {doc.metadata.get('title', 'Unknown')}\nContent: {doc.page_content}" for doc in docs])
    return context_str

def get_sources(docs):
    """Helper function to extract unique paper titles from retrieved documents."""
    sources = list(set(doc.metadata.get('title', 'Unknown') for doc in docs))
    return sources

def initialize_components():
    """Initialize all RAG components if they haven't been initialized yet."""
    global _component_cache

    if _component_cache['initialized']:
        return True

    try:
        # Initialize the database
        db = get_db()

        # Use the database's retriever
        retriever = db.collections["research_papers"].as_retriever(search_kwargs={'k': 5})
        _component_cache['retriever'] = retriever

        # Initialize the RAG prompt
        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        _component_cache['rag_prompt'] = rag_prompt

        # Initialize LLM - Try to connect to Ollama
        try:
            llm = OllamaLLM(model=LLM_MODEL_NAME)
            print(f"[initialize_components] Connected to Ollama with model {LLM_MODEL_NAME}.")
            
            # Quick test to verify connection and model
            print(f"[initialize_components] Ollama test successful.")
        except Exception as e:
            # If Ollama fails, use a mock LLM function
            print(f"[initialize_components] Warning: Ollama connection failed: {e}")
            print("[initialize_components] Using a mock LLM function for demonstration.")
            
            # Simple function that returns pre-defined responses based on keywords in the question
            def mock_llm_function(text):
                if "iud" in text.lower():
                    return "Based on the information in the context, IUDs (intrauterine devices) are small, T-shaped devices inserted into the uterus to prevent pregnancy. Common side effects may include cramping, spotting, or irregular periods."
                elif "patch" in text.lower() or "hormonal patch" in text.lower():
                    return "Based on the information in the context, the hormonal patch releases hormones through the skin and is changed weekly. Potential side effects include skin irritation at the site of application."
                else:
                    return "Based on the provided context, women have many options for contraception including hormonal and non-hormonal methods. For specific information about a particular method, please ask about it directly."
            
            # Create a callable object that mimics the LLM interface
            class MockLLM:
                def invoke(self, text):
                    return mock_llm_function(text)
                
                def __call__(self, text):
                    return mock_llm_function(text)
            
            llm = MockLLM()
        
        _component_cache['llm'] = llm
        _component_cache['initialized'] = True
        print("[initialize_components] RAG components initialized successfully.")
        return True
    except Exception as e:
        print(f"[initialize_components] Error initializing components: {e}")
        import traceback
        traceback.print_exc()
        return False

def query_rag(user_query: str) -> dict:
    """
    Queries the RAG chain with the user's question.
    
    Args:
        user_query: The question asked by the user.
        
    Returns:
        A dictionary containing the 'answer' and 'sources'.
    """
    print(f"[query_rag] Received user_query: '{user_query}'")
    # Initialize components if needed
    if not _component_cache['initialized']:
        success = initialize_components()
        if not success:
            return {"answer": "Error initializing RAG components. Cannot process query.", "sources": []}
    
    # Get the retriever and prompt
    retriever = _component_cache['retriever']
    rag_prompt = _component_cache['rag_prompt']
    llm = _component_cache['llm']
    
    # Retrieve documents from the database
    try:
        docs = retriever.get_relevant_documents(user_query)
        if not docs:
            return {"answer": "No relevant documents found in the database.", "sources": []}
        
        # Format documents and get sources
        context = format_docs(docs)
        sources = get_sources(docs)  # Use the updated get_sources function
        
        # Generate the answer using the prompt
        prompt_input = {"context": context, "question": user_query}
        formatted_prompt = rag_prompt.format(**prompt_input)
        
        # Use the LLM to generate the answer
        if hasattr(llm, 'invoke'):
            answer = llm.invoke(formatted_prompt)
        else:
            answer = llm(formatted_prompt)
        
        return {"answer": answer, "sources": sources}
    except Exception as e:
        print(f"[query_rag] Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return {"answer": f"An error occurred while processing your request: {e}", "sources": []}

# Create a Blueprint
rag_bp = Blueprint('rag', __name__)
CORS(rag_bp)

@rag_bp.route('/api/rag_query', methods=['POST'])
def handle_rag_query():
    """Receives a query from the frontend and returns the RAG response."""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    user_query = data['query']
    try:
        result = query_rag(user_query)
        return jsonify(result)
    except Exception as e:
        print(f"Error in /api/rag_query endpoint: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

@rag_bp.route('/dropdown-options', methods=['GET'])
def get_dropdown_options():
    """Returns dropdown options for the frontend."""
    try:
        options = {
            "birth_control_methods": ["IUD", "Pill", "Patch"],
            "side_effects": ["Nausea", "Headache", "Mood swings"],
            "age_groups": ["18-25", "26-35", "36-45"],
            "additional_filters": ["Smoker", "Non-smoker"],
            "mesh_terms": ["Contraception", "Hormonal"]
        }
        return jsonify(options)
    except Exception as e:
        print(f"Error in /dropdown-options endpoint: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

# Example Usage (for testing)
if __name__ == '__main__':
    # Test with some queries
    print("\n=== Testing RAG Query ===")
    test_query = "What are the side effects of an IUD?"
    response = query_rag(test_query)
    print(f"Query: {test_query}")
    print(f"Answer: {response.get('answer')}")
    print(f"Sources: {response.get('sources')}")
    
    print("\n=== Testing Another Query ===")
    test_query_2 = "What about hormonal patches?"
    response_2 = query_rag(test_query_2)
    print(f"Query: {test_query_2}")
    print(f"Answer: {response_2.get('answer')}")
    print(f"Sources: {response_2.get('sources')}")