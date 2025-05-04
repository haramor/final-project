import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

# Load environment variables 
load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3"  # Or another model you have downloaded in Ollama

# Initialize once at module level to print first-run debug logs
print("RAG module loaded, components will initialize on first query.")

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
    'embedding_function': None,
    'vectorstore': None,
    'llm': None,
    'retriever': None,
    'rag_chain_with_sources': None
}

def format_docs(docs):
    """Helper function to format retrieved documents for the prompt."""
    context_str = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs])
    return context_str

def get_sources(docs):
    """Helper function to extract unique source names from retrieved documents."""
    return list(set(doc.metadata.get('source', 'Unknown') for doc in docs))

def initialize_components():
    """Initialize all RAG components if they haven't been initialized yet."""
    global _component_cache
    
    # Skip if already initialized
    if _component_cache['initialized']:
        return True
    
    try:
        # Initialize embedding function
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("[initialize_components] Embedding function initialized.")
        
        # Initialize vector store with sample documents
        vectorstore = InMemoryVectorStore(embedding=embedding_function)
        documents = [
            Document(
                page_content="This is a dummy document about contraception. Women have many options for contraception including hormonal and non-hormonal methods.",
                metadata={"source": "dummy_source", "title": "Contraception Overview"}
            ),
            Document(
                page_content="IUDs (intrauterine devices) are small, T-shaped devices inserted into the uterus to prevent pregnancy. Common side effects may include cramping, spotting, or irregular periods.",
                metadata={"source": "dummy_source", "title": "IUD Information"}
            ),
            Document(
                page_content="The hormonal patch releases hormones through the skin and is changed weekly. Potential side effects include skin irritation at the site of application.",
                metadata={"source": "dummy_source", "title": "Birth Control Patch"}
            )
        ]
        vectorstore.add_documents(documents)
        print(f"[initialize_components] Vector store initialized with {len(documents)} documents.")
        
        # Initialize LLM - Try to connect to Ollama
        try:
            llm = OllamaLLM(model=LLM_MODEL_NAME)
            print(f"[initialize_components] Connected to Ollama with model {LLM_MODEL_NAME}.")
            
            # Quick test to verify connection and model
            test_response = llm.invoke("Hello there!")
            print(f"[initialize_components] Ollama test successful.")
        except Exception as e:
            # If Ollama fails, we'll use a mock function for demonstration purposes
            print(f"[initialize_components] Warning: Ollama connection failed: {e}")
            print("[initialize_components] Using a mock LLM function for demonstration.")
            
            # Simple function that returns pre-defined responses based on keywords in the question
            def mock_llm_function(text):
                # Simple keyword-based answering
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
            
        # Initialize retriever
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        print("[initialize_components] Retriever initialized.")
        
        # Initialize RAG prompt
        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        # Initialize RAG chain
        # Simplify the chain construction to avoid potential issues
        def process_query(query_dict):
            question = query_dict["question"]
            
            # Get documents
            docs = retriever.invoke(question)
            
            # Format documents and get sources
            context = format_docs(docs)
            sources = get_sources(docs)
            
            # Get answer from LLM
            prompt_input = {"context": context, "question": question}
            formatted_prompt = rag_prompt.format(**prompt_input)
            
            # Check if we should use the invoke method or call directly
            if hasattr(llm, 'invoke'):
                answer = llm.invoke(formatted_prompt)
            else:
                answer = llm(formatted_prompt)
            
            # Return result
            return {"answer": answer, "sources": sources}
        
        # This gives us a simple function that can be used without complex LangChain chaining
        rag_chain_with_sources = process_query
        
        print("[initialize_components] RAG chain initialized.")
        
        # Store everything in cache
        _component_cache.update({
            'initialized': True,
            'embedding_function': embedding_function,
            'vectorstore': vectorstore,
            'llm': llm,
            'retriever': retriever,
            'rag_chain_with_sources': rag_chain_with_sources
        })
        
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
    # Initialize components if needed
    if not _component_cache['initialized']:
        success = initialize_components()
        if not success:
            return {"answer": "Error initializing RAG components. Cannot process query.", "sources": []}
    
    # Get the chain from cache
    rag_chain_with_sources = _component_cache['rag_chain_with_sources']
    
    # Process the query
    print(f"[query_rag] Processing query: {user_query}")
    try:
        # Call the function directly instead of using .invoke()
        result = rag_chain_with_sources({"question": user_query})
        print(f"[query_rag] Result: {result}")
        return result
    except Exception as e:
        print(f"[query_rag] Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return {"answer": f"An error occurred while processing your request: {e}", "sources": []}

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