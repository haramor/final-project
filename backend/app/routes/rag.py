from flask import Blueprint, request, jsonify
from ..rag.rag_service import query_rag

rag_bp = Blueprint('rag_bp', __name__)

@rag_bp.route('/api/rag_query', methods=['POST'])
def handle_rag_query():
    """Receives a query from the frontend and returns the RAG response."""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    user_query = data['query']

    # --- Optional: Handle additional context if sent from frontend --- 
    # Example: You might add fields like 'current_method', 'priorities'
    # current_method = data.get('current_method') 
    # priorities = data.get('priorities')
    # You would then potentially pass these to query_rag or modify the query/prompt
    # --- End Optional --- 

    try:
        result = query_rag(user_query)
        return jsonify(result) # Result already contains {"answer": ..., "sources": ...}
    except Exception as e:
        print(f"Error in /api/rag_query endpoint: {e}") # Log the exception
        return jsonify({"error": "An internal server error occurred"}), 500 