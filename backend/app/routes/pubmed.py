from flask import Blueprint, request, jsonify
from ..services.pubmed_service import PubMedService

pubmed_bp = Blueprint('pubmed', __name__)
pubmed_service = PubMedService()

@pubmed_bp.route('/', methods=['GET'])
def home():
    return "Welcome to the XX PubMed API"


@pubmed_bp.route('/search', methods=['POST'])
def search_pubmed():
    """
    Get article IDs from PubMed according to items in dropdown and natural language query
    
    Request body:
    {
        "filters": {
            "birth_control": ["IUD Hormonal", "Nexplanon"],
            "side_effects": ["Weight Gain", "Mood Changes"],
            "age_group": ["Adolescent"],
            "additional": ["Efficacy"],
            "mesh_terms": ["Contraception"]
        },
        "natural_language_query": "What are the side effects of Nexplanon?"
    }

    Returns:
    {
        "articles": [
            {
                "title": "Title of Article",
                "abstract": "Abstract of Article",
                "pmid": "1234567890"
            }
        ]
    }
    """
    data = request.json
    filters = data.get('filters', {})
    print("filters in search", filters)
    natural_language_query = data.get('natural_language_query', '')
    
    response, error = pubmed_service.search_articles(filters, natural_language_query)

    if error:
        return jsonify({'error': error}), 400
        
    return jsonify({'response': response}), 200

@pubmed_bp.route('/dropdown-options', methods=['GET'])
def get_dropdown_options():
    """Get all options for the dropdown menus"""
    options = pubmed_service.get_dropdown_options()
    return jsonify(options), 200