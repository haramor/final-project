from flask import Blueprint, request, jsonify
from ..rag.rag_service import query_rag

rag_bp = Blueprint('rag_bp', __name__)

@rag_bp.route('/api/rag_query', methods=['POST'])
def handle_rag_query():
    """Receives a query and user profile from the frontend and returns the RAG response."""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    user_query = data['query']
    # Extract user profile data from the request, with defaults
    current_birth_control = data.get('current_birth_control', 'Not specified')
    current_side_effects = data.get('current_side_effects', 'Not specified')
    current_age_group = data.get('current_age_group', 'Not specified')
    primary_reason = data.get('primary_reason', 'Not specified')

    # --- Optional: Handle additional context if sent from frontend --- 
    # Example: You might add fields like 'current_method', 'priorities'
    # current_method = data.get('current_method') 
    # priorities = data.get('priorities')
    # You would then potentially pass these to query_rag or modify the query/prompt
    # --- End Optional --- 

    try:
        result = query_rag(
            user_query,
            current_birth_control=current_birth_control,
            current_side_effects=current_side_effects,
            current_age_group=current_age_group,
            primary_reason=primary_reason
        )
        return jsonify(result) # Result already contains {"answer": ..., "sources": ...}
    except Exception as e:
        print(f"Error in /api/rag_query endpoint: {e}") # Log the exception
        return jsonify({"error": "An internal server error occurred"}), 500 

@rag_bp.route('/dropdown-options', methods=['GET'])
def get_dropdown_options_route():
    """Returns dropdown options for the frontend."""
    try:
        options = {
            "birth_control_methods": ["IUD (Intrauterine Device)", "The Pill (Combined Oral Contraceptive)", "The Patch (Transdermal Contraceptive Patch)", "The Ring (Vaginal Ring)", "The Shot (Depo-Provera)", "Implant (Nexplanon)", "Condoms (Male)", "Condoms (Female)", "Spermicide", "Fertility Awareness-Based Methods", "Emergency Contraception", "Not Currently Using Any"],
            "side_effects": ["Nausea", "Headache", "Mood swings", "Irregular bleeding", "Spotting", "Weight gain", "Weight loss", "Acne", "Decreased libido", "Increased libido", "Breast tenderness", "No notable side effects"],
            "age_groups": ["Under 18", "18-24", "25-29", "30-34", "35-39", "40-44", "45+", "Prefer not to say"],
            "primary_reason": ["Prevent Pregnancy", "Manage Menstrual Symptoms (e.g., heavy bleeding, pain)", "Hormone Regulation (e.g., for acne, PCOS)", "Perimenopause Symptom Management", "Gender Affirming Care", "Other", "Prefer not to say"],
            "additional_filters": ["Efficacy", "Safety", "Cost-effectiveness", "Long-term Effects", "Breastfeeding", "Postpartum", "Drug Interactions", "Hormonal Effects", "Non-contraceptive Benefits", "Compliance", "Continuation Rates", "Patient Satisfaction"],
            "mesh_terms": ["Contraception", "Contraceptive Agents", "Contraceptive Devices", "Contraceptives, Oral", "Contraceptives, Oral, Hormonal", "Intrauterine Devices", "Contraceptive Agents, Female", "Contraceptive Agents, Male", "Family Planning Services"]
        }
        print("Backend /dropdown-options preparing to send:", options)
        return jsonify(options)
    except Exception as e:
        print(f"Error in /dropdown-options endpoint: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500 