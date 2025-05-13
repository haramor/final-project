from flask import Flask
from app.rag.rag_service import rag_bp  # Import the Blueprint

def create_app():
    app = Flask(__name__)

    # Register the Blueprint
    app.register_blueprint(rag_bp, url_prefix='/rag')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5173, host='0.0.0.0')