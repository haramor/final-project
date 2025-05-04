from flask import Flask, make_response, request
from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    from .config import Config
    app.config.from_object(Config)

    # Configure CORS properly
    app.config['CORS_HEADERS'] = 'Content-Type'
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:5173"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
        }
    })

    # Add CORS headers to all responses
    @app.after_request
    def after_request(response):
        header = response.headers
        header['Access-Control-Allow-Origin'] = 'http://localhost:5173'
        header['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        header['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    # Handle OPTIONS requests
    @app.before_request
    def handle_preflight():
        if request.method == "OPTIONS":
            response = make_response()
            header = response.headers
            header['Access-Control-Allow-Origin'] = 'http://localhost:5173'
            header['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            header['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
    

    with app.app_context():  # Create context
        from .routes.pubmed import pubmed_bp
        from .routes.rag import rag_bp
        app.register_blueprint(pubmed_bp)
        app.register_blueprint(rag_bp)

    return app
