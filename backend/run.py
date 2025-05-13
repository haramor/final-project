from app import create_app # Import create_app from the app package (app/__init__.py)

app = create_app() # Create the app instance using the centralized function

if __name__ == '__main__':
    # Use a different port if 5173 is for the frontend, e.g., 5001 or Flask's default 5000
    # The previous run.py used 5001. Let's stick to that for consistency with earlier debugging.
    app.run(debug=True, port=5001, host='0.0.0.0')