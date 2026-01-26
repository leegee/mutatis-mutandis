# src/search_service.py
"""
curl "http://127.0.0.1:5000/search?q=liberty&author=Milton&year=1640"

curl "http://127.0.0.1:5000/documents/A02601"
"""

from flask import Flask
from flask_cors import CORS

from src.routes.search_routes import search_bp
from src.routes.document_routes import documents_bp

searchApp = Flask(__name__)
CORS(searchApp)

searchApp.register_blueprint(search_bp)
searchApp.register_blueprint(documents_bp)

if __name__ == "__main__":
    searchApp.run(debug=True)
