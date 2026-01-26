# src/search_service.py
"""
curl "http://127.0.0.1:5000/search?q=liberty&author=Milton&year=1640"

curl "http://127.0.0.1:5000/documents/A02601"
"""

from flask import Flask
from src.routes.search_routes import search_bp
from src.routes.document_routes import documents_bp

search = Flask(__name__)

# register blueprints
search.register_blueprint(search_bp)
search.register_blueprint(documents_bp)

if __name__ == "__main__":
    search.run(debug=True)
