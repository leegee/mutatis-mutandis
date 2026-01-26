# src/routes/document_routes.py
from flask import Blueprint, jsonify
from src.services.document_service import get_document_by_id

documents_bp = Blueprint("documents_bp", __name__)

@documents_bp.route("/documents/<doc_id>", methods=["GET"])
def get_document(doc_id):
    doc = get_document_by_id(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    return jsonify(doc)
