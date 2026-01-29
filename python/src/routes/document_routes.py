# src/routes/document_routes.py
from flask import Blueprint, jsonify, send_file, abort
from pathlib import Path

from src.services.document_service import get_document_by_id
import src.lib.eebo_config as eebo_config

documents_bp = Blueprint("documents_bp", __name__)

@documents_bp.route("/documents/<doc_id>", methods=["GET"])
def get_document_json(doc_id):
    """
    Returns JSON representation of a document.
    """
    doc = get_document_by_id(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404

    return jsonify(doc)


@documents_bp.route("/documents/<doc_id>/xml", methods=["GET", "HEAD"])
def get_document_xml(doc_id):
    """
    Returns XML file for a document, ready to be displayed in an <iframe>.
    """
    xml_path = eebo_config.XML_ROOT_DIR / f"{doc_id}.P4.xml"

    if not xml_path.exists():
        print(f"No XML found at {xml_path}")
        abort(404, description="XML document not found")

    if request.method == "HEAD":
        return Response(status=200)

    return send_file(
        xml_path,
        mimetype="application/xml",
        as_attachment=False
    )
