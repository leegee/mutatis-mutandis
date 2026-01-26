# src/routes/search_routes.py
from flask import Blueprint, request, jsonify
from src.services.document_service import search_documents

search_bp = Blueprint("search_bp", __name__)

@search_bp.route("/search", methods=["GET"])
def search():
    q = request.args.get("q")
    author = request.args.get("author")
    year = request.args.get("year")
    place = request.args.get("place")
    title = request.args.get("title")
    limit = int(request.args.get("size", 20))
    offset = int(request.args.get("from", 0))

    print("=== /search called ===")
    print(f"Params -> q: {q}, author: {author}, year: {year}, place: {place}, title: {title}")
    print(f"Limit: {limit}, Offset: {offset}")

    result = search_documents(
        q=q,
        author=author,
        year=year,
        place=place,
        title=title,
        limit=limit,
        offset=offset
    )

    print(f"Result total: {result['total']}")

    return jsonify({
        "took": 0,
        "total": result["total"],
        "hits": result["hits"]
    })
