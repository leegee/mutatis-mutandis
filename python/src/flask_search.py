"""
flask_search.py

curl "http://127.0.0.1:5000/search?q=liberty&author=Milton&year=1640"
"""

from flask import Flask, request, jsonify
from lib.eebo_db import get_connection
from lib.eebo_logging import logger

app = Flask(__name__)

@app.route("/search", methods=["GET"])
def search_documents():
    """
    OpenSearch-style search endpoint:
    /search?author=...&year=...&place=...&title=...
    Returns JSON with "total" and "hits" array.
    """
    # Extract query parameters
    author = request.args.get("author")
    year = request.args.get("year")
    place = request.args.get("place")
    title = request.args.get("title")
    limit = int(request.args.get("size", 20))  # OpenSearch-style limit
    offset = int(request.args.get("from", 0))  # pagination

    # Base query
    query = "SELECT doc_id, title, author, pub_year, pub_place FROM documents"
    filters = []
    params = []

    # Add filters based on params
    if author:
        filters.append("LOWER(author) LIKE %s")
        params.append(f"%{author.lower()}%")
    if year:
        filters.append("pub_year = %s")
        params.append(year)
    if place:
        filters.append("LOWER(pub_place) LIKE %s")
        params.append(f"%{place.lower()}%")
    if title:
        filters.append("LOWER(title) LIKE %s")
        params.append(f"%{title.lower()}%")

    if filters:
        query += " WHERE " + " OR ".join(filters)

    # Add ordering and pagination
    query += " ORDER BY pub_year, title"
    query += " LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    # Execute query
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

            # Get total hits ignoring pagination
            count_query = "SELECT COUNT(*) FROM documents"
            if filters:
                count_query += " WHERE " + " AND ".join(filters)
            cur.execute(count_query, params[:-2])  # exclude limit/offset
            total = cur.fetchone()[0]

    hits = []
    for row in rows:
        hits.append({
            "_id": row[0],
            "_source": {
                "title": row[1],
                "author": row[2],
                "year": row[3],
                "place": row[4]
            }
        })

    return jsonify({
        "took": 0,  # could add timing if needed
        "total": total,
        "hits": hits
    })

@app.route("/documents/<doc_id>", methods=["GET"])
def get_document(doc_id: str):
    """
    Fetch a full document by ID.
    Returns metadata + reconstructed text.
    OpenSearch-style single-document fetch.
    """

    with get_connection() as conn:
        with conn.cursor() as cur:

            # 1️⃣ Fetch metadata
            cur.execute("""
                SELECT doc_id, title, author, pub_year, pub_place, publisher
                FROM documents
                WHERE doc_id = %s
            """, (doc_id,))
            meta = cur.fetchone()

            if not meta:
                return jsonify({"error": "Document not found"}), 404

            # 2️⃣ Fetch tokens in order
            cur.execute("""
                SELECT token
                FROM tokens
                WHERE doc_id = %s
                ORDER BY token_idx
            """, (doc_id,))
            tokens = [row[0] for row in cur.fetchall()]

    # Reconstruct surface text (simple join — can get fancier later)
    text = " ".join(tokens)

    return jsonify({
        "_id": meta[0],
        "_source": {
            "title": meta[1],
            "author": meta[2],
            "year": meta[3],
            "place": meta[4],
            "publisher": meta[5],
            "text": text
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
