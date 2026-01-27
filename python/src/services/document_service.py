# src/services/document_service.py

from typing import Optional, Dict, Any
from src.lib.eebo_db import get_connection

def search_documents(q=None, author=None, year=None, place=None, title=None, limit=20, offset=0):
    filters = []
    params = []

    # Base query: search in block-level materialized view
    query = """
    SELECT doc_id, title, author, pub_year, pub_place, COUNT(*) AS block_count
    FROM document_search
    """
    count_query = """
    SELECT COUNT(DISTINCT doc_id)
    FROM document_search
    """

    # Full-text search
    if q:
        filters.append("tsv @@ plainto_tsquery(%s)")
        params.append(q)

    # Metadata filters
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

    # Combine filters
    if filters:
        where_clause = " AND ".join(filters)
        query += " WHERE " + where_clause
        count_query += " WHERE " + where_clause

    # Group by doc_id to collapse multiple blocks into one document
    query += " GROUP BY doc_id, title, author, pub_year, pub_place"
    query += " ORDER BY pub_year, title LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Execute main query
            cur.execute(query, params)
            rows = cur.fetchall()

            # Execute count query
            cur.execute(count_query, params[:-2])
            row = cur.fetchone()
            assert row is not None  # Mypy
            total = row[0]

    hits = [{
        "_id": r[0],
        "_source": {
            "title": r[1],
            "author": r[2],
            "year": r[3],
            "place": r[4],
            "matching_blocks": r[5]  # number of blocks matching the query
        }
    } for r in rows]

    return {"total": total, "hits": hits}


def get_document_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Returns metadata + reconstructed text for a single document.

    Returns None if no document is found.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Fetch document metadata
            cur.execute("""
                SELECT doc_id, title, author, pub_year, pub_place, publisher
                FROM documents
                WHERE doc_id = %s
            """, (doc_id,))
            meta: Optional[tuple[str, str, str, int, str, str]] = cur.fetchone()

            if meta is None:
                return None

            # Fetch tokens
            cur.execute("""
                SELECT token
                FROM tokens
                WHERE doc_id = %s
                ORDER BY token_idx
            """, (doc_id,))
            tokens: list[str] = [row[0] for row in cur.fetchall()]

    text = " ".join(tokens)

    return {
        "_id": meta[0],
        "_source": {
            "title": meta[1],
            "author": meta[2],
            "year": meta[3],
            "place": meta[4],
            "publisher": meta[5],
            "text": text
        }
    }
