# src/services/document_service.py
from src.lib.eebo_db import get_connection

def search_documents(q=None, author=None, year=None, place=None, title=None, limit=20, offset=0):
    query = "SELECT d.doc_id, d.title, d.author, d.pub_year, d.pub_place FROM documents d"
    count_query = "SELECT COUNT(*) FROM documents d"
    filters = []
    params = []

    # Full-text search using tsvector + GIN
    if q:
        # This replaces the JOIN + LIKE search
        filters.append("d.tsv @@ plainto_tsquery(%s)")
        params.append(q)

    # Other metadata filters
    if author:
        filters.append("LOWER(d.author) LIKE %s")
        params.append(f"%{author.lower()}%")
    if year:
        filters.append("d.pub_year = %s")
        params.append(year)
    if place:
        filters.append("LOWER(d.pub_place) LIKE %s")
        params.append(f"%{place.lower()}%")
    if title:
        filters.append("LOWER(d.title) LIKE %s")
        params.append(f"%{title.lower()}%")

    if filters:
        where_clause = " AND ".join(filters)
        query += " WHERE " + where_clause
        count_query += " WHERE " + where_clause

    query += " ORDER BY d.pub_year, d.title LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Execute main query
            cur.execute(query, params)
            rows = cur.fetchall()

            # Execute count query
            cur.execute(count_query, params[:-2])
            total = cur.fetchone()[0]

    hits = [{
        "_id": r[0],
        "_source": {
            "title": r[1],
            "author": r[2],
            "year": r[3],
            "place": r[4]
        }
    } for r in rows]

    return {"total": total, "hits": hits}


def get_document_by_id(doc_id: str):
    """
    Returns metadata + reconstructed text for a single document.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT doc_id, title, author, pub_year, pub_place, publisher
                FROM documents
                WHERE doc_id = %s
            """, (doc_id,))
            meta = cur.fetchone()

            if not meta:
                return None

            cur.execute("""
                SELECT token
                FROM tokens
                WHERE doc_id = %s
                ORDER BY token_idx
            """, (doc_id,))
            tokens = [row[0] for row in cur.fetchall()]

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
