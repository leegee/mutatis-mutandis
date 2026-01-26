# src/services/document_service.py
from src.lib.eebo_db import get_connection

def search_documents(author=None, year=None, place=None, title=None, limit=20, offset=0):
    """
    Returns a list of documents matching search filters and total count.
    """
    query = "SELECT doc_id, title, author, pub_year, pub_place FROM documents"
    filters = []
    params = []

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

    query += " ORDER BY pub_year, title"
    query += " LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

            # get total count ignoring limit/offset
            count_query = "SELECT COUNT(*) FROM documents"
            if filters:
                count_query += " WHERE " + " AND ".join(filters)
            cur.execute(count_query, params[:-2])
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
