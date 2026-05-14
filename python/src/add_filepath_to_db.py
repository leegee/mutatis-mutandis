from pathlib import Path
from lib.eebo_db import get_connection
from lib.eebo_config import XML_ROOT_DIR, OUT_DIR
from lib.eebo_logging import logger

def build_file_map(base_dir: str):
    """
    Build mapping: filename (no extension) -> full absolute path
    """
    file_map = {}

    for path in Path(base_dir).rglob("*.xml"):
        if path.is_file():
            key = path.name.split(".", 1)[0]
            if key in file_map:
                logger.info(f"Warning: duplicate filename detected for '{key}'")
            else:
                file_map[key] = str(path.resolve())

    return file_map


def fetch_doc_ids(conn):
    """
    Get all document IDs from DB
    """
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM documents")
        return [row[0] for row in cur.fetchall()]


def update_paths(conn, updates):
    """
    Batch update file paths in Postgres
    """
    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE documents
            SET filepath = %s
            WHERE doc_id = %s
            """,
            updates
        )
    conn.commit()


def main():
    base_dir = XML_ROOT_DIR

    logger.info("Building file index...")
    file_map = build_file_map(base_dir)

    logger.info(f"Indexed {len(file_map)} files")

    conn = get_connection()

    try:
        logger.info("Fetching document IDs from database...")
        doc_ids = fetch_doc_ids(conn)

        updates = []
        missing = []

        logger.info("Matching files...")
        for doc_id in doc_ids:
            path = file_map.get(doc_id)
            if path:
                updates.append((path, doc_id))
            else:
                missing.append(doc_id)

        logger.info(f"Matched: {len(updates)}")
        logger.info(f"Missing: {len(missing)}")

        if updates:
            logger.info("Updating database...")
            update_paths(conn, updates)
            logger.info("Update complete.")

        if missing:
            fp = OUT_DIR / "missing_docs.txt"
            with open(fp, "w") as f:
                for doc_id in missing:
                    f.write(doc_id + "\n")
            logger.info(f"Missing IDs written to {fp}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
