from psycopg import Connection


def get_processed_concepts(conn: Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT concept FROM tier2.concepts")
        return {row[0] for row in cur.fetchall()}

