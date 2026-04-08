from lib.eebo_db import get_autocommit_connection, create_tiered_token_indexes

conn = get_autocommit_connection()

with conn.cursor() as cur:
    # Drop existing views
    cur.execute("DROP MATERIALIZED VIEW IF EXISTS document_search CASCADE;")
    cur.execute("DROP MATERIALIZED VIEW IF EXISTS pamphlet_tokens CASCADE;")
    cur.execute("DROP MATERIALIZED VIEW IF EXISTS pamphlet_corpus CASCADE;")

# Recreate the views + indexes (your existing function)
create_tiered_token_indexes(conn)
