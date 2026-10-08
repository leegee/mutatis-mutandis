import lib.corpus_db as corpus_db

with corpus_db.get_connection() as conn:
    corpus_db.create_views(conn)
    corpus_db.create_tiered_token_indexes(conn)

corpus_db.create_concurrent_indexes()