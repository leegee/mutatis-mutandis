import duckdb

root = r"D:\src\mutatis-mutandis\out\events"

con = duckdb.connect()

rows = con.execute(f"""
    SELECT event_id, COUNT(*) AS n
    FROM read_parquet(
        '{root}/**/*.parquet',
        hive_partitioning=true,
        union_by_name=true
    )
    WHERE year = 1530
      AND emb_local IS NOT NULL
    GROUP BY event_id
    HAVING COUNT(*) > 1
    ORDER BY n DESC
    LIMIT 20
""").fetchall()

for row in rows:
    print(row)
