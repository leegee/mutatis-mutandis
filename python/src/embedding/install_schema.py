from pathlib import Path

from lib.corpus_db import get_connection


def install_schema() -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    sql = schema_path.read_text(encoding="utf-8")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)

        conn.commit()


if __name__ == "__main__":
    install_schema()
