#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as etree

import lib.eebo_config as config
import lib.corpus_db as corpus_db


def extract_doc_id(xml_path: Path) -> str | None:
    try:
        tree = etree.parse(str(xml_path))
    except Exception:
        return None

    elem = tree.find(".//HEADER//IDNO[@TYPE='DLPS']")
    if elem is None or not elem.text:
        return None

    return elem.text.strip()


def compute_relative_path(xml_path: Path) -> str:
    return xml_path.relative_to(config.XML_ROOT_DIR).as_posix()


def main():
    xml_files = list(Path(config.XML_ROOT_DIR).rglob("*.xml"))

    updates = []

    for path in xml_files:
        doc_id = extract_doc_id(path)
        if not doc_id:
            continue

        new_filepath = compute_relative_path(path)
        updates.append((new_filepath, doc_id))

    print(f"Prepared {len(updates)} updates")

    with corpus_db.get_connection() as conn:
        cur = conn.cursor()

        for filepath, doc_id in updates:
            cur.execute(
                """
                UPDATE documents
                SET filepath = %s
                WHERE doc_id = %s
                """,
                (filepath, doc_id),
            )

        conn.commit()


if __name__ == "__main__":
    main()
