import sys

import eebo_config as config
import eebo_db

def make_slices():
    # Ensure slices directory exists
    try:
        config.SLICES_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Cannot create slices directory: {e}")
        sys.exit(1)


    cursor = eebo_db.dbh.cursor()

    # Process each slice
    for start_year, end_year in config.SLICES:
        slice_name = f"{start_year}-{end_year}"
        out_path = config.SLICES_DIR / f"{slice_name}.txt"
        print(f"[INFO] Processing slice {slice_name}")

        cursor.execute(
            "SELECT doc_id FROM documents WHERE pub_year BETWEEN ? AND ?",
            (start_year, end_year)
        )
        doc_ids = [row[0] for row in cursor.fetchall()]
        print(f"[INFO] {len(doc_ids)} documents in slice {slice_name}")

        # Concatenate text files
        texts = []
        missing_count = 0
        for doc_id in doc_ids:
            text_path = config.PLAIN_DIR / f"{doc_id}.txt"
            if text_path.exists():
                texts.append(text_path.read_text(encoding="utf-8"))
            else:
                missing_count += 1
                print(f"[WARN] Missing text file for {doc_id}")

        if missing_count:
            print(f"[WARN] {missing_count} missing files in slice {slice_name}")

        # Write concatenated slice
        try:
            out_path.write_text("\n".join(texts), encoding="utf-8")
            print(f"[DONE] Wrote slice file: {out_path} ({len(texts)} texts)")
        except Exception as e:
            print(f"[ERROR] Failed to write slice {slice_name}: {e}")

    eebo_db.dbh.close()
    print("[INFO] All slices processed.")


if __name__ == "__main__":
    make_slices()
