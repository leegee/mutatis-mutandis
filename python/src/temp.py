# Check for the apoalypse

from pathlib import Path
import lancedb

REPO_ROOT = Path(__file__).resolve().parents[1]
LANCE_DIR = REPO_ROOT.parent / "out" / "indexes" / "lance_albinism"

TARGET_DOC = "Campbell-Apocalypse-Export-05-English-Prose-tei"

db = lancedb.connect(str(LANCE_DIR))
names = db.list_tables().tables

print("Target:", TARGET_DOC)
print("Lance directory:", LANCE_DIR)
print("Tables:", len(names))

total_matches = 0

for name in sorted(names):
    table = db.open_table(name)
    df = table.to_arrow().to_pandas()

    matches = df[df["doc_id"].astype(str) == TARGET_DOC]

    if not matches.empty:
        total_matches += len(matches)
        print(f"\n{name}: {len(matches)} passages")
        print(matches[[
            "passage_id", "corpus", "doc_id", "pub_year",
            "word_start", "word_end", "token_start", "token_end"
        ]].to_string(index=False))

print(f"\nTotal indexed passages for target: {total_matches}")
