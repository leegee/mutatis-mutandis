# parse_eebo_tei.py

from pathlib import Path
from lxml import etree
import re

# Paths
INPUT_DIR = Path(r"../eebo_all/eebo_phase1/P4_XML_TCP")
OUT_DIR = Path(r"../out/plain")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_early_modern(text: str) -> str:
    text = text.lower()

    # long s → s
    text = text.replace("ſ", "s")

    # remove line-break hyphenation
    text = re.sub(r'-/s+', '', text)

    # v → u at word start when followed by vowel (vnity → unity)
    text = re.sub(r'/bv(?=[aeiou])', 'u', text)

    # j → i at word start when followed by vowel (iames is correct EME)
    text = re.sub(r'/bj(?=[aeiou])', 'i', text)

    # collapse whitespace
    text = re.sub(r'/s+', ' ', text)

    return text.strip()


def extract_text_from_tei(xml_path: Path) -> str:
    """
    Extracts all textual content from a TEI XML file.
    Works for full texts and fragments.
    """
    try:
        tree = etree.parse(str(xml_path))
    except Exception as e:
        print(f"[ERROR] Failed to parse {xml_path.name}: {e}")
        return ""

    # Collect all text nodes
    texts = tree.xpath("//text()")

    # Join and return
    return " ".join(t.strip() for t in texts if t.strip())


def main():
    xml_files = list(INPUT_DIR.glob("*.xml"))
    print(f"Found {len(xml_files)} TEI files")

    processed = 0

    for xml_file in xml_files:
        raw_text = extract_text_from_tei(xml_file)

        if not raw_text:
            continue

        normalized = normalize_early_modern(raw_text)

        if len(normalized) < 100:
            # Skip extremely small fragments
            continue

        out_file = OUT_DIR / f"{xml_file.stem}.txt"
        out_file.write_text(normalized, encoding="utf-8")

        processed += 1

        if processed % 500 == 0:
            print(f"Processed {processed} files")

    print(f"Done. Wrote {processed} text files to {OUT_DIR}")


if __name__ == "__main__":
    main()
