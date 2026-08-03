#!/usr/bin/env python

from pathlib import Path
import re
import xml.etree.ElementTree as etree
import unicodedata

import lib.eebo_ocr_fixes as eebo_ocr_fixes


XML_FILE = Path(
    "s:/src/pamphlets/corpus/ecco_all/ecco/XML/xml-200510/K000934.000.xml"
)

HEADER_ROOT = Path(
    "s:/src/pamphlets/corpus/ecco_all/ecco/headers"
)


ALLOWED_PUNCT = r"\.\,\;\:\!\?\'\"\-\(\)"


def safe_text(x):
    return x.text.strip() if x is not None and x.text else None


def normalize_early_modern(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(\w)[’‘ʼ′´](\w)", r"\1'\2", text)
    text = text.replace("ſ", "s")

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    text = re.sub(r"-\s*", " ", text)
    text = re.sub(r"\bv(?=[aeiou])", "u", text)
    text = re.sub(r"\bj(?=[aeiou])", "i", text)
    text = re.sub(r"tv\b", "ty", text)

    text = re.sub(rf"[^{ALLOWED_PUNCT}a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def render_text(node):
    parts = []

    if node.text:
        parts.append(node.text)

    for child in node:
        if child.tag.upper() == "GAP":
            extent = child.attrib.get("EXTENT", "")
            m = re.search(r"(\d+)", extent)
            n = int(m.group(1)) if m else 1
            parts.append("_" * n)
        else:
            parts.append(render_text(child))

        if child.tail:
            parts.append(child.tail)

    return "".join(parts)


def find_header(doc_id):
    matches = list(
        HEADER_ROOT.rglob(f"{doc_id}.hdr")
    )

    if not matches:
        raise FileNotFoundError(
            f"No header for {doc_id}"
        )

    return matches[0]


def extract_header_metadata(header_path):

    tree = etree.parse(str(header_path))

    title = tree.findtext(".//TITLESTMT/TITLE")
    author = tree.findtext(".//TITLESTMT/AUTHOR")

    pub = tree.find(".//SOURCEDESC//PUBLICATIONSTMT")

    publisher = None
    pub_place = None
    date_raw = None

    if pub is not None:
        publisher = pub.findtext("PUBLISHER")
        pub_place = pub.findtext("PUBPLACE")
        date_raw = pub.findtext("DATE")

    year = None
    if date_raw:
        m = re.search(r"\b(\d{4})\b", date_raw)
        if m:
            year = int(m.group(1))

    return {
        "title": title,
        "author": author,
        "publisher": publisher,
        "pub_place": pub_place,
        "date_raw": date_raw,
        "pub_year": year,
    }


def process_ecco_file(xml_path):

    tree = etree.parse(str(xml_path))

    idg = tree.find(".//EEBO/IDG")

    if idg is None:
        raise RuntimeError("No IDG")

    doc_id = idg.attrib["ID"]

    header_path = find_header(doc_id)

    metadata = extract_header_metadata(header_path)

    body = tree.findall(".//EEBO/TEXT/BODY")

    raw_text = " ".join(
        render_text(b)
        for b in body
    )

    normalized = normalize_early_modern(
        eebo_ocr_fixes.apply_ocr_fixes(raw_text)
    )

    tokens = re.findall(
        r"\w+|[^\w\s]",
        normalized
    )

    return {
        "doc_id": doc_id,
        **metadata,
        "lang": tree.find(".//EEBO/TEXT").attrib.get("LANG"),
        "chars": len(normalized),
        "tokens": len(tokens),
        "sample": tokens[:100],
    }


if __name__ == "__main__":

    result = process_ecco_file(XML_FILE)

    for k, v in result.items():
        print("\n", k)
        print(v)
