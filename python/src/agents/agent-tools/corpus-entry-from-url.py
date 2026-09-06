from __future__ import annotations

"""
agent-tools/corpus-entry-from-url.py - download a classic-literature.net work and save its textual content as UTF-8.

Output layout:

    OUTPUT_DIR/
        <first-letter>/
            <first-two-letters>/
                <url-filename>.txt

For example:

    the-strange-adventures-of-andrew-battell-of-leigh-in-angola-and-the-adjoining-regions

becomes:

    OUTPUT_DIR/t/th/
        the-strange-adventures-of-andrew-battell-of-leigh-in-angola-and-the-adjoining-regions.txt

The output preserves paragraph and heading boundaries.

Usage:

    python fetch_classic_literature.py https://classic-literature.net/andrew-battell-1560-1613/the-strange-adventures-of-andrew-battell-of-leigh-in-angola-and-the-adjoining-regions/

or:

    python fetch_classic_literature.py https://classic-literature.net/andrew-battell-1560-1613/the-strange-adventures-of-andrew-battell-of-leigh-in-angola-and-the-adjoining-regions/  --output-dir D:/corpus/classic-literature
"""

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("D:/corpus/classic-literature")

USER_AGENT = (
    "Mutatis-Mutandis corpus acquisition tool "
    "(research use; contact information may be added later)"
)

REQUEST_TIMEOUT = 60


# HTML elements whose textual contents we want to preserve as separate
# structural units.
BLOCK_ELEMENTS = {
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
}


# ---------------------------------------------------------------------------
# URL / filename handling
# ---------------------------------------------------------------------------

def filename_from_url(url: str) -> str:
    """
    Extract the filename-like component from a URL.

    Example:

        https://example.com/foo/bar/

    ->

        bar
    """
    parsed = urlparse(url)

    path = parsed.path.rstrip("/")

    if not path:
        raise ValueError(f"URL has no usable path: {url}")

    name = path.split("/")[-1]

    # Be conservative about what can become a filesystem filename.
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    name = name.strip("-.")

    if not name:
        raise ValueError(f"Could not derive a filename from URL: {url}")

    return name


def output_path(url: str, output_dir: Path) -> Path:
    """
    Construct the deterministic output path.

    The directory structure is based on the first one and two characters
    of the URL filename.
    """
    filename = filename_from_url(url)

    if len(filename) < 2:
        raise ValueError(
            f"URL filename must contain at least two characters: {filename!r}"
        )

    first = filename[0].lower()
    first_two = filename[:2].lower()

    directory = output_dir / first / first_two

    return directory / f"{filename}.txt"


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalise whitespace without destroying word boundaries."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def extract_text(html: str) -> str:
    """
    Extract the textual content of the work.

    Headings and paragraphs are retained as separate lines/paragraphs.

    We deliberately avoid using soup.get_text() over the entire document,
    because that would also collect navigation, menus, metadata, etc.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove things which can contain visible but non-textual/navigation
    # material.
    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()

    # classic-literature.net currently presents the book as a sequence of
    # headings and paragraphs. Find the main textual container where possible.
    #
    # If the site changes its markup, falling back to <body> still gives us
    # a useful acquisition rather than silently producing an empty file.
    main = soup.find("main")

    if main is None:
        main = soup.find("article")

    if main is None:
        main = soup.body

    if main is None:
        raise ValueError("Could not find a document body in the downloaded HTML")

    blocks: list[str] = []

    for element in main.find_all(BLOCK_ELEMENTS):
        text = clean_text(element.get_text(" ", strip=True))

        if text:
            blocks.append(text)

    if not blocks:
        raise ValueError("No textual blocks were extracted from the page")

    # Consecutive HTML block elements become paragraphs/lines in the output.
    #
    # We use blank lines between blocks so that paragraph boundaries survive
    # later processing.
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download(url: str) -> str:
    """Download URL and return its HTML."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    # requests normally detects this correctly, but classic literary sites
    # can occasionally have imperfect HTTP charset metadata. apparent_encoding
    # provides a useful fallback.
    if not response.encoding:
        response.encoding = response.apparent_encoding

    return response.text


# ---------------------------------------------------------------------------
# Main acquisition operation
# ---------------------------------------------------------------------------

def fetch_and_store(url: str, output_dir: Path) -> Path:
    """
    Download, extract and store one work.

    Returns the path of the resulting text file.
    """
    destination = output_path(url, output_dir)

    print(f"Downloading: {url}")
    print(f"Destination: {destination}")

    html = download(url)

    text = extract_text(html)

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Explicit UTF-8, no platform-dependent encoding.
    destination.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )

    print(f"Wrote {len(text):,} characters")

    return destination


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a classic-literature.net work as UTF-8 text."
    )

    parser.add_argument(
        "url",
        help="URL of the work to download",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    fetch_and_store(
        url=args.url,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
