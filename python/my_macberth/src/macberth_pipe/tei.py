from lxml import etree
from dataclasses import dataclass
from typing import Dict
import os

@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    meta: Dict[str, str]

def load_tei(path: str) -> Doc:
    tree = etree.parse(path)
    text_nodes = tree.xpath("//text()")
    text = " ".join([t.strip() for t in text_nodes if t.strip()])
    meta = {
        "filename": os.path.basename(path),
        "title": tree.findtext(".//title") or "",
        "author": tree.findtext(".//author") or ""
    }
    return Doc(id=os.path.basename(path), text=text, meta=meta)
