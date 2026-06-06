#!/usr/bin/env python3

import re
import xml.etree.ElementTree as etree


def render_text(node):
    """
    Preserve EEBO GAP elements as underscores.

    Invariant:
    - Every missing letter represented by a GAP contributes one "_".
    - Tree structure is flattened into continuous running text.
    """

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


xml = """
<BODY>
    liberty

    libe<GAP DESC="illegible"
              RESP="pdcc"
              EXTENT="1 letter"
              DISP="•"/>ty

    go<GAP EXTENT="3 letters"/>d

    <P>
        The kingd<GAP EXTENT="1 letter"/>m of England.
    </P>

    <P>
        A<GAP EXTENT="7 letters"/>Z
    </P>
</BODY>
"""

tree = etree.parse("s:/src/pamphlets/eebo_all/eebo_phase1/P4_XML_TCP/A57609.P4.xml")
body = tree.find(".//EEBO//TEXT//BODY")

print(render_text(body))

# root = etree.fromstring(xml)
# print("ORIGINAL XML:")
# print(xml)
# print("\nRENDERED:")
# print(render_text(root))
