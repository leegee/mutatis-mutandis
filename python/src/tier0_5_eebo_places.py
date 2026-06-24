#!/usr/bin/env python3

import re
import unicodedata
from collections import defaultdict

from lib.eebo_db import get_connection
from lib.eebo_logging import logger


# ---------------------------------------------------------------------------
# VARIANT → CANONICAL MAP (lowercase keys)
# ---------------------------------------------------------------------------

PLACE_MAP = {
    "london": "London",
    "londres": "London",
    "llundain": "London",
    "oxford": "Oxford",
    "oxon": "Oxford",
    "yorke": "York",
    "york": "York",
    "edinburgh": "Edinburgh",
    "edenburgh": "Edinburgh",
    "amsterdam": "Amsterdam",
    "amsterodam": "Amsterdam",
    "antwerp": "Antwerp",
    "antwerpen": "Antwerp",
    "bruxelles": "Brussels",
    "brussels": "Brussels",
    "dublin": "Dublin",
    "waterford": "Waterford",
    "leiden": "Leiden",
    "leyden": "Leiden",
    "ghent": "Ghent",
    "gant": "Ghent",
    "paris": "Paris",
    "rouen": "Rouen",
    "douai": "Douai",
    "saint omer": "Saint-Omer",
    "hamburgh": "Hamburg",
    "hamburg": "Hamburg",
    "rotterdam":"Rotterdam",
    "roterdame":"Rotterdam",
    "rhydychen": "Oxford",
    "norwich": "Norwich",
    "nottingham": "Nottingham",

    # Cambridge
    "cambridge": "Cambridge",
    "cantabrigi": "Cambridge",
    "cantabrigiae": "Cambridge",
    "cambridg": "Cambridge",

    # Boston
    "boston": "Boston, Mass.",

    # Aberdeen
    "aberdeen": "Aberdeen",
    "aberdene": "Aberdeen",

    # Edinburgh
    "edinburg": "Edinburgh",
    "edinbvrgh": "Edinburgh",
    "edenborough": "Edinburgh",
    "edenbrough": "Edinburgh",
    "edenburg": "Edinburgh",
    "edenburgi": "Edinburgh",
    "edynburgh": "Edinburgh",

    # Hague
    "hague": "The Hague",
    "hagh": "The Hague",
    "haghe": "The Hague",
    "hage": "The Hague",
    "gravenhagh": "The Hague",

    # Waterford
    "vvaterford": "Waterford",

    # Dublin
    "dvblin": "Dublin",
    "dubdin": "Dublin",

    # Leiden
    "leyden": "Leiden",

    # Delft
    "delft": "Delft",
    "delf": "Delft",
    "delff": "Delft",
    "delph": "Delft",

    # Brussels
    "bruxells": "Brussels",

    # Middelburg
    "middelburgh": "Middelburg",
    "middleburg": "Middelburg",

    # Antwerp
    "anwerp": "Antwerp",
    "antwarp": "Antwerp",

    # Ghent
    "ghent": "Ghent",

    # Douai
    "doway": "Douai",
    "dovvay": "Douai",

    # Aberdeen
    "aberderdene": "Aberdeen",   # if present elsewhere
    "abderdene": "Aberdeen",
    "aberden": "Aberdeen",
    "aberdoni": "Aberdeen",
    "aberdoniis": "Aberdeen",
    "abredeis": "Aberdeen",
    "abredoni": "Aberdeen",

    # Amsterdam
    "amstelodami": "Amsterdam",
    "amstelredam": "Amsterdam",
    "amsterledam": "Amsterdam",
    "amstersdam": "Amsterdam",

    # Antwerp
    "antworp": "Antwerp",

    # Saint-Omer
    "audomari": "Saint-Omer",
    "omer": "Saint-Omer",
    "omers": "Saint-Omer",

    # Beauvais
    "beauvais": "Beauvais",

    # Breda
    "breda": "Breda",

    # Bristol
    "bristol": "Bristol",
    "bristoll": "Bristol",

    # Bruges
    "bruges": "Bruges",

    # Caen
    "caen": "Caen",

    # Colchester
    "colechester": "Colchester",

    # Cologne
    "cologne": "Cologne",

    # Constantinople
    "constantinople": "Constantinople",

    # Cork
    "corck": "Cork",
    "corcke": "Cork",
    "cork": "Cork",
    "corke": "Cork",

    # Dordrecht
    "dordrecht": "Dordrecht",
    "dort": "Dordrecht",

    # Douai
    "douay": "Douai",

    # Durham
    "durham": "Durham",

    # Edinburgh
    "edinbourg": "Edinburgh",
    "edinburh": "Edinburgh",
    "edingburgh": "Edinburgh",

    # Exeter
    "exeter": "Exeter",
    "exon": "Exeter",

    # Falmouth
    "falmouth": "Falmouth",

    # Franckfort
    "franckfort": "Frankfurt",

    # Geneva
    "genevah": "Geneva",

    # Glasgow
    "glasgow": "Glasgow",
    "glasgu": "Glasgow",

    # Haarlem
    "haarlem": "Haarlem",

    # Hague
    "hag": "The Hague",
    "haye": "The Hague",

    # Hamburg
    "hamborough": "Hamburg",

    # Ipswich
    "ipswich": "Ipswich",

    # Kilkenny
    "kilkenny": "Kilkenny",

    # Leith
    "leith": "Leith",

    # Liège
    "liege": "Liège",
    "lvyck": "Liège",

    # Lille
    "lille": "Lille",

    # London variants
    "llyndain": "London",
    "lodnon": "London",
    "lodon": "London",
    "lond": "London",
    "londdn": "London",
    "londen": "London",
    "londgn": "London",
    "londinensis": "London",
    "londini": "London",
    "londn": "London",
    "londnon": "London",
    "londod": "London",
    "londou": "London",
    "londre": "London",
    "lonkon": "London",
    "lonndo": "London",
    "lonndon": "London",
    "lonnon": "London",
    "lonon": "London",
    "lonpuo": "London",
    "loudon": "London",
    "loydon": "London",
    "ludd": "London",
    "lundain": "London",
    "ondon": "London",

    # Loreto
    "loreto": "Loreto",

    # Louvain / Leuven
    "louain": "Leuven",
    "louanii": "Leuven",
    "louvain": "Leuven",
    "lovain": "Leuven",

    # Newcastle
    "newcastle": "Newcastle",

    # Oxford
    "oxen": "Oxford",   # OCR variant

    # Reading
    "reading": "Reading",

    # Rochester
    "rochester": "Rochester",

    # Rouen
    "roan": "Rouen",
    "rouan": "Rouen",

    # Salisbury
    "salisbury": "Salisbury",

    # Savoy
    "savoy": "Savoy, London",

    # Shrewsbury
    "shrewsbury": "Shrewsbury",

    # Swarthmore
    "swarthmore": "Swarthmore",

    # Troyes
    "troyes": "Troyes",

    # Waterford
    "warerford": "Waterford",

    # Westminster
    "westminster": "Westminster",

    # Worcester
    "worcester": "Worcester",

    "aire": "Aire-sur-la-Lys",      # often linked with Saint-Omer printing
    "buckden": "Buckden",
    "chelmsford": "Chelmsford",
    "chester": "Chester",
    "cirencester": "Cirencester",
    "finsbury": "Finsbury, London",
    "gateshead": "Gateshead",
    "pomadie": "Pomadie",           # check source; may be a pseudonymous imprint
    "sicilia": "Sicily",
    "smithfield": "Smithfield, London",
    "wakefield": "Wakefield",

    "gateside": "Gateside, Scotland",
    "carolopoli": "Charleville, France",
    "gottenberge": "Gothenburg, Denmark",
    "edinb": "Edinburgh",
    "edin": "Edinburgh",
    "england": "england",
    "scotland": "Scotland",
    "ireland": "Ireland",
    "europ": "Europe",
    "europe": "Europe",
    "franca": "France",
    "france": "France",
    "holland": "Netherlands",
    "netherlands": "Netherlands",
    "lon": "London",
    "abroad": "Sine Loco",
    "s.l.": "Sine Loco",
    "[s.l.": "Sine Loco",
    "[s.l": "Sine Loco",
    "[s.l,": "Sine Loco",
    "s. l.": "Sine Loco",
    "[S.1.": "Sine Loco",
    "n.p.": "Sine Loco",
    "n. p.": "Sine Loco",
    "doüay": "Douai, France",
    "roüen": "Rouen",
    "lo[ndon": "London",
    "L[ondo]n": "London",
    "[liège?": "Liège",
    "[lo]ndon": "London",
    "nod-nol.": "Nödinge-Nol, Västra Götaland",
}


def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()

    # remove combining diacritics (Doüay → douay, roüen → rouen)
    s = "".join(c for c in s if not unicodedata.combining(c))

    # normalize punctuation variants
    s = re.sub(r"[\[\]().,:;?']", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


CLEAN_PLACE_MAP = {
    norm_key(k): v
    for k, v in PLACE_MAP.items()
}


QUALIFIERS = {
    "england",
    "scotland",
    "ireland",
    "london",
    "netherlands",
    "holland",
    "mass",
    "massachusetts",
    "yorkshire",
    "oxfordshire",
    "cambridgeshire",
}




UNKNOWN = defaultdict(str)


def normalize_pub_place(raw: str | None):
    if not raw:
        return None

    s = unicodedata.normalize("NFKC", raw).lower()

    found = []

    # -------------------------------------------------------
    # pure variant scan (your requested approach)
    # -------------------------------------------------------
    key = norm_key(s)

    for variant, canonical in CLEAN_PLACE_MAP.items():
        if variant in key:
            found.append(canonical)

    # dedupe preserving order
    found = list(dict.fromkeys(found))

    # -------------------------------------------------------
    # if multiple, drop London (or other low-priority)
    # -------------------------------------------------------
    if len(found) > 1:
        found = [
            f for f in found
            if norm_key(f) not in QUALIFIERS
        ] or found

    # -------------------------------------------------------
    # logging unknowns (only for iteration)
    # -------------------------------------------------------
    if not found:
        for token in re.findall(r"[a-z]+", s):
            if token not in CLEAN_PLACE_MAP:
                UNKNOWN[token] += f"{token} "
        return None

    return found


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    conn = get_connection(application_name="normalize_places")

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS place_normalization (
                raw_place TEXT PRIMARY KEY,
                normalized_places TEXT[]
            )
        """)

        cur.execute("""
            SELECT DISTINCT pub_place
            FROM documents
            WHERE pub_place IS NOT NULL
            ORDER BY pub_place
        """)

        rows = cur.fetchall()
        logger.info("Processing %d rows", len(rows))

        total = 0
        matched = 0
        unmatched = []

        for (raw,) in rows:
            total += 1

            norm = normalize_pub_place(raw)

            cur.execute("""
                INSERT INTO place_normalization (raw_place, normalized_places)
                VALUES (%s, %s)
                ON CONFLICT (raw_place)
                DO UPDATE SET normalized_places = EXCLUDED.normalized_places
            """, (raw, norm))

            if norm:
                matched += 1
            else:
                unmatched.append(raw)

            logger.info("%s -> %s", raw, norm)

    conn.commit()

    # -------------------------------------------------------
    # coverage
    # -------------------------------------------------------

    logger.info("")
    logger.info("=== COVERAGE ===")
    logger.info("Total rows:     %d", total)
    logger.info("Matched rows:   %d", matched)
    logger.info("Unmatched rows: %d", len(unmatched))
    logger.info("Coverage:       %.1f%%", 100.0 * matched / total)

    # -------------------------------------------------------
    # unmatched examples
    # -------------------------------------------------------

    logger.info("")
    logger.info("=== UNMATCHED ROWS ===")

    for raw in unmatched[:200]:
        logger.info(raw)

    # -------------------------------------------------------
    # token diagnostics
    # -------------------------------------------------------

    # logger.info("")
    # logger.info("=== UNKNOWN TOKENS ===")

    # for token, values in sorted(
    #     UNKNOWN.items(),
    #     key=lambda x: len(x[1]),
    #     reverse=True,
    # )[:200]:
    #     logger.info("%s -> %s", token, " ".join(values))


if __name__ == "__main__":
    main()