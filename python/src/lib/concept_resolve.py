# from difflib import get_close_matches
from lib.corpus_config import CONCEPT_SETS

def resolve_concepts(*, concept, forms=None, false_positives=None, concept_sets=CONCEPT_SETS):
    if not concept:
        return concept_sets.items()

    concept_name = concept.upper()

    fps = (
        {f.strip().lower() for f in false_positives.split(",")}
        if false_positives else None
    )

    if forms:
        forms_set = {f.strip().lower() for f in forms.split(",")}
        return [(concept_name, {
            "forms": forms_set,
            "false_positives": fps or set(),
        })]

    if concept_name in concept_sets:
        base = concept_sets[concept_name]
        result = dict(base)
        if fps is not None:
            result["false_positives"] = fps
        return [(concept_name, result)]

    # Ad-hoc: concept name itself becomes the (single) form
    return [(concept_name, {
        "forms": {concept_name.lower()},
        "false_positives": fps or set(),
    })]

