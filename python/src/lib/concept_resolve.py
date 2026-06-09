from difflib import get_close_matches
from lib.eebo_config import CONCEPT_SETS

def resolve_concepts(*, concept, false_positives, concept_sets=CONCEPT_SETS):
    if not concept:
        return concept_sets.items()

    concept_name = concept.upper()

    if hasattr(args, "forms") and forms:
        # Ad-hoc concept — no config entry required
        forms = {f.strip() for f in forms.split(",")}
        false_positives = set()
        if hasattr(args, "false_positives") and false_positives:
            false_positives = {f.strip() for f in false_positives.split(",")}
        return [(concept_name, {"forms": forms, "false_positives": false_positives})]

    # Named concept from config
    if concept_name not in concept_sets:
        suggestion = get_close_matches(concept_name, concept_sets.keys(), n=1)
        msg = f"Unknown concept '{concept_name}'."
        if suggestion:
            msg += f" Did you mean '{suggestion[0]}'?"
        raise ValueError(msg)

    return [(concept_name, concept_sets[concept_name])]
