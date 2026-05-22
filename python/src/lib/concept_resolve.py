from difflib import get_close_matches
from lib.eebo_config import CONCEPT_SETS

def resolve_concepts(args, concept_sets = CONCEPT_SETS):
    """
    Returns an iterable of (concept_name, concept_spec)
    based on CLI args, with validation.
    """
    if not args.concept:
        return concept_sets.items()

    args.concept = args.concept.upper()

    concept_name = args.concept

    if concept_name not in concept_sets:
        from difflib import get_close_matches

        suggestion = get_close_matches(concept_name, concept_sets.keys(), n=1)

        msg = f"Unknown concept '{concept_name}'."
        if suggestion:
            msg += f" Did you mean '{suggestion[0]}'?"

        raise ValueError(msg)

    return [(concept_name, concept_sets[concept_name])]
