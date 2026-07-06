"""
slicing.py

Defines evaluation scopes that select the set of event IDs that participate in one clustering run.

"""

from __future__ import annotations


def build_scope(
    substrate,
    scope_type: str = "all",
    scope_value=None,
    concept: str | None = None,
):
    """
    Return the event IDs participating in one clustering run.

    Parameters
    ----------
    substrate
        Read-only semantic substrate.

    scope_type
        "all", "year", or "concept".

    scope_value
        Value associated with the scope type.

    concept
        Optional concept filter. May be combined with the "year" scope.
    """

    if scope_type == "all":
        return substrate.get_events(concept=concept)

    if scope_type == "year":
        return substrate.get_events(
            year=int(scope_value),
            concept=concept,
        )

    if scope_type == "concept":
        return substrate.get_events(
            concept=str(scope_value),
        )

    raise ValueError(f"Unknown scope_type: {scope_type}")
