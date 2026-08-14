# test_search_space_unit.py

from __future__ import annotations

import pytest

from retrieval.models import SearchSpace


def test_accepts_single_year_and_scale() -> None:
    space = SearchSpace(
        years=1625,
        scale="local",
    )

    assert space.years == (1625, 1625)
    assert space.scale == ("local",)


def test_accepts_year_range_and_multiple_scales() -> None:
    space = SearchSpace(
        years=(1625, 1725),
        scale=("local", "medium"),
    )

    assert space.years == (1625, 1725)
    assert space.scale == ("local", "medium")


def test_accepts_none_for_all_years_and_scales() -> None:
    space = SearchSpace(
        years=None,
        scale=None,
    )

    assert space.years is None
    assert space.scale is None


def test_accepts_single_scale_as_tuple() -> None:
    space = SearchSpace(
        years=1625,
        scale=("local",),
    )

    assert space.scale == ("local",)


def test_rejects_year_range_with_wrong_length() -> None:
    with pytest.raises(ValueError, match="exactly two years"):
        SearchSpace(
            years=(1625,),
            scale="local",
        )


def test_rejects_reversed_year_range() -> None:
    with pytest.raises(
        ValueError,
        match="ascending order",
    ):
        SearchSpace(
            years=(1725, 1625),
            scale="local",
        )


def test_rejects_non_integer_year_range() -> None:
    with pytest.raises(
        TypeError,
        match="year range must contain integers",
    ):
        SearchSpace(
            years=("1625", "1725"),
            scale="local",
        )


def test_rejects_invalid_years_type() -> None:
    with pytest.raises(
        TypeError,
        match="years must be an int",
    ):
        SearchSpace(
            years="1625",
            scale="local",
        )


def test_rejects_empty_scale_tuple() -> None:
    with pytest.raises(
        ValueError,
        match="at least one scale",
    ):
        SearchSpace(
            years=1625,
            scale=(),
        )


def test_rejects_invalid_scale() -> None:
    with pytest.raises(
        ValueError,
        match="invalid scales",
    ):
        SearchSpace(
            years=1625,
            scale="nonsense",
        )


def test_rejects_invalid_scale_in_tuple() -> None:
    with pytest.raises(
        ValueError,
        match="invalid scales",
    ):
        SearchSpace(
            years=1625,
            scale=("local", "nonsense"),
        )


def test_rejects_non_string_scale_selection() -> None:
    with pytest.raises(
        TypeError,
        match="scale selection must contain strings",
    ):
        SearchSpace(
            years=1625,
            scale=("local", 123),
        )
