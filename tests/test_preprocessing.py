"""Tests for the sentiment text-preprocessing functions."""

from src.preprocessing import clean_review


def test_removes_twitter_handle() -> None:
    """Twitter account names should be removed."""
    cleaned = clean_review(
        "@Airline The service was excellent."
    )

    assert "@airline" not in cleaned
    assert "service" in cleaned


def test_removes_url() -> None:
    """Website URLs should not remain in cleaned reviews."""
    cleaned = clean_review(
        "See the details at https://example.com/page"
    )

    assert "https" not in cleaned
    assert "example.com" not in cleaned


def test_preserves_negation() -> None:
    """Important negation terms should remain."""
    cleaned = clean_review(
        "The service was not good."
    )

    assert "not" in cleaned


def test_decodes_html_entities() -> None:
    """Encoded HTML characters should be decoded."""
    cleaned = clean_review(
        "The staff were kind &amp; helpful."
    )

    assert "amp" not in cleaned
    assert "staff" in cleaned
    assert "helpful" in cleaned


def test_empty_review_returns_empty_string() -> None:
    """An empty review should remain empty."""
    assert clean_review("") == ""
    assert clean_review(None) == ""