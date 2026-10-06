"""Tests for the controlled model-retraining utilities."""

import pandas as pd
import pytest

from src.retrain_model import (
    LABEL_COLUMN_CANDIDATES,
    TEXT_COLUMN_CANDIDATES,
    find_column,
    prepare_new_labeled_data,
)


def test_detects_retraining_columns() -> None:
    """Common review and sentiment columns should be detected."""
    dataframe = pd.DataFrame(
        {
            "feedback": [
                "Excellent service.",
            ],
            "label": [
                "positive",
            ],
        }
    )

    text_column = find_column(
        dataframe=dataframe,
        requested_column=None,
        candidates=TEXT_COLUMN_CANDIDATES,
        column_description="review-text",
    )

    label_column = find_column(
        dataframe=dataframe,
        requested_column=None,
        candidates=LABEL_COLUMN_CANDIDATES,
        column_description="sentiment-label",
    )

    assert text_column == "feedback"
    assert label_column == "label"


def test_prepares_valid_labeled_reviews() -> None:
    """Valid labelled reviews should be cleaned successfully."""
    dataframe = pd.DataFrame(
        {
            "review": [
                "@Airline The service was not good!",
                "The staff were very helpful.",
            ],
            "sentiment": [
                "negative",
                "positive",
            ],
        }
    )

    prepared = prepare_new_labeled_data(
        dataframe=dataframe,
        text_column="review",
        label_column="sentiment",
    )

    assert len(prepared) == 2
    assert "clean_review" in prepared.columns
    assert "not" in prepared.iloc[0]["clean_review"]


def test_rejects_invalid_sentiment_label() -> None:
    """Only positive, neutral and negative labels are allowed."""
    dataframe = pd.DataFrame(
        {
            "review": [
                "The service was satisfactory.",
            ],
            "sentiment": [
                "mixed",
            ],
        }
    )

    with pytest.raises(ValueError):
        prepare_new_labeled_data(
            dataframe=dataframe,
            text_column="review",
            label_column="sentiment",
        )


def test_requested_missing_column_raises_error() -> None:
    """An explicitly requested missing column should fail clearly."""
    dataframe = pd.DataFrame(
        {
            "feedback": [
                "Good service.",
            ]
        }
    )

    with pytest.raises(ValueError):
        find_column(
            dataframe=dataframe,
            requested_column="review",
            candidates=TEXT_COLUMN_CANDIDATES,
            column_description="review-text",
        )