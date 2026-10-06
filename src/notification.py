"""Notifications for high levels of negative customer sentiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALERT_LOG_PATH = (
    PROJECT_ROOT
    / "reports"
    / "negative_review_alerts.log"
)

DEFAULT_NEGATIVE_THRESHOLD = 50.0
DEFAULT_MINIMUM_REVIEWS = 5


@dataclass(frozen=True)
class NegativeReviewAlert:
    """Summary of a negative-sentiment threshold check."""

    total_reviews: int
    negative_reviews: int
    negative_percentage: float
    threshold_percentage: float
    triggered: bool
    message: str


def assess_negative_reviews(
    dataframe: pd.DataFrame,
    sentiment_column: str = "predicted_sentiment",
    threshold_percentage: float = DEFAULT_NEGATIVE_THRESHOLD,
    minimum_reviews: int = DEFAULT_MINIMUM_REVIEWS,
) -> NegativeReviewAlert:
    """
    Check whether negative reviews exceed the configured threshold.

    Only rows with processing_status='processed' are counted when that
    column is available.
    """
    if sentiment_column not in dataframe.columns:
        raise ValueError(
            f"Sentiment column '{sentiment_column}' was not found."
        )

    results = dataframe.copy()

    if "processing_status" in results.columns:
        results = results[
            results["processing_status"].eq("processed")
        ].copy()

    results = results.dropna(
        subset=[sentiment_column]
    )

    total_reviews = len(results)

    negative_reviews = int(
        results[sentiment_column]
        .astype(str)
        .str.lower()
        .eq("negative")
        .sum()
    )

    negative_percentage = (
        negative_reviews / total_reviews * 100
        if total_reviews > 0
        else 0.0
    )

    triggered = (
        total_reviews >= minimum_reviews
        and negative_percentage >= threshold_percentage
    )

    if total_reviews < minimum_reviews:
        message = (
            f"No alert generated. Only {total_reviews:,} processed "
            f"reviews were available; at least {minimum_reviews:,} "
            "are required."
        )

    elif triggered:
        message = (
            "HIGH NEGATIVE SENTIMENT ALERT: "
            f"{negative_reviews:,} of {total_reviews:,} reviews "
            f"({negative_percentage:.1f}%) were classified as "
            f"negative. The configured threshold is "
            f"{threshold_percentage:.1f}%."
        )

    else:
        message = (
            f"No alert generated. {negative_reviews:,} of "
            f"{total_reviews:,} reviews "
            f"({negative_percentage:.1f}%) were negative."
        )

    return NegativeReviewAlert(
        total_reviews=total_reviews,
        negative_reviews=negative_reviews,
        negative_percentage=negative_percentage,
        threshold_percentage=threshold_percentage,
        triggered=triggered,
        message=message,
    )


def record_alert(
    alert: NegativeReviewAlert,
    source_name: str | None = None,
) -> None:
    """Append a triggered alert to the project alert log."""
    if not alert.triggered:
        return

    ALERT_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    source_text = source_name or "Unknown source"

    log_entry = (
        f"[{timestamp}] "
        f"Source: {source_text} | "
        f"Negative: {alert.negative_reviews:,}/"
        f"{alert.total_reviews:,} "
        f"({alert.negative_percentage:.1f}%) | "
        f"Threshold: {alert.threshold_percentage:.1f}%\n"
    )

    with ALERT_LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as log_file:
        log_file.write(log_entry)


def notify_negative_reviews(
    dataframe: pd.DataFrame,
    source_name: str | None = None,
    sentiment_column: str = "predicted_sentiment",
    threshold_percentage: float = DEFAULT_NEGATIVE_THRESHOLD,
    minimum_reviews: int = DEFAULT_MINIMUM_REVIEWS,
) -> NegativeReviewAlert:
    """Assess feedback, display a notification and record alerts."""
    alert = assess_negative_reviews(
        dataframe=dataframe,
        sentiment_column=sentiment_column,
        threshold_percentage=threshold_percentage,
        minimum_reviews=minimum_reviews,
    )

    print(alert.message)

    if alert.triggered:
        record_alert(
            alert=alert,
            source_name=source_name,
        )

    return alert