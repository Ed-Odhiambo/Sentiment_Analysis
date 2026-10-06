"""Controlled retraining workflow for the sentiment-analysis model."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.preprocessing import clean_review
from src.predict import MODEL_PATH, VECTORIZER_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_reviews_cleaned.csv"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)

BACKUP_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "backups"
)

RETRAINING_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "retraining"
)

RETRAINING_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retraining"
)

VALID_SENTIMENTS = {
    "negative",
    "neutral",
    "positive",
}

TEXT_COLUMN_CANDIDATES = [
    "review",
    "text",
    "feedback",
    "comment",
    "comments",
    "customer_review",
    "detailed review",
    "content",
    "message",
]

LABEL_COLUMN_CANDIDATES = [
    "sentiment",
    "label",
    "sentiment_label",
    "airline_sentiment",
    "classification",
    "category",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


def timestamp_value() -> str:
    """Return a timestamp suitable for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_baseline_dataset() -> pd.DataFrame:
    """Load the original cleaned training dataset."""
    if not BASELINE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Baseline dataset not found:\n{BASELINE_DATA_PATH}"
        )

    dataframe = pd.read_csv(BASELINE_DATA_PATH)

    required_columns = {
        "review",
        "clean_review",
        "sentiment",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "The baseline dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.dropna(
        subset=[
            "review",
            "clean_review",
            "sentiment",
        ]
    ).copy()

    dataframe["sentiment"] = (
        dataframe["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return dataframe.reset_index(drop=True)


def read_labeled_file(
    file_path: Path,
) -> pd.DataFrame:
    """Read a labelled CSV or JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"New labelled dataset not found:\n{file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".csv":
        try:
            dataframe = pd.read_csv(
                file_path,
                encoding="utf-8",
            )
        except UnicodeDecodeError:
            dataframe = pd.read_csv(
                file_path,
                encoding="ISO-8859-1",
            )

    elif extension == ".json":
        try:
            dataframe = pd.read_json(file_path)
        except ValueError:
            dataframe = pd.read_json(
                file_path,
                lines=True,
            )

    else:
        raise ValueError(
            "Only CSV and JSON retraining files are supported."
        )

    if dataframe.empty:
        raise ValueError(
            "The retraining file contains no records."
        )

    return dataframe


def find_column(
    dataframe: pd.DataFrame,
    requested_column: str | None,
    candidates: list[str],
    column_description: str,
) -> str:
    """Find a requested or commonly named dataframe column."""
    if requested_column is not None:
        if requested_column not in dataframe.columns:
            raise ValueError(
                f"The requested {column_description} column "
                f"'{requested_column}' was not found.\n"
                f"Available columns: {list(dataframe.columns)}"
            )

        return requested_column

    normalized_columns = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

    raise ValueError(
        f"The {column_description} column could not be identified.\n"
        f"Available columns: {list(dataframe.columns)}"
    )


def prepare_new_labeled_data(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    """Validate and clean newly labelled reviews."""
    prepared = pd.DataFrame(
        {
            "review": dataframe[text_column],
            "sentiment": dataframe[label_column],
        }
    )

    prepared["review"] = (
        prepared["review"]
        .astype("string")
        .str.strip()
    )

    prepared["sentiment"] = (
        prepared["sentiment"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    prepared = prepared.dropna(
        subset=[
            "review",
            "sentiment",
        ]
    ).copy()

    invalid_sentiments = (
        set(prepared["sentiment"].unique())
        - VALID_SENTIMENTS
    )

    if invalid_sentiments:
        raise ValueError(
            "Invalid sentiment labels found: "
            f"{sorted(invalid_sentiments)}.\n"
            "Allowed labels are negative, neutral and positive."
        )

    prepared["clean_review"] = (
        prepared["review"]
        .apply(clean_review)
    )

    prepared = prepared[
        prepared["clean_review"]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    prepared = prepared.drop_duplicates(
        subset=[
            "clean_review",
            "sentiment",
        ]
    )

    if prepared.empty:
        raise ValueError(
            "No usable labelled reviews remained after cleaning."
        )

    return prepared[
        [
            "review",
            "clean_review",
            "sentiment",
        ]
    ].reset_index(drop=True)


def remove_conflicting_labels(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove review text that appears with more than one sentiment label.

    Conflicting labels introduce ambiguous training examples.
    """
    label_counts = (
        dataframe.groupby("clean_review")["sentiment"]
        .nunique()
    )

    conflicting_reviews = set(
        label_counts[
            label_counts > 1
        ].index
    )

    cleaned = dataframe[
        ~dataframe["clean_review"].isin(
            conflicting_reviews
        )
    ].copy()

    cleaned = cleaned.drop_duplicates(
        subset=["clean_review"],
        keep="last",
    )

    return (
        cleaned.reset_index(drop=True),
        len(conflicting_reviews),
    )


def create_vectorizer(
    feature_name: str,
) -> Any:
    """Create the selected feature-extraction method."""
    common_parameters = {
        "max_features": 20_000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
    }

    if feature_name == "bag_of_words":
        return CountVectorizer(
            **common_parameters,
        )

    if feature_name == "tfidf":
        return TfidfVectorizer(
            **common_parameters,
            sublinear_tf=True,
        )

    raise ValueError(
        f"Unsupported feature extractor: {feature_name}"
    )


def create_model(
    model_name: str,
) -> Any:
    """Create the selected classification model."""
    if model_name == "multinomial_naive_bayes":
        return MultinomialNB(
            alpha=0.5,
        )

    if model_name == "linear_svm":
        return LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5_000,
        )

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


def calculate_metrics(
    true_labels: pd.Series,
    predicted_labels: Any,
) -> dict[str, float]:
    """Calculate evaluation metrics."""
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def evaluate_model(
    model: Any,
    vectorizer: Any,
    reviews: pd.Series,
    sentiments: pd.Series,
) -> dict[str, float]:
    """Evaluate a trained model on review text."""
    features = vectorizer.transform(reviews)

    predictions = model.predict(features)

    return calculate_metrics(
        sentiments,
        predictions,
    )


def load_metadata() -> dict[str, Any]:
    """Load details about the current production model."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata was not found:\n{METADATA_PATH}"
        )

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as metadata_file:
        return json.load(metadata_file)


def backup_current_artifacts(
    timestamp: str,
) -> None:
    """Back up the existing production model files."""
    BACKUP_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts = [
        MODEL_PATH,
        VECTORIZER_PATH,
        METADATA_PATH,
    ]

    for artifact in artifacts:
        if artifact.exists():
            backup_name = (
                f"{artifact.stem}_{timestamp}"
                f"{artifact.suffix}"
            )

            shutil.copy2(
                artifact,
                BACKUP_DIRECTORY / backup_name,
            )


def save_retraining_report(
    report: dict[str, Any],
    timestamp: str,
) -> Path:
    """Save one retraining-attempt report."""
    RETRAINING_REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        RETRAINING_REPORT_DIRECTORY
        / f"retraining_report_{timestamp}.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=4,
        )

    return report_path


def print_metrics(
    heading: str,
    metrics: dict[str, float],
) -> None:
    """Print model metrics in a readable format."""
    print(f"\n{heading}")
    print("-" * 60)

    print(
        f"Accuracy:        {metrics['accuracy']:.4f}"
    )

    print(
        f"Macro precision: {metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro recall:    {metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1-score:  {metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:     {metrics['weighted_f1']:.4f}"
    )


def run_retraining(
    input_path: Path,
    text_column: str | None,
    label_column: str | None,
    minimum_improvement: float,
    force: bool,
) -> None:
    """Run the controlled retraining process."""
    timestamp = timestamp_value()

    metadata = load_metadata()

    feature_name = metadata["feature_extraction"]
    model_name = metadata["model"]

    baseline = load_baseline_dataset()

    baseline_train, baseline_test = train_test_split(
        baseline,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=baseline["sentiment"],
    )

    incoming_dataframe = read_labeled_file(
        input_path
    )

    detected_text_column = find_column(
        dataframe=incoming_dataframe,
        requested_column=text_column,
        candidates=TEXT_COLUMN_CANDIDATES,
        column_description="review-text",
    )

    detected_label_column = find_column(
        dataframe=incoming_dataframe,
        requested_column=label_column,
        candidates=LABEL_COLUMN_CANDIDATES,
        column_description="sentiment-label",
    )

    new_data = prepare_new_labeled_data(
        dataframe=incoming_dataframe,
        text_column=detected_text_column,
        label_column=detected_label_column,
    )

    # Prevent new records from exposing the original test set
    # during candidate-model training.
    test_review_text = set(
        baseline_test["clean_review"]
    )

    overlap_mask = new_data[
        "clean_review"
    ].isin(test_review_text)

    excluded_test_overlaps = int(
        overlap_mask.sum()
    )

    training_new_data = new_data[
        ~overlap_mask
    ].copy()

    if training_new_data.empty:
        raise ValueError(
            "All new reviews overlap with the original test set. "
            "No safe retraining records remain."
        )

    candidate_training_data = pd.concat(
        [
            baseline_train[
                [
                    "review",
                    "clean_review",
                    "sentiment",
                ]
            ],
            training_new_data,
        ],
        ignore_index=True,
    )

    (
        candidate_training_data,
        candidate_conflicts,
    ) = remove_conflicting_labels(
        candidate_training_data
    )

    current_model = joblib.load(MODEL_PATH)
    current_vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    current_metrics = evaluate_model(
        model=current_model,
        vectorizer=current_vectorizer,
        reviews=baseline_test["clean_review"],
        sentiments=baseline_test["sentiment"],
    )

    candidate_vectorizer = create_vectorizer(
        feature_name
    )

    candidate_model = create_model(
        model_name
    )

    candidate_training_features = (
        candidate_vectorizer.fit_transform(
            candidate_training_data["clean_review"]
        )
    )

    candidate_model.fit(
        candidate_training_features,
        candidate_training_data["sentiment"],
    )

    candidate_metrics = evaluate_model(
        model=candidate_model,
        vectorizer=candidate_vectorizer,
        reviews=baseline_test["clean_review"],
        sentiments=baseline_test["sentiment"],
    )

    required_score = (
        current_metrics["macro_f1"]
        + minimum_improvement
    )

    promoted = (
        force
        or candidate_metrics["macro_f1"]
        >= required_score
    )

    deployment_records = 0
    deployment_conflicts = 0

    if promoted:
        deployment_data = pd.concat(
            [
                baseline[
                    [
                        "review",
                        "clean_review",
                        "sentiment",
                    ]
                ],
                new_data,
            ],
            ignore_index=True,
        )

        (
            deployment_data,
            deployment_conflicts,
        ) = remove_conflicting_labels(
            deployment_data
        )

        deployment_records = len(
            deployment_data
        )

        deployment_vectorizer = create_vectorizer(
            feature_name
        )

        deployment_model = create_model(
            model_name
        )

        deployment_features = (
            deployment_vectorizer.fit_transform(
                deployment_data["clean_review"]
            )
        )

        deployment_model.fit(
            deployment_features,
            deployment_data["sentiment"],
        )

        backup_current_artifacts(timestamp)

        joblib.dump(
            deployment_model,
            MODEL_PATH,
        )

        joblib.dump(
            deployment_vectorizer,
            VECTORIZER_PATH,
        )

        updated_metadata = {
            "selected_configuration": (
                f"{feature_name} + {model_name}"
            ),
            "feature_extraction": feature_name,
            "model": model_name,
            "selection_metric": "macro_f1",
            "metrics": {
                key: round(value, 6)
                for key, value
                in candidate_metrics.items()
            },
            "labels": sorted(VALID_SENTIMENTS),
            "training_records": deployment_records,
            "evaluation_records": len(
                baseline_test
            ),
            "new_labeled_records": len(new_data),
            "source_file": input_path.name,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "retrained_at": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        with METADATA_PATH.open(
            "w",
            encoding="utf-8",
        ) as metadata_file:
            json.dump(
                updated_metadata,
                metadata_file,
                indent=4,
            )

        RETRAINING_DATA_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        deployment_data.to_csv(
            RETRAINING_DATA_DIRECTORY
            / f"training_data_{timestamp}.csv",
            index=False,
            encoding="utf-8",
        )

    report = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "source_file": str(input_path),
        "detected_text_column": detected_text_column,
        "detected_label_column": detected_label_column,
        "feature_extraction": feature_name,
        "model": model_name,
        "baseline_records": len(baseline),
        "new_records_received": len(
            incoming_dataframe
        ),
        "new_records_usable": len(new_data),
        "new_records_used_for_candidate": len(
            training_new_data
        ),
        "excluded_test_overlaps": (
            excluded_test_overlaps
        ),
        "candidate_label_conflicts_removed": (
            candidate_conflicts
        ),
        "deployment_label_conflicts_removed": (
            deployment_conflicts
        ),
        "deployment_records": deployment_records,
        "minimum_required_improvement": (
            minimum_improvement
        ),
        "forced_promotion": force,
        "promoted": promoted,
        "current_metrics": current_metrics,
        "candidate_metrics": candidate_metrics,
    }

    report_path = save_retraining_report(
        report=report,
        timestamp=timestamp,
    )

    print("\nCONTROLLED MODEL RETRAINING")
    print("=" * 60)
    print(f"Source file: {input_path}")
    print(
        f"Detected review column: "
        f"{detected_text_column}"
    )
    print(
        f"Detected label column: "
        f"{detected_label_column}"
    )
    print(
        f"Usable new labelled reviews: "
        f"{len(new_data):,}"
    )
    print(
        f"New reviews used for candidate training: "
        f"{len(training_new_data):,}"
    )
    print(
        f"Original test-set overlaps excluded: "
        f"{excluded_test_overlaps:,}"
    )
    print(
        f"Candidate training records: "
        f"{len(candidate_training_data):,}"
    )

    print_metrics(
        "CURRENT MODEL PERFORMANCE",
        current_metrics,
    )

    print_metrics(
        "CANDIDATE MODEL PERFORMANCE",
        candidate_metrics,
    )

    score_change = (
        candidate_metrics["macro_f1"]
        - current_metrics["macro_f1"]
    )

    print("\nRETRAINING DECISION")
    print("-" * 60)
    print(f"Macro F1 change: {score_change:+.4f}")
    print(
        f"Minimum required improvement: "
        f"{minimum_improvement:.4f}"
    )

    if promoted:
        print(
            "Decision: CANDIDATE MODEL PROMOTED"
        )
        print(
            "The previous production model was backed up."
        )
        print(
            f"Deployment training records: "
            f"{deployment_records:,}"
        )
    else:
        print(
            "Decision: CURRENT MODEL RETAINED"
        )
        print(
            "The candidate did not meet the required "
            "performance threshold."
        )

    print(f"\nRetraining report saved to:\n{report_path}")


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the sentiment model using newly "
            "labelled customer reviews."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to the labelled CSV or JSON file."
        ),
    )

    parser.add_argument(
        "--text-column",
        default=None,
        help=(
            "Optional name of the review-text column."
        ),
    )

    parser.add_argument(
        "--label-column",
        default=None,
        help=(
            "Optional name of the sentiment-label column."
        ),
    )

    parser.add_argument(
        "--minimum-improvement",
        type=float,
        default=0.0,
        help=(
            "Required macro F1 improvement before model "
            "replacement. Default: 0.0."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace the model even when the candidate "
            "does not improve performance."
        ),
    )

    arguments = parser.parse_args()

    if arguments.minimum_improvement < 0:
        parser.error(
            "--minimum-improvement cannot be negative."
        )

    return arguments


def main() -> None:
    """Run model retraining from command-line arguments."""
    arguments = parse_arguments()

    run_retraining(
        input_path=Path(arguments.input),
        text_column=arguments.text_column,
        label_column=arguments.label_column,
        minimum_improvement=(
            arguments.minimum_improvement
        ),
        force=arguments.force,
    )


if __name__ == "__main__":
    main()