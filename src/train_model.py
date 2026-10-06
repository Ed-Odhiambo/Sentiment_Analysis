"""Train, compare and save sentiment classification models."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_reviews_cleaned.csv"
)

MODEL_DIRECTORY = PROJECT_ROOT / "models"
REPORT_DIRECTORY = PROJECT_ROOT / "reports"
FIGURE_DIRECTORY = REPORT_DIRECTORY / "figures"

MODEL_PATH = MODEL_DIRECTORY / "sentiment_model.pkl"
VECTORIZER_PATH = MODEL_DIRECTORY / "vectorizer.pkl"
METADATA_PATH = MODEL_DIRECTORY / "model_metadata.json"
COMPARISON_PATH = REPORT_DIRECTORY / "model_comparison.csv"

LABELS = ["negative", "neutral", "positive"]

TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load and validate the cleaned sentiment dataset."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {file_path}\n"
            "Run: python -m src.clean_dataset"
        )

    dataframe = pd.read_csv(file_path)

    required_columns = {
        "review",
        "clean_review",
        "sentiment",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The cleaned dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.dropna(
        subset=["clean_review", "sentiment"]
    ).copy()

    dataframe["clean_review"] = (
        dataframe["clean_review"]
        .astype(str)
        .str.strip()
    )

    dataframe["sentiment"] = (
        dataframe["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe = dataframe[
        dataframe["clean_review"] != ""
    ].copy()

    unexpected_labels = (
        set(dataframe["sentiment"].unique()) - set(LABELS)
    )

    if unexpected_labels:
        raise ValueError(
            "Unexpected sentiment labels found: "
            f"{sorted(unexpected_labels)}"
        )

    return dataframe.reset_index(drop=True)


def split_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create stratified training and testing datasets."""
    reviews = dataframe["clean_review"]
    sentiments = dataframe["sentiment"]

    return train_test_split(
        reviews,
        sentiments,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=sentiments,
    )


def create_vectorizers() -> dict[str, Any]:
    """Create Bag-of-Words and TF-IDF vectorizers."""
    common_parameters = {
        "max_features": 20_000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
    }

    return {
        "bag_of_words": CountVectorizer(
            **common_parameters,
        ),
        "tfidf": TfidfVectorizer(
            **common_parameters,
            sublinear_tf=True,
        ),
    }


def create_models() -> dict[str, Any]:
    """Create the machine-learning models to compare."""
    return {
        "multinomial_naive_bayes": MultinomialNB(
            alpha=0.5,
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=5_000,
        ),
    }


def create_slug(value: str) -> str:
    """Convert a model name into a safe filename."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value)
    return slug.strip("_").lower()


def calculate_metrics(
    true_labels: pd.Series,
    predicted_labels: Any,
) -> dict[str, float]:
    """Calculate performance metrics for one model."""
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
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def save_classification_report(
    true_labels: pd.Series,
    predicted_labels: Any,
    configuration_name: str,
) -> None:
    """Save a detailed classification report as a CSV file."""
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = pd.DataFrame(report).transpose()

    filename = (
        f"classification_report_"
        f"{create_slug(configuration_name)}.csv"
    )

    output_path = REPORT_DIRECTORY / filename

    report_dataframe.to_csv(
        output_path,
        index=True,
    )


def save_confusion_matrix(
    true_labels: pd.Series,
    predicted_labels: Any,
    configuration_name: str,
) -> None:
    """Save a confusion-matrix figure for one model."""
    figure, axis = plt.subplots(figsize=(7, 6))

    ConfusionMatrixDisplay.from_predictions(
        true_labels,
        predicted_labels,
        labels=LABELS,
        display_labels=[
            "Negative",
            "Neutral",
            "Positive",
        ],
        values_format=",d",
        ax=axis,
    )

    axis.set_title(
        f"Confusion Matrix\n{configuration_name}"
    )

    figure.tight_layout()

    filename = (
        f"confusion_matrix_"
        f"{create_slug(configuration_name)}.png"
    )

    output_path = FIGURE_DIRECTORY / filename

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def print_split_summary(
    training_labels: pd.Series,
    testing_labels: pd.Series,
) -> None:
    """Print the sizes and class distributions of each split."""
    print("\nDATA SPLIT")
    print("-" * 60)
    print(f"Training records: {len(training_labels):,}")
    print(f"Testing records:  {len(testing_labels):,}")

    print("\nTraining-class distribution:")
    print(training_labels.value_counts().reindex(LABELS))

    print("\nTesting-class distribution:")
    print(testing_labels.value_counts().reindex(LABELS))


def save_best_model(
    model: Any,
    vectorizer: Any,
    configuration_name: str,
    metrics: dict[str, float],
    training_records: int,
    testing_records: int,
) -> None:
    """Save the selected model, vectorizer and supporting metadata."""
    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH,
    )

    feature_method, model_name = configuration_name.split(
        " + ",
        maxsplit=1,
    )

    metadata = {
        "selected_configuration": configuration_name,
        "feature_extraction": feature_method,
        "model": model_name,
        "selection_metric": "macro_f1",
        "metrics": {
            key: round(value, 6)
            for key, value in metrics.items()
        },
        "labels": LABELS,
        "training_records": training_records,
        "testing_records": testing_records,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
        )


def main() -> None:
    """Train all configurations and save the best-performing model."""
    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_dataset(DATA_PATH)

    (
        reviews_train,
        reviews_test,
        sentiments_train,
        sentiments_test,
    ) = split_dataset(dataframe)

    print("\nSENTIMENT MODEL TRAINING")
    print("=" * 60)
    print(f"Complete dataset: {len(dataframe):,} reviews")

    print_split_summary(
        sentiments_train,
        sentiments_test,
    )

    vectorizers = create_vectorizers()

    comparison_results: list[dict[str, Any]] = []

    best_score = -1.0
    best_model = None
    best_vectorizer = None
    best_configuration = ""
    best_metrics: dict[str, float] = {}

    for vectorizer_name, vectorizer in vectorizers.items():
        print("\n" + "=" * 60)
        print(
            f"Fitting vectorizer: {vectorizer_name}"
        )

        training_features = vectorizer.fit_transform(
            reviews_train
        )

        testing_features = vectorizer.transform(
            reviews_test
        )

        print(
            "Vocabulary size: "
            f"{len(vectorizer.vocabulary_):,}"
        )

        models = create_models()

        for model_name, model in models.items():
            configuration_name = (
                f"{vectorizer_name} + {model_name}"
            )

            print("\nTraining:")
            print(configuration_name)

            model.fit(
                training_features,
                sentiments_train,
            )

            predicted_sentiments = model.predict(
                testing_features
            )

            metrics = calculate_metrics(
                sentiments_test,
                predicted_sentiments,
            )

            result = {
                "feature_extraction": vectorizer_name,
                "model": model_name,
                **metrics,
            }

            comparison_results.append(result)

            print(
                f"Accuracy:        {metrics['accuracy']:.4f}"
            )
            print(
                "Macro precision: "
                f"{metrics['macro_precision']:.4f}"
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

            save_classification_report(
                sentiments_test,
                predicted_sentiments,
                configuration_name,
            )

            save_confusion_matrix(
                sentiments_test,
                predicted_sentiments,
                configuration_name,
            )

            if metrics["macro_f1"] > best_score:
                best_score = metrics["macro_f1"]
                best_model = model
                best_vectorizer = vectorizer
                best_configuration = configuration_name
                best_metrics = metrics

    comparison_dataframe = pd.DataFrame(
        comparison_results
    )

    comparison_dataframe = comparison_dataframe.sort_values(
        by="macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    comparison_dataframe.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    if best_model is None or best_vectorizer is None:
        raise RuntimeError(
            "No model was successfully trained."
        )

    save_best_model(
        model=best_model,
        vectorizer=best_vectorizer,
        configuration_name=best_configuration,
        metrics=best_metrics,
        training_records=len(reviews_train),
        testing_records=len(reviews_test),
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    display_columns = [
        "feature_extraction",
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    print(
        comparison_dataframe[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nBEST MODEL")
    print("-" * 60)
    print(f"Configuration: {best_configuration}")
    print(f"Macro F1-score: {best_score:.4f}")

    print("\nSaved files:")
    print(MODEL_PATH)
    print(VECTORIZER_PATH)
    print(METADATA_PATH)
    print(COMPARISON_PATH)


if __name__ == "__main__":
    main()