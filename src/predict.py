"""Load the trained model and predict sentiments for new reviews."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocessing import clean_review


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "vectorizer.pkl"

VALID_SENTIMENTS = {
    "negative",
    "neutral",
    "positive",
}


class SentimentPredictor:
    """Load and use the trained sentiment-analysis model."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        vectorizer_path: Path = VECTORIZER_PATH,
    ) -> None:
        """Load the saved model and text vectorizer."""
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)

        self.model = self._load_artifact(
            self.model_path,
            artifact_name="sentiment model",
        )

        self.vectorizer = self._load_artifact(
            self.vectorizer_path,
            artifact_name="vectorizer",
        )

        self._validate_artifacts()

    @staticmethod
    def _load_artifact(
        file_path: Path,
        artifact_name: str,
    ) -> Any:
        """Load a saved model artifact."""
        if not file_path.exists():
            raise FileNotFoundError(
                f"The {artifact_name} was not found at:\n"
                f"{file_path}\n\n"
                "Run the model training script first:\n"
                "python -m src.train_model"
            )

        try:
            return joblib.load(file_path)
        except Exception as error:
            raise RuntimeError(
                f"Unable to load the {artifact_name} from:\n"
                f"{file_path}"
            ) from error

    def _validate_artifacts(self) -> None:
        """Check that the loaded artifacts support prediction."""
        if not hasattr(self.model, "predict"):
            raise TypeError(
                "The loaded model does not provide a predict method."
            )

        if not hasattr(self.vectorizer, "transform"):
            raise TypeError(
                "The loaded vectorizer does not provide "
                "a transform method."
            )

    def predict_one(self, review: object) -> dict[str, str]:
        """
        Predict the sentiment of one review.

        Returns
        -------
        dict
            Original review, cleaned review, predicted sentiment,
            and processing status.
        """
        results = self.predict_many([review])
        result = results.iloc[0]

        if result["processing_status"] != "processed":
            raise ValueError(
                "The review is empty or contains no usable text."
            )

        return {
            "review": str(result["review"]),
            "clean_review": str(result["clean_review"]),
            "sentiment": str(result["sentiment"]),
            "processing_status": str(
                result["processing_status"]
            ),
        }

    def predict_many(
        self,
        reviews: Iterable[object],
    ) -> pd.DataFrame:
        """
        Predict sentiments for multiple reviews efficiently.

        Empty reviews are retained in the output but marked as skipped.
        """
        review_values = list(reviews)

        if not review_values:
            return pd.DataFrame(
                columns=[
                    "review",
                    "clean_review",
                    "sentiment",
                    "processing_status",
                ]
            )

        original_reviews = [
            "" if review is None else str(review)
            for review in review_values
        ]

        cleaned_reviews = [
            clean_review(review)
            for review in review_values
        ]

        results = pd.DataFrame(
            {
                "review": original_reviews,
                "clean_review": cleaned_reviews,
            }
        )

        valid_mask = (
            results["clean_review"]
            .astype(str)
            .str.strip()
            .ne("")
        )

        results["sentiment"] = pd.NA
        results["processing_status"] = "skipped_empty"

        if valid_mask.any():
            features = self.vectorizer.transform(
                results.loc[valid_mask, "clean_review"]
            )

            predictions = self.model.predict(features)

            unexpected_predictions = (
                set(predictions) - VALID_SENTIMENTS
            )

            if unexpected_predictions:
                raise ValueError(
                    "The model returned unexpected sentiment labels: "
                    f"{sorted(unexpected_predictions)}"
                )

            results.loc[
                valid_mask,
                "sentiment",
            ] = predictions

            results.loc[
                valid_mask,
                "processing_status",
            ] = "processed"

        return results

    def predict_file(
        self,
        input_path: str | Path,
        text_column: str = "review",
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Predict sentiments for reviews contained in a CSV or JSON file.

        Parameters
        ----------
        input_path:
            Path to a CSV or JSON file.

        text_column:
            Name of the column containing review text.

        output_path:
            Optional location for saving the processed results.
        """
        input_file = Path(input_path)

        if not input_file.exists():
            raise FileNotFoundError(
                f"Input file not found:\n{input_file}"
            )

        file_extension = input_file.suffix.lower()

        if file_extension == ".csv":
            dataframe = pd.read_csv(input_file)
        elif file_extension == ".json":
            dataframe = pd.read_json(input_file)
        else:
            raise ValueError(
                "Unsupported file format. "
                "Only CSV and JSON files are accepted."
            )

        if text_column not in dataframe.columns:
            raise ValueError(
                f"The column '{text_column}' was not found.\n"
                f"Available columns: {list(dataframe.columns)}"
            )

        predictions = self.predict_many(
            dataframe[text_column].tolist()
        )

        result = dataframe.copy()

        result["clean_review"] = predictions[
            "clean_review"
        ]

        result["predicted_sentiment"] = predictions[
            "sentiment"
        ]

        result["processing_status"] = predictions[
            "processing_status"
        ]

        if output_path is not None:
            output_file = Path(output_path)

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if output_file.suffix.lower() == ".json":
                result.to_json(
                    output_file,
                    orient="records",
                    indent=4,
                )
            else:
                result.to_csv(
                    output_file,
                    index=False,
                    encoding="utf-8",
                )

        return result


def main() -> None:
    """Demonstrate predictions using several sample reviews."""
    predictor = SentimentPredictor()

    sample_reviews = [
        "The service was excellent and the staff were very helpful.",
        "My flight was delayed and nobody provided any explanation.",
        "I travelled from Boston to Chicago yesterday.",
        "The experience was not good and I will never use them again.",
        "",
    ]

    predictions = predictor.predict_many(
        sample_reviews
    )

    print("\nSAMPLE SENTIMENT PREDICTIONS")
    print("=" * 80)

    print(
        predictions.to_string(
            index=False,
        )
    )


if __name__ == "__main__":
    main()