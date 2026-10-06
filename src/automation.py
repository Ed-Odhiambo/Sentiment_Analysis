"""Automatically process incoming customer-review files."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.notification import notify_negative_reviews
from src.predict import SentimentPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INCOMING_DIRECTORY = PROJECT_ROOT / "data" / "incoming"

PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "automated"
)

ARCHIVE_DIRECTORY = PROJECT_ROOT / "data" / "archive"

FAILED_DIRECTORY = ARCHIVE_DIRECTORY / "failed"

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".json",
}

TEXT_COLUMN_CANDIDATES = [
    "review",
    "text",
    "feedback",
    "comment",
    "comments",
    "customer_review",
    "customer feedback",
    "detailed review",
    "content",
    "message",
]


def ensure_directories() -> None:
    """Create all folders used by the automation process."""
    for directory in [
        INCOMING_DIRECTORY,
        PROCESSED_DIRECTORY,
        ARCHIVE_DIRECTORY,
        FAILED_DIRECTORY,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def read_review_file(
    file_path: Path,
) -> pd.DataFrame:
    """Read a CSV or JSON customer-review file."""
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
            f"Unsupported file type: {extension}"
        )

    if dataframe.empty:
        raise ValueError(
            "The incoming file does not contain any records."
        )

    return dataframe


def find_text_column(
    dataframe: pd.DataFrame,
) -> str:
    """Identify the column containing the customer-review text."""
    normalized_columns = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

    available_columns = ", ".join(
        str(column)
        for column in dataframe.columns
    )

    raise ValueError(
        "A customer-review text column could not be identified. "
        "Use a column name such as review, text, feedback, "
        f"comment or Detailed Review. Available columns: "
        f"{available_columns}"
    )


def timestamp_value() -> str:
    """Return a timestamp suitable for filenames."""
    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def save_processed_results(
    dataframe: pd.DataFrame,
    source_file: Path,
) -> Path:
    """Save classified reviews to the automated-results folder."""
    timestamp = timestamp_value()

    output_name = (
        f"{source_file.stem}_processed_{timestamp}.csv"
    )

    output_path = (
        PROCESSED_DIRECTORY
        / output_name
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return output_path


def archive_file(
    source_file: Path,
    failed: bool = False,
) -> Path:
    """Move an incoming file to the completed or failed archive."""
    destination_directory = (
        FAILED_DIRECTORY
        if failed
        else ARCHIVE_DIRECTORY
    )

    archive_name = (
        f"{source_file.stem}_"
        f"{timestamp_value()}"
        f"{source_file.suffix.lower()}"
    )

    archive_path = (
        destination_directory
        / archive_name
    )

    shutil.move(
        str(source_file),
        str(archive_path),
    )

    return archive_path


def process_file(
    file_path: Path,
    predictor: SentimentPredictor,
) -> Path:
    """Process one incoming customer-review file."""
    print("\n" + "=" * 70)
    print(f"Processing incoming file: {file_path.name}")

    dataframe = read_review_file(file_path)

    text_column = find_text_column(
        dataframe
    )

    print(
        f"Detected review column: {text_column}"
    )

    predictions = predictor.predict_many(
        dataframe[text_column].tolist()
    )

    results = dataframe.copy()

    results["clean_review"] = (
        predictions["clean_review"].values
    )

    results["predicted_sentiment"] = (
        predictions["sentiment"].values
    )

    results["processing_status"] = (
        predictions["processing_status"].values
    )

    output_path = save_processed_results(
        dataframe=results,
        source_file=file_path,
    )

    alert = notify_negative_reviews(
        dataframe=results,
        source_name=file_path.name,
    )

    archive_path = archive_file(
        source_file=file_path,
    )

    processed_count = int(
        results["processing_status"]
        .eq("processed")
        .sum()
    )

    skipped_count = int(
        results["processing_status"]
        .ne("processed")
        .sum()
    )

    print(f"Processed reviews: {processed_count:,}")
    print(f"Skipped reviews:   {skipped_count:,}")
    print(f"Alert triggered:   {alert.triggered}")
    print(f"Results saved to:  {output_path}")
    print(f"Original archived: {archive_path}")

    return output_path


def scan_incoming_directory(
    predictor: SentimentPredictor,
) -> int:
    """Process every supported file currently in the incoming folder."""
    incoming_files = sorted(
        file_path
        for file_path in INCOMING_DIRECTORY.iterdir()
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )

    if not incoming_files:
        print(
            "No incoming CSV or JSON files were found."
        )
        return 0

    successful_files = 0

    for file_path in incoming_files:
        try:
            process_file(
                file_path=file_path,
                predictor=predictor,
            )

            successful_files += 1

        except Exception as error:
            print(
                f"\nUnable to process {file_path.name}: "
                f"{error}"
            )

            try:
                failed_path = archive_file(
                    source_file=file_path,
                    failed=True,
                )

                print(
                    f"Failed file moved to: {failed_path}"
                )

            except Exception as archive_error:
                print(
                    "The failed file could not be archived: "
                    f"{archive_error}"
                )

    return successful_files


def monitor_incoming_directory(
    interval_seconds: float,
) -> None:
    """Continuously monitor the incoming folder for new files."""
    ensure_directories()

    predictor = SentimentPredictor()

    print("\nAUTOMATED SENTIMENT PROCESSOR")
    print("=" * 70)
    print(f"Monitoring folder: {INCOMING_DIRECTORY}")
    print(
        f"Checking every {interval_seconds:g} seconds."
    )
    print("Press Control+C to stop.\n")

    try:
        while True:
            scan_incoming_directory(
                predictor=predictor,
            )

            time.sleep(
                interval_seconds
            )

    except KeyboardInterrupt:
        print(
            "\nAutomation stopped by the user."
        )


def run_once() -> None:
    """Process all files currently available and then stop."""
    ensure_directories()

    predictor = SentimentPredictor()

    processed_files = scan_incoming_directory(
        predictor=predictor,
    )

    print(
        f"\nAutomation run completed. "
        f"{processed_files:,} file(s) processed successfully."
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Automatically classify customer reviews "
            "from incoming CSV or JSON files."
        )
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Process available files once and then stop."
        ),
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help=(
            "Seconds between folder checks when monitoring "
            "continuously. Default: 5."
        ),
    )

    arguments = parser.parse_args()

    if arguments.interval <= 0:
        parser.error(
            "--interval must be greater than zero."
        )

    return arguments


def main() -> None:
    """Run the incoming-review automation."""
    arguments = parse_arguments()

    if arguments.once:
        run_once()
    else:
        monitor_incoming_directory(
            interval_seconds=arguments.interval,
        )


if __name__ == "__main__":
    main()