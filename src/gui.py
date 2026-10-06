"""Tkinter dashboard for the sentiment analysis application."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.predict import SentimentPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIRECTORY = PROJECT_ROOT / "data" / "processed"

DISPLAY_LIMIT = 500

# A batch alert is triggered when at least five reviews are processed
# and 50% or more of them are classified as negative.
NEGATIVE_ALERT_PERCENTAGE = 50.0
NEGATIVE_ALERT_MIN_REVIEWS = 5

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


class SentimentAnalysisApp(tk.Tk):
    """Desktop dashboard for analysing customer-review sentiment."""

    def __init__(self) -> None:
        """Initialize the application and load the trained model."""
        super().__init__()

        self.title("Customer Review Sentiment Analysis")
        self.geometry("1250x800")
        self.minsize(1050, 700)

        self.predictor = SentimentPredictor()

        self.current_results = pd.DataFrame(
            columns=[
                "review",
                "clean_review",
                "sentiment",
                "processing_status",
            ]
        )

        self.chart_canvas: FigureCanvasTkAgg | None = None

        self.positive_count = tk.StringVar(value="0")
        self.neutral_count = tk.StringVar(value="0")
        self.negative_count = tk.StringVar(value="0")
        self.total_count = tk.StringVar(value="0")

        self.single_result = tk.StringVar(
            value="Enter a customer review to analyse its sentiment."
        )

        self.status_message = tk.StringVar(
            value="Application ready."
        )

        self._configure_styles()
        self._build_interface()
        self._draw_empty_chart()

    def _configure_styles(self) -> None:
        """Configure reusable Tkinter styles."""
        style = ttk.Style(self)

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Title.TLabel",
            font=("Helvetica", 22, "bold"),
        )

        style.configure(
            "Subtitle.TLabel",
            font=("Helvetica", 11),
        )

        style.configure(
            "CardTitle.TLabel",
            font=("Helvetica", 11, "bold"),
        )

        style.configure(
            "CardValue.TLabel",
            font=("Helvetica", 24, "bold"),
        )

        style.configure(
            "Result.TLabel",
            font=("Helvetica", 14, "bold"),
        )

        style.configure(
            "Status.TLabel",
            font=("Helvetica", 10),
        )

        style.configure(
            "Treeview",
            rowheight=28,
        )

        style.configure(
            "Treeview.Heading",
            font=("Helvetica", 10, "bold"),
        )

    def _build_interface(self) -> None:
        """Create all interface sections."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        self._build_header()
        self._build_input_section()
        self._build_statistics_section()
        self._build_results_section()
        self._build_status_bar()

    def _build_header(self) -> None:
        """Create the application heading."""
        header = ttk.Frame(
            self,
            padding=(20, 16, 20, 8),
        )

        header.grid(
            row=0,
            column=0,
            sticky="ew",
        )

        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Customer Review Sentiment Analysis",
            style="Title.TLabel",
        ).grid(
            row=0,
            column=0,
            sticky="w",
        )

        ttk.Label(
            header,
            text=(
                "Classify reviews as positive, neutral or negative "
                "using the trained TF-IDF and Linear SVM model."
            ),
            style="Subtitle.TLabel",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(4, 0),
        )

    def _build_input_section(self) -> None:
        """Create manual review and file-upload controls."""
        input_frame = ttk.LabelFrame(
            self,
            text="Analyse Customer Feedback",
            padding=12,
        )

        input_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=20,
            pady=(5, 10),
        )

        input_frame.columnconfigure(0, weight=1)

        self.review_input = tk.Text(
            input_frame,
            height=4,
            wrap="word",
            font=("Helvetica", 12),
        )

        self.review_input.grid(
            row=0,
            column=0,
            columnspan=5,
            sticky="ew",
            pady=(0, 10),
        )

        ttk.Button(
            input_frame,
            text="Analyse Review",
            command=self.analyse_single_review,
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 8),
        )

        ttk.Button(
            input_frame,
            text="Upload CSV or JSON",
            command=self.upload_review_file,
        ).grid(
            row=1,
            column=1,
            sticky="w",
            padx=8,
        )

        ttk.Button(
            input_frame,
            text="Export Results",
            command=self.export_results,
        ).grid(
            row=1,
            column=2,
            sticky="w",
            padx=8,
        )

        ttk.Button(
            input_frame,
            text="Clear Dashboard",
            command=self.clear_dashboard,
        ).grid(
            row=1,
            column=3,
            sticky="w",
            padx=8,
        )

        ttk.Label(
            input_frame,
            textvariable=self.single_result,
            style="Result.TLabel",
        ).grid(
            row=2,
            column=0,
            columnspan=5,
            sticky="w",
            pady=(12, 0),
        )

    def _build_statistics_section(self) -> None:
        """Create cards showing the sentiment totals."""
        statistics_frame = ttk.Frame(
            self,
            padding=(20, 0, 20, 10),
        )

        statistics_frame.grid(
            row=2,
            column=0,
            sticky="ew",
        )

        for column in range(4):
            statistics_frame.columnconfigure(
                column,
                weight=1,
                uniform="statistics",
            )

        self._create_statistic_card(
            statistics_frame,
            column=0,
            title="Total Processed",
            value_variable=self.total_count,
        )

        self._create_statistic_card(
            statistics_frame,
            column=1,
            title="Positive",
            value_variable=self.positive_count,
        )

        self._create_statistic_card(
            statistics_frame,
            column=2,
            title="Neutral",
            value_variable=self.neutral_count,
        )

        self._create_statistic_card(
            statistics_frame,
            column=3,
            title="Negative",
            value_variable=self.negative_count,
        )

    @staticmethod
    def _create_statistic_card(
        parent: ttk.Frame,
        column: int,
        title: str,
        value_variable: tk.StringVar,
    ) -> None:
        """Create one sentiment statistic card."""
        card = ttk.LabelFrame(
            parent,
            padding=12,
        )

        card.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=5,
        )

        ttk.Label(
            card,
            text=title,
            style="CardTitle.TLabel",
        ).pack()

        ttk.Label(
            card,
            textvariable=value_variable,
            style="CardValue.TLabel",
        ).pack(
            pady=(5, 0),
        )

    def _build_results_section(self) -> None:
        """Create the review table and distribution chart."""
        results_container = ttk.Panedwindow(
            self,
            orient=tk.HORIZONTAL,
        )

        results_container.grid(
            row=3,
            column=0,
            sticky="nsew",
            padx=20,
            pady=(0, 10),
        )

        table_frame = ttk.LabelFrame(
            results_container,
            text="Processed Reviews",
            padding=8,
        )

        chart_frame = ttk.LabelFrame(
            results_container,
            text="Sentiment Distribution",
            padding=8,
        )

        results_container.add(
            table_frame,
            weight=3,
        )

        results_container.add(
            chart_frame,
            weight=2,
        )

        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.chart_frame = chart_frame
        self.chart_frame.rowconfigure(0, weight=1)
        self.chart_frame.columnconfigure(0, weight=1)

        columns = (
            "review",
            "clean_review",
            "sentiment",
            "processing_status",
        )

        self.results_table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
        )

        self.results_table.heading(
            "review",
            text="Original Review",
        )

        self.results_table.heading(
            "clean_review",
            text="Processed Text",
        )

        self.results_table.heading(
            "sentiment",
            text="Sentiment",
        )

        self.results_table.heading(
            "processing_status",
            text="Status",
        )

        self.results_table.column(
            "review",
            width=340,
            minwidth=220,
        )

        self.results_table.column(
            "clean_review",
            width=280,
            minwidth=180,
        )

        self.results_table.column(
            "sentiment",
            width=100,
            anchor="center",
        )

        self.results_table.column(
            "processing_status",
            width=110,
            anchor="center",
        )

        vertical_scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.results_table.yview,
        )

        horizontal_scrollbar = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.results_table.xview,
        )

        self.results_table.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )

        self.results_table.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        vertical_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        horizontal_scrollbar.grid(
            row=1,
            column=0,
            sticky="ew",
        )

    def _build_status_bar(self) -> None:
        """Create the status message area."""
        status_bar = ttk.Frame(
            self,
            padding=(20, 4, 20, 10),
        )

        status_bar.grid(
            row=4,
            column=0,
            sticky="ew",
        )

        ttk.Label(
            status_bar,
            textvariable=self.status_message,
            style="Status.TLabel",
        ).pack(
            anchor="w",
        )

    def analyse_single_review(self) -> None:
        """Classify a manually entered review."""
        review = self.review_input.get(
            "1.0",
            tk.END,
        ).strip()

        if not review:
            messagebox.showwarning(
                "Review Required",
                "Enter a customer review before analysing it.",
            )
            return

        try:
            result = self.predictor.predict_one(review)
        except Exception as error:
            messagebox.showerror(
                "Prediction Error",
                str(error),
            )
            return

        result_dataframe = pd.DataFrame([result])

        self.current_results = pd.concat(
            [
                self.current_results,
                result_dataframe,
            ],
            ignore_index=True,
        )

        sentiment = result["sentiment"]
        emoji = self._sentiment_emoji(sentiment)

        self.single_result.set(
            f"{emoji} Predicted sentiment: {sentiment.upper()}"
        )

        self.status_message.set(
            "Review processed successfully."
        )

        self._update_dashboard()

    def upload_review_file(self) -> None:
        """Load and automatically classify reviews from CSV or JSON."""
        selected_file = filedialog.askopenfilename(
            title="Select a customer review file",
            filetypes=[
                ("CSV files", "*.csv"),
                ("JSON files", "*.json"),
                ("All supported files", "*.csv *.json"),
            ],
        )

        if not selected_file:
            return

        file_path = Path(selected_file)

        try:
            dataframe = self._read_input_file(file_path)
            text_column = self._find_text_column(dataframe)

            predictions = self.predictor.predict_many(
                dataframe[text_column].tolist()
            )

            self.current_results = predictions

        except Exception as error:
            messagebox.showerror(
                "File Processing Error",
                str(error),
            )
            return

        processed_count = (
            self.current_results[
                "processing_status"
            ]
            .eq("processed")
            .sum()
        )

        self.single_result.set(
            f"Processed file: {file_path.name}"
        )

        self.status_message.set(
            f"{processed_count:,} reviews were processed. "
            f"The table displays up to {DISPLAY_LIMIT:,} records."
        )

        self._update_dashboard()
        self._check_negative_review_alert()

        messagebox.showinfo(
            "Processing Complete",
            f"{processed_count:,} reviews were processed successfully.",
        )

    @staticmethod
    def _read_input_file(
        file_path: Path,
    ) -> pd.DataFrame:
        """Read a CSV or JSON review file."""
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
                "Unsupported file type. Select a CSV or JSON file."
            )

        if dataframe.empty:
            raise ValueError(
                "The selected file does not contain any records."
            )

        return dataframe

    @staticmethod
    def _find_text_column(
        dataframe: pd.DataFrame,
    ) -> str:
        """Identify the most likely review-text column."""
        normalized_columns = {
            str(column).strip().lower(): column
            for column in dataframe.columns
        }

        for candidate in TEXT_COLUMN_CANDIDATES:
            if candidate in normalized_columns:
                return str(normalized_columns[candidate])

        available_columns = ", ".join(
            str(column)
            for column in dataframe.columns
        )

        raise ValueError(
            "A review-text column could not be identified.\n\n"
            "Use a column name such as 'review', 'text', "
            "'feedback', 'comment' or 'Detailed Review'.\n\n"
            f"Available columns: {available_columns}"
        )

    def export_results(self) -> None:
        """Export all current predictions to CSV or JSON."""
        if self.current_results.empty:
            messagebox.showwarning(
                "No Results",
                "There are no processed results to export.",
            )
            return

        EXPORT_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = filedialog.asksaveasfilename(
            title="Export sentiment results",
            initialdir=EXPORT_DIRECTORY,
            initialfile="sentiment_results.csv",
            defaultextension=".csv",
            filetypes=[
                ("CSV file", "*.csv"),
                ("JSON file", "*.json"),
            ],
        )

        if not output_file:
            return

        output_path = Path(output_file)

        try:
            if output_path.suffix.lower() == ".json":
                self.current_results.to_json(
                    output_path,
                    orient="records",
                    indent=4,
                )
            else:
                self.current_results.to_csv(
                    output_path,
                    index=False,
                    encoding="utf-8",
                )

        except Exception as error:
            messagebox.showerror(
                "Export Error",
                str(error),
            )
            return

        self.status_message.set(
            f"Results exported to {output_path.name}."
        )

        messagebox.showinfo(
            "Export Complete",
            f"Results were saved to:\n{output_path}",
        )

    def clear_dashboard(self) -> None:
        """Clear entered text and all displayed results."""
        self.review_input.delete(
            "1.0",
            tk.END,
        )

        self.current_results = pd.DataFrame(
            columns=[
                "review",
                "clean_review",
                "sentiment",
                "processing_status",
            ]
        )

        self.single_result.set(
            "Enter a customer review to analyse its sentiment."
        )

        self.status_message.set(
            "Dashboard cleared."
        )

        self._update_dashboard()

    def _update_dashboard(self) -> None:
        """Refresh statistics, table and chart."""
        processed_results = self.current_results[
            self.current_results[
                "processing_status"
            ].eq("processed")
        ].copy()

        sentiment_counts = (
            processed_results["sentiment"]
            .value_counts()
            .reindex(
                [
                    "positive",
                    "neutral",
                    "negative",
                ],
                fill_value=0,
            )
        )

        total_processed = int(
            sentiment_counts.sum()
        )

        self.total_count.set(
            f"{total_processed:,}"
        )

        self.positive_count.set(
            f"{int(sentiment_counts['positive']):,}"
        )

        self.neutral_count.set(
            f"{int(sentiment_counts['neutral']):,}"
        )

        self.negative_count.set(
            f"{int(sentiment_counts['negative']):,}"
        )

        self._refresh_table()
        self._refresh_chart(sentiment_counts)

    def _refresh_table(self) -> None:
        """Display the latest prediction records."""
        for item in self.results_table.get_children():
            self.results_table.delete(item)

        display_dataframe = self.current_results.tail(
            DISPLAY_LIMIT
        )

        for row in display_dataframe.itertuples(
            index=False
        ):
            sentiment = (
                ""
                if pd.isna(row.sentiment)
                else str(row.sentiment).title()
            )

            self.results_table.insert(
                "",
                tk.END,
                values=(
                    row.review,
                    row.clean_review,
                    sentiment,
                    row.processing_status,
                ),
            )

    def _draw_empty_chart(self) -> None:
        """Display an empty chart when no results exist."""
        empty_counts = pd.Series(
            {
                "positive": 0,
                "neutral": 0,
                "negative": 0,
            }
        )

        self._refresh_chart(empty_counts)

    def _refresh_chart(
        self,
        sentiment_counts: pd.Series,
    ) -> None:
        """Redraw the sentiment-distribution bar chart."""
        if self.chart_canvas is not None:
            self.chart_canvas.get_tk_widget().destroy()

        labels = [
            "Positive",
            "Neutral",
            "Negative",
        ]

        values = [
            int(sentiment_counts.get("positive", 0)),
            int(sentiment_counts.get("neutral", 0)),
            int(sentiment_counts.get("negative", 0)),
        ]

        figure = Figure(
            figsize=(5, 4),
            dpi=100,
        )

        axis = figure.add_subplot(111)

        bars = axis.bar(
            labels,
            values,
            color=[
                "#2e8b57",
                "#d4a017",
                "#b22222",
            ],
        )

        axis.set_title(
            "Sentiment Distribution"
        )

        axis.set_xlabel(
            "Sentiment"
        )

        axis.set_ylabel(
            "Number of Reviews"
        )

        axis.grid(
            axis="y",
            alpha=0.25,
        )

        for bar, value in zip(
            bars,
            values,
            strict=True,
        ):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:,}",
                ha="center",
                va="bottom",
            )

        figure.tight_layout()

        self.chart_canvas = FigureCanvasTkAgg(
            figure,
            master=self.chart_frame,
        )

        self.chart_canvas.draw()

        self.chart_canvas.get_tk_widget().grid(
            row=0,
            column=0,
            sticky="nsew",
        )

    def _check_negative_review_alert(self) -> None:
        """Show an alert when negative feedback exceeds the threshold."""
        processed_results = self.current_results[
            self.current_results[
                "processing_status"
            ].eq("processed")
        ]

        total_reviews = len(processed_results)

        if total_reviews < NEGATIVE_ALERT_MIN_REVIEWS:
            return

        negative_reviews = int(
            processed_results["sentiment"]
            .eq("negative")
            .sum()
        )

        negative_percentage = (
            negative_reviews
            / total_reviews
            * 100
        )

        if negative_percentage >= NEGATIVE_ALERT_PERCENTAGE:
            messagebox.showwarning(
                "High Negative Sentiment Alert",
                (
                    f"{negative_reviews:,} of "
                    f"{total_reviews:,} reviews "
                    f"({negative_percentage:.1f}%) "
                    "were classified as negative.\n\n"
                    "Management should review the negative "
                    "feedback and identify recurring concerns."
                ),
            )

    @staticmethod
    def _sentiment_emoji(
        sentiment: str,
    ) -> str:
        """Return an emoji for a sentiment result."""
        emoji_map = {
            "positive": "🙂",
            "neutral": "😐",
            "negative": "🙁",
        }

        return emoji_map.get(
            sentiment,
            "❓",
        )


def run_gui() -> None:
    """Launch the sentiment-analysis dashboard."""
    application = SentimentAnalysisApp()
    application.mainloop()


if __name__ == "__main__":
    run_gui()