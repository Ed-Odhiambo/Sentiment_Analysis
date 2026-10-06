"""Text preprocessing utilities for the sentiment analysis project."""

import html
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer


# Negation words can reverse sentiment and should not be removed.
NEGATION_WORDS = {
    "no",
    "nor",
    "not",
    "never",
    "none",
    "cannot",
    "couldn't",
    "didn't",
    "doesn't",
    "don't",
    "hadn't",
    "hasn't",
    "haven't",
    "isn't",
    "mustn't",
    "shouldn't",
    "wasn't",
    "weren't",
    "won't",
    "wouldn't",
}

TOKENIZER = TweetTokenizer(
    preserve_case=False,
    reduce_len=True,
    strip_handles=False,
)

LEMMATIZER = WordNetLemmatizer()


@lru_cache(maxsize=1)
def get_stopwords() -> set[str]:
    """Return English stopwords while retaining important negation words."""
    english_stopwords = set(stopwords.words("english"))
    return english_stopwords.difference(NEGATION_WORDS)


def verify_nltk_resources() -> None:
    """Confirm that the required NLTK resources can be loaded."""

    missing_resources = []

    try:
        stopwords.words("english")
    except LookupError:
        missing_resources.append("stopwords")

    try:
        from nltk.corpus import wordnet

        wordnet.synsets("good")
    except LookupError:
        missing_resources.append("wordnet")

    if missing_resources:
        missing_text = ", ".join(missing_resources)

        raise LookupError(
            "The following NLTK resources are missing or unavailable: "
            f"{missing_text}.\n"
            "Download them using:\n"
            "python -m nltk.downloader stopwords wordnet omw-1.4"
        )


def clean_review(text: object) -> str:
    """
    Clean and normalize one customer review.

    Parameters
    ----------
    text:
        Review text. Non-string values are safely converted to strings.

    Returns
    -------
    str
        Cleaned review text ready for feature extraction.
    """
    if text is None:
        return ""

    review = str(text).strip()

    if not review:
        return ""

    # Convert entities such as &amp; and &quot; to normal characters.
    review = html.unescape(review)

    # Remove HTML tags.
    review = re.sub(r"<[^>]+>", " ", review)

    # Remove web links.
    review = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        review,
        flags=re.IGNORECASE,
    )

    # Remove Twitter account mentions.
    review = re.sub(r"@\w+", " ", review)

    # Keep hashtag text while removing the hash symbol.
    review = re.sub(r"#(\w+)", r"\1", review)

    # Separate common negation contractions before punctuation removal.
    review = re.sub(
        r"\b(can't)\b",
        "can not",
        review,
        flags=re.IGNORECASE,
    )
    review = re.sub(
        r"n['’]t\b",
        " not",
        review,
        flags=re.IGNORECASE,
    )

    # Retain letters, apostrophes, and spaces.
    review = re.sub(r"[^a-zA-Z'\s]", " ", review)

    # Normalize repeated whitespace.
    review = re.sub(r"\s+", " ", review).strip().lower()

    tokens = TOKENIZER.tokenize(review)
    stop_words = get_stopwords()

    cleaned_tokens = []

    for token in tokens:
        token = token.strip("'")

        if not token:
            continue

        if len(token) == 1 and token not in {"i", "a"}:
            continue

        if token in stop_words:
            continue

        lemmatized_token = LEMMATIZER.lemmatize(token, pos="v")

        if lemmatized_token:
            cleaned_tokens.append(lemmatized_token)

    return " ".join(cleaned_tokens)


def preprocess_reviews(reviews: list[object]) -> list[str]:
    """Clean multiple reviews using the same preprocessing procedure."""
    return [clean_review(review) for review in reviews]


if __name__ == "__main__":
    verify_nltk_resources()

    sample_reviews = [
        "@VirginAmerica I didn't enjoy the delayed flight at all!",
        "The service was amazing &amp; the staff were very helpful.",
        "Visit https://example.com for more details.",
        "#GreatService but the waiting time was not acceptable.",
    ]

    print("PREPROCESSING TEST")
    print("-" * 60)

    for original_review in sample_reviews:
        cleaned_review = clean_review(original_review)

        print(f"Original: {original_review}")
        print(f"Cleaned:  {cleaned_review}")
        print()