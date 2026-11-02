"""A small module for counting words in a piece of text."""

import string


def count_words(text: str) -> dict[str, int]:
    """Count the occurrences of each word in text.

    Words are split on whitespace, lowercased, and stripped of leading
    and trailing punctuation. Empty tokens are ignored.
    """
    counts: dict[str, int] = {}
    for word in text.split():
        word = word.strip(string.punctuation).lower()
        if not word:
            continue
        counts[word] = counts.get(word, 0) + 1
    return counts
