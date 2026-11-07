"""Tests for the wordcount module."""

from wordcount import count_words


def test_empty_string():
    assert count_words("") == {}


def test_empty_after_split():
    assert count_words("") == {}
    # Whitespace only yields no words.
    assert count_words("   \n\t  ") == {}


def test_repeated_words():
    result = count_words("cat cat dog dog cat")
    assert result == {"cat": 3, "dog": 2}


def test_mixed_case():
    assert count_words("The the THE") == {"the": 3}


def test_punctuation():
    result = count_words("hello, world!")
    assert result == {"hello": 1, "world": 1}


def test_extra_whitespace_and_newlines():
    text = "one\ttwo\nthree  four\n\nfive"
    assert count_words(text) == {"one": 1, "two": 1, "three": 1, "four": 1, "five": 1}


def test_punctuation_only_tokens_are_skipped():
    # Tokens made up solely of punctuation vanish after stripping.
    assert count_words("!!! hello ???") == {"hello": 1}


def test_apostrophes_inside_words_are_kept():
    assert count_words("don't stop, don't") == {"don't": 2, "stop": 1}
