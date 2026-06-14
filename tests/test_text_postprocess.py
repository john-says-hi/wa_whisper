"""Tests for the text post-processing helpers."""

from wa_whisper.text_postprocess import postprocess_text


def test_banned_phrases_are_removed_case_sensitive() -> None:
    text = "Start. Thank you."
    assert postprocess_text(text) == "start"


def test_lowercases_single_sentence_and_drops_period() -> None:
    text = "THIS IS A TEST."
    assert postprocess_text(text) == "this is a test"


def test_single_sentence_longer_than_threshold_keeps_case() -> None:
    text = "This sentence is definitely longer."
    assert postprocess_text(text) == "This sentence is definitely longer."


def test_question_sentence_keeps_question_mark() -> None:
    text = "Is This Working?"
    assert postprocess_text(text) == "Is This Working?"


def test_multiple_sentences_remain_unchanged_by_lowercase_rule() -> None:
    text = "First sentence. Second sentence."
    assert postprocess_text(text) == "First sentence. Second sentence."


def test_paragraph_without_punctuation_keeps_appended_period() -> None:
    text = "This is sentence one this is sentence two"
    assert postprocess_text(text) == "This is sentence one this is sentence two."


def test_curly_apostrophe_phrase_is_removed() -> None:
    text = "We\u2019ll be right back. Resume normal service."
    assert postprocess_text(text) == "Resume normal service."


def test_single_sentence_without_period_is_unchanged() -> None:
    text = "One sentence only"
    assert postprocess_text(text, ensure_punctuation=False) == "One sentence only"


def test_blank_text_does_not_turn_into_a_space() -> None:
    assert postprocess_text("", append_space=True) == ""
