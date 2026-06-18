"""
tests/test_utils.py — Tests for CER, WER, and utility functions.
"""

import pytest
from src.utils import compute_cer, compute_wer, flag_needs_review


def test_cer_perfect():
    assert compute_cer(["hello"], ["hello"]) == 0.0


def test_cer_one_substitution():
    # "helo" vs "hello" = 1 insertion / 5 chars = 0.2
    cer = compute_cer(["helo"], ["hello"])
    assert abs(cer - 0.2) < 1e-6


def test_cer_completely_wrong():
    cer = compute_cer(["xxxx"], ["abcd"])
    assert cer == 1.0


def test_cer_multiple_lines():
    preds = ["li rois", "dist que"]
    refs  = ["li rois", "dist que"]
    assert compute_cer(preds, refs) == 0.0


def test_cer_length_mismatch():
    with pytest.raises(ValueError):
        compute_cer(["a", "b"], ["a"])


def test_wer_perfect():
    assert compute_wer(["li rois dist"], ["li rois dist"]) == 0.0


def test_wer_one_word_wrong():
    # "li rois dist" vs "li rois que" = 1/3
    wer = compute_wer(["li rois dist"], ["li rois que"])
    assert abs(wer - 1/3) < 1e-6


def test_wer_always_gte_cer():
    preds = ["li rois dist que nul"]
    refs  = ["li rois dist que mul"]
    cer = compute_cer(preds, refs)
    wer = compute_wer(preds, refs)
    assert wer >= cer


def test_flag_needs_review_low_confidence():
    assert flag_needs_review("Item Johannes", 0.4) is True


def test_flag_needs_review_high_confidence():
    assert flag_needs_review("Item Johannes de Sancto", 0.9) is False


def test_flag_needs_review_short_text():
    assert flag_needs_review("ab", 0.9) is True


def test_flag_needs_review_empty():
    assert flag_needs_review("", 0.9) is True
