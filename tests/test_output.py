"""
tests/test_output.py — Tests for JSON output schema validation.
"""

import pytest
from src.output import build_output_record, validate_record


def make_valid_record() -> dict:
    return build_output_record(
        line_id="endp_r001_p042_l007",
        image_source="gallica_ark_12148_xxx",
        page=42,
        transcription="Item Johannes de Sancto Germano canonicus",
        confidence=0.87,
        model_name="kraken-cremma-finetuned-v1",
        polygon=[[120, 340], [890, 338], [892, 365], [118, 367]],
        train_set_sha256="a3f4b2c1d0e9f8a7b6c5d4e3f2a1b0c9",
    )


def test_valid_record_passes_schema():
    record = make_valid_record()
    assert validate_record(record) is True


def test_missing_required_field():
    record = make_valid_record()
    del record["transcription"]
    assert validate_record(record) is False


def test_confidence_out_of_range():
    record = make_valid_record()
    record["confidence"] = 1.5  # invalid
    assert validate_record(record) is False


def test_needs_review_is_boolean():
    record = make_valid_record()
    assert isinstance(record["needs_review"], bool)


def test_polygon_format():
    record = make_valid_record()
    assert record["polygon"]["format"] == "PAGE_XML"
    assert record["polygon"]["coordinate_system"] == "pixels_top_left_origin"
    assert isinstance(record["polygon"]["coordinates"], list)


def test_low_confidence_flagged_for_review():
    record = build_output_record(
        line_id="test_001",
        image_source="test",
        page=1,
        transcription="Item",
        confidence=0.3,
        model_name="test-model",
        polygon=[[0, 0], [100, 0], [100, 20], [0, 20]],
        train_set_sha256="abc123",
    )
    assert record["needs_review"] is True


def test_conventions_field():
    record = make_valid_record()
    assert record["transcription_conventions"] == "semi-diplomatic-catmus-v1"
