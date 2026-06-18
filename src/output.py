"""
output.py — Build and validate the final JSON data contract output.

The JSON output is delivered to the NLP team (Volet 2).
Each transcribed line becomes one record with polygon coordinates,
confidence score, needs_review flag, and metadata.
"""

import json
import jsonschema
from pathlib import Path
from src.utils import flag_needs_review, compute_sha256


# ---------------------------------------------------------------------------
# JSON schema (data contract)
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "line_id", "image_source", "page", "transcription",
        "confidence", "needs_review", "model", "language",
        "polygon", "sha256_train_set", "transcription_conventions"
    ],
    "properties": {
        "line_id": {"type": "string"},
        "image_source": {"type": "string"},
        "page": {"type": "integer", "minimum": 1},
        "transcription": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "needs_review": {"type": "boolean"},
        "model": {"type": "string"},
        "century": {"type": "string"},
        "language": {"type": "string"},
        "polygon": {
            "type": "object",
            "required": ["format", "coordinates", "coordinate_system"],
            "properties": {
                "format": {"type": "string"},
                "coordinates": {"type": "array"},
                "coordinate_system": {"type": "string"}
            }
        },
        "sha256_train_set": {"type": "string"},
        "transcription_conventions": {"type": "string"},
    }
}


# ---------------------------------------------------------------------------
# Build output records
# ---------------------------------------------------------------------------

def build_output_record(line_id: str,
                         image_source: str,
                         page: int,
                         transcription: str,
                         confidence: float,
                         model_name: str,
                         polygon: list,
                         train_set_sha256: str,
                         century: str = "14",
                         language: str = "frm") -> dict:
    """Builds a single output record conforming to the data contract.

    Args:
        line_id: Unique identifier for this line.
            Format: {manuscript_id}_p{page:03d}_l{line:03d}
        image_source: Gallica ARK identifier or file path of source image.
        page: Page number in the manuscript (1-indexed).
        transcription: The final transcription string.
        confidence: Model confidence score between 0 and 1.
        model_name: Name of the model that produced this transcription.
        polygon: List of [x, y] pixel coordinate pairs defining the
            line boundary on the source image.
        train_set_sha256: SHA-256 hash of the training set file.
        century: Century of the manuscript as string (e.g. "14").
        language: ISO 639-3 language code (frm=Middle French,
            fro=Old French, lat=Latin).

    Returns:
        Dictionary conforming to OUTPUT_SCHEMA.

    Example:
        >>> record = build_output_record(
        ...     line_id="endp_r001_p042_l007",
        ...     image_source="gallica_ark_12148_xxx",
        ...     page=42,
        ...     transcription="Item Johannes de Sancto Germano",
        ...     confidence=0.87,
        ...     model_name="kraken-cremma-finetuned-v1",
        ...     polygon=[[120,340],[890,338],[892,365],[118,367]],
        ...     train_set_sha256="a3f4b2..."
        ... )
    """
    return {
        "line_id": line_id,
        "image_source": image_source,
        "page": page,
        "transcription": transcription,
        "confidence": round(float(confidence), 4),
        "needs_review": flag_needs_review(transcription, confidence),
        "model": model_name,
        "century": century,
        "language": language,
        "polygon": {
            "format": "PAGE_XML",
            "coordinates": polygon,
            "coordinate_system": "pixels_top_left_origin"
        },
        "sha256_train_set": train_set_sha256,
        "transcription_conventions": "semi-diplomatic-catmus-v1",
    }


def validate_record(record: dict) -> bool:
    """Validates a single output record against the data contract schema.

    Args:
        record: Output record dictionary to validate.

    Returns:
        True if valid, False if schema validation fails.

    Example:
        >>> assert validate_record(record), "Invalid record!"
    """
    try:
        jsonschema.validate(record, OUTPUT_SCHEMA)
        return True
    except jsonschema.ValidationError as e:
        print(f"[INVALID] {record.get('line_id', '?')}: {e.message}")
        return False


# ---------------------------------------------------------------------------
# Save output dataset
# ---------------------------------------------------------------------------

def save_output_dataset(records: list[dict],
                         output_path: str,
                         validate: bool = True) -> dict:
    """Saves the complete output dataset as JSON.

    Validates all records against the schema before saving.
    Reports statistics: total lines, needs_review rate, etc.

    Args:
        records: List of output record dictionaries.
        output_path: Path to save the JSON file.
        validate: Whether to validate each record before saving.

    Returns:
        Statistics dictionary with counts and rates.

    Raises:
        ValueError: If validate=True and any records are invalid.

    Example:
        >>> stats = save_output_dataset(records, "dataset_nlp/output.json")
        >>> print(f"Saved {stats['total']} lines, "
        ...       f"{stats['needs_review_rate']:.1%} need review")
    """
    if validate:
        invalid = [r for r in records if not validate_record(r)]
        if invalid:
            raise ValueError(
                f"{len(invalid)} invalid records. Fix before saving."
            )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Compute statistics
    total = len(records)
    needs_review = sum(1 for r in records if r["needs_review"])
    llm_corrected = sum(1 for r in records if r.get("llm_corrected", False))
    avg_confidence = sum(r["confidence"] for r in records) / total if total else 0

    stats = {
        "total": total,
        "needs_review": needs_review,
        "needs_review_rate": needs_review / total if total else 0,
        "llm_corrected": llm_corrected,
        "avg_confidence": avg_confidence,
        "output_path": output_path,
    }

    print(f"\nDataset saved to {output_path}")
    print(f"  Total lines:       {total}")
    print(f"  Needs review:      {needs_review} ({stats['needs_review_rate']:.1%})")
    print(f"  LLM corrected:     {llm_corrected}")
    print(f"  Avg confidence:    {avg_confidence:.3f}")

    return stats
