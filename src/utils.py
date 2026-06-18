"""
utils.py — Shared utilities: seeds, hashing, logging, CER/WER.
"""

import os
import random
import hashlib
import json
import editdistance
import numpy as np
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def fix_seeds(seed: int = 42) -> None:
    """Fixes all random seeds for reproducibility.

    Args:
        seed: Integer seed value. Default 42.

    Example:
        >>> fix_seeds(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# SHA-256 hashing (for sealing the test set)
# ---------------------------------------------------------------------------

def compute_sha256(file_path: str) -> str:
    """Computes SHA-256 hash of a file.

    Used to seal the test set and prove it was never modified.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hex string of the SHA-256 hash.

    Example:
        >>> sha = compute_sha256("data/test_set.json")
        >>> print(sha)  # save this string permanently
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(file_path: str, expected_hash: str) -> bool:
    """Verifies a file matches its expected SHA-256 hash.

    Args:
        file_path: Path to the file to verify.
        expected_hash: The hash computed when the file was sealed.

    Returns:
        True if the file is unchanged, False if it was modified.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    actual = compute_sha256(file_path)
    return actual == expected_hash


# ---------------------------------------------------------------------------
# Experiment journal
# ---------------------------------------------------------------------------

def log_experiment(journal_path: str, record: dict) -> None:
    """Appends an experiment record to the JSONL journal.

    Every training run should be logged here with its hyperparameters
    and results. A result not logged is a result lost.

    Args:
        journal_path: Path to the .jsonl journal file.
        record: Dictionary with experiment details. Should include at
            minimum: run_id, model, cer_val, timestamp.

    Example:
        >>> log_experiment("experiments/journal.jsonl", {
        ...     "run_id": "kraken_ft_001",
        ...     "model": "cremma-medieval",
        ...     "cer_val": 0.142,
        ...     "epochs": 20,
        ...     "notes": "first fine-tuning run"
        ... })
    """
    record["timestamp"] = datetime.utcnow().isoformat()
    Path(journal_path).parent.mkdir(parents=True, exist_ok=True)
    with open(journal_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CER and WER
# ---------------------------------------------------------------------------

def compute_cer(predictions: list[str],
                references: list[str]) -> float:
    """Computes Character Error Rate (CER).

    CER = total edit distance / total reference characters.
    Lower is better. 0.0 = perfect, 1.0 = completely wrong.

    Args:
        predictions: List of predicted transcription strings.
        references: List of ground truth transcription strings.

    Returns:
        CER as a float between 0 and 1.

    Raises:
        ValueError: If predictions and references have different lengths.

    Example:
        >>> compute_cer(["li rois dist"], ["li rois dist que"])
        0.25
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions "
            f"vs {len(references)} references"
        )
    total_errors = sum(
        editdistance.eval(p, r)
        for p, r in zip(predictions, references)
    )
    total_chars = sum(len(r) for r in references)
    if total_chars == 0:
        return 0.0
    return total_errors / total_chars


def compute_wer(predictions: list[str],
                references: list[str]) -> float:
    """Computes Word Error Rate (WER).

    WER = total word-level edit distance / total reference words.
    Always higher than CER for the same text.

    Args:
        predictions: List of predicted transcription strings.
        references: List of ground truth transcription strings.

    Returns:
        WER as a float between 0 and 1.

    Example:
        >>> compute_wer(["li rois dist"], ["li rois dist que"])
        0.25
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions "
            f"vs {len(references)} references"
        )
    total_errors = sum(
        editdistance.eval(p.split(), r.split())
        for p, r in zip(predictions, references)
    )
    total_words = sum(len(r.split()) for r in references)
    if total_words == 0:
        return 0.0
    return total_errors / total_words


def bootstrap_cer_ci(predictions: list[str],
                     references: list[str],
                     n_bootstrap: int = 1000,
                     alpha: float = 0.05) -> tuple[float, float]:
    """Computes bootstrap confidence interval for CER.

    Resamples the test set 1000 times to estimate uncertainty.
    Report as: CER = X% ± Y% (95% CI).

    Args:
        predictions: List of predicted transcription strings.
        references: List of ground truth transcription strings.
        n_bootstrap: Number of bootstrap resampling iterations.
        alpha: Significance level. 0.05 gives 95% CI.

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval.

    Example:
        >>> lo, hi = bootstrap_cer_ci(preds, refs)
        >>> print(f"CER = {cer:.1%} [{lo:.1%}, {hi:.1%}]")
    """
    n = len(predictions)
    cer_samples = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        sample_preds = [predictions[i] for i in idx]
        sample_refs = [references[i] for i in idx]
        cer_samples.append(compute_cer(sample_preds, sample_refs))
    lower = float(np.percentile(cer_samples, 100 * alpha / 2))
    upper = float(np.percentile(cer_samples, 100 * (1 - alpha / 2)))
    return lower, upper


def mcnemar_test(model_a_correct: list[bool],
                 model_b_correct: list[bool]) -> float:
    """McNemar test to compare two models statistically.

    Tests whether the difference between two models is significant
    or could be due to chance.

    Args:
        model_a_correct: List of booleans, True if model A got line right.
        model_b_correct: List of booleans, True if model B got line right.

    Returns:
        p-value. If < 0.05, models are significantly different.

    Example:
        >>> p = mcnemar_test(kraken_correct, trocr_correct)
        >>> print("Significant" if p < 0.05 else "Not significant")
    """
    from statsmodels.stats.contingency_tables import mcnemar

    b = sum(1 for a, b in zip(model_a_correct, model_b_correct)
            if a and not b)
    c = sum(1 for a, b in zip(model_a_correct, model_b_correct)
            if not a and b)
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    return float(result.pvalue)


# ---------------------------------------------------------------------------
# needs_review flagging
# ---------------------------------------------------------------------------

def flag_needs_review(transcription: str,
                      confidence: float,
                      conf_threshold: float = 0.6,
                      min_length: int = 3) -> bool:
    """Flags a transcription as needing human review.

    Args:
        transcription: The predicted text string.
        confidence: Model confidence score between 0 and 1.
        conf_threshold: Lines below this confidence are flagged.
        min_length: Lines shorter than this (in chars) are flagged.

    Returns:
        True if the line should be reviewed by a human.

    Example:
        >>> flag_needs_review("Item", 0.45)
        True
    """
    return (
        confidence < conf_threshold
        or len(transcription.strip()) < min_length
    )
