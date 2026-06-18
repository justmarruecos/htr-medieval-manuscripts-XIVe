"""
data_loader.py — Download and prepare CREMMA, e-NDP, and CATMuS datasets.

Steps:
    1. Download CATMuS from HuggingFace, filter to French
    2. Clone CREMMA Médiéval from GitHub
    3. Download e-NDP from Zenodo
    4. Split by manuscript (not by line) into train/val/test
    5. Seal test set with SHA-256
"""

import os
import json
import hashlib
import subprocess
import requests
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from src.utils import compute_sha256, fix_seeds


# ---------------------------------------------------------------------------
# CATMuS Medieval
# ---------------------------------------------------------------------------

def load_catmus_french(cache_dir: str = "data/catmus") -> dict:
    """Downloads CATMuS Medieval and filters to French languages only.

    Filters to Old French (fro) and Middle French (frm) lines.
    These are the languages matching our target corpus.

    Args:
        cache_dir: Local directory to cache the downloaded dataset.

    Returns:
        Dictionary with keys 'train', 'validation', 'test', each
        containing a HuggingFace Dataset of French lines.

    Example:
        >>> splits = load_catmus_french()
        >>> print(f"Train lines: {len(splits['train'])}")
    """
    from datasets import load_dataset

    print("Downloading CATMuS Medieval from HuggingFace...")
    ds = load_dataset("CATMuS/medieval", cache_dir=cache_dir)

    french_languages = {"Old French", "Middle French", "fro", "frm"}

    filtered = {}
    for split_name, split_data in ds.items():
        filtered[split_name] = split_data.filter(
            lambda x: x.get("language", "") in french_languages,
            desc=f"Filtering {split_name} to French"
        )
        print(f"  {split_name}: {len(filtered[split_name])} French lines")

    return filtered


# ---------------------------------------------------------------------------
# CREMMA Médiéval
# ---------------------------------------------------------------------------

def download_cremma(output_dir: str = "data/cremma") -> str:
    """Clones the CREMMA Médiéval repository from GitHub.

    Contains ~1,500 annotated lines of Old French literary manuscripts
    (13th-14th century, Gothic textualis) in PAGE XML format.

    Args:
        output_dir: Local directory to clone the repository into.

    Returns:
        Path to the cloned repository.

    Raises:
        RuntimeError: If git clone fails.
    """
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"CREMMA already exists at {output_dir}, skipping clone.")
        return str(output_path)

    print("Cloning CREMMA Médiéval from GitHub...")
    result = subprocess.run(
        ["git", "clone",
         "https://github.com/HTR-United/cremma-medieval",
         str(output_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr}")

    print(f"CREMMA cloned to {output_dir}")
    return str(output_path)


# ---------------------------------------------------------------------------
# e-NDP (Notre-Dame de Paris registers)
# ---------------------------------------------------------------------------

def download_endp(output_dir: str = "data/endp",
                  zenodo_url: str = "https://zenodo.org/record/7575693") -> str:
    """Downloads the e-NDP dataset from Zenodo.

    Contains ~3,000 annotated lines from Notre-Dame chapter registers
    (1326-1504, Gothic cursiva, Latin + French) in PAGE XML format.

    Args:
        output_dir: Local directory to save the downloaded files.
        zenodo_url: Zenodo record URL for the e-NDP dataset.

    Returns:
        Path to the downloaded dataset directory.

    Note:
        Zenodo requires manual download for large files.
        This function prints the download URL if automatic
        download is not possible.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\ne-NDP dataset must be downloaded manually from:")
    print(f"  {zenodo_url}")
    print(f"Download the ZIP file and extract to: {output_dir}")
    print(f"The ZIP contains PAGE XML files and line images.\n")

    return str(output_path)


# ---------------------------------------------------------------------------
# Kraken CREMMA model
# ---------------------------------------------------------------------------

def download_kraken_model(output_dir: str = "models",
                           zenodo_url: str = "https://zenodo.org/records/5617783") -> str:
    """Downloads the pre-trained CREMMA Kraken model from Zenodo.

    This model was trained on medieval French manuscripts and serves
    as the starting point for fine-tuning. Much better than starting
    from a generic Kraken model.

    Args:
        output_dir: Directory to save the .mlmodel file.
        zenodo_url: Zenodo record URL for the CREMMA Kraken model.

    Returns:
        Path to the downloaded model file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nKraken CREMMA model must be downloaded from:")
    print(f"  {zenodo_url}")
    print(f"Download the .mlmodel file and place in: {output_dir}/")
    print(f"Rename it to: cremma-medieval.mlmodel\n")

    return str(output_path / "cremma-medieval.mlmodel")


# ---------------------------------------------------------------------------
# Dataset splitting (by manuscript, not by line)
# ---------------------------------------------------------------------------

def split_by_manuscript(records: list[dict],
                         manuscript_key: str = "manuscript_id",
                         val_size: float = 0.15,
                         test_size: float = 0.15,
                         seed: int = 42) -> tuple[list, list, list]:
    """Splits data into train/val/test by manuscript, not by line.

    Critical: splitting by line would put pages from the same manuscript
    in both train and test, letting the model memorize handwriting styles.
    Splitting by manuscript ensures genuine generalization.

    Args:
        records: List of dicts, each representing one transcribed line.
            Must contain the manuscript_key field.
        manuscript_key: Field name identifying which manuscript a line
            belongs to.
        val_size: Fraction of manuscripts for validation (0.0-1.0).
        test_size: Fraction of manuscripts for test (0.0-1.0).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_records, val_records, test_records).

    Raises:
        KeyError: If manuscript_key is not found in records.

    Example:
        >>> train, val, test = split_by_manuscript(records)
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    fix_seeds(seed)

    manuscript_ids = [r[manuscript_key] for r in records]
    unique_manuscripts = list(set(manuscript_ids))

    # First split: separate test set
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    trainval_idx, test_idx = next(
        gss_test.split(records, groups=manuscript_ids)
    )

    trainval_records = [records[i] for i in trainval_idx]
    test_records = [records[i] for i in test_idx]
    trainval_ms_ids = [manuscript_ids[i] for i in trainval_idx]

    # Second split: separate val from train
    val_fraction = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    train_idx, val_idx = next(
        gss_val.split(trainval_records, groups=trainval_ms_ids)
    )

    train_records = [trainval_records[i] for i in train_idx]
    val_records = [trainval_records[i] for i in val_idx]

    print(f"Split complete:")
    print(f"  Train: {len(train_records)} lines")
    print(f"  Val:   {len(val_records)} lines")
    print(f"  Test:  {len(test_records)} lines")

    # Report manuscript distribution
    train_ms = set(r[manuscript_key] for r in train_records)
    val_ms = set(r[manuscript_key] for r in val_records)
    test_ms = set(r[manuscript_key] for r in test_records)
    print(f"  Manuscripts — Train: {len(train_ms)}, "
          f"Val: {len(val_ms)}, Test: {len(test_ms)}")

    return train_records, val_records, test_records


def seal_test_set(test_records: list[dict],
                  output_path: str = "data/test_set.json") -> str:
    """Saves and seals the test set with SHA-256.

    After calling this, never modify the test set file.
    The SHA-256 hash proves the test set was not changed.

    Args:
        test_records: List of test set records.
        output_path: Where to save the sealed test set JSON.

    Returns:
        SHA-256 hash of the saved file. Save this string permanently.

    Example:
        >>> sha = seal_test_set(test_records)
        >>> print(f"Test set sealed: {sha}")
        >>> # Write this hash to README.md and never change it
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)

    sha = compute_sha256(output_path)
    print(f"\nTest set sealed at: {output_path}")
    print(f"SHA-256: {sha}")
    print("Save this hash in your README.md — never modify the test set.\n")
    return sha
