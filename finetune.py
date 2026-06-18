"""
finetune.py — Fine-tune Kraken or TrOCR on medieval French manuscript data.

Usage:
    # Fine-tune Kraken (starting from CREMMA model)
    python finetune.py kraken \
        --train_dir data/cremma/data/ \
        --base_model models/cremma-medieval.mlmodel \
        --output models/kraken-finetuned.mlmodel \
        --epochs 50

    # Fine-tune TrOCR with LoRA
    python finetune.py trocr \
        --train_dir data/cremma/data/ \
        --output models/trocr-finetuned \
        --lora_r 8 \
        --epochs 30 \
        --batch_size 8

    # Fine-tune TrOCR with CATMuS French data
    python finetune.py trocr \
        --use_catmus \
        --output models/trocr-catmus-finetuned \
        --lora_r 16 \
        --epochs 20
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.utils import fix_seeds, log_experiment, compute_cer
from src.recognition import get_device, fine_tune_kraken


# ---------------------------------------------------------------------------
# Dataset preparation for TrOCR
# ---------------------------------------------------------------------------

def load_line_images_from_xml(data_dir: str) -> list[dict]:
    """Load line images and transcriptions from ALTO or PAGE XML ground truth.

    Supports both ALTO XML (CREMMA format) and PAGE XML formats.
    Parses XML files, extracts line coordinates and transcriptions,
    then crops line images from source pages.

    Args:
        data_dir: Directory containing XML files and images (searched recursively).

    Returns:
        List of dicts with keys: 'image' (PIL Image), 'text' (str),
        'source' (str), 'line_id' (str), 'manuscript' (str).
    """
    import xml.etree.ElementTree as ET

    data_dir = Path(data_dir)
    records = []

    # Find all XML files (skip chocomufin variants)
    xml_files = sorted(
        f for f in data_dir.rglob("*.xml")
        if "chocomufin" not in f.name
    )
    if not xml_files:
        print(f"[WARNING] No XML files found in {data_dir}")
        return records

    for xml_path in tqdm(xml_files, desc="Loading XML ground truth"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            continue

        # Detect namespace
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        # Detect format: ALTO or PAGE
        is_alto = "alto" in ns.lower() or root.tag.lower().endswith("alto")

        if is_alto:
            new_records = _load_alto_xml(xml_path, root, ns)
        else:
            new_records = _load_page_xml(xml_path, root, ns, data_dir)

        records.extend(new_records)

    print(f"Loaded {len(records)} line images from {len(xml_files)} XML files")
    return records


def _load_alto_xml(xml_path: Path, root, ns: str) -> list[dict]:
    """Parse ALTO XML (CREMMA format) and extract line images + text.

    Args:
        xml_path: Path to the ALTO XML file.
        root: Parsed XML root element.
        ns: XML namespace string.

    Returns:
        List of record dicts.
    """
    records = []

    # Find source image filename
    filename_el = root.find(f".//{ns}fileName")
    if filename_el is None or not filename_el.text:
        return records

    img_filename = filename_el.text.strip()
    img_path = xml_path.parent / img_filename
    if not img_path.exists():
        # Try without directory prefix
        img_path = xml_path.parent / Path(img_filename).name
    if not img_path.exists():
        return records

    try:
        page_img = Image.open(img_path).convert("RGB")
    except Exception:
        return records

    # Extract manuscript name from parent directory
    manuscript = xml_path.parent.name

    # Parse TextLines
    for i, text_line in enumerate(root.iter(f"{ns}TextLine")):
        # Get text from String elements
        parts = []
        for string_el in text_line.iter(f"{ns}String"):
            content = string_el.get("CONTENT", "")
            if content:
                parts.append(content)

        text = " ".join(parts).strip()
        if not text:
            continue

        # Get bounding box from Shape/Polygon or HPOS/VPOS attributes
        shape_el = text_line.find(f"{ns}Shape")
        if shape_el is not None:
            polygon_el = shape_el.find(f"{ns}Polygon")
            if polygon_el is not None:
                points_str = polygon_el.get("POINTS", "")
                if points_str:
                    try:
                        coords = [
                            (int(p.split()[0]) if " " in p else int(p.split(",")[0]),
                             int(p.split()[1]) if " " in p else int(p.split(",")[1]))
                            for p in points_str.strip().split() if len(p.split()) == 1
                        ]
                        # ALTO POINTS format: "x1 y1 x2 y2 ..."
                        nums = points_str.strip().split()
                        if len(nums) >= 4 and all(n.isdigit() for n in nums[:4]):
                            coords = [
                                (int(nums[j]), int(nums[j+1]))
                                for j in range(0, len(nums) - 1, 2)
                            ]
                    except (ValueError, IndexError):
                        coords = []

                    if coords:
                        xs = [c[0] for c in coords]
                        ys = [c[1] for c in coords]
                        x0, x1_coord = max(0, min(xs)), max(xs)
                        y0, y1_coord = max(0, min(ys)), max(ys)

                        if x1_coord > x0 and y1_coord > y0:
                            line_img = page_img.crop((x0, y0, x1_coord, y1_coord))
                            line_id = f"{xml_path.stem}_l{i:03d}"
                            records.append({
                                "image": line_img,
                                "text": text,
                                "source": str(xml_path),
                                "line_id": line_id,
                                "manuscript": manuscript,
                            })
                            continue

        # Fallback: use HPOS/VPOS/WIDTH/HEIGHT attributes
        try:
            hpos = float(text_line.get("HPOS", 0))
            vpos = float(text_line.get("VPOS", 0))
            width = float(text_line.get("WIDTH", 0))
            height = float(text_line.get("HEIGHT", 0))
        except (ValueError, TypeError):
            continue

        if width <= 0 or height <= 0:
            continue

        x0 = max(0, int(hpos))
        y0 = max(0, int(vpos))
        x1_coord = int(hpos + width)
        y1_coord = int(vpos + height)

        line_img = page_img.crop((x0, y0, x1_coord, y1_coord))

        line_id = f"{xml_path.stem}_l{i:03d}"
        records.append({
            "image": line_img,
            "text": text,
            "source": str(xml_path),
            "line_id": line_id,
            "manuscript": manuscript,
        })

    return records


def _load_page_xml(xml_path: Path, root, ns: str, data_dir: Path) -> list[dict]:
    """Parse PAGE XML and extract line images + text.

    Args:
        xml_path: Path to the PAGE XML file.
        root: Parsed XML root element.
        ns: XML namespace string.
        data_dir: Base data directory for resolving image paths.

    Returns:
        List of record dicts.
    """
    records = []

    # Find the source image
    page_el = root.find(f".//{ns}Page")
    if page_el is None:
        return records

    img_filename = page_el.get("imageFilename", "")
    if not img_filename:
        return records

    # Resolve image path
    img_path = data_dir / img_filename
    if not img_path.exists():
        img_path = xml_path.parent / img_filename
    if not img_path.exists():
        img_path = xml_path.parent / Path(img_filename).name
    if not img_path.exists():
        return records

    try:
        page_img = Image.open(img_path).convert("RGB")
    except Exception:
        return records

    manuscript = xml_path.parent.name

    # Extract lines
    for i, text_line in enumerate(root.iter(f"{ns}TextLine")):
        # Get transcription from TextEquiv/Unicode
        unicode_el = text_line.find(f".//{ns}Unicode")
        if unicode_el is None or not unicode_el.text:
            continue
        text = unicode_el.text.strip()
        if not text:
            continue

        # Get coordinates for cropping
        coords_el = text_line.find(f"{ns}Coords")
        if coords_el is None:
            continue
        points_str = coords_el.get("points", "")
        if not points_str:
            continue

        try:
            points = [
                (int(p.split(",")[0]), int(p.split(",")[1]))
                for p in points_str.strip().split()
            ]
        except (ValueError, IndexError):
            continue

        if len(points) < 3:
            continue

        # Crop bounding box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, x1_coord = max(0, min(xs)), max(xs)
        y0, y1_coord = max(0, min(ys)), max(ys)

        if x1_coord <= x0 or y1_coord <= y0:
            continue

        line_img = page_img.crop((x0, y0, x1_coord, y1_coord))

        line_id = f"{xml_path.stem}_l{i:03d}"
        records.append({
            "image": line_img,
            "text": text,
            "source": str(xml_path),
            "line_id": line_id,
            "manuscript": manuscript,
        })

    return records


def prepare_trocr_dataset(records: list[dict], processor):
    """Convert line image/text records to a HuggingFace Dataset.

    Args:
        records: List of dicts with 'image' and 'text' keys.
        processor: TrOCRProcessor instance.

    Returns:
        HuggingFace Dataset with 'pixel_values' and 'labels' columns.
    """
    from datasets import Dataset

    pixel_values_list = []
    labels_list = []

    for record in tqdm(records, desc="Preparing TrOCR dataset"):
        img = record["image"].convert("RGB")

        # Process image
        pixel_values = processor(img, return_tensors="pt").pixel_values[0]
        pixel_values_list.append(pixel_values.numpy())

        # Tokenize text
        labels = processor.tokenizer(
            record["text"],
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        # Replace pad tokens with -100 for loss computation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels_list.append(labels.numpy())

    dataset = Dataset.from_dict({
        "pixel_values": pixel_values_list,
        "labels": labels_list,
    })
    dataset.set_format("torch")
    return dataset


# ---------------------------------------------------------------------------
# TrOCR fine-tuning with LoRA (standalone implementation)
# ---------------------------------------------------------------------------

def finetune_trocr(
    train_records: list[dict],
    val_records: list[dict],
    output_dir: str = "models/trocr-finetuned",
    lora_r: int = 8,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    seed: int = 42,
) -> dict:
    """Fine-tune TrOCR with LoRA on medieval manuscript line images.

    Args:
        train_records: Training data (list of dicts with 'image', 'text').
        val_records: Validation data (same format).
        output_dir: Where to save the fine-tuned model.
        lora_r: LoRA rank (8 or 16 recommended).
        epochs: Max training epochs.
        batch_size: Training batch size.
        learning_rate: AdamW learning rate.
        seed: Random seed.

    Returns:
        Dictionary with training metrics (final CER, etc.).
    """
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        EarlyStoppingCallback,
    )
    from peft import LoraConfig, get_peft_model

    fix_seeds(seed)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Fine-tuning TrOCR with LoRA (r={lora_r}) on {device}")
    print(f"Train: {len(train_records)} lines | Val: {len(val_records)} lines")
    print(f"{'='*60}\n")

    # Load processor and model
    model_name = "microsoft/trocr-base-handwritten"
    print(f"Loading base model: {model_name}")
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 4,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Set decoder config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Prepare datasets
    print("\nPreparing training dataset...")
    train_dataset = prepare_trocr_dataset(train_records, processor)
    print("Preparing validation dataset...")
    val_dataset = prepare_trocr_dataset(val_records, processor)

    # Metrics
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Replace -100 with pad for decoding
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = compute_cer(pred_str, label_str)
        return {"cer": cer}

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=(device == "cuda"),
        seed=seed,
        logging_steps=50,
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        weight_decay=0.01,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Save
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Final evaluation
    eval_results = trainer.evaluate()
    final_cer = eval_results.get("eval_cer", -1)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Final CER: {final_cer:.4f} ({final_cer:.1%})")
    print(f"  Model saved to: {output_dir}")
    print(f"{'='*60}\n")

    # Log experiment
    log_experiment("experiments/journal.jsonl", {
        "run_id": f"trocr_lora_r{lora_r}",
        "model": "trocr-base-handwritten",
        "method": f"LoRA r={lora_r}",
        "train_lines": len(train_records),
        "val_lines": len(val_records),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "cer_val": final_cer,
        "output_dir": output_dir,
    })

    return {
        "cer": final_cer,
        "train_loss": train_result.training_loss,
        "output_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# CATMuS data loading
# ---------------------------------------------------------------------------

def load_catmus_for_trocr(cache_dir: str = "data/catmus") -> tuple[list, list]:
    """Load CATMuS Medieval French data for TrOCR fine-tuning.

    Downloads the CATMuS dataset from HuggingFace, filters to French,
    and prepares line image/text pairs.

    Args:
        cache_dir: Local cache directory for the downloaded dataset.

    Returns:
        Tuple of (train_records, val_records).
    """
    from datasets import load_dataset

    print("Loading CATMuS Medieval (French) from HuggingFace...")
    ds = load_dataset("CATMuS/medieval", cache_dir=cache_dir)

    french_languages = {"Old French", "Middle French", "fro", "frm"}

    train_records = []
    val_records = []

    for split_name, split_data in ds.items():
        filtered = split_data.filter(
            lambda x: x.get("language", "") in french_languages,
            desc=f"Filtering {split_name} to French"
        )
        print(f"  {split_name}: {len(filtered)} French lines")

        for item in tqdm(filtered, desc=f"Processing {split_name}"):
            record = {
                "image": item["image"].convert("RGB") if hasattr(item["image"], "convert") else Image.open(item["image"]).convert("RGB"),
                "text": item.get("text", item.get("transcription", "")),
                "line_id": item.get("id", "unknown"),
                "source": "catmus",
            }
            if not record["text"]:
                continue

            if split_name == "train":
                train_records.append(record)
            else:
                val_records.append(record)

    print(f"CATMuS loaded: {len(train_records)} train, {len(val_records)} val")
    return train_records, val_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_kraken(args):
    """Fine-tune Kraken HTR model."""
    print(f"\n{'='*60}")
    print("Fine-tuning Kraken")
    print(f"  Base model: {args.base_model}")
    print(f"  Train dir:  {args.train_dir}")
    print(f"  Output:     {args.output}")
    print(f"  Epochs:     {args.epochs}")
    print(f"{'='*60}\n")

    fine_tune_kraken(
        train_data_dir=args.train_dir,
        base_model_path=args.base_model,
        output_model_path=args.output,
        epochs=args.epochs,
        lag=args.lag,
        augment=not args.no_augment,
    )

    log_experiment("experiments/journal.jsonl", {
        "run_id": f"kraken_ft_{Path(args.output).stem}",
        "model": "kraken",
        "base_model": args.base_model,
        "train_dir": args.train_dir,
        "epochs": args.epochs,
        "output": args.output,
    })


def cmd_trocr(args):
    """Fine-tune TrOCR with LoRA."""
    fix_seeds(args.seed)

    if args.use_catmus:
        train_records, val_records = load_catmus_for_trocr(
            cache_dir=args.catmus_cache
        )
    else:
        if not args.train_dir:
            print("ERROR: --train_dir is required when not using --use_catmus")
            return

        print(f"Loading training data from {args.train_dir}...")
        all_records = load_line_images_from_xml(args.train_dir)

        if not all_records:
            print("ERROR: No training data found. Check your --train_dir path.")
            return

        # Split by source (manuscript-level split)
        from sklearn.model_selection import train_test_split
        train_records, val_records = train_test_split(
            all_records,
            test_size=args.val_split,
            random_state=args.seed,
        )

    if not train_records:
        print("ERROR: No training records found.")
        return

    results = finetune_trocr(
        train_records=train_records,
        val_records=val_records,
        output_dir=args.output,
        lora_r=args.lora_r,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )

    print(f"\nFinal results: CER = {results['cer']:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune HTR models on medieval French manuscripts"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- Kraken ---
    p_kraken = subparsers.add_parser("kraken", help="Fine-tune Kraken model")
    p_kraken.add_argument("--train_dir", required=True,
                          help="Directory with PAGE XML training files")
    p_kraken.add_argument("--base_model", default="models/cremma-medieval.mlmodel",
                          help="Path to base Kraken model")
    p_kraken.add_argument("--output", default="models/kraken-finetuned.mlmodel",
                          help="Output model path")
    p_kraken.add_argument("--epochs", type=int, default=50)
    p_kraken.add_argument("--lag", type=int, default=5,
                          help="Early stopping patience")
    p_kraken.add_argument("--no_augment", action="store_true")

    # --- TrOCR ---
    p_trocr = subparsers.add_parser("trocr", help="Fine-tune TrOCR with LoRA")
    p_trocr.add_argument("--train_dir",
                         help="Directory with PAGE XML training files")
    p_trocr.add_argument("--use_catmus", action="store_true",
                         help="Use CATMuS Medieval French from HuggingFace")
    p_trocr.add_argument("--catmus_cache", default="data/catmus",
                         help="Cache dir for CATMuS download")
    p_trocr.add_argument("--output", default="models/trocr-finetuned",
                         help="Output directory for fine-tuned model")
    p_trocr.add_argument("--lora_r", type=int, default=8,
                         help="LoRA rank (8 or 16)")
    p_trocr.add_argument("--epochs", type=int, default=30)
    p_trocr.add_argument("--batch_size", type=int, default=8)
    p_trocr.add_argument("--lr", type=float, default=5e-5,
                         help="Learning rate")
    p_trocr.add_argument("--val_split", type=float, default=0.15)
    p_trocr.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "kraken":
        cmd_kraken(args)
    elif args.command == "trocr":
        cmd_trocr(args)
    else:
        parser.print_help()
