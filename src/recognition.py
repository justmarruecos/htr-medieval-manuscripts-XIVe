"""
recognition.py — HTR inference and fine-tuning for Kraken and TrOCR.

Two models:
    - Kraken: CNN+LSTM, fine-tuned with ketos train
    - TrOCR: Vision Transformer, fine-tuned with LoRA via HuggingFace peft
"""

import os
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Returns the best available compute device.

    Checks for CUDA (NVIDIA GPU), then MPS (Apple Silicon),
    then falls back to CPU.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.

    Example:
        >>> device = get_device()
        >>> print(f"Using: {device}")
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Kraken inference
# ---------------------------------------------------------------------------

def run_kraken_inference(line_images: list,
                         model_path: str) -> list[str]:
    """Runs Kraken HTR on a list of line images.

    Args:
        line_images: List of PIL Image objects, one per text line.
        model_path: Path to the .mlmodel Kraken model file.

    Returns:
        List of transcription strings, one per input image.

    Raises:
        FileNotFoundError: If model_path does not exist.
        ImportError: If kraken is not installed.

    Example:
        >>> images = [Image.open(p) for p in line_paths]
        >>> texts = run_kraken_inference(images, "models/cremma-medieval.mlmodel")
    """
    try:
        from kraken import rpred
        from kraken.lib import models
        from kraken.containers import BBoxLine, Segmentation
    except ImportError:
        raise ImportError("kraken not installed. Run: pip install kraken")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = models.load_any(model_path)
    transcriptions = []

    for img in tqdm(line_images, desc="Kraken inference"):
        try:
            w, h = img.size
            line = BBoxLine(
                id="line_0",
                bbox=(0, 0, w, h),
                text_direction="horizontal-lr",
            )
            bounds = Segmentation(
                lines=[line],
                imagename="line",
                type="bbox",
                text_direction="horizontal-lr",
                script_detection=False,
                line_orders=[],
                regions={},
            )
            pred = rpred.rpred(model, img, bounds)
            text = "".join([r.prediction for r in pred])
            transcriptions.append(text)
        except Exception as e:
            print(f"[WARN] Kraken inference failed on line: {e}")
            transcriptions.append("")

    return transcriptions


def fine_tune_kraken(train_data_dir: str,
                     base_model_path: str,
                     output_model_path: str,
                     epochs: int = 50,
                     lag: int = 5,
                     augment: bool = True) -> None:
    """Fine-tunes a Kraken model using ketos train.

    Runs ketos train as a subprocess. The training data must be
    in ALTO XML or PAGE XML format with paired line images.

    Args:
        train_data_dir: Directory containing PAGE XML training files.
        base_model_path: Path to the starting .mlmodel checkpoint.
            Use the CREMMA medieval model for best results.
        output_model_path: Where to save the fine-tuned model.
        epochs: Maximum number of training epochs.
        lag: Early stopping patience — stops if no improvement
            for this many epochs.
        augment: Whether to apply data augmentations during training.

    Example:
        >>> fine_tune_kraken(
        ...     "data/train/",
        ...     "models/cremma-medieval.mlmodel",
        ...     "models/kraken-finetuned.mlmodel"
        ... )
    """
    import subprocess

    xml_files = list(Path(train_data_dir).glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {train_data_dir}")

    cmd = [
        "ketos", "train",
        "-f", "page",
        "--load", base_model_path,
        "-o", output_model_path,
        "--lag", str(lag),
        "--epochs", str(epochs),
    ]
    if augment:
        cmd.append("--augment")

    cmd += [str(f) for f in xml_files]

    print(f"Starting Kraken fine-tuning on {len(xml_files)} files...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Fine-tuning complete. Model saved to {output_model_path}")


# ---------------------------------------------------------------------------
# TrOCR inference
# ---------------------------------------------------------------------------

def run_trocr_inference(line_images: list,
                        model_name_or_path: str = "microsoft/trocr-base-handwritten",
                        batch_size: int = 8) -> list[str]:
    """Runs TrOCR inference on a list of line images.

    Args:
        line_images: List of PIL Image objects, one per text line.
        model_name_or_path: HuggingFace model name or local path.
            Use 'microsoft/trocr-base-handwritten' for the base model
            or a local path for your fine-tuned model.
        batch_size: Number of images to process at once.

    Returns:
        List of transcription strings, one per input image.

    Example:
        >>> images = [Image.open(p) for p in line_paths]
        >>> texts = run_trocr_inference(images)
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    device = get_device()
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    print(f"Loading TrOCR from {model_name_or_path} on {device}...")
    processor = TrOCRProcessor.from_pretrained(model_name_or_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
    model = model.to(device)
    model.eval()

    transcriptions = []

    for i in tqdm(range(0, len(line_images), batch_size),
                  desc="TrOCR inference"):
        batch = line_images[i:i + batch_size]
        batch_rgb = [img.convert("RGB") for img in batch]

        pixel_values = processor(
            batch_rgb, return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        transcriptions.extend(texts)

    return transcriptions


# ---------------------------------------------------------------------------
# TrOCR fine-tuning with LoRA
# ---------------------------------------------------------------------------

def fine_tune_trocr_lora(train_dataset,
                          val_dataset,
                          output_dir: str = "models/trocr-finetuned",
                          lora_r: int = 8,
                          epochs: int = 30,
                          batch_size: int = 8,
                          learning_rate: float = 5e-5,
                          seed: int = 42) -> None:
    """Fine-tunes TrOCR with LoRA adapters on medieval manuscript data.

    LoRA = Low-Rank Adaptation. Only trains small adapter matrices
    instead of all 334M parameters. Much faster and cheaper.
    Start with r=8, try r=16 if CER plateaus.

    Args:
        train_dataset: HuggingFace Dataset with 'pixel_values' and
            'labels' columns (line images + transcriptions).
        val_dataset: Validation dataset, same format.
        output_dir: Directory to save checkpoints and final model.
        lora_r: LoRA rank. Higher = more parameters = potentially
            better but slower. Try 8 first, then 16.
        epochs: Maximum training epochs.
        batch_size: Training batch size. Reduce if OOM on GPU.
        learning_rate: AdamW learning rate.
        seed: Random seed for reproducibility.

    Example:
        >>> fine_tune_trocr_lora(
        ...     train_ds, val_ds,
        ...     output_dir="models/trocr-r8",
        ...     lora_r=8
        ... )
    """
    from transformers import (TrOCRProcessor, VisionEncoderDecoderModel,
                               Seq2SeqTrainer, Seq2SeqTrainingArguments,
                               default_data_collator, EarlyStoppingCallback)
    from peft import LoraConfig, get_peft_model, TaskType
    import evaluate
    from src.utils import fix_seeds, compute_cer

    fix_seeds(seed)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    device = get_device()
    print(f"Fine-tuning TrOCR with LoRA r={lora_r} on {device}")

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 4,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        cer = compute_cer(pred_str, label_str)
        return {"cer": cer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=(device == "cuda"),
        seed=seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Fine-tuning complete. Model saved to {output_dir}")