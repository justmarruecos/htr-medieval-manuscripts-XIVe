"""
run_pipeline.py — Full end-to-end HTR pipeline.

Usage:
    # Step 1: Preprocess images
    python run_pipeline.py preprocess --input data/raw/ --output data/preprocessed/

    # Step 2: Segment pages
    python run_pipeline.py segment --input data/preprocessed/ --output segmentations/

    # Step 3: Run HTR inference
    python run_pipeline.py transcribe --model kraken --model-path models/cremma-medieval.mlmodel --input segmentations/ --output results/

    # Step 4: Evaluate on test set
    python run_pipeline.py evaluate --predictions results/kraken_output.json --references data/test_set.json
"""

import argparse
import json
from pathlib import Path
from src.utils import fix_seeds, log_experiment, compute_cer, compute_wer, bootstrap_cer_ci

fix_seeds(42)


def cmd_preprocess(args):
    from src.preprocessing import preprocess_batch
    processed = preprocess_batch(args.input, args.output)
    print(f"Done. {len(processed)} images preprocessed.")


def cmd_segment(args):
    from src.segmentation import segment_batch
    xml_files = segment_batch(args.input, args.output)
    print(f"Done. {len(xml_files)} PAGE XML files saved.")


def cmd_transcribe(args):
    from PIL import Image
    import os

    # Load line images from segmentation output
    seg_dir = Path(args.input)
    line_images = []
    line_ids = []

    for xml_file in sorted(seg_dir.glob("*.xml")):
        from src.segmentation import _load_polygons_from_xml
        # In a real run, you'd crop line images from the page using polygons
        # For now, load any pre-cropped line images
        img_dir = seg_dir / xml_file.stem
        if img_dir.exists():
            for img_path in sorted(img_dir.glob("*.jpg")):
                line_images.append(Image.open(img_path))
                line_ids.append(img_path.stem)

    if not line_images:
        print("No line images found. Make sure line images are cropped.")
        return

    if args.model == "kraken":
        from src.recognition import run_kraken_inference
        texts = run_kraken_inference(line_images, args.model_path)
    elif args.model == "trocr":
        from src.recognition import run_trocr_inference
        texts = run_trocr_inference(line_images, args.model_path)
    else:
        raise ValueError(f"Unknown model: {args.model}. Use 'kraken' or 'trocr'")

    output = [{"line_id": lid, "transcription": t, "confidence": 0.8}
              for lid, t in zip(line_ids, texts)]

    out_path = Path(args.output) / f"{args.model}_output.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Transcriptions saved to {out_path}")



def cmd_evaluate(args):
    with open(args.predictions, encoding="utf-8") as f:
        pred_records = json.load(f)
    with open(args.references, encoding="utf-8") as f:
        ref_records = json.load(f)

    # Match by line_id
    ref_map = {r["line_id"]: r["transcription"] for r in ref_records}
    preds, refs = [], []
    for r in pred_records:
        if r["line_id"] in ref_map:
            preds.append(r["transcription"])
            refs.append(ref_map[r["line_id"]])

    if not preds:
        print("No matching line IDs found between predictions and references.")
        return

    cer = compute_cer(preds, refs)
    wer = compute_wer(preds, refs)
    ci_lo, ci_hi = bootstrap_cer_ci(preds, refs)

    print(f"\nEvaluation Results ({len(preds)} lines):")
    print(f"  CER: {cer:.1%}  [{ci_lo:.1%}, {ci_hi:.1%}] (95% CI)")
    print(f"  WER: {wer:.1%}")

    log_experiment("experiments/journal.jsonl", {
        "run_id": f"eval_{Path(args.predictions).stem}",
        "predictions_file": args.predictions,
        "n_lines": len(preds),
        "cer": cer,
        "wer": wer,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTR Medieval French Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # preprocess
    p = subparsers.add_parser("preprocess")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    # segment
    p = subparsers.add_parser("segment")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    # transcribe
    p = subparsers.add_parser("transcribe")
    p.add_argument("--model", choices=["kraken", "trocr"], required=True)
    p.add_argument("--model-path", dest="model_path",
                   default="models/cremma-medieval.mlmodel")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="results/")

    # evaluate
    p = subparsers.add_parser("evaluate")
    p.add_argument("--predictions", required=True)
    p.add_argument("--references", required=True)

    args = parser.parse_args()

    commands = {
        "preprocess": cmd_preprocess,
        "segment": cmd_segment,
        "transcribe": cmd_transcribe,
        "evaluate": cmd_evaluate,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
