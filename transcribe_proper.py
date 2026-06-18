from pathlib import Path
from PIL import Image
from kraken import rpred
from kraken.lib import models
from kraken.lib.xml import parse_page
from kraken.containers import Segmentation
from lxml import etree
import json
from tqdm import tqdm

model = models.load_any("models/cremma-medieval.mlmodel")
seg_dir = Path("segmentations")
img_dir = Path("data/preprocessed")
results = []

for xml_path in tqdm(sorted(seg_dir.glob("*.xml"))):
    img_path = img_dir / (xml_path.stem + ".jpg")
    if not img_path.exists():
        continue
    try:
        img = Image.open(img_path).convert("RGB")
        doc = etree.parse(str(xml_path))
        parsed = parse_page(doc, xml_path, "baselines")
        bounds = Segmentation(
            lines=list(parsed["lines"].values()),
            imagename=img_path,
            type="baselines",
            text_direction="horizontal-lr",
            script_detection=False,
            line_orders=[],
            regions=parsed["regions"],
        )
        preds = rpred.rpred(model, img, bounds)
        for pred in preds:
            results.append({
                "page": xml_path.stem,
                "transcription": pred.prediction,
            })
    except Exception as e:
        print(f"[SKIP] {xml_path.stem}: {e}")

Path("results").mkdir(exist_ok=True)
with open("results/kraken_output.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Done. {len(results)} lines transcribed.")
