from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

seg_dir = Path("segmentations")
img_dir = Path("data/preprocessed")

for xml_path in sorted(seg_dir.glob("*.xml")):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    img_path = img_dir / (xml_path.stem + ".jpg")
    if not img_path.exists():
        print(f"[SKIP] No image for {xml_path.stem}")
        continue

    img = Image.open(img_path).convert("RGB")
    out_dir = seg_dir / xml_path.stem
    out_dir.mkdir(exist_ok=True)

    lines = root.findall(f".//{ns}TextLine")
    for i, line in enumerate(lines):
        coords = line.find(f"{ns}Coords")
        if coords is None:
            continue
        pts_str = coords.get("points", "")
        if not pts_str:
            continue
        try:
            pts = [(int(p.split(",")[0]), int(p.split(",")[1]))
                   for p in pts_str.strip().split()]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            box = (min(xs), min(ys), max(xs), max(ys))
            crop = img.crop(box)
            crop.save(out_dir / f"line_{i+1:04d}.jpg")
        except Exception as e:
            print(f"[SKIP] {xml_path.stem} line {i}: {e}")

    print(f"Cropped {len(lines)} lines → {out_dir}")
