from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import json
import sys

# Pick a page to visualize (change this to any page name)
page_name = "0_36c2c_default"

# Load transcriptions for this page
with open("results/kraken_output.json") as f:
    all_results = json.load(f)

page_lines = [r for r in all_results if r["page"] == page_name]
print(f"Found {len(page_lines)} transcribed lines for {page_name}")

# Load image
img_path = Path("data/preprocessed") / (page_name + ".jpg")
if not img_path.exists():
    img_path = Path("data/raw") / (page_name + ".jpg")
img = Image.open(img_path).convert("RGB")

# Load line coordinates from XML
xml_path = Path("segmentations") / (page_name + ".xml")
tree = ET.parse(str(xml_path))
root = tree.getroot()
ns = ""
if root.tag.startswith("{"):
    ns = root.tag.split("}")[0] + "}"

lines_coords = []
for line in root.iter(f"{ns}TextLine"):
    coords = line.find(f"{ns}Coords")
    if coords is not None:
        pts_str = coords.get("points", "")
        if pts_str:
            pts = [(int(p.split(",")[0]), int(p.split(",")[1]))
                   for p in pts_str.strip().split()]
            lines_coords.append(pts)

# Draw on image
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
except:
    font = ImageFont.load_default()

for i, (coords, line_data) in enumerate(zip(lines_coords, page_lines)):
    if not coords:
        continue
    # Draw polygon
    draw.polygon(coords, outline=(255, 0, 0), width=3)
    # Draw transcription above the line
    x = min(p[0] for p in coords)
    y = min(p[1] for p in coords) - 35
    text = line_data["transcription"]
    if text.strip():
        draw.rectangle([x, max(0, y), x + len(text)*17, y+32],
                       fill=(255, 255, 200))
        draw.text((x, max(0, y)), text, fill=(0, 0, 180), font=font)

# Save scaled down version
scale = 0.4
new_w = int(img.width * scale)
new_h = int(img.height * scale)
img_small = img.resize((new_w, new_h), Image.LANCZOS)
out_path = f"viz_{page_name}.jpg"
img_small.save(out_path, quality=90)
print(f"Saved to {out_path}")
