"""
visualize_segmentation.py — Show ground truth vs predicted segmentation on manuscript pages.
"""
from pathlib import Path
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import numpy as np


def load_page_xml_polygons(xml_path):
    """Load polygons from our generated PAGE XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"
    polygons = []
    for coords in root.iter(f"{ns}Coords"):
        pts_str = coords.get("points", "")
        if not pts_str:
            continue
        try:
            pts = [(int(p.split(",")[0]), int(p.split(",")[1]))
                   for p in pts_str.strip().split()]
            if len(pts) >= 3:
                polygons.append(pts)
        except:
            continue
    return polygons


def load_alto_polygons(xml_path):
    """Load polygons from CREMMA ALTO XML ground truth."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'
    polygons = []
    for line in root.iter(f'{ns}TextLine'):
        shape = line.find(f'{ns}Shape')
        if shape is not None:
            poly = shape.find(f'{ns}Polygon')
            if poly is not None:
                pts_str = poly.get('POINTS', '')
                nums = pts_str.strip().split()
                pts = [(int(nums[i]), int(nums[i+1]))
                       for i in range(0, len(nums)-1, 2)]
                if len(pts) >= 3:
                    polygons.append(pts)
        else:
            hpos = int(line.get('HPOS', 0))
            vpos = int(line.get('VPOS', 0))
            w = int(line.get('WIDTH', 0))
            h = int(line.get('HEIGHT', 0))
            if w > 0 and h > 0:
                polygons.append([
                    (hpos, vpos), (hpos+w, vpos),
                    (hpos+w, vpos+h), (hpos, vpos+h)
                ])
    return polygons


def draw_polygons(img, polygons, color, width=4, fill_alpha=40):
    """Draw polygons on image with fill and outline."""
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_outline = ImageDraw.Draw(img)
    for poly in polygons:
        if len(poly) < 3:
            continue
        flat = [coord for pt in poly for coord in pt]
        draw_overlay.polygon(flat, fill=(*color, fill_alpha))
        draw_outline.line(flat + flat[:2], fill=(*color, 255), width=width)
    img = Image.alpha_composite(img.convert('RGBA'), overlay)
    return img.convert('RGB')


def visualize_page(page_name, img_dir, seg_dir, gt_dir, out_dir, scale=0.35):
    img_path = Path(img_dir) / (page_name + ".jpg")
    pred_xml = Path(seg_dir) / (page_name + ".xml")

    # Find GT xml
    gt_xml = None
    for p in Path(gt_dir).rglob(page_name + ".xml"):
        gt_xml = p
        break

    if not img_path.exists():
        print(f"[SKIP] No image: {img_path}")
        return
    if not pred_xml.exists():
        print(f"[SKIP] No prediction XML: {pred_xml}")
        return

    img_orig = Image.open(img_path).convert("RGB")

    # Original image (no annotations)
    img_plain = img_orig.copy()

    # Ground truth (blue polygons)
    img_gt = img_orig.copy()
    if gt_xml and gt_xml.exists():
        gt_polys = load_alto_polygons(str(gt_xml))
        img_gt = draw_polygons(img_gt, gt_polys, color=(30, 100, 255))
    else:
        print(f"[WARN] No GT xml found for {page_name}")

    # Predicted (red polygons)
    img_pred = img_orig.copy()
    pred_polys = load_page_xml_polygons(str(pred_xml))
    img_pred = draw_polygons(img_pred, pred_polys, color=(220, 50, 50))

    # Overlay both on same image
    img_both = img_orig.copy()
    if gt_xml and gt_xml.exists():
        img_both = draw_polygons(img_both, gt_polys, color=(30, 100, 255), fill_alpha=30)
    img_both = draw_polygons(img_both, pred_polys, color=(220, 50, 50), fill_alpha=30)

    # Resize all
    new_w = int(img_orig.width * scale)
    new_h = int(img_orig.height * scale)
    size = (new_w, new_h)
    imgs = [img_plain.resize(size, Image.LANCZOS),
            img_gt.resize(size, Image.LANCZOS),
            img_pred.resize(size, Image.LANCZOS),
            img_both.resize(size, Image.LANCZOS)]

    # Stitch side by side
    total_w = new_w * 4 + 30
    combined = Image.new('RGB', (total_w, new_h + 60), (240, 240, 240))

    titles = ['Original', 'Ground Truth (blue)', 'Predicted (red)', 'Overlay']
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(combined)
    for j, (tile, title) in enumerate(zip(imgs, titles)):
        x = j * (new_w + 10)
        combined.paste(tile, (x, 50))
        draw.text((x + new_w//2 - len(title)*7, 10), title,
                  fill=(40, 40, 40), font=font)

    Path(out_dir).mkdir(exist_ok=True)
    out_path = Path(out_dir) / f"seg_viz_{page_name}.jpg"
    combined.save(out_path, quality=90)
    print(f"Saved → {out_path}")


# Pages to visualize — pick ones that have GT
pages = [
    "btv1b84473026_f5",
    "btv1b84473026_f10",
    "f.157r",
    "343_74602_default",
]

for page in pages:
    visualize_page(
        page_name=page,
        img_dir="data/preprocessed",
        seg_dir="segmentations",
        gt_dir="data/cremma/data",
        out_dir="viz_segmentation",
        scale=0.3,
    )

print("\nDone. Open the viz_segmentation/ folder.")
