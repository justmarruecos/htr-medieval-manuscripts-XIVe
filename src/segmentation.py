"""
segmentation.py — Page layout analysis and line segmentation.
Pipeline:
    1. Run Kraken BLLA on preprocessed page image
    2. Extract line polygons
    3. Save to PAGE XML immediately
    4. Compute IoU against reference annotations
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image
from shapely.geometry import Polygon
# ---------------------------------------------------------------------------
# Kraken BLLA segmentation
# ---------------------------------------------------------------------------
def segment_page(image_path: str,
                 model_path: str = None) -> dict:
    """Finds all text lines on a manuscript page using Kraken BLLA.
    BLLA = Baseline Layout Analysis. Detects the baseline of each
    text line and computes a surrounding polygon. Works directly on
    the preprocessed binary image.
    Args:
        image_path: Path to the preprocessed page image.
        model_path: Path to a custom BLLA model. If None, uses the
            default Kraken BLLA model.
    Returns:
        Kraken segmentation result dict containing:
            - 'lines': list of line dicts with 'baseline' and 'boundary'
            - 'regions': detected text/margin/illustration regions
            - 'type': 'baselines'
    Raises:
        ImportError: If kraken is not installed.
        FileNotFoundError: If image_path does not exist.
    Example:
        >>> seg = segment_page("data/preprocessed/page_001.jpg")
        >>> print(f"Found {len(seg['lines'])} lines")
    """
    try:
        from kraken import blla
        from kraken.lib import vgsl
    except ImportError:
        raise ImportError("kraken not installed. Run: pip install kraken")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    if model_path and Path(model_path).exists():
        model = vgsl.TorchVGSLModel.load_model(model_path)
        result = blla.segment(img, model=model)
    else:
        result = blla.segment(img)
    return result
def segment_batch(image_dir: str,
                  output_dir: str,
                  model_path: str = None) -> list[str]:
    """Segments all pages in a directory and saves PAGE XML files.
    Args:
        image_dir: Directory containing preprocessed page images.
        output_dir: Directory where PAGE XML files will be saved.
        model_path: Optional path to custom BLLA model.
    Returns:
        List of paths to generated PAGE XML files.
    Example:
        >>> xml_files = segment_batch("data/preprocessed/", "segmentations/")
    """
    from tqdm import tqdm
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    xml_files = []
    for img_path in tqdm(images, desc="Segmenting pages"):
        try:
            seg = segment_page(str(img_path), model_path)
            xml_path = output_dir / f"{img_path.stem}.xml"
            save_page_xml(str(img_path), seg, str(xml_path))
            xml_files.append(str(xml_path))
        except Exception as e:
            print(f"[SKIP] {img_path.name}: {e}")
    print(f"Segmented {len(xml_files)}/{len(images)} pages → {output_dir}")
    return xml_files
# ---------------------------------------------------------------------------
# PAGE XML export
# ---------------------------------------------------------------------------
def save_page_xml(image_path: str,
                  segmentation,
                  output_path: str) -> None:
    """Saves Kraken segmentation results to PAGE XML format.
    PAGE XML is the standard format for storing line polygons and
    transcriptions in HTR. Compatible with eScriptorium and Kraken.
    Save immediately after segmentation — do not wait.
    Args:
        image_path: Path to the source image (stored in XML metadata).
        segmentation: Kraken segmentation result from segment_page().
        output_path: Path where the .xml file will be saved.
    Example:
        >>> seg = segment_page("page.jpg")
        >>> save_page_xml("page.jpg", seg, "segmentations/page.xml")
    """
    try:
        from kraken.lib import xml as kraken_xml
        kraken_xml.write_pagexml(segmentation, output_path)
    except Exception:
        _write_minimal_page_xml(image_path, segmentation, output_path)
def _write_minimal_page_xml(image_path: str,
                             segmentation,
                             output_path: str) -> None:
    """Fallback PAGE XML writer if kraken.lib.xml is unavailable.
    Supports both kraken 5.x (dict) and kraken 7.x (Segmentation object).
    Args:
        image_path: Source image path.
        segmentation: Kraken segmentation dict or object.
        output_path: Output XML path.
    """
    from PIL import Image as PILImage
    import xml.etree.ElementTree as ET
    from datetime import datetime
    img = PILImage.open(image_path)
    w, h = img.size
    root = ET.Element("PcGts", {
        "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    })
    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata, "Creator").text = "htr-medieval-french"
    ET.SubElement(metadata, "Created").text = datetime.utcnow().isoformat()
    page = ET.SubElement(root, "Page", {
        "imageFilename": str(image_path),
        "imageWidth": str(w),
        "imageHeight": str(h)
    })
    text_region = ET.SubElement(page, "TextRegion", {"id": "r1"})
    ET.SubElement(text_region, "Coords",
                  {"points": f"0,0 {w},0 {w},{h} 0,{h}"})

    # Support both kraken 5.x (dict) and kraken 7.x (Segmentation object)
    if isinstance(segmentation, dict):
        lines = segmentation.get("lines", [])
    else:
        lines = getattr(segmentation, "lines", [])

    for i, line in enumerate(lines):
        text_line = ET.SubElement(text_region, "TextLine", {"id": f"l{i+1}"})

        if isinstance(line, dict):
            boundary = line.get("boundary", [])
            baseline = line.get("baseline", [])
        else:
            boundary = getattr(line, "boundary", None) or []
            baseline = getattr(line, "baseline", None) or []

        if boundary:
            pts = " ".join(f"{int(x)},{int(y)}" for x, y in boundary)
            ET.SubElement(text_line, "Coords", {"points": pts})
        if baseline:
            pts = " ".join(f"{int(x)},{int(y)}" for x, y in baseline)
            ET.SubElement(text_line, "Baseline", {"points": pts})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
# ---------------------------------------------------------------------------
# IoU evaluation
# ---------------------------------------------------------------------------
def compute_iou(pred_polygon: list[tuple],
                ref_polygon: list[tuple]) -> float:
    """Computes Intersection over Union between two polygons.
    IoU = area of overlap / area of union.
    1.0 = perfect match, 0.75 = project minimum target.
    Args:
        pred_polygon: List of (x, y) pixel coordinates from your model.
        ref_polygon: List of (x, y) pixel coordinates from ground truth.
    Returns:
        IoU score between 0.0 and 1.0.
    Example:
        >>> iou = compute_iou([(0,0),(100,0),(100,20),(0,20)],
        ...                   [(5,0),(105,0),(105,20),(5,20)])
        >>> print(f"IoU: {iou:.3f}")
    """
    if len(pred_polygon) < 3 or len(ref_polygon) < 3:
        return 0.0
    pred = Polygon(pred_polygon)
    ref = Polygon(ref_polygon)
    if not pred.is_valid or not ref.is_valid:
        return 0.0
    intersection = pred.intersection(ref).area
    union = pred.union(ref).area
    return float(intersection / union) if union > 0 else 0.0
def evaluate_segmentation(pred_xml_path: str,
                           ref_xml_path: str) -> dict:
    """Evaluates segmentation quality against reference PAGE XML.
    Matches predicted lines to reference lines by overlap and
    computes mean IoU across all matched pairs.
    Args:
        pred_xml_path: Path to your predicted PAGE XML file.
        ref_xml_path: Path to the reference (ground truth) PAGE XML.
    Returns:
        Dictionary with:
            - 'mean_iou': Average IoU across all matched lines
            - 'n_pred': Number of predicted lines
            - 'n_ref': Number of reference lines
            - 'n_matched': Number of successfully matched lines
    Example:
        >>> results = evaluate_segmentation("pred.xml", "ref.xml")
        >>> print(f"Mean IoU: {results['mean_iou']:.3f}")
    """
    pred_polygons = _load_polygons_from_xml(pred_xml_path)
    ref_polygons = _load_polygons_from_xml(ref_xml_path)
    if not pred_polygons or not ref_polygons:
        return {"mean_iou": 0.0, "n_pred": 0, "n_ref": 0, "n_matched": 0}
    ious = []
    used_pred = set()
    for ref_poly in ref_polygons:
        best_iou = 0.0
        best_idx = -1
        for i, pred_poly in enumerate(pred_polygons):
            if i in used_pred:
                continue
            iou = compute_iou(pred_poly, ref_poly)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0 and best_iou > 0.1:
            ious.append(best_iou)
            used_pred.add(best_idx)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return {
        "mean_iou": mean_iou,
        "n_pred": len(pred_polygons),
        "n_ref": len(ref_polygons),
        "n_matched": len(ious)
    }
def _load_polygons_from_xml(xml_path: str) -> list[list[tuple]]:
    """Extracts line boundary polygons from a PAGE XML file.
    Args:
        xml_path: Path to PAGE XML file.
    Returns:
        List of polygons, each as a list of (x, y) tuples.
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"
    polygons = []
    for coords in root.iter(f"{ns}Coords"):
        points_str = coords.get("points", "")
        if not points_str:
            continue
        try:
            pts = [
                (int(p.split(",")[0]), int(p.split(",")[1]))
                for p in points_str.strip().split()
            ]
            if len(pts) >= 3:
                polygons.append(pts)
        except (ValueError, IndexError):
            continue
    return polygons