"""
evaluate_segmentation.py — Evaluate segmentation IoU against CREMMA ground truth.
"""
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.segmentation import _load_polygons_from_xml, compute_iou
from pathlib import Path


def load_alto_polygons(xml_path):
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


def evaluate_page(pred_xml, gt_xml):
    pred_polys = _load_polygons_from_xml(str(pred_xml))
    ref_polys = load_alto_polygons(str(gt_xml))
    if not pred_polys or not ref_polys:
        return None, 0, 0
    ious = []
    used = set()
    for ref in ref_polys:
        best, best_i = 0, -1
        for i, pred in enumerate(pred_polys):
            if i in used:
                continue
            iou = compute_iou(pred, ref)
            if iou > best:
                best, best_i = iou, i
        if best_i >= 0 and best > 0.1:
            ious.append(best)
            used.add(best_i)
    mean = float(np.mean(ious)) if ious else 0.0
    return mean, len(ious), len(ref_polys)


def main():
    seg_dir = Path('segmentations')
    gt_dir = Path('data/cremma/data')
    names, scores, matched_counts, ref_counts = [], [], [], []

    for gt_xml in sorted(gt_dir.rglob('*.xml')):
        pred_xml = seg_dir / gt_xml.name
        if not pred_xml.exists():
            continue
        mean_iou, n_matched, n_ref = evaluate_page(pred_xml, gt_xml)
        if mean_iou is None:
            continue
        names.append(gt_xml.stem)
        scores.append(mean_iou)
        matched_counts.append(n_matched)
        ref_counts.append(n_ref)
        print(f'{gt_xml.name}: IoU={mean_iou:.3f} ({n_matched}/{n_ref} matched)')

    if not scores:
        print('No results found.')
        return

    mean_iou = float(np.mean(scores))
    n_pass = sum(1 for s in scores if s >= 0.75)
    print(f'\nPages evaluated     : {len(scores)}')
    print(f'Pages passing >=0.75: {n_pass}/{len(scores)}')
    print(f'Mean IoU            : {mean_iou:.3f}')
    print(f'Min IoU             : {min(scores):.3f}')
    print(f'Max IoU             : {max(scores):.3f}')

    short_names = [n[:12] for n in names]
    idx = np.argsort(scores)
    sorted_names = [short_names[i] for i in idx]
    sorted_scores = [scores[i] for i in idx]
    sorted_matched = [matched_counts[i] for i in idx]
    sorted_ref = [ref_counts[i] for i in idx]

    # ── Chart 1: IoU vs Ground Truth line count ──────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    x = np.arange(len(sorted_names))
    width = 0.4

    bars_ref = ax1.bar(x - width/2, sorted_ref, width,
                       color='#B0C4DE', label='Ground Truth Lines', zorder=2)
    bars_match = ax1.bar(x + width/2, sorted_matched, width,
                         color=['#2ecc71' if s >= 0.75 else '#e67e22' if s >= 0.5 else '#e74c3c'
                                for s in sorted_scores],
                         label='Matched Lines', zorder=2)

    ax2 = ax1.twinx()
    ax2.plot(x, sorted_scores, color='#2c3e50', marker='o',
             markersize=5, linewidth=2, label='IoU Score', zorder=3)
    ax2.axhline(0.75, color='red', linestyle='--', linewidth=1.5,
                label='Target (0.75)', zorder=3)
    ax2.axhline(mean_iou, color='blue', linestyle=':', linewidth=1.5,
                label=f'Mean IoU ({mean_iou:.3f})', zorder=3)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('IoU Score', fontsize=12)

    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_names, rotation=90, fontsize=7)
    ax1.set_ylabel('Number of Lines', fontsize=12)
    ax1.set_title('Segmentation Quality: Ground Truth Lines vs Matched Lines + IoU Score',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_facecolor('#f8f9fa')
    ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)

    lines2, labels2 = ax2.get_legend_handles_labels()
    patch_ref = mpatches.Patch(color='#B0C4DE', label='Ground Truth Lines')
    patch_match = mpatches.Patch(color='#2ecc71', label='Matched Lines (green=pass)')
    ax1.legend(handles=[patch_ref, patch_match] + lines2,
               labels=['Ground Truth Lines', 'Matched Lines (green=pass)'] + labels2,
               loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig('iou_vs_groundtruth.png', dpi=150, bbox_inches='tight')
    print('Chart 1 saved to iou_vs_groundtruth.png')

    # ── Chart 2: IoU scores only, sorted, coloured ───────────────────────────
    fig2, ax3 = plt.subplots(figsize=(14, 5))

    bar_colors = ['#2ecc71' if s >= 0.75 else '#e67e22' if s >= 0.5 else '#e74c3c'
                  for s in sorted_scores]
    bars = ax3.barh(sorted_names, sorted_scores, color=bar_colors,
                    edgecolor='white', height=0.7)

    ax3.axvline(0.75, color='red', linestyle='--', linewidth=2,
                label='Target (0.75)', zorder=3)
    ax3.axvline(mean_iou, color='#2c3e50', linestyle=':', linewidth=2,
                label=f'Mean IoU ({mean_iou:.3f})', zorder=3)

    for bar, score in zip(bars, sorted_scores):
        ax3.text(score + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{score:.3f}', va='center', ha='left', fontsize=7,
                 color='#2c3e50')

    ax3.set_xlim(0, 1.08)
    ax3.set_xlabel('IoU Score', fontsize=12)
    ax3.set_title('IoU Score per Page (sorted) — Kraken BLLA Zero-Shot Segmentation',
                  fontsize=13, fontweight='bold', pad=15)
    ax3.set_facecolor('#f8f9fa')
    ax3.grid(axis='x', linestyle='--', alpha=0.4)
    ax3.tick_params(axis='y', labelsize=7)

    green_patch = mpatches.Patch(color='#2ecc71', label=f'Pass >= 0.75 ({n_pass} pages)')
    orange_patch = mpatches.Patch(color='#e67e22', label=f'Partial 0.50-0.75 ({sum(1 for s in scores if 0.5 <= s < 0.75)} pages)')
    red_patch = mpatches.Patch(color='#e74c3c', label=f'Fail < 0.50 ({sum(1 for s in scores if s < 0.5)} pages)')
    ax3.legend(handles=[green_patch, orange_patch, red_patch],
               loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig('iou_scores.png', dpi=150, bbox_inches='tight')
    print('Chart 2 saved to iou_scores.png')


if __name__ == '__main__':
    main()
