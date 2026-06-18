"""
evaluate_transcription.py — Compute CER and WER against CREMMA ground truth.
"""
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from src.utils import compute_cer, compute_wer


def load_gt_transcriptions(alto_xml_path):
    tree = ET.parse(alto_xml_path)
    root = tree.getroot()
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'
    lines = []
    for line in root.iter(f'{ns}TextLine'):
        parts = []
        for string in line.iter(f'{ns}String'):
            content = string.get('CONTENT', '')
            if content:
                parts.append(content)
        text = ' '.join(parts).strip()
        if text:
            lines.append(text)
    return lines


def main():
    with open('results/kraken_output.json') as f:
        predictions = json.load(f)

    pred_by_page = {}
    for p in predictions:
        pred_by_page.setdefault(p['page'], []).append(p['transcription'])

    gt_dir = Path('data/cremma/data')
    page_cers, page_wers, page_names = [], [], []
    all_preds, all_refs = [], []

    for gt_xml in sorted(gt_dir.rglob('*.xml')):
        page = gt_xml.stem
        if page not in pred_by_page:
            continue
        gt_lines = load_gt_transcriptions(str(gt_xml))
        pred_lines = pred_by_page[page]
        if not gt_lines or not pred_lines:
            continue
        n = min(len(gt_lines), len(pred_lines))
        gt_lines = gt_lines[:n]
        pred_lines = pred_lines[:n]
        cer = compute_cer(pred_lines, gt_lines)
        wer = compute_wer(pred_lines, gt_lines)
        page_cers.append(cer)
        page_wers.append(wer)
        page_names.append(page[:18])
        all_preds.extend(pred_lines)
        all_refs.extend(gt_lines)
        print(f'{page}: CER={cer:.3f} WER={wer:.3f} ({n} lines)')

    if not page_cers:
        print('No results found.')
        return

    overall_cer = compute_cer(all_preds, all_refs)
    overall_wer = compute_wer(all_preds, all_refs)

    print(f'\n{"="*50}')
    print(f'Pages evaluated : {len(page_cers)}')
    print(f'Overall CER     : {overall_cer*100:.1f}%')
    print(f'Overall WER     : {overall_wer*100:.1f}%')
    print(f'Mean page CER   : {np.mean(page_cers)*100:.1f}%')
    print(f'Best CER        : {min(page_cers)*100:.1f}%')
    print(f'Worst CER       : {max(page_cers)*100:.1f}%')
    print(f'{"="*50}')

    idx = np.argsort(page_cers)
    sorted_names = [page_names[i] for i in idx]
    sorted_cers = [page_cers[i] for i in idx]
    sorted_wers = [page_wers[i] for i in idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#f0f0f0')

    # ── Chart 1: CER histogram ────────────────────────────────────────────────
    bins = np.linspace(0, max(sorted_cers) + 0.05, 25)
    n_good = sum(1 for c in page_cers if c <= 0.10)
    n_mid  = sum(1 for c in page_cers if 0.10 < c <= 0.50)
    n_bad  = sum(1 for c in page_cers if c > 0.50)

    axes[0].hist([c for c in page_cers if c <= 0.10],
                 bins=bins, color='#2ecc71', label=f'CER ≤ 10% ({n_good} pages)')
    axes[0].hist([c for c in page_cers if 0.10 < c <= 0.50],
                 bins=bins, color='#e67e22', label=f'CER 10-50% ({n_mid} pages)')
    axes[0].hist([c for c in page_cers if c > 0.50],
                 bins=bins, color='#e74c3c', label=f'CER > 50% ({n_bad} pages)')
    axes[0].axvline(0.10, color='green', linestyle='--', linewidth=2,
                    label='Target (10%)')
    axes[0].axvline(overall_cer, color='#2c3e50', linestyle=':', linewidth=2,
                    label=f'Overall CER ({overall_cer*100:.1f}%)')
    axes[0].set_xlabel('Character Error Rate', fontsize=12)
    axes[0].set_ylabel('Number of Pages', fontsize=12)
    axes[0].set_title('Distribution of CER across Pages', fontsize=13,
                      fontweight='bold')
    axes[0].set_facecolor('#f8f9fa')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0].legend(fontsize=9)

    # annotate best/worst
    axes[0].annotate(f'Best: {min(page_cers)*100:.1f}%',
                     xy=(min(page_cers), 0), xytext=(min(page_cers)+0.02, 3),
                     arrowprops=dict(arrowstyle='->', color='green'),
                     color='green', fontsize=9)

    # ── Chart 2: Top 15 best and worst pages ─────────────────────────────────
    top_n = min(10, len(sorted_cers))
    best_names = sorted_names[:top_n]
    best_cers  = sorted_cers[:top_n]
    worst_names = sorted_names[-top_n:][::-1]
    worst_cers  = sorted_cers[-top_n:][::-1]

    combined_names = [f'✓ {n}' for n in best_names] + [f'✗ {n}' for n in worst_names]
    combined_cers  = best_cers + worst_cers
    combined_colors = ['#2ecc71'] * top_n + ['#e74c3c'] * top_n

    y = np.arange(len(combined_names))
    bars = axes[1].barh(y, combined_cers, color=combined_colors,
                        edgecolor='white', height=0.7)
    axes[1].axvline(0.10, color='green', linestyle='--', linewidth=2,
                    label='Target (10%)')
    axes[1].axvline(overall_cer, color='#2c3e50', linestyle=':', linewidth=2,
                    label=f'Overall ({overall_cer*100:.1f}%)')
    for bar, cer in zip(bars, combined_cers):
        axes[1].text(cer + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{cer*100:.1f}%', va='center', fontsize=8,
                     color='#2c3e50', fontweight='bold')
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(combined_names, fontsize=8)
    axes[1].set_xlabel('Character Error Rate', fontsize=12)
    axes[1].set_title(f'Top {top_n} Best vs Worst Pages', fontsize=13,
                      fontweight='bold')
    axes[1].set_facecolor('#f8f9fa')
    axes[1].grid(axis='x', linestyle='--', alpha=0.4)
    axes[1].legend(fontsize=9)

    # divider line between best and worst
    axes[1].axhline(top_n - 0.5, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)
    axes[1].text(max(combined_cers)*0.5, top_n - 0.5 + 0.1,
                 '── worst pages below ──', ha='center', fontsize=8,
                 color='gray')

    plt.suptitle(
        f'Kraken Zero-Shot Transcription  |  CER: {overall_cer*100:.1f}%  |  WER: {overall_wer*100:.1f}%\n'
        f'Target: CER < 10%  (fine-tuning required to reach target)',
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig('transcription_results.png', dpi=150, bbox_inches='tight')
    print('\nSaved to transcription_results.png')


if __name__ == '__main__':
    main()
