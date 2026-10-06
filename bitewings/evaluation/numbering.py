"""Tooth detection and numbering accuracy, split into front and back teeth.

Matches predicted teeth to ground-truth teeth by mask IoU (greedy, highest first) and
reports, separately for front teeth (incisors + canines) and back teeth:

    found       share of ground-truth teeth matched by a prediction
    numbered    share of found teeth whose FDI number is right
    overall     share of ground-truth teeth found AND numbered right
    extra       predictions that match no ground-truth tooth, per image

Wrong numbers are broken down into left/right mix-ups (same tooth, other side of the
same jaw, e.g. 11 vs 21), off-by-one (neighbour in the same quadrant) and other.

    python bitewings/evaluation/numbering.py --gt <annotations_gt.json> --pred <detections.pkl>

Predictions are the pickle written by config_inference.py (DumpMulticlassDetResults).
The default IoU threshold is 0.3 because DENTEX outlines are coarse polygons while the
model draws tight masks; a looser threshold avoids scoring outline style as a miss.
"""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pycocotools.mask as maskUtils

FDIS = [q * 10 + n for q in (1, 2, 3, 4) for n in range(1, 9)]


def is_front(fdi: int) -> bool:
    return fdi % 10 <= 3


def error_kind(gt: int, pred: int) -> str:
    gq, gn, pq, pn = gt // 10, gt % 10, pred // 10, pred % 10
    if gn == pn and {gq, pq} in ({1, 2}, {3, 4}):
        return 'left/right'
    if gq == pq and abs(gn - pn) == 1:
        return 'off by one'
    return 'other'


def load_predictions(path: Path, score_thr: float) -> dict:
    preds = {}
    for result in pickle.load(open(path, 'rb')):
        inst = result['pred_instances']
        teeth = []
        for i in range(len(inst['labels'])):
            if float(inst['scores'][i]) < score_thr:
                continue
            layers = inst['masks'][i]
            # tooth = union of all layers except calculus, as in mmdet2coco.py
            rle = maskUtils.merge(layers[:-1]) if isinstance(layers, list) else layers
            teeth.append((FDIS[int(inst['labels'][i])], rle))
        preds[Path(str(result['img_path'])).name] = teeth
    return preds


def evaluate(gt_path: Path, pred_path: Path, score_thr: float, iou_thr: float) -> dict:
    gt = json.loads(gt_path.read_text())
    fdi_of = {c['id']: int(c['name'][-2:]) for c in gt['categories']}
    gt_teeth = {img['id']: [] for img in gt['images']}
    for ann in gt['annotations']:
        gt_teeth[ann['image_id']].append((fdi_of[ann['category_id']], ann['segmentation']))
    preds = load_predictions(pred_path, score_thr)

    counts = Counter()
    errors = Counter()
    missing_predictions = 0
    for img in gt['images']:
        if img['file_name'] not in preds:
            missing_predictions += 1
            continue
        g, p = gt_teeth[img['id']], preds[img['file_name']]
        kind = img.get('crop_type', 'all')
        matched_g, matched_p = set(), set()
        if g and p:
            ious = maskUtils.iou([rle for _, rle in p], [rle for _, rle in g], [0] * len(g))
            for pi, gi in sorted(np.argwhere(ious >= iou_thr), key=lambda ij: -ious[ij[0], ij[1]]):
                if pi in matched_p or gi in matched_g:
                    continue
                matched_p.add(pi)
                matched_g.add(gi)
                gt_fdi, pred_fdi = g[gi][0], p[pi][0]
                region = 'front' if is_front(gt_fdi) else 'back'
                for key in (region, f'{kind} crops', 'all'):
                    counts[key, 'found'] += 1
                    counts[key, 'correct'] += gt_fdi == pred_fdi
                if gt_fdi != pred_fdi:
                    errors[region, error_kind(gt_fdi, pred_fdi)] += 1
        for gt_fdi, _ in g:
            region = 'front' if is_front(gt_fdi) else 'back'
            for key in (region, f'{kind} crops', 'all'):
                counts[key, 'gt'] += 1
        for key in (f'{kind} crops', 'all'):
            counts[key, 'images'] += 1
            counts[key, 'extra'] += len(p) - len(matched_p)

    summary = {'score_thr': score_thr, 'iou_thr': iou_thr, 'images_without_predictions': missing_predictions}
    for key in ('front', 'back', 'front crops', 'back crops', 'all'):
        n_gt = counts[key, 'gt']
        if not n_gt:
            continue
        found, correct = counts[key, 'found'], counts[key, 'correct']
        row = {'gt_teeth': n_gt, 'found': found / n_gt, 'numbered': correct / found if found else 0.0,
               'overall': correct / n_gt}
        if counts[key, 'images']:
            row['extra_per_image'] = counts[key, 'extra'] / counts[key, 'images']
        summary[key] = row
    summary['wrong_numbers'] = {f'{region} {kind}': n for (region, kind), n in sorted(errors.items())}
    return summary


def print_summary(summary: dict, title: str = ''):
    if title:
        print(f'== {title}')
    print(f"{'':12s} {'teeth':>6s} {'found':>7s} {'numbered':>9s} {'overall':>8s} {'extra/img':>9s}")
    for key in ('front', 'back', 'front crops', 'back crops', 'all'):
        if key in summary:
            r = summary[key]
            extra = f"{r['extra_per_image']:9.2f}" if 'extra_per_image' in r else ' ' * 9
            print(f"{key:12s} {r['gt_teeth']:6d} {r['found']:7.1%} {r['numbered']:9.1%} {r['overall']:8.1%} {extra}")
    if summary['wrong_numbers']:
        print('wrong numbers:', ', '.join(f'{k}: {v}' for k, v in summary['wrong_numbers'].items()))
    if summary['images_without_predictions']:
        print(f"WARNING: {summary['images_without_predictions']} images have no predictions")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--gt', type=Path, required=True, help='annotations_gt.json (tooth_<FDI> masks)')
    parser.add_argument('--pred', type=Path, required=True, help='detections.pkl from config_inference.py')
    parser.add_argument('--score-thr', type=float, default=0.5)
    parser.add_argument('--iou-thr', type=float, default=0.3)
    parser.add_argument('--out', type=Path, help='optional JSON file for the summary')
    parser.add_argument('--title', default='')
    args = parser.parse_args()
    summary = evaluate(args.gt, args.pred, args.score_thr, args.iou_thr)
    print_summary(summary, args.title)
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
