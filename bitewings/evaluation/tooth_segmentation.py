import argparse
from collections import defaultdict
from pathlib import Path
import pickle
import sys

import pycocotools.mask as maskUtils
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from bitewings import paths


def evaluate_hierarchical(
    work_dir: Path,
    score_thr: float=0.5,
    remove_labels: torch.tensor=torch.tensor([0, 1, 8, 9, 16, 17, 24, 25]),
    # remove_labels: torch.tensor=torch.tensor([50]),
    keep_masks: slice=slice(0, 7),
    min_width: float=0.1,
    iou_thr: float=0.75,
):
    with open(work_dir / 'detections.pkl', 'rb') as f:
        results = pickle.load(f)

    n, fn, tp, fp = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    for result in results:
        gt_labels = result['gt_instances']['labels']
        gt_rles = [maskUtils.merge(rles[keep_masks]) for rles in result['gt_instances']['masks']]
        gt_widths = torch.tensor([maskUtils.toBbox(rle)[2] for rle in gt_rles])
        gt_keep = (
            (~torch.any(gt_labels[:, None] == remove_labels, dim=1))
            & (gt_widths >= result['ori_shape'][1] * min_width)
        )

        pred_scores = result['pred_instances']['scores']
        pred_labels = result['pred_instances']['labels']
        pred_rles = [maskUtils.merge(rles[keep_masks]) for rles in result['pred_instances']['masks']]
        pred_widths = torch.tensor([maskUtils.toBbox(rle)[2] for rle in pred_rles])
        pred_keep = (
            (pred_scores >= score_thr)
            & (~torch.any(pred_labels[:, None] == remove_labels, dim=1))
            & (pred_widths >= result['ori_shape'][1] * min_width)
        )

        gt_labels = gt_labels[gt_keep]
        gt_rles = [gt_rles[i] for i in torch.nonzero(gt_keep)[:, 0]]

        pred_labels = pred_labels[pred_keep]
        pred_rles = [pred_rles[i] for i in torch.nonzero(pred_keep)[:, 0]]

        ious = maskUtils.iou(gt_rles, pred_rles, [0]*len(pred_rles))
        same_label = gt_labels[:, None] == pred_labels

        matches = same_label & (ious >= iou_thr)

        n[-1] += matches.shape[0]
        fn[-1] += (matches.sum(1) == 0).sum()
        tp[-1] += matches.shape[0] - (matches.sum(1) == 0).sum()
        fp[-1] += (matches.sum(0) == 0).sum()


        for label in [None, *list(range(32))]:
            gt_keep = gt_labels == label if label is not None else torch.ones_like(gt_labels)
            gt_rles_label = [gt_rles[i] for i in torch.nonzero(gt_keep)[:, 0]]

            pred_keep = pred_labels == label if label is not None else torch.ones_like(pred_labels)
            pred_rles_label = [pred_rles[i] for i in torch.nonzero(pred_keep)[:, 0]]

            label = label if label is not None else 0
            tooth_type = label % 8
            if gt_rles_label and not pred_rles_label:
                n[100 + label] += len(gt_rles_label)
                n[tooth_type] += len(gt_rles_label)
                fn[tooth_type] += len(gt_rles_label)
                continue
            elif not gt_rles_label and pred_rles_label:
                fp[tooth_type] += len(pred_rles_label)
                continue
            elif not gt_rles_label and not pred_rles_label:
                continue

            ious = maskUtils.iou(gt_rles_label, pred_rles_label, [0]*len(pred_rles_label))
            matches = ious >= iou_thr

            n[100 + label] += matches.shape[0]
            n[tooth_type] += matches.shape[0]
            fn[tooth_type] += (matches.sum(1) == 0).sum()
            tp[tooth_type] += matches.shape[0] - (matches.sum(1) == 0).sum()
            fp[tooth_type] += (matches.sum(0) == 0).sum()

    for key in sorted(n):
        if key not in tp:
            continue

        prec = tp[key] / (tp[key] + fp[key])
        sens = tp[key] / (tp[key] + fn[key])
        f1 = 2 * tp[key] / (2 * tp[key] + fp[key] + fn[key])

        print()
        print(key)
        print('n', n[key])
        print('Precision', prec)
        print('Sensitivity', sens)
        print('F1-score', f1)


def evaluate_flat(
    work_dir: Path,
    score_thr: float=0.5,
    remove_labels: torch.tensor=torch.tensor([0, 1, 8, 9, 16, 17, 24, 25]),
    # remove_labels: torch.tensor=torch.tensor([50]),
    min_width: float=0.1,
    iou_thr: float=0.75,
):
    with open(work_dir / 'detections.pkl', 'rb') as f:
        results = pickle.load(f)

    n, fn, tp, fp = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    for result in results:
        gt_rles = result['gt_instances']['masks']
        gt_widths = torch.tensor([maskUtils.toBbox(rle)[2] for rle in gt_rles])
        gt_labels = result['gt_instances']['labels']
        gt_keep = (
            (gt_labels < 32)
            & (~torch.any(gt_labels[:, None] == remove_labels, dim=1))
            & (gt_widths >= result['ori_shape'][1] * min_width)
        )

        pred_scores = result['pred_instances']['scores']
        pred_labels = result['pred_instances']['labels']
        pred_rles = result['pred_instances']['masks']
        pred_widths = torch.tensor([maskUtils.toBbox(rle)[2] for rle in pred_rles])
        pred_keep = (
            (pred_labels < 32)
            & (pred_scores >= score_thr)
            & (~torch.any(pred_labels[:, None] == remove_labels, dim=1))
            & (pred_widths >= result['ori_shape'][1] * min_width)
        )


        gt_labels = gt_labels[gt_keep]
        gt_rles = [gt_rles[i] for i in torch.nonzero(gt_keep)[:, 0]]

        pred_labels = pred_labels[pred_keep]
        pred_rles = [pred_rles[i] for i in torch.nonzero(pred_keep)[:, 0]]

        ious = maskUtils.iou(gt_rles, pred_rles, [0]*len(pred_rles))
        same_label = gt_labels[:, None] == pred_labels

        matches = same_label & (ious >= iou_thr)

        n[-1] += matches.shape[0]
        fn[-1] += (matches.sum(1) == 0).sum()
        tp[-1] += matches.shape[0] - (matches.sum(1) == 0).sum()
        fp[-1] += (matches.sum(0) == 0).sum()


        for label in [None, *list(range(32))]:
            gt_keep = gt_labels == label if label is not None else torch.ones_like(gt_labels)
            gt_rles_label = [gt_rles[i] for i in torch.nonzero(gt_keep)[:, 0]]

            pred_keep = pred_labels == label if label is not None else torch.ones_like(pred_labels)
            pred_rles_label = [pred_rles[i] for i in torch.nonzero(pred_keep)[:, 0]]

            label = 0 if label is None else label
            tooth_type = label % 8
            if gt_rles_label and not pred_rles_label:
                n[100 + label] += len(gt_rles_label)
                n[tooth_type] += len(gt_rles_label)
                fn[tooth_type] += len(gt_rles_label)
                continue
            elif not gt_rles_label and pred_rles_label:
                fp[tooth_type] += len(pred_rles_label)
                continue
            elif not gt_rles_label and not pred_rles_label:
                continue

            ious = maskUtils.iou(gt_rles_label, pred_rles_label, [0]*len(pred_rles_label))
            matches = ious >= iou_thr

            n[100 + label] += matches.shape[0]
            n[tooth_type] += matches.shape[0]
            fn[tooth_type] += (matches.sum(1) == 0).sum()
            tp[tooth_type] += matches.shape[0] - (matches.sum(1) == 0).sum()
            fp[tooth_type] += (matches.sum(0) == 0).sum()

    for key in sorted(n):
        if key not in tp:
            continue

        prec = tp[key] / (tp[key] + fp[key])
        sens = tp[key] / (tp[key] + fn[key])
        f1 = 2 * tp[key] / (2 * tp[key] + fp[key] + fn[key])

        print()
        print(key)
        print('n', n[key])
        print('Precision', prec)
        print('Sensitivity', sens)
        print('F1-score', f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tooth segmentation and labelling metrics.')
    parser.add_argument(
        '--work-dirs', type=Path, default=paths.work_dirs(),
        help='Folder with the chart_filing_<method> run folders (default: $BITEWINGS_WORK_DIRS '
             f'or {paths.DEFAULT_WORK_DIRS}).',
    )
    base = parser.parse_args().work_dirs

    evaluate_hierarchical(paths.method_work_dir('hierarchical', base))
    evaluate_flat(paths.method_work_dir('maskrcnn', base))
    evaluate_flat(paths.method_work_dir('maskdino', base))
    evaluate_flat(paths.method_work_dir('sparseinst', base))
