import argparse
from pathlib import Path
import pickle
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as maskUtils
from sklearn.metrics import(
    roc_curve, roc_auc_score,
    ConfusionMatrixDisplay, confusion_matrix,
)
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from bitewings import paths


def draw_confusion_matrix(cm, labels, ax, recolor: bool=False):
    for i, label in enumerate(labels):
        if ' ' not in label:
            continue

        labels[i] = ' '.join([
            label.split(' ')[0],
            label.split(' ')[1].lower(),
        ])    

    if not recolor:
        norm_cm = cm / cm.sum(axis=1, keepdims=True)
    else:
        norm_cm = cm

    disp = ConfusionMatrixDisplay(norm_cm, display_labels=labels)
    disp.plot(cmap='magma', ax=ax, colorbar=False)

    # draw colorbar according to largest non-TN value
    if recolor:
        cm_without_tn = cm.copy()
        cm_without_tn[-1, -1] = 0
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=cm_without_tn.max())
        disp.ax_.images[0].set_norm(normalize)
        disp.text_[0, 0].set_color(disp.im_.cmap(0.0))
    else:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                disp.text_[i, j].set_text(cm[i, j])
    
    # draw y ticklabels vertically
    offset = matplotlib.transforms.ScaledTranslation(-0.1, 0, disp.figure_.dpi_scale_trans)
    for label in disp.ax_.get_yticklabels():
        label.set_rotation(90)
        label.set_transform(label.get_transform() + offset)
        label.set_ha('center')
        label.set_rotation_mode('anchor')


def optimal_thresholds(
    gts,
    scores,
):
    f1s_list = []
    thresholds = np.linspace(0, 1, 1000)
    for thr in thresholds:
        tps = (gts & (scores >= thr)).sum(0)
        fps = (~gts & (scores >= thr)).sum(0)
        fns = (gts & (scores < thr)).sum(0)

        f1s = 2 * tps / (2 * tps + fps + fns)
        f1s_list.append(f1s)

    f1s = np.stack(f1s_list)
    thrs = thresholds[np.argmax(f1s, axis=0)]

    return torch.from_numpy(thrs)


def compute_metrics(gts, scores):
    thrs = optimal_thresholds(gts, scores)
    print(thrs)
    preds = scores >= thrs
    
    tps = (gts & preds).sum(0)
    tns = (~gts & ~preds).sum(0)
    fps = (~gts & preds).sum(0)
    fns = (gts & ~preds).sum(0)

    precs = tps / (tps + fps)
    senss = tps / (tps + fns)
    specs = tns / (tns + fps)
    f1s = 2 * tps / (2 * tps + fps + fns)
    
    print('n', gts.sum(0))
    print('Precisions', precs, 'mean', precs.mean())
    print('Sensitivities', senss, 'mean', senss.mean())
    print('Specificities', specs, 'mean', specs.mean())
    print('F1-scores', f1s, 'mean', f1s.mean())

    cms = []
    for i in range(gts.shape[1]):
        if gts[:, i].sum() == 0:
            cm = np.array([[gts.shape[0], 0], [0, 0]])
        else:
            cm = confusion_matrix(gts[:, i], preds[:, i])
        cms.append(cm)

    return np.stack(cms)



def extract_tooth_findings(
    instances,
    score_thr: float=0.5,
    remove_labels: torch.tensor=torch.tensor([0, 1, 8, 9, 16, 17, 24, 25]),
    # remove_labels: torch.tensor=torch.tensor([50]),
    keep_masks: slice=slice(0, 8),
    min_width: float=0.1,
):
    labels = instances['labels']
    scores = instances['scores'] if 'scores' in instances else torch.ones_like(labels)
    rles = [maskUtils.merge(rles[keep_masks]) for rles in instances['masks']]
    widths = torch.tensor([maskUtils.toBbox(rle)[2] for rle in rles])
    keep = (
        (scores >= score_thr)
        & (~torch.any(labels[:, None] == remove_labels, dim=1))
        & (widths >= rles[0]['size'][1] * min_width)
    )

    out = instances['multilogits' if 'multilogits' in instances else 'multilabels'][keep][:, 1:]
    rles = [rles[i] for i in torch.nonzero(keep)[:, 0]]

    return out, rles


def evaluate_hierarchical(
    work_dir: Path,
    iou_thr: float=0.75,
):
    with open(work_dir / 'detections.pkl', 'rb') as f:
        results = pickle.load(f)

    gt_idxs_list, gts, scores = [], [], []
    for result in results:
        if 'hfT3aTlDQlqBLaOB' in result['img_path'].name:
            k = 3
        gt_findings, gt_rles = extract_tooth_findings(result['gt_instances'])
        pred_scores, pred_rles = extract_tooth_findings(result['pred_instances'])
        
        ious = maskUtils.iou(gt_rles, pred_rles, [0]*len(pred_rles))
        matches = ious >= iou_thr
        gt_idxs, pred_idxs = np.nonzero(matches)

        gt_findings = gt_findings[gt_idxs]
        pred_scores = pred_scores[pred_idxs]

        gt_idxs_list.append(gt_idxs)
        gts.extend(gt_findings)
        scores.extend(pred_scores)

    gts = torch.stack(gts) == 1
    scores = torch.stack(scores)

    gt_idxs = np.column_stack((
        np.concatenate(gt_idxs_list),
        np.array([idx for i, gt_list in enumerate(gt_idxs_list) for idx in [i]*len(gt_list)]),
    ))

    return gt_idxs, gts, scores


def match_tooth_findings(
    results,
    remove_labels: torch.tensor=torch.tensor([0, 1, 8, 9, 16, 17, 24, 25]),
    # remove_labels: torch.tensor=torch.tensor([50]),
    min_width: float=0.1,
    score_thr: float=0.5,
):
    rles = results['masks']
    areas = np.array([maskUtils.area(rle) for rle in rles])

    # remove zero-area annotations
    rles = [rles[i] for i in np.nonzero(areas > 0)[0]]
    labels = results['labels'][areas > 0]
    pred_scores = results['scores'][areas > 0] if 'scores' in results else torch.ones_like(labels)

    tooth_rles = [rles[i] for i in torch.nonzero(labels < 32)[:, 0]]
    tooth_areas = np.array([maskUtils.area(rle) for rle in tooth_rles])
    tooth_scores = pred_scores[labels < 32]
    finding_rles = [rles[i] for i in torch.nonzero(labels >= 32)[:, 0]]
    finding_areas = np.array([maskUtils.area(rle) for rle in finding_rles])
    finding_scores = pred_scores[labels >= 32]

    # make calculus intersect with corresponding tooth
    finding_labels = labels[labels >= 32]
    finding_masks = [maskUtils.decode(rle) for rle in finding_rles]
    finding_masks = [
        np.roll(mask, -50 + 100 * (label == 38), axis=1) if label >= 38 else mask
        for mask, label in zip(finding_masks, finding_labels)
    ]
    finding_rles = [maskUtils.encode(np.asfortranarray(mask)) for mask in finding_masks]

    # determine the matched teeth and tooth findings
    ious = maskUtils.iou(tooth_rles, finding_rles, [0]*len(finding_rles))
    overlaps = ious / (finding_areas / tooth_areas[:, None])
    tooth2findings = torch.zeros(len(tooth_rles), 7)
    for i, overlap_scores in enumerate(overlaps):
        if not np.any(overlap_scores >= 0.5):
            continue

        finding_idxs = np.nonzero(overlap_scores >= 0.5)[0]
        for idx in finding_idxs:
            label = min(6, finding_labels[idx] - 32)
            tooth2findings[i, label] = max(tooth2findings[i, label], finding_scores[idx])
        
    
    # remove the irrelevant teeth
    tooth_widths = torch.tensor([maskUtils.toBbox(rle)[2] for rle in tooth_rles])
    tooth_labels = labels[labels < 32]
    tooth_keep = (
        (tooth_scores >= score_thr)
        & (~torch.any(tooth_labels[:, None] == remove_labels, dim=1))
        & (tooth_widths >= rles[0]['size'][1] * min_width)
    )
    tooth_rles = [tooth_rles[i] for i in torch.nonzero(tooth_keep)[:, 0]]
    tooth2findings = tooth2findings[tooth_keep]

    return tooth_rles, tooth2findings


def evaluate_flat(
    hierarchical_work_dir: Path,
    work_dir: Path,
    score_thr: float=0.5,
    iou_thr: float=0.75,
):
    with open(hierarchical_work_dir / 'detections.pkl', 'rb') as f:
        hierarchical_results = pickle.load(f)

    with open(work_dir / 'detections.pkl', 'rb') as f:
        results = pickle.load(f)

    gt_idxs_list, ns, gts, scores = [], [], [], []
    for hierarchical_result, result in zip(hierarchical_results, results):
        gt_findings, gt_tooth_rles = extract_tooth_findings(hierarchical_result['gt_instances'])
        pred_tooth_rles, pred_scores = match_tooth_findings(result['pred_instances'])

        ious = maskUtils.iou(gt_tooth_rles, pred_tooth_rles, [0]*len(pred_tooth_rles))

        matches = ious >= iou_thr
        gt_idxs, pred_idxs = np.nonzero(matches)

        ns.append(len(gt_tooth_rles))
        gt_findings = gt_findings[gt_idxs]
        pred_scores = pred_scores[pred_idxs]

        gt_idxs_list.append(gt_idxs)
        gts.extend(gt_findings)
        scores.extend(pred_scores)

    ns = sum(ns)
    gts = torch.stack(gts) == 1
    scores = torch.stack(scores)

    gt_idxs = np.column_stack((
        np.concatenate(gt_idxs_list),
        np.array([idx for i, gt_list in enumerate(gt_idxs_list) for idx in [i]*len(gt_list)]),
    ))

    return gt_idxs, gts, scores


def evaluate_model(work_dir: Path):
    if 'hierarchical' in work_dir.name:
        return evaluate_hierarchical(work_dir)
    else:
        return evaluate_flat(work_dir.parent / 'chart_filing_hierarchical', work_dir)



def compute_gts_scores(base: Path):
    work_dirs = [
        paths.method_work_dir('sparseinst', base),
        paths.method_work_dir('maskrcnn', base),
        paths.method_work_dir('maskdino', base),
        paths.method_work_dir('hierarchical', base),
    ]
    gt_idxs, gts, scores = [], [], []
    for work_dir in work_dirs:
        gt_idx, gt, score = evaluate_model(work_dir)
        gt_idxs.append(gt_idx)
        gts.append(gt)
        scores.append(score)

    all_gt_idxs = np.concatenate(gt_idxs)

    unique, inverse, counts = np.unique(
        all_gt_idxs, axis=0, return_inverse=True, return_counts=True,
    )

    keep_tooth = (counts == len(gt_idxs))[inverse]
    keep_teeth = np.split(keep_tooth, np.cumsum([gt.shape[0] for gt in gts[:-1]]))

    gt_idxs_out, gts_out, scores_out = [], [], []
    for keep, gt_idxs, gt, score, work_dir in zip(keep_teeth, gt_idxs, gts, scores, work_dirs):
        gt_idxs_out.append(gt_idxs[keep])
        gts_out.append(gt[keep])
        scores_out.append(score[keep])
        
    return work_dirs, gt_idxs_out, gts_out, scores_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROC curves and confusion matrices for tooth findings.')
    parser.add_argument(
        '--work-dirs', type=Path, default=paths.work_dirs(),
        help='Folder with the chart_filing_<method> run folders (default: $BITEWINGS_WORK_DIRS '
             f'or {paths.DEFAULT_WORK_DIRS}).',
    )
    args = parser.parse_args()

    classes = ['Implant', 'Crown', 'Pontic', 'Filling', 'Root canal treatment', 'Caries lesion', 'Calculus deposit']

    fig_roc, axs_roc = plt.subplots(2, 4, figsize=(17.5, 8.5))
    fig_cm, axs_cm = plt.subplots(2, 4, figsize=(16, 8.5))
    axs_roc, axs_cm = axs_roc.flatten(), axs_cm.flatten()
    for name, work_dir, _, gt, score in zip(
        ['SparseInst', 'Mask R-CNN', 'Mask DINO', 'Hierarchical'],
        *compute_gts_scores(args.work_dirs),
    ):
        for i, label in enumerate(classes):
            if gt[:, i].sum() <= 1:
                continue
            roc = roc_curve(gt[:, i], score[:, i])
            auc = roc_auc_score(gt[:, i], score[:, i])
            axs_roc[i].set_title(label)
            axs_roc[i].plot(roc[0], roc[1], label=f'{name} (AUC={auc:.3f})')
        
        cms = compute_metrics(gt, score)

        if name != 'Hierarchical':
            continue

        for i, label in enumerate(classes):
            axs_cm[i].set_title(label)
            draw_confusion_matrix(cms[i], ['Absent', 'Present'], axs_cm[i])

    for ax in axs_roc[:-1]:
        ax.plot([0, 1], [0, 1], ls='--', c='k', label='Random (AUC=0.5)')
        ax.set_xlabel('True positive rate')
        ax.set_ylabel('False positive rate')
        ax.legend()
        ax.grid()
    
    axs_roc[-1].axis('off')
    axs_cm[-1].axis('off')

    fig_roc.tight_layout()
    fig_roc.savefig('rocs.png', dpi=300, bbox_inches='tight', pad_inches=None)
    plt.show()

    fig_cm.tight_layout()
    fig_cm.savefig('cms.png', dpi=300, bbox_inches='tight', pad_inches=None)
    plt.show()
