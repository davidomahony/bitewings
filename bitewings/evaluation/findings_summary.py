"""Per-finding scores (implant, crown, ..., caries, calculus) for one hierarchical run.

Uses the authors' tooth-level matching from tooth_findings.py on the detections.pkl
that config_hierarchical.py's test run dumps, and reports per finding:

    n      teeth with the finding in the ground truth
    AUC    area under the ROC curve (threshold-free; best for comparing runs)
    F1     F1 at the threshold that maximises it, as in the paper (optimistic, but the same
           procedure for every run)

    python bitewings/evaluation/findings_summary.py <dir with detections.pkl> [--out summary.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[2]))

from bitewings.evaluation.tooth_findings import evaluate_hierarchical, optimal_thresholds  # noqa: E402

FINDINGS = ['implant', 'crown', 'pontic', 'filling', 'root canal', 'caries', 'calculus']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('work_dir', type=Path, help='folder containing detections.pkl')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()

    _, gts, scores = evaluate_hierarchical(args.work_dir)
    gts, scores = np.asarray(gts).astype(bool), np.asarray(scores, dtype=float)
    thresholds = np.asarray(optimal_thresholds(gts, scores))
    preds = scores >= thresholds

    summary = {}
    print(f"{'finding':12s} {'n':>5s} {'AUC':>6s} {'F1':>6s}")
    for i, name in enumerate(FINDINGS):
        n = int(gts[:, i].sum())
        auc = float(roc_auc_score(gts[:, i], scores[:, i])) if 0 < n < len(gts) else float('nan')
        tp = int((gts[:, i] & preds[:, i]).sum())
        f1 = 2 * tp / (2 * tp + int((~gts[:, i] & preds[:, i]).sum()) + int((gts[:, i] & ~preds[:, i]).sum())) \
            if n else float('nan')
        summary[name] = {'n': n, 'auc': auc, 'f1': f1}
        print(f'{name:12s} {n:5d} {auc:6.3f} {f1:6.3f}')
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
