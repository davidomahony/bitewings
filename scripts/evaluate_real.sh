#!/usr/bin/env bash
# Score one checkpoint on held-out REAL periapicals (DenPAR val + test, never trained on).
#
#     bash scripts/evaluate_real.sh <checkpoint.pth> <name>
#
# Runs config_inference.py on the real periapicals and bitewings/evaluation/numbering.py
# against DenPAR's FDI-numbered tooth masks: teeth found / correctly numbered, split into
# front (incisors + canines) and back teeth, and into anterior vs side images.
# Needs BITEWINGS_EVAL_REAL (printed by scripts/fetch_data.sh, built by
# bitewings/denpar/make_eval_set.py). Results go to ${BITEWINGS_WORK_DIRS:-work_dirs}/eval/<name>/.
# Complements scripts/evaluate.sh (Radboud bitewings + DENTEX crops).
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="$(realpath "${1:?usage: evaluate_real.sh <checkpoint.pth> <name>}")"
NAME="${2:?usage: evaluate_real.sh <checkpoint.pth> <name>}"
: "${BITEWINGS_EVAL_REAL:?set BITEWINGS_EVAL_REAL}"
OUT="$(realpath -m "${BITEWINGS_WORK_DIRS:-work_dirs}/eval/$NAME")"
mkdir -p "$OUT"

echo "=== $NAME: real periapicals (DenPAR val + test)"
# config_inference.py writes detections.pkl into the images folder; move it out straight away
BITEWINGS_IMAGES="$BITEWINGS_EVAL_REAL/images" python onedl-mmdetection/tools/test.py \
    bitewings/configs/config_inference.py "$CKPT" > "$OUT/real_eval.log" 2>&1
mv "$BITEWINGS_EVAL_REAL/images/detections.pkl" "$OUT/real_eval.pkl"
for score in 0.5 0.3; do
    python bitewings/evaluation/numbering.py --gt "$BITEWINGS_EVAL_REAL/annotations_gt.json" \
        --pred "$OUT/real_eval.pkl" --score-thr "$score" --out "$OUT/real_eval_score$score.json" \
        --title "$NAME, real periapicals, score >= $score" | tee "$OUT/real_eval_score$score.txt"
done
echo "=== results in $OUT"
