#!/usr/bin/env bash
# Score one checkpoint on both benchmarks, identically for the baseline and every trained model.
#
#     bash scripts/evaluate.sh <checkpoint.pth> <name>
#
# 1. Radboud fold-1 validation bitewings (config_hierarchical.py, original post-processing):
#    tooth numbering mAP and per-finding metrics, as reported by the authors' evaluators.
# 2. DENTEX fixed evaluation crops (config_inference.py + bitewings/evaluation/numbering.py):
#    front / back teeth found and correctly numbered.
# Needs BITEWINGS_DATA_ROOT and BITEWINGS_EVAL_CROPS (printed by scripts/fetch_data.sh).
# Results go to ${BITEWINGS_WORK_DIRS:-work_dirs}/eval/<name>/. SKIP_RADBOUD=1 runs only part 2
# (part 1's original post-processing needs more memory than CPU inference usually has).
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="$(realpath "${1:?usage: evaluate.sh <checkpoint.pth> <name>}")"
NAME="${2:?usage: evaluate.sh <checkpoint.pth> <name>}"
: "${BITEWINGS_DATA_ROOT:?set BITEWINGS_DATA_ROOT}" "${BITEWINGS_EVAL_CROPS:?set BITEWINGS_EVAL_CROPS}"
OUT="$(realpath -m "${BITEWINGS_WORK_DIRS:-work_dirs}/eval/$NAME")"
mkdir -p "$OUT"

if [ "${SKIP_RADBOUD:-0}" != 1 ]; then
    echo "=== $NAME: Radboud fold-1 validation"
    BITEWINGS_WORK_DIRS="$OUT/radboud" python onedl-mmdetection/tools/test.py \
        bitewings/configs/config_hierarchical.py "$CKPT" > "$OUT/radboud_val.log" 2>&1
    grep -E "Epoch\(test\).*\[[0-9]+/[0-9]+\] +[a-z_]+/" "$OUT/radboud_val.log" | tail -1 \
        | tee "$OUT/radboud_val_metrics.txt"
fi

echo "=== $NAME: DENTEX front/back numbering"
# config_inference.py writes detections.pkl into the images folder; move it out straight away
BITEWINGS_IMAGES="$BITEWINGS_EVAL_CROPS/images" python onedl-mmdetection/tools/test.py \
    bitewings/configs/config_inference.py "$CKPT" > "$OUT/dentex_eval.log" 2>&1
mv "$BITEWINGS_EVAL_CROPS/images/detections.pkl" "$OUT/dentex_eval.pkl"
python bitewings/evaluation/numbering.py --gt "$BITEWINGS_EVAL_CROPS/annotations_gt.json" \
    --pred "$OUT/dentex_eval.pkl" --out "$OUT/dentex_eval.json" --title "$NAME" | tee "$OUT/dentex_eval.txt"

echo "=== results in $OUT"
