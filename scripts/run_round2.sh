#!/usr/bin/env bash
# Round 2 end to end on a rented GPU: setup -> data -> train -> score -> auto-stop, with no manual
# steps in between (round 1 lost hours to idle gaps and hand-run steps).
#
#     nohup bash scripts/run_round2.sh /root/data /root/runs > /root/round2.log 2>&1 &
#
# Optional environment:
#     RUN_CONTROL=1   also train the control run without DENTEX (Radboud + DenPAR only)
#     AUTOSTOP=0      do not stop the Vast.ai instance afterwards (default: stop 90 min after finishing)
#     EPOCHS=12       training epochs per run
#
# Each run is scored with its best and final checkpoints on all three test sets:
# Radboud bitewings + DENTEX crops (scripts/evaluate.sh) and held-out REAL periapicals
# (scripts/evaluate_real.sh). A re-run resumes an interrupted training from its last checkpoint.
# The final summary is written to <runs>/SUMMARY.txt.
set -uo pipefail
cd "$(dirname "$0")/.."
DATA="$(mkdir -p "${1:-/root/data}" && cd "${1:-/root/data}" && pwd)"
RUNS="$(mkdir -p "${2:-/root/runs}" && cd "${2:-/root/runs}" && pwd)"
LOG="${ROUND2_LOG:-/root/round2.log}"
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }

if [ "${AUTOSTOP:-1}" = 1 ]; then
    nohup bash scripts/vast_autostop.sh "$LOG" $$ 90 > "$RUNS/autostop.log" 2>&1 &
    stamp "auto-stop watcher started (90 min after the pipeline ends)"
fi

stamp "SETUP"
# the PyTorch wheels are large and some hosts' links to PyPI are slow: wait longer, retry setup
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-180}" PIP_RETRIES="${PIP_RETRIES:-10}"
for attempt in 1 2 3; do
    if [ -x /opt/venv/bin/python ] && /opt/venv/bin/python -c "import mmdet, mmcv.ops" 2>/dev/null; then break; fi
    [ "$attempt" = 1 ] || stamp "setup attempt $attempt"
    bash scripts/setup_gpu_native.sh && break
    [ "$attempt" = 3 ] && { stamp "SETUP FAILED"; exit 1; }
done
# shellcheck disable=SC1091
source /opt/venv/bin/activate

stamp "DATA"
bash scripts/fetch_data.sh "$DATA" > "$RUNS/fetch_data.log" 2>&1 || { stamp "DATA FAILED"; tail -30 "$RUNS/fetch_data.log"; exit 1; }
grep -E "DIFFERENT|WARNING" "$RUNS/fetch_data.log" || stamp "all data fingerprints match"
eval "$(grep '^export BITEWINGS_' "$RUNS/fetch_data.log")"
export BITEWINGS_WORK_DIRS="$RUNS" BITEWINGS_BATCH_SIZE=1 BITEWINGS_ACCUMULATE=2
export BITEWINGS_EPOCHS="${EPOCHS:-12}" BITEWINGS_NUM_WORKERS=4

train_and_score() {  # $1 run name; remaining args: extra env assignments
    local name=$1; shift
    stamp "TRAIN $name"
    # --resume only when there is a checkpoint: tools/train.py --resume also clears load_from,
    # so on a fresh run it would silently train from random weights instead of the released model
    local resume=()
    [ -f "$RUNS/$name/last_checkpoint" ] && resume=(--resume) && stamp "resuming $name"
    env "$@" BITEWINGS_RUN_NAME="$name" python onedl-mmdetection/tools/train.py \
        bitewings/configs/config_finetune_round2.py "${resume[@]}" >> "$RUNS/$name.train.log" 2>&1 \
        || { stamp "TRAIN $name FAILED"; tail -25 "$RUNS/$name.train.log"; return 1; }
    local best last
    best=$(ls -t "$RUNS/$name"/best_*.pth 2>/dev/null | head -1)
    last=$(ls "$RUNS/$name"/epoch_*.pth 2>/dev/null | sort -V | tail -1)
    for ckpt in "$best" "$last"; do
        [ -n "$ckpt" ] || continue
        local tag="${name}_$( [ "$ckpt" = "$best" ] && echo best || echo last )"
        stamp "SCORE $tag"
        bash scripts/evaluate.sh "$ckpt" "$tag" > "$RUNS/$tag.evaluate.log" 2>&1 || stamp "SCORE $tag (Radboud/DENTEX) FAILED"
        bash scripts/evaluate_real.sh "$ckpt" "$tag" > "$RUNS/$tag.evaluate_real.log" 2>&1 || stamp "SCORE $tag (real) FAILED"
    done
}

train_and_score round2
[ "${RUN_CONTROL:-0}" = 1 ] && train_and_score round2_nodentex BITEWINGS_USE_DENTEX=0

stamp "SUMMARY"
{
    for d in "$RUNS"/eval/*/; do
        tag=$(basename "$d")
        echo "== $tag"
        grep -oE "fdi_label/segm_mAP: [0-9.]+" "$d/radboud_val_metrics.txt" 2>/dev/null
        grep -E "^(caries|calculus) " "$d/findings.txt" 2>/dev/null
        echo "-- DENTEX crops:"; grep -E "^(front|back) " "$d/dentex_eval.txt" 2>/dev/null
        for s in 0.5 0.3; do
            echo "-- REAL periapicals, score >= $s:"
            grep -E "^(front|back|front crops|back crops) " "$d/real_eval_score$s.txt" 2>/dev/null
        done
    done
} | tee "$RUNS/SUMMARY.txt"
stamp PIPELINE_COMPLETE
