#!/usr/bin/env bash
# Download and prepare all training/evaluation data, verifying every file.
#
#     source /opt/venv/bin/activate        # environment from setup_gpu_native.sh (or Docker)
#     bash scripts/fetch_data.sh /workspace/data
#
# Produces, under the given folder (re-running skips finished steps):
#     Netherlands/                Radboud bitewings + annotations_fdi.json + splits/   (CC BY 4.0)
#     checkpoints/hierarchical_chartfiling.pth                                         (CC BY 4.0)
#     dentex/quadrant_enumeration/  DENTEX numbered panoramics + splits/               (CC BY-NC-SA 4.0)
#     dentex/eval_crops/          fixed periapical-style test crops + annotations_gt.json
# and prints the BITEWINGS_* variables to export. Needs ~20 GB free during the DENTEX step.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mkdir -p "${1:?usage: fetch_data.sh <data dir>}" && cd "$1" && pwd)"
log() { echo -e "\n=== $*"; }

RADBOUD="https://webdav.data.ru.nl/rumc/carrest_t0000028a_dsc_813_v2"
STUDY="Automated%20Chart%20Filing%20on%20Bitewings%20Using%20Deep%20Learning%20Enhancing%20Clinical%20Diagnosis%20in%20a%20Multi-Center%20Study"
DENTEX_ZIP="https://huggingface.co/datasets/ibrahimhamamci/DENTEX/resolve/main/DENTEX/training_data.zip"

# SHA-256 of the files the published baseline was computed on
declare -A EXPECTED=(
    [dentex/quadrant_enumeration/splits/train_fdi.json]=b5c47732b991642d19f768ea037ddbe6d675f7bc8e6396c0219158951fd7a72a
    [dentex/quadrant_enumeration/splits/val_fdi.json]=8bd9062f49b4e858c8eae499f696ba3ef35f09fb832df564d3fdb722d7758108
    [dentex/quadrant_enumeration/splits/test_fdi.json]=b8c72329181ce86d6de3e08946d631ed7492d7ef1cc4eb584c4e7d85b9e21744
    [dentex/eval_crops/annotations_gt.json]=46a746ee8ad174b2e301ac7fc24589b1de38cff87b8d48fe068856b629e6d50f
    [dentex/eval_crops/images/coco.json]=07bf989ca0d225a7e6b394ffa3fce843070590fc853a4b875fa065acfd9da65c
)

log "Radboud bitewings and checkpoint -> $OUT"
mkdir -p "$OUT/Netherlands/images" "$OUT/checkpoints"
curl -sS --fail "$RADBOUD/MANIFEST.txt" -o "$OUT/MANIFEST.txt"
# MANIFEST lines: "<sha256> <study folder>/<relative path>"; keep this study's data and checkpoint
grep "Multi-Center Study/" "$OUT/MANIFEST.txt" \
    | sed -E 's#^([0-9a-f]+) [^/]+/#\1 #' \
    | grep -E ' (data/Netherlands/|checkpoints/hierarchical_chartfiling\.pth$)' > "$OUT/radboud_files.txt"
echo "$(wc -l < "$OUT/radboud_files.txt") files listed"
fetched=0
while read -r sha rel; do
    dest="$OUT/${rel#data/}"
    if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
        continue
    fi
    url_rel="$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$rel")"
    # the repository rate-limits (HTTP 429): pace requests and back off on retries
    sleep 0.3
    curl -sS --fail --retry 8 --retry-delay 5 -o "$dest" "$RADBOUD/$STUDY/$url_rel"
    [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ] || { echo "CHECKSUM MISMATCH: $rel"; exit 1; }
    fetched=$((fetched + 1))
done < "$OUT/radboud_files.txt"
echo "downloaded $fetched new files; all $(wc -l < "$OUT/radboud_files.txt") match MANIFEST.txt"

log "Radboud preprocessing (FDI matching + 5-fold splits)"
if [ ! -f "$OUT/Netherlands/splits/train_fdi_1.json" ]; then
    python "$REPO/bitewings/preprocess/preprocess.py" --data-root "$OUT/Netherlands"
fi
ls "$OUT/Netherlands/splits" | head -3

log "DENTEX numbered panoramics"
QE="$OUT/dentex/quadrant_enumeration"
if [ ! -f "$QE/train_quadrant_enumeration.json" ]; then
    mkdir -p "$OUT/dentex"
    curl -sSL --fail -C - --retry 5 --retry-delay 10 -o "$OUT/dentex/training_data.zip" "$DENTEX_ZIP"
    python3 - "$OUT/dentex" <<'EOF'
import sys, zipfile
from pathlib import Path
out = Path(sys.argv[1])
prefix = 'training_data/quadrant_enumeration/'
with zipfile.ZipFile(out / 'training_data.zip') as z:
    for name in z.namelist():
        if name.startswith(prefix) and not name.endswith('/'):
            target = out / 'quadrant_enumeration' / name[len(prefix):]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
print('extracted', len(list((out / 'quadrant_enumeration' / 'xrays').iterdir())), 'panoramics')
EOF
    rm -f "$OUT/dentex/training_data.zip"
fi
[ -f "$QE/splits/train_fdi.json" ] || python "$REPO/bitewings/dentex/convert.py" --dentex-root "$QE"

log "Fixed DENTEX evaluation crops"
[ -f "$OUT/dentex/eval_crops/annotations_gt.json" ] || \
    python "$REPO/bitewings/dentex/make_eval_crops.py" --dentex-root "$QE" --out-dir "$OUT/dentex/eval_crops"

log "Checking the files the baseline was computed on"
status=0
for rel in "${!EXPECTED[@]}"; do
    actual="$(sha256sum "$OUT/$rel" | cut -d' ' -f1)"
    if [ "$actual" = "${EXPECTED[$rel]}" ]; then
        echo "  OK        $rel"
    else
        echo "  DIFFERENT $rel"; status=1
    fi
done
[ $status -eq 0 ] || echo "WARNING: some files differ from the baseline run; scores are not directly comparable"

log "Done. Export:"
cat <<EOF
export BITEWINGS_DATA_ROOT=$OUT/Netherlands
export BITEWINGS_CHECKPOINTS=$OUT/checkpoints
export BITEWINGS_DENTEX_ROOT=$QE
export BITEWINGS_EVAL_CROPS=$OUT/dentex/eval_crops
EOF
