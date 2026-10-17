#!/usr/bin/env bash
# Stop this Vast.ai instance (its disk is kept) a grace period after a pipeline finishes or dies,
# so a rented GPU never sits idle. Run it in the background next to the pipeline:
#
#     nohup bash scripts/vast_autostop.sh <pipeline log> <pipeline pid> [grace minutes, default 90] &
#
# It stops after the log contains PIPELINE_COMPLETE, or when the pipeline process exits without
# it. Touch /root/keep_alive to hold off (e.g. while copying results). Uses the instance's own
# scoped key (CONTAINER_API_KEY in PID 1's environment), not an account key.
set -uo pipefail
LOG="${1:?usage: vast_autostop.sh <pipeline log> <pipeline pid> [grace minutes]}"
PID="${2:?pipeline pid}"
GRACE_MIN="${3:-90}"
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(CONTAINER_ID|CONTAINER_API_KEY)=' | sed 's/^/export /')"
[ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ] || { stamp "not a Vast.ai container; exiting"; exit 0; }

while true; do
    if grep -q PIPELINE_COMPLETE "$LOG" 2>/dev/null; then stamp "pipeline complete"; break; fi
    if ! kill -0 "$PID" 2>/dev/null; then stamp "pipeline process $PID gone without completing"; break; fi
    sleep 60
done
stamp "stopping in $GRACE_MIN min unless /root/keep_alive exists"
sleep $((GRACE_MIN * 60))
while [ -f /root/keep_alive ]; do sleep 300; done
stamp "stopping instance $CONTAINER_ID"
curl -s -X PUT -H "Authorization: Bearer $CONTAINER_API_KEY" -H "Content-Type: application/json" \
    -d '{"state": "stopped"}' "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/"
echo
