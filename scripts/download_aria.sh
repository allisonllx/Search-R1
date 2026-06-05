#!/usr/bin/env bash
# Robust parallel download of the Search-R1 index + corpus via aria2c.
# - 16 parallel connections per file  -> high aggregate throughput on a throttled link
# - -c + .aria2 control files          -> resume across restarts/stalls
# - --lowest-speed-limit               -> a stalled connection is dropped & retried
# - --max-tries=0                      -> retry forever
# Files land directly in index_document/ with the names the downstream code expects.
set -u
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1   # repo root, wherever this is cloned
OUT=index_document
mkdir -p "$OUT"

cat > "$OUT/.aria_urls.txt" <<'EOF'
https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_aa
  dir=index_document
  out=part_aa
https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_ab
  dir=index_document
  out=part_ab
https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz
  dir=index_document
  out=wiki-18.jsonl.gz
EOF

attempt=0
while true; do
  attempt=$((attempt+1))
  echo "=== aria2c attempt $attempt @ $(date '+%F %T') ==="
  aria2c \
    -i "$OUT/.aria_urls.txt" \
    -c \
    -x16 -s16 --max-connection-per-server=16 \
    --min-split-size=10M \
    --max-tries=0 --retry-wait=5 \
    --timeout=30 --connect-timeout=30 \
    --lowest-speed-limit=30K \
    --file-allocation=none \
    --auto-file-renaming=false --allow-overwrite=false \
    --summary-interval=30 --console-log-level=warn
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "=== ALL DOWNLOADS COMPLETE @ $(date '+%F %T') after $attempt attempt(s) ==="
    break
  fi
  echo "=== aria2c exited rc=$rc @ $(date '+%F %T'); restarting in 10s (resumes via .aria2) ==="
  sleep 10
done
