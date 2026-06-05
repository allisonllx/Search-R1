#!/usr/bin/env bash
# Self-healing wrapper around scripts/download.py for a flaky HF connection.
# Plain (resumable) downloader + short read timeout + retry loop:
# a stalled read raises after the timeout and the loop restarts, resuming
# from the .incomplete partial instead of hanging forever.
set -u
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1   # repo root, wherever this is cloned

export HF_HUB_ENABLE_HF_TRANSFER=0   # plain downloader => correct resume
export HF_HUB_DOWNLOAD_TIMEOUT=30    # no data for 30s => raise, then retry

attempt=0
while true; do
  attempt=$((attempt+1))
  echo "=== attempt $attempt @ $(date '+%F %T') ==="
  python scripts/download.py --save_path index_document
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "=== SUCCESS @ $(date '+%F %T') after $attempt attempt(s) ==="
    break
  fi
  echo "=== exited rc=$rc @ $(date '+%F %T'); retrying in 10s (resumes from partial) ==="
  sleep 10
done
