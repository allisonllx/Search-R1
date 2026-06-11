#!/bin/bash

# Resolve paths relative to this script; override INDEX_DIR in .env if the index
# lives elsewhere. Keeps machine-specific absolute paths out of the repo.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && { set -a; . "$SCRIPT_DIR/.env"; set +a; }
file_path="${INDEX_DIR:-$SCRIPT_DIR/index_document}"
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

# Default port, can be overridden with RETRIEVER_PORT env var
PORT=${RETRIEVER_PORT:-8002}

# Minimum free GPU memory (MiB) needed to host the index on the GPU.
# Defaults to the index file size + a margin for the model and faiss work buffers.
# Override the whole thing with REQUIRED_GPU_MEM_MIB, or just the margin with GPU_MEM_MARGIN_MIB.
INDEX_BYTES=$(stat -c %s "$index_file")
INDEX_MIB=$(( INDEX_BYTES / 1024 / 1024 ))
MARGIN_MIB=${GPU_MEM_MARGIN_MIB:-4096}
REQUIRED_MIB=${REQUIRED_GPU_MEM_MIB:-$(( INDEX_MIB + MARGIN_MIB ))}

# Override auto-selection: pin the retriever to specific GPU(s) by physical index,
# e.g.  FORCE_RETRIEVER_GPUS=1 ./retrieval_launch_smart.sh   (comma-separate to shard,
# e.g.  FORCE_RETRIEVER_GPUS=1,2). Skips the free-memory best-fit/shard search below.
# Use this to keep the retriever on a card disjoint from the training job's GPUs.
if [ -n "${FORCE_RETRIEVER_GPUS:-}" ]; then
  GPU_SET=$FORCE_RETRIEVER_GPUS
  echo "[retriever] FORCE_RETRIEVER_GPUS set -> pinning to GPU(s) $GPU_SET"
fi

# --- Backend selection -------------------------------------------------------
# RETRIEVER_BACKEND picks how the index is searched:
#   faiss_gpu : faiss on GPU. BROKEN on this H200 — faiss-gpu 1.7.2 has no working
#               sm_90 search kernel, aborts with "cublas failed (13)" in fp16 AND
#               fp32. Kept for other machines.
#   faiss_cpu : faiss on CPU (~33 s/query batch but stable). Encoder still on GPU,
#               so only a tiny card is needed.
#   torch_gpu : brute-force IndexFlatIP search reimplemented in PyTorch on GPU
#               (retrieval_server_torch.py). Uses PyTorch's working cublas -> fast
#               (~ms) AND sidesteps the faiss-gpu bug. Needs one dedicated ~46 GB card.
# Back-compat: if RETRIEVER_BACKEND is unset, RETRIEVER_CPU=1 -> faiss_cpu else faiss_gpu.
if [ -z "${RETRIEVER_BACKEND:-}" ]; then
  if [ -n "${RETRIEVER_CPU:-}" ]; then RETRIEVER_BACKEND=faiss_cpu; else RETRIEVER_BACKEND=faiss_gpu; fi
fi

SERVER_SCRIPT="search_r1/search/retrieval_server.py"
FAISS_GPU_FLAG="--faiss_gpu"
TORCH_EXTRA_ARGS=""
ALLOW_SHARD=1
case "$RETRIEVER_BACKEND" in
  faiss_gpu)
    echo "[retriever] backend=faiss_gpu (WARNING: broken on this H200 sm_90)";;
  faiss_cpu)
    FAISS_GPU_FLAG=""; REQUIRED_MIB=2048
    echo "[retriever] backend=faiss_cpu -> faiss on CPU; encoder needs only ~1 GB GPU";;
  torch_gpu)
    SERVER_SCRIPT="search_r1/search/retrieval_server_torch.py"
    FAISS_GPU_FLAG=""          # the torch server manages the GPU itself
    ALLOW_SHARD=0             # single dedicated card only (no faiss sharding)
    REQUIRED_MIB=${TORCH_REQUIRED_MIB:-46000}   # fp16 DB ~32 GB + sim buffer + encoder
    [ -n "${RETRIEVER_DB_FP32:-}" ] && TORCH_EXTRA_ARGS="$TORCH_EXTRA_ARGS --db_fp32"
    [ -n "${GPU_QUERY_BATCH:-}" ]   && TORCH_EXTRA_ARGS="$TORCH_EXTRA_ARGS --gpu_query_batch $GPU_QUERY_BATCH"
    echo "[retriever] backend=torch_gpu -> PyTorch brute-force IP search on one GPU (~46 GB)";;
  *)
    echo "ERROR: unknown RETRIEVER_BACKEND='$RETRIEVER_BACKEND' (use faiss_gpu|faiss_cpu|torch_gpu)" >&2
    exit 1;;
esac

# Snapshot free memory per GPU once (MiB, spaces stripped: "id,free" per line).
GPU_FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sed 's/ //g')

# 1) Prefer a SINGLE card: among GPUs with at least REQUIRED_MIB free, pick the one
#    with the LEAST free (best fit). This tucks the job into the tightest-fitting GPU
#    and leaves the larger empty GPUs available for other work. Skipped when pinned.
if [ -z "$GPU_SET" ]; then
GPU_SET=$(echo "$GPU_FREE" | awk -F',' -v req="$REQUIRED_MIB" '
  { if ($2 + 0 >= req && (best == "" || $2 + 0 < best)) { best = $2 + 0; id = $1 } }
  END { if (id != "") print id }')
fi

# 2) Fall back to SHARDING across the FEWEST cards when no single card fits. faiss
#    (co.shard=True) splits the index into N roughly-equal shards across all visible
#    cards, so each card must hold INDEX_MIB/N plus a per-card overhead (CUDA context
#    + faiss temp; the encoder also lands on the first card). Take the most-free cards
#    first so the smallest card in the set still holds its even shard.
if [ -z "$GPU_SET" ] && [ "$ALLOW_SHARD" = "1" ]; then
  PER_CARD_MIB=${SHARD_PER_CARD_MIB:-1536}
  GPU_SET=$(echo "$GPU_FREE" | sort -t',' -k2 -n -r | \
    awk -F',' -v idx="$INDEX_MIB" -v pc="$PER_CARD_MIB" '
      { n += 1; ids = (ids == "" ? $1 : ids","$1);
        if ($2 + 0 >= idx / n + pc) { print ids; found = 1; exit } }
      END { if (!found) exit 1 }')
fi

if [ -z "$GPU_SET" ]; then
  echo "ERROR: GPUs cannot host the index: need ${REQUIRED_MIB} MiB (index ${INDEX_MIB} MiB + margin ${MARGIN_MIB} MiB), even sharded across all cards." >&2
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader >&2
  exit 1
fi

case "$GPU_SET" in
  *,*) echo "Index needs ~${INDEX_MIB} MiB; no single card fits — sharding across GPUs $GPU_SET" ;;
  *)   echo "Index needs ~${REQUIRED_MIB} MiB free; selected GPU $GPU_SET (best fit — smallest GPU that still fits)" ;;
esac
echo "Using port $PORT"

# --- Re-check guard: close the time-of-check/time-of-use race ----------------
# Selection above sampled free memory once; on a shared box another job can grab a
# chosen card in the seconds before this server allocates its DB + reserved sim
# buffer. Re-query the cards we are about to use and abort cleanly if any no longer
# has enough free, so we fail fast with a clear message instead of OOMing on the
# first query (HTTP 500 -> training abort). Mirrors run.sh's guard. This runs even
# when FORCE_RETRIEVER_GPUS is set — pinning to a card that is already full is the
# exact mistake we want to catch. Set SKIP_GPU_RECHECK=1 to bypass.
if [ -z "${SKIP_GPU_RECHECK:-}" ]; then
  NCARDS=$(echo "$GPU_SET" | tr ',' '\n' | grep -c .)
  if [ "$NCARDS" -gt 1 ]; then
    # Sharded: each card holds ~INDEX_MIB/N plus per-card overhead.
    PER_CARD_REQ=$(( INDEX_MIB / NCARDS + ${SHARD_PER_CARD_MIB:-1536} ))
  else
    PER_CARD_REQ=$REQUIRED_MIB
  fi
  FREE_NOW=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sed 's/ //g')
  for id in $(echo "$GPU_SET" | tr ',' ' '); do
    free=$(echo "$FREE_NOW" | awk -F',' -v g="$id" '$1==g{print $2+0}')
    if [ "${free:-0}" -lt "$PER_CARD_REQ" ]; then
      echo "ERROR: GPU $id dropped to ${free:-0} MiB free (< ${PER_CARD_REQ} needed) between selection and launch." >&2
      echo "       Another job likely grabbed it. Free a card or re-run to re-select. Current free memory:" >&2
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader >&2
      exit 1
    fi
  done
  echo "[retriever] re-check OK: GPU(s) [$GPU_SET] still have >= ${PER_CARD_REQ} MiB free each"
fi

export CUDA_VISIBLE_DEVICES=$GPU_SET

if [ "$RETRIEVER_BACKEND" = "torch_gpu" ]; then
  python "$SERVER_SCRIPT" --index_path "$index_file" \
                          --corpus_path "$corpus_file" \
                          --topk 3 \
                          --retriever_name "$retriever_name" \
                          --retriever_model "$retriever_path" \
                          --port "$PORT" \
                          --device cuda \
                          $TORCH_EXTRA_ARGS
else
  python "$SERVER_SCRIPT" --index_path "$index_file" \
                          --corpus_path "$corpus_file" \
                          --topk 3 \
                          --retriever_name "$retriever_name" \
                          --retriever_model "$retriever_path" \
                          --port "$PORT" \
                          $FAISS_GPU_FLAG
fi
