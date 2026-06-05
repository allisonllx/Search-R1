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

# Best fit: among GPUs with at least REQUIRED_MIB free, pick the one with the LEAST
# free memory. This tucks the job into the tightest-fitting GPU and leaves the
# larger empty GPUs available for other work.
GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F',' -v req="$REQUIRED_MIB" '
    { gsub(/ /, "", $1); gsub(/ /, "", $2);
      if ($2 + 0 >= req && (best == "" || $2 + 0 < best)) { best = $2 + 0; id = $1 } }
    END { if (id == "") exit 1; print id }')

if [ -z "$GPU_ID" ]; then
  echo "ERROR: no GPU has the required ${REQUIRED_MIB} MiB free (index ${INDEX_MIB} MiB + margin ${MARGIN_MIB} MiB)." >&2
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader >&2
  exit 1
fi

echo "Index needs ~${REQUIRED_MIB} MiB free; selected GPU $GPU_ID (best fit — smallest GPU that still fits)"
echo "Using port $PORT"

export CUDA_VISIBLE_DEVICES=$GPU_ID

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --port $PORT \
                                            --faiss_gpu
