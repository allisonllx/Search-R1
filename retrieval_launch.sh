
# Resolve paths relative to this script; override INDEX_DIR in .env if the index
# lives elsewhere. Keeps machine-specific absolute paths out of the repo.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && { set -a; . "$SCRIPT_DIR/.env"; set +a; }
file_path="${INDEX_DIR:-$SCRIPT_DIR/index_document}"
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
