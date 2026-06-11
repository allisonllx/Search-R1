"""
GPU retrieval server that does the brute-force inner-product search in PyTorch
instead of faiss-gpu.

Why this exists
---------------
The wiki e5 index is an `IndexFlatIP` (exact inner-product brute force over
~21M normalized 768-d vectors). On this H200 box, faiss-gpu 1.7.2 has no working
Hopper/sm_90 cublas kernel, so its GPU search aborts the process on the first
query with `cublas failed (13)` in BOTH fp16 and fp32 (see
`memory/faiss-gpu-hopper-fp16.md`). Running faiss on CPU works but is ~33 s per
query batch, which starves the training GPUs.

A FlatIP search is just `scores = Q @ DB.T ; topk(scores)`. PyTorch's cublas works
fine on the H200 (training uses it), so we reconstruct the index vectors into a
GPU tensor once and do the matmul + topk ourselves. Latency drops from ~33 s to
a few ms, with no dependency on the broken faiss-gpu kernel.

Drop-in API
-----------
Exposes the same `POST /retrieve` contract as `retrieval_server.py`
(`{"queries": [...], "topk": k, "return_scores": bool}` -> `{"result": ...}`), so
the trainer's `retriever.url` works unchanged — only the launch command differs.

Memory (single H200, ~143 GB)
-----------------------------
  DB tensor   : 21M * 768 * 2 B (fp16) = ~32 GB   (fp32 = ~64 GB)
  sim buffer  : gpu_query_batch * 21M * 2 B       (256 -> ~10 GB, reserved up front)
  e5 encoder  : ~1 GB
Fits comfortably on one dedicated card. The DB is built in chunks so host RAM
peaks at ~(faiss index 64 GB + one small chunk), not 128 GB.

The similarity buffer is allocated ONCE at startup and held for the server's
lifetime (not allocated/freed per request). On a shared box this matters: it makes
the retriever claim its full ~46 GB footprint immediately — like vLLM/FSDP do — so a
co-located job can never grab the per-query scratch between requests and OOM us.
"""

import os
import sys
import argparse
from typing import List, Optional

import faiss
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Reuse the encoder + corpus helpers from the sibling faiss server (same dir is on
# sys.path[0] when this file is run as a script). Keeps query encoding / corpus
# formatting byte-for-byte identical to the original retriever.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_server import Encoder, load_corpus, load_docs  # noqa: E402


class TorchFlatIPRetriever:
    """Brute-force inner-product (cosine, for normalized e5) search on GPU in PyTorch."""

    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        retriever_name: str,
        retriever_model: str,
        topk: int = 3,
        pooling_method: str = "mean",
        query_max_length: int = 256,
        encoder_fp16: bool = True,
        device: str = "cuda",
        db_fp16: bool = True,
        gpu_query_batch: int = 256,
        db_load_chunk: int = 1_000_000,
    ):
        self.topk = topk
        self.device = device
        self.db_dtype = torch.float16 if db_fp16 else torch.float32
        self.gpu_query_batch = gpu_query_batch

        # 1) Read the faiss IndexFlatIP on CPU and stream its vectors onto the GPU
        #    in chunks so host RAM doesn't hold two full 64 GB copies at once.
        print(f"[torch-retriever] reading index {index_path} ...", flush=True)
        index = faiss.read_index(index_path)
        ntotal, dim = index.ntotal, index.d
        print(f"[torch-retriever] index: ntotal={ntotal:,} dim={dim} "
              f"-> building {self.db_dtype} DB tensor on {device}", flush=True)
        self.db = torch.empty((ntotal, dim), dtype=self.db_dtype, device=device)
        for start in range(0, ntotal, db_load_chunk):
            stop = min(start + db_load_chunk, ntotal)
            vecs = index.reconstruct_n(start, stop - start)  # (chunk, dim) fp32 numpy
            self.db[start:stop] = torch.from_numpy(vecs).to(device=device, dtype=self.db_dtype)
            del vecs
            if (start // db_load_chunk) % 5 == 0:
                print(f"[torch-retriever]   loaded {stop:,}/{ntotal:,}", flush=True)
        del index
        self.ntotal, self.dim = ntotal, dim
        print(f"[torch-retriever] DB on GPU: "
              f"{self.db.element_size() * self.db.nelement() / 1e9:.1f} GB", flush=True)

        # 1b) Reserve the peak (gpu_query_batch x ntotal) similarity buffer ONCE and
        #     hold it for the server's lifetime. Every request writes into this buffer
        #     (torch.matmul(..., out=self.sim_buf[:n])) instead of allocating a fresh
        #     ~10 GB block and freeing it. Pre-claiming it here means the retriever's
        #     full ~46 GB footprint is taken at startup — so a co-located job can't
        #     steal the scratch between requests and OOM the next matmul (the failure
        #     that returned HTTP 500 and aborted training).
        self.sim_buf = torch.empty((gpu_query_batch, ntotal), dtype=self.db_dtype, device=device)
        print(f"[torch-retriever] reserved sim buffer: "
              f"{self.sim_buf.element_size() * self.sim_buf.nelement() / 1e9:.1f} GB "
              f"(gpu_query_batch={gpu_query_batch})", flush=True)

        # 2) Corpus (for turning doc ids into text) and the e5 query encoder (on GPU).
        self.corpus = load_corpus(corpus_path)
        self.encoder = Encoder(
            model_name=retriever_name,
            model_path=retriever_model,
            pooling_method=pooling_method,
            max_length=query_max_length,
            use_fp16=encoder_fp16,
        )
        print("[torch-retriever] ready", flush=True)

    @torch.no_grad()
    def batch_search(self, query_list, num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results, scores = [], []
        # Sub-batch the queries so the (n x ntotal) similarity buffer stays bounded.
        for start in range(0, len(query_list), self.gpu_query_batch):
            qb = query_list[start:start + self.gpu_query_batch]
            emb = self.encoder.encode(qb)  # (n, dim) fp32 numpy, already normalized
            q = torch.from_numpy(emb).to(device=self.device, dtype=self.db_dtype)

            # Write similarities into the buffer reserved at startup instead of
            # allocating a fresh ~10 GB block here. self.sim_buf[:n] is a contiguous
            # view, so no new large allocation happens per request — which is what
            # stops a co-located job from stealing the scratch and OOMing us.
            sim = self.sim_buf[:q.shape[0]]
            torch.matmul(q, self.db.T, out=sim)       # (n, ntotal)
            topk_scores, topk_idx = torch.topk(sim, k=num, dim=1)  # (n, num)

            idxs = topk_idx.cpu().tolist()
            scs = topk_scores.float().cpu().tolist()

            flat = sum(idxs, [])
            docs = load_docs(self.corpus, flat)
            chunked = [docs[i * num:(i + 1) * num] for i in range(len(idxs))]
            results.extend(chunked)
            scores.extend(scs)

            # NB: do NOT free self.sim_buf or call torch.cuda.empty_cache() here —
            # holding the reserved buffer for the server's lifetime is the whole point.
            # Only drop the small per-request temporaries.
            del emb, q, topk_scores, topk_idx

        if return_score:
            return results, scores
        return results


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    topk = request.topk or config_topk
    # Always ask for scores internally, then shape the response by return_scores.
    # (Avoids the unpack bug in the original server when return_scores is False.)
    results, scores = retriever.batch_search(
        query_list=request.queries, num=topk, return_score=True
    )
    resp = []
    for single_result, single_scores in zip(results, scores):
        if request.return_scores:
            resp.append([{"document": doc, "score": sc}
                         for doc, sc in zip(single_result, single_scores)])
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch-GPU brute-force FlatIP retriever (faiss-gpu-free).")
    parser.add_argument("--index_path", type=str, required=True, help="Path to the faiss IndexFlatIP file.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus jsonl.")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retriever_name", type=str, default="e5")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda (single visible card) or e.g. cuda:0.")
    parser.add_argument("--db_fp32", action="store_true",
                        help="Store the DB matrix in fp32 (~64 GB) for exact parity with faiss; "
                             "default is fp16 (~32 GB), which is plenty for top-k cosine ranking.")
    parser.add_argument("--gpu_query_batch", type=int, default=256,
                        help="Queries per GPU matmul; bounds the (n x ntotal) similarity buffer.")
    args = parser.parse_args()

    config_topk = args.topk
    retriever = TorchFlatIPRetriever(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retriever_name=args.retriever_name,
        retriever_model=args.retriever_model,
        topk=args.topk,
        device=args.device,
        db_fp16=not args.db_fp32,
        gpu_query_batch=args.gpu_query_batch,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)
