# Running Search-R1 with Docker

Two images, one network:

- **`retriever`** — e5 encoder (GPU) + FAISS search (CPU). Serves `POST /retrieve`.
- **`train`** — verl + vLLM + flash-attn. Runs PPO and calls the retriever over the
  compose bridge network at `http://retriever:$RETRIEVER_PORT/retrieve`.

Data (FAISS index, wiki corpus, base model, checkpoints, HF cache) is **host-mounted**,
never baked into the images — so the images stay small and checkpoints persist on the host.

## Prerequisites

- Docker + Docker Compose v2.
- **NVIDIA Container Toolkit** on the host (`nvidia-ctk`), so containers see the GPUs.
  Quick check: `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi`.
- The FAISS index + corpus and the base model already on disk (see the main README's
  "Quick start" for downloading them).

## Configure

```bash
cp .env.example .env
```
Fill in (these are mounted into the containers at the **same** path):

| var | meaning |
|---|---|
| `BASE_MODEL` | path to the actor/critic/rollout base model dir |
| `INDEX_DIR` | dir holding `e5_Flat.index` + the corpus |
| `RAY_TEMP_DIR` | Ray spill dir on a roomy disk |
| `RETRIEVER_PORT` | retriever port (default 8002) |
| `RETRIEVER_BACKEND` | `faiss_cpu` (default) / `torch_gpu` / `faiss_gpu` |
| `HF_HOME` | host dir for the HF cache (default `./.hf-cache`) |
| `FORCE_RETRIEVER_GPUS` | optional: pin the encoder to a physical GPU index |
| `WANDB_API_KEY` | optional: passed into the training container |

## Build & run

```bash
docker compose build                      # flash-attn build makes the train image slow (tens of min)

docker compose up -d retriever            # loads index+corpus (minutes); becomes "healthy" when ready
docker compose logs -f retriever          # watch until it is serving

docker compose up -d train                # waits for retriever to be healthy
docker compose exec train bash run.sh     # launch PPO training
```

Checkpoints/logs land in `./verl_checkpoints` and `*.log` on the host (the repo is mounted
at `/app`). Stop everything with `docker compose down`.

## GPU placement

Both containers see all GPUs (`count: all`). The existing scripts coordinate placement the
same way they do on bare metal:

- `run.sh` auto-selects training GPUs by free memory (its ladder/guards still run inside the
  container via `nvidia-smi`).
- the retriever picks an encoder GPU via `retrieval_launch_smart.sh`; pin it off the training
  cards with `FORCE_RETRIEVER_GPUS=<idx>` in `.env`.

To hard-partition instead, replace `count: all` with explicit `device_ids: ['...']` per
service in `docker-compose.yml`.

## Notes / gotchas

- **faiss-cpu on purpose.** faiss-gpu 1.7.x/1.8.0 has no working sm_90 kernel and aborts on
  H200; the retriever image uses faiss-cpu (encoder still on GPU). To use `torch_gpu` instead,
  set `RETRIEVER_BACKEND=torch_gpu` (needs ~46 GB on one card).
- **CUDA tag / arch.** The images target CUDA 12.8 and build flash-attn for `sm_90` (Hopper).
  For other GPUs, edit `TORCH_CUDA_ARCH_LIST` in `docker/Dockerfile.train` and, if needed, the
  `nvidia/cuda:12.8.1-*` base tags.
- **Version set.** torch is installed from the cu128 wheel index so its bundled NCCL (2.27.3)
  matches — this is what avoids both the `Replicate` import error and the
  `undefined symbol: ncclCommResume` NCCL mismatch.
- **pyserini/BM25** is excluded from the retriever image (needs a JVM). Uncomment it in
  `requirements_retriever_pip.txt` and add a JDK to `docker/Dockerfile.retriever` if you need it.
