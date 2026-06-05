#!/bin/bash
# make the dataset

# Load machine-specific paths (BASE_MODEL, RAY_TEMP_DIR, ...) from a gitignored
# .env so they never get committed. Copy .env.example to .env and fill it in.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && { set -a; . "$SCRIPT_DIR/.env"; set +a; }

# ---------------------------------------------------------------------------
# Dynamic GPU selection.
# verl divides every micro/mini batch size by the world size, so per-GPU memory
# is fixed by (global batch size) / (num GPUs). We therefore (1) pick the
# largest power-of-2 number of GPUs (8 -> 4 -> 2 -> 1) for which that many GPUs
# each have at least MIN_FREE_MIB free, and (2) scale the global micro-batches
# linearly with that count so the per-GPU micro batch stays constant.
# Override the auto-pick by exporting FORCE_GPUS, e.g.  FORCE_GPUS=4,7 ./run.sh
# ---------------------------------------------------------------------------

# Per rank this job needs: vLLM ~= gpu_memory_utilization(0.4) * 143771 ~= 57 GB,
# plus FSDP training (~25-30 GB with offload + grad checkpointing) ~= 95-110 GB.
# Floor is 120 GB: comfortably above a rank's real need AND above a GPU that is
# already hosting the retriever (~33 GB used -> ~110 GB free), so auto-pick never
# co-locates training onto the retriever's card. An empty H200 reports ~138 GB
# free, so genuinely-empty cards still clear this floor with ~18 GB to spare.
export MIN_FREE_MIB=${MIN_FREE_MIB:-120000}

if [ -n "${FORCE_GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$FORCE_GPUS
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    echo "[run.sh] FORCE_GPUS set -> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ($N_GPUS GPU(s))"
else
    ELIGIBLE_IDS=()
    while IFS=',' read -r id free; do
        id=$(echo "$id" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
        [ "$free" -ge "$MIN_FREE_MIB" ] && ELIGIBLE_IDS+=("$id")
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t',' -k2 -rn)

    AVAIL=${#ELIGIBLE_IDS[@]}
    if   [ "$AVAIL" -ge 8 ]; then N_GPUS=8
    elif [ "$AVAIL" -ge 4 ]; then N_GPUS=4
    elif [ "$AVAIL" -ge 2 ]; then N_GPUS=2
    elif [ "$AVAIL" -ge 1 ]; then N_GPUS=1
    else
        echo "ERROR: no GPU has >= ${MIN_FREE_MIB} MiB free. Current free memory:" >&2
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader >&2
        exit 1
    fi

    # Take the N most-free GPUs (list is sorted by free mem desc), tidy ascending.
    export CUDA_VISIBLE_DEVICES=$(printf '%s\n' "${ELIGIBLE_IDS[@]:0:$N_GPUS}" | sort -n | paste -sd, -)
    echo "[run.sh] auto-selected $N_GPUS/${AVAIL} eligible GPU(s) -> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# ---------------------------------------------------------------------------
# Re-check guard: close the time-of-check/time-of-use race.
# Selection above sampled free memory once; on a shared box another job (or the
# retriever) can grab a selected card during the ~30-60 s before vLLM allocates.
# Re-query free memory on the GPUs we are about to use and abort cleanly if any
# of them no longer has MIN_FREE_MIB free, so we fail fast instead of OOMing
# mid-run. Skipped when FORCE_GPUS is set (user took explicit responsibility).
if [ -z "${FORCE_GPUS:-}" ]; then
    declare -A FREE_NOW=()
    while IFS=',' read -r id free; do
        id=$(echo "$id" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
        FREE_NOW[$id]=$free
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

    for id in $(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' '); do
        free=${FREE_NOW[$id]:-0}
        if [ "$free" -lt "$MIN_FREE_MIB" ]; then
            echo "ERROR: GPU $id dropped to ${free} MiB free (< ${MIN_FREE_MIB}) between selection and launch." >&2
            echo "       Another job likely grabbed it. Re-run to re-select. Current free memory:" >&2
            nvidia-smi --query-gpu=index,memory.free --format=csv,noheader >&2
            exit 1
        fi
    done
    echo "[run.sh] re-check OK: all of [$CUDA_VISIBLE_DEVICES] still have >= ${MIN_FREE_MIB} MiB free"
fi

# Scale global micro batches with the GPU count so per-GPU stays constant.
# 8-GPU baseline per GPU: actor micro 4, log_prob 16, critic micro 1.
PPO_MICRO=$((4 * N_GPUS))       # actor  ppo_micro_batch_size       (8 GPUs -> 32)
LOGPROB_MICRO=$((16 * N_GPUS))  # rollout/ref log_prob_micro_batch  (8 GPUs -> 128)
CRITIC_MICRO=$((1 * N_GPUS))    # critic ppo_micro_batch_size       (8 GPUs -> 8)
echo "[run.sh] batches -> actor_micro=$PPO_MICRO log_prob_micro=$LOGPROB_MICRO critic_micro=$CRITIC_MICRO (mini=256, train=512 held constant)"

# export DATA_DIR='data/nq_hotpotqa_train'

WAND_PROJECT='Search-R1-synthetic-data'

# Retriever server configuration
export RETRIEVER_PORT=${RETRIEVER_PORT:-8002}
export RETRIEVER_HOST=127.0.0.1

# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'  # HF download hung at ~64MB/5.75GB; switched to local 7B
# BASE_MODEL comes from .env (gitignored). Copy .env.example -> .env and set it.
: "${BASE_MODEL:?BASE_MODEL not set — copy .env.example to .env and fill in your model path}"
export BASE_MODEL
export MAX_TURN=8
export train_file_name=10turn_2to10_steps_second_round_feedback_min_2_steps_20k
export EXPERIMENT_NAME=${train_file_name}-qwen2.5-7b-it-llm-judge-max-${MAX_TURN}-turn
export epoch=15
# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

# Keep Ray's temp/spill dir off the near-full root disk. Set RAY_TEMP_DIR in .env
# to a path on a disk with lots of free space; falls back to /tmp/ray if unset.
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray}"

# Clear any stale/dead Ray cluster left over from an interrupted run so ray.init() starts fresh
ray stop --force 2>/dev/null

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=data/gemini_agentic_question_gen/${train_file_name}_train.parquet \
    data.val_files=data/gemini_agentic_question_gen/${train_file_name}_test_combined.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.max_start_length=2048 \
    data.max_obs_length=1000 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$PPO_MICRO \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOGPROB_MICRO \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$LOGPROB_MICRO \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=$CRITIC_MICRO \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$epoch \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=$MAX_TURN \
    +reward_model.reward_manager=llm_judge \
    retriever.url="http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log