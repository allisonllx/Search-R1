#!/bin/bash
# Run validation-only evaluation on a trained actor checkpoint.
# Logs metrics to both console and WandB.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && { set -a; . "$SCRIPT_DIR/.env"; set +a; }

: "${BASE_MODEL:?BASE_MODEL not set — copy .env.example to .env and fill in your model path}"

export MAX_TURN=8
export train_file_name=10turn_2to10_steps_second_round_feedback_min_2_steps_20k
MODEL_TAG=$(basename "$BASE_MODEL")
export EXPERIMENT_NAME=${train_file_name}-${MODEL_TAG}-llm-judge-max-${MAX_TURN}-turn
CHECKPOINT_STEP=${CHECKPOINT_STEP:-550}
ACTOR_CHECKPOINT="${SCRIPT_DIR}/verl_checkpoints/${EXPERIMENT_NAME}/actor/global_step_${CHECKPOINT_STEP}"

if [ ! -d "$ACTOR_CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $ACTOR_CHECKPOINT" >&2
    exit 1
fi

# Optional: set WANDB_API_KEY in .env or export before running.
if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_BIN="${WANDB_BIN:-/root/dataDisk/home/allison/miniconda3/envs/searchr1/bin/wandb}"
    if ! "$WANDB_BIN" login --relogin "$WANDB_API_KEY"; then
        echo "[evaluate.sh] WARNING: WANDB_API_KEY login failed; falling back to ~/.netrc credentials" >&2
    fi
fi

export RETRIEVER_PORT=${RETRIEVER_PORT:-8002}
export RETRIEVER_HOST=${RETRIEVER_HOST:-127.0.0.1}
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WAND_PROJECT='Search-R1-synthetic-data'
EVAL_RUN_NAME="${EXPERIMENT_NAME}-eval-step${CHECKPOINT_STEP}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Re-use run.sh GPU selection when FORCE_GPUS is not set.
if [ -z "${FORCE_GPUS:-}" ]; then
    GPU_LADDER=${GPU_LADDER:-"8:90000:0.3 4:90000:0.3 2:90000:0.3"}
    GPU_FREE_SORTED=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | sort -t',' -k2 -rn)
    for tier in $GPU_LADDER; do
        cnt=${tier%%:*}; minf=$(echo "$tier" | cut -d: -f2); util=${tier##*:}
        ELIGIBLE=$(echo "$GPU_FREE_SORTED" | awk -F',' -v m="$minf" '$2+0>=m{print $1}')
        AVAIL=$(printf '%s\n' "$ELIGIBLE" | grep -c .)
        if [ "$AVAIL" -ge "$cnt" ]; then
            export CUDA_VISIBLE_DEVICES=$(printf '%s\n' "$ELIGIBLE" | head -n "$cnt" | sort -n | paste -sd, -)
            N_GPUS=$cnt
            GPU_MEM_UTIL=${GMU:-$util}
            echo "[evaluate.sh] using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ($N_GPUS GPU(s)), gpu_mem_util=$GPU_MEM_UTIL"
            break
        fi
    done
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "ERROR: no GPUs available for evaluation" >&2
        exit 1
    fi
else
    export CUDA_VISIBLE_DEVICES=$FORCE_GPUS
    N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
    GPU_MEM_UTIL=${GMU:-0.3}
    echo "[evaluate.sh] FORCE_GPUS=$CUDA_VISIBLE_DEVICES ($N_GPUS GPU(s))"
fi

PPO_MICRO=$((4 * N_GPUS))
LOGPROB_MICRO=$((16 * N_GPUS))
CRITIC_MICRO=$((1 * N_GPUS))

echo "[evaluate.sh] actor checkpoint: $ACTOR_CHECKPOINT"
echo "[evaluate.sh] wandb run: $WAND_PROJECT / $EVAL_RUN_NAME"

ray stop --force 2>/dev/null

PYTHON="${PYTHON:-/root/dataDisk/home/allison/miniconda3/envs/searchr1/bin/python3}"
PYTHONUNBUFFERED=1 "$PYTHON" -m verl.trainer.main_ppo \
    data.train_files=data/gemini_agentic_question_gen/${train_file_name}_train.parquet \
    data.val_files=data/gemini_agentic_question_gen/${train_file_name}_test_combined.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.max_start_length=2048 \
    data.max_obs_length=2048 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path="$ACTOR_CHECKPOINT" \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.free_cache_engine=False \
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
    trainer.logger="['console','wandb']" \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EVAL_RUN_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=$MAX_TURN \
    +reward_model.reward_manager=llm_judge \
    +reward_model.judge_model="${VLLM_JUDGE_MODEL:-qwen-judge}" \
    +reward_model.judge_base_url="${VLLM_JUDGE_BASE_URL:-http://127.0.0.1:8001/v1}" \
    +reward_model.judge_temperature=0 \
    +reward_model.judge_timeout="${VLLM_JUDGE_TIMEOUT:-60}" \
    +reward_model.judge_max_retries="${VLLM_JUDGE_MAX_RETRIES:-3}" \
    +reward_model.judge_cache_dir="${VLLM_JUDGE_CACHE_DIR:-outputs/llm_judge_cache}" \
    retriever.url="http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/retrieve" \
    retriever.topk=3 \
    2>&1 | tee "${EVAL_RUN_NAME}_${TIMESTAMP}.log"
