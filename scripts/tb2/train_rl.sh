#!/usr/bin/env bash
# Stage C (TB2.1): GRPO editor training. The advisor patches the winner harness (kira_auto_test); the
# frozen Sonnet-4.5 student runs the patched harness via harbor; reward = binary task success. Needs the
# SkyRL venv + harbor engine (TB2_ENGINE_DIR or HARBOR_BIN) + Docker/tmux + Vertex creds. See docs.
#
# Build the dataset first:
#   python -m turbo_harness.rl.build_tb2_rl_dataset \
#     --split turbo_harness/terminal_bench/data/train_tasks.json --output data/rl/tb2_train.parquet
set -x
set -u
cd "$(dirname "$0")/../.." || exit 1
REPO="$(pwd)"
: "${VERTEXAI_PROJECT:?set VERTEXAI_PROJECT (Sonnet student via Vertex)}"; export VERTEXAI_PROJECT
: "${VERTEXAI_LOCATION:=global}"; export VERTEXAI_LOCATION
if [ -z "${TB2_ENGINE_DIR:-}" ] && [ -z "${HARBOR_BIN:-}" ]; then
  echo "set TB2_ENGINE_DIR or HARBOR_BIN (see docs/DEPENDENCIES.md)"; exit 1
fi
export TB2_ENGINE_DIR="${TB2_ENGINE_DIR:-}"

export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="${REPO}/SkyRL:${REPO}:${PYTHONPATH:-}"
: "${HF_HOME:=$HOME/.cache/huggingface}"; export HF_HOME
: "${CUDA_HOME:=/usr/local/cuda-13.3}"; export CUDA_HOME; export PATH="${CUDA_HOME}/bin:${PATH}"
: "${TILELANG_CACHE_DIR:=/tmp/tilelang_cache}"; export TILELANG_CACHE_DIR; mkdir -p "$TILELANG_CACHE_DIR"

# Reward env: frozen Sonnet student + base winner-harness dir + abstain penalty (0.05, reported).
export TB2_REWARD_STUDENT="${TB2_REWARD_STUDENT:-vertex_ai/claude-sonnet-4-5}"
export TB2_HARNESS="${TB2_HARNESS:-$REPO/turbo_harness/terminal_bench}"
export TB2_NO_PATCH_PENALTY="${TB2_NO_PATCH_PENALTY:-0.05}"
export TB2_RL_JOBS_DIR="${TB2_RL_JOBS_DIR:-/tmp/tb2rl_jobs}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-tb2}"

NUM_GPUS="${NUM_GPUS:-8}"
RUN_NAME="${RUN_NAME:-tb2_advisor_grpo_qwen35_autotest}"
DATA_DIR="${DATA_DIR:-${REPO}/data/rl}"
CKPT_PATH="${CKPT_PATH:-${REPO}/checkpoints/${RUN_NAME}}"
EXPORT_PATH="${EXPORT_PATH:-${REPO}/exports/${RUN_NAME}}"

cd "$REPO/SkyRL" || exit 1
"${VENV_PY:-.venv/bin/python}" -m turbo_harness.rl.main \
    data.train_data="['${DATA_DIR}/tb2_train.parquet']" \
    data.val_data="['${DATA_DIR}/tb2_train.parquet']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.model.path="Qwen/Qwen3.5-9B" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp \
    trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
    trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
    generator.inference_engine.num_engines="$NUM_GPUS" \
    generator.inference_engine.tensor_parallel_size=1 \
    trainer.epochs=5 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=16 \
    trainer.policy_mini_batch_size=2 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length=32768 \
    generator.max_input_length=32768 \
    generator.sampling_params.max_generate_length=8192 \
    generator.sampling_params.temperature=0.8 \
    generator.sampling_params.top_p=0.999 \
    generator.eval_sampling_params.temperature=0.6 \
    generator.eval_sampling_params.top_p=0.95 \
    generator.eval_sampling_params.top_k=20 \
    generator.eval_sampling_params.max_generate_length=8192 \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.kl_loss_coef=0.001 \
    trainer.algorithm.grpo_norm_by_std=true \
    generator.batched=false \
    environment.env_class=tb2_harness_patch \
    generator.n_samples_per_prompt=16 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.70 \
    generator.zero_reward_on_non_stop=true \
    trainer.remove_microbatch_padding=false \
    trainer.logger="wandb" \
    trainer.project_name="turbo-harness" \
    trainer.run_name="$RUN_NAME" \
    trainer.ckpt_interval=5 \
    trainer.max_ckpts_to_keep=10 \
    trainer.eval_interval=999 \
    generator.eval_n_samples_per_prompt=1 \
    trainer.resume_mode=null \
    trainer.ckpt_path="$CKPT_PATH" \
    trainer.hf_save_interval=5 \
    trainer.export_path="$EXPORT_PATH" \
    "$@"
