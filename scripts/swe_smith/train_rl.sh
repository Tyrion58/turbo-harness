#!/usr/bin/env bash
# Stage C: GRPO training of the harness editor (Qwen3.5-9B) for SWE-smith-MR.
# The frozen student runs in-process (mini-swe-agent DockerEnvironment); patch scoring goes through
# eval_server:5152. Needs the SkyRL training venv (SkyRL/.venv), rootless Docker, and ~8x H100 GPUs.
# See docs/DEPENDENCIES.md for the training-venv + node toolchain (CUDA 13 / CCCL) setup.
#
# Prereqs:
#   1) Build the RL dataset (once):
#        python -m turbo_harness.rl.build_rl_dataset \
#          --data-file data/swe_smith/train_multi_repo.json \
#          --harness-dir artifacts/swe_smith/haiku/general \
#          --playbook   artifacts/swe_smith/haiku/playbook.json \
#          --output     data/rl/train.parquet
#   2) Start eval_server:  python -m turbo_harness.infra.eval_server --host 0.0.0.0 --port 5152
#   3) Run this script.
set -u
cd "$(dirname "$0")/../.." || exit 1
REPO="$(pwd)"
# shellcheck source=/dev/null
source scripts/env.sh

# Point the REWARD env at the SAME general harness the dataset/prompt bakes (must match
# build_rl_dataset --harness-dir), and propagate the frozen student model. See turbo_harness/rl/env.py.
export SWE_HARNESS_DIR="${SWE_HARNESS_DIR:-$REPO/$ART/general}"
export SWE_STUDENT_MODEL="${SWE_STUDENT_MODEL:-$STUDENT_MODEL}"

export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="${REPO}/SkyRL:${REPO}:${PYTHONPATH:-}"
: "${HF_HOME:=$HOME/.cache/huggingface}"; export HF_HOME

# --- Node-specific toolchain (EXAMPLES — adjust for your machine; see docs/DEPENDENCIES.md) ---
# CUDA 13.x is required for TileLang JIT of Qwen3.5 fla (gated-delta-rule) kernels in the policy
# backward; rootless-Docker users also export DOCKER_HOST / MSWEA_DOCKER_EXECUTABLE.
: "${CUDA_HOME:=/usr/local/cuda-13.3}"; export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
: "${TILELANG_CACHE_DIR:=/tmp/tilelang_cache}"; export TILELANG_CACHE_DIR; mkdir -p "$TILELANG_CACHE_DIR"

# --- wandb (online; auth via `wandb login`, never a hardcoded key) ---
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-swe}"

NUM_GPUS="${NUM_GPUS:-8}"
RUN_NAME="${RUN_NAME:-harness_advisor_grpo_qwen35}"
DATA_DIR="${DATA_DIR:-${REPO}/data/rl}"
CKPT_PATH="${CKPT_PATH:-${REPO}/checkpoints/${RUN_NAME}}"
EXPORT_PATH="${EXPORT_PATH:-${REPO}/exports/${RUN_NAME}}"

cd "$REPO/SkyRL" || exit 1
"${VENV_PY:-.venv/bin/python}" -m turbo_harness.rl.main \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/train.parquet']" \
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
    trainer.train_batch_size=4 \
    trainer.policy_mini_batch_size=2 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length=16384 \
    generator.sampling_params.max_generate_length=4096 \
    generator.sampling_params.temperature=0.8 \
    generator.sampling_params.top_p=0.999 \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.kl_loss_coef=0.001 \
    trainer.algorithm.grpo_norm_by_std=true \
    generator.batched=false \
    environment.env_class=harness_advisor \
    generator.n_samples_per_prompt=8 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.85 \
    generator.zero_reward_on_non_stop=true \
    trainer.remove_microbatch_padding=false \
    trainer.logger="wandb" \
    trainer.project_name="turbo-harness" \
    trainer.run_name="$RUN_NAME" \
    trainer.ckpt_interval=10 \
    trainer.max_ckpts_to_keep=2 \
    trainer.eval_interval=0 \
    trainer.resume_mode="${RESUME_MODE:-null}" \
    trainer.ckpt_path="$CKPT_PATH" \
    trainer.hf_save_interval=10 \
    trainer.export_path="$EXPORT_PATH" \
    "$@"
