#!/usr/bin/env bash
# Stage C (ALFWorld): GRPO editor training. The FROZEN Qwen student must be served SEPARATELY on
# other GPUs (scripts/alfworld/serve_student.sh, tool-calling enabled); this trains on the rest.
# Reward = ALFWorld outcome (won in {0,1}) read from the trusted env. See docs/DEPENDENCIES.md
# (SkyRL venv, CUDA 13, ALFWORLD_DATA) and docs/REPRODUCE.md.
#
# Build the dataset first (train + a held-out val slice of harness_r1_train; NEVER use harness_r1_eval
# as val — that is the held-out TEST split):
#   python -m turbo_harness.rl.build_alfworld_rl_dataset --split harness_r1_train \
#     --harness-dir artifacts/alfworld/general \
#     --playbook   artifacts/alfworld/playbook.json \
#     --output     data/rl/alfworld_train.parquet
set -x
set -u
cd "$(dirname "$0")/../.." || exit 1
REPO="$(pwd)"
: "${ALFWORLD_DATA:?set ALFWORLD_DATA (see docs/DEPENDENCIES.md)}"; export ALFWORLD_DATA
: "${ALFWORLD_WORKER_PY:=$(command -v python)}"; export ALFWORLD_WORKER_PY

export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="${REPO}/SkyRL:${REPO}:${PYTHONPATH:-}"
: "${HF_HOME:=$HOME/.cache/huggingface}"; export HF_HOME
# Trainer GPUs (the frozen student holds the others, e.g. 4-7); node toolchain (CUDA 13 / CCCL) as in
# scripts/swe_smith/train_rl.sh + docs/DEPENDENCIES.md:
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"; export CUDA_VISIBLE_DEVICES
: "${CUDA_HOME:=/usr/local/cuda-13.3}"; export CUDA_HOME; export PATH="${CUDA_HOME}/bin:${PATH}"
: "${TILELANG_CACHE_DIR:=/tmp/tilelang_cache}"; export TILELANG_CACHE_DIR; mkdir -p "$TILELANG_CACHE_DIR"

# Reward env: base harness the advisor patches + frozen student name + abstain penalty (0.05, reported).
export ALF_HARNESS_DIR="${ALF_HARNESS_DIR:-$REPO/artifacts/alfworld/general}"
export ALF_REWARD_STUDENT="${ALF_REWARD_STUDENT:-Qwen3.5-9B}"
export ALF_NO_PATCH_PENALTY="${ALF_NO_PATCH_PENALTY:-0.05}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-alfworld}"

NUM_GPUS="${NUM_GPUS:-4}"
RUN_NAME="${RUN_NAME:-alfworld_advisor_grpo_qwen35}"
DATA_DIR="${DATA_DIR:-${REPO}/data/rl}"
CKPT_PATH="${CKPT_PATH:-${REPO}/checkpoints/${RUN_NAME}}"
EXPORT_PATH="${EXPORT_PATH:-${REPO}/exports/${RUN_NAME}}"

cd "$REPO/SkyRL" || exit 1
"${VENV_PY:-.venv/bin/python}" -m turbo_harness.rl.main \
    data.train_data="['${DATA_DIR}/alfworld_train.parquet']" \
    data.val_data="['${DATA_DIR}/alfworld_val.parquet']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.model.path="Qwen/Qwen3.5-9B" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp \
    trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
    trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
    generator.inference_engine.num_engines="$NUM_GPUS" \
    generator.inference_engine.tensor_parallel_size=1 \
    trainer.epochs=3 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=8 \
    trainer.policy_mini_batch_size=2 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length=16384 \
    generator.max_input_length=16384 \
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
    environment.env_class=alfworld_harness_patch \
    generator.n_samples_per_prompt=8 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization=0.70 \
    generator.zero_reward_on_non_stop=true \
    trainer.remove_microbatch_padding=false \
    trainer.logger="wandb" \
    trainer.project_name="turbo-harness" \
    trainer.run_name="$RUN_NAME" \
    trainer.ckpt_interval=10 \
    trainer.max_ckpts_to_keep=2 \
    trainer.eval_interval=15 \
    generator.eval_n_samples_per_prompt=1 \
    trainer.resume_mode=null \
    trainer.ckpt_path="$CKPT_PATH" \
    trainer.hf_save_interval=10 \
    trainer.export_path="$EXPORT_PATH" \
    "$@"
