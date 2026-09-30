# Dependencies & Environment Setup

Turbo Harness uses **two Python environments**, mirroring the original project:

1. **Dev / eval venv** (this `pyproject.toml`) — evaluation, the SWE agent/eval
   servers, dataset construction, Meta-Harness search, and the playbook pipeline.
2. **Training venv** (the vendored `SkyRL/`) — GRPO RL training of the harness
   editor, with its own heavy stack (PyTorch, vLLM, Ray, FSDP). Kept separate on
   purpose; see "Training venv" below.

## What each dependency is for

| Dependency | Needed for | How it ships |
|---|---|---|
| **SkyRL** | RL training of the editor (Stage C, all benchmarks) | **vendored** under `SkyRL/` (its own venv) |
| **mini-swe-agent** | coding-benchmark scaffold (SWE-smith, SWE-bench Verified) | pip (`dependencies`), used unmodified |
| **harbor + terminal-bench** | Terminal-Bench-2.1 runtime (train + eval) | pip extra: `pip install -e '.[tb2]'` |
| **vLLM** | RL-training rollouts (used inside SkyRL) **and** serving the trained editor for eval/demo | ships with the SkyRL venv (`uv sync --extra fsdp`); or `pip install vllm` standalone just to serve |
| Vertex AI (`anthropic[vertex]`, `google-cloud-aiplatform`, `litellm`) | frontier proposals + student model calls | pip (`dependencies`) |
| `swesmith`, `swebench==4.1.0` | scoring submitted code patches | pip (`dependencies`) |

## 1. Dev / eval venv

```bash
uv sync                                        # or: pip install -e .
source .venv/bin/activate
pip install -e turbo_harness/infra/mini-swe-agent   # vendored scaffold (+ its deps)
cp .env.example .env   # then edit: VERTEXAI_PROJECT, server URLs, storage roots
```

`mini-swe-agent` is **vendored** under `turbo_harness/infra/mini-swe-agent/` (the code
imports it from there); installing it editable pulls its runtime dependencies.
`swebench` is pinned to `4.1.0`: `swesmith` imports constants that were removed in
`swebench` 5.x.

## 2. Training venv (SkyRL, vendored)

SkyRL is vendored under `SkyRL/` (its source; the upstream `docs/` are omitted). It uses its own venv:

```bash
cd SkyRL
uv sync --extra fsdp
source .venv/bin/activate
uv pip install litellm tenacity google-cloud-aiplatform docker requests
# vllm-router is skipped by the fsdp extra; on glibc <2.35 install the PyPI wheel:
uv pip install "vllm-router==0.1.15"
cd ..
```

Requires NVIDIA GPUs (H100-class), CUDA 13.x (for TileLang JIT of the Qwen3.5 kernels), and a
working (rootless) Docker. On some nodes the TileLang JIT needs CCCL headers symlinked into the
venv's `nvidia/cu13/include`; see the header of `scripts/swe_smith/train_rl.sh`. RL training reads
`data/rl/train.parquet` (built via `turbo_harness.rl.build_rl_dataset`) and needs `pandas` +
`transformers` (both provided by the SkyRL `fsdp` extra). Per-benchmark steps: `docs/REPRODUCE.md`.

## 3. Terminal-Bench-2.1 (harbor engine)

```bash
uv pip install -e '.[tb2]'   # harbor + terminal-bench
```

TB2.1 drives the **harbor** engine with a frozen Claude Sonnet 4.5 student on Vertex, and needs Docker
and `tmux` on the host (bind-mounted into task containers via
`turbo_harness/terminal_bench/docker-compose-tmux.yaml`). Point the harness at harbor with **either**
`HARBOR_BIN=$(which harbor)` (from the extra above) **or** `TB2_ENGINE_DIR=/path/to/terminal_bench_2`
(a project with harbor in its `.venv`). The task set is pinned via
`turbo_harness/terminal_bench/data/registry_tb21.json`. See `docs/REPRODUCE.md`.

## 4. ALFWorld (agentic domain)

ALFWorld needs the `alfworld` package and its game-data download:

```bash
pip install alfworld            # pulls in textworld
alfworld-download               # downloads game data (json_2.1.1, PDDL, detectors)
export ALFWORLD_DATA=/path/to/alfworld/data
cp data/alfworld/harness_r1_*.json "$ALFWORLD_DATA"/   # split lists co-located with the games
```

The frozen **Qwen3.5-9B student** is served via vLLM with tool-calling
(`scripts/alfworld/serve_student.sh`). The ALFWorld env runs in a subprocess; if `alfworld` is not in
your active venv, point `ALFWORLD_WORKER_PY` at an interpreter that has it. The vendored
`turbo_harness/alfworld/configs/base_config.yaml` and `prompts/` are resolved automatically (from the
package directory), so no external repo is required.

## Environment variables

See `.env.example`. Everything the code reads is surfaced in
`turbo_harness/config.py`; nothing node-specific is hardcoded.
