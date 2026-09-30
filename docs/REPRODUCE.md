# Reproducing Turbo Harness

Each benchmark follows the same four-stage recipe:

```
Stage A: Meta-Harness search   → global harness H*        (frontier model; expensive)
Stage B: playbook          → playbook.json            (frontier model)
Stage C: RL train the editor   → editor checkpoint        (SkyRL / GRPO, GPUs)
Stage D: Evaluate              → Default / Meta / Turbo pass rates
```

We ship the **Stage-A/B outputs** (`H*` in `artifacts/`, `playbook.json`) and the
**data splits** (`data/`), so you can run **Stage C training** and **Stage D
evaluation** without repeating the expensive frontier search. Stage A/B scripts
are included for completeness (to regenerate those inputs).

## Execution budget (coding benchmarks)

SWE-smith-MR and SWE-bench Verified are run with **mini-swe-agent at a 40-step /
\$3-per-issue budget**, and we report the controlled **Default → Meta-Harness →
Turbo** deltas at this fixed budget.

## Per-benchmark instructions

<!-- Filled in as each module lands. -->

### SWE-smith-MR

**Prereqs:** dev venv installed (`docs/DEPENDENCIES.md`), a working (rootless) Docker, `.env`
filled in, and the servers running:

```bash
bash scripts/start_servers.sh     # eval_server:5152 (+ agent_server:8081, needed for Default)
```

Shipped inputs (no regeneration needed): the global harness `H*` and the playbook are in
`artifacts/swe_smith/{haiku,gemini37}/{general/,playbook.json}`; the 50/50 splits are in
`data/swe_smith/{train,test}_multi_repo.json`. Pick the executor with `STUDENT=haiku` or
`STUDENT=gemini37`.

**Default / Meta / Turbo:**

```bash
STUDENT=haiku bash scripts/swe_smith/eval_default.sh          # Default (mini-swe-agent, no opt)
STUDENT=haiku bash scripts/swe_smith/eval_meta.sh             # Meta-Harness (global H*)
STUDENT=haiku ADVISOR_CKPT=/path/to/editor/global_step_60/policy \
  bash scripts/swe_smith/eval_turbo.sh                        # Turbo (RL editor patches H*)
```

Turbo needs the trained editor checkpoint served on vLLM. Results go to
`results/swe_smith_<student>_{meta,turbo}.json` (Default prints to stdout). Expected pass rates
(paper): Haiku **42.0 / 50.7 / 64.0**, Gemini **44.0 / 70.7 / 88.0** (Default / Meta / Turbo).

**Optional — regenerate the shipped inputs** (needs frontier API budget):

```bash
STUDENT=haiku bash scripts/swe_smith/evolve.sh                            # Stage A: search -> H*
STUDENT=haiku RUN=swe_smith_haiku bash scripts/swe_smith/build_playbook.sh  # Stage B: playbook
```

**Editor training (Stage C, GRPO).** Trains the Qwen3.5-9B harness editor (needs the SkyRL
training venv + GPUs; see `docs/DEPENDENCIES.md`). With `eval_server:5152` up:

```bash
python -m turbo_harness.rl.build_rl_dataset \
  --data-file data/swe_smith/train_multi_repo.json \
  --harness-dir artifacts/swe_smith/haiku/general \
  --playbook   artifacts/swe_smith/haiku/playbook.json \
  --output     data/rl/train.parquet
STUDENT=haiku bash scripts/swe_smith/train_rl.sh   # GRPO; ckpts -> checkpoints/, exports -> exports/
```

Then run Turbo with an exported checkpoint:
`ADVISOR_CKPT=exports/<run>/global_step_60/policy bash scripts/swe_smith/eval_turbo.sh`.
The paper reports the final step-60 checkpoint (no test-set selection).

### SWE-bench Verified

Reuses the SWE-smith code with `--repo verified`; **scoring is in-process** (swebench), so **no
servers are needed** (a working Docker with the Verified task images is required). Shipped inputs:
`artifacts/verified/{haiku,gemini37}/{general/,playbook.json}` and the 251/99/150 splits
`data/swe_smith/{train,val,test}_verified.json`.

```bash
STUDENT=haiku bash scripts/verified/eval_default.sh    # Default (mini-swe-agent, no opt)
STUDENT=haiku bash scripts/verified/eval_meta.sh       # Meta-Harness (global H*)
STUDENT=haiku ADVISOR_CKPT=/path/to/editor/global_step_10/policy \
  bash scripts/verified/eval_turbo.sh                  # Turbo (RL editor patches H*)
```

Expected pass rates (paper, @40-step budget): Haiku **31.6 / 56.7 / 59.3**, Gemini
**24.4 / 38.4 / 54.4** (Default / Meta / Turbo). Checkpoints are validation-selected on the 99-issue
val split (Gemini step_20; Haiku step_10 — the entropy-collapsed step_60 is excluded).

**Editor training (Stage C).** Build the Verified RL dataset, then train (`BENCH=verified`):

```bash
python -m turbo_harness.rl.build_rl_dataset \
  --data-file data/swe_smith/train_verified.json \
  --harness-dir artifacts/verified/haiku/general \
  --playbook   artifacts/verified/haiku/playbook.json \
  --output     data/rl_verified/train.parquet
STUDENT=haiku RUN_NAME=harness_advisor_grpo_qwen35_verified DATA_DIR=data/rl_verified \
  bash scripts/verified/train_rl.sh
```

**Optional — regenerate inputs:** `scripts/verified/{evolve,build_playbook}.sh` (needs frontier API +
the Verified Docker images).

### ALFWorld

Frozen **Qwen3.5-9B** student over the ALFWorld env. Needs the `alfworld` package + its **game-data
download** (`ALFWORLD_DATA`) and a vLLM-served student (see `docs/DEPENDENCIES.md`). Shipped inputs:
`artifacts/alfworld/{general/,playbook.json}` and the game-list splits
`data/alfworld/harness_r1_{train,val,eval}.json` (train / val=`valid_seen` / test=`harness_r1_eval`, 150).

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/alfworld/serve_student.sh   # frozen Qwen3.5-9B on :8110
export ALFWORLD_DATA=/path/to/alfworld/data                           # your game-data download
cp data/alfworld/harness_r1_*.json "$ALFWORLD_DATA"/                  # split lists sit beside the games
bash scripts/alfworld/eval_default.sh                                 # Default
bash scripts/alfworld/eval_meta.sh                                    # Meta-Harness (global H*)
ADVISOR_CKPT=/path/to/editor/global_step_60/policy bash scripts/alfworld/eval_turbo.sh   # Turbo
```

Expected success % (paper): **40.7 / 60.7 / 70.7** (Default / Meta / Turbo). Turbo's checkpoint is
val-selected on the `valid_seen` split (`harness_r1_val`, step 60).

**Editor training (Stage C).** Build the dataset, then GRPO-train (student served separately on its
own GPUs; the abstention penalty defaults to 0.05):

```bash
python -m turbo_harness.rl.build_alfworld_rl_dataset --split harness_r1_train \
  --harness-dir artifacts/alfworld/general --playbook artifacts/alfworld/playbook.json \
  --output data/rl/alfworld_train.parquet
bash scripts/alfworld/train_rl.sh
```

### Terminal-Bench-2.1

Frozen **Claude Sonnet 4.5** student driving the **harbor** terminal engine. Needs harbor/terminal-bench
(the `.[tb2]` extra) + Docker + tmux + Vertex creds (see `docs/DEPENDENCIES.md`). The harnesses are Python
agents in `turbo_harness/terminal_bench/agents/` (KIRA, Terminus-2, and the evolution winner
`kira_auto_test`); the playbook is `artifacts/tb2/playbook.json`; the 45/44 split is
`turbo_harness/terminal_bench/data/{train,test}_tasks.json`.

```bash
export VERTEXAI_PROJECT=your-gcp-project
export TB2_ENGINE_DIR=/path/to/terminal_bench_2      # or: export HARBOR_BIN=$(which harbor)
bash scripts/tb2/eval_default.sh                                              # KIRA (Terminus-Kira)
AGENT=agents.baseline_terminus2:AgentHarness bash scripts/tb2/eval_default.sh # Terminus-2
bash scripts/tb2/eval_meta.sh                                                 # Meta (kira_auto_test)
ADVISOR_CKPT=/path/to/editor/global_step_5/policy bash scripts/tb2/eval_turbo.sh  # Turbo
```

Expected pass % (paper, n=5, held-out TEST-44): Turbo **55.5** > KIRA 52.3 > Terminus-2 50.9 ≈ Meta 50.5.
TB2.1 is noise-limited (one binary temp=1 rollout); the paper leads with the efficiency sweep. Reported
Turbo checkpoint = **step 5**.

**Editor training (Stage C).** Build the dataset, then GRPO-train:

```bash
python -m turbo_harness.rl.build_tb2_rl_dataset \
  --split turbo_harness/terminal_bench/data/train_tasks.json --output data/rl/tb2_train.parquet
bash scripts/tb2/train_rl.sh
```
