# Live demo

`run_demo.py` runs the **real** Turbo Harness pipeline on a single SWE-smith issue and prints each
step — the editor emits a per-instance patch, it is applied to the global harness `H*`, and the frozen
student runs the patched harness in Docker — formatted for screen-recording into a GIF. Nothing is
staged: it calls the same `generate_patch` / `apply_patch` / `executor.run_instance` the evaluation uses.

## Quick start (one command)

`demo/run.sh` starts the eval server + serves the editor, runs the demo, then cleans up. Configure it
with env vars (all documented in the script header):

```bash
# serve the trained editor from a checkpoint (set GPU to a free index):
EDITOR_CKPT=/path/to/<run>/global_step_N/policy GPU=0 bash demo/run.sh

# or use a frontier editor via API (no GPU / no vLLM):
ADVISOR_MODEL=vertex_ai/claude-opus-4-6 bash demo/run.sh
```

Capture a run and render the README image:

```bash
OUT=/tmp/run.txt SPEED=1.2 EDITOR_CKPT=... bash demo/run.sh   # capture a clean run
bash demo/render.sh /tmp/run.txt                              # -> assets/demo.{svg,png}
```

Handy overrides: `STUDENT`, `INSTANCE`, `WITH_DEFAULT=1`, `REPO_FLAG=verified`, `HARNESS_DIR`,
`PLAYBOOK`, `KEEP_SERVERS=1` (reuse servers across runs). Details below.

## Prerequisites (same as `scripts/swe_smith/eval_turbo.sh`)

- rootless Docker available — the frozen student runs in a container, and the single instance is
  **scored in-process** (local Docker via `swesmith`), so no eval server is needed
- an **editor**, either:
  - the trained Qwen editor served with **vLLM**, then pass `--advisor-url http://127.0.0.1:8120/v1`
    (vLLM ships with the SkyRL venv — `uv sync --extra fsdp` — or `pip install vllm` standalone; here
    you only use vLLM to *serve* the editor, you are not running training):
    ```bash
    python -m vllm.entrypoints.openai.api_server \
      --model /path/to/editor/global_step_60/policy --served-model-name advisor \
      --port 8120 --dtype bfloat16 --max-model-len 32768
    ```
  - or a **frontier editor via API** (no serving / no GPU): omit `--advisor-url` and pass e.g.
    `--advisor-model vertex_ai/claude-opus-4-6`.
- Vertex credentials for the student (`VERTEXAI_PROJECT` / `VERTEXAI_LOCATION`)

## Run

```bash
# trained Qwen editor served on :8120 (vLLM):
python demo/run_demo.py --advisor-url http://127.0.0.1:8120/v1
# or a frontier editor via API (no vLLM / no GPU serving):
python demo/run_demo.py --advisor-model vertex_ai/claude-opus-4-6
# options:  --instance <id>   --with-default (Default vs Turbo contrast)   --speed 1.2
```

## Make a GIF for the README

**From a captured run** (no re-run; this is how `assets/demo.gif` was made) — capture once,
then render a top-down screencast of that real run:

```bash
OUT=/tmp/run.txt bash demo/run.sh          # capture one real run's output
pip install rich cairosvg pillow
python demo/make_gif.py /tmp/run.txt       # -> assets/demo.gif
```

**Live terminal recording** (authentic asciinema capture) — alternative:

```bash
pip install asciinema                      # also install agg: https://github.com/asciinema/agg
ADVISOR_URL=http://127.0.0.1:8120/v1 bash demo/record.sh   # writes assets/demo.gif
```

Embed it in the repo `README.md`:

```markdown
![Turbo Harness demo](assets/demo.gif)
```
