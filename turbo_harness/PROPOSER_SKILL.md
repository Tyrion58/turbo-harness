# SWE Harness Evolution (Meta-Harness)

Run ONE iteration of harness scaffold evolution for SWE-smith bug-fix tasks.

**You do NOT run benchmarks.** You analyze results + failed agent trajectories, propose harness variants, and implement them. The outer loop (`meta_harness.py`) handles benchmarking.

## CRITICAL CONSTRAINTS

- You MUST produce 1 new harness variant every iteration.
- Do NOT write "the baseline is optimal" or "stop iterating", or abort early.

### Anti-overfitting rules

- **No task-specific hints.** Do not hardcode knowledge about specific issues. Harnesses must be general-purpose.
- **Never mention issue IDs** in harness code, prompts, or comments. No references like "if issue contains 'AttributeError'" or "for rewrite tasks." If your improvement only helps one issue, it's too specific.
- **General guidance is OK.** Rules like "always run failing tests before editing" are fine — they apply broadly. The test: would this advice be useful to a developer working on MANY unfamiliar issues in this repo?
- If in doubt, make it more general.

## CONTEXT

You are evolving a **harness** (agent scaffold) for solving SWE-smith bug-fix tasks. The harness wraps a **fixed, frozen student model** (given in the task prompt) running inside `minisweagent` (mini-swe-agent).

### What a "harness" IS

The harness is the **agent loop / scaffold** around the fixed model: how it manages context and memory, what it stores/retrieves/shows each step, its control flow, tools, and workflow. It is NOT the wording of a system prompt. Rewording instructions = prompt optimization = NOT your job. You optimize the **scaffold**.

### Search space

You may **subclass** `minisweagent.agents.default.DefaultAgent` and override any of its methods:

- `run(task)` — main episode loop (structure, memory across steps, when to stop)
- `step()` / `query()` — how the message/context is constructed for each model call
- `get_observation(response)` — how tool output is summarized/injected back
- `parse_action(response)` / `execute_action(action)` — action parsing / execution behavior
- `has_finished(output)` — termination / acceptance logic
- `add_message(...)` — memory policy (what is kept in the running context)
- the config templates (`system_template`, `instance_template`, ...) — allowed, but templates alone are prompt optimization; the value is in the **loop / context / memory / control** changes above.

Good harness changes (examples): maintain a structured ledger of hypotheses tried + test results across steps; re-inject the failing test + target function before each edit; a reproduce→localize→fix→test→verify workflow with an acceptance gate; revert-on-regression fallback; truncate/summarize stale tool output to control context; retry/repair on format errors.

### Model capability constraint

The harness is executed by a **fixed student model** (name given in task prompt). Design for its capability level: a weak model on a tight step budget needs a **lean, efficient** harness — NOT a complex, process-heavy one it cannot execute well.

## CANDIDATE DESIGN

Each candidate harness is a **directory** containing `harness.py`, `RULES_MEMORY.md`, `SUMMARY.md`.

You should read prior iteration harnesses in `artifacts/<repo>/iter*/` as starting points. **Copy and modify** rather than writing from scratch.

### Output artifact

1. **`harness.py`** — must expose:
   ```python
   def build_agent(model, env, step_limit: int, cost_limit: float):
       """Return a DefaultAgent (or subclass) configured as the harness."""
   ```
   Self-contained (import from `minisweagent...`; do not import other candidate artifacts).
2. **`RULES_MEMORY.md`** — repo-level knowledge/rules. General, not task-specific.
3. **`SUMMARY.md`** — design rationale: what changed vs baseline/prior iteration and why.

### What you can and cannot modify

- **CAN**: write any Python code in your `harness.py` — subclass `DefaultAgent`, override any method.
- **CAN**: change config templates, add context management, modify the agent loop.
- **CANNOT**: modify the baseline `DefaultAgent` source, `meta_harness.py`, or any infrastructure code.
- **CANNOT**: change `step_limit` or `cost_limit` (must respect the budgets passed in).

### Design principles

- Your primary goal is to improve the agent's pass rate on the evaluation issues.
- **One mechanism per candidate.** Each candidate tests exactly one hypothesis. If you're tempted to add "and also..." — that's a second candidate.
- **Mechanism-first.** Identify a specific failure mode or hypothesis from trajectories, then design changes that target it. Never add changes speculatively.

Constraints: `harness.py` must import cleanly; respect `step_limit`/`cost_limit`; no external network.

## WORKFLOW

### Step 1: Analyze (1 subagent)

Launch ONE Agent subagent (subagent_type: "general-purpose"). It should:

1. Read state files:
   - `frontier.json` — current best harness per issue (path given in task prompt)
   - `evolution_summary.jsonl` — what's been tried, what worked/didn't (path given in task prompt)
2. **Deep-read failed AND successful agent trajectories.** Most important step.
   - Trajectory files are under `trajectories/` (path given in task prompt).
   - Each JSONL file contains the full step-by-step agent execution: thoughts, commands, outputs.
   - Focus on understanding **WHY** the agent fails: does it loop? make wrong edits? run out of steps? misunderstand the bug?
   - Also read successful trajectories to understand what patterns work.
3. Read prior harness implementations in `artifacts/<repo>/iter*/harness.py`
4. Read the baseline scaffold: `DefaultAgent` source + config yaml (paths given in task prompt)
5. Return:

```
STATE: <5-line summary: current scores, what's been tried, common failure modes observed>

HYPOTHESIS: "<falsifiable claim about what will improve scores>"
CANDIDATE: name=<snake_case>, changes="<specific changes targeting observed failure modes>", prediction="<expected pass rate improvement>"
```

### Step 2: Implement (1 subagent)

Launch **1 Agent subagent** (subagent_type: "general-purpose"). The prompt must include the candidate name, specific changes from Step 1, and the output directory.

The subagent should:

1. Read the baseline `DefaultAgent` source to understand the overridable methods.
2. Write `harness.py`, `RULES_MEMORY.md`, `SUMMARY.md` into the output directory.
3. **Smoke test**: validate import (`python -c "import importlib.util; spec = importlib.util.spec_from_file_location('h', '<path>/harness.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('OK:', hasattr(m, 'build_agent'))"`)
4. Return: file path, validation status

### Step 3: Write pending_eval.json

Write `pending_eval.json` to the path specified in the task prompt:

```json
{
  "iteration": <N>,
  "candidates": [
    {
      "name": "<name>",
      "harness_dir": "<absolute path to the candidate's artifact directory>",
      "hypothesis": "<falsifiable claim>",
      "changes": "<what was changed and why>"
    }
  ]
}
```

Output: `CANDIDATES: <name>`

## IMPORTANT NOTES

- **Mechanism-first.** Identify a specific failure mode from trajectories, then design changes that target it. Never add changes speculatively.
- **One mechanism per candidate.** Each candidate tests exactly one hypothesis. If you're tempted to add "and also..." — that's a second candidate.
- The harness must work with `minisweagent`'s `DefaultAgent` framework. The agent is constructed via `build_agent(model, env, step_limit, cost_limit)` and run via `agent.run(problem, **task_vars)`.
