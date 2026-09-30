# CONCEPT.md Generation Template

Read the meta-harness evolution data and write a CONCEPT.md document following this structure.

## Data to read

1. `evolution_summary.jsonl` — each line is one iteration with: iteration, agent name, avg_rate, delta_vs_frontier, hypothesis, changes
2. `trajectories/` — per-agent subdirectories with JSONL files containing agent messages, resolved, status, steps
3. `harness.py` and `SUMMARY.md` for each iteration's artifact directory

## Required sections

### 1. Evolution Scorecard
Table showing all iterations: iter number, candidate name, train rate, delta vs frontier, whether it became frontier.

### 2. What Works
For each iteration that improved the frontier: what was changed, why it worked, per-issue evidence from the data. Cite specific rates and deltas.

### 3. What Hurts
For each iteration that regressed significantly: what was changed, why it failed, per-issue evidence. Cite specific rates and deltas.

### 4. Agent Behavior Patterns
Analyze the raw trajectories to identify recurring failure/success patterns in the agent's behavior. Cite specific commands and steps from trajectory data.

### 5. Per-Issue Best Harness Distribution
From per_issue data in evolution_summary.jsonl, compute which harness is best for each issue. Show the distribution — this motivates per-instance adaptation.

### 6. Guidelines and Key Takeaways
Based on ALL the above analysis, derive actionable guidelines for a per-instance advisor. What should the advisor do and avoid? Every guideline must be grounded in specific data from the evolution.

## Rules

- Every claim must cite a specific iteration, rate, or delta from the data
- Do not inject assumptions beyond what the data shows
- Be specific — name iterations, rates, issue indices
- Target 150-300 lines total
