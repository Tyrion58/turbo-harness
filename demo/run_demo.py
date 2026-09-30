"""Turbo Harness — live single-instance demo (record this to a GIF for the README).

Runs the REAL Turbo pipeline on ONE SWE-smith issue with the trained editor:

    instance  ->  editor reads H* + playbook  ->  emits a SEARCH/REPLACE patch
              ->  apply -> instance harness H_x  ->  frozen student runs H_x in Docker  ->  resolved?

This calls the same functions the evaluation uses (`generate_patch`, `apply_patch`,
`executor.run_instance`); nothing here is staged. It just drives ONE instance with
recording-friendly output.

Prereqs:
  - rootless Docker available (the student runs in a container; the single instance is scored
    in-process via swesmith + local Docker, so NO eval server is needed)
  - the trained editor served on vLLM, e.g. on :8120  (--advisor-url http://127.0.0.1:8120/v1)
  - Vertex creds for the student (VERTEXAI_PROJECT / VERTEXAI_LOCATION)

Example:
  python demo/run_demo.py --advisor-url http://127.0.0.1:8120/v1 --speed 1.0

Record to a GIF: see demo/record.sh.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import logging
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

# Keep the demo output clean for screen-recording: silence library banners/logs/warnings.
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "turbo_harness/infra/mini-swe-agent/src"))

# mini-swe-agent prints a banner at import; swallow import-time stdout/stderr so it stays off the demo.
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from turbo_harness.eval.patch_eval import load_instances  # noqa: E402
    from turbo_harness.patch_advisor import generate_patch, apply_patch  # noqa: E402
    from turbo_harness.proposer import validate_artifact  # noqa: E402
    from turbo_harness.playbook.memory_advisor import format_playbook_context  # noqa: E402
    from turbo_harness.playbook.schemas import Playbook  # noqa: E402
    from turbo_harness import executor as EX  # noqa: E402


def _c(code, t):
    return f"\033[{code}m{t}\033[0m"


BOLD = lambda t: _c("1", t)          # noqa: E731
DIM = lambda t: _c("2", t)           # noqa: E731
GRN = lambda t: _c("32", t)          # noqa: E731
RED = lambda t: _c("31", t)          # noqa: E731
YEL = lambda t: _c("33", t)          # noqa: E731
ORA = lambda t: _c("38;5;208", t)    # noqa: E731

_SLEEP = time.sleep
PAUSE = 1.0


def pause(s=1.0):
    if PAUSE:
        _SLEEP(s * PAUSE)


def hr(title):
    print("\n" + ORA("== " + title + " " + "=" * max(4, 64 - len(title))))


def _scored_cleanly(res):
    """True only if scoring actually produced a pass/fail (not an infra/timeout error)."""
    status = res.get("status") or ""
    return res.get("resolved") is not None and not (
        status.startswith(("SCORE_ERROR", "ERROR")) or "TIMEOUT" in status
    )


def result_tag(res):
    """Honest status: solved vs genuinely-not-solved vs could-not-score (infra/timeout)."""
    if not _scored_cleanly(res):
        return YEL("ran; scoring unavailable [!]")
    return GRN("RESOLVED [OK]") if res.get("resolved") else RED("not resolved [x]")


def main():
    ap = argparse.ArgumentParser(description="Turbo Harness single-instance live demo")
    ap.add_argument("--data-file", default=str(REPO / "data/swe_smith/test_multi_repo.json"))
    ap.add_argument("--harness-dir", default=str(REPO / "artifacts/swe_smith/haiku/general"))
    ap.add_argument("--playbook", default=str(REPO / "artifacts/swe_smith/haiku/playbook.json"))
    ap.add_argument("--advisor-url", default=None,
                    help="served-editor endpoint (vLLM), e.g. http://127.0.0.1:8120/v1; "
                         "omit to call a frontier editor via API (set --advisor-model accordingly)")
    ap.add_argument("--advisor-model", default="openai/advisor")
    ap.add_argument("--student", default="vertex_ai/claude-haiku-4-5")
    ap.add_argument("--instance", default=None, help="instance_id to demo (default: the first)")
    ap.add_argument("--repo", default="multi_repo")
    ap.add_argument("--with-default", action="store_true", help="also run the unpatched Default for contrast")
    ap.add_argument("--speed", type=float, default=1.0, help="pause multiplier (0 = no pauses)")
    ap.add_argument("--reasoning-lines", type=int, default=8,
                    help="lines of the editor's chain-of-thought to show (0 = hide)")
    args = ap.parse_args()

    global PAUSE
    PAUSE = args.speed
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    # Score this ONE instance in-process (local Docker). The eval_server only exists to serialize
    # scoring across many concurrent RL rollouts, so it is unneeded here; skipping the HTTP hop also
    # removes a transient connection-failure class. (The Verified path already scores in-process.)
    os.environ.pop("EVAL_SERVER_URL", None)

    print(ORA(BOLD("\n  TURBO HARNESS  --  live demo")))
    print(DIM("  instance-adaptive harness optimization: the editor patches the global harness H* per task\n"))

    instances = load_instances(args.data_file)
    gt = instances[0]
    if args.instance:
        matches = [i for i in instances if i.get("instance_id") == args.instance]
        if not matches:
            sys.exit(f"instance '{args.instance}' not found in {args.data_file}")
        gt = matches[0]

    # 1. the task
    hr("1. Task instance")
    print("  " + BOLD(gt["instance_id"]))
    lines = gt["problem_statement"].strip().splitlines()
    for ln in lines[:10]:
        print("  " + DIM(ln[:96]))
    if len(lines) > 10:
        print("  " + DIM("..."))
    pause(1.5)

    # 2. editor emits a per-instance patch
    hr("2. Editor reads H* + playbook -> emits a per-instance patch")
    pb = Playbook.load(args.playbook)
    ctx = format_playbook_context(pb)
    print(DIM(f"  H*: {Path(args.harness_dir).name}    playbook: {len(pb.entries)} strategies"))
    print(DIM(f"  editor: {args.advisor_model}" + (f" @ {args.advisor_url}" if args.advisor_url else " (via API)")))
    print("  generating patch ...", flush=True)
    t0 = time.time()
    edits, reasoning = generate_patch(
        args.harness_dir,
        gt["problem_statement"],
        model_name=args.advisor_model,
        api_base=args.advisor_url,
        playbook_context=ctx,
        return_reasoning=True,
    )
    print(DIM(f"  ({time.time() - t0:.1f}s)"))
    # the editor's REAL chain-of-thought (thinking-mode), truncated; skipped if none was emitted
    if reasoning and args.reasoning_lines > 0:
        rlines = [ln for ln in reasoning.splitlines() if ln.strip()]
        print(DIM("  editor reasoning (chain-of-thought, truncated):"))
        for ln in rlines[: args.reasoning_lines]:
            print("  " + DIM("  | " + ln[:90]))
        if len(rlines) > args.reasoning_lines:
            print("  " + DIM("  | ..."))
        pause(1.0)
    if not edits:
        print(YEL("  editor returned NO_PATCH_NEEDED  ->  runs the global H* unchanged"))
    else:
        print(GRN(f"  patch: {len(edits)} SEARCH/REPLACE edit(s)"))
        # Show only what each edit actually CHANGES (a line diff of search->replace,
        # with 1 line of context) so shared anchor lines don't drown out the edit.
        for i, (search, replace) in enumerate(edits[:3], 1):
            print(DIM(f"  --- edit {i} ---"))
            # unified_diff's first two lines are the ---/+++ headers: drop them by POSITION,
            # then skip only @@ markers. (Filtering by "---" prefix would also eat a removed
            # markdown "---" line, which unified_diff renders as "----".)
            diff = list(
                difflib.unified_diff(
                    search.splitlines(), replace.splitlines(), lineterm="", n=1
                )
            )
            hunk = [ln for ln in diff[2:] if not ln.startswith("@@")]
            if not hunk:
                print(DIM("  (no textual change)"))
            for shown, ln in enumerate(hunk):
                if shown >= 8:
                    print(DIM("    ..."))
                    break
                color = GRN if ln.startswith("+") else RED if ln.startswith("-") else DIM
                print("  " + color(ln[:96]))
    pause(1.5)

    # 3. apply -> H_x
    hr("3. Apply patch -> instance-specific harness H_x")
    harness_dir = args.harness_dir  # default: run H* unchanged (no-patch fallback)
    if edits:
        out_dir = Path(tempfile.mkdtemp(prefix="turbo_demo_hx_"))
        ok, msg = apply_patch(edits, args.harness_dir, str(out_dir))
        if ok:
            vok, vmsg = validate_artifact(str(out_dir))
            if vok:
                harness_dir = str(out_dir)
                print(GRN(f"  H_x ready at {out_dir}"))
            else:
                print(YEL(f"  patch validation failed ({vmsg}) -> falling back to H*"))
        else:
            print(YEL(f"  patch did not apply ({msg}) -> falling back to H*"))
    pause(1.0)

    # 4. frozen student runs H_x
    hr("4. Frozen student runs the harness in Docker")
    print(DIM(f"  student: {args.student}    harness: {Path(harness_dir).name}"))
    print("  running ...", flush=True)
    res = EX.run_instance(gt, args.student, harness=harness_dir, repo=args.repo)
    print("  " + BOLD(result_tag(res))
          + DIM(f"    steps={res.get('steps')}  cost=${res.get('agent_cost', 0):.3f}"))
    if not _scored_cleanly(res):
        print("  " + DIM("  (infra/scoring error — not a pass/fail signal; see server logs)"))

    # optional contrast
    d = None
    if args.with_default:
        hr("For contrast: the unoptimized Default harness")
        print("  running Default ...", flush=True)
        d = EX.run_instance(gt, args.student, harness="default", repo=args.repo)
        print("  " + BOLD("Default: ") + BOLD(result_tag(d)) + DIM(f"    steps={d.get('steps')}"))

    # one-line contrast (only when BOTH scored cleanly) — the headline a screenshot viewer reads first
    if d is not None and _scored_cleanly(res) and _scored_cleanly(d):
        st = "solved" if res.get("resolved") else "not solved"
        sd = "solved" if d.get("resolved") else "not solved"
        print(ORA(f"\n  Turbo H_x: {st} ({res.get('steps')} steps)"
                  f"   |   Default H*: {sd} ({d.get('steps')} steps)"))

    print(ORA(BOLD("\n  done.\n")))


if __name__ == "__main__":
    main()
