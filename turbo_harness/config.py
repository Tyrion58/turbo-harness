"""Central configuration for Turbo Harness.

All node-specific values — GCP project ids, model ids, server URLs, and storage
roots — are read from environment variables (see ``.env.example``) so the code is
portable across machines. Import these values here instead of hardcoding paths or
identifiers in modules or scripts.

This module imports only the standard library so it is safe to import before any
third-party dependency is installed.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (…/turbo-harness), independent of the current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(env_var: str, default: Path) -> Path:
    """Return an env-var path override, else ``default``.

    A relative override is resolved against ``REPO_ROOT`` so behavior does not
    depend on the current working directory (matches the unset default).
    """
    value = os.environ.get(env_var)
    if not value:
        return default
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p)


# --- Vertex AI / model providers -------------------------------------------------
VERTEXAI_PROJECT = os.environ.get("VERTEXAI_PROJECT", "your-gcp-project")
VERTEXAI_LOCATION = os.environ.get("VERTEXAI_LOCATION", "us-east5")

# Frontier model used for Meta-Harness proposals and playbook curation.
FRONTIER_MODEL = os.environ.get("TURBO_FRONTIER_MODEL", "vertex_ai/claude-sonnet-4-5")

# --- SWE coding servers ----------------------------------------------------------
AGENT_SERVER_URL = os.environ.get("AGENT_SERVER_URL", "http://localhost:8081")
EVAL_SERVER_URL = os.environ.get("EVAL_SERVER_URL", "http://localhost:5152")

# --- Storage roots ---------------------------------------------------------------
ARTIFACTS_DIR = _path("TURBO_ARTIFACTS_DIR", REPO_ROOT / "artifacts")
DATA_DIR = _path("TURBO_DATA_DIR", REPO_ROOT / "data")
ADVISOR_CKPT_DIR = _path("TURBO_ADVISOR_CKPT_DIR", REPO_ROOT / "checkpoints")
ADVISOR_EXPORT_DIR = _path("TURBO_ADVISOR_EXPORT_DIR", REPO_ROOT / "exports")


__all__ = [
    "REPO_ROOT",
    "VERTEXAI_PROJECT",
    "VERTEXAI_LOCATION",
    "FRONTIER_MODEL",
    "AGENT_SERVER_URL",
    "EVAL_SERVER_URL",
    "ARTIFACTS_DIR",
    "DATA_DIR",
    "ADVISOR_CKPT_DIR",
    "ADVISOR_EXPORT_DIR",
]
