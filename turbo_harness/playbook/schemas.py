"""Data schemas for the playbook advisor pipeline.

Defines the structured records that flow through:
  ExperienceRecord → Reflection → PlaybookEntry → Playbook
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class StrategyType(str, Enum):
    TEXT_INJECTION = "TEXT_INJECTION"
    SCAFFOLD_MODIFICATION = "SCAFFOLD_MODIFICATION"


@dataclass
class ExperienceRecord:
    instance_id: str
    repo: str
    problem_statement: str
    contrast_type: str  # "contrastive", "all_pass", "all_fail"
    pass_harness: str = ""
    fail_harness: str = ""
    pass_trajectory: str = ""
    fail_trajectory: str = ""
    pass_steps: int = 0
    fail_steps: int = 0
    fail_status: str = ""
    harness_diff: str = ""
    # Legacy fields kept for backward compat with curator/reflector
    iteration: int = 0
    baseline_result: bool = False
    evolved_result: bool = False
    steps_taken: int = 0
    submission_status: str = ""
    delta: str = ""  # "improved", "degraded", "maintained"


@dataclass
class Reflection:
    instance_id: str
    issue_characteristics: list[str] = field(default_factory=list)
    strategy: str = ""
    rationale: str = ""
    confidence: float = 0.0
    outcome: str = ""  # "pass" or "fail"


@dataclass
class PlaybookEntry:
    strategy_id: str
    condition: str
    strategy: str
    evidence: dict = field(default_factory=lambda: {"helpful": 0, "harmful": 0})
    confidence: float = 0.0
    harness_diff_template: str = ""
    strategy_type: StrategyType = StrategyType.TEXT_INJECTION
    anti_overspecification_note: str = ""


@dataclass
class Playbook:
    entries: list[PlaybookEntry] = field(default_factory=list)
    general_notes: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        for entry in d["entries"]:
            entry["strategy_type"] = (
                entry["strategy_type"].value
                if isinstance(entry["strategy_type"], StrategyType)
                else entry["strategy_type"]
            )
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> Playbook:
        d = json.loads(text)
        entries = []
        for e in d.get("entries", []):
            st = e.pop("strategy_type", "TEXT_INJECTION")
            entries.append(
                PlaybookEntry(
                    **{k: v for k, v in e.items() if k != "strategy_type"},
                    strategy_type=StrategyType(st),
                )
            )
        return cls(
            entries=entries,
            general_notes=d.get("general_notes", []),
            anti_patterns=d.get("anti_patterns", []),
            metadata=d.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> Playbook:
        return cls.from_json(Path(path).read_text())
