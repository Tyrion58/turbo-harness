"""Playbook builder: distill meta-harness search trajectories into a harness-editing playbook.

Experience-driven harness adaptation: extracts issue-conditional strategies
from meta-harness evolution data and injects them into the patch-advisor
pipeline as a structured playbook.

Pipeline:
  extract_experiences → reflector → curator → memory_advisor
  (offline, once)       (offline)   (offline)  (online, per-issue)
"""
