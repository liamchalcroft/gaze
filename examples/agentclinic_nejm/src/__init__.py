"""AgentClinic NEJM example: Multi-turn diagnostic reasoning.

This example implements the AgentClinic NEJM environment using the
verifiers package for multi-turn clinical case evaluation.

The assistant acts as a clinician, gathering information from a
simulated patient through HISTORY, EXAM, TESTS, and IMAGE requests
before making a final diagnosis.

Based on: https://github.com/SamuelSchmidgall/AgentClinic
Dataset: agentclinic_nejm_extended.jsonl
"""

from __future__ import annotations

from .environment import AgentClinicNEJMMultiTurn
from .environment import load_environment

__all__ = [
    "AgentClinicNEJMMultiTurn",
    "load_environment",
]
