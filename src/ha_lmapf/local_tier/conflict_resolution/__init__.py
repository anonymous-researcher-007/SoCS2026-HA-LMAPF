"""
Conflict Resolution Strategies.

This package contains algorithms for decentralized agent-agent deconfliction (Tier-2).
These resolvers determine which agent yields when a potential collision is detected
during local execution.

Modules:
  - base: Abstract base class and shared conflict detection logic.
  - token_passing: Communication-based resolution (Algorithm 3A).
  - priority_rules: Communication-free deterministic resolution (Algorithm 3B).
  - pibt: Priority Inheritance with Backtracking (optional extension).
"""

from .base import (
    BaseConflictResolver,
    ImminentConflict,
    detect_imminent_conflict,
)

from .token_passing import TokenPassingResolver
from .priority_rules import PriorityRulesResolver
from .pibt import PIBTResolver

__all__ = [
    "BaseConflictResolver",
    "ImminentConflict",
    "detect_imminent_conflict",
    "TokenPassingResolver",
    "PriorityRulesResolver",
    "PIBTResolver",
]