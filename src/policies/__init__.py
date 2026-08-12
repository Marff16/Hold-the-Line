"""Team policies (baselines now, learned RL policies later)."""

from src.policies.base import Policy
from src.policies.obstacle_avoiding_policy import ObstacleAvoidingPolicy
from src.policies.random_policy import RandomPolicy

__all__ = ["Policy", "RandomPolicy", "ObstacleAvoidingPolicy"]

# LearnedPolicy needs torch (the "rl" extra) - keep it optional so importing
# src.policies doesn't force a torch install on installs that only need the
# lightweight heuristic policies (e.g. the "web" extra alone).
try:
    from src.policies.learned_policy import LearnedPolicy

    __all__.append("LearnedPolicy")
except ImportError:
    LearnedPolicy = None  # type: ignore[assignment,misc]
