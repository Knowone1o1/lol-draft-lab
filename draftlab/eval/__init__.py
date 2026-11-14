"""Walk-forward evaluation harness for pre-game win models (Track A)."""
from .dataset import load_eval_games, tier_of
from .harness import evaluate

__all__ = ["evaluate", "load_eval_games", "tier_of"]
