"""Pre-game win models behind one interface (base.WinModel). The app and the harness pick one by name."""
from .base import WinModel, daily_refit
from .baselines import AlwaysBlue, EloLeagueLogit, EloLogit
from .draft import DRAFT_MODELS
from .strength import (StrengthGold15, StrengthGold15Tier, StrengthGold15TierCal, StrengthGold25,
                       StrengthLength, StrengthModel, StrengthNoForm)

REGISTRY = {m.name: m for m in (AlwaysBlue, EloLogit, EloLeagueLogit, StrengthModel, StrengthNoForm,
                                    StrengthGold15, StrengthGold25, StrengthLength,
                                    StrengthGold15Tier, StrengthGold15TierCal, *DRAFT_MODELS)}


def get_model(name: str, **params) -> WinModel:
    if name not in REGISTRY:
        raise KeyError(f"unknown win model '{name}'; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


__all__ = ["REGISTRY", "WinModel", "daily_refit", "get_model"]
