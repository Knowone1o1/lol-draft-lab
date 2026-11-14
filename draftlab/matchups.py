"""Cross-season lane matchup analysis: which champions are consistently good into which.

For every lane matchup (A vs B in the same role) and every season, we compare A's actual wins with
the wins expected from A's and B's overall strength *in that season and role*. Summing over seasons
gives a lift that isn't fooled by "A was just meta that year", plus how many seasons the edge held.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ROLE_LABELS, ROLES


def lane_rows(games: pd.DataFrame) -> pd.DataFrame:
    from .stats import fearless_game_weights
    parts = []
    year = games["date"].dt.year
    gw = fearless_game_weights(games)
    for r in ROLES:
        for side, opp, win in (("Blue", "Red", games["blue_win"]), ("Red", "Blue", 1 - games["blue_win"])):
            parts.append(pd.DataFrame({"role": r, "champ": games[f"{side}_{r}_champ"].values,
                                       "vs": games[f"{opp}_{r}_champ"].values,
                                       "win": np.asarray(win), "year": year.values, "w": gw}))
    return pd.concat(parts, ignore_index=True)


def matchup_table(games: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """Weighted by fearless game number: late-series games (drafted from leftovers) count less."""
    lr = lane_rows(games)
    lr["ww"] = lr["w"] * lr["win"]
    # Champion strength per season and role (weighted, shrunk towards 50% with 20 pseudo-games).
    cs = lr.groupby(["year", "role", "champ"])[["ww", "w"]].sum()
    wr = (cs["ww"] + 10) / (cs["w"] + 20)
    lr = lr.join(wr.rename("wr_a"), on=["year", "role", "champ"])
    lr = lr.join(wr.rename_axis(["year", "role", "vs"]).rename("wr_b"), on=["year", "role", "vs"])
    lr["exp"] = (0.5 + 0.5 * (lr["wr_a"] - lr["wr_b"])).clip(0.05, 0.95)
    lr["we"] = lr["w"] * lr["exp"]
    lr["wvar"] = lr["w"] ** 2 * lr["exp"] * (1 - lr["exp"])

    per_year = lr.groupby(["role", "champ", "vs", "year"]).agg(g=("win", "size"), w=("ww", "sum"), e=("we", "sum"))
    per_year["edge"] = per_year["w"] - per_year["e"]
    yrs = per_year[per_year["g"] >= 3].groupby(level=[0, 1, 2]).agg(
        seasons=("edge", "size"), seasons_ahead=("edge", lambda s: int((s > 0).sum())))

    t = lr.groupby(["role", "champ", "vs"]).agg(games=("win", "size"), raw_wins=("win", "sum"),
                                                wgames=("w", "sum"), wins=("ww", "sum"),
                                                expected=("we", "sum"), var=("wvar", "sum"),
                                                first=("year", "min"), last=("year", "max"))
    t = t[t["games"] >= min_games].join(yrs, how="left").fillna({"seasons": 0, "seasons_ahead": 0})
    t["win_rate"] = t["raw_wins"] / t["games"]
    t["lift"] = (t["wins"] - t["expected"]) / t["wgames"]
    t["z"] = (t["wins"] - t["expected"]) / np.sqrt(t["var"].clip(lower=1e-9))
    t = t.drop(columns=["var", "raw_wins", "wins", "wgames"]).reset_index()
    t["seasons"] = t["seasons"].astype(int)
    t["seasons_ahead"] = t["seasons_ahead"].astype(int)
    return t.sort_values("z", ascending=False)


def team_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Every champion vs every champion on the enemy team (25 pairs per side per game)."""
    from .stats import fearless_game_weights
    parts = []
    year = games["date"].dt.year.values
    gw = fearless_game_weights(games)
    for side, opp, win in (("Blue", "Red", games["blue_win"].values), ("Red", "Blue", 1 - games["blue_win"].values)):
        for r in ROLES:
            for r2 in ROLES:
                parts.append(pd.DataFrame({"role": "team", "champ": games[f"{side}_{r}_champ"].values,
                                           "vs": games[f"{opp}_{r2}_champ"].values, "vs_role": r2,
                                           "win": win, "year": year, "w": gw}))
    return pd.concat(parts, ignore_index=True)


def team_matchup_table(games: pd.DataFrame, min_games: int = 15) -> pd.DataFrame:
    """Champion vs any enemy champion, any role (e.g. Sylas vs Malphite), fearless-weighted."""
    lr = team_rows(games)
    lr["ww"] = lr["w"] * lr["win"]
    # Season strength from each champion's own games (each game appears 5x per champion here, fine for a rate).
    cs = lr.groupby(["year", "champ"])[["ww", "w"]].sum()
    wr = (cs["ww"] + 50) / (cs["w"] + 100)
    lr = lr.join(wr.rename("wr_a"), on=["year", "champ"])
    lr = lr.join(wr.rename_axis(["year", "vs"]).rename("wr_b"), on=["year", "vs"])
    lr["exp"] = (0.5 + (lr["wr_a"] - 0.5) - (lr["wr_b"] - 0.5)).clip(0.05, 0.95)
    lr["we"] = lr["w"] * lr["exp"]
    lr["wvar"] = lr["w"] ** 2 * lr["exp"] * (1 - lr["exp"])
    per_year = lr.groupby(["champ", "vs", "year"]).agg(g=("win", "size"), w=("ww", "sum"), e=("we", "sum"))
    per_year["edge"] = per_year["w"] - per_year["e"]
    yrs = per_year[per_year["g"] >= 3].groupby(level=[0, 1]).agg(
        seasons=("edge", "size"), seasons_ahead=("edge", lambda s: int((s > 0).sum())))
    t = lr.groupby(["champ", "vs"]).agg(games=("win", "size"), raw_wins=("win", "sum"), wgames=("w", "sum"),
                                        wins=("ww", "sum"), expected=("we", "sum"), var=("wvar", "sum"),
                                        first=("year", "min"), last=("year", "max"),
                                        vs_role=("vs_role", lambda s: s.mode().iloc[0]))
    t = t[t["games"] >= min_games].join(yrs, how="left").fillna({"seasons": 0, "seasons_ahead": 0})
    t["win_rate"] = t["raw_wins"] / t["games"]
    t["lift"] = (t["wins"] - t["expected"]) / t["wgames"]
    t["z"] = (t["wins"] - t["expected"]) / np.sqrt(t["var"].clip(lower=1e-9))
    t = t.drop(columns=["var", "raw_wins", "wins", "wgames"]).reset_index()
    t["role"] = "team"
    t["seasons"] = t["seasons"].astype(int)
    t["seasons_ahead"] = t["seasons_ahead"].astype(int)
    return t.sort_values("z", ascending=False)


def counters_report(t: pd.DataFrame, current_champs: set, top: int = 12, min_games: int = 20) -> str:
    lines = ["# Lane counters that hold up across seasons", "",
             "For each lane matchup we compare actual wins with the wins expected from both champions' "
             "strength *in that season and role*. **Lift** is the extra win rate beyond that expectation; "
             "**z** is how many standard errors it is from zero (|z| > 2 is unlikely to be noise); "
             "**Seasons ahead** counts the seasons (with 3+ games) where the edge was positive. "
             "In fearless series, later games are down-weighted (G1 1.0 → G5 0.35) because they are "
             "drafted from leftover pools; win rate is the raw rate.", "",
             f"Only matchups with at least {min_games} pro games and champions still played in the current "
             "meta are listed. Full table: `reports/matchups.csv`.", ""]
    for r in ROLES:
        sub = t[(t["role"] == r) & (t["games"] >= min_games) & t["champ"].isin(current_champs) & t["vs"].isin(current_champs)]
        sub = sub[sub["z"] > 0].head(top)
        lines += [f"## {ROLE_LABELS[r]}", "", "| Champion | Good into | Games | Win rate | Lift | z | Seasons ahead | Years |",
                  "|---|---|---|---|---|---|---|---|"]
        for x in sub.itertuples():
            lines.append(f"| {x.champ} | {x.vs} | {x.games} | {x.win_rate:.0%} | {x.lift:+.1%} | {x.z:.1f} | "
                         f"{x.seasons_ahead}/{x.seasons} | {x.first}-{x.last} |")
        lines.append("")
    team = t[(t["role"] == "team") & (t["games"] >= min_games * 2) & t["champ"].isin(current_champs)
             & t["vs"].isin(current_champs) & (t["z"] > 0)].head(top * 2)
    if len(team):
        lines += ["## Against the whole enemy team (any role)", "",
                  "Cross-role interactions a lane table can't see, e.g. Sylas stealing Malphite's ultimate.", "",
                  "| Champion | Good vs (enemy, usual role) | Games | Win rate | Lift | z | Seasons ahead | Years |",
                  "|---|---|---|---|---|---|---|---|"]
        for x in team.itertuples():
            lines.append(f"| {x.champ} | {x.vs} ({ROLE_LABELS.get(x.vs_role, x.vs_role)}) | {x.games} | {x.win_rate:.0%} | "
                         f"{x.lift:+.1%} | {x.z:.1f} | {x.seasons_ahead}/{x.seasons} | {x.first}-{x.last} |")
        lines.append("")
    lines += ["## Caveats", "",
              "- Pro samples are small: even 40 games leaves roughly ±8% uncertainty on a win rate. Use z and seasons, not lift alone.",
              "- Reworks and item changes can break old matchups; check that the years column includes recent seasons.",
              "- Draft context matters: a counter pick is often a last-pick decision, so it's partly selection bias "
              "(teams counter-pick when the matchup is favourable for reasons the data doesn't see)."]
    return "\n".join(lines)


def best_answers(t: pd.DataFrame, champ: str, role: str | None = None, min_games: int = 8, k: int = 10):
    """Champions that do best into `champ` (in its lane), and the ones it beats."""
    sub = t[t["games"] >= min_games]
    sub = sub[sub["role"] == role] if role else sub[sub["role"] != "team"]
    into = sub[sub["vs"] == champ].sort_values("z", ascending=False).head(k)
    beats = sub[sub["champ"] == champ].sort_values("z", ascending=False).head(k)
    cols = ["role", "champ", "vs", "games", "win_rate", "lift", "z", "seasons_ahead", "seasons", "first", "last"]
    return into[cols].to_dict("records"), beats[cols].to_dict("records")
