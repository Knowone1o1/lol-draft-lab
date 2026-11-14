"""Win conditions: which champions need a lead, when the comp wants the game to end, and its objective path.

Built from Oracle's Elixir columns: game length, dragons (4+ = soul), Barons, Elders, Atakhans per team,
and each player's damage share and gold share.

- Carry table (champion, role): team win rate when that lane is 500+ gold ahead at 15 vs 500+ behind
  ("lead dependency"), plus typical damage and gold share. Shrunk towards the role average.
- Similar comps: for a draft, the K most similar historical team comps (timing score + style mix)
  give win rate by game length and how often their wins had soul / Baron / Atakhan / Elder,
  compared with their losses and with all comps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .comp import comp_profile
from .config import RAW_DIR, ROLE_LABELS, ROLES

STYLES = ["Front-to-back", "Hard engage", "Dive", "Poke / siege", "Pick", "Split push",
          "Protect the carry", "Disengage", "Skirmish"]
OBJ_PATH = {"soul": "Dragon soul (4+ drakes)", "baron": "Baron", "atakhan": "Atakhan", "elder": "Elder dragon"}


def load_extra(paths=None):
    paths = list(paths) if paths else sorted(RAW_DIR.glob("*.csv"))
    want = {"gameid", "side", "position", "gamelength", "dragons", "opp_dragons", "barons", "opp_barons",
            "elders", "opp_elders", "atakhans", "opp_atakhans", "damageshare", "earnedgoldshare"}
    frames = []
    for p in paths:
        cols = pd.read_csv(p, nrows=0).columns
        frames.append(pd.read_csv(p, usecols=[c for c in cols if c in want], low_memory=False))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["gameid", "side", "position"])
    return df[df["position"] == "team"], df[df["position"].isin(ROLES)]


def attach_extra(g: pd.DataFrame, team: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    for side in ("Blue", "Red"):
        t = team[team["side"] == side].set_index("gameid")
        pre = side.lower()
        for c in ("dragons", "barons", "elders", "atakhans"):
            if c in t:
                g[f"{pre}_{c}"] = g["gameid"].map(t[c])
        if side == "Blue" and "gamelength" in t:
            g["minutes"] = g["gameid"].map(t["gamelength"]) / 60
        p = pl[pl["side"] == side].set_index(["gameid", "position"])
        for r in ROLES:
            for c, short in (("damageshare", "dmg"), ("earnedgoldshare", "gold")):
                if c in p:
                    g[f"{side}_{r}_{short}"] = g["gameid"].map(p[c].xs(r, level="position"))
    return g


def style_vector(profile: dict) -> np.ndarray:
    pts = profile.get("style_points", {})
    v = np.array([pts.get(s, 0.0) for s in STYLES], float)
    return v / v.sum() if v.sum() else v


def team_rows(g: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with comp profile, outcome, length and objectives."""
    cache = {}

    def prof(champs):
        k = tuple(sorted(champs))
        if k not in cache:
            p = comp_profile(list(k))
            cache[k] = (p["timing_score"], style_vector(p))
        return cache[k]

    out = []
    recs = g.to_dict("records")
    for side, opp in (("Blue", "Red"), ("Red", "Blue")):
        rows = []
        for r in recs:
            ts, sv = prof([r[f"{side}_{x}_champ"] for x in ROLES])
            ots, osv = prof([r[f"{opp}_{x}_champ"] for x in ROLES])
            won = r["blue_win"] if side == "Blue" else 1 - r["blue_win"]
            pre, opre = side.lower(), opp.lower()
            rows.append({"gameid": r["gameid"], "date": r["date"], "side": side, "own": ts, "opp": ots,
                         "sv": sv, "osv": osv, "win": won, "minutes": r.get("minutes"),
                         "soul": float((r.get(f"{pre}_dragons") or 0) >= 4),
                         "baron": float((r.get(f"{pre}_barons") or 0) > 0),
                         "elder": float((r.get(f"{pre}_elders") or 0) > 0),
                         "atakhan": float((r.get(f"{pre}_atakhans") or 0) > 0),
                         "gd": (r.get("blue_gd15") if side == "Blue" else -r["blue_gd15"]) if r.get("blue_gd15") == r.get("blue_gd15") else np.nan,
                         "elo": (r["blue_elo"] - r["red_elo"]) * (1 if side == "Blue" else -1)})
        out.append(pd.DataFrame(rows))
    return pd.concat(out, ignore_index=True)


def carry_table(g: pd.DataFrame, k: float = 30.0, min_games: int = 20) -> pd.DataFrame:
    """Per (champion, role): win rate when that lane is ahead / behind at 15, damage and gold share."""
    parts = []
    for r in ROLES:
        col = f"blue_{r}_gd15"
        if col not in g:
            continue
        for side, sign in (("Blue", 1), ("Red", -1)):
            win = g["blue_win"] if side == "Blue" else 1 - g["blue_win"]
            parts.append(pd.DataFrame({"champ": g[f"{side}_{r}_champ"], "role": r, "lane_gd": sign * g[col],
                                       "win": win, "dmg": g.get(f"{side}_{r}_dmg"), "gold": g.get(f"{side}_{r}_gold")}))
    d = pd.concat(parts, ignore_index=True).dropna(subset=["lane_gd"])
    d["ahead"], d["behind"] = d["lane_gd"] >= 500, d["lane_gd"] <= -500
    role = d.groupby("role").agg(r_ahead=("win", lambda s: s[d.loc[s.index, "ahead"]].mean()),
                                 r_behind=("win", lambda s: s[d.loc[s.index, "behind"]].mean()),
                                 r_dmg=("dmg", "mean"), r_gold=("gold", "mean"))
    agg = d.groupby(["champ", "role"]).agg(n=("win", "size"), wr=("win", "mean"), dmg=("dmg", "mean"), gold=("gold", "mean"),
                                           n_ahead=("ahead", "sum"), w_ahead=("win", lambda s: s[d.loc[s.index, "ahead"]].sum()),
                                           n_behind=("behind", "sum"), w_behind=("win", lambda s: s[d.loc[s.index, "behind"]].sum()))
    agg = agg[agg["n"] >= min_games].join(role, on="role")
    agg["wr_ahead"] = (agg["w_ahead"] + k * agg["r_ahead"]) / (agg["n_ahead"] + k)
    agg["wr_behind"] = (agg["w_behind"] + k * agg["r_behind"]) / (agg["n_behind"] + k)
    agg["lead_dependency"] = agg["wr_ahead"] - agg["wr_behind"]
    agg["role_dependency"] = agg["r_ahead"] - agg["r_behind"]
    return agg.reset_index()


GOLD_BINS = [-1e9, -3000, -1500, -500, 500, 1500, 3000, 1e9]


class SimilarComps:
    """Nearest historical team comps by timing score and style mix (opponent timing as a weaker term).

    Bias control: early leads cause short games and objectives (a team ahead at 15 takes drakes, towers and
    Baron because it is ahead). So every comparison is made against what an AVERAGE comp with the SAME gold
    at 15 did: expected win rate per game-length bucket and expected objective rate in wins come from all
    comps in the same gold-at-15 band. What is left over is the comp type's own tendency. It is still an
    association, not proof that the objective causes the win (post-15 events are not observed).
    """

    def __init__(self, rows: pd.DataFrame, k: int = 400):
        r = rows[rows["minutes"].notna() & rows["gd"].notna()].reset_index(drop=True)
        r["gbin"] = pd.cut(r["gd"], GOLD_BINS, labels=False).astype(int)
        r["lbin"] = np.select([r["minutes"] < 27, r["minutes"] > 33], [0, 2], 1)
        self.rows = r
        self.k = k
        self.X = np.column_stack([r["own"].to_numpy() * 1.5, np.vstack(r["sv"].to_numpy()), r["opp"].to_numpy() * 0.5])
        # Baselines over all comps, by gold-at-15 band (and length bucket for win rate).
        self.base_wr = r.groupby(["gbin", "lbin"])["win"].mean().to_dict()
        # Only objectives that exist in this data (e.g. Atakhan was in 2025 but removed for 2026).
        self.active = [o for o in OBJ_PATH if r[o].mean() > 0.02]
        w = r[r["win"] == 1]
        self.base_obj = {o: w.groupby("gbin")[o].mean().to_dict() for o in self.active}
        self.base = self._summary(r)

    def query(self, own_ts, sv, opp_ts):
        q = np.r_[own_ts * 1.5, sv, opp_ts * 0.5]
        d = np.sqrt(((self.X - q) ** 2).sum(1))
        idx = np.argsort(d)[: self.k]
        return self._summary(self.rows.iloc[idx])

    def _summary(self, r):
        w = r[r["win"] == 1]
        out = {"n": len(r), "win_rate": float(r["win"].mean()),
               "win_minutes": float(w["minutes"].median()) if len(w) else None}
        for lb, name in ((0, "short"), (2, "long")):
            s = r[r["lbin"] == lb]
            exp = np.mean([self.base_wr.get((gb, lb), 0.5) for gb in s["gbin"]]) if len(s) else np.nan
            out[f"wr_{name}"] = float(s["win"].mean()) if len(s) else None
            out[f"exp_{name}"] = float(exp) if len(s) else None
            out[f"n_{name}"] = int(len(s))
        for o in self.active:
            out[f"{o}_in_wins"] = float(w[o].mean()) if len(w) else 0.0
            out[f"{o}_expected"] = float(np.mean([self.base_obj[o].get(gb, 0.0) for gb in w["gbin"]])) if len(w) else 0.0
        out["n_wins"] = int(len(w))
        return out


CAVEAT = ("Compared with comps that had the same gold at 15, so an early lead alone doesn't explain it. "
          "Still an association: objectives taken after 15 are partly a result of how the game is going.")


def describe_path(sim: dict, base: dict) -> dict:
    """End-early/late tendency and objective tendency, each measured against comps with the same gold at 15."""
    ex_s = (sim["wr_short"] - sim["exp_short"]) if sim.get("wr_short") is not None and sim["n_short"] >= 30 else None
    ex_l = (sim["wr_long"] - sim["exp_long"]) if sim.get("wr_long") is not None and sim["n_long"] >= 30 else None
    if ex_s is not None and ex_l is not None:
        se = float(np.sqrt(0.25 / sim["n_short"] + 0.25 / sim["n_long"]))
        diff = ex_l - ex_s
        detail = (f"Against comps with the same gold at 15, similar comps do {ex_s * 100:+.0f} pts in games under 27 min "
                  f"and {ex_l * 100:+.0f} pts in games over 33.")
        if diff <= -max(0.04, 1.5 * se):
            timing = ("end_early", f"Wants to end early. {detail}")
        elif diff >= max(0.04, 1.5 * se):
            timing = ("end_late", f"Can play long. {detail}")
        else:
            timing = ("flexible", f"No clear game-length preference once early gold is accounted for. {detail}")
    else:
        timing = ("flexible", "Not enough similar games in short and long games to call a game-length preference.")
    paths = []
    nw = max(sim.get("n_wins", 0), 1)
    for o, label in OBJ_PATH.items():
        if f"{o}_in_wins" not in sim:  # objective not in the current season
            continue
        act, exp = sim[f"{o}_in_wins"], sim[f"{o}_expected"]
        se = float(np.sqrt(max(exp * (1 - exp), 0.01) / nw))
        paths.append({"key": o, "label": label, "in_wins": act, "expected_same_gold": exp,
                      "lift": act - exp, "clear": abs(act - exp) > max(0.04, 1.5 * se)})
    rank = sorted([p for p in paths if p["clear"] and p["lift"] > 0], key=lambda p: -p["lift"])
    if rank:
        top = rank[0]
        what = {"soul": "drakes and soul", "baron": "Baron", "atakhan": "Atakhan", "elder": "late Elder fights"}[top["key"]]
        msg = (f"Their wins lean on {what}: {top['label'].lower()} in {top['in_wins']:.0%} of wins, against "
               f"{top['expected_same_gold']:.0%} for other comps' wins with the same gold at 15.")
        if top["key"] == "elder":
            msg += " If the enemy gets soul, this comp can still flip the game on Elder."
    else:
        msg = "No objective stands out for this comp type once early gold is accounted for: play the standard path."
    return {"timing": timing[0], "timing_text": timing[1], "objective_text": msg, "objectives": paths,
            "win_minutes": sim.get("win_minutes"), "n_similar": sim["n"], "caveat": CAVEAT}


def carries_for(comp: dict, table: pd.DataFrame, top: int = 2):
    """comp: {role: champion}. Champions whose lane lead matters most for the team, with their numbers."""
    out = []
    for r, c in comp.items():
        if not c:
            continue
        row = table[(table["champ"] == c) & (table["role"] == r)]
        if row.empty:
            continue
        x = row.iloc[0]
        dmg = float(x.dmg) if x.dmg == x.dmg else 0.0
        excess = float(x.lead_dependency - x.role_dependency)
        out.append({"champion": c, "role": r, "role_label": ROLE_LABELS[r], "wr_ahead": float(x.wr_ahead),
                    "wr_behind": float(x.wr_behind), "lead_dependency": float(x.lead_dependency),
                    "vs_role": excess, "dmg_share": dmg, "gold_share": float(x.gold) if x.gold == x.gold else None,
                    "games": int(x.n),
                    # Win condition = carries the damage AND rides on its lane more than its role usually does.
                    # (Support/jungle gold leads mostly mirror the whole game, so the raw gap is not used alone.)
                    "score": dmg + 1.5 * excess})
    out.sort(key=lambda d: -d["score"])
    return out[:top], out
