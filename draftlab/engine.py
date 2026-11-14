"""Runtime draft engine: recommendations, opponent AI and live win probability."""
from __future__ import annotations

import math
import pickle

import numpy as np

from .config import MODEL_DIR, ROLE_LABELS, ROLES
from .data import other
from .features import BAN_FEATURES, PICK_FEATURES, DraftState, StateContext, ban_matrix, comp_features, pick_matrix

ROLE_KEYS = ROLES


class _WithPick(DraftState):
    """A draft state with one extra (hypothetical) pick for `side`, used to ask "what if they get it?"."""

    def __init__(self, base: DraftState, side: str, champ: str):
        super().__init__(base.blue_team, base.red_team, base.blue_roster, base.red_roster, list(base.actions), base.locked,
                         base.blue_first)
        self._extra_side, self._extra = side, champ

    def picks(self, side):
        p = super().picks(side)
        return p + [self._extra] if side == self._extra_side else p


class DraftEngine:
    def __init__(self, path=None):
        path = path or MODEL_DIR / "draftlab.pkl"
        with open(path, "rb") as f:
            art = pickle.load(f)
        self.snap = art["snapshot"]
        self.models = art["models"]
        self.elo = art["elo"]
        self.matchups = art.get("matchups")
        self.info = {k: art.get(k) for k in ("built", "data_range", "train_leagues", "patch")}
        self.champions = self.snap.champions
        from .config import league_elo
        self.league = league_elo()  # league strength offsets for cross-league matchups
        # Optional: pair stats for the latest patches (scripts/build_recent.py), for "this patch" reasons.
        try:
            with open(MODEL_DIR / "recent.pkl", "rb") as f:
                self.recent = pickle.load(f)
        except FileNotFoundError:
            self.recent = None
        # Win model (A6): A3 strength + A4 draft layer when configured and built, else the original model.
        from .config import WIN_MODEL
        self.win_model = None
        if WIN_MODEL == "a3a4" and (MODEL_DIR / "a3a4_win.json").exists():
            from .winmodels.runtime import A3A4WinModel
            self.win_model = A3A4WinModel(MODEL_DIR / "a3a4_win.json")
        self.info["win_model"] = self.win_model.describe() if self.win_model else {"name": "legacy"}

    # ------------------------------------------------------------ helpers
    def name(self, pkey):
        return self.snap.player_name.get(pkey, pkey) if pkey else "?"

    def candidates(self, state: DraftState):
        bad = state.unavailable()
        return [c for c in self.champions if c not in bad]

    def model_for(self, state: DraftState, kind: str):
        if kind == "pick":
            return self.models["pick"], "pick"
        return (self.models["ban1"] if len(state.actions) < 6 else self.models["ban2"]), "ban"

    def team_elo(self, team):
        """Team Elo plus its home league's strength offset (so an ERL champion isn't rated above Gen.G)."""
        off = self.league["offsets"].get(self.league["home"].get(team), 0.0)
        return self.elo.get(team, 1500.0) + off

    def win_prob(self, state: DraftState, ctx=None) -> float:
        if self.win_model is not None:
            x = comp_features(state, self.snap, 0.0, 0.0, ctx=ctx)[1:]
            return self.win_model.prob(state.blue_team, state.red_team, state.blue_first, x)
        x = comp_features(state, self.snap, self.team_elo(state.blue_team), self.team_elo(state.red_team), ctx=ctx)
        return float(self.models["win"].proba(x)[0])

    def level_win_prob(self, state: DraftState, ctx=None):
        """Blue's chance from the draft alone, both teams equally strong, no side / pick-order term (A4).
        None with the legacy model, which has no separate draft layer."""
        if self.win_model is None:
            return None
        return self.win_model.level_prob(comp_features(state, self.snap, 0.0, 0.0, ctx=ctx)[1:])

    def roles(self, state: DraftState):
        ctx = StateContext(state, self.snap)
        return {s: {c: ROLES[r] for c, r in ctx.roles[s].items()} for s in ("Blue", "Red")}

    def action_probs(self, state: DraftState, side: str, kind: str, temperature: float = 1.0):
        """Imitation-model probabilities: 'what would this team do here?'"""
        cands = self.candidates(state)
        ctx = StateContext(state, self.snap)
        model, k = self.model_for(state, kind)
        if k == "pick":
            X = pick_matrix(state, side, cands, self.snap, ctx)
            fit = np.array([ctx.best_role(side, c)[0] for c in cands])
            ok = fit >= 0.03
            if ok.any():
                cands = [c for c, o in zip(cands, ok) if o]
                X = X[ok]
        else:
            X = ban_matrix(state, side, cands, self.snap, ctx)
        # A model saved before new features were added only knows its own columns: use just those.
        names = PICK_FEATURES if k == "pick" else BAN_FEATURES
        if list(model.feature_names) != list(names):
            X = X[:, [names.index(n) for n in model.feature_names]]
        return cands, model.proba(X, temperature), X, ctx

    # ------------------------------------------------------------ AI opponent
    def ai_move(self, state: DraftState, temperature: float = 0.7, top_k: int = 8, rng=None, mode: str = "imitate"):
        """mode "imitate": draft like the real team (pick/ban habits). "smart": take one of the top
        recommendations (win-chance gain for picks, threat x damage for bans), with a little variety."""
        side, kind, _ = state.next_action()
        rng = rng or np.random.default_rng()
        if mode == "smart":
            recs = self.recommend(state, k=3)
            if recs:
                w = np.array([0.6, 0.25, 0.15][:len(recs)])
                r = recs[rng.choice(len(recs), p=w / w.sum())]
                return r["champion"], float(r.get("model_prob", 0.0))
        cands, p, _, _ = self.action_probs(state, side, kind, temperature)
        order = np.argsort(-p)[:top_k]
        pp = p[order] / p[order].sum()
        i = order[rng.choice(len(order), p=pp)]
        return cands[i], float(p[i])

    # ------------------------------------------------------------ recommendations
    def recommend(self, state: DraftState, k: int = 8, games_remaining: int = 0, sort: str = "pro"):
        """sort: "blend" (win-chance gain + pro priority), "win" (win-chance gain only), "pro" (what pros pick)."""
        nxt = state.next_action()
        if nxt is None:
            return []
        side, kind, _ = nxt
        if kind == "pick":
            return self._recommend_pick(state, side, k, games_remaining, sort)
        return self._recommend_ban(state, side, k)

    def _comfort_rank(self, player, champ, unavailable) -> int | None:
        """Rank of champ in the player's (recency-weighted) pool among champions still available."""
        if not player:
            return None
        if not hasattr(self, "_pools"):
            pools = {}
            for (p, c), g in self.snap.player_champ.items():
                if g >= 1.0:
                    pools.setdefault(p, []).append((g, c))
            self._pools = {p: [c for _, c in sorted(v, reverse=True)] for p, v in pools.items()}
        rank = 0
        for c in self._pools.get(player, []):
            if c in unavailable:
                continue
            rank += 1
            if c == champ:
                return rank
        return None

    def _fearless_notes(self, state, side, c, ctx, games_remaining):
        """(warnings, reasons): using your comfort pick is a COST; denying theirs is a benefit."""
        warnings, reasons = [], []
        unavailable = state.unavailable()
        fit, ri = ctx.best_role(side, c)
        if ri is not None:
            player = state.roster(side).get(ROLES[ri])
            rank = self._comfort_rank(player, c, unavailable - {c})
            if rank is not None and rank <= 3:
                n = games_remaining
                warnings.append(f"Uses {self.name(player)}'s #{rank} comfort pick: gone for the next {n} game{'s' if n > 1 else ''}")
        enemy = other(side)
        e_roster = state.roster(enemy)
        best = None
        for r in ctx.open[enemy]:
            pl = e_roster.get(ROLES[r])
            rank = self._comfort_rank(pl, c, unavailable - {c})
            if rank is not None and rank <= 3 and (best is None or rank < best[0]):
                best = (rank, pl)
        if best:
            reasons.append(f"Also takes {self.name(best[1])}'s #{best[0]} comfort pick away for the series")
        return warnings, reasons

    @staticmethod
    def edge_label(delta: float) -> str:
        """The draft's effect on win chance is small and not significant in backtests: a label, not a number."""
        if delta >= 0.02:
            return "slight edge"
        if delta <= -0.02:
            return "slight drawback"
        return "no clear edge"

    def _recommend_pick(self, state, side, k, games_remaining=0, sort="pro"):
        cands, p, X, ctx = self.action_probs(state, side, "pick")
        base = self.win_prob(state, ctx)
        sign = 1 if side == "Blue" else -1
        # Only evaluate win impact for the plausible top of the list (keeps it fast).
        order = np.argsort(-p)
        pro_rank = {cands[i]: n + 1 for n, i in enumerate(order)}
        out = []
        for i in order[:40]:
            c = cands[i]
            st = DraftState(state.blue_team, state.red_team, state.blue_roster, state.red_roster,
                            state.actions + [c], state.locked, state.blue_first)
            delta = sign * (self.win_prob(st) - base)
            prio = math.log(p[i] / p.max() + 1e-6)
            score = {"win": delta, "pro": float(p[i])}.get(sort, delta + 0.02 * prio)
            out.append((score, c, delta, float(p[i]), X[i]))
        out.sort(key=lambda t: -t[0])
        recs = []
        for score, c, delta, prob, x in out[:k]:
            headline, reasons = self._pick_reasons(state, side, c, ctx)
            warnings = []
            if games_remaining > 0:
                w, r = self._fearless_notes(state, side, c, ctx, games_remaining)
                warnings, reasons = w, reasons + r
            recs.append({"champion": c, "score": round(float(score), 4), "win_delta": round(float(delta), 4),
                         "edge": self.edge_label(delta), "model_prob": round(prob, 4), "pro_rank": pro_rank[c],
                         "headline": headline, "reasons": reasons, "warnings": warnings})
        return recs

    def _pick_reasons(self, state, side, c, ctx):
        """(headline, details) in plain words. Matchup numbers carry their sample size: most are within noise."""
        snap = self.snap
        fit, ri = ctx.best_role(side, c)
        reasons, heads = [], []          # heads: (priority, text) candidates for the headline
        if ri is not None:
            role = ROLES[ri]
            player = state.roster(side).get(role)
            g = snap.player_games_on(player, c)
            if g >= 0.5:
                reasons.append(f"{self.name(player)} plays it: ~{g:.0f} recent pro games, won {snap.player_wr_on(player, c):.0%}")
                if g >= 3:
                    heads.append((3, f"{self.name(player)}'s champion (~{g:.0f} recent games)"))
            else:
                reasons.append(f"{self.name(player)} has no recent pro games on it")
            opp = ctx.by_role[other(side)].get(ri)
            if opp and snap.lane_games(c, opp, ri) >= 20:
                cnt, n = snap.counter(c, opp, ri), snap.lane_games(c, opp, ri)
                if abs(cnt) >= 0.02:
                    verb = "do well" if cnt > 0 else "struggle"
                    reasons.append(f"Tends to {verb} vs {opp} in lane ({cnt:+.0%}, ~{n:.0f} games, could be chance)")
                    if cnt > 0:
                        heads.append((1, f"Possible lane counter to {opp}"))
        own = state.picks(side)
        if own:
            pal = max(own, key=lambda a: snap.copick_lift(c, a))
            if snap.copick_lift(c, pal) >= 0.26:
                reasons.append(f"Pros often pair it with {pal}")
                heads.append((2, f"Pros pair it with your {pal}"))
            if self.recent:
                rp = self.recent["pairs"]
                hits = [(rp[tuple(sorted((c, a)))], a) for a in own if tuple(sorted((c, a))) in rp]
                hits = [(h, a) for h, a in hits if h["games"] >= 8 and h["lift"] >= 1.5]
                if hits:
                    h, a = max(hits, key=lambda t: t[0]["lift"])
                    reasons.append(f"This patch ({self.recent['label']}): paired with {a} {h['games']} times, "
                                   f"won {h['wins']} of {h['games']}")
        enemy_picks = state.picks(other(side))
        if enemy_picks:
            tgt = max(enemy_picks, key=lambda e: snap.vspick_lift(c, e))
            if snap.vspick_lift(c, tgt) >= 0.26:
                reasons.append(f"Pros often pick it into {tgt}")
        reasons.append(f"Picked or banned in {snap.presence(c):.0%} of recent pro games")
        flex = [ROLE_LABELS[r] for r, v in zip(ROLES, snap.role_p(c)) if v > 0.15]
        if len(flex) > 1:
            reasons.append("Flex pick: " + " / ".join(flex))
            heads.append((0, "Flex pick (" + " / ".join(flex) + ")"))
        headline = max(heads)[1] if heads else f"Common pick ({snap.presence(c):.0%} picked or banned)"
        return headline, reasons

    def _counter_to_ours(self, state, side, ctx, c, min_games: float = 30.0):
        """Strongest LANE counter of champion c against one of our picks, if the enemy can still play it in
        that lane and there are at least `min_games` weighted games. Whole-team (any role) matchups are not
        used: their spread matches pure noise (judge report M3). Returns (value, our champion, games)."""
        best = (0.0, None, 0.0)
        rp = self.snap.role_p(c)
        e_open = ctx.open[other(side)]
        if not any(rp[r] > 0.2 for r in e_open):
            return best
        for o, ri in ctx.roles[side].items():
            n = self.snap.lane_games(c, o, ri)
            if ri in e_open and rp[ri] > 0.2 and n >= min_games:
                v = self.snap.counter(c, o, ri)
                if v > best[0]:
                    best = (v, o, n)
        return best

    def _recommend_ban(self, state, side, k):
        enemy = other(side)
        own_cands, own_p, _, ctx = self.action_probs(state, side, "ban")
        # Threat: how likely the enemy would pick each champion if it were their pick now.
        e_cands, e_p, _, _ = self.action_probs(state, enemy, "pick")
        threat = dict(zip(e_cands, e_p))
        habit = dict(zip(own_cands, own_p))
        # Damage: how much our win chance would drop if the enemy got it. Used to rank only, not shown as a
        # number, because the draft's effect on win chance is not significant in backtests.
        sign = 1 if side == "Blue" else -1
        base = self.win_prob(state, ctx)
        # Possible lane counters to OUR picks get a floor (threat >= general pick rate, damage >= half the
        # counter value), but only with 30+ games and an open enemy lane.
        counters = {c: self._counter_to_ours(state, side, ctx, c) for c in own_cands}
        pre = sorted(own_cands, key=lambda c: -(0.6 * threat.get(c, 0.0) + 0.4 * habit[c]))[:30]
        pre += [c for c, v in counters.items() if v[0] >= 0.04 and c not in pre]
        scored = []
        for c in pre:
            t = threat.get(c, 0.0)
            dmg = sign * (base - self.win_prob(_WithPick(state, enemy, c)))
            cv, victim, n = counters[c]
            t_adj, d_adj = t, dmg
            if cv >= 0.04:
                t_adj, d_adj = max(t, self.snap.pick_rate(c)), max(dmg, 0.5 * cv)
            score = t_adj * max(d_adj, 0.0) + 0.002 * habit[c] / max(own_p.max(), 1e-9)
            scored.append((score, c, t, dmg, cv, victim, n))
        scored.sort(key=lambda x: -x[0])
        recs = []
        for score, c, t, dmg, cv, victim, n in scored[:k]:
            reasons = []
            headline = f"They'd likely pick it ({t:.0%})" if t >= 0.05 else "Strong for them if left open"
            if cv >= 0.04:
                headline = f"Possible counter to your {victim}"
                reasons.append(f"Tends to beat {victim} in lane ({cv:+.0%}, ~{n:.0f} games, could be chance)")
            reasons.append(f"Chance they'd pick it now: {t:.0%}")
            rp = self.snap.role_p(c)
            roster = state.roster(enemy)
            best = max(ROLES, key=lambda r: rp[ROLES.index(r)] * self.snap.player_games_on(roster.get(r), c))
            g = self.snap.player_games_on(roster.get(best), c)
            if g >= 0.5:
                reasons.append(f"{self.name(roster.get(best))} plays it: ~{g:.0f} recent pro games, "
                               f"won {self.snap.player_wr_on(roster.get(best), c):.0%}")
                if g >= 5 and cv < 0.04:
                    headline = f"{self.name(roster.get(best))}'s champion (~{g:.0f} recent games)"
            tp = self.snap.team_pick_count(state.team(enemy), c)
            if tp >= 0.5:
                reasons.append(f"{state.team(enemy)} picked it ~{tp:.0f} times recently")
            e_picks = state.picks(enemy)
            if e_picks:
                pal = max(e_picks, key=lambda e: self.snap.copick_lift(c, e))
                if self.snap.copick_lift(c, pal) >= 0.26:
                    reasons.append(f"Pros often pair it with their {pal}")
            o_picks = state.picks(side)
            if o_picks:
                tgt = max(o_picks, key=lambda o: self.snap.vspick_lift(c, o))
                if self.snap.vspick_lift(c, tgt) >= 0.26:
                    reasons.append(f"Pros often pick it into your {tgt}")
            reasons.append(f"Picked or banned in {self.snap.presence(c):.0%} of recent pro games")
            recs.append({"champion": c, "score": round(float(score), 5), "model_prob": round(float(habit[c]), 4),
                         "threat": round(float(t), 4), "damage": round(float(dmg), 4), "headline": headline,
                         "reasons": reasons, "warnings": []})
        return recs

    # ------------------------------------------------------------ setup data
    def teams(self, active_days: int = 180):
        """Teams that played within `active_days` of the most recent game in the data."""
        snap = self.snap
        latest = max(snap.team_last.values())
        out = []
        for team, roster in snap.team_roster.items():
            if (latest - snap.team_last[team]).days > active_days:
                continue
            out.append({
                "name": team, "league": self.league["home"].get(team) or snap.team_league.get(team),
                "elo": round(self.team_elo(team)),
                "last_game": str(snap.team_last[team].date()),
                "roster": {r: {"key": p, "name": self.name(p)} for r, p in roster.items()},
            })
        out.sort(key=lambda t: (t["league"] or "", -t["elo"]))
        return out

    def players(self):
        snap = self.snap
        out = []
        for p, g in snap.player_games.items():
            out.append({"key": p, "name": self.name(p), "role": snap.player_main_role(p), "games": round(g, 1)})
        out.sort(key=lambda x: x["name"].lower())
        return out

    def champion_info(self):
        return [{"name": c, "roles": [r for r, v in zip(ROLES, self.snap.role_p(c)) if v > 0.1],
                 "presence": round(self.snap.presence(c), 3)} for c in self.champions]
