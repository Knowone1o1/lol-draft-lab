"""Feature construction for draft actions (pick / ban candidates) and team compositions."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .config import ROLES, SIDES
from .data import DRAFT_ORDER, draft_order, other
from .stats import Snapshot, log1p

PICK_FEATURES = [
    "meta_presence",      # log presence (pick+ban rate)
    "champ_strength",     # shrunk win rate - 0.5
    "role_fit",           # log P(best open role | champ)
    "player_comfort",     # log(1 + player games on champ), player in that role
    "player_share",       # share of player's games on champ
    "team_habit",         # log(1 + team picks of champ)
    "synergy",            # mean synergy with own picks
    "lane_counter",       # counter vs enemy laner already picked in that role
    "flex",               # log number of viable roles (flex picks go early)
    "flex_x_early",       # flex value matters more in early picks
    "comfort_x_late",     # counter-pick / comfort pick matters more late
    "copick_max",         # strongest pick pattern with an ally (pros pair them, e.g. Ori + Noc)
    "copick_sum",         # total pairing pull from all allies
    "vspick_max",         # strongest "picked into" pattern vs an enemy pick (e.g. Syndra into LeBlanc)
    "synergy_max",        # best win-rate lift with a single ally (e.g. Ori + J4)
    "counter_max",        # best win-rate lift vs any enemy pick in the same lane
    "team_counter_max",   # best win-rate lift vs any enemy champion, any role (Sylas vs Malphite)
    "team_counter_sum",   # total win-rate lift vs the whole enemy team so far
    "duo_copick",         # bot/support only: how much pros pair it with our already-picked duo partner
    "duo_synergy",        # bot/support only: win-rate lift of that bot-lane duo
]

BAN_FEATURES = [
    "meta_presence",
    "ban_rate",           # log ban rate
    "champ_strength",
    "enemy_comfort",      # best enemy player comfort on champ (open roles)
    "enemy_habit",        # log(1 + enemy team picks of champ)
    "enemy_role_fit",     # log P(enemy open role | champ)
    "own_comfort",        # own players' comfort (should be negative: don't ban your own champs)
    "own_ban_habit",      # log(1 + this team's bans of champ)
    "counters_own",       # mean lane-counter of champ vs our already picked champs (phase 2)
    "enemy_copick_max",   # pairs with something the enemy already has (they have Noc -> ban Ori)
    "into_us_max",        # pros pick it into one of our picks (ban the answer)
    "enemy_synergy_max",  # win-rate lift with an enemy pick
    "beats_us_max",       # win-rate lift vs one of our picks, any role (we have Malphite -> ban Sylas)
    "enemy_duo_copick",   # pros pair it with the enemy's already-picked bot/support (ban the duo partner)
    "pick_rate",          # log pick rate (with ban_rate, a non-collinear split of presence; judge M9)
]

WIN_FEATURES = ["elo_diff", "champ_strength", "player_comfort", "synergy", "lane_counter", "team_matchups"]


@dataclass
class DraftState:
    blue_team: str
    red_team: str
    blue_roster: dict            # role -> player key
    red_roster: dict
    actions: list = field(default_factory=list)   # champions (or None) in DRAFT_ORDER order
    locked: frozenset = frozenset()               # fearless: unavailable from earlier games
    blue_first: bool = True                       # 2026 First Selection: False when red picks first

    @property
    def order(self) -> list:
        return draft_order(self.blue_first)

    def team(self, side):
        return self.blue_team if side == "Blue" else self.red_team

    def roster(self, side):
        return self.blue_roster if side == "Blue" else self.red_roster

    def picks(self, side) -> list:
        return [c for (s, k, _), c in zip(self.order, self.actions) if s == side and k == "pick" and c]

    def bans(self, side) -> list:
        return [c for (s, k, _), c in zip(self.order, self.actions) if s == side and k == "ban" and c]

    def unavailable(self) -> set:
        return {c for c in self.actions if c} | set(self.locked)

    def next_action(self):
        return self.order[len(self.actions)] if len(self.actions) < len(self.order) else None

    def pick_number(self) -> int:
        """How many picks have already happened (0..9) — the draft 'stage'."""
        return sum(1 for (s, k, _), c in zip(self.order, self.actions) if k == "pick")


def assign_roles(champs: list, snap: Snapshot, fixed: dict | None = None) -> dict:
    """Most likely role per champion (champ -> role index), each role used once."""
    if fixed:
        return dict(fixed)
    if not champs:
        return {}
    logs = {c: np.log(snap.role_p(c) + 1e-4) for c in champs}
    best, best_s = None, -1e18
    for perm in itertools.permutations(range(5), len(champs)):
        s = sum(logs[c][r] for c, r in zip(champs, perm))
        if s > best_s:
            best, best_s = perm, s
    return dict(zip(champs, best))


class StateContext:
    """Per-state precomputation shared by all candidates."""

    def __init__(self, state: DraftState, snap: Snapshot, role_hint: dict | None = None):
        self.state = state
        self.snap = snap
        self.roles = {}
        self.open = {}
        self.by_role = {}
        for s in SIDES:
            picks = state.picks(s)
            fixed = None
            if role_hint and s in role_hint:
                fixed = {c: role_hint[s][c] for c in picks if c in role_hint[s]}
                if len(fixed) != len(picks):
                    fixed = None
            r = assign_roles(picks, snap, fixed)
            self.roles[s] = r
            self.open[s] = [i for i in range(5) if i not in r.values()]
            self.by_role[s] = {ri: c for c, ri in r.items()}

    def best_role(self, side, c):
        rp = self.snap.role_p(c)
        opens = self.open[side]
        if not opens:
            return 0.0, None
        ri = max(opens, key=lambda i: rp[i])
        return float(rp[ri]), ri


def pick_matrix(state: DraftState, side: str, cands: list, snap: Snapshot, ctx: StateContext | None = None):
    ctx = ctx or StateContext(state, snap)
    enemy = other(side)
    own = state.picks(side)
    roster = state.roster(side)
    team = state.team(side)
    enemy_picks = state.picks(enemy)
    stage = state.pick_number() / 9.0
    X = np.zeros((len(cands), len(PICK_FEATURES)), dtype=np.float32)
    for i, c in enumerate(cands):
        fit, ri = ctx.best_role(side, c)
        player = roster.get(ROLES[ri]) if ri is not None else None
        rp = snap.role_p(c)
        flex = max(1, int((rp > 0.15).sum()))
        comfort = log1p(snap.player_games_on(player, c))
        syns = [snap.syn(c, a) for a in own]
        syn = float(np.mean(syns)) if syns else 0.0
        opp = ctx.by_role[enemy].get(ri) if ri is not None else None
        lane = snap.counter(c, opp, ri) if opp else 0.0
        cps = [snap.copick_lift(c, a) for a in own]
        vss = [snap.vspick_lift(c, e) for e in enemy_picks]
        # Best lane counter over every open role c can realistically play (flex counter-picks).
        cms = [snap.counter(c, ctx.by_role[enemy][r], r) for r in ctx.open[side]
               if rp[r] > 0.1 and r in ctx.by_role[enemy]]
        tcs = [snap.vs_team(c, e) for e in enemy_picks]
        duo_cp = duo_syn = 0.0
        if ri is not None and ROLES[ri] in ("bot", "sup"):
            partner = ctx.by_role[side].get(ROLES.index("sup" if ROLES[ri] == "bot" else "bot"))
            if partner:
                duo_cp, duo_syn = snap.copick_lift(c, partner), snap.syn(c, partner)
        X[i] = (
            math.log(snap.presence(c) + 1e-3),
            snap.wr(c) - 0.5,
            math.log(fit + 1e-3),
            comfort,
            snap.player_share(player, c),
            log1p(snap.team_pick_count(team, c)),
            syn,
            lane,
            math.log(flex),
            math.log(flex) * (1 - stage),
            comfort * stage,
            max(cps) if cps else 0.0,
            sum(cps),
            max(vss) if vss else 0.0,
            max(syns) if syns else 0.0,
            max(cms) if cms else 0.0,
            max(tcs) if tcs else 0.0,
            sum(tcs),
            duo_cp,
            duo_syn,
        )
    return X


def ban_matrix(state: DraftState, side: str, cands: list, snap: Snapshot, ctx: StateContext | None = None):
    ctx = ctx or StateContext(state, snap)
    enemy = other(side)
    own_roster, enemy_roster = state.roster(side), state.roster(enemy)
    own_team, enemy_team = state.team(side), state.team(enemy)
    X = np.zeros((len(cands), len(BAN_FEATURES)), dtype=np.float32)
    for i, c in enumerate(cands):
        rp = snap.role_p(c)
        e_open = ctx.open[enemy] or list(range(5))
        o_open = ctx.open[side] or list(range(5))
        e_comf = max(rp[r] * log1p(snap.player_games_on(enemy_roster.get(ROLES[r]), c)) for r in e_open)
        o_comf = max(rp[r] * log1p(snap.player_games_on(own_roster.get(ROLES[r]), c)) for r in o_open)
        e_fit = max(rp[r] for r in e_open)
        cnt = [snap.counter(c, a, ri) for a, ri in ctx.roles[side].items()]
        e_picks, o_picks = state.picks(enemy), state.picks(side)
        e_cp = [snap.copick_lift(c, e) for e in e_picks]
        into = [snap.vspick_lift(c, o) for o in o_picks]
        e_syn = [snap.syn(c, e) for e in e_picks]
        beats = [snap.vs_team(c, o) for o in o_picks]
        e_duo = 0.0
        for duo_role, partner_role in (("bot", "sup"), ("sup", "bot")):
            ri_d = ROLES.index(duo_role)
            partner = ctx.by_role[enemy].get(ROLES.index(partner_role))
            if partner and ri_d in ctx.open[enemy] and rp[ri_d] > 0.2:
                e_duo = max(e_duo, snap.copick_lift(c, partner))
        X[i] = (
            math.log(snap.presence(c) + 1e-3),
            math.log(snap.ban_rate(c) + 1e-3),
            snap.wr(c) - 0.5,
            e_comf,
            log1p(snap.team_pick_count(enemy_team, c)),
            math.log(e_fit + 1e-3),
            o_comf,
            log1p(snap.team_ban_count(own_team, c)),
            float(np.mean(cnt)) if cnt else 0.0,
            max(e_cp) if e_cp else 0.0,
            max(into) if into else 0.0,
            max(e_syn) if e_syn else 0.0,
            max(beats) if beats else 0.0,
            e_duo,
            math.log(snap.pick_rate(c) + 1e-3),
        )
    return X


def comp_features(state: DraftState, snap: Snapshot, blue_elo: float, red_elo: float,
                  role_hint: dict | None = None, ctx: StateContext | None = None) -> np.ndarray:
    """Blue-minus-red composition features. Sums are divided by 5 so partial drafts scale smoothly."""
    ctx = ctx or StateContext(state, snap, role_hint)
    vals = {}
    for s in SIDES:
        picks = state.picks(s)
        roster = state.roster(s)
        strength = sum(snap.wr(c) - 0.5 for c in picks) / 5
        comfort = sum(log1p(snap.player_games_on(roster.get(ROLES[ctx.roles[s][c]]), c)) for c in picks) / 5
        syn = sum(snap.syn(a, b) for a, b in itertools.combinations(picks, 2)) / 10
        vals[s] = (strength, comfort, syn)
    lane = 0.0
    for ri in range(5):
        b, r = ctx.by_role["Blue"].get(ri), ctx.by_role["Red"].get(ri)
        if b and r:
            lane += snap.counter(b, r, ri)
    lane /= 5
    # Whole-team matchups: every blue champion vs every red champion (25 pairs), blue perspective.
    bp, rp_ = state.picks("Blue"), state.picks("Red")
    team_m = sum(snap.vs_team(b, r) for b in bp for r in rp_) / 25
    return np.array([
        (blue_elo - red_elo) / 400.0,
        vals["Blue"][0] - vals["Red"][0],
        vals["Blue"][1] - vals["Red"][1],
        vals["Blue"][2] - vals["Red"][2],
        lane,
        team_m,
    ], dtype=np.float64)


def state_from_game(row: dict, n_actions: int = 0) -> DraftState:
    from .data import blue_first_of, game_actions
    acts = [c for _, _, c in game_actions(row)][:n_actions]
    return DraftState(
        blue_team=row["Blue_teamname"], red_team=row["Red_teamname"],
        blue_roster={r: row[f"Blue_{r}_player"] for r in ROLES},
        red_roster={r: row[f"Red_{r}_player"] for r in ROLES},
        actions=acts, locked=row.get("locked") or frozenset(), blue_first=blue_first_of(row),
    )


def actual_roles(row: dict) -> dict:
    return {s: {row[f"{s}_{r}_champ"]: i for i, r in enumerate(ROLES)} for s in SIDES}
