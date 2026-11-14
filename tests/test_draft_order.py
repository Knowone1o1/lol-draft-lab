"""C0: 2026 First Selection lets red pick first; the draft sequence must follow the first-pick side."""
from draftlab.data import DRAFT_ORDER, draft_order, game_actions
from draftlab.features import state_from_game


def _row(blue_first):
    r = {"Blue_teamname": "B", "Red_teamname": "R", "blue_first_pick": 1.0 if blue_first else 0.0}
    for s in ("Blue", "Red"):
        for i in range(1, 6):
            r[f"{s}_ban{i}"] = f"{s[0]}ban{i}"
            r[f"{s}_pick{i}"] = f"{s[0]}pick{i}"
        for role in ("top", "jng", "mid", "bot", "sup"):
            r[f"{s}_{role}_player"] = f"{s}{role}"
    return r


def test_blue_first_unchanged():
    acts = game_actions(_row(True))
    assert [(s, k) for s, k, _ in acts] == [(s, k) for s, k, _ in DRAFT_ORDER]
    assert acts[0] == ("Blue", "ban", "Bban1") and acts[6] == ("Blue", "pick", "Bpick1")


def test_red_first_pick_takes_the_first_pick_sequence():
    acts = game_actions(_row(False))
    assert acts[0] == ("Red", "ban", "Rban1")            # the first-pick team also bans first
    assert acts[6] == ("Red", "pick", "Rpick1")          # first pick of the draft
    assert [c for _, _, c in acts[7:9]] == ["Bpick1", "Bpick2"]
    assert acts[19] == ("Blue", "pick", "Bpick5")         # last pick goes to the second-pick team
    assert draft_order(False)[6][0] == "Red"


def test_state_from_red_first_game():
    st = state_from_game(_row(False), 9)
    assert st.picks("Red") == ["Rpick1"] and st.picks("Blue") == ["Bpick1", "Bpick2"]
    assert st.next_action() == ("Red", "pick", 1)
    full = state_from_game(_row(False), 20)
    assert full.picks("Red") == [f"Rpick{i}" for i in range(1, 6)]
    assert full.bans("Blue") == [f"Bban{i}" for i in range(1, 6)]
