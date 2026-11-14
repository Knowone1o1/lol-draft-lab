import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def synthetic_games(tmp_path_factory):
    """Games table built from the synthetic Oracle's-Elixir-format CSV (LCK is fearless in 2026)."""
    import make_synthetic
    from draftlab.data import build_games, load_raw
    d = tmp_path_factory.mktemp("raw")
    make_synthetic.main(d)
    return build_games(load_raw([d / "synthetic.csv"]))


@pytest.fixture()
def db(tmp_path):
    from draftlab import store
    with store.connect(tmp_path / "test.duckdb") as con:
        yield con
