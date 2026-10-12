"""Smoke tests: every Streamlit page must execute without raising."""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = [ROOT / "app" / "main.py"] + sorted((ROOT / "app" / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_page_runs(page):
    at = AppTest.from_file(str(page), default_timeout=300).run()
    assert not at.exception, f"{page.name}: {[str(e.value) for e in at.exception]}"


def _run(page):
    return AppTest.from_file(str(ROOT / "app" / "pages" / page), default_timeout=300).run()


def test_game_explorer_all_view_modes():
    at = _run("01_Game_Explorer.py")
    assert not at.exception
    modes = at.radio[0].options
    assert set(modes) == {"Animate", "Single frame", "Movement trails"}
    for m in modes:
        at.radio[0].set_value(m).run()
        assert not at.exception, f"view mode {m}: {[str(e.value) for e in at.exception]}"


def test_game_explorer_every_game_loads():
    at = _run("01_Game_Explorer.py")
    for label in at.selectbox[0].options:
        at.selectbox[0].set_value(label).run()
        assert not at.exception, f"game {label}: {[str(e.value) for e in at.exception]}"


def test_why_page_both_modes():
    at = _run("09_Why_Did_This_Happen.py")
    for m in at.radio[0].options:
        at.radio[0].set_value(m).run()
        assert not at.exception, f"{m}: {[str(e.value) for e in at.exception]}"


def test_scouting_every_team():
    at = _run("10_Scouting_Report.py")
    for t in at.selectbox[0].options:
        at.selectbox[0].set_value(t).run()
        assert not at.exception, f"team {t}: {[str(e.value) for e in at.exception]}"
