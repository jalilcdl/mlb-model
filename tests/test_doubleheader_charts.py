"""
Regression tests for the doubleheader bug.

WHAT WENT WRONG (2026-07-22, real money on the line)
BAL@BOS was a doubleheader. The charts keyed rows on (away_team, home_team)
and kept only the first match, so both PDFs showed ONE row for the matchup.
A reader could not tell there were two games, which one the row described, or
that a second existed. The two games had different starters -- Kremer/Bennett
in the opener, Bradish/Rivera in the nightcap -- and therefore genuinely
different win probabilities: BOS 61.4% in G1 versus 53.9% in G2. Against the
market that flipped the recommendation's sign, from +4.2 points of edge to
-2.6. Someone read the single row as "tonight's game" and bet the wrong number.

A second, subtler bug surfaced while fixing the first: _select_event returned
a lone candidate WITHOUT checking its start time. Books pull the opener's line
once it starts, so a doubleheader routinely has exactly one live odds event --
and the nightcap's line was being pinned onto the completed opener.

    python -m pytest tests/test_doubleheader_charts.py -q
"""
import pandas as pd
import pytest

from src.odds import odds_adapter
from src.reports import play_chart


def _pred(game_pk, utc, home_sp="Home SP", away_sp="Away SP", status="Scheduled",
          home_wp=0.60):
    return {
        "game_pk": game_pk, "game_datetime_utc": utc, "status": status,
        "away_team": "BAL", "home_team": "BOS",
        "home_probable_pitcher": home_sp, "away_probable_pitcher": away_sp,
        "home_win_prob": home_wp, "away_win_prob": 1 - home_wp,
        "expected_home_runs": 4.8, "expected_away_runs": 4.2, "overdispersion": 2.25,
    }


def _ml_event(start, home_ml=-145, away_ml=125):
    return {
        "away_team": "BAL", "home_team": "BOS", "start_time_utc": start,
        "consensus_home_ml": home_ml, "consensus_away_ml": away_ml,
        "best_home_ml": home_ml, "best_away_ml": away_ml,
    }


G1_UTC, G2_UTC = "2026-07-22T17:35:00Z", "2026-07-22T23:10:00Z"


def test_doubleheader_yields_a_row_per_game_not_one_merged_row():
    """The core bug: two games must produce four moneyline rows, not two."""
    preds = pd.DataFrame([
        _pred(824735, G1_UTC, home_wp=0.6135),
        _pred(824732, G2_UTC, home_wp=0.5391),
    ])
    rows, _ = play_chart.build_rows(preds, [_ml_event(G1_UTC), _ml_event(G2_UTC)], "2026-07-22")

    assert len(rows) == 4, "both games of a doubleheader must render"
    assert {0.6135, 0.5391} <= set(rows["model"].round(4)), \
        "each game must carry its OWN probability, not a shared one"


def test_each_doubleheader_row_is_labelled_with_game_number_and_start_time():
    """A reader must be able to tell which game a row is without outside info."""
    preds = pd.DataFrame([_pred(824735, G1_UTC), _pred(824732, G2_UTC)])
    rows, _ = play_chart.build_rows(preds, [_ml_event(G1_UTC), _ml_event(G2_UTC)], "2026-07-22")

    labels = " ".join(rows["label"])
    assert "(G1, 1:35pm ET)" in labels
    assert "(G2, 7:10pm ET)" in labels


def test_single_games_are_not_given_a_game_number_suffix():
    """The tag is doubleheader-only; ordinary games stay clean."""
    preds = pd.DataFrame([_pred(824735, G1_UTC)])
    rows, _ = play_chart.build_rows(preds, [_ml_event(G1_UTC)], "2026-07-22")
    assert all("(G1" not in lbl for lbl in rows["label"])


def test_lone_odds_event_is_not_pinned_onto_the_other_half_of_a_doubleheader():
    """The second bug. Only the nightcap has a line (the opener's was pulled).
    The opener must be dropped, NOT silently given the nightcap's price."""
    preds = pd.DataFrame([_pred(824735, G1_UTC), _pred(824732, G2_UTC)])
    rows, unmatched = play_chart.build_rows(preds, [_ml_event(G2_UTC)], "2026-07-22")

    assert len(rows) == 2, "only the game with real odds may chart"
    assert all("G2" in lbl for lbl in rows["label"]), "the charted game must be the nightcap"
    assert any("G1" in u for u in unmatched), "the dropped opener must be reported, not vanish"


def test_select_event_requires_a_time_match_when_told_to():
    """Unit-level guard on the shortcut that caused it."""
    row = {"game_datetime_utc": G1_UTC}
    only_nightcap = [_ml_event(G2_UTC)]

    assert odds_adapter._select_event(only_nightcap, row) is not None, \
        "default behaviour (single game) is unchanged"
    assert odds_adapter._select_event(only_nightcap, row, require_time=True) is None, \
        "a 5.5-hour gap is far outside tolerance and must not match"


def test_unannounced_starter_is_flagged_rather_than_hidden():
    """A TBD starter is modelled at league average; the chart must say so."""
    preds = pd.DataFrame([_pred(824732, G2_UTC, home_sp=None)])
    rows, _ = play_chart.build_rows(preds, [_ml_event(G2_UTC)], "2026-07-22")
    assert all("*" in lbl for lbl in rows["label"])


def test_completed_game_is_flagged_as_not_bettable():
    preds = pd.DataFrame([_pred(824735, G1_UTC, status="Final")])
    rows, _ = play_chart.build_rows(preds, [_ml_event(G1_UTC)], "2026-07-22")
    assert all("[FINAL]" in lbl for lbl in rows["label"])


def test_excluded_games_are_named_on_the_chart_footnote():
    """A game absent from the chart must be visible as absent."""
    assert "NOT SHOWN (1)" in play_chart._excluded_note(["BAL@BOS (G1, 1:35pm ET)"])
    assert play_chart._excluded_note([]) == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
