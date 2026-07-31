"""
Game-detail view model: everything the BetQL-style game page renders, computed
from this project's real model outputs.

Design rule for this module: every number shown must trace to something we
actually compute or fetch. Where BetQL shows data we do not have, we either cut
the element or replace it with an honestly-labelled equivalent. Specifically:

  - NO first-half / first-5-innings projections. The Monte Carlo engine samples
    whole-game run totals; it has no inning structure, so any "first half"
    number would be invented.
  - NO sharp-money or ticket percentages. That is licensed sportsbook flow data
    which none of our free sources provide. `model_diagnostics()` replaces that
    panel with our own internal-agreement metrics, and the UI labels it as such.
  - Letter grades are derived from real ratings with the methodology documented
    on each function, not assigned by feel.

The star rating deliberately does NOT mean "bigger edge = better". Backtesting
found our largest model-vs-market disagreements are usually artifacts (a clamped
pitcher factor, a clipped park), and the one live bet that lost worst came from
that quadrant. So the rating peaks in the credible band and is *downgraded* for
implausibly large edges and for red-flag diagnostics.
"""
import numpy as np
import pandas as pd

from src import config

# Demonstrated out-of-sample moneyline edge over a naive baseline: 55.5% vs
# 53.0% (see README Backtest). Everything about "is this edge plausible?" is
# scaled against this, not against a marketing scale.
DEMONSTRATED_EDGE_PTS = 2.7

MARKET_VALIDATION = {
    "moneyline": ("VALIDATED", "55.5% vs 53.0% naive baseline, ECE 2.1% (6,273 games)"),
    "total": ("NO EDGE", "50.25% ATS vs 52.38% break-even on 11,706 real closing lines; "
                          "recalibration did not fix it (AUC 0.503)"),
    "run_line": ("NO EDGE", "64.5% vs a 64.4% always-take-the-underdog baseline"),
}


# --------------------------------------------------------------------------
# Team logos (ESPN public CDN)
# --------------------------------------------------------------------------
_LOGO_URL = "https://a.espncdn.com/i/teamlogos/mlb/500/scoreboard/{}.png"

# ESPN's slug matches our lowercased code for 29 of 30 clubs; the White Sox are
# "chw" there vs our canonical "CWS". Verified all 30 return HTTP 200 image/png.
_LOGO_SLUG_OVERRIDES = {"CWS": "chw"}


def logo_url(team_code):
    """Public ESPN CDN logo for a canonical team code, or None if unmapped.

    Decoration only -- nothing downstream depends on it, and Streamlit simply
    renders no image if the URL ever stops resolving, so a CDN change degrades
    the header rather than breaking the page.
    """
    if not team_code:
        return None
    slug = _LOGO_SLUG_OVERRIDES.get(team_code, str(team_code).lower())
    return _LOGO_URL.format(slug)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
def team_records(games, season=None):
    """Real W-L per team from the game log. Ties are impossible in MLB."""
    season = season or config.CURRENT_SEASON
    cur = games[games["season"] == season]
    rec = {}
    for g in cur.itertuples(index=False):
        home_won = g.home_score > g.away_score
        rec.setdefault(g.home_team, [0, 0])
        rec.setdefault(g.away_team, [0, 0])
        rec[g.home_team][0 if home_won else 1] += 1
        rec[g.away_team][1 if home_won else 0] += 1
    return {t: (w, l) for t, (w, l) in rec.items()}


def format_record(rec, team):
    w, l = rec.get(team, (0, 0))
    return f"{w}-{l}"


# --------------------------------------------------------------------------
# Grades
# --------------------------------------------------------------------------
_QUINTILE_LETTERS = ["F", "D", "C", "B", "A"]


def percentile_grade(value, population, higher_is_better=True):
    """Letter grade by quintile rank within the 30-team population.

    Quintiles (6 teams per band in a 30-team league) rather than absolute
    thresholds, because the ratings are relative measures whose spread changes
    through a season. Returns (letter, percentile 0-100).
    """
    pop = np.asarray(list(population), dtype=float)
    if len(pop) == 0 or not np.isfinite(value):
        return "N/A", None
    pct = float((pop < value).mean() * 100)
    if not higher_is_better:
        pct = 100 - pct
    idx = min(int(pct // 20), 4)
    return _QUINTILE_LETTERS[idx], pct


def pitcher_grade(factor, innings_pitched=None):
    """Grade a starting pitcher from their run-prevention factor.

    The factor is already a ratio to league average (1.00 = average, lower is
    better), so fixed bands are meaningful here without needing a population.
    Returns (letter, list_of_caveats) -- caveats matter more than the letter:
    this component has NO validated predictive lift (see the pitcher-isolation
    backtest), and the model clamps extreme values at 0.50/1.80, so a grade
    sitting on a clamp is the model extrapolating past its own guardrail.
    """
    caveats = []
    if factor is None or (isinstance(factor, float) and not np.isfinite(factor)):
        return "N/A", ["No probable starter announced, or under the "
                       f"{config.PITCHER_MIN_IP_FOR_ADJUSTMENT}-IP minimum — "
                       "no pitcher adjustment applied; team pitching used instead."]
    f = float(factor)
    if f <= 0.501:
        caveats.append("Factor is CLAMPED at the 0.50 floor — model is extrapolating past its guardrail.")
    if f >= 1.799:
        caveats.append("Factor is CLAMPED at the 1.80 ceiling — model is extrapolating past its guardrail.")
    if innings_pitched is not None and innings_pitched < 40:
        caveats.append(f"Only {innings_pitched:.0f} IP this season — small sample, expect regression.")
    caveats.append("Pitcher adjustment is UNVALIDATED: isolation backtest found no significant "
                   "improvement in any market (all CIs cross zero).")

    if f <= 0.80:
        letter = "A"
    elif f <= 0.92:
        letter = "B"
    elif f <= 1.08:
        letter = "C"
    elif f <= 1.20:
        letter = "D"
    else:
        letter = "F"
    return letter, caveats


FIELDING_CAVEATS = [
    "Savant OAA counts only QUALIFIED fielders (~8 per club vs 13+ who actually "
    "field), so team totals undercount; players who changed teams are dropped entirely.",
    "OAA is SEASON-TO-DATE with no rolling window, so unlike the run ratings it "
    "cannot be recency-weighted -- an early-season stretch stays baked in all year.",
]

PITCHING_CAVEATS = [
    "Graded on FIP (HR, BB, HBP, K only), which is defense-independent by "
    "construction -- it deliberately ignores what happens once a ball is in play, "
    "so it will disagree with ERA when a club's defense or luck is unusual.",
    "Park-normalized by each club's actual schedule exposure, since FIP is "
    "park-influenced through home runs.",
]


def pitching_grade(team, pitching_df, exposure=None):
    """Letter grade from park-adjusted team FIP (lower is better)."""
    if pitching_df is None or pitching_df.empty or team not in set(pitching_df["team"]):
        return "N/A", None, ["No team pitching stats available."]
    exposure = exposure or {}
    adj = pitching_df.set_index("team")["FIP"] / pitching_df.set_index("team").index.map(
        lambda t: exposure.get(t, 1.0))
    letter, pct = percentile_grade(adj.get(team), adj.values, higher_is_better=False)
    return letter, float(adj.get(team)), list(PITCHING_CAVEATS)


def fielding_grade(team, fielding_df):
    """Letter grade from Savant fielding runs prevented (higher is better)."""
    if fielding_df is None or fielding_df.empty or "runs_prevented" not in fielding_df:
        return "N/A", None, ["No fielding data available."]
    src = fielding_df.dropna(subset=["runs_prevented"])
    if team not in set(src["team"]):
        return "N/A", None, ["No qualified fielders for this club in the Savant sample."]
    vals = src.set_index("team")["runs_prevented"]
    letter, pct = percentile_grade(vals.get(team), vals.values, higher_is_better=True)
    return letter, float(vals.get(team)), list(FIELDING_CAVEATS)


def team_grades(team, elo, ratings, pitching_df=None, fielding_df=None, exposure=None):
    """Team / Offense / Run-prevention letter grades from real model ratings.

    Offense and run-prevention use the PARK-ADJUSTED ratings
    (`off_rating_pa` / `def_rating_pa`): each game's runs are divided by that
    venue's park factor before averaging. Without this, half of Colorado's
    schedule at Coors manufactured an inflated offense grade and a deflated
    run-prevention grade purely from altitude, and the reverse for pitcher-park
    teams -- the grade spread was substantially a stadium artifact.

    Falls back to the raw ratings if a caller passes an older TeamRunRatings
    instance that predates the park-adjusted fields.
    """
    off_src = getattr(ratings, "off_rating_pa", None) or ratings.off_rating
    def_src = getattr(ratings, "def_rating_pa", None) or ratings.def_rating

    elo_pop = [elo.rating(t) for t in ratings.off_rating]
    team_letter, team_pct = percentile_grade(elo.rating(team), elo_pop, higher_is_better=True)
    off_letter, off_pct = percentile_grade(
        off_src.get(team, 1.0), off_src.values(), higher_is_better=True)
    # Run-prevention rating is runs ALLOWED relative to average: lower is better.
    pit_letter, pit_pct = percentile_grade(
        def_src.get(team, 1.0), def_src.values(), higher_is_better=False)
    pit_grade, fip_val, pit_cav = pitching_grade(team, pitching_df, exposure)
    fld_grade, fld_val, fld_cav = fielding_grade(team, fielding_df)

    return {
        "team": {"grade": team_letter, "percentile": team_pct, "elo": elo.rating(team)},
        "offense": {"grade": off_letter, "percentile": off_pct,
                    "rating": off_src.get(team), "park_adjusted": True},
        # Runs actually allowed: pitching AND defense AND luck combined. Kept
        # under an honest label rather than being called "pitching".
        "run_prevention": {"grade": pit_letter, "percentile": pit_pct,
                           "rating": def_src.get(team), "park_adjusted": True},
        # Pitching proper: defense-independent (FIP).
        "pitching": {"grade": pit_grade, "fip_park_adj": fip_val, "caveats": pit_cav},
        # Defense proper: Savant runs prevented.
        "fielding": {"grade": fld_grade, "runs_prevented": fld_val, "caveats": fld_cav},
    }


# --------------------------------------------------------------------------
# Rating / recommendation
# --------------------------------------------------------------------------
def star_rating(edge_pts, market, diagnostics=None):
    """Translate an edge into a 1-5 actionability rating.

    NOT a raw edge ranking. Peaks in the credible band around our demonstrated
    ~2.7pt skill and falls off for implausibly large edges, because backtesting
    showed the biggest disagreements are usually model artifacts rather than
    market mistakes. Red-flag diagnostics and unvalidated markets deduct.
    """
    diagnostics = diagnostics or {}
    e = float(edge_pts)

    if e < 1.0:
        stars, note = 1, "Edge below noise — no play."
    elif e < DEMONSTRATED_EDGE_PTS:
        stars, note = 2, f"Edge under our demonstrated {DEMONSTRATED_EDGE_PTS}pt skill — thin."
    elif e <= 5.0:
        stars, note = 4, "Edge in the credible band (at or modestly above demonstrated skill)."
    elif e <= 8.0:
        stars, note = 3, "Edge is ~2x our demonstrated skill — plausible but claiming a lot."
    else:
        stars, note = 2, ("Edge is implausibly large (>3x demonstrated skill). Historically this "
                          "pattern is a model artifact, not market error.")

    penalties = []
    if diagnostics.get("pitcher_clamped"):
        penalties.append("clamped pitcher factor")
    if diagnostics.get("park_clipped"):
        penalties.append("park factor at clip ceiling")
    if diagnostics.get("elo_mc_gap_pts", 0) >= 15:
        penalties.append(f"Elo/sim disagree by {diagnostics['elo_mc_gap_pts']:.0f}pts")
    if penalties:
        stars = max(1, stars - 1)
        note += " Downgraded: " + ", ".join(penalties) + "."

    status, _ = MARKET_VALIDATION.get(market, ("UNKNOWN", ""))
    if status == "NO EDGE":
        stars = min(stars, 2)
        note += " Capped: this market has no validated edge."
    return stars, note


def edge_label(edge_pts):
    """Compact machine-readable verdict for an edge, shared by the dashboard and
    the JSON export so the two can never disagree."""
    e = float(edge_pts)
    if e < 1.0:
        return "no play"
    if e < DEMONSTRATED_EDGE_PTS:
        return "thin"
    if e <= 5.0:
        return "playable"
    if e <= 8.0:
        return "sizeable"
    return "suspicious"


def odds_comparability(status):
    """Whether a pre-game model probability can honestly be compared to the
    currently-posted price.

    Our simulation is a PRE-GAME model: it knows the starters and team ratings
    but nothing about the score. Once a game is underway the market price has
    absorbed live information the model has not, so subtracting one from the
    other produces a meaningless "edge". Postponed games aren't playable at all.
    """
    s = (status or "").strip().lower()
    if "postpon" in s or "suspend" in s or "cancel" in s:
        return False, f"Game is {status} — not playable."
    pregame = ("scheduled", "pre-game", "pregame", "warmup", "delayed start")
    if any(p in s for p in pregame):
        return True, None
    if "final" in s or "game over" in s:
        return False, f"Game is {status} — market closed; edge is historical only."
    return False, (f"Game is in progress ({status}). The model is PRE-GAME only and does not know "
                   "the score, so its probability is not comparable to the live price — "
                   "any 'edge' shown here is invalid.")


def recommendation_text(team, model_prob, market_prob, best_odds, best_book, market):
    """Plain-English advice, phrased around what the numbers actually support."""
    from src.odds.odds_adapter import prob_to_american
    fair = prob_to_american(model_prob)
    edge = (model_prob - market_prob) * 100
    fair_s = f"{fair:+d}" if fair is not None else "n/a"
    price_s = f"{int(best_odds):+d}" if best_odds is not None else "n/a"

    lead = (f"Model makes **{team}** {fair_s} ({model_prob*100:.1f}%). "
            f"Best available is **{price_s}**"
            + (f" at {best_book}" if best_book else "")
            + f" ({market_prob*100:.1f}% implied, de-vigged) — a **{edge:+.1f} point** difference.")

    if edge < 1.0:
        verdict = "No play. The difference is inside noise."
    elif edge < DEMONSTRATED_EDGE_PTS:
        verdict = (f"Thin. Below the ~{DEMONSTRATED_EDGE_PTS}pt edge we've actually demonstrated "
                   "out-of-sample, so treat it as a lean at most.")
    elif edge <= 5.0:
        verdict = "Playable. This is the band where our measured skill actually lives."
    elif edge <= 8.0:
        verdict = "Sizeable — roughly double our demonstrated skill. Check the diagnostics before acting."
    else:
        verdict = ("Treat with suspicion. Edges this large have historically come from model "
                   "artifacts, not market error — the market usually knows something we don't.")

    status, evidence = MARKET_VALIDATION.get(market, ("UNKNOWN", ""))
    if status == "NO EDGE":
        verdict += f" ⚠️ This market has NO validated edge ({evidence})."
    return lead, verdict


# --------------------------------------------------------------------------
# Diagnostics (replaces BetQL's licensed money-flow panel)
# --------------------------------------------------------------------------
def model_diagnostics(g, ratings):
    """Our own internal-agreement metrics. This is NOT market money-flow data.

    BetQL's "Sharp Bettor Report" shows % of money vs % of tickets, which is
    licensed sportsbook data we do not have and will not fabricate. These are
    the checks we can actually run: do our independent components agree, and is
    any input sitting on a guardrail?
    """
    elo_p = float(g.get("elo_home_win_prob", np.nan))
    mc_p = float(g.get("mc_home_win_prob", np.nan))
    cf_p = float(g.get("cf_home_win_prob", np.nan))
    gap = abs(elo_p - mc_p) * 100 if np.isfinite(elo_p) and np.isfinite(mc_p) else np.nan

    hf, af = g.get("home_pitcher_factor"), g.get("away_pitcher_factor")
    clamped = any(pd.notna(f) and (float(f) <= 0.501 or float(f) >= 1.799) for f in (hf, af))

    park = ratings.park_factor.get(g["home_team"], 1.0)
    lo, hi = (0.85, 1.15)
    clipped = park >= hi - 1e-9 or park <= lo + 1e-9

    return {
        "elo_home_prob": elo_p,
        "mc_home_prob": mc_p,
        "cf_home_prob": cf_p,
        "elo_mc_gap_pts": gap,
        "agreement": ("strong" if gap < 5 else "moderate" if gap < 15 else "weak"),
        "pitcher_clamped": clamped,
        "park_factor": park,
        "park_clipped": clipped,
        "overdispersion": g.get("overdispersion"),
        "n_sims": g.get("n_sims"),
    }
