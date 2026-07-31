# MLB Prediction Model

A statistical MLB prediction model — moneyline win probability, total runs
(over/under), and run line (-1.5/+1.5) — with a local Streamlit dashboard for
daily use and real multi-sportsbook edge-finding.

Built for real analysis, not a toy: real historical data (2023-present), a
Monte Carlo simulation engine as the primary prediction method (with
closed-form math as an independent cross-check), and an honest backtest with
baseline comparisons. Read the **Limitations** section before betting
anything on it.

## Quickstart (Windows)

First time only:
```
setup.bat
```
This creates a virtual environment, installs dependencies, pulls ~4 seasons
of historical game data from Baseball-Reference (takes several minutes),
and runs an initial backtest.

Then, any time:
```
run_dashboard.bat
```
Opens the dashboard at http://localhost:8501.

To refresh the historical data later (new games played since last pull):
```
refresh_data.bat
```

(If you're not on Windows: `python -m venv venv`, `pip install -r requirements.txt`,
`python -m src.data.fetch_historical`, `python -m src.backtest.backtest`, then
`streamlit run dashboard/app.py`.)

## Deploy to Streamlit Community Cloud (free, phone-accessible URL)

The dashboard runs as a hosted website on [Streamlit Community Cloud](https://share.streamlit.io)
for free — a real URL you can open from a phone, no local machine needed. The
repo is already deploy-ready: `requirements.txt` is the exact install list,
`data/processed/games.csv` (the one data file the app hard-requires) is committed,
and the dark theme in `.streamlit/config.toml` ships with it.

**One-time deploy (you do this — it needs your GitHub + Streamlit login):**

1. Go to **https://share.streamlit.io** and sign in with the GitHub account that
   owns this repo.
2. Click **New app** → **Deploy a public app from a repo** (a private repo works
   too on the free tier).
3. Set:
   - **Repository:** `<your-github-username>/mlb-model`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
4. Click **Deploy**. First build takes a few minutes (it installs
   `requirements.txt` and pulls `pybaseball`).

**Secrets — add these in the app, never in the repo.** Both are OPTIONAL; the
app runs with neither. After the app is created, open **Manage app → Settings →
Secrets** and paste only the lines you want (TOML format, same keys as
`.streamlit/secrets.toml.example`):

```toml
# STRONGLY recommended for a public URL — otherwise anyone with the link can open it.
MLB_DASHBOARD_PASSWORD = "choose-a-password"

# Optional: enables the in-app "Fetch live odds" button. Free key at highlightly.net.
HIGHLIGHTLY_API_KEY = "your-highlightly-key"
```

Streamlit Cloud exposes each secret both via `st.secrets` and as an environment
variable, which is exactly how this app reads them (`os.environ.get(...)`), so
no code change is needed. Save secrets and the app auto-reboots.

**What works on Cloud vs. local:** moneyline/total/run-line projections, the
scoreboard, game detail, team ratings, model performance, and the bet log all
work as-is (they read the committed `games.csv` and refit on load). The
"Rebuild historical dataset" button re-scrapes Baseball-Reference and can be
unreliable from a cloud IP — prefer refreshing data locally and pushing, rather
than rebuilding on Cloud.

**Never commit real secrets.** `.gitignore` already excludes `.env`,
`.streamlit/secrets.toml`, and the personal `bet_log.csv`. A previously
hard-coded Odds-API key was removed from `src/run_comparison.py` (it now reads
`THE_ODDS_API_KEY` from the environment). If you rotate keys, put them in the
Cloud Secrets panel or your local `.streamlit/secrets.toml` — never in tracked
files.

## How it works

Three components:

1. **Elo rating system** (`src/models/elo.py`) — every team starts at 1500 and
   is updated after each game based on the result, a home-field-advantage
   adjustment (+24 Elo points to the home team), and a margin-of-victory
   multiplier (blowouts move ratings more, dampened for expected blowouts,
   amplified for upsets). Ratings partially regress to the mean at the start
   of each season.

2. **Team offense/defense/park ratings** (`src/models/run_model.py`,
   `TeamRunRatings`) feed expected runs (`mu_home`, `mu_away`) for each side of
   a matchup: offense/defense rate relative to league average (blended
   recent-form + full-sample, shrunk toward league average for small
   samples), a park factor for the home stadium, and — for live/upcoming
   games only — the actual probable starting pitcher's season ERA and a
   FIP-style component, blended with the team's overall pitching rate (the
   starter is assumed to account for ~56% of a game's defensive innings).

3. **Monte Carlo simulation is the primary prediction method**
   (`src/models/monte_carlo.py`). Each game is simulated (10,000 times by
   default, configurable in the dashboard sidebar and via `n_sims`/
   `config.MC_DEFAULT_SIMS`): both teams' runs are sampled from a negative
   binomial distribution built from `mu_home`/`mu_away` and an **empirically
   estimated overdispersion factor** — measured directly from real
   historical team-game data (`TeamRunRatings.overdispersion`), not assumed.
   In this project's training data, real MLB team-game run totals run
   ~2.25x the variance a pure Poisson would predict. Moneyline win
   probability, the total-runs distribution, and run-line cover probability
   are then read off directly as **empirical frequencies** from the
   simulated outcomes.

   The closed-form Poisson/Skellam math (`run_model.game_probabilities`,
   `run_model.cover_probability`, `run_model.total_probabilities`) still runs
   alongside every simulation as a fast, independent cross-check — it's pure
   Poisson (no overdispersion), so a moderate, persistent gap between it and
   Monte Carlo on spread-sensitive numbers (like the total-runs predictive
   interval) is expected and explainable, not a bug. Both numbers are shown
   side by side throughout the dashboard.

4. **Ensemble** (`src/models/ensemble.py`) — Elo's win probability and the
   Monte Carlo engine's win probability are averaged 50/50
   (`config.ELO_BLEND_WEIGHT`) into the final moneyline number. This weight
   is a reasonable default, not tuned against a validation set.

## Data sources

- **Historical training data**: Baseball-Reference game logs via `pybaseball`
  (`src/data/fetch_historical.py`), covering `config.HISTORICAL_SEASONS` plus
  the current season-to-date. ~8,700 games as of the initial build
  (2023-2026).
- **Today's schedule & probable pitchers**: the free public MLB Stats API
  (`statsapi.mlb.com`), no API key required (`src/data/statsapi_client.py`).
- **Live odds**: real multi-sportsbook snapshots, handed to the project
  periodically as files (no live API connection on this project's end). See
  "Odds and edge-finding" below.

Team codes are canonicalized in `src/data/team_mapping.py` (MLB Stats API
abbreviations are used as the canonical codes everywhere) to handle the
mismatch between data sources and franchise relocations (e.g. the Athletics'
Baseball-Reference code changed from `OAK` to `ATH` starting in 2025).

## Keeping probable pitchers fresh — a workflow, not just a feature

The model is only as good as its starting-pitcher input, and probable
pitchers can change last-minute (rainouts, bullpen games, injuries,
last-minute scratches). This isn't fully solvable technically — MLB itself
doesn't always know until close to first pitch — so the dashboard is built
to make staleness *visible* and *cheap to fix*, and this is meant to be
checked, not just built once and trusted:

- **Every prediction shows when its pitcher data was pulled.** Look for the
  "pitchers checked HH:MM:SS" caption next to each game's probable-pitcher
  line (and a summary version above the games table). If it's been a while,
  don't trust the pick without refreshing.
- **Predictions auto-refresh every 15 minutes** on their own (the dashboard's
  cache has a built-in TTL) — so simply leaving the dashboard open and
  reloading periodically keeps pitcher data reasonably current without any
  action.
- **The sidebar's "🔄 Refresh probable pitchers & re-run projections" button**
  forces an immediate re-pull of just today's schedule, probable pitchers,
  and pitcher stats from the live MLB Stats API, and re-runs projections —
  a few seconds, no historical rescrape. Use this instead of waiting on the
  15-minute cache when it matters (see recommended cadence below).
- **If a probable pitcher changed since the last check, you'll see a
  warning** — both a summary banner above the games table and an inline
  warning on the specific game, naming the old and new pitcher. Projections
  shown are always for the *current* pitcher; the warning exists so a
  last-minute swap doesn't get missed silently, especially if you'd already
  formed a view (or placed a bet) based on the earlier name. This uses a
  small on-disk file (`data/processed/pitcher_watch.json`) that just
  remembers the last-seen probable pitcher per game — no history, no
  database, nothing to maintain.

**Recommended check-in cadence:**
- **Re-check probable pitchers within 1-2 hours of first pitch**, and again
  right before you act on a pick — this is the single highest-value habit
  here, especially for afternoon games (day-of scratches are more common
  than evening games where the pitcher's been public since the day before)
  and any game with rain in the forecast (bullpen games get called on short
  notice).
- **Don't trust a prediction pulled more than a few hours ago** for
  same-day betting decisions without hitting refresh first — a lot can
  change, and the whole point of the pitcher adjustment is that it's
  pitcher-specific.
- **If you see the "pitcher changed" warning, re-read the projection before
  acting** — it already reflects the new pitcher (projections aren't stale,
  only the *awareness* that something changed could have been), but any
  edge/EV number you'd mentally locked in from an earlier look is now
  based on the wrong pitcher and should be treated as void until you've
  looked again.

## Odds and edge-finding

This project doesn't have its own live odds API connection. Instead it reads
real odds from JSON files:

**Preferred: multi-sportsbook snapshots** (`odds/snapshots/`) — one JSON
object (or a list of them) per file, in this shape:

```json
{
  "event_id": "fef71589-...",
  "offer_type": "moneyline",
  "away_team": "nym", "home_team": "phi",
  "odds": {
    "DraftKings": [{"side": "away", "line": 0, "odds": 115}, {"side": "home", "line": 0, "odds": -139}],
    "FanDuel":    [{"side": "away", "line": 0, "odds": 110}, {"side": "home", "line": 0, "odds": -130}],
    "Pinnacle":   [{"side": "away", "line": 0, "odds": 117}, {"side": "home", "line": 0, "odds": -127}]
  }
}
```

`offer_type` is `"moneyline"`, `"spread"` (the run line — side is home/away
with a real signed line, not assumed to be exactly ±1.5), or `"total"` (side
is over/under with a real line). See `odds/snapshots_example/` for working
examples and `odds/snapshots_example/README.txt` for how to try them. Drop
real snapshot files into `odds/snapshots/` (create the folder) and reload
the dashboard — no restart needed, just "Refresh predictions for this date."

For each event+market, the adapter (`src/odds/odds_adapter.py`) computes,
per side: the single **best price** across every book (what you'd actually
get shopping lines) and a de-vigged **consensus probability** averaged
across every book's own no-vig number. Edges are reported both ways — edge
vs. consensus (does the model disagree with the market's collective fair
value?) and expected value (EV) vs. the best price (what you'd actually earn
betting the single best line available) — using the Monte Carlo engine's
probability at that market's *actual* line (which varies per book/game and
isn't assumed to be some default).

**Fallback: single-book manual entry** (`odds/odds.json`) — simpler, for
quick manual testing. Copy `odds/odds_example.json` to `odds/odds.json` and
fill in real numbers from any one sportsbook. Both formats are normalized
into the same internal shape, so the dashboard doesn't need to know which
one is in use; snapshots are preferred if both are present.

### Highlightly live odds (automated source — needs a PAID plan)

`src/data/highlightly.py` pulls a full slate's moneyline / run line / totals
from [Highlightly](https://highlightly.net) and writes it straight into
`odds/snapshots/` — a hands-off replacement for pasting in ESPN lines by hand.

> **Verified caveat (tested with a real key 2026-07-17):** Highlightly's
> marketing lists odds under the **free "Basic" ($0)** plan, but the live API
> rejects them — `GET /odds` returns `401 {"message":"Odds are not available in
> Basic plan. Please upgrade your plan."}`. Auth and the schedule endpoint work
> fine on Basic (the `/matches` parse and team-code resolution were verified
> against live data, 7/7 games mapped), but **odds require a paid tier** — PRO
> is **$7.99/mo** (7,500 req/day), then Ultra/Mega. The integration is built
> and ready; on Basic the dashboard button and CLI will surface that exact
> "upgrade your plan" error rather than silently failing. The `/odds` *parser*
> is built to Highlightly's published schema but has **not** been run against a
> live odds response (it's paywalled) — run `--probe` once on a paid plan to
> confirm the field names before trusting it.

Setup: get a key (a **paid plan for odds**), then expose it as an environment
variable before launching (preferred over pasting into `config.py`):
```
set HIGHLIGHTLY_API_KEY=your-key-here      # Windows
export HIGHLIGHTLY_API_KEY=your-key-here    # macOS/Linux
```
Then either click **"📥 Fetch live odds (Highlightly)"** in the dashboard
sidebar, or run:
```
python -m src.data.highlightly            # today's slate
python -m src.data.highlightly 2026-07-20
python -m src.data.highlightly --probe    # dump raw API JSON (see note below)
```

Every fetch also **appends each game's totals line to
`data/raw/historical_totals.csv`** (deduped), so from today forward we build
our *own* real closing-totals dataset organically — the file the totals
backtest consumes.

> **This fixes live-odds reliability (Problem A) — it does NOT unlock the
> totals backtest (Problem B).** A live-odds API only gives odds from now
> forward, not years of history. The organically-accumulated file above won't
> be a big enough sample to backtest meaningfully for many months. Validating
> the totals model against real history *still* requires either a **paid
> historical odds API** (e.g. The Odds API's paid tiers) or the free
> **Sportsbook Reviews Online / Kaggle** season files — see "Totals vs. the
> real market" above. Nothing about the Highlightly integration changes that.

> **First-run note:** the request/response shapes are built from Highlightly's
> published docs but were not run against a live key in this build. Run
> `--probe` once with your key to dump the raw JSON; if any field names differ,
> the fix is isolated to the `parse`/`_collect` helpers in `highlightly.py`.
> Any team name that doesn't resolve to a code is reported (not silently
> mis-mapped) so you can add it to `team_mapping._NAME_KEYWORDS`.

**Other manual fallbacks** remain available: multi-book snapshot files dropped
into `odds/snapshots/`, or single-book `odds/odds.json` (copy from
`odds/odds_example.json`). All are normalized to the same internal shape;
if several are present they're matched per game by team + start time.

## Backtest

Run `python -m src.backtest.backtest` (or use the dashboard's "Run backtest"
button, or `refresh_data.bat`). It walks the model forward through history
with **no lookahead** — the run-model ratings used to predict any given date
are re-fit using only games strictly before that date, using Monte Carlo
(fewer sims per game than live, `config.BACKTEST_MC_SIMS`, purely for
runtime) alongside the closed-form cross-check — and compares predictions to
actual results, evaluated on 2024-2026(partial) with 2023 as warm-up
(~6,300 games).

**Read the results next to their naive baselines, not in isolation** (the
dashboard's Model Performance tab does this automatically):

| Market | Model accuracy | Naive baseline | Calibration (ECE) | Real signal? |
|---|---|---|---|---|
| Moneyline | ~56% (blended) | ~53% (always pick home) | 2.2% (10-bin Expected Calibration Error) | Yes — modest but real (+2.7 pts, better Brier/log-loss too) |
| Run line (-1.5/+1.5) | ~64% | ~64% (always pick underdog) | 2.1% | Barely — MLB underdogs cover +1.5 in ~2/3 of games by construction, so raw accuracy here is mostly base rate, not skill |
| Total runs | 50.25% ATS vs real closing lines | ~50.5% (always under) | over-prob ECE 7.4% | **No — now tested against 11,706 real closing lines (2015-2019) and it does not beat the market** (see "Totals vs. the real market" below). Internally well-calibrated (PIT deviation 0.3 pts, interval coverage ~56%/84%) but that's coherence, not market-beating. |

### Starting-pitcher adjustment: isolated and tested separately

`src/backtest/pitcher_backtest.py` isolates whether the starting-pitcher
adjustment (season ERA/FIP blended into mu, live-only) actually improves on
the team-level-only baseline above, using real historical probable-pitcher
data and genuine as-of-date pitcher stats (no lookahead) — see the module
docstring for the full methodology. Run it with
`python -m src.backtest.pitcher_backtest` (needs
`python -m src.data.fetch_historical_pitchers` first to pull the
per-game starter and pitcher-log data, ~10-15 minutes, network-heavy).

**Honest result, on 4,822 games (2024-2025) with a bootstrapped significance
check on every comparison:**

- **A real, quantified bug was found and fixed along the way.** The FIP
  component's blend used the wrong additive constant (`league_avg_era`
  instead of the actual FIP constant, `config.LEAGUE_AVG_FIP_CONST`),
  which inflated the average pitcher factor to ~1.08 (it should center near
  1.00 -- confirmed by direct comparison against the real observed starter
  ERA distribution). Before the fix, this made the pitcher adjustment
  **significantly worse** than the team-only baseline for total runs (MAE
  degraded by 0.072 runs on average, 95% CI [0.051, 0.094], entirely
  positive -- a real, not noisy, effect). After the fix, that harm
  disappears (MAE difference -0.0006 runs, CI [-0.022, 0.020] -- a wash).
- **Even after the fix, the adjustment shows no statistically significant
  improvement anywhere** -- moneyline and run-line Brier scores both trend
  slightly *worse* with the adjustment on (not significant: 95% CIs include
  zero), and total-runs MAE is statistically indistinguishable from the
  baseline. Split by season for robustness: 2025 is a wash across all three
  markets; 2024 shows a **statistically significant harm to moneyline**
  specifically (Brier CI [-0.0055, -0.0002], entirely negative) even with
  the bug fixed.
- **Bottom line: as currently built, the starting-pitcher adjustment does
  not demonstrate a measurable benefit over the team-level-only baseline in
  this holdout**, despite being intuitively reasonable and despite fixing a
  real implementation bug along the way. It's not actively recommended to
  trust live picks as "sharper" because of it. Plausible reasons (not
  tested further, to avoid overfitting this exact holdout): the 15-IP
  minimum still admits fairly noisy early-season sample sizes; the team
  defense rating may already implicitly capture recent staff performance;
  the 50/50 ERA/FIP blend and 56% starter-innings-share weights were
  reasonable a priori choices, not fit to data, and may not be the right
  ones. See `data/processed/pitcher_backtest_summary.json` for the full
  numbers (overall, active-only, by-season, by-season-active-only).

### Totals vs. the real market — RUN, and the answer is no

This gap is now closed with real data. `src/data/historical_odds.py` pulls the
free, public [`pwu97/bettingtools`](https://github.com/pwu97/bettingtools)
dataset (SBR-sourced MLB closing lines + final scores, 2014-2019), which
doubles as both the odds source and the game log. Because free odds archives
(~2010-2021) don't overlap our live 2023-2026 `games.csv`, the backtest runs
entirely within that era: the model is fit walk-forward on 2014-2019 games and
its over/under picks are graded against the real 2014-2019 closing lines — a
genuine out-of-sample test on a period never touched during development.

Reproduce: `python -m src.data.historical_odds --run-backtest` (needs
`pip install pyreadr`).

**Result, on 11,706 real closing lines (2015-2019, 2014 as warm-up):**

| Metric | Value | Read |
|---|---|---|
| Model ATS win rate | **50.25%** | vs 50.45% "always under" / 49.55% "always over" — **no edge** |
| vs-baseline significance | not significant (95% CI [−1.4, +1.0] pp) | indistinguishable from a coin flip against the number |
| ROI (flat, −110 assumed) | **−4.07%** | roughly the vig — you'd lose at the hold |
| Over-prob calibration ECE | **7.4%** | and **overconfident on overs**: when it predicts ~64% over, the actual rate is ~48% |

**Honest conclusion: the totals model does not beat the market.** Its internal
PIT calibration was good, but internal coherence ≠ market-beating, and now we
have the evidence. Notably its *mean* total (8.94) was actually closer to the
actual average (9.05) than the market line (8.59) — but being right on the
aggregate mean doesn't translate to game-by-game edge, because the closing
line embeds per-game information (weather, lineups, bullpen) the model lacks.
The overconfidence-on-overs finding (ECE 7.4%) is a concrete lead for future
calibration work. Caveat: 2015-2019 is a different run environment than the
model's 2023-2026 tuning; the model re-fits per-era, but this is somewhat
out-of-distribution. It is, however, the only free market-graded test
available, and the result is consistent with priors.

**Bottom line for betting: trust moneyline (validated edge), not totals or the
run line.** The full summary is in `data/processed/totals_backtest_summary.json`.

> The live accumulator (`accumulate_totals`) also keeps appending today's
> real lines to `data/raw/historical_totals.csv` going forward, so eventually
> the same backtest can be re-run on the *current* era once enough have
> accrued — months away, but building.

## Bet tracker — the real forward test

The backtest is in-sample history; the honest test is whether the picks you
*actually place* make money going forward. `data/processed/bet_log.csv` (1u flat stakes) is a
hand-maintained log of exactly those, and `python -m src.bet_tracker`
(or the dashboard's **Bet Log** tab) turns it into a running record, win rate
by market, and flat-stake ROI.

**Adding a bet:** open `bet_log.csv` and add a row. The only columns you must
fill are `odds` (American, e.g. `-116`), `model_prob` (the model's probability
for the side you took, at the time you took it), and — once the game ends —
`result` (`win` / `loss` / `push`; leave blank until it's final). `date`,
`matchup`, `market`, `pick`, `stake`, and `notes` are for your own reading.

**What's derived vs. what you enter:** `implied_prob` and `edge_pct` are
recomputed from `odds` and `model_prob` every time the tracker loads, so you
never have to do the math — if you edit the odds, the summary just stays
correct. They live in the CSV only so the file reads sensibly on its own.

**Conventions and assumptions (so the numbers mean one consistent thing):**

- `model_prob` is **yours to record** — nothing recomputes it, because the
  model's inputs move day to day. For moneyline use the blended Elo+MC number;
  for totals and the run line use the pure Monte Carlo probability (those
  markets aren't blended). Whatever you enter is what the edge is measured from.
- `implied_prob` is the **raw break-even** from the odds, vig included — the
  actual bar a bet must clear to profit. It is deliberately **not** the
  de-vigged consensus (that measures market disagreement, a different question
  than "did this bet make money").
- `edge_pct = (model_prob − implied_prob) × 100`, in percentage points.
- **ROI is unit-normalized: 1u flat per bet**, not dollars. What matters for
  judging the model is return per unit risked, not how much happened to be on
  it that day. A win pays the odds, a loss loses the unit, a push returns it.
  ROI = net profit ÷ units actually at risk; **pushes and still-pending bets
  are excluded from turnover**, not counted as losses.
- Change `stake` per row only if you genuinely sized a bet differently.
- **Kalshi / binary-contract entries**: a Kalshi price *is* a probability (a
  56¢ contract = 56% implied), so it's converted to the American-odds
  equivalent on entry with the original price kept in `notes`. Economically
  it's the same bet, and it keeps one consistent math path for every venue.

**Seed data (already in the log), with the assumptions I had to make:**

- **2026-07-16 · NYM +1.5 · win.** You didn't give the exact price, so I used
  the best number we had recorded that day, **−170 (BookMaker)** from the
  20+-book feed. Edit it if you actually took a different line. `model_prob`
  is the push-corrected Monte Carlo cover probability (67.7%).
- **2026-07-17 · BOS ML −116 · pending.** You said it was *currently* winning,
  not final, so per your "leave blank until known" instruction its `result` is
  blank — it counts as pending, not a win, until the game ends. `model_prob` is
  the blended 57.6% (pure MC was 59.9%, noted in the row).

**The point is honesty over time, not a scoreboard.** With one or two settled
bets the ROI is pure noise — the tracker says so out loud until there are
enough bets (~50–100+) to mean anything. Let the sample grow before believing
any of it.

## Limitations — read this before betting anything on it

- **The starting-pitcher adjustment doesn't demonstrate a measurable
  benefit** in a proper, bug-fixed, statistically-tested holdout (see
  above) -- it's still used for live predictions (a reasonable model even
  without proven lift), but don't assume live picks are meaningfully
  sharper than the backtested team-only numbers because of it.
- **Run-line "accuracy" is mostly base rate.** See the table above. Use the
  run-line *probabilities* to compare against real market prices; don't
  trust a standalone win/loss record on this market.
- **Backtest scope**: the main backtest table above evaluates the
  *team-level* Elo + Monte Carlo engine, at a lower sim count than the live
  default purely for runtime. The pitcher adjustment is evaluated
  separately (see above) rather than folded into this table, since isolating
  it cleanly required a different, smaller-sample pipeline.
- **No historical odds data**: there's no historical sportsbook closing-line
  data in this build, so the total (over/under) market can't be backtested
  against real market lines — only against the model's own predicted mean
  (MAE/RMSE) and its own predictive interval coverage (computed from each
  game's actual simulated distribution). Moneyline and run line don't have
  this problem since they only need the actual outcome.
- **Odds snapshots have no date field**: the real snapshot format has no
  game date, so odds are matched to predictions by team matchup only — two
  games between the same two teams on different dates in a short series
  can't be told apart. Fine for same-day snapshots (the common case).
- **Small, recent sample**: training data covers only a few seasons.
- **Whole-game simulation, not inning-by-inning**: Monte Carlo samples each
  team's full-game run total, not inning by inning — Baseball-Reference
  doesn't give inning-by-inning box scores without a much heavier data pull,
  and the scoring-rate inputs are themselves game-level, so inning-level
  simulation wouldn't add real information.
- **Overdispersion is one global estimate**, not fit per team.
- **Extra innings** are modeled as a fixed 52%-home-team tiebreak in both
  engines, not simulated inning by inning.
- **Elo constants** (home-field advantage, K-factor, margin-of-victory
  scaling) are reasonable, commonly-used starting values, not fitted by
  optimizing against a validation set.
- **Park factors** are estimated from a limited number of home games per
  park and aren't split by handedness or weather.
- **Not licensed betting or financial advice.** A model output labeled
  "edge" is a hypothesis to sanity-check, not a guarantee.

## Project structure

```
mlb-model/
  src/
    config.py            constants (seasons, Elo/MC/backtest params)
    pipeline.py           orchestrates fitting + prediction generation
    data/
      team_mapping.py      canonical team codes across data sources
      fetch_historical.py  Baseball-Reference historical game log scraper
      statsapi_client.py   MLB Stats API client (schedule, pitcher stats)
    models/
      elo.py               Elo rating system
      run_model.py          team offense/defense/park ratings + closed-form cross-check math
      monte_carlo.py         Monte Carlo simulation engine (primary prediction method)
      ensemble.py           blends Elo + Monte Carlo win probabilities
    backtest/
      backtest.py           walk-forward backtest + honest metrics
    odds/
      odds_adapter.py        snapshot + manual odds loading, de-vigging, edge/EV calc
  dashboard/
    app.py                  Streamlit dashboard
  odds/
    odds_example.json        template — copy to odds.json for single-book manual entry
    snapshots_example/       example real-format multi-book snapshot files
    snapshots/                drop real snapshot files here (create this folder)
  data/processed/           cached data + model outputs (gitignored-style; regenerate via setup.bat)
  setup.bat / run_dashboard.bat / refresh_data.bat
```

## Updating the model over time

- **Re-scrape data periodically** (weekly is plenty during the season) via
  `refresh_data.bat` so ratings reflect recent form.
- **`config.py`** is the place to retune things: `HISTORICAL_SEASONS` and
  `CURRENT_SEASON` need bumping every offseason; Elo/Monte Carlo constants
  can be adjusted if you want to experiment (no formal parameter search was
  run). `MC_DEFAULT_SIMS` trades precision for speed on live predictions;
  `BACKTEST_MC_SIMS` does the same for the (much larger) backtest run.
