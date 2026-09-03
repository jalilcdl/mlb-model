"""
Live / in-game betting signal (PROTOTYPE).

This package is a proof-of-concept, separate from the validated pregame pipeline.
Its question is narrow: does a real-time gap between our in-game win-probability
estimate and the live market's de-vigged price EXIST and look exploitable? It is
NOT a production betting system and shares none of the pregame model's validation.

Three pieces:
  win_expectancy.py -- in-game win probability from base/out/inning/score state,
                       seeded with this repo's team run ratings as the prior.
  live_odds.py      -- live odds fetcher (SharpAPI free tier), key-pluggable,
                       with a mock provider so the logic is testable with no key.
  signal.py         -- de-vig the live two-way, compare to the model, flag gaps.

Run `python -m src.live.demo` for an end-to-end walk-through on mock odds.
"""
