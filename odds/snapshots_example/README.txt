Example odds snapshot files, matching the real multi-sportsbook format the
project ingests from odds/snapshots/ (see src/odds/odds_adapter.py's module
docstring for the full spec).

To try the edge-finding feature with these examples: copy this folder's
*.json files (not this README) into odds/snapshots/, then reload the
dashboard. They're for the Mets @ Phillies game structure -- change
away_team/home_team to canonical team codes (lowercase is fine, matched
case-insensitively) to match whatever's actually on the schedule for the
date you're looking at.

One file per event+market here (moneyline/spread/total), but a single file
containing a JSON list of multiple such objects also works -- see
load_snapshots() in src/odds/odds_adapter.py.
