# Live Betting Signals

Unified, always-on dashboard for the MLB and CFB live in-game signal
prototypes (mlb-model, cfb-model). **OBSERVE-ONLY** -- this detects and logs
disagreements between an in-game win-probability model and the de-vigged live
market. It never places a bet, sizes a stake, or recommends action, and there
is no order-placement code anywhere in this repo. That's a hard rule inherited
from both source repos, not a preference, and it must stay true for any sport
added here in the future.

## Architecture

This app lives at `live-betting-app/` inside the **mlb-model** GitHub repo (a
subfolder, sharing that repo/remote rather than a separate one) -- it doesn't
touch the MLB model's own code, and the MLB dashboard's own Streamlit Cloud
deploy is unaffected. Two independent free services, sharing this repo as
their "database":

1. **GitHub Actions** (`.github/workflows/poll-live-signals.yml`, at the repo
   ROOT -- GitHub only discovers workflow files there, not in subfolders) runs
   `live-betting-app/core/poller.py` on a ~5 minute cron. It polls every
   registered sport, appends any new game states to
   `live-betting-app/data/live_signal_log.jsonl`, sends a Telegram push for
   any newly-flagged game, and commits the changes back to this repo -- but
   only when something actually changed (no live games right now = no
   commit). This is fully independent of anything mlb-model's own code does.

2. **Streamlit Community Cloud** hosts `live-betting-app/app.py` as its own
   separate deployed app (a second Streamlit Cloud app pointed at the same
   repo, alongside whatever app already serves `dashboard/app.py`). It reads
   `data/live_signal_log.jsonl` from its own checkout. A commit from step 1
   triggers Streamlit Cloud to auto-redeploy with the fresh data -- which is
   why "nothing happens for hours, then a few quick redeploys during a live
   game" is expected behavior, not a bug.

No paid hosting, no tunnel, no localhost. Real HTTPS URL, reachable from your
phone, no manual restart.

**Note on git history:** the poller's auto-commits (author `live-signal-poller`)
will land on `main` alongside your normal mlb-model commits, since that's the
branch GitHub Actions' cron requires the workflow file to live on. They're
easy to filter out (`git log --invert-grep --author=live-signal-poller`) or
ignore, but if that ends up being annoying in practice, moving the data commits
to a dedicated branch is a reasonable follow-up -- just ask.

## Adding a sport (e.g. NFL later)

1. Create `sports/nfl/adapter.py` implementing `poll()` and `state_key()`
   (see `sports/base.py` for the interface, `sports/mlb/adapter.py` /
   `sports/cfb/adapter.py` for real examples).
2. Vendor whatever minimal state/odds-fetching code it needs under
   `sports/nfl/vendor/`.
3. Add one line to `sports/registry.py`.

Nothing else -- the poller, storage, Telegram formatting, and dashboard tabs
all pick it up automatically.

## One-time setup (you need to do these steps)

### 1. Push this to GitHub
This lives inside the existing `mlb-model` repo (https://github.com/jalilcdl/mlb-model),
pushed as a subfolder alongside the model code. Nothing for you to create here.

### 2. Add repo secrets (GitHub → jalilcdl/mlb-model → Settings → Secrets and variables → Actions → New repository secret)
- `SHARPAPI_API_KEY` -- your SharpAPI key (same one used by mlb-model /
  cfb-model; find it in `cfb-model/.env`, or generate a fresh one free at
  https://sharpapi.io if you'd rather not reuse it). One key works across
  sports.
- `TELEGRAM_BOT_TOKEN` -- see step 3 below.
- `TELEGRAM_CHAT_ID` -- see step 3 below.

### 3. Create your Telegram bot and get your chat ID
1. In Telegram, message **@BotFather**.
2. Send `/newbot` and follow the prompts (pick any name/username).
3. BotFather replies with a token like `123456789:AAExampleTokenHere`. Copy it.
4. Open your new bot in Telegram (BotFather gives you a link) and send it any
   message, e.g. "hi".
5. On your computer, run:
   ```
   pip install requests   # if not already installed
   python scripts/get_telegram_chat_id.py <paste your bot token here>
   ```
   It prints your `chat_id`. Use that as `TELEGRAM_CHAT_ID`.
6. Add both values as repo secrets (step 2).

### 4. Deploy the dashboard on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. "New app" → pick the **mlb-model** repo, branch `main`,
   **main file path `live-betting-app/app.py`** (this creates a second,
   independent deployed app from the same repo -- it won't affect or replace
   whatever app already serves `dashboard/app.py`).
3. Deploy -- no secrets needed here. The dashboard only *reads* the log file
   the poller already wrote; it never fetches odds or calls SharpAPI/Telegram
   itself, so it needs none of the keys from step 2.
4. You'll get a URL like `https://<something>.streamlit.app` -- that's your
   permanent link, bookmark it on your phone.

### 5. Verify
- Actions tab (on the mlb-model repo) → "Poll live signals" → Run workflow
  (manual trigger) to test the poller immediately instead of waiting for the
  next cron tick.
- Check the Actions log for `[poller] mlb: ... [poller] cfb: ...` output.
- Once a real game is live and a signal flags, you should get a Telegram
  push, and the dashboard should show it within a few minutes.

## Local development

```
cd live-betting-app
pip install -r requirements.txt
python -m core.poller          # one poll cycle, prints what it found
streamlit run app.py           # dashboard, reads data/live_signal_log.jsonl
```

Copy `.env.example` to `.env` for local secrets (never committed -- see
`.gitignore`).
