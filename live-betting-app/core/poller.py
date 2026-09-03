"""One poll cycle across every registered sport -- OBSERVE / LOG ONLY.

Run standalone (`python -m core.poller`) or on a schedule (see
.github/workflows/poll.yml, which runs this every few minutes via cron and
commits any new data back to the repo).

*** THIS TOOL NEVER PLACES A BET OR TAKES ANY REAL-MONEY ACTION. ***
It only detects and logs signals (model prob vs market-implied prob, flagged
mispricings) and, when a game newly crosses the edge threshold, pushes a
Telegram notification. There is no order-placement code anywhere in this
project by design, matching the hard rule in both source repos (mlb-model,
cfb-model) this app's logic was vendored from.
"""
from __future__ import annotations

import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # local dev only; CI/Cloud use platform secrets

from core import storage, telegram
from sports.base import validate_row
from sports.registry import SPORTS


def run_once() -> dict:
    """Poll every sport once. Returns a summary dict for logging/CI output."""
    now_ts = time.time()
    notified_state = storage.load_notified_state()
    summary = {"sports": {}, "new_rows": 0, "notifications_sent": 0}

    for sport_key, adapter in SPORTS.items():
        label, icon = adapter.SPORT_LABEL, adapter.SPORT_ICON
        try:
            rows = adapter.poll()
        except Exception as e:
            print(f"[poller] {sport_key}: poll() raised {type(e).__name__}: {e}")
            summary["sports"][sport_key] = {"error": str(e)}
            continue

        last_by_game = storage.load_log_tail_by_game(sport_key)
        new_rows = []
        for row in rows:
            row.setdefault("sport", sport_key)
            validate_row(row)  # hard-stops on mode != OBSERVE_ONLY -- see sports/base.py
            prev = last_by_game.get(row["game_id"])
            if prev is not None and adapter.state_key(prev) == adapter.state_key(row):
                continue  # unchanged since last logged row -- not new
            new_rows.append(row)

        storage.append_rows(new_rows)
        summary["new_rows"] += len(new_rows)

        n_notified = 0
        for row in new_rows:
            if not row["flagged"] or not row.get("pick_team"):
                continue
            if not storage.should_notify(notified_state, sport_key, row["game_id"],
                                         row["pick_team"], now_ts):
                continue
            text = telegram.format_flag_message(label, icon, row)
            if telegram.send_message(text):
                storage.record_notified(notified_state, sport_key, row["game_id"],
                                        row["pick_team"], now_ts)
                n_notified += 1

        summary["notifications_sent"] += n_notified
        summary["sports"][sport_key] = {
            "live_games": len(rows), "new_rows": len(new_rows), "notified": n_notified,
        }
        print(f"[poller] {sport_key}: {len(rows)} live, {len(new_rows)} new row(s), "
              f"{n_notified} notification(s) sent")

    storage.save_notified_state(notified_state)
    return summary


if __name__ == "__main__":
    print(f"[poller] OBSERVE-ONLY run starting -- {', '.join(SPORTS)}")
    result = run_once()
    print(f"[poller] done: {result['new_rows']} new row(s) total, "
          f"{result['notifications_sent']} notification(s) sent")
