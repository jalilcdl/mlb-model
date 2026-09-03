"""Telegram push notifications for flagged live signals.

Free, real push-to-phone via a bot. Needs two secrets (never committed):
  TELEGRAM_BOT_TOKEN  -- from @BotFather
  TELEGRAM_CHAT_ID    -- the chat to push into (see scripts/get_telegram_chat_id.py)

This module only SENDS a text message. It has no bet-placing capability and
must never gain any -- it is a notifier, not an actor.
"""
from __future__ import annotations

import os

import requests

API_BASE = "https://api.telegram.org"
_TIMEOUT = 10


def is_configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN")) and bool(os.environ.get("TELEGRAM_CHAT_ID"))


def send_message(text: str) -> bool:
    """Best-effort push. Returns True on success, False (never raises) on failure
    so a Telegram outage can't take down the poller."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[telegram] not configured (missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID); skipping push")
        return False
    url = f"{API_BASE}/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown",
                  "disable_web_page_preview": True},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"[telegram] send failed: {type(e).__name__}: {e}")
        return False


def format_flag_message(sport_label: str, sport_icon: str, row: dict) -> str:
    edge_pct = row["edge"] * 100
    return (
        f"{sport_icon} *{sport_label} signal flagged*\n"
        f"{row['matchup']}\n"
        f"{row.get('state_desc', '')}\n"
        f"Model {row['model_home_wp']:.0%} vs Market {row['market_home_wp']:.0%} "
        f"(home win) -- edge {edge_pct:+.1f}pts\n"
        f"Model favors: *{row.get('pick_team', '?')}*\n"
        f"_Observe-only signal. Not a bet recommendation._"
    )
