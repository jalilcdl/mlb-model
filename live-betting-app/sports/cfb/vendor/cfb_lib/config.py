"""Trimmed config for the vendored CFB live-signal path only.

Vendored from cfb-model/src/config.py. The only thing this path actually
needs from it is triggering .env loading so SHARPAPI_API_KEY lands on
os.environ (checked live in os.environ directly by live_odds.py) -- see the
dependency trace that produced this vendor copy. The full CFB model config
(CFBD data files, historical seasons, etc.) lives in the source repo.
"""
from pathlib import Path
from dotenv import load_dotenv as _load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
_load_dotenv(ROOT_DIR / ".env")
