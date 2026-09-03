"""Registered sport adapters. Add a new sport (e.g. NFL) by writing
sports/nfl/adapter.py with poll()/state_key()/SPORT_KEY/SPORT_LABEL/SPORT_ICON
and adding one line here -- nothing else in the app needs to change.
"""
from sports.mlb import adapter as mlb_adapter
from sports.cfb import adapter as cfb_adapter

SPORTS = {
    mlb_adapter.SPORT_KEY: mlb_adapter,
    cfb_adapter.SPORT_KEY: cfb_adapter,
}
