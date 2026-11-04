"""
Strategies look at one game's pre-game view and return "home", "away" or None (no pick).

    view = {"home_team", "away_team", "game_date", "season_type",
            "home_games", "home_win_pct", "home_recent_pct", "home_avg_margin", "home_rest_days",
            "away_...same...",
            "model_home_prob", "model_away_prob",          # CALCULATED (Model.EloModel)
            "home_moneyline", "away_moneyline",            # only if the game record has odds
            "market_home_prob", "market_away_prob"}        # de-vigged, only if odds present

The view never contains the score, so a strategy can't peek at the result.
"""

from typing import Optional


class AlwaysHome:
    """Baseline: pick the home team every game."""
    name = "always_home"
    description = "Pick the home team every game."

    def Decide(self, view: dict) -> Optional[str]:
        return "home"


class BetterRecord:
    """Pick the team with the better season-to-date win % (ties -> home). Skips games until both
    teams have played `min_games`."""

    def __init__(self, min_games: int = 5):
        self.min_games = min_games
        self.name = "better_record"
        self.description = f"Pick the better season-to-date win % (ties -> home), once both teams have {min_games}+ games."

    def Decide(self, view: dict) -> Optional[str]:
        if min(view["home_games"], view["away_games"]) < self.min_games:
            return None
        return "away" if view["away_win_pct"] > view["home_win_pct"] else "home"


class RestAdvantage:
    """Pick the more-rested team when the rest gap is at least `min_gap` days (e.g. the other side
    is on a back-to-back); otherwise no pick."""

    def __init__(self, min_gap: int = 1):
        self.min_gap = min_gap
        self.name = "rest_advantage"
        self.description = f"Pick the team with {min_gap}+ more days of rest; otherwise no pick."

    def Decide(self, view: dict) -> Optional[str]:
        home_rest, away_rest = view["home_rest_days"], view["away_rest_days"]
        if home_rest is None or away_rest is None:
            return None
        gap = min(home_rest, 4) - min(away_rest, 4)   # cap: a week off isn't better than 4 days
        if gap >= self.min_gap:
            return "home"
        if -gap >= self.min_gap:
            return "away"
        return None


class EloPick:
    """Pick whichever side the Elo model favours."""
    name = "elo_pick"
    description = "Pick the side with Elo win prob > 50% (Elo includes home court)."

    def Decide(self, view: dict) -> Optional[str]:
        return "home" if view["model_home_prob"] >= 0.5 else "away"


class ModelEdge:
    """Only when odds are present: pick the side whose model prob beats the de-vigged market prob
    by at least `min_edge`. Without odds it never picks."""

    def __init__(self, min_edge: float = 0.05):
        self.min_edge = min_edge
        self.name = f"model_edge_{int(round(min_edge * 100))}pct"
        self.description = f"Needs moneylines: pick the side where model - market prob >= {min_edge * 100:.0f} pts."

    def Decide(self, view: dict) -> Optional[str]:
        if view.get("market_home_prob") is None:
            return None
        home_edge = view["model_home_prob"] - view["market_home_prob"]
        away_edge = view["model_away_prob"] - view["market_away_prob"]
        if home_edge >= self.min_edge and home_edge >= away_edge:
            return "home"
        if away_edge >= self.min_edge:
            return "away"
        return None


def DefaultStrategies() -> list:
    return [AlwaysHome(), BetterRecord(), RestAdvantage(), EloPick(), ModelEdge()]
