"""
Season-to-date team features built ONLY from games already played (the engine feeds a game in
after it has been decided and settled, so nothing here can see the game being predicted).
"""

from collections import deque
from datetime import date

RECENT_GAMES = 10


class TeamState:
    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.points_for = 0
        self.points_against = 0
        self.recent = deque(maxlen=RECENT_GAMES)   # 1 = win, 0 = loss
        self.last_game_date = None

    @property
    def games(self) -> int:
        return self.wins + self.losses

    def WinPct(self, default: float = 0.5) -> float:
        return self.wins / self.games if self.games else default

    def RecentPct(self, default: float = 0.5) -> float:
        return sum(self.recent) / len(self.recent) if self.recent else default

    def AvgMargin(self) -> float:
        return (self.points_for - self.points_against) / self.games if self.games else 0.0

    def RestDays(self, game_date: str):
        """Days off before game_date (0 = back-to-back). None before a team's first game."""
        if self.last_game_date is None:
            return None
        return (date.fromisoformat(game_date) - date.fromisoformat(self.last_game_date)).days - 1


class FeatureBook:
    """All teams' running state for one season."""

    def __init__(self):
        self.teams = {}

    def Team(self, abbr: str) -> TeamState:
        return self.teams.setdefault(abbr, TeamState())

    def View(self, game: dict) -> dict:
        """Pre-game features for both sides. No scores."""
        view = {}
        for side in ("home", "away"):
            team = self.Team(game[f"{side}_team"])
            view.update({
                f"{side}_games": team.games,
                f"{side}_win_pct": round(team.WinPct(), 4),
                f"{side}_recent_pct": round(team.RecentPct(), 4),
                f"{side}_avg_margin": round(team.AvgMargin(), 2),
                f"{side}_rest_days": team.RestDays(game["game_date"]),
            })
        return view

    def Apply(self, game: dict) -> None:
        """Feed a finished game into both teams' state."""
        home, away = self.Team(game["home_team"]), self.Team(game["away_team"])
        home_won = game["home_score"] > game["away_score"]
        for team, scored, allowed, won in ((home, game["home_score"], game["away_score"], home_won),
                                           (away, game["away_score"], game["home_score"], not home_won)):
            team.wins += won
            team.losses += not won
            team.points_for += scored
            team.points_against += allowed
            team.recent.append(int(won))
            team.last_game_date = game["game_date"]
