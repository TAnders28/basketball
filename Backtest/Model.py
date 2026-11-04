"""
Elo model for calculated win probabilities. Updated only after a game day is settled.

    home win prob = 1 / (1 + 10 ^ (-(home_elo + HOME_COURT_ELO - away_elo) / 400))

Ratings can be seeded from the previous (finished) season, regressed CARRYOVER of the way back
toward the mean, so early-season predictions aren't coin flips.
"""

import math

BASE_ELO = 1500.0
K_FACTOR = 20.0
HOME_COURT_ELO = 70.0     # ~2.5-3 pts of home court, the usual NBA Elo setting
CARRYOVER = 0.75          # keep 75% of last season's distance from the mean


class EloModel:
    def __init__(self, prior_ratings: dict = None):
        self.ratings = {}
        for team, rating in (prior_ratings or {}).items():
            self.ratings[team] = BASE_ELO + CARRYOVER * (rating - BASE_ELO)

    def Rating(self, team: str) -> float:
        return self.ratings.get(team, BASE_ELO)

    def HomeWinProb(self, home: str, away: str) -> float:
        diff = self.Rating(home) + HOME_COURT_ELO - self.Rating(away)
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def Apply(self, game: dict) -> None:
        home, away = game["home_team"], game["away_team"]
        expected = self.HomeWinProb(home, away)
        margin = game["home_score"] - game["away_score"]
        actual = 1.0 if margin > 0 else 0.0
        # FiveThirtyEight-style margin-of-victory multiplier, damped for favourites
        elo_diff = self.Rating(home) + HOME_COURT_ELO - self.Rating(away)
        favourite_diff = elo_diff if margin > 0 else -elo_diff
        multiplier = ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * favourite_diff)
        shift = K_FACTOR * multiplier * (actual - expected)
        self.ratings[home] = self.Rating(home) + shift
        self.ratings[away] = self.Rating(away) - shift


def SeasonFinalRatings(games: list) -> dict:
    """Run a finished season through a fresh model and return final ratings (for seeding next year)."""
    model = EloModel()
    for game in sorted((g for g in games if g.get("home_score") is not None), key=lambda g: (g["game_date"], g["game_id"])):
        model.Apply(game)
    return dict(model.ratings)
