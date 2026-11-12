"""
State models for the matchup research. Every class is fed a game only AFTER it is decided (the walk in
Matchups.WalkForward enforces the day barrier), and predicts from its current state.

MinutesBook   - who is on each roster and how many minutes they play (EWMA over games played).
RapmModel     - M1: minutes-weighted ridge player ratings on game margins (RAPM-lite), time-decayed.
MatchupModel  - M2: defender-offender assignment shares + Poisson points-per-possession per pairing
                with empirical-Bayes (gamma) shrinkage to offender / defender / league priors.
"""

import math
from collections import defaultdict

import numpy as np

POS_CODE = {"G": 1.0, "F": 2.0, "C": 3.0}
DEFAULT_MINUTES = 15.0
MINUTES_ALPHA = 0.25
PACE_ALPHA = 0.10
LEAGUE_PACE = 99.0


class PlayerIndex:
    def __init__(self):
        self.ids = {}

    def Get(self, player_id: int) -> int:
        if player_id not in self.ids:
            self.ids[player_id] = len(self.ids)
        return self.ids[player_id]


class MinutesBook:
    def __init__(self):
        self.ewma = {}           # player -> EWMA minutes when he plays
        self.games = defaultdict(int)
        self.last_team = {}
        self.last_date = {}
        self.team_last_roster = {}   # team -> [players in its most recent game]
        self.team_pace = {}
        self.names = {}

    def Expected(self, player_id: int) -> float:
        return self.ewma.get(player_id, DEFAULT_MINUTES)

    def PregameRoster(self, team: str, game_date: str) -> dict:
        """Knowable before tip: players from the team's most recent game still listed on that team."""
        roster = [p for p in self.team_last_roster.get(team, []) if self.last_team.get(p) == team]
        if not roster:   # season opener: anyone whose latest team is this one
            roster = sorted((p for p, t in self.last_team.items() if t == team),
                            key=lambda p: -self.Expected(p))[:13]
        return self.Scale({p: self.Expected(p) for p in roster})

    def OracleRoster(self, players: list) -> dict:
        """Who actually appeared (an injury-report proxy, NOT knowable for sure pregame), with PRE-game minutes."""
        return self.Scale({p: self.Expected(p) for p in players})

    @staticmethod
    def Scale(minutes: dict) -> dict:
        total = sum(minutes.values())
        return {p: m * 240.0 / total for p, m in minutes.items()} if total > 0 else {}

    def Pace(self, team: str) -> float:
        return self.team_pace.get(team, LEAGUE_PACE)

    def Apply(self, game: dict, player_rows: list, possessions: dict) -> None:
        by_team = defaultdict(list)
        for r in player_rows:
            p = r["player_id"]
            self.ewma[p] = r["minutes"] if p not in self.ewma else (
                (1 - MINUTES_ALPHA) * self.ewma[p] + MINUTES_ALPHA * r["minutes"])
            self.games[p] += 1
            self.last_team[p] = r["team"]
            self.last_date[p] = game["game_date"]
            self.names[p] = r.get("name")
            by_team[r["team"]].append(p)
        for team, players in by_team.items():
            self.team_last_roster[team] = players
        for side in ("home", "away"):
            team = game[f"{side}_team"]
            poss = possessions.get((game["game_id"], team))
            if poss:
                self.team_pace[team] = poss if team not in self.team_pace else (
                    (1 - PACE_ALPHA) * self.team_pace[team] + PACE_ALPHA * poss)


class RapmModel:
    """margin_home = hca + sum_home (min/48) r_i - sum_away (min/48) r_j, ridge(lam), decayed by half-life."""

    def __init__(self, lam: float, half_life_days: float, capacity: int = 2500):
        self.lam, self.half_life = lam, half_life_days
        self.n = capacity + 1                      # column 0 = home court
        self.A = np.zeros((self.n, self.n))
        self.b = np.zeros(self.n)
        self.index = PlayerIndex()
        self.ratings = np.zeros(self.n)
        self.dirty = False

    def Decay(self, days: float) -> None:
        if days > 0:
            f = 0.5 ** (days / self.half_life)
            self.A *= f
            self.b *= f

    def Apply(self, game: dict, player_rows: list) -> None:
        cols, vals = [0], [1.0]
        for r in player_rows:
            sign = 1.0 if r["team"] == game["home_team"] else -1.0 if r["team"] == game["away_team"] else 0.0
            if sign:
                cols.append(1 + self.index.Get(r["player_id"]))
                vals.append(sign * r["minutes"] / 48.0)
        cols, vals = np.array(cols), np.array(vals)
        margin = game["home_score"] - game["away_score"]
        self.A[np.ix_(cols, cols)] += np.outer(vals, vals)
        self.b[cols] += vals * margin
        self.dirty = True

    def Fit(self) -> None:
        if not self.dirty:
            return
        m = 1 + len(self.index.ids)
        reg = np.full(m, self.lam)
        reg[0] = 1e-3
        self.ratings[:m] = np.linalg.solve(self.A[:m, :m] + np.diag(reg), self.b[:m])
        self.dirty = False

    def Rating(self, player_id: int) -> float:
        i = self.index.ids.get(player_id)
        return 0.0 if i is None else float(self.ratings[1 + i])

    def PredictMargin(self, home_minutes: dict, away_minutes: dict) -> float:
        return (self.ratings[0] + sum(m / 48.0 * self.Rating(p) for p, m in home_minutes.items())
                - sum(m / 48.0 * self.Rating(p) for p, m in away_minutes.items()))


class MatchupModel:
    """
    Offender o guarded by defender d for c possessions scores y ~ Poisson(c * exp(alpha + a_o + b_d) * g_od).
      exp(a_o) ~ Gamma prior (mean 1, strength k): o's scoring per possession vs an average defender.
      exp(b_d) ~ same: d's suppression multiplier.  g_od: pair interaction, shrunk with strength k_pair.
    Assignment: share s(o, d) of o's possessions guarded by d =
      (c_hist(o,d) + K0 * s0(o,d)) / (sum_d c_hist + K0),  s0 ~ defender minutes x position kernel.
    """

    def __init__(self, k: float, half_life_days: float, k_pair: float = None, k_assign: float = 10.0,
                 pos_width: float = 0.6, capacity: int = 600_000, target: str = "player"):
        self.k, self.half_life = k, half_life_days
        self.target = target          # "player": offender's own points; "team": team points on those possessions
        self.k_pair = k if k_pair is None else k_pair
        self.k_assign, self.pos_width = k_assign, pos_width
        self.pair_of = {}
        self.o = np.zeros(capacity, dtype=np.int64)
        self.d = np.zeros(capacity, dtype=np.int64)
        self.c = np.zeros(capacity)
        self.y = np.zeros(capacity)
        self.m = 0
        self.index = PlayerIndex()
        self.pos_sum = np.zeros(4000)
        self.pos_n = np.zeros(4000)
        self.a = np.zeros(4000)
        self.bd = np.zeros(4000)
        self.alpha = math.log(0.25)
        self.attrib_team = 1.0          # decayed sum of team points; ratio to attributed points
        self.attrib_player = 1.0
        self.dirty = False
        self.pos = np.full(4000, 2.0)
        self.off_y = np.zeros(1)

    def Decay(self, days: float) -> None:
        if days > 0:
            f = 0.5 ** (days / self.half_life)
            self.c[:self.m] *= f
            self.y[:self.m] *= f
            self.pos_sum *= f
            self.pos_n *= f
            self.attrib_team *= f
            self.attrib_player *= f

    def Pair(self, off: int, dfn: int) -> int:
        key = (off, dfn)
        j = self.pair_of.get(key)
        if j is None:
            j = self.m
            self.pair_of[key] = j
            self.o[j], self.d[j] = self.index.Get(off), self.index.Get(dfn)
            self.m += 1
        return j

    def AddRows(self, rows: list, weight: float = 1.0) -> None:
        for r in rows:
            if r["poss"] <= 0:
                continue
            j = self.Pair(r["off"], r["def"])
            self.c[j] += weight * r["poss"]
            self.y[j] += weight * (r["team_pts"] if self.target == "team" and "team_pts" in r else r["pts"])
            code = POS_CODE.get(r.get("pos", ""))
            if code:
                i = self.index.Get(r["off"])
                self.pos_sum[i] += weight * code * r["poss"]
                self.pos_n[i] += weight * r["poss"]
        self.dirty = True

    def AddTeamPoints(self, team_points: float, attributed: float) -> None:
        self.attrib_team += team_points
        self.attrib_player += attributed

    def Fit(self, iterations: int = 4) -> None:
        if not self.dirty:
            return
        m, n = self.m, len(self.index.ids)
        o, d, c, y = self.o[:m], self.d[:m], self.c[:m], self.y[:m]
        self.alpha = math.log(max(y.sum(), 1e-9) / max(c.sum(), 1e-9))
        self.off_y = np.bincount(o, weights=y, minlength=n)
        k = self.k
        for _ in range(iterations):
            eb = np.exp(self.bd[d])
            num = np.bincount(o, weights=y, minlength=n) + k
            den = np.bincount(o, weights=c * math.exp(self.alpha) * eb, minlength=n) + k
            self.a[:n] = np.log(num / den)
            ea = np.exp(self.a[o])
            num = np.bincount(d, weights=y, minlength=n) + k
            den = np.bincount(d, weights=c * math.exp(self.alpha) * ea, minlength=n) + k
            self.bd[:n] = np.log(num / den)
        # position score: starters' listed position, then guarding history for everyone else
        known = self.pos_n[:n] > 0.5
        pos = np.full(n, 2.0)
        pos[known] = self.pos_sum[:n][known] / self.pos_n[:n][known]
        for _ in range(2):
            guarded = np.bincount(d, weights=c * pos[o], minlength=n) / np.maximum(np.bincount(d, weights=c, minlength=n), 1e-9)
            has = np.bincount(d, weights=c, minlength=n) > 5
            pos = np.where(known, pos, np.where(has, guarded, 2.0))
        self.pos[:n] = pos
        self.dirty = False

    def Lambda(self, off: int, dfn: int) -> float:
        io, idf = self.index.ids.get(off), self.index.ids.get(dfn)
        a = self.a[io] if io is not None else 0.0
        b = self.bd[idf] if idf is not None else 0.0
        lam = math.exp(self.alpha + a + b)
        j = self.pair_of.get((off, dfn))
        if j is not None:
            lam *= (self.y[j] + self.k_pair) / (self.c[j] * lam + self.k_pair)
        return lam

    def OffenseRate(self, off: int) -> float:
        io = self.index.ids.get(off)
        return math.exp(self.alpha + (self.a[io] if io is not None else 0.0))

    def Position(self, player: int) -> float:
        i = self.index.ids.get(player)
        return 2.0 if i is None else float(self.pos[i])

    def Shares(self, off: int, defenders: dict, mode: str = "full") -> dict:
        """defenders: {player: expected minutes}. Returns {defender: share}, summing to 1."""
        po = self.Position(off)
        prior = {d: m * math.exp(-((po - self.Position(d)) ** 2) / (2 * self.pos_width ** 2))
                 for d, m in defenders.items()}
        z = sum(prior.values()) or 1.0
        prior = {d: v / z for d, v in prior.items()}
        if mode == "prior":
            return prior
        hist = {}
        for d in defenders:
            j = self.pair_of.get((off, d))
            hist[d] = self.c[j] if j is not None else 0.0
        w = sum(hist.values())
        if mode == "hist":
            return {d: hist[d] / w for d in defenders} if w > 0 else prior
        shares = {d: (hist[d] + self.k_assign * prior[d]) / (w + self.k_assign) for d in defenders}
        z = sum(shares.values())
        return {d: s / z for d, s in shares.items()}

    def AttributionRatio(self) -> float:
        if self.target == "team":     # every possession appears once per on-court offender (5x)
            return 0.2
        return self.attrib_team / max(self.attrib_player, 1e-9)

    def ExpectedPoints(self, offense: dict, defense: dict, pace: float, detail: bool = False):
        """offense/defense: {player: expected minutes}. Returns expected team points (and pair detail)."""
        total, pairs = 0.0, []
        for o, mins in offense.items():
            poss_o = mins / 48.0 * pace
            for d, s in self.Shares(o, defense).items():
                lam = self.Lambda(o, d)
                total += poss_o * s * lam
                if detail:
                    pairs.append((o, d, poss_o * s, lam))
        total *= self.AttributionRatio()
        return (total, pairs) if detail else total
