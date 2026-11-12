r"""
Availability scenarios (reading: availability-aware team strength, EPM / NOVA style).

StatusModel learns, from games already played, P(player plays next team game | k consecutive team games missed)
with a Beta-binomial per bucket of k (0, 1, 2, 3, 4-9, 10+), pooled league-wide. There is no injury-report feed,
so this is the "status probability" stand-in: it is calibrated on history and walk-forward safe.

Scenarios: the most uncertain rotation players (by expected minutes) are enumerated (2^k, k <= 6); the rest are
set to their most likely state. Each scenario redistributes minutes over the players present (pro rata on their
usual minutes, capped at 42, total 240), the team rating is computed per scenario, and the mixture gives the
scenario-averaged mean and variance (law of total variance).

OracleAvailability(game) returns who actually played (from the box score). It is NOT pregame-knowable in general
(late scratches), so any model run with it is labelled "@oracle" and reported separately.
"""

from collections import defaultdict
from itertools import product

import numpy as np

from Matchups.V2 import DataLayer as D
from Matchups.V2.Bayes import BetaBinomial

BUCKETS = (0, 1, 2, 3, 4, 10)
PRIOR_PLAY = {0: 0.93, 1: 0.55, 2: 0.45, 3: 0.4, 4: 0.3, 10: 0.1}


def Bucket(k):
    b = 0
    for x in BUCKETS:
        if k >= x:
            b = x
    return b


class StatusModel:
    def __init__(self, window=15, max_enum=6):
        self.p_play = {b: BetaBinomial(PRIOR_PLAY[b], 30.0) for b in BUCKETS}
        self.missed = defaultdict(int)            # (team, player) -> consecutive team games missed
        self.seen = defaultdict(int)              # (team, player) -> team games since last appearance
        self.mins = defaultdict(lambda: 24.0)     # player -> EW minutes when he plays
        self.window, self.max_enum = window, max_enum
        self.diag = []                            # (predicted p, played) for calibration reporting

    def Roster(self, team):
        return [p for (t, p), s in self.seen.items() if t == team and s < self.window]

    def Probs(self, team):
        return {p: self.p_play[Bucket(self.missed[(team, p)])].Mean() for p in self.Roster(team)}

    def Update(self, team, box_rows):
        played = {r["player_id"]: r["minutes"] for r in box_rows if r["team"] == team and (r["minutes"] or 0) > 0}
        for p in self.Roster(team):
            b = Bucket(self.missed[(team, p)])
            hit = p in played
            self.diag.append((self.p_play[b].Mean(), int(hit)))
            self.p_play[b].Add(int(hit), 1)
        for p in self.Roster(team):
            if p not in played:
                self.missed[(team, p)] += 1
                self.seen[(team, p)] += 1
        for p, m in played.items():
            # moving teams: forget the old team entry
            for key in [k for k in self.seen if k[1] == p and k[0] != team]:
                self.seen.pop(key, None)
                self.missed.pop(key, None)
            self.missed[(team, p)] = 0
            self.seen[(team, p)] = 0
            self.mins[p] += 0.3 * (m - self.mins[p])

    def Minutes(self, present):
        m = {p: max(self.mins[p], 1.0) for p in present}
        for _ in range(10):
            tot = sum(m.values())
            if tot <= 0:
                return {}
            m = {p: min(v * 240.0 / tot, 42.0) for p, v in m.items()}
            if abs(sum(m.values()) - 240.0) < 0.5:
                break
        return m

    def Scenarios(self, team):
        """[(weight, {player: minutes})] over availability scenarios."""
        probs = self.Probs(team)
        if not probs:
            return []
        certain = [p for p, q in probs.items() if q >= 0.97]
        unsure = sorted([p for p, q in probs.items() if 0.03 < q < 0.97], key=lambda p: -self.mins[p] * probs[p])
        enum, rest = unsure[:self.max_enum], unsure[self.max_enum:]
        base = certain + [p for p in rest if probs[p] >= 0.5]
        out = []
        for states in product((0, 1), repeat=len(enum)):
            w = 1.0
            present = list(base)
            for p, s in zip(enum, states):
                w *= probs[p] if s else 1 - probs[p]
                if s:
                    present.append(p)
            if w < 1e-4 or len(present) < 5:
                continue
            out.append((w, self.Minutes(present)))
        z = sum(w for w, _ in out) or 1.0
        return [(w / z, m) for w, m in out]


def Mixture(parts):
    """[(weight, mean, var)] -> mixture mean and variance."""
    if not parts:
        return 0.0, 0.0
    w = np.array([a for a, _, _ in parts])
    m = np.array([b for _, b, _ in parts])
    v = np.array([c for _, _, c in parts])
    mean = float(w @ m)
    return mean, float(w @ (v + m * m) - mean * mean)


def OracleAvailability(game):
    """ORACLE: players who actually played in this game -> {team: {player: 1.0}}. Labelled @oracle in reports."""
    out = defaultdict(dict)
    for r in D.PlayerBox(game["season"]).get(game["game_id"], []):
        if (r["minutes"] or 0) > 0:
            out[r["team"]][r["player_id"]] = 1.0
    return dict(out)
