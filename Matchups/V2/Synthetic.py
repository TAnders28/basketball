r"""
Synthetic league generator for unit tests: a small, internally consistent fake season with every table an
approach can consume (games, box, matchups, possessions, stints, shots, events). Truth is known (team and
player strengths), so tests can check recovery, calibration and posterior predictive coverage.
"""

import math
from datetime import date, timedelta

import numpy as np

from Matchups.V2.Base import GameRecord

ZONES = ["Restricted Area", "In The Paint (Non-RA)", "Mid-Range", "Above the Break 3", "Left Corner 3"]


def MakeLeague(n_teams=6, n_players=9, n_days=24, games_per_day=3, seed=0, season=2024, team_sd=4.0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    real = ["ATL", "BOS", "CHI", "DEN", "MIA", "PHX", "LAL", "NYK", "DAL", "UTA"]
    teams = real[:n_teams]
    players = {t: [1000 * (i + 1) + j for j in range(n_players)] for i, t in enumerate(teams)}
    skill = {p: rng.normal(0, 2.0) for t in teams for p in players[t]}
    team_strength = {t: rng.normal(0, team_sd) for t in teams}
    games, recs = [], {}
    gid = 0
    for day in range(n_days):
        day_s = (date(season, 11, 1) + timedelta(days=day)).isoformat()
        order = rng.permutation(teams)
        for k in range(games_per_day):
            home, away = order[2 * k], order[2 * k + 1]
            gid += 1
            game_id = f"00224{gid:05d}"
            recs[game_id] = _Game(rng, game_id, day_s, season, home, away, players, skill, team_strength)
            games.append(recs[game_id].game)
    return games, recs, {"team_strength": team_strength, "skill": skill, "players": players}


def _Game(rng, game_id, day_s, season, home, away, players, skill, team_strength):
    starters = {t: players[t][:5] for t in (home, away)}
    bench = {t: players[t][5:] for t in (home, away)}
    stints, possessions, box = [], [], {}
    score = {home: 0, away: 0}
    t0 = 0
    for k in range(8):
        fives = {t: tuple(sorted(starters[t] if k % 2 == 0 else list(starters[t][:2]) + list(rng.choice(bench[t], 3, replace=False))))
                 for t in (home, away)}
        st = {"start": t0, "end": t0 + 3600, "seconds": 360.0, "home5": fives[home], "away5": fives[away],
              "home_pts": 0, "away_pts": 0, "home_poss": 0, "away_poss": 0}
        for i in range(24):
            off = home if i % 2 == 0 else away
            dfn = away if off == home else home
            edge = (team_strength[off] - team_strength[dfn] + (1.5 if off == home else -1.5)
                    + sum(skill[p] for p in fives[off]) / 5 - sum(skill[p] for p in fives[dfn]) / 5) / 100.0
            tov = rng.random() < 0.13
            pts, shots, oreb = 0, [], 0
            if not tov:
                three = rng.random() < 0.38
                p_make = (0.36 if three else 0.53) + edge
                made = rng.random() < p_make
                shooter = int(rng.choice(fives[off]))
                shots.append({"player": shooter, "value": 3 if three else 2, "made": bool(made), "dist": 25 if three else 5, "sub": ""})
                pts = (3 if three else 2) if made else 0
                if not made and rng.random() < 0.25:
                    oreb = 1
                    if rng.random() < 0.5:
                        pts = 2
                        shots.append({"player": shooter, "value": 2, "made": True, "dist": 2, "sub": "putback"})
            margin = score[home] - score[away]
            poss = {"offense": off, "defense": dfn, "start": t0 + i * 150, "end": t0 + i * 150 + 140,
                    "start_type": "after_dreb" if i % 3 == 0 else "after_make", "pts": pts,
                    "fga": len(shots), "fg3a": sum(s["value"] == 3 for s in shots), "fta": 0, "oreb": oreb,
                    "tov": int(tov), "first_shot_t": None, "assisted": pts > 0 and rng.random() < 0.6,
                    "assister": int(fives[off][0]) if pts > 0 else None, "scorer": shots[-1]["player"] if pts > 0 else None,
                    "clutch": k == 7 and abs(margin) <= 5, "margin_start_home": margin, "off5": fives[off],
                    "def5": fives[dfn], "shots": shots, "transition": i % 3 == 0 and rng.random() < 0.4}
            possessions.append(poss)
            score[off] += pts
            st["home_pts" if off == home else "away_pts"] += pts
            st["home_poss" if off == home else "away_poss"] += 1
        stints.append(st)
        t0 += 3600
    for st in stints:
        for team, key in ((home, "home5"), (away, "away5")):
            for p in st[key]:
                b = box.setdefault(p, {"player_id": p, "team": team, "name": f"P{p}", "minutes": 0.0, "pts": 0, "fgm": 0,
                                       "fga": 0, "fg3m": 0, "fg3a": 0, "ftm": 0, "fta": 0, "oreb": 0, "dreb": 0, "ast": 0,
                                       "stl": 0, "blk": 0, "tov": 0, "pf": 0, "plus_minus": 0})
                b["minutes"] += 6.0
    for poss in possessions:
        for s in poss["shots"]:
            b = box[s["player"]]
            b["fga"] += 1
            b["fgm"] += s["made"]
            b["fg3a"] += s["value"] == 3
            b["fg3m"] += s["value"] == 3 and s["made"]
            b["pts"] += s["value"] if s["made"] else 0
        if poss["tov"]:
            box[poss["off5"][0]]["tov"] += 1
        box[poss["def5"][-1]]["dreb"] += 1 if not poss["tov"] and poss["pts"] == 0 else 0
        if poss["assister"]:
            box[poss["assister"]]["ast"] += 1
    matchups = []
    for poss in possessions:
        for i, o in enumerate(poss["off5"]):
            d = poss["def5"][i]
            pts = poss["pts"] if poss["scorer"] == o else 0
            matchups.append({"team": poss["offense"], "off": o, "def": d, "poss": 1.0, "pts": float(pts),
                             "team_pts": float(poss["pts"]), "switches": float(rng.random() < 0.1), "fga": 0.0, "fg3a": 0.0, "pos": ""})
    shots = []
    for poss in possessions:
        for s in poss["shots"]:
            zone = ZONES[3] if s["value"] == 3 else ZONES[0]
            shots.append({"team": poss["offense"], "player_id": s["player"], "zone": zone, "area": "", "dist": s["dist"],
                          "three": s["value"] == 3, "action": "", "made": int(s["made"]), "period": 1})
    game = {"game_id": game_id, "game_date": day_s, "season": season, "home_team": home, "away_team": away,
            "home_score": score[home], "away_score": score[away]}
    events = {home: {"timeouts": int(rng.integers(4, 8))}, away: {"timeouts": int(rng.integers(4, 8))}}
    return GameRecord(game, box=list(box.values()), matchups=matchups, possessions=possessions, stints=stints,
                      shots=shots, events=events, pbp=[{"x": 1}], rotation=None)


def Perturb(recs: dict, from_date: str, seed=1):
    """Copy of recs where every game on/after from_date gets different scores and possession points."""
    rng = np.random.default_rng(seed)
    out = {}
    for gid, r in recs.items():
        if r.game["game_date"] >= from_date:
            g = dict(r.game)
            g["home_score"], g["away_score"] = int(rng.integers(70, 140)), int(rng.integers(70, 140))
            poss = [dict(p, pts=int(rng.integers(0, 4))) for p in r.possessions]
            st = [dict(s, home_pts=int(rng.integers(0, 60)), away_pts=int(rng.integers(0, 60))) for s in r.stints]
            mu = [dict(m, pts=float(rng.integers(0, 4))) for m in (r.matchups or [])]
            sh = [dict(s, made=int(rng.random() < 0.5)) for s in r.shots]
            out[gid] = GameRecord(g, box=r.box, matchups=mu, possessions=poss, stints=st, shots=sh,
                                  events=r.events, pbp=r.pbp, rotation=None)
        else:
            out[gid] = r
    return out
