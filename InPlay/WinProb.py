"""In-game win-probability model: leak-free PBP features, GBM, and a possession MC grid.

Pregame prior preference: Kalshi T-60 de-vig mid, else stack_global, else a de-vigged
book/pickcenter moneyline, each flagged. ESPN files and InPlay loaders are reused; no fetches.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys

# Cap BLAS/OpenMP before numpy loads.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np

from InPlay.Align import EspnStates, IsoTs, SeasonOf

REG_END = 2880.0
EPS = 1e-6
N_WORKERS = max(1, (os.cpu_count() or 8) // 4)
FT_RE = re.compile(r"Free Throw.*(\d+)\s+of\s+(\d+)", re.I)
TIMEOUTS_START = 7  # NBA regular-season allotment; used as a cap on remaining


def ClipP(p, lo=EPS, hi=1.0 - EPS) -> float:
    return min(max(float(p), lo), hi)


def Logit(p) -> float:
    p = ClipP(p)
    return math.log(p / (1.0 - p))


def Sigmoid(z) -> float:
    z = float(np.clip(z, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-z))


def LogLoss(p, y) -> float:
    p = ClipP(p)
    return -math.log(p if y else (1.0 - p))


def Brier(p, y) -> float:
    return (float(p) - float(y)) ** 2


def ImpliedProb(american) -> float:
    a = float(american)
    if a > 0:
        return 100.0 / (a + 100.0)
    return (-a) / ((-a) + 100.0)


def DevigMl(home_ml, away_ml):
    try:
        h, a = ImpliedProb(home_ml), ImpliedProb(away_ml)
    except (TypeError, ValueError):
        return None
    s = h + a
    if s <= 0:
        return None
    return ClipP(h / s)


def SetNormalPriority() -> None:
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00000020)
    except Exception:
        pass


def LimitThreads() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def PhaseOf(el, per) -> str:
    if el is None:
        return "unknown"
    if int(per or 0) >= 5:
        return "OT"
    rem = max(0.0, REG_END - float(el))
    if rem <= 300 and int(per or 0) >= 4:
        return "last5"
    return f"Q{min(max(int(per or 1), 1), 4)}"


# --- play classification -------------------------------------------------------

def _Type(pl) -> str:
    return (pl.get("type") or "").replace("\n", " ").strip()


def IsTimeout(typ: str) -> bool:
    t = typ.lower()
    return "timeout" in t and "official" not in t


def IsOfficialTimeout(typ: str) -> bool:
    return "official" in typ.lower() and "timeout" in typ.lower()


def IsTurnover(typ: str) -> bool:
    return "turnover" in typ.lower()


def IsOffensiveFoul(typ: str) -> bool:
    t = typ.lower()
    return "offensive foul" in t


def IsTeamFoul(typ: str) -> bool:
    """Personal/shooting/loose-ball/technical count toward the bonus; offensive do not."""
    t = typ.lower()
    if "foul" not in t:
        return False
    if "offensive foul" in t:
        return False
    if "technical" in t and "offensive" in t:
        return False
    return True


def IsDefRebound(typ: str) -> bool:
    return typ.lower().startswith("defensive rebound")


def IsOffRebound(typ: str) -> bool:
    return typ.lower().startswith("offensive rebound")


def IsJumpball(typ: str) -> bool:
    return "jumpball" in typ.lower().replace(" ", "") or typ.lower() == "jump ball"


def FtIndex(typ: str):
    m = FT_RE.search(typ)
    if not m:
        if typ.lower().startswith("free throw"):
            return (1, 1)  # technical / 1 of 1 style
        return None
    return int(m.group(1)), int(m.group(2))


def IsLastFt(typ: str) -> bool:
    ix = FtIndex(typ)
    return bool(ix) and ix[0] == ix[1]


def SideOf(team, home_id, away_id):
    if team is None or team == "":
        return None
    t = str(team)
    if t == str(home_id):
        return "home"
    if t == str(away_id):
        return "away"
    return None


def Opponent(side):
    return {"home": "away", "away": "home"}.get(side)


# --- leak-free state walk ------------------------------------------------------

def WalkPlayState(game: dict) -> list:
    """Per-play state after each play (possession, team fouls, timeouts, run).

    Same-play updates are applied in sequence; a play never sees a later play.
    """
    home_id, away_id = game.get("home_id"), game.get("away_id")
    states = EspnStates(game)
    # Map wall+seq states back onto the raw monotone plays for type/team.
    from InPlay.Align import MonotoneWall, Elapsed
    raw = []
    for ts, p in MonotoneWall(game.get("plays") or []):
        el = Elapsed(p.get("period"), p.get("clock"))
        if el is None:
            continue
        raw.append((ts, p, el))
    # EspnStates filters the same way; lengths should match. If not, align by (t,el,hs,as).
    if len(raw) != len(states):
        index = {(s["t"], s["el"], s["hs"], s["as"]): s for s in states}
        use = []
        for ts, p, el in raw:
            s = index.get((ts, el, p.get("hs") or 0, p.get("as") or 0))
            if s is not None:
                use.append((ts, p, s))
        pairs = use
    else:
        pairs = [(ts, p, s) for (ts, p, el), s in zip(raw, states)]

    poss = None
    tf = {"home": 0, "away": 0}
    to_used = {"home": 0, "away": 0}
    period = None
    out = []
    for ts, p, s in pairs:
        typ = _Type(p)
        per = int(p.get("period") or s["per"])
        if period is None or per != period:
            # New period: team fouls reset. Possession unknown until jump / first action.
            tf = {"home": 0, "away": 0}
            poss = None
            period = per
        side = SideOf(p.get("team"), home_id, away_id)

        # Attribute this play, then record the *after* state (known at the end of the play).
        if IsTeamFoul(typ) and side in tf:
            tf[side] += 1
        if IsTimeout(typ) and not IsOfficialTimeout(typ) and side in to_used:
            to_used[side] += 1

        # Possession update (after the play).
        if IsJumpball(typ) and side in ("home", "away"):
            poss = side
        elif IsDefRebound(typ) and side in ("home", "away"):
            poss = side
        elif IsOffRebound(typ) and side in ("home", "away"):
            poss = side
        elif IsTurnover(typ) and side in ("home", "away"):
            poss = Opponent(side)
        elif typ.lower().startswith("free throw") or "free throw" in typ.lower():
            if IsLastFt(typ):
                made = bool(p.get("scoring"))
                if made:
                    poss = Opponent(side) if side else poss
                # miss: wait for rebound
            else:
                if side in ("home", "away"):
                    poss = side
        elif p.get("shoot") and p.get("scoring") and side in ("home", "away"):
            # Made FG. And-1 (shooting foul next) is resolved on later plays; we flip now
            # and a subsequent shooting foul / FT will set possession back.
            poss = Opponent(side)
        elif p.get("shoot") and not p.get("scoring"):
            # Missed FG: possession unresolved until rebound.
            pass

        bonus = {k: int(tf[k] >= 5) for k in ("home", "away")}
        to_left = {k: max(0, TIMEOUTS_START - to_used[k]) for k in ("home", "away")}
        row = {
            "t": s["t"], "el": s["el"], "per": per,
            "hs": s["hs"], "as": s["as"], "margin": int(s["hs"]) - int(s["as"]),
            "wp_espn": s.get("wp"), "run": s.get("run", 0),
            "type": typ, "team_side": side,
            "poss": poss, "poss_unknown": int(poss is None),
            "home_tf": tf["home"], "away_tf": tf["away"],
            "home_bonus": bonus["home"], "away_bonus": bonus["away"],
            "home_to_used": to_used["home"], "away_to_used": to_used["away"],
            "home_to_left": to_left["home"], "away_to_left": to_left["away"],
            "scoring": bool(p.get("scoring")), "sv": int(p.get("sv") or 0),
            "shoot": bool(p.get("shoot")),
        }
        out.append(row)
    return out


def PickcenterHomeProb(game: dict):
    """De-vigged home ML from ESPN pickcenter, preferring consensus then any provider."""
    pc = game.get("pickcenter") or []
    if not isinstance(pc, list):
        return None
    ordered = sorted(pc, key=lambda x: 0 if str(x.get("provider", "")).lower() == "consensus" else 1)
    for row in ordered:
        if not isinstance(row, dict):
            continue
        if row.get("homeML") in (None, "") or row.get("awayML") in (None, ""):
            continue
        p = DevigMl(row["homeML"], row["awayML"])
        if p is not None:
            return p
    return None


def FeatureRow(state, prior, prior_source, y):
    el = float(state["el"])
    rem_s = max(0.0, REG_END - el) if int(state["per"]) <= 4 else max(0.0, 300.0 - ((el - REG_END) % 300.0))
    rem_m = rem_s / 60.0
    margin = float(state["margin"])
    z = margin / math.sqrt(rem_m + 1.0)
    poss = state.get("poss")
    return {
        "logit_prior": Logit(prior),
        "margin": margin,
        "margin_z": z,
        "min_left": rem_m,
        "period": float(state["per"]),
        "poss_home": 1.0 if poss == "home" else 0.0,
        "poss_away": 1.0 if poss == "away" else 0.0,
        "poss_unknown": 1.0 if poss not in ("home", "away") else 0.0,
        "home_bonus": float(state["home_bonus"]),
        "away_bonus": float(state["away_bonus"]),
        "foul_diff": float(state["home_tf"] - state["away_tf"]),
        "to_diff": float(state["home_to_left"] - state["away_to_left"]),
        "home_to_left": float(state["home_to_left"]),
        "away_to_left": float(state["away_to_left"]),
        "run": float(state.get("run") or 0),
        "prior_is_kalshi": 1.0 if prior_source == "kalshi" else 0.0,
        "prior_is_stack": 1.0 if prior_source == "stack_global" else 0.0,
        "prior_is_book": 1.0 if prior_source == "book" else 0.0,
        "y": int(y),
        "prior": float(prior),
        "prior_source": prior_source,
        "phase": PhaseOf(el, state["per"]),
        "el": el, "t": state["t"], "per": state["per"],
        "wp_espn": state.get("wp_espn"),
    }


FEATURE_NAMES = [
    "logit_prior", "margin", "margin_z", "min_left", "period",
    "poss_home", "poss_away", "poss_unknown",
    "home_bonus", "away_bonus", "foul_diff",
    "to_diff", "home_to_left", "away_to_left", "run",
    "prior_is_kalshi", "prior_is_stack", "prior_is_book",
]


def LoadPriors():
    """Join Kalshi T-60, stack_global, and book/pickcenter keys on (date, home, away)."""
    import pickle
    from pathlib import Path
    from collections import defaultdict

    pri = {}  # (date, home, away) -> list of (rank, source, p)
    # Kalshi k_pre from the existing in-play games table (s0, no PM shift).
    gpath = Path("data/inplay/games_s0.pkl")
    if gpath.exists():
        games = pickle.loads(gpath.read_bytes())
        for r in games.itertuples(index=False):
            p = getattr(r, "k_pre", None)
            if p is not None and p == p:
                pri.setdefault((str(r.date_et)[:10], r.home, r.away), []).append((0, "kalshi", float(p)))

    # stack_global walk-forward pregame (NBA ids; join on date+teams).
    spath = Path("data/inplay/stack_global.json")
    if spath.exists():
        sg = json.loads(spath.read_text(encoding="utf-8"))
        for r in sg.get("rows") or []:
            key = (str(r["game_date"])[:10], r["home"], r["away"])
            pri.setdefault(key, []).append((1, "stack_global", float(r["p_home"])))

    # Closing/book ML on the games table when present (2025-26 onward).
    if gpath.exists():
        games = pickle.loads(gpath.read_bytes())
        for r in games.itertuples(index=False):
            hml, aml = getattr(r, "book_home_ml", None), getattr(r, "book_away_ml", None)
            if hml is None or aml is None:
                continue
            try:
                if hml != hml or aml != aml:
                    continue
            except Exception:
                continue
            p = DevigMl(hml, aml)
            if p is not None:
                pri.setdefault((str(r.date_et)[:10], r.home, r.away), []).append((2, "book", p))
    return pri


def ResolvePrior(date, home, away, game_json, priors):
    """Return (p, source). Prefer Kalshi, then stack_global, then book, then pickcenter."""
    from InPlay.Align import Team
    try:
        home_a = Team(home) if home and not (isinstance(home, float)) else home
        away_a = Team(away) if away and not (isinstance(away, float)) else away
    except Exception:
        home_a, away_a = home, away
    # ESPN games store team as abbr already often
    keys = []
    for h, a in ((home, away), (home_a, away_a)):
        if h and a and date:
            keys.append((str(date)[:10], str(h), str(a)))
    cands = []
    for k in keys:
        cands.extend(priors.get(k) or [])
    if cands:
        cands.sort(key=lambda x: x[0])
        return cands[0][2], cands[0][1]
    p = PickcenterHomeProb(game_json) if game_json else None
    if p is not None:
        return p, "book"
    return None, None


def GameDateAndTeams(game, path=None):
    from InPlay.Align import EtDate, Team
    ts = IsoTs(game.get("date"))
    day = EtDate(ts) if ts else None
    home, away = game.get("home"), game.get("away")
    try:
        home = Team(home) if home else home
        away = Team(away) if away else away
    except Exception:
        pass
    season = SeasonOf(ts) if ts else None
    won = game.get("home_won")
    return day, home, away, season, won, ts


def ParseOneEspn(path, priors):
    game = json.loads(open(path, encoding="utf-8").read())
    if game.get("home_won") is None:
        return None
    day, home, away, season, won, ts = GameDateAndTeams(game, path)
    prior, source = ResolvePrior(day, home, away, game, priors)
    if prior is None:
        # Last-resort: ESPN opening WP, flagged. Used only so long-history GBM
        # has a prior; market comparisons drop these via source flag.
        states0 = []
        try:
            from InPlay.Align import EspnStates
            st = EspnStates(game)
            for s in st:
                if s.get("wp") is not None:
                    prior, source = float(s["wp"]), "espn_wp"
                    break
        except Exception:
            prior = None
    if prior is None:
        return None
    states = WalkPlayState(game)
    if len(states) < 50:
        return None
    y = int(bool(won))
    rows = []
    # One row per ~30s of game time to keep the train set manageable and
    # less auto-correlated than every play, plus the last play.
    last_el = -999
    for i, s in enumerate(states):
        take = (s["el"] - last_el >= 30) or (i == len(states) - 1)
        if not take:
            continue
        last_el = s["el"]
        fr = FeatureRow(s, prior, source, y)
        fr.update({
            "game_id": str(game.get("id")),
            "date": day, "home": home, "away": away,
            "season": season, "path": str(path),
        })
        rows.append(fr)
    return {
        "game_id": str(game.get("id")),
        "date": day, "home": home, "away": away,
        "season": int(season) if season else None,
        "won": y, "prior": prior, "prior_source": source,
        "n_states": len(states), "rows": rows, "states": states,
        "tip_ts": states[0]["t"] if states else None,
        "end_ts": states[-1]["t"] if states else None,
    }


_PRIORS = None


def _InitParse(priors):
    global _PRIORS
    LimitThreads()
    SetNormalPriority()
    _PRIORS = priors


def _ParsePath(path):
    try:
        return ParseOneEspn(path, _PRIORS)
    except Exception:
        return None


def ListEspnPaths():
    return sorted(glob.glob(os.path.join("data", "inplay", "espn", "games", "*", "*.json")))


def ParseAllEspn(paths=None, priors=None, workers=N_WORKERS):
    from concurrent.futures import ProcessPoolExecutor
    paths = paths or ListEspnPaths()
    priors = priors if priors is not None else LoadPriors()
    out = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers)),
                             initializer=_InitParse, initargs=(priors,)) as ex:
        for i, rec in enumerate(ex.map(_ParsePath, paths, chunksize=20), 1):
            if rec is not None:
                out.append(rec)
            if i % 1000 == 0:
                print(f"parsed {i}/{len(paths)} kept={len(out)}", flush=True)
    out.sort(key=lambda r: (r["date"] or "", r["game_id"]))
    return out


def MatrixFromRows(rows, names=FEATURE_NAMES):
    X = np.zeros((len(rows), len(names)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    init = np.zeros(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, n in enumerate(names):
            X[i, j] = float(r.get(n) or 0.0)
        y[i] = float(r["y"])
        init[i] = float(r["logit_prior"])
    return X, y, init


def FitGbm(X, y, init, params=None):
    import lightgbm as lgb
    params = params or {}
    dtrain = lgb.Dataset(X, label=y, init_score=init, free_raw_data=False,
                         feature_name=FEATURE_NAMES)
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_threads": 1,
        "feature_pre_filter": False,
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "num_leaves": int(params.get("num_leaves", 31)),
        "min_child_samples": int(params.get("min_child_samples", 80)),
        "max_depth": int(params.get("max_depth", 6)),
        "lambda_l2": float(params.get("lambda_l2", 1.0)),
        "bagging_freq": 0,
    }
    n_est = int(params.get("n_estimators", 200))
    return lgb.train(lgb_params, dtrain, num_boost_round=n_est)


def PredictGbm(booster, X, init):
    raw = booster.predict(X, raw_score=True)
    return 1.0 / (1.0 + np.exp(-np.clip(init + raw, -40.0, 40.0)))


def WalkForwardGbm(games, params=None):
    """Game-grouped walk-forward by season (ending year). Train only on earlier seasons."""
    by = {}
    for g in games:
        by.setdefault(g["season"], []).append(g)
    seasons = sorted(k for k in by if k)
    assigned = {}
    models = {}
    for s in seasons:
        train_games = [g for t in seasons if t < s for g in by[t]]
        test_games = by[s]
        if not train_games:
            # Burn-in season: no model yet.
            for g in test_games:
                g["p_gbm_rows"] = None
            continue
        rows = [r for g in train_games for r in g["rows"]]
        X, y, init = MatrixFromRows(rows)
        booster = FitGbm(X, y, init, params)
        models[s] = booster
        trows = [r for g in test_games for r in g["rows"]]
        if not trows:
            continue
        Xt, _, initt = MatrixFromRows(trows)
        pred = PredictGbm(booster, Xt, initt)
        k = 0
        for g in test_games:
            n = len(g["rows"])
            ps = pred[k:k + n]
            for r, p in zip(g["rows"], ps):
                r["p_gbm"] = float(p)
            g["p_gbm_rows"] = ps
            k += n
        print(f"wf season={s} train_games={len(train_games)} test={len(test_games)}", flush=True)
    return models


# --- possession-level Monte Carlo grid ----------------------------------------

TIME_BINS = [
    (0, 30), (30, 60), (60, 120), (120, 180), (180, 240), (240, 300),
    (300, 360), (360, 480), (480, 600), (600, 720),
    (720, 960), (960, 1200), (1200, 1440), (1440, 1920), (1920, 2400), (2400, 2880),
]


def TimeBin(rem_s: float) -> int:
    rem = max(0.0, min(float(rem_s), 2879.9))
    for i, (a, b) in enumerate(TIME_BINS):
        if a <= rem < b:
            return i
    return len(TIME_BINS) - 1


def EstimateRates(games):
    """League possession rates from training-season PBP.

    ESPN often labels made threes as a generic 'Jump Shot' with sv=3, and missed
    threes the same way with sv=0, so 3PA cannot be counted from type+sv alone.
    We count turnovers, FT trips, ORBs and made 2s/3s, then back out attempt
    mix from league-typical FG percentages (fg2=0.54, fg3=0.36). Pace is
    regulation seconds over counted possessions per game, not play-to-play gaps
    (those include subs and fouls and are too short).
    """
    n_to = n_2 = n_3 = n_orb = n_ft_trip = n_miss_any = 0
    poss_by_game = {}
    for g in games:
        gid = g["game_id"]
        n_poss = 0
        for s in g["states"]:
            typ = (s.get("type") or "").lower()
            if "turnover" in typ:
                n_to += 1
                n_poss += 1
            elif s.get("shoot") and "free throw" not in typ:
                n_poss += 1
                sv = int(s.get("sv") or 0)
                if s.get("scoring") and sv >= 3:
                    n_3 += 1
                elif s.get("scoring"):
                    n_2 += 1
                else:
                    n_miss_any += 1
            elif typ.startswith("offensive rebound"):
                n_orb += 1
            elif typ.startswith("free throw") and "1 of" in typ:
                n_ft_trip += 1
        poss_by_game[gid] = poss_by_game.get(gid, 0) + n_poss
    fg2, fg3 = 0.54, 0.36
    a2 = n_2 / fg2
    a3 = n_3 / fg3
    n_poss = n_to + a2 + a3
    # ORB rate over implied missed FGs
    miss = max(1.0, (a2 - n_2) + (a3 - n_3))
    orb = n_orb / miss
    mean_poss = float(np.mean(list(poss_by_game.values()))) if poss_by_game else 100.0
    # Counted possessions above use FGA+TO only (no 0.44*FTA). Scale to a full game.
    sec = 2880.0 / max(80.0, mean_poss)
    sec = min(18.0, max(12.0, sec))
    return {
        "p_to": n_to / max(1.0, n_poss),
        "p_3att": a3 / max(1.0, n_poss),
        "p_2att": a2 / max(1.0, n_poss),
        "fg2": fg2,
        "fg3": fg3,
        "orb": min(0.40, max(0.18, orb)),
        "p_ft": n_ft_trip / max(1.0, n_poss),
        "sec_per_poss": sec,
        "ft_pts": 1.5,
        "n_train_games": len(poss_by_game),
        "made2": n_2, "made3": n_3, "tos": n_to,
    }


def _HomeAdvantagePpp(prior, rates):
    """Convert a home pregame WP into a points-per-possession edge."""
    # Margin ~ N(mu, 12); mu such that P(home win) = prior.
    from math import sqrt
    # inverse normal via scipy if present, else a rational approx
    p = ClipP(prior, 0.01, 0.99)
    try:
        import scipy.stats
        z = float(scipy.stats.norm.ppf(p))
    except Exception:
        # Abramowitz & Stegun approximation
        t = math.sqrt(-2.0 * math.log(min(p, 1 - p)))
        z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
            1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
        if p < 0.5:
            z = -z
    mu = z * 12.0  # expected margin
    n_poss = 2880.0 / rates["sec_per_poss"]
    return mu / max(n_poss, 50.0)


def SimulateCell(rem_s, poss_home, bonus_home, rates, n_sims, ppp_home_edge, rng):
    """Vectorized remaining-game sims. Returns P(home win) for margins -20..20.

    One call covers all margins: we simulate the *change* in margin from 0.
    """
    sec = max(1.0, float(rates["sec_per_poss"]))
    n_poss = int(math.ceil(max(0.0, rem_s) / sec))
    n_poss = min(n_poss, 160)
    # Offensive efficiency: league ~1.10 PPP implied by fg mix, plus edge on home possessions.
    p_to = ClipP(rates["p_to"], 0.05, 0.25)
    p_3 = ClipP(rates["p_3att"], 0.15, 0.55)
    p_ft = ClipP(rates["p_ft"], 0.05, 0.30)
    # renormalize attempt mix on non-TO possessions later
    fg2, fg3 = ClipP(rates["fg2"], 0.35, 0.65), ClipP(rates["fg3"], 0.25, 0.50)
    orb = ClipP(rates["orb"], 0.15, 0.40)
    ft_pts = float(rates.get("ft_pts", 1.5))
    # Mild bonus bump: in bonus, more FT trips.
    if bonus_home:
        p_ft = min(0.40, p_ft + 0.04)

    margin = np.zeros(n_sims, dtype=np.int16)
    poss = np.full(n_sims, 1 if poss_home else 0, dtype=np.int8)  # 1=home
    for _ in range(n_poss):
        u = rng.random(n_sims)
        # turnover
        is_to = u < p_to
        u2 = rng.random(n_sims)
        # among non-TO: FT / 3 / 2
        rest = ~is_to
        is_ft = rest & (u2 < p_ft)
        is_3 = rest & (~is_ft) & (u2 < p_ft + p_3)
        is_2 = rest & (~is_ft) & (~is_3)
        pts = np.zeros(n_sims, dtype=np.float64)
        # 2pt
        make2 = is_2 & (rng.random(n_sims) < fg2)
        pts = pts + np.where(make2, 2.0, 0.0)
        # 3pt
        make3 = is_3 & (rng.random(n_sims) < fg3)
        pts = pts + np.where(make3, 3.0, 0.0)
        # FT trip
        pts = pts + np.where(is_ft, rng.binomial(2, min(0.85, max(0.6, ft_pts / 2.0)), size=n_sims).astype(float), 0.0)
        # home edge: add a small extra make probability already baked into ppp via a point nudge
        # Apply edge as a deterministic PPP add on home possessions (half a point sometimes)
        # Simpler: after the poss, add Bernoulli edge.
        home_poss = poss == 1
        # miss + ORB keeps possession
        missed = ((is_2 & ~make2) | (is_3 & ~make3))
        keep = missed & (rng.random(n_sims) < orb)
        scored_or_to = (pts > 0) | is_to
        delta = np.where(home_poss, pts, -pts)
        # Expected edge: shift home scoring by ppp_home_edge in probability of +1
        if ppp_home_edge:
            extra = (rng.random(n_sims) < min(0.20, abs(ppp_home_edge))) * np.sign(ppp_home_edge)
            delta = delta + np.where(home_poss, extra, 0.0)
        margin = margin + delta.astype(np.int16)
        # flip poss unless ORB keep
        poss = np.where(keep, poss, 1 - poss)
    # P(win | current margin m) = P(m + change > 0) + 0.5 P(==)
    change = margin.astype(np.int32)
    out = {}
    for m in range(-20, 21):
        fin = m + change
        out[m] = float(np.mean((fin > 0) + 0.5 * (fin == 0)))
    return out


_RATES = None
_NSIMS = 1_000_000
_SEED = 7


def _InitSim(payload):
    global _RATES, _NSIMS, _SEED
    LimitThreads()
    SetNormalPriority()
    _RATES, _NSIMS, _SEED = payload


def _SimJob(job):
    tbin, poss_home, bonus, edge_key, edge = job
    rem = 0.5 * (TIME_BINS[tbin][0] + TIME_BINS[tbin][1])
    rng = np.random.default_rng(_SEED + 17 * tbin + 3 * int(poss_home) + 5 * int(bonus) + sum(ord(c) for c in str(edge_key)))
    grid = SimulateCell(rem, poss_home, bonus, _RATES, _NSIMS, edge, rng)
    return job, grid


def BuildMcGrid(rates, n_sims, edges, workers=N_WORKERS):
    """edges: dict name -> ppp_home_edge, including 0.0 for a neutral grid.

    Neutral grid plus a few prior-edge slices. Lookup interpolates on edge.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = []
    for name, edge in edges.items():
        for tbin in range(len(TIME_BINS)):
            for poss_home in (True, False):
                for bonus in (False, True):
                    jobs.append((tbin, poss_home, bonus, name, edge))
    print(f"MC grid jobs={len(jobs)} n_sims={n_sims} workers={workers}", flush=True)
    out = {}
    with ProcessPoolExecutor(max_workers=max(1, int(workers)),
                             initializer=_InitSim, initargs=((rates, n_sims, 7),)) as ex:
        futs = [ex.submit(_SimJob, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            job, grid = fut.result()
            tbin, poss_home, bonus, name, edge = job
            out[(tbin, poss_home, bonus, name)] = grid
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"mc {done}/{len(jobs)}", flush=True)
    return out


def McLookup(grid, rates, rem_s, poss, bonus_home, margin, prior, edges):
    tbin = TimeBin(rem_s)
    ph = poss == "home"
    # If possession unknown, average home/away.
    ppp = _HomeAdvantagePpp(prior, rates)
    # nearest edge slice
    name = min(edges, key=lambda n: abs(edges[n] - ppp))
    m = int(max(-20, min(20, round(margin))))
    if poss in ("home", "away"):
        cell = grid.get((tbin, ph, bool(bonus_home), name))
        if not cell:
            return None
        return cell[m]
    c1 = grid.get((tbin, True, bool(bonus_home), name))
    c2 = grid.get((tbin, False, bool(bonus_home), name))
    if not c1 or not c2:
        return None
    return 0.5 * (c1[m] + c2[m])


def AttachGbmToStates(games, models):
    """Predict p_gbm on every play state (not just the 30s training subsample)."""
    for g in games:
        booster = models.get(g["season"])
        if booster is None or not g.get("states"):
            g["state_p_gbm"] = None
            continue
        prior, src, y = g["prior"], g["prior_source"], g["won"]
        rows = [FeatureRow(s, prior, src, y) for s in g["states"]]
        X, _, init = MatrixFromRows(rows)
        pred = PredictGbm(booster, X, init)
        for s, p in zip(g["states"], pred):
            s["p_gbm"] = float(p)
        g["state_p_gbm"] = pred
    return games


def AttachMcToStates(games, grid, rates, edges):
    for g in games:
        if not g.get("states"):
            continue
        for s in g["states"]:
            rem = max(0.0, REG_END - float(s["el"])) if int(s["per"]) <= 4 else 150.0
            p = McLookup(grid, rates, rem, s.get("poss"), s.get("home_bonus"),
                         s["margin"], g["prior"], edges)
            s["p_mc"] = p
    return games
