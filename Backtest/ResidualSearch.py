"""Walk-forward residual-on-market search: models may only learn deviations from Kalshi T-60.

The market logit is an offset / base margin, never a free coefficient. Features are
leak-free and computed as of before each game. Injury-report counts are skipped:
the local PDF stash is incomplete and there is no parsed count series.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time

# Cap BLAS/OpenMP before numpy loads. 7 workers x 1 thread stays near 25% of 28 cores.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np

MAIN = Path(r"E:\Basketball\basketball")
OUT_DIR = Path(r"E:\AgentDocs\Basketbot\2026-11-13_residual-search")
BASE = "2026-11-13_residual-search"
EDGES_IN = Path(r"E:\AgentDocs\Basketbot\2026-11-13_bankroll-10k\2026-11-13_bankroll-10k_edges.csv")
FF_IN = Path(r"E:\AgentDocs\Basketbot\2026-11-13_four-factors\2026-11-13_four-factors.csv")
SEASON_DIR = MAIN / "data" / "season"
KALSHI_CANDLES = MAIN / "data" / "inplay" / "kalshi" / "candles"
KALSHI_MARKETS = MAIN / "data" / "inplay" / "kalshi" / "markets.json"
GAMES_META = MAIN / "data" / "inplay" / "games.pkl"

FEE = 0.02
FEE_BASIS = "tim_est_0.02"
PRICE_BASIS = "kalshi_T60"
EDGE_CUT = 0.03
START_BANK = 10000.0
FLAT_STAKE = 100.0
QKELLY_FRAC = 0.25
QKELLY_CAP = 0.02
BURNIN_END = "2025-12-02"  # exclusive; first ~6 weeks of 2025-26
STALE_SEC = 1800
T60_SEC = 3600
EPS = 1e-6
RESID_PRIOR = 10.0
BOOT_GAP_REPS = 10000
BOOT_EDGE_REPS = 10000
SPA_REPS = 10000
PERM_REPS = 200
MC_PATHS = 10000
BOOT_SEED = 7
GRID_SEED = 7
N_WORKERS = max(1, (os.cpu_count() or 8) // 4)
MEAN_BLOCK = 5  # stationary-bootstrap mean block length in days

# Official NBA arenas (lat, lon). Travel is great-circle miles from a team's
# previous game venue to this game's venue.
ARENA_LATLON = {
    "ATL": (33.7573, -84.3963), "BOS": (42.3662, -71.0621), "BKN": (40.6828, -73.9758),
    "CHA": (35.2251, -80.8392), "CHI": (41.8807, -87.6742), "CLE": (41.4965, -81.6882),
    "DAL": (32.7905, -96.8103), "DEN": (39.7487, -105.0077), "DET": (42.3411, -83.0550),
    "GSW": (37.7680, -122.3877), "HOU": (29.7508, -95.3621), "IND": (39.7640, -86.1555),
    "LAC": (34.0430, -118.2673), "LAL": (34.0430, -118.2673), "MEM": (35.1382, -90.0506),
    "MIA": (25.7814, -80.1870), "MIL": (43.0451, -87.9172), "MIN": (44.9795, -93.2762),
    "NOP": (29.9490, -90.0821), "NYK": (40.7505, -73.9934), "OKC": (35.4634, -97.5151),
    "ORL": (28.5392, -81.3839), "PHI": (39.9012, -75.1720), "PHX": (33.4457, -112.0712),
    "POR": (45.5316, -122.6668), "SAC": (38.5802, -121.4997), "SAS": (29.4270, -98.4375),
    "TOR": (43.6435, -79.3791), "UTA": (40.7683, -111.9011), "WAS": (38.8981, -77.0209),
}

FEATURE_NAMES = [
    "stack_logit_gap", "elo_logit_gap", "ff_logit_gap",
    "rest_diff", "home_b2b", "away_b2b", "rest_missing",
    "travel_diff", "travel_missing",
    "line_move", "line_missing",
    "resid_home", "resid_away",
]

FEATURE_SETS = {
    "all": list(FEATURE_NAMES),
    "models_only": ["stack_logit_gap", "elo_logit_gap", "ff_logit_gap"],
    "models_stack_elo": ["stack_logit_gap", "elo_logit_gap"],
    "models_stack_ff": ["stack_logit_gap", "ff_logit_gap"],
    "situational": ["rest_diff", "home_b2b", "away_b2b", "rest_missing",
                    "travel_diff", "travel_missing", "line_move", "line_missing"],
    "no_ff": [n for n in FEATURE_NAMES if n != "ff_logit_gap"],
    "no_models": ["rest_diff", "home_b2b", "away_b2b", "rest_missing",
                  "travel_diff", "travel_missing", "line_move", "line_missing",
                  "resid_home", "resid_away"],
    "no_resid": [n for n in FEATURE_NAMES if not n.startswith("resid_")],
    "no_line": [n for n in FEATURE_NAMES if n not in ("line_move", "line_missing")],
    "stack_situational": ["stack_logit_gap", "rest_diff", "home_b2b", "away_b2b",
                          "rest_missing", "travel_diff", "travel_missing",
                          "line_move", "line_missing"],
    "resid_line": ["resid_home", "resid_away", "line_move", "line_missing"],
    "models_resid": ["stack_logit_gap", "elo_logit_gap", "ff_logit_gap",
                     "resid_home", "resid_away"],
}

TREE_FEATURE_SETS = [
    "all", "models_only", "no_ff", "no_models", "no_resid", "no_line",
    "stack_situational", "models_resid",
]


def ClipP(p, lo=EPS, hi=1.0 - EPS) -> float:
    return min(max(float(p), lo), hi)


def Logit(p) -> float:
    p = ClipP(p)
    return math.log(p / (1.0 - p))


def Sigmoid(z) -> float:
    z = float(np.clip(z, -40.0, 40.0))
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def IgnBits(p, y) -> float:
    p = ClipP(p)
    return -math.log2(p if y else (1.0 - p))


def LogLossNats(p, y) -> float:
    p = ClipP(p)
    return -math.log(p if y else (1.0 - p))


def Mid(bid, ask):
    if bid is None or ask is None:
        return None
    return 0.5 * (float(bid) + float(ask))


def HaversineMiles(a, b) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def MonthKey(day: str) -> str:
    return day[:7]


def GitHash(repo: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo), text=True).strip()


def LimitThreads() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def SetNormalPriority() -> None:
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00000020)
    except Exception:
        pass


def WriteJson(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def WriteCsv(path: Path, rows: list, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for k in fields:
                v = r.get(k, "")
                if v is None:
                    out[k] = ""
                elif isinstance(v, float):
                    out[k] = f"{v:.10g}"
                elif isinstance(v, (np.floating,)):
                    out[k] = f"{float(v):.10g}"
                elif isinstance(v, (np.integer,)):
                    out[k] = int(v)
                elif isinstance(v, (bool, np.bool_)):
                    out[k] = "True" if v else "False"
                else:
                    out[k] = v
            w.writerow(out)


# --- line movement (candles at or before T-60 only) ---------------------------

def LineMoveFromCandles(candles, sched_ts: int, t60_sec: int = T60_SEC):
    """Earliest valid mid to last mid at or before T-60. Ignores later candles."""
    target = int(sched_ts) - int(t60_sec)
    valid = []
    for c in candles or []:
        t = c.get("t")
        if t is None or int(t) > target:
            continue
        bid, ask = c.get("bid"), c.get("ask")
        mid = Mid(bid, ask)
        if mid is None or not (0.0 < mid < 1.0):
            continue
        if not (0.0 <= float(bid) <= 1.0 and 0.0 <= float(ask) <= 1.0):
            continue
        valid.append((int(t), float(mid)))
    if not valid:
        return None
    valid.sort(key=lambda x: x[0])
    t0, m0 = valid[0]
    t1, m1 = valid[-1]
    return {
        "line_move": m1 - m0,
        "mid_open": m0,
        "mid_t60": m1,
        "cover_sec": t1 - t0,
        "n_candles": len(valid),
    }


# --- rest / travel / residual books (prior games only; same-day does not leak)

class ScheduleBook:
    """Rest, back-to-backs and travel from the prior game only."""

    def __init__(self):
        self.last_date = {}
        self.last_venue = {}

    def View(self, day: str, home: str, away: str, venue: str) -> dict:
        def rest_days(team):
            prev = self.last_date.get(team)
            if prev is None:
                return None
            return (date.fromisoformat(day) - date.fromisoformat(prev)).days - 1

        def travel(team):
            prev = self.last_venue.get(team)
            here = ARENA_LATLON.get(venue)
            if prev is None or here is None:
                return None
            return HaversineMiles(prev, here)

        rh, ra = rest_days(home), rest_days(away)
        th, ta = travel(home), travel(away)
        return {
            "home_rest": rh, "away_rest": ra,
            "home_b2b": 1.0 if rh == 0 else 0.0,
            "away_b2b": 1.0 if ra == 0 else 0.0,
            "rest_diff": (0.0 if rh is None or ra is None else float(rh - ra)),
            "rest_missing": 1.0 if rh is None or ra is None else 0.0,
            "home_travel": th, "away_travel": ta,
            "travel_diff": (0.0 if th is None or ta is None else (th - ta) / 1000.0),
            "travel_missing": 1.0 if th is None or ta is None else 0.0,
        }

    def Apply(self, day: str, home: str, away: str, venue: str) -> None:
        loc = ARENA_LATLON.get(venue)
        for team in (home, away):
            self.last_date[team] = day
            if loc is not None:
                self.last_venue[team] = loc


class ResidualBook:
    """Shrunk team residual vs the market, updated only after a day's games."""

    def __init__(self, prior_n: float = RESID_PRIOR):
        self.prior_n = float(prior_n)
        self.sum = defaultdict(float)
        self.n = defaultdict(float)

    def View(self, home: str, away: str):
        def shrunk(team):
            return self.sum[team] / (self.n[team] + self.prior_n)
        return shrunk(home), shrunk(away)

    def ApplyDay(self, games: list) -> None:
        for g in games:
            y = float(g["home_win"])
            p = float(g["p_mkt"])
            home, away = g["home_team"], g["away_team"]
            self.sum[home] += y - p
            self.n[home] += 1.0
            self.sum[away] += (1.0 - y) - (1.0 - p)
            self.n[away] += 1.0


def AttachScheduleFeatures(priced: list, season_years=(2024, 2025, 2026)) -> None:
    """Walk all local season games so opening-night rest/travel see the prior season."""
    all_games = []
    for year in season_years:
        path = SEASON_DIR / f"games_{year}.json"
        if not path.exists():
            continue
        for g in json.loads(path.read_text(encoding="utf-8")):
            if g.get("home_score") is None or g.get("away_score") is None:
                continue
            all_games.append(g)
    all_games.sort(key=lambda g: (g["game_date"], g["game_id"]))
    wanted = {g["game_id"] for g in priced}
    book = ScheduleBook()
    for g in all_games:
        day, home, away = g["game_date"], g["home_team"], g["away_team"]
        if g["game_id"] in wanted:
            view = book.View(day, home, away, home)
            # stash on the priced row later
            g["_sched_view"] = view
        book.Apply(day, home, away, home)
    by_id = {g["game_id"]: g for g in all_games if "_sched_view" in g}
    for row in priced:
        src = by_id.get(row["game_id"])
        view = src["_sched_view"] if src else {
            "home_rest": None, "away_rest": None, "home_b2b": 0.0, "away_b2b": 0.0,
            "rest_diff": 0.0, "rest_missing": 1.0,
            "home_travel": None, "away_travel": None,
            "travel_diff": 0.0, "travel_missing": 1.0,
        }
        row.update(view)
        row["home_team"] = src["home_team"] if src else row.get("home_team")
        row["away_team"] = src["away_team"] if src else row.get("away_team")


def AttachTeamsFromEvent(priced: list) -> None:
    for g in priced:
        if g.get("home_team") and g.get("away_team"):
            continue
        ev = g.get("event") or ""
        if " @ " in ev:
            away, home = ev.split(" @ ", 1)
            g["away_team"] = away.strip()
            g["home_team"] = home.strip()


def AttachFourFactors(priced: list) -> None:
    by = {}
    with FF_IN.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            by[str(r["event_id"])] = r
    missing = 0
    for g in priced:
        r = by.get(g["game_id"])
        if r is None:
            missing += 1
            g["p_ff"] = 0.5
            g["p_elo"] = g.get("p_elo", 0.5)
            continue
        g["p_ff"] = float(r["p_four_factors_gap"])
        if r.get("p_elo_predictday_v1"):
            g["p_elo"] = float(r["p_elo_predictday_v1"])
        if r.get("p_stack_global") and "p_stack" not in g:
            g["p_stack"] = float(r["p_stack_global"])
    if missing:
        print(f"four-factors missing={missing}", flush=True)


def AttachLineMovement(priced: list) -> None:
    """Join T-60 ticker and earliest-to-T-60 mid move. Local candles only."""
    import sys
    if str(MAIN) not in sys.path:
        sys.path.insert(0, str(MAIN))
    from Backtest.ExportOpsVision import LoadSchedIndex, LoadMarketIndex

    sched_index = LoadSchedIndex()
    market_index = LoadMarketIndex()
    n_miss = 0
    cache = {}
    for g in priced:
        key = (g["game_date"], g["home_team"], g["away_team"])
        sched = sched_index.get(key, {})
        markets = market_index.get(key, {})
        home_m = markets.get("home") if markets else None
        sched_ts = sched.get("sched_ts")
        ticker = sched.get("kalshi_ticker") or (home_m["ticker"] if home_m else None)
        if sched_ts is None and g.get("start_time"):
            from datetime import datetime
            try:
                sched_ts = int(datetime.fromisoformat(g["start_time"]).timestamp())
            except Exception:
                sched_ts = None
        g["sched_ts"] = sched_ts
        g["ticker"] = ticker
        if not ticker or sched_ts is None:
            g["line_move"] = 0.0
            g["line_missing"] = 1.0
            n_miss += 1
            continue
        if ticker not in cache:
            path = KALSHI_CANDLES / f"{ticker}.json"
            if path.exists():
                cache[ticker] = json.loads(path.read_text(encoding="utf-8")).get("candles") or []
            else:
                cache[ticker] = []
        info = LineMoveFromCandles(cache[ticker], int(sched_ts))
        if info is None:
            g["line_move"] = 0.0
            g["line_missing"] = 1.0
            n_miss += 1
        else:
            g["line_move"] = info["line_move"]
            g["line_missing"] = 0.0
            g["line_cover_sec"] = info["cover_sec"]
            g["line_n_candles"] = info["n_candles"]
    print(f"line movement attached, missing={n_miss}/{len(priced)}", flush=True)


def AttachResiduals(priced: list, y_field: str = "home_win") -> None:
    """Leak-free team residuals vs Kalshi: same-day games share the prior-day book."""
    rows = sorted(priced, key=lambda g: (g["game_date"], g["game_id"]))
    book = ResidualBook()
    i = 0
    n = len(rows)
    while i < n:
        day = rows[i]["game_date"]
        j = i
        while j < n and rows[j]["game_date"] == day:
            j += 1
        batch = rows[i:j]
        for g in batch:
            rh, ra = book.View(g["home_team"], g["away_team"])
            g["resid_home"] = rh
            g["resid_away"] = ra
        # Apply with possibly permuted labels sitting in y_field, copied to home_win view
        applied = []
        for g in batch:
            applied.append({
                "home_team": g["home_team"], "away_team": g["away_team"],
                "home_win": g[y_field], "p_mkt": g["p_mkt"],
            })
        book.ApplyDay(applied)
        i = j


def LoadPanel() -> list:
    """1469-game kalshi_T60 set plus leak-free features from local files only."""
    from Backtest.ExportMarketBlend import LoadPriced

    print("loading priced set", flush=True)
    priced = LoadPriced()
    AttachTeamsFromEvent(priced)
    AttachFourFactors(priced)
    if any("p_elo" not in g for g in priced):
        from Backtest.ExportMarketBlend import AttachElo
        AttachElo(priced)
    AttachScheduleFeatures(priced)
    AttachLineMovement(priced)
    AttachResiduals(priced, y_field="home_win")
    for g in priced:
        g["offset"] = Logit(g["p_mkt"])
        g["stack_logit_gap"] = Logit(g["p_stack"]) - g["offset"]
        g["elo_logit_gap"] = Logit(g.get("p_elo", 0.5)) - g["offset"]
        g["ff_logit_gap"] = Logit(g.get("p_ff", 0.5)) - g["offset"]
        g["month"] = MonthKey(g["game_date"])
        g["scored"] = g["game_date"] >= BURNIN_END
    print(f"panel n={len(priced)} scored={sum(1 for g in priced if g['scored'])}", flush=True)
    return priced


def FeatureMatrix(games: list, names=None):
    names = list(names or FEATURE_NAMES)
    x = np.zeros((len(games), len(names)), dtype=np.float64)
    for i, g in enumerate(games):
        for j, name in enumerate(names):
            v = g.get(name)
            x[i, j] = 0.0 if v is None else float(v)
    return x, names


def MonthFolds(games: list, burnin_end: str = BURNIN_END):
    """Train on earlier months only. Burn-in months are training data, not scored."""
    months = []
    for g in games:
        m = g["month"]
        if m not in months:
            months.append(m)
    months.sort()
    folds = []
    for m in months:
        if m + "-01" < burnin_end and not any(
                g["month"] == m and g["game_date"] >= burnin_end for g in games):
            # pure burn-in month (e.g. 2025-10, 2025-11)
            continue
        # A month is a test fold if it has any scored (post-burn-in) games.
        test_idx = [i for i, g in enumerate(games)
                    if g["month"] == m and g["game_date"] >= burnin_end]
        if not test_idx:
            continue
        first_test = min(games[i]["game_date"] for i in test_idx)
        train_idx = [i for i, g in enumerate(games) if g["game_date"] < first_test]
        if not train_idx:
            continue
        folds.append({"month": m, "train_idx": train_idx, "test_idx": test_idx})
    return folds


# --- models (market logit is an offset, never a free coefficient) -------------

def FitLogisticOffset(X, y, offset, l2=1.0, fit_intercept=True, max_iter=50):
    """L2-penalized logistic MLE with a frozen offset. Penalty does not touch intercept."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    n = X.shape[0]
    if fit_intercept:
        Xd = np.column_stack([np.ones(n), X])
    else:
        Xd = X
    k = Xd.shape[1]
    beta = np.zeros(k, dtype=np.float64)
    penalty = np.full(k, float(l2), dtype=np.float64)
    if fit_intercept:
        penalty[0] = 0.0
    for _ in range(max_iter):
        z = offset + Xd @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        grad = Xd.T @ (y - p) - penalty * beta
        h = Xd.T @ (w[:, None] * Xd)
        h[np.diag_indices(k)] += penalty
        try:
            step = np.linalg.solve(h, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(h, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def PredictLogisticOffset(beta, X, offset, fit_intercept=True):
    X = np.asarray(X, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    if fit_intercept:
        Xd = np.column_stack([np.ones(len(X)), X])
    else:
        Xd = X
    z = offset + Xd @ beta
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def StandardizeTrain(Xtr, Xte):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def FitPredictLogistic(Xtr, ytr, off_tr, Xte, off_te, l2=1.0, fit_intercept=True):
    Xtr_s, Xte_s = StandardizeTrain(Xtr, Xte)
    beta = FitLogisticOffset(Xtr_s, ytr, off_tr, l2=l2, fit_intercept=fit_intercept)
    return PredictLogisticOffset(beta, Xte_s, off_te, fit_intercept=fit_intercept)


def FitPredictLgbm(Xtr, ytr, off_tr, Xte, off_te, params):
    import lightgbm as lgb
    dtrain = lgb.Dataset(Xtr, label=ytr, init_score=off_tr, free_raw_data=False)
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_threads": 1,
        "feature_pre_filter": False,
        "num_leaves": int(params["num_leaves"]),
        "min_child_samples": int(params["min_child_samples"]),
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "lambda_l2": float(params["reg_lambda"]),
        "bagging_freq": 0,
    }
    booster = lgb.train(lgb_params, dtrain, num_boost_round=int(params["n_estimators"]))
    raw = booster.predict(Xte, raw_score=True)
    return 1.0 / (1.0 + np.exp(-np.clip(off_te + raw, -40.0, 40.0)))


def FitPredictXgb(Xtr, ytr, off_tr, Xte, off_te, params):
    import xgboost as xgb
    dtrain = xgb.DMatrix(Xtr, label=ytr, base_margin=off_tr)
    dtest = xgb.DMatrix(Xte, base_margin=off_te)
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "nthread": 1,
        "max_depth": int(params["max_depth"]),
        "min_child_weight": float(params["min_child_weight"]),
        "eta": float(params["learning_rate"]),
        "lambda": float(params["reg_lambda"]),
        "subsample": float(params["subsample"]),
        "tree_method": "hist",
        "verbosity": 0,
    }
    booster = xgb.train(xgb_params, dtrain, num_boost_round=int(params["n_estimators"]))
    return np.asarray(booster.predict(dtest), dtype=np.float64)


def BuildGrid():
    """At least 2,000 pre-registered configs: logistic, LightGBM, XGBoost."""
    configs = []
    cid = 0
    l2_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    for fs in FEATURE_SETS:
        for l2 in l2_grid:
            for intercept in (True, False):
                cid += 1
                configs.append({
                    "config_id": f"C{cid:04d}",
                    "family": "logistic",
                    "feature_set": fs,
                    "l2": l2,
                    "fit_intercept": intercept,
                    "num_leaves": "",
                    "min_child_samples": "",
                    "learning_rate": "",
                    "n_estimators": "",
                    "max_depth": "",
                    "reg_lambda": "",
                    "min_child_weight": "",
                    "subsample": "",
                })
    for fs in TREE_FEATURE_SETS:
        for num_leaves in (8, 16, 31):
            for min_child in (20, 50, 100):
                for lr in (0.03, 0.1):
                    for n_est in (50, 100, 200):
                        for depth in (2, 3, 4):
                            cid += 1
                            configs.append({
                                "config_id": f"C{cid:04d}",
                                "family": "lgbm",
                                "feature_set": fs,
                                "l2": "",
                                "fit_intercept": "",
                                "num_leaves": num_leaves,
                                "min_child_samples": min_child,
                                "learning_rate": lr,
                                "n_estimators": n_est,
                                "max_depth": depth,
                                "reg_lambda": 1.0,
                                "min_child_weight": "",
                                "subsample": "",
                            })
    for fs in TREE_FEATURE_SETS:
        for depth in (2, 3, 4):
            for min_cw in (1, 10):
                for lr in (0.03, 0.1):
                    for n_est in (50, 150):
                        for reg in (1.0, 5.0):
                            for subsample in (0.8, 1.0):
                                cid += 1
                                configs.append({
                                    "config_id": f"C{cid:04d}",
                                    "family": "xgb",
                                    "feature_set": fs,
                                    "l2": "",
                                    "fit_intercept": "",
                                    "num_leaves": "",
                                    "min_child_samples": "",
                                    "learning_rate": lr,
                                    "n_estimators": n_est,
                                    "max_depth": depth,
                                    "reg_lambda": reg,
                                    "min_child_weight": min_cw,
                                    "subsample": subsample,
                                })
    return configs


def ColIndex(names, feature_set):
    want = FEATURE_SETS[feature_set]
    return [names.index(n) for n in want]


def WalkOneConfig(cfg, X, y, offset, folds, names):
    """Walk-forward predictions for one config. Train months are strictly earlier."""
    cols = ColIndex(names, cfg["feature_set"])
    Xs = X[:, cols]
    pred = np.full(len(y), np.nan, dtype=np.float64)
    family = cfg["family"]
    for fold in folds:
        tr, te = fold["train_idx"], fold["test_idx"]
        Xtr, Xte = Xs[tr], Xs[te]
        ytr = y[tr]
        off_tr, off_te = offset[tr], offset[te]
        try:
            if family == "logistic":
                p = FitPredictLogistic(
                    Xtr, ytr, off_tr, Xte, off_te,
                    l2=float(cfg["l2"]),
                    fit_intercept=bool(cfg["fit_intercept"]),
                )
            elif family == "lgbm":
                p = FitPredictLgbm(Xtr, ytr, off_tr, Xte, off_te, cfg)
            elif family == "xgb":
                p = FitPredictXgb(Xtr, ytr, off_tr, Xte, off_te, cfg)
            else:
                raise ValueError(family)
        except Exception:
            p = 1.0 / (1.0 + np.exp(-np.clip(off_te, -40.0, 40.0)))
        pred[te] = np.clip(p, EPS, 1.0 - EPS)
    return pred


# Worker globals (set in process initializer; arrays are small).
_W = {}


def _InitWorker(payload):
    LimitThreads()
    SetNormalPriority()
    _W.update(payload)


def _EvalConfig(cfg):
    pred = WalkOneConfig(cfg, _W["X"], _W["y"], _W["offset"], _W["folds"], _W["names"])
    scored = _W["scored_idx"]
    p = pred[scored]
    y = _W["y"][scored]
    p_mkt = _W["p_mkt"][scored]
    days = _W["days"][scored]
    ign_m = np.array([IgnBits(a, int(b)) for a, b in zip(p, y)])
    ign_k = np.array([IgnBits(a, int(b)) for a, b in zip(p_mkt, y)])
    gap = float(ign_m.mean() - ign_k.mean()) if len(y) else float("nan")
    ll = float(np.mean([LogLossNats(a, int(b)) for a, b in zip(p, y)])) if len(y) else float("nan")
    daily = {}
    for d, gm, gk in zip(days, ign_m, ign_k):
        daily.setdefault(d, []).append(gk - gm)  # positive => model better (SPA)
    day_keys = _W["day_keys"]
    day_diff = np.array([
        (float(np.mean(daily[d])) if d in daily else 0.0) for d in day_keys
    ], dtype=np.float64)
    # Qualifying edge vs ask + $0.02 (Tim's estimate), 3c bar.
    edges, edge_days = QualifyingEdges(
        p, y, p_mkt, _W["home_ask"][scored], _W["away_ask"][scored], days)
    mean_edge = float(np.mean(edges)) if edges else float("nan")
    n_bets = len(edges)
    return {
        "config_id": cfg["config_id"],
        "pred": pred,
        "gap_bits": gap,
        "log_loss": ll,
        "n_oos": int(len(y)),
        "n_bets": n_bets,
        "mean_edge": mean_edge,
        "day_diff": day_diff,
        "edges": edges,
        "edge_days": edge_days,
    }


def QualifyingEdges(p_home, y_home, p_mkt, home_ask, away_ask, days, fee=FEE, cut=EDGE_CUT):
    """Claimed edge_vs_fill on 3c+ bets (used in the trials registry)."""
    edges, edge_days = [], []
    for p, y, mkt, ha, aa, d in zip(p_home, y_home, p_mkt, home_ask, away_ask, days):
        e_h = float(p) - (float(ha) + fee)
        e_a = (1.0 - float(p)) - (float(aa) + fee)
        if e_h >= e_a:
            edge, won = e_h, int(y)
        else:
            edge, won = e_a, 1 - int(y)
        if edge >= cut:
            edges.append(edge)
            edge_days.append(d)
    return edges, edge_days


def RealizedQualifying(p_home, y_home, home_ask, away_ask, days, fee=FEE, cut=EDGE_CUT):
    """Realized P&L per contract (win $1, cost = ask + fee) on 3c+ claimed-edge bets."""
    pnls, pnl_days = [], []
    for p, y, ha, aa, d in zip(p_home, y_home, home_ask, away_ask, days):
        e_h = float(p) - (float(ha) + fee)
        e_a = (1.0 - float(p)) - (float(aa) + fee)
        if e_h >= e_a:
            fill, won, edge = float(ha), int(y), e_h
        else:
            fill, won, edge = float(aa), 1 - int(y), e_a
        if edge >= cut:
            pnls.append(won - (fill + fee))
            pnl_days.append(d)
    return pnls, pnl_days


def DayBlockBootstrap(values, clusters, reps=10000, seed=BOOT_SEED, level=0.90):
    """Bootstrap the mean, resampling clusters (days) with replacement."""
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0}
    point = float(vals.mean())
    if clusters is None:
        groups = {i: [vals[i]] for i in range(len(vals))}
    else:
        groups = defaultdict(list)
        for v, c in zip(vals, clusters):
            groups[c].append(float(v))
    keys = list(groups)
    bags = [np.asarray(groups[k], dtype=np.float64) for k in keys]
    sizes = np.array([len(b) for b in bags], dtype=np.int64)
    total = int(sizes.sum())
    rng = np.random.default_rng(seed)
    K = len(bags)
    boot = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        idx = rng.integers(0, K, size=K)
        s = 0.0
        n = 0
        for i in idx:
            s += float(bags[i].sum())
            n += int(sizes[i])
        boot[b] = s / n if n else point
    lo_q = (1.0 - level) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "mean": point,
        "lo": float(np.quantile(boot, lo_q)),
        "hi": float(np.quantile(boot, hi_q)),
        "n": int(len(vals)),
        "n_clusters": int(K),
    }


def StationaryBootstrapIndices(T, mean_block, rng):
    p = 1.0 / max(1.0, float(mean_block))
    idx = np.empty(T, dtype=np.int64)
    i = 0
    while i < T:
        start = int(rng.integers(0, T))
        # geometric length, at least 1
        length = 1
        while rng.random() > p and i + length < T:
            length += 1
        for j in range(length):
            if i >= T:
                break
            idx[i] = (start + j) % T
            i += 1
    return idx


def HansenSpa(day_diffs, reps=SPA_REPS, seed=BOOT_SEED, mean_block=MEAN_BLOCK):
    """White's Reality Check and Hansen SPA (consistent) on daily loss differentials.

    day_diffs: array shape (T, K), entry = mean_t (L_mkt - L_model). Positive
    means the model beats the market that day.
    Returns p-values for the max studentized mean (one-sided: better than market).
    """
    D = np.asarray(day_diffs, dtype=np.float64)
    T, K = D.shape
    if T < 2 or K < 1:
        return {"white_p": None, "spa_p": None, "spa_lower_p": None, "T": T, "K": K}
    mean = D.mean(axis=0)
    # HAC-ish variance via the observed daily series (stationary bootstrap uses the same)
    var = D.var(axis=0, ddof=1)
    sigma = np.sqrt(np.maximum(var, 1e-18))
    tstat = np.sqrt(T) * mean / sigma
    obs = float(np.max(tstat))
    # Hansen consistent threshold
    a_consistent = math.sqrt(2.0 * math.log(math.log(max(T, 3)))) / math.sqrt(T)
    try:
        from threadpoolctl import threadpool_limits
        _limit = threadpool_limits(limits=1)
    except Exception:
        _limit = None
    rng = np.random.default_rng(seed)
    white_count = 0
    spa_count = 0
    lower_count = 0
    for _ in range(reps):
        idx = StationaryBootstrapIndices(T, mean_block, rng)
        boot = D[idx].mean(axis=0)
        # White: recenter every model
        t_white = np.sqrt(T) * (boot - mean) / sigma
        if float(np.max(t_white)) >= obs:
            white_count += 1
        # Hansen consistent: recenter only models not clearly below zero
        recenter = mean >= (-a_consistent * sigma)
        t_spa = np.sqrt(T) * (boot - mean * recenter) / sigma
        if float(np.max(t_spa)) >= obs:
            spa_count += 1
        # Lower: do not recenter (anti-conservative under the least favorable null)
        t_lo = np.sqrt(T) * boot / sigma
        if float(np.max(t_lo)) >= obs:
            lower_count += 1
    if _limit is not None:
        try:
            _limit.unregister()
        except Exception:
            pass
    return {
        "white_p": (white_count + 1) / (reps + 1),
        "spa_p": (spa_count + 1) / (reps + 1),
        "spa_lower_p": (lower_count + 1) / (reps + 1),
        "obs_t": obs,
        "best_k": int(np.argmax(tstat)),
        "best_mean_diff": float(mean[int(np.argmax(tstat))]),
        "T": T,
        "K": K,
        "mean_block": mean_block,
        "reps": reps,
    }


def PermuteLabels(games, rng):
    """Return a shallow-copied panel with shuffled home_win, residuals rebuilt."""
    y = [g["home_win"] for g in games]
    rng.shuffle(y)
    out = []
    for g, yy in zip(games, y):
        h = dict(g)
        h["home_win"] = int(yy)
        out.append(h)
    AttachResiduals(out, y_field="home_win")
    # residual features changed; model logit gaps etc. stay (they don't use labels)
    return out


def PermutationPValue(best_cfg, games, X_names, folds, observed_gap, reps=PERM_REPS, seed=BOOT_SEED,
                      n_workers=N_WORKERS):
    """Rerun the best config's full walk-forward on label-permuted panels."""
    rng = random.Random(seed + 99)
    gaps = []
    # Serial in the worker pool we will already be using the main process after the grid.
    # Run in a process pool of n_workers, but build perms here (residuals are cheap).
    payloads = []
    names = list(X_names)
    for i in range(reps):
        perm_games = PermuteLabels(games, rng)
        Xp, _ = FeatureMatrix(perm_games, names)
        y = np.array([g["home_win"] for g in perm_games], dtype=np.float64)
        offset = np.array([g["offset"] for g in perm_games], dtype=np.float64)
        p_mkt = np.array([g["p_mkt"] for g in perm_games], dtype=np.float64)
        scored = [i for i, g in enumerate(perm_games) if g["scored"]]
        payloads.append((best_cfg, Xp, y, offset, folds, names, p_mkt, scored))

    def _one(args):
        cfg, X, y, offset, folds, names, p_mkt, scored = args
        pred = WalkOneConfig(cfg, X, y, offset, folds, names)
        ign_m = [IgnBits(pred[i], int(y[i])) for i in scored]
        ign_k = [IgnBits(p_mkt[i], int(y[i])) for i in scored]
        return float(np.mean(ign_m) - np.mean(ign_k))

    if n_workers <= 1:
        gaps = [_one(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_one, p) for p in payloads]
            for fut in as_completed(futs):
                gaps.append(fut.result())
    gaps = np.asarray(gaps, dtype=np.float64)
    # observed_gap is model-minus-market (negative is better)
    count = int(np.sum(gaps <= observed_gap))
    p = (count + 1) / (len(gaps) + 1)
    return {"p": p, "reps": len(gaps), "perm_mean": float(gaps.mean()),
            "perm_lo": float(np.quantile(gaps, 0.05)),
            "perm_hi": float(np.quantile(gaps, 0.95)),
            "gaps": gaps}


def ReplayBankroll(bets, strategy: str, start=START_BANK):
    """bets: list of dicts with date, p, fill, result (1/0), already time-ordered."""
    equity = start
    sod = start
    last_day = None
    ruined = False
    path = []
    for b in bets:
        if last_day is not None and b["date"] != last_day:
            sod = equity
        last_day = b["date"]
        c = float(b["fill"]) + FEE
        p = float(b["p"])
        if strategy == "flat100":
            if equity < FLAT_STAKE:
                ruined = True
                path.append(equity)
                continue
            stake = FLAT_STAKE
        else:
            if equity <= 0:
                ruined = True
                path.append(equity)
                continue
            kelly = 0.0 if c >= 1.0 or p <= c else (p - c) / (1.0 - c)
            stake = min(QKELLY_FRAC * kelly * sod, QKELLY_CAP * sod)
            stake = max(0.0, stake)
            if stake < 1e-9:
                path.append(equity)
                continue
        contracts = stake / c
        pnl = contracts * int(b["result"]) - stake
        equity = equity + pnl
        if equity < FLAT_STAKE:
            ruined = True
        path.append(equity)
    return {"equity": equity, "ruined": ruined, "path": path}


def QualifyingBets(games, pred):
    bets = []
    for g, p in zip(games, pred):
        if not g.get("scored") or p is None or (isinstance(p, float) and math.isnan(p)):
            continue
        e_h = float(p) - (float(g["home_ask"]) + FEE)
        e_a = (1.0 - float(p)) - (float(g["away_ask"]) + FEE)
        if e_h >= e_a:
            side, p_side, mkt_side, fill, edge_fill = "home", float(p), float(g["p_mkt"]), float(g["home_ask"]), e_h
            result = int(g["home_win"])
        else:
            side, p_side, mkt_side, fill, edge_fill = "away", 1.0 - float(p), 1.0 - float(g["p_mkt"]), float(g["away_ask"]), e_a
            result = 1 - int(g["home_win"])
        qualifies = edge_fill >= EDGE_CUT
        edge_mid = p_side - mkt_side - FEE
        stake_pnl = (FLAT_STAKE / (fill + FEE)) * result - FLAT_STAKE
        bets.append({
            "date": g["game_date"],
            "start_time": g.get("start_time", ""),
            "sport": "nba",
            "season": g.get("season_label", ""),
            "split": g.get("split", ""),
            "event_id": g["game_id"],
            "event": g.get("event", ""),
            "side": side,
            "model": "residual_search",
            "model_prob": p_side,
            "market_prob_devig": mkt_side,
            "fill_price": fill,
            "price_basis": PRICE_BASIS,
            "price_time": g.get("price_time", ""),
            "fee": FEE,
            "fee_basis": FEE_BASIS,
            "edge": edge_mid,
            "edge_vs_fill": edge_fill,
            "qualifies": qualifies,
            "result": result,
            "pnl_at_flat_stake": stake_pnl,
            "p": p_side,
            "fill": fill,
        })
    return bets


def MonteCarlo(bets, paths=MC_PATHS, seed=BOOT_SEED, mean_block=MEAN_BLOCK):
    """Stationary day-block bootstrap of the qualifying-bet sequence from $10k."""
    qualified = [b for b in bets if b["qualifies"]]
    if not qualified:
        return {
            "n_bets": 0,
            "flat100": {"p_ruin": None, "p5": None, "p50": None, "p95": None, "mean": None},
            "qkelly": {"p_ruin": None, "p5": None, "p50": None, "p95": None, "mean": None},
        }
    by_day = defaultdict(list)
    for b in qualified:
        by_day[b["date"]].append(b)
    days = sorted(by_day)
    T = len(days)
    rng = np.random.default_rng(seed + 3)

    def run(strategy):
        end = np.empty(paths, dtype=np.float64)
        ruin = 0
        for i in range(paths):
            idx = StationaryBootstrapIndices(T, mean_block, rng) if T > 1 else np.zeros(1, dtype=np.int64)
            seq = []
            for j in idx:
                seq.extend(by_day[days[int(j)]])
            got = ReplayBankroll(seq, strategy)
            end[i] = got["equity"]
            if got["ruined"]:
                ruin += 1
        return {
            "p_ruin": ruin / paths,
            "p5": float(np.quantile(end, 0.05)),
            "p50": float(np.quantile(end, 0.50)),
            "p95": float(np.quantile(end, 0.95)),
            "mean": float(end.mean()),
            "n_bets": len(qualified),
            "n_days": T,
            "paths": paths,
        }

    return {"n_bets": len(qualified), "n_days": T, "flat100": run("flat100"), "qkelly": run("qkelly")}


def EdgeRows21(bets, model_name):
    rows = []
    for b in bets:
        r = dict(b)
        r["model"] = model_name
        r["qualifies"] = bool(b["qualifies"])
        rows.append(r)
    return rows


def Fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def FmtCi(d, nd=4):
    if not d or d.get("mean") is None:
        return "NA"
    return f"{Fmt(d['mean'], nd)} [{Fmt(d['lo'], nd)}, {Fmt(d['hi'], nd)}]"


def WriteMd(path, best, gap_ci, edge_ci, spa, perm, mc, n_grid, n_scored, n_panel,
            folds, hash_main, notes, skipped):
    spa_p = spa.get("spa_p") if spa else None
    perm_p = perm.get("p") if perm else None
    gap_below0 = gap_ci and gap_ci.get("hi") is not None and gap_ci["hi"] < 0
    edge_above0 = edge_ci and edge_ci.get("lo") is not None and edge_ci["lo"] > 0
    spa_sig = spa_p is not None and spa_p < 0.05
    verdict = "PASS"
    if spa_sig and gap_below0 and edge_above0:
        verdict = "FAIL-TO-PASS"  # should not use this wording
    # PASS unless SPA p < 0.05 AND gap CI below 0 AND edge CI above 0
    if spa_sig and gap_below0 and edge_above0:
        verdict = "REJECT MARKET (provisional)"
    else:
        verdict = "PASS"

    lines = []
    lines.append(f"# NBA residual-on-market search (2026-11-13)")
    lines.append("")
    lines.append("Simulation only. No wagers were placed. Fee is Tim's $0.02 working estimate (`fee_basis=tim_est_0.02`), not a verified exchange fee.")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Panel: {n_panel} kalshi_T60 priced games; walk-forward scored N={n_scored} after burn-in {BURNIN_END} (exclusive).")
    lines.append(f"- Offset / base margin: logit(de-vigged Kalshi T-60 mid). Models can only learn deviations from the market.")
    lines.append("- Features (leak-free, pre-game): stack_global logit gap, Elo logit gap, four-factor GAP logit gap, rest / back-to-back, travel (arena great-circle miles), earliest-candle-to-T-60 line move, shrunk team residuals vs the market (prior days only).")
    lines.append(f"- Grid: {n_grid} configs (logistic + LightGBM + XGBoost), monthly walk-forward, train on earlier months only. {len(folds)} folds: " + ", ".join(f['month'] for f in folds) + ".")
    lines.append("- SPA: Hansen consistent p-value, 10,000 stationary bootstrap resamples of days. White's Reality Check also reported.")
    lines.append(f"- Permutation: {PERM_REPS} label-shuffles, residuals rebuilt, best config's full walk-forward rerun.")
    lines.append("- Bankroll: $10k, flat100 and quarter-Kelly (2% cap), qualify at ask + $0.02 >= 3c.")
    lines.append(f"- Code: local branch `feature/residual-search` at `{hash_main}`, not pushed.")
    if skipped:
        lines.append("- Skipped features: " + "; ".join(skipped))
    lines.append("")
    lines.append("## Best config")
    lines.append(f"- id: `{best['config_id']}`  family: `{best['family']}`  feature_set: `{best['feature_set']}`")
    extras = []
    for k in ("l2", "fit_intercept", "num_leaves", "min_child_samples", "learning_rate",
              "n_estimators", "max_depth", "reg_lambda", "min_child_weight", "subsample"):
        v = best.get(k, "")
        if v != "" and v is not None:
            extras.append(f"{k}={v}")
    if extras:
        lines.append("- params: " + ", ".join(extras))
    lines.append(f"- holdout+forward gap vs Kalshi (bits/game, model minus market, 90% day-block CI): {FmtCi(gap_ci)}")
    lines.append(f"- mean edge vs ask+$0.02 on 3c+ bets (95% day-block CI): {FmtCi(edge_ci)}  N_bets={best.get('n_bets', 0)}")
    lines.append(f"- SPA p (Hansen consistent) = {Fmt(spa_p, 4)}; White RC p = {Fmt(spa.get('white_p') if spa else None, 4)}; permutation p = {Fmt(perm_p, 4)}")
    lines.append("")
    lines.append("## Monte Carlo ($10k, 10,000 day-block paths)")
    if mc and mc.get("n_bets"):
        for name, label in (("flat100", "flat100"), ("qkelly", "quarter-Kelly")):
            s = mc[name]
            lines.append(
                f"- {label}: P(ruin)={Fmt(s['p_ruin'], 3)}; ending bankroll 5/50/95 = "
                f"${s['p5']:.0f} / ${s['p50']:.0f} / ${s['p95']:.0f}"
            )
    else:
        lines.append("- No qualifying 3c+ bets; Monte Carlo not run.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("- stack_global, Elo and four-factors are not injury-adjusted.")
    lines.append("- Kalshi candles cover only about 1-2 hours before T-60, so line movement is late drift, not the full open.")
    lines.append("- Candles have no size; fills at the T-60 ask are optimistic for large stakes.")
    lines.append("- Away ask is implied as 1 - home bid (home YES book only).")
    lines.append("- The best config is selected on the same walk-forward window it is scored on; SPA and the permutation test are the checks for that search.")
    if notes:
        for n in notes:
            lines.append(f"- {n}")
    lines.append("")
    lines.append(f"**Verdict: {verdict}.** "
                 "PASS unless SPA p < 0.05 AND the gap CI is entirely below 0 AND the 3c+ edge CI is entirely above 0 after the fee.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return verdict


def PeakSampler(stop_flag, out_path: Path, interval=2.0):
    """Record peak machine CPU % while the job runs."""
    peak = 0.0
    samples = 0
    try:
        import subprocess
        while not stop_flag["stop"]:
            try:
                raw = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-Counter '\\Processor(_Total)\\% Processor Time' -ErrorAction SilentlyContinue).CounterSamples.CookedValue"],
                    text=True, timeout=15)
                val = float(raw.strip().split()[-1])
                peak = max(peak, val)
                samples += 1
                out_path.write_text(json.dumps({"peak_cpu_pct": peak, "samples": samples}), encoding="utf-8")
            except Exception:
                pass
            time.sleep(interval)
    except Exception:
        pass
    return peak


def _PermOne(payload):
    """Top-level worker: walk the best config on one permuted panel."""
    LimitThreads()
    cfg, X, y, offset, folds, names, p_mkt, scored = payload
    pred = WalkOneConfig(cfg, X, y, offset, folds, names)
    ign_m = [IgnBits(pred[i], int(y[i])) for i in scored]
    ign_k = [IgnBits(p_mkt[i], int(y[i])) for i in scored]
    return float(np.mean(ign_m) - np.mean(ign_k))


def PermutationPValue(best_cfg, games, names, folds, observed_gap,
                      reps=PERM_REPS, seed=BOOT_SEED, n_workers=N_WORKERS):
    rng = random.Random(seed + 99)
    payloads = []
    for _ in range(reps):
        perm_games = PermuteLabels(games, rng)
        Xp, _ = FeatureMatrix(perm_games, names)
        y = np.array([g["home_win"] for g in perm_games], dtype=np.float64)
        offset = np.array([g["offset"] for g in perm_games], dtype=np.float64)
        p_mkt = np.array([g["p_mkt"] for g in perm_games], dtype=np.float64)
        scored = [i for i, g in enumerate(perm_games) if g["scored"]]
        payloads.append((best_cfg, Xp, y, offset, folds, names, p_mkt, scored))
    workers = max(1, int(n_workers))
    if workers <= 1:
        gaps = [_PermOne(p) for p in payloads]
    else:
        gaps = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_PermOne, p) for p in payloads]
            for fut in as_completed(futs):
                gaps.append(fut.result())
    gaps = np.asarray(gaps, dtype=np.float64)
    count = int(np.sum(gaps <= observed_gap))
    p = (count + 1) / (len(gaps) + 1)
    return {
        "p": p, "reps": len(gaps),
        "perm_mean": float(gaps.mean()),
        "perm_lo": float(np.quantile(gaps, 0.05)),
        "perm_hi": float(np.quantile(gaps, 0.95)),
    }


def RunGrid(games, configs, n_workers=N_WORKERS, status_path=None):
    X, names = FeatureMatrix(games, FEATURE_NAMES)
    y = np.array([g["home_win"] for g in games], dtype=np.float64)
    offset = np.array([g["offset"] for g in games], dtype=np.float64)
    p_mkt = np.array([g["p_mkt"] for g in games], dtype=np.float64)
    home_ask = np.array([g["home_ask"] for g in games], dtype=np.float64)
    away_ask = np.array([g["away_ask"] for g in games], dtype=np.float64)
    days = np.array([g["game_date"] for g in games])
    scored_idx = np.array([i for i, g in enumerate(games) if g["scored"]], dtype=np.int64)
    day_keys = sorted({games[i]["game_date"] for i in scored_idx})
    folds = MonthFolds(games)
    print(f"grid configs={len(configs)} folds={len(folds)} workers={n_workers} scored={len(scored_idx)}",
          flush=True)
    payload = {
        "X": X, "y": y, "offset": offset, "folds": folds, "names": names,
        "scored_idx": scored_idx, "p_mkt": p_mkt, "days": days,
        "home_ask": home_ask, "away_ask": away_ask, "day_keys": day_keys,
    }
    results = []
    t0 = time.time()
    done = 0
    workers = max(1, int(n_workers))
    with ProcessPoolExecutor(max_workers=workers, initializer=_InitWorker, initargs=(payload,)) as ex:
        futs = {ex.submit(_EvalConfig, cfg): cfg for cfg in configs}
        for fut in as_completed(futs):
            cfg = futs[fut]
            rec = fut.result()
            rec["cfg"] = cfg
            results.append(rec)
            done += 1
            if done % 50 == 0 or done == len(configs):
                msg = f"grid {done}/{len(configs)} in {time.time()-t0:.1f}s"
                print(msg, flush=True)
                if status_path:
                    WriteJson(status_path, {
                        "phase": "grid", "done": done, "total": len(configs),
                        "elapsed_s": time.time() - t0,
                    })
    results.sort(key=lambda r: (r["gap_bits"] if r["gap_bits"] == r["gap_bits"] else 9e9,
                                r["cfg"]["config_id"]))
    return {
        "results": results, "folds": folds, "names": names, "X": X, "y": y,
        "offset": offset, "day_keys": day_keys, "scored_idx": scored_idx,
        "p_mkt": p_mkt, "days": days, "home_ask": home_ask, "away_ask": away_ask,
    }


def BestAndMetrics(games, pack):
    results = pack["results"]
    best_rec = results[0]
    best_cfg = best_rec["cfg"]
    pred = WalkOneConfig(best_cfg, pack["X"], pack["y"], pack["offset"], pack["folds"], pack["names"])
    scored = pack["scored_idx"]
    p = pred[scored]
    y = pack["y"][scored]
    p_mkt = pack["p_mkt"][scored]
    days = pack["days"][scored]
    ign_m = np.array([IgnBits(a, int(b)) for a, b in zip(p, y)])
    ign_k = np.array([IgnBits(a, int(b)) for a, b in zip(p_mkt, y)])
    gaps = ign_m - ign_k
    gap_ci = DayBlockBootstrap(gaps, days, reps=BOOT_GAP_REPS, seed=BOOT_SEED, level=0.90)
    edges, edge_days = RealizedQualifying(
        p, y, pack["home_ask"][scored], pack["away_ask"][scored], days)
    edge_ci = DayBlockBootstrap(edges, edge_days, reps=BOOT_EDGE_REPS, seed=BOOT_SEED + 1, level=0.95)
    D = np.column_stack([r["day_diff"] for r in results])
    return best_cfg, best_rec, pred, gap_ci, edge_ci, D


def ConfigRows(results):
    rows = []
    for r in results:
        cfg = r["cfg"]
        rows.append({
            **cfg,
            "n_oos": r["n_oos"],
            "log_loss": r["log_loss"],
            "gap_bits": r["gap_bits"],
            "n_bets": r["n_bets"],
            "mean_edge": r["mean_edge"],
        })
    return rows


def McRows(mc, best_id):
    rows = []
    for strat in ("flat100", "qkelly"):
        s = (mc or {}).get(strat) or {}
        rows.append({
            "date": "2026-11-13",
            "sport": "nba",
            "series": f"nba_residual_search_{strat}",
            "strategy": strat,
            "config_id": best_id,
            "n_bets": (mc or {}).get("n_bets", 0),
            "n_days": (mc or {}).get("n_days", 0),
            "paths": MC_PATHS,
            "p_ruin": s.get("p_ruin"),
            "end_p5": s.get("p5"),
            "end_p50": s.get("p50"),
            "end_p95": s.get("p95"),
            "end_mean": s.get("mean"),
            "start_bank": START_BANK,
            "fee_basis": FEE_BASIS,
        })
    return rows


def SummaryRows(best, gap_ci, edge_ci, spa, perm, mc, verdict, n_grid, n_scored):
    return [{
        "date": "2026-11-13",
        "sport": "nba",
        "series": "nba_residual_search",
        "model": f"residual_{best['family']}_{best['feature_set']}",
        "config_id": best["config_id"],
        "family": best["family"],
        "feature_set": best["feature_set"],
        "n_grid": n_grid,
        "n_oos": n_scored,
        "gap_bits": gap_ci.get("mean") if gap_ci else None,
        "gap_lo90": gap_ci.get("lo") if gap_ci else None,
        "gap_hi90": gap_ci.get("hi") if gap_ci else None,
        "n_bets": best.get("n_bets"),
        "mean_edge": edge_ci.get("mean") if edge_ci else None,
        "edge_lo95": edge_ci.get("lo") if edge_ci else None,
        "edge_hi95": edge_ci.get("hi") if edge_ci else None,
        "spa_p": spa.get("spa_p") if spa else None,
        "white_p": spa.get("white_p") if spa else None,
        "perm_p": perm.get("p") if perm else None,
        "mc_flat100_p_ruin": (mc or {}).get("flat100", {}).get("p_ruin"),
        "mc_flat100_p50": (mc or {}).get("flat100", {}).get("p50"),
        "mc_qkelly_p_ruin": (mc or {}).get("qkelly", {}).get("p_ruin"),
        "mc_qkelly_p50": (mc or {}).get("qkelly", {}).get("p50"),
        "verdict": verdict,
        "fee_basis": FEE_BASIS,
    }]


def Main() -> int:
    LimitThreads()
    SetNormalPriority()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = OUT_DIR / "_status.json"
    cpu_path = OUT_DIR / "_cpu_peak.json"
    print(f"python={sys.version.split()[0]} workers={N_WORKERS} git={GitHash(MAIN)}", flush=True)
    WriteJson(status_path, {"phase": "panel", "workers": N_WORKERS})

    import threading
    stop = {"stop": False}
    sampler = threading.Thread(target=PeakSampler, args=(stop, cpu_path), daemon=True)
    sampler.start()
    t0 = time.time()
    skipped = [
        "injury-report counts: local PDFs are a partial stash with no parsed per-game count series, so the feature was skipped rather than delaying the run"
    ]
    games = LoadPanel()
    configs = BuildGrid()
    assert len(configs) >= 2000, len(configs)
    WriteJson(status_path, {"phase": "grid", "n_configs": len(configs), "n_games": len(games)})
    pack = RunGrid(games, configs, n_workers=N_WORKERS, status_path=status_path)
    cfg_rows = ConfigRows(pack["results"])
    WriteCsv(OUT_DIR / f"{BASE}_configs.csv", cfg_rows,
             ["config_id", "family", "feature_set", "l2", "fit_intercept",
              "num_leaves", "min_child_samples", "learning_rate", "n_estimators",
              "max_depth", "reg_lambda", "min_child_weight", "subsample",
              "n_oos", "log_loss", "gap_bits", "n_bets", "mean_edge"])
    print("configs written", flush=True)

    WriteJson(status_path, {"phase": "post", "elapsed_s": time.time() - t0})
    best_cfg, best_rec, pred, gap_ci, edge_ci, D = BestAndMetrics(games, pack)
    best = dict(best_cfg)
    best["n_bets"] = best_rec["n_bets"]
    best["gap_bits"] = best_rec["gap_bits"]
    print(f"best {best['config_id']} {best['family']} {best['feature_set']} gap={best_rec['gap_bits']:.4f}",
          flush=True)

    # SPA (main process) and permutation (remaining workers) after the grid.
    WriteJson(status_path, {"phase": "spa_perm_mc", "best": best["config_id"]})
    spa = HansenSpa(D, reps=SPA_REPS, seed=BOOT_SEED, mean_block=MEAN_BLOCK)
    print(f"SPA spa_p={spa['spa_p']:.4f} white_p={spa['white_p']:.4f}", flush=True)

    bets = QualifyingBets(games, pred)
    mc = MonteCarlo(bets, paths=MC_PATHS, seed=BOOT_SEED)
    print(f"MC n_bets={mc.get('n_bets')}", flush=True)

    perm = PermutationPValue(
        best_cfg, games, pack["names"], pack["folds"], best_rec["gap_bits"],
        reps=PERM_REPS, seed=BOOT_SEED, n_workers=N_WORKERS)
    print(f"perm p={perm['p']:.4f}", flush=True)

    model_name = f"residual_{best['family']}_{best['feature_set']}"
    edge_rows = EdgeRows21(bets, model_name)
    EDGE_FIELDS = [
        "date", "start_time", "sport", "season", "split", "event_id", "event", "side",
        "model", "model_prob", "market_prob_devig", "fill_price", "price_basis",
        "price_time", "fee", "fee_basis", "edge", "edge_vs_fill", "qualifies",
        "result", "pnl_at_flat_stake",
    ]
    WriteCsv(OUT_DIR / f"{BASE}_edges.csv", edge_rows, EDGE_FIELDS)
    WriteCsv(OUT_DIR / f"{BASE}_mc.csv", McRows(mc, best["config_id"]),
             ["date", "sport", "series", "strategy", "config_id", "n_bets", "n_days",
              "paths", "p_ruin", "end_p5", "end_p50", "end_p95", "end_mean",
              "start_bank", "fee_basis"])
    n_scored = int(sum(1 for g in games if g["scored"]))
    # verdict written inside WriteMd
    notes = [
        f"Peak CPU is in {cpu_path.name}; workers={N_WORKERS} of {os.cpu_count()} (target <= 40%).",
        f"Elapsed {time.time()-t0:.0f}s.",
    ]
    # Temporary verdict inputs
    verdict = WriteMd(
        OUT_DIR / f"{BASE}.md", best, gap_ci, edge_ci, spa, perm, mc,
        len(configs), n_scored, len(games), pack["folds"], GitHash(MAIN), notes, skipped,
    )
    WriteCsv(OUT_DIR / f"{BASE}.csv",
             SummaryRows(best, gap_ci, edge_ci, spa, perm, mc, verdict, len(configs), n_scored),
             ["date", "sport", "series", "model", "config_id", "family", "feature_set",
              "n_grid", "n_oos", "gap_bits", "gap_lo90", "gap_hi90", "n_bets",
              "mean_edge", "edge_lo95", "edge_hi95", "spa_p", "white_p", "perm_p",
              "mc_flat100_p_ruin", "mc_flat100_p50", "mc_qkelly_p_ruin", "mc_qkelly_p50",
              "verdict", "fee_basis"])
    stop["stop"] = True
    peak = None
    try:
        if cpu_path.exists():
            peak = json.loads(cpu_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    WriteJson(status_path, {
        "phase": "done", "verdict": verdict, "best": best["config_id"],
        "gap_bits": best_rec["gap_bits"], "spa_p": spa.get("spa_p"),
        "perm_p": perm.get("p"), "elapsed_s": time.time() - t0, "peak_cpu": peak,
    })
    done_path = OUT_DIR / "_DONE.txt"
    done_path.write_text(f"verdict={verdict}\nbest={best['config_id']}\n", encoding="utf-8")
    print(f"DONE verdict={verdict} best={best['config_id']} elapsed={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(Main())
