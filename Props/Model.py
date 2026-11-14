"""Leak-free per-player 3PM count model (Poisson vs negative binomial) and a same-team joint."""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import nbinom, norm, poisson
from scipy.stats import multivariate_normal

TRAIN_SEASON = 2024  # 2024-25
HOLD_SEASON = 2025   # 2025-26
CURRENT_SEASON = 2026
THRESHOLDS = (2, 3, 4, 5)
MIN_PROJ = 12.0
EWMA_HALFLIFE = 8.0
RIDGE = 1e-6
RHO_CLIP = 0.5


def ParseDate(s):
    s = str(s)[:10]
    return datetime.strptime(s, "%Y-%m-%d")


def Possessions(fg2a_or_fga, oreb, tov, fta, fga=None):
    """Standard possession estimate. `fga` is field-goal attempts."""
    fga = fga if fga is not None else fg2a_or_fga
    return float(fga or 0) - float(oreb or 0) + float(tov or 0) + 0.44 * float(fta or 0)


def ParseMatchup(matchup):
    m = matchup or ""
    if " vs. " in m:
        team, opp = m.split(" vs. ", 1)
        return team.strip(), opp.strip(), True
    if " @ " in m:
        team, opp = m.split(" @ ", 1)
        return team.strip(), opp.strip(), False
    return None, None, None


def Shrink(num, den, mu, k):
    k = max(float(k), 0.0)
    den = float(den)
    if den + k <= 0:
        return float(mu)
    return (float(num) + k * float(mu)) / (den + k)


def EwmaAlpha(halflife):
    return 1.0 - 0.5 ** (1.0 / max(float(halflife), 1e-6))


def PoissonGe(k, lam):
    lam = max(float(lam), 0.0)
    if k <= 0:
        return 1.0
    return float(1.0 - poisson.cdf(int(k) - 1, lam))


def PoissonPmF(k, lam):
    return float(poisson.pmf(int(k), max(float(lam), 0.0)))


def NbParams(lam, alpha):
    lam = max(float(lam), 0.0)
    alpha = float(alpha)
    if alpha <= 1e-12:
        return None
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * lam)
    return n, p


def NbGe(k, lam, alpha):
    got = NbParams(lam, alpha)
    if got is None:
        return PoissonGe(k, lam)
    if k <= 0:
        return 1.0
    n, p = got
    return float(1.0 - nbinom.cdf(int(k) - 1, n, p))


def NbCdf(k, lam, alpha):
    got = NbParams(lam, alpha)
    if got is None:
        return float(poisson.cdf(int(k), max(float(lam), 0.0)))
    n, p = got
    return float(nbinom.cdf(int(k), n, p))


def NbPmf(k, lam, alpha):
    got = NbParams(lam, alpha)
    if got is None:
        return PoissonPmF(k, lam)
    n, p = got
    return float(nbinom.pmf(int(k), n, p))


def GaussianCopulaC(u, v, rho):
    """C(u, v) = P(U<=u, V<=v) for a Gaussian copula with correlation rho."""
    u = float(min(max(u, 1e-12), 1.0 - 1e-12))
    v = float(min(max(v, 1e-12), 1.0 - 1e-12))
    rho = float(min(max(rho, -0.999), 0.999))
    if abs(rho) < 1e-12:
        return u * v
    z = [norm.ppf(u), norm.ppf(v)]
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    return float(multivariate_normal.cdf(z, mean=[0.0, 0.0], cov=cov))


def JointGe(kx, ky, lamx, lamy, alpha, rho, family="nb"):
    """P(X>=kx, Y>=ky) via a Gaussian copula on the discrete CDFs."""
    if family == "poisson" or alpha is None or alpha <= 1e-12:
        fx = 1.0 - PoissonGe(kx, lamx)  # F(k-1) = 1 - P(>=k)
        fy = 1.0 - PoissonGe(ky, lamy)
    else:
        fx = 1.0 - NbGe(kx, lamx, alpha)
        fy = 1.0 - NbGe(ky, lamy, alpha)
    return max(0.0, min(1.0, 1.0 - fx - fy + GaussianCopulaC(fx, fy, rho)))


def MidPit(y, lam, alpha, family="nb"):
    """Mid-p PIT for a discrete count, clipped off {0,1}."""
    y = int(y)
    if family == "poisson" or alpha is None or alpha <= 1e-12:
        f_lo = 0.0 if y <= 0 else float(poisson.cdf(y - 1, max(lam, 0.0)))
        pmf = PoissonPmF(y, lam)
    else:
        f_lo = 0.0 if y <= 0 else NbCdf(y - 1, lam, alpha)
        pmf = NbPmf(y, lam, alpha)
    u = f_lo + 0.5 * pmf
    return min(max(u, 1e-9), 1.0 - 1e-9)


def FitPoissonOffset(y, offset, x, ridge=RIDGE, max_iter=40):
    """IRLS Poisson GLM: log ? = offset + X?. Returns ?."""
    y = np.asarray(y, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    x = np.c_[np.ones(len(y)), np.asarray(x, dtype=np.float64)]
    n, p = x.shape
    beta = np.zeros(p, dtype=np.float64)
    for _ in range(max_iter):
        eta = offset + x @ beta
        lam = np.exp(np.clip(eta, -20.0, 20.0))
        w = lam
        z = eta + (y - lam) / np.maximum(lam, 1e-8)
        xtw = x.T * w
        a = xtw @ x + ridge * np.eye(p)
        b = xtw @ (z - offset)
        try:
            new = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            break
        if float(np.max(np.abs(new - beta))) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def ApplyOffset(offset, x, beta):
    x = np.c_[np.ones(len(offset)), np.asarray(x, dtype=np.float64)]
    eta = np.asarray(offset, dtype=np.float64) + x @ np.asarray(beta, dtype=np.float64)
    return np.exp(np.clip(eta, -20.0, 20.0))


def MomentAlpha(y, lam):
    """NB2 dispersion: Var = ? + ? ?^2. Returns 0 if underdispersed."""
    y = np.asarray(y, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    num = (y - lam) ** 2 - lam
    den = np.maximum(lam ** 2, 1e-8)
    alpha = float(np.mean(num / den))
    return max(0.0, alpha)


def EmpiricalBayesK(successes, trials, min_trials=80):
    """Beta/binomial prior strength k from players with enough trials. mu, k."""
    rows = [(float(s), float(t)) for s, t in zip(successes, trials) if t and t >= min_trials]
    if len(rows) < 20:
        tot_s = sum(s for s, t in rows) if rows else 0.0
        tot_t = sum(t for s, t in rows) if rows else 0.0
        mu = tot_s / tot_t if tot_t else 0.36
        return mu, 200.0
    rates = np.array([s / t for s, t in rows], dtype=np.float64)
    sizes = np.array([t for s, t in rows], dtype=np.float64)
    mu = float(np.sum([s for s, t in rows]) / np.sum(sizes))
    var = float(np.var(rates, ddof=1))
    sampling = float(np.mean(mu * (1.0 - mu) / sizes))
    tau2 = max(var - sampling, 1e-6)
    k = mu * (1.0 - mu) / tau2 - 1.0
    return mu, float(min(max(k, 20.0), 800.0))


def EmpiricalBayesKRate(counts, exposure, min_exp=200.0):
    """EB strength for a rate (3PA per minute). mu, k in exposure units."""
    rows = [(float(c), float(e)) for c, e in zip(counts, exposure) if e and e >= min_exp]
    if len(rows) < 20:
        tot_c = sum(c for c, e in rows) if rows else 0.0
        tot_e = sum(e for c, e in rows) if rows else 0.0
        mu = tot_c / tot_e if tot_e else 0.12
        return mu, 250.0
    rates = np.array([c / e for c, e in rows], dtype=np.float64)
    exps = np.array([e for c, e in rows], dtype=np.float64)
    mu = float(np.sum([c for c, e in rows]) / np.sum(exps))
    var = float(np.var(rates, ddof=1))
    # Game-level Poisson-ish sampling of attempts: var ? mu/exp
    sampling = float(np.mean(mu / exps))
    tau2 = max(var - sampling, 1e-8)
    k = mu / tau2
    return mu, float(min(max(k, 40.0), 800.0))


def _TeamFromSeasonGame(g, side):
    st = g.get(f"{side}_stats") or {}
    team = g[f"{side}_team"]
    poss = Possessions(st.get("fga"), st.get("oreb"), st.get("tov"), st.get("fta"), st.get("fga"))
    mins = float(st.get("min") or 240)
    return {
        "game_id": g["game_id"],
        "date": str(g["game_date"])[:10],
        "team": team,
        "opp": g["away_team"] if side == "home" else g["home_team"],
        "home": side == "home",
        "poss": poss,
        "min": mins,
        "fg3a": float(st.get("fg3a") or 0),
        "fg3m": float(st.get("fg3m") or 0),
        "pace": 48.0 * poss / max(mins / 5.0, 1e-6) if mins else 0.0,
    }


def TeamGameRows(season_games):
    rows = []
    for g in season_games:
        rows.append(_TeamFromSeasonGame(g, "home"))
        rows.append(_TeamFromSeasonGame(g, "away"))
    return rows


def BuildFeatures(player_rows, season_games):
    """Walk chronologically. Each row's features use only games on earlier dates."""
    team_rows = TeamGameRows(season_games)
    by_date_teams = defaultdict(list)
    for r in team_rows:
        by_date_teams[r["date"]].append(r)
    by_date_players = defaultdict(list)
    for r in player_rows:
        by_date_players[str(r["GAME_DATE"])[:10]].append(r)
    dates = sorted(set(by_date_players) | set(by_date_teams))

    # Running pregame state (updated at the end of each date).
    pst = {}  # pid -> accumulators
    tst = {}  # team -> accumulators
    alpha = EwmaAlpha(EWMA_HALFLIFE)
    out = []

    def PState(pid, name):
        if pid not in pst:
            pst[pid] = {
                "name": name, "n": 0, "min": 0.0, "fg3m": 0.0, "fg3a": 0.0,
                "pts": 0.0, "reb": 0.0, "ast": 0.0,
                "ewma_min": None, "last_date": None, "mins": [],
            }
        return pst[pid]

    def TState(team):
        if team not in tst:
            tst[team] = {
                "n": 0, "poss": 0.0, "min": 0.0, "fg3a_for": 0.0, "fg3a_against": 0.0,
                "poss_against": 0.0, "last_date": None,
            }
        return tst[team]

    for d in dates:
        # Snapshot team pregame metrics for this date.
        team_pre = {}
        league_poss = league_min = league_3a = 0.0
        n_team = 0
        for team, st in tst.items():
            if st["n"] <= 0:
                continue
            pace = 48.0 * st["poss"] / max(st["min"] / 5.0, 1e-6)
            opp_3pa100 = 100.0 * st["fg3a_against"] / max(st["poss_against"], 1e-6)
            team_pre[team] = {
                "n": st["n"], "pace": pace, "opp_3pa100": opp_3pa100,
                "last_date": st["last_date"],
            }
            league_poss += st["poss"]
            league_min += st["min"]
            league_3a += st["fg3a_against"]
            n_team += 1
        league_pace = 48.0 * league_poss / max(league_min / 5.0, 1e-6) if league_min else 100.0
        league_opp3 = 100.0 * league_3a / max(league_poss, 1e-6) if league_poss else 35.0

        for r in by_date_players[d]:
            pid = r["PLAYER_ID"]
            name = r["PLAYER_NAME"]
            st = PState(pid, name)
            team, opp, home = ParseMatchup(r.get("MATCHUP"))
            team = team or r.get("TEAM_ABBREVIATION")
            if home is None:
                home = " vs. " in (r.get("MATCHUP") or "")
            rest = None
            if st["last_date"] is not None:
                rest = (ParseDate(d) - ParseDate(st["last_date"])).days
            b2b = bool(rest == 1)
            tp = team_pre.get(team) or {}
            op = team_pre.get(opp) or {}
            pace = None
            if tp.get("pace") and op.get("pace"):
                pace = 0.5 * (tp["pace"] + op["pace"])
            elif tp.get("pace"):
                pace = tp["pace"]
            elif op.get("pace"):
                pace = op["pace"]
            opp_3pa100 = op.get("opp_3pa100")
            proj_min = st["ewma_min"]
            if proj_min is None and st["n"] == 0:
                proj_min = None
            out.append({
                "player_id": pid,
                "player_name": name,
                "team": team,
                "opp": opp,
                "game_id": r["GAME_ID"],
                "date": d,
                "season": int(str(r.get("SEASON_ID") or "0")[-4:] or 0) if r.get("SEASON_ID") else None,
                "home": bool(home),
                "rest_days": rest,
                "b2b": b2b,
                "actual_min": float(r.get("MIN") or 0),
                "actual_3pm": float(r.get("FG3M") or 0),
                "actual_3pa": float(r.get("FG3A") or 0),
                "prior_n": st["n"],
                "prior_min": st["min"],
                "prior_3pm": st["fg3m"],
                "prior_3pa": st["fg3a"],
                "prior_pts": st.get("pts") or 0.0,
                "prior_reb": st.get("reb") or 0.0,
                "prior_ast": st.get("ast") or 0.0,
                "actual_pts": float(r.get("PTS") or 0),
                "actual_reb": float(r.get("REB") or 0),
                "actual_ast": float(r.get("AST") or 0),
                "proj_min_raw": proj_min,
                "pace": pace,
                "opp_3pa100": opp_3pa100,
                "league_pace": league_pace,
                "league_opp3": league_opp3,
            })

        # End-of-date updates (no same-day lookahead).
        for r in by_date_players[d]:
            st = PState(r["PLAYER_ID"], r["PLAYER_NAME"])
            mins = float(r.get("MIN") or 0)
            st["n"] += 1
            st["min"] += mins
            st["fg3m"] += float(r.get("FG3M") or 0)
            st["fg3a"] += float(r.get("FG3A") or 0)
            st["pts"] = st.get("pts", 0.0) + float(r.get("PTS") or 0)
            st["reb"] = st.get("reb", 0.0) + float(r.get("REB") or 0)
            st["ast"] = st.get("ast", 0.0) + float(r.get("AST") or 0)
            st["mins"].append(mins)
            if st["ewma_min"] is None:
                st["ewma_min"] = mins
            else:
                st["ewma_min"] = alpha * mins + (1.0 - alpha) * st["ewma_min"]
            st["last_date"] = d
            st["name"] = r["PLAYER_NAME"]
        # Pair team rows on this date so each side's "against" is the other.
        by_gid = defaultdict(list)
        for tr in by_date_teams[d]:
            by_gid[tr["game_id"]].append(tr)
        for gid, recs in by_gid.items():
            if len(recs) != 2:
                for tr in recs:
                    ts = TState(tr["team"])
                    ts["n"] += 1
                    ts["poss"] += tr["poss"]
                    ts["min"] += tr["min"]
                    ts["fg3a_for"] += tr["fg3a"]
                    ts["last_date"] = d
                continue
            a, b = recs
            for me, opp in ((a, b), (b, a)):
                ts = TState(me["team"])
                ts["n"] += 1
                ts["poss"] += me["poss"]
                ts["min"] += me["min"]
                ts["fg3a_for"] += me["fg3a"]
                ts["fg3a_against"] += opp["fg3a"]
                ts["poss_against"] += opp["poss"]
                ts["last_date"] = d

    # Attach season from game_id when SEASON_ID missing: 00224.... -> 2024
    for row in out:
        if not row["season"]:
            gid = str(row["game_id"])
            if len(gid) >= 5 and gid[3:5].isdigit():
                yy = int(gid[3:5])
                row["season"] = 2000 + yy if yy >= 50 else 2000 + yy
                if yy < 50:
                    row["season"] = 2000 + yy
                else:
                    row["season"] = 1900 + yy
            # NBA game ids are 0022600001 -> season 2026 (starts year)
            if len(gid) >= 5 and gid.startswith("002"):
                row["season"] = 2000 + int(gid[3:5])
    return out, pst, tst


def AttachPredictions(rows, params):
    """Apply frozen shrinkage, minutes cap, and GLM adjustments. Mutates and returns rows."""
    mu_pct = params["mu_pct"]
    k_pct = params["k_pct"]
    mu_rate = params["mu_rate"]
    k_rate = params["k_rate"]
    beta = np.asarray(params["beta"], dtype=np.float64)
    alpha = params["alpha"]
    league_pace = params.get("league_pace") or 100.0
    league_opp3 = params.get("league_opp3") or 35.0
    pace_sd = params.get("pace_sd") or 3.0
    opp3_sd = params.get("opp3_sd") or 4.0

    for row in rows:
        proj = row["proj_min_raw"]
        if proj is None:
            # No NBA minutes history: start at a cautious rotation default, flagged.
            proj = 16.0
            row["minutes_defaulted"] = True
        else:
            row["minutes_defaulted"] = False
        proj = float(min(max(proj, 0.0), 42.0))
        row["proj_min"] = proj
        row["rate_3pa"] = Shrink(row["prior_3pa"], row["prior_min"], mu_rate, k_rate)
        row["pct_3p"] = Shrink(row["prior_3pm"], row["prior_3pa"], mu_pct, k_pct)
        base = max(proj * row["rate_3pa"] * row["pct_3p"], 1e-6)
        row["base_lam"] = base
        pace = row["pace"] if row["pace"] is not None else league_pace
        opp3 = row["opp_3pa100"] if row["opp_3pa100"] is not None else league_opp3
        row["pace_z"] = (pace - league_pace) / max(pace_sd, 1e-6)
        row["opp3_z"] = (opp3 - league_opp3) / max(opp3_sd, 1e-6)
        x = np.array([[float(row["home"]), float(row["b2b"]), row["pace_z"], row["opp3_z"]]], dtype=np.float64)
        row["lam"] = float(ApplyOffset([math.log(base)], x, beta)[0])
        row["alpha"] = alpha
        for k in THRESHOLDS:
            row[f"p_pois_{k}"] = PoissonGe(k, row["lam"])
            row[f"p_nb_{k}"] = NbGe(k, row["lam"], alpha)
    return rows


def FitParams(train_rows):
    """Estimate shrinkage, GLM adjustments, and NB dispersion on 2024-25 rows only."""
    # Player-season (within train) totals for EB, using actuals ? this is the prior
    # strength, not a per-game leak into holdout features. Features themselves stay
    # leak-free via BuildFeatures.
    by_p = defaultdict(lambda: {"min": 0.0, "fg3m": 0.0, "fg3a": 0.0})
    for r in train_rows:
        rec = by_p[r["player_id"]]
        rec["min"] += r["actual_min"]
        rec["fg3m"] += r["actual_3pm"]
        rec["fg3a"] += r["actual_3pa"]
    mu_pct, k_pct = EmpiricalBayesK(
        [v["fg3m"] for v in by_p.values()],
        [v["fg3a"] for v in by_p.values()],
        min_trials=80,
    )
    mu_rate, k_rate = EmpiricalBayesKRate(
        [v["fg3a"] for v in by_p.values()],
        [v["min"] for v in by_p.values()],
        min_exp=200.0,
    )

    fit_rows = [r for r in train_rows
                if r["prior_n"] >= 5 and (r["proj_min_raw"] or 0) >= MIN_PROJ
                and r["prior_3pa"] >= 10]
    paces = [r["pace"] for r in fit_rows if r["pace"] is not None]
    opp3s = [r["opp_3pa100"] for r in fit_rows if r["opp_3pa100"] is not None]
    league_pace = float(np.mean(paces)) if paces else 100.0
    league_opp3 = float(np.mean(opp3s)) if opp3s else 35.0
    pace_sd = float(np.std(paces, ddof=1)) if len(paces) > 2 else 3.0
    opp3_sd = float(np.std(opp3s, ddof=1)) if len(opp3s) > 2 else 4.0
    pace_sd = max(pace_sd, 1.0)
    opp3_sd = max(opp3_sd, 1.0)

    # Temporary params so we can form the offset from shrunk rates on train.
    tmp = {
        "mu_pct": mu_pct, "k_pct": k_pct, "mu_rate": mu_rate, "k_rate": k_rate,
        "beta": np.zeros(5), "alpha": 0.0,
        "league_pace": league_pace, "league_opp3": league_opp3,
        "pace_sd": pace_sd, "opp3_sd": opp3_sd,
    }
    # Attach base_lam / z without using the (zero) adjustment.
    prepared = []
    for row in fit_rows:
        proj = float(min(max(row["proj_min_raw"], 0.0), 42.0))
        rate = Shrink(row["prior_3pa"], row["prior_min"], mu_rate, k_rate)
        pct = Shrink(row["prior_3pm"], row["prior_3pa"], mu_pct, k_pct)
        base = max(proj * rate * pct, 1e-6)
        pace = row["pace"] if row["pace"] is not None else league_pace
        opp3 = row["opp_3pa100"] if row["opp_3pa100"] is not None else league_opp3
        prepared.append({
            "y": row["actual_3pm"],
            "offset": math.log(base),
            "x": [float(row["home"]), float(row["b2b"]),
                  (pace - league_pace) / pace_sd, (opp3 - league_opp3) / opp3_sd],
            "base": base,
        })
    y = [p["y"] for p in prepared]
    offset = [p["offset"] for p in prepared]
    x = [p["x"] for p in prepared]
    beta = FitPoissonOffset(y, offset, x) if prepared else np.zeros(5)
    lam = ApplyOffset(offset, x, beta) if prepared else np.array([])
    alpha = MomentAlpha(y, lam) if len(y) else 0.0
    params = {
        "mu_pct": mu_pct, "k_pct": k_pct, "mu_rate": mu_rate, "k_rate": k_rate,
        "beta": [float(v) for v in beta],
        "alpha": float(alpha),
        "league_pace": league_pace, "league_opp3": league_opp3,
        "pace_sd": pace_sd, "opp3_sd": opp3_sd,
        "n_fit": len(prepared),
        "ewma_halflife": EWMA_HALFLIFE,
        "train_season": TRAIN_SEASON,
    }
    return params


def EstimateRho(rows, params, family="nb"):
    """Gaussian-copula rho from mid-p PIT of same-team rotation pairs in `rows`."""
    by_game_team = defaultdict(list)
    for r in rows:
        if (r.get("lam") is None) or (r.get("proj_min") or 0) < MIN_PROJ:
            continue
        if r["prior_3pa"] < 10:
            continue
        by_game_team[(r["game_id"], r["team"])].append(r)
    zs_a, zs_b = [], []
    alpha = params["alpha"]
    n_pairs = 0
    for recs in by_game_team.values():
        if len(recs) < 2:
            continue
        recs = sorted(recs, key=lambda r: r["player_id"])
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                ua = MidPit(recs[i]["actual_3pm"], recs[i]["lam"], alpha, family)
                ub = MidPit(recs[j]["actual_3pm"], recs[j]["lam"], alpha, family)
                zs_a.append(norm.ppf(ua))
                zs_b.append(norm.ppf(ub))
                n_pairs += 1
    if n_pairs < 30:
        return {"rho": 0.0, "n_pairs": n_pairs, "pearson": None}
    a = np.asarray(zs_a, dtype=np.float64)
    b = np.asarray(zs_b, dtype=np.float64)
    # Clip tails so a single 0/1 PIT cannot dominate.
    a = np.clip(a, -4.0, 4.0)
    b = np.clip(b, -4.0, 4.0)
    pearson = float(np.corrcoef(a, b)[0, 1])
    if not math.isfinite(pearson):
        pearson = 0.0
    rho = float(min(max(pearson, -RHO_CLIP), RHO_CLIP))
    return {"rho": rho, "n_pairs": n_pairs, "pearson": pearson}


def FilterCalib(rows):
    return [r for r in rows
            if (r.get("proj_min") or 0) >= MIN_PROJ and r.get("prior_n", 0) >= 5
            and (r.get("prior_3pa") or 0) >= 10 and r.get("lam") is not None]


def TonightState(player_rows, season_games, asof_date):
    """Build leak-free running state using only games with date < asof_date."""
    pcut = [r for r in player_rows if str(r["GAME_DATE"])[:10] < asof_date]
    tcut = [g for g in season_games if str(g["game_date"])[:10] < asof_date]
    _, pst, tst = BuildFeatures(pcut, tcut)
    return pst, tst, pcut, tcut


def PregameRow(pid, name, team, opp, home, date, pst, tst, last_team_dates=None):
    """Assemble a single pregame feature row from the running state."""
    st = pst.get(pid) or {
        "name": name, "n": 0, "min": 0.0, "fg3m": 0.0, "fg3a": 0.0,
        "pts": 0.0, "reb": 0.0, "ast": 0.0,
        "ewma_min": None, "last_date": None, "mins": [],
    }
    def team_metrics(abbr):
        ts = tst.get(abbr)
        if not ts or ts["n"] <= 0:
            return None
        pace = 48.0 * ts["poss"] / max(ts["min"] / 5.0, 1e-6)
        opp_3pa100 = 100.0 * ts["fg3a_against"] / max(ts["poss_against"], 1e-6)
        return {"pace": pace, "opp_3pa100": opp_3pa100, "n": ts["n"], "last_date": ts["last_date"]}
    tp, op = team_metrics(team), team_metrics(opp)
    league_poss = sum(ts["poss"] for ts in tst.values() if ts["n"])
    league_min = sum(ts["min"] for ts in tst.values() if ts["n"])
    league_3a = sum(ts["fg3a_against"] for ts in tst.values() if ts["n"])
    league_pace = 48.0 * league_poss / max(league_min / 5.0, 1e-6) if league_min else 100.0
    league_opp3 = 100.0 * league_3a / max(league_poss, 1e-6) if league_poss else 35.0
    pace = None
    if tp and op:
        pace = 0.5 * (tp["pace"] + op["pace"])
    elif tp:
        pace = tp["pace"]
    elif op:
        pace = op["pace"]
    rest = None
    if st.get("last_date"):
        rest = (ParseDate(date) - ParseDate(st["last_date"])).days
    return {
        "player_id": pid,
        "player_name": name or st.get("name"),
        "team": team,
        "opp": opp,
        "game_id": f"pregame-{date}-{team}",
        "date": date,
        "season": CURRENT_SEASON,
        "home": bool(home),
        "rest_days": rest,
        "b2b": bool(rest == 1),
        "actual_min": None,
        "actual_3pm": None,
        "actual_3pa": None,
        "prior_n": st.get("n") or 0,
        "prior_min": st.get("min") or 0.0,
        "prior_3pm": st.get("fg3m") or 0.0,
        "prior_3pa": st.get("fg3a") or 0.0,
        "prior_pts": st.get("pts") or 0.0,
        "prior_reb": st.get("reb") or 0.0,
        "prior_ast": st.get("ast") or 0.0,
        "proj_min_raw": st.get("ewma_min"),
        "recent_mins": list(st.get("mins") or [])[-10:],
        "pace": pace,
        "opp_3pa100": (op or {}).get("opp_3pa100"),
        "league_pace": league_pace,
        "league_opp3": league_opp3,
        "last_played": st.get("last_date"),
    }


def RangeRow(row, params, min_lo, min_hi, k_mult_lo=2.0, k_mult_hi=0.5):
    """Low/high ? by widening minutes and shrinkage for a low-sample player."""
    mu_pct, mu_rate = params["mu_pct"], params["mu_rate"]
    beta = params["beta"]
    def lam_at(proj, k_pct, k_rate):
        rate = Shrink(row["prior_3pa"], row["prior_min"], mu_rate, k_rate)
        pct = Shrink(row["prior_3pm"], row["prior_3pa"], mu_pct, k_pct)
        base = max(proj * rate * pct, 1e-6)
        x = np.array([[float(row["home"]), float(row["b2b"]),
                       row.get("pace_z") or 0.0, row.get("opp3_z") or 0.0]])
        return float(ApplyOffset([math.log(base)], x, beta)[0]), rate, pct, proj
    lo, rate_lo, pct_lo, min_l = lam_at(min_lo, params["k_pct"] * k_mult_lo, params["k_rate"] * k_mult_lo)
    hi, rate_hi, pct_hi, min_h = lam_at(min_hi, params["k_pct"] * k_mult_hi, params["k_rate"] * k_mult_hi)
    if lo > hi:
        lo, hi = hi, lo
        rate_lo, rate_hi = rate_hi, rate_lo
        pct_lo, pct_hi = pct_hi, pct_lo
        min_l, min_h = min_h, min_l
    return {"lam_lo": lo, "lam_hi": hi, "min_lo": min_l, "min_hi": min_h,
            "rate_lo": rate_lo, "rate_hi": rate_hi, "pct_lo": pct_lo, "pct_hi": pct_hi}



STAT_KEYS = {
    "pts": ("prior_pts", "actual_pts"),
    "reb": ("prior_reb", "actual_reb"),
    "ast": ("prior_ast", "actual_ast"),
    "pra": None,  # derived
}


def _StatPair(row, family):
    if family == "3pm":
        return row.get("prior_3pm") or 0.0, row.get("actual_3pm") or 0.0
    if family == "pra":
        prior = (row.get("prior_pts") or 0.0) + (row.get("prior_reb") or 0.0) + (row.get("prior_ast") or 0.0)
        actual = (row.get("actual_pts") or 0.0) + (row.get("actual_reb") or 0.0) + (row.get("actual_ast") or 0.0)
        return prior, actual
    keys = STAT_KEYS.get(family)
    if not keys:
        return 0.0, 0.0
    return row.get(keys[0]) or 0.0, row.get(keys[1]) or 0.0


def FitCountParams(train_rows, family):
    """NB/EB per-minute rate model for PTS, REB, AST, or PRA. Frozen on train_rows."""
    by_p = {}
    for r in train_rows:
        rec = by_p.setdefault(r["player_id"], {"min": 0.0, "stat": 0.0})
        rec["min"] += r.get("actual_min") or 0.0
        rec["stat"] += _StatPair(r, family)[1]
    mu_rate, k_rate = EmpiricalBayesKRate(
        [v["stat"] for v in by_p.values()],
        [v["min"] for v in by_p.values()],
        min_exp=200.0,
    )
    fit_rows = [r for r in train_rows
                if r["prior_n"] >= 5 and (r["proj_min_raw"] or 0) >= MIN_PROJ]
    paces = [r["pace"] for r in fit_rows if r["pace"] is not None]
    opp3s = [r["opp_3pa100"] for r in fit_rows if r["opp_3pa100"] is not None]
    league_pace = float(np.mean(paces)) if paces else 100.0
    league_opp3 = float(np.mean(opp3s)) if opp3s else 35.0
    pace_sd = max(float(np.std(paces, ddof=1)) if len(paces) > 2 else 3.0, 1.0)
    opp3_sd = max(float(np.std(opp3s, ddof=1)) if len(opp3s) > 2 else 4.0, 1.0)
    prepared = []
    for row in fit_rows:
        proj = float(min(max(row["proj_min_raw"] or 0.0, 0.0), 42.0))
        prior_stat, _ = _StatPair(row, family)
        rate = Shrink(prior_stat, row["prior_min"], mu_rate, k_rate)
        base = max(proj * rate, 1e-6)
        pace = row["pace"] if row["pace"] is not None else league_pace
        opp3 = row["opp_3pa100"] if row["opp_3pa100"] is not None else league_opp3
        prepared.append({
            "y": _StatPair(row, family)[1],
            "offset": math.log(base),
            "x": [float(row["home"]), float(row["b2b"]),
                  (pace - league_pace) / pace_sd, (opp3 - league_opp3) / opp3_sd],
        })
    y = [p["y"] for p in prepared]
    offset = [p["offset"] for p in prepared]
    x = [p["x"] for p in prepared]
    beta = FitPoissonOffset(y, offset, x) if prepared else np.zeros(5)
    lam = ApplyOffset(offset, x, beta) if prepared else np.array([])
    alpha = MomentAlpha(y, lam) if len(y) else 0.0
    return {
        "family": family,
        "mu_rate": mu_rate, "k_rate": k_rate,
        "beta": [float(v) for v in beta],
        "alpha": float(alpha),
        "league_pace": league_pace, "league_opp3": league_opp3,
        "pace_sd": pace_sd, "opp3_sd": opp3_sd,
        "n_fit": len(prepared),
        "ewma_halflife": EWMA_HALFLIFE,
        "train_season": TRAIN_SEASON,
    }


def AttachCountPredictions(rows, params, family=None):
    """Set lam_<family> on each row from a frozen count-model params dict."""
    family = family or params.get("family")
    mu_rate, k_rate = params["mu_rate"], params["k_rate"]
    beta = params["beta"]
    league_pace = params.get("league_pace") or 100.0
    league_opp3 = params.get("league_opp3") or 35.0
    pace_sd = params.get("pace_sd") or 3.0
    opp3_sd = params.get("opp3_sd") or 4.0
    for row in rows:
        proj = row.get("proj_min")
        if proj is None:
            raw = row.get("proj_min_raw")
            proj = 16.0 if raw is None else float(min(max(raw, 0.0), 42.0))
            row["proj_min"] = proj
        prior_stat, _ = _StatPair(row, family)
        rate = Shrink(prior_stat, row.get("prior_min") or 0.0, mu_rate, k_rate)
        base = max(proj * rate, 1e-6)
        pace = row["pace"] if row.get("pace") is not None else league_pace
        opp3 = row["opp_3pa100"] if row.get("opp_3pa100") is not None else league_opp3
        pace_z = (pace - league_pace) / max(pace_sd, 1e-6)
        opp3_z = (opp3 - league_opp3) / max(opp3_sd, 1e-6)
        x = np.array([[float(row.get("home") or 0), float(row.get("b2b") or 0), pace_z, opp3_z]])
        row[f"lam_{family}"] = float(ApplyOffset([math.log(base)], x, beta)[0])
        row[f"rate_{family}"] = rate
        row[f"alpha_{family}"] = params["alpha"]
    return rows


def FairGe(row, family, k, params=None):
    """P(count >= k) from the attached lambda and frozen alpha."""
    if family == "3pm":
        lam = row.get("lam")
        alpha = row.get("alpha") if row.get("alpha") is not None else (params or {}).get("alpha", 0.0)
    else:
        lam = row.get(f"lam_{family}")
        alpha = row.get(f"alpha_{family}")
        if alpha is None and params is not None:
            alpha = params.get("alpha", 0.0)
    if lam is None:
        return None
    return NbGe(k, lam, alpha or 0.0)
