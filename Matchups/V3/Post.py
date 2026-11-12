r"""
Post-processing on held-out (pregame) prediction rows; every fit uses only rows from strictly earlier days.

Emos (Gneiting et al. 2005, weather MOS/EMOS): margin ~ N(a + b * mu, exp(c) + exp(d) * var), fitted by minimum
mean CRPS (Nelder-Mead) on a trailing window, refit weekly. Before min_rows trailing games it passes through.

Temperature (Guo et al. 2017 / Platt): p' = sigmoid(t0 + t1 * logit p), Bayesian logistic (Laplace) on a trailing
window with prior N([0, 1], 0.5^2), refit weekly. The margin distribution is re-scaled to match p'.

ContextStack (Yao et al. 2018, 2021 stacking and hierarchical stacking of predictive distributions): mixture of
member margin distributions with weights maximizing the trailing log predictive density (EM), a global weight
vector with a Dirichlet(1 + 0.1) prior and context weights (early season / roster churn / back-to-back / normal)
shrunk to the global weights with Dirichlet pseudo-counts alpha_c * w_global. Refit weekly; rows of the target
day are never in the fit.
"""

import math
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

from Matchups.V2 import DataLayer as D
from Matchups.V2.Bayes import LaplaceLogistic
from Matchups.V3.Evaluate import InvPhi, NormalCrps, Phi


def NelderMead(f, x0, step=0.3, iters=300, tol=1e-7):
    n = len(x0)
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + step * np.eye(n)[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], 0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    return pts[int(np.argmin(vals))]


def _Days(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["game_date"]].append(r)
    return sorted(by.items())


def Trailing(history, day, window):
    lo = (date.fromisoformat(day) - timedelta(days=window)).isoformat()
    return [r for r in history if lo <= r["game_date"] < day]


def Emos(rows, window=150, min_rows=300, refit_every=7):
    out, history, theta, last_fit, fitted = [], [], np.array([0.0, 1.0, 0.0, 0.0]), None, False
    for day, day_rows in _Days(rows):
        if last_fit is None or (date.fromisoformat(day) - date.fromisoformat(last_fit)).days >= refit_every:
            tr = Trailing(history, day, window)
            if len(tr) >= min_rows:
                y = np.array([r["margin"] for r in tr], float)
                mu = np.array([r["margin_mean"] for r in tr], float)
                v = np.array([r["margin_var"] for r in tr], float)

                def F(t):
                    var = math.exp(t[2]) + math.exp(t[3]) * v
                    return float(np.mean(NormalCrps(y, t[0] + t[1] * mu, var)))
                theta = NelderMead(F, theta if fitted else np.array([0.0, 1.0, 0.0, 0.0]))
                fitted = True
            last_fit = day
        for r in day_rows:
            m = theta[0] + theta[1] * r["margin_mean"]
            v = math.exp(theta[2]) + math.exp(theta[3]) * r["margin_var"] if fitted else r["margin_var"]
            out.append(dict(r, margin_mean=float(m), margin_var=float(v), p_home=float(Phi(m / math.sqrt(v))),
                            emos=[round(float(x), 4) for x in theta]))
        history.extend(day_rows)
    return out


def Temperature(rows, window=150, min_rows=300, refit_every=7):
    out, history, w, last_fit = [], [], np.array([0.0, 1.0]), None
    for day, day_rows in _Days(rows):
        if last_fit is None or (date.fromisoformat(day) - date.fromisoformat(last_fit)).days >= refit_every:
            tr = Trailing(history, day, window)
            if len(tr) >= min_rows:
                p = np.clip([r["p_home"] for r in tr], 1e-6, 1 - 1e-6)
                X = np.c_[np.ones(len(tr)), np.log(p / (1 - p))]
                w, _ = LaplaceLogistic(X, np.array([r["home_win"] for r in tr], float), [0.0, 1.0], 0.25)
            last_fit = day
        for r in day_rows:
            p = min(max(r["p_home"], 1e-6), 1 - 1e-6)
            q = 1 / (1 + math.exp(-(w[0] + w[1] * math.log(p / (1 - p)))))
            sd = math.sqrt(r["margin_var"])
            out.append(dict(r, p_home=q, margin_mean=float(sd * InvPhi(q)), temperature=[round(float(x), 4) for x in w]))
        history.extend(day_rows)
    return out


def GameContexts(games):
    """Pregame context per game from games already played: early (first 30 days of a season), churn (>= 2 of
    the two teams' top-8 minute players missed their previous game), b2b (either team played yesterday), normal."""
    ctx, last_day, first_day = {}, {}, {}
    top = defaultdict(lambda: defaultdict(float))
    last_played = {}
    for g in sorted(games, key=lambda g: (g["game_date"], g["game_id"])):
        s, d = g["season"], g["game_date"]
        first_day.setdefault(s, d)
        early = (date.fromisoformat(d) - date.fromisoformat(first_day[s])).days < 30
        churn = 0
        for t in (g["home_team"], g["away_team"]):
            t8 = sorted(top[t], key=lambda p: -top[t][p])[:8]
            lp = last_played.get(t, set())
            churn += sum(1 for p in t8 if lp and p not in lp)
        y = (date.fromisoformat(d) - timedelta(days=1)).isoformat()
        b2b = last_day.get(g["home_team"]) == y or last_day.get(g["away_team"]) == y
        ctx[g["game_id"]] = "early" if early else "churn" if churn >= 2 else "b2b" if b2b else "normal"
        rows = D.PlayerBox(s).get(g["game_id"], [])
        for t in (g["home_team"], g["away_team"]):
            last_day[t] = d
            played = {r["player_id"] for r in rows if r["team"] == t and (r["minutes"] or 0) > 0}
            if played:
                last_played[t] = played
            for p in top[t]:
                top[t][p] *= 0.9
            for r in rows:
                if r["team"] == t:
                    top[t][r["player_id"]] += r["minutes"] or 0
    return ctx


def _Dens(y, mu, var):
    return np.exp(-0.5 * (y - mu) ** 2 / var) / np.sqrt(2 * math.pi * var)


def EmWeights(F, prior, iters=2000):
    """F: games x members densities. MAP mixture weights under Dirichlet(prior) pseudo-counts."""
    K = F.shape[1]
    w = np.full(K, 1.0 / K)
    for _ in range(iters):
        R = F * w
        R /= np.maximum(R.sum(1, keepdims=True), 1e-300)
        w_new = R.sum(0) + prior
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < 1e-7:
            w = w_new
            break
        w = w_new
    return w


def ContextStack(member_rows, contexts, window=365, min_rows=300, refit_every=7, alpha_c=30.0, contextual=True):
    """member_rows: {name: [rows]} on the same games. Returns stacked rows with a Normal-mixture margin."""
    names = sorted(member_rows)
    by_game = defaultdict(dict)
    for n in names:
        for r in member_rows[n]:
            by_game[r["game_id"]][n] = r
    games = [g for g, d in by_game.items() if len(d) == len(names)]
    rows = sorted((by_game[g][names[0]] for g in games), key=lambda r: (r["game_date"], r["game_id"]))
    K = len(names)
    w_glob, w_ctx, last_fit, hist = np.full(K, 1 / K), {}, None, []
    out = []
    for day, day_rows in _Days(rows):
        if last_fit is None or (date.fromisoformat(day) - date.fromisoformat(last_fit)).days >= refit_every:
            tr = Trailing(hist, day, window)
            if len(tr) >= min_rows:
                F = np.array([[_Dens(r["margin"], by_game[r["game_id"]][n]["margin_mean"], by_game[r["game_id"]][n]["margin_var"])
                               for n in names] for r in tr])
                w_glob = EmWeights(F, np.full(K, 0.1))
                w_ctx = {}
                if contextual:
                    cl = np.array([contexts.get(r["game_id"], "normal") for r in tr])
                    for c in set(cl):
                        w_ctx[c] = EmWeights(F[cl == c], alpha_c * w_glob)
            last_fit = day
        for r in day_rows:
            w = w_ctx.get(contexts.get(r["game_id"], "normal"), w_glob) if contextual else w_glob
            mus = np.array([by_game[r["game_id"]][n]["margin_mean"] for n in names])
            vs = np.array([by_game[r["game_id"]][n]["margin_var"] for n in names])
            mean = float(w @ mus)
            var = float(w @ (vs + mus ** 2) - mean ** 2)
            p = float(np.sum(w * Phi(mus / np.sqrt(vs))))
            out.append(dict(r, margin_mean=mean, margin_var=var, p_home=p, mix=(w.tolist(), mus.tolist(), vs.tolist()),
                            context=contexts.get(r["game_id"], "normal")))
        hist.extend(day_rows)
    return out, names, w_glob, w_ctx
