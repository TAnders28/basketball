"""Load the panel into per-venue Game lists and define splits."""

import os
import pickle

from InPlay.Engine import Game

ROOT = os.path.join("data", "inplay")
SPLITS = {"tune": (2024, 2025), "holdout": (2026,), "forward": (2027,)}
PM_SPREAD = 0.01  # assumed Polymarket quoted spread (no historical book): bid = p - 0.005, ask = p + 0.005
PM_MAX_AGE = 120


def SplitOf(season):
    for name, seasons in SPLITS.items():
        if season in seasons:
            return name
    return None


def _F(x):
    if x is None:
        return None
    try:
        import math
        x = float(x)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def BuildGames(panel, games):
    """Return {"kalshi": [Game], "pm": [Game]} using in-play bars only."""
    meta = games.set_index("game_id")
    out = {"kalshi": [], "pm": []}
    for gid, df in panel[panel.inplay == 1].sort_values(["game_id", "t"]).groupby("game_id", sort=False):
        g = meta.loc[gid]
        if g.get("reason") is not None and isinstance(g.get("reason"), str):
            continue
        won = _F(g["home_won"])
        if won is None:
            continue
        base = dict(gid=gid, day=g["date_et"], season=int(g["season"]), stype=int(g["season_type"]),
                    won=int(won), t=df.t.tolist(), el=df.el.astype(float).tolist(), per=df.per.astype(int).tolist(),
                    sd=df.sd.astype(int).tolist(), wp=[_F(x) for x in df.wp], total=_F(g.get("book_total")),
                    spread=_F(g.get("book_spread")))
        kpre = _F(g.get("k_pre"))
        if kpre is not None and g.get("kalshi_ticker") is not None and df.k_mid.notna().sum() > 30:
            pm_vals = [(_F(p) if (a is not None and _F(a) is not None and _F(a) <= 60) else None)
                       for p, a in zip(df.pm, df.pm_age)]
            out["kalshi"].append(Game(**base, venue="kalshi", pre=kpre, bid=[_F(x) for x in df.k_bid],
                                      ask=[_F(x) for x in df.k_ask], mid=[_F(x) for x in df.k_mid],
                                      trlo=[_F(x) for x in df.k_trlo], trhi=[_F(x) for x in df.k_trhi],
                                      pm=pm_vals, pmage=[_F(x) for x in df.pm_age]))
        ppre = _F(g.get("pm_pre"))
        if ppre is not None and g.get("pm_slug") is not None and df.pm.notna().sum() > 30:
            mid = [(_F(p) if (_F(a) is not None and _F(a) <= PM_MAX_AGE) else None) for p, a in zip(df.pm, df.pm_age)]
            bid = [None if m is None else max(0.0, m - PM_SPREAD / 2) for m in mid]
            ask = [None if m is None else min(1.0, m + PM_SPREAD / 2) for m in mid]
            n = len(mid)
            out["pm"].append(Game(**base, venue="pm", pre=ppre, bid=bid, ask=ask, mid=mid, trlo=[None] * n,
                                  trhi=[None] * n, pm=[None] * n, pmage=[None] * n))
    return out


def LoadGames(path=os.path.join(ROOT, "games_obj.pkl")):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def SaveGames(obj, path=os.path.join(ROOT, "games_obj.pkl")):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)