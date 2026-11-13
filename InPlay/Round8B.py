r"""Round 8B (9:09 PM steer, item 2; registered before scoring). Kalshi live totals (KXNBATOTAL) vs a simple pace projection.

Forward-window games only (live API, top-6 strikes by volume per game; free 1-minute candles). Same date split as Round 7C:
games before the median date tune ("f-early"), later games score once ("f-late"). No 2025-26 holdout (historical endpoint).
  * Market median total at minute t: probit fit of P(total > k) across strikes with mids in [0.03, 0.97] (>= 2 strikes).
  * Pace projection at checkpoint (end of Q1 / Q2 / Q3 = el 720 / 1440 / 2160): points so far + remaining regulation
    minutes x blended rate, rate = w * (points / el) + (1 - w) * (market median at tip / 2880), w = el / 2880.
  * Checks: mean absolute error of the final total for market vs pace; slope of (final - market) on (pace - market).
  * Trade: if pace - market >= g points (or <= -g), buy the YES (over) / NO side of the strike nearest the market median at
    the next minute's ask + 0.5c and hold to settlement. g in (3, 6, 10) x checkpoint (Q1, Q2, Q3).

    .venv\Scripts\python.exe -m InPlay.Round8B
"""

import glob
import json
import os
import sys
from collections import defaultdict
from statistics import NormalDist

import numpy as np
import pandas as pd

from InPlay.Align import AsOf, EspnStates
from InPlay.CiAudit import Audit
from InPlay.Grid import TrialId
from InPlay.Load import LoadGames
from InPlay.Research import AppendMd, AppendTrials, CsvRows, Now, SaveRes, _Short
from InPlay.Stats import RatioBootstrap

ND = NormalDist()
DROOT = os.path.join("data", "inplay", "kalshi", "derivs", "candles")


def Ladders():
    km = json.load(open(os.path.join("data", "inplay", "kalshi", "markets.json"), encoding="utf-8"))
    ev2home = {r["event_ticker"]: r["ticker"] for r in km if r["is_home"]}
    out = defaultdict(list)
    for p in glob.glob(os.path.join(DROOT, "KXNBATOTAL-*.json")):
        d = json.load(open(p, encoding="utf-8"))
        m = d["meta"]
        ev = m["event"].replace("KXNBATOTAL", "KXNBAGAME")
        if ev not in ev2home or m.get("strike") is None:
            continue
        c = sorted(d["candles"], key=lambda x: x["t"])
        out[ev2home[ev]].append(dict(strike=float(m["strike"]), result=m.get("result"), t=[x["t"] for x in c], c=c))
    return out


def Quotes(lad, t):
    q = []
    for L in lad:
        c = AsOf(L["t"], L["c"], t)
        if c is None or t - c["t"] > 60 or c["bid"] is None or c["ask"] is None or c["ask"] - c["bid"] > 0.10:
            continue
        q.append((L, c))
    return q


def Median(q):
    pts = [(L["strike"], (c["bid"] + c["ask"]) / 2) for L, c in q if 0.03 <= (c["bid"] + c["ask"]) / 2 <= 0.97]
    if len(pts) < 2 or np.ptp([k for k, _ in pts]) < 1:
        return None
    ks = np.array([k for k, _ in pts])
    zs = np.array([ND.inv_cdf(p) for _, p in pts])
    b, a = np.polyfit(ks, zs, 1)
    if b >= 0:
        return None
    return -a / b


def Agg(vals):
    if not vals:
        return dict(n=0, mean=float("nan"), lo=float("nan"), hi=float("nan"))
    ks = sorted({k for k, _ in vals})
    ix = {k: i for i, k in enumerate(ks)}
    num, den = np.zeros(len(ks)), np.zeros(len(ks))
    for k, v in vals:
        num[ix[k]] += v
        den[ix[k]] += 1
    m, lo, hi = RatioBootstrap(num, den, reps=1000)
    return dict(n=len(vals), mean=m, lo=lo, hi=hi)


def Main(argv=None):
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    G = LoadGames(os.path.join("data", "inplay", "games_obj_s0.pkl"))
    meta = pd.read_pickle(os.path.join("data", "inplay", "games_s0.pkl")).dropna(subset=["kalshi_ticker"])
    tick = dict(zip(meta.game_id.astype(str), meta.kalshi_ticker))
    lads = Ladders()
    files = {os.path.basename(p)[:-5]: p for p in glob.glob(os.path.join("data", "inplay", "espn", "games", "2027", "*.json"))}
    obs = []  # per game per checkpoint
    for g in G["kalshi"]:
        gid = str(g.gid)
        if gid not in tick or tick[gid] not in lads or gid not in files:
            continue
        lad = lads[tick[gid]]
        game = json.load(open(files[gid], encoding="utf-8"))
        st = EspnStates(game)
        st_t = [s["t"] for s in st]
        final = float(game.get("home_score") or 0) + float(game.get("away_score") or 0)
        if not final:
            continue
        i0 = next((k for k in range(g.n) if g.el[k] >= 0), None)
        m0 = Median(Quotes(lad, g.t[i0])) if i0 is not None else None
        if m0 is None:
            continue
        for cp in (720.0, 1440.0, 2160.0):
            k = next((j for j in range(g.n) if g.el[j] >= cp), None)
            if k is None or k + 1 >= g.n:
                continue
            s = AsOf(st_t, st, g.t[k])
            q = Quotes(lad, g.t[k])
            mk = Median(q)
            if s is None or mk is None:
                continue
            pts, el = s["hs"] + s["as"], max(s["el"], 1.0)
            w = min(el / 2880.0, 1.0)
            rate = w * pts / el + (1 - w) * m0 / 2880.0
            pace = pts + rate * max(2880.0 - el, 0.0)
            qn = Quotes(lad, g.t[k + 1])
            obs.append(dict(gid=g.gid, day=g.day, cp=int(cp), final=final, market=mk, pace=pace, pre=m0, q_next=qn))
    days = sorted({o["day"] for o in obs})
    med = days[len(days) // 2] if days else ""
    md = ["## Round 8B: Kalshi live totals vs a simple pace projection (forward-window games; registered before scoring)", "",
          f"Games with a usable totals ladder: {len({o['gid'] for o in obs})} (f-early < {med} <= f-late).", "",
          "| checkpoint | games | MAE of final total: market median / pace | slope of (final - market) on (pace - market) [95% game CI] |", "|---|---|---|---|"]
    rows, trials, res = [], [], {}
    for cp in (720, 1440, 2160):
        O = [o for o in obs if o["cp"] == cp]
        if len(O) < 10:
            continue
        x = np.array([o["pace"] - o["market"] for o in O])
        y = np.array([o["final"] - o["market"] for o in O])
        slope = float(np.polyfit(x, y, 1)[0])
        rng = np.random.default_rng(3)
        bs = []
        for _ in range(500):
            ix = rng.integers(0, len(O), len(O))
            if np.ptp(x[ix]) > 0:
                bs.append(np.polyfit(x[ix], y[ix], 1)[0])
        lo, hi = np.quantile(bs, [0.025, 0.975])
        mae_m, mae_p = float(np.mean(np.abs(y))), float(np.mean(np.abs([o["final"] - o["pace"] for o in O])))
        res[f"cp{cp}"] = dict(n=len(O), mae_market=mae_m, mae_pace=mae_p, slope=slope, lo=float(lo), hi=float(hi))
        rows.append(dict(section="round8B_totals_pace", venue="kalshi", split="forward", family="totals_pace", fee_basis="none", metric=f"slope_cp{cp}",
                         value=slope, lo=float(lo), hi=float(hi), n=len(O)))
        md.append(f"| end of Q{cp // 720} | {len(O)} | {mae_m:.1f} / {mae_p:.1f} | {slope:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    md += ["", "(slope 0 = the market already prices the pace information; positive = market under-reacts to pace.)", "",
           "| rule | split | trades | P/L per trade, Kalshi taker [conservative CI] | $0.02 fee |", "|---|---|---|---|---|"]

    def Trades(P, early):
        out = []
        for o in obs:
            if o["cp"] != P["cp"] or (o["day"] < med) != early:
                continue
            d = o["pace"] - o["market"]
            if abs(d) < P["gap"] or not o["q_next"]:
                continue
            L, c = min(o["q_next"], key=lambda lc: abs(lc[0]["strike"] - o["market"]))
            if L["result"] not in ("yes", "no"):
                continue
            yes = 1.0 if L["result"] == "yes" else 0.0
            side = 1 if d > 0 else -1
            entry = (c["ask"] if side == 1 else 1 - c["bid"]) + 0.005
            if entry >= 0.99:
                continue
            out.append({"gid": o["gid"], "day": o["day"], "season": 2027, "side": side, "entry": entry, "exit": yes if side == 1 else 1 - yes,
                        "settled": True, "why": "settle", "el": P["cp"], "maker": False})
        return out

    tune = []
    for gap in (3, 6, 10):
        for cp in (720, 1440, 2160):
            P = dict(gap=gap, cp=cp)
            m = Audit(Trades(P, True), "kalshi_taker")
            tune.append((P, m))
            trials.append(dict(trial_id=TrialId("totals_pace", P, "kalshi", "taker"), created_at=Now(), family="totals_pace", params=P, venue="kalshi",
                               exec="taker", tag="round8B_registered_before_scoring", split_scored="f-early", tune=m))
            if m:
                md.append(f"| {_Short(P)} | f-early (tune) | {m['n']} | {m['mean']:+.4f} [{m['cons_lo']:+.4f}, {m['cons_hi']:+.4f}] | |")
    ok = [x for x in tune if x[1] and x[1]["n"] >= 15] or [x for x in tune if x[1]]
    if ok:
        P, _ = max(ok, key=lambda x: x[1]["mean"])
        tr = Trades(P, False)
        mk, mt = Audit(tr, "kalshi_taker"), Audit(tr, "tim_est_0.02")
        res["trade"] = dict(params=P, kalshi_taker=mk, tim=mt)
        trials.append(dict(trial_id=TrialId("totals_pace", P, "kalshi", "taker"), created_at=Now(), family="totals_pace", params=P, venue="kalshi",
                           exec="taker", tag="round8B_selected_on_f-early", split_scored="f-late", res=res["trade"]))
        if mk:
            md.append(f"| SELECTED {_Short(P)} | f-late | {mk['n']} | {mk['mean']:+.4f} [{mk['cons_lo']:+.4f}, {mk['cons_hi']:+.4f}] | {mt['mean']:+.4f} |")
            for b, m in (("kalshi_taker", mk), ("tim_est_0.02", mt)):
                rows.append(dict(section="round8B_totals_pace", venue="kalshi", split="forward_late", family="totals_pace", trial_id=TrialId("totals_pace", P, "kalshi", "taker"),
                                 fee_basis=b, metric="pnl_per_trade", value=m["mean"], lo=m["cons_lo"], hi=m["cons_hi"], n=m["n"], label=_Short(P)))
    md.append("")
    AppendTrials(trials)
    CsvRows(rows)
    AppendMd("\n".join(md))
    SaveRes("round8B", res)
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())

