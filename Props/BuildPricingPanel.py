"""Build the leak-free Kalshi player-prop pricing panel (stage 1 of the 2026-11-14 prop-pricing study).

Tip time is the scheduled tip from games_meta (ExportOpsVision.LoadSchedIndex), NOT the Kalshi
expected_expiration_time / occurrence_datetime, which often fall after the real tip (leak).
Model: frozen NB/EB count models fit on 2024-25 only (Props.Model), features from earlier dates only.
Markets whose player has no box-score row that day (DNP or name mismatch) use the player's most recent
earlier pregame row (carry_forward), so DNP losses stay in the sample.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import bisect
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

from Props.ExportPropsBacktest import (
    FAMILY_OF, LoadSeasons, BuildIndex, MatchRow, NormName, StrikeBucket, LowerToNormal,
)
from Props.Model import (
    TRAIN_SEASON, AttachCountPredictions, AttachPredictions, BuildFeatures, FairGe,
    FitCountParams, FitParams,
)
from Props.PullProps import ROOT, TARGET_SERIES, IsoToTs
from InPlay.Http import ReadJson

EVAL_START = "2025-04-14"
PANEL_PATH = ROOT / "pricing_panel_2026-11-14.pkl"
OFFSETS_MIN = (240, 120, 60, 30, 15, 2)   # quote times: minutes before scheduled tip
STALE_S = 3600
ALIAS = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "PHO": "PHX", "WSH": "WAS", "UTAH": "UTA", "BRK": "BKN"}


def Log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def Team(t):
    t = str(t or "").upper()
    return ALIAS.get(t, t)


def SchedLookup():
    from Backtest.ExportOpsVision import LoadSchedIndex
    s = LoadSchedIndex()
    exact, by_team = {}, {}
    for (d, h, a), v in s.items():
        if v.get("sched_ts") is None:
            continue
        exact[(d, Team(h), Team(a))] = v["sched_ts"]
        by_team[(d, Team(h))] = v["sched_ts"]
        by_team[(d, Team(a))] = v["sched_ts"]
    return exact, by_team


def QuoteAt(candles, target, stale_s=STALE_S):
    best = None
    for c in candles:
        t = c.get("t")
        if t is None or t > target:
            continue
        if best is None or t > best["t"]:
            best = c
    if best is None or target - best["t"] > stale_s:
        return None
    return best


def Pack(c, target):
    if c is None:
        return None
    ask, bid = c.get("ask"), c.get("bid")
    return {
        "t": int(c["t"]), "age_s": int(target - c["t"]) if target is not None else None,
        "ask": None if ask is None else float(ask), "bid": None if bid is None else float(bid),
        "last": c.get("last"), "vol": c.get("vol"), "oi": c.get("oi"),
    }


def Main():
    LowerToNormal()
    t0 = time.time()
    players, games = LoadSeasons()
    feats, _, _ = BuildFeatures(players, games)
    train = [r for r in feats if r["season"] == TRAIN_SEASON and r.get("date") and r["date"] < EVAL_START]
    Log(f"features all={len(feats)} train={len(train)}")
    params_by = {"3pm": FitParams(train)}
    AttachPredictions(feats, params_by["3pm"])
    for fam in ("pts", "reb", "ast", "pra"):
        params_by[fam] = FitCountParams(train, fam)
        AttachCountPredictions(feats, params_by[fam], fam)
    Log("fit done")
    by_name, by_last = BuildIndex(feats)
    hist = defaultdict(list)
    for r in feats:
        if r.get("date"):
            hist[NormName(r["player_name"])].append((r["date"], r))
    for k in hist:
        hist[k].sort(key=lambda x: x[0])
    exact, by_team = SchedLookup()

    out, skipped = [], defaultdict(int)
    cdir = ROOT / "candles"
    for series in TARGET_SERIES:
        markets = ReadJson(str(ROOT / f"markets_{series}.json"))
        fam = FAMILY_OF[series]
        for m in markets:
            res = str(m.get("result") or "").lower()
            if res not in ("yes", "no"):
                skipped["no_result"] += 1
                continue
            d = m.get("date")
            if not d or d < EVAL_START or not m.get("k"):
                skipped["no_date_k"] += 1
                continue
            sched = exact.get((d, Team(m.get("home")), Team(m.get("away"))))
            if sched is None:
                sched = by_team.get((d, Team(m.get("home")))) or by_team.get((d, Team(m.get("away"))))
            if sched is None:
                skipped["no_sched_tip"] += 1
                continue
            cpath = cdir / (m["ticker"] + ".json")
            if not cpath.exists():
                skipped["no_candle_file"] += 1
                continue
            candles = (ReadJson(str(cpath)).get("candles") or [])
            if not candles:
                skipped["empty_candles"] += 1
                continue
            prow, how = MatchRow(m, by_name, by_last)
            if prow is None:
                lst = hist.get(NormName(m.get("player_name"))) or []
                i = bisect.bisect_left([x[0] for x in lst], d) - 1
                if i >= 0 and lst[i][0] >= str(int(d[:4]) - 1) + d[4:]:
                    prow, how = lst[i][1], "carry_forward:" + how
            if prow is None:
                skipped["no_model_row"] += 1
                continue
            p = FairGe(prow, fam, int(m["k"]), params_by[fam])
            if p is None:
                skipped["no_fair"] += 1
                continue
            pre = [c for c in candles if c.get("t") is not None and c["t"] <= sched]
            first = min(pre, key=lambda c: c["t"]) if pre else None
            quotes = {"first": Pack(first, None)}
            for off in OFFSETS_MIN:
                tgt = sched - off * 60
                quotes[f"T{off}"] = Pack(QuoteAt(candles, tgt), tgt)
            exp_ts = IsoToTs(m.get("expected_expiration_time"))
            occ_ts = IsoToTs(m.get("occurrence_datetime"))
            out.append({
                "date": d, "series": series, "family": fam, "protected": bool(m.get("protected")),
                "ticker": m["ticker"], "event_ticker": m.get("event_ticker"),
                "player_name": m.get("player_name"), "k": int(m["k"]), "strike_bucket": StrikeBucket(m["k"]),
                "home": m.get("home"), "away": m.get("away"), "sched_ts": int(sched),
                "exp_minus_sched_min": None if exp_ts is None else (exp_ts - sched) / 60.0,
                "occ_minus_sched_min": None if occ_ts is None else (occ_ts - sched) / 60.0,
                "model_p": float(p), "match": how, "proj_min": prow.get("proj_min"),
                "prior_n": prow.get("prior_n"), "yes": 1 if res == "yes" else 0,
                "volume": float(m.get("volume") or 0), "quotes": quotes,
            })
        Log(f"{series} done rows={len(out)} skipped={dict(skipped)}")
    with open(PANEL_PATH, "wb") as f:
        pickle.dump({"rows": out, "skipped": dict(skipped), "params": params_by,
                     "built_at": time.strftime("%Y-%m-%d %H:%M:%S ET")}, f)
    Log(f"DONE rows={len(out)} elapsed={time.time()-t0:.0f}s -> {PANEL_PATH}")


if __name__ == "__main__":
    Main()
