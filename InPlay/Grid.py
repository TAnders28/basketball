"""The registered strategy grid. Every variant here is a trial in the registry, scored or not."""

import hashlib
import json

EXITS_ABS = [dict(tgt=0.55), dict(tgt=0.60), dict(tgt=0.70), dict(tgt=0.55, stop=0.30), dict(tgt=0.60, stop=0.30),
             dict(tgt=0.70, stop=0.30), dict(tgt=0.60, stop=0.35), dict(tgt=0.60, t_exit=2160), dict()]


def TrialId(family, params, venue, exec_mode):
    key = json.dumps({"f": family, "p": params, "v": venue, "x": exec_mode}, sort_keys=True)
    return family + "-" + hashlib.sha1(key.encode()).hexdigest()[:10]


def Variants():
    v = []
    for pre in [(0.55, 0.65), (0.50, 0.60), (0.60, 0.75)]:
        for entry in (0.40, 0.45):
            for tmax in (1440, 2160):
                for ex in EXITS_ABS:
                    v.append(("dip_fav", dict(pre_lo=pre[0], pre_hi=pre[1], entry=entry, tmax=tmax, **ex)))
    for entry in (0.30, 0.40, 0.45):
        for tmax in (1440, 2160):
            for tgt_d in (0.10, 0.15, 0.20):
                for stop_d in (None, 0.10):
                    for mt in (1, 3):
                        v.append(("dip_any", dict(entry=entry, tmax=tmax, tgt_d=tgt_d, stop_d=stop_d, max_tr=mt)))
    for fam in ("fade_run", "momentum"):
        for look in (3, 5, 10):
            for drop in (0.10, 0.15, 0.20):
                for hold in (5, 10, 20, None):
                    v.append((fam, dict(look=look, drop=drop, tmax=2520, hold=hold, max_tr=3)))
    for fam in ("late_fav", "late_dog"):
        for mins in (3, 6, 12):
            for lo, hi in ((0.70, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 0.98)):
                v.append((fam, dict(mins=mins, lo=lo, hi=hi)))
    for k in (0.10, 0.15, 0.20):
        for tgt_d in (0.10, None):
            for stop_d in (None, 0.10):
                v.append(("half_reset", dict(k=k, tgt_d=tgt_d, stop_d=stop_d)))
    for k in (0.10, 0.15, 0.20, 0.25):
        for tmax in (1440, 2160):
            for tgt_d in (0.10, 0.15):
                for stop_d in (None, 0.15):
                    v.append(("anchor", dict(k=k, tmax=tmax, tgt_d=tgt_d, stop_d=stop_d)))
    for gap in (0.05, 0.08, 0.12):
        for d in (1, -1):
            for hold in (2, 5, 10):
                v.append(("espn_lag", dict(gap=gap, dir=d, hold=hold, close_gap=True, tmax=2820, max_tr=3)))
    for gap in (0.02, 0.03, 0.05):
        for hold in (2, 5, 10):
            v.append(("xvenue", dict(gap=gap, hold=hold, close_gap=True, tmax=2820, max_tr=3)))
    # Tim's comeback hypothesis (evenly matched games) and its mismatched-game control.
    leads = [dict(mode="price", lead_lo=0.75, lead_hi=0.90), dict(mode="price", lead_lo=0.80, lead_hi=0.90),
             dict(mode="price", lead_lo=0.75, lead_hi=0.85), dict(mode="pts", pts_lo=10, pts_hi=18),
             dict(mode="pts", pts_lo=12, pts_hi=18), dict(mode="pts", pts_lo=10, pts_hi=14)]
    cb_exits = [dict(), dict(tgt=0.40), dict(tgt=0.50), dict(tgt=0.40, stop=0.08), dict(tgt=0.50, stop=0.10)]
    for lead in leads:
        for tmax in (2520, 2160):
            for ex in cb_exits:
                v.append(("comeback", dict(close=0.05, tmax=tmax, **lead, **ex)))
                for who in ("fav", "dog"):
                    v.append(("comeback_ctrl", dict(mis_lo=0.10, mis_hi=0.30, leader_is=who, tmax=tmax, **lead, **ex)))
    # volatility regime on the headline dip rule: mean |1-min mid change| over the last 10 minutes
    # (pregame book totals are missing for the whole tuning split, so pace uses in-game realized volatility).
    for rmin, rmax in ((0.02, None), (None, 0.012)):
        for ex in (dict(tgt=0.60), dict(tgt=0.60, stop=0.30), dict()):
            v.append(("dip_fav_pace", dict(pre_lo=0.55, pre_hi=0.65, entry=0.40, tmax=1440, rv_min=rmin, rv_max=rmax, **ex)))
            v.append(("dip_fav_pace", dict(pre_lo=0.55, pre_hi=0.65, entry=0.45, tmax=2160, rv_min=rmin, rv_max=rmax, **ex)))
    for run_min in (6, 8, 10):
        for hold in (3, 6, 10):
            for fam in ("fade_score_run", "timeout_fade"):
                for tgt_d in (None, 0.10):
                    p = dict(run_min=run_min, hold=hold, tgt_d=tgt_d, tmax=2520, max_tr=3)
                    if fam == "timeout_fade":
                        p["to_window"] = 90
                    v.append((fam, p))
    for look in (1, 2, 3):
        for jump in (0.08, 0.15):
            for laggap in (0.05, 0.10):
                for hold in (2, 5):
                    v.append(("espn_jump", dict(look=look, jump=jump, laggap=laggap, hold=hold, tmax=2820, max_tr=3)))
    for persist in (2, 3):
        for gap in (0.02, 0.03, 0.05):
            for hold in (2, 5, 10):
                v.append(("xvenue", dict(gap=gap, hold=hold, close_gap=True, tmax=2820, max_tr=3, persist=persist)))
    return v

# Pre-registered representative rules (fixed before any holdout scoring).
REPRESENTATIVE = [
    ("dip_fav", dict(pre_lo=0.55, pre_hi=0.65, entry=0.40, tmax=1440, tgt=0.60, stop=0.30)),
    ("dip_fav", dict(pre_lo=0.55, pre_hi=0.65, entry=0.40, tmax=1440)),
    ("dip_any", dict(entry=0.40, tmax=1440, tgt_d=0.20, stop_d=0.10, max_tr=3)),
    ("fade_run", dict(look=5, drop=0.15, tmax=2520, hold=10, max_tr=3)),
    ("momentum", dict(look=5, drop=0.15, tmax=2520, hold=10, max_tr=3)),
    ("late_fav", dict(mins=6, lo=0.90, hi=0.95)),
    ("late_dog", dict(mins=6, lo=0.90, hi=0.95)),
    ("half_reset", dict(k=0.15, tgt_d=0.10, stop_d=None)),
    ("anchor", dict(k=0.15, tmax=1440, tgt_d=0.10, stop_d=None)),
    ("espn_lag", dict(gap=0.08, dir=1, hold=5, close_gap=True, tmax=2820, max_tr=3)),
    ("xvenue", dict(gap=0.03, hold=5, close_gap=True, tmax=2820, max_tr=3)),
    ("comeback", dict(close=0.05, tmax=2520, mode="price", lead_lo=0.75, lead_hi=0.90)),
    ("comeback", dict(close=0.05, tmax=2520, mode="pts", pts_lo=10, pts_hi=18, tgt=0.40, stop=0.08)),
    ("comeback_ctrl", dict(mis_lo=0.10, mis_hi=0.30, leader_is="fav", tmax=2520, mode="price", lead_lo=0.75, lead_hi=0.90)),
    ("comeback_ctrl", dict(mis_lo=0.10, mis_hi=0.30, leader_is="dog", tmax=2520, mode="price", lead_lo=0.75, lead_hi=0.90)),
]
MAKER_FAMILIES = ("dip_fav", "dip_any", "comeback", "anchor", "half_reset")
CONTROL_FAMILIES = ("comeback_ctrl",)


def EngineFamily(fam):
    return "dip_fav" if fam == "dip_fav_pace" else fam