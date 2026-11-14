"""Live game view: python -m Tools.LiveView --date YYYY-MM-DD [--game DEN] [--watch 60] [--out PATH]

Score, clock, ESPN win probability (source: ESPN summary winprobability), the Kalshi moneyline (no-vig from
yes asks) and the swing since the last run. State is kept in data\\live\\liveview_state_<date>.json.
Information only: never an edge, never an order.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from Tools import Data
from Tools.Data import FmtEt, Money, NowEt

STATE_DIR = Data.ROOT / "data" / "live"


def LoadState(day):
    p = STATE_DIR / f"liveview_state_{day}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def SaveState(day, state):
    Data.WriteAtomic(STATE_DIR / f"liveview_state_{day}.json", json.dumps(state, indent=1))


def Swing(now_val, prev_val):
    if now_val is None or prev_val is None:
        return None
    return round((now_val - prev_val) * 100, 1)


def Snapshot(day, game_filter=None, state=None):
    state = state if state is not None else {}
    rows = []
    for g in Data.EspnScoreboard(day):
        if game_filter:
            f = Data.Std(game_filter)
            if f not in (g["home"], g["away"]) and str(game_filter) != str(g["espn_id"]):
                continue
        summ = Data.EspnSummary(g["espn_id"], refresh=True) if g["state"] in ("in", "post") else Data.EspnSummary(g["espn_id"])
        wp = Data.EspnWinProb(summ)
        pre = Data.EspnPredictor(summ) if not wp else None
        kg = Data.KalshiGame(day, g["away"], g["home"])
        k_home = None if not kg else kg.get("home_fair_ask")
        if kg and k_home is None:
            k_home = kg.get("home_fair_mid")
        cur = {"t": NowEt().isoformat(timespec="seconds"), "home_score": g["home_info"].get("score"), "away_score": g["away_info"].get("score"),
               "espn_wp_home": None if not wp else wp["home_win_prob"], "kalshi_fair_home": k_home,
               "kalshi_ask_home": None if not kg or not kg.get("home") else kg["home"].get("yes_ask"),
               "kalshi_ask_away": None if not kg or not kg.get("away") else kg["away"].get("yes_ask"),
               "period": g.get("period"), "clock": g.get("clock"), "state": g.get("state")}
        prev = state.get(str(g["espn_id"]))
        rows.append({"game": g, "cur": cur, "prev": prev, "wp": wp, "pre": pre, "kalshi": kg,
                     "swing_espn_c": Swing(cur["espn_wp_home"], (prev or {}).get("espn_wp_home")),
                     "swing_kalshi_c": Swing(cur["kalshi_fair_home"], (prev or {}).get("kalshi_fair_home")),
                     "gap_espn_minus_kalshi_c": Swing(cur["espn_wp_home"], cur["kalshi_fair_home"])})
        state[str(g["espn_id"])] = cur
    return rows, state


def Render(day, rows):
    L = [f"# Live view - {day} (snapshot {FmtEt(NowEt())})", "",
         "Win probabilities are information, not an edge. ESPN WP = ESPN summary `winprobability` (ESPN's model). "
         "Kalshi = public API KXNBAGAME yes asks, no-vig across both teams. Swings are vs the previous run of this tool.", "",
         "| game | state | score | clock | ESPN WP (home) | Kalshi fair (home) | Kalshi asks home/away | ESPN - Kalshi | swing ESPN | swing Kalshi | prev run |",
         "|---|---|---|---|---:|---:|---|---:|---:|---:|---|"]
    for r in rows:
        g, c, prev = r["game"], r["cur"], r["prev"] or {}
        state = {"pre": "pregame", "in": "LIVE", "post": "final"}.get(g["state"], g["state"])
        score = f"{g['away']} {int(c['away_score'] or 0)} - {g['home']} {int(c['home_score'] or 0)}" if g["state"] != "pre" else f"{g['away']} @ {g['home']}"
        clock = g.get("detail") if g["state"] != "in" else f"Q{g.get('period')} {g.get('clock')}"
        wp = "n/a" if c["espn_wp_home"] is None else f"{100*c['espn_wp_home']:.0f}%"
        if c["espn_wp_home"] is None and r.get("pre"):
            wp = f"pregame BPI {100*r['pre']['home_win_prob']:.0f}%"
        kf = "n/a" if c["kalshi_fair_home"] is None else f"{100*c['kalshi_fair_home']:.0f}%"
        sw = lambda x: "n/a" if x is None else f"{x:+.0f}c"
        L.append(f"| {g['away']} @ {g['home']} | {state} | {score} | {clock} | {wp} | {kf} | {Money(c['kalshi_ask_home'])}/{Money(c['kalshi_ask_away'])} | "
                 f"{sw(r['gap_espn_minus_kalshi_c'])} | {sw(r['swing_espn_c'])} | {sw(r['swing_kalshi_c'])} | {FmtEt(prev.get('t'), False) if prev else 'first run'} |")
    if not rows:
        L.append("| no games found | | | | | | | | | | |")
    L += ["", "A gap between ESPN's WP and Kalshi is two models disagreeing, not a mispricing; Kalshi quotes can also lag the game by seconds."]
    if Data.errors:
        L += ["", "Data gaps: " + "; ".join(Data.errors[-5:])]
    return "\n".join(L) + "\n"


def Main(argv=None):
    ap = argparse.ArgumentParser(description="Live NBA game view")
    ap.add_argument("--date", default=NowEt().strftime("%Y-%m-%d"))
    ap.add_argument("--game", default=None, help="team code (e.g. DEN) or ESPN event id")
    ap.add_argument("--watch", type=int, default=0, help="refresh every N seconds (min 30); Ctrl-C to stop")
    ap.add_argument("--max-updates", type=int, default=0, help="stop after this many refreshes (0 = until Ctrl-C / all final)")
    ap.add_argument("--out", default=None, help="also write the latest snapshot Markdown here")
    a = ap.parse_args(argv)
    n = 0
    while True:
        state = LoadState(a.date)
        rows, state = Snapshot(a.date, a.game, state)
        SaveState(a.date, state)
        text = Render(a.date, rows)
        print(text, flush=True)
        if a.out:
            Data.WriteAtomic(a.out, text)
        n += 1
        if a.watch <= 0 or (a.max_updates and n >= a.max_updates) or (rows and all(r["game"]["state"] == "post" for r in rows)):
            break
        try:
            time.sleep(max(30, a.watch))
        except KeyboardInterrupt:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())