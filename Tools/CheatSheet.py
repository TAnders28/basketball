"""Game-day cheat sheet: python -m Tools.CheatSheet --date YYYY-MM-DD [--out PATH]

One skimmable Markdown file for the slate (about 30 seconds per game). Keeps "who is more likely to win"
separate from "price vs fair". Never claims an edge. Shares its fetchers with Tools.Board (Tools.Data).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from Tools import Data
from Tools.Data import FmtEt, Money, NowEt, FmtAmerican

STATUS_ORDER = {"Out": 0, "Doubtful": 1, "Questionable": 2, "Probable": 3, "Available": 4}


def Pct(p):
    return "n/a" if p is None else f"{100 * p:.0f}%"


def F(x, nd=1, pct=False):
    if x is None:
        return "n/a"
    return f"{100 * x:.{nd}f}%" if pct else f"{x:.{nd}f}"


def Name(n):
    n = str(n or "")
    if "," in n:
        last, first = n.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return n


def GameSection(g, model_prob=None):
    e, home, away, k = g["espn"], g["home"], g["away"], g["kalshi"] or {}
    L = []
    state = {"pre": "scheduled", "in": "LIVE", "post": "final"}.get(e.get("state"), e.get("state") or "?")
    score = ""
    if e.get("state") in ("in", "post"):
        score = f" | {away} {int(e['away_info'].get('score') or 0)} - {home} {int(e['home_info'].get('score') or 0)} ({e.get('detail')})"
    L.append(f"## {away} @ {home} - {FmtEt(e.get('start'), with_date=False)} ({state}){score}")
    L.append("")
    # --- likely to win
    fav = None
    if k.get("home_fair_ask") is not None:
        fav = home if k["home_fair_ask"] >= 0.5 else away
    b = g.get("book")
    pr = g.get("predictor")
    bits = []
    if k.get("home_fair_ask") is not None:
        bits.append(f"Kalshi no-vig {home} {Pct(k['home_fair_ask'])} / {away} {Pct(k['away_fair_ask'])}")
    if b and b.get("home_fair") is not None:
        bits.append(f"{b['provider']} no-vig {home} {Pct(b['home_fair'])} / {away} {Pct(b['away_fair'])}")
    if pr:
        bits.append(f"ESPN BPI {home} {Pct(pr['home_win_prob'])}")
    if model_prob is not None:
        bits.append(f"our stack_global (untested, lost to market, no injury input) {home} {Pct(model_prob)}")
        if k.get("home_fair_ask") is not None and abs(model_prob - k["home_fair_ask"]) >= 0.10:
            bits.append("big model-vs-market gap: most likely the model is missing injury/rotation news")
    L.append(f"**Most likely to win:** {fav or 'n/a'}. " + ("; ".join(bits) if bits else "no market or model probability available") + ".")
    L.append("")
    # --- price vs fair
    L.append("**Price vs fair** ($ to win $1; fair = no-vig. Paying above fair is the vig, not a signal):")
    L.append("")
    L.append("| side | Kalshi yes ask | bid | fair (asks) | fair (mids) | ask - fair | book ML | book no-vig |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for side, row_key, fk, fm, bml, bf in ((away, "away", "away_fair_ask", "away_fair_mid", "away_ml", "away_fair"),
                                           (home, "home", "home_fair_ask", "home_fair_mid", "home_ml", "home_fair")):
        r = k.get(row_key) or {}
        gap = None if (r.get("yes_ask") is None or k.get(fk) is None) else (r["yes_ask"] - k[fk]) * 100
        ml = None if not b else b.get(bml)
        L.append(f"| {side} | {Money(r.get('yes_ask'))} | {Money(r.get('yes_bid'))} | {Money(k.get(fk))} | {Money(k.get(fm))} | "
                 f"{'n/a' if gap is None else f'{gap:+.1f}c'} | {'n/a' if ml is None else f'{int(ml):+d}'} | {Money(None if not b else b.get(bf))} |")
    src = []
    if k:
        src.append(f"Kalshi {k.get('event_ticker')} as of {FmtEt(k.get('as_of'), False)}")
    if b:
        src.append(f"{b['provider']} via ESPN pickcenter as of {FmtEt(b.get('as_of'), False)} (spread {b.get('spread')}, total {b.get('total')}, hold {F(b.get('hold'), 1, True)})")
    if src:
        L.append("")
        L.append("Sources: " + "; ".join(src) + ".")
    if not k:
        L.append("")
        L.append("Kalshi moneyline not found for this game.")
    # --- line move
    mv = g.get("move_home")
    L.append("")
    if mv and mv.get("first_price") is not None:
        L.append(f"**Line move:** Kalshi {home} yes {Money(mv['first_price'])} at first hourly candle ({FmtEt(mv['first_t_et'])}, {mv['hours_back']} h ago) "
                 f"-> {Money(mv.get('now_yes_ask'))} now ({mv['move_c']:+.0f}c). Market opened {FmtEt(mv.get('market_open_et'))}; 'open' here is the first candle with a price.")
    else:
        L.append("**Line move:** no Kalshi candle history returned.")
    # --- injuries
    L.append("")
    inj = g.get("injuries_official") or []
    if inj:
        L.append(f"**Injuries** (NBA official report, slot {FmtEt(inj[0].get('report_slot_et'))}):")
    else:
        inj = [x for x in g.get("injuries_espn") or [] if x["team"] in (home, away)]
        L.append("**Injuries** (ESPN summary; official report had no rows for this game):" if inj else "**Injuries:** none listed.")
    for team in (away, home):
        rows = sorted([x for x in inj if x.get("team") == team], key=lambda x: STATUS_ORDER.get(x.get("status"), 9))
        real = [x for x in rows if not str(x.get("note") or "").startswith("GLeague")]
        if not rows:
            continue
        txt = ", ".join(f"{Name(x['player'])} {x['status']}" + (f" ({x['note'].replace('Injury/Illness-', '')})" if x.get("note") else "")
                        + (f" [ESPN {FmtEt(x.get('updated_et'))}]" if x.get("updated_et") else "") for x in real) or "only G League / two-way listings"
        L.append(f"- {team}: {txt}")
    L.append("- Lineups: confirmed starters aren't in free pregame feeds; check at tip.")
    # --- form
    L.append("")
    espn_rec = {home: e["home_info"].get("record"), away: e["away_info"].get("record")}
    L.append("**Form and rest** (local games + ESPN scores; if the local count differs from ESPN's record, a game is missing locally):")
    L.append("")
    L.append("| team | ESPN record | local record (n) | last 10 | L10 margin | season margin | rest days | B2B | streak | home / road | last game |")
    L.append("|---|---|---|---|---:|---:|---:|---|---|---|---|")
    for team in (away, home):
        f = g["form"].get(team)
        if not f:
            L.append(f"| {team} | {espn_rec.get(team) or 'n/a'} | n/a | | | | | | | | |")
            continue
        L.append(f"| {team} | {espn_rec.get(team) or 'n/a'} | {f['record']} ({f['n_games']}) | {f['last10']} | {f['last10_margin']:+.1f} | {f['season_margin']:+.1f} | {f['rest_days']} | "
                 f"{'yes' if f['back_to_back'] else 'no'} | {f['streak']} | {f['home_record']} / {f['road_record']} | {f['last_game']} |")
    # --- matchup stats
    L.append("")
    st_a, st_h = g["stats"].get(away), g["stats"].get(home)
    if st_a and st_h:
        L.append(f"**Matchup stats** (season to date, local box scores through {st_h['through']}):")
        L.append("")
        L.append(f"| stat | {away} | {home} |")
        L.append("|---|---:|---:|")
        for lab, key, nd, pct in (("Off rating", "ortg", 1, False), ("Def rating", "drtg", 1, False), ("Net rating", "net", 1, False),
                                  ("Pace (poss/g)", "pace", 1, False), ("eFG%", "efg", 1, True), ("Opp eFG%", "opp_efg", 1, True),
                                  ("TOV%", "tov_pct", 1, True), ("Opp TOV%", "opp_tov_pct", 1, True), ("OREB%", "orb_pct", 1, True),
                                  ("FT rate", "ftr", 2, False), ("3PA rate", "three_rate", 1, True), ("3P%", "three_pct", 1, True),
                                  ("Opp 3PA rate", "opp_three_rate", 1, True)):
            L.append(f"| {lab} | {F(st_a.get(key), nd, pct)} | {F(st_h.get(key), nd, pct)} |")
    else:
        L.append("**Matchup stats:** not available from local data.")
    # --- h2h + trends
    L.append("")
    h2h = g.get("h2h") or []
    L.append("**Head to head (recent):** " + ("; ".join(h2h) if h2h else ("; ".join(g.get("season_series") or []) or "none found")))
    L.append("")
    L.append("**Context, not an edge** (small samples; none of these were tested as betting rules):")
    for t in g.get("trends") or ["no trend lines available"]:
        L.append(f"- {t}")
    if pr:
        L.append(f"- ESPN Matchup Predictor has {home} at {Pct(pr['home_win_prob'])} (ESPN's model, untested by us).")
    if g.get("props"):
        L.append(f"- Kalshi lists {len(g['props'])} player/stat prop main lines for this game (see board_latest.json). No prop model is validated.")
    L.append("")
    return "\n".join(L)


def Render(bundle, model=None):
    model = model or {}
    now = NowEt()
    ir = bundle.get("injury_report")
    head = [f"# NBA cheat sheet - {bundle['date']}",
            "",
            f"Generated {FmtEt(bundle['generated_at'])}. For Tim's own judgment. **Nothing here is an edge.** "
            "Our rules/models did not beat the market in testing (2026-11-13/14 studies); market prices are the best probability estimates we have.",
            "",
            "Sources: Kalshi public API (KXNBAGAME moneylines, hourly candlesticks); ESPN public site API (scoreboard, summary: pickcenter odds, injuries, predictor); "
            + (f"NBA official injury report PDF ({ir['url'].rsplit('/', 1)[-1]}, slot {FmtEt(ir['report_slot_et'])}); " if ir else "NBA official injury report: none found for today yet; ")
            + f"local games data through {bundle.get('games_info', {}).get('local_through')} plus ESPN scores for later dates.",
            "",
            "How to read prices: $0.62 = pay 62c to win $1. 'Fair' = no-vig price (both sides scaled to sum to $1). Fee on Kalshi per Tim's working estimate: $0.02 per contract.",
            ""]
    body = [GameSection(g, model.get((g["home"], g["away"]))) for g in sorted(bundle["games"], key=lambda g: g["espn"].get("start") or "")]
    tail = []
    if bundle.get("errors"):
        tail = ["## Data gaps", ""] + [f"- {e}" for e in bundle["errors"]]
    if not bundle["games"]:
        body = ["No games found on ESPN's scoreboard for this date."]
    return "\n".join(head + body + tail) + "\n"


def Main(argv=None):
    ap = argparse.ArgumentParser(description="Write the game-day cheat sheet")
    ap.add_argument("--date", default=NowEt().strftime("%Y-%m-%d"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(rf"E:\AgentDocs\Basketbot\{a.date}_cheatsheet\{a.date}_cheatsheet.md")
    bundle = Data.SlateBundle(a.date, with_props=True)
    try:
        from Tools.Board import LoadModel
        model, _ = LoadModel(a.date)
    except Exception:
        model = {}
    Data.WriteAtomic(out, Render(bundle, model))
    print(f"wrote {out} games={len(bundle['games'])} errors={len(bundle['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())