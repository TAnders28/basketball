"""Slip checker: compare what you paid with the market's own fair price. Never an edge claim.

    python -m Tools.SlipCheck --leg KXNBAGAME-26NOV14DENNOP-DEN --price 0.84 --contracts 10
    python -m Tools.SlipCheck --leg DEN --leg "Jamal Murray 25+ points" --price 0.40 --contracts 5
    python -m Tools.SlipCheck --file slip.json
      slip.json: {"date": "2026-11-14", "price": 0.40, "contracts": 5,
                  "legs": [{"ticker": "KXNBAGAME-26NOV14DENNOP-DEN"}, {"desc": "Jamal Murray 25+ points", "side": "yes"}]}

A leg can be a Kalshi market ticker (optionally TICKER:no), a team code/name for that date's moneyline,
or "Player Name K+ points|rebounds|assists|threes|pra".
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from Tools import Data
from Tools.Data import FEE, FEE_LABEL, Money, NowEt, FmtAmerican, FmtEt

PROP_RE = re.compile(r"^(?P<name>.+?)\s+(?P<k>\d+)\+?\s*(?P<stat>points?|pts|rebounds?|reb|assists?|ast|threes?|3pm|3pt|pra|pts\+reb\+ast)$", re.I)
STAT_SERIES = {"point": "KXNBAPTS", "pts": "KXNBAPTS", "rebound": "KXNBAREB", "reb": "KXNBAREB", "assist": "KXNBAAST", "ast": "KXNBAAST",
               "three": "KXNBA3PT", "3pm": "KXNBA3PT", "3pt": "KXNBA3PT", "pra": "KXNBAPRA", "pts+reb+ast": "KXNBAPRA"}


def GameKey(ticker):
    """KXNBAPTS-26NOV14DENNOP-... -> 26NOV14DENNOP (the date+teams part shared by all markets of one game)."""
    parts = str(ticker or "").split("-")
    return parts[1] if len(parts) >= 2 else None


def LegFromMarket(row, side="yes", pair=None, label=None):
    """Fair probability for one leg. pair = the other side's market for a two-way moneyline."""
    if row is None:
        return {"label": label, "error": "market not found"}
    side = (side or "yes").lower()
    ask_yes, bid_yes = row.get("yes_ask"), row.get("yes_bid")
    method = None
    fair_yes = None
    if pair is not None:
        nv = Data.NoVig([ask_yes, pair.get("yes_ask")])
        if nv:
            fair_yes, method = nv[0], "no-vig across both teams' yes asks"
    if fair_yes is None and Data.Mid(row) is not None:
        fair_yes, method = Data.Mid(row), "yes bid/ask midpoint"
    if fair_yes is None and ask_yes is not None:
        fair_yes, method = ask_yes, "yes ask only (no two-sided book; fair is overstated)"
    if side == "no":
        fair = None if fair_yes is None else 1 - fair_yes
        ask = None if bid_yes is None else 1 - bid_yes
    else:
        fair, ask = fair_yes, ask_yes
    return {"label": label or row.get("title"), "ticker": row.get("ticker"), "side": side, "ask": ask, "fair": fair,
            "method": method, "game": GameKey(row.get("ticker")), "as_of": row.get("as_of"), "status": row.get("status")}


def ResolveLeg(spec, day, scoreboard_cache):
    side = (spec.get("side") or "yes").lower()
    t = spec.get("ticker")
    desc = spec.get("desc")
    if not t and desc and str(desc).upper().startswith("KX") and "-" in desc:
        t = desc
    if t:
        if ":" in t:
            t, side = t.split(":", 1)
        row = Data.KalshiMarket(t)
        pair = None
        if row and t.startswith("KXNBAGAME-"):
            others = [r for r in Data.KalshiEventMarkets(row["event_ticker"]) if r["ticker"] != t]
            pair = others[0] if len(others) == 1 else None
        return LegFromMarket(row, side, pair, label=t)
    if not desc:
        return {"label": None, "error": "empty leg"}
    if "board" not in scoreboard_cache:
        scoreboard_cache["board"] = Data.EspnScoreboard(day)
    board = scoreboard_cache["board"]
    m = PROP_RE.match(desc.strip())
    if m:
        stat = m.group("stat").lower().rstrip("s") if m.group("stat").lower() not in ("pts", "3pm", "pra", "pts+reb+ast") else m.group("stat").lower()
        series = STAT_SERIES.get(stat) or STAT_SERIES.get(stat.rstrip("s"))
        name, k = m.group("name").strip().lower(), int(m.group("k"))
        for g in board:
            for ev in (f"{series}-{Data.DateCode(day)}{g['away']}{g['home']}", f"{series}-{Data.DateCode(day)}{g['home']}{g['away']}"):
                rows = Data.KalshiEventMarkets(ev)
                hit = [r for r in rows if name in str(r.get("title") or "").lower() and re.search(rf"\b{k}\+", str(r.get("title") or ""))]
                if hit:
                    return LegFromMarket(hit[0], side, None, label=desc)
                if rows:
                    break
        return {"label": desc, "error": f"no {series} market matching '{desc}' on {day}"}
    team = Data.Std(desc)
    for g in board:
        names = {g["home"]: g["home_info"].get("name"), g["away"]: g["away_info"].get("name")}
        for code, full in names.items():
            if team == code or (full and desc.lower() in full.lower()):
                kg = Data.KalshiGame(day, g["away"], g["home"])
                if not kg:
                    return {"label": desc, "error": "Kalshi moneyline not found"}
                me = kg["home"] if code == g["home"] else kg["away"]
                other = kg["away"] if code == g["home"] else kg["home"]
                return LegFromMarket(me, side, other, label=f"{code} to win ({g['away']} @ {g['home']})")
    return {"label": desc, "error": f"could not resolve '{desc}' to a market on {day}"}


def Ticket(legs, price, contracts, fee=FEE):
    """Pure math for a ticket. Fair = product of leg fairs (assumes independence)."""
    ok = [l for l in legs if l.get("fair") is not None]
    fair = None
    if ok and len(ok) == len(legs):
        fair = 1.0
        for l in ok:
            fair *= l["fair"]
    asks = [l.get("ask") for l in legs]
    ask_product = None if any(a is None for a in asks) or not asks else float(__import__("math").prod(asks))
    games = [l.get("game") for l in legs if l.get("game")]
    same = sorted({g for g in games if games.count(g) > 1})
    n = contracts
    out = {"n_legs": len(legs), "fair_prob": fair, "fair_price": fair, "fair_american": FmtAmerican(fair),
           "ask_product": ask_product, "price": price, "contracts": n,
           "cost": None if price is None else round(price * n, 2), "payout_if_win": float(n),
           "profit_if_win": None if price is None else round(n * (1 - price), 2),
           "fee_total": round(fee * n, 2), "breakeven_prob_with_fee": None if price is None else price + fee,
           "price_minus_fair_c": None if (price is None or fair is None) else round((price - fair) * 100, 1),
           "same_game_keys": same, "independence_assumed": len(legs) > 1}
    if price is not None and fair is not None:
        out["value_vs_fair_total"] = round(n * (fair - price - fee), 2)
    return out


def Render(legs, tk, day):
    L = [f"# Slip check - {day} (checked {FmtEt(NowEt())})", "",
         "Compares your price with the market's own no-vig fair price. **Not an edge estimate**: the market is the benchmark, "
         "and paying under fair is rare after fees.", "",
         "| leg | side | market ask | market fair prob | fair odds | method | as of |", "|---|---|---:|---:|---:|---|---|"]
    for l in legs:
        if l.get("error"):
            L.append(f"| {l.get('label')} | | | | | ERROR: {l['error']} | |")
            continue
        L.append(f"| {l['label']} ({l.get('ticker')}) | {l['side']} | {Money(l.get('ask'))} | "
                 f"{'n/a' if l.get('fair') is None else f'{100*l['fair']:.1f}%'} | {FmtAmerican(l.get('fair'))} | {l.get('method')} | {FmtEt(l.get('as_of'), False)} |")
    L.append("")
    if tk["independence_assumed"]:
        L.append("Ticket fair value multiplies the legs' fair probabilities, **assuming the legs are independent**.")
    if tk["same_game_keys"]:
        L.append(f"**Same-game warning:** legs share game(s) {', '.join(tk['same_game_keys'])}. Same-game outcomes are usually correlated "
                 "(e.g. a team winning and its star scoring a lot), so the independent fair value can be badly off in either direction.")
    L += ["", "| item | value |", "|---|---:|",
          f"| market-fair win probability | {'n/a' if tk['fair_prob'] is None else f'{100*tk['fair_prob']:.1f}%'} ({tk['fair_american']}) |",
          f"| market-fair price per $1 | {Money(tk['fair_price'])} |",
          f"| buying each leg at its ask (product) | {Money(tk['ask_product'])} |",
          f"| price you paid per contract | {Money(tk['price'])} |",
          f"| your price minus fair | {'n/a' if tk['price_minus_fair_c'] is None else f'{tk['price_minus_fair_c']:+.1f}c per contract'} |",
          f"| contracts | {tk['contracts']} |",
          f"| cost | {Money(tk['cost'])} |",
          f"| payout if it wins | {Money(tk['payout_if_win'])} (profit {Money(tk['profit_if_win'])} before fees) |",
          f"| fees ({FEE_LABEL}) | {Money(tk['fee_total'])} |",
          f"| win probability needed to break even incl. fee | {'n/a' if tk['breakeven_prob_with_fee'] is None else f'{100*tk['breakeven_prob_with_fee']:.1f}%'} |"]
    if "value_vs_fair_total" in tk:
        L.append(f"| expected result if the market-fair probability is right (after fee) | {Money(tk['value_vs_fair_total'])} |")
    L += ["", "Sources: Kalshi public API (market quotes); ESPN public scoreboard (to find the game for team/prop descriptions)."]
    return "\n".join(L) + "\n"


def Main(argv=None):
    ap = argparse.ArgumentParser(description="Check a slip against market fair prices")
    ap.add_argument("--leg", action="append", default=[], help="ticker[:no] | team | 'Player K+ stat'")
    ap.add_argument("--side", action="append", default=[], help="yes/no per --leg (optional, same order)")
    ap.add_argument("--price", type=float, default=None, help="price paid per contract for the whole ticket ($)")
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--date", default=NowEt().strftime("%Y-%m-%d"))
    ap.add_argument("--file", default=None)
    ap.add_argument("--out", default=None, help="also write the Markdown here")
    a = ap.parse_args(argv)
    specs, price, n, day = [], a.price, a.contracts, a.date
    if a.file:
        body = json.loads(Path(a.file).read_text(encoding="utf-8"))
        specs = body.get("legs", [])
        price = body.get("price", price)
        n = int(body.get("contracts", n))
        day = body.get("date", day)
    for i, leg in enumerate(a.leg):
        specs.append({"desc": leg, "side": a.side[i] if i < len(a.side) else "yes"})
    if not specs:
        ap.error("give at least one --leg or --file")
    cache = {}
    legs = [ResolveLeg(s, day, cache) for s in specs]
    if price is None and len(legs) == 1 and legs[0].get("ask") is not None:
        price = legs[0]["ask"]
    tk = Ticket(legs, price, n)
    text = Render(legs, tk, day)
    print(text)
    if a.out:
        Data.WriteAtomic(a.out, text)
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())