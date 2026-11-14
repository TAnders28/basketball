"""Shared read-only fetchers and math for the decision tools (CheatSheet, Board, SlipCheck, LiveView).

Sources (all free / public, read-only):
- Kalshi public trade API v2: https://api.elections.kalshi.com/trade-api/v2 (markets, candlesticks)
- ESPN public site API: https://site.api.espn.com/apis/site/v2/sports/basketball/nba (scoreboard, summary)
- NBA official injury report PDFs: https://official.nba.com/referee/injury/Injury-Report_<date>_<hh_mmAM>.pdf
- Local games data: E:\\Basketball\\basketball\\data\\season\\games_<season>.json

Every network call can fail; every function returns None / [] on failure and records why in `errors`.
Nothing here places orders or claims an edge.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
INJURY_BASE = "https://official.nba.com/referee/injury"
INJURY_DIR = ROOT / "data" / "injury" / "reports"
FEE = 0.02
FEE_LABEL = "Tim's working estimate of $0.02 per contract (not a verified exchange fee)"
PROP_SERIES = {"KXNBAPTS": "points", "KXNBAREB": "rebounds", "KXNBAAST": "assists", "KXNBA3PT": "threes", "KXNBAPRA": "pts+reb+ast"}
# ESPN / other short codes -> NBA standard codes used by Kalshi and the local data.
ALIAS = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "PHO": "PHX", "WSH": "WAS", "UTAH": "UTA", "BRK": "BKN", "NOR": "NOP"}
errors: list = []


def Std(abbr):
    a = str(abbr or "").upper().strip()
    return ALIAS.get(a, a) or None


def Num(x):
    try:
        if x is None or x == "":
            return None
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def NowEt():
    return datetime.now(ET)


def ToEt(x):
    """datetime | ISO string | unix ts -> aware ET datetime (or None)."""
    if x is None or x == "":
        return None
    try:
        if isinstance(x, datetime):
            dt = x
        elif isinstance(x, (int, float)):
            dt = datetime.fromtimestamp(float(x), tz=timezone.utc)
        else:
            s = str(x).strip().replace("Z", "+00:00")
            if len(s) == 16 + 6 and s[10] == "T" and s.count(":") == 2:  # 2026-11-15T00:00+00:00
                s = s[:16] + ":00" + s[16:]
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ET)
    except Exception:
        return None


def FmtEt(x, with_date=True):
    dt = ToEt(x)
    if dt is None:
        return "unknown time"
    h = dt.strftime("%I:%M %p").lstrip("0")
    return (dt.strftime("%a %b %d ") + h + " ET") if with_date else (h + " ET")


def IsoEt(x):
    dt = ToEt(x)
    return None if dt is None else dt.isoformat(timespec="seconds")


def Money(p):
    if p is None:
        return "n/a"
    return f"-${abs(p):.2f}" if p < 0 else f"${p:.2f}"


# ---------------------------------------------------------------- odds math
def AmericanToProb(ml):
    v = Num(ml)
    if v is None or v == 0 or -100 < v < 100:
        return None
    return 100.0 / (v + 100.0) if v > 0 else -v / (-v + 100.0)


def NoVig(probs):
    """Normalize implied probabilities so they sum to 1. Returns None if any is missing."""
    if not probs or any(p is None or p <= 0 for p in probs):
        return None
    s = float(sum(probs))
    return [p / s for p in probs]


def ProbToAmerican(p):
    if p is None or p <= 0 or p >= 1:
        return None
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def FmtAmerican(p):
    a = ProbToAmerican(p)
    return "n/a" if a is None else (f"+{a}" if a > 0 else str(a))


def EdgeCents(prob, ask, fee=FEE):
    """(prob - ask - fee) x 100. Arithmetic only; a positive number is NOT a validated edge."""
    if prob is None or ask is None:
        return None
    return round((prob - ask - fee) * 100.0, 1)


# ---------------------------------------------------------------- HTTP
def _Get(url, params=None):
    try:
        from InPlay.Http import GetJson
        return GetJson(url, params=params, tries=3, timeout=20)
    except Exception as ex:
        errors.append(f"{url.split('?')[0]}: {ex}")
        return None


def DateCode(day):
    d = day if isinstance(day, date) else datetime.strptime(str(day), "%Y-%m-%d").date()
    return d.strftime("%y%b%d").upper()


# ---------------------------------------------------------------- ESPN
_summary_cache: dict = {}


def EspnScoreboard(day):
    fetched = NowEt()
    body = _Get(f"{ESPN}/scoreboard", {"dates": str(day).replace("-", "")})
    games = []
    for e in (body or {}).get("events", []) or []:
        try:
            c = e["competitions"][0]
            side = {}
            for t in c.get("competitors", []):
                recs = {r.get("type") or r.get("name"): r.get("summary") for r in t.get("records", []) or []}
                side[t.get("homeAway")] = {
                    "abbr": Std(t["team"].get("abbreviation")), "name": t["team"].get("displayName"),
                    "score": Num(t.get("score")), "record": recs.get("total") or recs.get("overall"),
                    "home_record": recs.get("home"), "road_record": recs.get("road"),
                }
            st = e.get("status", {})
            games.append({
                "espn_id": e.get("id"), "start": IsoEt(e.get("date")), "short": e.get("shortName"),
                "home": side.get("home", {}).get("abbr"), "away": side.get("away", {}).get("abbr"),
                "home_info": side.get("home", {}), "away_info": side.get("away", {}),
                "state": (st.get("type") or {}).get("state"), "detail": (st.get("type") or {}).get("detail"),
                "period": st.get("period"), "clock": st.get("displayClock"),
                "completed": bool((st.get("type") or {}).get("completed")),
                "fetched_at": fetched.isoformat(timespec="seconds"),
            })
        except Exception as ex:
            errors.append(f"espn scoreboard parse: {ex}")
    return games


def EspnSummary(espn_id, refresh=False):
    if espn_id is None:
        return None
    if refresh or espn_id not in _summary_cache:
        _summary_cache[espn_id] = (_Get(f"{ESPN}/summary", {"event": espn_id}), NowEt())
    return _summary_cache[espn_id]


def EspnInjuries(summary):
    body, fetched = summary or (None, None)
    out = []
    for team in (body or {}).get("injuries", []) or []:
        abbr = Std((team.get("team") or {}).get("abbreviation"))
        for inj in team.get("injuries", []) or []:
            ath = inj.get("athlete") or {}
            det = inj.get("details") or {}
            note = " ".join(x for x in [det.get("type"), det.get("location"), det.get("detail"), det.get("side")] if x) or (inj.get("shortComment") or None)
            out.append({"team": abbr, "player": ath.get("displayName"), "status": inj.get("status") or (inj.get("type") or {}).get("description"),
                        "note": note, "updated_et": IsoEt(inj.get("date")), "source": "ESPN summary injuries"})
    return out


def EspnBookOdds(summary, home, away):
    """Sportsbook moneylines from ESPN pickcenter (provider named), with no-vig fair probabilities."""
    body, fetched = summary or (None, None)
    for pc in (body or {}).get("pickcenter", []) or []:
        hml = Num((pc.get("homeTeamOdds") or {}).get("moneyLine"))
        aml = Num((pc.get("awayTeamOdds") or {}).get("moneyLine"))
        if hml is None or aml is None:
            continue
        fair = NoVig([AmericanToProb(hml), AmericanToProb(aml)])
        return {"provider": (pc.get("provider") or {}).get("name"), "home_ml": hml, "away_ml": aml,
                "home_fair": None if not fair else fair[0], "away_fair": None if not fair else fair[1],
                "hold": None if not fair else AmericanToProb(hml) + AmericanToProb(aml) - 1,
                "spread": pc.get("details"), "total": pc.get("overUnder"),
                "as_of": IsoEt(fetched), "source": "ESPN summary pickcenter"}
    return None


def EspnPredictor(summary):
    body, fetched = summary or (None, None)
    pred = (body or {}).get("predictor") or {}
    h = Num((pred.get("homeTeam") or {}).get("gameProjection"))
    return None if h is None else {"home_win_prob": h / 100.0, "source": "ESPN Matchup Predictor (summary.predictor)", "as_of": IsoEt(fetched)}


def EspnWinProb(summary):
    body, fetched = summary or (None, None)
    wp = (body or {}).get("winprobability") or []
    if not wp:
        return None
    last = wp[-1]
    h = Num(last.get("homeWinPercentage"))
    return None if h is None else {"home_win_prob": h, "play_id": last.get("playId"), "n_points": len(wp),
                                   "source": "ESPN summary winprobability", "as_of": IsoEt(fetched)}


def EspnSeasonSeries(summary):
    body, _ = summary or (None, None)
    out = []
    for ss in (body or {}).get("seasonseries", []) or []:
        if ss.get("summary"):
            out.append(ss.get("summary"))
    return out


# ---------------------------------------------------------------- Kalshi
def _MarketRow(m):
    ask, bid = Num(m.get("yes_ask_dollars")), Num(m.get("yes_bid_dollars"))
    if ask is None and m.get("yes_ask") is not None:
        ask = Num(m.get("yes_ask")) / 100.0
    if bid is None and m.get("yes_bid") is not None:
        bid = Num(m.get("yes_bid")) / 100.0
    return {"ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "title": m.get("title"),
            "yes_ask": ask if ask and ask > 0 else None, "yes_bid": bid if bid and bid > 0 else None,
            "last": Num(m.get("last_price_dollars")), "volume": Num(m.get("volume_fp") or m.get("volume")),
            "status": m.get("status"), "result": m.get("result") or None, "open_time": m.get("open_time"),
            "yes_sub_title": m.get("yes_sub_title"), "floor_strike": m.get("floor_strike")}


def Mid(row):
    if row is None or row.get("yes_ask") is None or row.get("yes_bid") is None:
        return None
    return 0.5 * (row["yes_ask"] + row["yes_bid"])


def KalshiMarket(ticker):
    fetched = NowEt()
    body = _Get(f"{KALSHI}/markets/{ticker}")
    m = (body or {}).get("market")
    if not m:
        return None
    r = _MarketRow(m)
    r["as_of"] = fetched.isoformat(timespec="seconds")
    return r


def KalshiEventMarkets(event_ticker):
    fetched = NowEt()
    body = _Get(f"{KALSHI}/markets", {"event_ticker": event_ticker, "limit": 1000})
    rows = [_MarketRow(m) for m in (body or {}).get("markets", []) or []]
    for r in rows:
        r["as_of"] = fetched.isoformat(timespec="seconds")
    return rows


def KalshiGame(day, away, home):
    """Moneyline pair for one game. Kalshi event ticker = KXNBAGAME-<yyMONdd><AWAY><HOME>."""
    ev = f"KXNBAGAME-{DateCode(day)}{away}{home}"
    rows = KalshiEventMarkets(ev)
    if not rows:  # some events list home first; try the reverse once
        ev2 = f"KXNBAGAME-{DateCode(day)}{home}{away}"
        rows = KalshiEventMarkets(ev2)
        ev = ev2 if rows else ev
    sides = {}
    for r in rows:
        team = str(r["ticker"]).rsplit("-", 1)[-1]
        sides[Std(team)] = r
    if not sides:
        return None
    h, a = sides.get(home), sides.get(away)
    fair_ask = NoVig([h and h["yes_ask"], a and a["yes_ask"]])
    fair_mid = NoVig([Mid(h), Mid(a)])
    return {"event_ticker": ev, "home": h, "away": a,
            "home_fair_ask": fair_ask and fair_ask[0], "away_fair_ask": fair_ask and fair_ask[1],
            "home_fair_mid": fair_mid and fair_mid[0], "away_fair_mid": fair_mid and fair_mid[1],
            "overround_asks": None if not (h and a and h["yes_ask"] and a["yes_ask"]) else h["yes_ask"] + a["yes_ask"] - 1,
            "as_of": (h or a or {}).get("as_of"), "source": "Kalshi public API (KXNBAGAME markets)"}


def KalshiCandles(ticker, start_ts, end_ts, interval=60):
    series = str(ticker).split("-", 1)[0]
    body = _Get(f"{KALSHI}/series/{series}/markets/{ticker}/candlesticks",
                {"start_ts": int(start_ts), "end_ts": int(end_ts), "period_interval": interval})
    try:
        from Props.PullProps import NormalizeCandle
        raw = (body or {}).get("candlesticks") or []
        return [c for c in (NormalizeCandle(x) for x in raw) if c.get("t") is not None]
    except Exception as ex:
        errors.append(f"candles {ticker}: {ex}")
        return []


def LineMove(market_row, now=None):
    """First available hourly candle (since the market opened) vs the current quote."""
    if not market_row or not market_row.get("ticker"):
        return None
    now = now or NowEt()
    open_dt = ToEt(market_row.get("open_time")) or (now - timedelta(days=3))
    start = max(open_dt, now - timedelta(days=7))
    candles = KalshiCandles(market_row["ticker"], start.timestamp(), now.timestamp(), 60)
    candles = [c for c in candles if c.get("ask") is not None or c.get("last") is not None]
    if not candles:
        return {"first": None, "note": "no candles returned", "market_open_et": IsoEt(open_dt)}
    first = candles[0]
    first_price = first.get("ask") if first.get("ask") is not None else first.get("last")
    now_price = market_row.get("yes_ask")
    return {"first_t_et": IsoEt(first["t"]), "first_yes_ask": first.get("ask"), "first_yes_bid": first.get("bid"),
            "first_price": first_price, "now_yes_ask": now_price,
            "move_c": None if (now_price is None or first_price is None) else round((now_price - first_price) * 100, 1),
            "hours_back": round((now - ToEt(first["t"])).total_seconds() / 3600, 1),
            "market_open_et": IsoEt(open_dt), "n_candles": len(candles),
            "note": "hourly candles; 'open' = first candle with a price, which can be later than the market open"}


def KalshiProps(day, away, home, main_line_only=True):
    """Kalshi player-prop ladders for one game. Returns the strike nearest 50c per player/stat by default."""
    out = []
    for series, stat in PROP_SERIES.items():
        rows = []
        for ev in (f"{series}-{DateCode(day)}{away}{home}", f"{series}-{DateCode(day)}{home}{away}"):
            rows = KalshiEventMarkets(ev)
            if rows:
                break
        by_player = defaultdict(list)
        for r in rows:
            title = r.get("title") or ""
            if ":" in title:
                player, rest = title.split(":", 1)
            else:
                player, rest = title, ""
            k = None
            for tok in rest.replace("+", " ").split():
                if tok.isdigit():
                    k = int(tok)
                    break
            if k is None:
                tail = str(r["ticker"]).rsplit("-", 1)[-1]
                k = int(tail) if tail.isdigit() else None
            r.update({"player": player.strip(), "stat": stat, "k": k, "series": series})
            by_player[r["player"]].append(r)
        for player, lst in by_player.items():
            lst = [x for x in lst if x.get("yes_ask") is not None]
            if not lst:
                continue
            if main_line_only:
                lst = [min(lst, key=lambda x: abs((Mid(x) or x["yes_ask"]) - 0.5))]
            out.extend(lst)
    return out


# ---------------------------------------------------------------- injuries (official PDF)
def _Slots(day, now):
    d = datetime.strptime(str(day), "%Y-%m-%d").date()
    t = datetime(d.year, d.month, d.day, 23, 45, tzinfo=ET)
    if now.date() == d:
        t = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    for i in range(0, 4 * 14):
        s = t - timedelta(minutes=15 * i)
        if s.date() != d:
            break
        yield s


def OfficialInjuryReport(day, now=None, max_tries=40):
    """Latest official NBA injury report PDF for `day` at or before now. Cached under data\\injury\\reports."""
    now = now or NowEt()
    try:
        import requests
        from Backtest.InjuryReports import ParsePdf
    except Exception as ex:
        errors.append(f"injury parser unavailable: {ex}")
        return None
    INJURY_DIR.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"User-Agent": "BasketbotDecisionTools/1.0 (read-only; personal research)"})
    for n, s in enumerate(_Slots(day, now)):
        if n >= max_tries:
            break
        name = f"Injury-Report_{s.strftime('%Y-%m-%d')}_{s.strftime('%I_%M%p')}.pdf"
        path = INJURY_DIR / name
        url = f"{INJURY_BASE}/{name}"
        if not path.exists():
            try:
                time.sleep(0.25)
                r = sess.get(url, timeout=20)
            except Exception as ex:
                errors.append(f"injury fetch {name}: {ex}")
                continue
            if r.status_code != 200 or not r.content.startswith(b"%PDF"):
                continue
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(r.content)
            os.replace(tmp, path)
        try:
            parsed = ParsePdf(path)
        except Exception as ex:
            errors.append(f"injury parse {name}: {ex}")
            continue
        return {"url": url, "file": str(path), "report_slot_et": s.isoformat(timespec="minutes"),
                "header_ts": str(parsed.get("header_ts")) if parsed.get("header_ts") else None,
                "rows": parsed.get("rows") or [], "source": "NBA official injury report PDF"}
    return None


def InjuriesForGame(report, away, home):
    if not report:
        return []
    out = []
    for r in report["rows"]:
        m = str(r.get("matchup") or "")
        if m.replace(" ", "") in (f"{away}@{home}",) or (Std(r.get("team")) in (away, home) and away in m and home in m):
            out.append({"team": Std(r.get("team")), "player": r.get("player"), "status": r.get("status"),
                        "note": r.get("reason") or None, "source": report["source"], "report_slot_et": report["report_slot_et"]})
    return out


# ---------------------------------------------------------------- local games + form
def LoadGames(day, back_days=400):
    """Completed games before `day`: local season files, topped up from ESPN scoreboards for missing recent dates."""
    d = datetime.strptime(str(day), "%Y-%m-%d").date()
    rows = []
    for season in (d.year - 2, d.year - 1, d.year):
        p = ROOT / "data" / "season" / f"games_{season}.json"
        if not p.exists():
            continue
        try:
            for g in json.loads(p.read_text(encoding="utf-8")):
                if g.get("game_date") and g["game_date"] < str(day) and g.get("home_score") is not None:
                    rows.append({"date": g["game_date"], "home": Std(g["home_team"]), "away": Std(g["away_team"]),
                                 "hs": g["home_score"], "as": g["away_score"], "hst": g.get("home_stats"), "ast": g.get("away_stats"),
                                 "type": g.get("season_type"), "src": "local"})
        except Exception as ex:
            errors.append(f"local games {p.name}: {ex}")
    last_local = max((r["date"] for r in rows), default=None)
    topup = []
    if last_local:
        have = {(r["date"], r["home"], r["away"]) for r in rows}
        cur = datetime.strptime(last_local, "%Y-%m-%d").date()  # inclusive: the last local day can be partial
        while cur < d and (d - cur).days <= 10:
            for g in EspnScoreboard(cur.isoformat()):
                if g["completed"] and g["home_info"].get("score") is not None and (cur.isoformat(), g["home"], g["away"]) not in have:
                    topup.append({"date": cur.isoformat(), "home": g["home"], "away": g["away"],
                                  "hs": int(g["home_info"]["score"]), "as": int(g["away_info"]["score"]), "hst": None, "ast": None,
                                  "type": "Regular Season", "src": "espn"})
            cur += timedelta(days=1)
    rows += topup
    rows.sort(key=lambda r: r["date"])
    return rows, {"local_through": last_local, "espn_topup_games": len(topup)}


def SeasonStart(day):
    d = datetime.strptime(str(day), "%Y-%m-%d").date()
    return f"{d.year if d.month >= 8 else d.year - 1}-08-01"


def TeamGames(team, games, since=None):
    out = []
    for g in games:
        if since and g["date"] < since:
            continue
        if g["home"] == team:
            out.append({"date": g["date"], "opp": g["away"], "home": True, "pf": g["hs"], "pa": g["as"], "st": g["hst"], "ost": g["ast"]})
        elif g["away"] == team:
            out.append({"date": g["date"], "opp": g["home"], "home": False, "pf": g["as"], "pa": g["hs"], "st": g["ast"], "ost": g["hst"]})
    return out


def _Poss(s):
    try:
        return s["fga"] - s["oreb"] + s["tov"] + 0.44 * s["fta"]
    except Exception:
        return None


def TeamForm(team, day, games):
    season = TeamGames(team, games, SeasonStart(day))
    if not season:
        return None
    last10 = season[-10:]
    w10 = sum(1 for g in last10 if g["pf"] > g["pa"])
    w = sum(1 for g in season if g["pf"] > g["pa"])
    d = datetime.strptime(str(day), "%Y-%m-%d").date()
    last = datetime.strptime(season[-1]["date"], "%Y-%m-%d").date()
    rest = (d - last).days - 1
    in7 = sum(1 for g in season if (d - datetime.strptime(g["date"], "%Y-%m-%d").date()).days <= 7)
    streak_n, streak_w = 0, None
    for g in reversed(season):
        won = g["pf"] > g["pa"]
        if streak_w is None:
            streak_w = won
        if won != streak_w:
            break
        streak_n += 1
    home_g = [g for g in season if g["home"]]
    road_g = [g for g in season if not g["home"]]
    return {"team": team, "record": f"{w}-{len(season)-w}", "last10": f"{w10}-{len(last10)-w10}",
            "last10_margin": round(sum(g["pf"] - g["pa"] for g in last10) / len(last10), 1),
            "season_margin": round(sum(g["pf"] - g["pa"] for g in season) / len(season), 1),
            "rest_days": rest, "back_to_back": rest == 0, "games_last_7d": in7,
            "streak": f"{'W' if streak_w else 'L'}{streak_n}",
            "home_record": f"{sum(1 for g in home_g if g['pf']>g['pa'])}-{sum(1 for g in home_g if g['pf']<g['pa'])}",
            "road_record": f"{sum(1 for g in road_g if g['pf']>g['pa'])}-{sum(1 for g in road_g if g['pf']<g['pa'])}",
            "last_game": f"{season[-1]['date']} {'vs' if season[-1]['home'] else '@'} {season[-1]['opp']} {season[-1]['pf']}-{season[-1]['pa']}",
            "n_games": len(season)}


def TeamStats(team, day, games):
    """Season-to-date four factors and ratings from local box scores (ESPN top-up games have no box)."""
    gs = [g for g in TeamGames(team, games, SeasonStart(day)) if g["st"] and g["ost"]]
    if not gs:
        return None
    tot = defaultdict(float)
    opp = defaultdict(float)
    poss = 0.0
    for g in gs:
        p1, p2 = _Poss(g["st"]), _Poss(g["ost"])
        if p1 is None or p2 is None:
            continue
        poss += 0.5 * (p1 + p2)
        for k in ("pts", "fgm", "fga", "fg3m", "fg3a", "fta", "ftm", "oreb", "dreb", "tov", "ast"):
            tot[k] += g["st"].get(k) or 0
            opp[k] += g["ost"].get(k) or 0
    if poss <= 0 or tot["fga"] <= 0:
        return None
    n = len(gs)
    return {"n_games": n, "through": gs[-1]["date"],
            "ortg": 100 * tot["pts"] / poss, "drtg": 100 * opp["pts"] / poss, "net": 100 * (tot["pts"] - opp["pts"]) / poss,
            "pace": poss / n, "efg": (tot["fgm"] + 0.5 * tot["fg3m"]) / tot["fga"],
            "opp_efg": (opp["fgm"] + 0.5 * opp["fg3m"]) / max(opp["fga"], 1),
            "tov_pct": tot["tov"] / poss, "opp_tov_pct": opp["tov"] / poss,
            "orb_pct": tot["oreb"] / max(tot["oreb"] + opp["dreb"], 1), "drb_pct": tot["dreb"] / max(tot["dreb"] + opp["oreb"], 1),
            "ftr": tot["fta"] / tot["fga"], "three_rate": tot["fg3a"] / tot["fga"], "three_pct": tot["fg3m"] / max(tot["fg3a"], 1),
            "opp_three_rate": opp["fg3a"] / max(opp["fga"], 1), "source": "local games data (box scores)"}


def HeadToHead(home, away, day, games, n=6):
    out = []
    for g in games:
        if {g["home"], g["away"]} == {home, away}:
            out.append(f"{g['date']}: {g['away']} {g['as']} @ {g['home']} {g['hs']}")
    return out[-n:]


def Trends(team, day, games):
    """Small-sample context lines. Context, not an edge."""
    season = TeamGames(team, games, SeasonStart(day))
    if not season:
        return []
    lines = []
    d = datetime.strptime(str(day), "%Y-%m-%d").date()
    dates = [datetime.strptime(g["date"], "%Y-%m-%d").date() for g in season]
    b2b = [g for i, g in enumerate(season) if i > 0 and (dates[i] - dates[i - 1]).days == 1]
    if b2b:
        lines.append(f"{team} on 2nd night of back-to-backs: {sum(1 for g in b2b if g['pf']>g['pa'])}-{sum(1 for g in b2b if g['pf']<g['pa'])} (n={len(b2b)})")
    close = [g for g in season if abs(g["pf"] - g["pa"]) <= 5]
    if close:
        lines.append(f"{team} in games decided by 5 or fewer: {sum(1 for g in close if g['pf']>g['pa'])}-{sum(1 for g in close if g['pf']<g['pa'])} (n={len(close)})")
    rec = {}
    for g in games:
        if g["date"] >= SeasonStart(day):
            for t, won in ((g["home"], g["hs"] > g["as"]), (g["away"], g["as"] > g["hs"])):
                rec.setdefault(t, [0, 0])[0 if won else 1] += 1
    good = [g for g in season if rec.get(g["opp"]) and rec[g["opp"]][0] > rec[g["opp"]][1]]
    if good:
        lines.append(f"{team} vs teams currently above .500: {sum(1 for g in good if g['pf']>g['pa'])}-{sum(1 for g in good if g['pf']<g['pa'])} (n={len(good)})")
    return lines


def WriteAtomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- one-call slate bundle
def SlateBundle(day, with_props=True, with_injury_pdf=True, with_moves=True, now=None):
    """Everything CheatSheet and Board need for one date, fetched once."""
    now = now or NowEt()
    del errors[:]
    board = EspnScoreboard(day)
    games, games_info = LoadGames(day)
    report = OfficialInjuryReport(day, now) if with_injury_pdf else None
    out = []
    for g in board:
        home, away = g["home"], g["away"]
        summ = EspnSummary(g["espn_id"])
        kg = KalshiGame(day, away, home)
        move = None
        if with_moves and kg and kg.get("home"):
            move = LineMove(kg["home"], now)
        out.append({
            "espn": g, "home": home, "away": away, "kalshi": kg, "move_home": move,
            "book": EspnBookOdds(summ, home, away), "predictor": EspnPredictor(summ), "wp": EspnWinProb(summ),
            "injuries_official": InjuriesForGame(report, away, home), "injuries_espn": EspnInjuries(summ),
            "season_series": EspnSeasonSeries(summ),
            "form": {home: TeamForm(home, day, games), away: TeamForm(away, day, games)},
            "stats": {home: TeamStats(home, day, games), away: TeamStats(away, day, games)},
            "h2h": HeadToHead(home, away, day, games),
            "trends": Trends(home, day, games) + Trends(away, day, games),
            "props": KalshiProps(day, away, home) if with_props else [],
        })
    return {"date": str(day), "generated_at": now.isoformat(timespec="seconds"), "games": out,
            "injury_report": None if not report else {k: report[k] for k in ("url", "report_slot_et", "header_ts", "source")},
            "games_info": games_info, "errors": list(errors)}