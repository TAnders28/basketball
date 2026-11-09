"""Public GET clients only. No order, cancel, or account endpoints.

Hosts are allow-listed. Redirects are rejected. A response that is not JSON
with status 200 is a source outage, not a quote.
"""

import json
import re
from datetime import datetime, timezone
from urllib.parse import quote, urlsplit

from LiveOdds.Teams import ResolveTeam
from LiveOdds.Time import EtDate, ParseInstant

ALLOWED_HOSTS = {
    "gamma-api.polymarket.com",
    "clob.polymarket.com",
    "cdn.nba.com",
    "external-api.kalshi.com",
}
# cdn.nba.com returns 403 without the same Referer and User-Agent nba_api's live
# scoreboard client sends. These are not credentials.
NBA_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:158.0) Gecko/20100101 Firefox/158.0",
}
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "basketball-live-odds-readonly/1.0",
}
NBA_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
POLYMARKET_SPORTS_URL = "https://gamma-api.polymarket.com/sports"
KALSHI_SERIES = "KXNBAGAME"
_SLUG = re.compile(r"^nba-([a-z]{3})-([a-z]{3})-(\d{4}-\d{2}-\d{2})$")
_TICKER = re.compile(r"^KXNBAGAME-(\d{2})([A-Z]{3})(\d{2})([A-Z]{3})([A-Z]{3})-([A-Z]{3})$")
_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
_MAX_PAGES = 10


def RequestsGet(url, headers, timeout):
    import requests
    response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=False)
    if response.status_code != 200:
        return response.status_code, None
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, None


class ReadOnlyHttp:
    """GET JSON from the allow-listed public hosts. There is no method that sends an order."""

    def __init__(self, get=None, clock=None, timeout=20):
        self._get = get or RequestsGet
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.timeout = timeout

    def GetJson(self, url, headers=None):
        observed = self._clock()
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.netloc not in ALLOWED_HOSTS:
            return {"ok": False, "error": "host_not_allowed", "url": url, "observed_at": observed}
        try:
            status, body = self._get(url, headers or DEFAULT_HEADERS, self.timeout)
        except Exception as exc:
            return {"ok": False, "error": "source_outage", "detail": type(exc).__name__,
                    "url": url, "observed_at": self._clock()}
        observed = self._clock()
        if status != 200 or not isinstance(body, (dict, list)):
            return {"ok": False, "error": "source_outage", "status": status, "url": url, "observed_at": observed}
        return {"ok": True, "status": status, "body": body, "url": url, "observed_at": observed}


def _JsonList(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            loaded = json.loads(value)
        except ValueError:
            return None
        if isinstance(loaded, list):
            return loaded
    return None


def _Schedule(raw):
    if not isinstance(raw, dict):
        return None
    return {"rate": raw.get("rate"), "exponent": raw.get("exponent"),
            "taker_only": raw.get("takerOnly", raw.get("taker_only")),
            "rebate_rate": raw.get("rebateRate", raw.get("rebate_rate"))}


def _DateText(game_date: str) -> list:
    parsed = datetime.strptime(game_date, "%Y-%m-%d")
    month, full, day, year = parsed.strftime("%b"), parsed.strftime("%B"), str(parsed.day), str(parsed.year)
    return [f"{month} {day}, {year}", f"{full} {day}, {year}"]


def ClassifyPolymarketMarket(raw: dict, game_date: str) -> dict:
    """Keep a moneyline only when slug, tip time, and outcomes name the same game.

    Other dates are skipped. A market dated today that disagrees with itself is rejected.
    """
    slug = raw.get("slug") or ""
    if raw.get("sportsMarketType") != "moneyline":
        return {"status": "skip", "reason": "not_moneyline", "slug": slug}
    match = _SLUG.fullmatch(slug)
    event = ((raw.get("events") or [None])[0]) or {}
    start, kind = ParseInstant(raw.get("gameStartTime"))
    signals = []
    if match:
        signals.append(("slug", match.group(3)))
    if kind == "naive":
        return {"status": "reject", "reason": "tip_timestamp_naive", "slug": slug}
    if start is not None:
        signals.append(("gameStartTime", EtDate(start)))
    if event.get("eventDate"):
        signals.append(("eventDate", str(event["eventDate"])[:10]))
    if not signals:
        return {"status": "skip", "reason": "date_unconfirmed", "slug": slug}
    dates = {value for _, value in signals}
    if dates == {game_date}:
        pass
    elif game_date not in dates:
        return {"status": "skip", "reason": "other_date", "slug": slug}
    else:
        return {"status": "reject", "reason": "date_mismatch", "slug": slug, "signals": signals}
    if start is None and not event.get("eventDate"):
        return {"status": "reject", "reason": "date_unconfirmed", "slug": slug, "signals": signals}
    if not match:
        return {"status": "reject", "reason": "slug_unparsed", "slug": slug, "signals": signals}
    away, home = match.group(1).upper(), match.group(2).upper()
    away_team, home_team = ResolveTeam(away), ResolveTeam(home)
    if not away_team["abbr"] or not home_team["abbr"]:
        return {"status": "reject", "reason": "slug_team", "slug": slug}
    names, tokens = _JsonList(raw.get("outcomes")), _JsonList(raw.get("clobTokenIds"))
    if not names or not tokens or len(names) != 2 or len(tokens) != 2:
        return {"status": "reject", "reason": "outcome_shape", "slug": slug}
    outcomes = []
    for name, token in zip(names, tokens):
        team = ResolveTeam(name)
        if not team["abbr"]:
            return {"status": "reject", "reason": team["reason"], "slug": slug, "input": name}
        outcomes.append({"name": name, "abbr": team["abbr"], "token_id": str(token),
                         "side": "away" if team["abbr"] == away else "home" if team["abbr"] == home else None})
    if {row["abbr"] for row in outcomes} != {away, home} or any(row["side"] is None for row in outcomes):
        return {"status": "reject", "reason": "outcome_team_mismatch", "slug": slug}
    description = raw.get("description") or event.get("description") or ""
    return {"status": "keep", "market": {
        "slug": slug, "away": away, "home": home, "date": game_date,
        "condition_id": raw.get("conditionId"), "question": raw.get("question"),
        "outcomes": outcomes, "token_ids": [row["token_id"] for row in outcomes],
        "closed": raw.get("closed"), "accepting_orders": raw.get("acceptingOrders"),
        "enable_order_book": raw.get("enableOrderBook"), "archived": raw.get("archived"),
        "resolution_status": raw.get("umaResolutionStatus"),
        "description": description, "resolution_source": raw.get("resolutionSource") or event.get("resolutionSource"),
        "fee_schedule": _Schedule(raw.get("feeSchedule")), "fee_type": raw.get("feeType"),
        "fees_enabled": raw.get("feesEnabled"), "maker_base_fee": raw.get("makerBaseFee"),
        "taker_base_fee": raw.get("takerBaseFee"),
        "indicative": {"best_bid": raw.get("bestBid"), "best_ask": raw.get("bestAsk"),
                       "outcome_prices": raw.get("outcomePrices"),
                       "tag": "indicative",
                       "note": "Gamma bestBid, bestAsk, and outcomePrices are not a verified bid/ask book."},
        "event": {"score": event.get("score"), "period": event.get("period"), "elapsed": event.get("elapsed"),
                  "live": event.get("live"), "ended": event.get("ended"), "updated_at": event.get("updatedAt"),
                  "finished_timestamp": event.get("finishedTimestamp"), "game_id": event.get("gameId")},
        "urls": {
            "event": f"https://polymarket.com/event/{slug}",
            "gamma_event": f"https://gamma-api.polymarket.com/events/slug/{slug}",
            "gamma_market": f"https://gamma-api.polymarket.com/markets/slug/{slug}",
        },
    }}


def _TickerDate(year_2, month, day) -> str:
    month_number = _MONTHS.get(month)
    if month_number is None:
        return ""
    return f"{2000 + int(year_2):04d}-{month_number:02d}-{int(day):02d}"


def ClassifyKalshiMarket(raw: dict, game_date: str) -> dict:
    ticker = raw.get("ticker") or ""
    match = _TICKER.fullmatch(ticker)
    if not match:
        return {"status": "skip", "reason": "ticker_unparsed", "ticker": ticker}
    ticker_date = _TickerDate(match.group(1), match.group(2), match.group(3))
    away, home, side = match.group(4), match.group(5), match.group(6)
    if ticker_date != game_date:
        return {"status": "skip", "reason": "other_date", "ticker": ticker}
    if raw.get("status") != "active":
        return {"status": "reject", "reason": "market_not_active", "ticker": ticker}
    if not raw.get("event_ticker"):
        return {"status": "reject", "reason": "missing_event_ticker", "ticker": ticker}
    if away == home or side not in (away, home):
        return {"status": "reject", "reason": "ticker_teams", "ticker": ticker}
    titled = ResolveTeam(raw.get("yes_sub_title") or raw.get("title"))
    if titled["abbr"] != side:
        return {"status": "reject", "reason": "outcome_team_mismatch", "ticker": ticker,
                "input": raw.get("yes_sub_title")}
    rules = "\n".join(part for part in (raw.get("rules_primary"), raw.get("rules_secondary")) if part)
    if not any(text in rules for text in _DateText(game_date)):
        return {"status": "reject", "reason": "rules_date_mismatch", "ticker": ticker}
    return {"status": "keep", "market": {
        "ticker": ticker, "event_ticker": raw.get("event_ticker"),
        "away": away, "home": home, "side": side, "yes_sub_title": raw.get("yes_sub_title"),
        "status": raw.get("status"), "rules": rules,
        "yes_bid": raw.get("yes_bid_dollars"), "yes_ask": raw.get("yes_ask_dollars"),
        "yes_bid_size": raw.get("yes_bid_size_fp"), "yes_ask_size": raw.get("yes_ask_size_fp"),
        "no_bid": raw.get("no_bid_dollars"), "no_ask": raw.get("no_ask_dollars"),
        "notional_value_dollars": raw.get("notional_value_dollars"),
        "metadata_updated_at": raw.get("updated_time"),
        "api_url": f"https://external-api.kalshi.com/trade-api/v2/markets/{ticker}",
        "orderbook_url": f"https://external-api.kalshi.com/trade-api/v2/markets/{ticker}/orderbook",
    }}


def _Pages(http, url, item_key, cursor_key, cursor_param):
    items, page_url = [], url
    for _ in range(_MAX_PAGES):
        got = http.GetJson(page_url)
        if not got["ok"]:
            return None, got["error"]
        body = got["body"]
        batch = body.get(item_key) if isinstance(body, dict) else None
        if not isinstance(batch, list):
            return None, "unexpected_payload"
        items.extend(batch)
        cursor = body.get(cursor_key) if isinstance(body, dict) else None
        if not cursor:
            return items, None
        joiner = "&" if "?" in url else "?"
        page_url = url + joiner + cursor_param + "=" + quote(str(cursor), safe="")
    return None, "truncated"


def FetchPolymarketMarkets(http) -> tuple:
    sports = http.GetJson(POLYMARKET_SPORTS_URL)
    if not sports["ok"] or not isinstance(sports["body"], list):
        return [], sports.get("error") or "source_outage"
    tag = None
    for row in sports["body"]:
        if str(row.get("sport", "")).lower() == "nba" and row.get("primaryTagId") is not None:
            tag = str(row["primaryTagId"])
            break
    if tag is None:
        return [], "nba_tag_missing"
    url = ("https://gamma-api.polymarket.com/markets/keyset"
           f"?sports_market_types=moneyline&closed=false&limit=50&tag_id={quote(tag)}")
    return _Pages(http, url, "markets", "next_cursor", "after_cursor")


def FetchBooksAndFees(http, markets) -> tuple:
    books, fees = {}, {}
    for market in markets:
        for token_id in market["token_ids"]:
            book_url = f"https://clob.polymarket.com/book?token_id={quote(token_id, safe='')}"
            got = http.GetJson(book_url)
            books[token_id] = {"url": book_url, "observed_at": got["observed_at"],
                               "body": got["body"] if got["ok"] else {"error": got.get("error") or "source_outage"}}
            fee_url = f"https://clob.polymarket.com/fee-rate?token_id={quote(token_id, safe='')}"
            fee = http.GetJson(fee_url)
            fees[token_id] = {"url": fee_url, "observed_at": fee["observed_at"],
                              "body": fee["body"] if fee["ok"] else {"error": fee.get("error") or "source_outage"}}
    return books, fees


def FetchScoreboard(http) -> dict:
    got = http.GetJson(NBA_SCOREBOARD_URL, NBA_HEADERS)
    if not got["ok"]:
        return {"error": got.get("error") or "source_outage", "observed_at": got.get("observed_at"), "body": None, "url": NBA_SCOREBOARD_URL}
    return {"error": None, "observed_at": got["observed_at"], "body": got["body"], "url": NBA_SCOREBOARD_URL}


def FetchKalshiMarkets(http) -> tuple:
    url = ("https://external-api.kalshi.com/trade-api/v2/markets"
           f"?series_ticker={KALSHI_SERIES}&status=open&limit=100")
    return _Pages(http, url, "markets", "cursor", "cursor")


def FetchKalshiBooks(http, markets) -> dict:
    books = {}
    for market in markets:
        url = market["orderbook_url"] + "?depth=5"
        got = http.GetJson(url)
        books[market["ticker"]] = {"url": url, "observed_at": got["observed_at"],
                                   "body": got["body"] if got["ok"] else {"error": got.get("error") or "source_outage"}}
    return books


def LoadBundle(game_date: str, http=None, pregame_model=None, live_sample: bool = False) -> dict:
    """Read public sources. Failures stay in the bundle; they are not retried as hidden endpoints."""
    http = http or ReadOnlyHttp()
    markets, poly_error = FetchPolymarketMarkets(http)
    kept = []
    if poly_error is None:
        for raw in markets:
            classified = ClassifyPolymarketMarket(raw, game_date)
            if classified["status"] == "keep":
                kept.append(classified["market"])
    books, fees = ({}, {})
    if poly_error is None:
        books, fees = FetchBooksAndFees(http, kept)
    scoreboard = FetchScoreboard(http)
    kalshi_raw, kalshi_error = FetchKalshiMarkets(http)
    kalshi_kept = []
    if kalshi_error is None:
        for raw in kalshi_raw:
            classified = ClassifyKalshiMarket(raw, game_date)
            if classified["status"] == "keep":
                kalshi_kept.append(classified["market"])
    return {
        "date": game_date,
        "live_sample": live_sample,
        "pregame_model": pregame_model,
        "polymarket_error": poly_error,
        "polymarket_markets": markets if poly_error is None else [],
        "books": books,
        "fees": fees,
        "scoreboard_error": scoreboard.get("error"),
        "scoreboard": scoreboard,
        "kalshi_error": kalshi_error,
        "kalshi_markets": kalshi_raw if kalshi_error is None else [],
        "kalshi_books": FetchKalshiBooks(http, kalshi_kept) if kalshi_error is None else {},
    }
