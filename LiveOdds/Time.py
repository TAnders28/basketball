"""UTC instants internally, Eastern presentation. Never treat retrieval time as a source timestamp."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

# Server clocks a few seconds ahead of ours are skew, not a future quote.
MAX_FUTURE_SKEW_SECONDS = 5
# In-play books must carry a snapshot this fresh. Pregame is looser; final games are not executable.
MAX_LIVE_QUOTE_AGE_SECONDS = 30
MAX_PREGAME_QUOTE_AGE_SECONDS = 180
MAX_GAME_STATE_AGE_SECONDS = 45
# Our fail-closed window for calling a quote and a score the same moment. Not an NBA SLA.
MAX_QUOTE_STATE_SKEW_SECONDS = 30
# Two outcome books are one snapshot only when their source timestamps agree this closely.
MAX_BOOK_TIMESTAMP_GAP_SECONDS = 2


def FormatUtc(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def FormatEt(moment: datetime) -> str:
    return moment.astimezone(EASTERN).strftime("%Y-%m-%d %H:%M:%S ET")


def ParseInstant(value):
    """Return (utc datetime or None, kind).

    kind is one of: missing, invalid, naive, unix_ms, unix_s, iso.
    Naive strings are not assigned a zone here.
    """
    if value is None or value == "":
        return None, "missing"
    if isinstance(value, bool):
        return None, "invalid"
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().isdigit()):
        number = int(value)
        if number >= 10 ** 12:
            return datetime.fromtimestamp(number / 1000.0, timezone.utc), "unix_ms"
        if number >= 10 ** 9:
            return datetime.fromtimestamp(number, timezone.utc), "unix_s"
        return None, "invalid"
    if not isinstance(value, str):
        return None, "invalid"
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if len(text) >= 3 and text[-3] in "+-" and text[-2:].isdigit():
        text = text + ":00"
    if "T" not in text and " " in text:
        text = text.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None, "invalid"
    if parsed.tzinfo is None:
        return None, "naive"
    return parsed.astimezone(timezone.utc), "iso"


def Freshness(source_time, observed: datetime, max_age_seconds: float) -> str:
    """fresh, stale, clock_skew, or unknown. Unknown when the source did not give a time."""
    if source_time is None:
        return "unknown"
    age = (observed - source_time).total_seconds()
    if age < -MAX_FUTURE_SKEW_SECONDS:
        return "clock_skew"
    if age > max_age_seconds:
        return "stale"
    return "fresh"


def QuoteMaxAge(phase: str) -> float:
    if phase == "scheduled":
        return MAX_PREGAME_QUOTE_AGE_SECONDS
    return MAX_LIVE_QUOTE_AGE_SECONDS


def ResolveNaiveWallClock(raw: str, observed: datetime):
    """Use a scoreboard clock only when it carries an explicit offset.

    NBA `meta.time` is a naive wall clock. It is not treated as UTC or Eastern,
    even when one of those readings sits near the observation. A 12-hour shift
    is recorded as a diagnostic and is not applied.
    """
    if raw is None or str(raw).strip() == "":
        return {"raw": raw, "utc": None, "et": None, "freshness": "unknown",
                "timezone_resolution": "missing"}
    text = str(raw).strip()
    if "T" not in text and " " in text:
        text = text.replace(" ", "T", 1)
    try:
        naive = datetime.fromisoformat(text)
    except ValueError:
        return {"raw": raw, "utc": None, "et": None, "freshness": "unknown",
                "timezone_resolution": "invalid"}
    if naive.tzinfo is not None:
        utc = naive.astimezone(timezone.utc)
        return {"raw": raw, "utc": utc, "et": FormatEt(utc),
                "freshness": Freshness(utc, observed, MAX_GAME_STATE_AGE_SECONDS),
                "timezone_resolution": "explicit_offset"}
    as_eastern = naive.replace(tzinfo=EASTERN).astimezone(timezone.utc)
    as_utc = naive.replace(tzinfo=timezone.utc)
    as_eastern_plus_12 = (naive + timedelta(hours=12)).replace(tzinfo=EASTERN).astimezone(timezone.utc)
    return {
        "raw": raw, "utc": None, "et": None, "freshness": "unknown",
        "timezone_resolution": "naive_no_offset",
        "delta_seconds_if_eastern_not_used": round(abs((observed - as_eastern).total_seconds()), 3),
        "delta_seconds_if_utc_not_used": round(abs((observed - as_utc).total_seconds()), 3),
        "delta_seconds_if_eastern_plus_12h_not_used": round(abs((observed - as_eastern_plus_12).total_seconds()), 3),
        "note": "No offset on the source clock. Zone was not inferred from the observation.",
    }


def EtDate(moment: datetime) -> str:
    return moment.astimezone(EASTERN).strftime("%Y-%m-%d")


def AgeSeconds(source_time, observed: datetime):
    if source_time is None:
        return None
    return round((observed - source_time).total_seconds(), 3)


def Earlier(left, right):
    if left is None or right is None:
        return None
    return left if left <= right else right


def GapSeconds(left, right):
    if left is None or right is None:
        return None
    return abs((left - right).total_seconds())
