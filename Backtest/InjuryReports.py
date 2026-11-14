"""Parse official NBA injury-report PDFs and detect status changes."""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pdfplumber

ET = ZoneInfo("America/New_York")
STATUSES = ("Questionable", "Probable", "Doubtful", "Available", "Out")
STATUS_RE = re.compile(r"^(?:Questionable|Probable|Doubtful|Available|Out)$")
HEADER_RE = re.compile(
    r"Injury\s+Report:\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2})\s+(AM|PM)",
    re.I,
)
DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
TIME_RE = re.compile(r"^(\d{1,2}:\d{2})\(ET\)$")
MATCHUP_RE = re.compile(r"^[A-Z]{2,3}@[A-Z]{2,3}$")
PAGE_RE = re.compile(r"^Page\s*\d+\s*of\s*\d+$", re.I)
CREATED_RE = re.compile(r"D:(\d{14})([+-]\d{2})'(\d{2})'")

# Column cuts from header x positions on the landscape GemBox reports.
CUT_TIME = 100
CUT_MATCHUP = 185
CUT_TEAM = 254
CUT_PLAYER = 410
CUT_STATUS = 575
CUT_REASON = 658

TEAM_ABBREV = {
    "AtlantaHawks": "ATL", "BostonCeltics": "BOS", "BrooklynNets": "BKN",
    "CharlotteHornets": "CHA", "ChicagoBulls": "CHI", "ClevelandCavaliers": "CLE",
    "DallasMavericks": "DAL", "DenverNuggets": "DEN", "DetroitPistons": "DET",
    "GoldenStateWarriors": "GSW", "HoustonRockets": "HOU", "IndianaPacers": "IND",
    "LAClippers": "LAC", "LosAngelesClippers": "LAC", "LosAngelesLakers": "LAL",
    "MemphisGrizzlies": "MEM", "MiamiHeat": "MIA", "MilwaukeeBucks": "MIL",
    "MinnesotaTimberwolves": "MIN", "NewOrleansPelicans": "NOP", "NewYorkKnicks": "NYK",
    "OklahomaCityThunder": "OKC", "OrlandoMagic": "ORL", "Philadelphia76ers": "PHI",
    "PhoenixSuns": "PHX", "PortlandTrailBlazers": "POR", "SacramentoKings": "SAC",
    "SanAntonioSpurs": "SAS", "TorontoRaptors": "TOR", "UtahJazz": "UTA",
    "WashingtonWizards": "WAS",
}


def ColOf(x0: float) -> str:
    if x0 < CUT_TIME:
        return "date"
    if x0 < CUT_MATCHUP:
        return "time"
    if x0 < CUT_TEAM:
        return "matchup"
    if x0 < CUT_PLAYER:
        return "team"
    if x0 < CUT_STATUS:
        return "player"
    if x0 < CUT_REASON:
        return "status"
    return "reason"


def ParseHeaderStamp(text: str):
    m = HEADER_RE.search(text)
    if not m:
        return None
    day, hm, ampm = m.group(1), m.group(2), m.group(3)
    for fmt in ("%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p"):
        try:
            dt = datetime.strptime(f"{day} {hm} {ampm}", fmt)
            return dt.replace(tzinfo=ET)
        except ValueError:
            continue
    return None


def ParseCreation(meta: dict):
    raw = (meta or {}).get("CreationDate") or ""
    m = CREATED_RE.search(str(raw))
    if not m:
        return None
    ts, off_h, off_m = m.group(1), int(m.group(2)), int(m.group(3))
    dt = datetime.strptime(ts, "%Y%m%d%H%M%S")
    # store as ET wall clock: the offset is already in the stamp
    sign = 1 if off_h >= 0 else -1
    # interpret as aware UTC then convert to ET
    from datetime import timezone, timedelta
    tz = timezone(timedelta(hours=off_h, minutes=sign * off_m))
    return dt.replace(tzinfo=tz).astimezone(ET)


def ExtractWords(path: Path):
    words = []
    header_text = ""
    creation = None
    with pdfplumber.open(str(path)) as pdf:
        creation = ParseCreation(pdf.metadata)
        for page in pdf.pages:
            if not header_text:
                header_text = page.extract_text() or ""
            for w in page.extract_words() or []:
                words.append(w)
    return words, header_text, creation


def ClusterLines(words, y_tol=4.0):
    ordered = sorted(words, key=lambda w: (w.get("doctop", w["top"]), w["x0"]))
    lines = []
    for w in ordered:
        y = float(w.get("doctop", w["top"]))
        if lines and abs(y - lines[-1]["y"]) <= y_tol:
            lines[-1]["words"].append(w)
            lines[-1]["y"] = (lines[-1]["y"] * (len(lines[-1]["words"]) - 1) + y) / len(lines[-1]["words"])
        else:
            lines.append({"y": y, "words": [w]})
    return lines


def LineFields(line):
    buckets = defaultdict(list)
    for w in line["words"]:
        buckets[ColOf(w["x0"])].append(w["text"])
    return {k: "".join(v) if k in ("team", "player", "reason") else " ".join(v) for k, v in buckets.items()}


FOOTER_RE = re.compile(r"NOTYETSUBMITTED|NOT\s*YET\s*SUBMITTED|Page\d+of\d+", re.I)

def CleanReason(s: str) -> str:
    s = FOOTER_RE.sub(" ", s or "")
    return re.sub(r"\s+", " ", s).strip()


def CleanName(s: str) -> str:
    s = re.sub(r"\s+", " ", s.replace(",", ", ")).strip()
    s = re.sub(r",\s+", ", ", s)
    return s


def ParsePdf(path: Path) -> dict:
    words, header_text, creation = ExtractWords(path)
    header_ts = ParseHeaderStamp(header_text) or creation
    title = " ".join(header_text.split()[:8]) if header_text else ""
    rows = []
    game_date = game_time = matchup = team = None
    pending_reason = []  # (y, text) fragments between player rows
    current = None
    current_y = None

    def attach(rec, parts):
        extra = " ".join(parts).strip()
        if rec is not None and extra:
            rec["reason"] = CleanReason((rec.get("reason", "") + " " + extra))

    def flush(parts=None):
        nonlocal current, current_y
        attach(current, parts or [])
        if current is not None:
            rows.append(current)
        current = None
        current_y = None

    for line in ClusterLines(words):
        fields = LineFields(line)
        texts = " ".join(w["text"] for w in line["words"]).replace(" ", "")
        if PAGE_RE.match(texts) or texts.startswith("InjuryReport:") or texts.startswith("GameDate"):
            continue
        if "date" in fields and DATE_RE.match(fields["date"].replace(" ", "")):
            game_date = fields["date"].replace(" ", "")
        if "time" in fields and TIME_RE.match(fields["time"].replace(" ", "")):
            game_time = fields["time"].replace(" ", "")
        if "matchup" in fields and MATCHUP_RE.match(fields["matchup"].replace(" ", "")):
            matchup = fields["matchup"].replace(" ", "")
        if "team" in fields and fields["team"] in TEAM_ABBREV:
            team = fields["team"]
        status = fields.get("status")
        player = fields.get("player")
        reason = fields.get("reason")
        is_player = bool(status and STATUS_RE.match(status) and player)
        if is_player:
            y = line["y"]
            to_prev, to_new = [], []
            for y_r, text in pending_reason:
                if current is None or current_y is None or abs(y_r - y) <= abs(y_r - current_y):
                    to_new.append(text)
                else:
                    to_prev.append(text)
            pending_reason = []
            flush(to_prev)
            current_y = y
            current = {
                "game_date": game_date,
                "game_time": game_time,
                "matchup": matchup,
                "team_name": team,
                "team": TEAM_ABBREV.get(team),
                "player_raw": player,
                "player": CleanName(player),
                "status": status,
                "reason": CleanReason(" ".join(x for x in (to_new + ([reason] if reason else [])) if x)),
            }
        else:
            if reason:
                pending_reason.append((line["y"], reason))
    leftover = [text for _y, text in pending_reason]
    pending_reason = []
    flush(leftover)
    return {
        "path": str(path),
        "name": Path(path).name,
        "header_ts": header_ts,
        "creation_ts": creation,
        "title": title,
        "n_rows": len(rows),
        "rows": rows,
    }


def StatusKey(row) -> tuple:
    return (row.get("game_date"), row.get("matchup"), row.get("team"), row.get("player"))