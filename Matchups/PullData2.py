r"""
Second polite, resumable data pull for the v2 approaches. One request at a time, sleep between calls,
everything cached under data/matchups2/ (gitignored). Safe to stop and restart: finished files are skipped.

    .venv\Scripts\python.exe -m Matchups.PullData2 --season-level 2022 2023 2024 2025 2026 --games 2026 2025 2024

Season level (small): player bio (height/weight/age/usage), team rosters + coaches, Synergy play types
(team + player, offense + defense), tracking dashboards (speed/distance, drives, passing, touches,
catch-and-shoot, pull-ups, rebounding, possessions, defense), closest-defender dashboards, hustle, and
per-team shot charts (per-shot rows WITH game dates, so they can be used walk-forward).
Season aggregates without dates are only ever used as priors for LATER seasons.

Per game: GameRotation (stints, tenths of a second) and PlayByPlayV3 (trimmed columns).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

OUT = os.path.join("data", "matchups2")
SLEEP = 0.6
TIMEOUT = 60
PLAY_TYPES = ["Isolation", "Transition", "PRBallHandler", "PRRollMan", "Postup", "Spotup", "Handoff", "Cut",
              "OffScreen", "OffRebound", "Misc"]
PT_MEASURES = ["SpeedDistance", "Defense", "Drives", "Passing", "CatchShoot", "PullUpShot", "PostTouch",
               "PaintTouch", "Rebounding", "Possessions"]
DEFEND_CATS = ["Overall", "3 Pointers", "2 Pointers", "Less Than 6Ft", "Greater Than 15Ft"]
PBP_COLS = ["actionNumber", "clock", "period", "teamTricode", "personId", "playerName", "shotResult", "isFieldGoal",
            "scoreHome", "scoreAway", "actionType", "subType", "description", "shotValue", "shotDistance",
            "xLegacy", "yLegacy", "location"]
TEAM_IDS = {"ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766, "CHI": 1610612741,
            "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743, "DET": 1610612765, "GSW": 1610612744,
            "HOU": 1610612745, "IND": 1610612754, "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763,
            "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
            "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756, "POR": 1610612757,
            "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764}


def Label(season):
    return f"{season}-{str(season + 1)[-2:]}"


class Puller:
    def __init__(self, log_path):
        self.log_path, self.calls, self.failed, self.t0 = log_path, 0, 0, time.time()

    def Log(self, msg):
        line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def Fetch(self, path, make, transform=None, tries=3):
        """Call make() (an nba_api endpoint) unless path exists; save frames as JSON records."""
        if os.path.exists(path):
            return "cached"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for attempt in range(tries):
            try:
                frames = make().get_data_frames()
                data = transform(frames) if transform else [f.to_dict("records") for f in frames]
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(tmp, path)
                self.calls += 1
                time.sleep(SLEEP)
                return "ok"
            except Exception as ex:
                wait = 5 * (attempt + 1) ** 2
                self.Log(f"retry {attempt + 1} {os.path.basename(path)} after {wait}s: {repr(ex)[:140]}")
                time.sleep(wait)
        self.failed += 1
        with open(path + ".failed", "w") as f:
            f.write("failed")
        return "failed"


def SeasonLevel(p: Puller, season: int):
    from nba_api.stats import endpoints as E
    s, base = Label(season), os.path.join(OUT, "season", str(season))
    p.Fetch(os.path.join(base, "player_bio.json"), lambda: E.LeagueDashPlayerBioStats(season=s, timeout=TIMEOUT))
    for kind in ("P", "T"):
        for grouping in ("offensive", "defensive"):
            for pt in PLAY_TYPES:
                p.Fetch(os.path.join(base, "synergy", f"{kind}_{grouping}_{pt}.json"),
                        lambda pt=pt, grouping=grouping, kind=kind: E.SynergyPlayTypes(
                            season=s, play_type_nullable=pt, player_or_team_abbreviation=kind,
                            type_grouping_nullable=grouping, season_type_all_star="Regular Season", timeout=TIMEOUT))
    for who in ("Player", "Team"):
        for m in PT_MEASURES:
            p.Fetch(os.path.join(base, "tracking", f"{who}_{m}.json"),
                    lambda m=m, who=who: E.LeagueDashPtStats(season=s, pt_measure_type=m, player_or_team=who,
                                                             per_mode_simple="Totals", timeout=TIMEOUT))
    for cat in DEFEND_CATS:
        p.Fetch(os.path.join(base, "defend", f"{cat.replace(' ', '_')}.json"),
                lambda cat=cat: E.LeagueDashPtDefend(season=s, defense_category=cat, per_mode_simple="Totals", timeout=TIMEOUT))
    p.Fetch(os.path.join(base, "hustle_player.json"), lambda: E.LeagueHustleStatsPlayer(season=s, per_mode_time="Totals", timeout=TIMEOUT))
    p.Fetch(os.path.join(base, "hustle_team.json"), lambda: E.LeagueHustleStatsTeam(season=s, per_mode_time="Totals", timeout=TIMEOUT))
    for abbr, tid in TEAM_IDS.items():
        p.Fetch(os.path.join(base, "roster", f"{abbr}.json"), lambda tid=tid: E.CommonTeamRoster(team_id=tid, season=s, timeout=TIMEOUT))
    if season >= 2023:
        keep = ["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "TEAM_ID", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING",
                "ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_DISTANCE", "LOC_X", "LOC_Y",
                "SHOT_MADE_FLAG", "GAME_DATE", "HTM", "VTM"]
        for abbr, tid in TEAM_IDS.items():
            p.Fetch(os.path.join(base, "shots", f"{abbr}.json"),
                    lambda tid=tid: E.ShotChartDetail(team_id=tid, player_id=0, season_nullable=s,
                                                      season_type_all_star="Regular Season", context_measure_simple="FGA",
                                                      timeout=120),
                    transform=lambda fr: fr[0][[c for c in keep if c in fr[0].columns]].to_dict("records"))
    p.Log(f"season-level {season} done (calls so far {p.calls}, failed {p.failed})")


def GameIds(season):
    games = json.load(open(os.path.join("data", "season", f"games_{season}.json"), encoding="utf-8"))
    return sorted(g["game_id"] for g in games if g["game_id"].startswith("002") and g.get("home_score") is not None)


def PerGame(p: Puller, season: int):
    from nba_api.stats import endpoints as E
    ids = GameIds(season)
    t0, done = time.time(), 0
    p.Log(f"per-game {season}: {len(ids)} games")
    for gid in ids:
        r1 = p.Fetch(os.path.join(OUT, "rotation", str(season), f"{gid}.json"),
                     lambda: E.GameRotation(game_id=gid, timeout=TIMEOUT),
                     transform=lambda fr: {"home": fr[1].to_dict("records"), "away": fr[0].to_dict("records")})
        r2 = p.Fetch(os.path.join(OUT, "pbp", str(season), f"{gid}.json"),
                     lambda: E.PlayByPlayV3(game_id=gid, timeout=TIMEOUT),
                     transform=lambda fr: fr[0][[c for c in PBP_COLS if c in fr[0].columns]].to_dict("records"))
        done += 1
        if done % 100 == 0:
            rate = (time.time() - t0) / done
            p.Log(f"per-game {season}: {done}/{len(ids)} ({rate:.2f}s/game, eta {rate * (len(ids) - done) / 60:.0f} min) last {r1}/{r2}")
    p.Log(f"per-game {season} done in {time.time() - t0:.0f}s (calls so far {p.calls}, failed {p.failed})")


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-level", type=int, nargs="*", default=[])
    ap.add_argument("--games", type=int, nargs="*", default=[])
    args = ap.parse_args()
    try:
        from Matchups.Cpu import LowerPriority
        LowerPriority()
    except Exception:
        pass
    os.makedirs(OUT, exist_ok=True)
    p = Puller(os.path.join("logs", f"{datetime.now():%Y-%m-%d}_matchups_pull2.log"))
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    p.Log(f"START python {sys.version.split()[0]} head {head} args {vars(args)}")
    for s in args.season_level:
        SeasonLevel(p, s)
    for s in args.games:
        PerGame(p, s)
    p.Log(f"END {time.time() - p.t0:.0f}s calls {p.calls} failed {p.failed}")


if __name__ == "__main__":
    Main()
