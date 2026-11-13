r"""Pull ESPN substitutions and box scores (participants) for the Kalshi trade-tape games. Resumable, read-only.

    .venv\Scripts\python.exe -m InPlay.PullSubs
"""

import os
import sys

import pandas as pd

from InPlay.Http import GetJson, HttpError, WriteJsonAtomic
from InPlay.PullEspn import SITE
from InPlay.TradeTape import TR

ROOT = os.path.join("data", "inplay", "espn", "subs")


def Compact(body):
    plays = []
    for p in body.get("plays", []):
        ids = [x.get("athlete", {}).get("id") for x in p.get("participants", []) or []]
        plays.append(dict(seq=int(p.get("sequenceNumber", 0)), per=p.get("period", {}).get("number"), clock=p.get("clock", {}).get("displayValue"),
                          wall=p.get("wallclock"), type=p.get("type", {}).get("text"), team=(p.get("team") or {}).get("id"), ids=ids,
                          text=p.get("text") if "ubstitut" in (p.get("type", {}).get("text") or "") or "foul" in (p.get("type", {}).get("text") or "").lower() else None))
    box = []
    for tm in body.get("boxscore", {}).get("players", []):
        tid = tm.get("team", {}).get("id")
        for st in tm.get("statistics", [])[:1]:
            names = st.get("names") or st.get("labels")
            for a in st.get("athletes", []):
                if not (a.get("athlete") or {}).get("id"):
                    continue
                s = dict(zip(names, a.get("stats") or []))
                box.append(dict(team=tid, id=a["athlete"]["id"], name=a["athlete"].get("displayName"), starter=a.get("starter"),
                                min=s.get("MIN"), pts=s.get("PTS"), pf=s.get("PF"), dnp=not a.get("stats")))
    return dict(plays=plays, box=box)


def Main(argv=None):
    meta = pd.read_pickle(os.path.join("data", "inplay", "games.pkl")).dropna(subset=["kalshi_ticker"])
    gids = [str(g) for g, t in zip(meta.game_id, meta.kalshi_ticker) if os.path.exists(os.path.join(TR, t + ".json"))]
    print(len(gids), "games", flush=True)
    done = err = 0
    for k, gid in enumerate(gids):
        p = os.path.join(ROOT, gid + ".json")
        if os.path.exists(p):
            continue
        try:
            WriteJsonAtomic(p, Compact(GetJson(SITE + "/summary", params={"event": gid})))
            done += 1
        except (HttpError, KeyError, TypeError) as exc:
            err += 1
            print("ERR", gid, exc, flush=True)
        if (k + 1) % 50 == 0:
            print(f"{k+1}/{len(gids)} done={done} err={err}", flush=True)
    print(f"DONE done={done} err={err}", flush=True)


if __name__ == "__main__":
    sys.exit(Main())

