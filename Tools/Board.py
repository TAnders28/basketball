"""GutCall board feed: python -m Tools.Board --date YYYY-MM-DD

Writes E:\\AgentDocs\\Basketbot\\_dashboard\\board_latest.json (atomic temp-file + rename) for Degen's GutCall
dashboard, plus a dated copy board_<date>.json. Schema per the Degen brief (no E:\\AgentWorkspaces\\GutCall\\gutcall\\SCHEMA.md
existed on 2026-11-14). Edges are (prob - ask - 0.02) x 100 in cents, where 0.02 is Tim's fee estimate. Every
model is marked tested=false: none has a validated holdout edge. Unknowns are null, never invented.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Tools import Data
from Tools.Data import EdgeCents, IsoEt, NowEt

DASH = Path(r"E:\AgentDocs\Basketbot\_dashboard")
GUTCALL_SCHEMA = Path(r"E:\AgentWorkspaces\GutCall\gutcall\SCHEMA.md")
PICK_DIRS = [DASH / "picks", Data.ROOT / "data" / "backtest" / "predictions"]


def LoadModel(day):
    for d in PICK_DIRS:
        p = d / f"picks_{day}.json"
        if p.exists():
            try:
                body = json.loads(p.read_text(encoding="utf-8"))
                out = {}
                for r in body.get("games", []):
                    prob = r.get("stack_global_home_prob", r.get("model_home_prob"))
                    out[(Data.Std(r.get("home_team")), Data.Std(r.get("away_team")))] = prob
                return out, {"path": str(p), "data_through": body.get("data_through"), "model_version": body.get("model_version")}
            except Exception as ex:
                Data.errors.append(f"picks {p}: {ex}")
    return {}, None


def Status(e):
    st = e.get("state")
    return {"pre": "scheduled", "in": "in_progress", "post": "final"}.get(st, st) if st else None


def BuildBoard(day, bundle, model, model_meta):
    games = []
    for g in bundle["games"]:
        e, home, away, k = g["espn"], g["home"], g["away"], g["kalshi"] or {}
        hrow, arow = k.get("home") or {}, k.get("away") or {}
        h_ask, a_ask = hrow.get("yes_ask"), arow.get("yes_ask")
        prob = model.get((home, away))
        mv = g.get("move_home") or {}
        score = None
        if e.get("state") in ("in", "post"):
            score = {"home": e["home_info"].get("score"), "away": e["away_info"].get("score"),
                     "period": e.get("period"), "clock": e.get("clock"), "detail": e.get("detail")}
        inj = g["injuries_official"] or []
        inj_src = "NBA official injury report PDF" if inj else ("ESPN summary injuries" if g["injuries_espn"] else None)
        inj = inj or [x for x in g["injuries_espn"] if x["team"] in (home, away)]
        notes = ["Context, not an edge. No model here has a validated holdout edge; tested=false everywhere."]
        if k:
            notes.append(f"Kalshi no-vig fair (from asks): {home} {Data.Money(k.get('home_fair_ask'))}, {away} {Data.Money(k.get('away_fair_ask'))}; "
                         f"asks sum to {None if k.get('overround_asks') is None else round(1 + k['overround_asks'], 2)}.")
        if g["book"]:
            b = g["book"]
            notes.append(f"{b['provider']} via ESPN: {home} {int(b['home_ml']):+d} / {away} {int(b['away_ml']):+d}; no-vig {home} {Data.Money(b['home_fair'])}, "
                         f"{away} {Data.Money(b['away_fair'])}; spread {b['spread']}, total {b['total']} (as of {b['as_of']}).")
        if g["predictor"]:
            notes.append(f"ESPN Matchup Predictor (BPI): {home} {g['predictor']['home_win_prob']:.1%} (untested here, context only).")
        if mv.get("first_price") is not None:
            notes.append(f"Kalshi {home} yes moved {mv['move_c']:+.0f}c since first hourly candle {mv['first_t_et']} ({mv['hours_back']} h back; not necessarily the true open).")
        if inj_src:
            notes.append(f"Injuries from {inj_src}" + (f", report slot {bundle['injury_report']['report_slot_et']}" if inj_src.startswith("NBA") and bundle.get("injury_report") else "") + ".")
        if prob is not None and k.get("home_fair_ask") is not None:
            gap = (float(prob) - k["home_fair_ask"]) * 100
            if abs(gap) >= 10:
                notes.append(f"Large gap: untested stack_global has {home} {float(prob):.0%} vs market fair {k['home_fair_ask']:.0%} ({gap:+.0f} pts). "
                             "The model has no injury/rotation input and its data stop at " + str((model_meta or {}).get("data_through")) +
                             "; in testing the market was more accurate, so read this as the model missing news, not as value.")
        notes += g["trends"]
        props = []
        for p in g["props"]:
            props.append({"player": p.get("player"), "market": p.get("stat"), "line": p.get("k"), "yes_ask": p.get("yes_ask"),
                          "yes_bid": p.get("yes_bid"), "ticker": p.get("ticker"), "model_prob": None, "edge_c": None, "tested": False})
        fh, fa = g["form"].get(home), g["form"].get(away)
        games.append({
            "game_id": e.get("espn_id"), "kalshi_event": k.get("event_ticker"), "start_time": e.get("start"),
            "home": home, "away": away, "status": Status(e), "status_detail": e.get("detail"), "score": score,
            "market": {"home_yes_ask": h_ask, "away_yes_ask": a_ask,
                       "home_yes_bid": hrow.get("yes_bid"), "away_yes_bid": arow.get("yes_bid"),
                       "home_fair_novig": k.get("home_fair_ask"), "away_fair_novig": k.get("away_fair_ask"),
                       "open_home_yes": mv.get("first_price"), "open_as_of": mv.get("first_t_et"),
                       "source": k.get("source"), "as_of": k.get("as_of")},
            "book": g["book"],
            "model": {"home_win_prob": None if prob is None else round(float(prob), 4),
                      "model_name": None if prob is None else "stack_global (Basketbot PredictDay)",
                      "tested": False, "data_through": None if not model_meta else model_meta.get("data_through")},
            "edge_home_c": EdgeCents(None if prob is None else float(prob), h_ask),
            "edge_away_c": EdgeCents(None if prob is None else 1 - float(prob), a_ask),
            "context": {
                "injuries": [{"team": x["team"], "player": x["player"], "status": x["status"], "note": x.get("note")} for x in inj],
                "injury_source": inj_src,
                "recent_form": {home: fh, away: fa},
                "rest_days": {home: None if not fh else fh["rest_days"], away: None if not fa else fa["rest_days"]},
                "head_to_head": g["h2h"] or g["season_series"] or [],
                "espn_predictor_home": None if not g["predictor"] else g["predictor"]["home_win_prob"],
                "notes": notes,
            },
            "props": props,
        })
    return {"sport": "NBA", "date": str(day), "generated_at": NowEt().isoformat(timespec="seconds"),
            "schema": "Degen brief 2026-11-14 (no GutCall SCHEMA.md present)" if not GUTCALL_SCHEMA.exists() else str(GUTCALL_SCHEMA),
            "fee_basis": Data.FEE_LABEL, "edge_definition": "(prob - ask - 0.02) x 100 cents; arithmetic only, never a validated edge",
            "model_source": model_meta, "injury_report": bundle.get("injury_report"),
            "sources": ["Kalshi public API (KXNBAGAME + KXNBAPTS/REB/AST/3PT/PRA markets, hourly candlesticks)",
                        "ESPN public site API (scoreboard, summary: pickcenter, injuries, predictor)",
                        "NBA official injury report PDFs (official.nba.com/referee/injury)",
                        "Local games data (data/season/games_*.json)" + (f", through {bundle['games_info']['local_through']}" if bundle.get("games_info") else "")],
            "games": games, "errors": bundle.get("errors", [])}


def Main(argv=None):
    ap = argparse.ArgumentParser(description="Write the GutCall board JSON")
    ap.add_argument("--date", default=NowEt().strftime("%Y-%m-%d"))
    ap.add_argument("--out", default=str(DASH / "board_latest.json"))
    ap.add_argument("--no-props", action="store_true")
    a = ap.parse_args(argv)
    bundle = Data.SlateBundle(a.date, with_props=not a.no_props)
    model, meta = LoadModel(a.date)
    board = BuildBoard(a.date, bundle, model, meta)
    text = json.dumps(board, indent=1, default=str)
    Data.WriteAtomic(a.out, text)
    Data.WriteAtomic(Path(a.out).with_name(f"board_{a.date}.json"), text)
    print(f"wrote {a.out} games={len(board['games'])} props={sum(len(g['props']) for g in board['games'])} "
          f"model={'yes' if model else 'none'} errors={len(board['errors'])} at {board['generated_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())