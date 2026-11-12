r"""
Leak-free stack_global forecasts for PredictDay.

Walks every non-oracle v3 base approach (plus the phase-1 Elo replica) through completed
games dated strictly before the target day, refits the global mixture weights on that
history with the same weekly schedule and priors as Matchups.V3.Post.ContextStack
(contextual=False), and mixes the members' pregame forecasts for the day's slate.

No EMOS or temperature is applied: v3 selected plain stack_global, and recalibration did
not help the good models. Default action is PASS unless a market ask is present and the
fee-adjusted edge (model_prob - ask - Tim's $0.02 estimate) is strictly positive.
"""

from datetime import date
from itertools import groupby

from Matchups import Cpu

Cpu.LimitThreads()

import Matchups.V2.Approaches  # noqa: F401
from Backtest.Model import EloModel, SeasonFinalRatings
from Backtest.Odds import FEE_BASIS, FEE_PER_CONTRACT, Edge
from Matchups import Data
from Matchups.V2.Base import REGISTRY, LoadGames, PregameView, RealRecords, RosterBook, ScheduleBook
from Matchups.V3 import Post as PP
from Matchups.V3.Evaluate import EloRows, InvPhi

SKIP_MEMBERS = {"coach_profile"}
RECORD_FROM = "2024-11-15"
MISSING = {"margin_mean": 0.0, "margin_var": 169.0, "p_home": 0.5}


def MemberNames():
    """Non-oracle registered approaches stacked in v3, excluding coach_profile."""
    return sorted(n for n in REGISTRY if n not in SKIP_MEMBERS)


def SeasonsFor(game_date: str, season: int) -> list:
    """Three-season window used by the v3 walk (warm-up, selection, current)."""
    return [season - 2, season - 1, season]


def FillMember(out: dict) -> dict:
    if not out:
        return dict(MISSING)
    if "margin_mean" not in out or not out.get("margin_var"):
        p = float(out.get("p_home", 0.5))
        p = min(max(p, 1e-6), 1 - 1e-6)
        return {"margin_mean": 13.0 * InvPhi(p), "margin_var": 169.0, "p_home": p}
    return {"margin_mean": float(out["margin_mean"]), "margin_var": float(out["margin_var"]),
            "p_home": float(out.get("p_home", 0.5))}


def WalkMembers(approaches: dict, games: list, records=RealRecords, record_from: str = RECORD_FROM,
                progress=None) -> dict:
    """One leak-free day loop shared by every member (records loaded once per game)."""
    rosters, schedule = RosterBook(), ScheduleBook()
    hist = {name: [] for name in approaches}
    last_day, n_days = None, 0
    for day, day_games in groupby(games, key=lambda g: g["game_date"]):
        day_games = list(day_games)
        gap = (date.fromisoformat(day) - date.fromisoformat(last_day)).days if last_day else 0
        for approach in approaches.values():
            approach.BeforeDay(day, gap)
        if record_from is None or day >= record_from:
            for g in day_games:
                view = PregameView(g, rosters, schedule)
                base = {"game_id": g["game_id"], "game_date": day, "season": g["season"],
                        "home": g["home_team"], "away": g["away_team"],
                        "margin": g["home_score"] - g["away_score"],
                        "home_win": int(g["home_score"] > g["away_score"])}
                for name, approach in approaches.items():
                    row = dict(base)
                    row.update(FillMember(approach.Predict(view)))
                    hist[name].append(row)
        recs = [records(g) for g in day_games]
        for approach in approaches.values():
            for rec in recs:
                approach.Update(rec)
        for rec in recs:
            rosters.Apply(rec.box)
            schedule.Apply(rec.game)
        last_day = day
        n_days += 1
        if progress and n_days % 30 == 0:
            progress(f"stack members: {day} ({n_days} days)")
    return {"rows": hist, "rosters": rosters, "schedule": schedule, "last_day": last_day}


def EloHistory(seasons, record_from: str, until: str) -> dict:
    """elo_phase1 rows (implied Normal margin) for completed games in [record_from, until)."""
    grow, probs = [], {}
    for s in seasons:
        prior = Data.AllGames(s - 1)
        model = EloModel(SeasonFinalRatings(prior) if prior else None)
        games = sorted((g for g in Data.AllGames(s) if g["game_date"] < until),
                       key=lambda g: (g["game_date"], g["game_id"]))
        for _d, day in groupby(games, key=lambda g: g["game_date"]):
            day = list(day)
            for g in day:
                probs[g["game_id"]] = model.HomeWinProb(g["home_team"], g["away_team"])
            for g in day:
                model.Apply(g)
        for g in games:
            if record_from is None or g["game_date"] >= record_from:
                rec = {"game_id": g["game_id"], "game_date": g["game_date"], "season": s,
                       "home": g["home_team"], "away": g["away_team"],
                       "margin": g["home_score"] - g["away_score"],
                       "home_win": int(g["home_score"] > g["away_score"])}
                grow.append(rec)
        last_model = model
    rows = EloRows(grow, probs)
    return {"rows": rows, "model": last_model, "probs": probs}


def ForecastSlate(approaches: dict, rosters: RosterBook, schedule: ScheduleBook,
                  elo: EloModel, slate: list, day: str, last_day: str) -> list:
    """Pregame member forecasts for unplayed slate games. Does not Update on results."""
    gap = (date.fromisoformat(day) - date.fromisoformat(last_day)).days if last_day else 0
    for approach in approaches.values():
        approach.BeforeDay(day, gap)
    out = []
    for game in slate:
        view = PregameView(game, rosters, schedule)
        members = {name: FillMember(approach.Predict(view)) for name, approach in approaches.items()}
        p = elo.HomeWinProb(game["home_team"], game["away_team"])
        members["elo_phase1"] = {"margin_mean": 13.0 * InvPhi(p), "margin_var": 169.0, "p_home": float(p)}
        out.append({"game": game, "members": members})
    return out


def SideFromHome(home_prob: float, home_team: str, away_team: str) -> tuple:
    if home_prob >= 0.5:
        return "home", home_team, float(home_prob)
    return "away", away_team, float(1.0 - home_prob)


def DecideAction(model_prob, market_price, fee: float = FEE_PER_CONTRACT) -> dict:
    """PASS unless a price exists and model_prob clears price + Tim's fee estimate."""
    if market_price is None:
        return {"action": "PASS", "edge": None, "reason": "no prices",
                "fee": fee, "fee_basis": FEE_BASIS}
    edge = Edge(model_prob, market_price, fee=fee)
    if edge > 0:
        return {"action": "BET", "edge": float(edge), "reason": "fee_adjusted_edge",
                "fee": fee, "fee_basis": FEE_BASIS}
    return {"action": "PASS", "edge": float(edge), "reason": "edge_not_positive",
            "fee": fee, "fee_basis": FEE_BASIS}


def PredictSlate(game_date: str, season: int, slate: list, seasons=None, record_from: str = RECORD_FROM,
                 records=RealRecords, games=None, approaches=None, progress=None,
                 elo_rows=None, elo_model=None) -> dict:
    """Fit stack_global on games before game_date and score the slate. Leak-free by construction."""
    Cpu.LowerPriority()
    Cpu.LimitThreads()
    seasons = seasons or SeasonsFor(game_date, season)
    if approaches is None:
        approaches = {name: REGISTRY[name]() for name in MemberNames()}
    if games is None:
        games = [g for g in LoadGames(seasons) if g["game_date"] < game_date]
    walked = WalkMembers(approaches, games, records=records, record_from=record_from, progress=progress)
    if elo_rows is None or elo_model is None:
        elo = EloHistory(seasons, record_from, game_date)
        if elo_rows is not None:
            elo["rows"] = elo_rows
        if elo_model is not None:
            elo["model"] = elo_model
    else:
        elo = {"rows": elo_rows, "model": elo_model}
    member_rows = dict(walked["rows"])
    member_rows["elo_phase1"] = elo["rows"]
    stack_names, w_glob, _w_ctx = PP.WeightsForDay(member_rows, game_date, contextual=False)
    for game in slate:
        game.setdefault("season", season)
        game.setdefault("game_date", game_date)
    forecasts = ForecastSlate(approaches, walked["rosters"], walked["schedule"],
                              elo["model"], slate, game_date, walked["last_day"])
    rows = []
    for item in forecasts:
        game = item["game"]
        mixed = PP.MixForecast(item["members"], stack_names, w_glob)
        kalman = item["members"].get("kalman_team", MISSING)
        elo_m = item["members"]["elo_phase1"]
        stack_side, stack_pick, stack_prob = SideFromHome(mixed["p_home"], game["home_team"], game["away_team"])
        k_side, k_pick, k_prob = SideFromHome(kalman["p_home"], game["home_team"], game["away_team"])
        e_side, e_pick, e_prob = SideFromHome(elo_m["p_home"], game["home_team"], game["away_team"])
        market = game.get("market_price")
        action = DecideAction(stack_prob, market)
        rows.append({
            "game_id": game.get("game_id"),
            "game_date": game_date,
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "stack_global_home_prob": round(float(mixed["p_home"]), 4),
            "kalman_team_home_prob": round(float(kalman["p_home"]), 4),
            "elo_predictday_v1_home_prob": round(float(elo_m["p_home"]), 4),
            "recommended_side": stack_side,
            "recommended": stack_pick,
            "recommended_prob": round(stack_prob, 4),
            "kalman_team_pick": k_pick,
            "kalman_team_prob": round(k_prob, 4),
            "elo_predictday_v1_pick": e_pick,
            "elo_predictday_v1_prob": round(e_prob, 4),
            "margin_mean": mixed["margin_mean"],
            "margin_var": mixed["margin_var"],
            **action,
        })
    played = games
    return {
        "rows": rows,
        "names": stack_names,
        "weights": [round(float(x), 4) for x in w_glob],
        "data_through": played[-1]["game_date"] if played else None,
        "games_used": len(played),
        "seasons": seasons,
    }