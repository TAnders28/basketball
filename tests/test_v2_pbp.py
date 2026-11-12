"""Play-by-play parser tests: a hand-built sequence (exact expectations) plus consistency checks on a few cached
real games (skipped when the per-game pull has not cached them)."""

import glob
import os
import unittest

from Matchups.V2 import Pbp


def Ev(n, period, clock, team, pid, name, typ, sub="", desc="", result="", value=0, sh="", sa="", loc=""):
    return {"actionNumber": n, "clock": clock, "period": period, "teamTricode": team, "personId": pid,
            "playerName": name, "shotResult": result, "scoreHome": sh, "scoreAway": sa, "actionType": typ,
            "subType": sub, "description": desc, "shotValue": value, "shotDistance": 0, "location": loc}


HOME, AWAY = "HHH", "AAA"
ROT = {"home": [{"PERSON_ID": p, "IN_TIME_REAL": 0, "OUT_TIME_REAL": 28800} for p in range(1, 6)],
       "away": [{"PERSON_ID": p, "IN_TIME_REAL": 0, "OUT_TIME_REAL": 14400} for p in range(11, 16)]
       + [{"PERSON_ID": 16, "IN_TIME_REAL": 14400, "OUT_TIME_REAL": 28800}]
       + [{"PERSON_ID": p, "IN_TIME_REAL": 14400, "OUT_TIME_REAL": 28800} for p in range(12, 16)]}
PBP = [
    Ev(1, 1, "PT11M50.00S", HOME, 1, "Ann", "Made Shot", "Jump Shot", "Ann 3PT (3 PTS) (Bob 1 AST)", "Made", 3, "3", "0"),
    Ev(2, 1, "PT11M40.00S", AWAY, 11, "Kim", "Missed Shot", "Layup", "MISS Kim Layup", "Missed", 2),
    Ev(3, 1, "PT11M39.00S", HOME, 2, "Bob", "Rebound", desc="Bob REBOUND"),
    Ev(4, 1, "PT11M35.00S", HOME, 2, "Bob", "Made Shot", "Dunk", "Bob Dunk (2 PTS)", "Made", 2, "5", "0"),
    Ev(5, 1, "PT11M20.00S", AWAY, 11, "Kim", "Turnover", "Bad Pass", "Kim Bad Pass TURNOVER (1 TO) Ann STEAL"),
    Ev(6, 1, "PT11M15.00S", "", 1610612700, "", "Timeout", "Regular", "HHH Timeout: Regular", loc="h"),
    Ev(7, 1, "PT11M00.00S", HOME, 3, "Cat", "Missed Shot", "Jump Shot", "MISS Cat Jumper", "Missed", 2),
    Ev(8, 1, "PT10M59.00S", "", 1610612701, "", "Rebound", desc="AAA Rebound", loc="v"),
    Ev(9, 1, "PT10M40.00S", AWAY, 12, "Lee", "Free Throw", "Free Throw 1 of 2", "Lee Free Throw 1 of 2 (1 PTS)", sh="5", sa="1"),
    Ev(10, 1, "PT10M40.00S", AWAY, 12, "Lee", "Free Throw", "Free Throw 2 of 2", "MISS Lee Free Throw 2 of 2"),
    Ev(11, 1, "PT10M39.00S", AWAY, 13, "Mo", "Rebound", desc="Mo REBOUND"),
    Ev(12, 1, "PT10M30.00S", AWAY, 13, "Mo", "Made Shot", "Layup", "Mo Layup (2 PTS)", "Made", 2, "5", "3"),
]


class TestHandBuilt(unittest.TestCase):
    def setUp(self):
        self.poss = Pbp.BuildPossessions(PBP, ROT, HOME, AWAY)

    def test_elapsed(self):
        self.assertEqual(Pbp.Elapsed(1, "PT12M00.00S"), 0)
        self.assertEqual(Pbp.Elapsed(2, "PT12M00.00S"), 7200)
        self.assertEqual(Pbp.Elapsed(5, "PT00M00.00S"), 28800 + 3000)
        self.assertEqual(Pbp.GameLength(6), 28800 + 6000)

    def test_possession_chain(self):
        p = self.poss
        self.assertEqual([x["offense"] for x in p], [HOME, AWAY, HOME, AWAY, HOME, AWAY])
        self.assertEqual([x["pts"] for x in p], [3, 0, 2, 0, 0, 3])
        self.assertEqual(p[2]["oreb"], 0)                      # HOME's possession #2 started on its defensive rebound
        self.assertEqual(p[5]["oreb"], 1)                      # Mo's offensive rebound after the missed FT
        self.assertEqual(p[5]["start_type"], "after_dreb")     # team rebound (location v) ended HOME's trip
        self.assertEqual(p[5]["fta"], 2)
        self.assertEqual(p[0]["assister"], 2)                  # "(Bob 1 AST)" -> Bob's id
        self.assertEqual(p[2]["start_type"], "after_dreb")
        self.assertTrue(p[2]["transition"])                    # shot 4 tenths*10 after the rebound
        self.assertEqual(sum(x["tov"] for x in p), 1)
        self.assertEqual(p[0]["off5"], (1, 2, 3, 4, 5))

    def test_turnover_starts_live_ball_clock(self):
        p = self.poss[4]   # HOME trip after Kim's steal-turnover; clock starts at the turnover, shot 20 s later
        self.assertEqual(p["start_type"], "after_steal")
        self.assertEqual(p["start"], Pbp.Elapsed(1, "PT11M20.00S"))
        self.assertFalse(p["transition"])

    def test_team_rows_assigned(self):
        ev = Pbp.GameEvents(PBP, HOME, AWAY)
        self.assertEqual(ev[HOME]["timeouts"], 1)

    def test_stints(self):
        st = Pbp.BuildStints(PBP, ROT, HOME, AWAY, self.poss)
        self.assertEqual(sum(s["seconds"] for s in st), 2880.0)
        self.assertEqual(sum(s["home_pts"] for s in st), 5)
        self.assertEqual(sum(s["away_pts"] for s in st), 3)
        self.assertTrue(all(len(s["home5"]) == 5 and len(s["away5"]) == 5 for s in st))


class TestRealGames(unittest.TestCase):
    def test_consistency_on_cached_games(self):
        from Matchups.V2 import DataLayer as D
        files = sorted(glob.glob(os.path.join(D.ROOT2, "pbp", "2026", "*.json")))[:5]
        if not files:
            self.skipTest("no cached PBP yet")
        games = {g["game_id"]: g for g in D.LoadJson("data/season/games_2026.json", [])}
        box = D.PlayerBox(2026)
        checked = 0
        for f in files:
            gid = os.path.basename(f)[:-5]
            g = games.get(gid)
            rot = D.Rotation(2026, gid)
            if not g or not rot or g.get("home_score") is None:
                continue
            with self.subTest(gid=gid):
                pbp = D.Pbp(2026, gid)
                poss = Pbp.BuildPossessions(pbp, rot, g["home_team"], g["away_team"])
                st = Pbp.BuildStints(pbp, rot, g["home_team"], g["away_team"], poss)
                for team, key in ((g["home_team"], "home_score"), (g["away_team"], "away_score")):
                    self.assertEqual(sum(p["pts"] for p in poss if p["offense"] == team), g[key])
                    n = sum(p["offense"] == team for p in poss)
                    self.assertTrue(85 <= n <= 125, n)
                    oreb = sum(r["oreb"] for r in box.get(gid, []) if r["team"] == team)
                    self.assertLessEqual(abs(sum(p["oreb"] for p in poss if p["offense"] == team) - oreb), 2)
                self.assertEqual(sum(s["home_pts"] for s in st), g["home_score"])
                self.assertTrue(all(len(s["home5"]) == 5 and len(s["away5"]) == 5 for s in st))
                checked += 1
        if not checked:
            self.skipTest("cached PBP present but no matching finished games")


if __name__ == "__main__":
    unittest.main()
