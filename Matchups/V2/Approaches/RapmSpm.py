r"""
RapmSpm: RAPM with an informative box-score (SPM) prior, exponentially decayed and refit daily (xRAPM / EPM style).

Builds on StintRapm (hierarchical O/D ratings on stints, sufficient statistics decayed 0.998 per day). The prior
mean for each player is no longer just a role mean: it is a statistical plus-minus (SPM) prediction from his
conjugate skill posteriors (Matchups.V3.Skills: per-36 Gamma-Poisson counts and Beta-binomial shooting, each with
its own decay). The SPM map (skills -> O and D) is a Bayesian ridge fitted every 30 days on the CURRENT RAPM
posteriors of players with >= 1,500 decayed possessions, i.e. only on information from games already played.
Until the first SPM fit (first ~30 days of the warm-up season) the v2 role-mean prior is used. The posterior is
refit every day (refit_every=1).
"""

import numpy as np

from Matchups.V2.Approaches.StintRapm import StintRapm
from Matchups.V2.Base import Register
from Matchups.V2.Bayes import BayesLinear
from Matchups.V3.Skills import FEATURES, SkillBook


@Register
class RapmSpm(StintRapm):
    name = "rapm_spm"
    idea = "Daily-refit decayed RAPM whose player priors come from a box-score SPM on conjugate skill posteriors."
    data_used = ("rotation", "pbp", "player_games", "player_bio")
    covers = ("team_strength", "possession", "shot_quality", "rebounding", "turnovers")
    priors = dict(StintRapm.priors, **{
        "player prior mean": ("SPM(skills) for O and D, clipped to +-4 per 100", "box-score priors nearly double APM accuracy (Sill 2010)"),
        "SPM coefficients": ("N(0, 1^2) per sd of each skill, refit every 30 days on current RAPM", "leak-free: targets are posteriors from earlier games"),
        "skill decays": ("per stat, 0.975 (steals) to 0.997 (FT%) per game", "fast-changing stats forget faster")})

    def __init__(self, refit_every=1, spm_every=30, min_poss=1500.0, **params):
        super().__init__(refit_every=refit_every, prior_provider=self.SpmPrior, **params)
        self.skills = SkillBook()
        self.spm, self.spm_days, self.spm_every, self.min_poss = None, 0, spm_every, min_poss
        self.skill_days = 999
        self.spm_info = {}

    def FitSpm(self):
        rows, o_t, d_t, w = [], [], [], []
        for p, i in self.ix.ids.items():
            j = 1 + 2 * i
            expo = self.reg.XtX[j, j]
            if expo < self.min_poss:
                continue
            rows.append(self.skills.Vector(p))
            o_t.append(self.reg.mean[j])
            d_t.append(self.reg.mean[j + 1])
            w.append(min(expo / 3000.0, 1.0))
        if len(rows) < 60:
            return
        X = np.array(rows)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - mu) / sd]
        coefs = []
        for y in (o_t, d_t):
            r = BayesLinear(Z.shape[1], np.r_[4.0, np.ones(Z.shape[1] - 1)], sigma2=1.0)
            for z, yy, ww in zip(Z, y, w):
                r.Add(z, yy, ww)
            r.Fit()
            coefs.append(r.mean.copy())
        self.spm = (mu, sd, coefs[0], coefs[1])
        self.spm_info = {"n_players": len(rows), "top_o": sorted(zip(FEATURES, coefs[0][1:]), key=lambda x: -abs(x[1]))[:3]}

    def SpmPrior(self, p):
        if self.spm is None or self.skills.Exposure(p) <= 0:
            return None
        mu, sd, co, cd = self.spm
        z = np.r_[1.0, (self.skills.Vector(p) - mu) / sd]
        return float(np.clip(z @ co, -4, 4)), float(np.clip(z @ cd, -4, 4))

    def BeforeDay(self, day, gap):
        self.spm_days += gap
        self.skill_days += gap
        if self.skill_days >= 7:
            self.skills.Refit()
            self.skill_days = 0
        if self.spm_days >= self.spm_every:
            self.FitSpm()
            self.spm_days = 0
        super().BeforeDay(day, gap)

    def Predict(self, view):
        out = super().Predict(view)
        out["spm_players"] = self.spm_info.get("n_players", 0)
        return out

    def Update(self, rec):
        super().Update(rec)
        self.skills.Update(rec.box)
