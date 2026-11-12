r"""
Lightweight Bayesian toolkit (numpy only, CPU-friendly). Every v2 approach builds on these pieces.

- GammaPoisson    conjugate rate (points / possession, turnovers / possession...) with exponential forgetting
- BetaBinomial    conjugate probability (make %, OREB %, TOV %...) with forgetting
- EmpiricalBayesBeta / EmpiricalBayesNormal   fit a group prior from many units (partial pooling by role/position/age)
- BayesLinear     Gaussian prior x Gaussian likelihood regression = ridge with a posterior covariance;
                  sufficient statistics are decayed, so it is a forgetting (dynamic) regression
- Kalman          dynamic linear state-space ratings (random-walk states, Gaussian observations)
- LaplaceLogistic / LaplacePoisson   MAP + Hessian Gaussian approximation for GLMs with Gaussian priors
- Metropolis      tiny random-walk MCMC, used only in smoke/unit tests to check the Laplace approximations
- NormalCdf, PredictiveWinProb   posterior predictive win probability from a margin mean and variance
"""

import math

import numpy as np

SQRT2 = math.sqrt(2.0)


def NormalCdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / SQRT2))


def PredictiveWinProb(margin_mean: float, margin_var: float) -> float:
    """P(home margin > 0) under a Normal posterior predictive (ties go to OT, treated as 50/50 by symmetry)."""
    return float(NormalCdf(margin_mean / math.sqrt(max(margin_var, 1e-9))))


class GammaPoisson:
    """rate ~ Gamma(shape a, rate b); counts y over exposure e ~ Poisson(rate * e). Forgetting multiplies (a-a0, b-b0)."""

    def __init__(self, prior_mean: float, prior_strength: float):
        self.a0, self.b0 = prior_mean * prior_strength, prior_strength
        self.y = 0.0
        self.e = 0.0

    def Decay(self, factor: float):
        self.y *= factor
        self.e *= factor

    def Add(self, y: float, exposure: float):
        self.y += y
        self.e += exposure

    @property
    def shape(self):
        return self.a0 + self.y

    @property
    def rate(self):
        return self.b0 + self.e

    def Mean(self):
        return self.shape / self.rate

    def Var(self):
        return self.shape / self.rate ** 2

    def PredictiveMoments(self, exposure: float):
        """Negative-binomial predictive count over `exposure`: mean and variance (parameter + Poisson noise)."""
        m = self.Mean() * exposure
        return m, m + self.Var() * exposure ** 2


class BetaBinomial:
    def __init__(self, prior_mean: float, prior_strength: float):
        self.a0, self.b0 = prior_mean * prior_strength, (1 - prior_mean) * prior_strength
        self.k = 0.0
        self.n = 0.0

    def Decay(self, factor: float):
        self.k *= factor
        self.n *= factor

    def Add(self, k: float, n: float):
        self.k += k
        self.n += n

    def Mean(self):
        return (self.a0 + self.k) / (self.a0 + self.b0 + self.n)

    def Var(self):
        a, b = self.a0 + self.k, self.b0 + self.n - self.k
        return a * b / ((a + b) ** 2 * (a + b + 1))

    def Interval(self, z: float = 1.645):
        m, s = self.Mean(), math.sqrt(self.Var())
        return max(0.0, m - z * s), min(1.0, m + z * s)


def EmpiricalBayesBeta(k, n, min_n: float = 1.0):
    """Method-of-moments Beta prior (mean, strength) from many units' successes k / trials n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    keep = n >= min_n
    if keep.sum() < 3:
        return float(k.sum() / max(n.sum(), 1e-9)) if n.sum() else 0.5, 10.0
    p = k[keep] / n[keep]
    w = n[keep]
    m = float(np.average(p, weights=w))
    var_obs = float(np.average((p - m) ** 2, weights=w))
    var_binom = float(np.mean(m * (1 - m) / w))
    var_true = max(var_obs - var_binom, 1e-6)
    strength = max(m * (1 - m) / var_true - 1.0, 1.0)
    return m, float(min(strength, 1e5))


def EmpiricalBayesNormal(x, se2):
    """Normal-Normal partial pooling: returns (prior mean, prior variance tau^2) and shrunk posteriors."""
    x, se2 = np.asarray(x, float), np.asarray(se2, float)
    w = 1.0 / np.maximum(se2, 1e-12)
    mu = float(np.sum(w * x) / np.sum(w))
    tau2 = max(float(np.mean((x - mu) ** 2 - se2)), 1e-9)
    shrink = tau2 / (tau2 + se2)
    post_mean = mu + shrink * (x - mu)
    post_var = 1.0 / (1.0 / tau2 + w)
    return mu, tau2, post_mean, post_var


class BayesLinear:
    """
    y = X beta + eps, eps ~ N(0, sigma2 / w); beta ~ N(m0, diag(v0)). Sufficient statistics (X'WX, X'Wy, y'Wy, sum w)
    are decayed for forgetting. Posterior: precision P = X'WX / sigma2 + diag(1/v0), mean = P^-1 (X'Wy / sigma2 + m0 / v0).
    sigma2 is estimated from the decayed residual sum of squares (empirical Bayes) unless fixed.
    """

    def __init__(self, dim: int, prior_var, prior_mean=None, sigma2: float = None):
        self.dim = dim
        self.v0 = np.full(dim, prior_var, float) if np.isscalar(prior_var) else np.asarray(prior_var, float).copy()
        self.m0 = np.zeros(dim) if prior_mean is None else np.asarray(prior_mean, float).copy()
        self.XtX = np.zeros((dim, dim))
        self.Xty = np.zeros(dim)
        self.yty = 0.0
        self.sw = 0.0
        self.fixed_sigma2 = sigma2
        self.mean = self.m0.copy()
        self.cov = np.diag(self.v0)
        self.dirty = False

    def Decay(self, factor: float):
        self.XtX *= factor
        self.Xty *= factor
        self.yty *= factor
        self.sw *= factor

    def AddSparse(self, idx, vals, y: float, w: float = 1.0):
        idx, vals = np.asarray(idx), np.asarray(vals, float)
        self.XtX[np.ix_(idx, idx)] += w * np.outer(vals, vals)
        self.Xty[idx] += w * vals * y
        self.yty += w * y * y
        self.sw += w
        self.dirty = True

    def Add(self, x, y: float, w: float = 1.0):
        x = np.asarray(x, float)
        nz = np.nonzero(x)[0]
        self.AddSparse(nz, x[nz], y, w)

    def Sigma2(self):
        if self.fixed_sigma2 is not None:
            return self.fixed_sigma2
        if self.sw < 5:
            return 1.0
        rss = self.yty - 2 * self.mean @ self.Xty + self.mean @ self.XtX @ self.mean
        return max(rss / self.sw, 1e-6)

    def SetPriorMean(self, m0):
        self.m0 = np.asarray(m0, float).copy()
        self.dirty = True

    def Fit(self, active: int = None, covariance: bool = True):
        if not self.dirty:
            return
        n = self.dim if active is None else active
        s2 = self.Sigma2()
        P = self.XtX[:n, :n] / s2 + np.diag(1.0 / self.v0[:n])
        rhs = self.Xty[:n] / s2 + self.m0[:n] / self.v0[:n]
        if covariance:
            cov = np.linalg.inv(P)
            self.mean[:n] = cov @ rhs
            self.cov = np.diag(self.v0).copy() if self.cov.shape[0] != self.dim else self.cov
            self.cov[:n, :n] = cov
        else:
            self.mean[:n] = np.linalg.solve(P, rhs)
        self.mean[n:] = self.m0[n:]
        self.dirty = False

    def Predict(self, x):
        """Posterior predictive mean and variance (parameter uncertainty + noise)."""
        x = np.asarray(x, float)
        return float(x @ self.mean), float(x @ self.cov @ x + self.Sigma2())

    def PredictSparse(self, idx, vals):
        idx, vals = np.asarray(idx), np.asarray(vals, float)
        sub = self.cov[np.ix_(idx, idx)]
        return float(vals @ self.mean[idx]), float(vals @ sub @ vals + self.Sigma2())


class Kalman:
    """Random-walk states x_t = x_{t-1} + N(0, q * days); observations y = h'x + N(0, r). Exact Gaussian updates."""

    def __init__(self, dim: int, prior_var: float, q_per_day: float, obs_var: float, prior_mean=None):
        self.x = np.zeros(dim) if prior_mean is None else np.asarray(prior_mean, float).copy()
        self.P = np.eye(dim) * prior_var
        self.q, self.r = q_per_day, obs_var

    def Predict(self, days: float):
        if days > 0:
            self.P = self.P + np.eye(len(self.x)) * self.q * days

    def Forecast(self, h):
        h = np.asarray(h, float)
        return float(h @ self.x), float(h @ self.P @ h + self.r)

    def Update(self, h, y: float, r: float = None):
        h = np.asarray(h, float)
        r = self.r if r is None else r
        s = float(h @ self.P @ h + r)
        k = (self.P @ h) / s
        innovation = y - float(h @ self.x)
        self.x = self.x + k * innovation
        self.P = self.P - np.outer(k, h @ self.P)
        self.P = 0.5 * (self.P + self.P.T)
        return innovation, s

    def Regress(self, toward, fraction: float):
        """Between seasons: pull means toward `toward` and inflate variance (new season uncertainty)."""
        self.x = toward + (1 - fraction) * (self.x - toward)


def _Sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def LaplaceLogistic(X, y, prior_mean, prior_var, iterations: int = 50):
    """Bayesian logistic regression: MAP by Newton, posterior ~ N(MAP, H^-1)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    m0 = np.asarray(prior_mean, float)
    v0 = np.full(X.shape[1], prior_var, float) if np.isscalar(prior_var) else np.asarray(prior_var, float)
    w = m0.copy()
    for _ in range(iterations):
        p = _Sigmoid(X @ w)
        g = X.T @ (p - y) + (w - m0) / v0
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(1 / v0)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = _Sigmoid(X @ w)
    H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(1 / v0)
    return w, np.linalg.inv(H)


def LaplacePoisson(X, y, exposure, prior_mean, prior_var, iterations: int = 50):
    """Bayesian log-linear Poisson GLM: y ~ Poisson(exposure * exp(X b)), b ~ N(m0, v0). Returns MAP, covariance."""
    X, y, e = np.asarray(X, float), np.asarray(y, float), np.asarray(exposure, float)
    m0 = np.asarray(prior_mean, float)
    v0 = np.full(X.shape[1], prior_var, float) if np.isscalar(prior_var) else np.asarray(prior_var, float)
    b = m0.copy()
    for _ in range(iterations):
        mu = e * np.exp(np.clip(X @ b, -20, 20))
        g = X.T @ (mu - y) + (b - m0) / v0
        H = (X * mu[:, None]).T @ X + np.diag(1 / v0)
        step = np.linalg.solve(H, g)
        b -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = e * np.exp(np.clip(X @ b, -20, 20))
    H = (X * mu[:, None]).T @ X + np.diag(1 / v0)
    return b, np.linalg.inv(H)


def Metropolis(log_post, x0, steps: int = 4000, scale: float = 0.1, seed: int = 0):
    """Random-walk Metropolis. Only for tiny smoke/unit checks of the Laplace approximations."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, float).copy()
    lp = log_post(x)
    out, accepted = [], 0
    for _ in range(steps):
        prop = x + rng.normal(0, scale, size=x.shape)
        lpp = log_post(prop)
        if math.log(rng.random()) < lpp - lp:
            x, lp = prop, lpp
            accepted += 1
        out.append(x.copy())
    return np.array(out), accepted / steps
