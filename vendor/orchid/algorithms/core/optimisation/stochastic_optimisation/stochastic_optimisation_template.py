"""
Stochastic Optimisation Template
================================
Pure-Python, zero-dependency toolkit for OPTIMISATION UNDER UNCERTAINTY and
optimisation of NOISY / STOCHASTIC objectives.  Where the sibling ``nonlinear_
programming`` / ``convex_optimisation`` modules assume a *deterministic* oracle,
here the objective, its gradient, or the data itself is RANDOM -- so the tools
are built around sampling, noise, variance reduction and risk.

It bundles, behind one small uniform API (every solver returns an ``OptResult``):

    1.  Stochastic first-order optimisers -- the machine-learning workhorses:
        SGD (+ momentum / Nesterov / weight decay), AdaGrad, RMSProp, AdaDelta,
        Adam, AdaMax, AMSGrad, Nadam -- plus learning-rate schedules, gradient
        clipping, Polyak-Ruppert averaging and a ``minimize_sgd`` driver.
    2.  Finite-sum / variance reduction -- mini-batch SGD, SVRG, SAGA.
    3.  Gradient-free stochastic approximation -- SPSA, Kiefer-Wolfowitz,
        Robbins-Monro root finding, finite-difference gradients.
    4.  Distribution-based global search -- the Cross-Entropy Method, separable
        CMA-ES and a Natural Evolution Strategy, for noisy black-box objectives.
    5.  Sampling & scenarios -- Monte-Carlo estimation (with control variates and
        importance sampling), Latin Hypercube, Halton / Sobol low-discrepancy
        sequences, antithetic variates and Kantorovich scenario reduction.
    6.  Stochastic programming -- Sample Average Approximation, two-stage
        problems with recourse, the newsvendor model, and the classic value
        metrics EVPI / VSS.
    7.  Risk-averse optimisation -- Value-at-Risk, Conditional VaR (expected
        shortfall), entropic and mean-variance risk, and CVaR minimisation via
        the Rockafellar-Uryasev stochastic subgradient.
    8.  Stochastic bandits -- epsilon-greedy, UCB1 and Thompson sampling.

Everything is standard-library only, so it drops straight into a competition /
hackathon / interview / research laptop -- just copy this module.  All randomised
routines take a ``seed`` (or ``rng``) for reproducibility.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple)

__version__ = "1.0.0"

Vector = List[float]
Scalar = float

__all__ = [
    # result type
    "OptResult",
    # 1. stochastic first-order optimisers
    "SGD", "AdaGrad", "RMSProp", "AdaDelta", "Adam", "AdaMax", "AMSGrad", "Nadam",
    "constant_lr", "step_decay", "exponential_decay", "inverse_time_decay",
    "cosine_annealing", "polynomial_decay", "warmup_cosine",
    "clip_by_norm", "clip_by_value", "minimize_sgd",
    # 2. finite-sum / variance reduction
    "minibatch_sgd", "svrg", "saga",
    # 3. gradient-free stochastic approximation
    "spsa", "kiefer_wolfowitz", "robbins_monro", "finite_difference_gradient",
    # 4. distribution-based global search
    "cross_entropy_method", "sep_cma_es", "evolution_strategy",
    # 5. sampling & scenarios
    "monte_carlo_estimate", "control_variate_estimate", "importance_sampling_estimate",
    "latin_hypercube", "van_der_corput", "halton_sequence", "sobol_sequence",
    "antithetic_normal", "stratified_uniform", "scenario_reduction",
    # 6. stochastic programming
    "sample_average_approximation", "two_stage_saa", "newsvendor",
    "stochastic_solution_metrics",
    # 7. risk-averse optimisation
    "value_at_risk", "conditional_value_at_risk", "entropic_risk",
    "mean_variance_objective", "minimize_cvar",
    # 8. stochastic bandits
    "EpsilonGreedy", "UCB1", "ThompsonSampling", "run_bandit",
]

_EPS = 1e-12


# ===========================================================================
# 0. VECTOR HELPERS + RESULT TYPE
# ===========================================================================
def _vec(x: Sequence[float]) -> Vector:
    return [float(v) for v in x]


def _add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x + y for x, y in zip(a, b)]


def _sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x - y for x, y in zip(a, b)]


def _scale(a: Sequence[float], s: float) -> Vector:
    return [s * x for x in a]


def _axpy(s: float, a: Sequence[float], b: Sequence[float]) -> Vector:
    """s*a + b."""
    return [s * x + y for x, y in zip(a, b)]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in a))


def _hadamard(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x * y for x, y in zip(a, b)]


def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return math.fsum(xs) / len(xs) if xs else 0.0


def _variance(xs: Sequence[float], ddof: int = 1) -> float:
    xs = list(xs)
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    m = _mean(xs)
    return math.fsum((x - m) ** 2 for x in xs) / (n - ddof)


def _all_finite(x: Sequence[float]) -> bool:
    return all(math.isfinite(v) for v in x)


def _as_rng(seed_or_rng: Any) -> random.Random:
    """Accept an int seed, a ``random.Random``, or ``None`` (fresh but seeded 0)."""
    if isinstance(seed_or_rng, random.Random):
        return seed_or_rng
    if seed_or_rng is None:
        return random.Random(0)
    return random.Random(seed_or_rng)


@dataclass
class OptResult:
    """Uniform result of a stochastic solve.

    x         : the best decision vector found.
    fun       : objective value at ``x`` (an *estimate* for noisy objectives).
    nit       : number of iterations / generations performed.
    nfev      : number of objective / gradient evaluations.
    converged : whether a stopping criterion (not the iteration cap) was met.
    message   : human-readable status.
    history   : per-iteration objective (or surrogate) trace, for plotting.
    meta      : solver-specific extras (std error, elites, paths, ...).
    """
    x: Vector
    fun: float
    nit: int = 0
    nfev: int = 0
    converged: bool = False
    message: str = ""
    history: List[float] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# 1. STOCHASTIC FIRST-ORDER OPTIMISERS
# ===========================================================================
# Each optimiser is a small stateful object with ``step(params, grad) -> params``
# returning a NEW parameter vector (it never mutates its input), mirroring the
# PyTorch / Keras optimiser pattern.  ``reset()`` clears the accumulated state.
class _BaseOptimizer:
    """Common machinery: dimension-lazy state and optional decoupled weight decay."""

    def __init__(self, lr: float, weight_decay: float = 0.0):
        if not lr > 0:
            raise ValueError("lr must be > 0.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.t = 0

    def reset(self) -> None:
        self.t = 0

    def _decay(self, params: Vector, grad: Vector) -> Vector:
        if self.weight_decay:
            return _axpy(self.weight_decay, params, grad)
        return list(grad)

    def step(self, params: Sequence[float], grad: Sequence[float]) -> Vector:
        raise NotImplementedError


class SGD(_BaseOptimizer):
    """Stochastic gradient descent with optional (Nesterov) momentum.

    ``momentum`` accumulates a velocity ``v = mu*v + g``; with ``nesterov`` the
    step looks ahead (``g + mu*v``).  The classic robust baseline."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.0,
                 nesterov: bool = False, weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if nesterov and momentum == 0.0:
            raise ValueError("nesterov momentum needs momentum > 0.")
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.v: Optional[Vector] = None

    def reset(self) -> None:
        super().reset()
        self.v = None

    def step(self, params: Sequence[float], grad: Sequence[float]) -> Vector:
        self.t += 1
        g = self._decay(_vec(params), grad)
        if self.momentum:
            if self.v is None:
                self.v = [0.0] * len(g)
            self.v = _axpy(self.momentum, self.v, g)          # v = mu v + g
            d = _axpy(self.momentum, self.v, g) if self.nesterov else self.v
        else:
            d = g
        return _axpy(-self.lr, d, _vec(params))


class AdaGrad(_BaseOptimizer):
    """Adaptive gradient: per-coordinate learning rate scaled by 1/sqrt(sum g^2).
    Strong on sparse features; the accumulator only grows, so the rate decays."""

    def __init__(self, lr: float = 0.01, eps: float = 1e-10,
                 weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        self.eps = float(eps)
        self.g2: Optional[Vector] = None

    def reset(self) -> None:
        super().reset()
        self.g2 = None

    def step(self, params: Sequence[float], grad: Sequence[float]) -> Vector:
        self.t += 1
        g = self._decay(_vec(params), grad)
        if self.g2 is None:
            self.g2 = [0.0] * len(g)
        self.g2 = [a + gi * gi for a, gi in zip(self.g2, g)]
        step = [self.lr * gi / (math.sqrt(a) + self.eps)
                for gi, a in zip(g, self.g2)]
        return _sub(params, step)


class RMSProp(_BaseOptimizer):
    """Root-mean-square propagation: an exponentially-decayed gradient-square
    accumulator, so the rate adapts without vanishing like AdaGrad's."""

    def __init__(self, lr: float = 0.01, rho: float = 0.9, eps: float = 1e-8,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        if not 0.0 < rho < 1.0:
            raise ValueError("rho must be in (0, 1).")
        self.rho = float(rho)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.g2: Optional[Vector] = None
        self.v: Optional[Vector] = None

    def reset(self) -> None:
        super().reset()
        self.g2 = self.v = None

    def step(self, params: Sequence[float], grad: Sequence[float]) -> Vector:
        self.t += 1
        g = self._decay(_vec(params), grad)
        if self.g2 is None:
            self.g2 = [0.0] * len(g)
            self.v = [0.0] * len(g)
        self.g2 = [self.rho * a + (1 - self.rho) * gi * gi
                   for a, gi in zip(self.g2, g)]
        upd = [self.lr * gi / (math.sqrt(a) + self.eps)
               for gi, a in zip(g, self.g2)]
        if self.momentum:
            self.v = _axpy(self.momentum, self.v, upd)
            upd = self.v
        return _sub(params, upd)


class AdaDelta(_BaseOptimizer):
    """AdaDelta: RMSProp with a running window of *update* magnitudes too, so it
    is (nearly) learning-rate free.  ``lr`` here is a plain global multiplier."""

    def __init__(self, lr: float = 1.0, rho: float = 0.95, eps: float = 1e-6,
                 weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        if not 0.0 < rho < 1.0:
            raise ValueError("rho must be in (0, 1).")
        self.rho = float(rho)
        self.eps = float(eps)
        self.g2: Optional[Vector] = None
        self.dx2: Optional[Vector] = None

    def reset(self) -> None:
        super().reset()
        self.g2 = self.dx2 = None

    def step(self, params: Sequence[float], grad: Sequence[float]) -> Vector:
        self.t += 1
        g = self._decay(_vec(params), grad)
        if self.g2 is None:
            self.g2 = [0.0] * len(g)
            self.dx2 = [0.0] * len(g)
        self.g2 = [self.rho * a + (1 - self.rho) * gi * gi
                   for a, gi in zip(self.g2, g)]
        dx = [-math.sqrt(dxa + self.eps) / math.sqrt(ga + self.eps) * gi
              for gi, ga, dxa in zip(g, self.g2, self.dx2)]
        self.dx2 = [self.rho * a + (1 - self.rho) * d * d
                    for a, d in zip(self.dx2, dx)]
        return _axpy(self.lr, dx, _vec(params))


class Adam(_BaseOptimizer):
    """Adam: bias-corrected first- and second-moment estimates.  The default
    optimiser for most stochastic problems.  ``amsgrad`` uses AMSGrad's
    non-decreasing second-moment (see the :class:`AMSGrad` alias)."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0, amsgrad: bool = False):
        super().__init__(lr, weight_decay)
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1, beta2 must be in [0, 1).")
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.amsgrad = bool(amsgrad)
        self.m: Optional[Vector] = None
        self.v: Optional[Vector] = None
        self.vmax: Optional[Vector] = None

    def reset(self) -> None:
        super().reset()
        self.m = self.v = self.vmax = None

    def step(self, params: Sequence[float], grad: Sequence[float]) -> Vector:
        self.t += 1
        g = self._decay(_vec(params), grad)
        if self.m is None:
            self.m = [0.0] * len(g)
            self.v = [0.0] * len(g)
            self.vmax = [0.0] * len(g)
        b1, b2 = self.beta1, self.beta2
        self.m = [b1 * mi + (1 - b1) * gi for mi, gi in zip(self.m, g)]
        self.v = [b2 * vi + (1 - b2) * gi * gi for vi, gi in zip(self.v, g)]
        mc = 1 - b1 ** self.t
        vc = 1 - b2 ** self.t
        out = []
        for i, p in enumerate(params):
            mhat = self.m[i] / mc
            vhat = self.v[i] / vc
            if self.amsgrad:
                self.vmax[i] = max(self.vmax[i], vhat)
                vhat = self.vmax[i]
            out.append(p - self.lr * mhat / (math.sqrt(vhat) + self.eps))
        return out


class AMSGrad(Adam):
    """Adam with the AMSGrad fix (non-decreasing second moment) -- restores the
    convergence guarantee Adam can lose on some online problems."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(lr, beta1, beta2, eps, weight_decay, amsgrad=True)


class AdaMax(_BaseOptimizer):
    """AdaMax: the Adam variant using an exponentially-weighted infinity norm of
    the gradients, which is more stable when gradients have heavy tails."""

    def __init__(self, lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1, beta2 must be in [0, 1).")
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.m: Optional[Vector] = None
        self.u: Optional[Vector] = None

    def reset(self) -> None:
        super().reset()
        self.m = self.u = None

    def step(self, params: Sequence[float], grad: Sequence[float]) -> Vector:
        self.t += 1
        g = self._decay(_vec(params), grad)
        if self.m is None:
            self.m = [0.0] * len(g)
            self.u = [0.0] * len(g)
        b1, b2 = self.beta1, self.beta2
        self.m = [b1 * mi + (1 - b1) * gi for mi, gi in zip(self.m, g)]
        self.u = [max(b2 * ui, abs(gi)) for ui, gi in zip(self.u, g)]
        mc = 1 - b1 ** self.t
        return [p - (self.lr / mc) * mi / (ui + self.eps)
                for p, mi, ui in zip(params, self.m, self.u)]


class Nadam(_BaseOptimizer):
    """Nadam: Adam with Nesterov-accelerated momentum, often a touch faster."""

    def __init__(self, lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(lr, weight_decay)
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1, beta2 must be in [0, 1).")
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.m: Optional[Vector] = None
        self.v: Optional[Vector] = None

    def reset(self) -> None:
        super().reset()
        self.m = self.v = None

    def step(self, params: Sequence[float], grad: Sequence[float]) -> Vector:
        self.t += 1
        g = self._decay(_vec(params), grad)
        if self.m is None:
            self.m = [0.0] * len(g)
            self.v = [0.0] * len(g)
        b1, b2, t = self.beta1, self.beta2, self.t
        self.m = [b1 * mi + (1 - b1) * gi for mi, gi in zip(self.m, g)]
        self.v = [b2 * vi + (1 - b2) * gi * gi for vi, gi in zip(self.v, g)]
        mc = 1 - b1 ** t
        vc = 1 - b2 ** t
        out = []
        for i, p in enumerate(params):
            mhat = self.m[i] / mc
            vhat = self.v[i] / vc
            nesterov_m = b1 * mhat + (1 - b1) * g[i] / mc
            out.append(p - self.lr * nesterov_m / (math.sqrt(vhat) + self.eps))
        return out


# --- learning-rate schedules: each returns a callable step -> lr ------------
def constant_lr(lr: float) -> Callable[[int], float]:
    """A flat learning rate."""
    return lambda t: lr


def step_decay(lr0: float, drop: float = 0.5, every: int = 100
               ) -> Callable[[int], float]:
    """Multiply by ``drop`` every ``every`` steps (staircase)."""
    if not every >= 1:
        raise ValueError("every must be >= 1.")
    return lambda t: lr0 * (drop ** (t // every))


def exponential_decay(lr0: float, rate: float = 0.01) -> Callable[[int], float]:
    """``lr0 * exp(-rate * t)`` -- smooth geometric decay."""
    return lambda t: lr0 * math.exp(-rate * t)


def inverse_time_decay(lr0: float, rate: float = 0.001) -> Callable[[int], float]:
    """``lr0 / (1 + rate * t)`` -- the classic Robbins-Monro 1/t schedule."""
    return lambda t: lr0 / (1.0 + rate * t)


def cosine_annealing(lr0: float, t_max: int, lr_min: float = 0.0
                     ) -> Callable[[int], float]:
    """Cosine decay from ``lr0`` to ``lr_min`` over ``t_max`` steps (then holds)."""
    if not t_max >= 1:
        raise ValueError("t_max must be >= 1.")
    return lambda t: lr_min + 0.5 * (lr0 - lr_min) * (
        1 + math.cos(math.pi * min(t, t_max) / t_max))


def polynomial_decay(lr0: float, t_max: int, power: float = 1.0,
                     lr_min: float = 0.0) -> Callable[[int], float]:
    """Polynomial decay to ``lr_min`` at ``t_max`` (``power=1`` is linear)."""
    if not t_max >= 1:
        raise ValueError("t_max must be >= 1.")
    return lambda t: (lr0 - lr_min) * (1 - min(t, t_max) / t_max) ** power + lr_min


def warmup_cosine(lr0: float, warmup: int, t_max: int, lr_min: float = 0.0
                  ) -> Callable[[int], float]:
    """Linear warm-up over ``warmup`` steps, then cosine annealing to ``lr_min``."""
    if not (warmup >= 0 and t_max > warmup):
        raise ValueError("need 0 <= warmup < t_max.")
    cos = cosine_annealing(lr0, t_max - warmup, lr_min)
    return lambda t: (lr0 * (t + 1) / warmup if warmup and t < warmup
                      else cos(t - warmup))


def clip_by_norm(grad: Sequence[float], max_norm: float) -> Vector:
    """Scale ``grad`` down so its L2 norm is at most ``max_norm``."""
    if not max_norm > 0:
        raise ValueError("max_norm must be > 0.")
    n = _norm(grad)
    if n > max_norm:
        return _scale(grad, max_norm / n)
    return list(grad)


def clip_by_value(grad: Sequence[float], lo: float, hi: float) -> Vector:
    """Clamp each gradient component into ``[lo, hi]``."""
    if lo > hi:
        raise ValueError("need lo <= hi.")
    return [min(hi, max(lo, g)) for g in grad]


def minimize_sgd(grad: Callable[[Vector], Sequence[float]], x0: Sequence[float], *,
                 optimizer: Optional[_BaseOptimizer] = None, n_iter: int = 1000,
                 lr_schedule: Optional[Callable[[int], float]] = None,
                 clip_norm: Optional[float] = None,
                 objective: Optional[Callable[[Vector], float]] = None,
                 averaging: bool = False, average_from: float = 0.5,
                 tol: Optional[float] = None, patience: int = 20,
                 callback: Optional[Callable[[int, Vector], None]] = None
                 ) -> OptResult:
    """Drive a stochastic first-order ``optimizer`` (default :class:`Adam`) with a
    noisy gradient oracle ``grad(x)``.

    ``lr_schedule(t)`` (optional) overrides ``optimizer.lr`` each step.  With
    ``averaging`` the returned ``x`` is the Polyak-Ruppert tail average (the last
    ``1 - average_from`` fraction of iterates), which sharply reduces the noise of
    a constant-rate run.  Pass ``objective(x)`` to record a convergence trace and
    enable early stopping (``tol`` over ``patience`` windows)."""
    x = _vec(x0)
    if optimizer is None:
        optimizer = Adam()
    optimizer.reset()
    history: List[float] = []
    nfev = 0
    avg: Optional[Vector] = None
    avg_count = 0
    start_avg = int(average_from * n_iter)
    best = float("inf")
    stall = 0
    converged = False
    for t in range(n_iter):
        if lr_schedule is not None:
            optimizer.lr = lr_schedule(t)
        g = list(grad(x))
        nfev += 1
        if len(g) != len(x):
            raise ValueError("gradient length must match x0.")
        if clip_norm is not None:
            g = clip_by_norm(g, clip_norm)
        x = optimizer.step(x, g)
        if not _all_finite(x):
            return OptResult(x, float("nan"), t + 1, nfev, False,
                             "diverged (non-finite iterate)", history)
        if averaging and t >= start_avg:
            avg = list(x) if avg is None else _add(avg, x)
            avg_count += 1
        if objective is not None:
            f = float(objective(x))
            history.append(f)
            if tol is not None:
                if f < best - tol:
                    best, stall = f, 0
                else:
                    stall += 1
                    if stall >= patience:
                        converged = True
                        if callback:
                            callback(t, x)
                        break
        if callback:
            callback(t, x)
    x_out = _scale(avg, 1.0 / avg_count) if (averaging and avg_count) else x
    f_out = float(objective(x_out)) if objective is not None else float("nan")
    return OptResult(x_out, f_out, t + 1, nfev, converged,
                     "converged" if converged else "max iterations reached",
                     history, {"averaged": averaging})


# ===========================================================================
# 2. FINITE-SUM / VARIANCE REDUCTION
# ===========================================================================
# For objectives F(x) = (1/n) sum_i f_i(x) with a per-component gradient oracle
# ``grad_i(x, i)``.  Variance-reduced methods (SVRG, SAGA) converge linearly on
# strongly-convex F, unlike plain SGD.
def minibatch_sgd(grad_i: Callable[[Vector, int], Sequence[float]],
                  x0: Sequence[float], n: int, *, batch_size: int = 1,
                  optimizer: Optional[_BaseOptimizer] = None, n_epochs: int = 50,
                  objective: Optional[Callable[[Vector], float]] = None,
                  seed: Any = 0) -> OptResult:
    """Mini-batch SGD over ``n`` components: each step averages ``batch_size``
    random per-component gradients and takes an ``optimizer`` step."""
    if not (1 <= batch_size <= n):
        raise ValueError("batch_size must be in [1, n].")
    rng = _as_rng(seed)
    x = _vec(x0)
    if optimizer is None:
        optimizer = SGD(lr=0.05)
    optimizer.reset()
    hist: List[float] = []
    nfev = 0
    steps = max(1, n // batch_size)
    for _ in range(n_epochs):
        for _ in range(steps):
            idx = [rng.randrange(n) for _ in range(batch_size)]
            g = [0.0] * len(x)
            for i in idx:
                g = _add(g, grad_i(x, i))
                nfev += 1
            g = _scale(g, 1.0 / batch_size)
            x = optimizer.step(x, g)
        if objective is not None:
            hist.append(float(objective(x)))
    f = float(objective(x)) if objective is not None else float("nan")
    return OptResult(x, f, n_epochs, nfev, False, "epochs completed", hist)


def svrg(grad_i: Callable[[Vector, int], Sequence[float]], x0: Sequence[float],
         n: int, *, lr: float = 0.1, n_epochs: int = 50, inner: Optional[int] = None,
         objective: Optional[Callable[[Vector], float]] = None,
         seed: Any = 0) -> OptResult:
    """Stochastic Variance-Reduced Gradient (Johnson & Zhang 2013).  Each epoch
    computes a full gradient at a snapshot, then takes ``inner`` corrected SGD
    steps ``g_i(x) - g_i(snap) + full`` -- the correction cancels the noise, giving
    linear convergence on strongly-convex finite sums."""
    if not lr > 0:
        raise ValueError("lr must be > 0.")
    rng = _as_rng(seed)
    x = _vec(x0)
    inner = inner or n
    hist: List[float] = []
    nfev = 0
    for _ in range(n_epochs):
        snap = list(x)
        full = [0.0] * len(x)
        for i in range(n):
            full = _add(full, grad_i(snap, i))
            nfev += 1
        full = _scale(full, 1.0 / n)
        for _ in range(inner):
            i = rng.randrange(n)
            gi_x = list(grad_i(x, i))
            gi_s = list(grad_i(snap, i))
            nfev += 2
            corrected = _add(_sub(gi_x, gi_s), full)
            x = _axpy(-lr, corrected, x)
        if objective is not None:
            hist.append(float(objective(x)))
    f = float(objective(x)) if objective is not None else float("nan")
    return OptResult(x, f, n_epochs, nfev, False, "epochs completed", hist)


def saga(grad_i: Callable[[Vector, int], Sequence[float]], x0: Sequence[float],
         n: int, *, lr: float = 0.1, n_epochs: int = 50,
         objective: Optional[Callable[[Vector], float]] = None,
         seed: Any = 0) -> OptResult:
    """SAGA (Defazio, Bach & Lacoste-Julien 2014): keep a table of the last
    gradient seen for each component and step with ``g_i(x) - table_i + mean(table)``.
    Variance-reduced like SVRG but without periodic full-gradient sweeps."""
    if not lr > 0:
        raise ValueError("lr must be > 0.")
    rng = _as_rng(seed)
    x = _vec(x0)
    d = len(x)
    table = [list(grad_i(x, i)) for i in range(n)]           # per-component memory
    nfev = n
    avg = [0.0] * d
    for row in table:
        avg = _add(avg, row)
    avg = _scale(avg, 1.0 / n)
    hist: List[float] = []
    for _ in range(n_epochs):
        for _ in range(n):
            i = rng.randrange(n)
            gi = list(grad_i(x, i))
            nfev += 1
            direction = _add(_sub(gi, table[i]), avg)
            x = _axpy(-lr, direction, x)
            # incremental update of the running average table
            avg = _add(avg, _scale(_sub(gi, table[i]), 1.0 / n))
            table[i] = gi
        if objective is not None:
            hist.append(float(objective(x)))
    f = float(objective(x)) if objective is not None else float("nan")
    return OptResult(x, f, n_epochs, nfev, False, "epochs completed", hist)


# ===========================================================================
# 3. GRADIENT-FREE STOCHASTIC APPROXIMATION
# ===========================================================================
def finite_difference_gradient(f: Callable[[Vector], float], x: Sequence[float], *,
                               h: float = 1e-5, central: bool = True) -> Vector:
    """Coordinate-wise finite-difference gradient of (possibly noisy) ``f``.
    ``central`` uses the O(h^2) symmetric difference (2n evals) rather than
    forward (n+1 evals)."""
    x = _vec(x)
    n = len(x)
    g = [0.0] * n
    f0 = None if central else float(f(x))
    for i in range(n):
        xp = list(x)
        xp[i] += h
        if central:
            xm = list(x)
            xm[i] -= h
            g[i] = (float(f(xp)) - float(f(xm))) / (2 * h)
        else:
            g[i] = (float(f(xp)) - f0) / h
    return g


def spsa(f: Callable[[Vector], float], x0: Sequence[float], *, n_iter: int = 1000,
         a: float = 0.1, c: float = 0.1, alpha: float = 0.602, gamma: float = 0.101,
         big_a: Optional[float] = None, bounds: Optional[Sequence[Tuple[float, float]]]
         = None, objective: Optional[Callable[[Vector], float]] = None,
         seed: Any = 0) -> OptResult:
    """Simultaneous Perturbation Stochastic Approximation (Spall 1992).

    Estimates the whole gradient from just TWO noisy function evaluations per
    iteration by perturbing every coordinate at once with a random +/-1 vector --
    so the cost per step is independent of dimension, ideal for high-dimensional
    noisy black boxes.  Gains follow Spall's recommended decays
    ``a_k = a/(A+k+1)^alpha``, ``c_k = c/(k+1)^gamma``."""
    if a <= 0 or c <= 0:
        # c = 0 -> ck = 0 -> divide-by-zero in the SP gradient; a <= 0 ascends.
        raise ValueError("SPSA gains a and c must be > 0.")
    rng = _as_rng(seed)
    x = _vec(x0)
    n = len(x)
    big_a = (0.1 * n_iter) if big_a is None else big_a
    hist: List[float] = []
    nfev = 0
    for k in range(n_iter):
        ak = a / (big_a + k + 1) ** alpha
        ck = c / (k + 1) ** gamma
        delta = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(n)]
        xp = [xi + ck * di for xi, di in zip(x, delta)]
        xm = [xi - ck * di for xi, di in zip(x, delta)]
        yp, ym = float(f(xp)), float(f(xm))
        nfev += 2
        ghat = [(yp - ym) / (2 * ck * di) for di in delta]
        x = [xi - ak * gi for xi, gi in zip(x, ghat)]
        x = _project_box(x, bounds)     # validated: rejects length-mismatch / lo > hi
        if not _all_finite(x):
            return OptResult(x, float("nan"), k + 1, nfev, False, "diverged", hist)
        if objective is not None:
            hist.append(float(objective(x)))
    f_out = float(objective(x)) if objective is not None else float(f(x))
    return OptResult(x, f_out, n_iter, nfev, False, "max iterations reached", hist)


def kiefer_wolfowitz(f: Callable[[Vector], float], x0: Sequence[float], *,
                     n_iter: int = 1000, a: float = 0.1, c: float = 0.1,
                     alpha: float = 1.0, gamma: float = 1.0 / 6,
                     bounds: Optional[Sequence[Tuple[float, float]]] = None,
                     objective: Optional[Callable[[Vector], float]] = None,
                     seed: Any = 0) -> OptResult:
    """Kiefer-Wolfowitz stochastic approximation: a full coordinate-wise
    finite-difference gradient (2n evals/step) with decaying gains
    ``a_k = a/(k+1)^alpha``, ``c_k = c/(k+1)^gamma``.  More evaluations than SPSA
    but a lower-variance gradient estimate."""
    if a <= 0 or c <= 0:
        # c = 0 -> ck = 0 -> divide-by-zero in the finite difference; a <= 0 ascends.
        raise ValueError("gains a and c must be > 0.")
    x = _vec(x0)
    n = len(x)
    hist: List[float] = []
    nfev = 0
    for k in range(n_iter):
        ak = a / (k + 1) ** alpha
        ck = c / (k + 1) ** gamma
        g = [0.0] * n
        for i in range(n):
            xp = list(x)
            xp[i] += ck
            xm = list(x)
            xm[i] -= ck
            g[i] = (float(f(xp)) - float(f(xm))) / (2 * ck)
            nfev += 2
        x = [xi - ak * gi for xi, gi in zip(x, g)]
        x = _project_box(x, bounds)     # validated: rejects length-mismatch / lo > hi
        if objective is not None:
            hist.append(float(objective(x)))
    f_out = float(objective(x)) if objective is not None else float(f(x))
    return OptResult(x, f_out, n_iter, nfev, False, "max iterations reached", hist)


def robbins_monro(response: Callable[[float], float], x0: float = 0.0, *,
                  target: float = 0.0, n_iter: int = 1000, a: float = 1.0,
                  gamma: float = 1.0, seed: Any = 0) -> OptResult:
    """Robbins-Monro root finding: solve ``E[response(x)] = target`` from noisy
    observations, ``x <- x - (a/k^gamma) (response(x) - target)``.  The original
    (1951) stochastic-approximation scheme -- dosage finding, quantile tracking,
    adaptive control."""
    if a <= 0:
        # a <= 0 flips the update sign, so the iterate diverges instead of converging.
        raise ValueError("Robbins-Monro gain a must be > 0.")
    x = float(x0)
    hist: List[float] = []
    nfev = 0
    for k in range(n_iter):
        ak = a / (k + 1) ** gamma
        y = float(response(x))
        nfev += 1
        x = x - ak * (y - target)
        hist.append(x)
    return OptResult([x], float("nan"), n_iter, nfev, False, "root estimate",
                     hist, {"root": x})


# ===========================================================================
# 4. DISTRIBUTION-BASED GLOBAL SEARCH  (noisy black-box, no gradient)
# ===========================================================================
def _project_box(x: Vector, bounds: Optional[Sequence[Tuple[float, float]]]) -> Vector:
    if bounds is None:
        return x
    if len(bounds) != len(x):
        # A mismatched bounds list silently TRUNCATED x via zip (so e.g. minimize_cvar
        # returned a lower-dimensional point) in some solvers and raised a cryptic
        # IndexError in others -- fail fast with one clear message instead.
        raise ValueError("bounds must have exactly one (lo, hi) pair per dimension "
                         "(got %d for %d)." % (len(bounds), len(x)))
    out = []
    for xi, (lo, hi) in zip(x, bounds):
        if lo > hi:                     # match clip_by_value, which rejects lo > hi
            raise ValueError("each bound must satisfy lo <= hi.")
        out.append(min(hi, max(lo, xi)))
    return out


def cross_entropy_method(f: Callable[[Vector], float], x0: Sequence[float], *,
                         sigma0: float = 1.0, pop_size: Optional[int] = None,
                         elite_frac: float = 0.2, n_iter: int = 100,
                         smoothing: float = 0.7, minimize: bool = True,
                         noise: Optional[float] = None,
                         bounds: Optional[Sequence[Tuple[float, float]]] = None,
                         tol: float = 1e-8, seed: Any = 0) -> OptResult:
    """The Cross-Entropy Method for continuous optimisation.  Maintain a diagonal
    Gaussian ``N(mu, diag(sigma^2))``, sample a population, keep the ``elite_frac``
    best, and refit ``mu, sigma`` to the elite (with ``smoothing``).  Robust on
    noisy / multi-modal objectives; needs no gradient.

    ``noise`` is an injected exploration standard deviation, added to the sampling
    spread and decayed linearly to zero over the run (default ``sigma0``).  It is
    the standard cure for CEM's premature collapse: it keeps the search alive long
    enough for ``mu`` to travel to a distant optimum before the elite variance
    shrinks."""
    rng = _as_rng(seed)
    dim = len(x0)
    mu = _vec(x0)
    sigma = [float(sigma0)] * dim
    lam = pop_size or max(10, 4 * dim)
    n_elite = max(1, int(round(elite_frac * lam)))
    sign = 1.0 if minimize else -1.0
    noise0 = float(sigma0) if noise is None else float(noise)
    hist: List[float] = []
    nfev = 0
    best_x, best_f = list(mu), sign * float(f(mu))
    nfev += 1
    for it in range(n_iter):
        inj = noise0 * max(0.0, 1.0 - it / n_iter)          # decaying exploration
        pop = []
        for _ in range(lam):
            cand = [mu[i] + (sigma[i] + inj) * rng.gauss(0, 1) for i in range(dim)]
            cand = _project_box(cand, bounds)
            fv = sign * float(f(cand))
            nfev += 1
            pop.append((fv, cand))
        pop.sort(key=lambda t: t[0])
        elite = [c for _, c in pop[:n_elite]]
        new_mu = [_mean([e[i] for e in elite]) for i in range(dim)]
        new_sigma = [math.sqrt(max(_EPS, _mean([(e[i] - new_mu[i]) ** 2
                                                for e in elite]))) for i in range(dim)]
        mu = _axpy(smoothing, new_mu, _scale(mu, 1 - smoothing))
        sigma = _axpy(smoothing, new_sigma, _scale(sigma, 1 - smoothing))
        if pop[0][0] < best_f:
            best_f, best_x = pop[0][0], list(pop[0][1])
        hist.append(sign * best_f)
        if max(sigma) + inj < tol:                          # effective spread tiny
            return OptResult(best_x, sign * best_f, it + 1, nfev, True,
                             "sigma below tol", hist, {"sigma": sigma})
    return OptResult(best_x, sign * best_f, n_iter, nfev, False,
                     "max iterations reached", hist, {"sigma": sigma})


def sep_cma_es(f: Callable[[Vector], float], x0: Sequence[float], *,
               sigma0: float = 0.3, pop_size: Optional[int] = None,
               n_iter: int = 200, minimize: bool = True,
               bounds: Optional[Sequence[Tuple[float, float]]] = None,
               tol: float = 1e-11, seed: Any = 0) -> OptResult:
    """Separable CMA-ES (Ros & Hansen 2008): the diagonal-covariance variant of
    CMA-ES -- weighted-recombination evolution paths adapt a per-coordinate scale
    and a global step size, with no O(n^2) covariance or eigendecomposition.  A
    state-of-the-art derivative-free optimiser that scales to high dimension."""
    rng = _as_rng(seed)
    n = len(x0)
    mean = _vec(x0)
    sigma = float(sigma0)
    sign = 1.0 if minimize else -1.0

    if pop_size is not None and pop_size < 2:
        # pop_size 1 -> mu = 1//2 = 0 -> empty recombination weights -> mu_eff = 1/0
        raise ValueError("pop_size must be >= 2 (CMA-ES needs >= 2 for weighted recombination).")
    lam = pop_size or (4 + int(math.floor(3 * math.log(n))))
    mu = lam // 2
    weights = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
    wsum = math.fsum(weights)
    weights = [w / wsum for w in weights]
    mu_eff = 1.0 / math.fsum(w * w for w in weights)

    cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    cs = (mu_eff + 2) / (n + mu_eff + 5)
    c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    # separable speed-up: inflate the covariance learning rates (Ros & Hansen)
    scale = (n + 2) / 3.0
    c1 = min(1.0, c1 * scale)
    cmu = min(1 - c1, cmu * scale)
    damps = 1 + 2 * max(0.0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
    chi_n = math.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))

    C = [1.0] * n                                            # diagonal covariance
    ps = [0.0] * n
    pc = [0.0] * n
    hist: List[float] = []
    best_x, best_f = list(mean), sign * float(f(mean))
    nfev = 1
    for gen in range(n_iter):
        sqrtC = [math.sqrt(ci) for ci in C]
        samples = []
        for _ in range(lam):
            z = [rng.gauss(0, 1) for _ in range(n)]
            y = [sqrtC[i] * z[i] for i in range(n)]
            x = [mean[i] + sigma * y[i] for i in range(n)]
            x = _project_box(x, bounds)
            fv = sign * float(f(x))
            nfev += 1
            samples.append((fv, y, x))
        samples.sort(key=lambda t: t[0])
        if samples[0][0] < best_f:
            best_f, best_x = samples[0][0], list(samples[0][2])
        hist.append(sign * best_f)

        # recombination of the mu best mutation steps y
        yw = [math.fsum(weights[k] * samples[k][1][i] for k in range(mu))
              for i in range(n)]
        mean = [mean[i] + sigma * yw[i] for i in range(n)]
        mean = _project_box(mean, bounds)

        # step-size path (uses C^{-1/2} = diag(1/sqrt(C)) in the separable case)
        cinv_yw = [yw[i] / sqrtC[i] for i in range(n)]
        ps = [(1 - cs) * ps[i] + math.sqrt(cs * (2 - cs) * mu_eff) * cinv_yw[i]
              for i in range(n)]
        ps_norm = _norm(ps)
        hsig = 1.0 if (ps_norm / math.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
                       / chi_n < 1.4 + 2.0 / (n + 1)) else 0.0
        pc = [(1 - cc) * pc[i] + hsig * math.sqrt(cc * (2 - cc) * mu_eff) * yw[i]
              for i in range(n)]

        # rank-one + rank-mu update of the diagonal covariance
        for i in range(n):
            rank_mu = math.fsum(weights[k] * samples[k][1][i] ** 2 for k in range(mu))
            C[i] = ((1 - c1 - cmu) * C[i]
                    + c1 * (pc[i] * pc[i] + (1 - hsig) * cc * (2 - cc) * C[i])
                    + cmu * rank_mu)
            C[i] = max(C[i], _EPS)
        sigma *= math.exp((cs / damps) * (ps_norm / chi_n - 1))
        if sigma * max(sqrtC) < tol or not math.isfinite(sigma):
            return OptResult(best_x, sign * best_f, gen + 1, nfev, True,
                             "step size below tol", hist, {"sigma": sigma})
    return OptResult(best_x, sign * best_f, n_iter, nfev, False,
                     "max iterations reached", hist, {"sigma": sigma})


def evolution_strategy(f: Callable[[Vector], float], x0: Sequence[float], *,
                       sigma: float = 0.1, pop_size: int = 50, lr: float = 0.05,
                       n_iter: int = 200, minimize: bool = True, antithetic: bool = True,
                       bounds: Optional[Sequence[Tuple[float, float]]] = None,
                       seed: Any = 0) -> OptResult:
    """Natural Evolution Strategy (the "OpenAI-ES" estimator): estimate the
    gradient of the Gaussian-smoothed objective from a population of perturbations
    and take an ascent/descent step.  Uses ANTITHETIC (mirrored) sampling and
    fitness-rank normalisation for a low-variance, scale-free update."""
    rng = _as_rng(seed)
    n = len(x0)
    x = _vec(x0)
    sign = 1.0 if minimize else -1.0
    hist: List[float] = []
    nfev = 0
    best_x, best_f = list(x), sign * float(f(x))
    nfev += 1
    if pop_size < 2:
        # antithetic pop_size 1 -> half = 0 -> empty population -> grad = 0/(0*sigma)
        raise ValueError("pop_size must be >= 2 (a gradient estimate needs >= 2 perturbations).")
    half = pop_size // 2 if antithetic else pop_size
    for it in range(n_iter):
        eps = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(half)]
        if antithetic:
            eps = eps + [[-e for e in row] for row in eps]
        scored = []
        for row in eps:
            cand = _project_box([x[i] + sigma * row[i] for i in range(n)], bounds)
            scored.append((sign * float(f(cand)), row))
            nfev += 1
        # rank transform fitness to [-0.5, 0.5] (minimisation -> lower is better)
        order = sorted(range(len(scored)), key=lambda k: scored[k][0])
        ranks = [0.0] * len(scored)
        m = len(scored)
        for r, k in enumerate(order):
            ranks[k] = (r / (m - 1) - 0.5) if m > 1 else 0.0
        # gradient estimate: sum_k rank_k * eps_k / (pop * sigma); descend it
        grad = [0.0] * n
        for k in range(m):
            w = ranks[k]
            for i in range(n):
                grad[i] += w * scored[k][1][i]
        grad = [g / (m * sigma) for g in grad]
        x = _project_box([x[i] - lr * grad[i] for i in range(n)], bounds)
        fv = sign * float(f(x))
        nfev += 1
        if fv < best_f:
            best_f, best_x = fv, list(x)
        hist.append(sign * best_f)
    return OptResult(best_x, sign * best_f, n_iter, nfev, False,
                     "max iterations reached", hist)


# ===========================================================================
# 5. SAMPLING & SCENARIOS
# ===========================================================================
def monte_carlo_estimate(sample_fn: Callable[[random.Random], float], n: int, *,
                         confidence: float = 0.95, seed: Any = 0) -> Dict[str, float]:
    """Estimate ``E[sample_fn]`` by Monte Carlo.  Returns the mean, sample
    variance, standard error and a normal-approximation confidence interval."""
    if n < 2:
        raise ValueError("need n >= 2 samples.")
    rng = _as_rng(seed)
    xs = [float(sample_fn(rng)) for _ in range(n)]
    mean = _mean(xs)
    var = _variance(xs)
    se = math.sqrt(var / n)
    z = _z_score(confidence)
    return {"mean": mean, "variance": var, "std_error": se, "n": n,
            "ci_low": mean - z * se, "ci_high": mean + z * se, "confidence": confidence}


def control_variate_estimate(pairs: Sequence[Tuple[float, float]],
                             control_mean: float) -> Dict[str, float]:
    """Variance-reduced mean of ``Y`` using a control variate ``X`` with known
    mean.  Estimates ``Y* = Y - b (X - E[X])`` with the optimal ``b = cov/var``,
    which never increases variance and usually shrinks it substantially."""
    ys = [p[0] for p in pairs]
    xs = [p[1] for p in pairs]
    n = len(ys)
    if n < 2:
        raise ValueError("need >= 2 pairs.")
    xbar, ybar = _mean(xs), _mean(ys)
    cov = math.fsum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / (n - 1)
    varx = _variance(xs)
    b = cov / varx if varx > _EPS else 0.0
    adj = [y - b * (x - control_mean) for x, y in zip(xs, ys)]
    mean = _mean(adj)
    var = _variance(adj)
    return {"mean": mean, "variance": var, "std_error": math.sqrt(var / n),
            "beta": b, "plain_mean": ybar, "plain_variance": _variance(ys)}


def importance_sampling_estimate(
        samples: Sequence[float], weights: Sequence[float], *,
        normalized: bool = False) -> Dict[str, float]:
    """Importance-sampling estimate of ``E_p[h]`` from samples drawn under ``q``
    with likelihood ratios ``w = p/q``.  ``normalized=True`` uses the
    self-normalised estimator (for unnormalised ``p``)."""
    if len(samples) != len(weights):
        raise ValueError("samples and weights must align.")
    n = len(samples)
    if n < 1:
        raise ValueError("need >= 1 sample.")
    # a likelihood ratio p/q is a ratio of densities and so is always >= 0; a negative
    # weight signals a buggy p/q, and would otherwise yield a silently-wrong estimate
    # and a meaningless (possibly negative) effective sample size.
    if any(w < 0 for w in weights):
        raise ValueError("importance weights (likelihood ratios p/q) must be >= 0.")
    wh = [w * h for w, h in zip(weights, samples)]
    if normalized:
        wsum = math.fsum(weights)
        if abs(wsum) < _EPS:
            raise ValueError("weights sum to ~0.")
        mean = math.fsum(wh) / wsum
        ess = (wsum ** 2) / math.fsum(w * w for w in weights)
    else:
        mean = math.fsum(wh) / n
        ess = (math.fsum(weights) ** 2) / max(_EPS, math.fsum(w * w for w in weights))
    return {"mean": mean, "effective_sample_size": ess, "n": n}


def latin_hypercube(n: int, dim: int, *, seed: Any = 0) -> List[Vector]:
    """Latin Hypercube design of ``n`` points in the unit cube ``[0,1]^dim``.
    Each axis is stratified into ``n`` equal bins with one sample per bin, giving
    far better coverage than plain Monte-Carlo for the same ``n``."""
    if n < 1 or dim < 1:
        raise ValueError("n and dim must be >= 1.")
    rng = _as_rng(seed)
    pts = [[0.0] * dim for _ in range(n)]
    for d in range(dim):
        perm = list(range(n))
        rng.shuffle(perm)
        for i in range(n):
            pts[i][d] = (perm[i] + rng.random()) / n
    return pts


def van_der_corput(n: int, base: int = 2) -> List[float]:
    """The van der Corput low-discrepancy sequence in ``base`` -- the 1-D building
    block of Halton."""
    if base < 2:
        raise ValueError("base must be >= 2.")
    out = []
    for i in range(1, n + 1):
        x, f, k = 0.0, 1.0 / base, i
        while k > 0:
            x += (k % base) * f
            k //= base
            f /= base
        out.append(x)
    return out


_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
           67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]


def halton_sequence(n: int, dim: int) -> List[Vector]:
    """Halton low-discrepancy sequence: a van der Corput sequence per dimension in
    the first ``dim`` prime bases.  Deterministic, seed-free quasi-Monte-Carlo."""
    if n < 1 or dim < 1:
        raise ValueError("n and dim must be >= 1.")
    if dim > len(_PRIMES):
        raise ValueError("halton supports up to %d dimensions." % len(_PRIMES))
    cols = [van_der_corput(n, _PRIMES[d]) for d in range(dim)]
    return [[cols[d][i] for d in range(dim)] for i in range(n)]


# Sobol direction numbers (Bratley & Fox, ACM TOMS 659) for the first dimensions.
# Each entry: (polynomial-degree s, a-coefficient bits, initial m_1..m_s).
_SOBOL_DIRS = [
    (1, 0, [1]),
    (2, 1, [1, 1]),
    (3, 1, [1, 3, 7]),
    (3, 2, [1, 1, 5]),
    (4, 1, [1, 3, 1, 1]),
    (4, 4, [1, 1, 3, 7]),
    (5, 2, [1, 3, 3, 9, 9]),
    (5, 13, [1, 3, 7, 13, 3]),
    (5, 7, [1, 1, 5, 11, 27]),
    (5, 14, [1, 3, 5, 1, 15]),
    (6, 1, [1, 1, 7, 3, 29, 51]),
]
_SOBOL_BITS = 30


def sobol_sequence(n: int, dim: int) -> List[Vector]:
    """Sobol' low-discrepancy sequence via the Gray-code construction with the
    classic Bratley-Fox direction numbers.  Excellent uniformity for
    quasi-Monte-Carlo integration; supports up to %d + 1 dimensions.""" \
        % (len(_SOBOL_DIRS))
    if dim < 1:
        raise ValueError("dim must be >= 1.")
    if dim > len(_SOBOL_DIRS) + 1:
        raise ValueError("sobol supports up to %d dimensions (extend the direction "
                         "number table for more)." % (len(_SOBOL_DIRS) + 1))
    bits = _SOBOL_BITS
    # build direction integers v[d][k], k = 1..bits
    V: List[List[int]] = []
    # dimension 0: van der Corput in base 2 -> v_k = 2^(bits-k)
    V.append([0] + [1 << (bits - k) for k in range(1, bits + 1)])
    for d in range(1, dim):
        s, a, m = _SOBOL_DIRS[d - 1]
        m = list(m)
        while len(m) < bits:
            k = len(m)                                       # 0-based index -> m_{k+1}
            val = m[k - s]                                   # m_{k-s+1}
            val ^= (m[k - s] << s)
            for j in range(1, s):
                if (a >> (s - 1 - j)) & 1:
                    val ^= (m[k - j] << j)
            m.append(val)
        V.append([0] + [m[k - 1] << (bits - k) for k in range(1, bits + 1)])
    points: List[Vector] = []
    x = [0] * dim
    # Emit the CANONICAL sequence, indices 0 .. n-1 (starting at the origin): for a
    # block of n = 2^m points this is a genuine (0, m, dim)-net -- exactly one point
    # per elementary dyadic box, the defining low-discrepancy property.  (Skipping the
    # origin, i.e. emitting indices 1..n, would silently break the net.)
    for i in range(n):
        points.append([x[d] / (1 << bits) for d in range(dim)])
        c = 1                                                # lowest zero bit of i (1-based)
        g = i
        while g & 1:
            g >>= 1
            c += 1
        for d in range(dim):
            x[d] ^= V[d][c]
    return points


def antithetic_normal(n: int, dim: int = 1, *, seed: Any = 0) -> List[Vector]:
    """``n`` standard-normal vectors using antithetic pairs (``z`` and ``-z``),
    which cancels the odd part of a smooth integrand and cuts variance for free."""
    if n < 1 or dim < 1:
        raise ValueError("n and dim must be >= 1.")
    rng = _as_rng(seed)
    out: List[Vector] = []
    for _ in range((n + 1) // 2):
        z = [rng.gauss(0, 1) for _ in range(dim)]
        out.append(z)
        out.append([-v for v in z])
    return out[:n]


def stratified_uniform(n: int, *, seed: Any = 0) -> List[float]:
    """``n`` stratified uniforms on ``[0,1]``: one draw per equal sub-interval,
    a 1-D variance-reduction staple."""
    if n < 1:
        raise ValueError("n must be >= 1.")
    rng = _as_rng(seed)
    return [(i + rng.random()) / n for i in range(n)]


def scenario_reduction(scenarios: Sequence[Sequence[float]],
                       probabilities: Optional[Sequence[float]], k: int, *,
                       metric: Optional[Callable[[Sequence[float],
                                                   Sequence[float]], float]] = None
                       ) -> Dict[str, Any]:
    """Reduce a discrete scenario set to ``k`` representatives by fast-forward
    selection (Dupacova, Growe-Kuska & Roemisch 2003), greedily minimising the
    Kantorovich (Wasserstein) distance to the original distribution.  Deleted
    scenarios' probabilities are redistributed to their nearest kept scenario."""
    m = len(scenarios)
    if not 1 <= k <= m:
        raise ValueError("k must be in [1, number of scenarios].")
    if probabilities is None:
        probabilities = [1.0 / m] * m
    if len(probabilities) != m:
        raise ValueError("probabilities must align with scenarios.")
    if any(p < 0.0 for p in probabilities):
        # a negative mass summing to 1 with the others used to pass the sum check and
        # silently corrupt the Kantorovich redistribution.
        raise ValueError("probabilities must be non-negative.")
    if abs(math.fsum(probabilities) - 1.0) > 1e-6:
        raise ValueError("probabilities must sum to 1.")
    if metric is None:
        def metric(a, b):
            return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b)))
    D = [[metric(scenarios[i], scenarios[j]) for j in range(m)] for i in range(m)]

    kept: List[int] = []
    remaining = set(range(m))
    # best distance of each scenario to the currently-selected set (inf initially)
    nearest = [math.inf] * m
    for _ in range(k):
        best_u, best_cost = None, math.inf
        for u in remaining:
            cost = 0.0
            for j in remaining:
                if j == u:
                    continue
                cost += probabilities[j] * min(nearest[j], D[j][u])
            if cost < best_cost:
                best_cost, best_u = cost, u
        kept.append(best_u)
        remaining.discard(best_u)
        for j in range(m):
            nearest[j] = min(nearest[j], D[j][best_u])

    new_probs = {i: probabilities[i] for i in kept}
    for j in range(m):
        if j in new_probs:
            continue
        closest = min(kept, key=lambda i: D[j][i])
        new_probs[closest] += probabilities[j]
    reduced = [(list(scenarios[i]), new_probs[i]) for i in kept]
    kant = math.fsum(probabilities[j] * min(D[j][i] for i in kept) for j in range(m))
    return {"scenarios": [r[0] for r in reduced],
            "probabilities": [r[1] for r in reduced],
            "kept_indices": kept, "kantorovich_distance": kant}


# ===========================================================================
# 6. STOCHASTIC PROGRAMMING
# ===========================================================================
def sample_average_approximation(
        sample_objective: Callable[[Vector, Any], float], x0: Sequence[float],
        scenario_sampler: Callable[[random.Random], Any], *, n_scenarios: int = 200,
        inner: str = "cma", bounds: Optional[Sequence[Tuple[float, float]]] = None,
        n_iter: int = 200, n_eval: int = 2000, seed: Any = 0,
        **inner_kwargs: Any) -> OptResult:
    """Sample Average Approximation (SAA): approximate ``min_x E_xi[F(x, xi)]`` by
    the sample-average ``(1/N) sum_i F(x, xi_i)`` over ``N`` fixed scenarios, and
    minimise it with a derivative-free ``inner`` solver (``"cma"``, ``"cem"`` or
    ``"spsa"``).  The returned ``fun`` is an INDEPENDENT out-of-sample estimate of
    the true objective (with a standard error in ``meta``)."""
    rng = _as_rng(seed)
    scenarios = [scenario_sampler(rng) for _ in range(n_scenarios)]

    def saa_obj(x: Vector) -> float:
        return _mean([sample_objective(x, s) for s in scenarios])

    if inner == "cma":
        res = sep_cma_es(saa_obj, x0, n_iter=n_iter, bounds=bounds,
                         seed=rng.randrange(1 << 30), **inner_kwargs)
    elif inner == "cem":
        res = cross_entropy_method(saa_obj, x0, n_iter=n_iter, bounds=bounds,
                                   seed=rng.randrange(1 << 30), **inner_kwargs)
    elif inner == "spsa":
        res = spsa(saa_obj, x0, n_iter=n_iter, bounds=bounds,
                   seed=rng.randrange(1 << 30), **inner_kwargs)
    else:
        raise ValueError("inner must be 'cma', 'cem' or 'spsa'.")

    # out-of-sample validation with fresh scenarios
    val = [sample_objective(res.x, scenario_sampler(rng)) for _ in range(n_eval)]
    mean, se = _mean(val), math.sqrt(_variance(val) / n_eval)
    res.fun = mean
    res.meta.update({"in_sample_obj": saa_obj(res.x), "std_error": se,
                     "n_scenarios": n_scenarios, "n_eval": n_eval})
    res.message = "SAA solved; out-of-sample estimate reported"
    return res


def two_stage_saa(first_stage_cost: Sequence[float],
                  recourse: Callable[[Vector, Any], float], x0: Sequence[float],
                  scenario_sampler: Callable[[random.Random], Any], *,
                  bounds: Optional[Sequence[Tuple[float, float]]] = None,
                  n_scenarios: int = 200, n_iter: int = 200, seed: Any = 0,
                  **inner_kwargs: Any) -> OptResult:
    """Two-stage stochastic linear/convex program with recourse:
    ``min_x  c^T x + E_xi[ Q(x, xi) ]``, where ``recourse(x, xi) = Q(x, xi)`` is the
    (already optimised) second-stage cost supplied by the caller.  Solved by SAA
    with a derivative-free inner optimiser -- so ``Q`` may be any oracle, LP or not."""
    c = _vec(first_stage_cost)

    def obj(x: Vector, xi: Any) -> float:
        return _dot(c, x) + recourse(x, xi)

    return sample_average_approximation(obj, x0, scenario_sampler,
                                        n_scenarios=n_scenarios, bounds=bounds,
                                        n_iter=n_iter, seed=seed, **inner_kwargs)


def newsvendor(demand_samples: Sequence[float], *, unit_cost: float,
               unit_price: float, salvage: float = 0.0,
               penalty: float = 0.0) -> Dict[str, float]:
    """The newsvendor problem -- the canonical single-stage stochastic program.
    Order ``q`` before observing demand ``D``; the optimum is the critical-fractile
    quantile ``q* = F^{-1}(cu / (cu + co))`` with underage ``cu = price - cost +
    penalty`` and overage ``co = cost - salvage``.  Returns the optimal order
    quantity (empirical quantile of the demand samples) and the expected profit."""
    if not (unit_price > unit_cost >= salvage >= 0):
        raise ValueError("need price > cost >= salvage >= 0.")
    cu = unit_price - unit_cost + penalty
    co = unit_cost - salvage
    crit = cu / (cu + co)
    d = sorted(float(x) for x in demand_samples)
    n = len(d)
    if n == 0:
        raise ValueError("need demand samples.")
    # empirical quantile (upper) at the critical ratio
    idx = min(n - 1, max(0, int(math.ceil(crit * n)) - 1))
    q = d[idx]

    def profit(order: float) -> float:
        vals = []
        for dem in d:
            sold = min(order, dem)
            leftover = max(0.0, order - dem)
            short = max(0.0, dem - order)
            vals.append(unit_price * sold + salvage * leftover
                        - unit_cost * order - penalty * short)
        return _mean(vals)

    return {"order_quantity": q, "critical_ratio": crit,
            "expected_profit": profit(q), "underage_cost": cu, "overage_cost": co}


def stochastic_solution_metrics(
        scenarios: Sequence[Any], probabilities: Optional[Sequence[float]],
        solve_scenario: Callable[[Any], Tuple[Vector, float]],
        evaluate: Callable[[Vector, Any], float], mean_scenario: Any,
        stochastic_solution: Vector) -> Dict[str, float]:
    """Compute the classic value-of-information metrics for a (minimisation)
    two-stage program:

    * WS  -- wait-and-see: ``E_xi[ min_x f(x, xi) ]`` (perfect information).
    * RP  -- recourse problem: ``E_xi[ f(x_RP, xi) ]`` at the given stochastic sol.
    * EEV -- expected result of using the expected-value (mean-scenario) solution.
    * EVPI = RP - WS  (>= 0): the price of not knowing the future.
    * VSS  = EEV - RP (>= 0): what modelling the uncertainty buys over a mean plan.
    """
    m = len(scenarios)
    if probabilities is None:
        probabilities = [1.0 / m] * m
    ws = math.fsum(p * solve_scenario(s)[1] for p, s in zip(probabilities, scenarios))
    rp = math.fsum(p * evaluate(stochastic_solution, s)
                   for p, s in zip(probabilities, scenarios))
    x_ev = solve_scenario(mean_scenario)[0]
    eev = math.fsum(p * evaluate(x_ev, s) for p, s in zip(probabilities, scenarios))
    return {"WS": ws, "RP": rp, "EEV": eev, "EVPI": rp - ws, "VSS": eev - rp,
            "expected_value_solution": x_ev}


# ===========================================================================
# 7. RISK-AVERSE OPTIMISATION
# ===========================================================================
def value_at_risk(losses: Sequence[float], alpha: float = 0.95) -> float:
    """Empirical Value-at-Risk at level ``alpha``: the ``alpha``-quantile of the
    loss distribution (the loss not exceeded with probability ``alpha``)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    xs = sorted(float(x) for x in losses)
    n = len(xs)
    if n == 0:
        raise ValueError("need losses.")
    idx = min(n - 1, max(0, int(math.ceil(alpha * n)) - 1))
    return xs[idx]


def conditional_value_at_risk(losses: Sequence[float], alpha: float = 0.95) -> float:
    """Empirical Conditional VaR / expected shortfall at ``alpha`` -- the mean loss in
    the worst ``1 - alpha`` tail.  A coherent risk measure, unlike VaR.

    Uses the exact Rockafellar-Uryasev value for the empirical distribution,
    ``CVaR = VaR + (1 / ((1-alpha) n)) * sum (L_i - VaR)+``, which is what
    :func:`minimize_cvar` optimises.  (The naive "mean of the losses >= VaR" divides by
    the tail COUNT, not by ``(1-alpha) n``, so it double-counts the boundary and biases
    the estimate low whenever ``(1-alpha) n`` is not an integer.)"""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    xs = sorted(float(x) for x in losses)
    n = len(xs)
    if n == 0:
        raise ValueError("need losses.")
    var = value_at_risk(xs, alpha)
    excess = math.fsum(max(0.0, x - var) for x in xs)
    return var + excess / ((1 - alpha) * n)


def entropic_risk(losses: Sequence[float], theta: float = 1.0) -> float:
    """Entropic (exponential) risk measure ``(1/theta) log E[exp(theta L)]`` -- a
    convex, risk-averse certainty equivalent; ``theta -> 0`` recovers the mean."""
    if theta <= 0:
        raise ValueError("theta must be > 0.")
    xs = [float(x) for x in losses]
    if not xs:                          # match value_at_risk/CVaR, not a raw max() error
        raise ValueError("need losses.")
    mx = max(xs)                                             # log-sum-exp for stability:
    # E[e^{theta L}] = e^{theta*mx} * mean_i e^{theta (x_i - mx)}, so the shift that is
    # factored out is theta*mx (NOT mx) -- dividing the log by theta then leaves mx.
    lse = theta * mx + math.log(_mean([math.exp(theta * (x - mx)) for x in xs]))
    return lse / theta


def mean_variance_objective(returns: Sequence[float], risk_aversion: float = 1.0
                            ) -> float:
    """Markowitz mean-variance objective ``E[R] - lambda Var[R]`` for a sample of
    returns (higher is better)."""
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be >= 0.")
    returns = list(returns)
    if not returns:                     # else _mean/_variance silently return 0.0
        raise ValueError("need returns.")
    return _mean(returns) - risk_aversion * _variance(returns)


def minimize_cvar(loss: Callable[[Vector, Any], float], x0: Sequence[float],
                  scenario_sampler: Callable[[random.Random], Any], *,
                  alpha: float = 0.95, n_scenarios: int = 500, lr: float = 0.05,
                  n_iter: int = 500,
                  loss_grad: Optional[Callable[[Vector, Any], Sequence[float]]] = None,
                  bounds: Optional[Sequence[Tuple[float, float]]] = None,
                  seed: Any = 0) -> OptResult:
    """Minimise the Conditional VaR of a loss ``L(x, xi)`` via the Rockafellar-
    Uryasev formulation

        ``CVaR_alpha = min_{x, eta}  eta + 1/(1-alpha) E[(L(x, xi) - eta)+]``,

    which is JOINTLY CONVEX in ``(x, eta)`` when ``L`` is convex in ``x``.  Once the
    scenario set is drawn this is a deterministic convex program, so it is solved by
    FULL-BATCH subgradient descent over ``(x, eta)`` using every scenario each step.
    (A one-scenario-per-step version only updates ``x`` on the ``1-alpha`` fraction of
    "active" tail scenarios, so at the usual ``alpha=0.95/0.99`` it converges an order
    of magnitude too slowly.)  Supply ``loss_grad`` for an exact subgradient, else it
    is estimated by finite differences (only over the active tail, so the cost stays
    proportional to ``(1-alpha) n``)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    rng = _as_rng(seed)
    scenarios = [scenario_sampler(rng) for _ in range(n_scenarios)]
    x = _vec(x0)
    dim = len(x)
    inv = 1.0 / (1.0 - alpha)
    eta = value_at_risk([loss(x, s) for s in scenarios], alpha)
    hist: List[float] = []
    nfev = n_scenarios
    opt_x = Adam(lr=lr)
    for t in range(n_iter):
        losses = [float(loss(x, s)) for s in scenarios]
        nfev += n_scenarios
        active = [i for i in range(n_scenarios) if losses[i] > eta]
        # full-batch subgradient of  eta + inv * mean_i (L_i - eta)+
        g_eta = 1.0 - inv * (len(active) / n_scenarios)     # d/d eta (exact)
        eta -= lr * g_eta
        if active:                                          # d/dx = inv * mean active grad
            gx = [0.0] * dim
            for i in active:
                s = scenarios[i]
                if loss_grad is not None:
                    gi = loss_grad(x, s)
                else:
                    gi = finite_difference_gradient(lambda z: loss(z, s), x,
                                                    central=False)
                    nfev += dim
                gx = _add(gx, gi)
            gx = _scale(gx, inv / n_scenarios)
            x = opt_x.step(x, gx)
            x = _project_box(x, bounds)
        hist.append(_cvar_value(loss, x, eta, scenarios, alpha))
    final = [loss(x, s) for s in scenarios]
    return OptResult(x, conditional_value_at_risk(final, alpha), n_iter, nfev, False,
                     "CVaR minimised", hist,
                     {"eta": eta, "alpha": alpha, "var": value_at_risk(final, alpha)})


def _cvar_value(loss, x, eta, scenarios, alpha) -> float:
    inv = 1.0 / (1.0 - alpha)
    return eta + inv * _mean([max(0.0, loss(x, s) - eta) for s in scenarios])


# ===========================================================================
# 8. STOCHASTIC BANDITS  (online stochastic optimisation)
# ===========================================================================
class _BanditPolicy:
    def select(self) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        raise NotImplementedError


class EpsilonGreedy(_BanditPolicy):
    """Epsilon-greedy: exploit the best-mean arm with probability ``1 - epsilon``,
    explore uniformly otherwise.  ``decay`` shrinks epsilon as ``eps/(1+decay*t)``."""

    def __init__(self, n_arms: int, epsilon: float = 0.1, decay: float = 0.0,
                 seed: Any = 0):
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        self.n = n_arms
        self.eps0 = float(epsilon)
        self.decay = float(decay)
        self.rng = _as_rng(seed)
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.t = 0

    def select(self) -> int:
        eps = self.eps0 / (1 + self.decay * self.t)
        if self.rng.random() < eps or min(self.counts) == 0:
            # explore (and guarantee each arm is tried once)
            untried = [a for a in range(self.n) if self.counts[a] == 0]
            return self.rng.choice(untried) if untried else self.rng.randrange(self.n)
        best = max(self.values)
        return self.rng.choice([a for a in range(self.n) if self.values[a] == best])

    def update(self, arm: int, reward: float) -> None:
        self.t += 1
        self.counts[arm] += 1
        c = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / c


class UCB1(_BanditPolicy):
    """UCB1 (Auer, Cesa-Bianchi & Fischer 2002): pick the arm maximising
    ``mean + c sqrt(2 ln t / n_arm)`` -- optimism under uncertainty, with
    logarithmic regret and no tuning beyond the exploration constant ``c``."""

    def __init__(self, n_arms: int, c: float = 1.0):
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1.")
        self.n = n_arms
        self.c = float(c)
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.t = 0

    def select(self) -> int:
        for a in range(self.n):
            if self.counts[a] == 0:
                return a                                     # try each arm once first
        return max(range(self.n), key=lambda a: self.values[a]
                   + self.c * math.sqrt(2 * math.log(self.t) / self.counts[a]))

    def update(self, arm: int, reward: float) -> None:
        self.t += 1
        self.counts[arm] += 1
        c = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / c


class ThompsonSampling(_BanditPolicy):
    """Thompson sampling: keep a posterior per arm and pick the arm with the
    highest sampled mean.  ``bernoulli=True`` uses a Beta(1,1) conjugate prior for
    0/1 rewards; otherwise a Normal-inverse-gamma posterior for real rewards."""

    def __init__(self, n_arms: int, bernoulli: bool = True, seed: Any = 0):
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1.")
        self.n = n_arms
        self.bernoulli = bool(bernoulli)
        self.rng = _as_rng(seed)
        if bernoulli:
            self.alpha = [1.0] * n_arms
            self.beta = [1.0] * n_arms
        else:
            self.counts = [0] * n_arms
            self.means = [0.0] * n_arms
            self.m2 = [0.0] * n_arms                         # sum of squared deviations

    def select(self) -> int:
        if self.bernoulli:
            samples = [self.rng.betavariate(self.alpha[a], self.beta[a])
                       for a in range(self.n)]
        else:
            samples = []
            for a in range(self.n):
                if self.counts[a] < 2:
                    samples.append(self.rng.gauss(self.means[a], 1.0))
                else:
                    var = self.m2[a] / (self.counts[a] - 1)
                    se = math.sqrt(max(_EPS, var) / self.counts[a])
                    samples.append(self.rng.gauss(self.means[a], se))
        return max(range(self.n), key=lambda a: samples[a])

    def update(self, arm: int, reward: float) -> None:
        if self.bernoulli:
            if not 0.0 <= reward <= 1.0:
                raise ValueError("bernoulli rewards must be in [0, 1].")
            self.alpha[arm] += reward
            self.beta[arm] += 1.0 - reward
        else:
            self.counts[arm] += 1
            delta = reward - self.means[arm]
            self.means[arm] += delta / self.counts[arm]
            self.m2[arm] += delta * (reward - self.means[arm])


def run_bandit(policy: _BanditPolicy, reward_fn: Callable[[int, random.Random], float],
               n_rounds: int, *, optimal_mean: Optional[float] = None,
               seed: Any = 0) -> Dict[str, Any]:
    """Run a bandit ``policy`` against a stochastic ``reward_fn(arm, rng)`` for
    ``n_rounds``.  Returns the pulls, total/mean reward and -- if ``optimal_mean``
    is known -- the cumulative regret trace."""
    rng = _as_rng(seed)
    rewards: List[float] = []
    arms: List[int] = []
    regret: List[float] = []
    cum_regret = 0.0
    for _ in range(n_rounds):
        a = policy.select()
        r = float(reward_fn(a, rng))
        policy.update(a, r)
        rewards.append(r)
        arms.append(a)
        if optimal_mean is not None:
            cum_regret += optimal_mean - r
            regret.append(cum_regret)
    return {"rewards": rewards, "arms": arms, "total_reward": math.fsum(rewards),
            "mean_reward": _mean(rewards), "cumulative_regret": regret,
            "pulls": [arms.count(a) for a in range(getattr(policy, "n", 0))]}


# ===========================================================================
# helpers
# ===========================================================================
def _z_score(confidence: float) -> float:
    """Two-sided normal critical value for a confidence level (rational approx)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")
    p = 1 - (1 - confidence) / 2
    return _inv_norm_cdf(p)


def _inv_norm_cdf(p: float) -> float:
    """Acklam's rational approximation to the standard-normal inverse CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1).")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ===========================================================================
# 9. DEMO
# ===========================================================================
def _demo() -> None:
    sep = "=" * 70
    print(sep)
    print("  STOCHASTIC OPTIMISATION (pure Python) - demo")
    print(sep)

    # --- stochastic gradient descent on a noisy quadratic -------------------
    # minimise f(x) = 0.5 ||x - x*||^2 with noisy gradients
    target = [3.0, -2.0, 1.0]
    rng = random.Random(0)

    def noisy_grad(x):
        return [(xi - ti) + rng.gauss(0, 0.5) for xi, ti in zip(x, target)]

    def obj(x):
        return 0.5 * sum((xi - ti) ** 2 for xi, ti in zip(x, target))

    print("\n-- stochastic first-order optimisers (noisy quadratic) --")
    for name, opt in (("SGD+mom", SGD(lr=0.05, momentum=0.9)),
                      ("AdaGrad", AdaGrad(lr=0.5)), ("RMSProp", RMSProp(lr=0.1)),
                      ("Adam", Adam(lr=0.1)), ("Nadam", Nadam(lr=0.1))):
        res = minimize_sgd(noisy_grad, [0.0, 0.0, 0.0], optimizer=opt,
                           n_iter=2000, objective=obj, averaging=True)
        print("  %-9s x~=%s  f=%.4f" % (name, [round(v, 2) for v in res.x], res.fun))

    # --- derivative-free on the (noisy) Rosenbrock --------------------------
    def rosen(x):
        return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                   for i in range(len(x) - 1)) + random.Random(hash(tuple(
                       round(v, 6) for v in x)) & 0xffff).gauss(0, 0.01)

    print("\n-- derivative-free global search (noisy Rosenbrock, 4-D) --")
    x0 = [0.0] * 4
    for name, fn in (("CEM", cross_entropy_method), ("sep-CMA-ES", sep_cma_es),
                     ("NES", evolution_strategy)):
        res = fn(rosen, x0, seed=1) if fn is not evolution_strategy else \
            fn(rosen, x0, sigma=0.3, n_iter=300, seed=1)
        print("  %-11s x~=%s  f=%.4f" % (name, [round(v, 2) for v in res.x], res.fun))

    # --- SPSA -------------------------------------------------------------
    res = spsa(rosen, x0, n_iter=2000, a=0.05, c=0.1, seed=2)
    print("  %-11s x~=%s  f=%.4f" % ("SPSA", [round(v, 2) for v in res.x], res.fun))

    # --- quasi-Monte-Carlo sampling ---------------------------------------
    print("\n-- low-discrepancy sampling (estimate area of unit quarter-circle) --")
    for name, pts in (("Monte-Carlo", [[random.Random(i).random(),
                                        random.Random(i + 999).random()]
                                       for i in range(2000)]),
                      ("Halton", halton_sequence(2000, 2)),
                      ("Sobol", sobol_sequence(2000, 2)),
                      ("LatinHC", latin_hypercube(2000, 2, seed=3))):
        inside = sum(1 for p in pts if p[0] ** 2 + p[1] ** 2 <= 1)
        print("  %-12s pi ~= %.4f" % (name, 4 * inside / len(pts)))

    # --- newsvendor & risk ------------------------------------------------
    print("\n-- stochastic programming: newsvendor --")
    demand = [random.Random(i).gauss(100, 20) for i in range(5000)]
    nv = newsvendor(demand, unit_cost=5, unit_price=8, salvage=2)
    print("  order q*=%.1f (critical ratio %.2f)  expected profit=%.2f"
          % (nv["order_quantity"], nv["critical_ratio"], nv["expected_profit"]))

    print("\n-- risk measures on a loss sample --")
    losses = [random.Random(i).gauss(0, 1) for i in range(10000)]
    print("  VaR(95%%)=%.3f  CVaR(95%%)=%.3f  entropic(1.0)=%.3f"
          % (value_at_risk(losses, 0.95), conditional_value_at_risk(losses, 0.95),
             entropic_risk(losses, 1.0)))

    # --- bandits ----------------------------------------------------------
    print("\n-- stochastic bandits (Bernoulli arms, best mean 0.7) --")
    means = [0.3, 0.5, 0.7, 0.2]

    def reward(a, rng):
        return 1.0 if rng.random() < means[a] else 0.0

    for name, pol in (("eps-greedy", EpsilonGreedy(4, epsilon=0.1, seed=0)),
                      ("UCB1", UCB1(4, c=1.0)),
                      ("Thompson", ThompsonSampling(4, seed=0))):
        out = run_bandit(pol, reward, 2000, optimal_mean=0.7, seed=0)
        print("  %-11s mean reward=%.3f  final regret=%.1f  pulls=%s"
              % (name, out["mean_reward"], out["cumulative_regret"][-1], out["pulls"]))

    print("\n" + sep)
    print("  demo OK")
    print(sep)


if __name__ == "__main__":
    _demo()
