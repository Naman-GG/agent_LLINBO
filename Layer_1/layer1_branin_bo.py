"""
Layer 1 — Classical Bayesian Optimization on the Branin function.

No BO library (no scikit-optimize, no bayes_opt). We build the two pieces
that matter by hand:
  1. Surrogate model  -> sklearn's GaussianProcessRegressor gives us a
     predicted mean + uncertainty (std) at any point. This part is just
     curve fitting, not worth reimplementing.
  2. Acquisition function -> Expected Improvement (EI), written out
     explicitly below. THIS is the actual "explore vs exploit" decision,
     and the part worth understanding by hand.

The loop: EI picks the next point -> we "run the experiment" (evaluate
Branin there) -> refit the GP on all points so far -> repeat.
"""

from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. The objective function (the thing we're pretending is expensive)
# ---------------------------------------------------------------------------
# Branin function: a standard 2D Bayesian-optimization benchmark.
# Domain: x1 in [-5, 10], x2 in [0, 15]
# Three known global minima, true minimum value ~= 0.397887
# We treat it as a black box: the optimizer never sees this formula,
# only (x, f(x)) pairs it has "measured".
def branin(x1, x2):
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi, 6.0, 10.0, 1.0 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

BOUNDS = np.array([[-5.0, 10.0], [0.0, 15.0]])  # [x1_range, x2_range]
TRUE_MIN = 0.397887

# ---------------------------------------------------------------------------
# 2. Expected Improvement, written out explicitly
# ---------------------------------------------------------------------------
def expected_improvement(X_candidates, gp, y_best, xi=0.01):
    """
    X_candidates: (n, 2) array of points we're considering trying next.
    gp: fitted GaussianProcessRegressor.
    y_best: best (lowest, since we're minimizing) objective value seen so far.
    xi: small exploration bonus - without it EI can get stuck exploiting
        too early.

    Returns EI(x) for each candidate: the expected amount by which trying
    this point beats our current best, weighted by how likely that is.
    """
    mu, sigma = gp.predict(X_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)  # avoid divide-by-zero at already-sampled points

    # Minimizing, so "improvement" is how far below y_best we expect to land.
    improvement = y_best - mu - xi
    z = improvement / sigma

    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    return ei

# ---------------------------------------------------------------------------
# 3. The BO loop
# ---------------------------------------------------------------------------
def run_bo(n_init=5, n_iter=20, grid_res=80):
    # --- initial random design (Latin-hypercube-ish via simple uniform here) ---
    X = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(n_init, 2))
    y = branin(X[:, 0], X[:, 1])

    # --- candidate grid we'll score with EI at every iteration ---
    g1 = np.linspace(*BOUNDS[0], grid_res)
    g2 = np.linspace(*BOUNDS[1], grid_res)
    G1, G2 = np.meshgrid(g1, g2)
    candidates = np.column_stack([G1.ravel(), G2.ravel()])

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-12, 1e-2))

    best_so_far = [y.min()]
    chosen_points = []  # points selected BY the acquisition function (post-init)

    for i in range(n_iter):
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=SEED
        )
        gp.fit(X, y)

        ei = expected_improvement(candidates, gp, y_best=y.min())
        next_x = candidates[np.argmax(ei)]

        next_y = branin(next_x[0], next_x[1])

        X = np.vstack([X, next_x])
        y = np.append(y, next_y)
        chosen_points.append(next_x)
        best_so_far.append(y.min())

        print(f"iter {i+1:2d}: tried x=({next_x[0]:6.2f}, {next_x[1]:6.2f})  "
              f"f(x)={next_y:8.3f}  best so far={y.min():8.3f}")

    return X, y, np.array(chosen_points), np.array(best_so_far), (g1, g2, G1, G2)

# ---------------------------------------------------------------------------
# 4. Run it and plot
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n_init = 5
    X, y, chosen, best_so_far, (g1, g2, G1, G2) = run_bo(n_init=n_init, n_iter=35)

    Z_true = branin(G1, G2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- left: true Branin surface + where we sampled ---
    ax = axes[0]
    cf = ax.contourf(G1, G2, Z_true, levels=40, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="Branin f(x1, x2)")
    ax.scatter(X[:n_init, 0], X[:n_init, 1], c="white", edgecolor="black",
               s=70, marker="o", label="initial random points", zorder=3)
    ax.scatter(chosen[:, 0], chosen[:, 1], c=np.arange(len(chosen)), cmap="autumn",
               edgecolor="black", s=70, marker="D", label="EI-chosen points", zorder=4)
    best_idx = np.argmin(y)
    ax.scatter(*X[best_idx], c="red", marker="*", s=350, edgecolor="black",
               label="best found", zorder=5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Where Expected Improvement chose to sample")
    ax.legend(loc="upper right", fontsize=8)

    # --- right: convergence curve ---
    ax = axes[1]
    iters = np.arange(len(best_so_far))
    ax.plot(iters, best_so_far, marker="o", color="tab:blue", label="best value found")
    ax.axhline(TRUE_MIN, color="gray", linestyle="--", label=f"true global min ({TRUE_MIN:.3f})")
    ax.axvline(n_init - 0.5, color="black", linestyle=":", alpha=0.5)
    ax.text(n_init - 0.5, ax.get_ylim()[1] * 0.9, " BO starts", fontsize=8, ha="left")
    ax.set_xlabel("iteration (0-4 = random init, 5+ = EI-chosen)")
    ax.set_ylabel("best f(x) found so far")
    ax.set_title("Convergence toward the known optimum")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "layer1_bo_result.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFinal best value found: {y.min():.4f}  (true minimum: {TRUE_MIN:.4f})")
    print(f"Saved plot to {out_path}")
