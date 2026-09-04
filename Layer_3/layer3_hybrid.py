"""
Layer 3 -- Hybrid: LLM proposes candidate points, a real GP + Expected
Improvement acquisition function picks among them (LLAMBO-style).

Unlike Layer 2 (the LLM's choice IS the next point, trusted directly), here
the LLM only narrows down WHERE to look. The actual "which point wins"
decision is made by the same principled EI math as Layer 1. This tests
whether LLM-generated candidates give EI a better starting point than a
blind random/grid search would -- without ever trusting the LLM's judgment
on its own.

Loop per iteration:
    1. Fit a GP on the history so far (same as Layer 1)
    2. Ask the LLM for N_CANDIDATES diverse candidate points
    3. Add a pool of random filler candidates too, so EI always has
       options even if every LLM candidate is unusable
    4. Score ALL candidates (LLM + random) with real EI over the fitted GP
    5. Evaluate the real objective at whichever candidate scored highest
    6. Log whether EI picked an LLM candidate or a random filler --
       this win rate is the actual measure of whether the LLM helped

Usage:
    python3 layer3_hybrid.py
    python3 layer3_hybrid.py --dry-run
    python3 layer3_hybrid.py --n-candidates 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # BayOAgent/ (project root)

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

from Layer_1.layer1_branin_bo import branin, BOUNDS, TRUE_MIN, expected_improvement, run_bo
from Layer_2.layer2_llm_proposer import extract_json_objects, _try_parse_one, format_history

load_dotenv()

SEED = 42
MODEL = "openai/gpt-oss-120b"  # served via Groq -- currently a Groq preview model, not GA
MAX_TOKENS = 2000
REASONING_EFFORT = "medium"  # qwen3.6-27b only accepts "none" (off) or "default" (thinking mode on)
N_CANDIDATES = 5  # how many candidate points the LLM proposes per iteration

SYSTEM_PROMPT = f"""You are proposing CANDIDATE points to evaluate next in a black-box \
optimization problem. The domain is x1 in [-5, 10] and x2 in [0, 15]. \
LOWER values are BETTER (we are minimizing). Given the history of points \
tried and their measured values, propose {N_CANDIDATES} DIVERSE candidate points \
worth considering next -- a mix of points near the best region found so far \
and points in unexplored areas. A separate scoring step will pick the best \
one from your candidates, so give a genuinely varied set rather than \
{N_CANDIDATES} nearly identical points.

Respond with ONLY a JSON object, no other text, in exactly this form:
{{"candidates": [{{"x1": <float>, "x2": <float>, "reasoning": "<why this point>"}}, ...]}}
with exactly {N_CANDIDATES} entries in the list."""


def make_gp():
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-12, 1e-2))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=SEED)


def parse_candidates_response(text):
    """Pull {"candidates": [...]} out of the response, reusing Layer 2's
    robust multi-block extraction (tries every balanced {...} found, last
    to first, so an echoed schema placeholder doesn't win over a real answer)."""
    blocks = extract_json_objects(text)
    if not blocks:
        raise ValueError(f"no balanced JSON object found in response: {text!r}")

    last_error = None
    for block in reversed(blocks):
        try:
            obj = _try_parse_one(block)
            points = [
                (float(c["x1"]), float(c["x2"]), c.get("reasoning", ""))
                for c in obj["candidates"]
            ]
            if not points:
                raise ValueError("candidates list was empty")
            return points
        except (json.JSONDecodeError, SyntaxError, ValueError, KeyError, TypeError) as e:
            last_error = e
            continue
    raise ValueError(
        f"found {len(blocks)} JSON-like block(s) but none parsed as a valid "
        f"candidates list (last error: {last_error}): {text!r}"
    )


def clip_to_bounds(x1, x2):
    x1c = float(np.clip(x1, *BOUNDS[0]))
    x2c = float(np.clip(x2, *BOUNDS[1]))
    return x1c, x2c


def propose_candidates_llm(history, client):
    """Real API call asking for N_CANDIDATES points. One retry if unparseable,
    then falls back to N_CANDIDATES random points (clearly tagged as such)."""
    prompt = f"History so far:\n{format_history(history)}\n\nPropose {N_CANDIDATES} candidate points."
    for attempt in range(2):
        kwargs = dict(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        if REASONING_EFFORT is not None:
            kwargs["reasoning_effort"] = REASONING_EFFORT
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content
        try:
            points = [(*clip_to_bounds(x1, x2), r) for x1, x2, r in parse_candidates_response(text)]
            return points, text
        except ValueError as e:
            if attempt == 0:
                prompt = (
                    f"{prompt}\n\nYour last response could not be parsed ({e}). "
                    f"Respond with ONLY the JSON object, nothing else."
                )
                continue
            rng = np.random.default_rng()
            fallback = [
                (*clip_to_bounds(*p), "FALLBACK: unparseable response twice")
                for p in rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(N_CANDIDATES, 2))
            ]
            return fallback, text


def propose_candidates_dry_run(rng):
    return [
        (*clip_to_bounds(*p), "dry-run stub")
        for p in rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(N_CANDIDATES, 2))
    ]


def run_hybrid_bo(n_init, n_iter, dry_run=False, n_random_candidates=20):
    rng = np.random.default_rng(SEED)

    client = None
    if not dry_run:
        import groq
        client = groq.Groq()

    X = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(n_init, 2))
    y = branin(X[:, 0], X[:, 1])

    best_so_far = [y.min()]
    reasoning_log = []
    llm_won_log = []  # True if EI picked an LLM candidate that iteration, False if a random filler won

    for i in range(n_iter):
        history = [{"x1": x[0], "x2": x[1], "y": yy} for x, yy in zip(X, y)]

        gp = make_gp()
        gp.fit(X, y)

        if dry_run:
            llm_points = propose_candidates_dry_run(rng)
            raw = ""
        else:
            llm_points, raw = propose_candidates_llm(history, client)

        random_points = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(n_random_candidates, 2))
        llm_xy = np.array([[p[0], p[1]] for p in llm_points])
        all_candidates = np.vstack([llm_xy, random_points])

        ei = expected_improvement(all_candidates, gp, y_best=y.min())
        best_idx = int(np.argmax(ei))
        next_x = all_candidates[best_idx]
        won_by_llm = best_idx < len(llm_points)

        next_y = float(branin(next_x[0], next_x[1]))
        X = np.vstack([X, next_x])
        y = np.append(y, next_y)
        best_so_far.append(float(y.min()))
        llm_won_log.append(won_by_llm)

        chosen_reasoning = llm_points[best_idx][2] if won_by_llm else "EI picked a random filler candidate over every LLM one"
        reasoning_log.append(chosen_reasoning)

        print(f"iter {i+1:2d}: tried x=({next_x[0]:6.2f}, {next_x[1]:6.2f})  f(x)={next_y:8.3f}  "
              f"best so far={y.min():8.3f}  [{'LLM' if won_by_llm else 'random'} candidate won]  | {chosen_reasoning}")

    return np.array(X), np.array(y), np.array(best_so_far), reasoning_log, llm_won_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    args = parser.parse_args()
    N_CANDIDATES = args.n_candidates  # note: only affects future calls if SYSTEM_PROMPT already built above

    print(f"=== Layer 3: hybrid LLM-candidates + real EI (dry_run={args.dry_run}) ===")
    X_h, y_h, best_hybrid, reasoning_log, llm_won_log = run_hybrid_bo(
        n_init=args.n_init, n_iter=args.n_iter, dry_run=args.dry_run,
    )

    print("\n=== Re-running Layer 1's classical GP+EI for comparison ===")
    _, _, _, best_classical, _ = run_bo(n_init=args.n_init, n_iter=args.n_iter)

    llm_win_rate = sum(llm_won_log) / len(llm_won_log) if llm_won_log else 0.0
    print(f"\nHybrid best found:    {y_h.min():.4f}")
    print(f"Classical best found: {best_classical[-1]:.4f}")
    print(f"True minimum:         {TRUE_MIN:.4f}")
    print(f"LLM candidate win rate: {llm_win_rate*100:.0f}% ({sum(llm_won_log)}/{len(llm_won_log)} iterations)")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    iters = np.arange(len(best_hybrid))
    ax.plot(iters, best_hybrid, marker="o", label="hybrid (LLM candidates + real EI)", color="tab:green")
    ax.plot(iters, best_classical, marker="s", label="classical GP + EI (Layer 1)", color="tab:blue")
    ax.axhline(TRUE_MIN, color="gray", linestyle="--", label=f"true global min ({TRUE_MIN:.3f})")
    ax.axvline(args.n_init - 0.5, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("iteration")
    ax.set_ylabel("best f(x) found so far")
    ax.set_title(f"Hybrid vs classical BO on Branin (LLM win rate: {llm_win_rate*100:.0f}%)")
    ax.legend(fontsize=9)
    plt.tight_layout()

    suffix = "_dryrun" if args.dry_run else ""
    out_path = Path(__file__).resolve().parent / f"layer3_comparison{suffix}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    results_path = Path(__file__).resolve().parent / f"layer3_results{suffix}.json"
    results_path.write_text(json.dumps({
        "dry_run": args.dry_run,
        "n_init": args.n_init,
        "n_iter": args.n_iter,
        "n_candidates": args.n_candidates,
        "best_so_far_hybrid": best_hybrid.tolist(),
        "best_so_far_classical": best_classical.tolist(),
        "llm_won_log": llm_won_log,
        "reasoning_log": reasoning_log,
    }, indent=2))
    print(f"Saved raw results to {results_path}")