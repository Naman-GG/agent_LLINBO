"""
Layer 2 -- LLM replaces the GP + Expected Improvement step from Layer 1.

Same loop as Layer 1:
    propose next point -> evaluate real objective -> add to history -> repeat
The only thing that changes is *who* proposes: instead of a fitted GP +
EI acquisition function, we hand the LLM the history as text and ask it
to reason about what to try next.

Includes the permuted-feedback sanity check from "LLMs for Bayesian
Optimization in Scientific Domains: Are We There Yet?" -- if the LLM is
actually reasoning from feedback, scrambling the y-values it sees should
visibly hurt its trajectory. If its behavior barely changes, that's
evidence it isn't really using the feedback at all.

Usage:
    Create a .env file next to this script containing:
        GROQ_API_KEY=your_key_here
    python3 layer2_llm_proposer.py                # real run
    python3 layer2_llm_proposer.py --dry-run       # test the harness, no API calls, no key needed
    python3 layer2_llm_proposer.py --permute       # sanity-check run
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # BayOAgent/ (project root)

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from Layer_1.layer1_branin_bo import branin, BOUNDS, TRUE_MIN, run_bo

load_dotenv()  # reads .env in the current directory (or a parent) into os.environ

SEED = 42
MODEL = "qwen/Qwen3.6-27B"  # served via Groq -- currently a Groq preview model, not GA
MAX_TOKENS = 500
REASONING_EFFORT = "default"  # qwen3.6-27b only accepts "none" (off) or "default" (thinking mode on)

SYSTEM_PROMPT = """You are proposing the next point to evaluate in a black-box \
optimization problem. The domain is x1 in [-5, 10] and x2 in [0, 15]. \
The objective is UNKNOWN to you -- you only see the history of points \
tried so far and their measured values. LOWER values are BETTER (we are \
minimizing). Given the history, propose the single next point you'd try, \
briefly reasoning about the tradeoff between exploring unseen regions \
and exploiting the best region found so far.

Respond with ONLY a JSON object, no other text, in exactly this form:
{"x1": <float>, "x2": <float>, "reasoning": "<one sentence>"}"""


def format_history(history):
    lines = [f"  x1={h['x1']:.3f}, x2={h['x2']:.3f} -> y={h['y']:.3f}" for h in history]
    return "\n".join(lines)


def extract_json_objects(text):
    """Find ALL balanced {...} objects in text, respecting quoted strings so
    a brace inside a string value doesn't get swallowed into a match.
    There can legitimately be more than one -- e.g. a reasoning model that
    echoes the format instructions before actually answering."""
    objects = []
    i = 0
    while True:
        start = text.find("{", i)
        if start == -1:
            break
        depth = 0
        in_string = False
        escape = False
        end = None
        for j in range(start, len(text)):
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
        if end is None:
            break  # unbalanced from here on -- stop scanning
        objects.append(text[start:end + 1])
        i = end + 1
    return objects


def _try_parse_one(candidate):
    """Parse a single candidate as JSON, falling back to a lenient
    Python-literal parse for single-quoted/Python-dict-style output."""
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        py_literal = re.sub(r"\btrue\b", "True", candidate)
        py_literal = re.sub(r"\bfalse\b", "False", py_literal)
        py_literal = re.sub(r"\bnull\b", "None", py_literal)
        return ast.literal_eval(py_literal)  # may itself raise SyntaxError/ValueError


def parse_llm_response(text):
    """Pull the JSON object out of the response, tolerating markdown fences,
    trailing commentary, and mild single-quoted/Python-dict-style slips.
    Tries every balanced {...} block found, from LAST to FIRST, since a
    model's real final answer -- if it echoes the schema or thinks aloud
    first -- almost always comes after any preamble, not before it."""
    candidates = extract_json_objects(text)
    if not candidates:
        raise ValueError(f"no balanced JSON object found in response: {text!r}")

    last_error = None
    for candidate in reversed(candidates):
        try:
            obj = _try_parse_one(candidate)
            return float(obj["x1"]), float(obj["x2"]), obj.get("reasoning", "")
        except (json.JSONDecodeError, SyntaxError, ValueError, KeyError, TypeError) as e:
            last_error = e
            continue
    raise ValueError(
        f"found {len(candidates)} JSON-like block(s) but none parsed successfully "
        f"(last error: {last_error}): {text!r}"
    )


def clip_to_bounds(x1, x2):
    x1c = float(np.clip(x1, *BOUNDS[0]))
    x2c = float(np.clip(x2, *BOUNDS[1]))
    return x1c, x2c


def propose_next_point_llm(history, client):
    """Real API call. One retry if the response doesn't parse as JSON."""
    prompt = f"History so far:\n{format_history(history)}\n\nPropose the next point."
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
            x1, x2, reasoning = parse_llm_response(text)
            return clip_to_bounds(x1, x2), reasoning, text
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            if attempt == 0:
                prompt = (
                    f"{prompt}\n\nYour last response could not be parsed as JSON "
                    f"({e}). Respond with ONLY the JSON object, nothing else."
                )
                continue
            # Naive fallback after one failed retry -- deliberately not a real
            # recovery strategy. This gap is exactly what Layer 3/the reliability
            # layer project is meant to close. Seeded off the history length so a
            # run that hits fallbacks is still reproducible.
            rng = np.random.default_rng(SEED + 100_000 + len(history))
            fallback = tuple(rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1]))
            return clip_to_bounds(*fallback), "FALLBACK: unparseable response twice", text


def propose_next_point_dry_run(history, rng):
    """Stand-in for the LLM call so the harness can be tested with no API key."""
    x1, x2 = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
    return clip_to_bounds(x1, x2), "dry-run stub: random point", ""


def run_llm_bo(n_init, n_iter, permute_feedback=False, dry_run=False):
    rng = np.random.default_rng(SEED)
    permute_rng = np.random.default_rng(SEED + 1)

    client = None
    if not dry_run:
        import groq
        client = groq.Groq()

    X_init = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(n_init, 2))
    y_init = branin(X_init[:, 0], X_init[:, 1])

    history_true = [{"x1": x[0], "x2": x[1], "y": y} for x, y in zip(X_init, y_init)]
    X, y = list(X_init), list(y_init)
    best_so_far = [min(y)]
    reasoning_log = []

    for i in range(n_iter):
        if permute_feedback:
            shuffled_y = permute_rng.permutation([h["y"] for h in history_true])
            history_shown = [
                {"x1": h["x1"], "x2": h["x2"], "y": yy}
                for h, yy in zip(history_true, shuffled_y)
            ]
        else:
            history_shown = history_true

        if dry_run:
            (x1, x2), reasoning, raw = propose_next_point_dry_run(history_shown, rng)
        else:
            (x1, x2), reasoning, raw = propose_next_point_llm(history_shown, client)

        y_new = float(branin(x1, x2))
        X.append([x1, x2])
        y.append(y_new)
        history_true.append({"x1": x1, "x2": x2, "y": y_new})
        best_so_far.append(min(y))
        reasoning_log.append(reasoning)

        print(f"iter {i+1:2d}: tried x=({x1:6.2f}, {x2:6.2f})  f(x)={y_new:8.3f}  "
              f"best so far={min(y):8.3f}   | {reasoning}")
        if reasoning.startswith("FALLBACK"):
            print(f"          raw response that failed to parse: {raw[:300]!r}")

    return np.array(X), np.array(y), np.array(best_so_far), reasoning_log


def make_plot(best_llm, best_classical, n_init, permute, n_fallback, out_path):
    """Convergence plot. Kept separate from the run so a saved results file can be
    re-plotted without spending API calls (see --replot)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    iters = np.arange(len(best_llm))
    label = "LLM proposer (permuted feedback)" if permute else "LLM proposer"
    ax.plot(iters, best_llm, marker="o", label=label, color="tab:orange")
    ax.plot(np.arange(len(best_classical)), best_classical, marker="s",
            label="classical GP + EI (Layer 1)", color="tab:blue")
    ax.axhline(TRUE_MIN, color="gray", linestyle="--", label=f"true global min ({TRUE_MIN:.3f})")
    # index 0 is the best over ALL n_init random points -- the proposer takes over at 1.
    ax.axvline(0.5, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel(f"iteration (0 = best of the {n_init} random init points, "
                  f"1+ = one proposal each)")
    ax.set_ylabel("best f(x) found so far")

    title = "LLM proposer vs classical BO on Branin" + (" -- permuted feedback" if permute else "")
    if n_fallback:
        # Unparseable responses fell back to uniform random points. Those iterations
        # are NOT the LLM reasoning, so the curve is part random search -- say so on
        # the figure, otherwise the permuted result reads stronger than it is.
        n_total = len(best_llm) - 1
        title += (f"\ncaveat: {n_fallback}/{n_total} proposals were random fallbacks "
                  f"(unparseable LLM response)")
        ax.set_title(title, fontsize=11)
    else:
        ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="test the harness with no API calls")
    parser.add_argument("--permute", action="store_true", help="run the permuted-feedback sanity check")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--replot", metavar="RESULTS_JSON",
                        help="rebuild the plot from a saved results file instead of running "
                             "(no API calls)")
    args = parser.parse_args()

    if args.replot:
        data = json.loads(Path(args.replot).read_text())
        n_fallback = sum(1 for r in data["reasoning_log"] if r.startswith("FALLBACK"))
        suffix = "_permuted" if data["permute_feedback"] else ""
        suffix += "_dryrun" if data["dry_run"] else ""
        out_path = make_plot(
            np.array(data["best_so_far_llm"]), np.array(data["best_so_far_classical"]),
            data["n_init"], data["permute_feedback"], n_fallback,
            Path(__file__).resolve().parent / f"layer2_comparison{suffix}.png",
        )
        print(f"Re-plotted {args.replot} -> {out_path}")
        print(f"  random fallbacks (unparseable LLM response): {n_fallback}/{data['n_iter']}")
        sys.exit(0)

    print(f"=== Layer 2: LLM proposer (dry_run={args.dry_run}, permute={args.permute}) ===")
    X_llm, y_llm, best_llm, reasoning_log = run_llm_bo(
        n_init=args.n_init, n_iter=args.n_iter,
        permute_feedback=args.permute, dry_run=args.dry_run,
    )

    print("\n=== Re-running Layer 1's classical GP+EI for comparison (no API cost) ===")
    _, _, _, best_classical, _ = run_bo(n_init=args.n_init, n_iter=args.n_iter)

    n_fallback = sum(1 for r in reasoning_log if r.startswith("FALLBACK"))
    print(f"\nLLM best found:       {y_llm.min():.4f}")
    print(f"Classical best found: {best_classical[-1]:.4f}")
    print(f"True minimum:         {TRUE_MIN:.4f}")
    if n_fallback:
        print(f"WARNING: {n_fallback}/{args.n_iter} proposals were random fallbacks "
              f"(unparseable LLM response) -- that fraction of this curve is random "
              f"search, not the LLM.")

    suffix = "_permuted" if args.permute else ""
    suffix += "_dryrun" if args.dry_run else ""
    out_path = make_plot(
        best_llm, best_classical, args.n_init, args.permute, n_fallback,
        Path(__file__).resolve().parent / f"layer2_comparison{suffix}.png",
    )
    print(f"Saved plot to {out_path}")

    results_path = Path(__file__).resolve().parent / f"layer2_results{suffix}.json"
    results_path.write_text(json.dumps({
        "permute_feedback": args.permute,
        "dry_run": args.dry_run,
        "n_init": args.n_init,
        "n_iter": args.n_iter,
        "n_fallback": n_fallback,
        "best_so_far_llm": best_llm.tolist(),
        "best_so_far_classical": best_classical.tolist(),
        "reasoning_log": reasoning_log,
    }, indent=2))
    print(f"Saved raw results to {results_path}")
