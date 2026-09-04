"""
Layer 4 -- Human-in-the-loop: the Layer 3 hybrid loop, wrapped in LangGraph,
with a real interrupt() gate before evaluating any candidate the GP is
genuinely uncertain about.

Why sigma (GP uncertainty) is the gate, not EI or "is this an LLM candidate":
EI already balances explore/exploit for us -- overriding it because a point
looks unusual would just be second-guessing math that's already accounting
for that. Sigma is a different, complementary signal: "how far is this from
anything we've actually measured." That's the real-world question a human
running an expensive experiment would ask before committing -- not "is the
math right" but "are we about to spend budget somewhere we have almost no
information." Calibrated against this project's actual GP: sigma sits
around 1 near an already-sampled point and 10-40 at genuinely unexplored
ones, so UNCERTAINTY_THRESHOLD=20 flags the clearly-unexplored tail without
flagging routine exploitation.

On reject: the risky candidate is dropped and the highest-EI candidate whose
sigma is already BELOW the threshold is run instead. Substituting a uniform
random point (the obvious first instinct) is actively wrong here -- a random
point in a barely-sampled domain is typically MORE uncertain than the one the
human just vetoed, so the veto would increase the very risk it exists to
contain. Falling back within the scored pool means "reject" says what a human
actually means by it: don't gamble, run the best option we already understand.
Only if no candidate clears the threshold do we resort to a random point.

Usage:
    python3 layer4_hitl.py                      # real run, prompts you at each interrupt
    python3 layer4_hitl.py --dry-run             # no API calls, still prompts for real interrupts
    python3 layer4_hitl.py --dry-run --auto-approve   # no API calls, no prompts -- for testing the graph itself
    python3 layer4_hitl.py --dry-run --auto-reject    # same, but always rejects -- tests the fallback path
"""

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # BayOAgent/ (project root)

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from Layer_1.layer1_branin_bo import branin, BOUNDS, TRUE_MIN, expected_improvement, run_bo
from Layer_3.layer3_hybrid import (
    make_gp, propose_candidates_llm, propose_candidates_dry_run,
    N_CANDIDATES as L3_N_CANDIDATES,
)

load_dotenv()

SEED = 42
UNCERTAINTY_THRESHOLD = 20.0  # calibrated against this project's GP -- see module docstring
N_RANDOM_CANDIDATES = 20
N_LLM_CANDIDATES = L3_N_CANDIDATES  # whatever Layer 3's proposer defaults to

_CLIENT = None


def get_client():
    """One client for the whole run -- propose_and_score_node fires once per
    iteration and rebuilding it each time is pure overhead."""
    global _CLIENT
    if _CLIENT is None:
        import groq
        _CLIENT = groq.Groq()
    return _CLIENT


class BOState(TypedDict):
    X: list
    y: list
    iteration: int
    n_iter: int
    dry_run: bool
    reasoning_log: list
    llm_won_log: list
    approval_log: list
    pending: dict
    ei_winner_llm: bool  # did an LLM candidate win EI scoring? recorded BEFORE any veto


def propose_and_score_node(state: BOState) -> dict:
    X = np.array(state["X"])
    y = np.array(state["y"])
    history = [{"x1": x[0], "x2": x[1], "y": yy} for x, yy in zip(X, y)]

    gp = make_gp()
    gp.fit(X, y)

    if state["dry_run"]:
        llm_points = propose_candidates_dry_run(np.random.default_rng(SEED + state["iteration"] + 200_000))
    else:
        llm_points, _ = propose_candidates_llm(history, get_client())

    rng = np.random.default_rng(SEED + state["iteration"])
    random_points = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(N_RANDOM_CANDIDATES, 2))
    llm_xy = np.array([[p[0], p[1]] for p in llm_points])
    all_candidates = np.vstack([llm_xy, random_points])

    ei = expected_improvement(all_candidates, gp, y_best=y.min())
    best_idx = int(np.argmax(ei))
    next_x = all_candidates[best_idx]
    won_by_llm = best_idx < len(llm_points)
    reasoning = llm_points[best_idx][2] if won_by_llm else "EI picked a random filler candidate over every LLM one"

    _, sigma_all = gp.predict(all_candidates, return_std=True)
    sigma = float(sigma_all[best_idx])

    # Best candidate the GP is ALREADY confident about, for the reject path. Picking
    # it here (rather than sampling a fresh random point at veto time) is what keeps
    # a rejection from landing somewhere even less explored than what was vetoed.
    safe = [j for j in np.argsort(ei)[::-1] if sigma_all[j] <= UNCERTAINTY_THRESHOLD]
    fallback = None
    if safe:
        j = int(safe[0])
        fallback = {
            "x1": float(all_candidates[j][0]), "x2": float(all_candidates[j][1]),
            "sigma": float(sigma_all[j]), "ei": float(ei[j]),
            "won_by_llm": bool(j < len(llm_points)),
        }

    pending = {
        "x1": float(next_x[0]), "x2": float(next_x[1]),
        "sigma": sigma, "ei": float(ei[best_idx]),
        "won_by_llm": won_by_llm, "reasoning": reasoning,
        "fallback": fallback,
    }
    return {"pending": pending, "ei_winner_llm": won_by_llm}


def route_after_scoring(state: BOState) -> str:
    return "human_review" if state["pending"]["sigma"] > UNCERTAINTY_THRESHOLD else "evaluate"


def human_review_node(state: BOState) -> dict:
    p = state["pending"]
    decision = interrupt({
        "message": "High-uncertainty candidate -- human approval needed before running this experiment.",
        "iteration": state["iteration"] + 1,
        **p,
    })
    approved = bool(decision.get("approved", False))
    log_entry = {"iteration": state["iteration"] + 1, **p, "human_approved": approved}

    if approved:
        return {"approval_log": state["approval_log"] + [log_entry]}

    fb = p.get("fallback")
    if fb is None:
        # Nothing in the pool cleared the threshold -- everything is unexplored, so a
        # random point is no worse than the alternatives. Rare, and flagged as such.
        rng = np.random.default_rng(SEED + state["iteration"] + 100_000)
        fallback_x = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
        fb = {"x1": float(fallback_x[0]), "x2": float(fallback_x[1]),
              "sigma": None, "ei": None, "won_by_llm": False}
        why = ("Human rejected the high-uncertainty candidate; NO candidate was under "
               "the threshold, so fell back to a random point.")
    else:
        why = (f"Human rejected the high-uncertainty candidate (sigma={p['sigma']:.1f}); "
               f"ran the best candidate under the threshold instead "
               f"(sigma={fb['sigma']:.1f}).")
    new_pending = {**fb, "fallback": None, "reasoning": why}
    return {"pending": new_pending, "approval_log": state["approval_log"] + [log_entry]}


def evaluate_node(state: BOState) -> dict:
    p = state["pending"]
    y_new = float(branin(p["x1"], p["x2"]))
    X = state["X"] + [[p["x1"], p["x2"]]]
    y = state["y"] + [y_new]
    reasoning_log = state["reasoning_log"] + [p["reasoning"]]
    # the point that WON EI scoring, before any human veto -- vetoing an LLM candidate
    # is a fact about the human, not evidence the LLM's candidate was worse.
    llm_won_log = state["llm_won_log"] + [state["ei_winner_llm"]]

    best_so_far = min(y)
    sigma_str = f"{p['sigma']:6.2f}" if p["sigma"] is not None else "  n/a "
    print(f"iter {state['iteration']+1:2d}: tried x=({p['x1']:6.2f}, {p['x2']:6.2f})  "
          f"f(x)={y_new:8.3f}  best so far={best_so_far:8.3f}  sigma={sigma_str}  | {p['reasoning']}")

    return {
        "X": X, "y": y, "iteration": state["iteration"] + 1,
        "reasoning_log": reasoning_log, "llm_won_log": llm_won_log,
    }


def route_after_evaluate(state: BOState) -> str:
    return "propose_and_score" if state["iteration"] < state["n_iter"] else END


def build_graph():
    graph = StateGraph(BOState)
    graph.add_node("propose_and_score", propose_and_score_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("evaluate", evaluate_node)

    graph.add_edge(START, "propose_and_score")
    graph.add_conditional_edges(
        "propose_and_score", route_after_scoring,
        {"human_review": "human_review", "evaluate": "evaluate"},
    )
    graph.add_edge("human_review", "evaluate")
    graph.add_conditional_edges(
        "evaluate", route_after_evaluate,
        {"propose_and_score": "propose_and_score", END: END},
    )
    return graph.compile(checkpointer=MemorySaver())


def run_with_human_in_the_loop(n_init, n_iter, dry_run=False, auto_approve=None, thread_id="layer4-run"):
    """auto_approve: None -> really prompt via input(). True/False -> skip the
    prompt and always answer that way (for testing the graph without a live human)."""
    rng = np.random.default_rng(SEED)
    X_init = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(n_init, 2))
    y_init = branin(X_init[:, 0], X_init[:, 1])

    app = build_graph()
    initial_state = {
        "X": X_init.tolist(), "y": y_init.tolist(),
        "iteration": 0, "n_iter": n_iter, "dry_run": dry_run,
        "reasoning_log": [], "llm_won_log": [], "approval_log": [], "pending": {},
        "ei_winner_llm": False,
    }
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(initial_state, config=config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n" + "=" * 70)
        print(f"HUMAN REVIEW REQUIRED -- iteration {payload['iteration']}")
        print(f"  Candidate: x=({payload['x1']:.2f}, {payload['x2']:.2f})")
        print(f"  GP uncertainty (sigma): {payload['sigma']:.2f}  (threshold: {UNCERTAINTY_THRESHOLD})")
        print(f"  Expected Improvement: {payload['ei']:.4f}")
        print(f"  Source: {'LLM candidate' if payload['won_by_llm'] else 'random filler'}")
        print(f"  Reasoning: {payload['reasoning']}")
        print("=" * 70)

        if auto_approve is not None:
            approved = auto_approve
            print(f"[non-interactive mode] decision: {'APPROVE' if approved else 'REJECT'}")
        else:
            ans = input("Approve this experiment? [y/n]: ").strip().lower()
            approved = ans.startswith("y")

        result = app.invoke(Command(resume={"approved": approved}), config=config)

    return np.array(result["X"]), np.array(result["y"]), result["reasoning_log"], result["llm_won_log"], result["approval_log"]


def make_plot(best_so_far, best_classical, approval_log, n_init, out_path):
    """Kept separate from the run so a saved results file can be re-plotted without
    spending API calls (see --replot)."""
    n_flagged = len(approval_log)
    n_approved = sum(1 for a in approval_log if a["human_approved"])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(np.arange(len(best_so_far)), best_so_far, marker="o",
            label="hybrid + HITL", color="tab:purple")
    ax.plot(np.arange(len(best_classical)), best_classical, marker="s",
            label="classical GP + EI (Layer 1)", color="tab:blue")
    # Label the first of EACH kind -- keying off approval_log[0] alone drops whichever
    # outcome didn't happen to come first from the legend entirely.
    labelled = set()
    for a in approval_log:
        ok = a["human_approved"]
        marker, color = ("^", "green") if ok else ("x", "red")
        ax.scatter(a["iteration"], best_so_far[min(a["iteration"], len(best_so_far) - 1)],
                   marker=marker, color=color, s=120, zorder=5,
                   label=None if ok in labelled else ("approved" if ok else "rejected"))
        labelled.add(ok)
    ax.axhline(TRUE_MIN, color="gray", linestyle="--", label=f"true global min ({TRUE_MIN:.3f})")
    # index 0 is the best over ALL n_init random points -- the loop takes over at 1.
    ax.axvline(0.5, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel(f"iteration (0 = best of the {n_init} random init points, "
                  f"1+ = one evaluation each)")
    ax.set_ylabel("best f(x) found so far")
    ax.set_title(f"Hybrid + HITL vs classical BO ({n_flagged} interrupts, {n_approved} approved)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto-approve", action="store_true", help="skip prompts, always approve (testing)")
    parser.add_argument("--auto-reject", action="store_true", help="skip prompts, always reject (testing)")
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--replot", metavar="RESULTS_JSON",
                        help="rebuild the plot from a saved results file instead of "
                             "running (no API calls)")
    args = parser.parse_args()

    if args.replot:
        d = json.loads(Path(args.replot).read_text())
        suffix = "_dryrun" if d["dry_run"] else ""
        out_path = make_plot(
            np.array(d["best_so_far"]), np.array(d["best_so_far_classical"]),
            d["approval_log"], d["n_init"],
            Path(__file__).resolve().parent / f"layer4_comparison{suffix}.png",
        )
        print(f"Re-plotted {args.replot} -> {out_path}")
        sys.exit(0)

    if args.auto_approve and args.auto_reject:
        parser.error("--auto-approve and --auto-reject are mutually exclusive")
    auto = True if args.auto_approve else (False if args.auto_reject else None)

    print(f"=== Layer 4: hybrid + LangGraph human-in-the-loop (dry_run={args.dry_run}) ===")
    X, y, reasoning_log, llm_won_log, approval_log = run_with_human_in_the_loop(
        n_init=args.n_init, n_iter=args.n_iter, dry_run=args.dry_run, auto_approve=auto,
    )
    best_so_far = [float(y[:args.n_init].min())]
    for yi in y[args.n_init:]:
        best_so_far.append(min(best_so_far[-1], float(yi)))
    best_so_far = np.array(best_so_far)

    print("\n=== Re-running Layer 1's classical GP+EI for comparison ===")
    _, _, _, best_classical, _ = run_bo(n_init=args.n_init, n_iter=args.n_iter)

    n_flagged = len(approval_log)
    n_approved = sum(1 for a in approval_log if a["human_approved"])
    print(f"\nHybrid+HITL best found: {y.min():.4f}")
    print(f"Classical best found:   {best_classical[-1]:.4f}")
    print(f"True minimum:           {TRUE_MIN:.4f}")
    print(f"Interrupts fired: {n_flagged}/{args.n_iter} iterations  "
          f"({n_approved} approved, {n_flagged - n_approved} rejected)")

    llm_win_rate = sum(llm_won_log) / len(llm_won_log) if llm_won_log else 0.0
    n_pool = N_LLM_CANDIDATES + N_RANDOM_CANDIDATES
    chance = N_LLM_CANDIDATES / n_pool
    print(f"LLM candidate win rate: {llm_win_rate*100:.0f}% "
          f"({sum(llm_won_log)}/{len(llm_won_log)} iterations)  -- chance baseline is "
          f"{chance*100:.0f}% ({N_LLM_CANDIDATES} LLM candidates out of {n_pool} total). "
          f"Measured on EI scoring, before human vetoes.")

    suffix = "_dryrun" if args.dry_run else ""
    out_path = make_plot(
        best_so_far, best_classical, approval_log, args.n_init,
        Path(__file__).resolve().parent / f"layer4_comparison{suffix}.png",
    )
    print(f"Saved plot to {out_path}")

    results_path = Path(__file__).resolve().parent / f"layer4_results{suffix}.json"
    results_path.write_text(json.dumps({
        "dry_run": args.dry_run,
        "n_init": args.n_init,
        "n_iter": args.n_iter,
        "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
        "llm_win_rate": llm_win_rate,
        "llm_win_rate_chance_baseline": chance,
        "best_so_far": best_so_far.tolist(),
        "best_so_far_classical": best_classical.tolist(),
        "approval_log": approval_log,
        "reasoning_log": reasoning_log,
        "llm_won_log": llm_won_log,
    }, indent=2))
    print(f"Saved raw results to {results_path}")