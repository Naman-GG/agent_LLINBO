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

On reject: the risky candidate is dropped and a random point is substituted
instead -- the human veto removes the risky choice, it doesn't get to
silently keep it.

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
from Layer_3.layer3_hybrid import make_gp, propose_candidates_llm, propose_candidates_dry_run

load_dotenv()

SEED = 42
UNCERTAINTY_THRESHOLD = 20.0  # calibrated against this project's GP -- see module docstring
N_RANDOM_CANDIDATES = 20


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


def propose_and_score_node(state: BOState) -> dict:
    X = np.array(state["X"])
    y = np.array(state["y"])
    history = [{"x1": x[0], "x2": x[1], "y": yy} for x, yy in zip(X, y)]

    gp = make_gp()
    gp.fit(X, y)

    if state["dry_run"]:
        llm_points = propose_candidates_dry_run(np.random.default_rng(SEED + state["iteration"] + 200_000))
    else:
        import groq
        client = groq.Groq()
        llm_points, _ = propose_candidates_llm(history, client)

    rng = np.random.default_rng(SEED + state["iteration"])
    random_points = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(N_RANDOM_CANDIDATES, 2))
    llm_xy = np.array([[p[0], p[1]] for p in llm_points])
    all_candidates = np.vstack([llm_xy, random_points])

    ei = expected_improvement(all_candidates, gp, y_best=y.min())
    best_idx = int(np.argmax(ei))
    next_x = all_candidates[best_idx]
    won_by_llm = best_idx < len(llm_points)
    reasoning = llm_points[best_idx][2] if won_by_llm else "EI picked a random filler candidate over every LLM one"

    _, sigma_arr = gp.predict(next_x.reshape(1, -1), return_std=True)
    sigma = float(sigma_arr[0])

    pending = {
        "x1": float(next_x[0]), "x2": float(next_x[1]),
        "sigma": sigma, "ei": float(ei[best_idx]),
        "won_by_llm": won_by_llm, "reasoning": reasoning,
    }
    return {"pending": pending}


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

    rng = np.random.default_rng(SEED + state["iteration"] + 100_000)
    fallback_x = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
    new_pending = {
        "x1": float(fallback_x[0]), "x2": float(fallback_x[1]),
        "sigma": None, "ei": None, "won_by_llm": False,
        "reasoning": "Human rejected the high-uncertainty candidate; fell back to a random point.",
    }
    return {"pending": new_pending, "approval_log": state["approval_log"] + [log_entry]}


def evaluate_node(state: BOState) -> dict:
    p = state["pending"]
    y_new = float(branin(p["x1"], p["x2"]))
    X = state["X"] + [[p["x1"], p["x2"]]]
    y = state["y"] + [y_new]
    reasoning_log = state["reasoning_log"] + [p["reasoning"]]
    llm_won_log = state["llm_won_log"] + [p["won_by_llm"]]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto-approve", action="store_true", help="skip prompts, always approve (testing)")
    parser.add_argument("--auto-reject", action="store_true", help="skip prompts, always reject (testing)")
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--n-init", type=int, default=5)
    args = parser.parse_args()

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
    print(f"LLM candidate win rate: {llm_win_rate*100:.0f}% ({sum(llm_won_log)}/{len(llm_won_log)} iterations)  "
          f"-- chance baseline is ~20% (5 LLM candidates out of 25 total)")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    iters = np.arange(len(best_so_far))
    ax.plot(iters, best_so_far, marker="o", label="hybrid + HITL", color="tab:purple")
    ax.plot(np.arange(len(best_classical)), best_classical, marker="s", label="classical GP + EI (Layer 1)", color="tab:blue")
    for a in approval_log:
        marker = "^" if a["human_approved"] else "x"
        color = "green" if a["human_approved"] else "red"
        ax.scatter(a["iteration"], best_so_far[min(a["iteration"], len(best_so_far) - 1)],
                    marker=marker, color=color, s=120, zorder=5,
                    label=("approved" if a["human_approved"] else "rejected") if a is approval_log[0] else None)
    ax.axhline(TRUE_MIN, color="gray", linestyle="--", label=f"true global min ({TRUE_MIN:.3f})")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best f(x) found so far")
    ax.set_title(f"Hybrid + HITL vs classical BO ({n_flagged} interrupts, {n_approved} approved)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    suffix = "_dryrun" if args.dry_run else ""
    out_path = Path(__file__).resolve().parent / f"layer4_comparison{suffix}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    results_path = Path(__file__).resolve().parent / f"layer4_results{suffix}.json"
    results_path.write_text(json.dumps({
        "dry_run": args.dry_run,
        "n_init": args.n_init,
        "n_iter": args.n_iter,
        "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
        "best_so_far": best_so_far.tolist(),
        "best_so_far_classical": best_classical.tolist(),
        "approval_log": approval_log,
        "reasoning_log": reasoning_log,
        "llm_won_log": llm_won_log,
    }, indent=2))
    print(f"Saved raw results to {results_path}")