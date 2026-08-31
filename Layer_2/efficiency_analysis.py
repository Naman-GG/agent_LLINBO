"""
Sample-efficiency comparison: how many evaluations did each method need to
get within a given tolerance of the true optimum, not just where they
ended up. Reads the results file layer2_llm_proposer.py already saves.

Usage:
    python3 efficiency_analysis.py                       # uses layer2_results.json
    python3 efficiency_analysis.py --file layer2_results_permuted.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Layer_1.layer1_branin_bo import TRUE_MIN


def iterations_to_reach(best_so_far, tolerance):
    """First iteration index (0-based) where best_so_far is within
    `tolerance` of the true global minimum. None if never reached."""
    threshold = TRUE_MIN + tolerance
    for i, v in enumerate(best_so_far):
        if v <= threshold:
            return i
    return None


def summarize(name, best_so_far, tolerances):
    print(f"\n{name}:")
    print(f"  final best value: {best_so_far[-1]:.4f}  (true min: {TRUE_MIN:.4f}, "
          f"gap: {best_so_far[-1] - TRUE_MIN:.4f})")
    for tol in tolerances:
        it = iterations_to_reach(best_so_far, tol)
        status = f"iteration {it}" if it is not None else "never reached"
        print(f"  within {tol:>5.2f} of true min: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="Layer_2/layer2_results.json")
    parser.add_argument("--tolerances", type=float, nargs="+", default=[0.5, 0.1, 0.05, 0.01])
    args = parser.parse_args()

    data = json.loads(Path(args.file).read_text())

    print(f"=== Sample efficiency: {args.file} ===")
    print(f"n_init={data['n_init']}, n_iter={data['n_iter']}, "
          f"permute_feedback={data.get('permute_feedback', False)}")

    summarize("Classical GP + EI (Layer 1)", data["best_so_far_classical"], args.tolerances)
    summarize("LLM proposer", data["best_so_far_llm"], args.tolerances)
