"""Run all four lab steps sequentially, or just one with --step N."""

import argparse
import importlib
import sys
import time

STEPS = {
    1: "01_langsmith_rag_pipeline",
    2: "02_prompt_hub_ab_routing",
    3: "03_ragas_evaluation",
    4: "04_guardrails_validator",
}


def run_step(n: int) -> None:
    name = STEPS[n]
    print(f"\n{'#' * 60}\n#   STEP {n}: {name}\n{'#' * 60}")
    t0 = time.time()
    mod = importlib.import_module(name)
    mod.main()
    print(f"\n[step {n} done in {time.time() - t0:.1f}s]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=STEPS.keys(),
                        help="Run only this step (1-4). Default: all.")
    args = parser.parse_args()

    steps = [args.step] if args.step else sorted(STEPS)
    for n in steps:
        try:
            run_step(n)
        except Exception as e:
            print(f"!! Step {n} failed: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
