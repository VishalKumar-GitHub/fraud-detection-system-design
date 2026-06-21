"""
Convenience runner — executes all 7 pipeline steps in order.

Equivalent to running each step manually:
    python src/step1_simulate_data.py
    python src/step2_feature_engineering.py
    ... etc

Use this for a full end-to-end run. For a daily working walkthrough, run the
step*.py files individually instead, so you can control each stage.
"""

import subprocess
import sys
from pathlib import Path

STEPS = [
    "step1_simulate_data.py",
    "step2_feature_engineering.py",
    "step3_train_test_split.py",
    "step4_train_models.py",
    "step5_evaluate_models.py",
    "step6_threshold_selection.py",
    "step7_risk_tiering_and_test_eval.py",
]

SRC_DIR = Path(__file__).parent / "src"


def main():
    for step in STEPS:
        script_path = SRC_DIR / step
        print(f"\n{'=' * 70}")
        print(f"Running {step}")
        print('=' * 70)
        result = subprocess.run([sys.executable, str(script_path)], cwd=Path(__file__).parent)
        if result.returncode != 0:
            print(f"\n{step} failed (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

    print(f"\n{'=' * 70}")
    print("Pipeline complete. See data/step7_final_scored_test.csv for results.")
    print('=' * 70)


if __name__ == "__main__":
    main()
