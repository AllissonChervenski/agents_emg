#!/usr/bin/env python3
"""Opt-in exploratory real DB2 S1/E1+E2+E3 benchmark. NOT scientific acceptance."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semg_dataset.loader import NinaProDB2Loader  # noqa: E402
from semg_baseline.feature004_multi_exercise import MultiExerciseWithinSubjectRunner  # noqa: E402
from semg_baseline.lineage import verify_feature_003_frozen_contracts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly authorize RAM-intensive real pilot")
    args = parser.parse_args()
    if not args.run:
        print("PILOT ONLY; no data processed. Pass --run to execute real S1/E1+E2+E3 LOTO.")
        return 0

    ok, errors = verify_feature_003_frozen_contracts(ROOT)
    if not ok:
        raise SystemExit("Feature 003 acceptance consistency failed: " + "; ".join(errors))
    print("Acceptance package: internally consistent (NOT externally pinned)", flush=True)
    loader = NinaProDB2Loader(raw_dir=ROOT / "data" / "raw" / "ninapro_db2")
    runner = MultiExerciseWithinSubjectRunner()
    result = runner.run_subject("S1", loader.load_recording)
    if result.test_evaluation_count != 1:
        raise RuntimeError("Expected exactly one TEST evaluation")

    print("\nEXPLORATORY PILOT — REAL NINAPRO DB2 / S1 / E1+E2+E3 / 50 CLASSES")
    print("Selected R:", result.selected_r)
    print("LOTO folds:", len(result.validation_folds))
    print("Train repetitions:", result.final_train_repetitions)
    print("Test repetitions:", result.final_test_repetitions)
    print("Admitted training windows:", result.n_final_train_samples)
    print("Test windows:", len(result.test_labels))
    print("Top-1 accuracy:", result.test_accuracy)
    print("Balanced accuracy:", result.test_balanced_accuracy)
    print("Macro F1:", result.test_macro_f1)
    print("TEST evaluation count:", result.test_evaluation_count)
    print("NOT ACCEPTED SCIENCE: independently pinned lineage, confounding audit, ")
    print("50-class shuffled-label control, evidence sealing and reviewer sign-off pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
