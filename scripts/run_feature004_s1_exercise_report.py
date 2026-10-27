#!/usr/bin/env python3
"""Opt-in real S1 50-class per-exercise test report; exploratory only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semg_baseline.feature004_exercise_metrics import stratify_final_test  # noqa: E402
from semg_baseline.feature004_multi_exercise import MultiExerciseWithinSubjectRunner  # noqa: E402
from semg_baseline.lineage import verify_feature_003_frozen_contracts  # noqa: E402
from semg_dataset.loader import NinaProDB2Loader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run one real S1 full LOTO plus final TEST")
    parser.add_argument("--save", action="store_true", help="Save an UNSIGNED exploratory JSON report")
    args = parser.parse_args()
    if not args.run:
        print("NO DATA LOADED: pass --run explicitly to execute the real S1 pilot.")
        return 0

    ok, errors = verify_feature_003_frozen_contracts(ROOT)
    if not ok:
        raise RuntimeError("Feature 003 acceptance consistency failed: " + "; ".join(errors))
    print("Feature 003: internal manifest consistency PASS (not externally pinned)", flush=True)
    loader = NinaProDB2Loader(raw_dir=ROOT / "data" / "raw" / "ninapro_db2")
    runner = MultiExerciseWithinSubjectRunner()
    result = runner.run_subject("S1", loader.load_recording)
    report = stratify_final_test(result, runner.test_exercise_ids)

    print("\nREAL NINAPRO DB2 S1 — EXPLORATORY 50-CLASS EXERCISE STRATIFICATION")
    print(f"Global Top-1: {report.global_top1_accuracy:.4%}; Balanced: {report.global_balanced_accuracy:.4%}; Macro-F1: {report.global_macro_f1:.4%}")
    print(f"Selected R: {report.selected_r} | TEST evaluations: {report.test_evaluation_count}")
    for part in report.by_exercise:
        print(f"\n{part.exercise_id}: {part.test_windows} TEST windows, REST={part.rest_windows}, active={part.active_windows}")
        print(f"  Top-1={part.top1_accuracy:.4%}; Balanced={part.balanced_accuracy:.4%}; Macro-F1={part.macro_f1:.4%}")
        print(f"  Active Top-1={part.active_top1_accuracy:.4%}; REST recall={part.rest_recall:.4%}; REST false alarm={part.rest_false_alarm_rate:.4%}")
        print(f"  Active->REST={part.active_to_rest_count} ({part.active_to_rest_rate:.4%}); outside exercise preds={part.outside_exercise_prediction_count} ({part.outside_exercise_prediction_rate:.4%})")
    print("\nNOT SCIENTIFIC ACCEPTANCE: positive REST confounding; unanchored lineage; no reviewer sign-off")
    if args.save:
        output = ROOT / "reports" / "feature_004_exploratory" / "s1_exercise_metrics.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately labeled exploratory, without an acceptance manifest or invented fingerprint.
        output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        print("UNSIGNED exploratory report:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
