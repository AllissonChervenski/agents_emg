#!/usr/bin/env python3
"""Exploratory real NinaPro DB2 S1, 50-class shuffled-label negative control.

NOT scientific acceptance. Uses the frozen repetition protocol and requires
an R selected on TRAIN-only LOTO beforehand. No synthetic recordings.
The existing control report has a placeholder fingerprint; it is not provenance.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from semg_dataset.loader import NinaProDB2Loader  # noqa: E402
from semg_baseline.feature004_multi_exercise import (  # noqa: E402
    MultiExerciseSubject,
    MultiExerciseWithinSubjectRunner,
)
from semg_baseline.lineage import verify_feature_003_frozen_contracts  # noqa: E402
from semg_baseline.controls import ShuffledLabelNegativeControl  # noqa: E402
from semg_baseline.thresholds import ThresholdCalibrator  # noqa: E402
from semg_baseline.features import TimeDomainFeatureExtractor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='store_true', help='Authorize real dataset processing')
    ap.add_argument('--selected-r', type=float, required=False, default=None,
                    help='R selected via TRAIN-only LOTO (e.g., 0.5 from S1 pilot)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if not args.run:
        print('DRY RUN: pass --run --selected-r 0.5 (if confirmed from TRAIN-only LOTO).')
        return 0
    if args.selected_r is None or not np.isfinite(args.selected_r) or args.selected_r <= 0:
        ap.error('--selected-r must be a positive finite, TRAIN-only preselected value')

    ok, errors = verify_feature_003_frozen_contracts(ROOT)
    if not ok:
        raise SystemExit('Frozen acceptance internally inconsistent: ' + '; '.join(errors))
    print('Feature 003 acceptance: internally consistent, NOT independently pinned', flush=True)

    loader = NinaProDB2Loader(raw_dir=ROOT / 'data' / 'raw' / 'ninapro_db2')
    runner = MultiExerciseWithinSubjectRunner()
    subject = MultiExerciseSubject('S1', loader.load_recording)

    # Process only training repetition units before reading/preparing TEST.
    print('Preparing TRAIN windows E1/E2/E3...', flush=True)
    train_windows, y_train, train_units, _ = runner.prepare_repetition_windows(
        subject, runner.TRAINING_REPETITIONS, validate_all_required=True
    )
    if set(np.unique(y_train).tolist()) != set(range(50)):
        raise RuntimeError('Missing nominal classes from TRAIN windows')
    if set(np.unique(train_units).tolist()) != set(runner.TRAINING_REPETITIONS):
        raise RuntimeError('TRAIN repetition units differ from frozen protocol')

    calibration = ThresholdCalibrator().calibrate(
        train_emg=train_windows,
        train_labels=y_train,
        rest_label=0,
        r_multiplier=args.selected_r,
    )
    extractor = TimeDomainFeatureExtractor(epsilon=calibration.epsilon_channels, num_channels=12)
    X_train = extractor.extract(train_windows)
    del train_windows

    print('Preparing TEST windows E1/E2/E3 (labels will remain untouched)...', flush=True)
    test_windows, y_test, test_units, _ = runner.prepare_repetition_windows(
        subject, runner.TEST_REPETITIONS, validate_all_required=False
    )
    if set(np.unique(y_test).tolist()) != set(range(50)):
        raise RuntimeError('Missing nominal classes from TEST windows')
    if set(np.unique(test_units).tolist()) != set(runner.TEST_REPETITIONS):
        raise RuntimeError('TEST repetition units differ from frozen protocol')
    X_test = extractor.extract(test_windows)
    del test_windows

    control = ShuffledLabelNegativeControl()
    outcome = control.run_from_features(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_classes=50,
        seed=args.seed,
        failure_threshold=0.05,
        protocol_id='within_subject',
        partition_id='S1',
        enforce_pass=False,
        model_taxonomy='global_50_class',
    )
    print('\nEXPLORATORY REAL S1 SHUFFLED-LABEL CONTROL', flush=True)
    print('Selected R (from prior TRAIN-only LOTO):', args.selected_r)
    print('Seed:', args.seed)
    print('TRAIN windows:', len(y_train))
    print('TEST windows:', len(y_test))
    print('50-class top-1:', outcome.test_top1_accuracy)
    print('Balanced accuracy:', outcome.test_balanced_accuracy)
    print('Macro-F1:', outcome.test_macro_f1)
    print('Nominal uniform chance:', outcome.chance_level)
    print('Pre-registered top-1 threshold:', outcome.failure_threshold)
    print('Control passed threshold:', outcome.passed)
    print('Diagnostic:', outcome.diagnostic_message)
    print('WARNING: controls.py emits a placeholder job_fingerprint; NOT cryptographic provenance.')
    print('WARNING: one seed and a positive confounding audit do not establish scientific acceptance.')
    return 0 if outcome.passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
