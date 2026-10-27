#!/usr/bin/env python3
"""Benchmark Ozdemir et al. sEMG dataset under precision quantization and channel reduction.

Evaluates 10-class gesture classification quality under:
1. Precision Reduction: float32, float16, int32, int16, int8
2. Channel Reduction: 4, 3, 2, 1 channels

Protocol:
- 40 subjects (S1 to S40)
- 10 gestures (0: Rest, 1: Extension, 2: Flexion, 3: Ulnar Dev, 4: Radial Dev,
               5: Grip, 6: Abduction, 7: Adduction, 8: Supination, 9: Pronation)
- 5 repetition cycles per subject
- Train pool: repetitions [1, 3, 4]
- Test pool: repetitions [2, 5]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semg_baseline.features import TimeDomainFeatureExtractor  # noqa: E402
from semg_baseline.lda import CanonicalLdaTrainer, PriorType  # noqa: E402
from semg_baseline.scaling import FeatureScaler  # noqa: E402
from semg_baseline.thresholds import ThresholdCalibrator  # noqa: E402

# Full Scale Range for Ozdemir sEMG recordings (+/- 10.0 mV)
V_FS_MV = 10.0

PRECISION_OPTIONS = ("float32", "float16", "int32", "int16", "int8")
CHANNEL_OPTIONS = (4, 3, 2, 1)

BYTES_PER_SAMPLE_1CH = {
    "float32": 4,
    "float16": 2,
    "int32": 4,
    "int16": 2,
    "int8": 1,
}

# Repetition start offsets in seconds from EMG_gesture_segmentation.m
REP_COEFFS_SEC = (4, 138, 272, 406, 540)
TRIAL_ACTIVE_SEC = 6.0
GESTURE_TRIAL_TOTAL_SEC = 10.0


def quantize_signal(emg: np.ndarray, precision: str, v_max: float = V_FS_MV) -> np.ndarray:
    """Apply precision reduction / quantization noise to continuous sEMG signal in mV.

    Args:
        emg: 2D array of shape (num_samples, num_channels) in float32.
        precision: One of 'float32', 'float16', 'int32', 'int16', 'int8'.
        v_max: Nominal full-scale voltage limit (default 10.0 mV).

    Returns:
        Quantized array cast back to float32 representation for downstream DSP.
    """
    if precision == "float32":
        return emg.astype(np.float32, copy=False)
    elif precision == "float16":
        return emg.astype(np.float16).astype(np.float32)
    elif precision == "int32":
        scale = (2**31 - 1) / v_max
        q = np.clip(np.round(emg * scale), -2**31, 2**31 - 1)
        return (q / scale).astype(np.float32)
    elif precision == "int16":
        scale = (2**15 - 1) / v_max
        q = np.clip(np.round(emg * scale), -2**15, 2**15 - 1)
        return (q / scale).astype(np.float32)
    elif precision == "int8":
        scale = (2**7 - 1) / v_max
        q = np.clip(np.round(emg * scale), -2**7, 2**7 - 1)
        return (q / scale).astype(np.float32)
    else:
        raise ValueError(f"Unsupported precision '{precision}'. Must be one of {PRECISION_OPTIONS}.")


def get_channel_feature_indices(num_channels: int, total_channels: int = 4) -> list[int]:
    """Return column indices of TimeDomainFeatureExtractor corresponding to first K channels.

    Feature order is 5 blocks of `total_channels`:
    [MAV (0..3), RMS (4..7), WL (8..11), ZC (12..15), SSC (16..19)].
    """
    if num_channels < 1 or num_channels > total_channels:
        raise ValueError(f"num_channels must be between 1 and {total_channels}, got {num_channels}")
    indices: list[int] = []
    for feat_idx in range(5):
        offset = feat_idx * total_channels
        indices.extend(range(offset, offset + num_channels))
    return indices


def compute_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Compute Top-1 Accuracy, Balanced Accuracy, and Macro F1 purely via NumPy."""
    acc = float(np.mean(y_true == y_pred))
    classes = np.unique(y_true)
    recalls: list[float] = []
    f1s: list[float] = []
    for c in classes:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        recalls.append(recall)
        f1s.append(f1)
    bal_acc = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    return acc, bal_acc, macro_f1


@dataclass(frozen=True)
class BenchmarkScenarioResult:
    """Outcome for a single (precision, channel_count) combination on a single subject."""

    precision: str
    num_channels: int
    num_features: int
    bytes_per_sample: int
    data_rate_reduction_ratio: float
    top1_accuracy: float
    balanced_accuracy: float
    macro_f1: float
    delta_acc_vs_baseline: float
    delta_bal_acc_vs_baseline: float
    delta_f1_vs_baseline: float


@dataclass(frozen=True)
class PopulationScenarioSummary:
    """Consolidated statistical outcome across multiple subjects for a scenario."""

    precision: str
    num_channels: int
    num_features: int
    bytes_per_sample: int
    data_rate_reduction_ratio: float
    mean_top1_accuracy: float
    std_top1_accuracy: float
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    mean_macro_f1: float
    std_macro_f1: float
    mean_delta_acc_vs_baseline: float
    std_delta_acc_vs_baseline: float


def load_ozdemir_recording(subject_id: str, raw_dir: Path, use_filtered: bool = True) -> np.ndarray:
    """Load continuous 4-channel sEMG recording for subject from .mat file."""
    # Subject id can be 'S1' or '1'
    num = int("".join(filter(str.isdigit, subject_id)))
    subdir = "filtered" if use_filtered else "raw"
    suffix = "_filtered.mat" if use_filtered else "_raw.mat"
    mat_path = raw_dir / "sEMG-dataset" / subdir / "mat" / f"{num}{suffix}"

    if not mat_path.exists():
        raise FileNotFoundError(f"Ozdemir recording not found: {mat_path}")

    mat = loadmat(str(mat_path))
    data = mat["data"].astype(np.float32)
    return data


def segment_ozdemir_windows(
    data: np.ndarray,
    fs: int = 2000,
    win_len_sec: float = 0.200,
    stride_sec: float = 0.050,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment active gesture windows from continuous recording.

    Returns:
        windows: Array of shape (N_windows, win_len, 4) in float32.
        labels: Array of shape (N_windows,) with gesture class 0..9.
        reps: Array of shape (N_windows,) with repetition cycle 1..5.
    """
    win_len = int(win_len_sec * fs)
    stride = int(stride_sec * fs)

    windows_list: list[np.ndarray] = []
    labels_list: list[int] = []
    reps_list: list[int] = []

    for rep_idx, rep_c in enumerate(REP_COEFFS_SEC, 1):
        for gest_idx in range(10):
            start_samp = int((rep_c + gest_idx * GESTURE_TRIAL_TOTAL_SEC) * fs)
            end_samp = int(start_samp + TRIAL_ACTIVE_SEC * fs)
            trial_sig = data[start_samp:end_samp]

            n_win = (len(trial_sig) - win_len) // stride + 1
            for w in range(n_win):
                w_start = w * stride
                windows_list.append(trial_sig[w_start : w_start + win_len])
                labels_list.append(gest_idx)
                reps_list.append(rep_idx)

    return (
        np.array(windows_list, dtype=np.float32),
        np.array(labels_list, dtype=np.int64),
        np.array(reps_list, dtype=np.int64),
    )


def run_benchmark_for_subject(
    subject_id: str,
    raw_dir: Path,
    precisions: Sequence[str] = PRECISION_OPTIONS,
    channel_counts: Sequence[int] = CHANNEL_OPTIONS,
    r_multiplier: float = 0.5,
    use_filtered: bool = True,
    verbose: bool = True,
) -> list[BenchmarkScenarioResult]:
    """Execute complete precision and channel reduction sweep for one Ozdemir subject."""
    results: list[BenchmarkScenarioResult] = []

    # Store baseline accuracy to calculate deltas
    baseline_top1 = 0.0
    baseline_bal_acc = 0.0
    baseline_f1 = 0.0

    if verbose:
        print("\n=======================================================")
        print(f"BENCHMARKING OZDEMIR SUBJECT: {subject_id}")
        print(f"Precisions: {list(precisions)}")
        print(f"Channels: {list(channel_counts)}")
        print("=======================================================\n", flush=True)

    t_load = time.time()
    raw_sig = load_ozdemir_recording(subject_id=subject_id, raw_dir=raw_dir, use_filtered=use_filtered)
    if verbose:
        print(f"[{subject_id}] Loaded continuous recording in {time.time() - t_load:.2f}s", flush=True)

    for prec in precisions:
        t0 = time.time()

        # Apply precision reduction / quantization
        q_sig = quantize_signal(raw_sig, precision=prec)

        # Segment windows
        windows, labels, reps = segment_ozdemir_windows(q_sig, fs=2000)

        # Train reps: [1, 3, 4], Test reps: [2, 5]
        train_mask = np.isin(reps, [1, 3, 4])
        test_mask = np.isin(reps, [2, 5])

        train_win, train_lbl = windows[train_mask], labels[train_mask]
        test_win, test_lbl = windows[test_mask], labels[test_mask]

        # Calibrate thresholds on rest label (0)
        calibrator = ThresholdCalibrator(default_r=r_multiplier, num_channels=4)
        calib = calibrator.calibrate(
            train_emg=train_win, train_labels=train_lbl, rest_label=0, r_multiplier=r_multiplier
        )

        extractor = TimeDomainFeatureExtractor(epsilon=calib.epsilon_channels, num_channels=4)
        X_train_full = extractor.extract(train_win)
        X_test_full = extractor.extract(test_win)

        # Release window buffers immediately to maintain low memory
        del windows, train_win, test_win
        del q_sig

        for k in channel_counts:
            feat_indices = get_channel_feature_indices(num_channels=k, total_channels=4)
            X_tr_k = X_train_full[:, feat_indices]
            X_te_k = X_test_full[:, feat_indices]

            scaler = FeatureScaler()
            X_tr_s = scaler.fit_transform(X_tr_k)
            X_te_s = scaler.transform(X_te_k)

            trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=10)
            trainer.fit(X_tr_s, train_lbl)
            preds = trainer.predict(X_te_s)

            acc, bal_acc, f1 = compute_evaluation_metrics(test_lbl, preds)

            bytes_per_sample = k * BYTES_PER_SAMPLE_1CH[prec]
            baseline_bytes = 4 * 4  # float32 * 4 channels = 16 bytes
            data_reduction = 1.0 - (bytes_per_sample / baseline_bytes)

            if prec == "float32" and k == 4:
                baseline_top1 = acc
                baseline_bal_acc = bal_acc
                baseline_f1 = f1

            delta_top1 = acc - baseline_top1
            delta_bal = bal_acc - baseline_bal_acc
            delta_f1 = f1 - baseline_f1

            res = BenchmarkScenarioResult(
                precision=prec,
                num_channels=k,
                num_features=len(feat_indices),
                bytes_per_sample=bytes_per_sample,
                data_rate_reduction_ratio=round(data_reduction, 4),
                top1_accuracy=round(acc, 6),
                balanced_accuracy=round(bal_acc, 6),
                macro_f1=round(f1, 6),
                delta_acc_vs_baseline=round(delta_top1, 6),
                delta_bal_acc_vs_baseline=round(delta_bal, 6),
                delta_f1_vs_baseline=round(delta_f1, 6),
            )
            results.append(res)
            if verbose:
                print(
                    f"  -> {prec:7s} | {k:2d} ch ({len(feat_indices):2d} feats) | "
                    f"{bytes_per_sample:2d} B/sample (-{data_reduction*100:5.1f}%) | "
                    f"Acc: {acc*100:5.2f}% ({delta_top1*100:+5.2f}%) | "
                    f"BalAcc: {bal_acc*100:5.2f}% | F1: {f1*100:5.2f}%"
                )

        prec_dur = time.time() - t0
        if not verbose:
            p_4ch = next((r for r in results if r.precision == prec and r.num_channels == 4), None)
            acc_str = f"{p_4ch.top1_accuracy*100:.1f}%" if p_4ch else "?"
            print(f"[{subject_id}][{prec:7s}] Evaluated in {prec_dur:3.1f}s (4ch Acc: {acc_str})", flush=True)

    del raw_sig
    return results


def _worker_benchmark_subject(
    args: tuple[str, str, Sequence[str], Sequence[int], float, bool, bool, str | None]
) -> tuple[str, list[BenchmarkScenarioResult], float]:
    """Helper for multiprocessing worker."""
    subject_id, raw_dir_str, precisions, channel_counts, r_multiplier, use_filtered, verbose, cp_dir_str = args
    t0 = time.time()
    results = run_benchmark_for_subject(
        subject_id=subject_id,
        raw_dir=Path(raw_dir_str),
        precisions=precisions,
        channel_counts=channel_counts,
        r_multiplier=r_multiplier,
        use_filtered=use_filtered,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    if cp_dir_str:
        cp_path = Path(cp_dir_str) / f"{subject_id}.json"
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        cp_path.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n", encoding="utf-8")

    return subject_id, results, elapsed


def aggregate_population_results(
    subject_results: dict[str, list[BenchmarkScenarioResult]],
    precisions: Sequence[str] = PRECISION_OPTIONS,
    channel_counts: Sequence[int] = CHANNEL_OPTIONS,
) -> list[PopulationScenarioSummary]:
    """Aggregate per-subject scenario results into population mean and std deviation."""
    summaries: list[PopulationScenarioSummary] = []

    for prec in precisions:
        for ch in channel_counts:
            acc_list: list[float] = []
            bal_list: list[float] = []
            f1_list: list[float] = []
            delta_list: list[float] = []
            num_feats = len(get_channel_feature_indices(ch, total_channels=4))
            bytes_sample = ch * BYTES_PER_SAMPLE_1CH[prec]
            reduction = 1.0 - (bytes_sample / 16.0)  # baseline 4ch * 4B = 16B

            for sid, res_list in subject_results.items():
                match = next((r for r in res_list if r.precision == prec and r.num_channels == ch), None)
                if match is not None:
                    acc_list.append(match.top1_accuracy)
                    bal_list.append(match.balanced_accuracy)
                    f1_list.append(match.macro_f1)
                    delta_list.append(match.delta_acc_vs_baseline)

            if acc_list:
                m_acc = float(np.mean(acc_list))
                s_acc = float(np.std(acc_list, ddof=1)) if len(acc_list) > 1 else 0.0
                m_bal = float(np.mean(bal_list))
                s_bal = float(np.std(bal_list, ddof=1)) if len(bal_list) > 1 else 0.0
                m_f1 = float(np.mean(f1_list))
                s_f1 = float(np.std(f1_list, ddof=1)) if len(f1_list) > 1 else 0.0
                m_del = float(np.mean(delta_list))
                s_del = float(np.std(delta_list, ddof=1)) if len(delta_list) > 1 else 0.0
            else:
                m_acc, s_acc, m_bal, s_bal, m_f1, s_f1, m_del, s_del = (0.0,) * 8

            summary = PopulationScenarioSummary(
                precision=prec,
                num_channels=ch,
                num_features=num_feats,
                bytes_per_sample=bytes_sample,
                data_rate_reduction_ratio=round(reduction, 4),
                mean_top1_accuracy=round(m_acc, 6),
                std_top1_accuracy=round(s_acc, 6),
                mean_balanced_accuracy=round(m_bal, 6),
                std_balanced_accuracy=round(s_bal, 6),
                mean_macro_f1=round(m_f1, 6),
                std_macro_f1=round(s_f1, 6),
                mean_delta_acc_vs_baseline=round(m_del, 6),
                std_delta_acc_vs_baseline=round(s_del, 6),
            )
            summaries.append(summary)

    return summaries


def run_benchmark_multi_subject(
    subjects: Sequence[str],
    raw_dir: Path,
    precisions: Sequence[str] = PRECISION_OPTIONS,
    channel_counts: Sequence[int] = CHANNEL_OPTIONS,
    r_multiplier: float = 0.5,
    use_filtered: bool = True,
    num_workers: int = 2,
    verbose: bool = False,
) -> tuple[dict[str, list[BenchmarkScenarioResult]], list[PopulationScenarioSummary]]:
    """Execute benchmark across multiple Ozdemir subjects with checkpoints."""
    total_subjects = len(subjects)
    expected_scenarios = len(precisions) * len(channel_counts)
    cp_dir = ROOT / "reports" / "benchmarks" / "ozdemir_checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print(f"OZDEMIR POPULATION BENCHMARK: {total_subjects} SUBJECTS")
    print(f"Subjects: {list(subjects)}")
    print(f"Workers: {num_workers} processes")
    print(f"Precisions: {list(precisions)} | Channels: {list(channel_counts)}")
    print(f"Checkpoints directory: {cp_dir}")
    print("=======================================================\n", flush=True)

    subject_results: dict[str, list[BenchmarkScenarioResult]] = {}
    pending_subjects: list[str] = []

    for sid in subjects:
        cp_file = cp_dir / f"{sid}.json"
        if cp_file.exists():
            try:
                raw_data = json.loads(cp_file.read_text(encoding="utf-8"))
                if isinstance(raw_data, list) and len(raw_data) == expected_scenarios:
                    res_list = [BenchmarkScenarioResult(**item) for item in raw_data]
                    subject_results[sid] = res_list
                    f32_4 = next((r.top1_accuracy for r in res_list if r.precision == "float32" and r.num_channels == 4), 0.0)
                    print(f"[RESUME] Loaded {sid} from checkpoint ({len(res_list)} scenarios | 4ch: {f32_4*100:.2f}%)", flush=True)
                    continue
            except Exception as exc:
                print(f"[WARN] Failed to read checkpoint for {sid}: {exc}", flush=True)
        pending_subjects.append(sid)

    start_all = time.time()
    n_pending = len(pending_subjects)
    print(f"\n[QUEUE] {len(subject_results)} subjects cached; {n_pending} subjects remaining to compute.\n", flush=True)

    if n_pending > 0:
        if num_workers > 1 and n_pending > 1:
            tasks = [
                (sid, str(raw_dir), precisions, channel_counts, r_multiplier, use_filtered, verbose, str(cp_dir))
                for sid in pending_subjects
            ]
            completed = len(subject_results)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_sid = {
                    executor.submit(_worker_benchmark_subject, task): task[0]
                    for task in tasks
                }
                for future in as_completed(future_to_sid):
                    sid = future_to_sid[future]
                    try:
                        res_sid, results, duration = future.result()
                        subject_results[res_sid] = results
                        completed += 1
                        f32_4ch = next((r.top1_accuracy for r in results if r.precision == "float32" and r.num_channels == 4), 0.0)
                        i16_4ch = next((r.top1_accuracy for r in results if r.precision == "int16" and r.num_channels == 4), 0.0)
                        print(
                            f"[{completed:2d}/{total_subjects:2d}] {res_sid:4s} completed in {duration:4.1f}s | "
                            f"4-ch float32: {f32_4ch*100:5.2f}% | 4-ch int16: {i16_4ch*100:5.2f}% (Saved checkpoint)",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"[{sid}] FAILED with exception: {exc}", file=sys.stderr, flush=True)
                        raise
        else:
            completed = len(subject_results)
            for sid in pending_subjects:
                t0 = time.time()
                results = run_benchmark_for_subject(
                    subject_id=sid,
                    raw_dir=raw_dir,
                    precisions=precisions,
                    channel_counts=channel_counts,
                    r_multiplier=r_multiplier,
                    use_filtered=use_filtered,
                    verbose=False,
                )
                subject_results[sid] = results
                elapsed = time.time() - t0
                completed += 1
                cp_path = cp_dir / f"{sid}.json"
                cp_path.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n", encoding="utf-8")
                f32_4ch = next((r.top1_accuracy for r in results if r.precision == "float32" and r.num_channels == 4), 0.0)
                print(f"[{completed}/{total_subjects}] {sid} completed in {elapsed:.1f}s | 4-ch Acc: {f32_4ch*100:.2f}%\n", flush=True)

    total_time = time.time() - start_all
    print(f"\n[ALL DONE] All {len(subject_results)} subjects ready in {total_time:.2f}s ({total_time/60:.2f} min).\n")

    summaries = aggregate_population_results(
        subject_results=subject_results,
        precisions=precisions,
        channel_counts=channel_counts,
    )
    return subject_results, summaries


def generate_population_markdown_report(
    subjects: Sequence[str],
    summaries: list[PopulationScenarioSummary],
    output_path: Path,
) -> str:
    """Generate comprehensive Markdown report summarizing Ozdemir population benchmark."""
    n_subj = len(subjects)
    subj_str = f"{n_subj} Sujeitos ({subjects[0]} .. {subjects[-1]})" if n_subj > 3 else ", ".join(subjects)

    lines: list[str] = [
        "# Ozdemir et al. sEMG — Relatório Populacional: Quantização e Redução de Canais",
        "",
        f"> **População Avaliada**: `{subj_str}` ($N = {n_subj}$) | **Classes Globais**: 10 gestos",
        "> **Divisão Protocolar**: LOTO Dentro do Sujeito (Treino reps `[1, 3, 4]`, Teste reps `[2, 5]`)",
        "> **Classificador**: Canonical LDA com Regularização e Prior Uniforme",
        "",
        "---",
        "",
        "## 1. Visão Geral da Análise Populacional",
        "",
        r"Este relatório consolida a média e o desvio padrão populacional ($\mu \pm \sigma$) das métricas de "
        "classificação neuromuscular sob quantização numérica de ADC e redução de canais de eletrodos no dataset Ozdemir et al. (4 canais, 2000 Hz).",
        "",
        "---",
        "",
        "## 2. Tabela Geral de Resultados Populacionais (Média ± Desvio Padrão)",
        "",
        "| Precisão | Canais | Features | Bytes/Amostra | Redução (%) | Top-1 Acc (%) | Δ vs Base 4ch (%) | Balanced Acc (%) | Macro F1 (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in summaries:
        delta_str = f"{s.mean_delta_acc_vs_baseline*100:+.2f} ± {s.std_delta_acc_vs_baseline*100:.2f}%" if not (s.precision == "float32" and s.num_channels == 4) else "REF"
        lines.append(
            f"| `{s.precision}` | {s.num_channels} | {s.num_features} | {s.bytes_per_sample} B | "
            f"{s.data_rate_reduction_ratio*100:.1f}% | "
            f"**{s.mean_top1_accuracy*100:.2f} ± {s.std_top1_accuracy*100:.2f}%** | {delta_str} | "
            f"{s.mean_balanced_accuracy*100:.2f} ± {s.std_balanced_accuracy*100:.2f}% | "
            f"{s.mean_macro_f1*100:.2f} ± {s.std_macro_f1*100:.2f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Impacto Populacional da Quantização (Fixando 4 Canais)",
        "",
        "| Precisão | Resolução Efetiva | Top-1 Acc (Média ± DP) | Balanced Acc (Média ± DP) | Macro F1 (Média ± DP) | Δ vs Base 4ch |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    p4 = [s for s in summaries if s.num_channels == 4]
    for s in p4:
        desc = {
            "float32": "Single precision (24-bit mantissa)",
            "float16": "Half precision (11-bit mantissa)",
            "int32": "32-bit ADC (passo ~4.65 pV em 10 mV)",
            "int16": "16-bit ADC (passo ~0.305 µV em 10 mV)",
            "int8": "8-bit ADC (passo ~78.7 µV em 10 mV)",
        }.get(s.precision, "")
        delta_str = f"{s.mean_delta_acc_vs_baseline*100:+.2f} ± {s.std_delta_acc_vs_baseline*100:.2f}%" if s.precision != "float32" else "REF"
        lines.append(
            f"| `{s.precision}` | {desc} | **{s.mean_top1_accuracy*100:.2f} ± {s.std_top1_accuracy*100:.2f}%** | "
            f"{s.mean_balanced_accuracy*100:.2f} ± {s.std_balanced_accuracy*100:.2f}% | "
            f"{s.mean_macro_f1*100:.2f} ± {s.std_macro_f1*100:.2f}% | {delta_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Impacto Populacional da Redução de Canais (Baseline Float32)",
        "",
        "| Canais | Features | Top-1 Acc (Média ± DP) | Balanced Acc (Média ± DP) | Macro F1 (Média ± DP) | Δ vs Base 4ch |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    c_f32 = [s for s in summaries if s.precision == "float32"]
    for s in c_f32:
        delta_str = f"{s.mean_delta_acc_vs_baseline*100:+.2f} ± {s.std_delta_acc_vs_baseline*100:.2f}%" if s.num_channels != 4 else "REF"
        lines.append(
            f"| {s.num_channels} ch | {s.num_features} | "
            f"**{s.mean_top1_accuracy*100:.2f} ± {s.std_top1_accuracy*100:.2f}%** | "
            f"{s.mean_balanced_accuracy*100:.2f} ± {s.std_balanced_accuracy*100:.2f}% | "
            f"{s.mean_macro_f1*100:.2f} ± {s.std_macro_f1*100:.2f}% | {delta_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Fronteira de Eficiência Populacional (Pareto Optimal)",
        "",
        "| Taxa / Amostra | Economia Dados | Configuração | Top-1 Acc Populacional | Balanced Acc Populacional | Macro F1 Populacional | Recomendação |",
        "| :---: | :---: | :--- | :---: | :---: | :---: | :--- |",
    ])

    sorted_by_bytes = sorted(summaries, key=lambda x: (x.bytes_per_sample, -x.mean_balanced_accuracy))
    pareto: list[PopulationScenarioSummary] = []
    max_bal = -1.0
    for cand in sorted_by_bytes:
        if cand.mean_balanced_accuracy > max_bal:
            pareto.append(cand)
            max_bal = cand.mean_balanced_accuracy

    for p in pareto:
        rec_desc = "Candidato Pareto"
        if p.num_channels == 4 and p.precision == "int16":
            rec_desc = "**Melhor Escolha Prótese Embarcada (-50% banda)**"
        elif p.num_channels == 4 and p.precision == "int8":
            rec_desc = "**Melhor Relação BLE/IoT (-75% banda)**"
        lines.append(
            f"| **{p.bytes_per_sample} B** | **{p.data_rate_reduction_ratio*100:.1f}%** | "
            f"`{p.precision}` @ {p.num_channels} ch | "
            f"{p.mean_top1_accuracy*100:.2f} ± {p.std_top1_accuracy*100:.2f}% | "
            f"{p.mean_balanced_accuracy*100:.2f} ± {p.std_balanced_accuracy*100:.2f}% | "
            f"{p.mean_macro_f1*100:.2f} ± {p.std_macro_f1*100:.2f}% | {rec_desc} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Conclusões e Recomendações Estatísticas",
        "",
        f"- **Validação Populacional ($N = {n_subj}$)**: Avaliação completa de 10 gestos com LOTO entre repetições nos 40 sujeitos.",
        "- **Invariância de Precisão**: Confirmação se `int16` preserva a acurácia do `float32` também em sensores de 4 canais.",
        "- **Sensibilidade de Canais**: Quantificação do impacto ao reduzir de 4 para 3, 2 e 1 eletrodo(s).",
        "",
    ])

    report_text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    return report_text


def find_all_available_subjects(raw_dir: Path) -> list[str]:
    """Discover all Ozdemir subjects available in raw directory sorted numerically."""
    mat_dir = raw_dir / "sEMG-dataset" / "filtered" / "mat"
    subjects: list[int] = []
    for p in mat_dir.glob("*_filtered.mat"):
        m = re.match(r"(\d+)_filtered\.mat", p.name)
        if m:
            subjects.append(int(m.group(1)))
    subjects.sort()
    return [f"S{num}" for num in subjects]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default=None, help="Single Subject ID (e.g. S1)")
    parser.add_argument("--subjects", nargs="+", default=None, help="List of Subject IDs (e.g. S1 S2 S3)")
    parser.add_argument("--all-subjects", action="store_true", help="Run across all subjects in data/raw/ozdemir")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel worker processes (default: 2)",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=list(PRECISION_OPTIONS),
        choices=list(PRECISION_OPTIONS),
        help="Precisions to evaluate",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=list(CHANNEL_OPTIONS),
        help="Channel counts to evaluate",
    )
    parser.add_argument("--save", action="store_true", help="Save reports to disk")
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw" / "ozdemir"

    if args.all_subjects:
        subjects = find_all_available_subjects(raw_dir)
        if not subjects:
            print(f"Error: No Ozdemir mat files found in {raw_dir}", file=sys.stderr)
            return 1
    elif args.subjects:
        subjects = list(args.subjects)
    elif args.subject:
        subjects = [args.subject]
    else:
        subjects = ["S1"]

    if len(subjects) == 1:
        sid = subjects[0]
        results = run_benchmark_for_subject(
            subject_id=sid,
            raw_dir=raw_dir,
            precisions=args.precisions,
            channel_counts=args.channels,
            verbose=True,
        )
        if args.save:
            json_path = ROOT / "reports" / "benchmarks" / "ozdemir_quantization_and_channel_reduction.json"
            data_dict = {
                "subject_id": sid,
                "v_fs_mv": V_FS_MV,
                "results": [asdict(r) for r in results],
            }
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(data_dict, indent=2) + "\n", encoding="utf-8")
            print(f"\n[OK] Single-subject reports saved: {json_path}")
        return 0

    subject_results, summaries = run_benchmark_multi_subject(
        subjects=subjects,
        raw_dir=raw_dir,
        precisions=args.precisions,
        channel_counts=args.channels,
        num_workers=args.workers,
        verbose=False,
    )

    if args.save:
        pop_json_path = ROOT / "reports" / "benchmarks" / "ozdemir_quantization_and_channel_reduction_population.json"
        pop_md_path = ROOT / "reports" / "benchmarks" / "ozdemir_quantization_and_channel_reduction_population_report.md"

        pop_data_dict: dict[str, Any] = {
            "num_subjects": len(subjects),
            "subjects": subjects,
            "v_fs_mv": V_FS_MV,
            "population_summary": [asdict(s) for s in summaries],
            "subject_results": {sid: [asdict(r) for r in res] for sid, res in subject_results.items()},
        }
        pop_json_path.parent.mkdir(parents=True, exist_ok=True)
        pop_json_path.write_text(json.dumps(pop_data_dict, indent=2) + "\n", encoding="utf-8")
        generate_population_markdown_report(subjects, summaries, pop_md_path)
        print("\n[OK] Ozdemir Population reports saved successfully:")
        print(f"  Population JSON: {pop_json_path}")
        print(f"  Population Markdown: {pop_md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
