# Acceptance and Verification Report — Feature 003

## Executive Summary

- **Feature**: `003-dataset-contract-and-splits`
- **Audit Date**: `2026-10-02T08:18:05.817541+00:00`
- **Git Commit**: `7dfceeb42211d7e2a70d4d1f18139e611cfceb2c`
- **Git Branch**: `feat/003-dataset-contract-and-splits`
- **Overall Status**: **`PASS`**
- **Total Audit Time**: `3.67 s`

---

## 7-Layer Independent Verification Results

| Layer | Component | Target | Result | Status |
|---|---|---|---|:---:|
| **1** | Raw Data & Hashes | 40 ZIPs, 120 .mat, 0 NaN/Inf | 100% matched baseline, 0 NaNs | **PASS** |
| **2** | Label Contracts | Vocabulary E1(0..17), E2(0,18..40), E3(0,41..49) | 100% within bounds | **PASS** |
| **3** | PROVE-OR-QUARANTINE | 18 E3 mismatches, S12 outlier | 18/18 proven start-anchored | **PASS** |
| **4** | Frozen Dataset Views | Refined vs Stimulus isolation | Anti-mixing strictly enforced | **PASS** |
| **5** | Split Algebra | Within (1,3,4,6 / 2,5), Cross (5 folds 24/8/8) | Zero leakage, complete | **PASS** |
| **6** | Adversarial Attacks | 8 deliberate leakage injections | 8/8 detected & rejected | **PASS** |
| **7** | Reproducibility | 5 consecutive manifest generation runs | Bit-to-bit identical hashes | **PASS** |
| **Gate**| Raw Immutability | `data/raw/ninapro_db2` | Clean, 0 writes/modifications | **PASS** |
| **Gate**| Quality Gates | pytest, ruff, mypy, compileall | All gates passed | **PASS** |

---

## S12 Outlier Scientific Alignment Evidence

```text
Recording ................... S12_E3_A1.mat
Original EMG Length ......... 875707 samples
Original Stimulus Length .... 875707 samples
Original Repetition Length .. 875707 samples
Original Restimulus Length .. 875435 samples
Original Rerepetition Length  875435 samples
Delta ....................... 272 samples (136.0 ms @ 2000 Hz)
Truncated Side .............. tail (cauda)
Discarded Interval .......... [875435:875707] (arrays: emg, stimulus, repetition)
Target Canonical Length ..... 875435 samples
Start Anchor (t=0) .......... PASS (restimulus[0]==0, stimulus[0]==0, reaction lag = 646.0 ms)
Tail Boundary (at cutoff) ... PASS (restimulus[875434] == 0)
Discarded Interval Proof .... PASS: Motor gesture (restimulus) ended at sample 872711. Physical rest persists for 2723 samples up to canonical cutoff at sample 875434. Discarded EMG interval [875435:875707] exhibits noise-floor RMS (0.000014 V vs 0.000093 V active gesture; baseline rest RMS 0.000019 V). Trailing visual prompt stimulus=49 reflects visual software prompt overrun after muscular cessation.
Final Engine Decision ....... ACCEPT_AUTO
Applied Policy .............. anchor_start_truncate_tail
```

---

## Adversarial Leakage Attacks (Layer 6)

| Attack ID | Description | Detected | Status |
|---|---|:---:|:---:|
| `ATK-01` | Inject subject overlap between train and test (S5 in both) | **True** | **PASS** |
| `ATK-02` | Inject repetition overlap between train and test (rep 2 in both) | **True** | **PASS** |
| `ATK-03` | Inject subject overlap between train and val (S3 in both) | **True** | **PASS** |
| `ATK-04` | Inject subject overlap between val and test (S4 in both) | **True** | **PASS** |
| `ATK-05` | Inject duplicate subject inside train partition ([S1, S1]) | **True** | **PASS** |
| `ATK-06` | Inject duplicate subject inside val partition ([S2, S2]) | **True** | **PASS** |
| `ATK-07` | Inject duplicate subject inside test partition ([S3, S3]) | **True** | **PASS** |
| `ATK-08` | Inject multiple overlapping repetitions ([2, 5] in both train and test) | **True** | **PASS** |

---

## Deterministic Quality Gates

- `pytest`: **PASS** (128/128 tests passing)
- `ruff`: **PASS** (0 lint violations)
- `mypy`: **PASS** (Strict static typing, 0 errors)
- `compileall`: **PASS** (0 syntax errors)
- `raw_immutability`: **PASS** (`data/raw/ninapro_db2` unmodified)

---

```text
RAW DATA INTEGRITY ............ PASS
LABEL CONTRACT ................ PASS
ALIGNMENT AUDIT ............... PASS
FROZEN VIEWS .................. PASS
WITHIN-SUBJECT SPLIT .......... PASS
CROSS-SUBJECT 5-FOLD .......... PASS
REST ASSIGNMENT ............... PASS
LEAKAGE NEGATIVE CONTROLS ..... PASS
MANIFEST REPRODUCIBILITY ...... PASS
RAW IMMUTABILITY .............. PASS
QUALITY GATES ................. PASS

FINAL STATUS:
PASS
```
