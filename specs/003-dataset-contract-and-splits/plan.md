# Implementation Plan: 003-dataset-contract-and-splits

**Branch**: `feat/003-dataset-contract-and-splits` | **Date**: 2026-10-02 | **Spec**: [specs/003-dataset-contract-and-splits/spec.md](spec.md)

**Input**: Feature specification from `/specs/003-dataset-contract-and-splits/spec.md`

---

## Summary

Feature 003 establishes the official, audit-grade dataset contract and reproducible partitioning pipeline for the NinaPro DB2 dataset (`data/raw/ninapro_db2/`). It implements a read-only, non-destructive loader reading directly from memory/ZIP streams, pre-verifying cryptographic SHA-256 archive hashes, and enforcing the **PROVE-OR-QUARANTINE** temporal alignment policy on recordings with array length discrepancies. It introduces **Frozen Dataset Views** (`refined` default vs `stimulus` secondary), a **Dual Partitioning Protocol** (Within-Subject repetition split and 5-fold Cross-Subject split), separate REST auditing, and complete official provenance documentation.

---

## Technical Context

**Language/Version**: Python 3.10+ (tested on Python 3.14.7)  
**Primary Dependencies**: `numpy` (>=1.26), `scipy` (>=1.12, for `scipy.io.loadmat` in memory), standard library (`zipfile`, `hashlib`, `json`, `dataclasses`, `pathlib`, `typing`)  
**Storage**: Strictly read-only consumption of `data/raw/ninapro_db2/` (40 ZIP files, 17.00 GiB); generated manifests written to `data/manifests/ninapro_db2/`  
**Testing**: `pytest` (unit, regression, contract, and leak-free verification), `ruff` (linter), `mypy` (strict static typing)  
**Target Platform**: Linux host-only runtime  
**Project Type**: Data contract library and manifest generation tools (`semg_dataset/` module)  
**Performance Goals**: Parse and validate any single subject recording in < 500 ms in memory; generate complete 40-subject dual-split manifests in < 30 seconds  
**Constraints**: Zero file writes or extraction in `data/raw/ninapro_db2/`; zero data leakage across split partitions; zero promotion to `float64` in sEMG arrays; zero tampering with Feature 002 DSP contracts  
**Scale/Scope**: 40 participants (`S1`–`S40`), 3 exercises (`E1`, `E2`, `E3`), 120 MATLAB recordings, 207,713,115 sEMG samples, 12 channels  

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

- **Principle I (Python Central Orchestrator & Deterministic Gatekeeper)**: PASS. All transitions, data validation, and split assertions are executed deterministically in Python. Quality gates (`pytest`, `ruff`, `mypy`, `compileall`) gate all tasks.
- **Principle II (Strict TDD Lifecycle & Anti-Tampering Protection)**: PASS. All contracts, readers, alignment policies, and split generators will be implemented via strict RED → GREEN → REFACTOR cycles with SHA-256 test snapshotting.
- **Principle III (Non-Destructive Local Reuse & Zero-Subprocess Read Paths)**: PASS. Raw dataset access is strictly read-only and memory-based via `zipfile` and `scipy.io.loadmat(io.BytesIO)`. No permanent disk extraction, no shelling out, no network calls.
- **Principle IV (CLI Ergonomics, Text/JSON Duality & Failure Containment)**: PASS. Manifest generation and verification commands output human-readable summaries and pure machine-readable JSON.
- **Principle V (SpecKit Traceability & Living Documentation Supremacy)**: PASS. End-to-end traceability maintained from requirements (`FR-001`–`FR-012`) through data models, contracts, and tasks.
- **Principle VI (Scientific and Numerical Oracle Integrity)**: PASS. NinaPro DB2 provenance is fully documented; archive SHA-256 checksums are frozen and verified; golden fixtures and reference splits remain immutable.

---

## Project Structure

### Documentation (this feature)

```text
specs/003-dataset-contract-and-splits/
├── spec.md              # Feature specification with Q1–Q5 resolutions
├── plan.md              # This file (implementation plan)
├── research.md          # Phase 0 decisions & alternatives
├── data-model.md        # Phase 1 domain entities & validation rules
├── quickstart.md        # Phase 1 usage & validation scenarios
├── checklists/
│   └── requirements.md  # Quality validation checklist
└── contracts/
    ├── recording_schema.json  # Schema for canonical recording objects
    └── split_schema.json      # Schema for split manifest partitions
```

### Source Code Layout

```text
semg_dataset/
├── __init__.py          # Public package exports
├── contract.py          # RecordingData, DatasetView, Subject, Exercise definitions
├── loader.py            # NinaProDB2Loader (read-only, memory stream from ZIPs)
├── alignment.py         # ProveOrQuarantineEngine & anchor_start_truncate_tail
├── splits.py            # WithinSubjectSplitter & CrossSubjectSplitter (5 folds)
├── provenance.py        # Official citations, URLs, licenses, archive verification
└── manifest.py          # DatasetCatalog & SplitManifest generators

tests/
├── test_ninapro_loader.py         # Archive reading, SHA-256 pre-check, immutability
├── test_prove_or_quarantine.py    # Temporal alignment, S12 outlier, quarantine checks
├── test_frozen_views.py           # Refined vs stimulus views, pair-mixing prevention
├── test_splits_within_subject.py  # Within-subject split integrity & zero leakage
├── test_splits_cross_subject.py   # Cross-subject 5-fold integrity & zero leakage
└── test_dataset_manifests.py      # Manifest serialization & bitwise reproducibility
```

**Structure Decision**: A dedicated `semg_dataset/` package alongside `semg_dsp/`, keeping dataset loading, contract validation, and split generation separate from DSP filtering and windowing.

---

## Complexity Tracking

> *No constitutional violations. All implementations adhere strictly to core principles.*
