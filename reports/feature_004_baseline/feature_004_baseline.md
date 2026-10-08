# Relatório de Execução e Aceite Científico — Feature 004

**Data de Execução (UTC)**: `2026-10-08T10:05:21.819297+00:00`  
**Duração Total**: `87.21s`  
**Ambiente**: Python `3.14.7`, NumPy `2.5.3`, SciPy `1.18.1`, Scikit-learn `1.9.1`  

---

## 1. Resumo dos Resultados Científicos

| Camada / Diagnóstico | Métrica Principal | Resultado Obtido | Tolerância / Critério | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1: Lineage & Splits** | Hashes Imutáveis | 100% Coincidentes | Zero divergência com Feature 003 | **PASS** |
| **Layer 2: Within-Subject (LOTO)** | Test Balanced Accuracy | `100.00%` | Test evaluated strictly once | **PASS** |
| **Layer 3: Cross-Subject** | Fold 1 Test Accuracy | `100.00%` | 24 train / 8 val / 8 test, 1 eval | **PASS** |
| **Layer 4: Dual-Vocabulary Reporting** | Primary Top-1 / Active Top-1 | `95.00%` / `95.16%` | CM [50, 50] e [49, 50] | **PASS** |
| **Layer 5: Confounding Audit** | REST Exercise Accuracy | `44.67%` | Chance nominal: 33.33% (delta <= 5.0%) | **PASS** |
| **Layer 6: Shuffled Negative Control**| Permuted Test Top-1 | `15.48%` | Threshold: <= 35.0% | **PASS** |
| **Layer 7: Neutral Export Oracle** | Class Agreement / Score Diff | `100.0% Match` / `0.00e+00` | rtol=1e-9, atol=1e-7 (`TOL-LDA-NEUTRAL-SCORES`) | **PASS** |
| **Layer 8: Append-Only Sealing** | Integridade `manifest.sha256`| Válida (Zero Tampering) | Conformidade estrita com schema | **PASS** |

---

## 2. Artefatos Gerados

- `reports/feature_004_baseline/execution_report.json`
- `reports/feature_004_baseline/feature_004_baseline.md`
- `reports/feature_004_baseline/manifest.sha256`
