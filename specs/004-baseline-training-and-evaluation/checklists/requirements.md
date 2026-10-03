# Specification Quality Checklist: 004-baseline-training-and-evaluation

**Purpose**: Validate specification completeness and quality before proceeding to CLARIFY and PLAN  
**Created**: 2026-10-02  
**Feature**: [specs/004-baseline-training-and-evaluation/spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs leaking into user requirements)
- [x] Focused on user value and business needs (scientific baseline benchmarking, reproducible evidence)
- [x] Written for scientific and engineering stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain (Q1–Q5 fully resolved in Session 2026-10-03)
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (verifiable outcomes)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded (explicit exclusions of INT8, C++, ESP32, architecture search)
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows (Within-subject, Cross-subject, Feature extraction, TrainingJob, Reporting)
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

- Clarifications Resolved (Session 2026-10-03):
  1. `[RESOLVED]`: Governança e aprovação de `scikit-learn` formalizada via ADR persistente em `docs/adr/ADR-004-001-scikit-learn-host-only.md` com gate automatizado e allowlists (FR-006).
  2. `[RESOLVED]`: Parâmetros de DSP causal (band-pass 20–450 Hz, notch 50 Hz omitido após auditoria espectral de $R_{50}=0.84$), janelas de 200 ms / 50 ms stride, política de janelas puras, thresholds adaptativos de treino e z-score customizado (FR-004, FR-012, FR-013).
  3. `[RESOLVED]`: Modelagem LDA com priors uniformes primários ($P=1/50$) e empíricos secundários em fits distintos, shrinkage de Ledoit-Wolf analítico, fronteira neutra de exportação numérica e oráculo de equivalência determinística (FR-005).
  4. `[RESOLVED]`: Taxonomia hierárquica de modelos: `global_50_class` (Primário), `E1/E2/E3 exercise-stratified` (Secundário) e `exercise_id_from_rest` (Auditoria diagnóstica de confounding) (FR-007).
  5. `[RESOLVED]`: Contratos contratuais em `contracts/`, persistência append-only em `artifacts/feature_004/jobs/<job_id>/` com fingerprint determinístico, fixtures sintéticas procedurais e três camadas de oráculos (FR-003, FR-011).
- Checklist 100% aprovado (16/16 itens passing). Ciclo CLARIFY encerrado. Pronto para iniciar `$speckit-plan`.
