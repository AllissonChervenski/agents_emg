# Feature Specification: 004-baseline-training-and-evaluation

**Feature Branch**: `feat/004-baseline-training-and-evaluation`  
**Created**: 2026-10-02  
**Status**: Draft (Ready for Clarify)  
**Input**: User description: "A Feature 004 deve estabelecer um baseline offline, reproduzível e auditável de classificação de gestos sEMG sobre o NinaPro DB2 usando os contratos congelados da Feature 003. Baseline inicial: classical sEMG time-domain features (MAV, RMS, WL, ZC, SSC) + regularized LDA. Vocabulário primário: 50 classes (REST 0 + gestos 1..49) e secundário: 49 classes ativas. Avaliação within-subject com leave-one-training-repetition-out em [1,3,4,6] e teste único em [2,5]. Avaliação cross-subject nos 5 folds congelados. Contrato formal de TrainingJob com rastreabilidade total de linhagem e hashes científicos. Pré-processamento e thresholds detalhados no CLARIFY."

## Clarifications

### Session 2026-10-03

- **Q1: Governança e inclusão de `scikit-learn` como dependência host-only (FR-006)**:
  - **Contexto de Governança Ativa**: Verificado em [`.specify/memory/constitution.md`](file:///home/allisson/Projects/agents_emg/.specify/memory/constitution.md) (Versão ativa: `1.1.0`, ratificada em 2026-09-28, alterada em 2026-09-29, commit Git ativo: `ce75be0`). A Linha 51 estipula: *"Features MUST NOT introduce new third-party dependencies without prior governance approval."* Nenhuma cláusula da Constituição v1.1.0 define "prior governance approval" como restrita ou obrigatoriamente equivalente a uma emenda direta do arquivo constitucional.
  - **Decisão Aprovada**: Opção C com **governança formal endurecida**. A aprovação de `scikit-learn` é materializada via Architecture Decision Record (ADR) persistente em [`docs/adr/ADR-004-001-scikit-learn-host-only.md`](file:///home/allisson/Projects/agents_emg/docs/adr/ADR-004-001-scikit-learn-host-only.md), sem criar Constitution v1.2.0.
  - **Cláusulas Verificáveis Obrigatórias**:
    1. *Aprovação Humana Formal*: O ADR inicia com status `PROPOSED`, recebe nome/ID do aprovador humano, data de aprovação e commit Git da aprovação; somente após isso atinge status `APPROVED`. O conteúdo aprovado é selado por hash/fingerprint criptográfico contra adulteração.
    2. *Finalidade e Isolamento Host-Only*: `scikit-learn` destina-se exclusivamente ao baseline offline e avaliação da Feature 004. É terminantemente proibido qualquer vínculo com `semg_dsp` ou runtimes embarcados.
    3. *Allowlist de Caminhos de Import*: Imports de `sklearn` permitidos exclusivamente sob `semg_baseline/**` e testes dedicados da Feature 004 (`tests/feature_004/**` ou pacote equivalente).
    4. *Allowlist de Superfície de API*: Permitidos apenas submódulos de `sklearn.discriminant_analysis`, `sklearn.covariance` e `sklearn.metrics`. Ficam estritamente proibidos `sklearn.model_selection`, `sklearn.pipeline.Pipeline`, `sklearn.preprocessing.StandardScaler` e quaisquer geradores de splits aleatórios. Splits vêm 100% dos manifests congelados da Feature 003.
    5. *Portão Determinístico Automatizado*: Gate com bloqueio `FAIL` (nunca warning) que valida status `APPROVED` do ADR, assinatura humana, pinagem de versão, integridade do hash do ADR e respeito às allowlists de paths e símbolos.
    6. *Reprodutibilidade & Locks*: Separação formal entre ADR (aprovação de faixa/escopo), Lockfile de ambiente (congelando `scikit-learn`, `numpy`, `scipy`, `joblib`, `threadpoolctl` e transitivas) e `TrainingJob` (registrando hashes e versões resolvidas).
    7. *Fronteira Estrita de Exportação*: Proibição absoluta de `pickle`, `joblib.dump` ou objetos sklearn como formato de intercâmbio de modelo. O baseline treinado deve ser exportado em formato neutro, determinístico e puramente numérico, consumível pela Feature 005 sem dependência de `sklearn`.
    8. *Ciclo de Mutação do `pyproject.toml`*: Sequência imutável: `PROPOSED ADR` → revisão/aprovação humana → commit da aprovação → verificação do gate de governança → alteração autorizada de `pyproject.toml` → lock de dependências → re-execução dos gates. O agente não assume ADR aprovado por tê-lo criado.

- **Q2: Parâmetros de DSP, Janelamento, Transições, Thresholds ZC/SSC e Normalização (FR-004, FR-012, FR-013)**:
  - **Decisão Aprovada**: **CUSTOM D — Baseline conservador com janelas puras e parâmetros aprendidos exclusivamente no TRAIN**.
  - **Resultado da Auditoria Espectral de 50 Hz (NinaPro DB2)**:
    - Executada auditoria espectral Welch (resolução 0.5 Hz) sobre os 12 canais sEMG em amostra cobrindo todos os 5 folds (`S1`, `S5`, `S10`, `S15`, `S20`, `S25`, `S30`, `S35`, `S40`).
    - Razão média de potência em 50 Hz vs bandas adjacentes (45–48 Hz e 52–55 Hz) = **0.84** (< 1.0, demonstrando que o sinal em 50 Hz é plano ou ligeiramente inferior à densidade espectral mioelétrica adjacente). Apenas 2 sujeitos apresentaram elevações pontuais em canais isolados (S5 ch9 = 2.54 / +4.05 dB, S15 ch7 = 2.04 / +3.09 dB), mas com médias globais de canais de 1.03 e 0.98.
    - **Conclusão Científica**: O NinaPro DB2 (adquirido com sensores ativos diferenciais Delsys Trigno) **não apresenta interferência sistemática de linha de 50 Hz**. Um filtro notch em 50 Hz atenuaria desnecessariamente a potência fisiológica do sEMG na faixa de maior densidade de disparo motor (20–150 Hz). Portanto, o **filtro notch de 50 Hz fica formalmente OMITIDO do baseline canônico**.
  - **Cláusulas Verificáveis Obrigatórias de Preprocessing & Janelamento**:
    1. *DSP Causal*: Filtro Butterworth causal de 4ª ordem, passa-faixa 20–450 Hz, executado via [`CausalSosFilter`](file:///home/allisson/Projects/agents_emg/semg_dsp/filter.py). Coeficientes SOS, resposta em frequência, versão e fixtures goldens são ativos próprios da Feature 004 (sem herança silenciosa da Feature 002).
    2. *Janelamento Temporal*: Sampling rate nominal = 2000.0 Hz, Janela $W = 200\,\text{ms}$ (400 amostras), Stride $S = 50\,\text{ms}$ (100 amostras), Overlap = 75%, Taxa de decisão = 20 Hz. Invariante estrito: nenhuma janela pode cruzar `repetition_unit` ou fronteira de partição de split.
    3. *Política de Transição (Janelas Estritamente Puras)*: Proibido o uso de voto majoritário no baseline canônico. Uma janela só é elegível se todas as 400 amostras pertencerem ao mesmo estado canônico e mesma unidade de repetição:
       - Janela 100% ACTIVE de classe $c \in \{1..49\}$ → incluída;
       - Janela 100% REST (0) → incluída;
       - Transição REST ↔ ACTIVE ou ACTIVE ↔ ACTIVE → descartada;
       - Janela que cruze `repetition_unit` → descartada.
       - Quantidade e duração de janelas descartadas registradas por sujeito/exercício para auditoria. A análise de transições é desacoplada do baseline steady-state.
    4. *Thresholds ZC/SSC Derivados do Treino*: Não há threshold estático universal. $\epsilon[\text{channel}] = R \times \text{RMS\_rest\_train}[\text{channel}]$. Grade congelada de candidatos: $R \in \{0.5, 1.0, 1.5, 2.0, 3.0\}$. A seleção de $R$ ocorre exclusivamente por validação interna no training pool (Within-subject: REST das repetições de treino `[1, 3, 4, 6]`; Cross-subject: REST dos 24 sujeitos de treino do fold). Um único $R$ selecionado por configuração/fold. Thresholds persistidos no `TrainingJob`.
    5. *Feature Scaling Z-Score Customizado*: $z_j = (x_j - \text{mean\_train}_j) / \text{std\_train}_j$. `mean` e `std` calculados unicamente no TRAIN. Validação e teste transformados sem estado. Se $\text{std}_j == 0$, fixar $\text{scale}_j = 1.0$ e registrar `zero_variance`. Proibido `sklearn.preprocessing.StandardScaler`. Persistir `mean[]` e `scale[]` como arrays numéricos neutros.
    6. *Invariante Anti-Vazamento*: Zero dependência de validação ou teste para RMS de repouso, thresholds ZC/SSC, estatísticas de scaling, parâmetros de shrinkage ou qualquer métrica aprendida.

- **Q3: Configuração de Priors de Classe, Shrinkage e Fronteira de Exportação do LDA (FR-005, FR-010)**:
  - **Decisão Aprovada**: **Opção A+ — Priors uniformes no baseline primário, Ledoit-Wolf analítico automático, semântica correta de métricas e oráculo de equivalência**.
  - **Cláusulas Verificáveis Obrigatórias de Modelagem & Inferência**:
    1. *Priors Uniformes Primários*: Baseline canônico de 50 classes configurado com $P(c) = 1/50$ para todo $c \in \{0..49\}$ (`uniform_prior_baseline`). O objetivo é avaliar a separabilidade intrínseca das 50 classes sem conceder vantagem bayesiana prévia à prevalência natural do REST. O efeito nas métricas é estritamente medido, sem alegação a priori de maximização de Balanced Accuracy/Macro-F1.
    2. *Modelo Secundário com Priors Empíricos*: Treinado e reportado separadamente como `empirical_prior_reference`, com $P(c) = N_c / N_{\text{train}}$ medido estritamente no TRAIN, para avaliar a sensibilidade do baseline à prevalência contínua. Nunca escolher entre os dois modelos usando TEST.
    3. *Fits Distintos Obrigatórios*: Os dois modelos devem ser ajustados como fits independentes, pois os priors afetam tanto a construção da covariância agrupada quanto o termo log-prior da função discriminante no scikit-learn.
    4. *Shrinkage Analítico de Ledoit-Wolf*: `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` tratado como estimador analítico determinístico de Ledoit-Wolf, sem busca em grid ou seleção via validation/test.
    5. *Evidências de Shrinkage & Metadados*: Não persistir um único `lambda_star` global (o sklearn estima covariâncias shrunk internamente por classe antes do agrupamento). Persistir obrigatoriamente no `TrainingJob`: `shrinkage_method = "ledoit_wolf_auto"`, versão exata do `scikit-learn`, solver, priors, `classes_`, `means_`, `covariance_`, `coef_`, `intercept_` e hashes SHA-256 dos arrays exportados.
    6. *Fronteira Neutra de Exportação (Feature 005)*: Nenhum objeto `sklearn` (pickle, joblib dump) pode ser formato de intercâmbio. O modelo treinado é exportado em formato numérico puro (`classes`, `coef`, `intercept`, `mean`, `scale`, configurações e hashes), permitindo que a Feature 005 execute inferência via $\text{scores} = X_{\text{scaled}} @ \text{coef}^T + \text{intercept}$ e $\text{argmax}$ sem importar `scikit-learn`.
    7. *Oráculo de Equivalência Determinística*: Teste unitário independente na Feature 004 validando $\text{sklearn.predict}(X) == \text{neutral\_numpy\_predict}(X)$ com igualdade exata de classes e tolerância numérica controlada nos scores de decisão sobre fixtures permitidos.
    8. *Imutabilidade dos Splits de Teste*: Nenhuma escolha de priors, shrinkage ou hiperparâmetro pode utilizar repetições `[2, 5]` ou folds de teste cross-subject.

- **Q4: Taxonomia de Modelos (Global 50 Classes vs Modelos por Exercício) e Auditoria de Confounding (FR-007, FR-010)**:
  - **Decisão Aprovada**: **Opção A+ — Modelo global de 50 classes como baseline primário, modelos por exercício como análises secundárias e auditoria de confounding de sessão/gravação**.
  - **Cláusulas Verificáveis Obrigatórias de Taxonomia & Diagnóstico**:
    1. *Baseline Primário Canônico (`global_50_class`)*: Vocabulário completo de 50 classes (REST 0 + ACTIVE 1..49) combinando dados de E1, E2 e E3 após validação dos contratos da Feature 003 (view, alinhamento, repetições, splits, janelas puras, zero leakage). Representa a avaliação do classificador global para o sistema protético.
    2. *Baselines Secundários por Exercício*: Treinar e reportar automaticamente três modelos complementares usando os labels locais congelados na Feature 003 (`get_local_labels()`):
       - `exercise_E1`: REST + 17 gestos (18 classes)
       - `exercise_E2`: REST + 23 gestos (24 classes)
       - `exercise_E3`: REST + 9 gestos (10 classes)
       Utilizam exatamente o mesmo preprocessing, janelamento, pure windows, features, thresholds ZC/SSC do treino, z-score train-only, LDA/shrinkage e splits correspondentes.
    3. *Status Científico dos Modelos por Exercício*: Descrever na documentação/relatórios como `exercise-stratified secondary baselines` ou `literature-aligned exercise-level analysis` (nunca "replicação exata de Atzori et al.", salvo se todas as condições de pré-processamento, janelamento e classificadores forem idênticas e versionadas).
    4. *Proibição de Média de Acurácia como Substituto*: Reportar separadamente `global_50_class`, `exercise_E1`, `exercise_E2` e `exercise_E3`. Proibido calcular a média das acurácias de E1, E2 e E3 e apresentá-la como substituta do baseline global.
    5. *Auditoria Obrigatória de Confounding (`exercise_id_from_rest`)*: Avaliação diagnóstica independente treinando um classificador para prever `exercise_id \in {E1, E2, E3}` exclusivamente a partir de janelas REST (sob o mesmo split). Desempenho próximo ao acaso (~33.3%) evidencia ausência de assinatura de gravação/sessão; desempenho substancialmente acima do acaso evidencia confounding de sessão/exercício que deve acompanhar explicitamente a interpretação do resultado global. Nenhum sujeito/recording é excluído com base nessa auditoria.
    6. *Hierarquia Científica Estrita*: (1) `global_50_class` [PRIMÁRIO], (2) `E1/E2/E3 exercise-stratified` [SECUNDÁRIO], (3) `exercise_id_from_rest` [DIAGNÓSTICO / AUDITORIA].

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Time-Domain Feature Extraction on Frozen Dataset Views (Priority: P1)

As a biosignal researcher or ML engineer, I need an automated, causal feature extraction pipeline that slides windows across continuous NinaPro DB2 recordings from the frozen `refined` dataset view (`restimulus` + `rerepetition`) to compute classical time-domain sEMG features (MAV, RMS, WL, ZC, SSC) per channel without temporal leakage or lookahead.

**Why this priority**: Essential foundation. Every downstream classifier depends on reliable, reproducible, and mathematically rigorous feature extraction from the 12-channel sEMG signals.

**Independent Test**: Can be tested independently by taking a synthetic or isolated recording, applying windowing with configurable window length $W$ and stride $S$, extracting the 5 time-domain features across all 12 channels, and verifying that the resulting feature matrix has shape `(num_windows, 60)`, contains zero NaN/Inf, and aligns strictly with the corresponding window ground-truth labels.

**Acceptance Scenarios**:
1. **Given** a continuous `RecordingData` object from Feature 003 in `refined` view, **When** feature extraction executes with window length $W$ and stride $S$, **Then** a 2D feature matrix `X` of shape `(num_windows, 12 * num_features)` and label vector `y` of shape `(num_windows,)` are generated with zero NaN or Inf.
2. **Given** window label assignment, **When** a window spans a transition between REST (0) and an active gesture (1..49), **Then** the window label is assigned deterministically according to the policy resolved in CLARIFY without guessing.
3. **Given** any feature scaling (e.g. z-score), **When** normalization statistics are calculated, **Then** statistics are fitted strictly on the training partition and applied statelessly to validation and test partitions with zero leakage.

---

### User Story 2 - Auditable TrainingJob Contract & Cryptographic Provenance (Priority: P1)

As an ML pipeline auditor or system orchestrator, I need every training and evaluation run to execute as an immutable, structured `TrainingJob` contract that captures full scientific provenance—including raw dataset manifest hash, dataset view hash, alignment record hash, split manifest hash, preprocessing configuration hash, model configuration hash, environment/dependency versions, and evaluation metrics.

**Why this priority**: Scientific and numerical oracle integrity (Constitution Principle VI). Without complete lineage tracking and verifiable hashes, baseline classification results cannot be audited, reproduced, or contrasted against future deep learning architectures.

**Independent Test**: Can be tested independently by launching a mock or unit `TrainingJob`, inspecting its JSON summary artifact, and validating that all lineage hashes, configuration signatures, environment versions, and execution status are recorded and verifiable.

**Acceptance Scenarios**:
1. **Given** an execution of a baseline pipeline run, **When** the job finishes (success or failure), **Then** a structured `TrainingJob` record is persisted to disk containing `job_id`, `git_commit`, all Feature 003 input hashes, configuration hashes, environment versions, execution timings, and deterministic output hashes.
2. **Given** an external provider reporting success, **When** postconditions or output hashes fail validation in Python, **Then** the `TrainingJob` status is recorded as failure, enforcing that provider success does not equate to job success.

---

### User Story 3 - Within-Subject Baseline Training with Leave-One-Repetition-Out Validation (Priority: P1)

As a researcher benchmarking individualized prosthetic control, I need to evaluate the regularized LDA baseline on each of the 40 subjects under the Within-Subject protocol, selecting model configuration using leave-one-training-repetition-out over repetitions `[1, 3, 4, 6]`, followed by a single final fit on all training repetitions `[1, 3, 4, 6]` and evaluation on the frozen test repetitions `[2, 5]`.

**Why this priority**: Represents the canonical within-user evaluation protocol of Atzori et al. (2014). It proves whether classical features + regularized LDA can distinguish the NinaPro DB2 gesture set per user under rigorous leakage-free conditions.

**Independent Test**: Can be tested independently by executing the Within-Subject pipeline on a single subject (e.g. `S1`), verifying that 4 internal validation folds rotate across repetitions `[1, 3, 4, 6]`, that final testing evaluates strictly on repetitions `[2, 5]`, and that no test window is ever exposed during training or hyperparameter selection.

**Acceptance Scenarios**:
1. **Given** a Within-Subject split for subject $S_i$, **When** model selection runs, **Then** exactly 4 validation rounds are evaluated (val 1 / train 3,4,6; val 3 / train 1,4,6; val 4 / train 1,3,6; val 6 / train 1,3,4) without observing test repetitions `[2, 5]`.
2. **Given** the selected baseline configuration, **When** final evaluation executes, **Then** the model is fitted on all windows from repetitions `[1, 3, 4, 6]` and evaluated once on repetitions `[2, 5]`.
3. **Given** evaluation completion, **When** results are computed, **Then** Top-1 Accuracy, Balanced Accuracy, Macro-F1, per-class precision/recall, and confusion matrix are generated.

---

### User Story 4 - Cross-Subject 5-Fold Benchmark Evaluation (Priority: P2)

As a machine learning researcher evaluating model generalization across unseen subjects, I need to run the baseline regularized LDA on the 5 frozen cross-subject folds from Feature 003 (24 train, 8 validation, 8 test subjects per fold) without regenerating or altering the fold partitions.

**Why this priority**: Validates inter-subject transferability of the classical baseline features, providing a sound benchmark for future cross-user domain adaptation models.

**Independent Test**: Can be tested independently by executing the cross-subject training runner on Fold 1, confirming that exactly subjects $G_1 \cup G_2 \cup G_3$ (24) are used for training, $G_4$ (8) for validation, and $G_5$ (8) for testing, with zero cross-subject contamination.

**Acceptance Scenarios**:
1. **Given** the Cross-Subject split manifest from Feature 003, **When** training executes for fold $k \in \{1..5\}$, **Then** the pipeline consumes exactly the frozen subject lists for that fold.
2. **Given** completion across all 5 folds, **When** aggregating metrics, **Then** per-fold metrics and aggregated summary statistics (mean, median, standard deviation) are produced.

---

### User Story 5 - Dual-Vocabulary Reporting: 50-Class Primary vs 49-Class Secondary (Priority: P2)

As a scientific reviewer or clinician, I need evaluation metrics reported primarily on the full 50-class vocabulary (REST class 0 + gestures 1..49) and secondarily on the active-only 49-class vocabulary (gestures 1..49), presented with strict segregation to avoid cross-study confusion.

**Why this priority**: NinaPro DB2 continuous recordings are dominated by REST intervals (~75% of duration). Reporting both 50-class and active-only 49-class metrics allows transparent comparison with both continuous prosthetic controller benchmarks and segmented gesture classification literature.

**Independent Test**: Can be tested independently by inspecting evaluation report summaries to confirm that 50-class and 49-class metrics are computed with distinct denomination, separate confusion matrices, and unmixed class counts.

**Acceptance Scenarios**:
1. **Given** model predictions on a test set, **When** generating primary evaluation reports, **Then** metrics are computed over all 50 classes (0..49), with REST explicitly tracked.
2. **Given** the same predictions, **When** generating secondary evaluation reports, **Then** windows labeled REST (0) are excluded or masked according to the active-only protocol, and metrics are computed strictly over classes 1..49.

---

### Edge Cases

- **Constant / Silent Channels**: What happens if an sEMG channel exhibits near-zero variance (e.g. poor electrode contact)? The feature extraction must guard against division-by-zero in SSC/ZC thresholds or feature scaling using a numerical epsilon ($\epsilon = 10^{-8}$).
- **Short or Truncated Repetitions**: What happens if a specific repetition contains fewer samples than window length $W$? The window buffer finalize policy (`drop`) excludes incomplete windows and logs the dropped duration.
- **Extreme Class Imbalance (REST vs Rare Gestures)**: REST (class 0) comprises the majority of sample time. Regularized LDA priors must be explicitly configured (uniform vs empirical) and evaluated via Balanced Accuracy and Macro-F1 to prevent majority-class collapse.
- **Singular Covariance in LDA**: When extracting multi-channel correlated features or when a feature has low variance, the pooled within-class covariance matrix may be ill-conditioned. Regularization (shrinkage / Ledoit-Wolf or empirical ridge parameter $\lambda$) must be applied to guarantee numerical stability.
- **Provider / Subprocess Failure**: If an agent or worker process fails, times out, or produces malformed JSON, the orchestrator catches the failure, records the partial log in `TrainingJob.stderr_summary`, and marks `execution_status = "FAILED"`.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST strictly consume frozen dataset artifacts and manifests from Feature 003 (`data/manifests/ninapro_db2/`, `semg_dataset`), never reading raw ZIPs directly in training routines or reinterpreting split boundaries.
- **FR-002**: System MUST default to the `refined` dataset view (`restimulus` + `rerepetition`) for ground truth labels, with optional support for the `stimulus` view without cross-pair contamination.
- **FR-003**: System MUST record in every `TrainingJob` artifact the following cryptographic provenance hashes:
  - `dataset_manifest_hash` (raw dataset baseline hash)
  - `dataset_view_hash` (hash of the active view)
  - `alignment_hash` (or alignment verification version)
  - `split_manifest_hash` (within-subject or cross-subject split manifest hash)
  - `protocol_id` (`"within_subject"` or `"cross_subject"`)
  - `fold_id` (identifier of the fold or subject evaluated)
  - `acceptance_package_hash` (Feature 003 acceptance package hash)
- **FR-004**: System MUST extract 5 classical sEMG time-domain features per channel:
  - Mean Absolute Value (MAV)
  - Root Mean Square (RMS)
  - Waveform Length (WL)
  - Zero Crossing count (ZC)
  - Slope Sign Change count (SSC)
  Yielding exactly $12 \times 5 = 60$ features per analysis window.
- **FR-005**: System MUST implement regularized Linear Discriminant Analysis (LDA) as the baseline classifier using Ledoit-Wolf covariance shrinkage, maintaining a strict neutral boundary:
  - **FR-005.1 (Primary Baseline with Uniform Priors)**: Canonical 50-class baseline MUST configure uniform priors $P(c) = 1/50$ for all classes $c \in \{0..49\}$ (`uniform_prior_baseline`), removing empirical REST prevalence from the decision rule.
  - **FR-005.2 (Secondary Reference with Empirical Priors)**: A distinct model MUST be trained with empirical priors $P(c) = N_c / N_{\text{train}}$ (`empirical_prior_reference`), evaluated and reported separately to demonstrate sensitivity to continuous class prevalence. Both models MUST be trained as distinct fits.
  - **FR-005.3 (Ledoit-Wolf Analytic Shrinkage)**: System MUST use `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` treated as Ledoit-Wolf analytic shrinkage without grid search or selection on validation/test.
  - **FR-005.4 (Model Parameters & Evidence Persistence)**: System MUST persist in `TrainingJob` artifacts: `shrinkage_method = "ledoit_wolf_auto"`, scikit-learn version, solver `"lsqr"`, priors used, `classes_`, `means_`, `covariance_`, `coef_`, `intercept_`, and SHA-256 hashes of neutral exported arrays.
  - **FR-005.5 (Neutral Parameter Export Boundary)**: Model artifacts MUST be exported in a neutral, deterministic, pure-numerical parameter format (`classes`, `coef`, `intercept`, `mean`, `scale`, metadata) without pickle or sklearn objects. Feature 005 runtime inference MUST execute via $\text{scores} = X_{\text{scaled}} @ \text{coef}^T + \text{intercept}$ and $\text{argmax}$ without importing `sklearn`.
  - **FR-005.6 (Equivalence Oracle Test)**: System MUST include an independent unit test asserting `sklearn.predict(X) == neutral_numpy_predict(X)` with exact predicted class match and numerical tolerance on decision scores for allowed validation/test fixtures.
  - **FR-005.7 (Test Split Immutability)**: No hyperparameter, prior, solver, or shrinkage choice can be made using Within-Subject test repetitions `[2, 5]` or Cross-Subject test folds.
- **FR-006**: System MUST govern `scikit-learn` strictly as a host-only baseline and evaluation dependency via an explicit Architecture Decision Record at [`docs/adr/ADR-004-001-scikit-learn-host-only.md`](file:///home/allisson/Projects/agents_emg/docs/adr/ADR-004-001-scikit-learn-host-only.md) complying with Constitution v1.1.0 (commit `ce75be0`):
  - **FR-006.1 (Human Approval Lifecycle)**: ADR MUST start as `PROPOSED`, receive human approval (name/ID, approval date, git commit), and transition to `APPROVED` sealed by a cryptographic SHA-256 fingerprint against silent mutation.
  - **FR-006.2 (Host-Only Scope & Import Allowlist)**: `sklearn` imports are permitted strictly within `semg_baseline/**` and Feature 004 tests (`tests/feature_004/**` or dedicated test package). Any import in `semg_dsp/`, dataset ingestion, or embedded targets is strictly prohibited.
  - **FR-006.3 (API Surface Allowlist)**: Usage is restricted to `sklearn.discriminant_analysis`, `sklearn.covariance`, and `sklearn.metrics`. Usage of `sklearn.model_selection`, `sklearn.pipeline.Pipeline`, `sklearn.preprocessing.StandardScaler`, or any split generator is prohibited; splits MUST derive exclusively from Feature 003 manifests.
  - **FR-006.4 (Automated Deterministic Gate)**: A verification gate MUST assert that: all dependencies in `pyproject.toml` have approved governance; ADR status is `APPROVED`; human signature is present; version is pinned; import paths and API surface comply with allowlists; and ADR hash fingerprint is intact. Any violation MUST trigger a hard `FAIL` (never warning).
  - **FR-006.5 (Export Boundary & Zero-Pickle Guarantee)**: Trained baseline models MUST be exported to a neutral, deterministic, pure-numerical parameter representation with structured metadata. Serialized Python objects (`pickle`, `joblib.dump`, or sklearn instances) are strictly prohibited as interchange formats; Feature 005 runtime inference MUST NOT import `sklearn`.
  - **FR-006.6 (pyproject.toml Mutation Sequence)**: Agents MUST NOT alter `pyproject.toml` without prior human commit of the approved ADR. The sequence is strictly: `PROPOSED ADR` → human review/approval commit → governance gate verification → authorized `pyproject.toml` edit → dependency lockfile generation → gate re-run.
- **FR-007**: System MUST establish a rigorous model taxonomy separating primary, secondary, and diagnostic baselines:
  - **FR-007.1 (Primary Global 50-Class Baseline)**: Canonical baseline `global_50_class` (REST=0, ACTIVE=1..49) combining E1, E2, E3 windows only after applying frozen Feature 003 contracts (view, alignment, repetition units, split manifests, preprocessing, pure windows, zero leakage).
  - **FR-007.2 (Secondary Exercise-Stratified Baselines)**: Automatically train and report three secondary baselines: `exercise_E1` (REST + 17 gestures = 18 classes), `exercise_E2` (REST + 23 gestures = 24 classes), `exercise_E3` (REST + 9 gestures = 10 classes) using derived local labels from Feature 003 (`get_local_labels()`). Described in reports strictly as `exercise-stratified secondary baselines` or `literature-aligned exercise-level analysis` (not "exact replication of Atzori et al." unless all historical conditions match).
  - **FR-007.3 (Mandatory Confounding Audit exercise_id_from_rest)**: Independent diagnostic audit training an auxiliary classifier to predict `exercise_id \in {E1, E2, E3}` strictly from REST windows under the same split constraints. Performance near chance (33.3%) indicates low session/recording signature in REST; performance substantially above chance indicates exercise/recording confounding which MUST accompany the interpretation of the global 50-class results without excluding recordings.
  - **FR-007.4 (Reporting and Non-Averaging Invariant)**: System MUST report global 50-class, E1 18-class, E2 24-class, and E3 10-class metrics separately. System MUST NOT compute an average accuracy across E1, E2, E3 and present it as substitute for the global 50-class baseline.
  - **FR-007.5 (Scientific Hierarchy)**: (1) `global_50_class` [Primary], (2) `E1/E2/E3 exercise-stratified` [Secondary], (3) `exercise_id_from_rest` [Diagnostic / Confounding Audit]. Secondary and diagnostics do not block the primary baseline unless revealing an objective leakage/integrity violation.
- **FR-008**: System MUST implement the Within-Subject validation protocol using leave-one-training-repetition-out over repetitions `[1, 3, 4, 6]`:
  - Fold 1: Val = 1, Train = `[3, 4, 6]`
  - Fold 2: Val = 3, Train = `[1, 4, 6]`
  - Fold 3: Val = 4, Train = `[1, 3, 6]`
  - Fold 4: Val = 6, Train = `[1, 3, 4]`
  Final model MUST be fitted on all training repetitions `[1, 3, 4, 6]` and evaluated once on test repetitions `[2, 5]`. Random window shuffling across repetitions is strictly prohibited.
- **FR-009**: System MUST execute Cross-Subject evaluation strictly over the 5 frozen folds generated in Feature 003 (24 train, 8 val, 8 test subjects per fold), reporting per-fold and aggregated cross-subject metrics.
- **FR-010**: System MUST report the following evaluation metrics for all benchmark evaluations:
  - Top-1 Accuracy (primary comparability metric)
  - Balanced Accuracy
  - Macro-F1
  - Per-class Precision and Recall
  - Multi-class Confusion Matrix
  - Per-subject breakdown and statistical aggregates (mean, median, standard deviation).
  System MUST NOT claim that Balanced Accuracy or Macro-F1 were part of Atzori's original benchmark without specific bibliographic citation.
- **FR-011**: System MUST define and persist a formal `TrainingJob` contract containing job metadata, configuration hashes, environment versions, timing, execution status, outputs, metrics, and artifact hashes.
- **FR-012**: System MUST enforce the following DSP preprocessing, windowing, and boundary rules:
  - **FR-012.1 (Causal DSP Band-Pass)**: Preprocessing MUST apply a 4th-order causal Butterworth band-pass filter (20–450 Hz) via [`CausalSosFilter`](file:///home/allisson/Projects/agents_emg/semg_dsp/filter.py) with SOS coefficients, frequency response, and golden fixtures owned explicitly by Feature 004 contracts. The 50 Hz notch filter is OMITTED based on empirical spectral audit (mean 50 Hz power ratio 0.84 vs flanking bands).
  - **FR-012.2 (Windowing Invariants)**: Analysis windows MUST have length $W = 200\,\text{ms}$ (400 samples @ 2000.0 Hz) and stride $S = 50\,\text{ms}$ (100 samples, 75% overlap, 20 Hz decision rate). Windows MUST NOT cross `repetition_unit` or partition boundaries.
  - **FR-012.3 (Pure Window Transition Exclusion)**: Windows spanning transitions (REST ↔ ACTIVE, ACTIVE ↔ ACTIVE, or crossing repetition units) MUST be excluded from the steady-state baseline. Only pure windows (100% samples belonging to the same state and repetition) are admitted. Discarded window counts and durations MUST be logged per subject/exercise for audit.
  - **FR-012.4 (Data-Driven ZC/SSC Thresholds)**: Deadzone thresholds MUST be computed as $\epsilon[\text{channel}] = R \times \text{RMS\_rest\_train}[\text{channel}]$, with candidate grid $R \in \{0.5, 1.0, 1.5, 2.0, 3.0\}$. Selection of $R$ MUST occur strictly via training-pool validation (Within-subject: train repetitions `[1, 3, 4, 6]`; Cross-subject: 24 train subjects). A single $R$ is selected per config/fold.
- **FR-013**: System MUST apply custom Z-score feature scaling ($z_j = (x_j - \mu_j) / \sigma_j$) fitted strictly on the training partition:
  - Validation and test partitions MUST be transformed statelessly using training $\mu$ and $\sigma$.
  - If $\sigma_j == 0$, system MUST set scale factor to $1.0$ and flag the feature as `zero_variance`.
  - `sklearn.preprocessing.StandardScaler` is strictly prohibited; scaling arrays (`mean[]`, `scale[]`) MUST be stored in neutral numerical formats.
  - Zero leakage invariant: Validation and test partitions MUST NEVER contribute to rest RMS, ZC/SSC thresholds, scaling statistics, or covariance shrinkage parameters.
- **FR-014**: System MUST NOT include INT8 quantization, embedded runtime deployment, C++, ESP32, FreeRTOS, Software-in-the-Loop (SIL), or broad deep learning architecture search in this feature.
- **FR-015**: System MUST NOT modify raw NinaPro DB2 data files or alter Feature 003 artifacts.

---

### Key Entities

- **`TrainingJob`**: Canonical execution record capturing all lineage hashes, configuration signatures, environment versions, random seeds, execution status, metrics, and artifact references.
- **`FeatureConfig`**: Immutable configuration specifying window length $W$, stride $S$, feature set (MAV, RMS, WL, ZC, SSC), and deadzone thresholds.
- **`ModelConfig`**: Configuration specifying classifier hyperparameters (regularized LDA shrinkage parameter, solver, priors).
- **`PreprocessingConfig`**: Specification of bandpass and notch filter stages applied via `semg_dsp` prior to windowing.
- **`EvaluationReport`**: Structured container holding per-fold, per-subject, and aggregated benchmark metrics across primary (50-class) and secondary (49-class) vocabularies.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of training and evaluation runs consume frozen Feature 003 manifests and views, logging verifiable SHA-256 hashes with zero runtime reinterpretation of raw archives.
- **SC-002**: Feature extraction across 12 channels produces a deterministic 60-dimensional feature matrix per window with exactly 0.0% NaN or Inf values.
- **SC-003**: 100% of Within-Subject evaluations strictly follow leave-one-repetition-out over `[1, 3, 4, 6]` for internal validation and evaluate exactly once on `[2, 5]` with zero temporal leakage.
- **SC-004**: 100% of Cross-Subject evaluations execute on the 5 frozen folds (24/8/8) with zero subject intersection between partitions.
- **SC-005**: All evaluation runs generate structured metrics (Top-1 Accuracy, Balanced Accuracy, Macro-F1, Confusion Matrix) reported separately for 50-class and 49-class vocabularies.
- **SC-006**: Generated `TrainingJob` artifacts are bitwise reproducible when executed with identical seeds and configurations on identical hardware environments.

---

## Assumptions

- NinaPro DB2 data is already validated and accessible via `semg_dataset` without modifying `data/raw/ninapro_db2/`.
- Sampling rate of sEMG is 2000.0 Hz nominal.
- `scikit-learn` is approved through the established governance mechanism as a host-only dependency for baseline model fitting and metrics computation, and will never be imported or linked in embedded/DSP runtimes.
- Regularized LDA serves as the baseline model; no neural networks or hyperparameter optimization sweeps are in scope for Feature 004.
- All evaluation occurs on host Python (3.10+); no embedded or target hardware deployment is performed.
