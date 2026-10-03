# Architecture Decision Record (ADR)

## ADR-004-001: Aprovação de Governança para Dependência Host-Only `scikit-learn` (Feature 004 Baseline & Evaluation)

- **ID**: `ADR-004-001`
- **Status**: `PROPOSED`
- **Data de Proposição**: 2026-10-03
- **Aprovador Humano**: `[PENDING HUMAN APPROVAL]`
- **Data de Aprovação Humana**: `[PENDING HUMAN APPROVAL]`
- **Commit Git de Aprovação**: `[PENDING HUMAN COMMIT]`
- **Fingerprint Criptográfico (SHA-256 pós-aprovação)**: `[PENDING LOCK]`

---

## 1. Contexto & Fundamentação Constitucional

Conforme estipulado em [`.specify/memory/constitution.md`](file:///home/allisson/Projects/agents_emg/.specify/memory/constitution.md) (Versão ativa: `1.1.0`, ratificada em 2026-09-28, commit Git `ce75be0`), Seção *Technical Constraints & Safe Execution Policies*:
> *"Scope Invariants: Core components—including ModelRouter, cost policies, StageRegistry, SkillDispatcher, and the TDD engine—MUST NOT be modified unless explicitly required by an authorized architectural initiative. Features MUST NOT introduce new third-party dependencies without prior governance approval."*

Este ADR materializa a **aprovação prévia de governança** exigida pela Constituição para a introdução controlada da dependência `scikit-learn`, estabelecendo contenção estrita de escopo, allowlists de imports e superfícies de API, e um oráculo de equivalência determinístico.

---

## 2. Dependência Aprovada & Escopo de Aplicação

- **Dependência Direta**: `scikit-learn`
- **Faixa de Versão**: `>=1.5.0, <2.0.0`
- **Finalidade Exclusiva**: Baseline offline e avaliação científica da Feature 004.
- **Isolamento de Runtime**: `scikit-learn` é uma dependência **estritamente host-only**. É categoricamente proibido importar, referenciar ou vincular `scikit-learn` em:
  - `semg_dsp/` (pipeline de processamento de sinal em tempo real);
  - `semg_dataset/` (contratos canônicos de ingestão e splits);
  - Futuros runtimes embarcados, C++, FreeRTOS ou alvos ESP32.

---

## 3. Allowlist Estrita de Caminhos de Import

Imports do módulo `sklearn` são autorizados exclusivamente dentro dos seguintes caminhos do repositório:
1. `semg_baseline/**` (módulos dedicados ao ajuste do baseline regularized LDA e cálculo de métricas);
2. `tests/test_baseline_*.py` ou `tests/feature_004/**` (suítes de testes unitários e de oráculos da Feature 004).

Qualquer import de `sklearn` fora destes caminhos constituirá uma violação imediata de governança (`SCOPE_VIOLATION`), bloqueando os portões determinísticos de verificação.

---

## 4. Allowlist de Superfície de API (Módulos Autorizados)

O uso de `scikit-learn` no baseline canônico é restrito aos seguintes submódulos essenciais:
- `sklearn.discriminant_analysis` (especificamente `LinearDiscriminantAnalysis` com solver `lsqr` e shrinkage `auto`);
- `sklearn.covariance` (mecanismos analíticos de covariância e encolhimento de Ledoit-Wolf);
- `sklearn.metrics` (cálculo de `confusion_matrix`, `accuracy_score`, `balanced_accuracy_score`, `f1_score`, `precision_score`, `recall_score`).

**APIs e Módulos Proibidos**:
- `sklearn.model_selection` (proibido: geradores de split, `train_test_split`, `KFold`, `GridSearchCV` — todos os splits devem derivar exclusivamente dos manifests congelados da Feature 003);
- `sklearn.pipeline.Pipeline` (proibido no baseline canônico);
- `sklearn.preprocessing.StandardScaler` (proibido: a normalização de features deve utilizar o `FeatureScaler` customizado da Feature 004, ajustado exclusivamente no treino);
- Qualquer função estocástica ou utilitário que randomize partições.

---

## 5. Fronteira Estrita de Exportação Numérica (Zero-Pickle)

Nenhum objeto serializado em Python (`pickle`, `joblib.dump`, instâncias de classes do sklearn) pode constituir formato de intercâmbio de modelo.
- O modelo treinado deve ser exportado exclusivamente como uma representação neutra puramente numérica:
  - `classes`: array `int32` de formato `(K,)`;
  - `coef`: array `float64` de formato `(K, 60)`;
  - `intercept`: array `float64` de formato `(K,)`;
  - `mean`: array `float64` de formato `(60,)`;
  - `scale`: array `float64` de formato `(60,)`;
  - Metadados estruturados e hashes SHA-256 em `model_manifest.json`.
- A Feature 005 implementará inferência via cálculo matricial direto:
  $$\text{scores} = X_{\text{scaled}} @ \text{coef}^T + \text{intercept}$$
  $$\hat{y} = \text{argmax}(\text{scores})$$
  sem importar ou depender de `scikit-learn`.

---

## 6. Oráculo de Equivalência Determinística

A Feature 004 deve incluir um teste de oráculo independente (`test_baseline_export_and_oracle.py`) comprovando que:
$$\text{sklearn.predict}(X) == \text{neutral\_numpy\_predict}(X)$$
com correspondência de classe 100% idêntica e tolerância numérica controlada ($\text{atol} \le 10^{-7}$) para os scores de decisão sobre os fixtures de teste. O `scikit-learn` atua como oráculo de treinamento e referência científica, nunca como acoplamento de runtime.

---

## 7. Portão Automatizado de Governança

Um teste determinístico automatizado (`tests/test_baseline_governance_gate.py`) deve verificar:
1. Toda dependência externa declarada em `pyproject.toml` possui ADR correspondente com status `APPROVED`;
2. A assinatura e aprovação humana estão presentes no ADR;
3. A versão de `scikit-learn` em `pyproject.toml` está pinada em conformidade com o ADR;
4. Todo import de `sklearn` no código-fonte respeita a allowlist de caminhos;
5. Todo import de `sklearn` respeita a allowlist de submódulos;
6. O fingerprint SHA-256 do ADR aprovado permanece íntegro e inalterado.
Qualquer violação produz um erro fatal `FAIL`, bloqueando o pipeline.

---

## 8. Ciclo de Vida e Mutação do `pyproject.toml`

O agente automatizado **não tem autoridade para considerar o ADR aprovado por tê-lo redigido**. A sequência obrigatória é:
1. Criação deste documento com status `PROPOSED`;
2. Revisão e assinatura humana (preenchimento dos metadados de aprovação e commit Git pelo usuário);
3. Execução do portão determinístico de governança;
4. Adição autorizada de `scikit-learn` em `pyproject.toml` sob `[project.optional-dependencies]`;
5. Bloqueio determinístico de versões transitivas no lockfile de ambiente;
6. Nova execução completa dos gates de qualidade (`pytest`, `ruff`, `mypy`, `compileall`).
