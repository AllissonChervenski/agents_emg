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
- **Faixa de Versão**: `>=1.5.0, <2.0.0` (exige obrigatoriamente tanto o limite inferior `>=1.5.0` quanto o superior `<2.0.0`).
- **Finalidade Exclusiva**: Baseline offline e avaliação científica da Feature 004.
- **Isolamento de Runtime**: `scikit-learn` é uma dependência **estritamente host-only**. É categoricamente proibido importar, referenciar ou vincular `scikit-learn` em:
  - `semg_dsp/` (pipeline de processamento de sinal em tempo real);
  - `semg_dataset/` (contratos canônicos de ingestão e splits);
  - `orchestrator/` e scripts de infraestrutura;
  - Futuros runtimes embarcados, C++, FreeRTOS ou alvos ESP32.

---

## 3. Allowlist Positiva Estrita de Caminhos de Import

Imports do módulo `sklearn` são autorizados exclusivamente dentro dos seguintes caminhos do repositório (allowlist positiva):
1. `semg_baseline/**` (módulos dedicados ao ajuste do baseline regularized LDA e cálculo de métricas);
2. `tests/test_baseline_*.py` (testes unitários e de integração do baseline);
3. `tests/feature_004/**` (suítes especializadas da Feature 004).

Qualquer import de `sklearn` fora destes caminhos constituirá uma violação imediata de governança (`SCOPE_VIOLATION`), bloqueando os portões determinísticos de verificação.

---

## 4. Allowlist de Superfície de API (Símbolos Autorizados)

O uso de `scikit-learn` no baseline canônico é restrito a símbolos específicos e documentados:

- `sklearn.discriminant_analysis`:
  - `LinearDiscriminantAnalysis` (com solver `lsqr` e shrinkage `auto`).
- `sklearn.covariance`:
  - `ledoit_wolf`, `LedoitWolf`, `empirical_covariance`, `shrunk_covariance`.
- `sklearn.metrics`:
  - `confusion_matrix`, `accuracy_score`, `balanced_accuracy_score`, `f1_score`, `precision_score`, `recall_score`.

**Regras de Sintaxe e APIs Proibidas**:
- Bare `import sklearn` é estritamente proibido (deve-se importar submódulos e símbolos autorizados).
- Imports com wildcard (`from sklearn... import *`) são estritamente proibidos.
- `sklearn.model_selection` é proibido (geradores de split, `train_test_split`, `KFold`, `GridSearchCV` — todos os splits derivam exclusivamente dos manifests congelados da Feature 003).
- `sklearn.pipeline.Pipeline` é proibido.
- `sklearn.preprocessing.StandardScaler` é proibido (deve-se usar o `FeatureScaler` customizado da Feature 004).
- Qualquer símbolo fora da allowlist acima produz `API_VIOLATION`.

---

## 5. Fronteira Estrita de Exportação Numérica (Zero-Pickle)

Nenhum objeto serializado em Python (`pickle`, `joblib.dump`, instâncias de classes do sklearn) pode constituir formato de intercâmbio de modelo.
- O modelo treinado deve ser exportado exclusivamente como uma representação neutra puramente numérica:
  - Os arrays numéricos binários persistidos em `model_params.npz` são **estritamente cinco**:
    - `classes`: array `int32` de formato `(K,)` (`semantic_role: class_labels`);
    - `coef`: array `float64` de formato `(K, 60)` (`semantic_role: discriminant_coefficients`);
    - `intercept`: array `float64` de formato `(K,)` (`semantic_role: discriminant_intercepts`);
    - `mean`: array `float64` de formato `(60,)` (`semantic_role: scaler_mean`);
    - `scale`: array `float64` de formato `(60,)` (`semantic_role: scaler_scale`).
    Nenhum sexto array é permitido no arquivo binário.
  - Metadados estruturados e hashes SHA-256 em `model_manifest.json` com integridade selada por sidecar `model_manifest.sha256`.
  - `zero_variance_indices`: metadata estruturada obrigatória em `model_manifest.json` (lista ordenada de inteiros não-negativos únicos), registrando features com variância de treino nula ($\sigma_j \le 10^{-12}$) para garantir que a transformação de teste zere estritamente essas colunas sem adulterar os arrays ou necessitar de heurísticas no preditor.
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
3. A versão de `scikit-learn` em `pyproject.toml` cumpre estritamente `>=1.5.0,<2.0.0`;
4. Todo import de `sklearn` no código-fonte respeita a allowlist positiva de caminhos;
5. Todo import de `sklearn` respeita a allowlist exata de símbolos autorizados;
6. O fingerprint SHA-256 do ADR aprovado e trancado permanece íntegro.

---

## 8. Ciclo de Vida de Aprovação em Duas Etapas e Mutação do `pyproject.toml`

Para eliminar autorreferência no hash Git e garantir auditoria estrita, a transição de governança segue um fluxo ordenado de 3 etapas:

```text
ADR PROPOSED (Status: PROPOSED, sklearn bloqueado)
      ↓
Humano decide aprovar
      ↓
[Etapa A] Commit de Aprovação Humana (Commit A):
  - Status = APPROVED
  - Aprovador Humano = <nome/ID do aprovador>
  - Data de Aprovação = <data ISO>
  - Commit Git de Aprovação = [PENDING COMMIT]
  - Fingerprint = [PENDING LOCK]
  (Nesta etapa, o gate reconhece aprovação preliminar mas mantém sklearn bloqueado: APPROVED_PENDING_LOCK)
      ↓
[Etapa B] Commit de Trancamento de Governança (Commit B):
  - Commit Git de Aprovação = <hash Git do Commit A>
  - Fingerprint Criptográfico = compute_adr_fingerprint(...)
  (Nesta etapa, o gate reconhece trancamento completo: status APPROVED, sklearn liberado)
      ↓
[Etapa C] Mutação Autorizada e Atualização do Lockfile:
  - Adição de "scikit-learn>=1.5.0,<2.0.0" sob dependências host/baseline
  - Geração/atualização do lockfile de dependências
  - Registro de dependency_lock_hash
  - Execução completa dos testes e gates (pytest, ruff, mypy, compileall)
      ↓
Desbloqueio formal para início de T014
```
