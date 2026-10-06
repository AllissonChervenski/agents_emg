# Architecture Decision Record (ADR)

## ADR-004-002: Aprovação de Governança para Dependência Host-Only `jsonschema` (Validação de Contratos da Feature 004)

- **ID**: `ADR-004-002`
- **Status**: `APPROVED`
- **Data de Proposição**: 2026-10-06
- **Aprovador Humano**: `Allisson Chervenski <allissonchervenski5@gmail.com>`
- **Data de Aprovação Humana**: `2026-10-06`
- **Commit Git de Aprovação**: `[PENDING HUMAN COMMIT]`
- **Fingerprint Criptográfico (SHA-256 pós-aprovação)**: `[PENDING LOCK]`

---

## 1. Contexto & Fundamentação Constitucional

Conforme estipulado em [`.specify/memory/constitution.md`](file:///home/allisson/Projects/agents_emg/.specify/memory/constitution.md) (Versão ativa: `1.1.0`, ratificada em 2026-09-28, commit Git `ce75be0`), Seção *Technical Constraints & Safe Execution Policies*:
> *"Scope Invariants: Core components—including ModelRouter, cost policies, StageRegistry, SkillDispatcher, and the TDD engine—MUST NOT be modified unless explicitly required by an authorized architectural initiative. Features MUST NOT introduce new third-party dependencies without prior governance approval."*

A Feature 004 estabelece quatro contratos formais em JSON Schema (especificação Draft 2020-12):
1. `training_job_schema.json`
2. `evaluation_report_schema.json`
3. `neutral_model_manifest_schema.json`
4. `preprocessing_config_schema.json`

A validação rigorosa de conformidade estrutural contra esses schemas durante a criação e execução de `TrainingJob` exige um validador Draft 2020-12 maduro e em conformidade com os padrões da IETF. Este ADR formaliza a governança host-only para a biblioteca `jsonschema`, impedindo dependências implícitas acidentais e assegurando contenção estrita de escopo.

---

## 2. Dependência Aprovada & Escopo de Aplicação

- **Dependência Direta**: `jsonschema`
- **Faixa de Versão**: `>=4.19.0, <5.0.0` (exige obrigatoriamente tanto o limite inferior `>=4.19.0` quanto o superior `<5.0.0`).
- **Finalidade Exclusiva**: Validação offline e em host dos contratos JSON Schema (Draft 2020-12) em `semg_baseline`.
- **Isolamento de Runtime**: `jsonschema` é uma dependência **estritamente host-only**. É categoricamente proibido importar, referenciar ou vincular `jsonschema` em:
  - `semg_dsp/` (pipeline de processamento de sinal em tempo real);
  - `semg_dataset/` (contratos canônicos de ingestão e splits);
  - `orchestrator/` (orquestrador de fluxo de trabalho);
  - Futuros runtimes embarcados, C++, FreeRTOS ou microcontroladores.

---

## 3. Allowlist Positiva de Caminhos de Import

Imports do módulo `jsonschema` são autorizados exclusivamente dentro dos seguintes caminhos do repositório (princípio do menor privilégio):
1. `semg_baseline/runner.py` (validação de metadados de `TrainingJob` contra schema);
2. `tests/**` (suítes de testes automatizados e validação de fixtures de contratos);
3. `scripts/validate_feature_004*.py` (se scripts dedicados de auditoria forense forem criados).

Qualquer import de `jsonschema` fora destes caminhos é uma violação imediata de governança (`SCOPE_VIOLATION`), bloqueando os portões determinísticos de verificação.

---

## 4. Allowlist de Superfície de API (Símbolos Autorizados)

O uso de `jsonschema` é restrito a validação estrutural:
- `jsonschema.Draft202012Validator`
- `jsonschema.validate`
- `jsonschema.exceptions.ValidationError`
- `jsonschema.exceptions.SchemaError`

---

## 5. Ciclo de Vida de Aprovação em Dois Estágios

Assim como no ADR-004-001, a autorização de `jsonschema` segue o protocolo rigoroso de dois estágios:
- **Estágio A (Assinatura Humana)**: `Status: APPROVED`, preenchimento de `Aprovador Humano` e `Data de Aprovação`. Enquanto commit e fingerprint estiverem pendentes, o status avaliado é `APPROVED_PENDING_LOCK` e a mutação do `pyproject.toml` permanece **bloqueada**.
- **Estágio B (Commit & Lock Criptográfico)**: Após commit da assinatura, preenche-se `Commit Git de Aprovação` e o `Fingerprint Criptográfico` (SHA-256). O status avaliado torna-se `APPROVED` / `PASS`, autorizando a adição de `jsonschema>=4.19.0,<5.0.0` ao `pyproject.toml`.

---

## 6. Estado Atual

- **Status**: `PROPOSED` (pendente de aprovação humana).
- A declaração de `jsonschema` em `pyproject.toml` permanece **bloqueada**.
