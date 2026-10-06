# Architecture Decision Record (ADR)

## ADR-004-003: Aprovação de Governança para Dependência Host-Only Condicional `tomli` (Compatibilidade com Python 3.10)

- **ID**: `ADR-004-003`
- **Status**: `APPROVED`
- **Data de Proposição**: 2026-10-06
- **Aprovador Humano**: `Allisson Chervenski <allissonchervenski5@gmail.com>`
- **Data de Aprovação Humana**: `2026-10-06`
- **Commit Git de Aprovação**: `[PENDING HUMAN COMMIT]`
- **Fingerprint Criptográfico (SHA-256 pós-aprovação)**: `[PENDING LOCK]`

---

## 1. Contexto & Fundamentação Constitucional

Conforme estipulado em [`.specify/memory/constitution.md`](file:///home/allisson/Projects/agents_emg/.specify/memory/constitution.md) (Versão ativa: `1.1.0`, ratificada em 2026-09-28, commit Git `ce75be0`), Seção *Technical Constraints & Safe Execution Policies*:
> *"Runtime Environment: Python 3.10+ standard library baseline. Optional development dependencies (pytest, PyYAML, ruff, mypy) are managed via pyproject.toml."*
> *"Scope Invariants: Core components—including ModelRouter, cost policies, StageRegistry, SkillDispatcher, and the TDD engine—MUST NOT be modified unless explicitly required by an authorized architectural initiative. Features MUST NOT introduce new third-party dependencies without prior governance approval."*

O módulo `tomllib` foi introduzido na biblioteca padrão do Python apenas a partir da versão 3.11. Para assegurar estrita conformidade com o baseline constitucional de suporte a Python 3.10+, o portão de governança de dependências da Feature 004 requer um mecanismo determinístico de leitura de arquivos TOML (`pyproject.toml`) em ambientes Python 3.10.

Este ADR governa a introdução formal e condicional da biblioteca `tomli` como fallback exclusivo para ambientes Python 3.10.

---

## 2. Dependência Aprovada & Escopo de Aplicação

- **Dependência Direta**: `tomli`
- **Condição de Ativação (Environment Marker)**: `python_version < "3.11"` (estritamente condicionada; em Python $\ge 3.11$, o carregamento utiliza exclusivamente `tomllib` da biblioteca padrão).
- **Faixa de Versão**: `>=2.0.0, <3.0.0` (exige obrigatoriamente tanto o limite inferior `>=2.0.0` quanto o superior `<3.0.0`).
- **Finalidade Exclusiva**: Fallback de leitura e parsing de `pyproject.toml` no portão de governança automatizado em ambientes host Python 3.10.
- **Isolamento de Runtime**: `tomli` é uma dependência **estritamente host-only e condicional**. É categoricamente proibido importar, referenciar ou vincular `tomli` em:
  - `semg_dsp/` (pipeline de processamento de sinal em tempo real);
  - `semg_dataset/` (contratos canônicos de ingestão e splits);
  - `orchestrator/` (orquestrador de fluxo de trabalho);
  - Módulos científicos de treinamento ou extração de features (`semg_baseline/lda.py`, `semg_baseline/pipeline.py`, etc.);
  - Futuros runtimes embarcados, C++, FreeRTOS ou microcontroladores.

---

## 3. Allowlist Positiva de Caminhos de Import

Imports do módulo `tomli` são autorizados exclusivamente dentro dos seguintes caminhos do repositório:
1. `semg_baseline/governance.py` (estritamente dentro do bloco de fallback `try: import tomllib except ImportError: import tomli as tomllib`);
2. `tests/**` (suítes de testes automatizados do portão de governança).

Qualquer import de `tomli` fora destes caminhos constitui uma violação imediata de governança (`SCOPE_VIOLATION`), bloqueando os portões determinísticos de verificação.

---

## 4. Allowlist de Superfície de API (Símbolos Autorizados)

O uso de `tomli` é restrito a leitura segura de arquivos TOML binários:
- `tomli.load`
- `tomli.loads`
- `tomli.TOMLDecodeError`

---

## 5. Ciclo de Vida de Aprovação em Dois Estágios

Assim como nos ADRs 004-001 e 004-002, a autorização de `tomli` segue o protocolo rigoroso de dois estágios:
- **Estágio A (Assinatura Humana)**: `Status: APPROVED`, preenchimento de `Aprovador Humano` e `Data de Aprovação`. Enquanto commit e fingerprint estiverem pendentes, o status avaliado é `APPROVED_PENDING_LOCK` e a mutação do `pyproject.toml` permanece **bloqueada**.
- **Estágio B (Commit & Lock Criptográfico)**: Após commit da assinatura, preenche-se `Commit Git de Aprovação` e o `Fingerprint Criptográfico` (SHA-256). O status avaliado torna-se `APPROVED` / `PASS`, autorizando a adição condicional da dependência ao `pyproject.toml`.

---

## 6. Estado Atual

- **Status**: `PROPOSED` (pendente de aprovação humana).
- A declaração de `tomli` em `pyproject.toml` permanece **bloqueada** até a conclusão formal do ciclo de governança.
