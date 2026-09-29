# Feature Specification: Comando `provider-summary`

**Feature Branch**: `001-provider-summary`  
**Created**: 2026-09-28  
**Status**: Ready for Planning  
**Input**: Adicionar um novo comando CLI somente de leitura: `python -m orchestrator provider-summary`. O comando deve apresentar um resumo dos providers atualmente conhecidos pelo orquestrador, reutilizando exclusivamente os dados e abstrações já existentes no projeto.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Resumo Textual dos Providers Conhecidos (Priority: P1)

Como operador do orquestrador, desejo executar um comando CLI para visualizar rapidamente a disponibilidade e contagem de modelos conhecidos de cada provider, para validar o estado do orquestrador sem disparar testes demorados ou subprocessos.

**Why this priority**: Funcionalidade principal do comando para inspeção humana rápida e validação de ambiente.

**Independent Test**: Executar `python -m orchestrator provider-summary` no terminal e verificar que o comando encerra com status 0 e exibe em stdout o nome, disponibilidade e contagem de modelos de cada provider conhecido.

**Acceptance Scenarios**:

1. **Given** o ambiente do orquestrador com providers conhecidos registrados, **When** o usuário executa `python -m orchestrator provider-summary`, **Then** o processo finaliza com código de saída 0 e imprime as informações de cada provider conhecido (nome, disponibilidade e contagem de modelos quando disponível).
2. **Given** um provider conhecido que não possua modelos descobertos, **When** o usuário executa `python -m orchestrator provider-summary`, **Then** a saída apresenta a contagem de modelos como 0 ou não disponível sem lançar exceções.

---

### User Story 2 - Saída Estruturada em JSON (Priority: P1)

Como desenvolvedor ou pipeline automatizado, desejo obter o resumo de providers em formato JSON através da flag `--json`, para que ferramentas externas e scripts possam consumir determinística e programaticamente as informações.

**Why this priority**: Interface programática essencial para integração e testes automatizados.

**Independent Test**: Executar `python -m orchestrator provider-summary --json`, capturar stdout e validar que o conteúdo consiste unicamente em JSON válido parseável.

**Acceptance Scenarios**:

1. **Given** o orquestrador configurado, **When** o usuário executa `python -m orchestrator provider-summary --json`, **Then** o processo finaliza com código de saída 0 e emite exclusivamente em stdout um documento JSON válido com lista de providers contendo nome, disponibilidade e quantidade de modelos.
2. **Given** a saída em stdout capturada no modo `--json`, **When** processada por um analisador JSON padrão, **Then** a análise é concluída com sucesso sem presença de texto, cabeçalhos ou logs extras antes ou após o payload.

---

### User Story 3 - Operação Estritamente Somente Leitura e Local (Priority: P1)

Como operador de sistema seguro, exijo que o comando execute estritamente em modo de leitura local, sem disparar CLIs de provedores externos, subprocessos, chamadas LLM ou testes ao vivo.

**Why this priority**: Atendimento mandatório ao Princípio III da Constituição (reuso local não destrutivo e zero subprocessos).

**Independent Test**: Executar `python -m orchestrator provider-summary` (e com `--json`) em ambiente monitorado garantindo que nenhuma chamada de rede, chamada de modelo ou subprocesso externo (`agy`, `codex`, `opencode`) seja disparada.

**Acceptance Scenarios**:

1. **Given** adapters de provedores disponíveis ou mockados para falhar se acionados, **When** `provider-summary` é executado, **Then** o comando completa com sucesso sem acionar nenhuma CLI externa ou subprocesso.
2. **Given** execução normal do comando, **When** inspecionadas as operações realizadas, **Then** zero chamadas a APIs de modelos LLM ou smoke tests em tempo de execução são executados.

---

### User Story 4 - Tratamento Controlado de Falhas Esperadas (Priority: P2)

Como usuário do orquestrador, espero que erros comuns (como argumentos inválidos ou falhas na leitura dos dados locais) produzam mensagens de erro claras e código de saída não-zero, sem expor tracebacks não tratados.

**Why this priority**: Ergonomia CLI e contenção de falhas (Princípio IV da Constituição).

**Independent Test**: Executar `python -m orchestrator provider-summary --flag-invalida` e verificar que o processo retorna código diferente de zero com mensagem explicativa em stderr sem traceback bruto.

**Acceptance Scenarios**:

1. **Given** passagem de opções não suportadas, **When** o comando for disparado, **Then** o processo encerra com código não-zero e imprime mensagem de uso em stderr sem traceback não tratado.

---

### Edge Cases

- **Provider sem modelos descobertos**: Quando uma abstração de provider estiver presente sem catálogo de modelos, o comando deve reportar contagem zero ou indicador correspondente de forma estável.
- **Inconsistência ou indisponibilidade na leitura de dados locais**: Se as abstrações locais do orquestrador encontrarem dados ilegíveis ou incompletos, o erro deve ser tratado graciosamente, reportando mensagem diagnóstica compreensível e código de saída não-zero sem crash bruto.
- **Flags e argumentos inválidos**: Argumentos desconhecidos devem ser rejeitados imediatamente pela camada CLI com código de saída padrão diferente de zero.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O sistema DEVE disponibilizar o comando `python -m orchestrator provider-summary` exibindo um resumo textual dos provedores conhecidos, contendo no mínimo: nome, disponibilidade conhecida e quantidade de modelos descobertos (quando disponível).
- **FR-002**: O sistema DEVE disponibilizar a opção `--json` (`python -m orchestrator provider-summary --json`), emitindo exclusivamente em stdout JSON válido e parseável com campos estáveis representando os provedores (`name`, `available`, `model_count` ou convenções equivalentes já adotadas pelo projeto).
- **FR-003**: O comando `provider-summary` DEVE ser estritamente somente de leitura, NÃO realizando chamadas a modelos LLM, live smoke tests nem execução de subprocessos externos (`agy`, `codex`, `opencode`).
- **FR-004**: A implementação DEVE reutilizar exclusivamente os componentes e abstrações já existentes no orquestrador para descoberta de providers, modelos, configuração e capability registry, sem duplicar lógica de descoberta.
- **FR-005**: Erros esperados DEVEM retornar exit code diferente de zero, produzir mensagem compreensível ao usuário e não exibir traceback bruto em uso normal.
- **FR-006 (Ciclo de Desenvolvimento TDD)**: O processo de implementação da feature DEVE seguir estritamente o ciclo TDD:
  - **Fase RED**: Criação prévia de testes automatizados dedicados em `tests/` que falhem comprovadamente pela ausência da funcionalidade (`EXPECTED_FAILURE`).
  - **Fase GREEN**: Implementação mínima e estrita nos arquivos permitidos para satisfazer os testes criados.
  - **Fase REFACTOR**: Refatoração opcional para qualidade e manutenibilidade sem enfraquecer nem alterar os testes já verdes (`TEST_TAMPERING`).
- **FR-007 (Restrições Arquiteturais e Invariantes de Escopo)**: A implementação da feature:
  - NÃO DEVE alterar `ModelRouter`.
  - NÃO DEVE alterar a política de custos.
  - NÃO DEVE alterar `StageRegistry`.
  - NÃO DEVE alterar `SkillDispatcher`.
  - NÃO DEVE alterar o `TDD engine`.
  - NÃO DEVE alterar o comportamento do comando existente `doctor`.
  - NÃO DEVE adicionar nenhum provider novo.
  - NÃO DEVE introduzir nenhuma dependência externa nova no projeto.
  - NÃO DEVE realizar chamadas LLM na implementação da feature.

### Acceptance Criteria

- **AC-001**: `python -m orchestrator provider-summary` retorna código 0 e apresenta em stdout os providers conhecidos de forma legível.
- **AC-002**: `python -m orchestrator provider-summary --json` produz JSON parseável sem qualquer texto, cabeçalho ou log adicional em stdout.
- **AC-003**: Os valores apresentados são derivados dinamicamente das abstrações existentes no orquestrador, sem utilização de listas estáticas ou duplicadas de providers.
- **AC-004**: Nenhum adapter ou componente executa subprocessos durante a execução do comando `provider-summary`.
- **AC-005**: Testes automatizados demonstram isoladamente que o comando funciona sem executar AGY, Codex, OpenCode ou chamadas LLM.
- **AC-006**: A suíte de testes de regressão existente permanece 100% verde (`pytest -q`).
- **AC-007**: A entrega é estruturada e verificável no ciclo formal RED → GREEN → REFACTOR, com testes dedicados comprovando a falha inicial esperada antes da implementação.
- **AC-008**: O diff da feature preserva intactos os arquivos e comportamentos de `ModelRouter`, política de custos, `StageRegistry`, `SkillDispatcher`, `TDD engine` e `doctor`, sem adicionar dependências ou providers ao sistema.
- **AC-009**: Em caso de falha esperada (como argumentos desconhecidos ou erro/indisponibilidade na leitura de dados locais do orquestrador), o comando encerra com código diferente de zero, exibindo mensagem de erro compreensível em stderr e sem expor traceback bruto.

### Key Entities

- **ProviderSummaryItem**: Entidade de dados representando o resumo de um provedor:
  - `name`: Nome identificador do provedor (ex.: `"agy"`, `"opencode"`, `"codex"`).
  - `available`: Estado booleano de disponibilidade conhecido.
  - `model_count`: Quantidade inteira de modelos descobertos conhecidos.
- **ProviderSummaryReport**: Agrupamento dos resumos de provedores para exibição textual e serialização JSON.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% das invocações nominais do comando `provider-summary` (em modo textual e com `--json`) concluem com sucesso (exit code 0) de forma síncrona e não interativa, sem requisição de entrada ao usuário.
- **SC-002**: 100% da saída capturada em stdout ao usar a flag `--json` é JSON válido e compatível com o schema contratado.
- **SC-003**: 100% dos testes da suíte de regressão passam juntamente com a nova suíte de testes TDD implementada para o comando (ciclo RED → GREEN → REFACTOR).
- **SC-004**: Zero chamadas a subprocessos externos, APIs de modelos ou live smoke tests registradas durante a execução do comando e de seus testes.

## Assumptions & Open Decisions

- O mapeamento concreto entre as abstrações/registros existentes no orquestrador e a montagem do `ProviderSummaryItem` será determinado na fase de planejamento técnico (`plan.md`).
- A formatação exata da tabela ou lista em modo texto pode seguir as convenções visuais vigentes na CLI do orquestrador.
- As decisões pendentes relativas à identificação exata dos módulos de armazenamento local ou cache serão consolidadas na fase de planejamento técnico sem antecipação prescritiva na especificação.
