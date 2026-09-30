# Feature Specification: 002-dsp-streaming-pipeline

**Feature Name**: `002-dsp-streaming-pipeline`  
**Feature Branch**: `feat/002-dsp-streaming-pipeline`  
**Created**: 2026-09-30  
**Status**: Ready for Clarification / Checklist  
**Input**: Especificação e implementação de um pipeline host-only, estritamente causal e streaming para pré-processamento DSP de sinais sEMG sintéticos, com estado explícito, janelamento determinístico e integridade de oráculo numérico independente (Princípio VI da Constituição), preparando portabilidade futura para C++/ESP32 sem depender de hardware real ou datasets externos.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Geração de Sinais Sintéticos Analíticos via `SampleSource` (Priority: P1)

Como engenheiro de sistemas biomédicos ou pesquisador de DSP, desejo que o sistema forneça uma fonte de sinal sintética (`SyntheticSampleSource`) compatível com uma interface conceitual mínima (`SampleSource`), gerando blocos (`chunks`) de amostras multicanal determinísticas em precisão `float32` (degrau, impulso, constante/DC, senoidal, multi-tom, zeros, saturação/clipping), para que o pipeline possa ser testado numericamente em streaming sem depender de hardware ou arquivos de dados reais.

**Why this priority**: É o ponto de partida do pipeline; sem uma fonte de amostras determinística e causal, nenhuma outra etapa de processamento ou teste numérico pode ser exercitada.

**Independent Test**: Instanciar `SyntheticSampleSource` com parâmetros analíticos conhecidos, solicitar leituras sucessivas de chunks com tamanhos variados e verificar conformidade exata das amostras contra as fórmulas matemáticas teóricas (L0), garantindo ausência de descontinuidade temporal e tipo `np.float32`.

**Acceptance Scenarios**:

1. **Given** um gerador analítico configurado com sinal degrau ou impulso unitário em precisão `float32`, **When** múltiplos chunks sucessivos de tamanhos arbitrários forem requisitados via `read_chunk()`, **Then** a sequência concatenada das amostras reproduz exatamente a função analítica L0 sem descontinuidades ou distorções de índice temporal.
2. **Given** um gerador senoidal puro configurado para frequência $f_0$ e taxa de amostragem provisória $F_s$, **When** chunks de tamanhos diferentes (ex.: 1 amostra, 7 amostras, 64 amostras) forem extraídos sequencialmente, **Then** as amostras geradas mantêm continuidade de fase estrita através das fronteiras dos chunks.
3. **Given** qualquer configuração de `SyntheticSampleSource`, **When** o array de dados gerado for inspecionado, **Then** o dtype é estritamente `float32` e nenhum elemento contém valores `NaN` ou `Inf`.

---

### User Story 2 - Filtragem Causal SOS/Biquad com Estado Explícito (Priority: P1)

Como engenheiro de firmware e DSP, desejo filtrar os sinais sEMG em streaming através de uma estrutura biquad/SOS (Second-Order Sections) causal com estado interno explícito mantido entre chunks, para garantir processamento causal em tempo real compatível com futuras implementações em C++ em microcontroladores.

**Why this priority**: A filtragem analógica/digital é a operação central do processamento sEMG. O estado explícito e a causalidade estrita garantem que a saída não dependa de dados futuros nem de processamento em lote completo.

**Independent Test**: Executar a filtragem de um sinal longo em uma única chamada de bloco grande e, separadamente, em uma sequência de pequenos blocos arbitrários; validar que a concatenação das saídas parciais é numericamente equivalente à saída do bloco único dentro da tolerância registrada (invariância por chunks).

**Acceptance Scenarios**:

1. **Given** coeficientes SOS conhecidos e um sinal de entrada sintético, **When** o filtro causal for alimentado com chunks sucessivos preservando seu estado interno (`sos_state`), **Then** a saída produzida é matematicamente causal ($y[n]$ depende apenas de $x[k]$ e $y[k]$ para $k \le n$) e sem dependência de amostras futuras.
2. **Given** uma sequência temporal contínua particionada arbitrariamente em $K$ chunks $\{C_1, C_2, \dots, C_K\}$, **When** o filtro processar a sequência preservando o estado interno entre chamadas sucessivas, **Then** a concatenação das saídas $\text{concat}(F(C_1), F(C_2), \dots, F(C_K))$ é numericamente equivalente a $F(\text{concat}(C_1, C_2, \dots, C_K))$ respeitando a tolerância numérica registrada `TOL-CHUNK-INVARIANCE`.
3. **Given** a execução do filtro em tempo de execução, **When** os tipos de entrada e saída forem verificados, **Then** as operações preservam precisão `float32` sem promoção silenciosa para `float64` nas estruturas de estado.

---

### User Story 3 - Janelamento Deslizante Causal com Estado (Stateful Windowing) (Priority: P1)

Como desenvolvedor de algoritmos de extração de características/classificação sEMG, desejo acumular e fatiar o fluxo contínuo de amostras filtradas em janelas temporais deslizantes determinísticas com comprimento (`window_length`) e passo (`stride`) configuráveis através de um buffer circular com estado explícito, para alimentar etapas posteriores de inferência sem perda ou descontinuidade de dados.

**Why this priority**: A transição entre amostras contínuas e janelas de análise para extração de atributos/inferência requer bufferização causal com retenção de histórico.

**Independent Test**: Enviar fluxos de amostras em parcelas irregulares para o `StatefulWindowBuffer` e verificar que a sequência ordenada de janelas emitida é exatamente idêntica, independentemente de como o mesmo stream contínuo for particionado em chunks. Amostras residuais a cada chunk devem ser retidas no estado e nunca descartadas prematuramente.

**Acceptance Scenarios**:

1. **Given** um buffer com `window_length = W` e `stride = S`, **When** um número de amostras inferior a $W$ é recebido via `process_chunk()`, **Then** nenhuma janela é emitida e todas as amostras permanecem retidas no buffer interno com estado aguardando os próximos chunks.
2. **Given** a chegada da amostra de índice $W + k \cdot S$, **When** o buffer processar o chunk, **Then** a respectiva janela temporal de comprimento $W$ é emitida imediatamente contendo os dados mais recentes em precisão `float32`.
3. **Given** a ingestão contínua de chunks, **When** amostras parciais não completarem uma nova janela, **Then** o buffer NUNCA descarta o residual durante o streaming.
4. **Given** a finalização explícita do stream (`finalize()`), **When** restarem amostras insuficientes para uma janela completa, **Then** o buffer aplica a política configurada (`partial_window_policy`), descartando deterministicamente sob a política padrão `"drop"`.

---

### User Story 4 - Integridade do Oráculo Numérico Independente e Verificação Fim-a-Fim (Priority: P1)

Como validador de conformidade científica e constitucional (Princípio VI), exijo que todos os estágios do pipeline DSP sejam verificados contra vetores de teste de ouro (golden vectors) com proveniência independente (SciPy ou formulações analíticas L0), utilizando uma tabela formal de tolerâncias numéricas registradas, sem que o código sob teste gere seu próprio gabarito de referência.

**Why this priority**: Cumprimento estrito e não-negociável do Princípio VI da Constituição (`Scientific and Numerical Oracle Integrity`). Previne falsos-positivos em testes numéricos onde o teste valida o erro do próprio algoritmo.

**Independent Test**: Executar os testes de oráculo independente comparando a saída das funções de filtragem causal contra o oráculo independente `scipy.signal.sosfilt` computado offline/no teste, verificando se a discrepância máxima satisfaz estritamente os contratos de tolerância registrados em `tolerance_id`.

**Acceptance Scenarios**:

1. **Given** um vetor de teste gerado pelo oráculo independente SciPy para coeficientes SOS e sinal L0, **When** comparado com a saída do motor de streaming sob teste, **Then** o erro relativo e absoluto atende às tolerâncias registradas (`rtol`, `atol`) de `TOL-SOS-FILTER-L1`.
2. **Given** qualquer arquivo de fixture ou golden vector em `tests/fixtures/`, **When** o ciclo de TDD (Green/Refactor) é executado pelo orquestrador, **Then** os arquivos de fixtures permanecem estritamente imutáveis sob verificação de hash SHA-256 (`fixture_files`).
3. **Given** a execução da suíte de testes de regressão, **When** o relatório de verificação numérica for gerado, **Then** cada teste numérico cita formalmente seu `tolerance_id` e a justificativa matemática para a tolerância aceita.

---

## Edge Cases & Error Handling

- **Entrada com dados não-finitos (NaN ou Inf)**: Se amostras de entrada corrompidas contiverem `NaN` ou `Inf`, o estágio DSP de filtragem e janelamento DEVE rejeitar o bloco com erro explícito (`ValueError`) ou emitir sinalização de saturação/erro sem propagar silenciamente estados instáveis.
- **Tamanho de chunk unitário ou variável**: O pipeline streaming DEVE produzir resultados idênticos se alimentado com chunks de tamanho 1 (amostra a amostra), chunks do tamanho do stride ou chunks arbitrários maiores que a janela.
- **Reinicialização de estado (Reset)**: Cada estágio com estado (`CausalSosFilter`, `StatefulWindowBuffer`) DEVE prover método explícito `reset()` que restaura o estado inicial a zeros de forma determinística, comprovado por testes.
- **Discrepância de Canais**: Se um chunk for injetado com número de canais diferente da configuração inicial do pipeline, o sistema DEVE levantar imediatamente `ValueError` descritivo.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 (Abstração `SampleSource`)**: O sistema DEVE definir uma interface conceitual mínima para fontes de amostras contendo ao menos o método `read_chunk(num_samples: int) -> ChunkData`, retornando dados em precisão `float32` acompanhados de metadados temporais (índice inicial da amostra e taxa de amostragem provisória).
- **FR-002 (`SyntheticSampleSource`)**: O sistema DEVE implementar `SyntheticSampleSource` capaz de sintetizar de forma determinística sinais analíticos L0:
  - `zeros`: sinal nulo multicanal;
  - `dc`: nível contínuo constante;
  - `impulse`: impulso unitário de Kronecker no índice configurado;
  - `step`: degrau unitário a partir do índice configurado;
  - `sine`: senoide com amplitude, frequência e fase configuráveis;
  - `multi_tone`: soma linear de componentes senoidais;
  - `saturation`: sinal analítico contendo ceifamento/saturação de amplitude nos limites da faixa de representação.
- **FR-003 (Filtragem Causal SOS com Estado)**: O sistema DEVE implementar o componente de filtragem `CausalSosFilter` que processa chunks de sinais utilizando seções de segunda ordem (SOS / biquads) no formato Transposta Direta II (Direct Form II Transposed) ou equivalente canônico causal, mantendo estado explícito `sos_state` entre invocações.
- **FR-004 (Causalidade Estrita no Runtime)**: O processamento em tempo de execução DEVE ser estritamente causal. Fica ESTRITAMENTE PROIBIDO no runtime:
  - Uso de filtragem bidirecional de fase zero (`scipy.signal.filtfilt`);
  - Amortecimento ou normalização baseada em toda a gravação (`whole-recording normalization`);
  - Janelamento com consulta a amostras futuras (*lookahead*).
- **FR-005 (Coeficientes SOS Versionados como Dados)**: Os coeficientes dos filtros SOS DEVEM ser recebidos como matrizes de dados numéricos pré-calculados e versionados (`shape = (n_sections, 6)`), e NUNCA projetados ou recalculados dinamicamente no laço crítico de streaming.
- **FR-006 (Janelador com Estado `StatefulWindowBuffer`)**: O sistema DEVE implementar um buffer deslizante com estado que armazena amostras recebidas em chunks contínuos e emite matrizes de janelas completas com formato `(window_length, num_channels)` em `float32` a cada intervalo de `stride` amostras.
- **FR-007 (Política de Janela Parcial)**: O `StatefulWindowBuffer` NUNCA descarta amostras residuais durante a ingestão contínua de chunks (`process_chunk`). As amostras parciais permanecem obrigatoriamente retidas no buffer com estado para serem combinadas com chunks subsequentes. No encerramento explícito do stream (`finalize`), a política `partial_window_policy` define o comportamento para amostras remanescentes insuficientes para uma janela completa (`"drop"` descarta deterministicamente; `"pad"` completa com zeros).
- **FR-008 (Tipagem Numérica Estrita `float32`)**: Todas as matrizes de entrada, coeficientes de filtro, buffers de estado e janelas de saída DEVEM manter precisão `numpy.float32`. Promoções implícitas para `float64` durante o processamento em streaming devem ser evitadas. `float32` é a referência numérica do pipeline host por ser um dtype explícito, reproduzível e adequado à futura migração para runtimes embedded com precisão simples.
- **FR-009 (Integridade de Oráculo Independente - Princípio VI)**:
  - Os vetores de teste esperados (golden vectors) DEVEM ser calculados por oráculo independente (SciPy ou equações matemáticas L0) no escopo dos testes ou fixtures geradas independentemente.
  - O código de produção (`orchestrator` ou `semg_dsp`) NUNCA deve ser invocado para gerar os próprios dados de teste esperados.
  - Fixtures de teste armazenadas em `tests/fixtures/` devem ser declaradas no campo `fixture_files` das tarefas para proteção de hash SHA-256 contra adulteração (`TEST_TAMPERING`).
- **FR-010 (Catálogo Formal de Tolerâncias Numéricas)**: Toda validação numérica DEVE associar asserções a um identificador formal de tolerância com metadados explícitos de proveniência, tipo, alvo de comparação, status e justificativa analítica:

| Tolerance ID | Runtime Dtype | Comparison Target | rtol | atol | Status | Rationale |
|---|---|---|---|---|---|---|
| `TOL-ANALYTICAL-L0` | `float32` | Fórmulas analíticas (degrau, DC, impulso, zeros) | `1e-6` | `1e-6` | `PROVISIONAL` | Resolução limite de máquina para mantissa de 24 bits IEEE 754 em precisão simples. |
| `TOL-SOS-FILTER-L1` | `float32` | Oráculo independente SciPy `scipy.signal.sosfilt` | `1e-5` | `1e-5` | `PROVISIONAL` | Compensa pequenas diferenças de acumulação de arredondamento entre o loop explícito DF2T em float32 e a rotina C do SciPy. |
| `TOL-CHUNK-INVARIANCE` | `float32` | Concatenação de saídas streaming vs processamento em bloco | `1e-6` | `1e-6` | `PROVISIONAL` | Garante que o estado interno do filtro retém continuidade exata sem deriva numérica através de fronteiras de chunks. |
| `TOL-WINDOW-ACCUMULATION` | `float32` | Janelamento contínuo em bloco vs janelamento streaming com estado | `0.0` | `0.0` | `PROVISIONAL` | Indexação temporal e cópias de buffers discretos devem ser estritamente bit a bit idênticos. |

- **FR-011 (Marcação de Tarefas `numeric_sensitive`)**: Todas as tarefas de especificação, planejamento e código envolvendo aritmética de filtros, janelamento, buffers circulares e transformações numéricas DEVEM conter `"numeric_sensitive": true` no metadata `harness-task`.
- **FR-012 (Isolamento de Famílias em Tarefas Numéricas)**: Para todas as tarefas com `"numeric_sensitive": true`, o orquestrador garantirá que a família do autor do teste (`test_designer`), do validador do teste (`test_validator`) e do implementador (`coder`) sejam mutuamente independentes (`test_designer` ≠ `test_validator` e `test_validator` ≠ `coder`).

---

### Acceptance Criteria

- **AC-001**: `SyntheticSampleSource` gera deterministicamente sinais analíticos L0 (zeros, DC, impulse, step, sine, multi-tone, saturation) em precisão `np.float32`, mantendo continuidade de fase e amostras idênticas entre chamadas sucessivas de chunks.
- **AC-002**: `CausalSosFilter` executa filtragem causal via Direct Form II Transposed com vetor de estado explícito `(n_sections, 2, num_channels)`, sem lookahead e sem uso de `filtfilt` em tempo de execução.
- **AC-003**: A invariância por chunks é satisfeita: para o filtro, $\text{concat}(F(C_1), \dots, F(C_k)) \approx F(\text{concat}(C_1, \dots, C_k))$ respeitando `TOL-CHUNK-INVARIANCE`; para o janelador, a sequência ordenada de janelas emitidas é exatamente idêntica, independentemente do particionamento dos chunks de entrada.
- **AC-004**: `StatefulWindowBuffer` emite janelas deslizantes de formato `(window_length, num_channels)` em `np.float32` conforme o `stride`, retendo amostras residuais no estado durante o streaming e descartando-as deterministicamente apenas quando `finalize()` for explicitamente invocado sob a política `"drop"`.
- **AC-005**: A ordenação de canais e sincronia temporal multicanal são estritamente preservadas através de todos os estágios do pipeline.
- **AC-006**: Injeção de dados contendo `NaN` ou `Inf`, ou discrepância no número de canais configurados, levanta deterministicamente `ValueError` descritivo.
- **AC-007**: A filtragem causal é comprovada contra o oráculo independente `scipy.signal.sosfilt` satisfazendo os limites do contrato de tolerância `TOL-SOS-FILTER-L1`.
- **AC-008**: Vetores de teste de ouro e arquivos em `tests/fixtures/dsp/` permanecem protegidos por hash SHA-256 durante todo o ciclo TDD (`fixture_files`).
- **AC-009**: Todas as tarefas de implementação e teste que manipulam aritmética de filtros, janelamento ou sinais são marcadas com `numeric_sensitive: true` e executadas sob separação de famílias de provedores (`test_designer` ≠ `test_validator` ≠ `coder`).
- **AC-010**: A implementação opera 100% host-only, sem invocar hardware ESP32, FreeRTOS, ADC/DMA ou introduzir novas bibliotecas em `pyproject.toml`.
- **AC-011**: O método `reset()` em cada componente com estado zera deterministicamente os buffers internos, retornando ao estado inicial idêntico ao pós-instanciação.
- **AC-012**: O ciclo de REFACTOR do orquestrador para tarefas marcadas com `numeric_sensitive: true` opera como safe no-op.

---

### Key Entities

- **SampleSource**: Protocolo abstrato / classe base para fontes de amostras de streaming (`read_chunk(n_samples) -> ChunkData`, `reset()`).
- **ChunkData**: Container imutável (`@dataclass(frozen=True)`) contendo dados multicanal `(num_samples, num_channels)` em `float32`, índice sequencial inicial e taxa provisória.
- **SyntheticSampleSource**: Emissor analítico determinístico de ondas L0 (degrau, impulso, senoidal, DC, etc.) para testes de bancada.
- **CausalSosFilter**: Estágio de filtragem causal DF2T mantendo matrizes de coeficientes `(n_sections, 6)` e estado explícito `(n_sections, 2, n_channels)`.
- **StatefulWindowBuffer**: Acumulador circular com estado mantendo histórico temporal para emissão de matrizes `(window_length, n_channels)`.
- **StreamingPipeline**: Orquestrador de streaming unindo fonte de dados, filtro SOS e janelador em um fluxo contínuo unificado.
- **ToleranceContract**: Contrato formal que vincula cada asserção de teste numérico a um `tolerance_id` (`rtol`, `atol`, justificativa matemática).

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% dos testes unitários e de integração de `semg_dsp` passam com tolerâncias numéricas registradas respeitadas contra os oráculos L0 e L1.
- **SC-002**: 100% dos chunks, estados internos e janelas emitidas possuem `dtype == np.float32`, sem promoção implícita para `float64`.
- **SC-003**: Invariância de chunk comprovada com erro absoluto máximo $\le 10^{-6}$ entre streaming fragmentado (chunks de 1 a 64 amostras) e execução em bloco.
- **SC-004**: Zero chamadas de `filtfilt`, zero lookahead e zero normalização baseada em gravações completas no runtime de produção.
- **SC-005**: 100% de conformidade com o Princípio VI da Constituição: zero golden vectors gerados pelo próprio código sob teste.

---

## Out of Scope

- Seleção ou ingestão de datasets sEMG reais (NinaPro, CapgMyo, etc.).
- Carregamento de dados de arquivos em disco ou gravação de voluntários.
- Definição final de frequências de corte de aplicação protética ou taxa de amostragem de hardware.
- Treinamento de redes neurais, CNNs, TFLite ou quantização INT8.
- Código C++, drivers ESP32, FreeRTOS, interrupções ADC, DMA, timers ou GPIO.
- Lógica de decisão de prótese ou controle de atuadores.

---

## Clarifications & Architectural Decisions

1. **Ordem dos Eixos e Formato de Dados**:
   - *Decisão*: Todas as matrizes de chunk e janelas adotam o formato temporal de primeira ordem: `(num_samples, num_channels)` para chunks contínuos e `(window_length, num_channels)` para janelas completas.
   - *Justificativa*: A indexação temporal no eixo 0 simplifica o buffer circular, operações de fatiamento sequencial e concatenação temporal contígua na memória C-contiguous.

2. **Precisão Numérica do Runtime**:
   - *Decisão*: Adotado `np.float32` como padrão uniforme para entradas, estados e saídas.
   - *Justificativa*: `float32` é a referência numérica do pipeline host por ser um dtype explícito, reproduzível e adequado à futura migração para runtimes embedded com precisão simples. Evita consumo desnecessário de memória de buffer e previne divergências numéricas em ponto flutuante.

3. **Interface do `SampleSource` e Continuidade Temporal**:
   - *Decisão*: Classe base / protocolo abstrato `SampleSource` com método `read_chunk(n_samples: int) -> ChunkData`. `SyntheticSampleSource` mantém um contador de amostra corrente interno que garante continuidade de fase estrita através de chamadas sucessivas a `read_chunk`, independente de tamanhos variáveis de chunks.
   - *Justificativa*: Permite testes rigorosos de invariância de streaming amostra a amostra sem quebras de fase.

4. **Arquitetura da Filtragem SOS**:
   - *Decisão*: Implementação da Direct Form II Transposed (DF2T) para a cascata de seções de segunda ordem, mantendo vetor de estado `sos_state` com shape `(n_sections, 2, n_channels)`.
   - *Justificativa*: A forma DF2T apresenta melhor imunidade a erros de quantização em ponto flutuante e permite avanço amostra a amostra com estado mínimo e causalidade perfeita.

5. **Semântica de Retorno e Retenção Residual do `StatefulWindowBuffer`**:
   - *Decisão*: O método `process_chunk(chunk: np.ndarray) -> list[np.ndarray]` consome o chunk de amostras e retorna uma lista ordenada contendo zero ou mais janelas completas (cada uma de shape `(window_length, num_channels)`). Amostras parciais insuficientes para uma janela completa NUNCA são descartadas pelo término do chunk; elas permanecem retidas no buffer de estado aguardando a chegada dos chunks subsequentes. O descarte de amostras residuais sob a política `"drop"` ocorre estritamente na finalização explícita do stream via `finalize()`.
   - *Justificativa*: Garante causalidade contínua e invariância estrita de janelamento independente do tamanho ou fragmentação dos chunks recebidos.

6. **Tratamento de Refatoração Numérica**:
   - *Decisão*: Em tarefas marcadas com `numeric_sensitive: true`, a fase de refatoração pelo orquestrador operará como *safe no-op*, mantendo a implementação aprovada e testada no GREEN até que um harness de snapshot numérico seja configurado.

---

## Deliberately Open Decisions (`DECISION_REQUIRED`)

Os seguintes itens permanecem deliberadamente em aberto como `DECISION_REQUIRED` e NÃO devem ter valores definitivos inventados nesta feature:

- `DECISION_REQUIRED-01 (Taxa de Amostragem Científica Definitiva)`: A taxa $F_s$ definitiva de amostragem de sEMG depende da especificação do hardware ADC e do dataset de gestos que será selecionado na feature de dataset/modelo. Para os testes e fixtures da Feature 002, utiliza-se o valor provisório `Fs = 1000.0 Hz` marcado estritamente como `FIXTURE_ONLY`.
- `DECISION_REQUIRED-02 (Número e Topologia Real de Canais)`: O número de canais definitivo (ex.: 8 canais de bracelete ou 16 canais de alta densidade) permanece em aberto. Para testes da Feature 002, adota-se `n_channels = 2` ou `n_channels = 4` estritamente como `FIXTURE_ONLY`.
- `DECISION_REQUIRED-03 (Banda de Passagem e Frequências de Corte Definitivas)`: As frequências de corte do filtro passa-faixa (ex.: 20–450 Hz) e filtro notch de rejeição de linha de 60 Hz serão determinadas na feature científica de modelagem de sinal. Para os testes unitários do SOS da Feature 002, usam-se filtros de teste biquad com coeficientes pré-calculados marcados como `PROVISIONAL_COEFFICIENTS`.
- `DECISION_REQUIRED-04 (Dimensões de Janela e Passo de Inferência)`: O comprimento temporal da janela (ex.: 200 ms) e o stride (ex.: 50 ms) dependem da latência aceitável para o atuador da prótese e dos requisitos da CNN. Nos testes da Feature 002, usam-se parâmetros arbitrários inteiros (ex.: `window_length = 64`, `stride = 16`) marcados como `FIXTURE_ONLY`.
- `DECISION_REQUIRED-05 (Adoção de Property-Based Testing com Hypothesis)`: A introdução da biblioteca `hypothesis` para testes baseados em propriedades numéricas permanece como `DECISION_REQUIRED`. Não será adicionada nesta feature para respeitar a política de zero novas dependências sem governança formal.
