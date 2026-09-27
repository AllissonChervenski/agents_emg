# Feature: Assignments CLI

Adicionar o comando:

python -m orchestrator assignments

## Requisitos

FR-001
O comando deve listar todos os roles do workflow e mostrar:
- role
- provider
- model
- tier
- selection_mode

FR-002
Adicionar:

python -m orchestrator assignments --json

A saída deve ser JSON válido contendo os mesmos assignments.

FR-003
O comando deve reutilizar o ModelRouter existente.
Não duplicar lógica de roteamento.

FR-004
O comando não deve executar nenhuma chamada LLM.

FR-005
Falhas esperadas devem produzir exit code != 0 e mensagem clara,
sem traceback cru.

## Critérios de aceitação

AC-001
`python -m orchestrator assignments` lista os roles configurados.

AC-002
`python -m orchestrator assignments --json` produz JSON parseável.

AC-003
O assignment retornado deve ser consistente com o ModelRouter.

AC-004
Testes comprovam que nenhuma CLI/provider é executado pelo comando.

AC-005
Toda a suíte existente continua passando.

## Fora de escopo

- alterar o ModelRouter;
- adaptive routing;
- novos providers;
- dashboard;
- edição manual dos assignments.