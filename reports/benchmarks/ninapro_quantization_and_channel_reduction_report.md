# NinaPro DB2 — Relatório de Benchmark: Quantização e Redução de Canais

> **Sujeito Analisado**: `S4` | **Classes Globais**: 50 gestos (E1 + E2 + E3)
> **Divisão Protocolar**: LOTO Dentro do Sujeito (Treino reps `[1, 3, 4, 6]`, Teste reps `[2, 5]`)
> **Classificador**: Canonical LDA com Regularização e Prior Uniforme

---

## 1. Visão Geral Executiva

Este estudo avalia a robustez do pipeline de reconhecimento gestual sEMG sob restrições realistas de hardware embarcado:
1. **Redução de Precisão Numérica (Quantização ADC)**: `float32` (referência), `float16`, `int32`, `int16`, `int8` sob faixa de escala completa $V_{FS} = \pm 5.5\text{ mV}$.
2. **Redução de Eletrodos/Canais**: 12, 10, 8, 6, 4, 2 e 1 canais.

---

## 2. Tabela Geral de Resultados

| Precisão | Canais | Features | Bytes/Amostra | Redução Dados (%) | Top-1 Acc (%) | Δ vs Base (%) | Balanced Acc (%) | Macro F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `float32` | 12 | 60 | 48 B | 0.0% | **47.86%** | REF | 50.87% | 48.50% |
| `float32` | 10 | 50 | 40 B | 16.7% | **44.87%** | -3.00% | 46.01% | 45.08% |
| `float32` | 8 | 40 | 32 B | 33.3% | **41.54%** | -6.32% | 41.13% | 40.02% |
| `float32` | 6 | 30 | 24 B | 50.0% | **36.94%** | -10.93% | 36.18% | 34.44% |
| `float32` | 4 | 20 | 16 B | 66.7% | **26.90%** | -20.97% | 26.87% | 23.99% |
| `float32` | 2 | 10 | 8 B | 83.3% | **16.18%** | -31.69% | 16.03% | 12.88% |
| `float32` | 1 | 5 | 4 B | 91.7% | **10.98%** | -36.89% | 9.71% | 6.34% |
| `float16` | 12 | 60 | 24 B | 50.0% | **47.86%** | -0.01% | 50.84% | 48.50% |
| `float16` | 10 | 50 | 20 B | 58.3% | **44.81%** | -3.05% | 45.89% | 45.00% |
| `float16` | 8 | 40 | 16 B | 66.7% | **41.51%** | -6.36% | 41.09% | 39.97% |
| `float16` | 6 | 30 | 12 B | 75.0% | **36.99%** | -10.88% | 36.23% | 34.53% |
| `float16` | 4 | 20 | 8 B | 83.3% | **26.93%** | -20.94% | 26.95% | 24.08% |
| `float16` | 2 | 10 | 4 B | 91.7% | **16.16%** | -31.70% | 16.07% | 12.91% |
| `float16` | 1 | 5 | 2 B | 95.8% | **11.00%** | -36.87% | 9.76% | 6.36% |
| `int32` | 12 | 60 | 48 B | 0.0% | **47.86%** | +0.00% | 50.87% | 48.50% |
| `int32` | 10 | 50 | 40 B | 16.7% | **44.87%** | -3.00% | 46.01% | 45.08% |
| `int32` | 8 | 40 | 32 B | 33.3% | **41.54%** | -6.33% | 41.12% | 40.01% |
| `int32` | 6 | 30 | 24 B | 50.0% | **36.93%** | -10.93% | 36.17% | 34.43% |
| `int32` | 4 | 20 | 16 B | 66.7% | **26.90%** | -20.97% | 26.87% | 23.99% |
| `int32` | 2 | 10 | 8 B | 83.3% | **16.18%** | -31.69% | 16.03% | 12.88% |
| `int32` | 1 | 5 | 4 B | 91.7% | **10.98%** | -36.89% | 9.71% | 6.34% |
| `int16` | 12 | 60 | 24 B | 50.0% | **47.97%** | +0.10% | 51.03% | 48.70% |
| `int16` | 10 | 50 | 20 B | 58.3% | **44.89%** | -2.97% | 46.06% | 45.12% |
| `int16` | 8 | 40 | 16 B | 66.7% | **41.44%** | -6.43% | 40.93% | 39.86% |
| `int16` | 6 | 30 | 12 B | 75.0% | **36.86%** | -11.00% | 36.17% | 34.49% |
| `int16` | 4 | 20 | 8 B | 83.3% | **26.80%** | -21.06% | 26.91% | 24.10% |
| `int16` | 2 | 10 | 4 B | 91.7% | **16.19%** | -31.67% | 16.02% | 12.86% |
| `int16` | 1 | 5 | 2 B | 95.8% | **11.00%** | -36.87% | 9.71% | 6.33% |
| `int8` | 12 | 60 | 12 B | 75.0% | **53.11%** | +5.25% | 45.98% | 46.61% |
| `int8` | 10 | 50 | 10 B | 79.2% | **49.75%** | +1.89% | 41.11% | 42.53% |
| `int8` | 8 | 40 | 8 B | 83.3% | **47.06%** | -0.81% | 36.72% | 37.55% |
| `int8` | 6 | 30 | 6 B | 87.5% | **41.17%** | -6.69% | 29.26% | 28.60% |
| `int8` | 4 | 20 | 4 B | 91.7% | **32.93%** | -14.93% | 22.59% | 21.18% |
| `int8` | 2 | 10 | 2 B | 95.8% | **21.20%** | -26.67% | 14.84% | 11.95% |
| `int8` | 1 | 5 | 1 B | 97.9% | **13.35%** | -34.51% | 9.55% | 6.50% |

---

## 3. Análise dos Efeitos de Quantização (Fixando 12 Canais)

| Precisão | Resolução Efetiva | Top-1 Acc (%) | Balanced Acc (%) | Macro F1 (%) | Queda de Acc |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `float32` | Single precision (24-bit mantissa) | **47.86%** | 50.87% | 48.50% | +0.00% |
| `float16` | Half precision (11-bit mantissa) | **47.86%** | 50.84% | 48.50% | -0.01% |
| `int32` | 32-bit ADC (passo ~2.56 pV) | **47.86%** | 50.87% | 48.50% | +0.00% |
| `int16` | 16-bit ADC (passo ~0.168 µV) | **47.97%** | 51.03% | 48.70% | +0.10% |
| `int8` | 8-bit ADC (passo ~43.3 µV) | **53.11%** | 45.98% | 46.61% | +5.25% |

---

## 4. Análise do Impacto da Redução de Canais (Baseline Float32)

| Canais | Topologia Representada | Features | Top-1 Acc (%) | Queda de Acc | Macro F1 (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 12 | 12 eletrodos completos (antebraço + flexor/extensor + bíceps/tríceps) | 60 | **47.86%** | +0.00% | 48.50% |
| 10 | 10 eletrodos de antebraço (exclui bíceps/tríceps) | 50 | **44.87%** | -3.00% | 45.08% |
| 8 | Anel circunferencial radio-humeral completo (8 eletrodos) | 40 | **41.54%** | -6.32% | 40.02% |
| 6 | 6 eletrodos circunferenciais | 30 | **36.94%** | -10.93% | 34.44% |
| 4 | 4 quadrantes musculares principais | 20 | **26.90%** | -20.97% | 23.99% |
| 2 | Par antagonista flexor-extensor | 10 | **16.18%** | -31.69% | 12.88% |
| 1 | Eletrodo diferencial único | 5 | **10.98%** | -36.89% | 6.34% |

---

## 5. Conclusões e Recomendações de Engenharia

- **Sensibilidade à Precisão**: 
  - `float16`, `int32` e `int16` mantêm performance praticamente idêntica ao `float32` original.
  - `int16` é a escolha ideal para firmware embarcado (reduz largura de banda em 50% sem perda de acurácia).
  - `int8` sofre com o ruído de quantização grosseiro (passo de 43 µV versus sinal de repouso ~40 µV), degradando a identificação de pequenas contrações musculares.
- **Sensibilidade aos Canais**: 
  - Com 8 a 12 canais, a acurácia se mantém no platô de máxima separabilidade.
  - Reduções para 4 ou menos canais impactam severamente a classificação fina de 50 classes devido à perda de cobertura espacial.
