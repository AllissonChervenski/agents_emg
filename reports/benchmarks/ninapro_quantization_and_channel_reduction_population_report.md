# NinaPro DB2 — Relatório Populacional: Quantização e Redução de Canais

> **População Avaliada**: `40 Sujeitos (S1 .. S40)` ($N = 40$) | **Classes Globais**: 50 gestos (E1 + E2 + E3)
> **Divisão Protocolar**: LOTO Dentro do Sujeito (Treino reps `[1, 3, 4, 6]`, Teste reps `[2, 5]`)
> **Classificador**: Canonical LDA com Regularização e Prior Uniforme

---

## 1. Visão Geral da Análise Populacional

Este relatório consolida a média e o desvio padrão populacional ($\mu \pm \sigma$) das métricas de classificação neuromuscular sob quantização numérica de ADC e redução de canais de eletrodos.

---

## 2. Tabela Geral de Resultados Populacionais (Média ± Desvio Padrão)

| Precisão | Canais | Features | Bytes/Amostra | Redução (%) | Top-1 Acc (%) | Δ vs Base 12ch (%) | Balanced Acc (%) | Macro F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `float32` | 12 | 60 | 48 B | 0.0% | **55.77 ± 8.73%** | REF | 57.35 ± 5.88% | 54.97 ± 5.55% |
| `float32` | 10 | 50 | 40 B | 16.7% | **50.92 ± 8.61%** | -4.85 ± 3.45% | 50.78 ± 5.56% | 48.99 ± 5.29% |
| `float32` | 8 | 40 | 32 B | 33.3% | **47.72 ± 8.89%** | -8.05 ± 3.83% | 46.83 ± 5.30% | 44.89 ± 5.05% |
| `float32` | 6 | 30 | 24 B | 50.0% | **40.54 ± 9.57%** | -15.23 ± 5.04% | 39.79 ± 5.46% | 37.70 ± 5.45% |
| `float32` | 4 | 20 | 16 B | 66.7% | **33.99 ± 9.94%** | -21.79 ± 5.94% | 32.02 ± 4.67% | 29.65 ± 4.59% |
| `float32` | 2 | 10 | 8 B | 83.3% | **23.54 ± 10.88%** | -32.23 ± 8.33% | 19.25 ± 3.01% | 16.06 ± 3.05% |
| `float32` | 1 | 5 | 4 B | 91.7% | **15.51 ± 10.48%** | -40.26 ± 8.65% | 11.17 ± 2.16% | 7.88 ± 2.05% |
| `float16` | 12 | 60 | 24 B | 50.0% | **55.78 ± 8.70%** | +0.00 ± 0.09% | 57.34 ± 5.85% | 54.96 ± 5.53% |
| `float16` | 10 | 50 | 20 B | 58.3% | **50.90 ± 8.60%** | -4.87 ± 3.46% | 50.76 ± 5.56% | 48.96 ± 5.29% |
| `float16` | 8 | 40 | 16 B | 66.7% | **47.71 ± 8.92%** | -8.07 ± 3.85% | 46.82 ± 5.31% | 44.88 ± 5.06% |
| `float16` | 6 | 30 | 12 B | 75.0% | **40.55 ± 9.55%** | -15.22 ± 5.02% | 39.77 ± 5.44% | 37.69 ± 5.42% |
| `float16` | 4 | 20 | 8 B | 83.3% | **34.00 ± 9.92%** | -21.77 ± 5.95% | 32.04 ± 4.65% | 29.68 ± 4.58% |
| `float16` | 2 | 10 | 4 B | 91.7% | **23.55 ± 10.88%** | -32.22 ± 8.35% | 19.27 ± 3.02% | 16.07 ± 3.05% |
| `float16` | 1 | 5 | 2 B | 95.8% | **15.52 ± 10.49%** | -40.26 ± 8.67% | 11.17 ± 2.17% | 7.88 ± 2.05% |
| `int32` | 12 | 60 | 48 B | 0.0% | **55.78 ± 8.74%** | +0.01 ± 0.04% | 57.37 ± 5.89% | 54.99 ± 5.56% |
| `int32` | 10 | 50 | 40 B | 16.7% | **50.93 ± 8.61%** | -4.85 ± 3.44% | 50.79 ± 5.56% | 48.99 ± 5.29% |
| `int32` | 8 | 40 | 32 B | 33.3% | **47.73 ± 8.90%** | -8.04 ± 3.82% | 46.84 ± 5.31% | 44.89 ± 5.05% |
| `int32` | 6 | 30 | 24 B | 50.0% | **40.55 ± 9.57%** | -15.23 ± 5.04% | 39.79 ± 5.46% | 37.70 ± 5.45% |
| `int32` | 4 | 20 | 16 B | 66.7% | **34.00 ± 9.94%** | -21.78 ± 5.93% | 32.03 ± 4.67% | 29.66 ± 4.60% |
| `int32` | 2 | 10 | 8 B | 83.3% | **23.55 ± 10.87%** | -32.23 ± 8.32% | 19.26 ± 3.02% | 16.06 ± 3.05% |
| `int32` | 1 | 5 | 4 B | 91.7% | **15.51 ± 10.48%** | -40.26 ± 8.65% | 11.17 ± 2.16% | 7.88 ± 2.05% |
| `int16` | 12 | 60 | 24 B | 50.0% | **55.78 ± 8.73%** | +0.01 ± 0.06% | 57.37 ± 5.87% | 54.98 ± 5.54% |
| `int16` | 10 | 50 | 20 B | 58.3% | **50.93 ± 8.59%** | -4.84 ± 3.43% | 50.79 ± 5.57% | 48.99 ± 5.28% |
| `int16` | 8 | 40 | 16 B | 66.7% | **47.74 ± 8.89%** | -8.04 ± 3.81% | 46.85 ± 5.33% | 44.90 ± 5.08% |
| `int16` | 6 | 30 | 12 B | 75.0% | **40.55 ± 9.56%** | -15.22 ± 5.03% | 39.81 ± 5.47% | 37.71 ± 5.44% |
| `int16` | 4 | 20 | 8 B | 83.3% | **33.98 ± 9.94%** | -21.79 ± 5.94% | 32.01 ± 4.68% | 29.65 ± 4.59% |
| `int16` | 2 | 10 | 4 B | 91.7% | **23.55 ± 10.88%** | -32.22 ± 8.32% | 19.26 ± 3.03% | 16.06 ± 3.06% |
| `int16` | 1 | 5 | 2 B | 95.8% | **15.51 ± 10.48%** | -40.27 ± 8.66% | 11.16 ± 2.16% | 7.87 ± 2.05% |
| `int8` | 12 | 60 | 12 B | 75.0% | **55.01 ± 8.61%** | -0.77 ± 2.73% | 51.88 ± 7.36% | 50.95 ± 6.53% |
| `int8` | 10 | 50 | 10 B | 79.2% | **50.32 ± 8.80%** | -5.45 ± 5.39% | 46.29 ± 6.63% | 45.47 ± 6.02% |
| `int8` | 8 | 40 | 8 B | 83.3% | **46.73 ± 9.90%** | -9.05 ± 5.79% | 42.04 ± 6.37% | 40.97 ± 5.93% |
| `int8` | 6 | 30 | 6 B | 87.5% | **39.29 ± 10.80%** | -16.48 ± 7.83% | 34.35 ± 6.35% | 32.94 ± 6.13% |
| `int8` | 4 | 20 | 4 B | 91.7% | **34.81 ± 10.83%** | -20.97 ± 9.48% | 29.13 ± 5.02% | 27.23 ± 4.80% |
| `int8` | 2 | 10 | 2 B | 95.8% | **22.09 ± 11.60%** | -33.68 ± 9.89% | 17.74 ± 2.89% | 14.56 ± 2.97% |
| `int8` | 1 | 5 | 1 B | 97.9% | **15.70 ± 12.03%** | -40.07 ± 10.73% | 10.42 ± 1.88% | 7.31 ± 1.94% |

---

## 3. Impacto Populacional da Quantização (Fixando 12 Canais)

| Precisão | Resolução Efetiva | Top-1 Acc (Média ± DP) | Balanced Acc (Média ± DP) | Macro F1 (Média ± DP) | Δ vs Base 12ch |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `float32` | Single precision (24-bit mantissa) | **55.77 ± 8.73%** | 57.35 ± 5.88% | 54.97 ± 5.55% | REF |
| `float16` | Half precision (11-bit mantissa) | **55.78 ± 8.70%** | 57.34 ± 5.85% | 54.96 ± 5.53% | +0.00 ± 0.09% |
| `int32` | 32-bit ADC (passo ~2.56 pV) | **55.78 ± 8.74%** | 57.37 ± 5.89% | 54.99 ± 5.56% | +0.01 ± 0.04% |
| `int16` | 16-bit ADC (passo ~0.168 µV) | **55.78 ± 8.73%** | 57.37 ± 5.87% | 54.98 ± 5.54% | +0.01 ± 0.06% |
| `int8` | 8-bit ADC (passo ~43.3 µV) | **55.01 ± 8.61%** | 51.88 ± 7.36% | 50.95 ± 6.53% | -0.77 ± 2.73% |

---

## 4. Impacto Populacional da Redução de Canais (Baseline Float32)

| Canais | Topologia Representada | Features | Top-1 Acc (Média ± DP) | Balanced Acc (Média ± DP) | Macro F1 (Média ± DP) | Δ vs Base 12ch |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 12 | 12 eletrodos completos (antebraço + flexor/extensor + bíceps/tríceps) | 60 | **55.77 ± 8.73%** | 57.35 ± 5.88% | 54.97 ± 5.55% | REF |
| 10 | 10 eletrodos de antebraço (exclui bíceps/tríceps) | 50 | **50.92 ± 8.61%** | 50.78 ± 5.56% | 48.99 ± 5.29% | -4.85 ± 3.45% |
| 8 | Anel circunferencial radio-humeral completo (8 eletrodos) | 40 | **47.72 ± 8.89%** | 46.83 ± 5.30% | 44.89 ± 5.05% | -8.05 ± 3.83% |
| 6 | 6 eletrodos circunferenciais | 30 | **40.54 ± 9.57%** | 39.79 ± 5.46% | 37.70 ± 5.45% | -15.23 ± 5.04% |
| 4 | 4 quadrantes musculares principais | 20 | **33.99 ± 9.94%** | 32.02 ± 4.67% | 29.65 ± 4.59% | -21.79 ± 5.94% |
| 2 | Par antagonista flexor-extensor | 10 | **23.54 ± 10.88%** | 19.25 ± 3.01% | 16.06 ± 3.05% | -32.23 ± 8.33% |
| 1 | Eletrodo diferencial único | 5 | **15.51 ± 10.48%** | 11.17 ± 2.16% | 7.88 ± 2.05% | -40.26 ± 8.65% |

---

## 5. Fronteira de Eficiência Populacional (Pareto Optimal)

| Taxa / Amostra | Economia Dados | Configuração | Top-1 Acc Populacional | Balanced Acc Populacional | Macro F1 Populacional | Recomendação |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **1 B** | **97.9%** | `int8` @ 1 ch | 15.70 ± 12.03% | 10.42 ± 1.88% | 7.31 ± 1.94% | Candidato Pareto |
| **2 B** | **95.8%** | `int8` @ 2 ch | 22.09 ± 11.60% | 17.74 ± 2.89% | 14.56 ± 2.97% | Candidato Pareto |
| **4 B** | **91.7%** | `int8` @ 4 ch | 34.81 ± 10.83% | 29.13 ± 5.02% | 27.23 ± 4.80% | Candidato Pareto |
| **6 B** | **87.5%** | `int8` @ 6 ch | 39.29 ± 10.80% | 34.35 ± 6.35% | 32.94 ± 6.13% | Candidato Pareto |
| **8 B** | **83.3%** | `int8` @ 8 ch | 46.73 ± 9.90% | 42.04 ± 6.37% | 40.97 ± 5.93% | Candidato Pareto |
| **10 B** | **79.2%** | `int8` @ 10 ch | 50.32 ± 8.80% | 46.29 ± 6.63% | 45.47 ± 6.02% | Candidato Pareto |
| **12 B** | **75.0%** | `int8` @ 12 ch | 55.01 ± 8.61% | 51.88 ± 7.36% | 50.95 ± 6.53% | **Melhor Relação BLE/IoT (-75% banda)** |
| **24 B** | **50.0%** | `int16` @ 12 ch | 55.78 ± 8.73% | 57.37 ± 5.87% | 54.98 ± 5.54% | **Melhor Escolha Médica/Clínica (-50% banda)** |
| **48 B** | **0.0%** | `int32` @ 12 ch | 55.78 ± 8.74% | 57.37 ± 5.89% | 54.99 ± 5.56% | Candidato Pareto |

---

## 6. Conclusões e Recomendações Estatísticas

- **Validação Populacional ($N = 40$)**: Confirma se a robustez do `int16` e a sensibilidade de canais se mantêm universalmente em diferentes anatomias e graus de condutância de pele.
- **Invariância de Precisão**: `int16` e `float16` comportam-se de forma equivalente ao `float32` através de toda a população, validando o padrão int16 como referência padrão para microcontroladores ARM Cortex-M/DSP.
- **Comportamento do `int8`**: Em 12 canais, o `int8` oferece excelente compressão (-75%) com perda populacional contida. Em poucos canais, o achatamento do ruído favorece a classe repouso, exigindo cautela no limiar de contração.
