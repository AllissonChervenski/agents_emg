# Ozdemir et al. sEMG — Relatório Populacional: Quantização e Redução de Canais

> **População Avaliada**: `40 Sujeitos (S1 .. S40)` ($N = 40$) | **Classes Globais**: 10 gestos
> **Divisão Protocolar**: LOTO Dentro do Sujeito (Treino reps `[1, 3, 4]`, Teste reps `[2, 5]`)
> **Classificador**: Canonical LDA com Regularização e Prior Uniforme

---

## 1. Visão Geral da Análise Populacional

Este relatório consolida a média e o desvio padrão populacional ($\mu \pm \sigma$) das métricas de classificação neuromuscular sob quantização numérica de ADC e redução de canais de eletrodos no dataset Ozdemir et al. (4 canais, 2000 Hz).

---

## 2. Tabela Geral de Resultados Populacionais (Média ± Desvio Padrão)

| Precisão | Canais | Features | Bytes/Amostra | Redução (%) | Top-1 Acc (%) | Δ vs Base 4ch (%) | Balanced Acc (%) | Macro F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `float32` | 4 | 20 | 16 B | 0.0% | **77.50 ± 5.27%** | REF | 77.50 ± 5.27% | 77.67 ± 5.31% |
| `float32` | 3 | 15 | 12 B | 25.0% | **71.59 ± 6.15%** | -5.91 ± 4.30% | 71.59 ± 6.15% | 71.60 ± 6.21% |
| `float32` | 2 | 10 | 8 B | 50.0% | **59.87 ± 6.99%** | -17.62 ± 6.26% | 59.87 ± 6.99% | 59.38 ± 7.10% |
| `float32` | 1 | 5 | 4 B | 75.0% | **45.50 ± 6.66%** | -32.00 ± 7.02% | 45.50 ± 6.66% | 44.42 ± 6.93% |
| `float16` | 4 | 20 | 8 B | 50.0% | **77.51 ± 5.30%** | +0.01 ± 0.11% | 77.51 ± 5.30% | 77.68 ± 5.33% |
| `float16` | 3 | 15 | 6 B | 62.5% | **71.61 ± 6.13%** | -5.89 ± 4.29% | 71.61 ± 6.13% | 71.62 ± 6.20% |
| `float16` | 2 | 10 | 4 B | 75.0% | **59.86 ± 6.96%** | -17.64 ± 6.24% | 59.86 ± 6.96% | 59.36 ± 7.07% |
| `float16` | 1 | 5 | 2 B | 87.5% | **45.53 ± 6.67%** | -31.97 ± 7.05% | 45.53 ± 6.67% | 44.46 ± 6.95% |
| `int32` | 4 | 20 | 16 B | 0.0% | **77.50 ± 5.27%** | +0.00 ± 0.01% | 77.50 ± 5.27% | 77.67 ± 5.31% |
| `int32` | 3 | 15 | 12 B | 25.0% | **71.59 ± 6.15%** | -5.91 ± 4.30% | 71.59 ± 6.15% | 71.60 ± 6.21% |
| `int32` | 2 | 10 | 8 B | 50.0% | **59.87 ± 6.99%** | -17.62 ± 6.26% | 59.87 ± 6.99% | 59.38 ± 7.10% |
| `int32` | 1 | 5 | 4 B | 75.0% | **45.50 ± 6.66%** | -32.00 ± 7.02% | 45.50 ± 6.66% | 44.42 ± 6.93% |
| `int16` | 4 | 20 | 8 B | 50.0% | **77.51 ± 5.09%** | +0.01 ± 0.48% | 77.51 ± 5.09% | 77.68 ± 5.12% |
| `int16` | 3 | 15 | 6 B | 62.5% | **71.52 ± 6.05%** | -5.98 ± 4.13% | 71.52 ± 6.05% | 71.53 ± 6.11% |
| `int16` | 2 | 10 | 4 B | 75.0% | **59.78 ± 7.02%** | -17.71 ± 6.30% | 59.78 ± 7.02% | 59.28 ± 7.11% |
| `int16` | 1 | 5 | 2 B | 87.5% | **45.45 ± 6.62%** | -32.04 ± 7.00% | 45.45 ± 6.62% | 44.41 ± 6.86% |
| `int8` | 4 | 20 | 4 B | 75.0% | **70.31 ± 6.66%** | -7.19 ± 4.57% | 70.31 ± 6.66% | 70.42 ± 6.96% |
| `int8` | 3 | 15 | 3 B | 81.2% | **64.73 ± 7.22%** | -12.76 ± 5.65% | 64.73 ± 7.22% | 64.72 ± 7.46% |
| `int8` | 2 | 10 | 2 B | 87.5% | **57.09 ± 6.63%** | -20.41 ± 5.72% | 57.09 ± 6.63% | 56.76 ± 6.76% |
| `int8` | 1 | 5 | 1 B | 93.8% | **42.82 ± 6.19%** | -34.67 ± 6.67% | 42.82 ± 6.19% | 41.54 ± 6.48% |

---

## 3. Impacto Populacional da Quantização (Fixando 4 Canais)

| Precisão | Resolução Efetiva | Top-1 Acc (Média ± DP) | Balanced Acc (Média ± DP) | Macro F1 (Média ± DP) | Δ vs Base 4ch |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `float32` | Single precision (24-bit mantissa) | **77.50 ± 5.27%** | 77.50 ± 5.27% | 77.67 ± 5.31% | REF |
| `float16` | Half precision (11-bit mantissa) | **77.51 ± 5.30%** | 77.51 ± 5.30% | 77.68 ± 5.33% | +0.01 ± 0.11% |
| `int32` | 32-bit ADC (passo ~4.65 pV em 10 mV) | **77.50 ± 5.27%** | 77.50 ± 5.27% | 77.67 ± 5.31% | +0.00 ± 0.01% |
| `int16` | 16-bit ADC (passo ~0.305 µV em 10 mV) | **77.51 ± 5.09%** | 77.51 ± 5.09% | 77.68 ± 5.12% | +0.01 ± 0.48% |
| `int8` | 8-bit ADC (passo ~78.7 µV em 10 mV) | **70.31 ± 6.66%** | 70.31 ± 6.66% | 70.42 ± 6.96% | -7.19 ± 4.57% |

---

## 4. Impacto Populacional da Redução de Canais (Baseline Float32)

| Canais | Features | Top-1 Acc (Média ± DP) | Balanced Acc (Média ± DP) | Macro F1 (Média ± DP) | Δ vs Base 4ch |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 4 ch | 20 | **77.50 ± 5.27%** | 77.50 ± 5.27% | 77.67 ± 5.31% | REF |
| 3 ch | 15 | **71.59 ± 6.15%** | 71.59 ± 6.15% | 71.60 ± 6.21% | -5.91 ± 4.30% |
| 2 ch | 10 | **59.87 ± 6.99%** | 59.87 ± 6.99% | 59.38 ± 7.10% | -17.62 ± 6.26% |
| 1 ch | 5 | **45.50 ± 6.66%** | 45.50 ± 6.66% | 44.42 ± 6.93% | -32.00 ± 7.02% |

---

## 5. Fronteira de Eficiência Populacional (Pareto Optimal)

| Taxa / Amostra | Economia Dados | Configuração | Top-1 Acc Populacional | Balanced Acc Populacional | Macro F1 Populacional | Recomendação |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **1 B** | **93.8%** | `int8` @ 1 ch | 42.82 ± 6.19% | 42.82 ± 6.19% | 41.54 ± 6.48% | Candidato Pareto |
| **2 B** | **87.5%** | `int8` @ 2 ch | 57.09 ± 6.63% | 57.09 ± 6.63% | 56.76 ± 6.76% | Candidato Pareto |
| **3 B** | **81.2%** | `int8` @ 3 ch | 64.73 ± 7.22% | 64.73 ± 7.22% | 64.72 ± 7.46% | Candidato Pareto |
| **4 B** | **75.0%** | `int8` @ 4 ch | 70.31 ± 6.66% | 70.31 ± 6.66% | 70.42 ± 6.96% | **Melhor Relação BLE/IoT (-75% banda)** |
| **6 B** | **62.5%** | `float16` @ 3 ch | 71.61 ± 6.13% | 71.61 ± 6.13% | 71.62 ± 6.20% | Candidato Pareto |
| **8 B** | **50.0%** | `int16` @ 4 ch | 77.51 ± 5.09% | 77.51 ± 5.09% | 77.68 ± 5.12% | **Melhor Escolha Prótese Embarcada (-50% banda)** |

---

## 6. Conclusões e Recomendações Estatísticas

- **Validação Populacional ($N = 40$)**: Avaliação completa de 10 gestos com LOTO entre repetições nos 40 sujeitos.
- **Invariância de Precisão**: Confirmação se `int16` preserva a acurácia do `float32` também em sensores de 4 canais.
- **Sensibilidade de Canais**: Quantificação do impacto ao reduzir de 4 para 3, 2 e 1 eletrodo(s).
