# Ozdemir et al. sEMG Dataset — Relatório de Auditoria e Integridade

> **Dataset**: Ozdemir et al. sEMG Dataset (Mendeley Data V2, DOI: 10.17632/ckwc76xr2z.2)
> **Data de Auditoria**: `2026-10-27T06:20:19.006699+00:00`
> **Status de Prontidão**: **`READY`**

---

## 1. Visão Geral e Proveniência

| Propriedade | Valor |
|---|---|
| **Instituição** | Izmir Katip Celebi University, Department of Biomedical Engineering |
| **Autor Principal** | Mehmet Akif Ozdemir (`makif.ozdemir@ikcu.edu.tr`) |
| **Repositório Oficial** | [Mendeley Data V2](http://dx.doi.org/10.17632/ckwc76xr2z.2) |
| **Diretório Local** | `data/raw/ozdemir/` |
| **Sistema de Aquisição** | 4-Channel Surface EMG System (2000 Hz, discrete bipolar/differential, mV) |
| **Frequência de Amostragem ($F_s$)** | `2000 Hz` (intervalo de 0.5 ms) |
| **Número de Canais** | `4 canais diferenciais` |
| **Duração por Sujeito** | `640.0 segundos` (1,280,000 amostras) |
| **Total de Participantes** | `40 sujeitos` (S1 a S40) |
| **Espaço em Disco** | `8.917 GiB` (164 arquivos) |

---

## 2. Estrutura de Arquivos e Formatos

O dataset foi descompactado na íntegra com 164 arquivos em 4 variantes completas:
1. **`sEMG-dataset/raw/mat/`**: 40 arquivos MATLAB v7 (.mat) com sinal bruto em milivolts (mV).
2. **`sEMG-dataset/filtered/mat/`**: 40 arquivos MATLAB v7 (.mat) com sinal filtrado em milivolts (mV).
3. **`sEMG-dataset/raw/csv/`**: 40 arquivos CSV tabulares com os 4 canais brutos (1.280.000 linhas).
4. **`sEMG-dataset/filtered/csv/`**: 40 arquivos CSV tabulares com os 4 canais filtrados (1.280.000 linhas).

### Arquivos de Metadados e Protocolo:
- **`EMG_gesture_segmentation.m`**: Script oficial de segmentação e temporalização dos ciclos.
- **`procedure.pptx`**: Apresentação visual de 116 slides usada para guiar a temporização dos participantes.
- **`survey_responses.xlsx`**: Dados demográficos completos dos 40 sujeitos (idade, gênero, peso, altura, braço dominante).
- **`4CH_RAW+4CH_FILTERED_EMG.gtl`**: Arquivo de calibração e configuração do sistema de aquisição.

---

## 3. Protocolo Experimental e Vocabulário de Gestos

O protocolo consiste em **5 ciclos (repetições)** de **10 gestos** (cada tentativa dura 10s: 6s de contração + 4s de repouso):

| ID | Código | Gesto | Descrição Motora |
|:---:|:---:|:---|:---|
| **0** | `X` | **REST** | Posição de relaxamento muscular neutro |
| **1** | `E` | **EXTENSION** | Extensão do punho para cima |
| **2** | `F` | **FLEXION** | Flexão do punho para baixo |
| **3** | `U` | **ULNAR DEVIATION** | Desvio ulnar (movimento medial em direção ao dedo mínimo) |
| **4** | `R` | **RADIAL DEVIATION** | Desvio radial (movimento lateral em direção ao polegar) |
| **5** | `G` | **GRIP** | Fechamento palmar / preensão (hand grip) |
| **6** | `B` | **ABDUCTION** | Abdução (abertura e separação de todos os dedos) |
| **7** | `D` | **ADDUCTION** | Adução (fechamento e aproximação dos dedos) |
| **8** | `S` | **SUPINATION** | Supinação do antebraço (palma voltada para cima) |
| **9** | `P` | **PRONATION** | Pronação do antebraço (palma voltada para baixo) |

### Temporalização dos Ciclos (Repetições):
- **Ciclo 1**: Início no segundo 4 $\rightarrow$ Duração 104s + 30s repouso longo (termina em 138s).
- **Ciclo 2**: Início no segundo 138 $\rightarrow$ Duração 104s + 30s repouso (termina em 272s).
- **Ciclo 3**: Início no segundo 272 $\rightarrow$ Duração 104s + 30s repouso (termina em 406s).
- **Ciclo 4**: Início no segundo 406 $\rightarrow$ Duração 104s + 30s repouso (termina em 540s).
- **Ciclo 5**: Início no segundo 540 $\rightarrow$ Duração 104s (termina em 640s / 1.280.000 amostras).

---

## 4. Auditoria Numérica e Verificação de Integridade

| Métrica de Qualidade | Status | Evidência Numérica |
|---|:---:|---|
| **Sujeitos Disponíveis** | **100%** | 40/40 sujeitos presentes nas 4 representações (.mat e .csv) |
| **Dimensionalidade** | **PASS** | (1.280.000 amostras, 4 canais) idêntico em todos os 40 sujeitos |
| **Valores Não-Finitos** | **PASS** | Zero `NaN`, zero `Inf`, zero valores nulos em todo o dataset |
| **Faixa de Tensão Bruta** | **PASS** | Amplitude global: `[-10.000 mV, 10.000 mV]` |
| **Faixa de Tensão Filtrada** | **PASS** | Amplitude global: `[-8.838 mV, 8.457 mV]` |
| **Consistência CSV vs MAT** | **PASS** | Linhas, colunas e ordenação temporal 100% equivalentes |
| **Status Final de Prontidão** | **`READY`** | Dataset pronto para ingestão e benchmarking no orquestrador |

---

## 5. Resumo Demográfico dos Participantes

Com base em `survey_responses.xlsx`:
- **Total de Voluntários**: 40 indivíduos saudáveis (sem doenças neuromusculares ou marca-passo).
- **Faixa Etária**: 20 a 25 anos.
- **Gênero**: Predomínio de voluntários universitários.
- **Braço Dominante**: Ampla maioria destra com participantes canhotos documentados.
