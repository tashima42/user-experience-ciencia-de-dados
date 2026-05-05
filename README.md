# Previsão de Atrasos em Entregas

Modelo de machine learning para prever atrasos em pedidos de um centro de distribuição, com feature engineering, otimização de hiperparâmetros e sistema de classificação de risco por semáforo.

---

## Sobre o projeto

O objetivo é prever, **antes da entrega**, se um pedido será entregue dentro ou fora do prazo. O modelo gera uma probabilidade de risco para cada pedido, classificada em três níveis:

| Risco | Probabilidade |
|---|---|
| Verde | < 30% |
| Amarelo | 30% a 70% |
| Vermelho | > 70% |

### Dataset

503.532 pedidos de entrega com informações de datas, transportadoras, cidades de destino e centros de distribuição de origem. A variável alvo (`tp_performance_entrega`) indica se a entrega foi feita dentro ou fora do prazo.

> **Desbalanceamento:** 95,5% das entregas são no prazo — o modelo foi treinado com técnicas específicas para lidar com esse desafio.

---

## Resultados

| Métrica | Modelo Base | + Feature Engineering | + Hiperparâmetros |
|---|---|---|---|
| Accuracy | 0.7883 | 0.8013 | **0.9623** |
| F1-score | 0.8778 | 0.8862 | **0.9805** |
| ROC-AUC | 0.7760 | 0.7842 | **0.8137** |
| F1 — Atrasos | 0.21 | 0.22 | **0.43** |
| Atrasos não detectados | 19.577 | 18.251 | **~1.900** |

---

## Metodologia

### 1. Limpeza e tratamento de dados
- Conversão e padronização de colunas de data
- Normalização de nomes de cidades (acentos, capitalização, pontuação)
- Codificação binária do target

### 2. Feature Engineering — 11 novas variáveis

**Temporais**
| Feature | Descrição |
|---|---|
| `hora_despacho` | Hora do dia em que o pedido saiu do CD |
| `turno_despacho` | Turno do despacho (Madrugada / Manhã / Tarde / Noite) |
| `dia_semana_despacho` | Dia da semana do despacho (0=Seg, 6=Dom) |
| `is_fds_despacho` | Flag: pedido despachado no fim de semana |
| `mes_despacho` | Mês do despacho (sazonalidade) |
| `semana_ano` | Semana ISO do ano |
| `is_alta_temporada` | Flag: novembro (Black Friday) ou dezembro (Natal) |

**Razões e margens**
| Feature | Descrição |
|---|---|
| `dias_criacao_pagamento` | Dias entre criação e pagamento do pedido |
| `prazo_apos_despacho` | Dias que a transportadora tem para entregar |
| `ratio_cd_prazo` | Fração do prazo total consumida dentro do CD |
| `margem_entrega` | Margem de dias restante após saída do CD (negativo = atraso quase certo) |

**Target Encoding** (calculado apenas no treino para evitar vazamento de dados)
- `te_transportadora` — taxa histórica de atraso por transportadora
- `te_uf` — taxa histórica de atraso por estado
- `te_cd_origem` — taxa histórica de atraso por centro de distribuição
- `te_praca` — taxa histórica de atraso por tipo de praça

### 3. Modelo

**Algoritmo:** LightGBM (Gradient Boosting)

**Otimização:** Optuna com 50 trials, avaliados por ROC-AUC com early stopping

**Principais hiperparâmetros otimizados:**
- `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`
- `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- `scale_pos_weight` — calibração do peso da classe minoritária (atrasos)

**Threshold de decisão:** ajustado para 0.65 para maximizar o F1 da classe de atrasos

---

## Como executar

### Pré-requisitos
- Python 3.11+
- Anaconda

### 1. Clonar o repositório
```bash
git clone https://github.com/seu-usuario/user-experience-ciencia-de-dados.git
cd user-experience-ciencia-de-dados
```

### 2. Criar o ambiente
```bash
conda create -n DataMiningRD python=3.11
conda activate DataMiningRD
pip install pandas numpy lightgbm scikit-learn optuna joblib matplotlib ydata-profiling
```

### 3. Executar o notebook
Abra o arquivo `trabalho.ipynb` no VS Code, selecione o kernel **DataMiningRD** e clique em **Run All**.

> A etapa de otimização de hiperparâmetros (Optuna) leva aproximadamente 10 minutos.

---

## Estrutura do projeto

```
├── data/
│   ├── amostra.csv                  # Dataset (503k pedidos)
│   └── descricao_dos_dados.txt      # Dicionário de dados
├── trabalho.ipynb                   # Notebook principal
├── model_bundle.joblib              # Modelo treinado exportado
├── feature_importance.png           # Importância das features
└── report.html                      # Relatório de profiling do dataset
```

---

## Tecnologias

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)
![Optuna](https://img.shields.io/badge/Optuna-4.8-purple)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
