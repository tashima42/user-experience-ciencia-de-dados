# Precompute Predictions v2 — Documentação

Este documento descreve o pipeline de inferência de risco de atraso implementado em `precompute_predictions_v2.py`, incluindo o contexto do que foi feito, como executar e o que esperar como saída.

---

## Contexto do Projeto

A equipe desenvolveu uma série de notebooks com objetivo de prever atrasos em pedidos logísticos:

| Arquivo | Autor | Descrição |
| :--- | :--- | :--- |
| `trabalho.ipynb` | Pedro | Treinamento inicial com todo o dataset; gera `model_bundle.joblib` original |
| `trabalho2.ipynb` | Matheus | Melhorias de features e análises exploratórias adicionais |
| `trabalho3.ipynb` | Daniel | Versão consolidada: base limpa (parquet), novas features de engenharia temporal, divisão holdout 70/30 e modelo LightGBM otimizado |
| `precompute_predictions.py` | Pedro | Script de inferência que consumia um CSV de amostra |
| **`precompute_predictions_v2.py`** | Daniel | **Script atualizado**, descrito neste documento |

---

## O que o `precompute_predictions_v2.py` faz

O script executa o pipeline completo de **inferência em lote**, produzindo um CSV pronto para a dashboard de semáforo de risco.

### Fluxo de Execução

```
pedidos_logistica_limpo.parquet
         │
         ▼
   Recriar Features de Engenharia
   (turno, ratios, flags temporais)
         │
         ▼
   Divisão Temporal (70/30)
   ┌─────────────┬────────────────────────┐
   │ Train (70%) │  Holdout (30%) → usado │
   │  (ignorado) │  como inferência real  │
   └─────────────┴────────────────────────┘
         │
         ▼
   Filtro de Intervalo de Datas
   (variável de ambiente)
         │
         ▼
   Pré-processamento para o Modelo
   (categóricas nativas, descarta datetime)
         │
         ▼
   model_bundle.joblib → predict_proba()
         │
         ▼
   data/precomputed_predictions_v2.csv
```

---

## Como Executar

### Pré-requisitos

- Ambiente `logistica-eda` ativado (ver [GUIA_UV.md](./GUIA_UV.md) ou `HOW_TO.md`)
- `pedidos_logistica_limpo.parquet` presente na raiz do projeto
- `model_bundle.joblib` presente na raiz do projeto (gerado pelo `trabalho3.ipynb`)

### Execução padrão (intervalo padrão: 01–08/dez/2023)

```bash
conda run -n logistica-eda python precompute_predictions_v2.py
```

### Sobrescrever o intervalo de datas via variável de ambiente

```bash
DATE_RANGE_START=2023-11-01 DATE_RANGE_END=2023-11-30 \
  conda run -n logistica-eda python precompute_predictions_v2.py
```

> [!IMPORTANT]
> O intervalo de datas deve estar **dentro do holdout** (últimos 30% do dataset, ordenado por `dt_criacao`).
> Com o dataset atual isso corresponde a **2023-11-18 → 2023-12-31**.
> Datas anteriores retornarão zero registros.

---

## Arquivos de Entrada

| Arquivo | Descrição |
| :--- | :--- |
| `pedidos_logistica_limpo.parquet` | Dataset limpo gerado por `data_preparation_final.py` (~490 mil linhas) |
| `model_bundle.joblib` | Bundle do modelo LightGBM gerado pelo `trabalho3.ipynb` |

### O que está dentro do `model_bundle.joblib`

```python
{
    "model": LGBMClassifier,            # Modelo treinado
    "selected_features": [...],         # 21 features (3 datas são descartadas em runtime)
    "date_cols": [...],                 # Colunas de data (descartadas antes do fit)
    "categorical_cols": [...],          # Categóricas usadas nativas pelo LightGBM
    "categorical_mappings": {...},      # Herdado do v1 — não usado na v2
    "threshold": 0.5                    # Threshold de classificação binária
}
```

> [!NOTE]
> O modelo internamente foi treinado com **18 features** (as 3 colunas de data estavam em `selected_features` mas foram descartadas antes do `model.fit` pois o LightGBM não aceita `datetime64`). A informação temporal é capturada pelas features de engenharia como `prazo_apos_despacho`, `semana_ano`, `is_alta_temporada`, etc.

---

## Features do Modelo (18 efetivas)

### Categóricas (nativas do LightGBM)
| Feature | Descrição |
| :--- | :--- |
| `cidade_destinatario` | Cidade de destino |
| `uf` | Estado de destino |
| `grp_transportadora` | Grupo da transportadora |
| `tp_praca` | Tipo de praça (Capital, Interior, Reg. Metropolitana) |
| `des_unidade_negocio` | Unidade de negócio (Mono, Multi) |
| `des_cd_origem` | Centro de distribuição de origem |
| `turno_despacho` | Turno do despacho (Madrugada, Manha, Tarde, Noite) |

### Numéricas — operacionais
| Feature | Descrição |
| :--- | :--- |
| `dias_gastos_cd` | Dias entre pagamento e despacho (tempo no CD) |
| `dias_restantes_prazo` | Dias entre pagamento e previsão de entrega |
| `prazo_apos_despacho` | Dias entre despacho e previsão de entrega |
| `ratio_cd_prazo` | Fração do prazo consumida no CD (clip 0–2) |
| `margem_entrega` | `prazo_apos_despacho - dias_gastos_cd` (negativo = já atrasado antes de sair) |
| `dias_criacao_pagamento` | Dias entre criação e pagamento do pedido |

### Numéricas — temporais (engenharia)
| Feature | Descrição |
| :--- | :--- |
| `dia_semana_despacho` | Dia da semana do despacho (0=seg, 6=dom) |
| `mes_despacho` | Mês do despacho |
| `semana_ano` | Semana ISO do ano |
| `is_fds_despacho` | Flag: despacho em fim de semana (1/0) |
| `is_alta_temporada` | Flag: novembro ou dezembro (1/0) |

---

## Arquivo de Saída

**Localização:** `data/precomputed_predictions_v2.csv`

### Colunas do CSV

| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `id` | int | ID interno do pedido |
| `cod_pedido` | str | Código do pedido |
| `uf` | str | Estado destino |
| `grp_transportadora` | str | Grupo transportadora |
| `tp_praca` | str | Tipo de praça |
| `des_unidade_negocio` | str | Unidade de negócio |
| `des_cd_origem` | str | CD de origem |
| `cidade_destinatario` | str | Cidade destino |
| `dt_criacao` | date | Data de criação do pedido |
| `dt_despacho_pedido` | datetime | Data/hora do despacho |
| `dt_previsao_entrega_cliente` | date | Previsão de entrega |
| `tp_performance_entrega` | int | **Gabarito real** (1=no prazo, 0=atrasado) |
| `predicao_probabilidade` | float | Probabilidade de entrega **no prazo** — saída bruta do modelo (0–1) |
| `probabilidade_atraso` | float | Risco de atraso = `1 - predicao_probabilidade` — **usado no semáforo** |
| `predicao_binaria` | int | Predição binária (threshold=0.5) |
| `risco_semaforo` | str | 🟢 Verde / 🟡 Amarelo / 🔴 Vermelho |
| `dias_gastos_cd` | float | Tempo no centro de distribuição |
| `dias_transito` | float | Tempo em trânsito |
| `dias_restantes_prazo` | float | Prazo restante |
| `prazo_apos_despacho` | int | Dias de prazo após o despacho |
| `margem_entrega` | float | Margem líquida antes do atraso |
| `ratio_cd_prazo` | float | Fração do prazo gasta no CD |
| `turno_despacho` | str | Turno do despacho |
| `is_alta_temporada` | int | Flag de alta temporada |
| `is_fds_despacho` | int | Flag de despacho em FDS |

### Lógica do Semáforo

O modelo foi treinado com alvo `tp_performance_entrega` onde **1 = entregue no prazo**.
Assim, `predict_proba[:, 1]` retorna a **probabilidade de ser entregue no prazo** — e não o risco de atraso.
O semáforo é calculado sobre o **complemento** dessa probabilidade:

```
probabilidade_atraso = 1 - predicao_probabilidade
```

| Probabilidade de atraso (`probabilidade_atraso`) | Semáforo |
| :---: | :--- |
| `< 30%` | 🟢 Verde — baixo risco |
| `30% a 70%` | 🟡 Amarelo — risco moderado |
| `> 70%` | 🔴 Vermelho — alto risco |

---

## Diferenças em relação ao `precompute_predictions.py` original

| Aspecto | v1 (original) | **v2 (atual)** |
| :--- | :--- | :--- |
| Fonte de dados | `data/amostra.csv` | `pedidos_logistica_limpo.parquet` |
| Conjunto de inferência | Todos os registros no período | **Holdout (30% final — dados não vistos)** |
| Features de engenharia | Não replicadas | Recriadas completas |
| Predição | Linha a linha | **Lote** (batch) — mais rápido |
| Saída | Features + probabilidade | + identificadores, gabarito, semáforo, contexto |
| Categóricas | `cat.codes` (encoding manual) | **`category` nativo** do LightGBM |

---

## Observações Técnicas

> [!NOTE]
> **Sobre o semáforo:** o modelo prevê a probabilidade de **entrega no prazo** (classe 1). O semáforo usa o **complemento** (`1 - predicao_probabilidade`) como risco de atraso. Ambas as colunas estão disponíveis no CSV: `predicao_probabilidade` (bruta) e `probabilidade_atraso` (usada no semáforo).

> [!TIP]
> **Resultado esperado em dez/2023:** ~85% Verde, ~14% Amarelo, ~2% Vermelho — condizente com o gabarito real do período onde 96.9% dos pedidos foram entregues no prazo. Compare sempre o semáforo com a coluna `tp_performance_entrega` para validar a acurácia.

> [!WARNING]
> **Alta temporada (nov/dez):** a feature `is_alta_temporada` captura o aumento de pressão operacional no período de Black Friday e Natal. Mesmo que a taxa de verde continue alta, os casos Vermelho nesse período merecem atenção redobrada da operação.

> [!TIP]
> Para explorar um mês de menor sazonalidade, tente:
> ```bash
> DATE_RANGE_START=2023-08-01 DATE_RANGE_END=2023-08-31 \
>   conda run -n logistica-eda python precompute_predictions_v2.py
> ```

> [!NOTE]
> **Ponto de atenção para versões futuras:** o `model_bundle.joblib` ainda registra as 3 colunas de data em `selected_features` e `date_cols`, mas o modelo nunca as utilizou (foram descartadas antes do `model.fit`). Para evitar confusão, numa próxima versão do bundle, remova essas colunas de `selected_features` antes de salvar.
