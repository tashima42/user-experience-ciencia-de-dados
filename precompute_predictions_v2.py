"""
precompute_predictions_v2.py
----------------------------
Gera previsões de risco de atraso usando:
  - Base de dados limpa: pedidos_logistica_limpo.parquet  (saída do data_preparation_final.py)
  - Modelo treinado:     model_bundle.joblib               (saída do trabalho3.ipynb)

Diferenças em relação ao precompute_predictions.py original:
  - Usa parquet limpo no lugar do CSV de amostra
  - Recria as features de engenharia temporal (turno, semana, flags) que o novo modelo exige
  - Utiliza o holdout (últimos 30% por dt_criacao) como conjunto de inferência —
    dados que o modelo nunca viu durante o treinamento
  - O LightGBM agora recebe colunas do tipo `category` nativas (não mais cat.codes)
  - O CSV de saída inclui identificadores (id, cod_pedido), gabarito real
    (tp_performance_entrega), probabilidade e semáforo — pronto para a dashboard

Como executar:
    python precompute_predictions_v2.py

    Opcional — sobrescrever intervalo de datas e arquivo de saída:
    DATE_RANGE_START=2023-11-01 DATE_RANGE_END=2023-11-30 python precompute_predictions_v2.py
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parquet limpo gerado por data_preparation_final.py
PARQUET_PATH = os.path.join(BASE_DIR, "pedidos_logistica_limpo.parquet")

# Bundle gerado pelo trabalho3.ipynb
MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, "model_bundle.joblib")

# Saída
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "precomputed_predictions_v2.csv")

# Intervalo de datas para filtro (pode ser sobrescrito via variável de ambiente)
DATE_RANGE_START = os.environ.get("DATE_RANGE_START", "2023-12-01")
DATE_RANGE_END   = os.environ.get("DATE_RANGE_END",   "2023-12-08")

# Fração usada no treino — o holdout começa após esse ponto
TRAIN_SPLIT_RATIO = 0.70

# ---------------------------------------------------------------------------
# Feature engineering — espelha exatamente o que o trabalho3.ipynb faz
# Deve ser chamado APÓS load_and_clean_data (que já cria dias_gastos_cd, etc.)
# ---------------------------------------------------------------------------
def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recria as features de engenharia do trabalho3.ipynb.
    A função é idempotente: se uma coluna já existir, ela é recalculada.
    """
    df = df.copy()

    # Garantir datetime
    df["dt_despacho_pedido"] = pd.to_datetime(df["dt_despacho_pedido"], errors="coerce")

    # --- Temporais do despacho ---
    df["hora_despacho"]        = df["dt_despacho_pedido"].dt.hour
    df["dia_semana_despacho"]  = df["dt_despacho_pedido"].dt.dayofweek
    df["mes_despacho"]         = df["dt_despacho_pedido"].dt.month
    df["semana_ano"]           = df["dt_despacho_pedido"].dt.isocalendar().week.astype("float32")

    # Flag final de semana (Sab=5, Dom=6)
    df["is_fds_despacho"]  = (df["dia_semana_despacho"] >= 5).astype("int8")

    # Alta temporada: novembro (Black Friday) e dezembro (Natal)
    df["is_alta_temporada"] = df["mes_despacho"].isin([11, 12]).astype("int8")

    # Turno do despacho
    df["turno_despacho"] = pd.cut(
        df["hora_despacho"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Madrugada", "Manha", "Tarde", "Noite"],
    ).astype("category")

    # --- Ratios derivados ---
    df["dias_criacao_pagamento"] = (
        df["dt_pagamento_pedido"] - df["dt_criacao"]
    ).dt.days

    df["prazo_apos_despacho"] = (
        df["dt_previsao_entrega_cliente"] - df["dt_despacho_pedido"]
    ).dt.days

    # Fração do prazo já consumida dentro do CD (clip [0,2] para evitar outliers)
    df["ratio_cd_prazo"] = (
        df["dias_gastos_cd"] / df["dias_restantes_prazo"].replace(0, np.nan)
    ).clip(0, 2)

    # Margem líquida: positivo = confortável, negativo = já estourou antes do despacho
    df["margem_entrega"] = df["prazo_apos_despacho"] - df["dias_gastos_cd"]

    return df


# ---------------------------------------------------------------------------
# Pré-processamento para inferência — espelha o que o modelo foi treinado
# ---------------------------------------------------------------------------
def preprocess_for_model(
    df: pd.DataFrame,
    selected_features: list[str],
    date_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """
    Prepara o DataFrame para predição espelhando exatamente o que o trabalho3.ipynb faz:
      - As colunas de data (dt_previsao_entrega_cliente, dt_criacao, dt_pagamento_pedido)
        estavam em selected_features mas foram DESCARTADAS antes do model.fit porque
        o LightGBM não aceita datetime — o modelo foi treinado SEM elas.
      - As features de engenharia temporal (prazo_apos_despacho, dias_criacao_pagamento,
        semana_ano, etc.) capturam a informação das datas de forma numérica.
      - Categóricas são mantidas como `category` nativo do LightGBM.
    """
    # Selecionar apenas as colunas necessárias (criar com NaN se ausente)
    for col in selected_features:
        if col not in df.columns:
            df[col] = np.nan

    X = df[selected_features].copy()

    # Categóricas → tipo category nativo (LightGBM >= 3.x)
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # Remover colunas de tipo object/datetime — O MODELO FOI TREINADO SEM ELAS.
    # O notebook descartou as date_cols antes do fit pois eram datetime64 não suportados.
    # As features de engenharia temporal já capturam a informação relevante.
    unsupported = X.select_dtypes(include=["object", "datetime64[ns]", "datetimetz"]).columns
    if len(unsupported) > 0:
        print(f"  Descartando colunas não suportadas (mesma lógica do treino): {list(unsupported)}")
        X = X.drop(columns=unsupported)

    return X


# ---------------------------------------------------------------------------
# Semáforo
# ---------------------------------------------------------------------------
def classify_semaforo(prob: float) -> str:
    if prob < 0.30:
        return "🟢 Verde"
    elif prob < 0.70:
        return "🟡 Amarelo"
    return "🔴 Vermelho"


# ---------------------------------------------------------------------------
# Rotina principal
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. Carregar o bundle do modelo
    print(f"Carregando modelo de {MODEL_BUNDLE_PATH} ...")
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    model             = bundle["model"]
    selected_features = bundle["selected_features"]
    date_cols         = bundle.get("date_cols", [])
    categorical_cols  = bundle.get("categorical_cols", [])

    print(f"  Features selecionadas ({len(selected_features)}): {selected_features}")
    print(f"  Colunas de data:       {date_cols}")
    print(f"  Colunas categóricas:   {categorical_cols}")

    # 2. Carregar base limpa
    print(f"\nCarregando base de dados de {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Registros carregados: {len(df)}")

    # Garantir datas
    date_base_cols = [
        "dt_criacao", "dt_despacho_pedido", "dt_previsao_entrega_cliente",
        "dt_pagamento_pedido", "dt_entrega_pedido",
    ]
    for col in date_base_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 3. Recriar features de engenharia
    print("\nRecriando features de engenharia ...")
    df = build_engineered_features(df)

    # 4. Divisão Temporal — usar apenas o holdout (30% final, ordenado por dt_criacao)
    #    Estes são os dados que o modelo NUNCA viu durante o treino
    df = df.sort_values("dt_criacao").reset_index(drop=True)
    split_idx    = int(len(df) * TRAIN_SPLIT_RATIO)
    holdout_data = df.iloc[split_idx:].copy()
    print(f"  Total de registros: {len(df)}")
    print(f"  Holdout (30% final): {len(holdout_data)} registros")
    print(f"  Período do holdout: {holdout_data['dt_criacao'].min().date()} → {holdout_data['dt_criacao'].max().date()}")

    # 5. Filtro de datas opcional (dentro do holdout)
    filtered = holdout_data[
        (holdout_data["dt_criacao"] >= DATE_RANGE_START) &
        (holdout_data["dt_criacao"] <= DATE_RANGE_END)
    ].reset_index(drop=True)

    if len(filtered) == 0:
        print(
            f"\n[aviso] Nenhum registro encontrado para o intervalo "
            f"{DATE_RANGE_START} → {DATE_RANGE_END} dentro do holdout.\n"
            f"Verifique DATE_RANGE_START / DATE_RANGE_END ou use todo o holdout."
        )
        return

    print(f"\nRegistros no intervalo {DATE_RANGE_START} → {DATE_RANGE_END}: {len(filtered)}")

    # 6. Pré-processar para o modelo
    X = preprocess_for_model(filtered, selected_features, date_cols, categorical_cols)

    # 7. Predição em lote (muito mais rápido que linha a linha)
    print("\nExecutando predições ...")
    probabilities    = model.predict_proba(X)[:, 1]
    predictions_bin  = model.predict(X)

    # 8. Montar DataFrame de saída
    output_df = pd.DataFrame()

    # Identificadores e contexto para a dashboard
    id_cols = ["id", "cod_pedido", "uf", "grp_transportadora",
               "tp_praca", "des_unidade_negocio", "des_cd_origem",
               "cidade_destinatario",
               "dt_criacao", "dt_despacho_pedido", "dt_previsao_entrega_cliente"]
    for col in id_cols:
        if col in filtered.columns:
            output_df[col] = filtered[col].values

    # Gabarito real — permite comparar predição × realidade
    if "tp_performance_entrega" in filtered.columns:
        output_df["tp_performance_entrega"] = filtered["tp_performance_entrega"].values

    # Predições
    output_df["predicao_probabilidade"] = probabilities
    output_df["predicao_binaria"]       = predictions_bin
    output_df["risco_semaforo"]         = [classify_semaforo(p) for p in probabilities]

    # Métricas de contexto úteis para a dashboard
    context_cols = [
        "dias_gastos_cd", "dias_transito", "dias_restantes_prazo",
        "prazo_apos_despacho", "margem_entrega", "ratio_cd_prazo",
        "turno_despacho", "is_alta_temporada", "is_fds_despacho",
    ]
    for col in context_cols:
        if col in filtered.columns:
            output_df[col] = filtered[col].values

    # 9. Salvar CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    # 10. Sumário
    total        = len(output_df)
    n_verde      = (output_df["risco_semaforo"] == "🟢 Verde").sum()
    n_amarelo    = (output_df["risco_semaforo"] == "🟡 Amarelo").sum()
    n_vermelho   = (output_df["risco_semaforo"] == "🔴 Vermelho").sum()
    print(f"\n=== Sumário das Predições ===")
    print(f"  Total de pedidos:    {total}")
    print(f"  🟢 Verde (<30%):     {n_verde}  ({n_verde/total:.1%})")
    print(f"  🟡 Amarelo (30-70%): {n_amarelo}  ({n_amarelo/total:.1%})")
    print(f"  🔴 Vermelho (>70%):  {n_vermelho}  ({n_vermelho/total:.1%})")
    print(f"\nArquivo salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
