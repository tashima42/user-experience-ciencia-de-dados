"""
precompute_predictions_v2.py
----------------------------
Gera previsões de risco de atraso usando:
  - Base de dados limpa: pedidos_logistica_limpo.parquet  (saída do data_preparation_final.py)
  - Modelo treinado:     model_bundle.joblib               (saída do trabalho3.ipynb)
  - Opcional (SHAP):     shap_out/shap_wide.csv            (saída do shap_updated.py)

Diferenças em relação ao precompute_predictions.py original:
  - Usa parquet limpo no lugar do CSV de amostra
  - Recria as features de engenharia temporal (turno, semana, flags) que o novo modelo exige
  - Utiliza o holdout (últimos 30% por dt_criacao) como conjunto de inferência —
    dados que o modelo nunca viu durante o treinamento
  - O LightGBM agora recebe colunas do tipo `category` nativas (não mais cat.codes)
  - O CSV/JSON de saída inclui identificadores (id, cod_pedido), gabarito real
    (tp_performance_entrega), probabilidade, semáforo e suporte opcional a Fatores SHAP.

Pipeline Completo:
  1. Extração/Limpeza -> data_preparation_final.py
  2. Treinamento ML   -> trabalho3.ipynb
  3. Interpretabilidade SHAP -> shap_updated.py
  4. Predição e Junção -> precompute_predictions_v2.py (este script)
  5. Dashboard Web/Mapa -> server_map.py

Como executar:

    # Execução padrão (CSV, período padrão: 01–08/dez/2023)
    - Conda:  conda run -n logistica-eda python precompute_predictions_v2.py
    - UV:     uv run python precompute_predictions_v2.py

    Parâmetros disponíveis:
      --start         DATE_RANGE_START  Data de início (YYYY-MM-DD)  [padrão: 2023-11-18]
      --end           DATE_RANGE_END    Data de fim    (YYYY-MM-DD)  [padrão: 2023-12-01]
      --format        OUTPUT_FORMAT     Formato de saída: csv | json [padrão: csv]
      --output                          Caminho customizado para o arquivo de saída (opcional)
      --include-shap                    Se flag for usada, injeta os Top 4 Fatores SHAP de risco (lendo do arquivo shap_wide)
      --shap-wide                       Caminho customizado para o arquivo SHAP (padrão: shap_out/shap_wide.csv)

    Exemplos:
      uv run python precompute_predictions_v2.py --start 2023-11-18 --end 2023-12-01
      uv run python precompute_predictions_v2.py --start 2023-11-18 --end 2023-12-01 --format json --include-shap

    Variáveis de ambiente (alternativa aos parâmetros):
      DATE_RANGE_START=2023-11-18 DATE_RANGE_END=2023-12-01 uv run python precompute_predictions_v2.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configurações e Argumentos
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Gera previsões de risco de atraso logístico.")
    
    parser.add_argument(
        "--format", 
        choices=["csv", "json"], 
        default=os.environ.get("OUTPUT_FORMAT", "csv"),
        help="Formato de saída (padrão: csv)"
    )
    parser.add_argument(
        "--start", 
        default=os.environ.get("DATE_RANGE_START", "2023-11-18"),
        help="Data de início (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", 
        default=os.environ.get("DATE_RANGE_END", "2023-12-01"),
        help="Data de fim (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output", 
        help="Caminho customizado para o arquivo de saída (opcional)"
    )
    parser.add_argument(
        "--include-shap",
        action="store_true",
        help="Se informado, inclui os Top Fatores SHAP para cada pedido (lendo de shap_wide.csv)"
    )
    parser.add_argument(
        "--shap-wide",
        default="shap_out/shap_wide.csv",
        help="Caminho para o arquivo shap_wide.csv (padrão: shap_out/shap_wide.csv)"
    )
    
    return parser.parse_args()

# Caminhos Base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH       = os.path.join(BASE_DIR, "pedidos_logistica_limpo.parquet")
MODEL_BUNDLE_PATH  = os.path.join(BASE_DIR, "model_bundle.joblib")
CITY_LOCAL_PATH    = os.path.join(BASE_DIR, "daniel", "logistica", "city_local.parquet")

# Fração usada no treino — o holdout começa após esse ponto
TRAIN_SPLIT_RATIO = 0.70

# ---------------------------------------------------------------------------
# Feature engineering — espelha exatamente o que o trabalho3.ipynb faz
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

# Mapeamento semáforo → cor hexadecimal (usado exclusivamente na saída JSON)
SEMAFORO_HEX: dict[str, str] = {
    "🟢 Verde":    "#2ECC71",
    "🟡 Amarelo":  "#F1C40F",
    "🔴 Vermelho": "#E74C3C",
}


def classify_semaforo(prob_no_prazo: float) -> str:
    """
    Classifica o risco de atraso com base na probabilidade de NÃO atrasar.
    O modelo retorna predict_proba[:, 1] = probabilidade de entrega NO PRAZO (classe 1).
    O risco de atraso é o complemento: 1 - probabilidade_no_prazo.

    Thresholds:
      - Risco de atraso < 30%  → 🟢 Verde   (confortável)
      - Risco de atraso 30–70% → 🟡 Amarelo (atenção)
      - Risco de atraso > 70%  → 🔴 Vermelho (crítico)
    """
    risco_atraso = 1.0 - prob_no_prazo
    if risco_atraso < 0.30:
        return "🟢 Verde"
    elif risco_atraso < 0.70:
        return "🟡 Amarelo"
    return "🔴 Vermelho"


# ---------------------------------------------------------------------------
# Rotina principal
# ---------------------------------------------------------------------------
def main() -> None:
    # 0. Processar argumentos
    args = parse_args()
    
    # Definir caminho de saída se não fornecido
    if args.output:
        output_path = args.output
    else:
        ext = args.format
        output_path = os.path.join(BASE_DIR, "data", f"precomputed_predictions_v2.{ext}")

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
        (holdout_data["dt_criacao"] >= args.start) &
        (holdout_data["dt_criacao"] <= args.end)
    ].reset_index(drop=True)

    if len(filtered) == 0:
        print(
            f"\n[aviso] Nenhum registro encontrado para o intervalo "
            f"{args.start} → {args.end} dentro do holdout.\n"
            f"Verifique --start / --end ou use todo o holdout."
        )
        return

    print(f"\nRegistros no intervalo {args.start} → {args.end}: {len(filtered)}")

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
    # predicao_probabilidade = prob. de ser entregue NO PRAZO (saída bruta do modelo)
    # probabilidade_atraso   = complemento = risco de atraso (usado no semáforo)
    output_df["predicao_probabilidade"] = probabilities
    output_df["probabilidade_atraso"]   = 1.0 - probabilities
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

    # 8.5 Incluir Fatores SHAP (Opcional)
    if args.include_shap:
        shap_path = os.path.join(BASE_DIR, args.shap_wide)
        if os.path.exists(shap_path):
            print(f"\nIntegrando Top Fatores SHAP de {shap_path} ...")
            try:
                import sys
                if BASE_DIR not in sys.path:
                    sys.path.append(BASE_DIR)
                from join_shap_wide import _load_shap_top
                
                shap_top = _load_shap_top(
                    Path(shap_path),
                    shap_key="row_id",
                    top_n=4,
                    exclude_cols={"base_value", "expected_value", "prediction", "pred", "model_output", "output_value", "shap_base_value"},
                    chunksize=50000
                )
                
                output_df = output_df.merge(
                    shap_top,
                    left_on="cod_pedido",
                    right_on="row_id",
                    how="left"
                ).drop(columns=["row_id"], errors="ignore")
                
                matched_shap = output_df["top_shap_columns"].notna().sum()
                print(f"  Fatores SHAP integrados com sucesso. (Matched: {matched_shap}/{len(output_df)})")
            except Exception as e:
                print(f"  [aviso] Falha ao integrar SHAP: {e}")
        else:
            print(f"  [aviso] Arquivo SHAP não encontrado em {shap_path}. Ignorando flag --include-shap.")

    # 9. Salvar Arquivo
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if args.format == "csv":
        output_df.to_csv(output_path, index=False)
    else:
        # Para JSON, converter timestamps para string para evitar erros de serialização
        output_json = output_df.copy()
        for col in output_json.select_dtypes(include=['datetime64']).columns:
            output_json[col] = output_json[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        # --- Campos extras exclusivos do JSON ---

        # 1. Cor do semáforo em hexadecimal
        output_json["cor_semaforo"] = output_json["risco_semaforo"].map(SEMAFORO_HEX)

        # 2. Coordenadas geográficas (lat/lon) via merge com city_local
        #    Join duplo (nome_normalizado + uf) para evitar cidades homônimas entre estados
        if os.path.exists(CITY_LOCAL_PATH):
            print(f"  Carregando city_local de {CITY_LOCAL_PATH} ...")
            city_df = pd.read_parquet(CITY_LOCAL_PATH)[["nome_normalizado", "uf", "lat", "lon"]]
            output_json = output_json.merge(
                city_df,
                left_on=["cidade_destinatario", "uf"],
                right_on=["nome_normalizado", "uf"],
                how="left",
            ).drop(columns=["nome_normalizado"])
            matched = output_json["lat"].notna().sum()
            print(f"  Coordenadas resolvidas: {matched}/{len(output_json)} pedidos")
        else:
            print(f"  [aviso] {CITY_LOCAL_PATH} não encontrado — lat/lon serão nulos.")
            output_json["lat"] = None
            output_json["lon"] = None

        output_json.to_json(output_path, orient="records", indent=2, force_ascii=False)

    # 10. Sumário
    total        = len(output_df)
    n_verde      = (output_df["risco_semaforo"] == "🟢 Verde").sum()
    n_amarelo    = (output_df["risco_semaforo"] == "🟡 Amarelo").sum()
    n_vermelho   = (output_df["risco_semaforo"] == "🔴 Vermelho").sum()
    print(f"\n=== Sumário das Predições ({args.format.upper()}) ===")
    print(f"  Total de pedidos:    {total}")
    print(f"  🟢 Verde (<30%):     {n_verde}  ({n_verde/total:.1%})")
    print(f"  🟡 Amarelo (30-70%): {n_amarelo}  ({n_amarelo/total:.1%})")
    print(f"  🔴 Vermelho (>70%):  {n_vermelho}  ({n_vermelho/total:.1%})")
    print(f"\nArquivo salvo em: {output_path}")


if __name__ == "__main__":
    main()
