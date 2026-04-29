from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Dashboard SHAP - Logística", layout="wide")

st.title("Dashboard de Risco de Atrasos (SHAP)")
st.markdown("Explique as probabilidades do LightGBM via drivers (valores SHAP).")


@st.cache_data
def load_shap_long(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, engine="pyarrow" if path.lower().endswith(".csv") else None)


default_path = os.path.join("shap", "out", "shap_long.csv")
path = st.sidebar.text_input("Caminho do shap_long (csv/parquet)", value=default_path)

if not os.path.exists(path):
    st.error(
        "Arquivo não encontrado. Gere primeiro com `python run_shap_lgbm.py --input <seu_arquivo>` "
        "e confirme que existe `shap/out/shap_long.csv`."
    )
    st.stop()

with st.spinner("Carregando dados SHAP..."):
    df_shap = load_shap_long(path)

required_cols = {"row_id", "feature", "feature_value_raw", "shap_value", "direction"}
missing = required_cols - set(df_shap.columns)
if missing:
    st.error(f"shap_long inválido. Colunas faltando: {sorted(missing)}")
    st.stop()


st.sidebar.divider()
all_features = sorted(df_shap["feature"].dropna().unique().tolist())
selected_features = st.sidebar.multiselect("Filtrar variáveis", options=all_features, default=all_features)

direction_choice = st.sidebar.radio(
    "Impacto",
    options=["Aumenta risco (pos)", "Reduz risco (neg)", "Ambos"],
    index=0,
)

df = df_shap[df_shap["feature"].isin(selected_features)].copy()
if direction_choice == "Aumenta risco (pos)":
    df = df[df["direction"] == "pos"]
elif direction_choice == "Reduz risco (neg)":
    df = df[df["direction"] == "neg"]


tab1, tab2, tab3 = st.tabs(["Visão global", "Raio-X por variável", "Ranking acionável"])

with tab1:
    st.subheader("1) Fatores globais")
    st.write("Quais variáveis, em média, mais empurram a previsão para maior/menor risco?")

    global_drivers = df.groupby("feature")["shap_value"].mean().reset_index()
    global_drivers = global_drivers.sort_values("shap_value", ascending=True)

    fig = px.bar(
        global_drivers,
        x="shap_value",
        y="feature",
        orientation="h",
        title="Média de impacto (SHAP) por variável",
        labels={"shap_value": "Impacto médio (SHAP)", "feature": "Variável"},
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("2) Raio-X por variável")
    st.write("Veja quais valores específicos (ex.: UF=SP, Transportadora=X) mais impactam o risco.")

    variavel = st.selectbox("Variável", options=all_features)
    df_var = df_shap[df_shap["feature"] == variavel].copy()

    agg = df_var.groupby("feature_value_raw")["shap_value"].mean().reset_index()
    agg = agg.sort_values("shap_value", ascending=True)

    top_n = st.slider("Top N valores", min_value=10, max_value=80, value=20, step=5)
    if len(agg) > top_n:
        # show extremes (bottom + top) to avoid hiding reductions when focusing on risk
        n = top_n // 2
        view = pd.concat([agg.head(n), agg.tail(top_n - n)], ignore_index=True)
        view = view.sort_values("shap_value", ascending=True)
    else:
        view = agg

    view["Cor"] = view["shap_value"].apply(lambda x: "Aumenta risco" if x > 0 else "Reduz risco")
    fig2 = px.bar(
        view,
        x="shap_value",
        y="feature_value_raw",
        orientation="h",
        color="Cor",
        color_discrete_map={"Aumenta risco": "#d62728", "Reduz risco": "#1f77b4"},
        title=f"Impacto específico: {variavel}",
        labels={"shap_value": "Impacto (SHAP)", "feature_value_raw": "Valor"},
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("3) Ranking acionável (condição específica)")
    st.write("Ranking por (variável = valor), com filtro mínimo de volume.")

    col1, col2 = st.columns([1, 2])
    with col1:
        top_n = st.slider("Quantidade no ranking", min_value=10, max_value=100, value=20, step=5, key="rank_topn")
        min_volume = st.number_input("Volume mínimo (linhas)", min_value=1, value=50, step=10)
    with col2:
        rank_features = st.multiselect("Variáveis no ranking", options=all_features, default=all_features)

    df_rank = df_shap[df_shap["feature"].isin(rank_features)].copy()
    df_rank["condicao"] = df_rank["feature"].astype(str) + " = " + df_rank["feature_value_raw"].astype(str)

    ranking = (
        df_rank.groupby("condicao")
        .agg(shap_value=("shap_value", "mean"), volume=("shap_value", "count"))
        .reset_index()
    )
    ranking = ranking[ranking["volume"] >= int(min_volume)].copy()
    ranking = ranking.sort_values("shap_value", ascending=False).head(int(top_n))
    ranking = ranking.sort_values("shap_value", ascending=True)

    if ranking.empty:
        st.warning("Nenhum dado atende aos filtros atuais.")
    else:
        ranking["rotulo"] = ranking["shap_value"].map(lambda x: f"{x:+.3f}")
        fig3 = px.bar(
            ranking,
            x="shap_value",
            y="condicao",
            orientation="h",
            text="rotulo",
            title=f"Top {top_n} condições por impacto médio (SHAP)",
            labels={"shap_value": "Impacto médio (SHAP)", "condicao": "Condição"},
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=max(450, int(top_n) * 28), margin=dict(r=60))
        st.plotly_chart(fig3, use_container_width=True)

