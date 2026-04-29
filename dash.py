import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Logistics Risk Dashboard", layout="wide")

st.title("📦 Dashboard de Risco de Atrasos (Análise SHAP)")
st.markdown("Identificação dos principais causadores de atraso na malha logística.")

# ==============================================================================
# 1. Load Data
# ==============================================================================
# Data is now natively clean thanks to run_shap_main.py!
@st.cache_data
def load_data():
    return pd.read_csv(r'shap\out\shap_long.csv', engine='pyarrow')

# Initialize the data
with st.spinner("Carregando dados..."):
    df_shap = load_data()

# Filter for positive impacts globally
df_risco = df_shap[df_shap['direction'] == 'pos']

# ==============================================================================
# GRAPH 1: Global Risk Factors
# ==============================================================================
st.subheader("1. Fatores Globais de Risco")
st.write("Quais características do pedido mais contribuem para o risco de atraso?")

global_drivers = df_risco.groupby('feature')['shap_value'].mean().reset_index()
global_drivers = global_drivers.sort_values('shap_value', ascending=True)

fig_global = px.bar(
    global_drivers, 
    x='shap_value', 
    y='feature', 
    orientation='h',
    title='Média de Impacto no Atraso por Variável',
    labels={'shap_value': 'Força do Atraso (SHAP)', 'feature': 'Variável'},
    color_discrete_sequence=['#d62728'] 
)
st.plotly_chart(fig_global, use_container_width=True)

st.divider()

# ==============================================================================
# GRAPH 2: Variable Drill-Down
# ==============================================================================
st.subheader("2. Raio-X por Variável")
st.write("Selecione uma variável acima para ver quais valores específicos mais causam problemas.")

variavel_selecionada = st.selectbox(
    "Escolha a variável para detalhar:", 
    options=df_shap['feature'].unique()
)

df_detalhe = df_shap[df_shap['feature'] == variavel_selecionada]
detalhe_agrupado = df_detalhe.groupby('feature_value_raw')['shap_value'].mean().reset_index()
detalhe_agrupado = detalhe_agrupado.sort_values('shap_value', ascending=True)

top_bottom = pd.concat([detalhe_agrupado.head(10), detalhe_agrupado.tail(10)])
top_bottom['Cor'] = top_bottom['shap_value'].apply(lambda x: 'Aumenta Risco' if x > 0 else 'Reduz Risco')
color_map = {'Aumenta Risco': '#d62728', 'Reduz Risco': '#1f77b4'}

fig_detalhe = px.bar(
    top_bottom, 
    x='shap_value', 
    y='feature_value_raw', 
    orientation='h',
    color='Cor',
    color_discrete_map=color_map,
    title=f'Impacto Específico: {variavel_selecionada}',
    labels={'shap_value': 'Impacto no Atraso', 'feature_value_raw': 'Valor'}
)
st.plotly_chart(fig_detalhe, use_container_width=True)

st.divider()

# ==============================================================================
# GRAPH 3: Absolute Ranking (Configurable View)
# ==============================================================================
st.subheader("3. O Ranking Absoluto de Gargalos (Visão Configurável)")
st.write("Quais condições operacionais específicas mais aumentam o risco de atraso na malha inteira?")

# Layout for controls
col1, col2 = st.columns([1, 3])

with col1:
    top_n = st.slider("Quantidade de ofensores para exibir:", min_value=10, max_value=100, value=15, step=5)

with col2:
    all_features = df_risco['feature'].unique().tolist()
    # Sensible defaults: select all features EXCEPT 'dt_criacao' and 'cidade_destinatario'
    default_features = [f for f in all_features if f not in ['dt_criacao', 'cidade_destinatario']]
    
    selected_features = st.multiselect(
        "Selecione as variáveis para incluir no ranking:",
        options=all_features,
        default=default_features
    )

# Filter the dataframe based on the user's multiselect choice
df_acionavel = df_risco[df_risco['feature'].isin(selected_features)].copy()

# Create the combined string (e.g., "uf = SP")
df_acionavel['Condicao_Especifica'] = df_acionavel['feature'] + " = " + df_acionavel['feature_value_raw'].astype(str)

# Group by the specific condition and apply volume filter (min 50 orders)
ranking_absoluto = df_acionavel.groupby('Condicao_Especifica').agg(
    shap_value=('shap_value', 'mean'),
    volume=('shap_value', 'count')
).reset_index()

ranking_absoluto = ranking_absoluto[ranking_absoluto['volume'] >= 50]

# Sort, limit, and format
ranking_absoluto = ranking_absoluto.sort_values('shap_value', ascending=False).head(top_n)
ranking_absoluto = ranking_absoluto.sort_values('shap_value', ascending=True)
ranking_absoluto['Rotulo'] = ranking_absoluto['shap_value'].apply(lambda x: f"+{x:.3f}")

# Protect against empty charts if the user removes all filters or volume is too low
if ranking_absoluto.empty:
    st.warning("Nenhum dado atende aos filtros selecionados. Tente adicionar mais variáveis.")
else:
    fig_ranking = px.bar(
        ranking_absoluto, 
        x='shap_value', 
        y='Condicao_Especifica', 
        orientation='h',
        text='Rotulo', 
        title=f'Top {top_n} Maiores Causadores Operacionais de Atraso',
        labels={'shap_value': 'Força Média de Atraso (Impacto SHAP)', 'Condicao_Especifica': 'Variável e Valor'},
        color_discrete_sequence=['#ff7f0e'] 
    )

    fig_ranking.update_traces(textposition='outside')
    fig_ranking.update_layout(height=max(400, top_n * 30), margin=dict(r=50))

    st.plotly_chart(fig_ranking, use_container_width=True)