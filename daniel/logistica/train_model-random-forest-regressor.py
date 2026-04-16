import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
import os

# Importar lógica de preparação de dados existente
from data_preparation import load_and_clean_data

def prepare_ml_data(df):
    """
    Limpeza e filtragem adicional específica para Machine Learning
    conforme requisitos do projeto.
    """
    print("\n--- Limpeza e Filtragem de Dados ---")
    inicial = len(df)
    
    # 1. Remover registros com nulos em colunas críticas da jornada
    # Justificativa: Se TAT ou datas estão nulas, a jornada está incompleta para aprendizado.
    cols_criticas = ['dt_despacho', 'dt_entrega', 'dt_pagamento', 'qtd_dias_tat']
    df = df.dropna(subset=cols_criticas)
    
    posterior = len(df)
    perda = 1 - (posterior / inicial)
    
    print(f"Registros antes: {inicial}")
    print(f"Registros após limpeza: {posterior}")
    print(f"Perda de dados: {perda:.2%}")

    # 2. Tratar outliers extremos em qtd_dias_tat (Percentil 99)
    # Justificativa: Evitar que atrasos anômalos (erros de sistema ou casos isolados de meses) desviem o modelo.
    limite_superior = df['qtd_dias_tat'].quantile(0.99)
    df = df[df['qtd_dias_tat'] <= limite_superior].copy()
    print(f"Registros filtrados por outliers (> {limite_superior:.1f} dias): {len(df[df['qtd_dias_tat'] > limite_superior])}")
    
    return df

def train_and_evaluate(df, model_name="Random Forest Regressor"):
    """
    Treina o modelo, avalia métricas e gera previsões.
    """
    print(f"\n--- Iniciando Modelagem: {model_name} ---")
    
    # Ordenar por data de criação para divisão temporal estrita
    df = df.sort_values('dt_criacao')
    
    # Definição de Features e Target
    # Escolhemos features que o gestor tem em mãos ANTES da entrega para predição preventiva
    features_cat = ['uf', 'grp_transportadora', 'tp_praca', 'unidade_negocio']
    features_num = ['qtd_dias_tat', 'dias_aprovacao', 'dias_processamento_cd', 'hora_despacho']
    target = 'dias_atraso_real'
    
    X = df[features_cat + features_num]
    y = df[target]
    
    # Divisão Temporal Estrita (70% Treino/Teste, 30% Hold-out/Simulação Real)
    split_idx = int(len(df) * 0.7)
    X_train_test, X_holdout = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_test, y_holdout = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Split Treino/Teste dentro dos 70% iniciais
    X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.2, random_state=42)
    
    print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]} | Hold-out: {X_holdout.shape[0]}")

    # Encoding Categórico (Target Encoding)
    # Escolhido para lidar com alta cardinalidade sem explodir o número de colunas (One-Hot)
    encoder = TargetEncoder(cols=features_cat)
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_test_encoded = encoder.transform(X_test)
    X_holdout_encoded = encoder.transform(X_holdout)

    # Configuração do Modelo (Random Forest Regressor)
    # Justificativa: Robusto para lidar com não-linearidade e interações entre variáveis categóricas/numéricas.
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    
    print("Treinando modelo...")
    model.fit(X_train_encoded, y_train)
    
    # Previsões
    y_pred = model.predict(X_test_encoded)
    
    # Avaliação
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Resultados {model_name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.2f} dias") # Erro médio em dias: fácil de explicar ao gestor.
    print(f"  RMSE: {rmse:.2f} dias")
    
    return model, encoder, X_holdout_encoded, y_holdout, features_cat + features_num

def plot_importance(model, features, output_path="docs/feature_importance.png"):
    """
    Gera gráfico de importância de variáveis.
    """
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis', hue=feat_imp.index, legend=False)
    plt.title('Importância das Features - Driver de Atrasos')
    plt.xlabel('Importância Relativa')
    plt.tight_layout()
    
    if not os.path.exists('docs'):
        os.makedirs('docs')
        
    plt.savefig(output_path)
    print(f"Gráfico de importância salvo em: {output_path}")

def run_simulation(model, encoder, X_holdout_encoded, y_holdout, df_original):
    """
    Função de simulação para o usuário final (UX).
    Compara a Previsão do Modelo vs Prazo Oficial no conjunto de 30% mais recente.
    """
    print("\n--- Simulação de Uso Real (Hold-out) ---")
    
    preds = model.predict(X_holdout_encoded)
    
    # Pegar as datas originais do conjunto de hold-out para comparação
    df_sim = df_original.iloc[-len(y_holdout):].copy()
    df_sim['previsao_atraso_modelo'] = preds
    
    # Simulação de Insight UX: "Pedido com X dias de risco de atraso"
    # Se a previsão de atraso for > 0, o modelo está alertando risco.
    
    print(f"Média de Atraso Real no Período: {y_holdout.mean():.2f} dias")
    print(f"Média de Atraso Previsto pelo Modelo: {preds.mean():.2f} dias")
    
    # Exemplo de Alerta no Dashboard
    print("\n[UX Suggestion] Como usar os resultados no Dashboard:")
    print("1. Bandeira Vermelha: Previsão de atraso > 2 dias.")
    print("2. Bandeira Amarela: Previsão de atraso entre 0.5 e 2 dias.")
    print("3. Verde: Entrega prevista dentro do prazo ou antecipada.")
    
    # Mostrar top 5 casos de maior risco previsto
    print("\nTop 5 Pedidos com Maior Risco de Atraso Identificado:")
    print(df_sim[['dt_previsao', 'uf', 'grp_transportadora', 'previsao_atraso_modelo']].sort_values('previsao_atraso_modelo', ascending=False).head(5))

if __name__ == "__main__":
    # 1. Carregar dados usando a estrutura padrão do projeto
    input_file = "pedidos_logistica.parquet"
    df = load_and_clean_data(input_file)
    
    if df is not None:
        # 2. Filtragem específica para ML
        df_ml = prepare_ml_data(df)
        
        # 3. Treinar e Avaliar
        model, encoder, X_holdout, y_holdout, feature_list = train_and_evaluate(df_ml)
        
        # 4. Importância de Features
        plot_importance(model, feature_list)
        
        # 5. Simulação UX
        run_simulation(model, encoder, X_holdout, y_holdout, df_ml)
