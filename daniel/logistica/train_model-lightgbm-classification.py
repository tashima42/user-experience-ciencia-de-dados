import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Importar lógica de preparação de dados existente
from data_preparation import load_and_clean_data

def prepare_data_for_lgbm(df):
    """
    Limpeza e filtragem sugerida para o modelo LightGBM.
    """
    print("\n--- Preparação de Dados para LightGBM ---")
    inicial = len(df)
    
    # 1. Limpeza Crítica: dropna em campos fundamentais
    # Requisito: dt_despacho, dt_entrega, dt_pagamento
    cols_limpeza = ['dt_despacho', 'dt_entrega', 'dt_pagamento', 'qtd_dias_tat']
    
    # --- OPÇÃO DE IMPUTAÇÃO (COMENTADA) ---
    # Se decidirmos não dropar os nulos de dt_pagamento (6.4%):
    # df['dt_pagamento'] = df['dt_pagamento'].fillna(method='ffill') # ou uma data fixa 'Pendente'
    # df['dias_aprovacao'] = df['dias_aprovacao'].fillna(-1) # Categoria para 'Pendente'
    # --------------------------------------
    
    df = df.dropna(subset=cols_limpeza).copy()
    
    posterior = len(df)
    print(f"Registros antes: {inicial}")
    print(f"Registros após limpeza (dropna): {posterior}")
    print(f"Perda de dados: {1 - (posterior/inicial):.2%}")

    # 2. Tratamento de Outliers (Percentil 99 em qtd_dias_tat)
    limite_99 = df['qtd_dias_tat'].quantile(0.99)
    df = df[df['qtd_dias_tat'] <= limite_99].copy()
    print(f"Outliers removidos (TAT > {limite_99:.1f} dias): {posterior - len(df)}")

    return df

def train_lgbm(df):
    """
    Treinamento do modelo LightGBM usando a API nativa e recursos de classificação.
    """
    # 1. Divisão Temporal 70/30
    df = df.sort_values('dt_criacao')
    split_idx = int(len(df) * 0.7)
    train_test_data = df.iloc[:split_idx].copy()
    holdout_data = df.iloc[split_idx:].copy()
    
    # 2. Definição de Features e Alvo
    # Categóricas (Nativas do LightGBM)
    cat_features = ['uf', 'grp_transportadora', 'tp_praca', 'unidade_negocio', 'dia_semana_despacho']
    # Numéricas
    num_features = ['qtd_dias_tat', 'dias_aprovacao', 'dias_processamento_cd', 'hora_despacho']
    
    target = 'is_atrasado'
    
    # Converter categóricas para o tipo 'category' do Pandas (Exigência do LightGBM)
    for col in cat_features:
        train_test_data[col] = train_test_data[col].astype('category')
        holdout_data[col] = holdout_data[col].astype('category')

    X = train_test_data[cat_features + num_features]
    y = train_test_data[target]
    
    # Split para Treino e Validação (dentro dos 70%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    # 3. Cálculo do scale_pos_weight para desbalanceamento
    # Justificativa: LightGBM lida melhor com classes minoritárias se aumentarmos o peso dos positivos (atrasos).
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    spw = neg_count / pos_count
    
    print(f"\nConfigurando scale_pos_weight: {spw:.2f} (Classe 'Atrasado' é minoritária)")

    # 4. Configuração do Modelo
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'scale_pos_weight': spw,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    print("Iniciando treinamento com LightGBM...")
    
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, categorical_feature=cat_features)
    
    model = lgb.train(
        params,
        train_set,
        valid_sets=[train_set, val_set],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )

    return model, holdout_data, cat_features + num_features

def run_daily_operational_simulation(model, holdout_data, features, start_date='2023-12-05', window_days=8):
    """
    Simulador diário: Executa o modelo para uma janela específica,
    focando na fila de pedidos pendentes (Já criados, mas não despachados).
    """
    print(f"\n--- Simulador Operacional (Janela: {window_days} dias a partir de {start_date}) ---")
    
    # Gerar o intervalo de datas para a simulação
    start_dt = pd.to_datetime(start_date)
    datas_simulacao = pd.date_range(start=start_dt, periods=window_days)
    
    daily_stats = []
    
    for dia_dt in datas_simulacao:
        # Filtro da Fila do Dia: 
        # Já entrou no sistema (dt_criacao <= dia) E ainda não saiu do CD (dt_despacho >= dia)
        mask = (holdout_data['dt_criacao'] <= dia_dt) & (holdout_data['dt_despacho'] >= dia_dt)
        df_fila = holdout_data[mask].copy()
        
        if len(df_fila) == 0:
            continue
            
        # Predizer Risco
        X_fila = df_fila[features]
        probs = model.predict(X_fila)
        df_fila['p_atraso'] = probs
        
        # Classificação UX
        df_fila['status'] = df_fila['p_atraso'].apply(
            lambda p: 'Crítico' if p > 0.70 else ('Atenção' if p > 0.40 else 'Normal')
        )
        
        c = df_fila['status'].value_counts()
        
        daily_stats.append({
            'Data Ref': dia_dt.date(),
            'Total Fila': len(df_fila),
            'Crítico': c.get('Crítico', 0),
            'Atenção': c.get('Atenção', 0),
            'Normal': c.get('Normal', 0),
            'Atrasos Reais': df_fila['is_atrasado'].sum()
        })
        
    resumo = pd.DataFrame(daily_stats)
    
    if not resumo.empty:
        print("\nSaúde Operacional da Fila (Snapshot Diário):")
        # Mostrar os últimos 15 dias do hold-out para não poluir o terminal
        print(resumo.tail(15).to_string(index=False))
        
        # Insight de Negócio
        pico_critico = resumo.loc[resumo['Crítico'].idxmax()]
        print(f"\n[DASHBOARD ALERT] Pico de risco detectado em {pico_critico['Data Ref']} "
              f"com {pico_critico['Crítico']} pedidos em estado Crítico na fila.")
    else:
        print("Sem dados suficientes para a simulação diária.")

def plot_lgbm_importance(model, output_path="docs/lightgbm_importance.png"):
    """
    Gráfico de importância nativo do LightGBM para storytelling.
    """
    plt.figure(figsize=(12, 8))
    lgb.plot_importance(model, max_num_features=15, importance_type='gain', precision=2)
    plt.title('Importância das Variáveis (LightGBM Gain)')
    plt.tight_layout()
    
    if not os.path.exists('docs'):
        os.makedirs('docs')
        
    plt.savefig(output_path)
    print(f"\nGráfico de importância salvo em: {output_path}")

if __name__ == "__main__":
    # 1. Carregar e Limpar (Manter IDs para Simulação de Dashboard)
    input_file = "pedidos_logistica.parquet"
    df = load_and_clean_data(input_file, drop_ids=False)
    
    if df is not None:
        df_lgbm = prepare_data_for_lgbm(df)
        
        # 2. Treinar
        model, holdout, features = train_lgbm(df_lgbm)
        
        # 3. Importância
        plot_lgbm_importance(model)
        
        # 4. Simulação Operacional Diária (Pedidos Pendentes no CD)
        run_daily_operational_simulation(model, holdout, features)

        # Notas de Storytelling:
        print("\n--- Notas de Storytelling e UX ---")
        print("1. Por que LightGBM? Supera o Random Forest em datasets desbalanceados devido ao Gradient Boosting")
        print("   e à técnica GOSS, que foca em instâncias com maiores gradientes (erros).")
        print("2. Por que Probabilidade? Permite que o gestor filtre por 'Crítico' (> 70%) para agir")
        print("   apenas nos casos de maior incerteza ou impacto.")
        print("3. Dados de Pagamento Nulos? Sugerimos imputação por mediana ou categoria 'Pendente' caso")
        print("   seja necessário manter os 6.4% de registros que atualmente são descartados.")
