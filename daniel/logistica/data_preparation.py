import pandas as pd
import os

def load_and_clean_data(input_file="pedidos_logistica.parquet"):
    """
    Carrega o arquivo parquet, realiza limpeza de dados, e aplica engenharia de variáveis
    de forma padronizada para as análises.
    """
    if not os.path.exists(input_file):
        print(f"Erro: {input_file} não achado. Favor rodar extract_data.py antes.")
        return None

    print(f"Carregando dados de {input_file}...")
    df = pd.read_parquet(input_file)

    # 1. Remover colunas de identificação e normalizar nomes de colunas
    cols_to_drop = ['id', 'cod_pedido', 'cidade_destinatario']
    cols_present = [c for c in cols_to_drop if c in df.columns]
    if cols_present:
        print(f"Removendo colunas desnecessárias: {cols_present}")
        df = df.drop(columns=cols_present)
    
    if 'cidade_destinatario_normalizada' in df.columns:
        print("Renomeando 'cidade_destinatario_normalizada' para 'cidade_destinatario'")
        df = df.rename(columns={'cidade_destinatario_normalizada': 'cidade_destinatario'})

    # 2. Verificar consistência lógica de datas (UX-Safety Check)
    inconsistentes = df[df['dt_entrega'] < df['dt_despacho']]
    if len(inconsistentes) > 0:
        print(f"Removendo registros inconsistentes (entrega antecede despacho): {len(inconsistentes)}")
        df = df[df['dt_entrega'] >= df['dt_despacho']]

    # 3. Remover duplicados
    qtd_antes = len(df)
    df = df.drop_duplicates()
    qtd_depois = len(df)
    if qtd_antes > qtd_depois:
        print(f"Removidos {qtd_antes - qtd_depois} registros duplicados.")

    # 4. Garantir que as colunas de data estão em formato datetime
    cols_data = ['dt_criacao', 'dt_pagamento', 'dt_despacho', 'dt_entrega', 'dt_previsao']
    for col in cols_data:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Filtro Temporal comentado como na análise original
    # No data_preparation.py, alterando a parte do filtro:
    # garantir periodo de 6 meses para análise
    df = df[(df['dt_criacao'] >= '2023-07-01') & (df['dt_criacao'] <= '2023-12-31')].copy()

    #-----------------------------------------------------------------------------------------
    # Engenharia de features temporais
    print("Realizando engenharia de features...")

    df['dt_despacho'] = pd.to_datetime(df['dt_despacho'])
    df['hora_despacho'] = df['dt_despacho'].dt.hour
    df['dia_semana_despacho'] = df['dt_despacho'].dt.day_name()
    df['mes_despacho'] = df['dt_despacho'].dt.month

    # Tempo de Aprovação
    df['dias_aprovacao'] = (df['dt_pagamento'] - df['dt_criacao']).dt.total_seconds() / 86400
    
    # Tempo de Processamento Interno
    df['dias_processamento_cd'] = (df['dt_despacho'] - df['dt_pagamento']).dt.total_seconds() / 86400
    
    # Tempo de Trânsito (Lead Time)
    df['dias_transito'] = (df['dt_entrega'] - df['dt_despacho']).dt.total_seconds() / 86400
    
    # Erro de Previsão
    df['dias_atraso_real'] = (df['dt_entrega'] - df['dt_previsao']).dt.total_seconds() / 86400
    
    # Tempo Total de Ciclo
    df['dias_ciclo'] = (df['dt_entrega'] - df['dt_criacao']).dt.total_seconds() / 86400
    
    # Variáveis alvo de atraso
    df['is_atrasado'] = (df['dt_entrega'] > df['dt_previsao']).astype(int)
    if 'performance_entrega' in df.columns:
        # TODO verificar se melhor não considerar esta variavel!
        # Definindo atraso (usando a coluna de performance oficial do projeto)
        df['is_atrasado_oficial'] = (df['performance_entrega'] == 'Fora do Prazo').astype(int)
    
    print(f"Processamento concluído. Formato final: {df.shape}")
    
    return df
