import pandas as pd
import os

#--------------------------------------------------------------------
# Colunas renomeadas para manter padrão da descricao_dos_dados.txt
#--------------------------------------------------------------------
#cod_pedido: Identificador único do pedido.
#cidade_destinatario: Cidade de destino do pedido.
#uf: Estado da cidade destino.
#dt_despacho_pedido: Data de saída do centro de distribuição.
#hr_despacho_pedido: Hora de saída do centro de distribuição.
#dt_entrega_pedido: Data de entrega ao destinatário.
#hr_entrega_pedido: Hora de entrega ao destinatário.
#dt_previsao_entrega_cliente: Data prometida de entrega.
#dt_criacao: Data de criação do pedido.
#dt_pagamento_pedido: Data de pagamento do pedido.
#flg_existem_ocorrencias: Indicador de ocorrências de entrega.
#tp_praca: Tipo de praça de entrega (Capital, Interior, Região Metropolitana).
#des_unidade_negocio: Classificação da unidade de negócio.
#des_cd_origem: Centro de distribuição de origem.
#qtd_dias_tat: Dias até a entrega.
#tp_performance_entrega: Status de entrega (dentro ou fora do prazo).
#--------------------------------------------------------------------

def load_and_clean_data(input_file="pedidos_logistica.parquet", drop_ids=True, output_file="pedidos_logistica_limpo.parquet"):
    """
    Carrega o arquivo parquet, realiza limpeza de dados, e aplica engenharia de variáveis
    de forma padronizada para as análises.
    """
    if not os.path.exists(input_file):
        print(f"Erro: {input_file} não achado. Favor rodar extract_data.py antes.")
        return None

    print(f"Carregando dados de {input_file}...")
    df = pd.read_parquet(input_file)

    # Renomear colunas
    # Voltar aos nomes originais para não perder o contexto - integração com a equipe
    df = df.rename(columns={"dt_previsao": "dt_previsao_entrega_cliente"})
    df = df.rename(columns={"dt_pagamento": "dt_pagamento_pedido"})
    df = df.rename(columns={"dt_entrega": "dt_entrega_pedido"})
    df = df.rename(columns={"dt_despacho": "dt_despacho_pedido"})
    df = df.rename(columns={"unidade_negocio": "des_unidade_negocio"})
    df = df.rename(columns={"cd_origem": "des_cd_origem"})
    df = df.rename(columns={"performance_entrega": "tp_performance_entrega"})
    df = df.rename(columns={"tp_praca": "tp_praca"})
    df = df.rename(columns={"flg_ocorrencias": "flg_existem_ocorrencias"})

    # Remover colunas de identificação e normalizar nomes de colunas
    if drop_ids:
        cols_to_drop = ['id', 'cidade_destinatario']
        cols_present = [c for c in cols_to_drop if c in df.columns]
        if cols_present:
            print(f"Removendo colunas desnecessárias: {cols_present}")
            df = df.drop(columns=cols_present)
    
    if 'cidade_destinatario_normalizada' in df.columns:
        if 'cidade_destinatario' in df.columns:
            df = df.drop(columns=['cidade_destinatario'])
        df = df.rename(columns={'cidade_destinatario_normalizada': 'cidade_destinatario'})

    print("Colunas após renomeação:")
    cols = df.columns
    for i in range(0, len(cols), 3):
        print(", ".join(cols[i:i+3]))

    # Verificar consistência lógica de datas (UX-Safety Check)
    inconsistentes = df[df['dt_entrega_pedido'] < df['dt_despacho_pedido']]
    if len(inconsistentes) > 0:
        print(f"Removendo registros inconsistentes (entrega antecede despacho): {len(inconsistentes)}")
        df = df[df['dt_entrega_pedido'] >= df['dt_despacho_pedido']]

    # Remover duplicados
    qtd_antes = len(df)
    df = df.drop_duplicates()
    qtd_depois = len(df)
    if qtd_antes > qtd_depois:
        print(f"Removidos {qtd_antes - qtd_depois} registros duplicados.")

    # Garantir que as colunas de data estão em formato datetime
    cols_data = ['dt_criacao', 'dt_pagamento_pedido', 'dt_despacho_pedido', 'dt_entrega_pedido', 'dt_previsao_entrega_cliente']
    for col in cols_data:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Filtro Temporal comentado como na análise original
    # No data_preparation.py, alterando a parte do filtro:
    # garantir periodo de 6 meses para análise
    df = df[(df['dt_criacao'] >= '2023-07-01') & (df['dt_criacao'] <= '2023-12-31')].copy()

    #-----------------------------------------------------------------------------------------
    # Engenharia de features
    print("Realizando engenharia de features...")    
    
    # Variáveis alvo de atraso    
    # df['is_atrasado'] = (df['dt_entrega_pedido'] > df['dt_previsao_entrega_cliente']).astype(int)
    # if 'tp_performance_entrega' in df.columns:
    #     # TODO verificar se melhor não considerar esta variavel!
    #     # Definindo atraso (usando a coluna de performance oficial do projeto)
    #     df['is_atrasado_oficial'] = (df['tp_performance_entrega'] == 'Fora do Prazo').astype(int)

    # Transformar tp_performance_entrega em binário
    if 'tp_performance_entrega' in df.columns:
        df['tp_performance_entrega'] = (
            df['tp_performance_entrega']
            .astype('string')
            .str.strip()
            .map({
                'Entregue no Prazo': 1,
                'Fora do Prazo': 0
        })
        .astype('Int64')
    )

    # df['hora_despacho_pedido'] = df['dt_despacho_pedido'].dt.hour    
    #df['dia_semana_despacho_pedido'] = df['dt_despacho_pedido'].dt.day_name()
    #df['mes_despacho_pedido'] = df['dt_despacho_pedido'].dt.month

    # Tempo de Aprovação
    # df['dias_aprovacao'] = (df['dt_pagamento_pedido'] - df['dt_criacao']).dt.total_seconds() / 86400
    
    # Tempo de Processamento Interno
    # df['dias_processamento_cd'] = (df['dt_despacho_pedido'] - df['dt_pagamento_pedido']).dt.total_seconds() / 86400
    df['dias_gastos_cd'] = (df['dt_despacho_pedido'] - df['dt_pagamento_pedido']).dt.total_seconds() / 86400
    
    # Tempo de Trânsito (Lead Time)
    df['dias_transito'] = (df['dt_entrega_pedido'] - df['dt_despacho_pedido']).dt.total_seconds() / 86400
    
    # Erro de Previsão
    df['dias_atraso_real'] = (df['dt_entrega_pedido'] - df['dt_previsao_entrega_cliente']).dt.total_seconds() / 86400
    
    # Tempo Total de Ciclo
    df['dias_ciclo'] = (df['dt_entrega_pedido'] - df['dt_criacao']).dt.total_seconds() / 86400

    # Dias restantes até a entrega
    df['dias_restantes_prazo'] = (df['dt_previsao_entrega_cliente'] - df['dt_pagamento_pedido']).dt.days

    # Transformar colunas em categoria
    categorical_features = ['uf', 'grp_transportadora', 'cidade_destinatario', 'tp_praca', 'des_unidade_negocio', 'des_cd_origem']
    for col in categorical_features:
        df[col] = df[col].astype('category')
    
    print(f"Processamento concluído. Formato final: {df.shape}")
    
    if output_file:
        print(f"Salvando dataset limpo em {output_file}...")
        df.to_parquet(output_file, index=False)

    return df
