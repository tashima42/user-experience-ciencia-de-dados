import pandas as pd
import os

def verify_risk_window():
    input_file = "pedidos_logistica.parquet"
    if not os.path.exists(input_file):
        print("Error: Parquet file not found.")
        return

    df = pd.read_parquet(input_file)
    df['dt_despacho'] = pd.to_datetime(df['dt_despacho'])
    df['hora_despacho'] = df['dt_despacho'].dt.hour
    
    # Using the same logic as the notebook to verify the 7.6% rate
    df['is_atraso'] = (df['dt_entrega'] > df['dt_previsao']).astype(int)
    
    atraso_por_hora = df.groupby('hora_despacho')['is_atraso'].mean() * 100
    
    print("Taxa de atraso por hora:")
    print(atraso_por_hora.sort_index())
    
    if 17 in atraso_por_hora.index:
        rate = atraso_por_hora.loc[17]
        print(f"\nTaxa de atraso entre 17h e 18h: {rate:.2f}%")
        if 7.0 <= rate <= 8.5:
            print("VREIFICADO: A taxa está dentro da faixa esperada (~7.6%)!")
        else:
            print(f"AVISO: A taxa encontrada ({rate:.2f}%) diverge do esperado (~7.6%).")
    else:
        print("Erro: Hora 17 não encontrada nos dados.")

if __name__ == "__main__":
    verify_risk_window()
