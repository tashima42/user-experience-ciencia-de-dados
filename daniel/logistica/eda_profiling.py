import pandas as pd
from ydata_profiling import ProfileReport

def run_profiling():
    input_file = "pedidos_logistica.parquet"
    
    from data_preparation import load_and_clean_data
    df = load_and_clean_data(input_file)
    if df is None:
        return    

    print("\n--------------------------------------------------------")
    print(f"Dados prontos para criação de perfis. Formato: {df.shape}")

    print("Gerando relatório de perfis (isso pode levar alguns minutos para 0.5M registros)...")
    profile = ProfileReport(df, title="Logistics EDA Report", explorative=True)

    print("\n")
    output_report = "docs/logistica_profiling_report.html"
    print(f"Salvando relatório em {output_report}...")
    profile.to_file(output_report)
    print("Relatório de perfis gerado com sucesso!")

if __name__ == "__main__":
    run_profiling()
