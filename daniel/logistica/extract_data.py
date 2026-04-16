import pandas as pd
from sqlalchemy import create_engine
import os

# Database connection details
DB_USER = "daniel"
DB_PASS = "a1s2d3f4"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "cda"

def extract_data():
    print("Connecting to PostgreSQL...")
    conn_str = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(conn_str)
    
    query = "SELECT * FROM pedidos_logistica"
    # TODO selecionar parte dos dados para análise, 4 meses de dados (deixar outra parte para implementação)
    
    print("Fetching data from pedidos_logistica...")
    # Using chunksize for large datasets if needed, but 0.5M should fit in memory (~100-200MB)
    df = pd.read_sql(query, engine)
    
    print(f"Extraction successful. Shape: {df.shape}")
    
    # Save to parquet for efficiency
    output_file = "pedidos_logistica.parquet"
    print(f"Saving to {output_file}...")
    df.to_parquet(output_file, engine="pyarrow")
    print("Done!")

if __name__ == "__main__":
    extract_data()
