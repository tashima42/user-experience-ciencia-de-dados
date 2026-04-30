import pandas as pd
from sqlalchemy import create_engine
import os

# Database connection details
DB_USER = "daniel"
DB_PASS = "a1s2d3f4"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "cda"

def extract_city_local():
    print("Connecting to PostgreSQL...")
    conn_str = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(conn_str)
    
    query = "SELECT * FROM city_local"
    
    print("Fetching data from city_local...")
    # Using pandas to read the table
    df = pd.read_sql(query, engine)
    
    print(f"Extraction successful. Shape: {df.shape}")
    
    # Save to parquet for efficiency
    output_file = "city_local.parquet"
    print(f"Saving to {output_file}...")
    df.to_parquet(output_file, engine="pyarrow", index=False)
    print("Done!")

if __name__ == "__main__":
    extract_city_local()
