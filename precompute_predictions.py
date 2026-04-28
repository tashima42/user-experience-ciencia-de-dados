from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, "model_bundle.joblib")
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, "data", "amostra.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "precomputed_predictions.csv")

# Date range for predictions
DATE_RANGE_START = "2023-12-01"
DATE_RANGE_END = "2023-12-08"


def _safe_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.to_datetime(value).date())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_payload(row_dict, selected_features):
    return {col: _safe_value(row_dict.get(col)) for col in selected_features}


def preprocess_one(payload, selected_features, date_cols, categorical_cols, categorical_mappings, numeric_cols):
    x = pd.DataFrame([payload])

    for column in selected_features:
        if column not in x.columns:
            x[column] = np.nan
    x = x[selected_features].copy()

    for column in date_cols:
        if column in x.columns:
            x[column] = pd.to_datetime(x[column], errors="coerce")
            x[column] = x[column].map(lambda value: value.toordinal() if pd.notna(value) else np.nan).astype("float32")

    for column in categorical_cols:
        if column in x.columns:
            mapping = categorical_mappings.get(column, {})
            x[column] = x[column].astype("string").map(mapping).astype("float32")

    for column in numeric_cols:
        if column in x.columns:
            x[column] = pd.to_numeric(x[column], errors="coerce").astype("float32")

    return x


def main() -> None:
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    model = bundle["model"]
    selected_features = bundle["selected_features"]
    date_cols = bundle["date_cols"]
    categorical_cols = bundle["categorical_cols"]
    categorical_mappings = bundle["categorical_mappings"]
    numeric_cols = [
        column for column in selected_features
        if column not in set(date_cols) and column not in set(categorical_cols)
    ]

    full_df = pd.read_csv(SAMPLE_CSV_PATH)
    if "dt_despacho_pedido" in full_df.columns:
        full_df["dt_despacho_pedido"] = pd.to_datetime(full_df["dt_despacho_pedido"], errors="coerce")

    full_df = full_df.reset_index(drop=False).rename(columns={"index": "_source_row_id"})

    # Filter by date range using dt_criacao
    if "dt_criacao" in full_df.columns:
        full_df["dt_criacao"] = pd.to_datetime(full_df["dt_criacao"], errors="coerce")
        full_df = full_df[
            (full_df["dt_criacao"] >= DATE_RANGE_START) &
            (full_df["dt_criacao"] <= DATE_RANGE_END)
        ].reset_index(drop=True)

    prediction_rows = []
    for _, row in full_df.iterrows():
        payload = _build_payload(row.to_dict(), selected_features)
        x = preprocess_one(payload, selected_features, date_cols, categorical_cols, categorical_mappings, numeric_cols)
        probability = float(model.predict_proba(x)[:, 1][0])
        
        # Build row with all selected features plus predictions
        row_dict = {col: _safe_value(row.get(col)) for col in selected_features}
        row_dict["_source_row_id"] = int(row.get("_source_row_id", -1))
        row_dict["predicao_probabilidade"] = probability
        prediction_rows.append(row_dict)

    output_df = pd.DataFrame(prediction_rows)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
