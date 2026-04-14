from flask import Flask, request, jsonify, render_template
import os
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, "model_bundle.joblib")
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, "data", "amostra.csv")
SAMPLE_SIZE = 200
SAMPLE_RANDOM_STATE = 42

bundle = joblib.load(MODEL_BUNDLE_PATH)

model = bundle["model"]
selected_features = bundle["selected_features"]
date_cols = bundle["date_cols"]
categorical_cols = bundle["categorical_cols"]
categorical_mappings = bundle["categorical_mappings"]
threshold = bundle["threshold"]
numeric_cols = [
    c for c in selected_features
    if c not in set(date_cols) and c not in set(categorical_cols)
]

sample_df = pd.DataFrame()
sample_rows = []
table_columns = []
sample_error = None


def _safe_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.to_datetime(value).date())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _row_to_payload(row_dict):
    return {col: _safe_value(row_dict.get(col)) for col in selected_features}


def _predict_payload(payload):
    x = preprocess_one(payload)
    proba = float(model.predict_proba(x)[:, 1][0])
    pred = int(proba >= threshold)
    classe_descricao = "No prazo" if pred == 1 else "Atraso"
    return {
        "probabilidade_atraso": proba,
        "classe_prevista": pred,
        "classe_descricao": classe_descricao,
    }


def _load_sample_data():
    global sample_df, sample_rows, table_columns, sample_error
    try:
        full_df = pd.read_csv(SAMPLE_CSV_PATH)
        if len(full_df) > SAMPLE_SIZE:
            sample_df = full_df.sample(n=SAMPLE_SIZE, random_state=SAMPLE_RANDOM_STATE).copy()
        else:
            sample_df = full_df.copy()

        sample_df = sample_df.reset_index(drop=False).rename(columns={"index": "_source_row_id"})
        table_columns = [c for c in selected_features if c in sample_df.columns]
        if "tp_performance_entrega" in sample_df.columns:
            table_columns.append("tp_performance_entrega")

        sample_rows = []
        for row_id, row in sample_df.iterrows():
            row_data = {"_row_id": int(row_id)}
            row_data["_source_row_id"] = int(row.get("_source_row_id", row_id))
            for col in table_columns:
                row_data[col] = _safe_value(row.get(col))
            sample_rows.append(row_data)

        sample_error = None
    except Exception as exc:
        sample_df = pd.DataFrame()
        sample_rows = []
        table_columns = []
        sample_error = str(exc)


_load_sample_data()

def preprocess_one(payload):
    x = pd.DataFrame([payload])

    # Ensure all expected columns exist
    for c in selected_features:
        if c not in x.columns:
            x[c] = np.nan
    x = x[selected_features].copy()

    # Dates to ordinal
    for c in date_cols:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors="coerce")
            x[c] = x[c].map(lambda d: d.toordinal() if pd.notna(d) else np.nan).astype("float32")

    # Categorical mapping (unknown -> NaN)
    for c in categorical_cols:
        if c in x.columns:
            m = categorical_mappings.get(c, {})
            x[c] = x[c].astype("string").map(m).astype("float32")

    # Numeric features must be numeric for LightGBM
    for c in numeric_cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype("float32")

    return x


@app.get("/")
def index():
    return render_template(
        "index.html",
        rows=sample_rows,
        table_columns=table_columns,
        sample_error=sample_error,
    )

@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    return jsonify(_predict_payload(payload))


@app.get("/random_row")
def random_row():
    if sample_df.empty:
        return jsonify({"error": "CSV not loaded."}), 404

    row_id = int(np.random.randint(0, len(sample_df)))
    row_dict = sample_df.iloc[row_id].to_dict()
    payload = _row_to_payload(row_dict)
    return jsonify({"row_id": row_id, "payload": payload})


@app.post("/predict_row/<int:row_id>")
def predict_row(row_id):
    if sample_df.empty:
        return jsonify({"error": "CSV not loaded."}), 404
    if row_id < 0 or row_id >= len(sample_df):
        return jsonify({"error": "Invalid row id."}), 400

    row_dict = sample_df.iloc[row_id].to_dict()
    payload = _row_to_payload(row_dict)
    result = _predict_payload(payload)
    result["row_id"] = row_id
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)