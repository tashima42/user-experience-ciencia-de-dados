from flask import Flask, request, render_template
import os
import math
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, "data", "amostra.csv")
PREDICTION_CACHE_PATH = os.path.join(BASE_DIR, "data", "precomputed_predictions.csv")

DISPLAY_EXCLUDED_COLUMNS = {
    "dt_despacho_pedido",
    "tp_performance_entrega",
    "qtd_dias_tat",
    "des_unidade_negocio",
    "flg_existem_ocorrencias",
    "row_id",
    "hr_despacho_pedido",
    "hr_entrega_pedido",
    "des_unidade_negocio",
    "dt_entrega_pedido"
}

full_df = pd.DataFrame()
sample_df = pd.DataFrame()
sample_rows = []
table_columns = []
des_cd_origem_options = []
sample_error = None
DEFAULT_PAGE_SIZE = 50
DEFAULT_FILTER_DATE = "2023-10-01"


def _safe_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def _parse_filter_date(value):
    if not value:
        return None
    parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def _load_prediction_cache(full_base_df):
    if not os.path.exists(PREDICTION_CACHE_PATH):
        raise FileNotFoundError(
            "Prediction cache file not found. Run precompute_predictions.py first."
        )

    cached_df = pd.read_csv(PREDICTION_CACHE_PATH)
    expected_columns = {"_source_row_id", "predicao_probabilidade", "predicao"}
    if not expected_columns.issubset(set(cached_df.columns)):
        raise ValueError(
            "Prediction cache has invalid columns. Rebuild with precompute_predictions.py."
        )

    if len(cached_df) != len(full_base_df):
        raise ValueError(
            "Prediction cache row count does not match dataset. Rebuild with precompute_predictions.py."
        )

    return cached_df


def _build_display_data(df):
    table_columns_local = ["predicao"] + [
        col for col in df.columns
        if col not in {"_source_row_id", "predicao"} and col not in DISPLAY_EXCLUDED_COLUMNS
    ]

    rows_local = []
    for row_id, (_, row) in enumerate(df.iterrows()):
        row_data = {"_row_id": int(row_id)}
        row_data["_source_row_id"] = int(row.get("_source_row_id", row_id))
        for col in table_columns_local:
            value = _safe_value(row.get(col))
            if col == "predicao_probabilidade" and value is not None:
                try:
                    probability = float(value)
                except (TypeError, ValueError):
                    row_data[col] = value
                else:
                    row_data[col] = f"{(1 - probability) * 100:.2f}%"
            else:
                row_data[col] = value
        rows_local.append(row_data)

    return rows_local, table_columns_local


def _build_filtered_sample(des_cd_origem=None, dt_despacho_pedido=None):
    if sample_df.empty:
        return pd.DataFrame()

    filtered_df = sample_df.copy()

    if "predicao" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["predicao"].astype("string") == "🔴 Ruim"]

    if des_cd_origem:
        filtered_df = filtered_df[filtered_df["des_cd_origem"].astype("string") == des_cd_origem]

    cutoff = _parse_filter_date(dt_despacho_pedido)
    if cutoff is not None and "dt_despacho_pedido" in filtered_df.columns:
        upper_bound = cutoff + pd.Timedelta(days=5)
        filtered_df = filtered_df[
            (filtered_df["dt_despacho_pedido"] >= cutoff)
            & (filtered_df["dt_despacho_pedido"] <= upper_bound)
        ]

    return filtered_df


def _build_priority_sample(filtered_df, limit=10):
    if filtered_df.empty or "predicao_probabilidade" not in filtered_df.columns:
        return [], []

    priority_df = filtered_df.copy()
    priority_df["predicao_probabilidade"] = pd.to_numeric(
        priority_df["predicao_probabilidade"], errors="coerce"
    )
    priority_df = priority_df.sort_values(by="predicao_probabilidade", ascending=True, na_position="last")
    priority_df = priority_df.head(limit)

    return _build_display_data(priority_df)


def _load_sample_data():
    global full_df, sample_df, sample_rows, table_columns, des_cd_origem_options, sample_error
    try:
        full_df = pd.read_csv(SAMPLE_CSV_PATH)

        if "dt_despacho_pedido" in full_df.columns:
            full_df["dt_despacho_pedido"] = pd.to_datetime(full_df["dt_despacho_pedido"], errors="coerce")

        if "des_cd_origem" in full_df.columns:
            des_cd_origem_options = sorted(full_df["des_cd_origem"].dropna().astype(str).unique().tolist())
        else:
            des_cd_origem_options = []

        full_base_df = full_df.reset_index(drop=False).rename(columns={"index": "_source_row_id"})
        prediction_df = _load_prediction_cache(full_base_df)

        sample_df = full_base_df.merge(prediction_df, on="_source_row_id", how="left")
        sample_df["predicao"] = sample_df["predicao"].fillna("🔴 Ruim")

        sample_rows, table_columns = _build_display_data(sample_df)
        sample_error = None
    except Exception as exc:
        full_df = pd.DataFrame()
        sample_df = pd.DataFrame()
        sample_rows = []
        table_columns = []
        des_cd_origem_options = []
        sample_error = str(exc)


_load_sample_data()


@app.get("/")
def index():
    des_cd_origem = request.args.get("des_cd_origem", "").strip()
    dt_despacho_pedido = request.args.get("dt_despacho_pedido", "").strip()
    page_raw = request.args.get("page", "1").strip()

    try:
        page = max(int(page_raw), 1)
    except ValueError:
        page = 1

    rows = sample_rows
    columns = table_columns
    filter_error = None
    priority_rows = []
    priority_columns = []
    total_rows = len(rows)
    total_pages = 1

    if not des_cd_origem and des_cd_origem_options:
        des_cd_origem = des_cd_origem_options[0]

    if not dt_despacho_pedido:
        dt_despacho_pedido = DEFAULT_FILTER_DATE

    if not sample_error:
        filtered_df = _build_filtered_sample(des_cd_origem or None, dt_despacho_pedido or None)
        total_rows = len(filtered_df)
        total_pages = max(math.ceil(total_rows / DEFAULT_PAGE_SIZE), 1)
        page = min(page, total_pages)

        priority_rows, priority_columns = _build_priority_sample(filtered_df, 10)

        start_idx = (page - 1) * DEFAULT_PAGE_SIZE
        end_idx = start_idx + DEFAULT_PAGE_SIZE
        paged_df = filtered_df.iloc[start_idx:end_idx]
        rows, columns = _build_display_data(paged_df)

        if dt_despacho_pedido and _parse_filter_date(dt_despacho_pedido) is None:
            filter_error = "Use the date format YYYY-mm-dd."

    return render_template(
        "index.html",
        rows=rows,
        table_columns=columns,
        des_cd_origem_options=des_cd_origem_options,
        selected_des_cd_origem=des_cd_origem,
        selected_dt_despacho_pedido=dt_despacho_pedido,
        filter_error=filter_error,
        sample_error=sample_error,
        priority_rows=priority_rows,
        priority_columns=priority_columns,
        page=page,
        total_pages=total_pages,
        total_rows=total_rows,
        page_size=DEFAULT_PAGE_SIZE,
    )


@app.get("/insights")
def insights():
    return render_template("insights.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
