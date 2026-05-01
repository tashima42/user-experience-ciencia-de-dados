from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "precomputed_predictions_v2_with_top_shap.csv")
DEFAULT_DATE_START = "2023-11-15"
DEFAULT_DATE_END = "2023-12-01"

app = Flask(__name__)


# ============================================================================
# Data Loading & Processing
# ============================================================================


def load_predictions_data() -> pd.DataFrame:
    """Load precomputed predictions from CSV."""
    if not os.path.exists(PREDICTIONS_PATH):
        return pd.DataFrame()
    
    df = pd.read_csv(PREDICTIONS_PATH)
    
    # Ensure date columns are datetime.
    for column_name in ("dt_criacao", "dt_despacho_pedido", "dt_previsao_entrega_cliente"):
        if column_name in df.columns:
            df[column_name] = pd.to_datetime(df[column_name], errors="coerce")

    # Normalize numeric fields used by the dashboard.
    for column_name in ("predicao_probabilidade", "probabilidade_atraso", "predicao_binaria"):
        if column_name in df.columns:
            df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    
    return df


def get_unique_origins(df: pd.DataFrame) -> list[str]:
    """Get unique des_cd_origem values from data."""
    if df.empty or "des_cd_origem" not in df.columns:
        return []
    
    origins = df["des_cd_origem"].dropna().unique().tolist()
    return sorted([str(o) for o in origins])


def filter_predictions(
    df: pd.DataFrame,
    date_start: str = None,
    origin: str = None,
) -> pd.DataFrame:
    """Filter predictions by date and origin."""
    result = df.copy()
    
    # Filter by date
    if date_start and "dt_criacao" in result.columns:
        try:
            date_obj = pd.to_datetime(date_start)
            result = result[result["dt_criacao"] >= date_obj]
        except Exception:
            pass
    
    # Filter by origin
    if origin and "des_cd_origem" in result.columns:
        result = result[result["des_cd_origem"].astype(str) == str(origin)]

    return result


def get_top_at_risk_orders(
    df: pd.DataFrame, 
    limit: int = 10
) -> list[dict]:
    """Get top N at-risk orders (highest probabilidade_atraso)."""
    if df.empty:
        return []

    risk_column = "probabilidade_atraso" if "probabilidade_atraso" in df.columns else "predicao_probabilidade"
    if risk_column not in df.columns:
        return []
    
    # Sort by risk (descending) and get top rows.
    ranked_df = df.copy()
    ranked_df[risk_column] = pd.to_numeric(ranked_df[risk_column], errors="coerce")
    sorted_df = ranked_df.sort_values(by=risk_column, ascending=False, na_position="last")
    if limit is not None:
        sorted_df = sorted_df.head(limit)
    top_orders = sorted_df.to_dict(orient="records")
    
    return top_orders


def get_sorted_by_risk(df: pd.DataFrame, direction: str = "desc") -> pd.DataFrame:
    """Sort dataframe by risk probability."""
    if df.empty:
        return df

    risk_column = "probabilidade_atraso" if "probabilidade_atraso" in df.columns else "predicao_probabilidade"
    if risk_column not in df.columns:
        return df

    ranked_df = df.copy()
    ranked_df[risk_column] = pd.to_numeric(ranked_df[risk_column], errors="coerce")
    ascending = str(direction).lower() == "asc"
    return ranked_df.sort_values(by=risk_column, ascending=ascending, na_position="last")


def apply_search_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Apply simple search filter across key columns."""
    if df.empty or not query:
        return df

    lowered = str(query).lower()
    columns = ["cod_pedido", "cidade_destinatario", "grp_transportadora"]
    available = [col for col in columns if col in df.columns]
    if not available:
        return df

    mask = pd.Series(False, index=df.index)
    for column_name in available:
        mask = mask | df[column_name].astype(str).str.lower().str.contains(lowered, na=False)

    return df[mask]


def format_order_for_display(order: dict) -> dict:
    """Format order data for frontend display."""
    risk_value = order.get("probabilidade_atraso", order.get("predicao_probabilidade", 0))
    return {
        "source_row_id": order.get("_source_row_id", ""),
        "cod_pedido": str(order.get("cod_pedido", order.get("_source_row_id", ""))),
        "cidade_destinatario": str(order.get("cidade_destinatario", "")),
        "grp_transportadora": str(order.get("grp_transportadora", "")),
        "dt_previsao_entrega_cliente": str(order.get("dt_previsao_entrega_cliente", ""))[:10],
        "probability": float(risk_value or 0),
        "prediction": str(order.get("risco_semaforo", order.get("predicao", ""))),
        "origin": str(order.get("des_cd_origem", "")),
        "creation_date": str(order.get("dt_criacao", ""))[:10],
        "raw_data": order,
    }


# ============================================================================
# SHAP Risk Factors
# ============================================================================


def _humanize_feature_name(name: str) -> str:
    cleaned = str(name).replace("_", " ").strip()
    return cleaned.title()


def _format_shap_value(value: str) -> tuple[str, float | None]:
    try:
        numeric = -float(str(value))
    except (TypeError, ValueError):
        return str(value), None
    return f"{numeric:+.4f}", numeric


def build_shap_risk_factors(order: dict) -> list[dict]:
    """Build risk factors from top SHAP columns and values."""
    raw_cols = order.get("top_shap_columns", "")
    raw_vals = order.get("top_shap_values", "")
    if not raw_cols or not raw_vals:
        return []

    col_list = [c for c in str(raw_cols).split("|") if c]
    val_list = [v for v in str(raw_vals).split("|") if v]
    count = min(len(col_list), len(val_list))
    factors = []
    for idx in range(count):
        feature = col_list[idx]
        impact_text, impact_value = _format_shap_value(val_list[idx])
        factors.append(
            {
                "factor": _humanize_feature_name(feature),
                "impact": impact_text,
            }
        )

    return factors


# ============================================================================
# Routes
# ============================================================================


@app.route("/")
def index():
    """Main dashboard page."""
    df = load_predictions_data()
    origins = get_unique_origins(df)
    
    # Default filters
    date_start = request.args.get("date", DEFAULT_DATE_START)
    date_end = request.args.get("date_end", DEFAULT_DATE_END)
    origin = request.args.get("origin", origins[0] if origins else None)
    
    filtered_df = filter_predictions(df, date_start=date_start, origin=origin)
    if date_end and "dt_criacao" in filtered_df.columns:
        try:
            end_obj = pd.to_datetime(date_end)
            filtered_df = filtered_df[filtered_df["dt_criacao"] <= end_obj]
        except Exception:
            pass
    top_orders = get_top_at_risk_orders(filtered_df, limit=10)
    
    orders_display = [format_order_for_display(o) for o in top_orders]
    
    return render_template(
        "risk_dashboard.html",
        orders=orders_display,
        origins=origins,
        selected_origin=origin or (origins[0] if origins else ""),
        selected_date=date_start,
        selected_date_end=date_end,
    )


@app.route("/api/orders", methods=["GET"])
def api_orders():
    """Get filtered at-risk orders."""
    df = load_predictions_data()
    
    date_start = request.args.get("date", DEFAULT_DATE_START)
    origin = request.args.get("origin")
    include_all = request.args.get("all") in {"1", "true", "True", "yes"}
    search_query = request.args.get("q", "")
    sort_direction = request.args.get("sort", "desc")
    page = request.args.get("page", "1")
    page_size = request.args.get("page_size", "10")
    date_end = request.args.get("date_end", None if include_all else DEFAULT_DATE_END)
    
    filtered_df = filter_predictions(df, date_start=date_start, origin=origin)
    if date_end and "dt_criacao" in filtered_df.columns:
        try:
            end_obj = pd.to_datetime(date_end)
            filtered_df = filtered_df[filtered_df["dt_criacao"] <= end_obj]
        except Exception:
            pass
    filtered_df = apply_search_filter(filtered_df, search_query) if include_all else filtered_df
    if include_all:
        sorted_df = get_sorted_by_risk(filtered_df, sort_direction)
        try:
            page_int = max(int(page), 1)
        except ValueError:
            page_int = 1
        try:
            page_size_int = max(min(int(page_size), 100), 1)
        except ValueError:
            page_size_int = 10
        start_idx = (page_int - 1) * page_size_int
        end_idx = start_idx + page_size_int
        paged_df = sorted_df.iloc[start_idx:end_idx]
        top_orders = paged_df.to_dict(orient="records")
        total_filtered = len(filtered_df)
        total_pages = max((total_filtered + page_size_int - 1) // page_size_int, 1)
    else:
        top_orders = get_top_at_risk_orders(filtered_df, limit=10)
        total_filtered = len(filtered_df)
        page_int = 1
        page_size_int = 10
        total_pages = 1
    
    orders_display = [format_order_for_display(o) for o in top_orders]
    
    return jsonify({
        "success": True,
        "orders": orders_display,
        "total_filtered": total_filtered,
        "page": page_int,
        "page_size": page_size_int,
        "total_pages": total_pages,
    })


@app.route("/api/risk-factors/<order_id>", methods=["GET"])
def api_risk_factors(order_id: str):
    """Get risk factors for an order."""
    df = load_predictions_data()
    factors = []
    if not df.empty and "cod_pedido" in df.columns:
        matches = df[df["cod_pedido"].astype(str) == str(order_id)]
        if not matches.empty:
            factors = build_shap_risk_factors(matches.iloc[0].to_dict())
    
    return jsonify({
        "success": True,
        "order_id": order_id,
        "risk_factors": factors,
    })


@app.route("/api/origins", methods=["GET"])
def api_origins():
    """Get unique origins."""
    df = load_predictions_data()
    origins = get_unique_origins(df)
    
    return jsonify({
        "success": True,
        "origins": origins,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
