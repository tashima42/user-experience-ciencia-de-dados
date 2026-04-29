from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "precomputed_predictions.csv")

app = Flask(__name__)


# ============================================================================
# Data Loading & Processing
# ============================================================================


def load_predictions_data() -> pd.DataFrame:
    """Load precomputed predictions from CSV."""
    if not os.path.exists(PREDICTIONS_PATH):
        return pd.DataFrame()
    
    df = pd.read_csv(PREDICTIONS_PATH)
    
    # Ensure date columns are datetime
    if "dt_criacao" in df.columns:
        df["dt_criacao"] = pd.to_datetime(df["dt_criacao"], errors="coerce")
    
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
    origin: str = None
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
    """Get top N at-risk orders (lowest predicao_probabilidade)."""
    if df.empty or "predicao_probabilidade" not in df.columns:
        return []
    
    # Sort by probability (ascending) and get top rows
    top_orders = (
        df
        .nsmallest(limit, "predicao_probabilidade")
        .to_dict(orient="records")
    )
    
    return top_orders


def format_order_for_display(order: dict) -> dict:
    """Format order data for frontend display."""
    return {
        "source_row_id": order.get("_source_row_id", ""),
        "probability": float(order.get("predicao_probabilidade", 0)),
        "prediction": str(order.get("predicao", "")),
        "origin": str(order.get("des_cd_origem", "")),
        "creation_date": str(order.get("dt_criacao", ""))[:10],
        "raw_data": order,
    }


# ============================================================================
# Mock Risk Factors
# ============================================================================


def get_mock_risk_factors(order_id: int) -> list[dict]:
    """Generate mock risk factors for an order."""
    import hashlib
    
    # Use order ID to seed reproducible mock data
    seed = int(hashlib.md5(str(order_id).encode()).hexdigest(), 16) % 100
    
    factors = [
        {
            "factor": "Long shipping distance",
            "impact": f"{60 + (seed % 20)}%",
            "description": "Order destination is far from distribution center"
        },
        {
            "factor": "High carrier workload",
            "impact": f"{50 + (seed % 30)}%",
            "description": "Selected carrier has many pending deliveries"
        },
        {
            "factor": "Tight delivery window",
            "impact": f"{70 - (seed % 30)}%",
            "description": "Customer requested expedited delivery with short deadline"
        },
        {
            "factor": "Weekend delivery constraint",
            "impact": f"{40 + (seed % 20)}%",
            "description": "Delivery date falls on weekend"
        },
        {
            "factor": "Complex handling required",
            "impact": f"{45 + (seed % 25)}%",
            "description": "Order contains fragile or special items"
        },
    ]
    
    return factors[:3]  # Return top 3 factors


# ============================================================================
# Routes
# ============================================================================


@app.route("/")
def index():
    """Main dashboard page."""
    df = load_predictions_data()
    origins = get_unique_origins(df)
    
    # Default filters
    date_start = request.args.get("date", "2023-12-01")
    origin = request.args.get("origin", origins[0] if origins else None)
    
    filtered_df = filter_predictions(df, date_start=date_start, origin=origin)
    top_orders = get_top_at_risk_orders(filtered_df, limit=10)
    
    orders_display = [format_order_for_display(o) for o in top_orders]
    
    return render_template(
        "risk_dashboard.html",
        orders=orders_display,
        origins=origins,
        selected_origin=origin or (origins[0] if origins else ""),
        selected_date=date_start,
    )


@app.route("/api/orders", methods=["GET"])
def api_orders():
    """Get filtered at-risk orders."""
    df = load_predictions_data()
    
    date_start = request.args.get("date", "2023-12-01")
    origin = request.args.get("origin")
    
    filtered_df = filter_predictions(df, date_start=date_start, origin=origin)
    top_orders = get_top_at_risk_orders(filtered_df, limit=10)
    
    orders_display = [format_order_for_display(o) for o in top_orders]
    
    return jsonify({
        "success": True,
        "orders": orders_display,
        "total_filtered": len(filtered_df),
    })


@app.route("/api/risk-factors/<int:order_id>", methods=["GET"])
def api_risk_factors(order_id: int):
    """Get risk factors for an order."""
    factors = get_mock_risk_factors(order_id)
    
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
