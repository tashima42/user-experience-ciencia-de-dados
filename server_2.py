from __future__ import annotations

import math
import os
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request

from join_shap_wide_2 import SUM_SHAP_COLUMNS, suffix_for_sum_shap_column

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(
    BASE_DIR, "data", "precomputed_predictions_v2_with_full_shap.csv"
)
SUGGESTIONS_PATH = os.path.join(BASE_DIR, "data", "suggestions.csv")
DEFAULT_DATE_START = "2023-11-15"
DEFAULT_DATE_END = "2023-12-01"

app = Flask(__name__)


def _sigmoid(x: float) -> float:
    x = max(min(float(x), 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))


def load_predictions_data() -> pd.DataFrame:
    if not os.path.exists(PREDICTIONS_PATH):
        return pd.DataFrame()

    df = pd.read_csv(PREDICTIONS_PATH, dtype={"cod_pedido": str})

    for column_name in ("dt_criacao", "dt_despacho_pedido", "dt_previsao_entrega_cliente"):
        if column_name in df.columns:
            df[column_name] = pd.to_datetime(df[column_name], errors="coerce")

    for column_name in ("predicao_probabilidade", "probabilidade_atraso", "predicao_binaria"):
        if column_name in df.columns:
            df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

    for column_name in ("sum_shap", "sigmoid_sum_shap"):
        if column_name in df.columns:
            df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

    for col in SUM_SHAP_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_suggestions_rows() -> list[dict[str, str]]:
    """Load business suggestions from data/suggestions.csv (semicolon-separated)."""
    if not os.path.exists(SUGGESTIONS_PATH):
        return []
    try:
        raw = pd.read_csv(SUGGESTIONS_PATH, sep=";", encoding="utf-8")
    except Exception:
        return []
    raw.columns = [str(c).strip().removeprefix("\ufeff") for c in raw.columns]
    raw = raw.dropna(axis=1, how="all")
    df = raw.rename(
        columns={
            "Coluna": "column",
            "Valor (base)": "value",
            "Solução": "solution",
        }
    )
    rows: list[dict[str, str]] = []
    for _, r in df.iterrows():
        def cell(name: str) -> str:
            if name not in df.columns:
                return ""
            v = r.get(name)
            if pd.isna(v):
                return ""
            return str(v).strip()

        col, val, sol = cell("column"), cell("value"), cell("solution")
        if not col and not val and not sol:
            continue
        rows.append({"column": col, "value": val, "solution": sol})
    return rows


def _norm_lookup_token(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _value_lookup_variants(display_value: str) -> list[str]:
    """Match CSV valores to dashboard valores (ex.: 7 vs 7.0)."""
    raw = str(display_value).strip()
    keys: set[str] = {_norm_lookup_token(raw)}
    normalized_num = raw.replace(",", ".")
    try:
        f = float(normalized_num)
        keys.add(_norm_lookup_token(normalized_num))
        keys.add(_norm_lookup_token(str(f)))
        rf = round(f)
        if abs(f - rf) < 1e-9:
            keys.add(_norm_lookup_token(str(int(rf))))
    except ValueError:
        pass
    return list(keys)


_suggestion_pairs_map: dict[tuple[str, str], str] | None = None


def _suggestion_pairs_lookup() -> dict[tuple[str, str], str]:
    global _suggestion_pairs_map
    if _suggestion_pairs_map is None:
        merged: dict[tuple[str, str], str] = {}
        for r in load_suggestions_rows():
            col_n = _norm_lookup_token(r["column"])
            sol = str(r["solution"]).strip()
            for vk in _value_lookup_variants(r["value"]):
                k = (col_n, vk)
                if k not in merged or (sol and len(sol) > len(merged.get(k, ""))):
                    merged[k] = sol
        _suggestion_pairs_map = merged
    return _suggestion_pairs_map


def suggestion_for_pair(feature_label: str, value_display: str) -> str:
    """Texto de Solução do CSV quando Coluna + Valor (base) batem com o grupo."""
    lu = _suggestion_pairs_lookup()
    cn = _norm_lookup_token(feature_label)
    for vk in _value_lookup_variants(value_display):
        txt = lu.get((cn, vk))
        if txt is not None:
            return txt
    return ""


def get_unique_origins(df: pd.DataFrame) -> list[str]:
    if df.empty or "des_cd_origem" not in df.columns:
        return []
    origins = df["des_cd_origem"].dropna().unique().tolist()
    return sorted([str(o) for o in origins])


def filter_predictions(
    df: pd.DataFrame,
    date_start: str | None = None,
    origin: str | None = None,
    date_end: str | None = None,
) -> pd.DataFrame:
    result = df.copy()

    if date_start and "dt_criacao" in result.columns:
        try:
            date_obj = pd.to_datetime(date_start)
            result = result[result["dt_criacao"] >= date_obj]
        except Exception:
            pass

    if date_end and "dt_criacao" in result.columns:
        try:
            end_obj = pd.to_datetime(date_end)
            result = result[result["dt_criacao"] <= end_obj]
        except Exception:
            pass

    if origin and "des_cd_origem" in result.columns:
        result = result[result["des_cd_origem"].astype(str) == str(origin)]

    return result


def _humanize_feature_name(name: str) -> str:
    cleaned = str(name).replace("_", " ").strip()
    return cleaned.title()


def _risk_column(df: pd.DataFrame) -> str | None:
    if "probabilidade_atraso" in df.columns:
        return "probabilidade_atraso"
    if "predicao_probabilidade" in df.columns:
        return "predicao_probabilidade"
    return None


def _format_cell_value(val: Any) -> str:
    if val is None:
        return "(vazio)"
    if isinstance(val, float) and pd.isna(val):
        return "(vazio)"
    if isinstance(val, pd.Timestamp):
        try:
            return val.date().isoformat()
        except Exception:
            return str(val)
    if hasattr(val, "isoformat") and callable(getattr(val, "isoformat", None)):
        try:
            iso = val.isoformat()
            return iso[:10] if len(iso) >= 10 else iso
        except Exception:
            pass
    if isinstance(val, bool):
        return str(int(val))
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val).strip()
    return s if s else "(vazio)"


def _build_value_groups(filtered: pd.DataFrame) -> list[dict[str, Any]]:
    """
    One group per (feature column name, raw dataset value): aggregate sum of SHAP for that
    feature across orders that share the same raw value.
    """
    risk_col = _risk_column(filtered)
    # (feature, value_str) -> accumulator
    buckets: dict[tuple[str, str], dict[str, Any]] = {}

    for _, row in filtered.iterrows():
        cod = str(row.get("cod_pedido", ""))
        prob = None
        if risk_col:
            rv = row.get(risk_col)
            if pd.notna(rv):
                prob = float(rv)
        sum_full = row.get("sum_shap")
        sum_full_f = float(sum_full) if sum_full is not None and pd.notna(sum_full) else None
        sig_full = row.get("sigmoid_sum_shap")
        sig_full_f = float(sig_full) if sig_full is not None and pd.notna(sig_full) else None

        for shap_col in SUM_SHAP_COLUMNS:
            if shap_col not in row.index:
                continue
            shap_v = row[shap_col]
            if pd.isna(shap_v):
                continue
            shap_f = float(shap_v)

            if shap_col == "expected_value":
                feat = "expected_value"
                raw_val = _format_cell_value(row.get("expected_value"))
            else:
                feat = suffix_for_sum_shap_column(shap_col)
                if feat in row.index:
                    raw_val = _format_cell_value(row[feat])
                else:
                    raw_val = "(sem coluna no dataset)"

            key = (feat, raw_val)
            if key not in buckets:
                buckets[key] = {
                    "feature": feat,
                    "value": raw_val,
                    "sum_shap_part": 0.0,
                    "order_keys": [],
                    "probs": [],
                    "sig_full_vals": [],
                }
            b = buckets[key]
            b["sum_shap_part"] += shap_f
            b["order_keys"].append(
                {
                    "cod_pedido": cod,
                    "shap_part": shap_f,
                    "sigmoid_shap_part": _sigmoid(shap_f),
                    "model_probability": prob,
                    "sum_shap_order": sum_full_f,
                    "sigmoid_sum_shap_order": sig_full_f,
                    "pct_of_sum_shap": _pct_for_order(row, feat),
                }
            )
            if prob is not None:
                b["probs"].append(prob)
            if sig_full_f is not None:
                b["sig_full_vals"].append(sig_full_f)

    out: list[dict[str, Any]] = []
    for key, b in buckets.items():
        feat, raw_val = key
        sum_part = b["sum_shap_part"]
        orders = b["order_keys"]
        n = len(orders)
        mean_prob = sum(b["probs"]) / len(b["probs"]) if b["probs"] else None
        mean_sig_full = sum(b["sig_full_vals"]) / len(b["sig_full_vals"]) if b["sig_full_vals"] else None
        feat_label = _humanize_feature_name(feat)
        sug = suggestion_for_pair(feat_label, raw_val)
        out.append(
            {
                "feature": feat,
                "feature_label": feat_label,
                "value": raw_val,
                "order_count": n,
                "sum_shap_part": sum_part,
                "sigmoid_sum_shap_part": _sigmoid(sum_part),
                "mean_model_probability": mean_prob,
                "mean_sigmoid_full_order": mean_sig_full,
                "orders_preview": [o["cod_pedido"] for o in orders[:5]],
                "suggestion": sug,
                "has_suggestion": bool(sug and sug.strip()),
            }
        )

    # Most positive SHAP first (pushes log-odds / delay risk up); most negative last.
    out.sort(key=lambda r: r["sum_shap_part"], reverse=True)
    return out


def _pct_for_order(row: pd.Series, feat: str) -> float | None:
    col = f"pct_of_sum_shap__{feat}"
    if col not in row.index:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    return float(v)


@app.route("/")
def index():
    df = load_predictions_data()
    origins = get_unique_origins(df)
    date_start = request.args.get("date", DEFAULT_DATE_START)
    origin = request.args.get("origin", origins[0] if origins else None)
    date_end = request.args.get("date_end", DEFAULT_DATE_END)

    filtered = filter_predictions(df, date_start=date_start, origin=origin, date_end=date_end)

    return render_template(
        "risk_dashboard_shap_by_feature.html",
        origins=origins,
        selected_origin=origin or (origins[0] if origins else ""),
        selected_date=date_start,
        selected_date_end=date_end,
        filtered_count=len(filtered),
    )


@app.route("/api/shap-value-groups", methods=["GET"])
def api_shap_value_groups():
    df = load_predictions_data()
    date_start = request.args.get("date", DEFAULT_DATE_START)
    origin = request.args.get("origin")
    date_end = request.args.get("date_end", DEFAULT_DATE_END)
    page = request.args.get("page", "1")
    page_size = request.args.get("page_size", "30")

    try:
        page_int = max(int(page), 1)
    except ValueError:
        page_int = 1
    try:
        page_size_int = max(min(int(page_size), 200), 1)
    except ValueError:
        page_size_int = 30

    filtered = filter_predictions(df, date_start=date_start, origin=origin, date_end=date_end)
    groups = _build_value_groups(filtered)
    total = len(groups)
    total_pages = max((total + page_size_int - 1) // page_size_int, 1)
    start = (page_int - 1) * page_size_int
    page_rows = groups[start : start + page_size_int]

    return jsonify(
        {
            "success": True,
            "groups": page_rows,
            "total_groups": total,
            "total_filtered_orders": len(filtered),
            "page": page_int,
            "page_size": page_size_int,
            "total_pages": total_pages,
        }
    )


@app.route("/api/shap-value-groups/orders", methods=["GET"])
def api_shap_value_group_orders():
    df = load_predictions_data()
    date_start = request.args.get("date", DEFAULT_DATE_START)
    origin = request.args.get("origin")
    date_end = request.args.get("date_end", DEFAULT_DATE_END)
    feature = request.args.get("feature", "")
    value = request.args.get("value", "")
    page = request.args.get("page", "1")
    page_size = request.args.get("page_size", "20")

    try:
        page_int = max(int(page), 1)
    except ValueError:
        page_int = 1
    try:
        page_size_int = max(min(int(page_size), 200), 1)
    except ValueError:
        page_size_int = 20

    filtered = filter_predictions(df, date_start=date_start, origin=origin, date_end=date_end)
    all_orders = _orders_for_group(filtered, feature, value)
    if not all_orders:
        return jsonify(
            {
                "success": False,
                "error": "group_not_found",
                "orders": [],
                "page": 1,
                "total_pages": 0,
            }
        )

    group_sum = sum(o["shap_part"] for o in all_orders)
    all_orders.sort(key=lambda o: o["shap_part"], reverse=True)
    total_o = len(all_orders)
    total_pages = max((total_o + page_size_int - 1) // page_size_int, 1)
    start = (page_int - 1) * page_size_int
    slice_o = all_orders[start : start + page_size_int]

    return jsonify(
        {
            "success": True,
            "feature": feature,
            "value": value,
            "group_sum_shap_part": group_sum,
            "group_sigmoid_shap_part": _sigmoid(group_sum),
            "orders": slice_o,
            "page": page_int,
            "page_size": page_size_int,
            "total_pages": total_pages,
            "total_filtered": total_o,
        }
    )


def _orders_for_group(filtered: pd.DataFrame, feature: str, value: str) -> list[dict[str, Any]]:
    risk_col = _risk_column(filtered)
    out: list[dict[str, Any]] = []
    for _, row in filtered.iterrows():
        cod = str(row.get("cod_pedido", ""))
        prob = None
        if risk_col:
            rv = row.get(risk_col)
            if pd.notna(rv):
                prob = float(rv)
        sum_full = row.get("sum_shap")
        sum_full_f = float(sum_full) if sum_full is not None and pd.notna(sum_full) else None
        sig_full = row.get("sigmoid_sum_shap")
        sig_full_f = float(sig_full) if sig_full is not None and pd.notna(sig_full) else None

        for shap_col in SUM_SHAP_COLUMNS:
            if shap_col not in row.index:
                continue
            shap_v = row[shap_col]
            if pd.isna(shap_v):
                continue
            shap_f = float(shap_v)

            if shap_col == "expected_value":
                feat = "expected_value"
                raw_val = _format_cell_value(row.get("expected_value"))
            else:
                feat = suffix_for_sum_shap_column(shap_col)
                raw_val = (
                    _format_cell_value(row[feat])
                    if feat in row.index
                    else "(sem coluna no dataset)"
                )

            if feat == feature and raw_val == value:
                out.append(
                    {
                        "cod_pedido": cod,
                        "shap_part": shap_f,
                        "sigmoid_shap_part": _sigmoid(shap_f),
                        "pct_of_sum_shap": _pct_for_order(row, feat),
                        "sum_shap_order": sum_full_f,
                        "sigmoid_sum_shap_order": sig_full_f,
                        "model_probability": prob,
                    }
                )
    return out


@app.route("/api/suggestions", methods=["GET"])
def api_suggestions():
    q = (request.args.get("q") or "").strip().lower()
    rows = load_suggestions_rows()
    if q:
        rows = [
            r
            for r in rows
            if q in r["column"].lower()
            or q in r["value"].lower()
            or q in r["solution"].lower()
        ]
    return jsonify(
        {
            "success": True,
            "rows": rows,
            "total": len(rows),
            "source": "data/suggestions.csv",
        }
    )


@app.route("/api/origins", methods=["GET"])
def api_origins():
    df = load_predictions_data()
    origins = get_unique_origins(df)
    return jsonify({"success": True, "origins": origins})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)
