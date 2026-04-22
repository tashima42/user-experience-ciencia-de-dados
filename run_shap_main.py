"""
Run SHAP explanations for the LightGBM model bundle from branch `main`.

This script is intentionally written to be executed manually by you later.
It does NOT run anything by itself unless you call it.

Inputs:
  - model_bundle.joblib (created by trabalho.ipynb in main)
  - a CSV with raw columns (e.g. data/amostra.csv)

Outputs (CSV):
  - predictions.csv : per-row predictions + top drivers
  - shap_wide.csv   : per-row SHAP values in wide format
  - shap_long.csv   : tidy per-(row,feature) SHAP table for pivoting

Notes:
  - Preprocessing is kept consistent with server.py (preprocess_one).
  - For tree models, SHAP is most naturally additive in the model output space
    (often log-odds). We still export the predicted probability alongside.
  - Business rules (City/State concatenation and Date math) are applied natively
    before CSV export to ensure business-ready data.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Bundle:
    model: Any
    selected_features: List[str]
    date_cols: List[str]
    categorical_cols: List[str]
    categorical_mappings: Dict[str, Dict[str, int]]
    threshold: float

    @staticmethod
    def load(path: str) -> "Bundle":
        b = joblib.load(path)
        return Bundle(
            model=b["model"],
            selected_features=list(b["selected_features"]),
            date_cols=list(b["date_cols"]),
            categorical_cols=list(b["categorical_cols"]),
            categorical_mappings=dict(b["categorical_mappings"]),
            threshold=float(b["threshold"]),
        )


def preprocess_frame(df_raw: pd.DataFrame, bundle: Bundle) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce X (encoded) consistent with server.py and also return a copy of the
    aligned raw frame (same column order) for reporting feature_value_raw.
    """
    x_raw = df_raw.copy()

    # Ensure all expected columns exist
    for c in bundle.selected_features:
        if c not in x_raw.columns:
            x_raw[c] = np.nan
    x_raw = x_raw[bundle.selected_features].copy()

    x = x_raw.copy()

    # Dates to ordinal
    for c in bundle.date_cols:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors="coerce")
            x[c] = x[c].map(lambda d: d.toordinal() if pd.notna(d) else np.nan).astype("float32")

    # Categorical mapping (unknown -> NaN)
    for c in bundle.categorical_cols:
        if c in x.columns:
            m = bundle.categorical_mappings.get(c, {})
            x[c] = x[c].astype("string").map(m).astype("float32")

    # Numeric features must be numeric for LightGBM
    numeric_cols = [
        c
        for c in bundle.selected_features
        if c not in set(bundle.date_cols) and c not in set(bundle.categorical_cols)
    ]
    for c in numeric_cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype("float32")

    return x, x_raw


def _safe_identifier_column(df: pd.DataFrame) -> str:
    """
    Choose a reasonable ID column if available; fallback to index.
    """
    for c in ["cod_pedido", "row_id", "_row_id", "_source_row_id"]:
        if c in df.columns:
            return c
    return "__index__"


def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _topk_drivers_row(
    feature_names: List[str],
    shap_row: np.ndarray,
    k: int,
) -> Dict[str, Any]:
    """
    Return top-k positive and negative drivers for one row.
    """
    order_pos = np.argsort(-shap_row)  # descending
    order_neg = np.argsort(shap_row)   # ascending (most negative first)

    out: Dict[str, Any] = {}
    for i in range(k):
        if i < len(feature_names):
            j = int(order_pos[i])
            out[f"top_pos_{i+1}_feature"] = feature_names[j]
            out[f"top_pos_{i+1}_shap"] = float(shap_row[j])
        else:
            out[f"top_pos_{i+1}_feature"] = None
            out[f"top_pos_{i+1}_shap"] = None

    for i in range(k):
        if i < len(feature_names):
            j = int(order_neg[i])
            out[f"top_neg_{i+1}_feature"] = feature_names[j]
            out[f"top_neg_{i+1}_shap"] = float(shap_row[j])
        else:
            out[f"top_neg_{i+1}_feature"] = None
            out[f"top_neg_{i+1}_shap"] = None

    return out


def _build_shap_explainer(model: Any, X_background: Optional[pd.DataFrame] = None):
    """
    Build a SHAP TreeExplainer. We import shap lazily so the repo doesn't break
    if shap isn't installed; the script will error at runtime with a clear message.
    """
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Package 'shap' is required. Install with: pip install shap"
        ) from exc

    # For tree models, TreeExplainer is appropriate.
    # background data is optional; providing a sample can stabilize expected_value.
    if X_background is not None and len(X_background) > 0:
        return shap.TreeExplainer(model, data=X_background, feature_perturbation="tree_path_dependent")
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")


def _compute_shap_values(explainer, X: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """
    Compute SHAP values for the positive class in binary classification.

    Depending on SHAP version/model wrapper, return shapes can vary:
      - (n, p) array
      - list of two arrays [class0, class1]
    """
    shap_vals = explainer.shap_values(X)
    expected = explainer.expected_value

    # Handle binary classification outputs
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_pos = np.asarray(shap_vals[1])
    else:
        shap_pos = np.asarray(shap_vals)

    # expected_value can be list-like in binary
    if isinstance(expected, (list, tuple, np.ndarray)) and not np.isscalar(expected):
        if len(expected) == 2:
            expected_pos = float(np.asarray(expected)[1])
        else:
            expected_pos = float(np.asarray(expected).ravel()[0])
    else:
        expected_pos = float(expected)

    return shap_pos, expected_pos


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute SHAP explanations for main's LightGBM bundle.")
    parser.add_argument("--input", required=True, help="Path to input CSV (e.g., data/amostra.csv).")
    parser.add_argument("--bundle", default="model_bundle.joblib", help="Path to model bundle joblib.")
    parser.add_argument("--outdir", default=os.path.join("shap", "out"), help="Output directory.")
    parser.add_argument("--topk", type=int, default=3, help="Top-k positive/negative drivers to export.")
    parser.add_argument(
        "--background",
        type=int,
        default=200,
        help="Rows to use as SHAP background dataset (sampled from input).",
    )
    args = parser.parse_args()

    bundle = Bundle.load(args.bundle)
    _ensure_outdir(args.outdir)

    df_raw = pd.read_csv(args.input)

    id_col = _safe_identifier_column(df_raw)
    if id_col == "__index__":
        df_raw = df_raw.reset_index(drop=False).rename(columns={"index": "__index__"})

    X, X_raw_aligned = preprocess_frame(df_raw, bundle)

    # Predict probabilities
    proba = bundle.model.predict_proba(X)[:, 1].astype("float64")
    pred = (proba >= bundle.threshold).astype("int32")

    # Background sample for expected value stability
    bg_n = int(max(0, args.background))
    X_bg = X.sample(n=min(bg_n, len(X)), random_state=42) if bg_n > 0 else None

    explainer = _build_shap_explainer(bundle.model, X_background=X_bg)
    shap_pos, expected_value = _compute_shap_values(explainer, X)

    feature_names = list(X.columns)

    # =========================================================================
    # BUSINESS LOGIC TRANSFORMATIONS (Geographical & Date Math)
    # =========================================================================
    
    # 1. City - State Concat
    if 'cidade_destinatario' in X_raw_aligned.columns and 'uf' in df_raw.columns:
        X_raw_aligned.loc[:, 'cidade_destinatario'] = (
            X_raw_aligned['cidade_destinatario'].astype(str) + " - " + df_raw['uf'].astype(str)
        )

    # 2. Date Math
    def classificar_dias(dias_series):
        bins = [-float('inf'), 5, 10, 15, 30, 45, 60, float('inf')]
        labels = ['0-5 dias', '6-10 dias', '11-15 dias', '16-30 dias', '31-45 dias', '46-60 dias', '61+ dias']
        return pd.cut(dias_series, bins=bins, labels=labels)

    if 'dt_criacao' in df_raw.columns:
        dt_base = pd.to_datetime(df_raw['dt_criacao'], errors='coerce')

        if 'dt_pagamento_pedido' in X_raw_aligned.columns:
            dt_pag = pd.to_datetime(X_raw_aligned['dt_pagamento_pedido'], errors='coerce')
            dias_pag = (dt_pag - dt_base).dt.days
            X_raw_aligned.loc[:, 'dt_pagamento_pedido'] = classificar_dias(dias_pag).astype(str)

        if 'dt_previsao_entrega_cliente' in X_raw_aligned.columns:
            dt_prev = pd.to_datetime(X_raw_aligned['dt_previsao_entrega_cliente'], errors='coerce')
            dias_prev = (dt_prev - dt_base).dt.days
            X_raw_aligned.loc[:, 'dt_previsao_entrega_cliente'] = classificar_dias(dias_prev).astype(str)

    # 3. Rename columns so predictions, shap_wide, and shap_long use the business names
    rename_map = {
        'dt_pagamento_pedido': 'tempo_aprovacao_pagamento',
        'dt_previsao_entrega_cliente': 'prazo_prometido_cliente'
    }
    feature_names = [rename_map.get(f, f) for f in feature_names]
    X_raw_aligned = X_raw_aligned.rename(columns=rename_map)
    X = X.rename(columns=rename_map)
    
    # =========================================================================

    # ---- predictions.csv ----
    pred_rows: List[Dict[str, Any]] = []
    for i in range(len(X)):
        row_id = df_raw.iloc[i][id_col]
        drivers = _topk_drivers_row(feature_names, shap_pos[i, :], k=int(args.topk))
        pred_rows.append(
            {
                "row_id": row_id,
                "probabilidade_atraso": float(proba[i]),
                "classe_prevista": int(pred[i]),
                "expected_value": float(expected_value),
                **drivers,
            }
        )
    pd.DataFrame(pred_rows).to_csv(os.path.join(args.outdir, "predictions.csv"), index=False)

    # ---- shap_wide.csv ----
    shap_wide = pd.DataFrame(shap_pos, columns=[f"shap__{c}" for c in feature_names])
    shap_wide.insert(0, "row_id", df_raw[id_col].values)
    shap_wide.insert(1, "expected_value", expected_value)
    shap_wide.insert(2, "probabilidade_atraso", proba)
    shap_wide.to_csv(os.path.join(args.outdir, "shap_wide.csv"), index=False)

    # ---- shap_long.csv ----
    long_parts: List[pd.DataFrame] = []
    row_ids = df_raw[id_col].values

    for j, fname in enumerate(feature_names):
        part = pd.DataFrame(
            {
                "row_id": row_ids,
                "feature": fname,
                "feature_value_raw": X_raw_aligned[fname].astype("string").fillna(pd.NA),
                "feature_value_encoded": X[fname].astype("float64"),
                "shap_value": shap_pos[:, j].astype("float64"),
            }
        )
        part["direction"] = np.where(part["shap_value"] >= 0, "pos", "neg")
        long_parts.append(part)

    shap_long = pd.concat(long_parts, ignore_index=True)
    shap_long.to_csv(os.path.join(args.outdir, "shap_long.csv"), index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())