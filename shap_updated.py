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
            date_cols=list(b.get("date_cols", [])),
            categorical_cols=list(b.get("categorical_cols", [])),
            categorical_mappings=dict(b.get("categorical_mappings", {})),
            threshold=float(b.get("threshold", 0.5)),
        )


def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _safe_identifier_column(df: pd.DataFrame) -> str:
    for c in ["cod_pedido", "row_id", "_row_id", "_source_row_id", "id"]:
        if c in df.columns:
            return c
    return "__index__"


def preprocess_frame(df_raw: pd.DataFrame, bundle: Bundle) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce X (encoded) consistent with the bundle and also return a copy of the
    aligned raw frame (same column order) for reporting feature_value_raw.
    """
    x_raw = df_raw.copy()

    for c in bundle.selected_features:
        if c not in x_raw.columns:
            x_raw[c] = np.nan
    x_raw = x_raw[bundle.selected_features].copy()

    x = x_raw.copy()

    # Dates to ordinal (convert to numeric)
    for c in bundle.date_cols:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors="coerce")
            x[c] = x[c].apply(lambda d: d.toordinal() if pd.notna(d) else np.nan).astype("float32")

    # Categorical columns as pandas category type (required by LightGBM)
    for c in bundle.categorical_cols:
        if c in x.columns:
            x[c] = x[c].astype("category")

    # Numeric features
    numeric_cols = [c for c in bundle.selected_features if c not in set(bundle.date_cols) and c not in set(bundle.categorical_cols)]
    for c in numeric_cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype("float32")

    return x, x_raw


def _topk_drivers_row(feature_names: List[str], shap_row: np.ndarray, k: int) -> Dict[str, Any]:
    order_pos = np.argsort(-shap_row)
    order_neg = np.argsort(shap_row)

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
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Package 'shap' is required. Install with: pip install shap") from exc

    # For models with categorical features, use simple explainer without background
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")


def _compute_shap_values(explainer, X: pd.DataFrame) -> Tuple[np.ndarray, float]:
    shap_vals = explainer.shap_values(X)
    expected = explainer.expected_value

    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_pos = np.asarray(shap_vals[1])
    else:
        shap_pos = np.asarray(shap_vals)

    if isinstance(expected, (list, tuple, np.ndarray)) and not np.isscalar(expected):
        if len(expected) == 2:
            expected_pos = float(np.asarray(expected)[1])
        else:
            expected_pos = float(np.asarray(expected).ravel()[0])
    else:
        expected_pos = float(expected)

    return shap_pos, expected_pos


def _encode_feature_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_categorical_dtype(series):
        return series.cat.codes.replace({-1: np.nan}).astype("float64")
    return pd.to_numeric(series, errors="coerce").astype("float64")


def main() -> int:
    # Resolve bundle path (looks in current directory or script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_bundle = "model_bundle.joblib"
    default_outdir = "shap_out"

    p = argparse.ArgumentParser(
        description="Compute SHAP explanations for LightGBM model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python shap_updated.py
    (uses default: data/prepared_data_for_shap.csv from prepare_raw_data_for_shap.ipynb)
  
  python shap_updated.py --input data/my_custom_data.csv
  python shap_updated.py --input data/my_data.csv --outdir ./my_results --topk 5
        """
    )
    p.add_argument(
        "--input", 
        default="data/prepared_data_for_shap.csv",
        help="Path to input data (CSV or Parquet). Default: data/prepared_data_for_shap.csv (output from prepare_raw_data_for_shap.ipynb)"
    )
    p.add_argument(
        "--bundle", 
        default=default_bundle, 
        help=f"Path to model bundle joblib (default: model_bundle.joblib)."
    )
    p.add_argument(
        "--outdir", 
        default=default_outdir, 
        help=f"Output directory (default: shap_out)."
    )
    p.add_argument(
        "--topk", 
        type=int, 
        default=3, 
        help="Top-k positive/negative drivers to export (default: 3)."
    )
    p.add_argument(
        "--background", 
        type=int, 
        default=200, 
        help="Rows to use as SHAP background dataset sampled (default: 200)."
    )
    args = p.parse_args()

    print(f"[INFO] Loading model bundle from: {args.bundle}")
    if not os.path.exists(args.bundle):
        print(f"[ERROR] Bundle not found at: {args.bundle}")
        print(f"[ERROR] Make sure model_bundle.joblib exists in the current directory")
        return 1

    bundle = Bundle.load(args.bundle)
    _ensure_outdir(args.outdir)

    print(f"[INFO] Loading input data from: {args.input}")
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        print(f"[ERROR] Make sure to run 'prepare_raw_data_for_shap.ipynb' first to generate this file")
        print(f"[ERROR] Or specify a different input with: --input <path_to_data>")
        return 1
    if args.input.lower().endswith(".parquet"):
        df_raw = pd.read_parquet(args.input)
    else:
        df_raw = pd.read_csv(args.input)

    print(f"[INFO] Loaded {len(df_raw)} rows")

    id_col = _safe_identifier_column(df_raw)
    if id_col == "__index__":
        df_raw = df_raw.reset_index(drop=False).rename(columns={"index": "__index__"})

    X, X_raw_aligned = preprocess_frame(df_raw, bundle)

    # Drop unsupported columns (same as training notebook does)
    unsupported_cols = X.select_dtypes(include=['object', 'datetime64[ns]', 'datetimetz']).columns
    if len(unsupported_cols) > 0:
        print(f"[INFO] Dropping unsupported columns: {list(unsupported_cols)}")
        X = X.drop(columns=unsupported_cols, errors='ignore')
    
    # Also drop date columns (they were dropped during training)
    date_cols_to_drop = [c for c in bundle.date_cols if c in X.columns]
    if date_cols_to_drop:
        print(f"[INFO] Dropping date columns (not used in training): {date_cols_to_drop}")
        X = X.drop(columns=date_cols_to_drop, errors='ignore')

    # Predict probabilities
    try:
        proba = bundle.model.predict_proba(X)[:, 1].astype("float64")
    except Exception:
        proba = np.asarray(bundle.model.predict(X)).astype("float64")
    pred = (proba >= bundle.threshold).astype("int32")

    # Background sample for expected value stability
    bg_n = int(max(0, args.background))
    X_bg = X.sample(n=min(bg_n, len(X)), random_state=42) if bg_n > 0 else None

    print(f"[INFO] Computing SHAP values (this may take a while)...")
    explainer = _build_shap_explainer(bundle.model, X_background=X_bg)
    shap_pos, expected_value = _compute_shap_values(explainer, X)

    feature_names = list(X.columns)

    # ---- predictions.csv ----
    print(f"[INFO] Generating predictions.csv with top-{args.topk} drivers...")
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
    predictions_csv = os.path.join(args.outdir, "predictions.csv")
    pd.DataFrame(pred_rows).to_csv(predictions_csv, index=False)
    print(f"[OK] Saved: {predictions_csv}")

    # ---- shap_wide.csv ----
    print(f"[INFO] Generating shap_wide.csv...")
    shap_wide = pd.DataFrame(shap_pos, columns=[f"shap__{c}" for c in feature_names])
    shap_wide.insert(0, "row_id", df_raw[id_col].values)
    shap_wide.insert(1, "expected_value", expected_value)
    shap_wide.insert(2, "probabilidade_atraso", proba)
    shap_wide_csv = os.path.join(args.outdir, "shap_wide.csv")
    shap_wide.to_csv(shap_wide_csv, index=False)
    print(f"[OK] Saved: {shap_wide_csv}")

    # ---- shap_long.csv ----
    print(f"[INFO] Generating shap_long.csv...")
    long_parts: List[pd.DataFrame] = []
    row_ids = df_raw[id_col].values

    for j, fname in enumerate(feature_names):
        part = pd.DataFrame(
            {
                "row_id": row_ids,
                "feature": fname,
                "feature_value_raw": X_raw_aligned[fname].astype("string").fillna(pd.NA),
                "feature_value_encoded": _encode_feature_values(X[fname]),
                "shap_value": shap_pos[:, j].astype("float64"),
            }
        )
        part["direction"] = np.where(part["shap_value"] >= 0, "pos", "neg")
        long_parts.append(part)

    shap_long = pd.concat(long_parts, ignore_index=True)
    shap_long_csv = os.path.join(args.outdir, "shap_long.csv")
    shap_long.to_csv(shap_long_csv, index=False)
    print(f"[OK] Saved: {shap_long_csv}")

    print(f"\n[SUCCESS] SHAP analysis complete. Output files in: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
