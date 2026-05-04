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
    threshold: float

    @staticmethod
    def load(path: str) -> "Bundle":
        b = joblib.load(path)
        return Bundle(
            model=b["model"],
            selected_features=list(b["selected_features"]),
            date_cols=list(b.get("date_cols", [])),
            categorical_cols=list(b.get("categorical_cols", [])),
            threshold=float(b.get("threshold", 0.5)),
        )


def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_identifier_column(df: pd.DataFrame) -> str:
    chosen = _pick_first_existing(df, ["cod_pedido", "row_id", "_row_id", "_source_row_id", "id"])
    return chosen if chosen is not None else "__index__"


def preprocess_frame(df_raw: pd.DataFrame, bundle: Bundle) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x_raw = df_raw.copy()

    for c in bundle.selected_features:
        if c not in x_raw.columns:
            x_raw[c] = np.nan
    x_raw = x_raw[bundle.selected_features].copy()

    x = x_raw.copy()

    for c in bundle.date_cols:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors="coerce")
            x[c] = x[c].map(lambda d: d.toordinal() if pd.notna(d) else np.nan).astype("float32")

    for c in bundle.categorical_cols:
        if c in x.columns:
            x[c] = x[c].astype("category")

    numeric_cols = [
        c
        for c in bundle.selected_features
        if c not in set(bundle.date_cols) and c not in set(bundle.categorical_cols)
    ]
    for c in numeric_cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype("float32")

    # Mirror inference-time constraints from training/inference scripts.
    unsupported = x.select_dtypes(include=["object", "datetime64[ns]", "datetimetz"]).columns
    if len(unsupported) > 0:
        x = x.drop(columns=list(unsupported), errors="ignore")

    date_cols_to_drop = [c for c in bundle.date_cols if c in x.columns]
    if date_cols_to_drop:
        x = x.drop(columns=date_cols_to_drop, errors="ignore")

    return x, x_raw


def _build_shap_explainer(model: Any, x_background: Optional[pd.DataFrame] = None):
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Package 'shap' is required. Install with: pip install shap") from exc

    candidates: List[Any] = [model]
    booster = getattr(model, "booster_", None)
    if booster is not None:
        candidates.append(booster)

    last_exc: Optional[Exception] = None
    for candidate in candidates:
        for use_background in (True, False):
            if use_background and (x_background is None or len(x_background) == 0):
                continue
            try:
                kwargs: Dict[str, Any] = {"feature_perturbation": "tree_path_dependent"}
                if use_background:
                    kwargs["data"] = x_background
                return shap.TreeExplainer(candidate, **kwargs)
            except Exception as exc:
                last_exc = exc
                continue

    raise RuntimeError("Unable to build SHAP TreeExplainer with current model/SHAP versions.") from last_exc


def _compute_shap_values(explainer, x_for_shap) -> Tuple[np.ndarray, float]:
    shap_vals = explainer.shap_values(x_for_shap)
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


def _topk_drivers_row(feature_names: List[str], shap_row: np.ndarray, k: int) -> Dict[str, Any]:
    order_pos = np.argsort(-shap_row)
    order_neg = np.argsort(shap_row)

    out: Dict[str, Any] = {}
    for i in range(k):
        if i < len(feature_names):
            j = int(order_pos[i])
            out[f"top_pos_{i + 1}_feature"] = feature_names[j]
            out[f"top_pos_{i + 1}_shap"] = float(shap_row[j])
        else:
            out[f"top_pos_{i + 1}_feature"] = None
            out[f"top_pos_{i + 1}_shap"] = None

    for i in range(k):
        if i < len(feature_names):
            j = int(order_neg[i])
            out[f"top_neg_{i + 1}_feature"] = feature_names[j]
            out[f"top_neg_{i + 1}_shap"] = float(shap_row[j])
        else:
            out[f"top_neg_{i + 1}_feature"] = None
            out[f"top_neg_{i + 1}_shap"] = None
    return out


def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _filter_rows_by_key(
    df_source: pd.DataFrame,
    filter_df: pd.DataFrame,
    source_key: Optional[str],
    filter_key: str,
) -> Tuple[pd.DataFrame, str]:
    resolved_source_key = source_key or _pick_first_existing(
        df_source,
        [filter_key, "cod_pedido", "row_id", "id"],
    )
    if resolved_source_key is None:
        raise ValueError("Could not infer source key column. Pass --input-key explicitly.")
    if filter_key not in filter_df.columns:
        raise ValueError(f"Filter key '{filter_key}' not found in filter CSV.")

    allowed = set(filter_df[filter_key].astype("string").dropna().tolist())
    source_key_values = df_source[resolved_source_key].astype("string")
    filtered = df_source.loc[source_key_values.isin(allowed)].copy()
    return filtered, resolved_source_key


def main() -> int:
    p = argparse.ArgumentParser(description="SHAP for model_bundle filtered by precomputed IDs.")
    p.add_argument("--input", default="data/prepared_data_for_shap.csv", help="Input CSV/Parquet with model features.")
    p.add_argument("--bundle", default="model_bundle.joblib", help="Path to model bundle joblib.")
    p.add_argument("--outdir", default="shap_out", help="Output directory.")
    p.add_argument("--topk", type=int, default=3, help="Top-k positive/negative drivers.")
    p.add_argument("--background", type=int, default=200, help="Rows sampled for SHAP background.")
    p.add_argument(
        "--filter-csv",
        default="data/precomputed_predictions_v2.csv",
        help="CSV used only to filter orders to explain.",
    )
    p.add_argument(
        "--filter-key",
        default="cod_pedido",
        help="Key column in --filter-csv used to keep matching rows from --input.",
    )
    p.add_argument(
        "--input-key",
        default=None,
        help="Optional key column in --input. If omitted, it is inferred.",
    )
    args = p.parse_args()

    if not os.path.exists(args.bundle):
        raise FileNotFoundError(f"Bundle not found: {args.bundle}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not os.path.exists(args.filter_csv):
        raise FileNotFoundError(f"Filter CSV not found: {args.filter_csv}")

    bundle = Bundle.load(args.bundle)
    _ensure_outdir(args.outdir)

    df_source = _read_table(args.input)
    filter_df = pd.read_csv(args.filter_csv)
    df_raw, source_key = _filter_rows_by_key(df_source, filter_df, args.input_key, args.filter_key)
    if len(df_raw) == 0:
        raise ValueError("Filtering produced zero rows. Check keys in input and filter CSV.")

    id_col = _safe_identifier_column(df_raw)
    if id_col == "__index__":
        df_raw = df_raw.reset_index(drop=False).rename(columns={"index": "__index__"})
        id_col = "__index__"

    x, x_raw_aligned = preprocess_frame(df_raw, bundle)

    model_feature_names = getattr(bundle.model, "feature_name_", None)
    if model_feature_names is not None:
        model_feature_names = list(model_feature_names)
        missing = [f for f in model_feature_names if f not in x.columns]
        if missing:
            raise ValueError(f"Model expects feature names not found in filtered input: {missing}")
        x = x.loc[:, model_feature_names].copy()

    # Convention aligned with precompute_predictions_v2.py:
    # predict_proba[:, 1] = probabilidade de entrega no prazo
    # probabilidade_atraso = 1 - probabilidade_no_prazo
    # Prefer DataFrame input to preserve categorical dtype and feature names (as in shap_updated.py).
    # Some LightGBM bundles may reject DataFrame due to categorical metadata mismatch; in that case,
    # fall back to a numeric matrix encoding categories as codes.
    x_for_model = x
    x_for_shap = x
    try:
        proba_no_prazo = bundle.model.predict_proba(x_for_model)[:, 1].astype("float64")
    except Exception:
        # Fallback: numeric matrix
        x_num = x.copy()
        for col in x_num.columns:
            if pd.api.types.is_categorical_dtype(x_num[col]):
                x_num[col] = x_num[col].cat.codes.replace({-1: np.nan}).astype("float32")
            else:
                x_num[col] = pd.to_numeric(x_num[col], errors="coerce").astype("float32")
        x_for_model = x_num.to_numpy()
        x_for_shap = x_for_model
        proba_no_prazo = bundle.model.predict_proba(x_for_model)[:, 1].astype("float64")

    proba_atraso = 1.0 - proba_no_prazo
    pred = (proba_atraso >= bundle.threshold).astype("int32")

    bg_n = int(max(0, args.background))
    if bg_n > 0:
        bg_df = x.sample(n=min(bg_n, len(x)), replace=False).copy()
        # Background must match the representation we pass to the explainer/model.
        if isinstance(x_for_shap, np.ndarray):
            # numeric fallback path
            bg_num = bg_df.copy()
            for col in bg_num.columns:
                if pd.api.types.is_categorical_dtype(bg_num[col]):
                    bg_num[col] = bg_num[col].cat.codes.replace({-1: np.nan}).astype("float32")
                else:
                    bg_num[col] = pd.to_numeric(bg_num[col], errors="coerce").astype("float32")
            x_bg = bg_num.to_numpy()
        else:
            # DataFrame path (keeps category dtype)
            x_bg = bg_df
    else:
        x_bg = None

    explainer = _build_shap_explainer(bundle.model, x_background=x_bg)
    shap_pos, expected_pos = _compute_shap_values(explainer, x_for_shap)

    # Convert explanation from class-1/no-prazo raw output to atraso raw output.
    # This keeps additivity in raw space:
    # expected_atraso + sum(shap_atraso_i) = - (expected_pos + sum(shap_pos_i))
    shap_atraso = -shap_pos
    expected_atraso = -expected_pos

    feature_names = list(x.columns)

    pred_rows: List[Dict[str, Any]] = []
    for i in range(len(x)):
        row_id = df_raw.iloc[i][id_col]
        drivers = _topk_drivers_row(feature_names, shap_atraso[i, :], k=int(args.topk))
        pred_rows.append(
            {
                "row_id": row_id,
                "source_key": source_key,
                "probabilidade_atraso": float(proba_atraso[i]),
                "probabilidade_no_prazo": float(proba_no_prazo[i]),
                "classe_prevista": int(pred[i]),
                "expected_value": float(expected_atraso),
                **drivers,
            }
        )
    pd.DataFrame(pred_rows).to_csv(os.path.join(args.outdir, "predictions.csv"), index=False)

    shap_wide = pd.DataFrame(shap_atraso, columns=[f"shap__{c}" for c in feature_names])
    shap_wide.insert(0, "row_id", df_raw[id_col].values)
    shap_wide.insert(1, "source_key", source_key)
    shap_wide.insert(2, "expected_value", expected_atraso)
    shap_wide.insert(3, "probabilidade_atraso", proba_atraso)
    shap_wide.insert(4, "probabilidade_no_prazo", proba_no_prazo)
    shap_wide.to_csv(os.path.join(args.outdir, "shap_wide.csv"), index=False)

    long_parts: List[pd.DataFrame] = []
    row_ids = df_raw[id_col].values
    for j, fname in enumerate(feature_names):
        part = pd.DataFrame(
            {
                "row_id": row_ids,
                "source_key": source_key,
                "feature": fname,
                "feature_value_raw": x_raw_aligned[fname].astype("string").fillna(pd.NA),
                "feature_value_encoded": _encode_feature_values(x[fname]),
                "shap_value": shap_atraso[:, j].astype("float64"),
            }
        )
        part["direction"] = np.where(part["shap_value"] >= 0, "pos", "neg")
        long_parts.append(part)
    shap_long = pd.concat(long_parts, ignore_index=True)
    shap_long.to_csv(os.path.join(args.outdir, "shap_long.csv"), index=False)

    print(f"[SUCCESS] SHAP done for {len(x)} filtered rows. Outputs in: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
