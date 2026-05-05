"""
Join SHAP wide outputs to precomputed predictions and add top-N SHAP features.

Run:
    python join_shap_wide.py

Custom paths and options:
    python join_shap_wide.py \
        --precomputed data/precomputed_predictions_v2.csv \
        --shap-wide shap_out/shap_wide.csv \
        --output data/precomputed_predictions_v2_with_top_shap.csv \
    --pre-key cod_pedido \
    --shap-key row_id \
        --top 4 \
        --shap-chunksize 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


DEFAULT_EXCLUDE_COLS = {
    "base_value",
    "expected_value",
    "prediction",
    "pred",
    "model_output",
    "output_value",
    "shap_base_value",
}


def _resolve_key(
    df: pd.DataFrame, key: str, allow_id_alias: bool = False
) -> pd.DataFrame:
    if allow_id_alias and key == "row_id" and key not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "row_id"})
    if key not in df.columns:
        raise ValueError(f"Missing key column: {key}")
    return df


def _select_feature_columns(
    df: pd.DataFrame, key: str, exclude: Iterable[str]
) -> List[str]:
    keys_set = {key}
    exclude_set = set(exclude)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [
        col for col in numeric_cols if col not in keys_set and col not in exclude_set
    ]
    if not feature_cols:
        raise ValueError("No numeric SHAP feature columns found.")
    return feature_cols


def _top_features(df: pd.DataFrame, feature_cols: List[str], top_n: int) -> pd.DataFrame:
    top_n = min(top_n, len(feature_cols))
    vals = df[feature_cols].abs().to_numpy()
    row_idx = np.arange(vals.shape[0])[:, None]

    # Fast top-N selection by absolute SHAP magnitude
    top_idx = np.argpartition(-vals, kth=list(range(top_n)), axis=1)[:, :top_n]
    top_vals = vals[row_idx, top_idx]
    order = np.argsort(-top_vals, axis=1)
    top_idx_sorted = top_idx[row_idx, order]

    # Clean feature names once to avoid repeating in every row
    cleaned_feature_names = np.array([c.replace("shap__", "") for c in feature_cols])
    top_feat_cleaned = cleaned_feature_names[top_idx_sorted]
    top_shap_sorted = df[feature_cols].to_numpy()[row_idx, top_idx_sorted]

    # Use list comprehensions instead of np.apply_along_axis to avoid string truncation
    # np.apply_along_axis infers return type from the first row, which leads to truncation
    joined_cols = ["|".join(row) for row in top_feat_cleaned]
    joined_vals = ["|".join([str(v) for v in row]) for row in top_shap_sorted]

    return pd.DataFrame(
        {"top_shap_columns": joined_cols, "top_shap_values": joined_vals},
        index=df.index,
    )


def _load_shap_top(
    shap_path: Path,
    shap_key: str,
    top_n: int,
    exclude_cols: Iterable[str],
    chunksize: int | None,
) -> pd.DataFrame:
    if chunksize is None:
        shap_df = pd.read_csv(shap_path)
        shap_df = _resolve_key(shap_df, shap_key)
        feature_cols = _select_feature_columns(shap_df, shap_key, exclude_cols)
        top_df = _top_features(shap_df, feature_cols, top_n)
        return pd.concat([shap_df[[shap_key]], top_df], axis=1)

    chunks = []
    for chunk in pd.read_csv(shap_path, chunksize=chunksize):
        chunk = _resolve_key(chunk, shap_key)
        feature_cols = _select_feature_columns(chunk, shap_key, exclude_cols)
        top_df = _top_features(chunk, feature_cols, top_n)
        chunks.append(pd.concat([chunk[[shap_key]], top_df], axis=1))
    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join SHAP wide outputs and add top-N SHAP features."
    )
    parser.add_argument(
        "--precomputed",
        default="data/precomputed_predictions_v2.csv",
        help="Path to precomputed predictions CSV.",
    )
    parser.add_argument(
        "--shap-wide",
        default="shap_out/shap_wide.csv",
        help="Path to SHAP wide CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/precomputed_predictions_v2_with_top_shap.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--pre-key",
        default="cod_pedido",
        help="Key column in precomputed predictions.",
    )
    parser.add_argument(
        "--shap-key",
        default="row_id",
        help="Key column in SHAP wide output.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=4,
        help="Number of top SHAP features to add per row.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=sorted(DEFAULT_EXCLUDE_COLS),
        help="Additional non-feature columns to exclude from SHAP selection.",
    )
    parser.add_argument(
        "--shap-chunksize",
        type=int,
        default=None,
        help="Optional chunk size for reading shap_wide.csv.",
    )

    args = parser.parse_args()

    precomputed_path = Path(args.precomputed)
    shap_path = Path(args.shap_wide)
    output_path = Path(args.output)

    pre_df = pd.read_csv(precomputed_path)
    pre_df = _resolve_key(pre_df, args.pre_key, allow_id_alias=True)

    shap_top = _load_shap_top(
        shap_path,
        shap_key=args.shap_key,
        top_n=args.top,
        exclude_cols=args.exclude,
        chunksize=args.shap_chunksize,
    )

    merged = pre_df.merge(
        shap_top,
        left_on=args.pre_key,
        right_on=args.shap_key,
        how="left",
    )
    if args.shap_key != args.pre_key and args.shap_key in merged.columns:
        merged = merged.drop(columns=[args.shap_key])
    merged.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
