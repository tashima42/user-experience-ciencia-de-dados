"""
Join full SHAP wide outputs to precomputed predictions (all features, no top-N).

Adds:
  - sum_shap: sum of expected_value + listed shap__* columns (per order)
  - sigmoid_sum_shap: 1 / (1 + exp(-sum_shap))
  - pct_of_sum_shap__*: each part as % of signed sum_shap (NaN if sum_shap == 0)
  - pct_of_abs_sum_shap__*: each part as % of sum(abs(parts)) (always defined if any part non-zero)

Run:
    python join_shap_wide_2.py

Custom paths:
    python join_shap_wide_2.py \\
        --precomputed data/precomputed_predictions_v2.csv \\
        --shap-wide shap_out/shap_wide.csv \\
        --output data/precomputed_predictions_v2_with_full_shap.csv \\
        --pre-key cod_pedido \\
        --shap-key row_id \\
        --shap-chunksize 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from join_shap_wide import _resolve_key

# Columns that define SUM_SHAP (additive SHAP / expected value terms per row).
SUM_SHAP_COLUMNS: tuple[str, ...] = (
    "expected_value",
    "shap__cidade_destinatario",
    "shap__uf",
    "shap__grp_transportadora",
    "shap__tp_praca",
    "shap__des_unidade_negocio",
    "shap__des_cd_origem",
    "shap__dias_gastos_cd",
    "shap__dias_restantes_prazo",
    "shap__turno_despacho",
    "shap__dia_semana_despacho",
    "shap__mes_despacho",
    "shap__semana_ano",
    "shap__is_fds_despacho",
    "shap__is_alta_temporada",
    "shap__dias_criacao_pagamento",
    "shap__prazo_apos_despacho",
    "shap__ratio_cd_prazo",
    "shap__margem_entrega",
)


def _suffix_for_sum_col(name: str) -> str:
    if name == "expected_value":
        return "expected_value"
    if name.startswith("shap__"):
        return name[len("shap__") :]
    return name


def suffix_for_sum_shap_column(name: str) -> str:
    """Public alias for CSV suffix used in pct_of_sum_shap__* columns."""
    return _suffix_for_sum_col(name)


def _load_full_shap(
    shap_path: Path,
    shap_key: str,
    chunksize: int | None,
) -> pd.DataFrame:
    read_kwargs: dict = {"dtype": {shap_key: str}}
    if chunksize is None:
        shap_df = pd.read_csv(shap_path, **read_kwargs)
        return _resolve_key(shap_df, shap_key)

    chunks = []
    for chunk in pd.read_csv(shap_path, chunksize=chunksize, **read_kwargs):
        chunks.append(_resolve_key(chunk, shap_key))
    return pd.concat(chunks, ignore_index=True)


def _append_sum_and_shares(shap_df: pd.DataFrame) -> pd.DataFrame:
    out = shap_df.copy()
    missing = [c for c in SUM_SHAP_COLUMNS if c not in out.columns]
    if missing:
        for c in missing:
            out[c] = np.nan

    parts = out[list(SUM_SHAP_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    parts = np.nan_to_num(parts, nan=0.0)
    sum_shap = parts.sum(axis=1)
    # Excel: 1/(1+EXP(-SUM_SHAP)) — clip logit for numerical stability
    clipped = np.clip(sum_shap, -60.0, 60.0)
    sigmoid = 1.0 / (1.0 + np.exp(-clipped))

    out["sum_shap"] = sum_shap
    out["sigmoid_sum_shap"] = sigmoid

    sum_abs = np.sum(np.abs(parts), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        signed_pct = np.where(
            np.abs(sum_shap)[:, None] > 1e-12,
            (parts / sum_shap[:, None]) * 100.0,
            np.nan,
        )
        abs_pct = np.where(
            sum_abs[:, None] > 1e-12,
            (np.abs(parts) / sum_abs[:, None]) * 100.0,
            np.nan,
        )

    for j, col in enumerate(SUM_SHAP_COLUMNS):
        suf = _suffix_for_sum_col(col)
        out[f"pct_of_sum_shap__{suf}"] = signed_pct[:, j]
        out[f"pct_of_abs_sum_shap__{suf}"] = abs_pct[:, j]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join full SHAP wide CSV and add sum_shap, sigmoid, and share columns."
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
        default="data/precomputed_predictions_v2_with_full_shap.csv",
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
        "--shap-chunksize",
        type=int,
        default=None,
        help="Optional chunk size for reading shap_wide.csv.",
    )

    args = parser.parse_args()

    precomputed_path = Path(args.precomputed)
    shap_path = Path(args.shap_wide)
    output_path = Path(args.output)

    pre_df = pd.read_csv(precomputed_path, dtype={args.pre_key: str})
    pre_df = _resolve_key(pre_df, args.pre_key, allow_id_alias=True)

    shap_df = _load_full_shap(shap_path, args.shap_key, args.shap_chunksize)
    shap_df = _append_sum_and_shares(shap_df)

    merged = pre_df.merge(
        shap_df,
        left_on=args.pre_key,
        right_on=args.shap_key,
        how="left",
    )
    if args.shap_key != args.pre_key and args.shap_key in merged.columns:
        merged = merged.drop(columns=[args.shap_key])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
