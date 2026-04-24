from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_preparation import load_and_clean_data


@dataclass(frozen=True)
class Bundle:
    model: Any
    selected_features: List[str]
    date_cols: List[str]
    categorical_cols: List[str]
    categorical_mappings: Dict[str, Dict[str, int]]
    threshold: float

    def dump(self, path: str) -> None:
        joblib.dump(
            {
                "model": self.model,
                "selected_features": self.selected_features,
                "date_cols": self.date_cols,
                "categorical_cols": self.categorical_cols,
                "categorical_mappings": self.categorical_mappings,
                "threshold": self.threshold,
            },
            path,
        )


def _prepare_data_for_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    cols_limpeza = ["dt_despacho", "dt_entrega", "dt_pagamento", "qtd_dias_tat"]
    df = df.dropna(subset=cols_limpeza).copy()

    limite_99 = df["qtd_dias_tat"].quantile(0.99)
    df = df[df["qtd_dias_tat"] <= limite_99].copy()
    return df


def _build_categorical_mappings(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict[str, int]]:
    mappings: Dict[str, Dict[str, int]] = {}
    for c in cat_cols:
        if c not in df.columns:
            mappings[c] = {}
            continue
        series = df[c].astype("string")
        cats = series.dropna().unique().tolist()
        # stable ordering for reproducibility
        cats = sorted([str(x) for x in cats])
        mappings[c] = {k: i for i, k in enumerate(cats)}
    return mappings


def _encode_frame(
    df: pd.DataFrame,
    selected_features: List[str],
    categorical_cols: List[str],
    categorical_mappings: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    x = df.copy()
    for c in selected_features:
        if c not in x.columns:
            x[c] = np.nan
    x = x[selected_features].copy()

    for c in categorical_cols:
        if c in x.columns:
            m = categorical_mappings.get(c, {})
            x[c] = x[c].astype("string").map(m).astype("float32")

    num_cols = [c for c in selected_features if c not in set(categorical_cols)]
    for c in num_cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype("float32")
    return x


def train_and_bundle(
    parquet_path: str,
    out_path: str,
    threshold: float,
) -> Tuple[Bundle, Dict[str, float]]:
    df = load_and_clean_data(parquet_path, drop_ids=False)
    if df is None:
        raise FileNotFoundError(
            f"Arquivo não encontrado: {parquet_path}. Rode `python extract_data.py` para gerar o parquet."
        )

    df = _prepare_data_for_lgbm(df)

    # Mirror the existing LightGBM classification script's feature set
    cat_features = ["uf", "grp_transportadora", "tp_praca", "unidade_negocio", "dia_semana_despacho"]
    num_features = ["dias_aprovacao"]
    target = "is_atrasado"
    selected_features = cat_features + num_features

    df = df.sort_values("dt_criacao")
    split_idx = int(len(df) * 0.7)
    train_test_data = df.iloc[:split_idx].copy()

    X_raw = train_test_data[selected_features].copy()
    y = train_test_data[target].astype("int32")

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y, test_size=0.15, random_state=42, stratify=y
    )

    # scale_pos_weight for imbalance
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - y_train.sum())
    spw = (neg_count / pos_count) if pos_count > 0 else 1.0

    categorical_mappings = _build_categorical_mappings(X_train_raw, cat_features)
    X_train = _encode_frame(X_train_raw, selected_features, cat_features, categorical_mappings)
    X_val = _encode_frame(X_val_raw, selected_features, cat_features, categorical_mappings)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "scale_pos_weight": spw,
        "learning_rate": 0.05,
        "num_leaves": 128,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, categorical_feature=cat_features, free_raw_data=False)

    model = lgb.train(
        params,
        train_set,
        valid_sets=[train_set, val_set],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)],
    )

    bundle = Bundle(
        model=model,
        selected_features=selected_features,
        date_cols=[],
        categorical_cols=cat_features,
        categorical_mappings=categorical_mappings,
        threshold=float(threshold),
    )
    bundle.dump(out_path)

    metrics = {
        "scale_pos_weight": float(spw),
        "best_iteration": float(getattr(model, "best_iteration", np.nan)),
        "best_auc_train": float(model.best_score.get("training", {}).get("auc", np.nan)),
        "best_auc_val": float(model.best_score.get("valid_1", {}).get("auc", np.nan)),
    }
    return bundle, metrics


def main() -> int:
    p = argparse.ArgumentParser(description="Treina LightGBM (classificação) e salva model_bundle.joblib.")
    p.add_argument("--parquet", default="pedidos_logistica.parquet", help="Caminho do arquivo parquet.")
    p.add_argument("--out", default="model_bundle.joblib", help="Saída do bundle joblib.")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold para classe prevista.")
    args = p.parse_args()

    _, metrics = train_and_bundle(args.parquet, args.out, args.threshold)
    print("\nBundle gerado com sucesso.")
    for k, v in metrics.items():
        print(f"- {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

