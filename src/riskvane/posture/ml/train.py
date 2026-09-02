"""Entrypoint: riskvane-train-model [--data PATH] [--model-out PATH] [--features-out PATH]

Trains a tabular XGBoost regressor on the (features, label) corpus steps 6-7 accumulate, and
saves it in the format posture/ml/inference.py expects to load. Requires the optional `ml` extra.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import xgboost as xgb

from riskvane.posture.ml.inference import DEFAULT_FEATURES_PATH, DEFAULT_MODEL_PATH

_DEFAULT_DATA_PATH = Path("var/training/posture_scans.jsonl")
_DEFAULT_MIN_ROWS = 10
_DEFAULT_TEST_SIZE = 0.2


def _train_test_split(rows, labels, test_size: float, seed: int = 0):
    """Deterministic shuffle-and-split (stdlib stand-in for sklearn's train_test_split)."""
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    n_test = max(1, round(len(indices) * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return (
        [rows[i] for i in train_idx],
        [rows[i] for i in test_idx],
        [labels[i] for i in train_idx],
        [labels[i] for i in test_idx],
    )


def _mean_absolute_error(actual, predicted) -> float:
    return sum(abs(a - p) for a, p in zip(actual, predicted, strict=True)) / len(actual)


def _load_corpus(path: Path) -> tuple[list[dict[str, float]], list[int], list[str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    feature_names = list(rows[0]["features"].keys())
    features = [row["features"] for row in rows]
    labels = [row["label"] for row in rows]
    return features, labels, feature_names


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="riskvane-train-model",
        description="Train a tabular risk model on the accumulated training-data corpus.",
    )
    parser.add_argument("--data", default=str(_DEFAULT_DATA_PATH))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--features-out", default=str(DEFAULT_FEATURES_PATH))
    parser.add_argument("--min-rows", type=int, default=_DEFAULT_MIN_ROWS)
    parser.add_argument("--test-size", type=float, default=_DEFAULT_TEST_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data_path = Path(args.data)

    features, labels, feature_names = _load_corpus(data_path)
    if len(features) < args.min_rows:
        print(
            f"Only {len(features)} training rows found at {data_path} (need at least "
            f"{args.min_rows}). Run riskvane-generate-synthetic-corpus and/or accumulate more "
            "real /scan traffic first.",
            file=sys.stderr,
        )
        return 1

    rows = [[row[name] for name in feature_names] for row in features]
    x_train, x_test, y_train, y_test = _train_test_split(rows, labels, test_size=args.test_size)

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_names)
    params = {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.1}
    booster = xgb.train(
        params, dtrain, num_boost_round=50, evals=[(dtest, "test")], verbose_eval=False
    )

    predictions = booster.predict(dtest)
    mae = _mean_absolute_error(y_test, predictions)
    print(
        f"Trained on {len(x_train)} rows, held out {len(x_test)}; test MAE={mae:.2f}"
    )

    model_out = Path(args.model_out)
    features_out = Path(args.features_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    features_out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_out))
    features_out.write_text(json.dumps(feature_names), encoding="utf-8")
    print(f"Saved model to {model_out} and feature order to {features_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
