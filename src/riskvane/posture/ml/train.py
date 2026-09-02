"""Entrypoint: riskvane-train-model [--data PATH] [--model-out PATH] [--features-out PATH]

Trains a tabular XGBoost regressor on the (features, label) corpus steps 6-7 accumulate, and
saves it in the format posture/ml/inference.py expects to load. Requires the optional `ml` extra.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from riskvane.posture.ml.inference import DEFAULT_FEATURES_PATH, DEFAULT_MODEL_PATH

_DEFAULT_DATA_PATH = Path("var/training/posture_scans.jsonl")
_DEFAULT_MIN_ROWS = 10
_DEFAULT_TEST_SIZE = 0.2


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
    x_train, x_test, y_train, y_test = train_test_split(
        rows, labels, test_size=args.test_size, random_state=0
    )

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_names)
    params = {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.1}
    booster = xgb.train(
        params, dtrain, num_boost_round=50, evals=[(dtest, "test")], verbose_eval=False
    )

    predictions = booster.predict(dtest)
    mae = mean_absolute_error(y_test, predictions)
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
