"""Main model: LightGBM binary classifier for P(miss), with light Optuna tuning on validation only."""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from cre.features.build import CATEGORICAL, FEATURES
from cre.logging_utils import log_event

log = logging.getLogger("cre.miss_prob")
BASE = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "bagging_freq": 1,
    "max_cat_to_onehot": 8,
    "num_threads": 0,
}


def _ds(df: pd.DataFrame, ref=None) -> lgb.Dataset:
    return lgb.Dataset(df[FEATURES], label=df["y"], categorical_feature=CATEGORICAL, reference=ref, free_raw_data=False)


def fit(train: pd.DataFrame, val: pd.DataFrame, params: dict, seed: int = 0, max_rounds: int = 3000) -> lgb.Booster:
    p = {**BASE, **params, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}
    dtr = _ds(train)
    dva = _ds(val, ref=dtr)
    return lgb.train(p, dtr, max_rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])


def tune(train: pd.DataFrame, val: pd.DataFrame, n_trials: int = 20, subsample: int = 400_000, seed: int = 0) -> dict:
    """Light tuning: a training subsample keeps it to minutes; early stopping on validation.
    Kept light on purpose: on tabular data the gain from heavy tuning is usually small compared with
    features and framing, and every extra trial is another look at validation."""
    tr = train.sample(min(subsample, len(train)), random_state=seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        b = fit(tr, val, params, seed=seed, max_rounds=1500)
        return b.best_score["valid_0"]["binary_logloss"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    log_event(log, "tuning_done", best_logloss=round(study.best_value, 5), params=study.best_params)
    return study.best_params


def predict(boosters: list[lgb.Booster], df: pd.DataFrame) -> np.ndarray:
    """Seed ensemble: mean of per-seed probabilities."""
    return np.mean([b.predict(df[FEATURES], num_iteration=b.best_iteration) for b in boosters], axis=0)
