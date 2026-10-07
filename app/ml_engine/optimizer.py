import logging
from typing import Dict, Any, Optional
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, root_mean_squared_error

logger = logging.getLogger("HybridAIEngine.Optimizer")
optuna.logging.set_verbosity(optuna.logging.WARNING)

class HyperparameterOptimizer:
    """
    Automated hyperparameter tuning for LightGBM models using Optuna.
    Runs a fast search over key parameters like learning_rate, num_leaves, etc.
    """
    def __init__(self, n_trials: int = 10, timeout: int = 30):
        self.n_trials = n_trials
        self.timeout = timeout

    def optimize(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        task: str = "binary"
    ) -> Dict[str, Any]:
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "binary" if task == "binary" else "regression",
                "metric": "auc" if task == "binary" else "rmse",
                "feature_pre_filter": False,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "verbosity": -1,
                "seed": 42
            }

            booster = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )

            preds = booster.predict(X_val)
            if task == "binary":
                # Maximize AUC
                return float(roc_auc_score(y_val, preds))
            else:
                # Minimize RMSE (return negative for maximization)
                return -float(root_mean_squared_error(y_val, preds))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = study.best_params
        best_params["objective"] = "binary" if task == "binary" else "regression"
        best_params["metric"] = "auc" if task == "binary" else "rmse"
        best_params["feature_pre_filter"] = False
        best_params["verbosity"] = -1
        best_params["seed"] = 42

        logger.info(f"Optuna completed {len(study.trials)} trials. Best score: {study.best_value:.4f}")
        return best_params
