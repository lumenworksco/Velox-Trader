"""ML-002: Model Training Pipeline.

Ensemble training with purged K-fold cross-validation, embargo periods,
and a stacking meta-learner.  Base models: LightGBM, XGBoost, Random Forest.

All ML library imports are conditional — the bot runs without them.
"""

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional ML imports
# ---------------------------------------------------------------------------

_HAS_LGBM = False
_HAS_XGB = False
_HAS_CATBOOST = False
_HAS_SKLEARN = False
_HAS_OPTUNA = False

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    lgb = None  # type: ignore[assignment]

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    xgb = None  # type: ignore[assignment]

# T5-003: CatBoost integration
try:
    import catboost as cb
    _HAS_CATBOOST = True
except ImportError:
    cb = None  # type: ignore[assignment]

# T5-003: Optuna for Bayesian hyperparameter optimization
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    optuna = None  # type: ignore[assignment]

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, f1_score, log_loss, mean_squared_error, roc_auc_score,
    )
    _HAS_SKLEARN = True
except ImportError:
    pass


def _check_ml_deps(require_all: bool = False) -> bool:
    """Check ML library availability."""
    if require_all:
        missing = []
        if not _HAS_LGBM:
            missing.append("lightgbm")
        if not _HAS_XGB:
            missing.append("xgboost")
        if not _HAS_CATBOOST:
            missing.append("catboost")
        if not _HAS_SKLEARN:
            missing.append("scikit-learn")
        if missing:
            logger.warning("Missing ML libraries: %s — install with pip", ", ".join(missing))
            return False
        return True
    return _HAS_SKLEARN


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """Metrics from model evaluation."""
    accuracy: float = 0.0
    auc_roc: float = 0.0
    f1: float = 0.0
    log_loss_val: float = 0.0
    mse: float = 0.0
    sharpe_of_predictions: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "f1": self.f1,
            "log_loss": self.log_loss_val,
            "mse": self.mse,
            "sharpe_of_predictions": self.sharpe_of_predictions,
            "n_folds": len(self.fold_metrics),
            "top_features": dict(sorted(
                self.feature_importance.items(),
                key=lambda x: abs(x[1]), reverse=True,
            )[:20]),
        }


@dataclass
class TrainedModel:
    """Container for a trained ensemble model."""
    base_models: List[Any] = field(default_factory=list)
    meta_model: Any = None
    scaler: Any = None
    feature_names: List[str] = field(default_factory=list)
    model_type: str = "classification"  # "classification" or "regression"
    version: str = ""
    trained_at: float = 0.0
    metrics: Optional[ModelMetrics] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions from the ensemble.

        Args:
            features: DataFrame with columns matching ``self.feature_names``.

        Returns:
            Array of predictions (probabilities for classification).
        """
        if not self.base_models:
            raise RuntimeError("Model has no trained base models")

        # Align features
        X = features.reindex(columns=self.feature_names, fill_value=0.0).values

        if self.scaler is not None:
            X = self.scaler.transform(X)

        if self.meta_model is not None:
            # Stacking: collect base predictions, feed to meta
            base_preds = self._base_predictions(X)
            return self.meta_model.predict(base_preds)
        else:
            # Simple averaging
            base_preds = self._base_predictions(X)
            return np.mean(base_preds, axis=1)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (classification only)."""
        if self.model_type != "classification":
            raise ValueError("predict_proba only available for classification models")

        X = features.reindex(columns=self.feature_names, fill_value=0.0).values
        if self.scaler is not None:
            X = self.scaler.transform(X)

        if self.meta_model is not None:
            base_preds = self._base_predictions_proba(X)
            if hasattr(self.meta_model, "predict_proba"):
                return self.meta_model.predict_proba(base_preds)
            return self.meta_model.predict(base_preds).reshape(-1, 1)
        else:
            preds_list = self._base_predictions_proba(X)
            return np.mean(preds_list, axis=1).reshape(-1, 1) if preds_list.ndim == 1 else preds_list

    def _base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Collect predictions from all base models."""
        preds = []
        for model in self.base_models:
            if hasattr(model, "predict"):
                preds.append(model.predict(X))
        if not preds:
            return np.zeros(X.shape[0])
        return np.column_stack(preds) if len(preds) > 1 else np.array(preds[0]).reshape(-1, 1)

    def _base_predictions_proba(self, X: np.ndarray) -> np.ndarray:
        """Collect probability predictions from all base models."""
        preds = []
        for model in self.base_models:
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)
                preds.append(p[:, 1] if p.ndim == 2 else p)
            elif hasattr(model, "predict"):
                preds.append(model.predict(X))
        if not preds:
            return np.zeros(X.shape[0])
        return np.column_stack(preds) if len(preds) > 1 else np.array(preds[0]).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Purged K-Fold
# ---------------------------------------------------------------------------

class PurgedKFold:
    """K-Fold cross-validation with purging and embargo.

    Removes overlapping samples between train and test splits to prevent
    lookahead bias, and adds an embargo period after each test fold.

    Based on Marcos Lopez de Prado's *Advances in Financial Machine Learning*.
    """

    def __init__(self, n_splits: int = 5, purge_window: int = 10, embargo: int = 5):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo = embargo

    def split(self, X: np.ndarray, y: np.ndarray = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index pairs.

        Args:
            X: Feature array (n_samples, n_features).
            y: Ignored, present for API compatibility.

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        n = len(X)
        indices = np.arange(n)
        fold_size = n // self.n_splits
        splits = []

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n

            test_idx = indices[test_start:test_end]

            # Purge: remove samples within purge_window of test boundaries
            purge_start = max(0, test_start - self.purge_window)
            purge_end = min(n, test_end + self.purge_window)

            # Embargo: extend purge after test set
            embargo_end = min(n, test_end + self.embargo)

            # Build train indices: everything outside purge+embargo zone
            train_mask = np.ones(n, dtype=bool)
            train_mask[purge_start:max(purge_end, embargo_end)] = False
            train_idx = indices[train_mask]

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits


# ---------------------------------------------------------------------------
# Model Trainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Train ensemble models with purged cross-validation.

    Usage::

        trainer = ModelTrainer()
        model = trainer.train(features_df, labels)
        metrics = trainer.evaluate(model, test_features, test_labels)
        trainer.save_model(model, "models/my_model")
    """

    # Default hyperparameters
    DEFAULT_LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_estimators": 300,
    }

    DEFAULT_XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 300,
        "verbosity": 0,
    }

    DEFAULT_RF_PARAMS = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "n_jobs": -1,
    }

    # T5-003: CatBoost default parameters
    DEFAULT_CATBOOST_PARAMS = {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "border_count": 128,
        "verbose": 0,
        "allow_writing_files": False,
        "random_seed": 42,
    }

    def __init__(
        self,
        model_type: str = "classification",
        lgbm_params: Optional[Dict] = None,
        xgb_params: Optional[Dict] = None,
        rf_params: Optional[Dict] = None,
        catboost_params: Optional[Dict] = None,
        use_stacking: bool = True,
    ):
        """Initialize the trainer.

        Args:
            model_type: ``"classification"`` or ``"regression"``.
            lgbm_params: Override default LightGBM parameters.
            xgb_params: Override default XGBoost parameters.
            rf_params: Override default Random Forest parameters.
            catboost_params: Override default CatBoost parameters.
            use_stacking: If True, train a meta-learner on base model outputs.
        """
        self.model_type = model_type
        self.use_stacking = use_stacking

        # Merge user params over defaults
        self.lgbm_params = {**self.DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
        self.xgb_params = {**self.DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.rf_params = {**self.DEFAULT_RF_PARAMS, **(rf_params or {})}
        self.catboost_params = {**self.DEFAULT_CATBOOST_PARAMS, **(catboost_params or {})}

        # Adjust for regression
        if model_type == "regression":
            self.lgbm_params["objective"] = "regression"
            self.lgbm_params["metric"] = "mse"
            self.xgb_params["objective"] = "reg:squarederror"
            self.xgb_params["eval_metric"] = "rmse"
            self.catboost_params["loss_function"] = "RMSE"

    def train(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        purge_window: int = 10,
        embargo: int = 5,
        n_splits: int = 5,
    ) -> TrainedModel:
        """Train an ensemble model with purged cross-validation.

        Args:
            features_df: Feature DataFrame (n_samples x n_features).
            labels: Target labels (binary for classification, continuous for regression).
            purge_window: Number of samples to purge around test boundaries.
            embargo: Number of samples to embargo after test set.
            n_splits: Number of CV folds.

        Returns:
            TrainedModel containing the trained ensemble.
        """
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for model training. pip install scikit-learn")

        logger.info("Starting model training: %d samples, %d features, type=%s",
                     len(features_df), len(features_df.columns), self.model_type)

        feature_names = list(features_df.columns)
        X = features_df.values.astype(np.float64)
        y = labels.values.astype(np.float64)

        # Handle NaN/Inf in features
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Purged K-Fold splits
        pkf = PurgedKFold(n_splits=n_splits, purge_window=purge_window, embargo=embargo)
        splits = pkf.split(X_scaled, y)

        if not splits:
            raise ValueError("No valid CV splits generated — dataset too small?")

        # Train base models on full data, evaluate via CV
        # HIGH-018: Track which base models succeed per fold so the stacking
        # meta-learner only receives columns that actually produced predictions.
        # T5-003: Added CatBoost as 4th ensemble member
        base_model_names = ["lgbm", "xgb", "rf", "catboost"]
        model_col_map = {"lgbm": 0, "xgb": 1, "rf": 2, "catboost": 3}
        fold_metrics = []
        oof_preds = np.zeros((len(X), 4))  # out-of-fold preds for each base model
        oof_mask = np.zeros(len(X), dtype=bool)
        # HIGH-018: Track which models produced OOF predictions across all folds
        oof_model_produced = {name: np.zeros(len(X), dtype=bool) for name in base_model_names}

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_preds = {}

            # --- LightGBM ---
            if _HAS_LGBM:
                try:
                    lgbm_model = self._train_lgbm(X_train, y_train)
                    pred = lgbm_model.predict(X_test)
                    fold_preds["lgbm"] = pred
                    oof_preds[test_idx, 0] = pred
                    oof_model_produced["lgbm"][test_idx] = True
                except Exception as e:
                    logger.warning("LightGBM fold %d failed: %s", fold_idx, e)

            # --- XGBoost ---
            if _HAS_XGB:
                try:
                    xgb_model = self._train_xgb(X_train, y_train)
                    pred = xgb_model.predict(X_test)
                    fold_preds["xgb"] = pred
                    oof_preds[test_idx, 1] = pred
                    oof_model_produced["xgb"][test_idx] = True
                except Exception as e:
                    logger.warning("XGBoost fold %d failed: %s", fold_idx, e)

            # --- Random Forest ---
            try:
                rf_model = self._train_rf(X_train, y_train)
                if self.model_type == "classification" and hasattr(rf_model, "predict_proba"):
                    pred = rf_model.predict_proba(X_test)[:, 1]
                else:
                    pred = rf_model.predict(X_test)
                fold_preds["rf"] = pred
                oof_preds[test_idx, 2] = pred
                oof_model_produced["rf"][test_idx] = True
            except Exception as e:
                logger.warning("Random Forest fold %d failed: %s", fold_idx, e)

            # --- T5-003: CatBoost ---
            if _HAS_CATBOOST:
                try:
                    cb_model = self._train_catboost(X_train, y_train)
                    if self.model_type == "classification" and hasattr(cb_model, "predict_proba"):
                        pred = cb_model.predict_proba(X_test)[:, 1]
                    else:
                        pred = cb_model.predict(X_test).flatten()
                    fold_preds["catboost"] = pred
                    oof_preds[test_idx, 3] = pred
                    oof_model_produced["catboost"][test_idx] = True
                except Exception as e:
                    logger.warning("CatBoost fold %d failed: %s", fold_idx, e)

            oof_mask[test_idx] = True

            # Fold metrics
            if fold_preds:
                avg_pred = np.mean(list(fold_preds.values()), axis=0)
                fm = self._compute_fold_metrics(y_test, avg_pred)
                fm["fold"] = fold_idx
                fm["models_trained"] = list(fold_preds.keys())
                fold_metrics.append(fm)

            logger.info("Fold %d/%d complete — %d train, %d test, models=%s",
                        fold_idx + 1, len(splits), len(train_idx), len(test_idx),
                        list(fold_preds.keys()))

        # Train final base models on all data
        base_models = []
        importances: Dict[str, float] = {}

        if _HAS_LGBM:
            try:
                final_lgbm = self._train_lgbm(X_scaled, y)
                base_models.append(final_lgbm)
                # Feature importance
                imp = final_lgbm.feature_importances_
                for name, val in zip(feature_names, imp):
                    importances[name] = importances.get(name, 0) + float(val)
            except Exception as e:
                logger.warning("Final LightGBM training failed: %s", e)

        if _HAS_XGB:
            try:
                final_xgb = self._train_xgb(X_scaled, y)
                base_models.append(final_xgb)
                imp = final_xgb.feature_importances_
                for name, val in zip(feature_names, imp):
                    importances[name] = importances.get(name, 0) + float(val)
            except Exception as e:
                logger.warning("Final XGBoost training failed: %s", e)

        try:
            final_rf = self._train_rf(X_scaled, y)
            base_models.append(final_rf)
            imp = final_rf.feature_importances_
            for name, val in zip(feature_names, imp):
                importances[name] = importances.get(name, 0) + float(val)
        except Exception as e:
            logger.warning("Final Random Forest training failed: %s", e)

        # T5-003: CatBoost final model
        if _HAS_CATBOOST:
            try:
                final_cb = self._train_catboost(X_scaled, y)
                base_models.append(final_cb)
                imp = final_cb.get_feature_importance()
                for name, val in zip(feature_names, imp):
                    importances[name] = importances.get(name, 0) + float(val)
            except Exception as e:
                logger.warning("Final CatBoost training failed: %s", e)

        # Normalize importances
        n_models = len(base_models)
        if n_models > 0:
            importances = {k: v / n_models for k, v in importances.items()}

        # --- Stacking meta-learner ---
        # HIGH-018: Only stack columns from models that actually produced
        # predictions across all folds.  A model column is included only if
        # it produced predictions for at least all OOF rows.
        meta_model = None
        if self.use_stacking and oof_mask.sum() > 10:
            try:
                # Determine which model columns have full OOF coverage
                usable_cols = []
                for name in base_model_names:
                    col_idx = model_col_map[name]
                    # Model produced predictions wherever oof_mask is True
                    if np.all(oof_model_produced[name][oof_mask]):
                        usable_cols.append(col_idx)
                    else:
                        logger.info("Stacking: excluding %s (incomplete OOF coverage)", name)

                if not usable_cols:
                    logger.warning("No base models with full OOF coverage — skipping stacking")
                else:
                    meta_X = oof_preds[oof_mask][:, usable_cols]
                    meta_y = y[oof_mask]
                    # V11.4: Clip and scale OOF predictions to prevent overflow
                    meta_X = np.clip(meta_X, -1e6, 1e6)
                    meta_X = np.nan_to_num(meta_X, nan=0.5, posinf=1.0, neginf=0.0)
                    _meta_scaler = StandardScaler()
                    meta_X = _meta_scaler.fit_transform(meta_X)
                    if self.model_type == "classification":
                        meta_model = LogisticRegression(
                            C=1.0, max_iter=1000, solver="lbfgs"
                        )
                    else:
                        meta_model = Ridge(alpha=1.0)
                    meta_model.fit(meta_X, meta_y)
                    logger.info("Stacking meta-learner trained on %d OOF samples, %d model columns",
                                len(meta_X), len(usable_cols))
            except Exception as e:
                logger.warning("Meta-learner training failed: %s", e)
                meta_model = None

        # Build version hash
        version = self._model_version(features_df, labels)

        # Aggregate metrics
        metrics = ModelMetrics(
            feature_importance=importances,
            fold_metrics=fold_metrics,
        )
        if fold_metrics:
            metrics.accuracy = float(np.mean([m.get("accuracy", 0) for m in fold_metrics]))
            metrics.auc_roc = float(np.mean([m.get("auc_roc", 0) for m in fold_metrics]))
            metrics.f1 = float(np.mean([m.get("f1", 0) for m in fold_metrics]))
            metrics.log_loss_val = float(np.mean([m.get("log_loss", 0) for m in fold_metrics]))
            metrics.mse = float(np.mean([m.get("mse", 0) for m in fold_metrics]))

        model = TrainedModel(
            base_models=base_models,
            meta_model=meta_model,
            scaler=scaler,
            feature_names=feature_names,
            model_type=self.model_type,
            version=version,
            trained_at=time.time(),
            metrics=metrics,
            config={
                "purge_window": purge_window,
                "embargo": embargo,
                "n_splits": n_splits,
                "n_samples": len(X),
                "n_features": len(feature_names),
                "base_model_count": len(base_models),
                "has_meta_learner": meta_model is not None,
            },
        )

        logger.info(
            "Training complete: %d base models, version=%s, avg AUC=%.4f",
            len(base_models), version[:12], metrics.auc_roc,
        )
        return model

    def evaluate(
        self,
        model: TrainedModel,
        test_features: pd.DataFrame,
        test_labels: pd.Series,
    ) -> ModelMetrics:
        """Evaluate a trained model on held-out test data.

        Args:
            model: Trained model to evaluate.
            test_features: Test feature DataFrame.
            test_labels: True test labels.

        Returns:
            ModelMetrics with evaluation results.
        """
        predictions = model.predict(test_features)
        y_true = test_labels.values.astype(np.float64)

        metrics = ModelMetrics()
        metrics_dict = self._compute_fold_metrics(y_true, predictions)
        metrics.accuracy = metrics_dict.get("accuracy", 0.0)
        metrics.auc_roc = metrics_dict.get("auc_roc", 0.0)
        metrics.f1 = metrics_dict.get("f1", 0.0)
        metrics.log_loss_val = metrics_dict.get("log_loss", 0.0)
        metrics.mse = metrics_dict.get("mse", 0.0)

        # Feature importance from model
        if model.metrics:
            metrics.feature_importance = model.metrics.feature_importance

        # Sharpe of predictions (signal quality proxy)
        if len(predictions) > 1:
            pred_rets = predictions * y_true  # agreement = positive
            if np.std(pred_rets) > 0:
                metrics.sharpe_of_predictions = float(
                    np.mean(pred_rets) / np.std(pred_rets) * np.sqrt(252)
                )

        return metrics

    # ------------------------------------------------------------------
    # Model serialization
    # ------------------------------------------------------------------

    def save_model(self, model: TrainedModel, path: str) -> str:
        """Save model to disk with versioning.

        Args:
            path: Directory path for model storage.

        Returns:
            Full path to the saved model file.
        """
        os.makedirs(path, exist_ok=True)
        version_str = model.version[:12] if model.version else "unknown"
        filename = f"model_{version_str}_{int(model.trained_at)}.pkl"
        filepath = os.path.join(path, filename)

        with open(filepath, "wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # Save metadata sidecar
        meta = {
            "version": model.version,
            "trained_at": model.trained_at,
            "model_type": model.model_type,
            "n_features": len(model.feature_names),
            "n_base_models": len(model.base_models),
            "has_meta_learner": model.meta_model is not None,
            "metrics": model.metrics.to_dict() if model.metrics else {},
            "config": model.config,
        }
        meta_path = filepath.replace(".pkl", "_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("Model saved: %s (%.1f KB)", filepath,
                     os.path.getsize(filepath) / 1024)
        return filepath

    @staticmethod
    def load_model(filepath: str) -> TrainedModel:
        """Load a saved model from disk.

        Args:
            filepath: Path to the .pkl model file.

        Returns:
            TrainedModel instance.
        """
        with open(filepath, "rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, TrainedModel):
            raise TypeError(f"Expected TrainedModel, got {type(model).__name__}")
        logger.info("Model loaded: %s (version=%s)", filepath, model.version[:12])
        return model

    # ------------------------------------------------------------------
    # Internal training methods
    # ------------------------------------------------------------------

    def _train_lgbm(self, X: np.ndarray, y: np.ndarray):
        """Train a LightGBM model."""
        params = dict(self.lgbm_params)
        n_est = params.pop("n_estimators", 300)
        if self.model_type == "classification":
            model = lgb.LGBMClassifier(n_estimators=n_est, **params)
        else:
            model = lgb.LGBMRegressor(n_estimators=n_est, **params)
        model.fit(X, y)
        return model

    def _train_xgb(self, X: np.ndarray, y: np.ndarray):
        """Train an XGBoost model."""
        params = dict(self.xgb_params)
        n_est = params.pop("n_estimators", 300)
        if self.model_type == "classification":
            model = xgb.XGBClassifier(n_estimators=n_est, **params)
        else:
            model = xgb.XGBRegressor(n_estimators=n_est, **params)
        model.fit(X, y)
        return model

    def _train_rf(self, X: np.ndarray, y: np.ndarray):
        """Train a Random Forest model."""
        params = dict(self.rf_params)
        if self.model_type == "classification":
            model = RandomForestClassifier(**params)
        else:
            model = RandomForestRegressor(**params)
        model.fit(X, y)
        return model

    def _train_catboost(self, X: np.ndarray, y: np.ndarray):
        """T5-003: Train a CatBoost model."""
        params = dict(self.catboost_params)
        if self.model_type == "classification":
            model = cb.CatBoostClassifier(**params)
        else:
            model = cb.CatBoostRegressor(**params)
        model.fit(X, y, verbose=False)
        return model

    # ------------------------------------------------------------------
    # T5-003: Bayesian Hyperparameter Optimization (Optuna)
    # ------------------------------------------------------------------

    def optimize_hyperparameters(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        n_trials: int = 50,
        purge_window: int = 10,
        embargo: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        """Run Bayesian hyperparameter optimization using Optuna.

        Optimizes LightGBM, XGBoost, and CatBoost hyperparameters using
        purged cross-validation to prevent lookahead bias.

        Args:
            features_df: Feature DataFrame.
            labels: Target labels.
            n_trials: Number of Optuna optimization trials.
            purge_window: Purge window for CV.
            embargo: Embargo period for CV.

        Returns:
            Dict mapping model name to best hyperparameters found.
        """
        if not _HAS_OPTUNA:
            logger.warning("T5-003: optuna not installed — skipping hyperparameter optimization")
            return {}

        if not _HAS_SKLEARN:
            logger.warning("T5-003: sklearn required for optimization")
            return {}

        # V11.4: Removed ML_ENSEMBLE_ENABLED gate — optimization should always
        # run when explicitly called (the gate was for the live signal path, not training)
        logger.info("T5-003: Starting Bayesian hyperparameter optimization (%d trials)", n_trials)

        X = features_df.values.astype(np.float64)
        y = labels.values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pkf = PurgedKFold(n_splits=3, purge_window=purge_window, embargo=embargo)
        splits = pkf.split(X_scaled, y)
        if not splits:
            logger.warning("T5-003: No valid CV splits for optimization")
            return {}

        best_params: Dict[str, Dict[str, Any]] = {}

        # --- Optimize LightGBM ---
        if _HAS_LGBM:
            def _lgbm_objective(trial):
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "objective": "binary" if self.model_type == "classification" else "regression",
                    "verbose": -1,
                    "boosting_type": "gbdt",
                }
                scores = []
                for train_idx, test_idx in splits:
                    if self.model_type == "classification":
                        m = lgb.LGBMClassifier(**params)
                    else:
                        m = lgb.LGBMRegressor(**params)
                    m.fit(X_scaled[train_idx], y[train_idx])
                    pred = m.predict(X_scaled[test_idx])
                    if self.model_type == "classification":
                        pred_binary = (pred > 0.5).astype(int)
                        scores.append(accuracy_score(y[test_idx].astype(int), pred_binary))
                    else:
                        scores.append(-mean_squared_error(y[test_idx], pred))
                return float(np.mean(scores))

            try:
                study = optuna.create_study(direction="maximize")
                study.optimize(_lgbm_objective, n_trials=n_trials, show_progress_bar=False)
                best_params["lgbm"] = study.best_params
                logger.info("T5-003: LightGBM best score=%.4f", study.best_value)
            except Exception as e:
                logger.warning("T5-003: LightGBM optimization failed: %s", e)

        # --- Optimize XGBoost ---
        if _HAS_XGB:
            def _xgb_objective(trial):
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "verbosity": 0,
                }
                scores = []
                for train_idx, test_idx in splits:
                    if self.model_type == "classification":
                        params["objective"] = "binary:logistic"
                        m = xgb.XGBClassifier(**params)
                    else:
                        params["objective"] = "reg:squarederror"
                        m = xgb.XGBRegressor(**params)
                    m.fit(X_scaled[train_idx], y[train_idx])
                    pred = m.predict(X_scaled[test_idx])
                    if self.model_type == "classification":
                        pred_binary = (pred > 0.5).astype(int)
                        scores.append(accuracy_score(y[test_idx].astype(int), pred_binary))
                    else:
                        scores.append(-mean_squared_error(y[test_idx], pred))
                return float(np.mean(scores))

            try:
                study = optuna.create_study(direction="maximize")
                study.optimize(_xgb_objective, n_trials=n_trials, show_progress_bar=False)
                best_params["xgb"] = study.best_params
                logger.info("T5-003: XGBoost best score=%.4f", study.best_value)
            except Exception as e:
                logger.warning("T5-003: XGBoost optimization failed: %s", e)

        # --- Optimize CatBoost ---
        if _HAS_CATBOOST:
            def _cb_objective(trial):
                params = {
                    "depth": trial.suggest_int("depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
                    "iterations": trial.suggest_int("iterations", 100, 500),
                    "border_count": trial.suggest_int("border_count", 32, 255),
                    "verbose": 0,
                    "allow_writing_files": False,
                    "random_seed": 42,
                }
                scores = []
                for train_idx, test_idx in splits:
                    if self.model_type == "classification":
                        m = cb.CatBoostClassifier(**params)
                    else:
                        m = cb.CatBoostRegressor(**params)
                    m.fit(X_scaled[train_idx], y[train_idx], verbose=False)
                    pred = m.predict(X_scaled[test_idx])
                    if self.model_type == "classification":
                        pred = pred.flatten()
                        pred_binary = (pred > 0.5).astype(int)
                        scores.append(accuracy_score(y[test_idx].astype(int), pred_binary))
                    else:
                        scores.append(-mean_squared_error(y[test_idx], pred.flatten()))
                return float(np.mean(scores))

            try:
                study = optuna.create_study(direction="maximize")
                study.optimize(_cb_objective, n_trials=n_trials, show_progress_bar=False)
                best_params["catboost"] = study.best_params
                logger.info("T5-003: CatBoost best score=%.4f", study.best_value)
            except Exception as e:
                logger.warning("T5-003: CatBoost optimization failed: %s", e)

        logger.info("T5-003: Optimization complete. Best params for %d models.", len(best_params))
        return best_params

    def train_with_optimized_params(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        optimized_params: Dict[str, Dict[str, Any]],
        **train_kwargs,
    ) -> "TrainedModel":
        """Train using Optuna-optimized hyperparameters.

        Args:
            features_df: Feature DataFrame.
            labels: Target labels.
            optimized_params: Dict from optimize_hyperparameters().
            **train_kwargs: Passed to self.train().

        Returns:
            TrainedModel trained with optimized parameters.
        """
        if "lgbm" in optimized_params:
            self.lgbm_params.update(optimized_params["lgbm"])
        if "xgb" in optimized_params:
            self.xgb_params.update(optimized_params["xgb"])
        if "catboost" in optimized_params:
            self.catboost_params.update(optimized_params["catboost"])

        logger.info("T5-003: Training with optimized hyperparameters")
        return self.train(features_df, labels, **train_kwargs)

    def _compute_fold_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute metrics for a single fold."""
        metrics: Dict[str, float] = {}

        if self.model_type == "classification":
            y_pred_binary = (y_pred > 0.5).astype(int)
            y_true_binary = y_true.astype(int)

            try:
                metrics["accuracy"] = accuracy_score(y_true_binary, y_pred_binary)
            except Exception:
                metrics["accuracy"] = 0.0

            try:
                if len(np.unique(y_true_binary)) > 1:
                    metrics["auc_roc"] = roc_auc_score(y_true_binary, y_pred)
                else:
                    metrics["auc_roc"] = 0.5
            except Exception:
                metrics["auc_roc"] = 0.5

            try:
                metrics["f1"] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            except Exception:
                metrics["f1"] = 0.0

            try:
                y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
                metrics["log_loss"] = log_loss(y_true_binary, y_pred_clipped)
            except Exception:
                metrics["log_loss"] = 1.0
        else:
            metrics["mse"] = float(mean_squared_error(y_true, y_pred))
            metrics["accuracy"] = 0.0
            metrics["auc_roc"] = 0.0

        return metrics

    @staticmethod
    def _model_version(features_df: pd.DataFrame, labels: pd.Series) -> str:
        """Generate a deterministic version hash for the training data."""
        h = hashlib.sha256()
        h.update(str(features_df.shape).encode())
        h.update(str(list(features_df.columns)[:10]).encode())
        h.update(str(float(labels.mean())).encode())
        h.update(str(float(labels.std())).encode())
        return h.hexdigest()
