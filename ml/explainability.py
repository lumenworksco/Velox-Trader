"""ADVML-005: XAI Layer — Model Explainability for Trading Predictions.

Provides SHAP-based local and global explanations for ML model
predictions, with a graceful fallback to built-in feature importance
when SHAP is unavailable.

Capabilities:
    - Per-prediction feature attributions (SHAP or permutation-based)
    - Global feature importance ranking
    - Counterfactual generation ("what would need to change to flip
      the prediction?")
    - Human-readable explanation summaries for the dashboard / logs
    - Audit trail integration for regulatory compliance

Usage:
    engine = ExplainabilityEngine()
    expl = engine.explain_prediction(model, features, prediction)
    global_imp = engine.get_global_importance(model, features_df)
    cf = engine.generate_counterfactual(model, features, target=1.0)

Dependencies:
    - numpy, pandas (always available)
    - shap (optional — ``pip install shap``)

References:
    - Lundberg & Lee (2017). "A Unified Approach to Interpreting Model
      Predictions."  NeurIPS.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional SHAP import
# ---------------------------------------------------------------------------

try:
    import shap

    _SHAP_AVAILABLE = True
    logger.debug("SHAP library available (version %s).", shap.__version__)
except ImportError:
    _SHAP_AVAILABLE = False
    logger.info(
        "SHAP not installed — falling back to built-in feature importance. "
        "Install with: pip install shap"
    )

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class Explanation:
    """Container for a single-prediction explanation.

    Attributes
    ----------
    feature_contributions : dict
        ``{feature_name: contribution_value}`` — signed contribution of
        each feature to the prediction (relative to the base value).
    base_value : float
        Expected model output (average prediction over background data).
    prediction : float
        Actual model output for the explained instance.
    top_positive : list of tuple
        Top features pushing the prediction *up*, sorted descending.
    top_negative : list of tuple
        Top features pushing the prediction *down*, sorted ascending.
    method : str
        Explanation method used (``"shap"`` or ``"fallback"``).
    """

    feature_contributions: Dict[str, float]
    base_value: float
    prediction: float
    top_positive: List[tuple] = field(default_factory=list)
    top_negative: List[tuple] = field(default_factory=list)
    method: str = "shap"

    def summary(self, top_k: int = 5) -> str:
        """Human-readable summary of the explanation."""
        lines = [
            f"Prediction: {self.prediction:.4f}  (base: {self.base_value:.4f})",
            f"Method: {self.method}",
        ]
        if self.top_positive:
            lines.append("Top positive drivers:")
            for name, val in self.top_positive[:top_k]:
                lines.append(f"  + {name}: {val:+.4f}")
        if self.top_negative:
            lines.append("Top negative drivers:")
            for name, val in self.top_negative[:top_k]:
                lines.append(f"  - {name}: {val:+.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------


class ModelExplainer:
    """Unified explainability layer for tree-based and generic models.

    Parameters
    ----------
    background_samples : int
        Number of background samples for SHAP KernelExplainer (used when
        the model is not natively supported by TreeExplainer).  Default 100.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        background_samples: int = 100,
        seed: int = 42,
    ) -> None:
        self.background_samples = background_samples
        self.seed = seed
        self._shap_explainer: Optional[Any] = None
        self._background_data: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None

    # ----- public API -------------------------------------------------------

    def explain_prediction(
        self,
        model: Any,
        features: Union[np.ndarray, pd.Series, pd.DataFrame, Dict[str, float]],
        prediction: Optional[float] = None,
        feature_names: Optional[List[str]] = None,
        background: Optional[pd.DataFrame] = None,
    ) -> Explanation:
        """Explain a single prediction.

        Parameters
        ----------
        model
            Fitted model with a ``predict`` method, or a callable.
        features
            Feature values for the instance to explain.  Accepts a dict,
            pandas Series, 1-D array, or single-row DataFrame.
        prediction : float, optional
            Pre-computed prediction.  If *None*, the model is called.
        feature_names : list of str, optional
            Feature names matching the columns.  Inferred from *features*
            when possible.
        background : pd.DataFrame, optional
            Background dataset for SHAP.  Cached internally after first use.

        Returns
        -------
        Explanation
        """
        features_arr, names = self._normalise_features(features, feature_names)

        if prediction is None:
            prediction = self._predict(model, features_arr)

        if _SHAP_AVAILABLE:
            try:
                return self._explain_shap(
                    model, features_arr, prediction, names, background
                )
            except Exception as exc:
                logger.warning(
                    "SHAP explanation failed (%s) — using fallback.", exc
                )

        return self._explain_fallback(model, features_arr, prediction, names)

    def get_global_importance(
        self,
        model: Any,
        features_df: pd.DataFrame,
        method: str = "auto",
        n_repeats: int = 5,
    ) -> Dict[str, float]:
        """Compute global feature importance.

        Parameters
        ----------
        model
            Fitted model.
        features_df : pd.DataFrame
            Feature matrix (rows = samples, columns = features).
        method : str
            ``"auto"`` tries SHAP first, then built-in importance, then
            permutation.  ``"shap"``, ``"builtin"``, ``"permutation"``
            force a specific method.
        n_repeats : int
            Number of permutation repeats (only for permutation method).

        Returns
        -------
        dict
            ``{feature_name: importance_score}`` sorted descending.
        """
        importances: Optional[Dict[str, float]] = None

        if method in ("auto", "shap") and _SHAP_AVAILABLE:
            try:
                importances = self._global_shap(model, features_df)
            except Exception as exc:
                logger.debug("SHAP global importance failed: %s", exc)

        if importances is None and method in ("auto", "builtin"):
            importances = self._builtin_importance(model, features_df.columns.tolist())

        if importances is None and method in ("auto", "permutation"):
            importances = self._permutation_importance(
                model, features_df, n_repeats=n_repeats
            )

        if importances is None:
            logger.warning("All importance methods failed — returning empty dict.")
            return {}

        # Sort descending by absolute value
        sorted_imp = dict(
            sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )
        return sorted_imp

    def generate_counterfactual(
        self,
        model: Any,
        features: Union[np.ndarray, pd.Series, Dict[str, float]],
        target: float,
        feature_names: Optional[List[str]] = None,
        max_features_to_change: int = 3,
        step_size: float = 0.1,
        max_iterations: int = 200,
    ) -> Dict[str, Any]:
        """Generate a greedy counterfactual explanation.

        Answers: "What minimal feature changes would shift the prediction
        to *target*?"

        Uses a simple greedy hill-climbing approach: at each step, perturb
        the feature whose gradient (finite difference) most reduces the
        gap to the target.

        Parameters
        ----------
        model
            Fitted model with a ``predict`` method.
        features
            Original feature values.
        target : float
            Desired prediction value.
        feature_names : list of str, optional
        max_features_to_change : int
            Maximum number of features to modify.
        step_size : float
            Relative step size for finite differences and perturbations.
        max_iterations : int
            Maximum optimisation steps.

        Returns
        -------
        dict
            Keys: ``original_prediction``, ``counterfactual_prediction``,
            ``target``, ``changes`` (dict of feature deltas),
            ``success`` (bool), ``iterations``.
        """
        features_arr, names = self._normalise_features(features, feature_names)
        x = features_arr.copy().astype(np.float64).ravel()
        original_pred = self._predict(model, x.reshape(1, -1))

        # Track which features we are allowed to change (pick top by gradient)
        changed_features: Dict[int, float] = {}  # idx -> cumulative delta

        current_pred = original_pred
        for _iteration in range(max_iterations):
            gap = target - current_pred
            if abs(gap) < 1e-6:
                break

            # Compute finite-difference gradient for all features
            grads = np.zeros(len(x))
            for j in range(len(x)):
                eps = max(abs(x[j]) * step_size, 1e-8)
                x_plus = x.copy()
                x_plus[j] += eps
                pred_plus = self._predict(model, x_plus.reshape(1, -1))
                grads[j] = (pred_plus - current_pred) / eps

            # Mask features we cannot change (already at limit)
            if len(changed_features) >= max_features_to_change:
                mask = np.zeros(len(x), dtype=bool)
                for idx in changed_features:
                    mask[idx] = True
                grads[~mask] = 0.0

            # Pick the feature whose gradient best reduces the gap
            # gap > 0 means we need to increase prediction → pick positive grad
            # gap < 0 means we need to decrease prediction → pick negative grad
            alignment = grads * np.sign(gap)
            best_idx = int(np.argmax(alignment))

            if alignment[best_idx] <= 0:
                # No feature can help — stop
                break

            # Step
            perturbation = step_size * np.sign(gap) * max(abs(x[best_idx]), 1.0)
            x[best_idx] += perturbation
            changed_features.setdefault(best_idx, 0.0)
            changed_features[best_idx] += perturbation

            current_pred = self._predict(model, x.reshape(1, -1))

        # Build result
        changes: Dict[str, float] = {}
        for idx, delta in changed_features.items():
            fname = names[idx] if idx < len(names) else f"feature_{idx}"
            changes[fname] = float(delta)

        success = abs(target - current_pred) < abs(target - original_pred) * 0.1

        result = {
            "original_prediction": float(original_pred),
            "counterfactual_prediction": float(current_pred),
            "target": float(target),
            "changes": changes,
            "success": success,
            "iterations": iteration + 1 if max_iterations > 0 else 0,
        }

        logger.info(
            "Counterfactual: %.4f -> %.4f (target %.4f), %d features changed, success=%s",
            original_pred,
            current_pred,
            target,
            len(changes),
            success,
        )
        return result

    # ----- SHAP-based methods -----------------------------------------------

    def _explain_shap(
        self,
        model: Any,
        features_arr: np.ndarray,
        prediction: float,
        names: List[str],
        background: Optional[pd.DataFrame],
    ) -> Explanation:
        """SHAP-based explanation (TreeExplainer or KernelExplainer)."""
        explainer = self._get_shap_explainer(model, background)
        shap_values = explainer.shap_values(features_arr)

        # Handle multi-output (classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        shap_values = np.asarray(shap_values).ravel()

        base_value = float(
            explainer.expected_value
            if isinstance(explainer.expected_value, (int, float, np.floating))
            else explainer.expected_value[1]
            if hasattr(explainer.expected_value, "__len__")
            and len(explainer.expected_value) > 1
            else explainer.expected_value[0]
        )

        contributions = {
            names[i]: float(shap_values[i]) for i in range(len(names))
        }

        return self._build_explanation(contributions, base_value, prediction, "shap")

    def _global_shap(
        self, model: Any, features_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Global importance via mean |SHAP|."""
        explainer = self._get_shap_explainer(model, features_df)
        shap_values = explainer.shap_values(features_df.values)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        return {
            col: float(mean_abs[i])
            for i, col in enumerate(features_df.columns)
        }

    def _get_shap_explainer(
        self, model: Any, background: Optional[pd.DataFrame]
    ) -> Any:
        """Obtain or cache a SHAP explainer for the given model."""
        # Try TreeExplainer for tree-based models
        model_type = type(model).__name__
        tree_types = (
            "RandomForestClassifier",
            "RandomForestRegressor",
            "GradientBoostingClassifier",
            "GradientBoostingRegressor",
            "XGBClassifier",
            "XGBRegressor",
            "LGBMClassifier",
            "LGBMRegressor",
            "CatBoostClassifier",
            "CatBoostRegressor",
            "DecisionTreeClassifier",
            "DecisionTreeRegressor",
            "ExtraTreesClassifier",
            "ExtraTreesRegressor",
        )
        if model_type in tree_types:
            try:
                return shap.TreeExplainer(model)
            except Exception:
                pass

        # Fallback to KernelExplainer
        if background is not None:
            bg = background
            if len(bg) > self.background_samples:
                bg = bg.sample(
                    self.background_samples, random_state=self.seed
                )
            bg_arr = bg.values if isinstance(bg, pd.DataFrame) else bg
        elif self._background_data is not None:
            bg_arr = self._background_data
        else:
            raise ValueError(
                "KernelExplainer requires background data. Pass background= "
                "on first call."
            )

        predict_fn = (
            model if callable(model) and not hasattr(model, "predict")
            else model.predict
        )
        self._background_data = bg_arr
        return shap.KernelExplainer(predict_fn, bg_arr)

    # ----- Fallback methods -------------------------------------------------

    def _explain_fallback(
        self,
        model: Any,
        features_arr: np.ndarray,
        prediction: float,
        names: List[str],
    ) -> Explanation:
        """Fallback explanation using built-in feature importance or
        permutation-based approach."""
        contributions: Dict[str, float] = {}

        # Try built-in feature_importances_
        importances = self._builtin_importance(model, names)
        if importances:
            # Sign contributions by feature value relative to zero
            vals = features_arr.ravel()
            for i, name in enumerate(names):
                imp = importances.get(name, 0.0)
                sign = 1.0 if (i < len(vals) and vals[i] >= 0) else -1.0
                contributions[name] = imp * sign
        else:
            # Permutation-based single-instance attribution
            contributions = self._single_instance_permutation(
                model, features_arr, prediction, names
            )

        base_value = prediction - sum(contributions.values())
        return self._build_explanation(
            contributions, base_value, prediction, "fallback"
        )

    @staticmethod
    def _builtin_importance(
        model: Any, feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Extract feature_importances_ if the model exposes them."""
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            imp = np.asarray(imp).ravel()
            if len(imp) == len(feature_names):
                return {
                    feature_names[i]: float(imp[i]) for i in range(len(imp))
                }
        # Try coef_ for linear models
        coef = getattr(model, "coef_", None)
        if coef is not None:
            coef = np.asarray(coef).ravel()
            if len(coef) == len(feature_names):
                return {
                    feature_names[i]: float(abs(coef[i]))
                    for i in range(len(coef))
                }
        return None

    def _permutation_importance(
        self,
        model: Any,
        features_df: pd.DataFrame,
        n_repeats: int = 5,
    ) -> Dict[str, float]:
        """Global permutation importance: shuffle each feature column,
        measure prediction change."""
        rng = np.random.RandomState(self.seed)
        X = features_df.values.copy()
        base_preds = self._predict_array(model, X)
        importances: Dict[str, float] = {}

        for j, col in enumerate(features_df.columns):
            score_drops = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                rng.shuffle(X_perm[:, j])
                perm_preds = self._predict_array(model, X_perm)
                # Importance = mean absolute prediction change
                score_drops.append(float(np.mean(np.abs(perm_preds - base_preds))))
            importances[col] = float(np.mean(score_drops))

        return importances

    def _single_instance_permutation(
        self,
        model: Any,
        features_arr: np.ndarray,
        prediction: float,
        names: List[str],
        n_permutations: int = 20,
    ) -> Dict[str, float]:
        """Estimate per-feature contributions for a single instance by
        zeroing each feature."""
        x = features_arr.ravel().copy()
        contributions: Dict[str, float] = {}

        for i, name in enumerate(names):
            x_mod = x.copy()
            x_mod[i] = 0.0
            pred_without = self._predict(model, x_mod.reshape(1, -1))
            contributions[name] = prediction - pred_without

        return contributions

    # ----- helpers ----------------------------------------------------------

    @staticmethod
    def _normalise_features(
        features: Union[np.ndarray, pd.Series, pd.DataFrame, Dict[str, float]],
        feature_names: Optional[List[str]],
    ) -> tuple:
        """Return (2-D numpy array, list of names)."""
        if isinstance(features, dict):
            names = list(features.keys())
            arr = np.array(list(features.values()), dtype=np.float64).reshape(1, -1)
        elif isinstance(features, pd.Series):
            names = list(features.index)
            arr = features.values.astype(np.float64).reshape(1, -1)
        elif isinstance(features, pd.DataFrame):
            names = list(features.columns)
            arr = features.values.astype(np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        else:
            arr = np.asarray(features, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            names = feature_names or [f"feature_{i}" for i in range(arr.shape[1])]

        if feature_names is not None:
            names = list(feature_names)

        return arr, names

    @staticmethod
    def _predict(model: Any, X: np.ndarray) -> float:
        """Get a scalar prediction from the model."""
        if callable(model) and not hasattr(model, "predict"):
            pred = model(X)
        else:
            pred = model.predict(X)
        pred = np.asarray(pred).ravel()
        return float(pred[0]) if len(pred) > 0 else 0.0

    @staticmethod
    def _predict_array(model: Any, X: np.ndarray) -> np.ndarray:
        """Get predictions for multiple samples."""
        if callable(model) and not hasattr(model, "predict"):
            return np.asarray(model(X)).ravel()
        return np.asarray(model.predict(X)).ravel()

    @staticmethod
    def _build_explanation(
        contributions: Dict[str, float],
        base_value: float,
        prediction: float,
        method: str,
    ) -> Explanation:
        """Construct an Explanation with sorted top features."""
        sorted_contribs = sorted(
            contributions.items(), key=lambda kv: kv[1], reverse=True
        )
        top_positive = [(k, v) for k, v in sorted_contribs if v > 0]
        top_negative = [(k, v) for k, v in sorted_contribs if v < 0]

        return Explanation(
            feature_contributions=contributions,
            base_value=base_value,
            prediction=prediction,
            top_positive=top_positive,
            top_negative=top_negative,
            method=method,
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_explainer = ModelExplainer()


def explain_prediction(
    model: Any,
    features: Union[np.ndarray, pd.Series, pd.DataFrame, Dict[str, float]],
    prediction: Optional[float] = None,
    **kwargs,
) -> Explanation:
    """Explain a single prediction using the default explainer.

    See :meth:`ModelExplainer.explain_prediction` for full parameter docs.
    """
    return _default_explainer.explain_prediction(
        model, features, prediction, **kwargs
    )


def get_global_importance(
    model: Any,
    features_df: pd.DataFrame,
    **kwargs,
) -> Dict[str, float]:
    """Compute global feature importance using the default explainer.

    See :meth:`ModelExplainer.get_global_importance` for full parameter docs.
    """
    return _default_explainer.get_global_importance(model, features_df, **kwargs)


def generate_counterfactual(
    model: Any,
    features: Union[np.ndarray, pd.Series, Dict[str, float]],
    target: float,
    **kwargs,
) -> Dict[str, Any]:
    """Generate a counterfactual explanation using the default explainer.

    See :meth:`ModelExplainer.generate_counterfactual` for full parameter docs.
    """
    return _default_explainer.generate_counterfactual(
        model, features, target, **kwargs
    )


def store_explanation(
    explanation: Explanation,
    signal_id: str,
    audit_store: Optional[Any] = None,
) -> dict:
    """Store an explanation alongside a signal in the audit trail.

    Args:
        explanation: Explanation object to store.
        signal_id: Unique identifier for the trading signal.
        audit_store: Optional audit trail storage backend.
                     If None, just returns the serialized record.

    Returns:
        Serialized explanation record.
    """
    record = {
        "signal_id": signal_id,
        "prediction": explanation.prediction,
        "base_value": explanation.base_value,
        "method": explanation.method,
        "feature_contributions": explanation.feature_contributions,
        "top_positive": explanation.top_positive[:5],
        "top_negative": explanation.top_negative[:5],
    }

    if audit_store is not None:
        try:
            if hasattr(audit_store, "log"):
                audit_store.log("signal_explanation", record)
            elif hasattr(audit_store, "store"):
                audit_store.store(record)
        except Exception as e:
            logger.warning("Failed to store explanation in audit trail: %s", e)

    return record


# Alias for spec compatibility
ExplainabilityEngine = ModelExplainer
