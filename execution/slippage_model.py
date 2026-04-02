"""EXEC-003: Pre-trade Slippage Prediction Model.

Uses market microstructure features to predict expected slippage:
- Spread at order time
- Volatility (intraday + daily)
- Order size / ADV ratio
- Time of day (open/close effects)
- VIX level (regime proxy)

Used for:
1. Pre-trade cost estimation (position sizing gate)
2. Backtest realism (realistic slippage assumptions)
3. Post-trade analysis (predicted vs actual comparison)
"""

import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, time

import config

logger = logging.getLogger(__name__)


@dataclass
class SlippageFeatures:
    """Feature vector for slippage prediction."""
    spread_bps: float = 0.0       # Bid-ask spread in basis points
    volatility: float = 0.0       # Daily annualized volatility
    size_adv_ratio: float = 0.0   # order_size / ADV
    time_of_day: float = 0.0      # Minutes since market open (0-390)
    vix: float = 0.0              # VIX level
    is_opening: bool = False      # First 30 minutes
    is_closing: bool = False      # Last 30 minutes
    order_type: str = "market"    # "market" or "limit"
    side: str = "buy"             # "buy" or "sell"


@dataclass
class SlippagePrediction:
    """Slippage prediction output."""
    expected_bps: float           # Expected slippage in basis points
    confidence_low_bps: float     # 10th percentile (optimistic)
    confidence_high_bps: float    # 90th percentile (pessimistic)
    features_used: dict           # Feature values used in prediction
    model_version: str = "v1"


@dataclass
class _ModelWeights:
    """Linear model weights for slippage prediction.

    slippage_bps = intercept
                 + w_spread * spread_bps
                 + w_volatility * vol_normalized
                 + w_size * log(size_adv_ratio)
                 + w_vix * vix_normalized
                 + w_opening * is_opening
                 + w_closing * is_closing
    """
    intercept: float = 0.5
    w_spread: float = 0.3         # Wider spread -> more slippage
    w_volatility: float = 15.0    # Higher vol -> more slippage
    w_size: float = 2.0           # Larger size -> more slippage (log scale)
    w_vix: float = 0.05           # Higher VIX -> more slippage
    w_opening: float = 2.0        # Opening auction premium
    w_closing: float = 1.5        # Closing auction premium
    w_market_order: float = 1.0   # Market orders slip more than limits


class SlippageModel:
    """Pre-trade slippage prediction model.

    Predicts expected execution slippage in basis points using a feature-based
    linear model that can be retrained from actual fill data.

    Usage:
        model = SlippageModel()
        prediction = model.predict_slippage(features)
        print(f"Expected slippage: {prediction.expected_bps:.1f} bps")

        # After fills come in, retrain periodically
        model.update_model(fill_records)
    """

    # Static fallback when model has not been trained
    _FALLBACK_BPS: float = 5.0

    def __init__(self):
        self._weights = _ModelWeights()
        self._fill_history: list[dict] = []
        self._max_history = 10_000
        self._last_retrain: datetime | None = None
        self._retrain_count = 0
        self._prediction_errors: list[float] = []  # Track prediction accuracy
        self._max_errors = 1000
        self._trained = False
        self._lock = threading.Lock()
        # V11.4: Load persisted weights on startup
        self.load_weights()

    def predict_slippage(self, features: SlippageFeatures) -> SlippagePrediction:
        """Predict expected slippage for an order.

        HIGH-017: Blends theoretical (linear model) and empirical (fill history
        average) slippage predictions.  The blend weight shifts toward the
        empirical model as more fill data accumulates.

        Args:
            features: Market microstructure features at order time.

        Returns:
            SlippagePrediction with expected slippage and confidence interval.
        """
        with self._lock:
            w_snapshot = _ModelWeights(
                intercept=self._weights.intercept,
                w_spread=self._weights.w_spread,
                w_volatility=self._weights.w_volatility,
                w_size=self._weights.w_size,
                w_vix=self._weights.w_vix,
                w_opening=self._weights.w_opening,
                w_closing=self._weights.w_closing,
                w_market_order=self._weights.w_market_order,
            )
            n_fills = len(self._fill_history)
            empirical_mean = self._compute_empirical_mean() if n_fills >= 10 else None
        w = w_snapshot

        # Normalize features
        vol_normalized = features.volatility / 0.20  # Normalize to ~20% annual vol
        size_log = math.log1p(features.size_adv_ratio)  # Log-scale
        vix_normalized = features.vix / 20.0  # Normalize to ~20 VIX

        # Theoretical (linear model) prediction
        theoretical_bps = (
            w.intercept
            + w.w_spread * features.spread_bps
            + w.w_volatility * vol_normalized
            + w.w_size * size_log
            + w.w_vix * vix_normalized
            + (w.w_opening if features.is_opening else 0.0)
            + (w.w_closing if features.is_closing else 0.0)
            + (w.w_market_order if features.order_type == "market" else 0.0)
        )
        theoretical_bps = max(0.0, theoretical_bps)

        # HIGH-017: Blend theoretical and empirical predictions
        # Weight empirical more as sample count grows: w_emp = n / (n + K)
        # K=100 means at 100 fills, empirical gets 50% weight
        _BLEND_K = 100
        if empirical_mean is not None and n_fills >= 10:
            w_empirical = n_fills / (n_fills + _BLEND_K)
            w_theoretical = 1.0 - w_empirical
            pred_bps = w_theoretical * theoretical_bps + w_empirical * empirical_mean
        else:
            pred_bps = theoretical_bps

        # Floor at 0 — negative slippage (price improvement) is possible but
        # we predict the expected cost, which is non-negative
        pred_bps = max(0.0, pred_bps)

        # MED-013: Cap maximum estimated slippage to prevent extreme values
        max_slippage_bps = getattr(config, "MAX_SLIPPAGE_BPS", 50)
        pred_bps = min(pred_bps, max_slippage_bps)

        # Confidence interval: +/- based on volatility and model uncertainty
        uncertainty_mult = 1.0 + vol_normalized * 0.5
        confidence_low = max(0.0, pred_bps * 0.5 / uncertainty_mult)
        confidence_high = pred_bps * 2.0 * uncertainty_mult

        features_dict = {
            "spread_bps": features.spread_bps,
            "volatility": features.volatility,
            "size_adv_ratio": features.size_adv_ratio,
            "time_of_day_min": features.time_of_day,
            "vix": features.vix,
            "is_opening": features.is_opening,
            "is_closing": features.is_closing,
            "order_type": features.order_type,
        }

        prediction = SlippagePrediction(
            expected_bps=round(pred_bps, 2),
            confidence_low_bps=round(confidence_low, 2),
            confidence_high_bps=round(confidence_high, 2),
            features_used=features_dict,
        )

        logger.debug(
            f"SlippageModel: predicted {pred_bps:.1f}bps "
            f"[{confidence_low:.1f}, {confidence_high:.1f}] "
            f"(spread={features.spread_bps:.0f}bps, size/ADV={features.size_adv_ratio:.4f})"
        )
        return prediction

    def predict_from_order(
        self,
        order_size: int,
        price: float,
        spread_bps: float = 3.0,
        adv: float = 1_000_000,
        volatility: float = 0.25,
        vix: float = 18.0,
        order_type: str = "market",
        side: str = "buy",
        current_time: datetime | None = None,
    ) -> SlippagePrediction:
        """Convenience method: predict slippage from raw order parameters.

        Args:
            order_size: Number of shares.
            price: Expected execution price.
            spread_bps: Current bid-ask spread in basis points.
            adv: Average daily volume.
            volatility: Annualized daily volatility.
            vix: Current VIX level.
            order_type: "market" or "limit".
            side: "buy" or "sell".
            current_time: Current time (for time-of-day effects).

        Returns:
            SlippagePrediction.
        """
        if current_time is None:
            current_time = datetime.now()

        # Compute time-of-day features
        market_open = time(9, 30)
        t = current_time.time()
        minutes_since_open = max(
            0.0,
            (t.hour - market_open.hour) * 60 + (t.minute - market_open.minute),
        )
        is_opening = minutes_since_open < 30
        is_closing = minutes_since_open > 360  # Last 30 min of 390 min session

        size_adv = order_size / adv if adv > 0 else 0.0

        features = SlippageFeatures(
            spread_bps=spread_bps,
            volatility=volatility,
            size_adv_ratio=size_adv,
            time_of_day=minutes_since_open,
            vix=vix,
            is_opening=is_opening,
            is_closing=is_closing,
            order_type=order_type,
            side=side,
        )
        return self.predict_slippage(features)

    def update_model(self, fill_data: list[dict]) -> None:
        """Update model weights from actual fill data (weekly retrain).

        Each fill_data dict should contain:
            - actual_slippage_bps: float (realized slippage)
            - spread_bps: float (spread at order time)
            - volatility: float (daily vol)
            - size_adv_ratio: float (order size / ADV)
            - time_of_day: float (minutes since open)
            - vix: float (VIX level)
            - is_opening: bool
            - is_closing: bool
            - order_type: str ("market" or "limit")
        """
        if not fill_data:
            return

        with self._lock:
            self._fill_history.extend(fill_data)
            if len(self._fill_history) > self._max_history:
                self._fill_history = self._fill_history[-self._max_history:]

            # Need minimum sample for stable regression
            if len(self._fill_history) < 50:
                logger.debug(
                    f"SlippageModel: {len(self._fill_history)}/50 fills needed for retrain"
                )
                return

            # Simple gradient descent on MSE
            # (In production, replace with scikit-learn Ridge regression)
            self._retrain_sgd(self._fill_history[-2000:])
            self._retrain_count += 1
            self._last_retrain = datetime.now()
            self._trained = True

        logger.info(
            f"SlippageModel retrained (#{self._retrain_count}) "
            f"on {len(self._fill_history)} fills"
        )
        # V11.4: Persist weights after retrain
        try:
            self.save_weights()
        except Exception as e:
            logger.warning("V11.4: Failed to save slippage weights: %s", e)

    def record_prediction_error(self, predicted_bps: float, actual_bps: float) -> None:
        """Record a prediction error for model monitoring.

        Args:
            predicted_bps: What we predicted.
            actual_bps: What actually happened.
        """
        error = actual_bps - predicted_bps
        with self._lock:
            self._prediction_errors.append(error)
            if len(self._prediction_errors) > self._max_errors:
                self._prediction_errors = self._prediction_errors[-self._max_errors:]

    def get_expected_cost(
        self,
        spread_bps: float = 3.0,
        volatility: float = 0.25,
        size_adv_pct: float = 0.01,
        time_of_day_bucket: str = "mid",
        vix: float = 18.0,
    ) -> float:
        """Get total expected execution cost in basis points.

        Combines the model's slippage prediction with the half-spread cost.
        If the model has not been trained yet, returns a static fallback of
        5 bps.

        Args:
            spread_bps: Current bid-ask spread in basis points.
            volatility: Annualized daily volatility.
            size_adv_pct: Order size as a percentage of ADV (e.g. 0.01 = 1%).
            time_of_day_bucket: One of "open", "mid", "close".
            vix: Current VIX level.

        Returns:
            Total expected cost in basis points.
        """
        with self._lock:
            trained = self._trained

        if not trained:
            logger.debug(
                "SlippageModel not trained yet, returning fallback %.1f bps",
                self._FALLBACK_BPS,
            )
            return self._FALLBACK_BPS

        # Map time-of-day bucket to minutes since open
        tod_map = {"open": 15.0, "mid": 195.0, "close": 375.0}
        tod_minutes = tod_map.get(time_of_day_bucket, 195.0)

        features = SlippageFeatures(
            spread_bps=spread_bps,
            volatility=volatility,
            size_adv_ratio=size_adv_pct,
            time_of_day=tod_minutes,
            vix=vix,
            is_opening=(time_of_day_bucket == "open"),
            is_closing=(time_of_day_bucket == "close"),
            order_type="market",
        )

        prediction = self.predict_slippage(features)

        # Total cost = model slippage + half-spread crossing cost
        half_spread_bps = spread_bps / 2.0
        total_cost = prediction.expected_bps + half_spread_bps

        logger.debug(
            "Expected cost: %.1f bps (slippage=%.1f + half_spread=%.1f)",
            total_cost, prediction.expected_bps, half_spread_bps,
        )
        return round(total_cost, 2)

    def _compute_empirical_mean(self) -> float:
        """Compute the mean actual slippage from fill history.

        Must be called while holding ``self._lock``.
        """
        if not self._fill_history:
            return 0.0
        slippages = [
            max(0.0, f.get("actual_slippage_bps", 0.0))
            for f in self._fill_history[-500:]
        ]
        return sum(slippages) / len(slippages) if slippages else 0.0

    def _retrain_sgd(self, fills: list[dict]) -> None:
        """Retrain model weights using stochastic gradient descent.

        Simple online learning with L2 regularization to prevent overfitting.
        """
        w = self._weights
        lr = 0.001  # Learning rate
        l2_reg = 0.01  # L2 regularization strength
        n_epochs = 5

        for _ in range(n_epochs):
            total_loss = 0.0
            for f in fills:
                actual = f.get("actual_slippage_bps", 0)
                if actual < 0:
                    actual = 0  # Treat price improvement as 0 slippage for training

                # Compute features
                vol_norm = f.get("volatility", 0.2) / 0.20
                size_log = math.log1p(f.get("size_adv_ratio", 0))
                vix_norm = f.get("vix", 18) / 20.0
                is_open = 1.0 if f.get("is_opening", False) else 0.0
                is_close = 1.0 if f.get("is_closing", False) else 0.0
                is_market = 1.0 if f.get("order_type", "market") == "market" else 0.0

                # Predict
                pred = (
                    w.intercept
                    + w.w_spread * f.get("spread_bps", 0)
                    + w.w_volatility * vol_norm
                    + w.w_size * size_log
                    + w.w_vix * vix_norm
                    + w.w_opening * is_open
                    + w.w_closing * is_close
                    + w.w_market_order * is_market
                )

                error = pred - actual
                total_loss += error * error

                # Gradient updates with L2 regularization
                w.intercept -= lr * (error + l2_reg * w.intercept)
                w.w_spread -= lr * (error * f.get("spread_bps", 0) + l2_reg * w.w_spread)
                w.w_volatility -= lr * (error * vol_norm + l2_reg * w.w_volatility)
                w.w_size -= lr * (error * size_log + l2_reg * w.w_size)
                w.w_vix -= lr * (error * vix_norm + l2_reg * w.w_vix)
                w.w_opening -= lr * (error * is_open + l2_reg * w.w_opening)
                w.w_closing -= lr * (error * is_close + l2_reg * w.w_closing)
                w.w_market_order -= lr * (error * is_market + l2_reg * w.w_market_order)

            avg_loss = total_loss / len(fills) if fills else 0
            logger.debug(f"SlippageModel SGD epoch loss: {avg_loss:.4f}")

    def save_weights(self, path: str = None) -> None:
        """V11.4: Persist learned weights to disk so they survive restarts."""
        import json
        import os
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "slippage_weights.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self._lock:
            data = {
                "intercept": self._weights.intercept,
                "w_spread": self._weights.w_spread,
                "w_volatility": self._weights.w_volatility,
                "w_size": self._weights.w_size,
                "w_vix": self._weights.w_vix,
                "w_opening": self._weights.w_opening,
                "w_closing": self._weights.w_closing,
                "w_market_order": self._weights.w_market_order,
                "retrain_count": self._retrain_count,
                "trained": self._trained,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("V11.4: Slippage weights saved to %s", path)

    def load_weights(self, path: str = None) -> bool:
        """V11.4: Load persisted weights from disk."""
        import json
        import os
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "slippage_weights.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            with self._lock:
                self._weights.intercept = data.get("intercept", self._weights.intercept)
                self._weights.w_spread = data.get("w_spread", self._weights.w_spread)
                self._weights.w_volatility = data.get("w_volatility", self._weights.w_volatility)
                self._weights.w_size = data.get("w_size", self._weights.w_size)
                self._weights.w_vix = data.get("w_vix", self._weights.w_vix)
                self._weights.w_opening = data.get("w_opening", self._weights.w_opening)
                self._weights.w_closing = data.get("w_closing", self._weights.w_closing)
                self._weights.w_market_order = data.get("w_market_order", self._weights.w_market_order)
                self._retrain_count = data.get("retrain_count", 0)
                self._trained = data.get("trained", False)
            logger.info("V11.4: Slippage weights loaded from %s (retrain_count=%d)", path, self._retrain_count)
            return True
        except Exception as e:
            logger.warning("V11.4: Failed to load slippage weights: %s", e)
            return False

    @property
    def stats(self) -> dict:
        """Model statistics and performance metrics."""
        errors = self._prediction_errors
        mae = sum(abs(e) for e in errors) / len(errors) if errors else 0
        bias = sum(errors) / len(errors) if errors else 0

        return {
            "retrain_count": self._retrain_count,
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "fill_history_size": len(self._fill_history),
            "prediction_errors_tracked": len(errors),
            "mean_absolute_error_bps": round(mae, 2),
            "bias_bps": round(bias, 2),  # Positive = overpredicting
            "weights": {
                "intercept": round(self._weights.intercept, 4),
                "spread": round(self._weights.w_spread, 4),
                "volatility": round(self._weights.w_volatility, 4),
                "size": round(self._weights.w_size, 4),
                "vix": round(self._weights.w_vix, 4),
                "opening": round(self._weights.w_opening, 4),
                "closing": round(self._weights.w_closing, 4),
                "market_order": round(self._weights.w_market_order, 4),
            },
        }
