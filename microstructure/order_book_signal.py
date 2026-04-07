"""T5-014: Order Book Microstructure Signal (DeepLOB-Inspired).

Computes order book features at 1-second frequency:
  - depth_imbalance = (bid_size_5 - ask_size_5) / (bid_size_5 + ask_size_5)
  - queue_depletion_rate: how fast bid queue is being consumed
  - sweep_indicator: trade size > top-of-book size

Includes OrderImbalanceModel (logistic regression on depth imbalance features)
that outputs a sizing multiplier:
  - Strong buy signal (>0.3)  -> 1.2x
  - Strong sell signal (<-0.3) -> 0.8x
  - Neutral -> 1.0x

Gated behind ``ORDER_BOOK_SIGNAL_ENABLED`` config flag.

Usage::

    extractor = OrderBookFeatureExtractor()
    extractor.update(bid_sizes=[100, 200, 150, 80, 50],
                     ask_sizes=[80, 150, 200, 120, 90],
                     last_trade_size=500, top_bid_size=100)

    features = extractor.get_features()
    # {'depth_imbalance': 0.12, 'queue_depletion_rate': -0.05, 'sweep_indicator': 1}

    model = OrderImbalanceModel()
    mult = model.get_sizing_multiplier("AAPL")
    # 1.2 if strong buy, 0.8 if strong sell, 1.0 otherwise
"""

import logging
import threading
import time as _time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEPTH_LEVELS = 5            # Top 5 price levels
HISTORY_SIZE = 300          # 5 minutes at 1-second frequency
IMBALANCE_THRESHOLD = 0.3   # Strong signal threshold
BUY_MULTIPLIER = 1.2
SELL_MULTIPLIER = 0.8


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OrderBookSnapshot:
    """Single order book snapshot."""
    timestamp: float
    bid_sizes: list[int]      # Sizes at top N bid levels
    ask_sizes: list[int]      # Sizes at top N ask levels
    depth_imbalance: float    # (total_bid - total_ask) / (total_bid + total_ask)
    queue_depletion_rate: float  # Rate of bid queue consumption
    sweep_indicator: int      # 1 if last trade swept top-of-book, else 0
    last_trade_size: int = 0
    top_bid_size: int = 0


@dataclass
class OrderBookFeatures:
    """Extracted features from the order book."""
    depth_imbalance: float           # Current depth imbalance [-1, 1]
    depth_imbalance_ema: float       # Exponential moving average of imbalance
    queue_depletion_rate: float      # Rate of bid queue depletion
    sweep_indicator: int             # Recent sweep count (last 30s)
    imbalance_trend: float           # Slope of imbalance over window
    bid_ask_size_ratio: float        # Total bid size / total ask size
    effective_signal: float          # Combined signal [-1, 1]


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class OrderBookFeatureExtractor:
    """Computes order book features at 1-second frequency.

    Tracks depth imbalance, queue depletion rate, and sweep indicators
    across top-5 price levels.

    Args:
        history_size: Number of snapshots to retain.
        ema_alpha: Smoothing factor for EMA (0 < alpha <= 1).
    """

    def __init__(self, history_size: int = HISTORY_SIZE, ema_alpha: float = 0.1):
        self._history: deque[OrderBookSnapshot] = deque(maxlen=history_size)
        self._per_symbol: dict[str, deque[OrderBookSnapshot]] = {}
        self._ema_alpha = ema_alpha
        self._ema_imbalance: dict[str, float] = {}
        self._lock = threading.Lock()
        self._prev_bid_total: dict[str, float] = {}

    def update(
        self,
        bid_sizes: list[int],
        ask_sizes: list[int],
        last_trade_size: int = 0,
        top_bid_size: int = 0,
        symbol: str | None = None,
        timestamp: float | None = None,
    ) -> OrderBookSnapshot:
        """Ingest a new order book snapshot and compute features.

        Args:
            bid_sizes: Sizes at top N bid price levels (nearest first).
            ask_sizes: Sizes at top N ask price levels (nearest first).
            last_trade_size: Size of the most recent trade.
            top_bid_size: Size at the best bid (for sweep detection).
            symbol: Optional symbol for per-symbol tracking.
            timestamp: Snapshot time (epoch seconds). Defaults to now.

        Returns:
            OrderBookSnapshot with computed features.
        """
        ts = timestamp or _time.time()

        # Pad to DEPTH_LEVELS if needed
        bid_sizes = (bid_sizes + [0] * DEPTH_LEVELS)[:DEPTH_LEVELS]
        ask_sizes = (ask_sizes + [0] * DEPTH_LEVELS)[:DEPTH_LEVELS]

        total_bid = sum(bid_sizes)
        total_ask = sum(ask_sizes)
        total = total_bid + total_ask

        # Depth imbalance
        depth_imbalance = (total_bid - total_ask) / total if total > 0 else 0.0

        # Queue depletion rate
        key = symbol or "__global__"
        prev_bid = self._prev_bid_total.get(key, total_bid)
        queue_depletion = (total_bid - prev_bid) / max(prev_bid, 1)
        self._prev_bid_total[key] = total_bid

        # Sweep indicator
        sweep = 1 if (last_trade_size > 0 and top_bid_size > 0
                      and last_trade_size > top_bid_size) else 0

        snapshot = OrderBookSnapshot(
            timestamp=ts,
            bid_sizes=bid_sizes,
            ask_sizes=ask_sizes,
            depth_imbalance=depth_imbalance,
            queue_depletion_rate=queue_depletion,
            sweep_indicator=sweep,
            last_trade_size=last_trade_size,
            top_bid_size=top_bid_size,
        )

        with self._lock:
            self._history.append(snapshot)
            if symbol:
                if symbol not in self._per_symbol:
                    self._per_symbol[symbol] = deque(maxlen=HISTORY_SIZE)
                self._per_symbol[symbol].append(snapshot)

            # Update EMA
            prev_ema = self._ema_imbalance.get(key, 0.0)
            self._ema_imbalance[key] = (
                self._ema_alpha * depth_imbalance +
                (1 - self._ema_alpha) * prev_ema
            )

        return snapshot

    def get_features(self, symbol: str | None = None) -> OrderBookFeatures:
        """Extract current features for a symbol (or global).

        Args:
            symbol: Optional symbol. Uses global history if None.

        Returns:
            OrderBookFeatures with all computed signals.
        """
        key = symbol or "__global__"

        with self._lock:
            history = list(
                self._per_symbol.get(symbol, self._history) if symbol
                else self._history
            )
            ema = self._ema_imbalance.get(key, 0.0)

        if not history:
            return OrderBookFeatures(
                depth_imbalance=0.0, depth_imbalance_ema=0.0,
                queue_depletion_rate=0.0, sweep_indicator=0,
                imbalance_trend=0.0, bid_ask_size_ratio=1.0,
                effective_signal=0.0,
            )

        latest = history[-1]

        # Sweep count in last 30 seconds
        now = latest.timestamp
        recent_sweeps = sum(
            1 for s in history
            if s.sweep_indicator and (now - s.timestamp) <= 30
        )

        # Imbalance trend (linear regression slope over last 60 snapshots)
        window = history[-60:] if len(history) >= 60 else history
        if len(window) >= 3:
            imbalances = np.array([s.depth_imbalance for s in window])
            x = np.arange(len(imbalances))
            try:
                coeffs = np.polyfit(x, imbalances, 1)
                trend = float(coeffs[0])
            except (np.linalg.LinAlgError, ValueError):
                trend = 0.0
        else:
            trend = 0.0

        # Bid/ask size ratio
        total_bid = sum(latest.bid_sizes)
        total_ask = sum(latest.ask_sizes)
        ratio = total_bid / total_ask if total_ask > 0 else 1.0

        # Combined effective signal: weighted combination of features
        effective = (
            0.5 * ema +
            0.2 * trend * 10 +
            0.2 * min(1.0, max(-1.0, (ratio - 1.0))) +
            0.1 * (recent_sweeps / 5.0 if recent_sweeps > 0 else 0.0)
        )
        effective = max(-1.0, min(1.0, effective))

        return OrderBookFeatures(
            depth_imbalance=latest.depth_imbalance,
            depth_imbalance_ema=ema,
            queue_depletion_rate=latest.queue_depletion_rate,
            sweep_indicator=recent_sweeps,
            imbalance_trend=trend,
            bid_ask_size_ratio=ratio,
            effective_signal=effective,
        )

    def reset(self, symbol: str | None = None):
        """Clear history for a symbol or all."""
        with self._lock:
            if symbol:
                self._per_symbol.pop(symbol, None)
                self._ema_imbalance.pop(symbol, None)
                self._prev_bid_total.pop(symbol, None)
            else:
                self._history.clear()
                self._per_symbol.clear()
                self._ema_imbalance.clear()
                self._prev_bid_total.clear()


# ---------------------------------------------------------------------------
# Order Imbalance Model
# ---------------------------------------------------------------------------

class OrderImbalanceModel:
    """Logistic regression on depth imbalance features.

    Learns to predict short-term price direction from order book features.
    Falls back to threshold-based rules if not fitted.

    Wires imbalance signal as sizing multiplier:
      - strong buy (>0.3)  -> 1.2x
      - strong sell (<-0.3) -> 0.8x
      - neutral -> 1.0x
    """

    def __init__(self):
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._fitted = False
        self._extractor = OrderBookFeatureExtractor()
        self._lock = threading.Lock()

    @property
    def extractor(self) -> OrderBookFeatureExtractor:
        """Access the underlying feature extractor."""
        return self._extractor

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01,
            epochs: int = 100) -> dict:
        """Fit logistic regression on order book features.

        Args:
            X: (n_samples, 7) feature matrix from OrderBookFeatures.
            y: (n_samples,) binary labels (1 = price went up, 0 = down).
            lr: Learning rate.
            epochs: Training iterations.

        Returns:
            Training metrics dict.
        """
        n_features = X.shape[1]
        weights = np.zeros(n_features)
        bias = 0.0

        # Normalize features for numerical stability
        feat_std = np.std(X, axis=0) + 1e-8
        feat_mean = np.mean(X, axis=0)
        X_norm = (X - feat_mean) / feat_std

        for _epoch in range(epochs):
            # Forward pass
            z = X_norm @ weights + bias
            preds = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

            # Gradient
            error = preds - y
            grad_w = X.T @ error / len(y)
            grad_b = np.mean(error)

            # Update
            weights -= lr * grad_w
            bias -= lr * grad_b

        with self._lock:
            self._weights = weights
            self._bias = bias
            self._feat_mean = feat_mean
            self._feat_std = feat_std
            self._fitted = True

        # Compute accuracy
        final_preds = (1.0 / (1.0 + np.exp(-np.clip(X_norm @ weights + bias, -500, 500)))) > 0.5
        accuracy = np.mean(final_preds == y)

        logger.info("T5-014: OrderImbalanceModel fitted (accuracy=%.3f, epochs=%d)",
                     accuracy, epochs)
        return {"accuracy": float(accuracy), "epochs": epochs}

    def predict_signal(self, symbol: str | None = None) -> float:
        """Predict directional signal from current order book state.

        Args:
            symbol: Optional symbol for per-symbol features.

        Returns:
            Signal in [-1.0, 1.0]. Positive = bullish.
        """
        features = self._extractor.get_features(symbol)

        if self._fitted and self._weights is not None:
            # Use logistic regression model
            x = np.array([
                features.depth_imbalance,
                features.depth_imbalance_ema,
                features.queue_depletion_rate,
                features.sweep_indicator / 5.0,
                features.imbalance_trend,
                features.bid_ask_size_ratio - 1.0,
                features.effective_signal,
            ])
            with self._lock:
                # Normalize using training statistics
                if self._feat_mean is not None and self._feat_std is not None:
                    x = (x - self._feat_mean) / self._feat_std
                z = float(np.dot(self._weights, x) + self._bias)
            prob = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            return float(prob * 2 - 1)  # Map [0,1] -> [-1,1]
        else:
            # Fallback: use effective_signal directly
            return features.effective_signal

    def get_sizing_multiplier(self, symbol: str | None = None) -> float:
        """T5-014: Get position sizing multiplier from order book signal.

        Returns:
            1.2 for strong buy (signal > 0.3)
            0.8 for strong sell (signal < -0.3)
            1.0 for neutral

        Gated behind ORDER_BOOK_SIGNAL_ENABLED config flag.
        """
        try:
            import config as _cfg
            if not getattr(_cfg, "ORDER_BOOK_SIGNAL_ENABLED", False):
                return 1.0
        except Exception:
            return 1.0

        signal = self.predict_signal(symbol)

        if signal > IMBALANCE_THRESHOLD:
            return BUY_MULTIPLIER
        elif signal < -IMBALANCE_THRESHOLD:
            return SELL_MULTIPLIER
        else:
            return 1.0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_model_instance: OrderImbalanceModel | None = None
_model_lock = threading.Lock()


def get_order_book_model() -> OrderImbalanceModel:
    """Get or create the global OrderImbalanceModel singleton."""
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                _model_instance = OrderImbalanceModel()
    return _model_instance
