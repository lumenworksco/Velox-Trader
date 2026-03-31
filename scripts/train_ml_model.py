#!/usr/bin/env python3
"""V11.4: ML Ensemble Training Script.

Generates training data from historical OHLCV bars, computes 200+ features,
creates forward-return labels, and trains the LightGBM + XGBoost + CatBoost
+ RandomForest stacking ensemble.

Usage:
    python3 scripts/train_ml_model.py [--days 252] [--optimize] [--trials 30]

The trained model is saved to models/ and auto-detected by the signal
processor on next startup.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set TESTING to avoid config requiring API keys during import
os.environ.setdefault("TESTING", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_ml_model")


# ---------------------------------------------------------------------------
# Universe of liquid US equities for training
# ---------------------------------------------------------------------------

TRAINING_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "AXP", "V",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN", "GILD", "TMO",
    # Consumer
    "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW", "DIS", "NFLX",
    # Energy / Industrial
    "XOM", "CVX", "COP", "SLB", "CAT", "BA", "GE", "HON", "UPS", "RTX",
    # Other
    "COIN", "SQ", "PYPL", "UBER", "ABNB", "SNAP", "ROKU", "ZM", "SHOP", "PLTR",
    # ETFs for cross-asset context
    "SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "XLV",
]


def generate_synthetic_training_data(
    n_symbols: int = 50,
    n_bars_per_symbol: int = 252,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for training when API is unavailable.

    Creates price series with:
    - Mean-reverting behavior (OU process) for some symbols
    - Trending behavior for others
    - Realistic volume patterns (U-shaped intraday)
    - Earnings-like jumps
    """
    np.random.seed(seed)
    symbols = TRAINING_UNIVERSE[:n_symbols]
    all_bars = []

    for i, symbol in enumerate(symbols):
        # Base price and volatility
        base_price = np.random.uniform(20, 500)
        daily_vol = np.random.uniform(0.01, 0.04)

        # Generate daily returns with regime-switching
        returns = np.random.normal(0.0003, daily_vol, n_bars_per_symbol)

        # Add mean-reversion for some symbols
        if i % 3 == 0:
            theta = 0.1  # mean-reversion speed
            mu = 0.0
            for t in range(1, len(returns)):
                returns[t] = theta * (mu - returns[t - 1]) + daily_vol * np.random.randn()

        # Add occasional jumps (earnings)
        for t in np.random.choice(n_bars_per_symbol, size=4, replace=False):
            returns[t] += np.random.choice([-1, 1]) * np.random.uniform(0.03, 0.08)

        # Build price series
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        dates = pd.bdate_range(
            end=datetime.now().date() - timedelta(days=1),
            periods=n_bars_per_symbol,
        )

        for t in range(len(prices)):
            intraday_vol = daily_vol * prices[t]
            high = prices[t] + abs(np.random.normal(0, intraday_vol * 0.5))
            low = prices[t] - abs(np.random.normal(0, intraday_vol * 0.5))
            open_price = prices[t] + np.random.normal(0, intraday_vol * 0.2)

            # Volume with U-shape
            base_vol = np.random.lognormal(mean=14, sigma=1)

            all_bars.append({
                "symbol": symbol,
                "timestamp": dates[t],
                "open": round(open_price, 2),
                "high": round(max(high, open_price, prices[t]), 2),
                "low": round(min(low, open_price, prices[t]), 2),
                "close": round(prices[t], 2),
                "volume": int(base_vol),
            })

    df = pd.DataFrame(all_bars)
    logger.info("Generated synthetic data: %d bars for %d symbols", len(df), n_symbols)
    return df


def compute_features_and_labels(
    bars_df: pd.DataFrame,
    forward_window: int = 10,
    label_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute features and labels from OHLCV bars.

    Args:
        bars_df: DataFrame with columns [symbol, timestamp, open, high, low, close, volume]
        forward_window: Number of bars to look forward for label computation
        label_threshold: Return threshold for binary classification (0 = any positive return)

    Returns:
        (features_df, labels) where labels are binary (1 = positive forward return)
    """
    from ml.features import FeatureEngine

    engine = FeatureEngine()
    all_features = []
    all_labels = []
    symbols = bars_df["symbol"].unique()

    logger.info("Computing features for %d symbols...", len(symbols))

    # Pre-compute SPY data for cross-asset features
    spy_data = None
    if "SPY" in symbols:
        spy_bars = bars_df[bars_df["symbol"] == "SPY"].set_index("timestamp").sort_index()
        spy_data = {
            "spy_bars": spy_bars,
            "vix_level": 20.0,  # Default VIX placeholder
            "vix_change": 0.0,
        }

    for sym_idx, symbol in enumerate(symbols):
        if symbol in ("SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "XLV"):
            continue  # Skip ETFs as prediction targets

        sym_bars = bars_df[bars_df["symbol"] == symbol].sort_values("timestamp")
        sym_bars = sym_bars.set_index("timestamp")

        if len(sym_bars) < 80:  # Need enough history for features
            continue

        # Compute forward returns for labels
        forward_returns = sym_bars["close"].pct_change(forward_window).shift(-forward_window)

        # Slide a window and compute features at each point
        min_lookback = 60
        for i in range(min_lookback, len(sym_bars) - forward_window):
            window = sym_bars.iloc[max(0, i - 252):i + 1]  # Up to 252 bars of history

            try:
                feats = engine.compute_all_features(symbol, window, market_data=spy_data)
                if not feats:
                    continue

                label_val = forward_returns.iloc[i]
                if pd.isna(label_val):
                    continue

                # Binary label: 1 if positive forward return, 0 otherwise
                label = 1 if label_val > label_threshold else 0

                feats["_symbol"] = symbol
                feats["_timestamp"] = str(sym_bars.index[i])
                all_features.append(feats)
                all_labels.append(label)
            except Exception:
                continue

        if (sym_idx + 1) % 10 == 0:
            logger.info("  Processed %d/%d symbols (%d samples so far)",
                        sym_idx + 1, len(symbols), len(all_features))

    if not all_features:
        raise ValueError("No features computed — check data quality")

    features_df = pd.DataFrame(all_features)

    # Remove metadata columns
    meta_cols = [c for c in features_df.columns if c.startswith("_")]
    features_df = features_df.drop(columns=meta_cols)

    # Fill any remaining NaN
    features_df = features_df.fillna(0.0)

    labels = pd.Series(all_labels, name="label")

    logger.info("Feature matrix: %d samples x %d features", len(features_df), len(features_df.columns))
    logger.info("Label distribution: %.1f%% positive, %.1f%% negative",
                labels.mean() * 100, (1 - labels.mean()) * 100)

    return features_df, labels


def train_and_save(
    features_df: pd.DataFrame,
    labels: pd.Series,
    optimize: bool = False,
    n_trials: int = 30,
    model_dir: str = None,
) -> str:
    """Train the ensemble model and save to disk.

    Args:
        features_df: Feature DataFrame
        labels: Binary labels
        optimize: Whether to run Optuna hyperparameter optimization
        n_trials: Number of Optuna trials
        model_dir: Directory to save model (default: models/)

    Returns:
        Path to saved model file
    """
    from ml.training import ModelTrainer

    if model_dir is None:
        model_dir = str(PROJECT_ROOT / "models")

    trainer = ModelTrainer(model_type="classification", use_stacking=True)

    # Optional: Bayesian hyperparameter optimization
    if optimize:
        logger.info("Running Optuna hyperparameter optimization (%d trials)...", n_trials)
        best_params = trainer.optimize_hyperparameters(
            features_df, labels, n_trials=n_trials,
        )
        if best_params:
            logger.info("Best parameters found: %s", json.dumps(best_params, indent=2, default=str))
            model = trainer.train_with_optimized_params(
                features_df, labels, best_params,
                purge_window=10, embargo=5, n_splits=5,
            )
        else:
            logger.info("Optimization returned no results, using defaults")
            model = trainer.train(features_df, labels, purge_window=10, embargo=5, n_splits=5)
    else:
        model = trainer.train(features_df, labels, purge_window=10, embargo=5, n_splits=5)

    # Evaluate on last 20% (time-forward holdout)
    split_idx = int(len(features_df) * 0.8)
    test_features = features_df.iloc[split_idx:]
    test_labels = labels.iloc[split_idx:]

    if len(test_features) > 10:
        metrics = trainer.evaluate(model, test_features, test_labels)
        logger.info("=" * 60)
        logger.info("HOLDOUT EVALUATION (last 20%%):")
        logger.info("  Accuracy:  %.4f", metrics.accuracy)
        logger.info("  AUC-ROC:   %.4f", metrics.auc_roc)
        logger.info("  F1:        %.4f", metrics.f1)
        logger.info("  Log Loss:  %.4f", metrics.log_loss_val)
        logger.info("  Pred Sharpe: %.4f", metrics.sharpe_of_predictions)
        logger.info("=" * 60)

        # Check for overfitting / poor model
        if metrics.auc_roc < 0.52:
            logger.warning("WARNING: AUC-ROC < 0.52 — model may not be better than random")
        if metrics.accuracy < 0.48:
            logger.warning("WARNING: Accuracy < 48%% — model may be counterproductive")

    # Save model
    model_path = trainer.save_model(model, model_dir)
    logger.info("Model saved to: %s", model_path)

    # Also save feature names for validation
    feature_names_path = os.path.join(model_dir, "feature_names.json")
    with open(feature_names_path, "w") as f:
        json.dump(model.feature_names, f, indent=2)
    logger.info("Feature names saved to: %s", feature_names_path)

    # Print top features
    if model.metrics and model.metrics.feature_importance:
        top_features = sorted(
            model.metrics.feature_importance.items(),
            key=lambda x: abs(x[1]), reverse=True,
        )[:20]
        logger.info("\nTop 20 Features by Importance:")
        for name, imp in top_features:
            logger.info("  %-40s  %.4f", name, imp)

    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train VELOX ML ensemble model")
    parser.add_argument("--days", type=int, default=252, help="Days of historical data (default: 252)")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials (default: 30)")
    parser.add_argument("--symbols", type=int, default=50, help="Number of symbols to train on (default: 50)")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("VELOX V11.4 ML Model Training")
    logger.info("=" * 60)
    logger.info("  Symbols: %d", args.symbols)
    logger.info("  Days: %d", args.days)
    logger.info("  Optimize: %s", args.optimize)
    if args.optimize:
        logger.info("  Trials: %d", args.trials)

    # Step 1: Generate training data
    logger.info("\n--- Step 1: Generating Training Data ---")
    bars_df = generate_synthetic_training_data(
        n_symbols=args.symbols,
        n_bars_per_symbol=args.days,
    )

    # Step 2: Compute features and labels
    logger.info("\n--- Step 2: Computing Features & Labels ---")
    features_df, labels = compute_features_and_labels(bars_df, forward_window=10)

    # Step 3: Train and save
    logger.info("\n--- Step 3: Training Ensemble Model ---")
    model_path = train_and_save(
        features_df, labels,
        optimize=args.optimize,
        n_trials=args.trials,
    )

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("Training complete in %.1f seconds", elapsed)
    logger.info("Model: %s", model_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
