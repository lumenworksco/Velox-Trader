"""Configuration — all settings and environment variable handling."""

import os
from datetime import time
from zoneinfo import ZoneInfo

# ============================================================
# BROKER & CONNECTIVITY
# ============================================================

ET = ZoneInfo("America/New_York")

PAPER_MODE = os.getenv("ALPACA_LIVE", "false") != "true"
API_KEY = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

# Validate API credentials at import time (skip in test environments)
if not os.getenv("TESTING") and not os.getenv("PYTEST_CURRENT_TEST"):
    if not API_KEY:
        raise RuntimeError("ALPACA_API_KEY environment variable is required")
    if not API_SECRET:
        raise RuntimeError("ALPACA_API_SECRET environment variable is required")

ALLOW_SHORT = os.getenv("ALLOW_SHORT", "true") == "true"

# MED-030: Removed BROKER_ABSTRACTION_ENABLED and PAPER_BROKER_SPREAD_BPS (dead code, never referenced)

# ============================================================
# MARKET HOURS (ET)
# ============================================================

MARKET_OPEN = time(9, 30)
TRADING_START = time(10, 0)
ORB_EXIT_TIME = time(15, 45)
MARKET_CLOSE = time(16, 0)
EOD_SUMMARY_TIME = time(16, 15)

# ============================================================
# SYMBOL UNIVERSE
# ============================================================

CORE_SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL",
    "META", "NFLX", "AMD", "INTC", "SOFI", "PLTR", "COIN", "PYPL",
    "SNAP", "BABA", "JD", "SHOP", "JPM", "BAC", "GS", "V", "MA",
    "UNH", "CVX", "XOM", "LLY", "PFE", "SQ", "UBER", "LYFT", "ABNB",
    "RBLX", "AFRM", "HOOD", "CRWD", "PANW", "ZS", "IWM", "DIA",
    "XLF", "XLE", "XLK", "XLV", "ARKK", "GLD", "SLV", "TLT",
]

LARGE_CAP_GROWTH = [
    "ADBE", "CRM", "NOW", "SNOW", "DDOG", "NET", "FTNT", "OKTA",
    "ZM", "DOCU", "TWLO", "MDB", "ESTC", "CFLT", "GTLB", "BILL",
    "HUBS", "VEEV", "WDAY", "ANSS", "TTD", "ROKU", "PINS", "ETSY",
    "W", "CHWY", "DASH", "APP",
]

SECTOR_ETFS = [
    "XLB", "XLI", "XLU", "XLRE", "XLC", "XLP", "XLY", "SOXX",
    "SMH", "IBB", "SOXL", "TQQQ", "SPXL", "UVXY", "SQQQ",
    "TNA", "FAS", "FAZ", "LABU", "LABD",
]

HIGH_MOMENTUM_MIDCAPS = [
    "CELH", "SMCI", "AXON", "PODD", "ENPH", "FSLR", "RUN", "BLNK",
    "CHPT", "BE", "IONQ", "RGTI", "QUBT", "ARQQ", "BBAI", "SOUN",
    "ASTS", "RDW", "RKLB", "LUNR", "DUOL", "CAVA", "BROS", "SHAK",
    "WING", "TXRH", "CMG", "DPZ", "DNUT", "JACK",
]

# Full universe
SYMBOLS = CORE_SYMBOLS + LARGE_CAP_GROWTH + SECTOR_ETFS + HIGH_MOMENTUM_MIDCAPS

# Leveraged ETFs — ONLY use VWAP on these, never ORB or Momentum
LEVERAGED_ETFS = {
    "SOXL", "TQQQ", "SPXL", "UVXY", "SQQQ", "TNA", "FAS", "FAZ", "LABU", "LABD",
}

# Non-leveraged symbols for ORB and Momentum
STANDARD_SYMBOLS = [s for s in SYMBOLS if s not in LEVERAGED_ETFS]

# Sector ETF mapping (for relative strength and pairs)
SECTOR_MAP = {
    # Technology
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "INTC": "XLK",
    "CRM": "XLK", "ADBE": "XLK", "NOW": "XLK", "PLTR": "XLK", "CRWD": "XLK",
    "PANW": "XLK", "ZS": "XLK", "SNOW": "XLK", "DDOG": "XLK", "NET": "XLK",
    "FTNT": "XLK", "OKTA": "XLK", "TWLO": "XLK", "MDB": "XLK", "ESTC": "XLK",
    "CFLT": "XLK", "GTLB": "XLK", "BILL": "XLK", "HUBS": "XLK", "VEEV": "XLK",
    "WDAY": "XLK", "ANSS": "XLK", "SMCI": "XLK", "IONQ": "XLK", "RGTI": "XLK",
    "QUBT": "XLK", "ARQQ": "XLK", "BBAI": "XLK", "SOUN": "XLK", "APP": "XLK",
    # Semiconductors
    "SOXX": "SMH", "SMH": "SMH", "SOXL": "SMH",
    # Communication Services
    "META": "XLC", "GOOGL": "XLC", "NFLX": "XLC", "SNAP": "XLC", "TTD": "XLC",
    "ROKU": "XLC", "PINS": "XLC", "ZM": "XLC", "DOCU": "XLC", "DUOL": "XLC",
    # Consumer Discretionary
    "TSLA": "XLY", "AMZN": "XLY", "SHOP": "XLY", "BABA": "XLY", "JD": "XLY",
    "UBER": "XLY", "LYFT": "XLY", "ABNB": "XLY", "RBLX": "XLY", "ETSY": "XLY",
    "W": "XLY", "CHWY": "XLY", "DASH": "XLY", "CAVA": "XLY", "BROS": "XLY",
    "SHAK": "XLY", "WING": "XLY", "TXRH": "XLY", "CMG": "XLY", "DPZ": "XLY",
    "DNUT": "XLY", "JACK": "XLY",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "V": "XLF", "MA": "XLF",
    "PYPL": "XLF", "SQ": "XLF", "SOFI": "XLF", "COIN": "XLF", "AFRM": "XLF",
    "HOOD": "XLF",
    # Healthcare
    "UNH": "XLV", "LLY": "XLV", "PFE": "XLV", "PODD": "XLV",
    # Energy
    "CVX": "XLE", "XOM": "XLE",
    # Clean Energy / EV
    "ENPH": "XLE", "FSLR": "XLE", "RUN": "XLE", "BLNK": "XLE",
    "CHPT": "XLE", "BE": "XLE",
    # Biotech
    "IBB": "IBB", "LABU": "IBB", "LABD": "IBB",
    # Aerospace / Space
    "AXON": "XLI", "ASTS": "XLI", "RDW": "XLI", "RKLB": "XLI", "LUNR": "XLI",
    # Consumer / Other
    "CELH": "XLP",
}

# Sector groups for pair selection
SECTOR_GROUPS = {
    'mega_tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],
    'semis': ['NVDA', 'AMD', 'INTC'],
    'banks': ['JPM', 'BAC', 'GS', 'V', 'MA'],
    'energy': ['XOM', 'CVX'],
    'fintech': ['PYPL', 'SQ', 'SOFI', 'COIN', 'AFRM', 'HOOD'],
    'cloud_saas': ['CRM', 'NOW', 'SNOW', 'DDOG', 'NET', 'CRWD', 'PANW', 'ZS'],
    'consumer': ['UBER', 'LYFT', 'ABNB', 'DASH'],
    'etf_pairs': [('SPY', 'QQQ'), ('IWM', 'QQQ'), ('XLK', 'QQQ'), ('GLD', 'SLV')],
}

# ============================================================
# STRATEGY PARAMETERS
# ============================================================

STRATEGY_ALLOCATIONS = {
    # V11.3 T10: Reallocated from dead strategies (COPULA_PAIRS, SECTOR_MOM, etc.)
    # to active ones. Total = 100%.
    'STAT_MR': 0.35,         # was 0.30 — highest Sharpe, most active
    'VWAP': 0.20,            # was 0.15 — second most active
    'KALMAN_PAIRS': 0.20,    # was 0.15 — diversifying, uncorrelated
    'ORB': 0.10,             # was 0.05 — now with trailing stops
    'MICRO_MOM': 0.10,       # was 0.05 — now with ATR-based stops
    'PEAD': 0.05,            # was 0.08 — event-driven, low frequency
}

# --- Statistical Mean Reversion ---
MR_ZSCORE_ENTRY = 1.5
MR_ZSCORE_EXIT_FULL = 0.2
MR_ZSCORE_EXIT_PARTIAL = 0.5
MR_ZSCORE_STOP = 2.5
MR_RSI_PERIOD = 7
MR_RSI_OVERSOLD = 40
MR_RSI_OVERBOUGHT = 60
MR_HURST_MAX = 0.52
MR_HALFLIFE_MIN_HOURS = 1
MR_HALFLIFE_MAX_HOURS = 48
MR_UNIVERSE_SIZE = 40
MR_UNIVERSE_PREP_TIME = time(9, 0)
MR_MIN_GAIN_PCT = 0.002
MR_MIN_RR_RATIO = 2.0              # Raised from 1.5 — only take trades with 2:1 R/R
MR_MIN_STOP_PCT = 0.005            # Minimum stop distance = 0.5% of price (prevents breakeven stops)

# --- VWAP Mean Reversion ---
VWAP_OU_ZSCORE_MIN   = 1.0        # OU z-score confirmation for entries
VWAP_MAX_SPREAD_PCT  = 0.002      # Skip if bid-ask spread > 0.2% (was 0.1% — too restrictive)
VWAP_VOLUME_RATIO    = 0.8        # Volume ratio vs 20-bar average
MAX_INTRADAY_MOVE_PCT = 0.03      # Skip if stock moved > 3% today
VWAP_BAND_STD        = 2.0        # Standard deviation multiplier for VWAP bands
VWAP_RSI_OVERSOLD    = 45         # RSI below this = oversold (buy signal, raised from 40)
VWAP_RSI_OVERBOUGHT  = 70         # RSI above this = overbought (short signal)
VWAP_CONFIRMATION_BARS = 1        # Bars confirming bounce (1 = disabled)
VWAP_STOP_EXTENSION  = 0.5        # Stop extension beyond band (in std devs)
VWAP_MIN_STOP_PCT    = 0.01       # Minimum 1.0% stop distance (was 0.5%)

# --- Kalman Pairs Trading ---
PAIRS_ZSCORE_ENTRY = 2.0
PAIRS_ZSCORE_EXIT = 0.2
PAIRS_ZSCORE_STOP = 3.0
PAIRS_MAX_HOLD_DAYS = 10
PAIRS_MAX_ACTIVE = 15
PAIRS_MIN_CORRELATION = 0.70       # Lowered from 0.80 — find more pairs
PAIRS_COINT_PVALUE = 0.10          # Relaxed from 0.05 — accept more pairs (still statistically meaningful)
KALMAN_DELTA = 1e-4
KALMAN_OBS_NOISE = 0.001
PAIRS_TP_PCT = 0.015   # V10: 1.5% take-profit (was 0.5% — negative EV after costs)
PAIRS_SL_PCT = 0.010   # V10: 1.0% stop-loss (was 1.5% — gives 1.5:1 R/R)

# --- Opening Range Breakout ---
ORB_ENABLED          = True
ORB_VOLUME_RATIO     = 1.3        # Volume confirmation ratio
ORB_MAX_GAP_PCT      = 0.04       # Skip if gap > 4%
ORB_MAX_RANGE_PCT    = 0.035      # Skip if range > 3.5%
ORB_MIN_STOP_PCT     = 0.008      # Minimum 0.8% stop distance (was 0.3%)
ORB_SCAN_SYMBOLS     = 15         # Top N by morning volume
ORB_ACTIVE_UNTIL     = time(12, 30)   # Extended from 11:30 — more signal window
ORB_BREAKOUT_BUFFER  = 0.001      # 0.1% confirmation buffer above/below ORB range
ORB_TP_MULT          = 1.5        # Take profit = entry ± 1.5x ORB range
ORB_SL_MULT          = 0.5        # Stop loss = entry ∓ 0.5x ORB range
ORB_TIME_STOP_HOURS  = 2          # Close after 2 hours

# --- Micro Momentum ---
MICRO_SPY_VOL_SPIKE_MULT = 1.5     # Was 2.0 — detect more micro-events
MICRO_SPY_MIN_MOVE_PCT = 0.0008    # Was 0.001 — 0.08% SPY move triggers event
MICRO_MAX_HOLD_MINUTES = 30          # V11.3: Extended from 15 to 30 — trailing stop manages risk
MICRO_STOP_PCT = 0.01                # Was 0.003 — 1% stop survives noise on beta=2 stocks
MICRO_TARGET_PCT = 0.02              # Was 0.006 — 2% target keeps 2:1 R/R
MICRO_MAX_TRADES_PER_EVENT = 2       # Was 3 — fewer, higher-conviction trades
MICRO_MAX_DAILY_GAIN_DISABLE = 0.015
MICRO_TOP_BETA_STOCKS = 5
MICRO_EVENT_COOLDOWN_SEC = 900    # 15-min cooldown between events
MICRO_EVENT_WINDOW_SEC = 300      # 5-min window after event for trades
MICRO_BETA_TABLE = {
    'NVDA': 1.6, 'AMD': 1.5, 'TSLA': 1.8, 'COIN': 2.0, 'SOFI': 1.6,
    'PLTR': 1.5, 'HOOD': 1.8, 'AFRM': 1.7, 'SMCI': 2.0, 'SNOW': 1.4,
    'DDOG': 1.3, 'NET': 1.4, 'SQ': 1.5, 'META': 1.3, 'NFLX': 1.2,
    'GOOGL': 1.15, 'AMZN': 1.2, 'AAPL': 1.1, 'MSFT': 1.05,
    'CRWD': 1.3, 'PANW': 1.2, 'ZS': 1.2, 'RBLX': 1.4,
    'CELH': 1.3, 'AXON': 1.2, 'ENPH': 1.5, 'FSLR': 1.4,
    'IONQ': 2.0, 'RGTI': 2.0, 'QUBT': 2.0, 'RKLB': 1.8,
    'LYFT': 1.5, 'UBER': 1.3, 'ABNB': 1.4, 'DASH': 1.3,
}

# --- Post-Earnings Announcement Drift ---
PEAD_ENABLED = True
PEAD_MIN_SURPRISE_PCT = 3.0        # Lowered from 5.0 — capture more earnings plays
PEAD_MIN_VOLUME_RATIO = 1.5       # Lowered from 2.0 — less restrictive volume gate
PEAD_HOLD_DAYS_MIN = 10
PEAD_HOLD_DAYS_MAX = 20
PEAD_TAKE_PROFIT = 0.05
PEAD_STOP_LOSS = 0.03
PEAD_MAX_POSITIONS = 5
PEAD_POSITION_SIZE_PCT = 0.02

# --- T5-001: PEAD Pre-Earnings Implied-Move Exploitation ---
PEAD_PRE_EARNINGS_ENABLED = os.getenv("PEAD_PRE_EARNINGS_ENABLED", "true") == "true"
PEAD_IMPLIED_MOVE_RATIO_THRESHOLD = 1.5   # realized > 1.5x implied to qualify
PEAD_PRE_ENTRY_DAYS = 2                   # enter 2 days before earnings
PEAD_POST_EXIT_DAYS = 2                   # exit within 2 days post-earnings
PEAD_PRE_SIZE_FRACTION = 0.50             # 50% of normal position size
PEAD_PRE_ATR_STOP_MULT = 1.5             # 1.5x ATR stop

# --- T5-002: Intraday Regime Switching (HMM) ---
INTRADAY_REGIME_ENABLED = os.getenv("INTRADAY_REGIME_ENABLED", "true") == "true"
INTRADAY_REGIME_UPDATE_MIN = 5            # update every 5 minutes
INTRADAY_REGIME_STATES = 3               # Trending Up, Trending Down, Mean-Reverting

# --- T5-003: ML Ensemble Upgrade ---
ML_ENSEMBLE_ENABLED = os.getenv("ML_ENSEMBLE_ENABLED", "true") == "true"
ML_BAYESIAN_OPTIM_TRIALS = 50            # Optuna trials for hyperparameter search

# --- T5-004: NLP Sentiment (FinBERT) ---
NLP_SENTIMENT_ENABLED = os.getenv("NLP_SENTIMENT_ENABLED", "true") == "true"

# --- T5-005: Options Skew Signal ---
OPTIONS_SKEW_ENABLED = os.getenv("OPTIONS_SKEW_ENABLED", "true") == "true"
SKEW_ROLLING_WINDOW = 90                 # 90-day rolling mean for z-score
SKEW_BULLISH_THRESHOLD = -1.5            # z < -1.5 -> bullish
SKEW_BEARISH_THRESHOLD = 1.5             # z > 1.5 -> bearish
SKEW_MAX_BOOST = 0.40                    # max +/-40% confidence adjustment

# --- T5-006: Dark Pool Volume Detection ---
DARK_POOL_ENABLED = os.getenv("DARK_POOL_ENABLED", "true") == "true"
DARK_POOL_ROLLING_MINUTES = 30           # rolling window
DARK_POOL_RATIO_THRESHOLD = 0.35         # 35% dark pool ratio
DARK_POOL_ALPHA_WEIGHT = 0.15            # confidence multiplier weight

# --- T5-008: PDT Protection ---
PDT_PROTECTION_ENABLED = os.getenv("PDT_PROTECTION_ENABLED", "true") == "true"

# ============================================================
# RISK MANAGEMENT
# ============================================================

# --- Slippage Modeling (backtester only) ---
SLIPPAGE_MODEL = "volume"  # "fixed", "volume", or "market_impact"
SLIPPAGE_FIXED_BPS = 1.0
SLIPPAGE_BASE_BPS = 0.5
SLIPPAGE_VOLUME_FACTOR = 0.1

# --- PDT Rule ---
PDT_ENFORCEMENT_ENABLED = os.getenv("PDT_ENFORCEMENT", "true") == "true"
PDT_EQUITY_THRESHOLD = 25_000.0

# --- Position Sizing ---
RISK_PER_TRADE_PCT = 0.005          # Risk 0.5% of portfolio per trade (was 0.8% — tighter)
MAX_POSITION_PCT = 0.05             # Hard cap: max 5% per position (was 8% — tighter)
MIN_POSITION_VALUE = 100            # Min $100 per trade
MAX_POSITIONS = 12
MAX_PORTFOLIO_DEPLOY = 0.55
DAILY_LOSS_HALT = -0.025

# --- Volatility Targeting ---
VOL_TARGET_DAILY = 0.01
VOL_TARGET_MAX = 0.015
VOL_SCALAR_MIN = 0.3
VOL_SCALAR_MAX = 1.5

# --- Kelly Criterion ---
KELLY_ENABLED = True
KELLY_MIN_TRADES = 30
KELLY_LOOKBACK = 100
KELLY_FRACTION_MULT = 0.5          # Half-Kelly
KELLY_MIN_RISK = 0.003
KELLY_MAX_RISK = 0.02

# --- T7-004: Bayesian Kelly Sizing ---
BAYESIAN_KELLY_ENABLED = True           # V11.4: Enable regime-weighted Kelly fractions

# --- Daily P&L Controls ---
PNL_GAIN_LOCK_PCT = 0.015
PNL_LOSS_HALT_PCT = -0.010
PNL_GAIN_LOCK_SIZE_MULT = 0.30

# --- Re-entry Cooldown ---
REENTRY_COOLDOWN_MIN = 30          # Block re-entry for 30 min after stop-loss (was 15 — too short)

# --- Per-Symbol Daily Loss Cap ---
MAX_SYMBOL_DAILY_LOSS = 200.0      # Max $200 loss per symbol per day

# --- VIX Scaling ---
VIX_RISK_SCALING_ENABLED = True
VIX_HALT_THRESHOLD = 40            # Halt all new positions above VIX 40
VIX_CACHE_SECONDS = 300            # Cache VIX value for 5 minutes

# --- Beta Neutralization ---
BETA_MAX_ABS = 0.3
BETA_CHECK_INTERVAL_MIN = 15
BETA_SKIP_FIRST_MINUTES = 15

# --- Market Regime ---
REGIME_CHECK_INTERVAL_MIN = 30
REGIME_EMA_PERIOD = 20
BEARISH_SIZE_CUT = 0.40

# --- HMM Regime Detection ---
HMM_REGIME_ENABLED = True
HMM_N_STATES = 5
HMM_RETRAIN_DAY = "sunday"
HMM_TRAINING_YEARS = 3
HMM_MIN_PROBABILITY = 0.4       # Below this, treat as uncertain — conservative sizing

# --- Portfolio Heat & Correlation ---
PORTFOLIO_HEAT_ENABLED = True
PORTFOLIO_HEAT_MAX = 0.60
CLUSTER_CORRELATION_THRESHOLD = 0.70
CLUSTER_MAX_HEAT = 0.20
HEAT_CORRELATION_LOOKBACK = 20     # trading days
CORRELATION_THRESHOLD = 0.92       # Skip if correlated > 92% with open position
MAX_PAIRWISE_CORRELATION = 0.70    # MED-035: Correlation limiter pairwise cap
MIN_EFFECTIVE_BETS = 2.0           # MED-035: Minimum effective independent bets
MAX_SECTOR_WEIGHT = 0.50           # MED-035: Max sector concentration weight

# --- Short Selling ---
SHORT_SIZE_MULTIPLIER = 0.75       # Short positions = 75% of equivalent long size
SHORT_HARD_STOP_PCT = 0.04         # Close short if goes against you > 4%
MOMENTUM_TRAILING_STOP_PCT = 0.02  # Trailing stop for position monitor
NO_SHORT_SYMBOLS = {"SPY", "QQQ", "IWM", "DIA"}

# V11.3: Symbols excluded from broker sync re-adoption (e.g., hedge positions)
BROKER_SYNC_EXCLUDE_SYMBOLS = {"SPY"}

# --- Dynamic Capital Allocation ---
DYNAMIC_ALLOCATION = os.getenv("DYNAMIC_ALLOCATION", "true") == "true"
ALLOCATION_LOOKBACK_DAYS = 20      # Rolling window for Sharpe-based allocation
# V10 CONFIG-002: Removed legacy 10% minimum (conflicted with ADAPTIVE_MIN_WEIGHT 3%)
ALLOCATION_MIN_WEIGHT = 0.03  # Aligned with ADAPTIVE_MIN_WEIGHT

# --- Adaptive Strategy Allocation ---
ADAPTIVE_ALLOCATION_ENABLED = True
ADAPTIVE_MIN_WEIGHT = 0.03
ADAPTIVE_MAX_WEIGHT = 0.60
ADAPTIVE_MAX_DAILY_CHANGE = 0.10
ADAPTIVE_SORTINO_LOOKBACK = 30    # trades
ADAPTIVE_SORTINO_WEIGHT = 0.40
ADAPTIVE_REGIME_WEIGHT = 0.30
ADAPTIVE_CORRELATION_WEIGHT = 0.30

# --- Monte Carlo Tail Risk ---
MONTE_CARLO_ENABLED = True
MONTE_CARLO_SIMULATIONS = 10000
MONTE_CARLO_HORIZON_DAYS = 21
MONTE_CARLO_CVAR_LIMIT = -0.08    # De-risk if cvar_99 exceeds -8%
MONTE_CARLO_DELEVERAGE_PCT = 0.20 # Reduce VOL_TARGET by 20%

# ============================================================
# EXECUTION
# ============================================================

# --- Kill Switch ---
KILL_SWITCH_BATCH_SIZE = 5         # MED-031: Positions to close per batch
KILL_SWITCH_BATCH_DELAY_SEC = 0.5  # MED-031: Delay between batches (seconds)

# --- Transaction Cost Model ---
COST_SPREAD_BPS = 1.0              # MED-035: Default spread cost (basis points)
COST_SLIPPAGE_BPS = 0.5            # MED-035: Default slippage cost (basis points)
COST_COMMISSION_PER_SHARE = 0.0035 # MED-035: Commission per share
COST_MIN_EXPECTED_RETURN_BPS = 5.0 # MED-035: Minimum expected return to justify trade

# --- Smart Order Routing ---
SMART_ROUTING_ENABLED = True
SPREAD_THRESHOLD_PCT = 0.0015     # Use limit if spread > 0.15%
CHASE_AFTER_SECONDS = 60
CHASE_CONVERT_MARKET_AFTER = 120
ADAPTIVE_TWAP_ENABLED = True

# --- Exit Management ---
# V11.4: Removed ADVANCED_EXITS_ENABLED and SCALED_TP_ENABLED (superseded)
ADAPTIVE_EXITS_ENABLED = True
TRAILING_STOP_PCT = 0.015
BREAKEVEN_STOP_ENABLED = True
ATR_EXPANSION_MULT = 2.5
RSI_EXIT_THRESHOLD = 70

# --- ATR Trailing Stops ---
ATR_TRAILING_ENABLED = True
ATR_TRAIL_MULT = {
    "STAT_MR": 1.5,
    "VWAP": 1.5,
    "KALMAN_PAIRS": 2.0,
    "ORB": 2.5,
    "MICRO_MOM": 1.0,
}
ATR_TRAIL_ACTIVATION = 0.5        # Activate after 0.5x ATR in profit

# --- Execution Analytics ---
EXECUTION_ANALYTICS_ENABLED = True
EXECUTION_SLIPPAGE_ALERT_PCT = 0.001

# --- T7-001: RL Execution Agent ---
RL_EXECUTION_ENABLED = False            # Enable deep RL execution agent

# --- T7-003: EDGAR 8-K Monitor ---
EDGAR_MONITOR_ENABLED = False           # Enable real-time 8-K filing monitor

# --- T7-005: Black-Litterman Portfolio Optimization ---
BLACK_LITTERMAN_ENABLED = True          # V11.4: Enable BL portfolio-level optimization

# --- Scan Configuration ---
SCAN_INTERVAL_SEC = 120
# MED-030: Removed CLOSE_UNKNOWN_POSITIONS (dead code, never referenced)

# --- Overnight Holds ---
OVERNIGHT_HOLD_ENABLED = True
OVERNIGHT_MAX_POSITIONS = 4
OVERNIGHT_MIN_PROFIT_PCT = 0.003    # Must be 0.3% in profit to hold
OVERNIGHT_SIZE_REDUCTION = 0.40     # Sell 40% at close
OVERNIGHT_GAP_STOP_PCT = 0.01      # Close if gaps against > 1%
OVERNIGHT_ELIGIBLE_STRATEGIES = ["PEAD", "STAT_MR", "KALMAN_PAIRS"]

# ============================================================
# DATA & FILTERING
# ============================================================

# --- Earnings & News ---
EARNINGS_FILTER_DAYS = 2           # Skip symbols with earnings within 2 days
NEWS_SENTIMENT_ENABLED = True
NEWS_CACHE_TTL_MIN = 30

# --- Multi-Timeframe Confirmation ---
MTF_CONFIRMATION_ENABLED = True
MTF_CACHE_SECONDS = 300
MTF_ENABLED_FOR = {
    'STAT_MR':      False,
    'VWAP':         False,
    'KALMAN_PAIRS': False,
    'ORB':          True,
    'MICRO_MOM':    True,
    'GAP_GO':       False,
}

# --- Multi-Timeframe Confluence ---
MTF_CONFLUENCE_ENABLED = True
MTF_MIN_CONFLUENCE_BREAKOUT = 0.66
MTF_MAX_CONFLUENCE_MEANREV = 0.33

# --- ADX Trend Strength Filter ---
ORB_ADX_FILTER_ENABLED = True
ORB_ADX_MIN = 25
ORB_ADX_PERIOD = 14
VWAP_ADX_FILTER_ENABLED = True
VWAP_ADX_MAX = 20

# --- OBV Divergence ---
OBV_DIVERGENCE_ENABLED = True
OBV_CONFIDENCE_BOOST = 0.1
OBV_CONFIDENCE_PENALTY = -0.1
OBV_LOOKBACK = 20

# --- Intraday Seasonality ---
INTRADAY_SEASONALITY_ENABLED = True
SEASONALITY_OPEN_AUCTION_BLOCK = True   # Skip first 15 min entirely (wide spreads)
SEASONALITY_ADAPTIVE_LEARNING = True    # Learn from own trade history
SEASONALITY_LEARNING_LOOKBACK = 60      # trading days

# --- Cross-Asset Signals ---
CROSS_ASSET_ENABLED = True
CROSS_ASSET_UPDATE_INTERVAL = 900   # 15 minutes
CROSS_ASSET_FLIGHT_REDUCTION = 0.30  # Reduce to 30% sizing on flight-to-safety
CROSS_ASSET_CREDIT_STRESS_THRESHOLD = 0.70

# --- Data Quality ---
DATA_QUALITY_ENABLED = True
DATA_QUALITY_MAX_STALENESS_SEC = 300
DATA_QUALITY_MAX_SINGLE_MOVE = 0.15
DATA_QUALITY_MIN_BARS = 50

# --- Data Caching ---
DATA_CACHE_ENABLED = True
DATA_CACHE_MAX_SIZE = 500

# --- Signal Ranking & Alpha Decay ---
SIGNAL_RANKING_ENABLED = True
ALPHA_DECAY_ENABLED = True
ALPHA_DECAY_CRITICAL_SHARPE = 0.3
ALPHA_DECAY_WARNING_SHARPE = 0.5
ALPHA_DECAY_AUTO_DEMOTE = True

# --- LLM Signal Scoring ---
LLM_SCORING_ENABLED    = os.getenv('LLM_SCORING_ENABLED', 'false') == 'true'
LLM_SCORE_THRESHOLD    = 0.45
LLM_SCORE_SIZE_MULT    = True
LLM_MAX_DAILY_COST_USD = 0.10
ANTHROPIC_API_KEY      = os.getenv('ANTHROPIC_API_KEY', '')

# --- Walk-Forward Validation ---
WALK_FORWARD_ENABLED    = True
WALK_FORWARD_MIN_SHARPE = 0.3

# --- Sortino Ratio ---
SORTINO_ENABLED = True
WALK_FORWARD_MIN_SORTINO = 0.5

# --- T5-011: Temporal Fusion Transformer ---
TFT_ENABLED = os.getenv("TFT_ENABLED", "false") == "true"

# --- T5-012: LLM Multi-Agent Alpha Mining ---
ALPHA_AGENTS_ENABLED = os.getenv("ALPHA_AGENTS_ENABLED", "false") == "true"

# --- T5-013: Cross-Asset Lead-Lag Signal ---
LEAD_LAG_ENABLED = os.getenv("LEAD_LAG_ENABLED", "true") == "true"

# --- T5-014: Order Book Microstructure Signal ---
ORDER_BOOK_SIGNAL_ENABLED = os.getenv("ORDER_BOOK_SIGNAL_ENABLED", "false") == "true"

# --- T5-015: Adaptive RL-Informed Scan Scheduling ---
ADAPTIVE_SCAN_ENABLED = os.getenv("ADAPTIVE_SCAN_ENABLED", "false") == "true"

# ============================================================
# MONITORING & ALERTS
# ============================================================

# --- Telegram ---
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false") == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- Web Dashboard ---
WEB_DASHBOARD_ENABLED = os.getenv("WEB_DASHBOARD_ENABLED", "true") == "true"
WEB_DASHBOARD_PORT = int(os.getenv("WEB_DASHBOARD_PORT", "8080"))
CORS_ORIGINS = ["http://localhost:3000"]
TRUSTED_PROXY_IPS: list[str] = []  # MED-034: IPs allowed to set X-Forwarded-For

# --- WebSocket Position Monitoring ---
WEBSOCKET_MONITORING = True
WEBSOCKET_RECONNECT_SEC = 5

# ============================================================
# REPLAY & A/B TESTING
# ============================================================

REPLAY_ENABLED = True
AB_TESTING_ENABLED = True
REPLAY_DATA_RETENTION_DAYS = 90

# ============================================================
# PERSISTENCE & BACKTESTING
# ============================================================

STATE_FILE = "state.json"
DB_FILE = "bot.db"
LOG_FILE = "bot.log"
AUDIT_LOG_FILE = "audit.log"
STATE_SAVE_INTERVAL_SEC = 60

# T4-004: Connection pool tuning — increased from 3 to 10 for market hours throughput
DB_POOL_SIZE = 10
DB_POOL_TIMEOUT = 5.0   # seconds to wait for a connection before overflow

# --- Watchdog & Reconciliation ---
WATCHDOG_ENABLED = True
WATCHDOG_CHECK_INTERVAL = 300        # 5 minutes
RECONCILIATION_ENABLED = True
RECONCILIATION_INTERVAL = 1800       # 30 minutes

# --- Structured Audit Trail ---
STRUCTURED_LOGGING_ENABLED = True
AUDIT_TRAIL_RETENTION_DAYS = 365

BACKTEST_SLIPPAGE = 0.0005         # 0.05% slippage per trade
BACKTEST_COMMISSION = 0.0035       # $0.0035 per share
BACKTEST_RISK_FREE_RATE = 0.045    # 4.5% annual
BACKTEST_TOP_N = 20                # Run on top 20 most liquid symbols

# --- Parameter Optimization ---
PARAM_OPTIMIZER_ENABLED = True
PARAM_OPTIMIZER_DAY = "sunday"
PARAM_OPTIMIZER_TRIALS = 100
PARAM_OPTIMIZER_MIN_IMPROVEMENT = 0.15   # 15% Sortino improvement required
PARAM_OPTIMIZER_APPLY_AUTO = False       # Manual approval by default

# ============================================================
# RUNTIME PARAMETERS
# ============================================================

import threading as _threading

_runtime_params: dict = {}
_runtime_lock = _threading.Lock()  # V10 CONFIG-001: Thread-safe access

def get_param(key: str, default=None):
    """Get a runtime parameter (optimizer-modified or config default)."""
    with _runtime_lock:
        return _runtime_params.get(key, default)

def set_param(key: str, value):
    """Set a runtime parameter (used by optimizer)."""
    with _runtime_lock:
        _runtime_params[key] = value


# ============================================================
# PROD-013: YAML Config Support with Hot Reload
# ============================================================

_yaml_config: dict = {}
_yaml_lock = _threading.Lock()
_yaml_watcher_started = False
_yaml_last_mtime: float = 0.0

YAML_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_yaml_config(path: str | None = None) -> dict:
    """PROD-013: Load configuration overrides from a YAML file.

    If a config.yaml exists, loads settings from it and applies them
    as overrides to the module-level variables. Settings in YAML take
    precedence over defaults but environment variables still win.

    Args:
        path: Path to YAML file. Defaults to config.yaml in project root.

    Returns:
        Dict of loaded YAML settings, or empty dict if file not found.
    """
    import logging as _logging
    _logger = _logging.getLogger("config")

    yaml_path = path or YAML_CONFIG_PATH

    if not os.path.exists(yaml_path):
        _logger.debug("PROD-013: No config.yaml found at %s", yaml_path)
        return {}

    try:
        import yaml
    except ImportError:
        _logger.warning(
            "PROD-013: PyYAML not installed — cannot load config.yaml. "
            "Install with: pip install PyYAML"
        )
        return {}

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        global _yaml_config
        with _yaml_lock:
            _yaml_config = data

        # Apply YAML overrides to module globals
        applied = _apply_yaml_overrides(data)
        _logger.info(
            "PROD-013: Loaded config.yaml (%d settings, %d applied)",
            len(data), applied,
        )
        return data

    except Exception as e:
        _logger.error("PROD-013: Failed to load config.yaml: %s", e)
        return {}


def _apply_yaml_overrides(data: dict) -> int:
    """Apply YAML config values as overrides to module globals.

    Only overrides values that:
    - Exist as module-level attributes in config.py
    - Are not environment-variable-controlled (API keys, etc.)
    - Match the expected type

    Returns count of applied overrides.
    """
    import logging as _logging
    _logger = _logging.getLogger("config")

    # Keys that should NEVER be overridden from YAML (security-sensitive)
    _protected = {"API_KEY", "API_SECRET", "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN"}

    import config as _self
    applied = 0

    for key, value in data.items():
        if key.startswith("_") or key in _protected:
            continue

        if not hasattr(_self, key):
            _logger.debug("PROD-013: Ignoring unknown YAML key: %s", key)
            continue

        current = getattr(_self, key)

        # Type check (allow int<->float coercion)
        if current is not None and not isinstance(value, type(current)):
            if isinstance(current, float) and isinstance(value, int):
                value = float(value)
            elif isinstance(current, int) and isinstance(value, float) and value == int(value):
                value = int(value)
            else:
                _logger.warning(
                    "PROD-013: Type mismatch for %s: expected %s, got %s — skipping",
                    key, type(current).__name__, type(value).__name__,
                )
                continue

        setattr(_self, key, value)
        applied += 1
        _logger.debug("PROD-013: Applied YAML override: %s = %r", key, value)

    return applied


def start_yaml_watcher(poll_interval_sec: float = 30.0):
    """PROD-013: Start a background thread that polls config.yaml for changes.

    Uses simple file mtime polling (no external dependencies required).
    When a change is detected, the YAML config is reloaded and overrides
    are re-applied.

    Args:
        poll_interval_sec: How often to check for file changes (default 30s).
    """
    import logging as _logging
    _logger = _logging.getLogger("config")

    global _yaml_watcher_started, _yaml_last_mtime

    if _yaml_watcher_started:
        _logger.debug("PROD-013: YAML watcher already running")
        return

    yaml_path = YAML_CONFIG_PATH
    if os.path.exists(yaml_path):
        _yaml_last_mtime = os.path.getmtime(yaml_path)

    def _watcher_loop():
        global _yaml_last_mtime
        while True:
            _threading.Event().wait(timeout=poll_interval_sec)
            try:
                if not os.path.exists(yaml_path):
                    continue
                mtime = os.path.getmtime(yaml_path)
                if mtime > _yaml_last_mtime:
                    _yaml_last_mtime = mtime
                    _logger.info("PROD-013: config.yaml changed, reloading...")
                    load_yaml_config(yaml_path)
            except Exception as e:
                _logger.warning("PROD-013: YAML watcher error: %s", e)

    watcher = _threading.Thread(
        target=_watcher_loop,
        name="YAMLConfigWatcher",
        daemon=True,
    )
    watcher.start()
    _yaml_watcher_started = True
    _logger.info(
        "PROD-013: YAML config watcher started (poll_interval=%ds)",
        poll_interval_sec,
    )


# =============================================================================
# T2-002: Structured Config Shim
# =============================================================================
#
# config.py is the legacy flat-variable entry point (44 files import it).
# config/settings.py is the canonical source with structured dataclass groups.
# This shim re-exports get_settings() so callers can migrate incrementally:
#
#   import config
#   settings = config.get_settings()   # typed, grouped access
#   config.MAX_POSITIONS               # legacy flat access (still works)

try:
    from config.settings import (
        get_settings,
        invalidate_settings_cache,
        RiskSettings,
        StrategySettings,
        MLSettings,
        ExecutionSettings,
        OperationalSettings,
        Settings,
    )
except ImportError:
    # Fallback if config/settings.py is not yet available (e.g., during tests)
    def get_settings():
        return None
    def invalidate_settings_cache():
        pass


def get_yaml_config() -> dict:
    """Get the currently loaded YAML config (read-only copy)."""
    with _yaml_lock:
        return dict(_yaml_config)


def validate():
    """Validate configuration at startup. Call from main.py before trading starts.

    Raises RuntimeError if critical configuration errors are found.
    """
    errors, warnings = validate_config()

    import logging as _logging
    _logger = _logging.getLogger("config")

    for w in warnings:
        _logger.warning("CONFIG WARNING: %s", w)

    if errors:
        for e in errors:
            _logger.error("CONFIG ERROR: %s", e)
        raise RuntimeError(f"Configuration validation failed with {len(errors)} error(s). See log.")

    _logger.info("Configuration validated successfully (%d warnings)", len(warnings))


def validate_config() -> tuple[list[str], list[str]]:
    """PROD-006: Comprehensive configuration validation.

    Checks:
    - All required parameters exist and have correct types.
    - Numeric ranges are valid (allocations sum <= 1.0, risk limits > 0, etc.).
    - Warns on suspicious but non-fatal values.

    Returns:
        Tuple of (errors: list[str], warnings: list[str]).
        Errors are fatal; warnings are logged but non-blocking.
    """
    import logging as _logging
    _logger = _logging.getLogger("config")
    errors: list[str] = []
    warnings: list[str] = []

    # --- API credentials (skip in test mode) ---
    if not os.getenv("TESTING") and not os.getenv("PYTEST_CURRENT_TEST"):
        if not API_KEY or not API_SECRET:
            errors.append("ALPACA_API_KEY and ALPACA_API_SECRET must be set")

    # --- Strategy allocations sum <= 1.0 ---
    total_alloc = sum(STRATEGY_ALLOCATIONS.values())
    if total_alloc > 1.0 + 1e-6:
        errors.append(
            f"Strategy allocations sum to {total_alloc:.6f} ({total_alloc:.2%}), must be <= 100%"
        )
    if total_alloc < 0.5:
        warnings.append(
            f"Strategy allocations sum to only {total_alloc:.2%} — "
            f"more than half of capital may remain undeployed"
        )

    # --- Individual allocation values ---
    for strat, weight in STRATEGY_ALLOCATIONS.items():
        if not isinstance(weight, (int, float)):
            errors.append(f"STRATEGY_ALLOCATIONS['{strat}'] must be numeric, got {type(weight).__name__}")
        elif weight < 0:
            errors.append(f"STRATEGY_ALLOCATIONS['{strat}'] = {weight} — cannot be negative")
        elif weight > 0.5:
            warnings.append(
                f"STRATEGY_ALLOCATIONS['{strat}'] = {weight:.2%} — "
                f"single strategy >50% creates concentration risk"
            )

    # --- Theoretical max deployment vs limit ---
    n_strategies = len(STRATEGY_ALLOCATIONS)
    theoretical_max_deploy = n_strategies * MAX_POSITION_PCT
    if theoretical_max_deploy > MAX_PORTFOLIO_DEPLOY:
        warnings.append(
            f"{n_strategies} strategies * {MAX_POSITION_PCT:.0%} max position = "
            f"{theoretical_max_deploy:.0%} theoretical max deployment, exceeds "
            f"MAX_PORTFOLIO_DEPLOY={MAX_PORTFOLIO_DEPLOY:.0%}. "
            f"Risk layer will enforce the deploy limit, but strategies may compete for capital."
        )

    # --- Risk limits must be positive ---
    _positive_checks = {
        "RISK_PER_TRADE_PCT": RISK_PER_TRADE_PCT,
        "MAX_POSITION_PCT": MAX_POSITION_PCT,
        "MIN_POSITION_VALUE": MIN_POSITION_VALUE,
        "MAX_POSITIONS": MAX_POSITIONS,
        "VOL_TARGET_DAILY": VOL_TARGET_DAILY,
        "SCAN_INTERVAL_SEC": SCAN_INTERVAL_SEC,
    }
    for name, value in _positive_checks.items():
        if not isinstance(value, (int, float)):
            errors.append(f"{name} must be numeric, got {type(value).__name__}")
        elif value <= 0:
            errors.append(f"{name} = {value} — must be > 0")

    # --- Range checks ---
    if RISK_PER_TRADE_PCT > 0.10:
        errors.append(f"RISK_PER_TRADE_PCT ({RISK_PER_TRADE_PCT}) > 10% — dangerously high")

    if MAX_POSITION_PCT > 0.50:
        errors.append(f"MAX_POSITION_PCT ({MAX_POSITION_PCT}) > 50% — dangerously high")

    if SCAN_INTERVAL_SEC < 5:
        errors.append(f"SCAN_INTERVAL_SEC ({SCAN_INTERVAL_SEC}) too low — minimum 5s")

    if MAX_PORTFOLIO_DEPLOY > 1.0:
        errors.append(f"MAX_PORTFOLIO_DEPLOY ({MAX_PORTFOLIO_DEPLOY}) > 100% — invalid")

    # --- Kelly criterion bounds ---
    if KELLY_MIN_RISK >= KELLY_MAX_RISK:
        errors.append(f"KELLY_MIN_RISK ({KELLY_MIN_RISK}) must be < KELLY_MAX_RISK ({KELLY_MAX_RISK})")

    if KELLY_FRACTION_MULT <= 0 or KELLY_FRACTION_MULT > 1.0:
        warnings.append(
            f"KELLY_FRACTION_MULT ({KELLY_FRACTION_MULT}) — "
            f"expected 0 < value <= 1.0 (half-Kelly = 0.5)"
        )

    # --- Daily P&L controls ---
    if PNL_LOSS_HALT_PCT >= 0:
        errors.append(f"PNL_LOSS_HALT_PCT ({PNL_LOSS_HALT_PCT}) must be negative (it's a loss threshold)")

    if DAILY_LOSS_HALT >= 0:
        errors.append(f"DAILY_LOSS_HALT ({DAILY_LOSS_HALT}) must be negative")

    if PNL_GAIN_LOCK_PCT <= 0:
        warnings.append(f"PNL_GAIN_LOCK_PCT ({PNL_GAIN_LOCK_PCT}) <= 0 — gain lock effectively disabled")

    # --- Volatility targeting ---
    if VOL_SCALAR_MIN >= VOL_SCALAR_MAX:
        errors.append(
            f"VOL_SCALAR_MIN ({VOL_SCALAR_MIN}) must be < VOL_SCALAR_MAX ({VOL_SCALAR_MAX})"
        )

    if VOL_TARGET_MAX < VOL_TARGET_DAILY:
        errors.append(
            f"VOL_TARGET_MAX ({VOL_TARGET_MAX}) must be >= VOL_TARGET_DAILY ({VOL_TARGET_DAILY})"
        )

    # --- VIX thresholds ---
    if VIX_HALT_THRESHOLD < 20:
        warnings.append(
            f"VIX_HALT_THRESHOLD ({VIX_HALT_THRESHOLD}) < 20 — "
            f"will halt trading during normal market conditions"
        )

    # --- Suspicious values ---
    if BACKTEST_RISK_FREE_RATE > 0.10:
        warnings.append(
            f"BACKTEST_RISK_FREE_RATE ({BACKTEST_RISK_FREE_RATE}) > 10% — verify this is correct"
        )

    if CORRELATION_THRESHOLD < 0.5:
        warnings.append(
            f"CORRELATION_THRESHOLD ({CORRELATION_THRESHOLD}) < 50% — "
            f"may block too many trades"
        )

    if MAX_PORTFOLIO_DEPLOY < 0.2:
        warnings.append(
            f"MAX_PORTFOLIO_DEPLOY ({MAX_PORTFOLIO_DEPLOY}) < 20% — "
            f"very conservative, most capital will sit idle"
        )

    # --- Type checks for key string params ---
    _string_checks = {
        "DB_FILE": DB_FILE,
        "LOG_FILE": LOG_FILE,
        "STATE_FILE": STATE_FILE,
    }
    for name, value in _string_checks.items():
        if not isinstance(value, str) or not value:
            errors.append(f"{name} must be a non-empty string")

    # --- Circuit breaker tiers (validate if module available) ---
    try:
        from risk.circuit_breaker import TieredCircuitBreaker
        TieredCircuitBreaker()
    except ImportError:
        pass  # Module not yet available during early startup
    except Exception as e:
        warnings.append(f"Circuit breaker validation failed: {e}")

    return errors, warnings
