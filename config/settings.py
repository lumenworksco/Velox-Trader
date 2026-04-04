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
# Validate API credentials at import time (skip in test environments)
if not os.getenv("TESTING") and not os.getenv("PYTEST_CURRENT_TEST"):
    if not API_KEY:
        raise RuntimeError("ALPACA_API_KEY environment variable is required")
    if not API_SECRET:
        raise RuntimeError("ALPACA_API_SECRET environment variable is required")

ALLOW_SHORT = os.getenv("ALLOW_SHORT", "false") == "true"

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
    "CHPT", "BE",
    # V12 AUDIT: Removed illiquid micro-caps: QUBT, IONQ, RGTI, SOUN, ARQQ, BBAI, RKLB, LUNR
    "ASTS", "RDW", "DUOL", "CAVA", "BROS", "SHAK",
    "WING", "TXRH", "CMG", "DPZ", "DNUT", "JACK",
]

# Full universe
SYMBOLS = CORE_SYMBOLS + LARGE_CAP_GROWTH + SECTOR_ETFS + HIGH_MOMENTUM_MIDCAPS

# Leveraged ETFs — ONLY use VWAP on these, never ORB or Momentum
# V11 REMOVE-006: For earnings exemption, prefer dynamic check via
# Alpaca asset classification API (data.universe.DynamicUniverse).
# This static list is kept as fallback for leveraged ETF detection.
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
    "WDAY": "XLK", "ANSS": "XLK", "SMCI": "XLK", "APP": "XLK",
    # V12 AUDIT: Removed IONQ, RGTI, QUBT, ARQQ, BBAI, SOUN (illiquid micro-caps)
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
    "AXON": "XLI", "ASTS": "XLI", "RDW": "XLI",
    # V12 AUDIT: Removed RKLB, LUNR (illiquid micro-caps)
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
    'STAT_MR': 0.25,         # V12 AUDIT: Reduced from 0.35 — crowded edge, high alpha decay
    'VWAP': 0.13,            # V12 AUDIT: Reduced from 0.20 — overlaps with STAT_MR
    'KALMAN_PAIRS': 0.27,    # V12 AUDIT: Increased from 0.20 — uncorrelated, diversifying
    'ORB': 0.12,             # V12 AUDIT: Increased from 0.10 — good risk mgmt
    'MICRO_MOM': 0.05,       # V12 AUDIT: Reduced from 0.10 — weak signals, high alpha decay
    'PEAD': 0.18,            # V12 AUDIT: Increased from 0.05 — orthogonal, proven academic edge
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
VWAP_MAX_SPREAD_PCT  = 0.0015     # Skip if bid-ask spread > 0.15% — tighter quality filter
VWAP_VOLUME_RATIO    = 1.0        # Volume ratio vs 20-bar average — require above-average volume
MAX_INTRADAY_MOVE_PCT = 0.03      # Skip if stock moved > 3% today
VWAP_BAND_STD        = 1.5        # Standard deviation multiplier for VWAP bands — tighter bands, more entries
VWAP_RSI_OVERSOLD    = 45         # RSI below this = oversold (buy signal, raised from 40)
VWAP_RSI_OVERBOUGHT  = 70         # RSI above this = overbought (short signal)
VWAP_CONFIRMATION_BARS = 1        # Bars confirming bounce (1 = disabled)
VWAP_STOP_EXTENSION  = 0.5        # Stop extension beyond band (in std devs)
VWAP_MIN_STOP_PCT    = 0.01       # Minimum 1.0% stop distance (was 0.5%)

# --- Kalman Pairs Trading ---
PAIRS_ZSCORE_ENTRY = 1.5
PAIRS_ZSCORE_EXIT = 0.2
PAIRS_ZSCORE_STOP = 3.0
PAIRS_MAX_HOLD_DAYS = 10
PAIRS_MAX_ACTIVE = 15
PAIRS_MIN_CORRELATION = 0.80       # Tightened from 0.70 — higher quality pairs only
PAIRS_COINT_PVALUE = 0.05          # Tightened from 0.10 — require stronger cointegration
KALMAN_DELTA = 1e-4
KALMAN_OBS_NOISE = 0.001
PAIRS_TP_PCT = 0.015   # V10: 1.5% take-profit (was 0.5% — negative EV after costs)
PAIRS_SL_PCT = 0.010   # V10: 1.0% stop-loss (was 1.5% — gives 1.5:1 R/R)

# --- Opening Range Breakout ---
ORB_ENABLED          = True
ORB_VOLUME_RATIO     = 2.0        # Volume confirmation ratio — require strong volume on breakout
ORB_MAX_GAP_PCT      = 0.04       # Skip if gap > 4%
ORB_MAX_RANGE_PCT    = 0.035      # Skip if range > 3.5%
ORB_MIN_STOP_PCT     = 0.008      # Minimum 0.8% stop distance (was 0.3%)
ORB_SCAN_SYMBOLS     = 15         # Top N by morning volume
ORB_ACTIVE_UNTIL     = time(12, 30)   # Extended from 11:30 — more signal window
ORB_BREAKOUT_BUFFER  = 0.0025     # 0.25% confirmation buffer above/below ORB range — reduce fakeouts
ORB_TP_MULT          = 2.0        # Take profit = entry ± 2.0x ORB range — wider target
ORB_SL_MULT          = 0.7        # Stop loss = entry ∓ 0.7x ORB range — wider stop, fewer whipsaws
ORB_TIME_STOP_HOURS  = 2          # Close after 2 hours

# --- Micro Momentum ---
MICRO_SPY_VOL_SPIKE_MULT = 2.0     # Tightened from 1.5 — require stronger volume spike
MICRO_SPY_MIN_MOVE_PCT = 0.0015    # Tightened from 0.0008 — 0.15% SPY move triggers event
MICRO_MAX_HOLD_MINUTES = 20          # Reduced from 30 to 20 — shorter hold, capture fast moves
MICRO_STOP_PCT = 0.01                # Was 0.003 — 1% stop survives noise on beta=2 stocks
MICRO_TARGET_PCT = 0.02              # Was 0.006 — 2% target keeps 2:1 R/R
MICRO_MAX_TRADES_PER_EVENT = 4       # Increased from 2 — capture more momentum on confirmed events
MICRO_MAX_DAILY_GAIN_DISABLE = 0.015
MICRO_TOP_BETA_STOCKS = 5
MICRO_EVENT_COOLDOWN_SEC = 900    # 15-min cooldown between events
MICRO_EVENT_WINDOW_SEC = 300      # 5-min window after event for trades
# V11 REMOVE-001: Hardcoded beta table — kept as FALLBACK only.
# Prefer dynamic beta from risk.factor_model.FactorRiskModel which
# computes 60-day rolling regression against SPY daily at market open.
MICRO_BETA_TABLE = {
    'NVDA': 1.6, 'AMD': 1.5, 'TSLA': 1.8, 'COIN': 2.0, 'SOFI': 1.6,
    'PLTR': 1.5, 'HOOD': 1.8, 'AFRM': 1.7, 'SMCI': 2.0, 'SNOW': 1.4,
    'DDOG': 1.3, 'NET': 1.4, 'SQ': 1.5, 'META': 1.3, 'NFLX': 1.2,
    'GOOGL': 1.15, 'AMZN': 1.2, 'AAPL': 1.1, 'MSFT': 1.05,
    'CRWD': 1.3, 'PANW': 1.2, 'ZS': 1.2, 'RBLX': 1.4,
    'CELH': 1.3, 'AXON': 1.2, 'ENPH': 1.5, 'FSLR': 1.4,
    # V12 FINAL: Removed IONQ, RGTI, QUBT, RKLB (dropped from universe)
    'LYFT': 1.5, 'UBER': 1.3, 'ABNB': 1.4, 'DASH': 1.3,
}

# --- Post-Earnings Announcement Drift ---
PEAD_ENABLED = True
PEAD_MIN_SURPRISE_PCT = 3.0        # Lowered from 5.0 — capture more earnings plays
PEAD_MIN_VOLUME_RATIO = 2.0       # Tightened from 1.5 — require strong volume confirmation
PEAD_HOLD_DAYS_MIN = 3             # Minimum hold = 3 days (capture initial drift)
PEAD_HOLD_DAYS_MAX = 10            # Maximum hold = 10 days (most drift captured by then)
PEAD_TAKE_PROFIT = 0.05
PEAD_STOP_LOSS = 0.02              # Tightened from 3% to 2% — cut losses faster
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

# --- V12 14.4: Alternative Data Sources (Free) ---
EDGAR_MONITOR_ENABLED = os.getenv("EDGAR_MONITOR_ENABLED", "true") == "true"
MACRO_SURPRISE_ENABLED = os.getenv("MACRO_SURPRISE_ENABLED", "true") == "true"
SHORT_INTEREST_ENABLED = os.getenv("SHORT_INTEREST_ENABLED", "false") == "true"  # requires API key

# --- V12 BONUS: Profit Maximization ---
INTRADAY_VOL_REGIME_ENABLED = os.getenv("INTRADAY_VOL_REGIME_ENABLED", "true") == "true"
WIN_STREAK_SIZING_ENABLED = os.getenv("WIN_STREAK_SIZING_ENABLED", "true") == "true"
CONVICTION_PYRAMIDING_ENABLED = os.getenv("CONVICTION_PYRAMIDING_ENABLED", "false") == "true"
DYNAMIC_STOP_TIGHTENING_ENABLED = os.getenv("DYNAMIC_STOP_TIGHTENING_ENABLED", "true") == "true"

# --- V12 AUDIT: ML Core Feature Selection ---
ML_CORE_FEATURES_ONLY = True  # Reduce ML features from 200+ to ~50 high-signal features

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
RISK_PER_TRADE_PCT = 0.008         # V12 AUDIT: Increased from 0.005 — 0.8% risk per trade
MAX_POSITION_PCT = 0.08            # V12 AUDIT: Increased from 0.05 — 8% max per position
MIN_POSITION_VALUE = 100            # Min $100 per trade
MAX_POSITIONS = 12
MAX_PORTFOLIO_DEPLOY = 0.55
DAILY_LOSS_HALT = -0.025

# V12 AUDIT: Position sizing multiplier cascade floor
POSITION_SIZE_MULTIPLIER_FLOOR = 0.30  # Never reduce position by more than 70%

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
PNL_GAIN_LOCK_SIZE_MULT = 0.70

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
MAX_SECTOR_EXPOSURE = 0.30         # V12 6.2: Max sector weight as fraction of portfolio (30%)
MAX_ACCEPTABLE_DRAWDOWN = 0.08     # V12 6.3: Drawdown-based sizing denominator (8%)
API_FAILURE_CIRCUIT_BREAKER_COUNT = 5   # V12 6.4: Consecutive failures to trigger kill switch
API_FAILURE_CIRCUIT_BREAKER_WINDOW = 300  # V12 6.4: Window in seconds (5 minutes)

# V12 FINAL: VIX spike detection for circuit breaker escalation
VIX_SPIKE_WINDOW_SEC = 900          # 15-minute rolling window
VIX_SPIKE_THRESHOLD_PCT = 0.20      # 20% VIX rise triggers ORANGE escalation

# V12 FINAL: Periodic refresh intervals
CORRELATION_REFRESH_INTERVAL_SEC = 3600   # Refresh correlation matrix every 60 min
CORP_ACTION_CHECK_INTERVAL_SEC = 1800     # Check corporate actions every 30 min

# --- Re-entry Cooldown ---
REENTRY_COOLDOWN_MIN = 30          # Block re-entry for 30 min after stop-loss (was 15 — too short)

# --- Per-Symbol Daily Loss Cap ---
MAX_SYMBOL_DAILY_LOSS = 200.0      # Max $200 loss per symbol per day

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

# --- Smart Order Routing ---
SMART_ROUTING_ENABLED = True
SPREAD_THRESHOLD_PCT = 0.0015     # Use limit if spread > 0.15%
CHASE_AFTER_SECONDS = 60
CHASE_CONVERT_MARKET_AFTER = 120
ADAPTIVE_TWAP_ENABLED = True

# --- Exit Management ---
ADVANCED_EXITS_ENABLED = False     # Superseded by ADAPTIVE_EXITS_ENABLED
SCALED_TP_ENABLED = False          # Superseded by adaptive exits
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

# --- Kill Switch ---
KILL_SWITCH_BATCH_SIZE = 5         # MED-031: Positions to close per batch
KILL_SWITCH_BATCH_DELAY_SEC = 0.5  # MED-031: Delay between batches (seconds)

# --- Transaction Cost Model ---
COST_SPREAD_BPS = 1.0              # MED-035: Default spread cost (basis points)
COST_SLIPPAGE_BPS = 0.5            # MED-035: Default slippage cost (basis points)
COST_COMMISSION_PER_SHARE = 0.0035 # MED-035: Commission per share
COST_MIN_EXPECTED_RETURN_BPS = 5.0 # MED-035: Minimum expected return to justify trade

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


# ============================================================
# PROD-013: YAML Config Support with Hot Reload
# ============================================================

_yaml_config: dict = {}
_yaml_lock = _threading.Lock()
_yaml_watcher_started = False
_yaml_last_mtime: float = 0.0

YAML_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")


def load_yaml_config(path: str | None = None) -> dict:
    """PROD-013: Load configuration overrides from a YAML file.

    If a config.yaml exists in the project root, loads settings from it
    and applies them as overrides to the module-level variables. Settings in
    YAML take precedence over defaults but environment variables still win.

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

    Only overrides values that exist as module-level attributes and
    are not security-sensitive (API keys, etc.).

    Returns count of applied overrides.
    """
    import logging as _logging
    _logger = _logging.getLogger("config")
    import sys

    _protected = {"API_KEY", "API_SECRET", "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN"}
    _self = sys.modules[__name__]
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
# T2-002: Structured Config Groups
# =============================================================================
#
# These dataclasses provide typed, grouped access to settings. They are populated
# from the module-level variables above, so config.yaml / env overrides are
# reflected automatically.  The legacy flat-variable interface remains canonical
# — these groups are a read-only convenience layer.

from dataclasses import dataclass as _dataclass, field as _field


@_dataclass(frozen=True)
class RiskSettings:
    """Grouped risk management parameters."""
    risk_per_trade_pct: float = 0.0
    max_position_pct: float = 0.0
    min_position_value: float = 0.0
    max_positions: int = 0
    max_portfolio_deploy: float = 0.0
    daily_loss_halt: float = 0.0
    vol_target_daily: float = 0.0
    vol_target_max: float = 0.0
    vol_scalar_min: float = 0.0
    vol_scalar_max: float = 0.0
    kelly_enabled: bool = False
    kelly_fraction_mult: float = 0.0
    kelly_min_risk: float = 0.0
    kelly_max_risk: float = 0.0
    pnl_gain_lock_pct: float = 0.0
    pnl_loss_halt_pct: float = 0.0
    vix_risk_scaling_enabled: bool = False
    vix_halt_threshold: int = 40
    beta_max_abs: float = 0.0
    correlation_threshold: float = 0.0


@_dataclass(frozen=True)
class StrategySettings:
    """Grouped strategy configuration."""
    strategy_allocations: dict = _field(default_factory=dict)
    mr_zscore_entry: float = 0.0
    mr_zscore_stop: float = 0.0
    vwap_band_std: float = 0.0
    vwap_min_stop_pct: float = 0.0
    pairs_zscore_entry: float = 0.0
    pairs_max_hold_days: int = 0
    orb_enabled: bool = False
    micro_max_hold_minutes: int = 0
    pead_enabled: bool = False


@_dataclass(frozen=True)
class MLSettings:
    """Grouped ML / analytics parameters."""
    hmm_regime_enabled: bool = False
    hmm_n_states: int = 5
    llm_scoring_enabled: bool = False
    llm_score_threshold: float = 0.0
    signal_ranking_enabled: bool = False
    alpha_decay_enabled: bool = False
    walk_forward_enabled: bool = False
    intraday_seasonality_enabled: bool = False
    cross_asset_enabled: bool = False


@_dataclass(frozen=True)
class ExecutionSettings:
    """Grouped execution / routing parameters."""
    smart_routing_enabled: bool = False
    spread_threshold_pct: float = 0.0
    adaptive_twap_enabled: bool = False
    scan_interval_sec: int = 0
    overnight_hold_enabled: bool = False
    overnight_max_positions: int = 0
    cost_spread_bps: float = 0.0
    cost_slippage_bps: float = 0.0


@_dataclass(frozen=True)
class OperationalSettings:
    """Grouped operational / monitoring parameters."""
    paper_mode: bool = True
    telegram_enabled: bool = False
    web_dashboard_enabled: bool = False
    websocket_monitoring: bool = False
    watchdog_enabled: bool = False
    reconciliation_enabled: bool = False
    structured_logging_enabled: bool = False
    db_file: str = "bot.db"
    log_file: str = "bot.log"


@_dataclass
class Settings:
    """Top-level composite settings object (T2-002)."""
    risk: RiskSettings = _field(default_factory=RiskSettings)
    strategy: StrategySettings = _field(default_factory=StrategySettings)
    ml: MLSettings = _field(default_factory=MLSettings)
    execution: ExecutionSettings = _field(default_factory=ExecutionSettings)
    ops: OperationalSettings = _field(default_factory=OperationalSettings)


def _build_settings() -> Settings:
    """Build a Settings snapshot from current module-level variables."""
    import sys
    _mod = sys.modules[__name__]

    def _g(name, default=None):
        return getattr(_mod, name, default)

    return Settings(
        risk=RiskSettings(
            risk_per_trade_pct=_g("RISK_PER_TRADE_PCT", 0.008),
            max_position_pct=_g("MAX_POSITION_PCT", 0.08),
            min_position_value=_g("MIN_POSITION_VALUE", 100),
            max_positions=_g("MAX_POSITIONS", 12),
            max_portfolio_deploy=_g("MAX_PORTFOLIO_DEPLOY", 0.55),
            daily_loss_halt=_g("DAILY_LOSS_HALT", -0.04),
            vol_target_daily=_g("VOL_TARGET_DAILY", 0.01),
            vol_target_max=_g("VOL_TARGET_MAX", 0.015),
            vol_scalar_min=_g("VOL_SCALAR_MIN", 0.3),
            vol_scalar_max=_g("VOL_SCALAR_MAX", 1.5),
            kelly_enabled=_g("KELLY_ENABLED", True),
            kelly_fraction_mult=_g("KELLY_FRACTION_MULT", 0.5),
            kelly_min_risk=_g("KELLY_MIN_RISK", 0.003),
            kelly_max_risk=_g("KELLY_MAX_RISK", 0.02),
            pnl_gain_lock_pct=_g("PNL_GAIN_LOCK_PCT", 0.015),
            pnl_loss_halt_pct=_g("PNL_LOSS_HALT_PCT", -0.01),
            vix_risk_scaling_enabled=_g("VIX_RISK_SCALING_ENABLED", True),
            vix_halt_threshold=_g("VIX_HALT_THRESHOLD", 40),
            beta_max_abs=_g("BETA_MAX_ABS", 0.3),
            correlation_threshold=_g("CORRELATION_THRESHOLD", 0.92),
        ),
        strategy=StrategySettings(
            strategy_allocations=dict(_g("STRATEGY_ALLOCATIONS", {})),
            mr_zscore_entry=_g("MR_ZSCORE_ENTRY", 1.5),
            mr_zscore_stop=_g("MR_ZSCORE_STOP", 2.5),
            vwap_band_std=_g("VWAP_BAND_STD", 2.0),
            vwap_min_stop_pct=_g("VWAP_MIN_STOP_PCT", 0.01),
            pairs_zscore_entry=_g("PAIRS_ZSCORE_ENTRY", 2.0),
            pairs_max_hold_days=_g("PAIRS_MAX_HOLD_DAYS", 10),
            orb_enabled=_g("ORB_ENABLED", True),
            micro_max_hold_minutes=_g("MICRO_MAX_HOLD_MINUTES", 15),
            pead_enabled=_g("PEAD_ENABLED", True),
        ),
        ml=MLSettings(
            hmm_regime_enabled=_g("HMM_REGIME_ENABLED", True),
            hmm_n_states=_g("HMM_N_STATES", 5),
            llm_scoring_enabled=_g("LLM_SCORING_ENABLED", False),
            llm_score_threshold=_g("LLM_SCORE_THRESHOLD", 0.45),
            signal_ranking_enabled=_g("SIGNAL_RANKING_ENABLED", True),
            alpha_decay_enabled=_g("ALPHA_DECAY_ENABLED", True),
            walk_forward_enabled=_g("WALK_FORWARD_ENABLED", True),
            intraday_seasonality_enabled=_g("INTRADAY_SEASONALITY_ENABLED", True),
            cross_asset_enabled=_g("CROSS_ASSET_ENABLED", True),
        ),
        execution=ExecutionSettings(
            smart_routing_enabled=_g("SMART_ROUTING_ENABLED", True),
            spread_threshold_pct=_g("SPREAD_THRESHOLD_PCT", 0.0015),
            adaptive_twap_enabled=_g("ADAPTIVE_TWAP_ENABLED", True),
            scan_interval_sec=_g("SCAN_INTERVAL_SEC", 120),
            overnight_hold_enabled=_g("OVERNIGHT_HOLD_ENABLED", True),
            overnight_max_positions=_g("OVERNIGHT_MAX_POSITIONS", 4),
            cost_spread_bps=_g("COST_SPREAD_BPS", 1.0),
            cost_slippage_bps=_g("COST_SLIPPAGE_BPS", 0.5),
        ),
        ops=OperationalSettings(
            paper_mode=_g("PAPER_MODE", True),
            telegram_enabled=_g("TELEGRAM_ENABLED", False),
            web_dashboard_enabled=_g("WEB_DASHBOARD_ENABLED", True),
            websocket_monitoring=_g("WEBSOCKET_MONITORING", True),
            watchdog_enabled=_g("WATCHDOG_ENABLED", True),
            reconciliation_enabled=_g("RECONCILIATION_ENABLED", True),
            structured_logging_enabled=_g("STRUCTURED_LOGGING_ENABLED", True),
            db_file=_g("DB_FILE", "bot.db"),
            log_file=_g("LOG_FILE", "bot.log"),
        ),
    )


_settings_cache: Settings | None = None
_settings_cache_lock = _threading.Lock()


def get_settings() -> Settings:
    """T2-002: Return the structured settings object (cached, rebuilt on YAML reload).

    Call this from new code that prefers grouped access over flat variables.
    The underlying module-level variables remain the source of truth.
    """
    global _settings_cache
    with _settings_cache_lock:
        if _settings_cache is None:
            _settings_cache = _build_settings()
        return _settings_cache


def invalidate_settings_cache():
    """Force rebuild of the structured settings object (call after YAML reload)."""
    global _settings_cache
    with _settings_cache_lock:
        _settings_cache = None


def get_yaml_config() -> dict:
    """Get the currently loaded YAML config (read-only copy)."""
    with _yaml_lock:
        return dict(_yaml_config)
