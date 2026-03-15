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

ALLOW_SHORT = os.getenv("ALLOW_SHORT", "false") == "true"
ASYNC_MODE = os.getenv("ASYNC_MODE", "false") == "true"

BROKER_ABSTRACTION_ENABLED = True
PAPER_BROKER_SPREAD_BPS = 5.0

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
    'STAT_MR': 0.40,
    'VWAP': 0.20,
    'KALMAN_PAIRS': 0.20,
    'PEAD': 0.10,
    'ORB': 0.05,
    'MICRO_MOM': 0.05,
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
MR_MIN_RR_RATIO = 1.5

# --- VWAP Mean Reversion ---
VWAP_OU_ZSCORE_MIN   = 1.0        # OU z-score confirmation for entries
VWAP_MAX_SPREAD_PCT  = 0.001      # Skip if bid-ask spread > 0.1%
VWAP_VOLUME_RATIO    = 0.8        # Volume ratio vs 20-bar average
MAX_INTRADAY_MOVE_PCT = 0.03      # Skip if stock moved > 3% today
VWAP_BAND_STD        = 2.0        # Standard deviation multiplier for VWAP bands
VWAP_RSI_OVERSOLD    = 30         # RSI below this = oversold (buy signal)
VWAP_RSI_OVERBOUGHT  = 70         # RSI above this = overbought (short signal)
VWAP_CONFIRMATION_BARS = 1        # Bars confirming bounce (1 = disabled)
VWAP_STOP_EXTENSION  = 0.5        # Stop extension beyond band (in std devs)
VWAP_MIN_STOP_PCT    = 0.005      # Minimum 0.5% stop distance

# --- Kalman Pairs Trading ---
PAIRS_ZSCORE_ENTRY = 2.0
PAIRS_ZSCORE_EXIT = 0.2
PAIRS_ZSCORE_STOP = 3.0
PAIRS_MAX_HOLD_DAYS = 10
PAIRS_MAX_ACTIVE = 15
PAIRS_MIN_CORRELATION = 0.80
PAIRS_COINT_PVALUE = 0.05
KALMAN_DELTA = 1e-4
KALMAN_OBS_NOISE = 0.001

# --- Opening Range Breakout ---
ORB_ENABLED          = True
ORB_VOLUME_RATIO     = 1.3        # Volume confirmation ratio
ORB_MAX_GAP_PCT      = 0.04       # Skip if gap > 4%
ORB_MAX_RANGE_PCT    = 0.035      # Skip if range > 3.5%
ORB_MIN_STOP_PCT     = 0.003      # Minimum 0.3% stop distance
ORB_SCAN_SYMBOLS     = 15         # Top N by morning volume
ORB_ACTIVE_UNTIL     = time(11, 30)
ORB_BREAKOUT_BUFFER  = 0.001      # 0.1% confirmation buffer above/below ORB range
ORB_TP_MULT          = 1.5        # Take profit = entry ± 1.5x ORB range
ORB_SL_MULT          = 0.5        # Stop loss = entry ∓ 0.5x ORB range
ORB_TIME_STOP_HOURS  = 2          # Close after 2 hours

# --- Micro Momentum ---
MICRO_SPY_VOL_SPIKE_MULT = 3.0
MICRO_SPY_MIN_MOVE_PCT = 0.0015
MICRO_MAX_HOLD_MINUTES = 8
MICRO_STOP_PCT = 0.003
MICRO_TARGET_PCT = 0.006
MICRO_MAX_TRADES_PER_EVENT = 3
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

# ============================================================
# RISK MANAGEMENT
# ============================================================

# --- Position Sizing ---
RISK_PER_TRADE_PCT = 0.008          # Risk 0.8% of portfolio per trade
MAX_POSITION_PCT = 0.08             # Hard cap: max 8% per position
MIN_POSITION_VALUE = 100            # Min $100 per trade
MAX_POSITIONS = 12
MAX_PORTFOLIO_DEPLOY = 0.55
DAILY_LOSS_HALT = -0.04

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

# --- Daily P&L Controls ---
PNL_GAIN_LOCK_PCT = 0.015
PNL_LOSS_HALT_PCT = -0.010
PNL_GAIN_LOCK_SIZE_MULT = 0.30

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

# --- Short Selling ---
SHORT_SIZE_MULTIPLIER = 0.75       # Short positions = 75% of equivalent long size
SHORT_HARD_STOP_PCT = 0.04         # Close short if goes against you > 4%
MOMENTUM_TRAILING_STOP_PCT = 0.02  # Trailing stop for position monitor
NO_SHORT_SYMBOLS = {"SPY", "QQQ", "IWM", "DIA"}

# --- Dynamic Capital Allocation ---
DYNAMIC_ALLOCATION = os.getenv("DYNAMIC_ALLOCATION", "true") == "true"
ALLOCATION_LOOKBACK_DAYS = 20      # Rolling window for Sharpe-based allocation
ALLOCATION_MIN_WEIGHT = 0.10       # Min 10% per strategy

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

# --- Scan Configuration ---
SCAN_INTERVAL_SEC = 120
CLOSE_UNKNOWN_POSITIONS = False    # Auto-close broker positions we didn't open

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

# --- WebSocket Position Monitoring ---
WEBSOCKET_MONITORING = True
WEBSOCKET_RECONNECT_SEC = 5

# ============================================================
# PERSISTENCE & BACKTESTING
# ============================================================

STATE_FILE = "state.json"
DB_FILE = "bot.db"
LOG_FILE = "bot.log"
STATE_SAVE_INTERVAL_SEC = 60

BACKTEST_SLIPPAGE = 0.0005         # 0.05% slippage per trade
BACKTEST_COMMISSION = 0.0035       # $0.0035 per share
BACKTEST_RISK_FREE_RATE = 0.045    # 4.5% annual
BACKTEST_TOP_N = 20                # Run on top 20 most liquid symbols

# ============================================================
# LEGACY (backward-compat for archived strategies)
# ============================================================

MAX_MOMENTUM_POSITIONS = 4
MAX_SECTOR_POSITIONS = 3
MAX_PAIRS_POSITIONS = 5
GAP_MAX_POSITIONS = 3
GAP_GO_ENABLED = False
SECTOR_ROTATION_ENABLED = False
SECTOR_POSITION_SIZE_PCT = 0.05
EMA_SCALP_MAX_POSITIONS = 3
NEWS_FILTER_ENABLED = False
NEWS_LOOKBACK_HOURS = 6

# ============================================================
# RUNTIME PARAMETERS
# ============================================================

_runtime_params: dict = {}

def get_param(key: str, default=None):
    """Get a runtime parameter (optimizer-modified or config default)."""
    return _runtime_params.get(key, default)

def set_param(key: str, value):
    """Set a runtime parameter (used by optimizer)."""
    _runtime_params[key] = value
