"""
╔══════════════════════════════════════════════════════════════════╗
║         STEPHEN'S AI TRADING SYSTEM                             ║
║         Phase 0 — Data Pipeline                                 ║
║         Collects, Cleans & Stores Market Data                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import time
import logging
import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════

# Stocks to trade
STOCKS = [
    "AAPL",   # Apple
    "TSLA",   # Tesla
    "GOOGL",  # Google
    "MSFT",   # Microsoft
    "AMZN",   # Amazon
    "META",   # Meta
    "NVDA",   # Nvidia
    "SPY",    # S&P 500 ETF
]

# Forex pairs
FOREX = [
    "EURUSD=X",
    "GBPUSD=X",
    "JPY=X",
    "AUDUSD=X",
    "CAD=X",
    "GC=F",   # Gold
]

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), "trading_data.db")

# Timeframes
TIMEFRAMES = {
    "1m":  {"period": "7d",  "interval": "1m"},
    "5m":  {"period": "60d", "interval": "5m"},
    "15m": {"period": "60d", "interval": "15m"},
    "1h":  {"period": "2y",  "interval": "1h"},
    "1d":  {"period": "5y",  "interval": "1d"},
}

# ══════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data_pipeline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
#  DATABASE SETUP
# ══════════════════════════════════════════════════════

def setup_database():
    """Create all database tables."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    # Price data table
    c.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol    TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open      REAL,
            high      REAL,
            low       REAL,
            close     REAL,
            volume    REAL,
            UNIQUE(symbol, timeframe, timestamp)
        )
    """)

    # Indicators table
    c.execute("""
        CREATE TABLE IF NOT EXISTS indicators (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol    TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            ema20     REAL,
            ema50     REAL,
            ema200    REAL,
            rsi       REAL,
            atr       REAL,
            adx       REAL,
            bb_upper  REAL,
            bb_lower  REAL,
            bb_mid    REAL,
            volume_ma REAL,
            UNIQUE(symbol, timeframe, timestamp)
        )
    """)

    # Market regimes table
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_regimes (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol    TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            regime    TEXT NOT NULL,
            confidence REAL,
            UNIQUE(symbol, timeframe, timestamp)
        )
    """)

    # Trades table
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol       TEXT NOT NULL,
            side         TEXT NOT NULL,
            entry_price  REAL,
            exit_price   REAL,
            quantity     REAL,
            entry_time   TEXT,
            exit_time    TEXT,
            profit_loss  REAL,
            profit_pct   REAL,
            strategy     TEXT,
            regime       TEXT,
            status       TEXT DEFAULT 'open',
            stop_loss    REAL,
            take_profit  REAL,
            trailing_stop REAL
        )
    """)

    # Performance table
    c.execute("""
        CREATE TABLE IF NOT EXISTS performance (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT NOT NULL UNIQUE,
            total_trades INTEGER DEFAULT 0,
            wins         INTEGER DEFAULT 0,
            losses       INTEGER DEFAULT 0,
            win_rate     REAL DEFAULT 0,
            total_pnl    REAL DEFAULT 0,
            best_trade   REAL DEFAULT 0,
            worst_trade  REAL DEFAULT 0,
            max_drawdown REAL DEFAULT 0,
            sharpe_ratio REAL DEFAULT 0
        )
    """)

    # News/Events table
    c.execute("""
        CREATE TABLE IF NOT EXISTS market_events (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol    TEXT,
            event     TEXT NOT NULL,
            impact    TEXT,
            actual    TEXT,
            forecast  TEXT
        )
    """)

    conn.commit()
    conn.close()
    log.info("Database setup complete!")


# ══════════════════════════════════════════════════════
#  DATA COLLECTION
# ══════════════════════════════════════════════════════

def fetch_data(symbol, period="3mo", interval="1h", retries=3):
    """Fetch market data with automatic retry."""
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(symbol)
            df     = ticker.history(period=period, interval=interval)

            if df is None or df.empty or len(df) < 10:
                log.warning("No data for " + symbol + " attempt " + str(attempt+1))
                time.sleep(5)
                continue

            df.columns = [c.lower() for c in df.columns]
            df = df.reset_index()

            # Rename datetime column
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "timestamp"})
            elif "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})

            # Convert types
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
            else:
                df["volume"] = 0

            df = df.dropna(subset=["open", "high", "low", "close"])
            df["timestamp"] = df["timestamp"].astype(str)

            log.info("Fetched " + str(len(df)) + " candles for " + symbol)
            return df

        except Exception as e:
            log.warning("Fetch error " + symbol + " attempt " + str(attempt+1) + ": " + str(e))
            time.sleep(10)

    return None


def save_to_database(df, symbol, timeframe):
    """Save price data to SQLite database."""
    if df is None or df.empty:
        return 0

    conn    = sqlite3.connect(DB_PATH)
    saved   = 0

    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT OR IGNORE INTO price_data
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                timeframe,
                str(row["timestamp"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row.get("volume", 0))
            ))
            saved += 1
        except Exception as e:
            pass

    conn.commit()
    conn.close()
    return saved


# ══════════════════════════════════════════════════════
#  INDICATOR CALCULATION
# ══════════════════════════════════════════════════════

def calculate_indicators(df):
    """Calculate all technical indicators."""
    if df is None or len(df) < 20:
        return df

    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    o = df["open"].astype(float)

    # EMAs
    df["ema20"]  = c.ewm(span=20,  adjust=False).mean()
    df["ema50"]  = c.ewm(span=50,  adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()

    # RSI
    delta    = c.diff()
    gain     = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss     = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # ATR
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # ADX
    hd  = h.diff()
    ld  = l.diff()
    pdm = hd.where((hd > 0) & (hd > -ld), 0.0)
    mdm = (-ld).where((-ld > 0) & (-ld > hd), 0.0)
    pdi = 100 * (pdm.ewm(span=14, adjust=False).mean() / (df["atr"] + 1e-10))
    mdi = 100 * (mdm.ewm(span=14, adjust=False).mean() / (df["atr"] + 1e-10))
    df["adx"] = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)).ewm(span=14, adjust=False).mean()

    # Bollinger Bands
    sma20          = c.rolling(20).mean()
    std20          = c.rolling(20).std()
    df["bb_upper"] = sma20 + std20 * 2
    df["bb_lower"] = sma20 - std20 * 2
    df["bb_mid"]   = sma20

    # Volume MA
    if "volume" in df.columns:
        df["volume_ma"] = df["volume"].rolling(20).mean()
    else:
        df["volume_ma"] = 0

    # MACD
    ema12        = c.ewm(span=12, adjust=False).mean()
    ema26        = c.ewm(span=26, adjust=False).mean()
    df["macd"]   = ema12 - ema26
    df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["signal_line"]

    # Price momentum
    df["momentum"] = c.pct_change(10)

    # Volatility
    df["volatility"] = c.rolling(20).std() / c.rolling(20).mean() * 100

    return df


def save_indicators(df, symbol, timeframe):
    """Save calculated indicators to database."""
    if df is None or df.empty:
        return

    conn = sqlite3.connect(DB_PATH)

    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT OR IGNORE INTO indicators
                (symbol, timeframe, timestamp, ema20, ema50, ema200,
                 rsi, atr, adx, bb_upper, bb_lower, bb_mid, volume_ma)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, timeframe, str(row.get("timestamp", "")),
                float(row.get("ema20",   0) or 0),
                float(row.get("ema50",   0) or 0),
                float(row.get("ema200",  0) or 0),
                float(row.get("rsi",     0) or 0),
                float(row.get("atr",     0) or 0),
                float(row.get("adx",     0) or 0),
                float(row.get("bb_upper",0) or 0),
                float(row.get("bb_lower",0) or 0),
                float(row.get("bb_mid",  0) or 0),
                float(row.get("volume_ma",0) or 0),
            ))
        except Exception:
            pass

    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════
#  DATA QUALITY CHECK
# ══════════════════════════════════════════════════════

def check_data_quality(df, symbol):
    """Check data for issues."""
    issues = []

    if df is None or df.empty:
        issues.append("No data")
        return issues

    # Check for missing values
    missing = df[["open","high","low","close"]].isnull().sum().sum()
    if missing > 0:
        issues.append(str(missing) + " missing values")

    # Check for zero prices
    zeros = (df["close"] == 0).sum()
    if zeros > 0:
        issues.append(str(zeros) + " zero prices")

    # Check for duplicate timestamps
    if "timestamp" in df.columns:
        dups = df["timestamp"].duplicated().sum()
        if dups > 0:
            issues.append(str(dups) + " duplicate timestamps")

    # Check for price anomalies
    pct_change = df["close"].pct_change().abs()
    spikes = (pct_change > 0.2).sum()
    if spikes > 0:
        issues.append(str(spikes) + " price spikes >20%")

    if not issues:
        log.info("Data quality OK for " + symbol)
    else:
        log.warning("Data issues for " + symbol + ": " + ", ".join(issues))

    return issues


# ══════════════════════════════════════════════════════
#  VOLUME ANALYSIS
# ══════════════════════════════════════════════════════

def detect_volume_anomalies(df):
    """Detect unusual volume spikes."""
    if df is None or "volume" not in df.columns:
        return df

    df["vol_ma20"]    = df["volume"].rolling(20).mean()
    df["vol_ratio"]   = df["volume"] / (df["vol_ma20"] + 1e-10)
    df["vol_anomaly"] = df["vol_ratio"] > 2.0  # 2x average = anomaly

    anomalies = df[df["vol_anomaly"] == True]
    if len(anomalies) > 0:
        log.info("Found " + str(len(anomalies)) + " volume anomalies!")

    return df


# ══════════════════════════════════════════════════════
#  REAL-TIME DATA STREAMING
# ══════════════════════════════════════════════════════

def get_latest_price(symbol):
    """Get the latest real-time price."""
    try:
        ticker = yf.Ticker(symbol)
        data   = ticker.history(period="1d", interval="1m")
        if data is not None and not data.empty:
            latest = data.iloc[-1]
            return {
                "symbol": symbol,
                "price":  float(latest["Close"]),
                "volume": float(latest["Volume"]) if "Volume" in latest else 0,
                "time":   str(data.index[-1])
            }
    except Exception as e:
        log.warning("Price fetch error for " + symbol + ": " + str(e))
    return None


def stream_prices(symbols, callback, interval=60):
    """Stream real-time prices for multiple symbols."""
    log.info("Starting price stream for " + str(len(symbols)) + " symbols...")
    while True:
        for symbol in symbols:
            price_data = get_latest_price(symbol)
            if price_data:
                callback(price_data)
        time.sleep(interval)


# ══════════════════════════════════════════════════════
#  DATA RETRIEVAL
# ══════════════════════════════════════════════════════

def get_data_from_db(symbol, timeframe, limit=500):
    """Retrieve data from database."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("""
        SELECT * FROM price_data
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, conn, params=(symbol, timeframe, limit))
    conn.close()
    return df.iloc[::-1].reset_index(drop=True)


def get_indicators_from_db(symbol, timeframe, limit=500):
    """Retrieve indicators from database."""
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("""
        SELECT * FROM indicators
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, conn, params=(symbol, timeframe, limit))
    conn.close()
    return df.iloc[::-1].reset_index(drop=True)


def get_performance_summary():
    """Get overall performance summary."""
    conn = sqlite3.connect(DB_PATH)

    # Total trades
    trades = pd.read_sql_query("""
        SELECT * FROM trades WHERE status = 'closed'
    """, conn)

    conn.close()

    if trades.empty:
        return {
            "total_trades": 0,
            "wins":         0,
            "losses":       0,
            "win_rate":     0,
            "total_pnl":    0,
            "best_trade":   0,
            "worst_trade":  0,
        }

    wins   = len(trades[trades["profit_loss"] > 0])
    losses = len(trades[trades["profit_loss"] <= 0])
    total  = len(trades)

    return {
        "total_trades": total,
        "wins":         wins,
        "losses":       losses,
        "win_rate":     round(wins / total * 100, 1) if total > 0 else 0,
        "total_pnl":    round(trades["profit_loss"].sum(), 2),
        "best_trade":   round(trades["profit_loss"].max(), 2),
        "worst_trade":  round(trades["profit_loss"].min(), 2),
    }


# ══════════════════════════════════════════════════════
#  MAIN DATA COLLECTION RUN
# ══════════════════════════════════════════════════════

def run_data_collection():
    """Run full data collection for all symbols."""
    log.info("="*50)
    log.info("STEPHEN AI TRADING SYSTEM")
    log.info("Phase 0 — Data Pipeline Starting")
    log.info("="*50)

    # Setup database
    setup_database()

    all_symbols = STOCKS + FOREX
    total_saved = 0

    for symbol in all_symbols:
        log.info("Processing " + symbol + "...")

        # Fetch 1h data (main timeframe)
        df = fetch_data(symbol, period="2y", interval="1h")

        if df is not None:
            # Check quality
            check_data_quality(df, symbol)

            # Detect volume anomalies
            df = detect_volume_anomalies(df)

            # Calculate indicators
            df = calculate_indicators(df)

            # Save to database
            saved = save_to_database(df, symbol, "1h")
            save_indicators(df, symbol, "1h")

            total_saved += saved
            log.info("Saved " + str(saved) + " records for " + symbol)

        # Small delay between requests
        time.sleep(2)

    log.info("="*50)
    log.info("Data collection complete!")
    log.info("Total records saved: " + str(total_saved))
    log.info("Database: " + DB_PATH)
    log.info("="*50)

    return total_saved


# ══════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    run_data_collection()
