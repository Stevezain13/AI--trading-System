"""
╔══════════════════════════════════════════════════════════════════╗
║         STEPHEN'S AI TRADING SYSTEM                             ║
║         Phase 2 — HMM Brain Engine                             ║
║         Detects Market Regimes, Patterns & Volume Anomalies     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════
#  MARKET REGIMES
# ══════════════════════════════════════════════════════

REGIMES = {
    0: "BULL_TREND",
    1: "BEAR_TREND",
    2: "SIDEWAYS",
    3: "HIGH_VOLATILITY"
}

# ══════════════════════════════════════════════════════
#  HMM ENGINE
# ══════════════════════════════════════════════════════

class HMMBrain:
    """
    Hidden Markov Model Brain
    Detects market regimes using:
    - Price action
    - Volume
    - Volatility
    - Momentum
    """

    def __init__(self):
        self.current_regime    = "SIDEWAYS"
        self.regime_confidence = 0.0
        self.regime_history    = []
        self.n_states          = 4

    def extract_features(self, df):
        """Extract features for HMM analysis."""
        if df is None or len(df) < 20:
            return None

        c = df["close"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)

        features = pd.DataFrame()

        # 1. Returns
        features["returns"]     = c.pct_change()
        features["returns_5"]   = c.pct_change(5)
        features["returns_20"]  = c.pct_change(20)

        # 2. Volatility
        features["volatility"]  = c.rolling(20).std() / c.rolling(20).mean()
        features["atr_ratio"]   = (h - l) / c

        # 3. Trend strength
        ema20  = c.ewm(span=20,  adjust=False).mean()
        ema50  = c.ewm(span=50,  adjust=False).mean()
        ema200 = c.ewm(span=200, adjust=False).mean()

        features["ema_trend"]   = (ema20 - ema50) / (ema50 + 1e-10)
        features["above_200"]   = (c > ema200).astype(float)

        # 4. RSI
        delta    = c.diff()
        gain     = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss     = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi      = 100 - (100 / (1 + gain / (loss + 1e-10)))
        features["rsi_norm"]    = (rsi - 50) / 50

        # 5. Volume ratio
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(20).mean()
            features["vol_ratio"] = df["volume"] / (vol_ma + 1e-10)
        else:
            features["vol_ratio"] = 1.0

        # 6. Momentum
        features["momentum"]    = c.pct_change(10)
        features["momentum_20"] = c.pct_change(20)

        # 7. Bollinger Band position
        sma20  = c.rolling(20).mean()
        std20  = c.rolling(20).std()
        bb_pos = (c - sma20) / (std20 * 2 + 1e-10)
        features["bb_position"] = bb_pos

        return features.dropna()

    def detect_regime(self, df):
        """
        Detect current market regime using rule-based HMM approach.

        Regimes:
        - BULL_TREND: Strong upward momentum
        - BEAR_TREND: Strong downward momentum
        - SIDEWAYS: Low volatility, ranging
        - HIGH_VOLATILITY: Extreme moves
        """
        features = self.extract_features(df)

        if features is None or features.empty:
            return "SIDEWAYS", 0.5

        last = features.iloc[-1]

        # Score each regime
        bull_score  = 0.0
        bear_score  = 0.0
        side_score  = 0.0
        volat_score = 0.0

        # Trend direction
        if last["ema_trend"] > 0.005:
            bull_score += 2
        elif last["ema_trend"] < -0.005:
            bear_score += 2

        # Above/below EMA200
        if last["above_200"] == 1:
            bull_score += 1
        else:
            bear_score += 1

        # RSI
        if last["rsi_norm"] > 0.2:
            bull_score += 1
        elif last["rsi_norm"] < -0.2:
            bear_score += 1
        else:
            side_score += 1

        # Momentum
        if last["momentum"] > 0.02:
            bull_score += 2
        elif last["momentum"] < -0.02:
            bear_score += 2
        else:
            side_score += 1

        # Volatility
        if last["volatility"] > 0.03:
            volat_score += 3
        elif last["volatility"] < 0.01:
            side_score += 2

        # BB position
        if last["bb_position"] > 0.8:
            volat_score += 1
        elif last["bb_position"] < -0.8:
            volat_score += 1

        # Volume
        if last["vol_ratio"] > 2.0:
            volat_score += 2

        # Determine regime
        scores = {
            "BULL_TREND":      bull_score,
            "BEAR_TREND":      bear_score,
            "SIDEWAYS":        side_score,
            "HIGH_VOLATILITY": volat_score
        }

        regime     = max(scores, key=scores.get)
        max_score  = max(scores.values())
        total      = sum(scores.values()) + 1e-10
        confidence = max_score / total

        self.current_regime    = regime
        self.regime_confidence = confidence
        self.regime_history.append({
            "time":       str(datetime.utcnow()),
            "regime":     regime,
            "confidence": confidence,
            "scores":     scores
        })

        log.info("Regime: " + regime + " (confidence: " + str(round(confidence*100)) + "%)")
        return regime, confidence

    def get_strategy_for_regime(self, regime):
        """Return best strategy for current regime."""
        strategies = {
            "BULL_TREND":      ["TREND_FOLLOWING", "BREAKOUT", "SMC_BUY"],
            "BEAR_TREND":      ["TREND_FOLLOWING", "BREAKOUT", "SMC_SELL"],
            "SIDEWAYS":        ["MEAN_REVERSION"],
            "HIGH_VOLATILITY": ["BREAKOUT", "AVOID"]
        }
        return strategies.get(regime, ["AVOID"])

    def should_trade(self, regime, confidence):
        """Decide if conditions are good enough to trade."""
        if confidence < 0.4:
            log.info("Low confidence " + str(round(confidence*100)) + "% — skipping")
            return False
        if regime == "HIGH_VOLATILITY" and confidence > 0.7:
            log.info("High volatility detected — caution!")
            return False
        return True


# ══════════════════════════════════════════════════════
#  PATTERN DETECTOR
# ══════════════════════════════════════════════════════

class PatternDetector:
    """Detects price patterns and candlestick formations."""

    def detect_breakout(self, df, lookback=20):
        """Detect price breakouts from consolidation."""
        if df is None or len(df) < lookback + 5:
            return None

        recent    = df.tail(lookback)
        last      = df.iloc[-1]
        high_zone = float(recent["high"].max())
        low_zone  = float(recent["low"].min())
        price     = float(last["close"])
        atr       = float(last.get("atr", (high_zone - low_zone) * 0.1))

        if price > high_zone - atr * 0.5:
            return {
                "type":      "BULLISH_BREAKOUT",
                "level":     high_zone,
                "strength":  "STRONG" if price > high_zone + atr else "WEAK"
            }
        elif price < low_zone + atr * 0.5:
            return {
                "type":      "BEARISH_BREAKOUT",
                "level":     low_zone,
                "strength":  "STRONG" if price < low_zone - atr else "WEAK"
            }
        return None

    def detect_consolidation(self, df, lookback=20):
        """Detect price consolidation zones."""
        if df is None or len(df) < lookback:
            return None

        recent    = df.tail(lookback)
        high_zone = float(recent["high"].max())
        low_zone  = float(recent["low"].min())
        close     = float(df.iloc[-1]["close"])
        rng       = high_zone - low_zone
        mid       = (high_zone + low_zone) / 2

        # Tight range = consolidation
        if rng / close < 0.02:
            return {
                "type":  "CONSOLIDATION",
                "high":  high_zone,
                "low":   low_zone,
                "mid":   mid,
                "range": rng
            }
        return None

    def detect_trend_structure(self, df):
        """Detect higher highs/lowers lows for trend structure."""
        if df is None or len(df) < 10:
            return "NEUTRAL"

        closes = df["close"].astype(float).values
        highs  = df["high"].astype(float).values
        lows   = df["low"].astype(float).values

        # Check last 5 swings
        recent_highs = []
        recent_lows  = []

        for i in range(2, min(len(df)-2, 20)):
            if highs[-i] > highs[-i-1] and highs[-i] > highs[-i+1]:
                recent_highs.append(highs[-i])
            if lows[-i] < lows[-i-1] and lows[-i] < lows[-i+1]:
                recent_lows.append(lows[-i])

        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            hh = recent_highs[-1] > recent_highs[-2]
            hl = recent_lows[-1]  > recent_lows[-2]
            lh = recent_highs[-1] < recent_highs[-2]
            ll = recent_lows[-1]  < recent_lows[-2]

            if hh and hl:
                return "UPTREND"
            elif lh and ll:
                return "DOWNTREND"

        return "NEUTRAL"

    def detect_smc_patterns(self, df, direction):
        """Detect SMC specific patterns."""
        if df is None or len(df) < 10:
            return {}

        patterns = {}

        # Order Block
        for i in range(len(df)-10, len(df)-2):
            if i < 1:
                continue
            c = df.iloc[i]
            n = df.iloc[i+1]
            body  = abs(float(c["close"]) - float(c["open"]))
            nbody = abs(float(n["close"]) - float(n["open"]))

            if direction == "BUY":
                if (float(c["close"]) < float(c["open"]) and
                    float(n["close"]) > float(n["open"]) and
                    nbody > body * 1.5):
                    patterns["order_block"] = {
                        "high": float(c["high"]),
                        "low":  float(c["low"]),
                        "type": "BULLISH_OB"
                    }
            else:
                if (float(c["close"]) > float(c["open"]) and
                    float(n["close"]) < float(n["open"]) and
                    nbody > body * 1.5):
                    patterns["order_block"] = {
                        "high": float(c["high"]),
                        "low":  float(c["low"]),
                        "type": "BEARISH_OB"
                    }

        # Fair Value Gap
        for i in range(1, len(df)-1):
            p = df.iloc[i-1]
            n = df.iloc[i+1]
            if direction == "BUY" and float(n["low"]) > float(p["high"]):
                gap = float(n["low"]) - float(p["high"])
                if gap >= 0.0003:
                    patterns["fvg"] = {
                        "top":    float(n["low"]),
                        "bottom": float(p["high"]),
                        "type":   "BULLISH_FVG"
                    }

            if direction == "SELL" and float(n["high"]) < float(p["low"]):
                gap = float(p["low"]) - float(n["high"])
                if gap >= 0.0003:
                    patterns["fvg"] = {
                        "top":    float(p["low"]),
                        "bottom": float(n["high"]),
                        "type":   "BEARISH_FVG"
                    }

        return patterns


# ══════════════════════════════════════════════════════
#  VOLUME ANOMALY DETECTOR
# ══════════════════════════════════════════════════════

class VolumeAnalyzer:
    """Analyzes volume for institutional activity."""

    def detect_anomalies(self, df):
        """Detect unusual volume activity."""
        if df is None or "volume" not in df.columns:
            return []

        vol    = df["volume"].astype(float)
        vol_ma = vol.rolling(20).mean()
        ratio  = vol / (vol_ma + 1e-10)

        anomalies = []
        last_ratio = float(ratio.iloc[-1])
        last_vol   = float(vol.iloc[-1])
        last_price = float(df["close"].iloc[-1])

        if last_ratio > 3.0:
            anomalies.append({
                "type":       "EXTREME_VOLUME",
                "ratio":      last_ratio,
                "volume":     last_vol,
                "price":      last_price,
                "signal":     "STRONG_MOVE_COMING"
            })
        elif last_ratio > 2.0:
            anomalies.append({
                "type":       "HIGH_VOLUME",
                "ratio":      last_ratio,
                "volume":     last_vol,
                "price":      last_price,
                "signal":     "INSTITUTIONAL_ACTIVITY"
            })
        elif last_ratio < 0.3:
            anomalies.append({
                "type":       "LOW_VOLUME",
                "ratio":      last_ratio,
                "volume":     last_vol,
                "price":      last_price,
                "signal":     "AVOID_TRADING"
            })

        return anomalies

    def get_volume_trend(self, df, periods=5):
        """Get volume trend direction."""
        if df is None or "volume" not in df.columns:
            return "NEUTRAL"

        recent_vol = df["volume"].tail(periods).astype(float)
        if recent_vol.is_monotonic_increasing:
            return "INCREASING"
        elif recent_vol.is_monotonic_decreasing:
            return "DECREASING"
        return "NEUTRAL"


# ══════════════════════════════════════════════════════
#  MAIN BRAIN CLASS
# ══════════════════════════════════════════════════════

class TradingBrain:
    """
    Master brain that combines:
    - HMM regime detection
    - Pattern recognition
    - Volume analysis
    """

    def __init__(self):
        self.hmm       = HMMBrain()
        self.patterns  = PatternDetector()
        self.volume    = VolumeAnalyzer()

    def analyse(self, df, symbol):
        """Full market analysis."""
        log.info("Brain analysing " + symbol + "...")

        # 1. Detect regime
        regime, confidence = self.hmm.detect_regime(df)

        # 2. Get trend structure
        structure = self.patterns.detect_trend_structure(df)

        # 3. Check for breakout
        breakout = self.patterns.detect_breakout(df)

        # 4. Check consolidation
        consolidation = self.patterns.detect_consolidation(df)

        # 5. Volume analysis
        vol_anomalies = self.volume.detect_anomalies(df)
        vol_trend     = self.volume.get_volume_trend(df)

        # 6. Get best strategies
        strategies = self.hmm.get_strategy_for_regime(regime)

        # 7. Should we trade?
        trade_ok = self.hmm.should_trade(regime, confidence)

        result = {
            "symbol":        symbol,
            "regime":        regime,
            "confidence":    confidence,
            "structure":     structure,
            "breakout":      breakout,
            "consolidation": consolidation,
            "vol_anomalies": vol_anomalies,
            "vol_trend":     vol_trend,
            "strategies":    strategies,
            "trade_ok":      trade_ok,
            "timestamp":     str(datetime.utcnow())
        }

        log.info("Analysis complete for " + symbol + ":")
        log.info("  Regime: "     + regime + " (" + str(round(confidence*100)) + "%)")
        log.info("  Structure: "  + structure)
        log.info("  Strategies: " + str(strategies))
        log.info("  Trade OK: "   + str(trade_ok))

        return result


if __name__ == "__main__":
    import yfinance as yf
    from data_pipeline import fetch_data, calculate_indicators

    brain = TradingBrain()

    # Test on AAPL
    print("Testing HMM Brain on AAPL...")
    df = fetch_data("AAPL", period="3mo", interval="1h")
    if df is not None:
        df = calculate_indicators(df)
        result = brain.analyse(df, "AAPL")
        print("Regime:", result["regime"])
        print("Confidence:", result["confidence"])
        print("Strategies:", result["strategies"])
        print("Trade OK:", result["trade_ok"])
