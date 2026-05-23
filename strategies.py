"""
╔══════════════════════════════════════════════════════════════════╗
║         STEPHEN'S AI TRADING SYSTEM                             ║
║         Phase 3+4 — Trading Strategies + Trailing Stop          ║
║         Day Trading Bot with 6 Strategies                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
#  SIGNAL CLASS
# ══════════════════════════════════════════════════════

class Signal:
    def __init__(self, symbol, side, strategy, score,
                 entry, stop_loss, take_profit1, take_profit2,
                 regime, confidence, reason):
        self.symbol      = symbol
        self.side        = side          # BUY or SELL
        self.strategy    = strategy      # Strategy name
        self.score       = score         # Signal quality 0-10
        self.entry       = entry
        self.stop_loss   = stop_loss
        self.take_profit1 = take_profit1
        self.take_profit2 = take_profit2
        self.regime      = regime
        self.confidence  = confidence
        self.reason      = reason
        self.timestamp   = str(datetime.utcnow())

    def to_dict(self):
        return {
            "symbol":       self.symbol,
            "side":         self.side,
            "strategy":     self.strategy,
            "score":        self.score,
            "entry":        self.entry,
            "stop_loss":    self.stop_loss,
            "take_profit1": self.take_profit1,
            "take_profit2": self.take_profit2,
            "regime":       self.regime,
            "confidence":   self.confidence,
            "reason":       self.reason,
            "timestamp":    self.timestamp
        }

    def __str__(self):
        return (
            self.side + " " + self.symbol + "\n"
            "Strategy: " + self.strategy + "\n"
            "Score: " + str(self.score) + "/10\n"
            "Entry: " + str(self.entry) + "\n"
            "SL: " + str(self.stop_loss) + "\n"
            "TP1: " + str(self.take_profit1) + "\n"
            "TP2: " + str(self.take_profit2) + "\n"
            "Regime: " + self.regime + "\n"
            "Reason: " + self.reason
        )


# ══════════════════════════════════════════════════════
#  BASE STRATEGY
# ══════════════════════════════════════════════════════

class BaseStrategy:
    """Base class for all strategies."""

    def __init__(self, name):
        self.name = name

    def get_signal(self, df, symbol, regime, brain_result):
        raise NotImplementedError

    def get_atr(self, df):
        if "atr" in df.columns:
            return float(df["atr"].iloc[-1])
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        return float(tr.rolling(14).mean().iloc[-1])

    def get_pip(self, symbol):
        if "JPY" in symbol:
            return 0.01
        if "XAU" in symbol or "GC" in symbol:
            return 0.1
        return 0.0001


# ══════════════════════════════════════════════════════
#  STRATEGY 1 — TREND FOLLOWING
# ══════════════════════════════════════════════════════

class TrendFollowingStrategy(BaseStrategy):
    """
    Follows strong trends using EMA crossover + ADX.
    Best in BULL_TREND and BEAR_TREND regimes.
    """

    def __init__(self):
        super().__init__("TREND_FOLLOWING")

    def get_signal(self, df, symbol, regime, brain_result):
        if regime not in ["BULL_TREND", "BEAR_TREND"]:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr  = self.get_atr(df)

        score   = 0
        reasons = []

        # EMA crossover
        cross_up   = float(prev["ema20"]) <= float(prev["ema50"]) and float(last["ema20"]) > float(last["ema50"])
        cross_down = float(prev["ema20"]) >= float(prev["ema50"]) and float(last["ema20"]) < float(last["ema50"])

        if not cross_up and not cross_down:
            return None

        if cross_up:
            side = "BUY"
            if regime == "BULL_TREND":
                score += 3
                reasons.append("EMA crossover in bull regime")
        else:
            side = "SELL"
            if regime == "BEAR_TREND":
                score += 3
                reasons.append("EMA crossover in bear regime")

        # ADX confirmation
        if float(last.get("adx", 0)) > 25:
            score += 2
            reasons.append("Strong trend ADX " + str(round(float(last["adx"]))))

        # Above/below EMA200
        if side == "BUY" and float(last["close"]) > float(last["ema200"]):
            score += 2
            reasons.append("Above EMA200")
        elif side == "SELL" and float(last["close"]) < float(last["ema200"]):
            score += 2
            reasons.append("Below EMA200")

        # RSI
        rsi = float(last.get("rsi", 50))
        if side == "BUY" and 40 < rsi < 65:
            score += 1
            reasons.append("RSI bullish " + str(round(rsi)))
        elif side == "SELL" and 35 < rsi < 60:
            score += 1
            reasons.append("RSI bearish " + str(round(rsi)))

        # Volume
        if "vol_ratio" in last.index and float(last["vol_ratio"]) > 1.2:
            score += 1
            reasons.append("Volume confirmed")

        if score < 5:
            return None

        entry = float(last["close"])
        if side == "BUY":
            sl  = round(entry - atr * 2, 5)
            tp1 = round(entry + atr * 2, 5)
            tp2 = round(entry + atr * 4, 5)
        else:
            sl  = round(entry + atr * 2, 5)
            tp1 = round(entry - atr * 2, 5)
            tp2 = round(entry - atr * 4, 5)

        return Signal(
            symbol=symbol, side=side,
            strategy=self.name, score=min(score, 10),
            entry=entry, stop_loss=sl,
            take_profit1=tp1, take_profit2=tp2,
            regime=regime, confidence=brain_result["confidence"],
            reason=" | ".join(reasons)
        )


# ══════════════════════════════════════════════════════
#  STRATEGY 2 — MEAN REVERSION
# ══════════════════════════════════════════════════════

class MeanReversionStrategy(BaseStrategy):
    """
    Trades when price is stretched too far from average.
    Best in SIDEWAYS regime.
    """

    def __init__(self):
        super().__init__("MEAN_REVERSION")

    def get_signal(self, df, symbol, regime, brain_result):
        if regime != "SIDEWAYS":
            return None

        last  = df.iloc[-1]
        atr   = self.get_atr(df)
        price = float(last["close"])
        rsi   = float(last.get("rsi", 50))

        score   = 0
        reasons = []
        side    = None

        # Bollinger Band extremes
        bb_upper = float(last.get("bb_upper", price + atr*2))
        bb_lower = float(last.get("bb_lower", price - atr*2))
        bb_mid   = float(last.get("bb_mid", price))

        if price <= bb_lower and rsi < 35:
            side = "BUY"
            score += 3
            reasons.append("Price at lower BB oversold RSI " + str(round(rsi)))
        elif price >= bb_upper and rsi > 65:
            side = "SELL"
            score += 3
            reasons.append("Price at upper BB overbought RSI " + str(round(rsi)))

        if side is None:
            return None

        # Extra confirmations
        if side == "BUY" and rsi < 30:
            score += 2
            reasons.append("Extremely oversold")
        elif side == "SELL" and rsi > 70:
            score += 2
            reasons.append("Extremely overbought")

        if score < 4:
            return None

        entry = price
        if side == "BUY":
            sl  = round(entry - atr * 1.5, 5)
            tp1 = round(bb_mid, 5)
            tp2 = round(bb_upper, 5)
        else:
            sl  = round(entry + atr * 1.5, 5)
            tp1 = round(bb_mid, 5)
            tp2 = round(bb_lower, 5)

        return Signal(
            symbol=symbol, side=side,
            strategy=self.name, score=min(score, 10),
            entry=entry, stop_loss=sl,
            take_profit1=tp1, take_profit2=tp2,
            regime=regime, confidence=brain_result["confidence"],
            reason=" | ".join(reasons)
        )


# ══════════════════════════════════════════════════════
#  STRATEGY 3 — BREAKOUT TRADING
# ══════════════════════════════════════════════════════

class BreakoutStrategy(BaseStrategy):
    """
    Trades confirmed breakouts from consolidation zones.
    Best in HIGH_VOLATILITY regime.
    """

    def __init__(self):
        super().__init__("BREAKOUT")

    def get_signal(self, df, symbol, regime, brain_result):
        breakout = brain_result.get("breakout")
        if not breakout:
            return None

        last  = df.iloc[-1]
        atr   = self.get_atr(df)
        price = float(last["close"])

        score   = 0
        reasons = []
        side    = None

        if breakout["type"] == "BULLISH_BREAKOUT":
            side = "BUY"
            score += 3
            reasons.append("Bullish breakout above " + str(round(breakout["level"], 5)))
            if breakout["strength"] == "STRONG":
                score += 2
                reasons.append("Strong breakout!")

        elif breakout["type"] == "BEARISH_BREAKOUT":
            side = "SELL"
            score += 3
            reasons.append("Bearish breakout below " + str(round(breakout["level"], 5)))
            if breakout["strength"] == "STRONG":
                score += 2
                reasons.append("Strong breakout!")

        if side is None or score < 4:
            return None

        # Volume confirmation
        if "vol_ratio" in last.index and float(last.get("vol_ratio", 1)) > 1.5:
            score += 2
            reasons.append("Volume spike confirmed")

        # ADX
        if float(last.get("adx", 0)) > 20:
            score += 1
            reasons.append("ADX " + str(round(float(last["adx"]))))

        entry = price
        if side == "BUY":
            sl  = round(entry - atr * 2, 5)
            tp1 = round(entry + atr * 3, 5)
            tp2 = round(entry + atr * 6, 5)
        else:
            sl  = round(entry + atr * 2, 5)
            tp1 = round(entry - atr * 3, 5)
            tp2 = round(entry - atr * 6, 5)

        return Signal(
            symbol=symbol, side=side,
            strategy=self.name, score=min(score, 10),
            entry=entry, stop_loss=sl,
            take_profit1=tp1, take_profit2=tp2,
            regime=regime, confidence=brain_result["confidence"],
            reason=" | ".join(reasons)
        )


# ══════════════════════════════════════════════════════
#  STRATEGY 4 — SMC STRATEGY
# ══════════════════════════════════════════════════════

class SMCStrategy(BaseStrategy):
    """
    Smart Money Concepts strategy.
    Uses Order Blocks, BOS, FVG from your existing knowledge!
    Works in all regimes.
    """

    def __init__(self):
        super().__init__("SMC")

    def get_signal(self, df, symbol, regime, brain_result):
        last  = df.iloc[-1]
        prev  = df.iloc[-2]
        atr   = self.get_atr(df)
        price = float(last["close"])

        score   = 0
        reasons = []
        side    = None

        # Determine direction from regime
        if regime == "BULL_TREND":
            side = "BUY"
        elif regime == "BEAR_TREND":
            side = "SELL"
        else:
            # Use EMA for direction in other regimes
            if float(last["ema20"]) > float(last["ema50"]):
                side = "BUY"
            else:
                side = "SELL"

        # Check SMC patterns
        smc_patterns = brain_result.get("smc_patterns", {})

        # Order Block
        if "order_block" in smc_patterns:
            ob = smc_patterns["order_block"]
            score += 3
            reasons.append("Order Block at " + str(round(ob["low"], 5)) + "-" + str(round(ob["high"], 5)))

        # FVG
        if "fvg" in smc_patterns:
            score += 2
            reasons.append("Fair Value Gap detected")

        # BOS
        structure = brain_result.get("structure", "NEUTRAL")
        if side == "BUY" and structure == "UPTREND":
            score += 2
            reasons.append("BOS bullish structure")
        elif side == "SELL" and structure == "DOWNTREND":
            score += 2
            reasons.append("BOS bearish structure")

        # HTF confirmation
        if side == "BUY" and float(last["close"]) > float(last["ema200"]):
            score += 1
            reasons.append("Above EMA200")
        elif side == "SELL" and float(last["close"]) < float(last["ema200"]):
            score += 1
            reasons.append("Below EMA200")

        # Premium/Discount
        recent = df.tail(50)
        high   = float(recent["high"].max())
        low    = float(recent["low"].min())
        mid    = (high + low) / 2

        if side == "BUY" and price < mid:
            score += 1
            reasons.append("In discount zone")
        elif side == "SELL" and price > mid:
            score += 1
            reasons.append("In premium zone")

        if score < 5:
            return None

        entry = price
        if side == "BUY":
            sl  = round(entry - atr * 2, 5)
            tp1 = round(entry + atr * 2, 5)
            tp2 = round(entry + atr * 4, 5)
        else:
            sl  = round(entry + atr * 2, 5)
            tp1 = round(entry - atr * 2, 5)
            tp2 = round(entry - atr * 4, 5)

        return Signal(
            symbol=symbol, side=side,
            strategy=self.name, score=min(score, 10),
            entry=entry, stop_loss=sl,
            take_profit1=tp1, take_profit2=tp2,
            regime=regime, confidence=brain_result["confidence"],
            reason=" | ".join(reasons)
        )


# ══════════════════════════════════════════════════════
#  STRATEGY 5 — VOLUME ANOMALY
# ══════════════════════════════════════════════════════

class VolumeAnomalyStrategy(BaseStrategy):
    """
    Trades on institutional volume spikes.
    Best for stocks.
    """

    def __init__(self):
        super().__init__("VOLUME_ANOMALY")

    def get_signal(self, df, symbol, regime, brain_result):
        anomalies = brain_result.get("vol_anomalies", [])
        if not anomalies:
            return None

        last  = df.iloc[-1]
        atr   = self.get_atr(df)
        price = float(last["close"])

        score   = 0
        reasons = []
        side    = None

        for anomaly in anomalies:
            if anomaly["type"] in ["EXTREME_VOLUME", "HIGH_VOLUME"]:
                score += 3
                reasons.append(anomaly["type"] + " ratio " + str(round(anomaly["ratio"], 1)) + "x")

                # Direction based on price action
                if float(last["close"]) > float(last["open"]):
                    side = "BUY"
                    reasons.append("Bullish candle on high volume")
                else:
                    side = "SELL"
                    reasons.append("Bearish candle on high volume")

            elif anomaly["type"] == "LOW_VOLUME":
                log.info("Low volume — skipping " + symbol)
                return None

        if side is None or score < 4:
            return None

        # Regime confirmation
        if (side == "BUY" and regime == "BULL_TREND") or \
           (side == "SELL" and regime == "BEAR_TREND"):
            score += 2
            reasons.append("Regime confirms direction")

        entry = price
        if side == "BUY":
            sl  = round(entry - atr * 2, 5)
            tp1 = round(entry + atr * 3, 5)
            tp2 = round(entry + atr * 5, 5)
        else:
            sl  = round(entry + atr * 2, 5)
            tp1 = round(entry - atr * 3, 5)
            tp2 = round(entry - atr * 5, 5)

        return Signal(
            symbol=symbol, side=side,
            strategy=self.name, score=min(score, 10),
            entry=entry, stop_loss=sl,
            take_profit1=tp1, take_profit2=tp2,
            regime=regime, confidence=brain_result["confidence"],
            reason=" | ".join(reasons)
        )


# ══════════════════════════════════════════════════════
#  TRAILING STOP MANAGER
# ══════════════════════════════════════════════════════

class TrailingStopManager:
    """
    Manages 4 types of trailing stops:
    1. ATR Trailing — trails 2x ATR behind price
    2. Percentage — trails 5% behind highest price
    3. Structure — moves to each swing level
    4. Break Even — moves to entry at 1R profit
    """

    def __init__(self):
        self.positions = {}  # symbol: position data

    def add_position(self, symbol, side, entry, stop_loss,
                     take_profit1, take_profit2, atr, trail_type="ATR"):
        """Add a new position to track."""
        self.positions[symbol] = {
            "symbol":       symbol,
            "side":         side,
            "entry":        entry,
            "stop_loss":    stop_loss,
            "take_profit1": take_profit1,
            "take_profit2": take_profit2,
            "atr":          atr,
            "trail_type":   trail_type,
            "highest":      entry,  # For percentage trailing
            "lowest":       entry,
            "be_set":       False,  # Break even set?
            "tp1_hit":      False,  # TP1 hit?
            "open_time":    str(datetime.utcnow())
        }
        log.info("Position added: " + symbol + " " + side + " @ " + str(entry))

    def update_trail(self, symbol, current_price, df=None):
        """Update trailing stop for a position."""
        if symbol not in self.positions:
            return None

        pos    = self.positions[symbol]
        side   = pos["side"]
        entry  = pos["entry"]
        atr    = pos["atr"]
        result = {"action": "HOLD", "new_sl": pos["stop_loss"]}

        # Update highest/lowest price
        if side == "BUY":
            if current_price > pos["highest"]:
                pos["highest"] = current_price
        else:
            if current_price < pos["lowest"]:
                pos["lowest"] = current_price

        trail_type = pos["trail_type"]

        # ── 1. BREAK EVEN FIRST ───────────────────────
        if not pos["be_set"]:
            if side == "BUY" and current_price >= entry + atr:
                new_sl = round(entry + 0.0001, 5)  # Just above entry
                if new_sl > pos["stop_loss"]:
                    pos["stop_loss"] = new_sl
                    pos["be_set"]    = True
                    result["action"] = "BREAKEVEN"
                    result["new_sl"] = new_sl
                    log.info("BREAK EVEN set for " + symbol + " @ " + str(new_sl))

            elif side == "SELL" and current_price <= entry - atr:
                new_sl = round(entry - 0.0001, 5)  # Just below entry
                if new_sl < pos["stop_loss"]:
                    pos["stop_loss"] = new_sl
                    pos["be_set"]    = True
                    result["action"] = "BREAKEVEN"
                    result["new_sl"] = new_sl
                    log.info("BREAK EVEN set for " + symbol + " @ " + str(new_sl))

        # ── 2. TP1 HIT → More aggressive trailing ─────
        if not pos["tp1_hit"]:
            if side == "BUY" and current_price >= pos["take_profit1"]:
                pos["tp1_hit"] = True
                result["action"] = "TP1_HIT"
                log.info("TP1 HIT for " + symbol + "!")

            elif side == "SELL" and current_price <= pos["take_profit1"]:
                pos["tp1_hit"] = True
                result["action"] = "TP1_HIT"
                log.info("TP1 HIT for " + symbol + "!")

        # ── 3. TRAILING TYPES ──────────────────────────

        if trail_type == "ATR" or not pos["tp1_hit"]:
            # ATR trailing
            multiplier = 1.5 if pos["tp1_hit"] else 2.0
            if side == "BUY":
                new_sl = round(current_price - atr * multiplier, 5)
                if new_sl > pos["stop_loss"]:
                    pos["stop_loss"] = new_sl
                    result["new_sl"] = new_sl
                    if result["action"] == "HOLD":
                        result["action"] = "TRAIL_UPDATE"

            else:
                new_sl = round(current_price + atr * multiplier, 5)
                if new_sl < pos["stop_loss"]:
                    pos["stop_loss"] = new_sl
                    result["new_sl"] = new_sl
                    if result["action"] == "HOLD":
                        result["action"] = "TRAIL_UPDATE"

        elif trail_type == "PERCENTAGE":
            # Percentage trailing (5%)
            pct = 0.03 if pos["tp1_hit"] else 0.05
            if side == "BUY":
                new_sl = round(pos["highest"] * (1 - pct), 5)
                if new_sl > pos["stop_loss"]:
                    pos["stop_loss"] = new_sl
                    result["new_sl"] = new_sl
                    if result["action"] == "HOLD":
                        result["action"] = "TRAIL_UPDATE"

            else:
                new_sl = round(pos["lowest"] * (1 + pct), 5)
                if new_sl < pos["stop_loss"]:
                    pos["stop_loss"] = new_sl
                    result["new_sl"] = new_sl
                    if result["action"] == "HOLD":
                        result["action"] = "TRAIL_UPDATE"

        # ── 4. CHECK STOP HIT ──────────────────────────
        if side == "BUY" and current_price <= pos["stop_loss"]:
            result["action"] = "STOP_HIT"
            result["exit_price"] = current_price
            pnl = current_price - entry
            result["pnl"] = round(pnl, 5)
            del self.positions[symbol]
            log.info("STOP HIT for " + symbol + " PnL: " + str(result["pnl"]))

        elif side == "SELL" and current_price >= pos["stop_loss"]:
            result["action"] = "STOP_HIT"
            result["exit_price"] = current_price
            pnl = entry - current_price
            result["pnl"] = round(pnl, 5)
            del self.positions[symbol]
            log.info("STOP HIT for " + symbol + " PnL: " + str(result["pnl"]))

        # ── 5. CHECK TP2 HIT ───────────────────────────
        if symbol in self.positions:
            pos = self.positions[symbol]
            if side == "BUY" and current_price >= pos["take_profit2"]:
                result["action"] = "TP2_HIT"
                result["exit_price"] = current_price
                pnl = current_price - entry
                result["pnl"] = round(pnl, 5)
                del self.positions[symbol]
                log.info("TP2 HIT for " + symbol + " PnL: " + str(result["pnl"]))

            elif side == "SELL" and current_price <= pos["take_profit2"]:
                result["action"] = "TP2_HIT"
                result["exit_price"] = current_price
                pnl = entry - current_price
                result["pnl"] = round(pnl, 5)
                del self.positions[symbol]
                log.info("TP2 HIT for " + symbol + " PnL: " + str(result["pnl"]))

        return result


# ══════════════════════════════════════════════════════
#  STRATEGY MANAGER
# ══════════════════════════════════════════════════════

class StrategyManager:
    """
    Manages all 5 strategies.
    Picks best strategy based on regime.
    """

    def __init__(self):
        self.strategies = {
            "TREND_FOLLOWING": TrendFollowingStrategy(),
            "MEAN_REVERSION":  MeanReversionStrategy(),
            "BREAKOUT":        BreakoutStrategy(),
            "SMC":             SMCStrategy(),
            "VOLUME_ANOMALY":  VolumeAnomalyStrategy(),
        }
        self.trailing = TrailingStopManager()
        self.min_score = 6

    def get_signal(self, df, symbol, brain_result):
        """Get best signal from all applicable strategies."""
        regime      = brain_result["regime"]
        strategies  = brain_result["strategies"]
        best_signal = None

        for strat_name in strategies:
            if strat_name not in self.strategies:
                continue

            strategy = self.strategies[strat_name]

            try:
                signal = strategy.get_signal(df, symbol, regime, brain_result)
                if signal and signal.score >= self.min_score:
                    if best_signal is None or signal.score > best_signal.score:
                        best_signal = signal
                        log.info("Signal from " + strat_name + " score " + str(signal.score) + "/10")
            except Exception as e:
                log.warning("Strategy " + strat_name + " error: " + str(e))

        return best_signal

    def update_positions(self, symbol, current_price, df=None):
        """Update trailing stops for open positions."""
        return self.trailing.update_trail(symbol, current_price, df)

    def has_position(self, symbol):
        """Check if we have an open position."""
        return symbol in self.trailing.positions

    def get_positions(self):
        """Get all open positions."""
        return self.trailing.positions


if __name__ == "__main__":
    print("Strategy Manager loaded successfully!")
    print("Available strategies:")
    sm = StrategyManager()
    for name in sm.strategies:
        print("  - " + name)
