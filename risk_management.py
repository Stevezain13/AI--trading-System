"""
╔══════════════════════════════════════════════════════════════════╗
║         STEPHEN'S AI TRADING SYSTEM                             ║
║         Phase 5 — Risk Management                               ║
║         Protects capital at all times                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_pipeline import DB_PATH

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
#  RISK CONFIGURATION
# ══════════════════════════════════════════════════════

RISK_CONFIG = {
    "max_risk_per_trade":    1.0,    # % of account per trade
    "max_daily_loss":        3.0,    # Stop trading if daily loss > 3%
    "max_drawdown":          10.0,   # Emergency stop if drawdown > 10%
    "max_open_positions":    5,      # Max simultaneous trades
    "max_correlation":       0.7,    # Max correlation between positions
    "max_account_heat":      5.0,    # Max % of account at risk at once
    "position_scale_up":     0.2,    # Scale up by 20% on winning streak
    "position_scale_down":   0.3,    # Scale down by 30% on losing streak
    "winning_streak_trigger": 5,     # Trades before scaling up
    "losing_streak_trigger":  3,     # Losses before scaling down
    "paper_trading":         True,   # Paper trading mode
    "starting_balance":      100000, # Paper trading balance
}


# ══════════════════════════════════════════════════════
#  POSITION SIZER
# ══════════════════════════════════════════════════════

class PositionSizer:
    """Calculates optimal position size using Kelly Criterion + Risk %."""

    def __init__(self, config=RISK_CONFIG):
        self.config = config

    def calculate_size(self, balance, entry, stop_loss,
                       win_rate=0.55, avg_win=1.5, avg_loss=1.0):
        """
        Calculate position size using:
        1. Risk % method (primary)
        2. Kelly Criterion (secondary confirmation)
        """
        # ── Method 1: Risk % ──────────────────────────
        risk_amount = balance * (self.config["max_risk_per_trade"] / 100)
        risk_per_unit = abs(entry - stop_loss)

        if risk_per_unit <= 0:
            log.warning("Invalid SL distance!")
            return 0

        size_by_risk = risk_amount / risk_per_unit

        # ── Method 2: Kelly Criterion ──────────────────
        kelly_pct = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        kelly_pct = max(0, min(kelly_pct, 0.25))  # Cap at 25%
        size_by_kelly = (balance * kelly_pct) / entry

        # Use the smaller of the two (safer)
        final_size = min(size_by_risk, size_by_kelly)

        # Round to reasonable lot
        final_size = round(final_size, 2)

        log.info("Position size: " + str(final_size) +
                 " (Risk: " + str(round(size_by_risk, 2)) +
                 " Kelly: " + str(round(size_by_kelly, 2)) + ")")

        return final_size

    def adjust_for_streak(self, base_size, consecutive_wins, consecutive_losses):
        """Adjust size based on performance streak."""
        if consecutive_losses >= self.config["losing_streak_trigger"]:
            adjusted = base_size * (1 - self.config["position_scale_down"])
            log.warning("Scaling DOWN due to " + str(consecutive_losses) + " losses")
            return round(adjusted, 2)

        if consecutive_wins >= self.config["winning_streak_trigger"]:
            adjusted = base_size * (1 + self.config["position_scale_up"])
            log.info("Scaling UP due to " + str(consecutive_wins) + " wins")
            return round(adjusted, 2)

        return base_size


# ══════════════════════════════════════════════════════
#  RISK MANAGER
# ══════════════════════════════════════════════════════

class RiskManager:
    """
    Master risk manager — protects your capital!
    """

    def __init__(self, config=RISK_CONFIG):
        self.config              = config
        self.balance             = config["starting_balance"]
        self.peak_balance        = config["starting_balance"]
        self.start_balance       = config["starting_balance"]
        self.daily_start_balance = config["starting_balance"]
        self.open_positions      = {}
        self.consecutive_wins    = 0
        self.consecutive_losses  = 0
        self.emergency_stop      = False
        self.daily_reset_date    = datetime.utcnow().date()
        self.sizer               = PositionSizer(config)

    def update_balance(self, new_balance):
        """Update account balance."""
        self.balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

    def reset_daily(self):
        """Reset daily tracking at start of new day."""
        today = datetime.utcnow().date()
        if today != self.daily_reset_date:
            self.daily_start_balance = self.balance
            self.daily_reset_date    = today
            log.info("Daily reset — Balance: $" + str(self.balance))

    def get_daily_pnl_pct(self):
        """Get daily P&L as percentage."""
        return ((self.balance - self.daily_start_balance) /
                self.daily_start_balance * 100)

    def get_drawdown_pct(self):
        """Get current drawdown from peak."""
        return ((self.peak_balance - self.balance) /
                self.peak_balance * 100)

    def check_daily_loss(self):
        """Check if daily loss limit hit."""
        daily_pnl = self.get_daily_pnl_pct()
        if daily_pnl <= -self.config["max_daily_loss"]:
            log.warning("DAILY LOSS LIMIT HIT: " + str(round(daily_pnl, 2)) + "%")
            return True
        return False

    def check_max_drawdown(self):
        """Check if max drawdown hit — emergency stop!"""
        dd = self.get_drawdown_pct()
        if dd >= self.config["max_drawdown"]:
            self.emergency_stop = True
            log.error("EMERGENCY STOP — Drawdown: " + str(round(dd, 2)) + "%")
            return True
        return False

    def check_max_positions(self):
        """Check if max positions reached."""
        if len(self.open_positions) >= self.config["max_open_positions"]:
            log.info("Max positions reached: " + str(len(self.open_positions)))
            return True
        return False

    def check_account_heat(self):
        """Check total account heat (% at risk)."""
        total_risk = 0
        for pos in self.open_positions.values():
            risk = abs(pos["entry"] - pos["stop_loss"]) * pos["size"]
            total_risk += risk

        heat_pct = (total_risk / self.balance * 100)
        if heat_pct >= self.config["max_account_heat"]:
            log.warning("Account heat too high: " + str(round(heat_pct, 2)) + "%")
            return True
        return False

    def can_trade(self, symbol=None):
        """
        Master check — can we open a new trade?
        Returns (allowed, reason)
        """
        if self.emergency_stop:
            return False, "EMERGENCY STOP ACTIVE"

        if self.check_max_drawdown():
            return False, "MAX DRAWDOWN EXCEEDED"

        if self.check_daily_loss():
            return False, "DAILY LOSS LIMIT HIT"

        if self.check_max_positions():
            return False, "MAX POSITIONS REACHED"

        if self.check_account_heat():
            return False, "ACCOUNT HEAT TOO HIGH"

        if symbol and symbol in self.open_positions:
            return False, "ALREADY HAVE POSITION IN " + symbol

        return True, "OK"

    def calculate_position_size(self, entry, stop_loss, signal_score=5):
        """Calculate position size with all adjustments."""
        # Base size
        size = self.sizer.calculate_size(
            balance    = self.balance,
            entry      = entry,
            stop_loss  = stop_loss
        )

        # Adjust for streak
        size = self.sizer.adjust_for_streak(
            base_size          = size,
            consecutive_wins   = self.consecutive_wins,
            consecutive_losses = self.consecutive_losses
        )

        # Adjust for signal quality
        quality_mult = 0.5 + (signal_score / 20)  # 0.5x to 1x based on score
        size = size * quality_mult

        # Adjust for drawdown
        dd = self.get_drawdown_pct()
        if dd > 5:
            size = size * 0.5
            log.warning("Reducing size due to drawdown " + str(round(dd, 2)) + "%")

        return max(1, round(size, 2))

    def record_trade_open(self, symbol, side, entry, stop_loss, size):
        """Record a new open position."""
        self.open_positions[symbol] = {
            "symbol":    symbol,
            "side":      side,
            "entry":     entry,
            "stop_loss": stop_loss,
            "size":      size,
            "open_time": str(datetime.utcnow())
        }
        log.info("Trade opened: " + side + " " + symbol +
                 " size " + str(size) + " @ " + str(entry))

    def record_trade_close(self, symbol, exit_price, pnl):
        """Record a closed trade and update stats."""
        if symbol in self.open_positions:
            del self.open_positions[symbol]

        self.balance += pnl

        if pnl > 0:
            self.consecutive_wins   += 1
            self.consecutive_losses  = 0
            log.info("WIN: " + symbol + " PnL: $" + str(round(pnl, 2)))
        else:
            self.consecutive_losses += 1
            self.consecutive_wins    = 0
            log.warning("LOSS: " + symbol + " PnL: $" + str(round(pnl, 2)))

        # Save to database
        self._save_trade(symbol, exit_price, pnl)

    def _save_trade(self, symbol, exit_price, pnl):
        """Save trade result to database."""
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""
                UPDATE trades SET
                    exit_price = ?,
                    profit_loss = ?,
                    exit_time = ?,
                    status = 'closed'
                WHERE symbol = ? AND status = 'open'
            """, (exit_price, pnl, str(datetime.utcnow()), symbol))
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("DB save error: " + str(e))

    def get_status(self):
        """Get full risk status."""
        return {
            "balance":           round(self.balance, 2),
            "peak_balance":      round(self.peak_balance, 2),
            "daily_pnl_pct":     round(self.get_daily_pnl_pct(), 2),
            "drawdown_pct":      round(self.get_drawdown_pct(), 2),
            "open_positions":    len(self.open_positions),
            "consecutive_wins":  self.consecutive_wins,
            "consecutive_losses":self.consecutive_losses,
            "emergency_stop":    self.emergency_stop,
            "paper_trading":     self.config["paper_trading"],
        }


# ══════════════════════════════════════════════════════
#  PORTFOLIO MANAGER
# ══════════════════════════════════════════════════════

class PortfolioManager:
    """
    Manages multiple positions across stocks and forex.
    Checks correlation to avoid over-concentration.
    """

    def __init__(self):
        self.positions    = {}
        self.correlations = {}

        # Known correlations
        self.corr_groups = {
            "USD_POSITIVE": ["EURUSD=X", "GBPUSD=X", "AUDUSD=X"],
            "USD_NEGATIVE": ["JPY=X", "CAD=X"],
            "TECH_STOCKS":  ["AAPL", "GOOGL", "MSFT", "META", "NVDA"],
            "GROWTH":       ["TSLA", "AMZN", "NVDA"],
        }

    def check_correlation(self, new_symbol, new_side):
        """
        Check if new trade is correlated with existing positions.
        Returns True if OK to trade, False if too correlated.
        """
        if not self.positions:
            return True, "No existing positions"

        for group_name, group_symbols in self.corr_groups.items():
            if new_symbol not in group_symbols:
                continue

            # Check if we already have position in same group
            for existing_symbol, pos in self.positions.items():
                if existing_symbol in group_symbols:
                    # Same group — check direction
                    if pos["side"] == new_side:
                        return False, "Correlated with " + existing_symbol + " in " + group_name

        return True, "No correlation conflict"

    def add_position(self, symbol, side, size, entry):
        """Add position to portfolio."""
        self.positions[symbol] = {
            "symbol": symbol,
            "side":   side,
            "size":   size,
            "entry":  entry
        }

    def remove_position(self, symbol):
        """Remove position from portfolio."""
        if symbol in self.positions:
            del self.positions[symbol]

    def get_exposure(self):
        """Get total portfolio exposure."""
        buy_exposure  = sum(p["size"] * p["entry"]
                           for p in self.positions.values()
                           if p["side"] == "BUY")
        sell_exposure = sum(p["size"] * p["entry"]
                           for p in self.positions.values()
                           if p["side"] == "SELL")
        return {
            "buy":   round(buy_exposure, 2),
            "sell":  round(sell_exposure, 2),
            "net":   round(buy_exposure - sell_exposure, 2),
            "total": round(buy_exposure + sell_exposure, 2)
        }


# ══════════════════════════════════════════════════════
#  EMERGENCY STOP
# ══════════════════════════════════════════════════════

class EmergencyStop:
    """
    Emergency stop system.
    Triggered by extreme conditions.
    """

    def __init__(self, risk_manager, telegram_token="", chat_id=""):
        self.risk           = risk_manager
        self.telegram_token = telegram_token
        self.chat_id        = chat_id
        self.triggered      = False

    def check_and_trigger(self, broker_client=None):
        """Check conditions and trigger emergency stop if needed."""
        status = self.risk.get_status()

        # Trigger conditions
        if status["drawdown_pct"] >= 10:
            self._trigger("MAX DRAWDOWN " + str(status["drawdown_pct"]) + "%", broker_client)
            return True

        if status["daily_pnl_pct"] <= -3:
            self._trigger("DAILY LOSS " + str(status["daily_pnl_pct"]) + "%", broker_client)
            return True

        if status["emergency_stop"]:
            self._trigger("EMERGENCY STOP FLAG SET", broker_client)
            return True

        return False

    def _trigger(self, reason, broker_client=None):
        """Execute emergency stop."""
        if self.triggered:
            return

        self.triggered = True
        log.error("EMERGENCY STOP TRIGGERED: " + reason)

        # Close all positions
        if broker_client:
            try:
                broker_client.close_all_positions()
                log.info("All positions closed!")
            except Exception as e:
                log.error("Failed to close positions: " + str(e))

        # Send Telegram alert
        self._send_alert(reason)

    def _send_alert(self, reason):
        """Send emergency alert to Telegram."""
        try:
            import requests
            if not self.telegram_token or not self.chat_id:
                return
            msg = "EMERGENCY STOP!\nReason: " + reason + "\nAll positions closed!"
            requests.post(
                "https://api.telegram.org/bot" + self.telegram_token + "/sendMessage",
                data={"chat_id": self.chat_id, "text": msg},
                timeout=10
            )
        except Exception as e:
            log.warning("Alert failed: " + str(e))


if __name__ == "__main__":
    print("Risk Management System loaded!")
    rm = RiskManager()
    print("Starting balance: $" + str(rm.balance))
    print("Max risk per trade: " + str(rm.config["max_risk_per_trade"]) + "%")
    print("Max daily loss: " + str(rm.config["max_daily_loss"]) + "%")
    print("Max drawdown: " + str(rm.config["max_drawdown"]) + "%")
    can, reason = rm.can_trade("AAPL")
    print("Can trade: " + str(can) + " — " + reason)
