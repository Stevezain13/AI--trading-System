"""
╔══════════════════════════════════════════════════════════════════╗
║         STEPHEN'S AI TRADING SYSTEM                             ║
║         Phase 7 — Main Loop & Orchestration                     ║
║         Day Trading Bot — Connects All Phases Together          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import time
import logging
import sqlite3
import requests
import threading
from datetime import datetime, timedelta
from data_pipeline import (
    fetch_data, calculate_indicators,
    setup_database, DB_PATH
)
from hmm_brain import TradingBrain
from strategies import StrategyManager
from risk_management import RiskManager, PortfolioManager, EmergencyStop
from alpaca_broker import AlpacaClient, OrderManager, PositionTracker

# ══════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("daytrading_bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════
#  BOT CONFIG
# ══════════════════════════════════════════════════════

BOT_CONFIG = {
    # Telegram
    "telegram_token": os.environ.get("TELEGRAM_TOKEN", ""),
    "telegram_chat":  os.environ.get("TELEGRAM_CHAT",  "7781270946"),

    # Trading symbols
    "stocks": ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "SPY"],
    "forex":  ["EURUSD=X", "GBPUSD=X", "JPY=X", "GC=F"],

    # Timing
    "check_interval":    300,   # Check every 5 minutes
    "data_refresh":      3600,  # Refresh data every hour

    # Strategy
    "min_signal_score":  6,     # Minimum signal score
    "trail_type":        "ATR", # Trailing stop type

    # Session (Eastern Time)
    "market_open_hour":  9,
    "market_open_min":   30,
    "market_close_hour": 16,
    "market_close_min":  0,

    # Paper trading
    "paper_trading": True,
}


# ══════════════════════════════════════════════════════
#  TELEGRAM ALERTS
# ══════════════════════════════════════════════════════

def send_telegram(message, token="", chat_id=""):
    """Send Telegram alert."""
    token   = token   or BOT_CONFIG["telegram_token"]
    chat_id = chat_id or BOT_CONFIG["telegram_chat"]

    if not token:
        log.warning("No Telegram token!")
        return

    try:
        url  = "https://api.telegram.org/bot" + token + "/sendMessage"
        data = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
        requests.post(url, data=data, timeout=10)
        log.info("Telegram sent!")
    except Exception as e:
        log.warning("Telegram failed: " + str(e))


def format_signal_alert(signal, position_size, regime, confidence):
    """Format a beautiful signal alert for Telegram."""
    side_emoji = "BUY" if signal.side == "BUY" else "SELL"

    msg = (
        "<b>" + side_emoji + " SIGNAL — " + signal.symbol + "</b>\n"
        "Strategy: " + signal.strategy + "\n"
        "Score: " + str(signal.score) + "/10\n"
        "Regime: " + regime + " (" + str(round(confidence*100)) + "%)\n"
        "---\n"
        "Entry:  " + str(signal.entry) + "\n"
        "SL:     " + str(signal.stop_loss) + "\n"
        "TP1:    " + str(signal.take_profit1) + "\n"
        "TP2:    " + str(signal.take_profit2) + "\n"
        "Size:   " + str(position_size) + "\n"
        "---\n"
        "Reason: " + signal.reason + "\n"
        "Time: " + str(datetime.utcnow().strftime("%H:%M UTC"))
    )
    return msg


def format_trade_update(symbol, action, pnl=None, price=None):
    """Format trade update alert."""
    actions = {
        "BREAKEVEN":   "BREAK EVEN SET — Trade is now risk-free!",
        "TP1_HIT":     "TP1 HIT — Partial profit secured!",
        "TP2_HIT":     "TP2 HIT — Full profit taken!",
        "STOP_HIT":    "STOP LOSS HIT",
        "TRAIL_UPDATE":"Trailing stop updated",
    }

    msg = (
        "<b>TRADE UPDATE — " + symbol + "</b>\n"
        "Action: " + actions.get(action, action) + "\n"
    )

    if price:
        msg += "Price: " + str(price) + "\n"
    if pnl is not None:
        emoji = "PROFIT" if pnl > 0 else "LOSS"
        msg += "PnL: $" + str(round(pnl, 2)) + " " + emoji + "\n"

    msg += "Time: " + str(datetime.utcnow().strftime("%H:%M UTC"))
    return msg


def send_daily_report(risk_manager, broker_client):
    """Send daily performance report."""
    status    = risk_manager.get_status()
    account   = broker_client.get_account()
    positions = broker_client.get_positions()

    msg = (
        "<b>STEPHEN AI BOT — DAILY REPORT</b>\n"
        "Date: " + datetime.utcnow().strftime("%d %B %Y") + "\n"
        "---\n"
        "Balance:     $" + str(status["balance"]) + "\n"
        "Daily PnL:   " + str(status["daily_pnl_pct"]) + "%\n"
        "Drawdown:    " + str(status["drawdown_pct"]) + "%\n"
        "Open Trades: " + str(status["open_positions"]) + "\n"
        "Win Streak:  " + str(status["consecutive_wins"]) + "\n"
        "Loss Streak: " + str(status["consecutive_losses"]) + "\n"
        "---\n"
        "Mode: " + ("PAPER" if status["paper_trading"] else "LIVE") + "\n"
        "Bot running normally!"
    )
    send_telegram(msg)


# ══════════════════════════════════════════════════════
#  STATE MACHINE
# ══════════════════════════════════════════════════════

class BotState:
    INITIALIZING = "INITIALIZING"
    RUNNING      = "RUNNING"
    PAUSED       = "PAUSED"
    STOPPED      = "STOPPED"
    EMERGENCY    = "EMERGENCY"


# ══════════════════════════════════════════════════════
#  MAIN DAY TRADING BOT
# ══════════════════════════════════════════════════════

class DayTradingBot:
    """
    Main Day Trading Bot.
    Orchestrates all components.
    """

    def __init__(self):
        self.state           = BotState.INITIALIZING
        self.brain           = TradingBrain()
        self.strategies      = StrategyManager()
        self.risk            = RiskManager()
        self.portfolio       = PortfolioManager()
        self.broker          = AlpacaClient()
        self.order_manager   = OrderManager(self.broker)
        self.tracker         = PositionTracker(self.broker)
        self.emergency       = EmergencyStop(
            self.risk,
            BOT_CONFIG["telegram_token"],
            BOT_CONFIG["telegram_chat"]
        )
        self.last_report     = datetime.utcnow().date()
        self.signals_today   = 0
        self.trades_today    = 0
        self.last_data_refresh = None

        log.info("Day Trading Bot initialized!")

    def startup(self):
        """Initialize all systems."""
        log.info("="*55)
        log.info("STEPHEN AI DAY TRADING BOT STARTING")
        log.info("="*55)

        # Setup database
        setup_database()

        # Check broker connection
        account = self.broker.get_account()
        if account:
            self.risk.update_balance(account["balance"])
            log.info("Broker connected! Balance: $" + str(account["balance"]))
        else:
            log.error("Broker connection failed!")
            self.state = BotState.STOPPED
            return False

        self.state = BotState.RUNNING

        # Send startup message
        mode = "PAPER" if BOT_CONFIG["paper_trading"] else "LIVE"
        msg = (
            "<b>STEPHEN AI DAY TRADING BOT STARTED!</b>\n"
            "---\n"
            "Mode: " + mode + " TRADING\n"
            "Stocks: " + str(len(BOT_CONFIG["stocks"])) + " symbols\n"
            "Forex: " + str(len(BOT_CONFIG["forex"])) + " pairs\n"
            "Brain: HMM + 5 Strategies\n"
            "Risk: " + str(self.risk.config["max_risk_per_trade"]) + "% per trade\n"
            "Balance: $" + str(round(self.risk.balance, 2)) + "\n"
            "---\n"
            "Bot is watching markets!"
        )
        send_telegram(msg)
        log.info("Startup complete!")
        return True

    def refresh_data(self, symbols):
        """Refresh market data for all symbols."""
        log.info("Refreshing market data...")
        data_cache = {}

        for symbol in symbols:
            df = fetch_data(symbol, period="3mo", interval="1h")
            if df is not None and len(df) > 50:
                df = calculate_indicators(df)
                data_cache[symbol] = df
                log.info("Data refreshed: " + symbol + " (" + str(len(df)) + " candles)")
            time.sleep(1)

        self.last_data_refresh = datetime.utcnow()
        return data_cache

    def analyse_symbol(self, symbol, df):
        """Run full analysis on a symbol."""
        if df is None or len(df) < 50:
            return None, None

        # Brain analysis
        brain_result = self.brain.analyse(df, symbol)

        # Check if tradeable
        if not brain_result["trade_ok"]:
            log.info(symbol + " — Brain says not tradeable")
            return brain_result, None

        # Get signal
        signal = self.strategies.get_signal(df, symbol, brain_result)

        return brain_result, signal

    def process_signal(self, signal, brain_result, df):
        """Process a trading signal."""
        if signal is None:
            return False

        symbol     = signal.symbol
        regime     = brain_result["regime"]
        confidence = brain_result["confidence"]

        # Risk check
        can_trade, reason = self.risk.can_trade(symbol)
        if not can_trade:
            log.info("Risk block for " + symbol + ": " + reason)
            return False

        # Portfolio correlation check
        corr_ok, corr_reason = self.portfolio.check_correlation(
            symbol, signal.side
        )
        if not corr_ok:
            log.info("Correlation block for " + symbol + ": " + corr_reason)
            return False

        # Calculate position size
        atr  = float(df["atr"].iloc[-1])
        size = self.risk.calculate_position_size(
            entry      = signal.entry,
            stop_loss  = signal.stop_loss,
            signal_score = signal.score
        )

        log.info("Signal: " + signal.side + " " + symbol +
                 " score=" + str(signal.score) +
                 " size=" + str(size))

        # Place order
        order = self.order_manager.open_trade(signal, size)

        if order:
            # Add to risk tracking
            self.risk.record_trade_open(
                symbol    = symbol,
                side      = signal.side,
                entry     = signal.entry,
                stop_loss = signal.stop_loss,
                size      = size
            )

            # Add to portfolio
            self.portfolio.add_position(
                symbol = symbol,
                side   = signal.side,
                size   = size,
                entry  = signal.entry
            )

            # Add to trailing stop manager
            self.strategies.trailing.add_position(
                symbol      = symbol,
                side        = signal.side,
                entry       = signal.entry,
                stop_loss   = signal.stop_loss,
                take_profit1 = signal.take_profit1,
                take_profit2 = signal.take_profit2,
                atr         = atr,
                trail_type  = BOT_CONFIG["trail_type"]
            )

            # Send Telegram alert
            alert = format_signal_alert(signal, size, regime, confidence)
            send_telegram(alert)

            self.trades_today  += 1
            self.signals_today += 1
            return True

        return False

    def update_positions(self):
        """Update all open positions with trailing stops."""
        positions = self.broker.get_positions()

        for pos in positions:
            symbol  = pos["symbol"]
            price   = pos["current"]

            # Update trailing stop
            result = self.strategies.update_positions(symbol, price)

            if result and result["action"] != "HOLD":
                action = result["action"]
                pnl    = result.get("pnl")

                # If stop or TP hit — record and alert
                if action in ["STOP_HIT", "TP2_HIT"]:
                    exit_price = result.get("exit_price", price)
                    pnl_amount = (pnl or 0) * pos["qty"]

                    self.risk.record_trade_close(symbol, exit_price, pnl_amount)
                    self.portfolio.remove_position(symbol)

                    alert = format_trade_update(symbol, action, pnl_amount, exit_price)
                    send_telegram(alert)

                elif action in ["BREAKEVEN", "TP1_HIT", "TRAIL_UPDATE"]:
                    alert = format_trade_update(symbol, action, None, price)
                    send_telegram(alert)

    def check_daily_report(self):
        """Send daily report once per day."""
        today = datetime.utcnow().date()
        if today != self.last_report:
            self.last_report = today
            send_daily_report(self.risk, self.broker)
            self.signals_today = 0
            self.trades_today  = 0
            self.risk.reset_daily()

    def is_trading_time(self):
        """Check if it is time to trade."""
        now     = datetime.utcnow()
        weekday = now.weekday()

        # No trading weekends
        if weekday >= 5:
            return False, "Weekend"

        # Market hours (approximate UTC)
        hour = now.hour
        if 13 <= hour < 21:  # 9:30am-4pm EST = 13:30-21 UTC approximately
            return True, "Market Open"

        return False, "Market Closed"

    def run(self):
        """Main bot loop."""
        if not self.startup():
            return

        log.info("Bot running! Press Ctrl+C to stop.")
        data_cache = {}

        while self.state == BotState.RUNNING:
            try:
                # Daily report check
                self.check_daily_report()

                # Emergency check
                if self.emergency.check_and_trigger(self.broker):
                    self.state = BotState.EMERGENCY
                    break

                # Update balance
                account = self.broker.get_account()
                if account:
                    self.risk.update_balance(account["balance"])

                # Update existing positions
                self.update_positions()

                # Check trading time
                trading_ok, time_reason = self.is_trading_time()
                if not trading_ok:
                    log.info("Not trading: " + time_reason)
                    time.sleep(BOT_CONFIG["check_interval"])
                    continue

                # Refresh data if needed
                need_refresh = (
                    self.last_data_refresh is None or
                    (datetime.utcnow() - self.last_data_refresh).seconds >
                    BOT_CONFIG["data_refresh"]
                )

                all_symbols = BOT_CONFIG["stocks"] + BOT_CONFIG["forex"]

                if need_refresh:
                    data_cache = self.refresh_data(all_symbols)

                if not data_cache:
                    log.warning("No data available!")
                    time.sleep(60)
                    continue

                # Analyse each symbol
                for symbol in all_symbols:
                    if symbol not in data_cache:
                        continue

                    # Skip if already have position
                    if self.strategies.has_position(symbol):
                        log.info(symbol + " — Position open, skipping")
                        continue

                    df = data_cache[symbol]

                    # Analyse
                    brain_result, signal = self.analyse_symbol(symbol, df)

                    if signal and signal.score >= BOT_CONFIG["min_signal_score"]:
                        log.info("Signal found: " + signal.side + " " +
                                 symbol + " score=" + str(signal.score))
                        self.process_signal(signal, brain_result, df)

                    time.sleep(1)  # Rate limiting

                log.info("Cycle complete. Next check in " +
                         str(BOT_CONFIG["check_interval"]) + "s")
                time.sleep(BOT_CONFIG["check_interval"])

            except KeyboardInterrupt:
                log.info("Bot stopped by user!")
                send_telegram("Day Trading Bot stopped manually!")
                self.state = BotState.STOPPED
                break

            except Exception as e:
                log.error("Main loop error: " + str(e))
                send_telegram("Bot error: " + str(e)[:100])
                time.sleep(60)

        log.info("Bot shutdown complete!")


# ══════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = DayTradingBot()
    bot.run()
