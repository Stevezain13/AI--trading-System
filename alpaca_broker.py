"""
╔══════════════════════════════════════════════════════════════════╗
║         STEPHEN'S AI TRADING SYSTEM                             ║
║         Phase 6 — Alpaca Broker Integration                     ║
║         Connects to Alpaca for Paper + Real Trading             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════
#  ALPACA CONFIG
# ══════════════════════════════════════════════════════

ALPACA_CONFIG = {
    "api_key":    os.environ.get("ALPACA_API_KEY",    "YOUR_ALPACA_KEY"),
    "secret_key": os.environ.get("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET"),
    "paper":      True,  # Paper trading = True, Real = False
}

# URLs
PAPER_URL  = "https://paper-api.alpaca.markets"
REAL_URL   = "https://api.alpaca.markets"
DATA_URL   = "https://data.alpaca.markets"


# ══════════════════════════════════════════════════════
#  ALPACA CLIENT
# ══════════════════════════════════════════════════════

class AlpacaClient:
    """
    Connects to Alpaca broker for:
    - Paper trading (practice)
    - Real trading (live)
    - Account info
    - Order management
    - Position tracking
    """

    def __init__(self, config=ALPACA_CONFIG):
        self.api_key    = config["api_key"]
        self.secret_key = config["secret_key"]
        self.paper      = config["paper"]
        self.base_url   = PAPER_URL if self.paper else REAL_URL

        self.headers = {
            "APCA-API-KEY-ID":     self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type":        "application/json"
        }

        mode = "PAPER" if self.paper else "REAL MONEY"
        log.info("Alpaca client initialized — " + mode + " trading")

    def _get(self, endpoint, params=None):
        """Make GET request to Alpaca API."""
        try:
            url = self.base_url + endpoint
            r   = requests.get(url, headers=self.headers, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            else:
                log.warning("Alpaca GET error " + str(r.status_code) + ": " + r.text)
                return None
        except Exception as e:
            log.error("Alpaca GET failed: " + str(e))
            return None

    def _post(self, endpoint, data=None):
        """Make POST request to Alpaca API."""
        try:
            url = self.base_url + endpoint
            r   = requests.post(url, headers=self.headers, json=data, timeout=10)
            if r.status_code in [200, 201]:
                return r.json()
            else:
                log.warning("Alpaca POST error " + str(r.status_code) + ": " + r.text)
                return None
        except Exception as e:
            log.error("Alpaca POST failed: " + str(e))
            return None

    def _delete(self, endpoint):
        """Make DELETE request to Alpaca API."""
        try:
            url = self.base_url + endpoint
            r   = requests.delete(url, headers=self.headers, timeout=10)
            return r.status_code in [200, 204]
        except Exception as e:
            log.error("Alpaca DELETE failed: " + str(e))
            return False

    # ── ACCOUNT ─────────────────────────────────────

    def get_account(self):
        """Get account information."""
        data = self._get("/v2/account")
        if data:
            return {
                "balance":       float(data.get("equity", 0)),
                "cash":          float(data.get("cash", 0)),
                "buying_power":  float(data.get("buying_power", 0)),
                "pnl":           float(data.get("unrealized_pl", 0)),
                "status":        data.get("status", "unknown"),
                "paper_trading": self.paper
            }
        return None

    def get_balance(self):
        """Get current account balance."""
        account = self.get_account()
        return float(account["balance"]) if account else 0

    # ── ORDERS ──────────────────────────────────────

    def place_market_order(self, symbol, side, qty,
                           stop_loss=None, take_profit=None):
        """
        Place a market order with optional SL and TP.
        side = 'buy' or 'sell'
        """
        order_data = {
            "symbol":        symbol,
            "qty":           str(qty),
            "side":          side.lower(),
            "type":          "market",
            "time_in_force": "day"
        }

        # Add bracket order (SL + TP) if provided
        if stop_loss and take_profit:
            order_data["order_class"]  = "bracket"
            order_data["stop_loss"]    = {"stop_price": str(round(stop_loss, 2))}
            order_data["take_profit"]  = {"limit_price": str(round(take_profit, 2))}

        result = self._post("/v2/orders", order_data)

        if result:
            log.info("Order placed: " + side.upper() + " " + symbol +
                     " qty=" + str(qty) + " id=" + result.get("id", ""))
            return {
                "order_id":  result.get("id"),
                "symbol":    symbol,
                "side":      side,
                "qty":       qty,
                "status":    result.get("status"),
                "timestamp": str(datetime.utcnow())
            }
        return None

    def place_limit_order(self, symbol, side, qty, limit_price,
                          stop_loss=None, take_profit=None):
        """Place a limit order."""
        order_data = {
            "symbol":        symbol,
            "qty":           str(qty),
            "side":          side.lower(),
            "type":          "limit",
            "limit_price":   str(round(limit_price, 2)),
            "time_in_force": "day"
        }

        if stop_loss and take_profit:
            order_data["order_class"] = "bracket"
            order_data["stop_loss"]   = {"stop_price": str(round(stop_loss, 2))}
            order_data["take_profit"] = {"limit_price": str(round(take_profit, 2))}

        result = self._post("/v2/orders", order_data)

        if result:
            log.info("Limit order placed: " + side.upper() + " " + symbol +
                     " @ " + str(limit_price))
            return result
        return None

    def cancel_order(self, order_id):
        """Cancel a specific order."""
        success = self._delete("/v2/orders/" + order_id)
        if success:
            log.info("Order cancelled: " + order_id)
        return success

    def cancel_all_orders(self):
        """Cancel all open orders."""
        success = self._delete("/v2/orders")
        if success:
            log.info("All orders cancelled!")
        return success

    def get_orders(self, status="open"):
        """Get all orders."""
        return self._get("/v2/orders", params={"status": status}) or []

    # ── POSITIONS ───────────────────────────────────

    def get_positions(self):
        """Get all open positions."""
        data = self._get("/v2/positions")
        if not data:
            return []

        positions = []
        for p in data:
            positions.append({
                "symbol":     p.get("symbol"),
                "side":       p.get("side"),
                "qty":        float(p.get("qty", 0)),
                "entry":      float(p.get("avg_entry_price", 0)),
                "current":    float(p.get("current_price", 0)),
                "pnl":        float(p.get("unrealized_pl", 0)),
                "pnl_pct":    float(p.get("unrealized_plpc", 0)) * 100,
                "market_val": float(p.get("market_value", 0))
            })
        return positions

    def get_position(self, symbol):
        """Get specific position."""
        data = self._get("/v2/positions/" + symbol)
        if data:
            return {
                "symbol":  data.get("symbol"),
                "side":    data.get("side"),
                "qty":     float(data.get("qty", 0)),
                "entry":   float(data.get("avg_entry_price", 0)),
                "current": float(data.get("current_price", 0)),
                "pnl":     float(data.get("unrealized_pl", 0))
            }
        return None

    def close_position(self, symbol):
        """Close a specific position."""
        result = self._delete("/v2/positions/" + symbol)
        if result:
            log.info("Position closed: " + symbol)
        return result

    def close_all_positions(self):
        """Close ALL open positions — emergency stop!"""
        result = self._delete("/v2/positions")
        if result:
            log.info("ALL positions closed!")
        return result

    # ── TRAILING STOP ───────────────────────────────

    def update_trailing_stop(self, order_id, trail_price=None, trail_percent=None):
        """Update trailing stop on existing order."""
        data = {}
        if trail_price:
            data["trail_price"] = str(trail_price)
        if trail_percent:
            data["trail_percent"] = str(trail_percent)

        try:
            url    = self.base_url + "/v2/orders/" + order_id
            result = requests.patch(url, headers=self.headers, json=data, timeout=10)
            if result.status_code == 200:
                log.info("Trailing stop updated for order " + order_id)
                return True
        except Exception as e:
            log.warning("Trail update failed: " + str(e))
        return False

    def place_trailing_stop_order(self, symbol, side, qty, trail_percent=2.0):
        """Place order with built-in trailing stop."""
        order_data = {
            "symbol":        symbol,
            "qty":           str(qty),
            "side":          side.lower(),
            "type":          "trailing_stop",
            "trail_percent": str(trail_percent),
            "time_in_force": "gtc"
        }
        result = self._post("/v2/orders", order_data)
        if result:
            log.info("Trailing stop order placed: " + symbol +
                     " trail " + str(trail_percent) + "%")
        return result

    # ── MARKET DATA ─────────────────────────────────

    def get_latest_price(self, symbol):
        """Get latest price from Alpaca."""
        try:
            url     = DATA_URL + "/v2/stocks/" + symbol + "/trades/latest"
            result  = requests.get(url, headers=self.headers, timeout=10)
            if result.status_code == 200:
                data = result.json()
                return float(data.get("trade", {}).get("p", 0))
        except Exception as e:
            log.warning("Price fetch failed for " + symbol + ": " + str(e))
        return None

    def is_market_open(self):
        """Check if market is currently open."""
        data = self._get("/v2/clock")
        if data:
            return data.get("is_open", False)
        return False

    def get_market_hours(self):
        """Get market open/close times."""
        data = self._get("/v2/clock")
        if data:
            return {
                "is_open":    data.get("is_open"),
                "next_open":  data.get("next_open"),
                "next_close": data.get("next_close")
            }
        return None

    def get_assets(self, asset_class="us_equity"):
        """Get list of tradeable assets."""
        data = self._get("/v2/assets", params={
            "status":      "active",
            "asset_class": asset_class
        })
        return data or []

    # ── PERFORMANCE ─────────────────────────────────

    def get_portfolio_history(self, period="1W", timeframe="1D"):
        """Get portfolio performance history."""
        data = self._get("/v2/account/portfolio/history", params={
            "period":    period,
            "timeframe": timeframe
        })
        if data:
            return {
                "timestamps":  data.get("timestamp", []),
                "equity":      data.get("equity", []),
                "profit_loss": data.get("profit_loss", []),
                "base_value":  data.get("base_value", 0)
            }
        return None

    def get_activities(self, activity_type="FILL"):
        """Get trading activities (filled orders)."""
        data = self._get("/v2/account/activities/" + activity_type)
        return data or []


# ══════════════════════════════════════════════════════
#  ORDER MANAGER
# ══════════════════════════════════════════════════════

class OrderManager:
    """
    High-level order management.
    Handles order lifecycle and tracking.
    """

    def __init__(self, client: AlpacaClient):
        self.client     = client
        self.open_orders = {}

    def open_trade(self, signal, position_size):
        """
        Open a trade from a signal.
        Returns order details or None.
        """
        symbol    = signal.symbol
        side      = signal.side.lower()
        entry     = signal.entry
        sl        = signal.stop_loss
        tp1       = signal.take_profit1
        tp2       = signal.take_profit2

        # Check market is open for stocks
        if not self._is_forex(symbol):
            if not self.client.is_market_open():
                log.info("Market closed — cannot place order for " + symbol)
                return None

        # Place order
        order = self.client.place_market_order(
            symbol      = symbol,
            side        = side,
            qty         = position_size,
            stop_loss   = sl,
            take_profit = tp2  # Use TP2 as main target
        )

        if order:
            self.open_orders[symbol] = {
                **order,
                "signal":   signal.to_dict(),
                "tp1":      tp1,
                "tp2":      tp2,
                "sl":       sl,
                "tp1_hit":  False
            }
            log.info("Trade opened: " + side.upper() + " " + symbol)

        return order

    def close_trade(self, symbol, reason="manual"):
        """Close a specific trade."""
        result = self.client.close_position(symbol)
        if result and symbol in self.open_orders:
            del self.open_orders[symbol]
            log.info("Trade closed: " + symbol + " reason: " + reason)
        return result

    def update_stops(self, symbol, new_sl, new_tp=None):
        """Update stop loss and take profit for a position."""
        orders = self.client.get_orders(status="open")
        for order in orders:
            if order.get("symbol") == symbol:
                order_id = order.get("id")
                self.client.update_trailing_stop(order_id)
                log.info("Stop updated for " + symbol + " new SL: " + str(new_sl))
                return True
        return False

    def _is_forex(self, symbol):
        """Check if symbol is forex."""
        forex = ["EURUSD", "GBPUSD", "JPY", "AUDUSD", "CAD", "GC"]
        return any(f in symbol for f in forex)

    def get_open_trades(self):
        """Get all currently open trades."""
        return self.client.get_positions()

    def get_summary(self):
        """Get trading summary."""
        account   = self.client.get_account()
        positions = self.client.get_positions()

        total_pnl = sum(p["pnl"] for p in positions)

        return {
            "account":      account,
            "positions":    positions,
            "total_pnl":    round(total_pnl, 2),
            "num_positions": len(positions),
            "timestamp":    str(datetime.utcnow())
        }


# ══════════════════════════════════════════════════════
#  POSITION TRACKER
# ══════════════════════════════════════════════════════

class PositionTracker:
    """Tracks all positions and their performance."""

    def __init__(self, client: AlpacaClient):
        self.client     = client
        self.history    = []

    def update(self):
        """Update position data."""
        positions = self.client.get_positions()
        account   = self.client.get_account()

        snapshot = {
            "timestamp": str(datetime.utcnow()),
            "balance":   account["balance"] if account else 0,
            "positions": positions,
            "total_pnl": sum(p["pnl"] for p in positions)
        }

        self.history.append(snapshot)
        return snapshot

    def get_best_position(self):
        """Get position with highest profit."""
        positions = self.client.get_positions()
        if not positions:
            return None
        return max(positions, key=lambda p: p["pnl"])

    def get_worst_position(self):
        """Get position with largest loss."""
        positions = self.client.get_positions()
        if not positions:
            return None
        return min(positions, key=lambda p: p["pnl"])

    def get_total_exposure(self):
        """Get total market exposure."""
        positions = self.client.get_positions()
        return sum(p["market_val"] for p in positions)


if __name__ == "__main__":
    print("Alpaca Broker Integration loaded!")
    print("Mode: PAPER TRADING")
    print("")
    print("To connect:")
    print("  export ALPACA_API_KEY=your_key")
    print("  export ALPACA_SECRET_KEY=your_secret")
    print("")

    client = AlpacaClient()
    account = client.get_account()
    if account:
        print("Connected! Balance: $" + str(account["balance"]))
        print("Status: " + account["status"])
    else:
        print("Not connected — add your API keys!")
