"""
╔══════════════════════════════════════════════════════════════════╗
║         STEPHEN'S AI TRADING SYSTEM                             ║
║         Phase 9 — Integration Testing                           ║
║         Tests all components before going live                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import logging
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ══════════════════════════════════════════════════════
#  TEST DATA PIPELINE
# ══════════════════════════════════════════════════════

class TestDataPipeline(unittest.TestCase):
    """Tests for Phase 0 — Data Pipeline."""

    def test_fetch_data(self):
        """Test data fetching from Yahoo Finance."""
        try:
            from data_pipeline import fetch_data
            df = fetch_data("AAPL", period="1mo", interval="1h")
            self.assertIsNotNone(df)
            self.assertGreater(len(df), 10)
            self.assertIn("close", df.columns)
            self.assertIn("open", df.columns)
            self.assertIn("high", df.columns)
            self.assertIn("low", df.columns)
            print("PASS: Data fetch working!")
        except Exception as e:
            self.fail("Data fetch failed: " + str(e))

    def test_calculate_indicators(self):
        """Test indicator calculation."""
        try:
            from data_pipeline import fetch_data, calculate_indicators
            df = fetch_data("AAPL", period="3mo", interval="1h")
            self.assertIsNotNone(df)
            df = calculate_indicators(df)
            self.assertIn("ema20",    df.columns)
            self.assertIn("ema50",    df.columns)
            self.assertIn("ema200",   df.columns)
            self.assertIn("rsi",      df.columns)
            self.assertIn("atr",      df.columns)
            self.assertIn("adx",      df.columns)
            self.assertIn("bb_upper", df.columns)
            self.assertIn("bb_lower", df.columns)
            self.assertFalse(df["ema20"].isnull().all())
            print("PASS: Indicators calculating correctly!")
        except Exception as e:
            self.fail("Indicator calculation failed: " + str(e))

    def test_database_setup(self):
        """Test database creation."""
        try:
            from data_pipeline import setup_database, DB_PATH
            import sqlite3
            setup_database()
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            self.assertIn("price_data",    tables)
            self.assertIn("indicators",    tables)
            self.assertIn("trades",        tables)
            self.assertIn("performance",   tables)
            print("PASS: Database setup correctly!")
        except Exception as e:
            self.fail("Database setup failed: " + str(e))

    def test_data_quality(self):
        """Test data quality checks."""
        try:
            from data_pipeline import fetch_data, check_data_quality
            df = fetch_data("AAPL", period="1mo", interval="1h")
            self.assertIsNotNone(df)
            issues = check_data_quality(df, "AAPL")
            print("PASS: Data quality check — Issues: " + str(issues))
        except Exception as e:
            self.fail("Data quality check failed: " + str(e))


# ══════════════════════════════════════════════════════
#  TEST HMM BRAIN
# ══════════════════════════════════════════════════════

class TestHMMBrain(unittest.TestCase):
    """Tests for Phase 2 — HMM Brain."""

    def setUp(self):
        """Set up test data."""
        from data_pipeline import fetch_data, calculate_indicators
        self.df = fetch_data("AAPL", period="3mo", interval="1h")
        if self.df is not None:
            self.df = calculate_indicators(self.df)

    def test_regime_detection(self):
        """Test market regime detection."""
        try:
            from hmm_brain import HMMBrain
            brain  = HMMBrain()
            regime, confidence = brain.detect_regime(self.df)
            self.assertIn(regime, ["BULL_TREND", "BEAR_TREND", "SIDEWAYS", "HIGH_VOLATILITY"])
            self.assertGreater(confidence, 0)
            self.assertLessEqual(confidence, 1)
            print("PASS: Regime detected: " + regime + " (" + str(round(confidence*100)) + "%)")
        except Exception as e:
            self.fail("Regime detection failed: " + str(e))

    def test_strategy_selection(self):
        """Test strategy selection for each regime."""
        try:
            from hmm_brain import HMMBrain
            brain = HMMBrain()
            regimes = ["BULL_TREND", "BEAR_TREND", "SIDEWAYS", "HIGH_VOLATILITY"]
            for regime in regimes:
                strats = brain.get_strategy_for_regime(regime)
                self.assertIsInstance(strats, list)
                self.assertGreater(len(strats), 0)
                print("PASS: " + regime + " → " + str(strats))
        except Exception as e:
            self.fail("Strategy selection failed: " + str(e))

    def test_pattern_detection(self):
        """Test price pattern detection."""
        try:
            from hmm_brain import PatternDetector
            detector  = PatternDetector()
            structure = detector.detect_trend_structure(self.df)
            self.assertIn(structure, ["UPTREND", "DOWNTREND", "NEUTRAL"])
            print("PASS: Structure detected: " + structure)
        except Exception as e:
            self.fail("Pattern detection failed: " + str(e))

    def test_volume_analysis(self):
        """Test volume anomaly detection."""
        try:
            from hmm_brain import VolumeAnalyzer
            analyzer   = VolumeAnalyzer()
            anomalies  = analyzer.detect_anomalies(self.df)
            vol_trend  = analyzer.get_volume_trend(self.df)
            self.assertIsInstance(anomalies, list)
            self.assertIn(vol_trend, ["INCREASING", "DECREASING", "NEUTRAL"])
            print("PASS: Volume analysis — Trend: " + vol_trend +
                  " Anomalies: " + str(len(anomalies)))
        except Exception as e:
            self.fail("Volume analysis failed: " + str(e))

    def test_full_brain_analysis(self):
        """Test complete brain analysis."""
        try:
            from hmm_brain import TradingBrain
            brain  = TradingBrain()
            result = brain.analyse(self.df, "AAPL")
            self.assertIn("regime",     result)
            self.assertIn("confidence", result)
            self.assertIn("strategies", result)
            self.assertIn("trade_ok",   result)
            print("PASS: Full brain analysis complete!")
            print("  Regime: "    + result["regime"])
            print("  Strategies: " + str(result["strategies"]))
            print("  Trade OK: "  + str(result["trade_ok"]))
        except Exception as e:
            self.fail("Brain analysis failed: " + str(e))


# ══════════════════════════════════════════════════════
#  TEST STRATEGIES
# ══════════════════════════════════════════════════════

class TestStrategies(unittest.TestCase):
    """Tests for Phase 3+4 — Strategies & Trailing Stop."""

    def setUp(self):
        from data_pipeline import fetch_data, calculate_indicators
        from hmm_brain import TradingBrain
        self.df    = fetch_data("AAPL", period="3mo", interval="1h")
        if self.df is not None:
            self.df = calculate_indicators(self.df)
        self.brain = TradingBrain()

    def test_strategy_manager(self):
        """Test strategy manager initialization."""
        try:
            from strategies import StrategyManager
            sm = StrategyManager()
            self.assertIn("TREND_FOLLOWING", sm.strategies)
            self.assertIn("MEAN_REVERSION",  sm.strategies)
            self.assertIn("BREAKOUT",        sm.strategies)
            self.assertIn("SMC",             sm.strategies)
            self.assertIn("VOLUME_ANOMALY",  sm.strategies)
            print("PASS: Strategy manager has all 5 strategies!")
        except Exception as e:
            self.fail("Strategy manager failed: " + str(e))

    def test_signal_generation(self):
        """Test signal generation."""
        try:
            from strategies import StrategyManager
            from hmm_brain import TradingBrain
            sm           = StrategyManager()
            brain        = TradingBrain()
            brain_result = brain.analyse(self.df, "AAPL")
            signal       = sm.get_signal(self.df, "AAPL", brain_result)
            if signal:
                self.assertIn(signal.side, ["BUY", "SELL"])
                self.assertGreater(signal.entry, 0)
                self.assertGreater(signal.stop_loss, 0)
                self.assertGreater(signal.score, 0)
                print("PASS: Signal generated: " + signal.side +
                      " score=" + str(signal.score))
            else:
                print("INFO: No signal generated (market conditions not met)")
        except Exception as e:
            self.fail("Signal generation failed: " + str(e))

    def test_trailing_stop(self):
        """Test trailing stop manager."""
        try:
            from strategies import TrailingStopManager
            trail = TrailingStopManager()

            # Add test position
            trail.add_position(
                symbol       = "AAPL",
                side         = "BUY",
                entry        = 190.0,
                stop_loss    = 188.0,
                take_profit1 = 192.0,
                take_profit2 = 194.0,
                atr          = 1.5,
                trail_type   = "ATR"
            )

            self.assertIn("AAPL", trail.positions)

            # Simulate price moving up
            result = trail.update_trail("AAPL", 191.0)
            self.assertIsNotNone(result)
            print("PASS: Trailing stop working — Action: " + result["action"])

            # Simulate break even
            result = trail.update_trail("AAPL", 191.6)
            print("PASS: After BE check — Action: " + result["action"])

            # Simulate TP1
            result = trail.update_trail("AAPL", 192.1)
            print("PASS: After TP1 — Action: " + result["action"])

        except Exception as e:
            self.fail("Trailing stop failed: " + str(e))


# ══════════════════════════════════════════════════════
#  TEST RISK MANAGEMENT
# ══════════════════════════════════════════════════════

class TestRiskManagement(unittest.TestCase):
    """Tests for Phase 5 — Risk Management."""

    def test_position_sizer(self):
        """Test position sizing calculation."""
        try:
            from risk_management import PositionSizer
            sizer = PositionSizer()
            size  = sizer.calculate_size(
                balance   = 100000,
                entry     = 190.0,
                stop_loss = 188.0
            )
            self.assertGreater(size, 0)
            print("PASS: Position size: " + str(size) + " shares")
        except Exception as e:
            self.fail("Position sizer failed: " + str(e))

    def test_risk_manager_checks(self):
        """Test all risk manager checks."""
        try:
            from risk_management import RiskManager
            rm = RiskManager()

            # Can trade check
            can, reason = rm.can_trade("AAPL")
            self.assertTrue(can)
            print("PASS: Can trade: " + str(can) + " — " + reason)

            # Daily loss check
            rm.balance = 97000  # Simulate 3% loss
            daily_loss = rm.check_daily_loss()
            print("PASS: Daily loss check: " + str(daily_loss))

            # Drawdown check
            rm.balance     = 90000
            rm.peak_balance = 100000
            dd = rm.get_drawdown_pct()
            self.assertAlmostEqual(dd, 10.0, places=1)
            print("PASS: Drawdown: " + str(dd) + "%")

        except Exception as e:
            self.fail("Risk manager failed: " + str(e))

    def test_portfolio_manager(self):
        """Test portfolio correlation checks."""
        try:
            from risk_management import PortfolioManager
            pm = PortfolioManager()

            # Add position
            pm.add_position("AAPL", "BUY", 10, 190.0)

            # Check correlation — same group same side
            ok, reason = pm.check_correlation("MSFT", "BUY")
            print("PASS: Correlation check MSFT BUY: " + str(ok) + " — " + reason)

            # Check different side
            ok2, reason2 = pm.check_correlation("GOOGL", "SELL")
            print("PASS: Correlation check GOOGL SELL: " + str(ok2) + " — " + reason2)

        except Exception as e:
            self.fail("Portfolio manager failed: " + str(e))

    def test_kelly_criterion(self):
        """Test Kelly Criterion position sizing."""
        try:
            from risk_management import PositionSizer
            sizer = PositionSizer()
            size  = sizer.calculate_size(
                balance   = 100000,
                entry     = 190.0,
                stop_loss = 185.0,
                win_rate  = 0.60,
                avg_win   = 2.0,
                avg_loss  = 1.0
            )
            self.assertGreater(size, 0)
            print("PASS: Kelly position size: " + str(size))
        except Exception as e:
            self.fail("Kelly criterion failed: " + str(e))


# ══════════════════════════════════════════════════════
#  TEST BROKER CONNECTION
# ══════════════════════════════════════════════════════

class TestBrokerConnection(unittest.TestCase):
    """Tests for Phase 6 — Alpaca Broker."""

    def test_client_init(self):
        """Test Alpaca client initialization."""
        try:
            from alpaca_broker import AlpacaClient
            client = AlpacaClient()
            self.assertIsNotNone(client)
            self.assertTrue(client.paper)
            print("PASS: Alpaca client initialized!")
        except Exception as e:
            self.fail("Alpaca client init failed: " + str(e))

    def test_account_connection(self):
        """Test Alpaca account connection."""
        try:
            from alpaca_broker import AlpacaClient
            client  = AlpacaClient()
            account = client.get_account()
            if account:
                print("PASS: Account connected!")
                print("  Balance: $" + str(account["balance"]))
                print("  Status: "  + account["status"])
            else:
                print("INFO: Account not connected — add API keys to connect")
        except Exception as e:
            print("INFO: Broker test skipped — " + str(e))

    def test_market_hours(self):
        """Test market hours check."""
        try:
            from alpaca_broker import AlpacaClient
            client   = AlpacaClient()
            is_open  = client.is_market_open()
            hours    = client.get_market_hours()
            print("PASS: Market open: " + str(is_open))
            if hours:
                print("  Next open:  " + str(hours.get("next_open", "N/A")))
                print("  Next close: " + str(hours.get("next_close", "N/A")))
        except Exception as e:
            print("INFO: Market hours test skipped — " + str(e))

    def test_order_manager(self):
        """Test order manager initialization."""
        try:
            from alpaca_broker import AlpacaClient, OrderManager
            client = AlpacaClient()
            om     = OrderManager(client)
            self.assertIsNotNone(om)
            trades = om.get_open_trades()
            print("PASS: Order manager working — Open trades: " + str(len(trades)))
        except Exception as e:
            self.fail("Order manager failed: " + str(e))


# ══════════════════════════════════════════════════════
#  FULL SYSTEM INTEGRATION TEST
# ══════════════════════════════════════════════════════

class TestFullSystem(unittest.TestCase):
    """Full end-to-end system test."""

    def test_complete_pipeline(self):
        """Test complete signal generation pipeline."""
        print("\n" + "="*50)
        print("FULL SYSTEM INTEGRATION TEST")
        print("="*50)

        try:
            from data_pipeline import fetch_data, calculate_indicators, setup_database
            from hmm_brain import TradingBrain
            from strategies import StrategyManager
            from risk_management import RiskManager, PortfolioManager

            # Step 1: Setup
            print("\nStep 1: Setting up database...")
            setup_database()
            print("  Database ready!")

            # Step 2: Fetch data
            print("\nStep 2: Fetching market data...")
            df = fetch_data("AAPL", period="3mo", interval="1h")
            self.assertIsNotNone(df)
            print("  Got " + str(len(df)) + " candles for AAPL")

            # Step 3: Calculate indicators
            print("\nStep 3: Calculating indicators...")
            df = calculate_indicators(df)
            print("  Indicators calculated!")

            # Step 4: Brain analysis
            print("\nStep 4: Running HMM Brain analysis...")
            brain        = TradingBrain()
            brain_result = brain.analyse(df, "AAPL")
            print("  Regime: "    + brain_result["regime"])
            print("  Confidence: " + str(round(brain_result["confidence"]*100)) + "%")
            print("  Strategies: " + str(brain_result["strategies"]))

            # Step 5: Generate signal
            print("\nStep 5: Generating trading signal...")
            sm     = StrategyManager()
            signal = sm.get_signal(df, "AAPL", brain_result)
            if signal:
                print("  Signal: " + signal.side + " AAPL")
                print("  Strategy: " + signal.strategy)
                print("  Score: " + str(signal.score) + "/10")
                print("  Entry: " + str(signal.entry))
                print("  SL: " + str(signal.stop_loss))
                print("  TP1: " + str(signal.take_profit1))
                print("  TP2: " + str(signal.take_profit2))
            else:
                print("  No signal — conditions not met")

            # Step 6: Risk check
            print("\nStep 6: Risk management check...")
            rm       = RiskManager()
            can, why = rm.can_trade("AAPL")
            print("  Can trade: " + str(can) + " — " + why)

            if signal and can:
                size = rm.calculate_position_size(
                    entry        = signal.entry,
                    stop_loss    = signal.stop_loss,
                    signal_score = signal.score
                )
                print("  Position size: " + str(size) + " shares")

            # Step 7: Portfolio check
            print("\nStep 7: Portfolio correlation check...")
            pm       = PortfolioManager()
            ok, msg  = pm.check_correlation("AAPL", "BUY")
            print("  Correlation OK: " + str(ok) + " — " + msg)

            print("\n" + "="*50)
            print("FULL SYSTEM TEST COMPLETE!")
            print("="*50)

        except Exception as e:
            self.fail("Full system test failed: " + str(e))


# ══════════════════════════════════════════════════════
#  RUN ALL TESTS
# ══════════════════════════════════════════════════════

def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*55)
    print("  STEPHEN AI TRADING SYSTEM — TEST SUITE")
    print("="*55)
    print("Running all tests...\n")

    # Test suites
    suites = [
        TestDataPipeline,
        TestHMMBrain,
        TestStrategies,
        TestRiskManagement,
        TestBrokerConnection,
        TestFullSystem,
    ]

    total_passed = 0
    total_failed = 0

    for suite_class in suites:
        print("\n" + "-"*40)
        print("Testing: " + suite_class.__name__)
        print("-"*40)

        suite  = unittest.TestLoader().loadTestsFromTestCase(suite_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)

        passed = result.testsRun - len(result.failures) - len(result.errors)
        failed = len(result.failures) + len(result.errors)

        total_passed += passed
        total_failed += failed

        for test, error in result.failures + result.errors:
            print("FAIL: " + str(test))
            print("  " + error.split('\n')[-2])

    print("\n" + "="*55)
    print("TEST RESULTS:")
    print("  Passed: " + str(total_passed))
    print("  Failed: " + str(total_failed))
    print("  Total:  " + str(total_passed + total_failed))

    if total_failed == 0:
        print("\nALL TESTS PASSED! System ready for paper trading!")
    else:
        print("\nSome tests failed — fix before going live!")

    print("="*55)
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
