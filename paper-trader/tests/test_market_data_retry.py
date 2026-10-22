from datetime import date
from unittest import TestCase
from unittest.mock import Mock, patch

import pandas as pd

import paper_trader
from market_data import expected_session_times


def candles(count=None, interval="5m"):
    times = list(expected_session_times(interval))
    if count is None:
        count = len(times)
    times = times[:count]
    idx = pd.DatetimeIndex([
        pd.Timestamp.combine(pd.Timestamp("2026-10-21").date(), t)
        for t in times
    ]).tz_localize(paper_trader.TZ)
    return pd.DataFrame({
        "Open": [100.0] * count,
        "High": [101.0] * count,
        "Low": [99.0] * count,
        "Close": [100.0] * count,
        "Volume": [1000.0] * count,
    }, index=idx)


class MarketDataRetryTests(TestCase):
    def test_validator_accepts_only_complete_valid_session(self):
        good, err = paper_trader._prepare_intraday_frame(candles(), date(2026, 10, 21))
        self.assertIsNone(err)
        self.assertEqual(len(good), 75)

        partial, err = paper_trader._prepare_intraday_frame(candles(74), date(2026, 10, 21))
        self.assertIsNone(partial)
        self.assertIn("Incomplete 5m candle grid", err)

        good15, err = paper_trader._prepare_intraday_frame(
            candles(interval="15m"), date(2026, 10, 21), "15m"
        )
        self.assertIsNone(err)
        self.assertEqual(len(good15), 25)

        partial15, err = paper_trader._prepare_intraday_frame(
            candles(24, interval="15m"), date(2026, 10, 21), "15m"
        )
        self.assertIsNone(partial15)
        self.assertIn("Incomplete 15m candle grid", err)

    def test_downloader_falls_back_to_history_and_keeps_strict_validation(self):
        partial = candles(74)
        complete = candles(75)
        ticker = Mock()
        ticker.history.return_value = complete
        with patch.object(paper_trader.yf, "download", return_value=partial), \
             patch.object(paper_trader.yf, "Ticker", return_value=ticker), \
             patch("time.sleep", return_value=None):
            frame, err = paper_trader.download_today("TEST.NS", date(2026, 10, 21))
        self.assertIsNone(err)
        self.assertEqual(len(frame), 75)
        ticker.history.assert_called_once()
        self.assertIs(ticker.history.call_args.kwargs["repair"], False)

    def test_no_provider_response_can_turn_incomplete_data_into_a_tradeable_frame(self):
        partial = candles(74)
        ticker = Mock()
        ticker.history.return_value = partial
        with patch.object(paper_trader.yf, "download", return_value=partial), \
             patch.object(paper_trader.yf, "Ticker", return_value=ticker), \
             patch("time.sleep", return_value=None):
            frame, err = paper_trader.download_today("TEST.NS", date(2026, 10, 21))
        self.assertIsNone(frame)
        self.assertIn("All free-source attempts failed strict validation", err)
        self.assertIn("Incomplete 5m candle grid", err)


    def test_watchlist_uses_15m_only_when_all_symbols_are_valid(self):
        today = date(2026, 10, 21)

        def fake_download(symbol, requested_date, interval="5m"):
            if interval == "5m":
                return None, "simulated missing 5m candle"
            return candles(interval="15m"), None

        with patch.object(paper_trader, "download_today", side_effect=fake_download):
            frames, interval, diagnostics = paper_trader.download_watchlist(today)
        self.assertEqual(interval, "15m")
        self.assertEqual(set(frames), set(paper_trader.SYMBOLS))
        self.assertTrue(any("switched to 15m" in item for item in diagnostics))

    def test_watchlist_rejects_partial_15m_universe(self):
        today = date(2026, 10, 21)

        def fake_download(symbol, requested_date, interval="5m"):
            if interval == "5m":
                return None, "simulated missing 5m candle"
            if symbol == paper_trader.SYMBOLS[-1]:
                return None, "simulated missing 15m candle"
            return candles(interval="15m"), None

        with patch.object(paper_trader, "download_today", side_effect=fake_download):
            frames, interval, diagnostics = paper_trader.download_watchlist(today)
        self.assertIsNone(interval)
        self.assertEqual(frames, {})
        self.assertTrue(any("15m: only 7/8 symbols passed" in item for item in diagnostics))
