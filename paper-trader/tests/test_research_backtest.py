import unittest

import pandas as pd

from research_backtest import filter_valid_ohlcv, indicators
from market_data import expected_session_times, is_complete_session
from paper_trader import indicators as paper_indicators, select_portfolio_trade


class ValidOhlcvTests(unittest.TestCase):
    def test_keeps_valid_candles_and_rejects_malformed_rows(self):
        frame = pd.DataFrame(
            {
                "Open":   [100, 100, 100, 100, 0],
                "High":   [103, 103, 99, 103, 103],
                "Low":    [99, 99, 99, 101, 99],
                "Close":  [102, 102, 102, 102, 102],
                "Volume": [1000, -1, 1000, 1000, 1000],
            },
            index=pd.date_range("2026-10-01 09:15", periods=5, freq="5min"),
        )
        clean = filter_valid_ohlcv(frame)
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean.index[0], frame.index[0])

    def test_coerces_numeric_strings_and_drops_missing_values(self):
        frame = pd.DataFrame(
            {
                "Open": ["100", "100"],
                "High": ["102", "102"],
                "Low": ["99", "99"],
                "Close": ["101", None],
                "Volume": ["500", "500"],
            }
        )
        clean = filter_valid_ohlcv(frame)
        self.assertEqual(len(clean), 1)
        self.assertEqual(float(clean.iloc[0]["Close"]), 101.0)

    def test_missing_required_column_returns_empty(self):
        frame = pd.DataFrame({"Open": [100], "Close": [101]})
        clean = filter_valid_ohlcv(frame)
        self.assertTrue(clean.empty)

    def test_expected_session_grids_match_nse_intraday_intervals(self):
        self.assertEqual(len(expected_session_times("5m")), 75)
        self.assertEqual(len(expected_session_times("15m")), 25)

    def test_accepts_complete_timezone_aware_session_for_both_intervals(self):
        for interval, frequency, count in (("5m", "5min", 75), ("15m", "15min", 25)):
            with self.subTest(interval=interval):
                index = pd.date_range(
                    "2026-10-01 09:15",
                    periods=count,
                    freq=frequency,
                    tz="Asia/Kolkata",
                )
                self.assertTrue(is_complete_session(index, interval))

    def test_rejects_session_with_missing_midday_candle(self):
        index = pd.date_range(
            "2026-10-01 09:15", periods=75, freq="5min", tz="Asia/Kolkata"
        ).delete(37)
        # This 74-bar session used to pass the old len(df) >= 60 check.
        self.assertFalse(is_complete_session(index, "5m"))

    def test_rejects_duplicate_and_out_of_order_timestamps(self):
        index = pd.date_range(
            "2026-10-01 09:15", periods=75, freq="5min", tz="Asia/Kolkata"
        )
        self.assertFalse(is_complete_session(index.insert(1, index[0]), "5m"))
        shuffled = index.delete(20).insert(20, index[21])
        self.assertFalse(is_complete_session(shuffled, "5m"))

    def test_indicators_reset_between_sessions(self):
        first = pd.date_range("2026-10-01 09:15", periods=24, freq="5min")
        second = pd.date_range("2026-10-02 09:15", periods=24, freq="5min")
        index = first.append(second)
        first_close = [100 + i for i in range(24)]
        second_close = [1000 + i for i in range(24)]
        close = first_close + second_close
        frame = pd.DataFrame(
            {
                "Open": close,
                "High": [value + 1 for value in close],
                "Low": [value - 1 for value in close],
                "Close": close,
                "Volume": [1000] * len(close),
            },
            index=index,
        )

        calculated = indicators(frame)
        first_bar_next_session = second[0]
        self.assertTrue(pd.isna(calculated.loc[first_bar_next_session, "EMA9"]))
        self.assertTrue(pd.isna(calculated.loc[first_bar_next_session, "EMA21"]))
        self.assertTrue(pd.isna(calculated.loc[first_bar_next_session, "PREV_CLOSE"]))
        self.assertEqual(calculated.loc[second[6], "ATR14"], 2.0)
        expected_ema21 = pd.Series(second_close[:21]).ewm(
            span=21, adjust=False, min_periods=21
        ).mean().iloc[-1]
        self.assertAlmostEqual(calculated.loc[second[20], "EMA21"], expected_ema21)

    def test_paper_simulator_requires_full_ema_warmup(self):
        session = pd.date_range("2026-10-01 09:15", periods=24, freq="5min")
        close = [100 + i for i in range(24)]
        frame = pd.DataFrame(
            {
                "Open": close,
                "High": [value + 1 for value in close],
                "Low": [value - 1 for value in close],
                "Close": close,
                "Volume": [1000] * len(close),
            },
            index=session,
        )
        calculated = paper_indicators(frame)
        self.assertTrue(pd.isna(calculated.iloc[0]["EMA9"]))
        self.assertTrue(pd.isna(calculated.iloc[0]["EMA21"]))
        self.assertTrue(pd.isna(calculated.iloc[19]["EMA21"]))
        self.assertFalse(pd.isna(calculated.iloc[20]["EMA21"]))

    def test_portfolio_selects_earliest_entry_not_watchlist_order(self):
        later = {
            "ticker": "RELIANCE.NS",
            "entry_time_ist": "2026-10-19T10:00:00+05:30",
        }
        earlier = {
            "ticker": "TCS.NS",
            "entry_time_ist": "2026-10-19T09:45:00+05:30",
        }
        self.assertEqual(select_portfolio_trade([later, earlier])["ticker"], "TCS.NS")
        self.assertEqual(select_portfolio_trade([earlier, later])["ticker"], "TCS.NS")

    def test_portfolio_selection_is_deterministic_for_equal_entry_times(self):
        a = {"ticker": "RELIANCE.NS", "entry_time_ist": "2026-10-19T10:00:00+05:30"}
        b = {"ticker": "INFY.NS", "entry_time_ist": "2026-10-19T10:00:00+05:30"}
        self.assertEqual(select_portfolio_trade([a, b])["ticker"], "INFY.NS")
        self.assertIsNone(select_portfolio_trade([]))



if __name__ == "__main__":
    unittest.main()
