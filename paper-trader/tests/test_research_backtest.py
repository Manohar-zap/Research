import unittest

import pandas as pd

from research_backtest import common_session_dates, filter_valid_ohlcv, indicators, merge_session_history, net_pnl_at_cost_rate
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

    def test_interval_selection_uses_shared_dates_not_union(self):
        first = pd.DataFrame(
            index=pd.to_datetime(["2026-10-01 09:15", "2026-10-02 09:15"])
        )
        second = pd.DataFrame(
            index=pd.to_datetime(["2026-10-02 09:15", "2026-10-03 09:15"])
        )
        self.assertEqual(
            [str(day) for day in common_session_dates({"A": first, "B": second})],
            ["2026-10-02"],
        )
        self.assertEqual(common_session_dates({}), [])

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

    def test_cost_sensitivity_reprices_fixed_fills_without_changing_gross_pnl(self):
        trades = [{"entry": 100.0, "exit": 110.0, "qty": 10}]
        self.assertAlmostEqual(net_pnl_at_cost_rate(trades, 0.001), 98.95)
        self.assertLess(
            net_pnl_at_cost_rate(trades, 0.002),
            net_pnl_at_cost_rate(trades, 0.001),
        )

    def test_cost_sensitivity_rejects_negative_or_nonfinite_rates(self):
        trades = [{"entry": 100.0, "exit": 101.0, "qty": 1}]
        for rate in (-0.001, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                net_pnl_at_cost_rate(trades, rate)

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


    def test_session_cache_accumulates_only_complete_dates(self):
        def make_session(day, start_price):
            index = pd.date_range(
                f"{day} 09:15", periods=25, freq="15min", tz="Asia/Kolkata"
            )
            close = [start_price + i for i in range(25)]
            return pd.DataFrame(
                {
                    "Open": close,
                    "High": [value + 1 for value in close],
                    "Low": [value - 1 for value in close],
                    "Close": close,
                    "Volume": [1000] * len(close),
                },
                index=index,
            )

        cached = make_session("2026-10-01", 100)
        fresh = make_session("2026-10-02", 200)
        merged = merge_session_history(cached, fresh, "15m")
        self.assertEqual(len(merged), 50)
        self.assertEqual(len(set(merged.index.date)), 2)
        # Fresh history with a missing candle must not enter the cache.
        incomplete = fresh.drop(fresh.index[10])
        merged_with_gap = merge_session_history(cached, incomplete, "15m")
        self.assertEqual(len(merged_with_gap), 25)
        self.assertEqual(set(merged_with_gap.index.date), {cached.index[0].date()})

    def test_session_cache_prefers_fresh_values_on_overlapping_timestamp(self):
        index = pd.date_range(
            "2026-10-01 09:15", periods=25, freq="15min", tz="Asia/Kolkata"
        )
        cached_close = [100 + i for i in range(25)]
        fresh_close = [200 + i for i in range(25)]

        def frame(closes):
            return pd.DataFrame(
                {
                    "Open": closes,
                    "High": [value + 1 for value in closes],
                    "Low": [value - 1 for value in closes],
                    "Close": closes,
                    "Volume": [1000] * len(closes),
                },
                index=index,
            )

        merged = merge_session_history(frame(cached_close), frame(fresh_close), "15m")
        self.assertEqual(len(merged), 25)
        self.assertEqual(float(merged.iloc[0]["Close"]), 200.0)



if __name__ == "__main__":
    unittest.main()
