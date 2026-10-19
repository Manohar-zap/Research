import unittest

import pandas as pd

from research_backtest import filter_valid_ohlcv, indicators


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

    def test_indicators_reset_between_sessions(self):
        first = pd.date_range("2026-10-01 09:15", periods=8, freq="5min")
        second = pd.date_range("2026-10-02 09:15", periods=8, freq="5min")
        index = first.append(second)
        close = [100 + i for i in range(8)] + [1000 + i for i in range(8)]
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
        self.assertEqual(calculated.loc[first_bar_next_session, "EMA9"], 1000.0)
        self.assertTrue(pd.isna(calculated.loc[first_bar_next_session, "PREV_CLOSE"]))
        self.assertEqual(calculated.loc[second[6], "ATR14"], 2.0)


if __name__ == "__main__":
    unittest.main()
