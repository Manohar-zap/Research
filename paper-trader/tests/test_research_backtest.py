import unittest

import pandas as pd

from research_backtest import filter_valid_ohlcv


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


if __name__ == "__main__":
    unittest.main()
