# NSE Cash-Equity Paper Trading

This project is a **simulation only**. It never places brokerage orders and does not connect to Groww. Starting virtual equity is ₹1,00,000.

## Schedule
GitHub Actions tries at 10:15, 10:30 and 10:45 UTC (15:45, 16:00 and 16:15 IST) on weekdays. Later runs retry when the free data feed is incomplete. Manual runs before 15:40 IST exit without changing the ledger. A data-limited run can be retried later the same day and replaces that date's provisional record rather than duplicating it.

## Data and limitations
- Uses Yahoo Finance via `yfinance` for NSE-listed symbols (`.NS`) and NIFTY 50 (`^NSEI`). This is an unofficial source, not an exchange-certified feed. Intraday data can be delayed, incomplete, or unavailable.
- A session is eligible only after 15:40 IST and only when recent 5-minute candles belong to the current IST date, include at least 60 candles, and extend to at least 15:25 IST. Otherwise no trade is simulated; data-limited reports can be retried after the close.
- The script includes the official 2026 NSE equity-holiday list and writes a no-session report on listed holidays without modifying the trading ledger. Update the list when NSE publishes the next year's calendar. Special sessions, including Muhurat trading, are not yet scheduled.
- Simulated execution assumes entry at the next candle open after a signal. Stop/target fills are approximations based on candle OHLC; if both stop and target occur in the same candle, stop is assumed first (conservative).
- Trading costs are an explicit rough estimate of 0.10% of traded notional round-trip, not a broker-specific fee quote. Replace with exact statutory charges before relying on results.
- The daily VWAP/EMA strategy remains an unvalidated paper-trading baseline until the research reports show repeated out-of-sample evidence. Research compares three fixed rule sets; it does not auto-switch strategies or claim that any rule is profitable. Daily targets are not forced.

## Strategy v1
5-minute VWAP + EMA momentum. A long signal requires EMA9 > EMA21, close above VWAP, and a close crossing above the previous candle high. Long-only cash equity; no leverage, short selling, options, or futures. ATR(14) sets a 1x ATR stop and 2R target. Max planned risk ₹250 per trade, max daily net loss ₹500, at most one simulated trade per day in v1, max ₹25,000 notional per position. No averaging down.

## Files
- `state/ledger.json`: persistent virtual equity and daily/trade history.
- `reports/`: dated Markdown reports.
- `research_backtest.py`: historical comparison of VWAP/EMA momentum, opening-range breakout, and VWAP mean reversion with a chronological 70/30 train/test split.
- `.github/workflows/strategy-research.yml`: schedules the research report at 5:00 PM IST on weekdays; reports are saved as `reports/research-YYYY-MM-DD.md`.

## Reports
Daily results are generated on trading weekdays. Friday reports include week-to-date figures; the final trading session of a calendar month includes month-to-date figures. The ledger starts with a baseline on the first successful run; missing prior records are not backfilled.

## Important
Do not add broker credentials, API keys, OTPs, or real account information. This repository was selected from the connected GitHub account and is currently public; keep reports free of personal information. For private records, move the project to a private repository in GitHub before relying on it.
