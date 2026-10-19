# NSE Cash-Equity Paper Trading

This project is a **simulation only**. It never places brokerage orders and does not connect to Groww. Starting virtual equity is ₹1,00,000.

## Schedule
GitHub Actions runs on weekdays at 10:15 UTC (15:45 IST) to allow a buffer after the NSE cash close. Runs may be delayed. Manual runs before 15:40 IST exit without changing the ledger. A data-limited run can be retried later the same day and replaces that date's provisional record rather than duplicating it.

## Data and limitations
- Uses Yahoo Finance via `yfinance` for NSE-listed symbols (`.NS`) and NIFTY 50 (`^NSEI`). This is an unofficial source, not an exchange-certified feed. Intraday data can be delayed, incomplete, or unavailable.
- A session is eligible only after 15:40 IST and only when recent 5-minute candles belong to the current IST date, include at least 60 candles, and extend to at least 15:25 IST. Otherwise no trade is simulated; the script records a data-limited report only after the post-close window.
- Simulated execution assumes entry at the next candle open after a signal. Stop/target fills are approximations based on candle OHLC; if both stop and target occur in the same candle, stop is assumed first (conservative).
- Trading costs are an explicit rough estimate of 0.10% of traded notional round-trip, not a broker-specific fee quote. Replace with exact statutory charges before relying on results.
- The strategy is an unvalidated research baseline, not a claim of profitability. Daily targets are not forced.

## Strategy v1
5-minute VWAP + EMA momentum. A long signal requires EMA9 > EMA21, close above VWAP, and a close crossing above the previous candle high. Long-only cash equity; no leverage, short selling, options, or futures. ATR(14) sets a 1x ATR stop and 2R target. Max planned risk ₹250 per trade, max daily net loss ₹500, at most one simulated trade per day in v1, max ₹25,000 notional per position. No averaging down.

## Files
- `state/ledger.json`: persistent virtual equity and daily/trade history.
- `reports/`: dated Markdown reports.

## Reports
Daily results are generated on trading weekdays. Friday reports include week-to-date figures; the final trading session of a calendar month includes month-to-date figures. The ledger starts with a baseline on the first successful run; missing prior records are not backfilled.

## Important
Do not add broker credentials, API keys, OTPs, or real account information. This repository was selected from the connected GitHub account and is currently public; keep reports free of personal information. For private records, move the project to a private repository in GitHub before relying on it.
