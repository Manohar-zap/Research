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
- This automation is a deterministic parameter-search loop, not an AI model retraining itself. The variant grid is fixed and versioned to reduce result-chasing. The daily VWAP/EMA strategy stays the baseline; research cannot automatically replace it. Any shadow candidate must also earn at least 30 genuine forward paper trades before a later review. Daily profit targets are not forced.

## Strategy v1
5-minute VWAP + EMA momentum. A long signal requires EMA9 > EMA21, close above VWAP, and a close crossing above the previous candle high. Long-only cash equity; no leverage, short selling, options, or futures. ATR(14) sets a 1x ATR stop and 2R target. Max planned risk ₹250 per trade, max daily net loss ₹500, at most one simulated trade per day in v1, max ₹25,000 notional per position. No averaging down.

## Files
- `state/ledger.json`: persistent virtual equity and daily/trade history.
- `reports/`: dated Markdown reports.
- `research_backtest.py`: runs a bounded 20-variant parameter search across VWAP/EMA momentum, opening-range breakout, and VWAP mean reversion. It uses chronological train/validation/holdout slices; variants are ranked on validation only, and only the validation winner is evaluated on the final holdout. It prefers 5-minute candles and retries the entire watchlist at 15-minute resolution when fewer than 15 complete dates are available.
- `state/experiment_history.json`: persistent experiment audit log, including all variant training/validation metrics, latest holdout result, data-end date, and status. Identical data windows update an existing audit record rather than count as new independent evidence.
- `state/shadow_paper_candidate.json`: created only when the validation-selected variant passes the separate holdout screen. This is a review/shadow candidate only; it does not switch the daily strategy or place orders.
- `.github/workflows/strategy-research.yml`: runs the parameter search on weekdays at 5:00 PM IST and when research code changes. Reports are saved as `reports/research-YYYY-MM-DD.md`; state files are committed to the repository.

## Reports
Daily results are generated on trading weekdays. Friday reports include week-to-date figures; the final trading session of a calendar month includes month-to-date figures. The ledger starts with a baseline on the first successful run; missing prior records are not backfilled.

## Important
Do not add broker credentials, API keys, OTPs, or real account information. This repository was selected from the connected GitHub account and is currently public; keep reports free of personal information. For private records, move the project to a private repository in GitHub before relying on it.
