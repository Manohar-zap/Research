from __future__ import annotations

import json
import math
from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

TZ = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
SYMBOLS = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
           "TCS.NS", "SBIN.NS", "LT.NS", "ITC.NS"]
START_EQUITY = 100_000.0
RISK_PER_TRADE = 250.0
MAX_NOTIONAL = 25_000.0
MAX_TRADES_PER_DAY = 2
DAILY_LOSS_LIMIT = 500.0
COST_RATE = 0.001  # approximate all-in round-trip cost, not a fee quote
STRATEGIES = ("VWAP_EMA", "OPENING_RANGE_BREAKOUT", "VWAP_MEAN_REVERSION")


def get_bars(symbol: str, interval: str) -> tuple[pd.DataFrame, str | None]:
    min_bars = 60 if interval == "5m" else 20
    min_last_time = dt_time(15, 25) if interval == "5m" else dt_time(15, 15)
    try:
        raw = yf.download(symbol, period="60d", interval=interval, auto_adjust=False,
                           progress=False, threads=False, prepost=False)
        if raw is None or raw.empty:
            return pd.DataFrame(), "No 5-minute data returned"
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not set(required).issubset(df.columns):
            return pd.DataFrame(), "Missing OHLCV columns"
        idx = pd.to_datetime(df.index)
        idx = idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)
        df.index = idx
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[required].apply(pd.to_numeric, errors="coerce").dropna()
        df = df[(df["Close"] > 0) & (df["High"] > 0) & (df["Low"] > 0)]
        df = df.between_time("09:15", "15:30")
        valid = []
        for d, day in df.groupby(df.index.date, sort=True):
            if len(day) >= min_bars and day.index[-1].time() >= min_last_time:
                valid.append(d)
        if not valid:
            return pd.DataFrame(), "No complete sessions passed candle-count/time checks"
        df = df[[ts.date() in set(valid) for ts in df.index]]
        df.attrs["interval"] = interval
        df.attrs["min_bars"] = min_bars
        df.attrs["min_last_time"] = min_last_time
        return df, None
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.attrs.update(df.attrs)
    session = pd.Series(x.index.date, index=x.index)
    typical = (x["High"] + x["Low"] + x["Close"]) / 3
    volume = x["Volume"].clip(lower=0)
    x["VWAP"] = (typical * volume).groupby(session).cumsum() / volume.groupby(session).cumsum().replace(0, float("nan"))
    x["EMA9"] = x["Close"].ewm(span=9, adjust=False).mean()
    x["EMA21"] = x["Close"].ewm(span=21, adjust=False).mean()
    prev_close = x["Close"].shift(1)
    tr = pd.concat([x["High"] - x["Low"], (x["High"] - prev_close).abs(),
                    (x["Low"] - prev_close).abs()], axis=1).max(axis=1)
    x["ATR14"] = tr.rolling(14, min_periods=7).mean()
    delta = x["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    x["RSI14"] = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))
    x["PREV_HIGH"] = x["High"].shift(1)
    x["PREV_CLOSE"] = x["Close"].shift(1)
    x["PREV_EMA9"] = x["EMA9"].shift(1)
    x["OR_HIGH"] = float("nan")
    x["VOL_MED20"] = float("nan")
    for _, day in x.groupby(session, sort=True):
        opening = day.loc[[ts.time() < dt_time(9, 45) for ts in day.index], "High"]
        or_high = float(opening.max()) if not opening.empty else float("nan")
        x.loc[day.index, "OR_HIGH"] = or_high
        x.loc[day.index, "VOL_MED20"] = day["Volume"].shift(1).rolling(20, min_periods=10).median()
    times = pd.Series([ts.time() for ts in x.index], index=x.index)
    active = (times >= dt_time(9, 45)) & (times <= dt_time(14, 45))
    x["SIG_VWAP_EMA"] = (active & (x["EMA9"] > x["EMA21"]) & (x["Close"] > x["VWAP"])
                         & (x["Close"] > x["PREV_HIGH"]) & (x["PREV_CLOSE"] <= x["PREV_HIGH"]))
    x["SIG_ORB"] = (active & (times <= dt_time(14, 30)) & (x["EMA9"] > x["EMA21"])
                    & (x["Close"] > x["OR_HIGH"]) & (x["PREV_CLOSE"] <= x["OR_HIGH"])
                    & (x["Volume"] >= 1.2 * x["VOL_MED20"]))
    x["SIG_MEANREV"] = (active & (x["Close"] < x["VWAP"] - x["ATR14"])
                        & (x["RSI14"] < 30) & (x["PREV_CLOSE"] <= x["PREV_EMA9"])
                        & (x["Close"] > x["EMA9"]))
    return x


def candidates_for(symbol: str, x: pd.DataFrame) -> list[dict]:
    out = []
    min_bars = int(x.attrs.get("min_bars", 60))
    min_last_time = x.attrs.get("min_last_time", dt_time(15, 25))
    col_for = {"VWAP_EMA": "SIG_VWAP_EMA",
               "OPENING_RANGE_BREAKOUT": "SIG_ORB",
               "VWAP_MEAN_REVERSION": "SIG_MEANREV"}
    for strategy, signal_col in col_for.items():
        for session_date, day in x.groupby(x.index.date, sort=True):
            if len(day) < min_bars or day.index[-1].time() < min_last_time:
                continue
            flags = day[signal_col].fillna(False).astype(bool).to_numpy()
            for i, signal in enumerate(flags[:-1]):
                if not signal:
                    continue
                signal_time, entry_time = day.index[i], day.index[i + 1]
                if entry_time.time() > dt_time(14, 50):
                    continue
                atr = float(day["ATR14"].iloc[i])
                entry = float(day["Open"].iloc[i + 1])
                if not (math.isfinite(atr) and atr > 0 and math.isfinite(entry) and entry > 0):
                    continue
                stop = entry - atr
                if strategy == "VWAP_MEAN_REVERSION":
                    vwap = float(day["VWAP"].iloc[i])
                    if not math.isfinite(vwap) or vwap <= entry + 0.25 * atr:
                        continue
                    target = min(vwap, entry + 2 * atr)
                else:
                    target = entry + 2 * atr
                if stop <= 0 or target <= entry:
                    continue
                exit_price, exit_time, reason = None, None, None
                for j in range(i + 1, len(day)):
                    bar, ts = day.iloc[j], day.index[j]
                    op, low, high = float(bar["Open"]), float(bar["Low"]), float(bar["High"])
                    if low <= stop:
                        exit_price = min(op, stop) if op < stop else stop
                        exit_time, reason = ts, "Stop-loss"
                        break
                    if high >= target:
                        exit_price, exit_time, reason = target, ts, "Target"
                        break
                    if ts.time() >= dt_time(15, 15):
                        exit_price, exit_time, reason = float(bar["Close"]), ts, "End-of-day exit"
                        break
                if exit_price is None:
                    exit_price, exit_time, reason = float(day["Close"].iloc[-1]), day.index[-1], "Last available candle"
                out.append({"date": str(session_date), "strategy": strategy, "ticker": symbol,
                            "signal_time": signal_time, "entry_time": entry_time,
                            "entry": entry, "exit_time": exit_time, "exit": exit_price,
                            "stop": stop, "target": target, "reason": reason})
    return out


def simulate(candidates: list[dict], dates: set) -> list[dict]:
    chosen = sorted((c for c in candidates if c["entry_time"].date() in dates),
                    key=lambda c: (c["entry_time"], c["ticker"]))
    trades, available_at, equity = [], None, START_EQUITY
    daily_counts: dict[str, int] = {}
    daily_pnl: dict[str, float] = {}
    for c in chosen:
        d = c["date"]
        if daily_counts.get(d, 0) >= MAX_TRADES_PER_DAY:
            continue
        if daily_pnl.get(d, 0.0) <= -DAILY_LOSS_LIMIT:
            continue
        if available_at is not None and c["entry_time"] <= available_at:
            continue  # one portfolio position at a time for this strategy
        risk = c["entry"] - c["stop"]
        if risk <= 0:
            continue
        qty = min(int(RISK_PER_TRADE / risk), int(MAX_NOTIONAL / c["entry"]), int(equity / c["entry"]))
        if qty < 1:
            continue
        gross = (c["exit"] - c["entry"]) * qty
        costs = (c["entry"] * qty + c["exit"] * qty) * COST_RATE / 2
        net = gross - costs
        trades.append({"date": c["date"], "ticker": c["ticker"], "entry_time": c["entry_time"],
                       "exit_time": c["exit_time"], "gross": gross, "costs": costs, "net": net,
                       "reason": c["reason"], "entry": c["entry"], "exit": c["exit"], "qty": qty})
        equity += net
        daily_counts[d] = daily_counts.get(d, 0) + 1
        daily_pnl[d] = daily_pnl.get(d, 0.0) + net
        available_at = c["exit_time"]
    return trades


def stats(trades: list[dict]) -> dict:
    p = [t["net"] for t in trades]
    win, loss = [v for v in p if v > 0], [v for v in p if v < 0]
    gross_loss = abs(sum(loss))
    equity = peak = START_EQUITY
    dd = 0.0
    for value in p:
        equity += value
        peak = max(peak, equity)
        dd = max(dd, (peak - equity) / peak * 100 if peak else 0)
    return {
        "n": len(p), "net": sum(p), "return": sum(p) / START_EQUITY * 100,
        "winrate": len(win) / len(p) * 100 if p else None,
        "avg": sum(p) / len(p) if p else None,
        "pf": sum(win) / gross_loss if gross_loss else None,
        "dd": dd, "costs": sum(t["costs"] for t in trades),
    }


def fnum(v, suffix=""):
    return "N/A" if v is None else f"{v:,.2f}{suffix}"


def nifty_benchmark(test_dates: set) -> str:
    try:
        raw = yf.download("^NSEI", period="60d", interval="1d", auto_adjust=False,
                          progress=False, threads=False)
        if raw is None or raw.empty:
            return "Unavailable: no daily NIFTY 50 candles returned."
        nd = raw.copy()
        if isinstance(nd.columns, pd.MultiIndex):
            nd.columns = nd.columns.get_level_values(0)
        if "Close" not in nd.columns:
            return "Unavailable: closing-price column missing."
        idx = pd.to_datetime(nd.index)
        idx = idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)
        nd.index = idx
        nd = nd[[stamp.date() in test_dates for stamp in nd.index]].dropna(subset=["Close"])
        if len(nd) < 2:
            return "Unavailable: fewer than two NIFTY 50 closes overlap the out-of-sample dates."
        first_date, last_date = nd.index[0].date(), nd.index[-1].date()
        first_close, last_close = float(nd["Close"].iloc[0]), float(nd["Close"].iloc[-1])
        change_pct = (last_close / first_close - 1) * 100 if first_close else float("nan")
        return (f"{first_date} close ₹{first_close:,.2f} to {last_date} close ₹{last_close:,.2f}; "
                f"close-to-close change {change_pct:.3f}% across the selected test dates. "
                "This is a benchmark reference, not a directly risk-matched strategy comparison.")
    except Exception as exc:
        return f"Unavailable: {type(exc).__name__}: {exc}"


def main():
    now = datetime.now(TZ)
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = [f"# Intraday Strategy Research — {now:%Y-%m-%d}", "",
              f"**Generated IST:** {now.isoformat()}",
              "**Mode:** Historical simulation only; no brokerage orders.",
              "**Initial equity per strategy/split:** ₹1,00,000", "",
              "## Methods",
              "- Compare three predeclared long-only strategies: VWAP/EMA momentum, opening-range breakout, and VWAP mean reversion.",
              "- 5-minute candles; enter at next candle open; one open portfolio position at a time per strategy.",
              "- Planned risk ₹250/trade, max notional ₹25,000/trade, max 2 trades/day and ₹500 daily-loss stop; stop-first if stop and target touch in the same candle.",
              "- Estimated 0.10% round-trip costs. This is a rough proxy, not an exact Groww charge schedule.",
              "- 70% chronological in-sample / 30% held-out out-of-sample split; no parameter search is performed.", "",
              "## Data coverage"]
    data, failures = {}, []
    for symbol in SYMBOLS:
        df, err = get_bars(symbol)
        if err:
            failures.append((symbol, err))
        else:
            x = indicators(df)
            data[symbol] = x
            dates = sorted(set(x.index.date))
            report.append(f"- {symbol}: {len(x)} candles, {len(dates)} complete sessions ({dates[0]} to {dates[-1]})")
    for symbol, error in failures:
        report.append(f"- {symbol}: not usable — {error}")
    all_dates = sorted({ts.date() for x in data.values() for ts in x.index})
    if not data or len(all_dates) < 15:
        report += ["", "## Decision", "**DATA-LIMITED:** insufficient verified session history to compare strategies. No strategy is promoted; no results are invented.",
                   "Retry later or supply a more reliable historical intraday feed."]
    else:
        cut = max(1, min(len(all_dates) - 1, int(len(all_dates) * 0.70)))
        train_dates, test_dates = set(all_dates[:cut]), set(all_dates[cut:])
        report += ["", "## Time split",
                   f"- In-sample: {min(train_dates)} to {max(train_dates)} ({len(train_dates)} session dates)",
                   f"- Held-out out-of-sample: {min(test_dates)} to {max(test_dates)} ({len(test_dates)} session dates)",
                   "", "## NIFTY 50 benchmark",
                   "- " + nifty_benchmark(test_dates),
                   "", "## Results",
                   "| Strategy | Period | Trades | Net P/L | Return | Win rate | Avg/trade | Profit factor | Max drawdown | Costs |",
                   "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        cache = {sym: candidates_for(sym, x) for sym, x in data.items()}
        results = {}
        for strategy in STRATEGIES:
            candidates = [c for rows in cache.values() for c in rows if c["strategy"] == strategy]
            tr, te = simulate(candidates, train_dates), simulate(candidates, test_dates)
            sm = stats(tr), stats(te)
            results[strategy] = sm[1]
            for label, m in (("In-sample", sm[0]), ("Out-of-sample", sm[1])):
                report.append(f"| {strategy} | {label} | {m['n']} | ₹{m['net']:,.2f} | {m['return']:.3f}% | "
                              f"{fnum(m['winrate'], '%')} | ₹{fnum(m['avg'])} | {fnum(m['pf'])} | "
                              f"{m['dd']:.3f}% | ₹{m['costs']:,.2f} |")
        qualified = [name for name, m in results.items()
                     if m["n"] >= 30 and m["net"] > 0 and m["pf"] is not None and m["pf"] >= 1.2]
        report += ["", "## Research decision"]
        if qualified:
            report.append("Minimum research screen passed (not a profitability guarantee): " + ", ".join(qualified) +
                          ". Continue with longer walk-forward testing and paper-trading confirmation before any strategy change.")
        else:
            report.append("No strategy is promoted. Minimum screen: at least 30 out-of-sample trades, positive net P/L after estimated costs, and profit factor at least 1.20.")
    report += ["", "## Limitations",
               "- Yahoo Finance via yfinance is an unofficial feed; historical 5-minute bars can be missing, revised, or delayed.",
               "- OHLC candles cannot reveal exact intrabar price sequence, bid-ask spread, queue priority, or actual fills.",
               "- The 0.10% cost estimate is approximate; confirm actual brokerage, STT, exchange, GST, stamp-duty and SEBI charges separately.",
               "- The stock universe is a fixed shortlist and has selection bias. A 60-day intraday sample is not enough to establish durable profitability.",
               "- This report does not automatically change the daily paper-trading strategy. Do not infer future returns from backtest results.",
               "- The ₹1,000–₹3,000 daily target is not forced; no-trade days are acceptable."]
    path = REPORTS / f"research-{now:%Y-%m-%d}.md"
    path.write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
