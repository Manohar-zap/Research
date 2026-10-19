from __future__ import annotations

import json
import math
from datetime import datetime, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from market_data import is_complete_session

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


def filter_valid_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with positive prices, nonnegative volume, and internally consistent OHLC."""
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(df.columns):
        return pd.DataFrame(columns=required, index=df.index[:0])
    clean = df[required].apply(pd.to_numeric, errors="coerce").dropna().copy()
    return clean[
        (clean["Open"] > 0) & (clean["High"] > 0) & (clean["Low"] > 0) & (clean["Close"] > 0)
        & (clean["Volume"] >= 0)
        & (clean["High"] >= clean[["Open", "Close", "Low"]].max(axis=1))
        & (clean["Low"] <= clean[["Open", "Close", "High"]].min(axis=1))
    ]


def session_ewm(series: pd.Series, span: int, min_periods: int = 1) -> pd.Series:
    """Calculate an EMA independently for each trading session."""
    result = pd.Series(float("nan"), index=series.index, dtype=float)
    for _, session_values in series.groupby(series.index.date, sort=True):
        result.loc[session_values.index] = session_values.ewm(
            span=span, adjust=False, min_periods=min_periods
        ).mean()
    return result


def complete_sessions_only(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Keep only complete, duplicate-free, internally valid intraday sessions."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    clean = filter_valid_ohlcv(df)
    if clean.empty:
        return clean
    valid_dates = [
        day_date
        for day_date, day in clean.groupby(clean.index.date, sort=True)
        if is_complete_session(day.index, interval)
    ]
    valid_date_set = set(valid_dates)
    return clean[[stamp.date() in valid_date_set for stamp in clean.index]].copy()


def merge_session_history(
    cached: pd.DataFrame,
    fresh: pd.DataFrame,
    interval: str,
    retention_days: int = 180,
) -> pd.DataFrame:
    """Merge only complete sessions, preferring the latest downloaded bars on overlap."""
    cached_valid = complete_sessions_only(cached, interval)
    fresh_valid = complete_sessions_only(fresh, interval)
    if cached_valid.empty and fresh_valid.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    combined = pd.concat([cached_valid, fresh_valid]).sort_index(kind="stable")
    # Overlap between a complete cached session and a complete refresh is expected;
    # stable sorting leaves fresh rows last for each shared timestamp.
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = complete_sessions_only(combined, interval)
    if not combined.empty and retention_days > 0:
        cutoff = combined.index.max() - pd.Timedelta(days=retention_days)
        combined = combined[combined.index >= cutoff]
    return combined


def _normalize_index_timezone(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index)
    idx = idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)
    out.index = idx
    return out


def get_bars(symbol: str, interval: str) -> tuple[pd.DataFrame, str | None]:
    min_bars = 60 if interval == "5m" else 20
    min_last_time = dt_time(15, 25) if interval == "5m" else dt_time(15, 15)
    required = ["Open", "High", "Low", "Close", "Volume"]
    cache_path = ROOT / "state" / "market_data_cache" / f"{symbol.replace('.', '_')}_{interval}.csv"
    cached = pd.DataFrame(columns=required)
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cached = _normalize_index_timezone(cached)
            cached = cached[required] if set(required).issubset(cached.columns) else pd.DataFrame(columns=required)
            cached = filter_valid_ohlcv(cached).between_time("09:15", "15:30")
            cached = complete_sessions_only(cached, interval)
        except Exception:
            # A damaged cache must never block a fresh market-data fetch.
            cached = pd.DataFrame(columns=required)

    fresh = pd.DataFrame(columns=required)
    fetch_error = None
    try:
        raw = yf.download(
            symbol, period="60d", interval=interval, auto_adjust=False,
            progress=False, threads=False, prepost=False,
        )
        if raw is None or raw.empty:
            fetch_error = f"No {interval} data returned; using validated cache if available"
        else:
            df = raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not set(required).issubset(df.columns):
                fetch_error = "Missing OHLCV columns; using validated cache if available"
            else:
                df = _normalize_index_timezone(df)
                # A duplicate timestamp invalidates its entire source session instead of
                # silently deduplicating data and treating a questionable session as valid.
                duplicate_rows = df.index.duplicated(keep=False)
                duplicate_dates = set(df.index[duplicate_rows].date)
                df = df.sort_index()
                df = filter_valid_ohlcv(df[required]).between_time("09:15", "15:30")
                if duplicate_dates:
                    df = df[[stamp.date() not in duplicate_dates for stamp in df.index]]
                fresh = complete_sessions_only(df, interval)
                if fresh.empty:
                    fetch_error = (
                        f"No complete {interval} sessions in current fetch; "
                        "using validated cache if available"
                    )
    except Exception as exc:
        fetch_error = f"{type(exc).__name__}: {exc}; using validated cache if available"

    df = merge_session_history(cached, fresh, interval, retention_days=180)
    if df.empty:
        expected_count = 75 if interval == "5m" else 25
        reason = fetch_error or f"No sessions passed the complete {interval} candle-grid check ({expected_count} expected timestamps per session)"
        return pd.DataFrame(), reason

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index_label="Timestamp", date_format="%Y-%m-%dT%H:%M:%S%z")
    df.attrs["interval"] = interval
    df.attrs["min_bars"] = min_bars
    df.attrs["min_last_time"] = min_last_time
    return df, None


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x.attrs.update(df.attrs)
    session = pd.Index(x.index.date)
    typical = (x["High"] + x["Low"] + x["Close"]) / 3
    volume = x["Volume"].clip(lower=0)
    volume_cum = volume.groupby(session).cumsum()
    x["VWAP"] = (typical * volume).groupby(session).cumsum() / volume_cum.replace(0, float("nan"))

    # Match the live paper simulator: indicators restart at each market session.
    # Do not treat a partially formed EMA as a fully warmed indicator.
    x["EMA9"] = session_ewm(x["Close"], span=9, min_periods=9)
    x["EMA21"] = session_ewm(x["Close"], span=21, min_periods=21)
    prev_close = x["Close"].groupby(session).shift(1)
    true_range = pd.concat(
        [
            x["High"] - x["Low"],
            (x["High"] - prev_close).abs(),
            (x["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["ATR14"] = true_range.groupby(session).transform(
        lambda values: values.rolling(14, min_periods=7).mean()
    )
    delta = x["Close"].groupby(session).diff()
    gain = delta.clip(lower=0).groupby(session).transform(
        lambda values: values.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    loss = (-delta.clip(upper=0)).groupby(session).transform(
        lambda values: values.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    x["RSI14"] = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))
    x["PREV_HIGH"] = x["High"].groupby(session).shift(1)
    x["PREV_CLOSE"] = x["Close"].groupby(session).shift(1)
    x["PREV_EMA9"] = x["EMA9"].groupby(session).shift(1)
    x["OR_HIGH"] = float("nan")
    x["VOL_MED20"] = float("nan")
    for _, day in x.groupby(session, sort=True):
        opening = day.loc[[ts.time() < dt_time(9, 45) for ts in day.index], "High"]
        or_high = float(opening.max()) if not opening.empty else float("nan")
        x.loc[day.index, "OR_HIGH"] = or_high
        x.loc[day.index, "VOL_MED20"] = day["Volume"].shift(1).rolling(20, min_periods=10).median()
    times = pd.Series([ts.time() for ts in x.index], index=x.index)
    active = (times >= dt_time(9, 45)) & (times <= dt_time(14, 45))
    x["SIG_VWAP_EMA"] = (
        active & (x["EMA9"] > x["EMA21"]) & (x["Close"] > x["VWAP"])
        & (x["Close"] > x["PREV_HIGH"]) & (x["PREV_CLOSE"] <= x["PREV_HIGH"])
    )
    x["SIG_ORB"] = (
        active & (times <= dt_time(14, 30)) & (x["EMA9"] > x["EMA21"])
        & (x["Close"] > x["OR_HIGH"]) & (x["PREV_CLOSE"] <= x["OR_HIGH"])
        & (x["Volume"] >= 1.2 * x["VOL_MED20"])
    )
    x["SIG_MEANREV"] = (
        active & (x["Close"] < x["VWAP"] - 0.5 * x["ATR14"])
        & (x["RSI14"] < 40) & (x["PREV_CLOSE"] <= x["PREV_EMA9"])
        & (x["Close"] > x["EMA9"])
    )
    return x


def candidates_for(symbol: str, x: pd.DataFrame, variant: dict) -> list[dict]:
    """Generate simulated candidate trades for one fixed parameter variant."""
    out = []
    fast_span, slow_span = int(variant["ema_fast"]), int(variant["ema_slow"])
    fast = session_ewm(x["Close"], span=fast_span, min_periods=fast_span)
    slow = session_ewm(x["Close"], span=slow_span, min_periods=slow_span)
    prev_fast = fast.groupby(fast.index.date).shift(1)
    times = pd.Series([ts.time() for ts in x.index], index=x.index)
    active = (times >= dt_time(9, 45)) & (times <= dt_time(14, 45))
    strategy = variant["strategy"]
    if strategy == "VWAP_EMA":
        signal = (active & (fast > slow) & (x["Close"] > x["VWAP"]) &
                  (x["Close"] > x["PREV_HIGH"]) & (x["PREV_CLOSE"] <= x["PREV_HIGH"]))
    elif strategy == "OPENING_RANGE_BREAKOUT":
        signal = (active & (times <= dt_time(14, 30)) & (fast > slow) &
                  (x["Close"] > x["OR_HIGH"]) & (x["PREV_CLOSE"] <= x["OR_HIGH"]) &
                  (x["Volume"] >= float(variant["volume_multiplier"]) * x["VOL_MED20"]))
    elif strategy == "VWAP_MEAN_REVERSION":
        signal = (active & (x["Close"] < x["VWAP"] - float(variant["deviation_atr"]) * x["ATR14"]) &
                  (x["RSI14"] < float(variant["rsi_max"])) &
                  (x["PREV_CLOSE"] <= prev_fast) & (x["Close"] > fast))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    min_bars = int(x.attrs.get("min_bars", 60))
    min_last_time = x.attrs.get("min_last_time", dt_time(15, 25))
    signal = signal.fillna(False).astype(bool)
    for session_date, day in x.groupby(x.index.date, sort=True):
        if len(day) < min_bars or day.index[-1].time() < min_last_time:
            continue
        flags = signal.loc[day.index].to_numpy()
        for i, triggered in enumerate(flags[:-1]):
            if not triggered:
                continue
            signal_time, entry_time = day.index[i], day.index[i + 1]
            if entry_time.time() > dt_time(14, 50):
                continue
            atr = float(day["ATR14"].iloc[i])
            entry = float(day["Open"].iloc[i + 1])
            if not (math.isfinite(atr) and atr > 0 and math.isfinite(entry) and entry > 0):
                continue
            risk = atr * float(variant["stop_atr"])
            stop = entry - risk
            target = entry + float(variant["target_r"]) * risk
            if strategy == "VWAP_MEAN_REVERSION":
                vwap = float(day["VWAP"].iloc[i])
                if not math.isfinite(vwap):
                    continue
                target = min(vwap, target)
            if stop <= 0 or target <= entry:
                continue
            exit_price, exit_time, reason = None, None, None
            for j in range(i + 1, len(day)):
                bar, ts = day.iloc[j], day.index[j]
                op, low, high = float(bar["Open"]), float(bar["Low"]), float(bar["High"])
                # Conservative fill assumption if stop and target touch in one candle.
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
            out.append({
                "date": str(session_date), "strategy": strategy, "variant_id": variant["id"],
                "parameters": variant, "ticker": symbol, "signal_time": signal_time,
                "entry_time": entry_time, "exit_time": exit_time, "entry": entry,
                "exit": exit_price, "stop": stop, "target": target, "reason": reason,
            })
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


def net_pnl_at_cost_rate(trades: list[dict], rate: float) -> float:
    """Reprice a fixed set of simulated fills at an alternative proportional cost rate."""
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("Cost rate must be a finite, nonnegative fraction")
    total = 0.0
    for trade in trades:
        entry, exit_price, quantity = (
            float(trade["entry"]),
            float(trade["exit"]),
            int(trade["qty"]),
        )
        gross = (exit_price - entry) * quantity
        estimated_costs = (entry * quantity + exit_price * quantity) * rate / 2
        total += gross - estimated_costs
    return total


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


def load_universe(interval: str):
    data, failures = {}, []
    for symbol in SYMBOLS:
        df, err = get_bars(symbol, interval)
        if err:
            failures.append((symbol, err))
        else:
            x = indicators(df)
            data[symbol] = x
    dates = sorted({ts.date() for x in data.values() for ts in x.index})
    return data, failures, dates


def common_session_dates(data: dict[str, pd.DataFrame]) -> list:
    """Return dates shared by every usable symbol; do not confuse this with a union."""
    date_sets = [set(frame.index.date) for frame in data.values()]
    return sorted(set.intersection(*date_sets)) if date_sets else []


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


def build_variants() -> list[dict]:
    """A bounded parameter grid fixed before evaluation; never expand based on holdout results."""
    variants = []
    ema_grid = [
        (5, 13, 1.0, 1.5), (5, 13, 1.0, 2.0),
        (9, 21, 1.0, 1.5), (9, 21, 1.0, 2.0),
        (9, 21, 1.5, 2.0), (12, 26, 1.0, 2.0),
        (12, 26, 1.5, 2.0), (8, 20, 1.25, 2.0),
    ]
    for fast, slow, stop_atr, target_r in ema_grid:
        variants.append({
            "id": f"vwapema_f{fast}_s{slow}_stop{stop_atr:g}_r{target_r:g}",
            "strategy": "VWAP_EMA", "ema_fast": fast, "ema_slow": slow,
            "stop_atr": stop_atr, "target_r": target_r,
            "volume_multiplier": 0.0, "deviation_atr": 0.0, "rsi_max": 0.0,
        })
    orb_grid = [
        (5, 13, 1.0, 1.5, 1.0), (5, 13, 1.0, 2.0, 1.2),
        (9, 21, 1.0, 1.5, 1.0), (9, 21, 1.0, 2.0, 1.2),
        (9, 21, 1.5, 2.0, 1.5), (12, 26, 1.0, 2.0, 1.2),
    ]
    for fast, slow, stop_atr, target_r, vol_mult in orb_grid:
        variants.append({
            "id": f"orb_f{fast}_s{slow}_stop{stop_atr:g}_r{target_r:g}_vol{vol_mult:g}",
            "strategy": "OPENING_RANGE_BREAKOUT", "ema_fast": fast, "ema_slow": slow,
            "stop_atr": stop_atr, "target_r": target_r,
            "volume_multiplier": vol_mult, "deviation_atr": 0.0, "rsi_max": 0.0,
        })
    mr_grid = [
        (0.5, 30, 1.0, 1.5), (0.5, 35, 1.0, 1.5),
        (0.5, 40, 1.0, 2.0), (0.75, 35, 1.0, 1.5),
        (0.75, 40, 1.0, 2.0), (1.0, 35, 1.5, 1.5),
    ]
    for deviation, rsi_max, stop_atr, target_r in mr_grid:
        variants.append({
            "id": f"meanrev_dev{deviation:g}_rsi{rsi_max}_stop{stop_atr:g}_r{target_r:g}",
            "strategy": "VWAP_MEAN_REVERSION", "ema_fast": 9, "ema_slow": 21,
            "stop_atr": stop_atr, "target_r": target_r,
            "volume_multiplier": 0.0, "deviation_atr": deviation, "rsi_max": rsi_max,
        })
    return variants


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def main():
    now = datetime.now(TZ)
    REPORTS.mkdir(parents=True, exist_ok=True)
    history_path = ROOT / "state" / "experiment_history.json"
    variants = build_variants()
    report = [
        f"# Automated Strategy Experiment — {now:%Y-%m-%d}", "",
        f"**Run time (IST):** {now.isoformat()}",
        "**Mode:** offline historical research and paper-candidate screening only; no brokerage orders.",
        f"**Experiment version:** grid-v2 · **Predeclared variants:** {len(variants)}", "",
        "## Automated experiment loop",
        "- Fetch free intraday OHLCV and validate complete sessions.",
        "- Run a fixed parameter grid across three strategy families.",
        "- Rank variants using the middle chronological validation slice only.",
        "- Test only the validation winner on the untouched final holdout slice.",
        "- Log results persistently; re-runs on unchanged data update an audit record rather than counting as new evidence.", "",
        "## Methods and controls",
        "- Prefer 5-minute data; if fewer than 15 complete dates are found, retry the whole universe at 15-minute resolution. Never mix intervals.",
        "- Use common complete session dates across symbols for comparable splits.",
        "- Next-candle-open entry; one open position at a time per variant; max 2 trades/day; ₹250 planned risk/trade; ₹25,000 maximum notional; ₹500 daily-loss limit.",
        "- If stop and target are both touched in one candle, assume the stop happened first; exit by 15:15 IST.",
        "- Estimated 0.10% round-trip costs; this is not an exact Groww fee quote.", "",
        "## Data coverage",
    ]

    data, failures, _ = load_universe("5m")
    used_interval = "5m"
    primary_common_dates = common_session_dates(data)
    if len(primary_common_dates) < 15:
        fallback_data, fallback_failures, _ = load_universe("15m")
        fallback_common_dates = common_session_dates(fallback_data)
        if len(fallback_common_dates) > len(primary_common_dates):
            data, failures, used_interval = fallback_data, fallback_failures, "15m"
            report.append(
                f"- Fallback used: 5-minute data had {len(primary_common_dates)} common complete sessions; "
                f"15-minute data had {len(fallback_common_dates)}."
            )
        else:
            report.append(
                f"- 15-minute fallback not selected: 5-minute data had {len(primary_common_dates)} "
                f"common complete sessions versus {len(fallback_common_dates)} at 15 minutes."
            )
    report.append(f"- Selected interval: {used_interval}")

    common_dates = common_session_dates(data)
    latest_common_date = max(common_dates) if common_dates else None
    data_age_days = (now.date() - latest_common_date).days if latest_common_date else None
    data_freshness_ok = data_age_days is not None and 0 <= data_age_days <= 5
    if latest_common_date is not None:
        report.append(
            f"- Latest common complete session: {latest_common_date} "
            f"({data_age_days} calendar day(s) before this run)."
        )
        if not data_freshness_ok:
            report.append(
                "- **Stale-data warning:** candidate qualification is blocked unless the "
                "latest common complete session is between 0 and 5 calendar days old."
            )
    if len(common_dates) >= 10:
        all_dates = common_dates
        common_set = set(common_dates)
        data = {
            symbol: frame[[ts.date() in common_set for ts in frame.index]].copy()
            for symbol, frame in data.items()
        }
        for symbol, frame in data.items():
            frame.attrs["min_bars"] = 60 if used_interval == "5m" else 20
            frame.attrs["min_last_time"] = dt_time(15, 25) if used_interval == "5m" else dt_time(15, 15)
            report.append(f"- {symbol}: {len(frame)} bars across {len(set(frame.index.date))} common sessions.")
    else:
        all_dates = []
        report.append(f"- Only {len(common_dates)} common session dates; insufficient for a comparable multi-symbol split.")
    for symbol, error in failures:
        report.append(f"- {symbol}: not usable — {error}")

    history = {"schema_version": 1, "runs": []}
    if history_path.exists():
        try:
            old_history = json.loads(history_path.read_text())
            if isinstance(old_history, dict) and isinstance(old_history.get("runs"), list):
                history = old_history
        except Exception:
            report.append("- Previous experiment history could not be read; starting a new audit file.")

    selected = None
    validation_rankings = []
    training_metrics = None
    validation_metrics = None
    holdout_metrics = None
    status = "DATA_LIMITED"
    data_end = str(max(all_dates)) if all_dates else None

    if len(all_dates) < 15 or not data:
        report += [
            "", "## Decision",
            f"**DATA-LIMITED:** {len(all_dates)} common complete dates; at least 15 are required to run the exploratory grid.",
            "No variant is selected or promoted, and no result is fabricated.",
        ]
    else:
        n = len(all_dates)
        if n < 30:
            report.append(f"- **Exploratory sample warning:** only {n} common sessions. Results can rank experiments for research, but cannot qualify a candidate for paper tracking.")
        train_end = int(n * 0.60)
        val_end = int(n * 0.80)
        train_dates = set(all_dates[:train_end])
        validation_dates = set(all_dates[train_end:val_end])
        holdout_dates = set(all_dates[val_end:])
        report += [
            "", "## Chronological split",
            f"- Train: {min(train_dates)} to {max(train_dates)} ({len(train_dates)} dates)",
            f"- Validation / selection: {min(validation_dates)} to {max(validation_dates)} ({len(validation_dates)} dates)",
            f"- Untouched final holdout: {min(holdout_dates)} to {max(holdout_dates)} ({len(holdout_dates)} dates)",
            "", "## Parameter experiments — ranked by validation only",
            "| Rank | Variant | Validation trades | Net P/L | PF | Max drawdown |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        experiments = []
        for variant in variants:
            candidates = [
                item for symbol, frame in data.items()
                for item in candidates_for(symbol, frame, variant)
            ]
            train_trades = simulate(candidates, train_dates)
            val_trades = simulate(candidates, validation_dates)
            train_stats, val_stats = stats(train_trades), stats(val_trades)
            experiments.append({
                "variant": variant,
                "train": train_stats,
                "validation": val_stats,
                "_candidates": candidates,
            })
        ranked = sorted(
            experiments,
            key=lambda item: (
                item["validation"]["n"] >= 8,
                item["validation"]["n"] > 0,
                item["validation"]["net"] > 0,
                item["validation"]["pf"] is not None and item["validation"]["pf"] >= 1.0,
                item["validation"]["net"] - 1000.0 * item["validation"]["dd"],
                item["validation"]["n"],
            ),
            reverse=True,
        )
        for rank, item in enumerate(ranked, start=1):
            m = item["validation"]
            validation_rankings.append({
                "rank": rank, "variant": item["variant"],
                "train": item["train"], "validation": m,
            })
            report.append(
                f"| {rank} | {item['variant']['id']} | {m['n']} | ₹{m['net']:,.2f} | "
                f"{fnum(m['pf'])} | {m['dd']:.3f}% |"
            )

        # Selection uses validation only; holdout P/L cannot determine the winner.
        min_validation_trades = max(5, min(8, len(validation_dates)))
        winner = next(
            (item for item in ranked
             if item["validation"]["n"] >= min_validation_trades
             and item["validation"]["net"] > 0
             and item["validation"]["pf"] is not None
             and item["validation"]["pf"] >= 1.0
             and item["validation"]["dd"] <= 2.5),
            None,
        )
        # Re-price the same validation fills at higher friction assumptions. This is
        # descriptive only: it never changes rankings or consults holdout performance.
        top_validation_trades = simulate(ranked[0]["_candidates"], validation_dates)
        report += [
            "",
            "## Cost sensitivity — top validation-ranked variant",
            "Same validation fills, repriced under alternative proportional cost assumptions; ranking is unchanged.",
            "| Assumed round-trip cost | Validation net P/L |",
            "|---:|---:|",
        ]
        for rate in (0.001, 0.0015, 0.002):
            scenario_net = net_pnl_at_cost_rate(top_validation_trades, rate)
            report.append(f"| {rate * 100:.2f}% | ₹{scenario_net:,.2f} |")

        report += ["", "## Final holdout and candidate decision"]
        if winner is None:
            status = "NO_VALIDATION_CANDIDATE"
            report.append("No variant cleared the validation gate. The final holdout was not used to rank variants; the daily baseline remains unchanged.")
        else:
            selected = winner["variant"]
            training_metrics = winner["train"]
            validation_metrics = winner["validation"]
            holdout_trades = simulate(winner["_candidates"], holdout_dates)
            holdout_metrics = stats(holdout_trades)
            h = holdout_metrics
            report.append(f"- Variant selected on validation: {selected['id']}.")
            report.append(f"- Untouched holdout: {h['n']} trades; net ₹{h['net']:,.2f}; return {h['return']:.3f}%; win rate {fnum(h['winrate'], '%')}; PF {fnum(h['pf'])}; max drawdown {h['dd']:.3f}%.")
            report.append("- NIFTY 50 benchmark: " + nifty_benchmark(holdout_dates))
            if len(all_dates) >= 30 and data_freshness_ok and h["n"] >= 15 and h["net"] > 0 and h["pf"] is not None and h["pf"] >= 1.2 and h["dd"] <= 2.0:
                status = "SHADOW_PAPER_CANDIDATE"
                report.append("Holdout gate passed for shadow paper tracking only. It is not a production strategy; 30 genuine forward paper trades are still required.")
            else:
                status = "NOT_QUALIFIED"
                report.append("Holdout gate not passed. Keep the baseline unchanged and continue testing on fresh data.")

    report += [
        "", "## Interpretation",
        "- Data quality filter: invalid OHLC relationships, non-positive prices, negative volume, duplicate timestamps, and incomplete sessions are excluded before indicators are calculated.",
        "- The experiment is not considered trained or accurate merely because one variant ranks first; validation and untouched holdout gates must both pass, then forward paper trades must confirm performance.",
        "- This is automated bounded parameter search, not self-modifying AI or neural-network retraining.",
        "- Variant ranking is based on validation; only the selected validation winner is evaluated on the final holdout.",
        "- The workflow never places live orders and never changes the active paper strategy automatically.",
        "- Yahoo Finance is unofficial, candle fills are approximate, and the 0.10% cost model is not exact.",
        "- Only validated complete sessions are retained in a rolling 180-calendar-day cache; stale data blocks candidate qualification.",
        "- A daily ₹1,000–₹3,000 target is not forced; no-trade days are acceptable.",
    ]

    record = {
        "experiment_version": "grid-v2",
        "run_at_ist": now.isoformat(),
        "latest_data_date": data_end,
        "interval": used_interval,
        "common_session_dates": len(all_dates),
        "variant_count": len(variants),
        "status": status,
        "selected_variant": selected,
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "holdout_metrics": holdout_metrics,
        "validation_rankings": validation_rankings,
    }
    record_key = (record["experiment_version"], record["interval"], record["latest_data_date"])
    previous = next(
        (item for item in history["runs"]
         if (item.get("experiment_version"), item.get("interval"), item.get("latest_data_date")) == record_key),
        None,
    )
    if previous is None:
        history["runs"].append(record)
    else:
        repeat_count = int(previous.get("repeat_check_count", 0)) + 1
        previous.update(record)
        previous["repeat_check_count"] = repeat_count
    history["schema_version"] = 1
    history["updated_at_ist"] = now.isoformat()
    history["latest"] = record
    history["runs"] = history["runs"][-180:]
    save_json(history_path, history)

    candidate_path = ROOT / "state" / "shadow_paper_candidate.json"
    if status == "SHADOW_PAPER_CANDIDATE" and selected:
        save_json(candidate_path, {
            "status": status,
            "experiment_version": "grid-v2",
            "selected_at_ist": now.isoformat(),
            "latest_data_date": data_end,
            "variant": selected,
            "validation_metrics": validation_metrics,
            "holdout_metrics": holdout_metrics,
            "next_step": "Start a separate shadow portfolio and require 30 genuine forward paper trades before considering any strategy change. Never place live orders.",
        })
    elif candidate_path.exists():
        candidate_path.unlink()

    report_path = REPORTS / f"research-{now:%Y-%m-%d}.md"
    report_path.write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
