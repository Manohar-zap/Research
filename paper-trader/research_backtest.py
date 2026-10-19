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
            return pd.DataFrame(), f"No {interval} data returned"
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
    x["SIG_MEANREV"] = (active & (x["Close"] < x["VWAP"] - 0.5 * x["ATR14"])
                        & (x["RSI14"] < 40) & (x["PREV_CLOSE"] <= x["PREV_EMA9"])
                        & (x["Close"] > x["EMA9"]))
    return x


def candidates_for(symbol: str, x: pd.DataFrame, variant: dict) -> list[dict]:
    """Generate simulated candidate trades for one fixed parameter variant."""
    out = []
    fast = x["Close"].ewm(span=int(variant["ema_fast"]), adjust=False).mean()
    slow = x["Close"].ewm(span=int(variant["ema_slow"]), adjust=False).mean()
    prev_fast = fast.shift(1)
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

    data, failures, union_dates = load_universe("5m")
    used_interval = "5m"
    primary_count = len(union_dates)
    if primary_count < 15:
        fallback_data, fallback_failures, fallback_dates = load_universe("15m")
        if len(fallback_dates) > primary_count:
            data, failures, union_dates, used_interval = fallback_data, fallback_failures, fallback_dates, "15m"
            report.append(f"- Fallback used: 5-minute data had {primary_count} complete dates; 15-minute data had {len(union_dates)}.")
    report.append(f"- Selected interval: {used_interval}")

    date_sets = [set(frame.index.date) for frame in data.values()]
    common_dates = sorted(set.intersection(*date_sets)) if date_sets else []
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
            if len(all_dates) >= 30 and h["n"] >= 15 and h["net"] > 0 and h["pf"] is not None and h["pf"] >= 1.2 and h["dd"] <= 2.0:
                status = "SHADOW_PAPER_CANDIDATE"
                report.append("Holdout gate passed for shadow paper tracking only. It is not a production strategy; 30 genuine forward paper trades are still required.")
            else:
                status = "NOT_QUALIFIED"
                report.append("Holdout gate not passed. Keep the baseline unchanged and continue testing on fresh data.")

    report += [
        "", "## Interpretation",
        "- This is automated bounded parameter search, not self-modifying AI or neural-network retraining.",
        "- Variant ranking is based on validation; only the selected validation winner is evaluated on the final holdout.",
        "- The workflow never places live orders and never changes the active paper strategy automatically.",
        "- Yahoo Finance is unofficial, candle fills are approximate, and the 0.10% cost model is not exact.",
        "- A daily ₹1,000–₹3,000 target is not forced; no-trade days are acceptable.",
    ]

    record = {
        "experiment_version": "grid-v1",
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
            "experiment_version": "grid-v1",
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
