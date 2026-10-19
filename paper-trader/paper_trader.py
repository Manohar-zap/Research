from __future__ import annotations
import json, math
from datetime import datetime, date, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

TZ = ZoneInfo('Asia/Kolkata')
ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'state' / 'ledger.json'
REPORTS = ROOT / 'reports'
START_CAPITAL = 100_000.0
RISK_PER_TRADE = 250.0
DAILY_LOSS_LIMIT = 500.0
MAX_TRADES = 1
MAX_NOTIONAL = 25_000.0
COST_RATE = 0.001  # rough 0.10% round-trip estimate; not broker-specific
SYMBOLS = ['RELIANCE.NS','HDFCBANK.NS','ICICIBANK.NS','INFY.NS','TCS.NS','SBIN.NS','LT.NS','ITC.NS']
STRATEGY = 'VWAP-EMA Momentum v1'

# Official NSE 2026 equities-segment holidays; update this list for each new year.
NSE_HOLIDAYS = {
    '2026-01-15': 'Municipal Corporation Election - Maharashtra',
    '2026-01-26': 'Republic Day',
    '2026-03-03': 'Holi',
    '2026-03-26': 'Shri Ram Navami',
    '2026-03-31': 'Shri Mahavir Jayanti',
    '2026-04-03': 'Good Friday',
    '2026-04-14': 'Dr. Baba Saheb Ambedkar Jayanti',
    '2026-05-01': 'Maharashtra Day',
    '2026-05-28': 'Bakri Id',
    '2026-06-26': 'Muharram',
    '2026-09-14': 'Ganesh Chaturthi',
    '2026-10-02': 'Mahatma Gandhi Jayanti',
    '2026-10-20': 'Dussehra',
    '2026-11-10': 'Diwali-Balipratipada',
    '2026-11-24': 'Prakash Gurpurb Sri Guru Nanak Dev',
    '2026-12-25': 'Christmas',
}
NSE_HOLIDAY_SOURCE = 'https://www.nseindia.com/resources/exchange-communication-holidays'


def load_ledger():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {'starting_equity': START_CAPITAL, 'equity': START_CAPITAL, 'trades': [], 'sessions': [], 'created_at': datetime.now(TZ).isoformat()}


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def download_today(symbol, today):
    try:
        df = yf.download(symbol, period='5d', interval='5m', auto_adjust=False, progress=False, threads=False)
        if df is None or df.empty:
            return None, 'No candles returned'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        idx = pd.to_datetime(df.index)
        if idx.tz is None:
            # Yahoo timestamps for NSE symbols should be interpreted as exchange-local
            # when timezone metadata is absent; do not shift them from UTC.
            idx = idx.tz_localize(TZ)
        else:
            idx = idx.tz_convert(TZ)
        df.index = idx
        df = df[~df.index.duplicated(keep='last')].sort_index()
        df = df[df.index.date == today]
        df = df.between_time('09:15', '15:30')
        required = {'Open','High','Low','Close','Volume'}
        if df.empty or not required.issubset(df.columns):
            return None, 'No verifiable current-session 5-minute candles'
        df = df[list(required)].apply(pd.to_numeric, errors='coerce').dropna()
        # Reject malformed OHLCV rows before computing indicators or simulated fills.
        df = df[
            (df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)
            & (df['Volume'] >= 0)
            & (df['High'] >= df[['Open', 'Close', 'Low']].max(axis=1))
            & (df['Low'] <= df[['Open', 'Close', 'High']].min(axis=1))
        ]
        if df.empty:
            return None, 'No valid OHLCV rows after price/volume consistency checks'
        # A completed NSE session should normally provide around 75 five-minute
        # candles. Refuse partial/stale feeds rather than make a decision from fragments.
        if len(df) < 60:
            return None, f'Only {len(df)} current-session candles; expected at least 60'
        if df.index[-1].time() < dt_time(15, 25):
            return None, f'Latest candle is too early ({df.index[-1].strftime("%H:%M")} IST); session data appears incomplete'
        return df.astype(float), None
    except Exception as e:
        return None, f'Data retrieval error: {type(e).__name__}: {e}'


def indicators(df):
    x = df.copy()
    typical = (x['High'] + x['Low'] + x['Close']) / 3
    vol = x['Volume'].clip(lower=0)
    x['VWAP'] = (typical * vol).cumsum() / vol.cumsum().replace(0, float('nan'))
    x['EMA9'] = x['Close'].ewm(span=9, adjust=False).mean()
    x['EMA21'] = x['Close'].ewm(span=21, adjust=False).mean()
    prev_close = x['Close'].shift(1)
    tr = pd.concat([x['High'] - x['Low'], (x['High'] - prev_close).abs(), (x['Low'] - prev_close).abs()], axis=1).max(axis=1)
    x['ATR14'] = tr.rolling(14, min_periods=7).mean()
    x['PREV_HIGH'] = x['High'].shift(1)
    x['SIGNAL'] = (x['EMA9'] > x['EMA21']) & (x['Close'] > x['VWAP']) & (x['Close'] > x['PREV_HIGH']) & (x['Close'].shift(1) <= x['PREV_HIGH'])
    return x


def simulate_symbol(symbol, df, equity, already_traded, session_pnl):
    x = indicators(df)
    trades = []
    if x['ATR14'].isna().all():
        return trades
    # Avoid first/last minutes; simulated entries occur at the next candle open.
    positions = [i for i in range(len(x)-1) if x.index[i].strftime('%H:%M') >= '09:45' and x.index[i].strftime('%H:%M') <= '14:45' and bool(x['SIGNAL'].iloc[i])]
    for i in positions:
        if len(trades) >= MAX_TRADES or session_pnl <= -DAILY_LOSS_LIMIT:
            break
        if already_traded:
            break
        entry_i = i + 1
        entry_time = x.index[entry_i]
        if entry_time.strftime('%H:%M') > '14:50':
            continue
        entry = float(x['Open'].iloc[entry_i])
        atr = float(x['ATR14'].iloc[i])
        if not math.isfinite(atr) or atr <= 0 or not math.isfinite(entry) or entry <= 0:
            continue
        stop = entry - atr
        target = entry + 2 * atr
        qty_by_risk = int(RISK_PER_TRADE / (entry - stop))
        qty_by_notional = int(min(MAX_NOTIONAL, equity) / entry)
        qty = min(qty_by_risk, qty_by_notional)
        if qty < 1:
            continue
        exit_price = None; exit_time = None; reason = None
        for j in range(entry_i, len(x)):
            bar = x.iloc[j]
            # Conservative stop-first assumption if stop and target hit within one candle.
            if float(bar['Low']) <= stop:
                exit_price = min(float(bar['Open']), stop) if float(bar['Open']) < stop else stop
                exit_time = x.index[j]; reason = 'Stop-loss'; break
            if float(bar['High']) >= target:
                exit_price = target; exit_time = x.index[j]; reason = 'Target'; break
            if x.index[j].strftime('%H:%M') >= '15:15':
                exit_price = float(bar['Close']); exit_time = x.index[j]; reason = 'End-of-day exit'; break
        if exit_price is None:
            exit_price = float(x['Close'].iloc[-1]); exit_time = x.index[-1]; reason = 'Last available candle exit'
        gross = (exit_price - entry) * qty
        costs = (entry * qty + exit_price * qty) * COST_RATE / 2
        net = gross - costs
        trade = {'date': str(x.index[entry_i].date()), 'strategy': STRATEGY, 'ticker': symbol, 'entry_time_ist': entry_time.isoformat(), 'entry_price': round(entry, 4), 'exit_time_ist': exit_time.isoformat(), 'exit_price': round(exit_price, 4), 'quantity': qty, 'capital_deployed': round(entry*qty, 2), 'stop_loss': round(stop, 4), 'target': round(target, 4), 'exit_reason': reason, 'gross_pnl': round(gross, 2), 'estimated_costs': round(costs, 2), 'net_pnl': round(net, 2), 'data_source': 'Yahoo Finance via yfinance; unofficial; 5-minute OHLCV', 'execution_note': 'Simulated from candle OHLC; intrabar ordering is unknowable; stop-first assumption.'}
        trades.append(trade)
        session_pnl += net
        already_traded = True  # one position at a time; this conservative v1 allows one trade across scan
        break
    return trades


def main():
    now = datetime.now(TZ); today = now.date(); today_s = str(today)
    ledger = load_ledger()
    REPORTS.mkdir(parents=True, exist_ok=True)
    if today.weekday() >= 5:
        print('Weekend: no regular NSE cash-equity session; no trades recorded.')
        return
    if today_s in NSE_HOLIDAYS:
        # Produce a dated no-session report, but do not add a fake trading session to the ledger.
        holiday_lines = [
            f'# NSE Paper Trading Report — {today_s}',
            '',
            f'**Generated (IST):** {now.isoformat()}',
            '**Status:** MARKET_HOLIDAY — no regular cash-equity session',
            f'**Reason:** {NSE_HOLIDAYS[today_s]}',
            '**Mode:** Simulation only; no brokerage orders placed.',
            '',
            '## Portfolio (unchanged)',
            f"- Virtual equity: ₹{float(ledger['equity']):,.2f}",
            f"- Cumulative realized net P/L: ₹{sum(t['net_pnl'] for t in ledger['trades']):,.2f}",
            f"- Recorded sessions: {len(ledger['sessions'])}",
            f"- Recorded trades: {len(ledger['trades'])}",
            '',
            'No trade was simulated and no session was added to the performance ledger.',
            f'Holiday calendar source: {NSE_HOLIDAY_SOURCE}',
            'This embedded holiday list covers the 2026 NSE equities calendar and must be updated when NSE publishes a new calendar.',
        ]
        holiday_path = REPORTS / f'{today_s}.md'
        holiday_path.write_text('\n'.join(holiday_lines) + '\n')
        print('\n'.join(holiday_lines))
        return
    # Do not let a manual diagnostic run before market close consume the date in the
    # ledger. NSE cash trading normally ends at 15:30 IST; allow a small data-finalization buffer.
    if now.time().replace(tzinfo=None) < dt_time(15, 40):
        print(f'Before the post-close window ({now.isoformat()}); no report or ledger entry written.')
        return
    existing_idx = next(
        (i for i, s in enumerate(ledger['sessions']) if s.get('date') == today_s),
        None,
    )
    if existing_idx is not None and ledger['sessions'][existing_idx].get('status') != 'DATA_LIMITED':
        print(f'Already finalized {today_s}; refusing a duplicate ledger entry.')
        return
    if existing_idx is not None:
        print(f'Retrying prior data-limited run for {today_s}; the existing session will be replaced, not duplicated.')
    # The scheduled workflow is after close. Do not run strategy for a date with stale/missing data.
    all_trades = []; errors = []; data_count = 0
    session_pnl = 0.0; already_traded = False
    for symbol in SYMBOLS:
        df, err = download_today(symbol, today)
        if err:
            errors.append(f'{symbol}: {err}'); continue
        data_count += 1
        trades = simulate_symbol(symbol, df, ledger['equity'], already_traded, session_pnl)
        if trades:
            all_trades.extend(trades)
            session_pnl += sum(t['net_pnl'] for t in trades)
            already_traded = True
        if len(all_trades) >= MAX_TRADES or session_pnl <= -DAILY_LOSS_LIMIT:
            break
    ledger['trades'].extend(all_trades)
    gross = round(sum(t['gross_pnl'] for t in all_trades), 2)
    costs = round(sum(t['estimated_costs'] for t in all_trades), 2)
    net = round(sum(t['net_pnl'] for t in all_trades), 2)
    ledger['equity'] = round(float(ledger['equity']) + net, 2)
    session = {'date': today_s, 'generated_at_ist': now.isoformat(), 'trades': len(all_trades), 'gross_pnl': gross, 'estimated_costs': costs, 'net_pnl': net, 'closing_equity': ledger['equity'], 'daily_return_pct': round(net / max(ledger['equity'] - net, 1) * 100, 4), 'data_symbols_validated': data_count, 'data_errors': errors, 'status': 'TRADED' if all_trades else ('NO_VALID_SIGNAL' if data_count == len(SYMBOLS) else 'DATA_LIMITED')}
    if existing_idx is None:
        ledger['sessions'].append(session)
    else:
        ledger['sessions'][existing_idx] = session
    wins = [t['net_pnl'] for t in ledger['trades'] if t['net_pnl'] > 0]
    losses = [t['net_pnl'] for t in ledger['trades'] if t['net_pnl'] < 0]
    total_wins = sum(wins); total_losses = abs(sum(losses))
    peak = START_CAPITAL; max_dd = 0.0
    for s in ledger['sessions']:
        peak = max(peak, float(s['closing_equity']))
        max_dd = max(max_dd, (peak - float(s['closing_equity'])) / peak * 100 if peak else 0)
    # NIFTY benchmark close-to-close only; explicitly separate from intraday strategy P/L.
    nifty_note = 'Unavailable'
    try:
        nd = yf.Ticker('^NSEI').history(period='5d', interval='1d')
        if not nd.empty:
            nifty_note = f"Latest returned daily close: {float(nd['Close'].iloc[-1]):.2f}; candle date: {nd.index[-1]} (verify session/date before comparison)"
    except Exception as e:
        nifty_note = f'Unavailable ({type(e).__name__})'
    lines = [f'# NSE Paper Trading Report — {today_s}', '', f"**Generated (IST):** {now.isoformat()}", f"**Status:** {session['status']}", '**Mode:** Simulation only; no brokerage orders placed.', '', '## Daily portfolio', f"- Starting virtual equity: ₹{START_CAPITAL:,.2f}", f"- Capital deployed (sum of trade notionals): ₹{sum(t['capital_deployed'] for t in all_trades):,.2f}", f"- Trades: {len(all_trades)}", f"- Gross P/L: ₹{gross:,.2f}", f"- Estimated costs: ₹{costs:,.2f}", f"- **Net P/L: ₹{net:,.2f}**", f"- Closing virtual equity: ₹{ledger['equity']:,.2f}", f"- Daily return: {session['daily_return_pct']:.4f}%", f"- Validated symbols with current-date candles: {data_count}/{len(SYMBOLS)}", '', '## Trades']
    if all_trades:
        for t in all_trades:
            lines += ['', f"### {t['ticker']} — net ₹{t['net_pnl']:,.2f}", f"- Strategy: {t['strategy']}", f"- Entry: {t['entry_time_ist']} at ₹{t['entry_price']:.2f}", f"- Exit: {t['exit_time_ist']} at ₹{t['exit_price']:.2f} ({t['exit_reason']})", f"- Quantity / notional: {t['quantity']} / ₹{t['capital_deployed']:,.2f}", f"- Stop / target: ₹{t['stop_loss']:.2f} / ₹{t['target']:.2f}", f"- Gross P/L / estimated costs / net P/L: ₹{t['gross_pnl']:.2f} / ₹{t['estimated_costs']:.2f} / ₹{t['net_pnl']:.2f}", f"- Data: {t['data_source']}", f"- Fill assumptions: {t['execution_note']}"]
    else:
        lines += ['', 'No trade recorded. This is not a fabricated zero-price trade; no validated strategy entry was executed in the simulation.']
    lines += ['', '## Cumulative statistics', f"- Recorded sessions: {len(ledger['sessions'])}", f"- Recorded trades: {len(ledger['trades'])}", f"- Cumulative realized net P/L: ₹{sum(t['net_pnl'] for t in ledger['trades']):,.2f}", f"- Win rate: {len(wins)/len(ledger['trades'])*100:.1f}%" if ledger['trades'] else '- Win rate: N/A (no trades)', f"- Average win: ₹{sum(wins)/len(wins):,.2f}" if wins else '- Average win: N/A', f"- Average loss: ₹{sum(losses)/len(losses):,.2f}" if losses else '- Average loss: N/A', f"- Profit factor: {total_wins/total_losses:.2f}" if total_losses else '- Profit factor: N/A (no realized losses)', f'- Maximum observed drawdown: {max_dd:.2f}%', f'- NIFTY 50 reference: {nifty_note}', '', '## Data limitations and assumptions', '- Source: Yahoo Finance via yfinance; unofficial, may be delayed/incomplete and is not an NSE-certified feed.', '- Costs: 0.10% of round-trip notional estimated; not an exact brokerage/tax calculation.', '- OHLC candle fills are approximate. Stop is assumed first if stop and target both occur in one candle.', '- A no-trade or data-limited day is valid. Daily profit targets are not forced.', '- This strategy has not been proven profitable; results are paper simulations, not investment advice.']
    # Weekly/monthly stats by dates in ledger; the current run does not backfill missing history.
    day_date = date.fromisoformat(today_s)
    week = [s for s in ledger['sessions'] if date.fromisoformat(s['date']).isocalendar()[:2] == day_date.isocalendar()[:2]]
    month = [s for s in ledger['sessions'] if s['date'][:7] == today_s[:7]]
    def summary(title, rows):
        pnl = sum(float(s['net_pnl']) for s in rows)
        return [f'## {title}', f'- Recorded sessions: {len(rows)}', f'- Net P/L: ₹{pnl:,.2f}', f"- Latest recorded equity: ₹{rows[-1]['closing_equity']:,.2f}" if rows else '- Equity: no records']
    if today.weekday() == 4:
        lines += ['', *summary('Weekly summary (recorded sessions only)', week)]
    import calendar
    month_last = date(today.year, today.month, calendar.monthrange(today.year, today.month)[1])
    final_weekday = month_last
    while final_weekday.weekday() >= 5:
        final_weekday -= timedelta(days=1)
    if today == final_weekday:
        lines += ['', *summary('Monthly summary (recorded sessions only; weekday estimate, exchange holidays may differ)', month)]
    (REPORTS / f'{today_s}.md').write_text('\n'.join(lines) + '\n')
    save_json(STATE, ledger)
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
