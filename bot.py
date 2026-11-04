import os
import time
import json
import ccxt
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- НАСТРОЙКИ ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
STATE_FILE = "fast_signals_state.json"

TIMEFRAMES = ['15m', '30m', '1h']
RSI_PERIOD = 14
PIVOT_LEFT = 4       # Минимум 4 свечи слева ниже пика
PIVOT_RIGHT = 1      # 1 закрытая свеча справа для мгновенного входа
MIN_PEAK_DIST = 5    # Мин. расстояние между двумя вершинами
MAX_PEAK_DIST = 45   # Макс. расстояние между вершинами

# --- ИНИЦИАЛИЗАЦИЯ БИРЖИ (MEXC) ---
try:
    exchange = ccxt.mexc({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    exchange.load_markets()
except Exception as e:
    print(f"❌ Ошибка инициализации биржи: {e}")
    exit(1)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"⚠ Ошибка сохранения состояния: {e}")

last_signals = load_state()

def send_telegram_message(message, symbol):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠ Не указан TELEGRAM_TOKEN или CHAT_ID!")
        return

    base, quote = symbol.split('/')
    # Прямая ссылка на фьючерсы BingX (формат: BTC-USDT)
    bingx_futures_url = f"https://bingx.com/ru-ru/perpetual/{base}-{quote}"

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message,
        'parse_mode': 'HTML',
        'disable_web_page_preview': True,
        'reply_markup': {
            'inline_keyboard': [
                [
                    {'text': '⚡ Открыть на BingX', 'url': bingx_futures_url}
                ]
            ]
        }
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Ошибка отправки в Telegram: {e}")

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def get_top_20_crypto_symbols():
    try:
        tickers = exchange.fetch_tickers()
        crypto_list = []
        stablecoins = {'USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'FDUSD', 'USDD', 'USDE', 'PYUSD'}
        exclude_patterns = ('3L', '3S', '5L', '5S', 'BEAR', 'BULL', 'UP', 'DOWN', 'HEDGE')

        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT'):
                base_currency = symbol.split('/')[0]
                if base_currency in stablecoins or any(p in base_currency for p in exclude_patterns):
                    continue
                volume = ticker.get('quoteVolume') or 0
                if volume > 0:
                    crypto_list.append({'symbol': symbol, 'volume': volume})

        crypto_list.sort(key=lambda x: x['volume'], reverse=True)
        return [item['symbol'] for item in crypto_list[:20]]
    except Exception as e:
        print(f"❌ Ошибка загрузки топ-20: {e}")
        return []

def find_rsi_pivots(rsi_values, left=PIVOT_LEFT, right=PIVOT_RIGHT):
    peaks = []
    troughs = []
    n = len(rsi_values)

    for i in range(left, n - right):
        window_left = rsi_values[i - left : i]
        window_right = rsi_values[i + 1 : i + 1 + right]
        val = rsi_values[i]

        if val >= np.max(window_left) and val > np.max(window_right):
            peaks.append(i)

        if val <= np.min(window_left) and val < np.min(window_right):
            troughs.append(i)

    return peaks, troughs

def check_fast_divergence(df):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    rsis = df['rsi'].values
    timestamps = df['timestamp'].values
    n = len(df)

    peaks, troughs = find_rsi_pivots(rsis, left=PIVOT_LEFT, right=PIVOT_RIGHT)
    signals = []

    # 1. МЕДВЕЖЬЯ ДИВЕРГЕНЦИЯ (SHORT)
    if len(peaks) >= 2:
        p2 = peaks[-1]
        if (n - 1 - p2) in (1, 2):
            for p1 in reversed(peaks[:-1]):
                dist = p2 - p1
                if MIN_PEAK_DIST <= dist <= MAX_PEAK_DIST:
                    price_higher = (highs[p2] > highs[p1]) or (closes[p2] > closes[p1])
                    rsi_lower = rsis[p2] < (rsis[p1] - 1.5)
                    rsi_zone_ok = (rsis[p1] >= 58) and (rsis[p2] >= 52)
                    candle_confirmed = closes[-1] < closes[p2]

                    if price_higher and rsi_lower and rsi_zone_ok and candle_confirmed:
                        entry_price = closes[-1]
                        stop_loss = max(highs[p2], highs[-1]) * 1.003
                        support_level = np.min(lows[p1:p2+1])
                        
                        signals.append({
                            'type': '🐻 BEARISH (ШОРТ)',
                            'peak_ts': int(timestamps[p2]),
                            'entry': entry_price,
                            'sl': stop_loss,
                            'tp1': support_level,
                            'rsi_p1': rsis[p1],
                            'rsi_p2': rsis[p2],
                            'rsi_now': rsis[-1]
                        })
                        break

    # 2. БЫЧЬЯ ДИВЕРГЕНЦИЯ (LONG)
    if len(troughs) >= 2:
        t2 = troughs[-1]
        if (n - 1 - t2) in (1, 2):
            for t1 in reversed(troughs[:-1]):
                dist = t2 - t1
                if MIN_PEAK_DIST <= dist <= MAX_PEAK_DIST:
                    price_lower = (lows[t2] < lows[t1]) or (closes[t2] < closes[t1])
                    rsi_higher = rsis[t2] > (rsis[t1] + 1.5)
                    rsi_zone_ok = (rsis[t1] <= 42) and (rsis[t2] <= 48)
                    candle_confirmed = closes[-1] > closes[t2]

                    if price_lower and rsi_higher and rsi_zone_ok and candle_confirmed:
                        entry_price = closes[-1]
                        stop_loss = min(lows[t2], lows[-1]) * 0.997
                        resistance_level = np.max(highs[t1:t2+1])

                        signals.append({
                            'type': '🐂 BULLISH (ЛОНГ)',
                            'peak_ts': int(timestamps[t2]),
                            'entry': entry_price,
                            'sl': stop_loss,
                            'tp1': resistance_level,
                            'rsi_p1': rsis[t1],
                            'rsi_p2': rsis[t2],
                            'rsi_now': rsis[-1]
                        })
                        break

    return signals

def scan_single_pair(coin, tf):
    try:
        ohlcv = exchange.fetch_ohlcv(coin, timeframe=tf, limit=120)
        if len(ohlcv) < 60:
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['rsi'] = calculate_rsi(df['close'], period=RSI_PERIOD)

        df_closed = df.iloc[:-1].reset_index(drop=True)
        found_signals = check_fast_divergence(df_closed)

        return (coin, tf, found_signals)
    except Exception:
        return None

def run_scanner():
    top_coins = get_top_20_crypto_symbols()
    if not top_coins:
        print("⚠ Не удалось получить список монет.")
        return

    tasks = [(coin, tf) for tf in TIMEFRAMES for coin in top_coins]
    print(f"⚡ Быстрое сканирование {len(tasks)} графиков — {time.strftime('%Y-%m-%d %H:%M:%S')}")

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(scan_single_pair, coin, tf) for coin, tf in tasks]
        for future in as_completed(futures):
            result = future.result()
            if not result:
                continue
            coin, tf, signals = result
            for sig in signals:
                signal_key = f"{coin}_{tf}_{sig['peak_ts']}_{sig['type']}"
                if last_signals.get(f"{coin}_{tf}") != signal_key:
                    last_signals[f"{coin}_{tf}"] = signal_key
                    save_state(last_signals)

                    risk_pct = abs(sig['entry'] - sig['sl']) / sig['entry'] * 100
                    reward_pct = abs(sig['tp1'] - sig['entry']) / sig['entry'] * 100

                    msg = (
                        f"⚡ <b>СИГНАЛ ВХОДА: {sig['type']}</b>\n"
                        f"Монета: <b>#{coin.replace('/', '')}</b> | ТФ: <b>{tf}</b>\n"
                        f"────────────────────\n"
                        f"🎯 <b>Точка входа (сейчас):</b> <code>{sig['entry']:.6g}</code>\n"
                        f"🛑 <b>Стоп-лосс (за пик):</b> <code>{sig['sl']:.6g}</code> (-{risk_pct:.2f}%)\n"
                        f"💰 <b>Тейк 1 (уровень):</b> <code>{sig['tp1']:.6g}</code> (+{reward_pct:.2f}%)\n"
                        f"────────────────────\n"
                        f"📉 <b>Дивергенция RSI:</b> {sig['rsi_p1']:.1f} ➔ {sig['rsi_p2']:.1f} (сейчас: {sig['rsi_now']:.1f})\n"
                        f"⏱ <i>Сформировано только что (задержка 1 свеча)!</i>"
                    )
                    print(f"🔥 НАЙДЕН СИГНАЛ: {coin} ({tf}) -> {sig['type']}")
                    send_telegram_message(msg, coin)

if __name__ == "__main__":
    print("🚀 Запуск сканера дивергенций...")
    try:
        run_scanner()
        print("✅ Проверка успешно завершена!")
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")
