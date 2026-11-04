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
PIVOT_RIGHT = 1      # 1 закрытая свеча справа для быстрого сигнала
MIN_PEAK_DIST = 5    # Мин. расстояние между двумя вершинами
MAX_PEAK_DIST = 45   # Макс. расстояние между вершинами

# --- ИНИЦИАЛИЗАЦИЯ БИРЖИ (MEXC) ---
try:
    exchange = ccxt.mexc({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    markets = exchange.load_markets()
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
                    {'text': '⚡ Открыть график на BingX', 'url': bingx_futures_url}
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

def get_top_25_by_market_cap():
    """
    Загружает Топ-25 монет по рыночной капитализации через CoinGecko
    и проверяет их доступность на бирже MEXC к USDT.
    """
    ignored_tokens = {
        'USDT', 'USDC', 'BUSD', 'DAI', 'FDUSD', 'TUSD', 'USDD', 'USDE', 'PYUSD', 'USDS',
        'WBTC', 'STETH', 'WSTETH', 'WEETH', 'WETH', 'CBETH', 'WBETH', 'LEO'
    }
    
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 60,
            'page': 1,
            'sparkline': 'false'
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        top_25 = []
        for coin in data:
            base = coin.get('symbol', '').upper()
            if base in ignored_tokens:
                continue
            
            pair = f"{base}/USDT"
            if pair in markets:
                top_25.append(pair)
                if len(top_25) == 25:
                    break

        if top_25:
            print(f"✅ Топ-25 по капитализации загружен: {', '.join(top_25)}")
            return top_25
    except Exception as e:
        print(f"⚠ Не удалось получить капитализацию с CoinGecko ({e}), берем резервный топ с MEXC...")

    # Резервный вариант, если API CoinGecko временно недоступен
    try:
        tickers = exchange.fetch_tickers()
        crypto_list = []
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT'):
                base = symbol.split('/')[0]
                if base in ignored_tokens:
                    continue
                vol = ticker.get('quoteVolume') or 0
                if vol > 0:
                    crypto_list.append({'symbol': symbol, 'volume': vol})
        crypto_list.sort(key=lambda x: x['volume'], reverse=True)
        return [item['symbol'] for item in crypto_list[:25]]
    except Exception as e:
        print(f"❌ Ошибка резервной загрузки: {e}")
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

    # 1. МЕДВЕЖЬЯ ДИВЕРГЕНЦИЯ (BEARISH)
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
                        signals.append({
                            'type': '🐻 Медвежья дивергенция (Bearish)',
                            'peak_ts': int(timestamps[p2]),
                            'price_now': closes[-1],
                            'rsi_p1': rsis[p1],
                            'rsi_p2': rsis[p2],
                            'rsi_now': rsis[-1]
                        })
                        break

    # 2. БЫЧЬЯ ДИВЕРГЕНЦИЯ (BULLISH)
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
                        signals.append({
                            'type': '🐂 Бычья дивергенция (Bullish)',
                            'peak_ts': int(timestamps[t2]),
                            'price_now': closes[-1],
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
    top_coins = get_top_25_by_market_cap()
    if not top_coins:
        print("⚠ Не удалось получить список монет.")
        return

    tasks = [(coin, tf) for tf in TIMEFRAMES for coin in top_coins]
    print(f"⚡ Сканирование {len(tasks)} графиков (Топ-25 по капитализации) — {time.strftime('%Y-%m-%d %H:%M:%S')}")

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

                    msg = (
                        f"🔔 <b>Обнаружен сигнал ({tf})</b>\n"
                        f"Монета: <b>#{coin.replace('/', '')}</b>\n"
                        f"Паттерн: <b>{sig['type']}</b>\n"
                        f"────────────────────\n"
                        f"💵 Текущая цена: <code>{sig['price_now']:.6g}</code>\n"
                        f"📊 RSI по пикам: <b>{sig['rsi_p1']:.1f} ➔ {sig['rsi_p2']:.1f}</b> (сейчас: {sig['rsi_now']:.1f})"
                    )
                    print(f"🔥 НАЙДЕН СИГНАЛ: {coin} ({tf}) -> {sig['type']}")
                    send_telegram_message(msg, coin)

if __name__ == "__main__":
    print("🚀 Запуск сканера дивергенций (Топ-25 Market Cap)...")
    try:
        run_scanner()
        print("✅ Проверка успешно завершена!")
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")
# === КОНЕЦ ФАЙЛА ===
