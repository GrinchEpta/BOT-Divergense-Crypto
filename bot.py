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
    mexc_spot_url = f"https://www.mexc.com/ru-RU/exchange/{base}_{quote}"
    mexc_futures_url = f"https://futures.mexc.com/ru-RU/exchange/{base}_{quote}"
    tv_url = f"https://www.tradingview.com/chart/?symbol=MEXC:{base}{quote}"

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message,
        'parse_mode': 'HTML',
        'disable_web_page_preview': True,
        'reply_markup': {
            'inline_keyboard': [
                [
                    {'text': '⚡ Фьючерсы MEXC', 'url': mexc_futures_url},
                    {'text': '📈 Спот MEXC', 'url': mexc_spot_url}
                ],
                [
                    {'text': '📊 Открыть в TradingView', 'url': tv_url}
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
                if MIN_PE
