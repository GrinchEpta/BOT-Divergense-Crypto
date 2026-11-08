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

# Список монет со скриншота
TARGET_COINS = [
    'XMR/USDT', 'BCH/USDT', 'ADA/USDT', 'BNB/USDT',
    'LTC/USDT', 'HYPE/USDT', 'XRP/USDT', 'SOL/USDT',
    'ZEC/USDT', 'BTC/USDT', 'ETH/USDT'
]

TIMEFRAMES = ['15m', '30m', '1h']
RSI_PERIOD = 14
PIVOT_LEFT = 5       # Минимум 5 свечей слева до пика
PIVOT_RIGHT = 1      # 1 закрытая свеча справа (подтверждение разворота)
MIN_PEAK_DIST = 5    # Мин. расстояние между двумя вершинами (в свечах)
MAX_PEAK_DIST = 45   # Макс. расстояние между вершинами (в свечах)

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
    """Загружает историю уже отправленных сигналов."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}

def save_state(state):
    """Сохраняет историю сигналов и очищает слишком старые записи (>1000)."""
    try:
        # Оставляем последние 1000 записей, чтобы файл не разрастался бесконечно
        if len(state) > 1000:
            keys_to_keep = list(state.keys())[-1000:]
            state = {k: state[k] for k in keys_to_keep}
            
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠ Ошибка сохранения состояния: {e}")

def send_telegram_message(message, symbol):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠ Не указан TELEGRAM_TOKEN или CHAT_ID!")
        return False

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
                [{'text': '⚡ Открыть график на BingX', 'url': bingx_futures_url}]
            ]
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Ошибка отправки в Telegram: {e}")
        return False

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

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

    # 1. МЕДВЕЖЬЯ ДИВЕРГЕНЦИЯ (Цена обновляет максимум, а RSI падает)
    if len(peaks) >= 2:
        p2 = peaks[-1]
        # Проверяем пик, который сформировался на предпоследней закрытой свече (или свечой ранее)
        if (n - 1 - p2) in (1, 2):
            for p1 in reversed(peaks[:-1]):
                dist = p2 - p1
                if MIN_PEAK_DIST <= dist <= MAX_PEAK_DIST:
                    # Цена на втором пике выше первого, а RSI ниже минимум на 2 пункта
                    price_higher = (highs[p2] > highs[p1]) and (closes[p2] >= closes[p1] * 0.998)
                    rsi_lower = rsis[p2] <= (rsis[p1] - 2.0)
                    rsi_zone_ok = (rsis[p1] >= 60) and (rsis[p2] >= 52)
                    candle_confirmed = closes[-1] < closes[p2]

                    if price_higher and rsi_lower and rsi_zone_ok and candle_confirmed:
                        signals.append({
                            'id': f"BEAR_{int(timestamps[p2])}", # Уникальный ID по времени свечи пика
                            'type': '🐻 Медвежья дивергенция (Шорт)',
                            'price_p1': highs[p1],
                            'price_p2': highs[p2],
                            'price_now': closes[-1],
                            'rsi_p1': rsis[p1],
                            'rsi_p2': rsis[p2],
                            'rsi_now': rsis[-1]
                        })
                        break # Берем только одну самую актуальную дивергенцию для этой вершины

    # 2. БЫЧЬЯ ДИВЕРГЕНЦИЯ (Цена обновляет минимум, а RSI растет)
    if len(troughs) >= 2:
        t2 = troughs[-1]
        if (n - 1 - t2) in (1, 2):
            for t1 in reversed(troughs[:-1]):
                dist = t2 - t1
                if MIN_PEAK_DIST <= dist <= MAX_PEAK_DIST:
                    # Цена на втором дне ниже первого, а RSI выше минимум на 2 пункта
                    price_lower = (lows[t2] < lows[t1]) and (closes[t2] <= closes[t1] * 1.002)
                    rsi_higher = rsis[t2] >= (rsis[t1] + 2.0)
                    rsi_zone_ok = (rsis[t1] <= 40) and (rsis[t2] <= 48)
                    candle_confirmed = closes[-1] > closes[t2]

                    if price_lower and rsi_higher and rsi_zone_ok and candle_confirmed:
                        signals.append({
                            'id': f"BULL_{int(timestamps[t2])}", # Уникальный ID по времени свечи дна
                            'type': '🐂 Бычья дивергенция (Лонг)',
                            'price_p1': lows[t1],
                            'price_p2': lows[t2],
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

        # Отбрасываем последнюю незакрытую свечу, чтобы не было ложных сигналов
        df_closed = df.iloc[:-1].reset_index(drop=True)
        found_signals = check_fast_divergence(df_closed)

        return (coin, tf, found_signals)
    except Exception:
        return None

def run_scanner():
    sent_signals = load_state()
    state_modified = False

    available_coins = [pair for pair in TARGET_COINS if pair in markets]
    tasks = [(coin, tf) for tf in TIMEFRAMES for coin in available_coins]
    print(f"⚡ Сканирование {len(tasks)} графиков — {time.strftime('%Y-%m-%d %H:%M:%S')}")

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(scan_single_pair, coin, tf) for coin, tf in tasks]
        for future in as_completed(futures):
            result = future.result()
            if not result:
                continue
            coin, tf, signals = result
            for sig in signals:
                # Уникальный ключ: Монета + Таймфрейм + Тип + Точное время свечи пика
                unique_signal_key = f"{coin}_{tf}_{sig['id']}"

                # Если этот сигнал уже отправлялся ранее — жестко пропускаем его
                if unique_signal_key in sent_signals:
                    continue

                msg = (
                    f"🔔 <b>Сигнал дивергенции ({tf})</b>\n"
                    f"Монета: <b>#{coin.replace('/', '')}</b>\n"
                    f"Паттерн: <b>{sig['type']}</b>\n"
                    f"────────────────────\n"
                    f"📈 Пики цены: <code>{sig['price_p1']:.6g}</code> ➔ <code>{sig['price_p2']:.6g}</code>\n"
                    f"📊 Пики RSI: <b>{sig['rsi_p1']:.1f} ➔ {sig['rsi_p2']:.1f}</b>\n"
                    f"💵 Цена входа: <code>{sig['price_now']:.6g}</code> (RSI: {sig['rsi_now']:.1f})"
                )
                print(f"🔥 НОВЫЙ СИГНАЛ: {unique_signal_key}")
                
                if send_telegram_message(msg, coin):
                    sent_signals[unique_signal_key] = int(time.time())
                    state_modified = True

    if state_modified:
        save_state(sent_signals)

if __name__ == "__main__":
    print("🚀 Запуск сканера дивергенций...")
    run_scanner()
    print("✅ Проверка завершена!")
