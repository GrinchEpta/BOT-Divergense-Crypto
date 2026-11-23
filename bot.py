import os
import time
import json
import ccxt
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')  # Для генерации картинок без GUI на сервере
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- НАСТРОЙКИ ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
STATE_FILE = "fast_signals_state.json"

TARGET_COINS = [
    'XMR/USDT', 'BCH/USDT', 'ADA/USDT', 'BNB/USDT',
    'LTC/USDT', 'HYPE/USDT', 'XRP/USDT', 'SOL/USDT',
    'ZEC/USDT', 'BTC/USDT', 'ETH/USDT'
]

TIMEFRAMES = ['30m', '1h']  # Только 30-минутный и часовой таймфреймы
RSI_PERIOD = 14
PIVOT_LEFT = 5       # Свечей слева до пика
PIVOT_RIGHT = 2      # 2 закрытые свечи справа для точности
MIN_PEAK_DIST = 5    # Мин. расстояние между экстремумами
MAX_PEAK_DIST = 45   # Макс. расстояние

try:
    exchange = ccxt.mexc({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    markets = exchange.load_markets()
except Exception as e:
    print(f"❌ Ошибка инициализации MEXC: {e}")
    exit(1)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}

def save_state(state):
    try:
        if len(state) > 1000:
            keys_to_keep = list(state.keys())[-1000:]
            state = {k: state[k] for k in keys_to_keep}
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠ Ошибка сохранения состояния: {e}")

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def find_pivots(values, left=PIVOT_LEFT, right=PIVOT_RIGHT):
    peaks, troughs = [], []
    n = len(values)
    for i in range(left, n - right):
        window_left = values[i - left : i]
        window_right = values[i + 1 : i + 1 + right]
        val = values[i]
        if val >= np.max(window_left) and val > np.max(window_right):
            peaks.append(i)
        if val <= np.min(window_left) and val < np.min(window_right):
            troughs.append(i)
    return peaks, troughs

def generate_divergence_chart(df, tf, coin, sig, filename="divergence.png"):
    """Создаёт 3-панельный скриншот: Цена, RSI, Объём с линиями дивергенции"""
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={'height_ratios': [2, 1, 1]}
    )

    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    dates = df['date']
    bar_width = (dates.iloc[1] - dates.iloc[0]) * 0.6

    # --- 1. ПАНЕЛЬ ЦЕНЫ ---
    ax1.plot(dates, df['close'], color='#5dade2', label='Цена закрытия', linewidth=1.5)
    ax1.fill_between(dates, df['low'], df['high'], color='#5dade2', alpha=0.1)

    # --- 2. ПАНЕЛЬ RSI ---
    ax2.plot(dates, df['rsi'], color='#af7ac5', label='RSI (14)', linewidth=1.5)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.axhline(50, color='gray', linestyle=':', alpha=0.3)
    ax2.set_ylim(10, 90)

    # --- 3. ПАНЕЛЬ ОБЪЕМОВ ---
    colors = ['#27ae60' if c >= o else '#e74c3c' for c, o in zip(df['close'], df['open'])]
    ax3.bar(dates, df['volume'], color=colors, alpha=0.8, width=bar_width)

    # Рисуем линии дивергенции по точкам
    p1_idx, p2_idx = sig['p1_idx'], sig['p2_idx']
    d1, d2 = dates.iloc[p1_idx], dates.iloc[p2_idx]

    if "🐻" in sig['type']:  # Медвежья (по пикам)
        ax1.plot([d1, d2], [df['high'].iloc[p1_idx], df['high'].iloc[p2_idx]], color='#f39c12', linewidth=2.5, marker='o')
        ax1.text(d2, df['high'].iloc[p2_idx], "  Максимумы ↑", color='#f39c12', fontsize=10, fontweight='bold')
        ax2.plot([d1, d2], [df['rsi'].iloc[p1_idx], df['rsi'].iloc[p2_idx]], color='#f39c12', linewidth=2.5, marker='o')
        ax2.text(d2, df['rsi'].iloc[p2_idx], "  RSI ↓", color='#f39c12', fontsize=10, fontweight='bold')
        if sig['vol_div']:
            ax3.plot([d1, d2], [df['volume'].iloc[p1_idx], df['volume'].iloc[p2_idx]], color='#e67e22', linewidth=2, linestyle='--', marker='o')
            ax3.text(d2, df['volume'].iloc[p2_idx], "  Объём ↓", color='#e67e22', fontsize=9)
    else:  # Бычья (по впадинам)
        ax1.plot([d1, d2], [df['low'].iloc[p1_idx], df['low'].iloc[p2_idx]], color='#2ecc71', linewidth=2.5, marker='o')
        ax1.text(d2, df['low'].iloc[p2_idx], "  Минимумы ↓", color='#2ecc71', fontsize=10, fontweight='bold')
        ax2.plot([d1, d2], [df['rsi'].iloc[p1_idx], df['rsi'].iloc[p2_idx]], color='#2ecc71', linewidth=2.5, marker='o')
        ax2.text(d2, df['rsi'].iloc[p2_idx], "  RSI ↑", color='#2ecc71', fontsize=10, fontweight='bold')
        if sig['vol_div']:
            ax3.plot([d1, d2], [df['volume'].iloc[p1_idx], df['volume'].iloc[p2_idx]], color='#2ecc71', linewidth=2, linestyle='--', marker='o')
            ax3.text(d2, df['volume'].iloc[p2_idx], "  Объём (Подтв.)", color='#2ecc71', fontsize=9)

    fig.suptitle(f"{coin} ({tf}) — {sig['type']}", color='white', fontsize=14, fontweight='bold')
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('#131722')
        ax.tick_params(colors='white')
        ax.grid(True, color='#2a2e39', linestyle=':', alpha=0.5)

    fig.patch.set_facecolor('#131722')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, facecolor='#131722')
    plt.close()

def send_telegram_with_photo(message, photo_path, symbol):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠ Токен Telegram или CHAT_ID не настроены!")
        return False

    base, quote = symbol.split('/')
    mexc_url = f"https://www.mexc.com/ru-RU/trade/{base}_{quote}"

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    keyboard = {
        'inline_keyboard': [[{'text': '📊 Открыть график на MEXC', 'url': mexc_url}]]
    }

    try:
        with open(photo_path, 'rb') as photo:
            payload = {
                'chat_id': CHAT_ID,
                'caption': message,
                'parse_mode': 'HTML',
                'reply_markup': json.dumps(keyboard)
            }
            files = {'photo': photo}
            resp = requests.post(url, data=payload, files=files, timeout=20)
            return resp.status_code == 200
    except Exception as e:
        print(f"❌ Ошибка отправки фото в Telegram: {e}")
        return False

def send_one_time_test(sent_signals):
    """Отправляет 1 тестовое сообщение с реальным графиком BTC/USDT, чтобы проверить формат"""
    test_key = "TEST_ALERT_SENT_V1"
    if test_key in sent_signals:
        return False

    try:
        coin, tf = 'BTC/USDT', '1h'
        ohlcv = exchange.fetch_ohlcv(coin, timeframe=tf, limit=80)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['rsi'] = calculate_rsi(df['close'], period=RSI_PERIOD)
        df_closed = df.iloc[:-1].reset_index(drop=True)

        p1, p2 = len(df_closed) - 25, len(df_closed) - 3
        test_sig = {
            'type': '🐂 Бычья дивергенция (Лонг) [ТЕСТ]',
            'p1_idx': p1,
            'p2_idx': p2,
            'price_p1': df_closed['low'].iloc[p1],
            'price_p2': df_closed['low'].iloc[p2],
            'price_now': df_closed['close'].iloc[-1],
            'rsi_p1': df_closed['rsi'].iloc[p1],
            'rsi_p2': df_closed['rsi'].iloc[p2],
            'rsi_now': df_closed['rsi'].iloc[-1],
            'vol_div': True
        }

        photo_name = "test_div.png"
        generate_divergence_chart(df_closed, tf, coin, test_sig, filename=photo_name)

        msg = (
            f"🧪 <b>ТЕСТОВОЕ ОПОВЕЩЕНИЕ ({tf})</b>\n"
            f"<i>Так будут выглядеть реальные сигналы бота:</i>\n\n"
            f"Монета: <b>#{coin.replace('/', '')} (MEXC)</b>\n"
            f"Паттерн: <b>{test_sig['type']}</b>\n"
            f"────────────────────────\n"
            f"📉 Пики цены: <code>{test_sig['price_p1']:.5g}</code> ➔ <code>{test_sig['price_p2']:.5g}</code>\n"
            f"📊 RSI (14): <b>{test_sig['rsi_p1']:.1f} ➔ {test_sig['rsi_p2']:.1f}</b> (Сейчас: {test_sig['rsi_now']:.1f})\n"
            f"🔊 Дивергенция объёма: <b>✅ Да (Подтверждено! 🔥)</b>\n"
            f"💵 Ориентировочный вход: <code>{test_sig['price_now']:.5g}</code>\n"
            f"⏰ Свеча закрыта: <i>{time.strftime('%H:%M', time.gmtime(df_closed['timestamp'].iloc[-1]/1000))} UTC</i>"
        )

        if send_telegram_with_photo(msg, photo_name, coin):
            sent_signals[test_key] = int(time.time())
            print("🧪 Тестовое сообщение успешно отправлено!")
            if os.path.exists(photo_name):
                os.remove(photo_name)
            return True
    except Exception as e:
        print(f"⚠ Ошибка отправки теста: {e}")
    return False

def check_divergences_and_volumes(df):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    rsis = df['rsi'].values
    volumes = df['volume'].values
    timestamps = df['timestamp'].values
    n = len(df)

    rsi_peaks, rsi_troughs = find_pivots(rsis)
    signals = []

    # МЕДВЕЖЬЯ (Шорт)
    if len(rsi_peaks) >= 2:
        p2 = rsi_peaks[-1]
        if (n - 1 - p2) in (1, 2):
            for p1 in reversed(rsi_peaks[:-1]):
                dist = p2 - p1
                if MIN_PEAK_DIST <= dist <= MAX_PEAK_DIST:
                    price_higher = (highs[p2] > highs[p1]) and (closes[p2] >= closes[p1] * 0.998)
                    rsi_lower = rsis[p2] <= (rsis[p1] - 1.5)
                    rsi_zone = (rsis[p1] >= 55)

                    if price_higher and rsi_lower and rsi_zone:
                        vol_div = volumes[p2] < volumes[p1]
                        signals.append({
                            'id': f"BEAR_{int(timestamps[p2])}",
                            'type': '🐻 Медвежья дивергенция (Шорт)',
                            'p1_idx': p1,
                            'p2_idx': p2,
                            'price_p1': highs[p1],
                            'price_p2': highs[p2],
                            'price_now': closes[-1],
                            'rsi_p1': rsis[p1],
                            'rsi_p2': rsis[p2],
                            'rsi_now': rsis[-1],
                            'vol_div': vol_div
                        })
                        break

    # БЫЧЬЯ (Лонг)
    if len(rsi_troughs) >= 2:
        t2 = rsi_troughs[-1]
        if (n - 1 - t2) in (1, 2):
            for t1 in reversed(rsi_troughs[:-1]):
                dist = t2 - t1
                if MIN_PEAK_DIST <= dist <= MAX_PEAK_DIST:
                    price_lower = (lows[t2] < lows[t1]) and (closes[t2] <= closes[t1] * 1.002)
                    rsi_higher = rsis[t2] >= (rsis[t1] + 1.5)
                    rsi_zone = (rsis[t1] <= 45)

                    if price_lower and rsi_higher and rsi_zone:
                        vol_div = volumes[t2] < volumes[t1]
                        signals.append({
                            'id': f"BULL_{int(timestamps[t2])}",
                            'type': '🐂 Бычья дивергенция (Лонг)',
                            'p1_idx': t1,
                            'p2_idx': t2,
                            'price_p1': lows[t1],
                            'price_p2': lows[t2],
                            'price_now': closes[-1],
                            'rsi_p1': rsis[t1],
                            'rsi_p2': rsis[t2],
                            'rsi_now': rsis[-1],
                            'vol_div': vol_div
                        })
                        break
    return signals

def scan_single_pair(coin, tf):
    try:
        ohlcv = exchange.fetch_ohlcv(coin, timeframe=tf, limit=100)
        if len(ohlcv) < 60:
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['rsi'] = calculate_rsi(df['close'], period=RSI_PERIOD)

        df_closed = df.iloc[:-1].reset_index(drop=True)
        signals = check_divergences_and_volumes(df_closed)
        return (coin, tf, signals, df_closed)
    except Exception as e:
        print(f"⚠ Ошибка сканирования {coin} {tf}: {e}")
        return None

def run_scanner():
    sent_signals = load_state()
    state_modified = False

    # Отправляем одноразовый тест, если он ещё не отправлялся
    if send_one_time_test(sent_signals):
        state_modified = True

    available_coins = [pair for pair in TARGET_COINS if pair in markets]
    tasks = [(coin, tf) for tf in TIMEFRAMES for coin in available_coins]
    print(f"⚡ Старт сканирования {len(tasks)} пар на MEXC — {time.strftime('%Y-%m-%d %H:%M:%S')}")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(scan_single_pair, coin, tf) for coin, tf in tasks]
        for future in as_completed(futures):
            res = future.result()
            if not res:
                continue
            coin, tf, signals, df_closed = res

            for sig in signals:
                unique_key = f"{coin}_{tf}_{sig['id']}"
                if unique_key in sent_signals:
                    continue

                photo_name = f"div_{coin.replace('/', '_')}_{tf}.png"
                generate_divergence_chart(df_closed, tf, coin, sig, filename=photo_name)

                vol_status = "✅ Да (Подтверждено! 🔥)" if sig['vol_div'] else "❌ Нет (Только RSI)"
                msg = (
                    f"🔔 <b>СИГНАЛ ДИВЕРГЕНЦИИ ({tf})</b>\n\n"
                    f"Монета: <b>#{coin.replace('/', '')} (MEXC)</b>\n"
                    f"Паттерн: <b>{sig['type']}</b>\n"
                    f"────────────────────────\n"
                    f"📉 Пики цены: <code>{sig['price_p1']:.5g}</code> ➔ <code>{sig['price_p2']:.5g}</code>\n"
                    f"📊 RSI (14): <b>{sig['rsi_p1']:.1f} ➔ {sig['rsi_p2']:.1f}</b> (Сейчас: {sig['rsi_now']:.1f})\n"
                    f"🔊 Дивергенция объёма: <b>{vol_status}</b>\n"
                    f"💵 Ориентировочный вход: <code>{sig['price_now']:.5g}</code>\n"
                    f"⏰ Свеча закрыта: <i>{time.strftime('%H:%M', time.gmtime(df_closed['timestamp'].iloc[-1]/1000))} UTC</i>"
                )

                print(f"🔥 Найден сигнал {unique_key}. Отправка...")
                if send_telegram_with_photo(msg, photo_name, coin):
                    sent_signals[unique_key] = int(time.time())
                    state_modified = True

                if os.path.exists(photo_name):
                    os.remove(photo_name)

    if state_modified:
        save_state(sent_signals)

if __name__ == "__main__":
    run_scanner()
    print("✅ Работа сканера завершена успешно!")
