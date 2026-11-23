import os
import time
import json
import ccxt
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')  # Для генерации картинок без GUI на сервере (headless)
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

TIMEFRAMES = ['15m', '30m', '1h']
RSI_PERIOD = 14
PIVOT_LEFT = 5       # Свечей слева до пика
PIVOT_RIGHT = 2      # Увеличили до 2 закрытых свечей справа для большей точности
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
                return json.load(f)
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
    """Поиск локальных пиков и впадин"""
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
    """Создаёт 3-панельный скриншот: Цена, RSI, Объём с нарисованными линиями дивергенции"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, 
                                       gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Подготовка индексов для отображения
    df = df.suffix_index_as_dates = df.copy()
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    dates = df['date']
    
    # --- 1. ПАНЕЛЬ ЦЕНЫ ---
    # Рисуем свечи (упрощенно линиями для скорости генерации на GitHub Actions)
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
    ax3.bar(dates, df['volume'], color=colors, alpha=0.8, width=0.005 * (dates.iloc[-1] - dates.iloc[0]))
    
    # Рисуем линии дивергенции по точкам
    p1_idx, p2_idx = sig['p1_idx'], sig['p2_idx']
    d1, d2 = dates[p1_idx], dates[p2_idx]
    
    if "🐻" in sig['type']: # Медвежья (по пикам)
        # На цене
        ax1.plot([d1, d2], [df['high'][p1_idx], df['high'][p2_idx]], color='#f39c12', linewidth=2.5, marker='o')
        ax1.text(d2, df['high'][p2_idx], "  Максимумы повышаются", color='#f39c12', fontsize=10, fontweight='bold')
        # На RSI
        ax2.plot([d1, d2], [df['rsi'][p1_idx], df['rsi'][p2_idx]], color='#f39c12', linewidth=2.5, marker='o')
        ax2.text(d2, df['rsi'][p2_idx], "  RSI понижается", color='#f39c12', fontsize=10, fontweight='bold')
        # На объемах
        if sig['vol_div']:
            ax3.plot([d1, d2], [df['volume'][p1_idx], df['volume'][p2_idx]], color='#e67e22', linewidth=2, linestyle='--', marker='o')
            ax3.text(d2, df['volume'][p2_idx], "  Объём падает (Дивер)", color='#e67e22', fontsize=9)
    else: # Бычья (по впадинам)
        # На цене
        ax1.plot([d1, d2], [df['low'][p1_idx], df['low'][p2_idx]], color='#2ecc71', linewidth=2.5, marker='o')
        ax1.text(d2, df['low'][p2_idx], "  Минимумы понижаются", color='#2ecc71', fontsize=10, fontweight='bold')
        # На RSI
        ax2.plot([d1, d2], [df['rsi'][p1_idx], df['rsi'][p2_idx]], color='#2ecc71', linewidth=2.5, marker='o')
        ax2.text(d2, df['rsi'][p2_idx], "  RSI повышается", color='#2ecc71', fontsize=10, fontweight='bold')
        # На объемах
        if sig['vol_div']:
            ax3.plot([d1, d2], [df['volume'][p1_idx], df['volume'][p2_idx]], color='#2ecc71', linewidth=2, linestyle='--', marker='o')
            ax3.text(d2, df['volume'][p2_idx], "  Объём растёт (Подтверждено)", color='#2ecc71', fontsize=9)

    # Оформление интерфейса
    fig.suptitle(f"📈 {coin} ({tf}) - {sig['type']}", color='white', fontsize=14, fontweight='bold')
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

def check_divergences_and_volumes(df):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    rsis = df['rsi'].values
    volumes = df['volume'].values
    timestamps = df['timestamp'].values
    n = len(df)

    # 1. Поиск пиков и впадин отдельно на RSI и Цене
    rsi_peaks, rsi_troughs = find_pivots(rsis)
    signals = []

    # МЕДВЕЖЬЯ (Шорт)
    if len(rsi_peaks) >= 2:
        p2 = rsi_peaks[-1]
        if (n - 1 - p2) in (1, 2):  # Свежий экстремум
            for p1 in reversed(rsi_peaks[:-1]):
                dist = p2 - p1
                if MIN_PEAK_DIST <= dist <= MAX_PEAK_DIST:
                    price_higher = (highs[p2] > highs[p1]) and (closes[p2] >= closes[p1] * 0.998)
                    rsi_lower = rsis[p2] <= (rsis[p1] - 1.5)
                    rsi_zone = (rsis[p1] >= 55)
                    
                    if price_higher and rsi_lower and rsi_zone:
                        # Проверка объёма (дивергенция по объемам: активность падает на втором пике)
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
                        # На бычьем тренде дивер объёма — это либо рост интереса, либо истощение продаж.
                        # Если объём на втором дне выше или угасает — фиксируем в сигнале.
                        vol_div = volumes[t2] < volumes[t1] # Снижение давления продаж
                        
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
        
        # Строго исключаем текущую формирующуюся свечу
        df_closed = df.iloc[:-1].reset_index(drop=True)
        signals = check_divergences_and_volumes(df_closed)
        return (coin, tf, signals, df_closed)
    except Exception as e:
        print(f"⚠ Ошибка сканирования {coin} {tf}: {e}")
        return None

def run_scanner():
    sent_signals = load_state()
    state_modified = False
    
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
                
                # Генерация графика
                photo_name = f"div_{coin.replace('/', '_')}_{tf}.png"
                generate_divergence_chart(df_closed, tf, coin, sig, filename=photo_name)
                
                # Текст сообщения
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
                
                # Чистим временный файл изображения
                if os.path.exists(photo_name):
                    os.remove(photo_name)
                    
    if state_modified:
        save_state(sent_signals)

if __name__ == "__main__":
    run_scanner()
    print("✅ Работа сканера завершена успешно!")
