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

TOP_N_COINS = 50        # Топ-50 самых ликвидных USDT-M фьючерсов MEXC
MAX_BTC_CORR = 0.50     # Макс. корреляция с BTC (ниже 0.50 = монета ходит сама по себе)
BTC_SYMBOL = 'BTC/USDT:USDT'
ALWAYS_INCLUDE = [
    'BTC/USDT:USDT',
    'ETH/USDT:USDT',
    'ZEC/USDT:USDT',
    'XMR/USDT:USDT'
]  # Эти монеты сканируем всегда, независимо от корреляции с BTC

TIMEFRAMES = ['30m', '1h']
RSI_PERIOD = 14
PIVOT_LEFT = 3          # Свечей слева до пика (ловит и короткие развороты)
PIVOT_RIGHT = 2         # 2 закрытые свечи справа для подтверждения
MIN_PEAK_DIST = 3       # Мин. расстояние между пиками
MAX_PEAK_DIST = 60      # Макс. расстояние между пиками
LOOKBACK_PIVOTS = 4     # Проверяем до 4 предыдущих пиков/впадин назад

# ФИЛЬТРЫ ПРОТИВ БОКОВИКА
MIN_PRICE_DIFF_PCT = 0.003  # Мин. разница между пиками цены: 0.3%
MIN_RSI_DIFF = 2.5          # Мин. разница между пиками RSI: 2.5 пункта

STABLECOINS = {
    'USDC', 'FDUSD', 'TUSD', 'DAI', 'USDE', 'BUSD',
    'USDD', 'PYUSD', 'EUR', 'USD1', 'USDP', 'RLUSD'
}

try:
    exchange = ccxt.mexc({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}  # Фьючерсный рынок MEXC (USDT-M)
    })
    markets = exchange.load_markets()
except Exception as e:
    print(f"❌ Ошибка инициализации MEXC Futures: {e}")
    exit(1)

def get_base_name(symbol):
    """Извлекает тикер монеты (например, 'XMR' из 'XMR/USDT:USDT')."""
    return symbol.split('/')[0]

def get_top_liquid_coins(limit=TOP_N_COINS):
    """Загружает топ-N самых ликвидных USDT-M фьючерсов на MEXC по суточному объёму."""
    try:
        tickers = exchange.fetch_tickers()
        valid_pairs = []
        for symbol, ticker in tickers.items():
            if not symbol.endswith('/USDT:USDT') or symbol not in markets:
                continue
            market = markets[symbol]
            if not market.get('active', True) or not market.get('swap', False):
                continue
            base = get_base_name(symbol)
            if base in STABLECOINS:
                continue
            vol = ticker.get('quoteVolume') or (
                (ticker.get('baseVolume') or 0) * (ticker.get('last') or 0)
            ) or 0
            if vol > 0:
                valid_pairs.append((symbol, vol))

        valid_pairs.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [pair[0] for pair in valid_pairs[:limit]]
        for c in ALWAYS_INCLUDE:
            if c in markets and c not in top_symbols:
                top_symbols.append(c)
        return top_symbols
    except Exception as e:
        print(f"⚠ Ошибка загрузки топ-{limit} фьючерсов: {e}")
        return [c for c in ALWAYS_INCLUDE if c in markets]

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

def calculate_btc_correlation(coin_df, btc_df):
    """Считает корреляцию Пирсона по доходностям (pct_change) с BTC Futures."""
    merged = pd.merge(
        coin_df[['timestamp', 'close']],
        btc_df[['timestamp', 'close']],
        on='timestamp',
        suffixes=('_coin', '_btc')
    )
    if len(merged) < 30:
        return 1.0
    ret_coin = merged['close_coin'].pct_change().dropna()
    ret_btc = merged['close_btc'].pct_change().dropna()
    corr = ret_coin.corr(ret_btc)
    return float(corr) if not np.isnan(corr) else 1.0

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

def is_line_clean(values, idx1, idx2, is_upper=True):
    """Проверяет, что на RSI между точками idx1 и idx2 нет «прокола» линии."""
    if idx2 - idx1 <= 1:
        return True
    y1, y2 = values[idx1], values[idx2]
    for i in range(idx1 + 1, idx2):
        ratio = (i - idx1) / (idx2 - idx1)
        line_val = y1 + (y2 - y1) * ratio
        if is_upper and values[i] > line_val + 1.0:
            return False
        if not is_upper and values[i] < line_val - 1.0:
            return False
    return True

def generate_divergence_chart(df, tf, coin, sig, filename="divergence.png"):
    """Создаёт 3-панельный скриншот: Цена, RSI, Объём с линиями дивергенции."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={'height_ratios': [2, 1, 1]}
    )

    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    dates = df['date']
    bar_width = (dates.iloc[1] - dates.iloc[0]) * 0.6

    ax1.plot(dates, df['close'], color='#5dade2', linewidth=1.5)
    ax1.fill_between(dates, df['low'], df['high'], color='#5dade2', alpha=0.12)

    ax2.plot(dates, df['rsi'], color='#af7ac5', linewidth=1.5)
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.axhline(50, color='gray', linestyle=':', alpha=0.3)
    ax2.set_ylim(10, 90)

    colors = ['#27ae60' if c >= o else '#e74c3c' for c, o in zip(df['close'], df['open'])]
    ax3.bar(dates, df['volume'], color=colors, alpha=0.8, width=bar_width)

    p1_idx, p2_idx = sig['p1_idx'], sig['p2_idx']
    d1, d2 = dates.iloc[p1_idx], dates.iloc[p2_idx]
    is_bear = sig['direction'] == 'BEAR'
    line_color = '#f39c12' if is_bear else '#2ecc71'

    if is_bear:
        y_price_1, y_price_2 = df['high'].iloc[p1_idx], df['high'].iloc[p2_idx]
    else:
        y_price_1, y_price_2 = df['low'].iloc[p1_idx], df['low'].iloc[p2_idx]

    ax1.plot([d1, d2], [y_price_1, y_price_2], color=line_color, linewidth=2.5, marker='o')
    price_arrow = "↑" if y_price_2 > y_price_1 else "↓"
    ax1.text(d2, y_price_2, f"  Цена {price_arrow}", color=line_color, fontsize=10, fontweight='bold')

    y_rsi_1, y_rsi_2 = df['rsi'].iloc[p1_idx], df['rsi'].iloc[p2_idx]
    ax2.plot([d1, d2], [y_rsi_1, y_rsi_2], color=line_color, linewidth=2.5, marker='o')
    rsi_arrow = "↑" if y_rsi_2 > y_rsi_1 else "↓"
    ax2.text(d2, y_rsi_2, f"  RSI {rsi_arrow}", color=line_color, fontsize=10, fontweight='bold')

    if sig['vol_div']:
        y_vol_1, y_vol_2 = df['volume'].iloc[p1_idx], df['volume'].iloc[p2_idx]
        ax3.plot([d1, d2], [y_vol_1, y_vol_2], color=line_color, linewidth=2, linestyle='--', marker='o')
        ax3.text(d2, y_vol_2, "  Объём (Подтв.)", color=line_color, fontsize=9)

    display_pair = f"{get_base_name(coin)}/USDT"
    fig.suptitle(f"{display_pair} Futures ({tf}) — {sig['type']}", color='white', fontsize=14, fontweight='bold')
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

    base = get_base_name(symbol)
    mexc_url = f"https://futures.mexc.com/ru-RU/exchange/{base}_USDT"

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    keyboard = {
        'inline_keyboard': [[{'text': '📊 Открыть фьючерс на MEXC', 'url': mexc_url}]]
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

def check_divergences_and_volumes(df, max_age=2):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    rsis = df['rsi'].values
    volumes = df['volume'].values
    timestamps = df['timestamp'].values
    n = len(df)

    rsi_peaks, rsi_troughs = find_pivots(rsis)
    signals = []

    # --- МЕДВЕЖЬИ (ПРОВЕРЯЕМ ДО 4 ПРЕДЫДУЩИХ ПИКОВ) ---
    if len(rsi_peaks) >= 2:
        p2 = rsi_peaks[-1]
        age = n - 1 - p2

        if 1 <= age <= max_age:
            prev_peaks = rsi_peaks[max(0, len(rsi_peaks) - 1 - LOOKBACK_PIVOTS) : -1]
            for p1 in prev_peaks:  # От более широкой к ближней
                dist = p2 - p1
                if not (MIN_PEAK_DIST <= dist <= MAX_PEAK_DIST):
                    continue
                if not is_line_clean(rsis, p1, p2, is_upper=True):
                    continue

                classic_bear = (
                    highs[p2] >= highs[p1] * (1 + MIN_PRICE_DIFF_PCT) and
                    rsis[p2] <= rsis[p1] - MIN_RSI_DIFF
                )
                hidden_bear = (
                    highs[p2] <= highs[p1] * (1 - MIN_PRICE_DIFF_PCT) and
                    rsis[p2] >= rsis[p1] + MIN_RSI_DIFF
                )

                if classic_bear or hidden_bear:
                    sig_kind = "BEAR_CLASSIC" if classic_bear else "BEAR_HIDDEN"
                    sig_label = (
                        "🐻 Классическая медвежья (Шорт)"
                        if classic_bear else
                        "🐻 Скрытая медвежья (Шорт по тренду)"
                    )
                    signals.append({
                        'id': f"{sig_kind}_{int(timestamps[p2])}",
                        'direction': 'BEAR',
                        'type': sig_label,
                        'p1_idx': p1,
                        'p2_idx': p2,
                        'age': age,
                        'price_p1': highs[p1],
                        'price_p2': highs[p2],
                        'price_now': closes[-1],
                        'rsi_p1': rsis[p1],
                        'rsi_p2': rsis[p2],
                        'rsi_now': rsis[-1],
                        'vol_div': volumes[p2] < volumes[p1]
                    })
                    break  # Берём самую широкую чистую линию для этой свечи

    # --- БЫЧЬИ (ПРОВЕРЯЕМ ДО 4 ПРЕДЫДУЩИХ ВПАДИН) ---
    if len(rsi_troughs) >= 2:
        t2 = rsi_troughs[-1]
        age = n - 1 - t2

        if 1 <= age <= max_age:
            prev_troughs = rsi_troughs[max(0, len(rsi_troughs) - 1 - LOOKBACK_PIVOTS) : -1]
            for t1 in prev_troughs:  # От более широкой к ближней
                dist = t2 - t1
                if not (MIN_PEAK_DIST <= dist <= MAX_PEAK_DIST):
                    continue
                if not is_line_clean(rsis, t1, t2, is_upper=False):
                    continue

                classic_bull = (
                    lows[t2] <= lows[t1] * (1 - MIN_PRICE_DIFF_PCT) and
                    rsis[t2] >= rsis[t1] + MIN_RSI_DIFF
                )
                hidden_bull = (
                    lows[t2] >= lows[t1] * (1 + MIN_PRICE_DIFF_PCT) and
                    rsis[t2] <= rsis[t1] - MIN_RSI_DIFF
                )

                if classic_bull or hidden_bull:
                    sig_kind = "BULL_CLASSIC" if classic_bull else "BULL_HIDDEN"
                    sig_label = (
                        "🐂 Классическая бычья (Лонг)"
                        if classic_bull else
                        "🐂 Скрытая бычья (Лонг по тренду)"
                    )
                    signals.append({
                        'id': f"{sig_kind}_{int(timestamps[t2])}",
                        'direction': 'BULL',
                        'type': sig_label,
                        'p1_idx': t1,
                        'p2_idx': t2,
                        'age': age,
                        'price_p1': lows[t1],
                        'price_p2': lows[t2],
                        'price_now': closes[-1],
                        'rsi_p1': rsis[t1],
                        'rsi_p2': rsis[t2],
                        'rsi_now': rsis[-1],
                        'vol_div': volumes[t2] < volumes[t1]
                    })
                    break  # Берём самую широкую чистую линию для этой свечи

    return signals

def fetch_df(coin, tf, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(coin, timeframe=tf, limit=limit)
        if len(ohlcv) < 60:
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['rsi'] = calculate_rsi(df['close'], period=RSI_PERIOD)
        return df.iloc[:-1].reset_index(drop=True)  # Только закрытые свечи
    except Exception:
        return None

def format_message(coin, tf, sig, df_closed, btc_corr):
    base = get_base_name(coin)
    vol_status = "✅ Да (Подтверждено! 🔥)" if sig['vol_div'] else "❌ Нет (Только RSI)"
    price_diff_pct = ((sig['price_p2'] - sig['price_p1']) / sig['price_p1']) * 100
    corr_line = "🔗 Базовый актив (BTC)\n" if coin == BTC_SYMBOL else f"🔗 Корреляция с BTC (1h): <b>{btc_corr:.2f}</b>\n"

    return (
        f"🔔 <b>СИГНАЛ ДИВЕРГЕНЦИИ ({tf})</b>\n\n"
        f"Монета: <b>#{base}USDT (MEXC Futures)</b>\n"
        f"Паттерн: <b>{sig['type']}</b>\n"
        f"{corr_line}"
        f"────────────────────────\n"
        f"📉 Экстремумы цены: <code>{sig['price_p1']:.5g}</code> ➔ <code>{sig['price_p2']:.5g}</code> ({price_diff_pct:+.2f}%)\n"
        f"📊 RSI (14): <b>{sig['rsi_p1']:.1f} ➔ {sig['rsi_p2']:.1f}</b> (Сейчас: {sig['rsi_now']:.1f})\n"
        f"🔊 Дивергенция объёма: <b>{vol_status}</b>\n"
        f"💵 Текущая цена: <code>{sig['price_now']:.5g}</code>\n"
        f"⏰ Свеча закрыта: <i>{time.strftime('%H:%M', time.gmtime(df_closed['timestamp'].iloc[-1]/1000))} UTC</i>"
    )

def run_scanner():
    sent_signals = load_state()
    state_modified = False

    # 1. Загружаем эталонный график BTC Futures (1h)
    btc_df_1h = fetch_df(BTC_SYMBOL, '1h', limit=100)
    if btc_df_1h is None:
        print("❌ Не удалось загрузить свечи BTC Futures")
        return

    top_coins = get_top_liquid_coins(TOP_N_COINS)
    print(f"⚡ Проверка корреляции с BTC для топ-{len(top_coins)} фьючерсов MEXC...")

    # 2. Загружаем 1h для всех топ-50 и отбираем некоррелирующие с BTC
    selected_coins = {}  # {coin: (corr, df_1h)}

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_map = {executor.submit(fetch_df, coin, '1h', 100): coin for coin in top_coins}
        for future in as_completed(future_map):
            coin = future_map[future]
            df_1h = future.result()
            if df_1h is None:
                continue

            corr = 1.0 if coin == BTC_SYMBOL else calculate_btc_correlation(df_1h, btc_df_1h)
            if coin in ALWAYS_INCLUDE or abs(corr) <= MAX_BTC_CORR:
                selected_coins[coin] = (corr, df_1h)

    decorrelated_list = [
        f"{get_base_name(c)} ({v[0]:.2f})"
        for c, v in selected_coins.items() if c != BTC_SYMBOL
    ]
    print(f"🎯 Отобрано независимых фьючерсов ({len(selected_coins)}): {', '.join(decorrelated_list)}")

    # 3. Проверяем дивергенции на 1h и догружаем 30m для отобранных
    results_to_check = []
    for coin, (corr, df_1h) in selected_coins.items():
        results_to_check.append((coin, '1h', corr, df_1h))

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_30m = {
            executor.submit(fetch_df, coin, '30m', 100): (coin, corr)
            for coin, (corr, _) in selected_coins.items()
        }
        for future in as_completed(future_30m):
            coin, corr = future_30m[future]
            df_30m = future.result()
            if df_30m is not None:
                results_to_check.append((coin, '30m', corr, df_30m))

    # 4. Ищем сигналы и отправляем в Telegram
    for coin, tf, corr, df_closed in results_to_check:
        signals = check_divergences_and_volumes(df_closed, max_age=2)
        for sig in signals:
            base = get_base_name(coin)
            unique_key = f"{base}_USDT_SWAP_{tf}_{sig['id']}"
            if unique_key in sent_signals:
                continue

            photo_name = f"div_{base}_USDT_{tf}.png"
            generate_divergence_chart(df_closed, tf, coin, sig, filename=photo_name)
            msg = format_message(coin, tf, sig, df_closed, corr)

            print(f"🔥 Найден сигнал {unique_key} (корр. BTC: {corr:.2f}). Отправка...")
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
