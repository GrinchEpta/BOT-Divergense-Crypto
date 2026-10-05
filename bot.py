import os
import time
import ccxt
import pandas as pd
import requests
from scipy.signal import find_peaks

# --- НАСТРОЙКИ (Берем из переменных окружения GitHub Actions) ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "ВАШ_ТОКЕН_БОТА_ИЗ_BOTFATHER")  
CHAT_ID = os.environ.get("CHAT_ID", "ВАШ_CHAT_ID")                       

# Подключаемся к Binance
exchange = ccxt.binance({
    'enableRateLimit': True,
})

# Словарь для запоминания последних отправленных сигналов (защита от спама)
last_signals = {}

def send_telegram_message(message):
    if TELEGRAM_TOKEN == "ВАШ_ТОКЕН_БОТА_ИЗ_BOTFATHER" or not TELEGRAM_TOKEN:
        print("⚠ Не указан Telegram Token или Chat ID!")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Ошибка отправки в Telegram: {e}")

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_top_15_symbols():
    print("🔄 Загружаем актуальный топ-15 монет с биржи...")
    exchange.load_markets()
    tickers = exchange.fetch_tickers()
    
    usdt_tickers = []
    for symbol, ticker in tickers.items():
        if '/USDT' in symbol and not any(x in symbol for x in ['UP', 'DOWN', 'BEAR', 'BULL']):
            volume = ticker.get('quoteVolume', 0)
            if volume:
                usdt_tickers.append((symbol, volume))
                
    usdt_tickers.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in usdt_tickers[:15]]

def check_divergence(df):
    closes = df['close'].values
    rsis = df['rsi'].values
    
    peak_indices, _ = find_peaks(closes, distance=5)
    trough_indices, _ = find_peaks(-closes, distance=5)
    
    signals = []

    if len(peak_indices) >= 2:
        p2, p1 = peak_indices[-1], peak_indices[-2]
        if closes[p2] > closes[p1] and rsis[p2] < rsis[p1]:
            signals.append('BEARISH (Медвежья)')

    if len(trough_indices) >= 2:
        t2, t1 = trough_indices[-1], trough_indices[-2]
        if closes[t2] < closes[t1] and rsis[t2] > rsis[t1]:
            signals.append('BULLISH (Бычья)')
            
    return signals

def run_bot_iteration():
    top_coins = get_top_15_symbols()
    print(f"\n🔍 Сканируем топ-15 монет (таймфрейм 1h) — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for coin in top_coins:
        try:
            ohlcv = exchange.fetch_ohlcv(coin, timeframe='1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['rsi'] = calculate_rsi(df['close'], period=14)
            
            signals = check_divergence(df)
            last_row = df.iloc[-2]  # Последняя закрытая свеча
            candle_time = last_row['timestamp']
            
            print(f"Монета: {coin:<10} | Цена: {last_row['close']:<10.4f} | RSI: {last_row['rsi']:.2f}")
            
            if signals:
                signal_str = ", ".join(signals)
                signal_key = f"{coin}_{candle_time}_{signal_str}"
                
                if last_signals.get(coin) != signal_key:
                    last_signals[coin] = signal_key
                    msg = (
                        f"🚨 <b>ВНИМАНИЕ! Дивергенция RSI (1h)</b>\n\n"
                        f"Монета: <b>{coin}</b>\n"
                        f"Сигнал: <b>{signal_str}</b>\n"
                        f"Цена: {last_row['close']}\n"
                        f"RSI: {last_row['rsi']:.2f}"
                    )
                    print(f"🔥 НАЙДЕН СИГНАЛ по {coin}: {signal_str}")
                    send_telegram_message(msg)
                
        except Exception as e:
            print(f"Ошибка для {coin}: {e}")

if __name__ == "__main__":
    print("🤖 Бот запущен в облаке GitHub Actions!")
    try:
        run_bot_iteration()
        print("✅ Проверка успешно завершена.")
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")
