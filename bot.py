import os
import time
import ccxt
import pandas as pd
import requests
from scipy.signal import find_peaks

# --- НАСТРОЙКИ ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")  
CHAT_ID = os.environ.get("CHAT_ID") 

# --- ПОДКЛЮЧЕНИЕ К БИРЖЕ (MEXC) ---
try:
    exchange = ccxt.mexc({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
        }
    })
    exchange.load_markets()
except Exception as e:
    print(f"Ошибка инициализации биржи: {e}")
    exit()

# Словарь для защиты от повторного спама
last_signals = {}

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
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
    # Метод сглаживания Уайлдера (как в TradingView)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_top_25_crypto_symbols():
    print("🔄 Загружаем топ-25 монет по объему с MEXC...")
    try:
        tickers = exchange.fetch_tickers()
        crypto_list = []
        stablecoins = ['USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'FDUSD', 'USDD']
        exclude_patterns = ['3L', '3S', '6L', '6S', 'BEAR', 'BULL', 'UP', 'DOWN', 'HEDGE']

        for symbol, ticker in tickers.items():
            if '/USDT' in symbol:
                base_currency = symbol.split('/')[0]
                if base_currency in stablecoins or any(p in symbol for p in exclude_patterns):
                    continue
                volume = ticker.get('quoteVolume', 0)
                if volume and volume > 0:
                    crypto_list.append({'symbol': symbol, 'volume': volume})
        
        crypto_list.sort(key=lambda x: x['volume'], reverse=True)
        top_25 = [item['symbol'] for item in crypto_list[:25]]
        
        if not top_25:
            print("❌ Не удалось получить список монет.")
            return []
            
        print(f"✅ Топ-25 по объему загружен: {', '.join(top_25)}")
        return top_25
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке рынков с MEXC: {e}")
        return []

def check_divergence(df):
    closes = df['close'].values
    rsis = df['rsi'].values
    total_bars = len(df)
    
    # Строгие фильтры для отсеивания рыночного шума
    peak_indices, _ = find_peaks(closes, distance=7, prominence=2.0) 
    trough_indices, _ = find_peaks(-closes, distance=7, prominence=2.0)
    
    signals = []

    # 1. Медвежья дивергенция (цена выше, RSI ниже, RSI > 58)
    if len(peak_indices) >= 2:
        p2, p1 = peak_indices[-1], peak_indices[-2]
        
        # Проверяем свежесть последнего пика (должен быть сформирован недавно, в пределах 12 баров)
        if (total_bars - p2) <= 12:
            if closes[p2] > closes[p1] and rsis[p2] < rsis[p1]:
                if rsis[p2] > 58:
                    signals.append('🐻 BEARISH')

    # 2. Бычья дивергенция (цена ниже, RSI выше, RSI < 42)
    if len(trough_indices) >= 2:
        t2, t1 = trough_indices[-1], trough_indices[-2]
        
        # Проверяем свежесть последней впадины
        if (total_bars - t2) <= 12:
            if closes[t2] < closes[t1] and rsis[t2] > rsis[t1]:
                if rsis[t2] < 42:
                    signals.append('🐂 BULLISH')
            
    return signals

def run_bot_iteration():
    top_coins = get_top_25_crypto_symbols()
    if not top_coins:
        print("⚠ Список монет пуст.")
        return

    timeframes = ['4h', '1h', '30m']

    print(f"\n🔍 Сканируем топ-25 по таймфреймам {timeframes} — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for tf in timeframes:
        print(f"⏱ Сканирование таймфрейма: {tf}")
        for coin in top_coins:
            try:
                ohlcv = exchange.fetch_ohlcv(coin, timeframe=tf, limit=150)
                if len(ohlcv) < 100:
                    continue
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['rsi'] = calculate_rsi(df['close'], period=14)
                
                signals = check_divergence(df)
                
                # Берем надежную закрытую свечу [-2]
                last_row = df.iloc[-2]  
                candle_time = last_row['timestamp']
                
                if signals:
                    signal_str = ", ".join(signals)
                    signal_key = f"{coin}_{tf}_{candle_time}_{signal_str}"
                    
                    if last_signals.get(f"{coin}_{tf}") != signal_key:
                        last_signals[f"{coin}_{tf}"] = signal_key
                        
                        msg = (
                            f"🚨 <b>Дивергенция RSI ({tf})</b>\n"
                            f"Биржа: MEXC (Топ-25)\n\n"
                            f"Монета: <b>{coin}</b>\n"
                            f"Сигнал: <b>{signal_str}</b>\n"
                            f"Цена закрытия: {last_row['close']:.6f}\n"
                            f"RSI: {last_row['rsi']:.2f}"
                        )
                        print(f"🔥 НАЙДЕН ТОЧНЫЙ СИГНАЛ | {coin} | TF: {tf} | {signal_str}")
                        send_telegram_message(msg)
                        
            except Exception as e:
                print(f"❌ Ошибка для {coin} ({tf}): {e}")
                time.sleep(0.5)
        print("-" * 40)

if __name__ == "__main__":
    print("🤖 Бот запущен в режиме высокой точности!")
    print("⏱ Проверка будет выполняться каждые 5 минут.")

    while True:
        try:
            run_bot_iteration()
            print("✅ Проверка завершена.")
            print("💤 Следующая проверка через 5 минут...")

        except Exception as e:
            print(f"❌ Общая ошибка: {e}")

        time.sleep(5 * 60)
