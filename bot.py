import os
import time
import ccxt
import pandas as pd
import requests
from scipy.signal import find_peaks

# --- НАСТРОЙКИ (Берем из секретов GitHub) ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")  
CHAT_ID = os.environ.get("CHAT_ID") 

# --- ПОДКЛЮЧЕНИЕ К БИРЖЕ (MEXC) ---
try:
    # Используем ccxt.mexc. Рекомендуется использовать unified api.
    exchange = ccxt.mexc({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot', # Только спотовый рынок
        }
    })
    # Предварительная загрузка рынков не обязательна, но полезна
    exchange.load_markets()
except Exception as e:
    print(f"Ошибка инициализации биржи: {e}")
    exit()

# Словарь для защиты от спама (запоминаем уникальный ключ сигнала)
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
    # Используем Wilder's Smoothing method (как в TradingView)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_top_20_crypto_symbols():
    print("🔄 Загружаем топ-20 монет по объему с MEXC...")
    try:
        # Загружаем все тикеры одним запросом
        tickers = exchange.fetch_tickers()
        
        crypto_list = []
        # Исключаем стейблкоины и токены с плечом из отбора
        stablecoins = ['USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'FDUSD', 'USDD']
        exclude_patterns = ['3L', '3S', '6L', '6S', 'BEAR', 'BULL', 'UP', 'DOWN', 'HEDGE']

        for symbol, ticker in tickers.items():
            # Проверяем, что это пара к USDT
            if '/USDT' in symbol:
                base_currency = symbol.split('/')[0]
                
                # Жестко фильтруем стейблкоины и деривативы/токены с плечом
                if base_currency in stablecoins or any(p in symbol for p in exclude_patterns):
                    continue
                
                # Берем объем торгов в USDT за 24 часа
                volume = ticker.get('quoteVolume', 0)
                
                # Если объем есть (торгуется), добавляем в список
                if volume and volume > 0:
                    crypto_list.append({'symbol': symbol, 'volume': volume})
        
        # Сортируем по объему (от большего к меньшему)
        crypto_list.sort(key=lambda x: x['volume'], reverse=True)
        
        # Берем топ-20
        top_20 = [item['symbol'] for item in crypto_list[:20]]
        
        if not top_20:
            print("❌ Не удалось получить список монет.")
            return []
            
        print(f"✅ Топ-20 по объему загружен: {', '.join(top_20)}")
        return top_20
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке рынков с MEXC: {e}")
        return []

def check_divergence(df):
    closes = df['close'].values
    rsis = df['rsi'].values
    
    # Настройки пиков. Дистанцию 5 можно менять (меньше - больше сигналов, но они шумнее)
    peak_indices, _ = find_peaks(closes, distance=5, prominence=1) 
    trough_indices, _ = find_peaks(-closes, distance=5, prominence=1)
    
    signals = []

    # Проверка Медвежьей дивергенции (цена выше, RSI ниже)
    if len(peak_indices) >= 2:
        p2, p1 = peak_indices[-1], peak_indices[-2]
        if closes[p2] > closes[p1] and rsis[p2] < rsis[p1]:
            # Дополнительный фильтр: RSI должен быть выше 60
            if rsis[p2] > 60:
                signals.append('🐻 BEARISH')

    # Проверка Бычьей дивергенции (цена ниже, RSI выше)
    if len(trough_indices) >= 2:
        t2, t1 = trough_indices[-1], trough_indices[-2]
        if closes[t2] < closes[t1] and rsis[t2] > rsis[t1]:
            # Дополнительный фильтр: RSI должен быть ниже 40
            if rsis[t2] < 40:
                signals.append('🐂 BULLISH')
            
    return signals

def run_bot_iteration():
    top_coins = get_top_20_crypto_symbols()
    if not top_coins:
        print("⚠ Список монет пуст.")
        return

    print(f"\n🔍 Сканируем топ-20 (таймфрейм 1h) — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for coin in top_coins:
        try:
            # Получаем 150 свечей для надежного расчета RSI и find_peaks
            ohlcv = exchange.fetch_ohlcv(coin, timeframe='1h', limit=150)
            if len(ohlcv) < 100:
                continue # Пропускаем монеты с недостаточной историей
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['rsi'] = calculate_rsi(df['close'], period=14)
            
            signals = check_divergence(df)
            last_row = df.iloc[-2]  # Используем последнюю закрытую свечу
            candle_time = last_row['timestamp']
            
            print(f"Монета: {coin:<12} | Цена: {last_row['close']:<10.6f} | RSI: {last_row['rsi']:.2f}")
            
            if signals:
                signal_str = ", ".join(signals)
                # Уникальный ключ для защиты от отправки одного и того же сигнала
                signal_key = f"{coin}_{candle_time}_{signal_str}"
                
                if last_signals.get(coin) != signal_key:
                    last_signals[coin] = signal_key
                    msg = (
                        f"🚨 <b>ВНИМАНИЕ! Дивергенция RSI (1h)</b>\n"
                        f"Биржа: MEXC (Топ-20 по объему)\n\n"
                        f"Монета: <b>{coin}</b>\n"
                        f"Сигнал: <b>{signal_str}</b>\n"
                        f"Цена: {last_row['close']:.6f}\n"
                        f"RSI: {last_row['rsi']:.2f}"
                    )
                    print(f"🔥 НАЙДЕН СИГНАЛ по {coin}: {signal_str}")
                    send_telegram_message(msg)
                else:
                    print(f"ℹ️ Сигнал по {coin} уже был отправлен.")
                    
        except Exception as e:
            print(f"❌ Ошибка для {coin}: {e}")
            time.sleep(1) # Небольшая пауза при ошибке API

if __name__ == "__main__":
    print("🤖 Бот запущен в облаке GitHub Actions (MEXC, топ-20 крипто-монет)!")
    try:
        run_bot_iteration()
        print("✅ Проверка завершена.")
    except Exception as e:
        print(f"❌ Общая ошибка: {e}")
