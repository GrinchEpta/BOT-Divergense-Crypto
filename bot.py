import os
import time
import json
import ccxt
import pandas as pd
import requests
from scipy.signal import find_peaks

# ============================================================
# НАСТРОЙКИ
# ============================================================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

CHECK_INTERVAL = 5 * 60
SIGNALS_FILE = "signals.json"

TOP_COINS = 30

TIMEFRAMES = ["4h", "1h", "30m"]

OHLCV_LIMIT = 150

# ============================================================
# ПОДКЛЮЧЕНИЕ К MEXC
# ============================================================

try:
    exchange = ccxt.mexc({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot"
        }
    })

    exchange.load_markets()

    print("✅ Подключение к MEXC успешно.")

except Exception as e:
    print(f"❌ Ошибка инициализации биржи: {e}")
    exit()

# ============================================================
# ЗАГРУЗКА ОТПРАВЛЕННЫХ СИГНАЛОВ
# ============================================================

def load_sent_signals():
    if not os.path.exists(SIGNALS_FILE):
        return {}

    try:
        with open(
            SIGNALS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, dict):
                return data

    except Exception as e:
        print(
            f"⚠ Ошибка загрузки "
            f"{SIGNALS_FILE}: {e}"
        )

    return {}


def save_sent_signals():
    try:
        with open(
            SIGNALS_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                last_signals,
                file,
                ensure_ascii=False,
                indent=4
            )

    except Exception as e:
        print(
            f"⚠ Ошибка сохранения "
            f"{SIGNALS_FILE}: {e}"
        )


last_signals = load_sent_signals()

# ============================================================
# TELEGRAM
# ============================================================

def send_telegram_message(message):

    if not TELEGRAM_TOKEN or not CHAT_ID:
        print(
            "⚠ Не указан Telegram Token "
            "или Chat ID!"
        )
        return

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=10
        )

        if response.status_code != 200:

            print(
                f"⚠ Ошибка Telegram: "
                f"{response.text}"
            )

    except Exception as e:

        print(
            f"❌ Ошибка отправки "
            f"в Telegram: {e}"
        )

# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (
        100 / (1 + rs)
    )

# ============================================================
# TOP-30 МОНЕТ MEXC ПО ОБОРОТУ
# ============================================================

def get_top_30_crypto_symbols():

    print(
        "🔄 Загружаем TOP-30 монет "
        "по обороту с MEXC..."
    )

    try:

        tickers = exchange.fetch_tickers()

        crypto_list = []

        stablecoins = [
            "USDT",
            "USDC",
            "BUSD",
            "TUSD",
            "DAI",
            "FDUSD",
            "USDD",
            "USDE",
            "PYUSD"
        ]

        exclude_patterns = [
            "3L",
            "3S",
            "5L",
            "5S",
            "6L",
            "6S",
            "BEAR",
            "BULL",
            "UP",
            "DOWN",
            "HEDGE"
        ]

        for symbol, ticker in tickers.items():

            # Только пары USDT
            if not symbol.endswith("/USDT"):
                continue

            base_currency = symbol.split("/")[0]

            # Исключаем стейблкоины
            if base_currency in stablecoins:
                continue

            # Исключаем ETF/левереджированные токены
            if any(
                pattern in symbol
                for pattern in exclude_patterns
            ):
                continue

            volume = ticker.get(
                "quoteVolume",
                0
            )

            if volume is None:
                continue

            try:
                volume = float(volume)
            except Exception:
                continue

            if volume <= 0:
                continue

            crypto_list.append({
                "symbol": symbol,
                "volume": volume
            })

        # Сортируем по обороту
        crypto_list.sort(
            key=lambda x: x["volume"],
            reverse=True
        )

        # Берём TOP-30
        top_30 = [
            item["symbol"]
            for item in crypto_list[:TOP_COINS]
        ]

        if not top_30:

            print(
                "❌ Не удалось получить "
                "список монет."
            )

            return []

        print(
            f"✅ TOP-{TOP_COINS} загружен:"
        )

        for index, symbol in enumerate(
            top_30,
            start=1
        ):

            item = next(
                (
                    x
                    for x in crypto_list
                    if x["symbol"] == symbol
                ),
                None
            )

            if item:

                print(
                    f"{index:2}. "
                    f"{symbol:<15} "
                    f"оборот: "
                    f"${item['volume']:,.0f}"
                )

        print()

        return top_30

    except Exception as e:

        print(
            f"❌ Ошибка при загрузке "
            f"рынков MEXC: {e}"
        )

        return []

# ============================================================
# ПОИСК ДИВЕРГЕНЦИЙ
# ============================================================

def check_divergence(df):

    closes = df["close"].values

    rsis = df["rsi"].values

    total_bars = len(df)

    # --------------------------------------------------------
    # ПИКИ
    # --------------------------------------------------------

    peak_indices, _ = find_peaks(
        closes,
        distance=7,
        prominence=2.0
    )

    # --------------------------------------------------------
    # ВПАДИНЫ
    # --------------------------------------------------------

    trough_indices, _ = find_peaks(
        -closes,
        distance=7,
        prominence=2.0
    )

    signals = []

    # ========================================================
    # BEAR — МЕДВЕЖЬЯ ДИВЕРГЕНЦИЯ
    # ========================================================

    if len(peak_indices) >= 2:

        p2 = peak_indices[-1]

        p1 = peak_indices[-2]

        if (
            total_bars - p2
        ) <= 12:

            if (
                closes[p2] > closes[p1]
                and
                rsis[p2] < rsis[p1]
                and
                rsis[p2] > 58
            ):

                signals.append("BEAR")

    # ========================================================
    # BULL — БЫЧЬЯ ДИВЕРГЕНЦИЯ
    # ========================================================

    if len(trough_indices) >= 2:

        t2 = trough_indices[-1]

        t1 = trough_indices[-2]

        if (
            total_bars - t2
        ) <= 12:

            if (
                closes[t2] < closes[t1]
                and
                rsis[t2] > rsis[t1]
                and
                rsis[t2] < 42
            ):

                signals.append("BULL")

    return signals

# ============================================================
# ОДИН ЦИКЛ СКАНИРОВАНИЯ
# ============================================================

def run_bot_iteration():

    top_coins = get_top_30_crypto_symbols()

    if not top_coins:

        print(
            "⚠ Список монет пуст."
        )

        return

    timeframes = [
        "4h",
        "1h",
        "30m"
    ]

    print()

    print(
        "🔍 Сканирование рынка | "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(
        f"📊 Монет: TOP-{TOP_COINS}"
    )

    print(
        f"⏱ Таймфреймы: "
        f"{', '.join(timeframes)}"
    )

    print()

    # ========================================================
    # СКАНИРОВАНИЕ
    # ========================================================

    for tf in timeframes:

        print(
            f"⏱ Сканирование {tf}..."
        )

        for coin in top_coins:

            try:

                # Получаем свечи
                ohlcv = exchange.fetch_ohlcv(
                    coin,
                    timeframe=tf,
                    limit=OHLCV_LIMIT
                )

                if len(ohlcv) < 100:

                    continue

                # Создаём DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=[
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume"
                    ]
                )

                # RSI
                df["rsi"] = calculate_rsi(
                    df["close"],
                    period=14
                )

                # Проверяем дивергенции
                signals = check_divergence(
                    df
                )

                # Последняя полностью
                # закрытая свеча
                last_row = df.iloc[-2]

                candle_time = int(
                    last_row["timestamp"]
                )

                # ====================================================
                # ЕСЛИ НАЙДЕН СИГНАЛ
                # ====================================================

                if signals:

                    signal_str = ", ".join(
                        signals
                    )

                    signal_key = (
                        f"{coin}_"
                        f"{tf}_"
                        f"{candle_time}_"
                        f"{signal_str}"
                    )

                    storage_key = (
                        f"{coin}_{tf}"
                    )

                    # Проверяем, не отправляли
                    # ли этот сигнал раньше
                    if (
                        last_signals.get(
                            storage_key
                        )
                        != signal_key
                    ):

                        # Сохраняем сигнал
                        last_signals[
                            storage_key
                        ] = signal_key

                        save_sent_signals()

                        # ==================================================
                        # TELEGRAM СООБЩЕНИЕ
                        # ==================================================

                        msg = (
                            "🚨 "
                            "<b>RSI DIVERGENCE</b>\n\n"

                            f"Монета: "
                            f"<b>{coin}</b>\n"

                            f"Таймфрейм: "
                            f"<b>{tf}</b>\n"

                            f"Сигнал: "
                            f"<b>{signal_str}</b>\n"

                            f"Цена: "
                            f"<b>{last_row['close']:.6f}</b>\n"

                            f"RSI: "
                            f"<b>{last_row['rsi']:.2f}</b>"
                        )

                        print()

                        print(
                            "🔥 НОВЫЙ СИГНАЛ | "
                            f"{coin} | "
                            f"{tf} | "
                            f"{signal_str}"
                        )

                        send_telegram_message(
                            msg
                        )

                # Небольшая пауза
                time.sleep(0.2)

            except Exception as e:

                print(
                    f"❌ Ошибка для "
                    f"{coin} ({tf}): {e}"
                )

                time.sleep(0.5)

        print(
            "-" * 50
        )

# ============================================================
# ЗАПУСК 24/7
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "🤖 RSI DIVERGENCE BOT"
    )

    print(
        "🟢 Режим: 24/7"
    )

    print(
        "⏱ Проверка: каждые 5 минут"
    )

    print(
        f"📊 Монеты: TOP-{TOP_COINS} MEXC"
    )

    print(
        "📈 Таймфреймы: 4h / 1h / 30m"
    )

    print(
        "🔕 Повторные сигналы: ЗАБЛОКИРОВАНЫ"
    )

    print(
        "=" * 60
    )

    while True:

        try:

            start_time = time.time()

            run_bot_iteration()

            elapsed = (
                time.time()
                - start_time
            )

            print()

            print(
                f"✅ Проверка завершена "
                f"за {elapsed:.1f} сек."
            )

            print(
                "💤 Следующая проверка "
                "через 5 минут..."
            )

            print()

        except KeyboardInterrupt:

            print()

            print(
                "🛑 Бот остановлен "
                "пользователем."
            )

            break

        except Exception as e:

            print()

            print(
                f"❌ Критическая ошибка: {e}"
            )

            print(
                "🔄 Бот продолжит работу..."
            )

        time.sleep(
            CHECK_INTERVAL
        )
