import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time


def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        # print('Fetched', len(ohlcv), symbol, 'candles from',
        # exchange.iso8601 (ohlcv[0][0]), 'to', exchange.iso8601
        # (ohlcv[-1][0]))
        return ohlcv
    except Exception:
        if num_retries > max_retries:
            # Exception('Failed to fetch', timeframe, symbol,
            # 'OHLCV in', max_retries, 'attempts')
            raise


def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    earliest_timestamp = exchange.milliseconds()
    timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    timedelta = limit * timeframe_duration_in_ms
    all_ohlcv = []
    while True:
        fetch_since = earliest_timestamp - timedelta
        ohlcv = retry_fetch_ohlcv(
            exchange, max_retries, symbol, timeframe, fetch_since, limit
        )
        # if we have reached the beginning of history
        if ohlcv[0][0] >= earliest_timestamp:
            break
        earliest_timestamp = ohlcv[0][0]
        all_ohlcv = ohlcv + all_ohlcv
        # print(len(all_ohlcv), 'candles in total from',
        # exchange.iso8601(all_ohlcv[0][0]), 'to',
        # exchange.iso8601(all_ohlcv[-1][0]))
        # if we have reached the checkpoint
        if fetch_since < since:
            break
    return exchange.filter_by_since_limit(all_ohlcv, since, None, key=0)


def scrape_candles_to_csv(
    filename, exchange_id, max_retries, symbol, timeframe, since, limit
):
    # instantiate the exchange by id
    exchange = getattr(ccxt, exchange_id)(
        {
            "enableRateLimit": True,
        }
    )
    # convert since from string to milliseconds integer if needed
    if isinstance(since, str):
        since = exchange.parse8601(since)
    # preload all markets from the exchange
    exchange.load_markets()
    # fetch all candles
    ohlcv = scrape_ohlcv(
        exchange, max_retries, symbol, timeframe, since, limit
    )
    # Creates Dataframe and save it to csv file
    read_file = f"{filename}"
    pd.DataFrame(ohlcv).to_csv(read_file)
    # print('Saved', len(ohlcv), 'candles from',
    # exchange.iso8601(ohlcv[0][0]), 'to',
    # exchange.iso8601(ohlcv[-1][0]), 'to', filename)


def fetch_cryptodata_from_exchange(
    loc_folder,
    exchange="binance",
    cryptos=["BTC/USDT"],
    sample_freq="1d",
    since_hours=48,
    page_limit=1000,
):
    # datetime_now = datetime.now().strftime("%Y-%m-%d")
    since = (
        datetime.today() - timedelta(hours=since_hours) - timedelta(hours=3)
    ).strftime("%Y-%m-%dT%H:%M:%S")
    print("Begin download...")

    for market_symbol in cryptos:
        scrape_candles_to_csv(
            filename="test.csv",
            exchange_id=exchange,
            max_retries=3,
            symbol=market_symbol,
            timeframe=sample_freq,
            since=since,
            limit=page_limit,
        )
        time.sleep(2)
        filename = "test.csv"
        df = pd.read_csv(filename)
        if market_symbol == cryptos[0]:

            df.drop(df.columns[[0]], axis=1, inplace=True)
            df["0"] = pd.to_datetime(df["0"], unit="ms")
            df.rename(
                columns={
                    "0": "Datetime",
                    "1": "Open",
                    "2": "High",
                    "3": "Low",
                    "4": "Close",
                    "5": "Volume",
                },
                inplace=True,
            )
            df = df.set_index("Datetime")
            dfx = df.copy()

        else:

            df.drop(df.columns[[0]], axis=1, inplace=True)
            df["0"] = pd.to_datetime(df["0"], unit="ms")
            df.rename(
                columns={
                    "0": "Datetime",
                    "1": "Open",
                    "2": "High",
                    "3": "Low",
                    "4": "Close",
                    "5": "Volume",
                },
                inplace=True,
            )
            df = df.set_index("Datetime")
            dfx = pd.merge(dfx, df, on=["Datetime"])

    dfx = dfx.loc[:, ~dfx.columns.duplicated()]
    dfx = dfx[~dfx.index.duplicated(keep="first")]
    # crypto = market_symbol.replace("/", "")

    print("Finished \n")
    # write_file = f'{loc_folder}/{crypto}-{sample_freq}-Alarm Price Data.csv'
    return dfx
