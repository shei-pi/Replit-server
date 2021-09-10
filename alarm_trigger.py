from os import system
system('pip install --upgrade pip')
from keep_alive import keep_alive
from datetime import datetime, timedelta
import time
import ccxt
import pandas as pd
import smtplib
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel, KeltnerChannel, UlcerIndex
from ta.momentum import RSIIndicator, StochRSIIndicator, TSIIndicator, UltimateOscillator, StochasticOscillator, \
    KAMAIndicator, ROCIndicator, AwesomeOscillatorIndicator, WilliamsRIndicator, PercentagePriceOscillator, \
    PercentageVolumeOscillator
from ta.trend import MACD, ADXIndicator, AroonIndicator, CCIIndicator, DPOIndicator, WMAIndicator, IchimokuIndicator, \
    KSTIndicator, MassIndex, STCIndicator, TRIXIndicator, VortexIndicator
from ta.volume import AccDistIndexIndicator, ChaikinMoneyFlowIndicator, EaseOfMovementIndicator, ForceIndexIndicator, \
    MFIIndicator, OnBalanceVolumeIndicator, VolumePriceTrendIndicator


def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        # print('Fetched', len(ohlcv), symbol, 'candles from', exchange.iso8601 (ohlcv[0][0]), 'to', exchange.iso8601 (ohlcv[-1][0]))
        return ohlcv
    except Exception:
        if num_retries > max_retries:
            raise  # Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')


def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    earliest_timestamp = exchange.milliseconds()
    timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    timedelta = limit * timeframe_duration_in_ms
    all_ohlcv = []
    while True:
        fetch_since = earliest_timestamp - timedelta
        ohlcv = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe,
                                  fetch_since, limit)
        # if we have reached the beginning of history
        if ohlcv[0][0] >= earliest_timestamp:
            break
        earliest_timestamp = ohlcv[0][0]
        all_ohlcv = ohlcv + all_ohlcv
        # print(len(all_ohlcv), 'candles in total from', exchange.iso8601(all_ohlcv[0][0]), 'to', exchange.iso8601(all_ohlcv[-1][0]))
        # if we have reached the checkpoint
        if fetch_since < since:
            break
    return exchange.filter_by_since_limit(all_ohlcv, since, None, key=0)


def scrape_candles_to_csv(filename, exchange_id, max_retries, symbol,
                          timeframe, since, limit):
    # instantiate the exchange by id
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,
    })
    # convert since from string to milliseconds integer if needed
    if isinstance(since, str):
        since = exchange.parse8601(since)
    # preload all markets from the exchange
    exchange.load_markets()
    # fetch all candles
    ohlcv = scrape_ohlcv(exchange, max_retries, symbol, timeframe, since,
                         limit)
    # Creates Dataframe and save it to csv file
    read_file = f'{filename}'
    pd.DataFrame(ohlcv).to_csv(read_file)
    # print('Saved', len(ohlcv), 'candles from', exchange.iso8601(ohlcv[0][0]), 'to', exchange.iso8601(ohlcv[-1][0]), 'to', filename)


def fetch_data(loc_folder,
               exchange='binance',
               cryptos=['BTC/USDT'],
               sample_freq='1d',
               since_hours=48,
               page_limit=1000):
    datetime_now = datetime.now().strftime('%Y-%m-%d')
    since = (datetime.today() - timedelta(hours=since_hours) -
             timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%S')
    print('Begin download...')

    for market_symbol in cryptos:
        scrape_candles_to_csv(filename='test.csv',
                              exchange_id=exchange,
                              max_retries=3,
                              symbol=market_symbol,
                              timeframe=sample_freq,
                              since=since,
                              limit=page_limit)
        time.sleep(2)
        filename = 'test.csv'
        df = pd.read_csv(filename)
        if market_symbol == cryptos[0]:

            df.drop(df.columns[[0]], axis=1, inplace=True)
            df['0'] = pd.to_datetime(df['0'], unit='ms')
            df.rename(columns={
                '0': 'Datetime',
                '1': 'Open',
                '2': 'High',
                '3': 'Low',
                '4': 'Close',
                '5': 'Volume'
            },
                inplace=True)
            df = df.set_index('Datetime')
            dfx = df.copy()

        else:

            df.drop(df.columns[[0]], axis=1, inplace=True)
            df['0'] = pd.to_datetime(df['0'], unit='ms')
            df.rename(columns={
                '0': 'Datetime',
                '1': 'Open',
                '2': 'High',
                '3': 'Low',
                '4': 'Close',
                '5': 'Volume'
            },
                inplace=True)
            df = df.set_index('Datetime')
            dfx = pd.merge(dfx, df, on=['Datetime'])

    dfx = dfx.loc[:, ~dfx.columns.duplicated()]
    dfx = dfx[~dfx.index.duplicated(keep='first')]
    crypto = market_symbol.replace('/', '')

    print(f'Finished \n')
    # write_file = f'{loc_folder}/{crypto}-{sample_freq}-Alarm Price Data.csv'
    return dfx


def import_csv(loc_folder, filename):
    read_file = f'{loc_folder}/{filename}'
    df = pd.read_csv(read_file, index_col='Datetime', parse_dates=True)
    return df


def get_psar(df, iaf=0.02, maxaf=0.2):
    length = len(df)
    high = df['High']
    low = df['Low']
    df['PSAR'] = df['Close'].copy()
    bull = True
    af = iaf
    hp = high.iloc[0]
    lp = low.iloc[0]

    for i in range(2, length):
        if bull:
            df.PSAR.iloc[i] = df.PSAR.iloc[i -
                                           1] + af * (hp - df.PSAR.iloc[i - 1])
        else:
            df.PSAR.iloc[i] = df.PSAR.iloc[i -
                                           1] + af * (lp - df.PSAR.iloc[i - 1])

        reverse = False

        if bull:
            if low.iloc[i] < df.PSAR.iloc[i]:
                bull = False
                reverse = True
                df.PSAR.iloc[i] = hp
                lp = low.iloc[i]
                af = iaf
        else:
            if high.iloc[i] > df.PSAR.iloc[i]:
                bull = True
                reverse = True
                df.PSAR.iloc[i] = lp
                hp = high.iloc[i]
                af = iaf

        if not reverse:
            if bull:
                if high.iloc[i] > hp:
                    hp = high.iloc[i]
                    af = min(af + iaf, maxaf)
                if low.iloc[i - 1] < df.PSAR.iloc[i]:
                    df.PSAR.iloc[i] = low[i - 1]
                if low.iloc[i - 2] < df.PSAR.iloc[i]:
                    df.PSAR.iloc[i] = low.iloc[i - 2]
            else:
                if low.iloc[i] < lp:
                    lp = low.iloc[i]
                    af = min(af + iaf, maxaf)
                if high.iloc[i - 1] > df.PSAR.iloc[i]:
                    df.PSAR.iloc[i] = high.iloc[i - 1]
                if high.iloc[i - 2] > df.PSAR.iloc[i]:
                    df.PSAR.iloc[i] = high.iloc[i - 2]
    return df.PSAR


def rsi_s(df, args):  # rsi_w
    rsi = RSIIndicator(close=df['Close'], window=args[0])
    df['RSI'] = rsi.rsi()
    df.loc[(df.RSI > args[2]), 'RSI_Z'] = 1
    df.loc[(df.RSI < args[1]), 'RSI_Z'] = -1
    df.drop(columns=['RSI'], inplace=True)
    return df


def tsi_s(df, args):  # tsi_fw, tsi_fw, tsi_sig, tsi_s
    tsi = TSIIndicator(close=df['Close'],
                       window_slow=args[0],
                       window_fast=args[1])
    df['TSI'] = tsi.tsi()
    df['TSI_S'] = df['TSI'].ewm(args[2],
                                min_periods=0,
                                adjust=False,
                                ignore_na=False).mean()
    df.loc[(df.TSI > args[3]), 'TSI_Z'] = 1
    df.loc[(df.TSI < -args[3]), 'TSI_Z'] = -1
    df.loc[(df.TSI < 0), 'TSI_T'] = 1
    df.loc[(df.TSI >= 0), 'TSI_T'] = -1
    df.loc[(df.TSI.shift(1) > df.TSI_S.shift(1)) & (df.TSI < df.TSI_S),
           'TSI_A'] = 1
    df.loc[(df.TSI.shift(1) < df.TSI_S.shift(1)) & (df.TSI > df.TSI_S),
           'TSI_A'] = -1
    df.drop(columns=['TSI', 'TSI_S'], inplace=True)
    return df


def kst_s(
        df, args
):  # kst_1, kst_2, kst_3, kst_4, kst_ns, kst_r1, kst_r2, kst_r3, kst_r4, kst_b, kst_s
    kst = KSTIndicator(close=df['Close'],
                       roc1=args[5],
                       roc2=args[6],
                       roc3=args[7],
                       roc4=args[8],
                       window1=args[0],
                       window2=args[1],
                       window3=args[2],
                       window4=args[3],
                       nsig=args[4])
    df['KST'] = kst.kst()
    df['KST_S'] = kst.kst_sig()
    df['KST_H'] = kst.kst_diff()
    df.loc[df.KST_H >= 0, 'KST_T'] = -1
    df.loc[df.KST_H < 0, 'KST_T'] = 1
    df.loc[(df.KST.shift(1) > df.KST_S.shift(1)) & (df.KST < df.KST_S) &
           (df.KST.shift(1) > args[10]), 'KST_A'] = 1
    df.loc[(df.KST.shift(1) < df.KST_S.shift(1)) & (df.KST > df.KST_S) &
           (df.KST.shift(1) < args[9]), 'KST_A'] = -1
    df.drop(columns=['KST', 'KST_S', 'KST_H'], inplace=True)
    return df


def stc_s(df, args):  # stc_fw, stc_sw, stc_c, stc_s1, stc_s2, stc_hl, stc_ll
    stc = STCIndicator(close=df['Close'],
                       window_fast=args[0],
                       window_slow=args[1],
                       cycle=args[2],
                       smooth1=args[3],
                       smooth2=args[4])
    df['STC'] = stc.stc()
    df.loc[df.STC < args[6], 'STC_T'] = 1
    df.loc[df.STC > args[5], 'STC_T'] = -1
    df.loc[(df.STC.shift(1) > args[5]) & (df.STC < args[5]), 'STC_A'] = 1
    df.loc[(df.STC.shift(1) < args[6]) & (df.STC > args[6]), 'STC_A'] = -1
    df.drop(columns=['STC'], inplace=True)
    return df


def cmf_s(df, args):  # cmf_w, cmf_l
    cmf = ChaikinMoneyFlowIndicator(high=df['High'],
                                    low=df['Low'],
                                    close=df['Close'],
                                    volume=df['Volume'],
                                    window=args[0])
    df['CMF'] = cmf.chaikin_money_flow()
    df.loc[(df.CMF > args[1]), 'CMF_Z'] = 1
    df.loc[(df.CMF < -args[1]), 'CMF_Z'] = -1
    df.loc[(df.CMF.shift(1) > 0) & (df.CMF < 0), 'CMF_A'] = 1
    df.loc[(df.CMF.shift(1) < 0) & (df.CMF > 0), 'CMF_A'] = -1
    df.drop(columns=['CMF'], inplace=True)
    return df


def fi_s(df, args):  # fi_w, std_m
    fi = ForceIndexIndicator(close=df['Close'],
                             volume=df['Volume'],
                             window=args[0])
    df['FI'] = fi.force_index()
    fi_s = int(df[df['FI'] > 0]['FI'].mean() +
               3.5 * df[df['FI'] > 0]['FI'].mean() * args[1])
    fi_b = int(df[df['FI'] < 0]['FI'].mean() +
               2.5 * df[df['FI'] < 0]['FI'].mean() * args[1])
    df.loc[df.FI > fi_s, 'FI_Z'] = 1
    df.loc[df.FI < fi_b, 'FI_Z'] = -1
    df.loc[(df.FI.shift(1) > fi_s) & (df.FI < fi_s), 'FI_A'] = 1
    df.loc[(df.FI.shift(1) < fi_b) & (df.FI > fi_b), 'FI_A'] = -1
    df.drop(columns=['FI'], inplace=True)
    return df


def mfi_s(df, args):  # mfi_w, mfi_b, mfi_s
    mfi = MFIIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       volume=df['Volume'],
                       window=args[0])
    df['MFI'] = mfi.money_flow_index()
    df.loc[(df.MFI > args[2]), 'MFI_Z'] = 1
    df.loc[(df.MFI < args[1]), 'MFI_Z'] = -1
    df.drop(columns=['MFI'], inplace=True)
    return df


def uo_s(df, args):  # uo_1, uo_2, uo_3, uo_b, uo_s
    uo = UltimateOscillator(high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            window1=args[0],
                            window2=args[1],
                            window3=args[2])
    df['UO'] = uo.ultimate_oscillator()
    df.loc[(df.UO > args[4]), 'UO_Z'] = 1
    df.loc[(df.UO < args[3]), 'UO_Z'] = -1
    df.drop(columns=['UO'], inplace=True)
    return df


def so_s(df, args):  # so_w	so_sw	so_b so_s
    so = StochasticOscillator(high=df['High'],
                              low=df['Low'],
                              close=df['Close'],
                              window=args[0],
                              smooth_window=args[1])
    df['SO'] = so.stoch()
    df['SOS'] = so.stoch_signal()
    df.loc[(df.SO > args[3]), 'SO_Z'] = 1
    df.loc[(df.SO < args[2]), 'SO_Z'] = -1
    df.loc[(df.SO.shift(1) > df.SOS.shift(1)) & (df.SO < df.SOS), 'SO_A'] = 1
    df.loc[(df.SO.shift(1) < df.SOS.shift(1)) & (df.SO > df.SOS), 'SO_A'] = -1
    df.drop(columns=['SO', 'SOS'], inplace=True)
    return df


def ki_s(df, args):  # ki_w, ki_p1, ki_p2, ki_p3, ki_b, ki_s
    ki = KAMAIndicator(close=df['Close'],
                       window=args[0],
                       pow1=args[1],
                       pow2=args[2])
    ki_sig = KAMAIndicator(close=df['Close'],
                           window=args[0],
                           pow1=args[3],
                           pow2=args[2])
    df['KI'] = ki.kama()
    df['KIS'] = ki_sig.kama()
    df.loc[(df.Close > df.KI * (1 + args[5])), 'KI_Z'] = 1
    df.loc[(df.Close < df.KI * (1 - args[4])), 'KI_Z'] = -1
    df.loc[(df.Close.shift(1) > df.KI.shift(1)) & (df.Close < df.KI *
                                                   (1 - args[4])), 'KI_T'] = 1
    df.loc[(df.Close.shift(1) < df.KI.shift(1)) & (df.Close > df.KI *
                                                   (1 + args[5])), 'KI_T'] = -1
    df['KI_T'] = df['KI_T'].fillna(method='ffill')
    df.loc[(df.KI > df.KIS.shift(1)) & (df.KI < df.KIS), 'KI_A'] = 1
    df.loc[(df.KI < df.KIS.shift(1)) & (df.KI > df.KIS), 'KI_A'] = -1
    df.drop(columns=['KI', 'KIS'], inplace=True)
    return df


def roc_s(df, args):  # roc_w, roc_b, roc_s
    roc = ROCIndicator(close=df['Close'], window=args[0])
    df['ROC'] = roc.roc()
    df.loc[(df.ROC > args[2]), 'ROC_Z'] = 1
    df.loc[(df.ROC < args[1]), 'ROC_Z'] = -1
    df.loc[(df.ROC < 0), 'ROC_T'] = 1
    df.loc[(df.ROC >= 0), 'ROC_T'] = -1
    df.drop(columns=['ROC'], inplace=True)
    return df


def ao_s(df, args):  # ao_1, ao_2
    ao = AwesomeOscillatorIndicator(high=df['High'],
                                    low=df['Low'],
                                    window1=args[0],
                                    window2=args[1])
    df['AO'] = ao.awesome_oscillator()
    df.loc[(df.AO < 0), 'AO_T'] = 1
    df.loc[(df.AO >= 0), 'AO_T'] = -1
    df.loc[((df.AO.shift(1) > 0) & (df.AO < 0)) |
           ((df.AO < 0) & (df.AO.shift(2) < df.AO.shift(1)) &
            (df.AO.shift(1) > df.AO)), 'AO_A'] = 1
    df.loc[((df.AO.shift(1) < 0) & (df.AO > 0)) |
           ((df.AO > 0) & (df.AO.shift(2) > df.AO.shift(1)) &
            (df.AO.shift(1) < df.AO)), 'AO_A'] = -1
    df.drop(columns=['AO'], inplace=True)
    return df


def wi_s(df, args):  # wi_w	wi_b	wi_s
    wi = WilliamsRIndicator(high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            lbp=args[0])
    df['WI'] = wi.williams_r()
    df.loc[(df.WI > args[2]), 'WI_Z'] = 1
    df.loc[(df.WI < args[1]), 'WI_Z'] = -1
    df.drop(columns=['WI'], inplace=True)
    return df


def srsi_s(df, args):  # srsi_w srsi_kw	srsi_dw	srsi_b	srsi_s
    srsi = StochRSIIndicator(close=df['Close'],
                             window=args[0],
                             smooth1=args[1],
                             smooth2=args[2])
    df['SRSI'] = srsi.stochrsi()
    df['SRSI_K'] = srsi.stochrsi_k()
    df['SRSI_D'] = srsi.stochrsi_d()
    df.loc[(df.SRSI > args[4]), 'SRSI_Z'] = 1
    df.loc[(df.SRSI < args[3]), 'SRSI_Z'] = -1
    df.loc[(df.SRSI_K.shift(1) > df.SRSI_D.shift(1)) & (df.SRSI_K < df.SRSI_D),
           'SRSI_A'] = 1
    df.loc[(df.SRSI_K.shift(1) < df.SRSI_D.shift(1)) & (df.SRSI_K > df.SRSI_D),
           'SRSI_A'] = -1
    df.drop(columns=['SRSI_K', 'SRSI_D', 'SRSI'], inplace=True)
    return df


def po_s(df, args):  # po_sw	po_fw	po_sg
    po = PercentagePriceOscillator(close=df['Close'],
                                   window_slow=args[0],
                                   window_fast=args[1],
                                   window_sign=args[2])
    df['PO'] = po.ppo()
    df['PO_S'] = po.ppo_signal()
    df['PO_H'] = po.ppo_hist()
    df.loc[(df.PO_H < 0), 'PO_T'] = 1
    df.loc[(df.PO_H >= 0), 'PO_T'] = -1
    df.loc[(df.PO.shift(1) > df.PO_S.shift(1)) & (df.PO < df.PO_S), 'PO_A'] = 1
    df.loc[(df.PO.shift(1) < df.PO_S.shift(1)) & (df.PO > df.PO_S),
           'PO_A'] = -1
    df.drop(columns=['PO', 'PO_S', 'PO_H'], inplace=True)
    return df


def pvo_s(df, args):  # pvo_sw	pvo_fw	pvo_sg
    pvo = PercentageVolumeOscillator(volume=df['Volume'],
                                     window_slow=args[0],
                                     window_fast=args[1],
                                     window_sign=args[2])
    df['PVO'] = pvo.pvo()
    df['PVO_S'] = pvo.pvo_signal()
    df['PVO_H'] = pvo.pvo_hist()
    df.loc[(df.PVO_H < 0), 'PVO_T'] = 1
    df.loc[(df.PVO_H >= 0), 'PVO_T'] = -1
    df.loc[(df.PVO.shift(1) > df.PVO_S.shift(1)) & (df.PVO < df.PVO_S),
           'PVO_A'] = 1
    df.loc[(df.PVO.shift(1) < df.PVO_S.shift(1)) & (df.PVO > df.PVO_S),
           'PVO_A'] = -1
    df.drop(columns=['PVO', 'PVO_S', 'PVO_H'], inplace=True)
    return df


def atr_s(df, args):  # atr_w atr_l rsi_w rsi_b rsi_s
    atr = AverageTrueRange(high=df['High'],
                           low=df['Low'],
                           close=df['Close'],
                           window=args[0])
    rsi = RSIIndicator(close=df['Close'], window=args[2])
    df['RSI'] = rsi.rsi()
    df['ATR'] = atr.average_true_range()
    df.loc[((df.ATR.shift(1) * (args[0] - 1) + df.ATR) / args[0] > args[1]) &
           (df.RSI > args[4]), 'ATR_Z'] = 1
    df.loc[((df.ATR.shift(1) * (args[0] - 1) + df.ATR) / args[0] > args[1]) &
           (df.RSI < args[3]), 'ATR_Z'] = -1
    df.drop(columns=['ATR', 'RSI'], inplace=True)
    return df


def bb_s(df, args):  # bb_w	bb_d bb_l
    bb = BollingerBands(close=df['Close'], window=args[0], window_dev=args[1])
    adx = ADXIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0])
    df['BB_H'] = bb.bollinger_hband_indicator()
    df['BB_L'] = bb.bollinger_lband_indicator()
    df['BB_W'] = bb.bollinger_wband()
    df['ADX_P'] = adx.adx_pos()
    df['ADX_N'] = adx.adx_neg()
    df.loc[(df.ADX_P < df.ADX_N), 'ADX_T'] = 1
    df.loc[(df.ADX_P > df.ADX_N), 'ADX_T'] = -1
    df.loc[(df.BB_W > args[2]), 'BB_Z'] = 1
    df.loc[(df.BB_W > args[2]), 'BB_Z'] = -1
    df.loc[(df.Close >= df.BB_H) & (df.ADX_T > 0), 'BB_T'] = 1
    df.loc[(df.Close <= df.BB_L) & (df.ADX_T < 0), 'BB_T'] = -1
    df['BB_T'] = df['BB_T'].fillna(method='ffill')
    df.loc[(df.Close >= df.BB_H), 'BB_A'] = 1
    df.loc[(df.Close <= df.BB_L), 'BB_A'] = -1
    df.drop(columns=['BB_H', 'BB_L', 'BB_W', 'ADX', 'ADX_P', 'ADX_N', 'ADX_T'],
            inplace=True)
    return df


def kc_s(df, args):  # kc_w	kc_a
    kc = KeltnerChannel(high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        window=args[0],
                        window_atr=args[1])
    df['KC_H'] = kc.keltner_channel_hband_indicator()
    df['KC_L'] = kc.keltner_channel_lband_indicator()
    df.loc[df.KC_L > 0, 'KC_T'] = 1
    df.loc[df.KC_H > 0, 'KC_T'] = -1
    df['KC_T'] = df['KC_T'].fillna(method='ffill')
    df.drop(columns=['KC_H', 'KC_L'], inplace=True)
    return df


def dc_s(df, args):  # dc_w	dc_l
    dc = DonchianChannel(high=df['High'],
                         low=df['Low'],
                         close=df['Close'],
                         window=args[0])
    adx = ADXIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0])
    df['DC_H'] = dc.donchian_channel_hband()
    df['DC_L'] = dc.donchian_channel_lband()
    df['DC_W'] = dc.donchian_channel_wband()
    df['ADX_P'] = adx.adx_pos()
    df['ADX_N'] = adx.adx_neg()
    df.loc[(df.ADX_P < df.ADX_N), 'ADX_T'] = 1
    df.loc[(df.ADX_P > df.ADX_N), 'ADX_T'] = -1
    df.loc[(df.DC_W.shift(1) < args[1]), 'DC_Z'] = 1
    df.loc[(df.DC_W.shift(1) > args[1]), 'DC_Z'] = -1
    df.loc[(df.High >= df.DC_H) & (df.ADX_T > 0), 'DC_T'] = 1
    df.loc[(df.Low <= df.DC_L) & (df.ADX_T < 0), 'DC_T'] = -1
    df['DC_T'] = df['DC_T'].fillna(method='ffill')
    df.loc[(df.Close >= df.DC_H), 'DC_A'] = 1
    df.loc[(df.Close <= df.DC_L), 'DC_A'] = -1
    df.drop(columns=['DC_H', 'DC_L', 'DC_W', 'ADX', 'ADX_P', 'ADX_N', 'ADX_T'],
            inplace=True)
    return df


def ui_s(df, args):  # ui_w	ui_b ui_s
    ui = UlcerIndex(close=df['Close'], window=args[0])
    df['UI'] = ui.ulcer_index()
    df.loc[(df.UI < args[2]), 'UI_Z'] = 1
    df.loc[(df.UI > args[1]), 'UI_Z'] = -1
    df.drop(columns=['UI'], inplace=True)
    return df


def macd_s(df, args):  # macd_fw	macd_sw	macd_sg
    macd = MACD(close=df['Close'],
                window_fast=args[0],
                window_slow=args[1],
                window_sign=args[2])
    df['MACD'] = macd.macd()
    df['MACD_S'] = macd.macd_signal()
    df['MACD_H'] = macd.macd_diff()
    df.loc[(df.MACD_H < 0), 'MACD_T'] = 1
    df.loc[(df.MACD_H >= 0), 'MACD_T'] = -1
    df.loc[(df.MACD.shift(1) > df.MACD_S.shift(1)) & (df.MACD < df.MACD_S),
           'MACD_A'] = 1
    df.loc[(df.MACD.shift(1) < df.MACD_S.shift(1)) & (df.MACD > df.MACD_S),
           'MACD_A'] = -1
    df.drop(columns=['MACD', 'MACD_S', 'MACD_H'], inplace=True)
    return df


def adx_s(df, args):  # adx_w	adx_l
    adx = ADXIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0])
    df['ADX'] = adx.adx()
    df['ADX_P'] = adx.adx_pos()
    df['ADX_N'] = adx.adx_neg()
    df.loc[(df.ADX_P < df.ADX_N), 'ADX_T'] = 1
    df.loc[(df.ADX_P > df.ADX_N), 'ADX_T'] = -1
    df.loc[(df.ADX_T > 0) & df.ADX > args[1], 'ADX_Z'] = 1
    df.loc[(df.ADX_T < 0) & df.ADX > args[1], 'ADX_Z'] = -1
    df.drop(columns=['ADX', 'ADX_P', 'ADX_N'], inplace=True)
    return df


def ai_s(df, args):  # ai_w
    ai = AroonIndicator(close=df['Close'], window=args[0])
    df['AI_U'] = ai.aroon_up()
    df['AI_D'] = ai.aroon_down()
    df.loc[(df.AI_U.shift(1) > df.AI_D.shift(1)) & (df.AI_U < df.AI_D),
           'AI_A'] = 1
    df.loc[(df.AI_U.shift(1) < df.AI_D.shift(1)) & (df.AI_U > df.AI_D),
           'AI_A'] = -1
    df.drop(columns=['AI_U', 'AI_D'], inplace=True)
    return df


def cci_s(df, args):  # cci_w cci_b cci_s
    cci = CCIIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0],
                       constant=0.015)
    df['CCI'] = cci.cci()
    df.loc[df.CCI > args[2], 'CCI_Z'] = 1
    df.loc[df.CCI < args[1], 'CCI_Z'] = -1
    df.loc[(df.CCI.shift(1) > args[2]) & (df.CCI < args[2]), 'CCI_A'] = 1
    df.loc[(df.CCI.shift(1) < args[1]) & (df.CCI > args[1]), 'CCI_A'] = -1
    df.drop(columns=['CCI'], inplace=True)
    return df


def dpo_s(df, args):  # dpo_w, dpo_s
    dpo = DPOIndicator(close=df['Close'], window=args[0])
    df['DPO'] = dpo.dpo()
    df.loc[(df.DPO > args[1]), 'DPO_Z'] = 1
    df.loc[(df.DPO < -args[1]), 'DPO_Z'] = -1
    df.drop(columns=['DPO'], inplace=True)
    return df


def mi_s(df, args):  # mi_fw	mi_sw std_m
    mi = MassIndex(high=df['High'],
                   low=df['Low'],
                   window_fast=args[0],
                   window_slow=args[1])
    df['MI'] = mi.mass_index()
    mi_s = df.MI.mean() + df.MI.std() * args[2]
    df.loc[df.MI > mi_s, 'MI_Z'] = 1
    df.loc[df.MI < mi_s, 'MI_Z'] = -1
    df.drop(columns=['MI'], inplace=True)
    return df


def ii_s(df, args):  # ii_1 ii_2 ii_3
    ii = IchimokuIndicator(high=df['High'],
                           low=df['Low'],
                           window1=args[0],
                           window2=args[1],
                           window3=args[2])
    df['II'] = ii.ichimoku_conversion_line()
    df['II_S'] = ii.ichimoku_base_line()
    df['II_LA'] = ii.ichimoku_a()
    df['II_LB'] = ii.ichimoku_b()
    df.loc[(df.II > df.II_S), 'II_Z'] = 1
    df.loc[(df.II < df.II_S), 'II_Z'] = -1
    df.loc[(df.Close < df.II_S), 'II_T'] = 1
    df.loc[(df.Close >= df.II_S), 'II_T'] = -1
    df.loc[(df.II.shift(1) > df.II_S.shift(1)) & (df.II < df.II_S) &
           (df.High > df.II_LA), 'II_A'] = 1
    df.loc[(df.II.shift(1) < df.II_S.shift(1)) & (df.II > df.II_S) &
           (df.Low < df.II_LB), 'II_A'] = -1
    df.drop(columns=['II', 'II_S', 'II_LA', 'II_LB'], inplace=True)
    return df


def trix_s(df, args):  # trix_w	trix_sw
    trix = TRIXIndicator(close=df['Close'], window=args[0])
    df['TRIX'] = trix.trix()
    df['TRIX_S'] = df['TRIX'].ewm(span=args[1], adjust=False).mean()
    df.loc[df.TRIX < 0, 'TRIX_T'] = 1
    df.loc[df.TRIX >= 0, 'TRIX_T'] = -1
    df.loc[(df.TRIX.shift(1) > df.TRIX_S.shift(1)) & (df.TRIX < df.TRIX_S),
           'TRIX_A'] = 1
    df.loc[(df.TRIX.shift(1) < df.TRIX_S.shift(1)) & (df.TRIX > df.TRIX_S),
           'TRIX_A'] = -1
    df.drop(columns=['TRIX', 'TRIX_S'], inplace=True)
    return df


def vi_s(df, args):  # vi_w
    vi = VortexIndicator(high=df['High'],
                         low=df['Low'],
                         close=df['Close'],
                         window=args[0])
    df['VI_P'] = vi.vortex_indicator_pos()
    df['VI_N'] = vi.vortex_indicator_neg()
    df['VI_H'] = vi.vortex_indicator_diff()
    df.loc[(df.VI_P < df.VI_N), 'VI_T'] = 1
    df.loc[(df.VI_P > df.VI_N), 'VI_T'] = -1
    df.loc[(df.VI_P.shift(1) > df.VI_N.shift(1)) & (df.VI_P < df.VI_N),
           'VI_A'] = 1
    df.loc[(df.VI_P.shift(1) < df.VI_N.shift(1)) & (df.VI_P > df.VI_N),
           'VI_A'] = -1
    df.drop(columns=['VI_H', 'VI_P', 'VI_N'], inplace=True)
    return df


def psar_s(df, args):  # psar_ms	psar_st
    df['PSAR'] = get_psar(df, args[1], args[0])
    df.loc[df.PSAR > df.Close, 'PSAR_T'] = 1
    df.loc[df.PSAR < df.Close, 'PSAR_T'] = -1
    df = df.fillna(0)
    df.loc[(df.PSAR_T.shift(1) < 0) & (df.PSAR_T > 0), 'PSAR_A'] = 1
    df.loc[(df.PSAR_T.shift(1) > 0) & (df.PSAR_T < 0), 'PSAR_A'] = -1
    df.drop(columns=['PSAR'], inplace=True)
    return df


def adi_s(df, args):  # adi_fw	adi_sw	adi_sg
    adi = AccDistIndexIndicator(high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                volume=df['Volume'])
    df['ADI'] = adi.acc_dist_index()
    df['ADI_MACD'] = df.ADI.ewm(
        span=args[0], adjust=False).mean() - df.ADI.ewm(span=args[1],
                                                        adjust=False).mean()
    df['ADI_MACD_S'] = df.ADI_MACD.rolling(args[2]).mean()
    df.loc[df.ADI_MACD < df.ADI_MACD_S, 'ADI_T'] = 1
    df.loc[df.ADI_MACD > df.ADI_MACD_S, 'ADI_T'] = -1
    df.loc[(df.ADI_MACD.shift(1) > df.ADI_MACD_S.shift(1)) &
           (df.ADI_MACD < df.ADI_MACD_S), 'ADI_A'] = 1
    df.loc[(df.ADI_MACD.shift(1) < df.ADI_MACD_S.shift(1)) &
           (df.ADI_MACD > df.ADI_MACD_S), 'ADI_A'] = -1
    df.drop(columns=['ADI', 'ADI_MACD', 'ADI_MACD_S'], inplace=True)
    return df


def obv_s(df, args):  # obv_fw obv_sw obv_sg
    obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
    df['OBV'] = obv.on_balance_volume()
    df['OBV_MACD'] = df.OBV.ewm(
        span=args[0], adjust=False).mean() - df.OBV.ewm(span=args[1],
                                                        adjust=False).mean()
    df['OBV_MACD_S'] = df.OBV_MACD.rolling(args[2]).mean()
    df.loc[df.OBV_MACD < df.OBV_MACD_S, 'OBV_T'] = 1
    df.loc[df.OBV_MACD > df.OBV_MACD_S, 'OBV_T'] = -1
    df.loc[(df.OBV_MACD.shift(1) > df.OBV_MACD_S.shift(1)) &
           (df.OBV_MACD < df.OBV_MACD_S), 'OBV_A'] = 1
    df.loc[(df.OBV_MACD.shift(1) < df.OBV_MACD_S.shift(1)) &
           (df.OBV_MACD > df.OBV_MACD_S), 'OBV_A'] = -1
    df.drop(columns=[
        'OBV',
        'OBV_MACD',
        'OBV_MACD_S',
    ], inplace=True)
    return df


def eom_s(df, args):  # eom_w eom_sma std_m
    eom = EaseOfMovementIndicator(high=df['High'],
                                  low=df['Low'],
                                  volume=df['Volume'],
                                  window=args[0])
    df['EOM'] = eom.ease_of_movement()
    df['EOM_S'] = df['EOM'].rolling(args[1]).mean()
    eom_s = round(df.EOM.mean() + df.EOM.std() * args[2], 2)
    eom_b = round(df.EOM.mean() - df.EOM.std() * args[2], 2)
    df.loc[(df.EOM > eom_s), 'EOM_Z'] = -1
    df.loc[(df.EOM < eom_b), 'EOM_Z'] = -1
    df.loc[(df.EOM.shift(1) > df.EOM_S) & (df.EOM < df.EOM_S), 'EOM_A'] = 1
    df.loc[(df.EOM.shift(1) < df.EOM_S) & (df.EOM > df.EOM_S), 'EOM_A'] = -1
    df.drop(columns=['EOM', 'EOM_S'], inplace=True)
    return df


def vpt_s(df, args):  # vpt_sma adx_l
    vpt = VolumePriceTrendIndicator(close=df['Close'], volume=df['Volume'])
    adx = ADXIndicator(high=df['High'],
                       low=df['Low'],
                       close=df['Close'],
                       window=args[0])
    df['VPT'] = vpt.volume_price_trend()
    df['ADX'] = adx.adx()
    df['VPT_S'] = df['VPT'].rolling(args[0]).mean()
    df.loc[(df.VPT < df.VPT_S) & df.ADX > args[1], 'VPT_T'] = 1
    df.loc[(df.VPT > df.VPT_S) & df.ADX > args[1], 'VPT_T'] = -1
    df.loc[(df.VPT.shift(1) > df.VPT_S) & (df.VPT < df.VPT_S), 'VPT_A'] = 1
    df.loc[(df.VPT.shift(1) < df.VPT_S) & (df.VPT > df.VPT_S), 'VPT_A'] = -1
    df.drop(columns=['VPT', 'VPT_S', 'ADX'], inplace=True)
    return df


def sma_s(df, args):  # sma_w
    df['SMA'] = df['Close'].rolling(args[0]).mean()
    df.loc[(df.Close > df.SMA * 1.04), 'SMA_Z'] = 1
    df.loc[(df.Close < df.SMA * 0.96), 'SMA_Z'] = -1
    df.loc[(df.Close.shift(1) > df.SMA * 1.04) & (df.Close < df.SMA),
           'SMA_A'] = 1
    df.loc[(df.Close.shift(1) < df.SMA * 0.96) & (df.Close > df.SMA),
           'SMA_A'] = -1
    df.drop(columns=['SMA'], inplace=True)
    return df


def wma_s(df, args):  # wma_w
    wma = WMAIndicator(close=df['Close'], window=args[0])
    df['WMA'] = wma.wma()
    df.loc[(df.Close > df.WMA * 1.04), 'WMA_Z'] = 1
    df.loc[(df.Close < df.WMA * 0.96), 'WMA_Z'] = -1
    df.loc[(df.Close.shift(1) > df.WMA * 1.04) & (df.Close < df.WMA),
           'WMA_A'] = 1
    df.loc[(df.Close.shift(1) < df.WMA * 0.96) & (df.Close > df.WMA),
           'WMA_A'] = -1
    df.drop(columns=['WMA'], inplace=True)
    return df


def ema_s(df, args):  # ema_w
    df['EMA'] = df['Close'].ewm(span=args[0],
                                min_periods=0,
                                adjust=False,
                                ignore_na=False).mean()
    df.loc[(df.Close > df.EMA * 1.04), 'EMA_Z'] = 1
    df.loc[(df.Close < df.EMA * 0.96), 'EMA_Z'] = -1
    df.loc[(df.Close.shift(1) > df.EMA * 1.04) & (df.Close < df.EMA),
           'EMA_A'] = 1
    df.loc[(df.Close.shift(1) < df.EMA * 0.96) & (df.Close > df.EMA),
           'EMA_A'] = -1

    df.drop(columns=['EMA'], inplace=True)
    return df


def set_indicators(df, name, str_dic):
    # Diccionario de funciones:

    f_dic = {
        'RSI_Z': rsi_s,
        'TSI_Z': tsi_s,
        'TSI_T': tsi_s,
        'TSI_A': tsi_s,
        'KST_T': kst_s,
        'KST_A': kst_s,
        'STC_T': stc_s,
        'STC_A': stc_s,
        'CMF_Z': cmf_s,
        'CMF_A': cmf_s,
        'FI_Z': fi_s,
        'FI_A': fi_s,
        'MFI_Z': mfi_s,
        'UO_Z': uo_s,
        'SO_Z': so_s,
        'SO_A': so_s,
        'KI_Z': ki_s,
        'KI_T': ki_s,
        'KI_A': ki_s,
        'ROC_Z': roc_s,
        'ROC_T': roc_s,
        'AO_T': ao_s,
        'AO_A': ao_s,
        'WI_Z': wi_s,
        'SRSI_Z': srsi_s,
        'SRSI_A': srsi_s,
        'PO_T': po_s,
        'PO_A': po_s,
        'PVO_T': pvo_s,
        'PVO_A': pvo_s,
        'ATR_Z': atr_s,
        'BB_Z': bb_s,
        'BB_T': bb_s,
        'BB_A': bb_s,
        'KC_T': kc_s,
        'DC_Z': dc_s,
        'DC_T': dc_s,
        'DC_A': dc_s,
        'UI_Z': ui_s,
        'MACD_T': macd_s,
        'MACD_A': macd_s,
        'ADX_T': adx_s,
        'ADX_Z': adx_s,
        'AI_A': ai_s,
        'CCI_Z': cci_s,
        'CCI_A': cci_s,
        'DPO_Z': dpo_s,
        'MI_Z': mi_s,
        'II_Z': ii_s,
        'II_T': ii_s,
        'II_A': ii_s,
        'TRIX_T': trix_s,
        'TRIX_A': trix_s,
        'VI_T': vi_s,
        'PSAR_T': psar_s,
        'PSAR_A': psar_s,
        'ADI_T': adi_s,
        'ADI_A': adi_s,
        'OBV_T': obv_s,
        'OBV_A': obv_s,
        'EOM_Z': eom_s,
        'EOM_A': eom_s,
        'VPT_T': vpt_s,
        'VPT_A': vpt_s,
        'SMA_Z': sma_s,
        'SMA_A': sma_s,
        'WMA_Z': wma_s,
        'WMA_A': wma_s,
        'EMA_Z': ema_s,
        'EMA_A': ema_s
    }

    idx = []
    for x in range(len(str_dic[name])):

        if str_dic[name][x][0] is None:
            continue
        else:
            df = f_dic[str_dic[name][x][0]](
                df, str_dic[name][x][1]
            )  # Aplica los argumentos a una función extraída del diccionario de funciones (f_dic)
            idx.append(
                str_dic[name][x][0]
            )  # Agrega a la lista idx los indicadores de a 1 (Si hay 1 agrega 1 si hay 3)

    df = df[df.columns.intersection(
        idx)]  # Selecciona las columnas que figuran en la lista
    df = df.fillna(0)
    df = df.reset_index()
    df = df.drop(columns=['Datetime'])
    return df


def signal(coin, type):
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login('gusrab@gmail.com', '33gatosgrises')
            subject = f'{coin} {type} Signal'
            body = 'You know what to do'
            msg = f'Subject: {subject}\n\n{body}'
            recipients = ['gusrab@gmail.com', 'Shei3.1416@tuta.io']
            smtp.sendmail('gusrab@gmail.com', recipients, msg)
            print(f'{coin} {type} Alarm Triggered - Email Sent')
    except:
        print('Something went wrong...')


#                   -------- Binance Historical OHLC Import Tool -------

cryptos = ['BTC/BUSD', 'ETH/BUSD', 'AVAX/BUSD', 'BNB/BUSD', 'SOL/BUSD', 'LUNA/BUSD', 'ADA/BUSD']
sample_freq = '1m'
since = 200  # Expressed in hours
#loc_folder = 'C:/Users/USER/PycharmProjects/pythonProject/'
loc_folder = 'Results/'
t_list = ('3h', '1h', '15min')
str_dic = {}

for x in range(len(cryptos)):
    for y in range(len(t_list)):
        file_n = f'TA Results {cryptos[x].replace("/", "")} - {t_list[y]} 0.7.csv'
        dt = pd.read_csv(f'{loc_folder}{file_n}')
        dt = dt.astype({
            'sma_w': int,
            'ema_w': int,
            'wma_w': int,
            'adi_fw': int,
            'adi_sw': int,
            'adi_sg': int,
            'adx_w': int,
            'adx_l': int,
            'ai_w': int,
            'ao_1': int,
            'ao_2': int,
            'atr_w': int,
            'atr_l': int,
            'bb_w': int,
            'bb_d': int,
            'bb_l': int,
            'cci_w': int,
            'cci_b': int,
            'cci_s': int,
            'cmf_w': int,
            'dc_w': int,
            'dc_l': int,
            'dpo_w': int,
            'dpo_s': int,
            'eom_w': int,
            'eom_sma': int,
            'eom_b': int,
            'eom_s': int,
            'fi_w': int,
            'fi_b': int,
            'fi_s': int,
            'ii_1': int,
            'ii_2': int,
            'ii_3': int,
            'kc_w': int,
            'kc_a': int,
            'ki_w': int,
            'ki_p1': int,
            'ki_p2': int,
            'ki_p3': int,
            'kst_1': int,
            'kst_2': int,
            'kst_3': int,
            'kst_4': int,
            'kst_ns': int,
            'kst_r1': int,
            'kst_r2': int,
            'kst_r3': int,
            'kst_r4': int,
            'kst_b': int,
            'kst_s': int,
            'macd_fw': int,
            'macd_sw': int,
            'macd_sg': int,
            'mfi_w': int,
            'mfi_b': int,
            'mfi_s': int,
            'mi_fw': int,
            'mi_sw': int,
            'obv_fw': int,
            'obv_sw': int,
            'obv_sg': int,
            'po_sw': int,
            'po_fw': int,
            'po_sg': int,
            'pvo_sw': int,
            'pvo_fw': int,
            'pvo_sg': int,
            'roc_w': int,
            'roc_b': int,
            'roc_s': int,
            'rsi_w': int,
            'rsi_b': int,
            'rsi_s': int,
            'so_w': int,
            'so_sw': int,
            'so_b': int,
            'so_s': int,
            'srsi_w': int,
            'srsi_kw': int,
            'srsi_dw': int,
            'stc_fw': int,
            'stc_sw': int,
            'stc_c': int,
            'stc_s1': int,
            'stc_s2': int,
            'stc_hl': int,
            'stc_ll': int,
            'trix_w': int,
            'trix_sw': int,
            'tsi_fw': int,
            'tsi_sw': int,
            'tsi_sig': int,
            'tsi_s': int,
            'ui_w': int,
            'ui_b': int,
            'ui_s': int,
            'uo_1': int,
            'uo_2': int,
            'uo_3': int,
            'uo_b': int,
            'uo_s': int,
            'vi_w': int,
            'vpt_sma': int,
            'wi_w': int,
            'wi_b': int,
            'wi_s': int
        })
        dt = dt.join(dt['adx_l'].rename('vpt_adx'))
        dt = dt.join(dt['std_m'].rename('mi_std'))
        dt = dt.join(dt['std_m'].rename('fi_std'))
        dt = dt.join(dt['std_m'].rename('eom_std'))
        dt = dt[[
            'Datetime', 'Criteria', 'Score', 'Test Perf. %', 'Transactions',
            'Accuracy', ' Final Equity', 'Market Return', 'Zone', 'Trend',
            'Signal', 'sma_w', 'ema_w', 'wma_w', 'adi_fw', 'adi_sw', 'adi_sg',
            'adx_w', 'adx_l', 'ai_w', 'ao_1', 'ao_2', 'atr_w', 'atr_l', 'bb_w',
            'bb_d', 'bb_l', 'cci_w', 'cci_b', 'cci_s', 'cmf_w', 'cmf_l',
            'dc_w', 'dc_l', 'dpo_w', 'dpo_s', 'eom_w', 'eom_sma', 'eom_b',
            'eom_s', 'eom_std', 'fi_w', 'fi_b', 'fi_s', 'fi_std', 'ii_1',
            'ii_2', 'ii_3', 'kc_w', 'kc_a', 'ki_w', 'ki_p1', 'ki_p2', 'ki_p3',
            'ki_b', 'ki_s', 'kst_1', 'kst_2', 'kst_3', 'kst_4', 'kst_ns',
            'kst_r1', 'kst_r2', 'kst_r3', 'kst_r4', 'kst_b', 'kst_s',
            'macd_fw', 'macd_sw', 'macd_sg', 'mfi_w', 'mfi_b', 'mfi_s',
            'mi_fw', 'mi_sw', 'mi_std', 'obv_fw', 'obv_sw', 'obv_sg', 'po_sw',
            'po_fw', 'po_sg', 'psar_ms', 'psar_st', 'pvo_sw', 'pvo_fw',
            'pvo_sg', 'roc_w', 'roc_b', 'roc_s', 'rsi_w', 'rsi_b', 'rsi_s',
            'so_w', 'so_sw', 'so_b', 'so_s', 'srsi_w', 'srsi_kw', 'srsi_dw',
            'srsi_b', 'srsi_s', 'stc_fw', 'stc_sw', 'stc_c', 'stc_s1',
            'stc_s2', 'stc_hl', 'stc_ll', 'trix_w', 'trix_sw', 'tsi_fw',
            'tsi_sw', 'tsi_sig', 'tsi_s', 'ui_w', 'ui_b', 'ui_s', 'uo_1',
            'uo_2', 'uo_3', 'uo_b', 'uo_s', 'vi_w', 'vpt_sma', 'vpt_adx',
            'wi_w', 'wi_b', 'wi_s', 'std_m'
        ]]
        dt.columns = [f' {x}' for x in dt.columns]
        dt = dt.drop_duplicates(subset=' Accuracy')
        r = len(dt) if len(dt) < 5 else 4
        a_idx = [dt[' Accuracy'].nlargest(x+1).index[-1] for x in range(r)]
        a_list = [dt[' Accuracy'][a_idx[x]] for x in range(r)]
        e_list = [dt.loc[a_idx[x]][5] for x in range(r)]

        if (a_list[0] - a_list[-1]) > 10:
            a_idx.pop()
            a_list.pop()
            e_list.pop()

        score_list = [(a_list[x] - a_list[-1])*(0.1*e_list[x])+e_list[x] for x in range(len(e_list))]
        idx_max = a_idx[score_list.index(max(score_list))]

        zone = dt.loc[idx_max][8]
        trend = dt.loc[idx_max][9]
        signal = dt.loc[idx_max][10]
        str_dic[f'{cryptos[x]}{t_list[y]}'] = [[None, None], [None, None],
                                               [None, None]]

        if zone == 'Null':
            str_dic[f'{cryptos[x]}{t_list[y]}'][0][0] = None
            str_dic[f'{cryptos[x]}{t_list[y]}'][0][1] = None
        else:
            zone_f = ' ' + zone[:-2].lower()
            str_dic[f'{cryptos[x]}{t_list[y]}'][0][0] = zone
            str_dic[f'{cryptos[x]}{t_list[y]}'][0][1] = list(
                dt.loc[idx_max].filter(like=f'{zone_f}'))

        if trend == 'Null':
            str_dic[f'{cryptos[x]}{t_list[y]}'][1][0] = None
            str_dic[f'{cryptos[x]}{t_list[y]}'][1][1] = None
        else:
            trend_f = ' ' + trend[:-2].lower()
            str_dic[f'{cryptos[x]}{t_list[y]}'][1][0] = trend
            str_dic[f'{cryptos[x]}{t_list[y]}'][1][1] = list(
                dt.loc[idx_max].filter(like=f'{trend_f}'))

        if signal == 'Null':
            str_dic[f'{cryptos[x]}{t_list[y]}'][2][0] = None
            str_dic[f'{cryptos[x]}{t_list[y]}'][2][1] = None
        else:
            signal_f = ' ' + signal[:-2].lower()
            str_dic[f'{cryptos[x]}{t_list[y]}'][2][0] = signal
            str_dic[f'{cryptos[x]}{t_list[y]}'][2][1] = list(
                dt.loc[idx_max].filter(like=f'{signal_f}'))

cdf = pd.DataFrame([0 for x in range(len(t_list) * len(cryptos))])
s_list = [0 for x in range(len(cryptos))]

keep_alive()

while True:
    try:
        for x in range(len(cryptos)):
            n = -1
            cdf = pd.DataFrame([0 for x in range(len(t_list))])
            print(cryptos[x])
            dfx = fetch_data(loc_folder,
                            exchange='binance',
                            cryptos=[cryptos[x]],
                            sample_freq=sample_freq,
                            since_hours=since,
                            page_limit=1000)

            print(f'{cryptos[x]} Dasaset Span: {int(len(dfx))/ 60} hours')

            for y in range(len(t_list)):
                if t_list[y] != '1min':
                    df = dfx.resample(t_list[y]).agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).fillna(method='ffill')
                else:
                    df = dfx

                if len(df) < 14:
                    print(
                        f'Dataset {t_list[y]} is to short to be analyzed at the defined sampling'
                    )
                    exit()

                df = set_indicators(df, f'{cryptos[x]}{t_list[y]}', str_dic)
                n += 1
                if df.iloc[-1].sum() == len(df.columns):
                    cdf.iloc[n] = 1
                elif df.iloc[-1].sum() == -len(df.columns):
                    cdf.iloc[n] = -1
            
            if int(cdf.sum()) == -len(t_list) and s_list[x] == 0:
                signal(cryptos[x], 'Buy')
                print(f'{cryptos[x]} Buy Alarm Triggered')
                s_list[x] == 1
            elif int(cdf.iloc[0]) == 1 and s_list[x] == 1:
                signal(cryptos[x], 'Sell')
                print(f'{cryptos[x]} Sell Alarm Triggered')
                s_list[x] == 0

        time.sleep(900)
    except:
      time.sleep(900)
      continue
