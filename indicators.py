from ta.volume import (
    AccDistIndexIndicator,
    ChaikinMoneyFlowIndicator,
    EaseOfMovementIndicator,
    ForceIndexIndicator,
    MFIIndicator,
    OnBalanceVolumeIndicator,
    VolumePriceTrendIndicator,
    DPOIndicator,
    WMAIndicator,
    IchimokuIndicator,
    KSTIndicator,
    MassIndex,
    STCIndicator,
    TRIXIndicator,
    VortexIndicator,
    UltimateOscillator,
    StochasticOscillator,
    KAMAIndicator,
    ROCIndicator,
    AwesomeOscillatorIndicator,
    WilliamsRIndicator,
    PercentagePriceOscillator,
    PercentageVolumeOscillator,
)


from ta.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    UlcerIndex,
)
from ta.momentum import RSIIndicator, StochRSIIndicator, TSIIndicator
from ta.trend import MACD, ADXIndicator, AroonIndicator, CCIIndicator


def rsi_s(df, args):  # rsi_w
    rsi = RSIIndicator(close=df["Close"], window=args[0])
    df["RSI"] = rsi.rsi()
    df.loc[(df.RSI > args[2]), "RSI_Z"] = 1
    df.loc[(df.RSI < args[1]), "RSI_Z"] = -1
    df.drop(columns=["RSI"], inplace=True)
    return df


def tsi_s(df, args):  # tsi_fw, tsi_fw, tsi_sig, tsi_s
    tsi = TSIIndicator(
        close=df["Close"], window_slow=args[0], window_fast=args[1]
    )
    df["TSI"] = tsi.tsi()
    df["TSI_S"] = (
        df["TSI"]
        .ewm(args[2], min_periods=0, adjust=False, ignore_na=False)
        .mean()
    )
    df.loc[(df.TSI > args[3]), "TSI_Z"] = 1
    df.loc[(df.TSI < -args[3]), "TSI_Z"] = -1
    df.loc[(df.TSI < 0), "TSI_T"] = 1
    df.loc[(df.TSI >= 0), "TSI_T"] = -1
    df.loc[
        (df.TSI.shift(1) > df.TSI_S.shift(1)) & (df.TSI < df.TSI_S), "TSI_A"
    ] = 1
    df.loc[
        (df.TSI.shift(1) < df.TSI_S.shift(1)) & (df.TSI > df.TSI_S), "TSI_A"
    ] = -1
    df.drop(columns=["TSI", "TSI_S"], inplace=True)
    return df


def kst_s(
    df, args
):  # kst_1, kst_2, kst_3, kst_4, kst_ns, kst_r1, kst_r2, kst_r3,
    # kst_r4, kst_b, kst_s
    kst = KSTIndicator(
        close=df["Close"],
        roc1=args[5],
        roc2=args[6],
        roc3=args[7],
        roc4=args[8],
        window1=args[0],
        window2=args[1],
        window3=args[2],
        window4=args[3],
        nsig=args[4],
    )
    df["KST"] = kst.kst()
    df["KST_S"] = kst.kst_sig()
    df["KST_H"] = kst.kst_diff()
    df.loc[df.KST_H >= 0, "KST_T"] = -1
    df.loc[df.KST_H < 0, "KST_T"] = 1
    df.loc[
        (df.KST.shift(1) > df.KST_S.shift(1))
        & (df.KST < df.KST_S)
        & (df.KST.shift(1) > args[10]),
        "KST_A",
    ] = 1
    df.loc[
        (df.KST.shift(1) < df.KST_S.shift(1))
        & (df.KST > df.KST_S)
        & (df.KST.shift(1) < args[9]),
        "KST_A",
    ] = -1
    df.drop(columns=["KST", "KST_S", "KST_H"], inplace=True)
    return df


def stc_s(df, args):  # stc_fw, stc_sw, stc_c, stc_s1, stc_s2, stc_hl, stc_ll
    stc = STCIndicator(
        close=df["Close"],
        window_fast=args[0],
        window_slow=args[1],
        cycle=args[2],
        smooth1=args[3],
        smooth2=args[4],
    )
    df["STC"] = stc.stc()
    df.loc[df.STC < args[6], "STC_T"] = 1
    df.loc[df.STC > args[5], "STC_T"] = -1
    df.loc[(df.STC.shift(1) > args[5]) & (df.STC < args[5]), "STC_A"] = 1
    df.loc[(df.STC.shift(1) < args[6]) & (df.STC > args[6]), "STC_A"] = -1
    df.drop(columns=["STC"], inplace=True)
    return df


def cmf_s(df, args):  # cmf_w, cmf_l
    cmf = ChaikinMoneyFlowIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"],
        window=args[0],
    )
    df["CMF"] = cmf.chaikin_money_flow()
    df.loc[(df.CMF > args[1]), "CMF_Z"] = 1
    df.loc[(df.CMF < -args[1]), "CMF_Z"] = -1
    df.loc[(df.CMF.shift(1) > 0) & (df.CMF < 0), "CMF_A"] = 1
    df.loc[(df.CMF.shift(1) < 0) & (df.CMF > 0), "CMF_A"] = -1
    df.drop(columns=["CMF"], inplace=True)
    return df


def fi_s(df, args):  # fi_w, std_m
    fi = ForceIndexIndicator(
        close=df["Close"], volume=df["Volume"], window=args[0]
    )
    df["FI"] = fi.force_index()
    fi_s = int(
        df[df["FI"] > 0]["FI"].mean()
        + 3.5 * df[df["FI"] > 0]["FI"].mean() * args[1]
    )
    fi_b = int(
        df[df["FI"] < 0]["FI"].mean()
        + 2.5 * df[df["FI"] < 0]["FI"].mean() * args[1]
    )
    df.loc[df.FI > fi_s, "FI_Z"] = 1
    df.loc[df.FI < fi_b, "FI_Z"] = -1
    df.loc[(df.FI.shift(1) > fi_s) & (df.FI < fi_s), "FI_A"] = 1
    df.loc[(df.FI.shift(1) < fi_b) & (df.FI > fi_b), "FI_A"] = -1
    df.drop(columns=["FI"], inplace=True)
    return df


def mfi_s(df, args):  # mfi_w, mfi_b, mfi_s
    mfi = MFIIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"],
        window=args[0],
    )
    df["MFI"] = mfi.money_flow_index()
    df.loc[(df.MFI > args[2]), "MFI_Z"] = 1
    df.loc[(df.MFI < args[1]), "MFI_Z"] = -1
    df.drop(columns=["MFI"], inplace=True)
    return df


def uo_s(df, args):  # uo_1, uo_2, uo_3, uo_b, uo_s
    uo = UltimateOscillator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window1=args[0],
        window2=args[1],
        window3=args[2],
    )
    df["UO"] = uo.ultimate_oscillator()
    df.loc[(df.UO > args[4]), "UO_Z"] = 1
    df.loc[(df.UO < args[3]), "UO_Z"] = -1
    df.drop(columns=["UO"], inplace=True)
    return df


def so_s(df, args):  # so_w	so_sw	so_b so_s
    so = StochasticOscillator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=args[0],
        smooth_window=args[1],
    )
    df["SO"] = so.stoch()
    df["SOS"] = so.stoch_signal()
    df.loc[(df.SO > args[3]), "SO_Z"] = 1
    df.loc[(df.SO < args[2]), "SO_Z"] = -1
    df.loc[(df.SO.shift(1) > df.SOS.shift(1)) & (df.SO < df.SOS), "SO_A"] = 1
    df.loc[(df.SO.shift(1) < df.SOS.shift(1)) & (df.SO > df.SOS), "SO_A"] = -1
    df.drop(columns=["SO", "SOS"], inplace=True)
    return df


def ki_s(df, args):  # ki_w, ki_p1, ki_p2, ki_p3, ki_b, ki_s
    ki = KAMAIndicator(
        close=df["Close"], window=args[0], pow1=args[1], pow2=args[2]
    )
    ki_sig = KAMAIndicator(
        close=df["Close"], window=args[0], pow1=args[3], pow2=args[2]
    )
    df["KI"] = ki.kama()
    df["KIS"] = ki_sig.kama()
    df.loc[(df.Close > df.KI * (1 + args[5])), "KI_Z"] = 1
    df.loc[(df.Close < df.KI * (1 - args[4])), "KI_Z"] = -1
    df.loc[
        (df.Close.shift(1) > df.KI.shift(1))
        & (df.Close < df.KI * (1 - args[4])),
        "KI_T",
    ] = 1
    df.loc[
        (df.Close.shift(1) < df.KI.shift(1))
        & (df.Close > df.KI * (1 + args[5])),
        "KI_T",
    ] = -1
    df["KI_T"] = df["KI_T"].fillna(method="ffill")
    df.loc[(df.KI > df.KIS.shift(1)) & (df.KI < df.KIS), "KI_A"] = 1
    df.loc[(df.KI < df.KIS.shift(1)) & (df.KI > df.KIS), "KI_A"] = -1
    df.drop(columns=["KI", "KIS"], inplace=True)
    return df


def roc_s(df, args):  # roc_w, roc_b, roc_s
    roc = ROCIndicator(close=df["Close"], window=args[0])
    df["ROC"] = roc.roc()
    df.loc[(df.ROC > args[2]), "ROC_Z"] = 1
    df.loc[(df.ROC < args[1]), "ROC_Z"] = -1
    df.loc[(df.ROC < 0), "ROC_T"] = 1
    df.loc[(df.ROC >= 0), "ROC_T"] = -1
    df.drop(columns=["ROC"], inplace=True)
    return df


def ao_s(df, args):  # ao_1, ao_2
    ao = AwesomeOscillatorIndicator(
        high=df["High"], low=df["Low"], window1=args[0], window2=args[1]
    )
    df["AO"] = ao.awesome_oscillator()
    df.loc[(df.AO < 0), "AO_T"] = 1
    df.loc[(df.AO >= 0), "AO_T"] = -1
    df.loc[
        ((df.AO.shift(1) > 0) & (df.AO < 0))
        | (
            (df.AO < 0)
            & (df.AO.shift(2) < df.AO.shift(1))
            & (df.AO.shift(1) > df.AO)
        ),
        "AO_A",
    ] = 1
    df.loc[
        ((df.AO.shift(1) < 0) & (df.AO > 0))
        | (
            (df.AO > 0)
            & (df.AO.shift(2) > df.AO.shift(1))
            & (df.AO.shift(1) < df.AO)
        ),
        "AO_A",
    ] = -1
    df.drop(columns=["AO"], inplace=True)
    return df


def wi_s(df, args):  # wi_w	wi_b	wi_s
    wi = WilliamsRIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], lbp=args[0]
    )
    df["WI"] = wi.williams_r()
    df.loc[(df.WI > args[2]), "WI_Z"] = 1
    df.loc[(df.WI < args[1]), "WI_Z"] = -1
    df.drop(columns=["WI"], inplace=True)
    return df


def srsi_s(df, args):  # srsi_w srsi_kw	srsi_dw	srsi_b	srsi_s
    srsi = StochRSIIndicator(
        close=df["Close"], window=args[0], smooth1=args[1], smooth2=args[2]
    )
    df["SRSI"] = srsi.stochrsi()
    df["SRSI_K"] = srsi.stochrsi_k()
    df["SRSI_D"] = srsi.stochrsi_d()
    df.loc[(df.SRSI > args[4]), "SRSI_Z"] = 1
    df.loc[(df.SRSI < args[3]), "SRSI_Z"] = -1
    df.loc[
        (df.SRSI_K.shift(1) > df.SRSI_D.shift(1)) & (df.SRSI_K < df.SRSI_D),
        "SRSI_A",
    ] = 1
    df.loc[
        (df.SRSI_K.shift(1) < df.SRSI_D.shift(1)) & (df.SRSI_K > df.SRSI_D),
        "SRSI_A",
    ] = -1
    df.drop(columns=["SRSI_K", "SRSI_D", "SRSI"], inplace=True)
    return df


def po_s(df, args):  # po_sw	po_fw	po_sg
    po = PercentagePriceOscillator(
        close=df["Close"],
        window_slow=args[0],
        window_fast=args[1],
        window_sign=args[2],
    )
    df["PO"] = po.ppo()
    df["PO_S"] = po.ppo_signal()
    df["PO_H"] = po.ppo_hist()
    df.loc[(df.PO_H < 0), "PO_T"] = 1
    df.loc[(df.PO_H >= 0), "PO_T"] = -1
    df.loc[(df.PO.shift(1) > df.PO_S.shift(1)) & (df.PO < df.PO_S), "PO_A"] = 1
    df.loc[
        (df.PO.shift(1) < df.PO_S.shift(1)) & (df.PO > df.PO_S), "PO_A"
    ] = -1
    df.drop(columns=["PO", "PO_S", "PO_H"], inplace=True)
    return df


def pvo_s(df, args):  # pvo_sw	pvo_fw	pvo_sg
    pvo = PercentageVolumeOscillator(
        volume=df["Volume"],
        window_slow=args[0],
        window_fast=args[1],
        window_sign=args[2],
    )
    df["PVO"] = pvo.pvo()
    df["PVO_S"] = pvo.pvo_signal()
    df["PVO_H"] = pvo.pvo_hist()
    df.loc[(df.PVO_H < 0), "PVO_T"] = 1
    df.loc[(df.PVO_H >= 0), "PVO_T"] = -1
    df.loc[
        (df.PVO.shift(1) > df.PVO_S.shift(1)) & (df.PVO < df.PVO_S), "PVO_A"
    ] = 1
    df.loc[
        (df.PVO.shift(1) < df.PVO_S.shift(1)) & (df.PVO > df.PVO_S), "PVO_A"
    ] = -1
    df.drop(columns=["PVO", "PVO_S", "PVO_H"], inplace=True)
    return df


def atr_s(df, args):  # atr_w atr_l rsi_w rsi_b rsi_s
    atr = AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=args[0]
    )
    rsi = RSIIndicator(close=df["Close"], window=args[2])
    df["RSI"] = rsi.rsi()
    df["ATR"] = atr.average_true_range()
    df.loc[
        ((df.ATR.shift(1) * (args[0] - 1) + df.ATR) / args[0] > args[1])
        & (df.RSI > args[4]),
        "ATR_Z",
    ] = 1
    df.loc[
        ((df.ATR.shift(1) * (args[0] - 1) + df.ATR) / args[0] > args[1])
        & (df.RSI < args[3]),
        "ATR_Z",
    ] = -1
    df.drop(columns=["ATR", "RSI"], inplace=True)
    return df


def bb_s(df, args):  # bb_w	bb_d bb_l
    bb = BollingerBands(close=df["Close"], window=args[0], window_dev=args[1])
    adx = ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=args[0]
    )
    df["BB_H"] = bb.bollinger_hband_indicator()
    df["BB_L"] = bb.bollinger_lband_indicator()
    df["BB_W"] = bb.bollinger_wband()
    df["ADX_P"] = adx.adx_pos()
    df["ADX_N"] = adx.adx_neg()
    df.loc[(df.ADX_P < df.ADX_N), "ADX_T"] = 1
    df.loc[(df.ADX_P > df.ADX_N), "ADX_T"] = -1
    df.loc[(df.BB_W > args[2]), "BB_Z"] = 1
    df.loc[(df.BB_W > args[2]), "BB_Z"] = -1
    df.loc[(df.Close >= df.BB_H) & (df.ADX_T > 0), "BB_T"] = 1
    df.loc[(df.Close <= df.BB_L) & (df.ADX_T < 0), "BB_T"] = -1
    df["BB_T"] = df["BB_T"].fillna(method="ffill")
    df.loc[(df.Close >= df.BB_H), "BB_A"] = 1
    df.loc[(df.Close <= df.BB_L), "BB_A"] = -1
    df.drop(
        columns=["BB_H", "BB_L", "BB_W", "ADX", "ADX_P", "ADX_N", "ADX_T"],
        inplace=True,
    )
    return df


def kc_s(df, args):  # kc_w	kc_a
    kc = KeltnerChannel(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=args[0],
        window_atr=args[1],
    )
    df["KC_H"] = kc.keltner_channel_hband_indicator()
    df["KC_L"] = kc.keltner_channel_lband_indicator()
    df.loc[df.KC_L > 0, "KC_T"] = 1
    df.loc[df.KC_H > 0, "KC_T"] = -1
    df["KC_T"] = df["KC_T"].fillna(method="ffill")
    df.drop(columns=["KC_H", "KC_L"], inplace=True)
    return df


def dc_s(df, args):  # dc_w	dc_l
    dc = DonchianChannel(
        high=df["High"], low=df["Low"], close=df["Close"], window=args[0]
    )
    adx = ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=args[0]
    )
    df["DC_H"] = dc.donchian_channel_hband()
    df["DC_L"] = dc.donchian_channel_lband()
    df["DC_W"] = dc.donchian_channel_wband()
    df["ADX_P"] = adx.adx_pos()
    df["ADX_N"] = adx.adx_neg()
    df.loc[(df.ADX_P < df.ADX_N), "ADX_T"] = 1
    df.loc[(df.ADX_P > df.ADX_N), "ADX_T"] = -1
    df.loc[(df.DC_W.shift(1) < args[1]), "DC_Z"] = 1
    df.loc[(df.DC_W.shift(1) > args[1]), "DC_Z"] = -1
    df.loc[(df.High >= df.DC_H) & (df.ADX_T > 0), "DC_T"] = 1
    df.loc[(df.Low <= df.DC_L) & (df.ADX_T < 0), "DC_T"] = -1
    df["DC_T"] = df["DC_T"].fillna(method="ffill")
    df.loc[(df.Close >= df.DC_H), "DC_A"] = 1
    df.loc[(df.Close <= df.DC_L), "DC_A"] = -1
    df.drop(
        columns=["DC_H", "DC_L", "DC_W", "ADX", "ADX_P", "ADX_N", "ADX_T"],
        inplace=True,
    )
    return df


def ui_s(df, args):  # ui_w	ui_b ui_s
    ui = UlcerIndex(close=df["Close"], window=args[0])
    df["UI"] = ui.ulcer_index()
    df.loc[(df.UI < args[2]), "UI_Z"] = 1
    df.loc[(df.UI > args[1]), "UI_Z"] = -1
    df.drop(columns=["UI"], inplace=True)
    return df


def macd_s(df, args):  # macd_fw	macd_sw	macd_sg
    macd = MACD(
        close=df["Close"],
        window_fast=args[0],
        window_slow=args[1],
        window_sign=args[2],
    )
    df["MACD"] = macd.macd()
    df["MACD_S"] = macd.macd_signal()
    df["MACD_H"] = macd.macd_diff()
    df.loc[(df.MACD_H < 0), "MACD_T"] = 1
    df.loc[(df.MACD_H >= 0), "MACD_T"] = -1
    df.loc[
        (df.MACD.shift(1) > df.MACD_S.shift(1)) & (df.MACD < df.MACD_S),
        "MACD_A",
    ] = 1
    df.loc[
        (df.MACD.shift(1) < df.MACD_S.shift(1)) & (df.MACD > df.MACD_S),
        "MACD_A",
    ] = -1
    df.drop(columns=["MACD", "MACD_S", "MACD_H"], inplace=True)
    return df


def adx_s(df, args):  # adx_w	adx_l
    adx = ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=args[0]
    )
    df["ADX"] = adx.adx()
    df["ADX_P"] = adx.adx_pos()
    df["ADX_N"] = adx.adx_neg()
    df.loc[(df.ADX_P < df.ADX_N), "ADX_T"] = 1
    df.loc[(df.ADX_P > df.ADX_N), "ADX_T"] = -1
    df.loc[(df.ADX_T > 0) & df.ADX > args[1], "ADX_Z"] = 1
    df.loc[(df.ADX_T < 0) & df.ADX > args[1], "ADX_Z"] = -1
    df.drop(columns=["ADX", "ADX_P", "ADX_N"], inplace=True)
    return df


def ai_s(df, args):  # ai_w
    ai = AroonIndicator(close=df["Close"], window=args[0])
    df["AI_U"] = ai.aroon_up()
    df["AI_D"] = ai.aroon_down()
    df.loc[
        (df.AI_U.shift(1) > df.AI_D.shift(1)) & (df.AI_U < df.AI_D), "AI_A"
    ] = 1
    df.loc[
        (df.AI_U.shift(1) < df.AI_D.shift(1)) & (df.AI_U > df.AI_D), "AI_A"
    ] = -1
    df.drop(columns=["AI_U", "AI_D"], inplace=True)
    return df


def cci_s(df, args):  # cci_w cci_b cci_s
    cci = CCIIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=args[0],
        constant=0.015,
    )
    df["CCI"] = cci.cci()
    df.loc[df.CCI > args[2], "CCI_Z"] = 1
    df.loc[df.CCI < args[1], "CCI_Z"] = -1
    df.loc[(df.CCI.shift(1) > args[2]) & (df.CCI < args[2]), "CCI_A"] = 1
    df.loc[(df.CCI.shift(1) < args[1]) & (df.CCI > args[1]), "CCI_A"] = -1
    df.drop(columns=["CCI"], inplace=True)
    return df


def dpo_s(df, args):  # dpo_w, dpo_s
    dpo = DPOIndicator(close=df["Close"], window=args[0])
    df["DPO"] = dpo.dpo()
    df.loc[(df.DPO > args[1]), "DPO_Z"] = 1
    df.loc[(df.DPO < -args[1]), "DPO_Z"] = -1
    df.drop(columns=["DPO"], inplace=True)
    return df


def mi_s(df, args):  # mi_fw	mi_sw std_m
    mi = MassIndex(
        high=df["High"],
        low=df["Low"],
        window_fast=args[0],
        window_slow=args[1],
    )
    df["MI"] = mi.mass_index()
    mi_s = df.MI.mean() + df.MI.std() * args[2]
    df.loc[df.MI > mi_s, "MI_Z"] = 1
    df.loc[df.MI < mi_s, "MI_Z"] = -1
    df.drop(columns=["MI"], inplace=True)
    return df


def ii_s(df, args):  # ii_1 ii_2 ii_3
    ii = IchimokuIndicator(
        high=df["High"],
        low=df["Low"],
        window1=args[0],
        window2=args[1],
        window3=args[2],
    )
    df["II"] = ii.ichimoku_conversion_line()
    df["II_S"] = ii.ichimoku_base_line()
    df["II_LA"] = ii.ichimoku_a()
    df["II_LB"] = ii.ichimoku_b()
    df.loc[(df.II > df.II_S), "II_Z"] = 1
    df.loc[(df.II < df.II_S), "II_Z"] = -1
    df.loc[(df.Close < df.II_S), "II_T"] = 1
    df.loc[(df.Close >= df.II_S), "II_T"] = -1
    df.loc[
        (df.II.shift(1) > df.II_S.shift(1))
        & (df.II < df.II_S)
        & (df.High > df.II_LA),
        "II_A",
    ] = 1
    df.loc[
        (df.II.shift(1) < df.II_S.shift(1))
        & (df.II > df.II_S)
        & (df.Low < df.II_LB),
        "II_A",
    ] = -1
    df.drop(columns=["II", "II_S", "II_LA", "II_LB"], inplace=True)
    return df


def trix_s(df, args):  # trix_w	trix_sw
    trix = TRIXIndicator(close=df["Close"], window=args[0])
    df["TRIX"] = trix.trix()
    df["TRIX_S"] = df["TRIX"].ewm(span=args[1], adjust=False).mean()
    df.loc[df.TRIX < 0, "TRIX_T"] = 1
    df.loc[df.TRIX >= 0, "TRIX_T"] = -1
    df.loc[
        (df.TRIX.shift(1) > df.TRIX_S.shift(1)) & (df.TRIX < df.TRIX_S),
        "TRIX_A",
    ] = 1
    df.loc[
        (df.TRIX.shift(1) < df.TRIX_S.shift(1)) & (df.TRIX > df.TRIX_S),
        "TRIX_A",
    ] = -1
    df.drop(columns=["TRIX", "TRIX_S"], inplace=True)
    return df


def vi_s(df, args):  # vi_w
    vi = VortexIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=args[0]
    )
    df["VI_P"] = vi.vortex_indicator_pos()
    df["VI_N"] = vi.vortex_indicator_neg()
    df["VI_H"] = vi.vortex_indicator_diff()
    df.loc[(df.VI_P < df.VI_N), "VI_T"] = 1
    df.loc[(df.VI_P > df.VI_N), "VI_T"] = -1
    df.loc[
        (df.VI_P.shift(1) > df.VI_N.shift(1)) & (df.VI_P < df.VI_N), "VI_A"
    ] = 1
    df.loc[
        (df.VI_P.shift(1) < df.VI_N.shift(1)) & (df.VI_P > df.VI_N), "VI_A"
    ] = -1
    df.drop(columns=["VI_H", "VI_P", "VI_N"], inplace=True)
    return df


def psar_s(df, args):  # psar_ms	psar_st
    df["PSAR"] = get_psar(df, args[1], args[0])
    df.loc[df.PSAR > df.Close, "PSAR_T"] = 1
    df.loc[df.PSAR < df.Close, "PSAR_T"] = -1
    df = df.fillna(0)
    df.loc[(df.PSAR_T.shift(1) < 0) & (df.PSAR_T > 0), "PSAR_A"] = 1
    df.loc[(df.PSAR_T.shift(1) > 0) & (df.PSAR_T < 0), "PSAR_A"] = -1
    df.drop(columns=["PSAR"], inplace=True)
    return df


def adi_s(df, args):  # adi_fw	adi_sw	adi_sg
    adi = AccDistIndexIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]
    )
    df["ADI"] = adi.acc_dist_index()
    df["ADI_MACD"] = (
        df.ADI.ewm(span=args[0], adjust=False).mean()
        - df.ADI.ewm(span=args[1], adjust=False).mean()
    )
    df["ADI_MACD_S"] = df.ADI_MACD.rolling(args[2]).mean()
    df.loc[df.ADI_MACD < df.ADI_MACD_S, "ADI_T"] = 1
    df.loc[df.ADI_MACD > df.ADI_MACD_S, "ADI_T"] = -1
    df.loc[
        (df.ADI_MACD.shift(1) > df.ADI_MACD_S.shift(1))
        & (df.ADI_MACD < df.ADI_MACD_S),
        "ADI_A",
    ] = 1
    df.loc[
        (df.ADI_MACD.shift(1) < df.ADI_MACD_S.shift(1))
        & (df.ADI_MACD > df.ADI_MACD_S),
        "ADI_A",
    ] = -1
    df.drop(columns=["ADI", "ADI_MACD", "ADI_MACD_S"], inplace=True)
    return df


def obv_s(df, args):  # obv_fw obv_sw obv_sg
    obv = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
    df["OBV"] = obv.on_balance_volume()
    df["OBV_MACD"] = (
        df.OBV.ewm(span=args[0], adjust=False).mean()
        - df.OBV.ewm(span=args[1], adjust=False).mean()
    )
    df["OBV_MACD_S"] = df.OBV_MACD.rolling(args[2]).mean()
    df.loc[df.OBV_MACD < df.OBV_MACD_S, "OBV_T"] = 1
    df.loc[df.OBV_MACD > df.OBV_MACD_S, "OBV_T"] = -1
    df.loc[
        (df.OBV_MACD.shift(1) > df.OBV_MACD_S.shift(1))
        & (df.OBV_MACD < df.OBV_MACD_S),
        "OBV_A",
    ] = 1
    df.loc[
        (df.OBV_MACD.shift(1) < df.OBV_MACD_S.shift(1))
        & (df.OBV_MACD > df.OBV_MACD_S),
        "OBV_A",
    ] = -1
    df.drop(
        columns=[
            "OBV",
            "OBV_MACD",
            "OBV_MACD_S",
        ],
        inplace=True,
    )
    return df


def eom_s(df, args):  # eom_w eom_sma std_m
    eom = EaseOfMovementIndicator(
        high=df["High"], low=df["Low"], volume=df["Volume"], window=args[0]
    )
    df["EOM"] = eom.ease_of_movement()
    df["EOM_S"] = df["EOM"].rolling(args[1]).mean()
    eom_s = round(df.EOM.mean() + df.EOM.std() * args[2], 2)
    eom_b = round(df.EOM.mean() - df.EOM.std() * args[2], 2)
    df.loc[(df.EOM > eom_s), "EOM_Z"] = -1
    df.loc[(df.EOM < eom_b), "EOM_Z"] = -1
    df.loc[(df.EOM.shift(1) > df.EOM_S) & (df.EOM < df.EOM_S), "EOM_A"] = 1
    df.loc[(df.EOM.shift(1) < df.EOM_S) & (df.EOM > df.EOM_S), "EOM_A"] = -1
    df.drop(columns=["EOM", "EOM_S"], inplace=True)
    return df


def vpt_s(df, args):  # vpt_sma adx_l
    vpt = VolumePriceTrendIndicator(close=df["Close"], volume=df["Volume"])
    adx = ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=args[0]
    )
    df["VPT"] = vpt.volume_price_trend()
    df["ADX"] = adx.adx()
    df["VPT_S"] = df["VPT"].rolling(args[0]).mean()
    df.loc[(df.VPT < df.VPT_S) & df.ADX > args[1], "VPT_T"] = 1
    df.loc[(df.VPT > df.VPT_S) & df.ADX > args[1], "VPT_T"] = -1
    df.loc[(df.VPT.shift(1) > df.VPT_S) & (df.VPT < df.VPT_S), "VPT_A"] = 1
    df.loc[(df.VPT.shift(1) < df.VPT_S) & (df.VPT > df.VPT_S), "VPT_A"] = -1
    df.drop(columns=["VPT", "VPT_S", "ADX"], inplace=True)
    return df


def sma_s(df, args):  # sma_w
    df["SMA"] = df["Close"].rolling(args[0]).mean()
    df.loc[(df.Close > df.SMA * 1.04), "SMA_Z"] = 1
    df.loc[(df.Close < df.SMA * 0.96), "SMA_Z"] = -1
    df.loc[
        (df.Close.shift(1) > df.SMA * 1.04) & (df.Close < df.SMA), "SMA_A"
    ] = 1
    df.loc[
        (df.Close.shift(1) < df.SMA * 0.96) & (df.Close > df.SMA), "SMA_A"
    ] = -1
    df.drop(columns=["SMA"], inplace=True)
    return df


def wma_s(df, args):  # wma_w
    wma = WMAIndicator(close=df["Close"], window=args[0])
    df["WMA"] = wma.wma()
    df.loc[(df.Close > df.WMA * 1.04), "WMA_Z"] = 1
    df.loc[(df.Close < df.WMA * 0.96), "WMA_Z"] = -1
    df.loc[
        (df.Close.shift(1) > df.WMA * 1.04) & (df.Close < df.WMA), "WMA_A"
    ] = 1
    df.loc[
        (df.Close.shift(1) < df.WMA * 0.96) & (df.Close > df.WMA), "WMA_A"
    ] = -1
    df.drop(columns=["WMA"], inplace=True)
    return df


def get_psar(df, iaf=0.02, maxaf=0.2):
    length = len(df)
    high = df["High"]
    low = df["Low"]
    df["PSAR"] = df["Close"].copy()
    bull = True
    af = iaf
    hp = high.iloc[0]
    lp = low.iloc[0]

    for i in range(2, length):
        if bull:
            df.PSAR.iloc[i] = df.PSAR.iloc[i - 1] + af * (
                hp - df.PSAR.iloc[i - 1]
            )
        else:
            df.PSAR.iloc[i] = df.PSAR.iloc[i - 1] + af * (
                lp - df.PSAR.iloc[i - 1]
            )

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


def ema_s(df, args):  # ema_w
    df["EMA"] = (
        df["Close"]
        .ewm(span=args[0], min_periods=0, adjust=False, ignore_na=False)
        .mean()
    )
    df.loc[(df.Close > df.EMA * 1.04), "EMA_Z"] = 1
    df.loc[(df.Close < df.EMA * 0.96), "EMA_Z"] = -1
    df.loc[
        (df.Close.shift(1) > df.EMA * 1.04) & (df.Close < df.EMA), "EMA_A"
    ] = 1
    df.loc[
        (df.Close.shift(1) < df.EMA * 0.96) & (df.Close > df.EMA), "EMA_A"
    ] = -1

    df.drop(columns=["EMA"], inplace=True)
    return df


INDICATORS_FUNC_DICT = {
    "RSI_Z": rsi_s,
    "TSI_Z": tsi_s,
    "TSI_T": tsi_s,
    "TSI_A": tsi_s,
    "KST_T": kst_s,
    "KST_A": kst_s,
    "STC_T": stc_s,
    "STC_A": stc_s,
    "CMF_Z": cmf_s,
    "CMF_A": cmf_s,
    "FI_Z": fi_s,
    "FI_A": fi_s,
    "MFI_Z": mfi_s,
    "UO_Z": uo_s,
    "SO_Z": so_s,
    "SO_A": so_s,
    "KI_Z": ki_s,
    "KI_T": ki_s,
    "KI_A": ki_s,
    "ROC_Z": roc_s,
    "ROC_T": roc_s,
    "AO_T": ao_s,
    "AO_A": ao_s,
    "WI_Z": wi_s,
    "SRSI_Z": srsi_s,
    "SRSI_A": srsi_s,
    "PO_T": po_s,
    "PO_A": po_s,
    "PVO_T": pvo_s,
    "PVO_A": pvo_s,
    "ATR_Z": atr_s,
    "BB_Z": bb_s,
    "BB_T": bb_s,
    "BB_A": bb_s,
    "KC_T": kc_s,
    "DC_Z": dc_s,
    "DC_T": dc_s,
    "DC_A": dc_s,
    "UI_Z": ui_s,
    "MACD_T": macd_s,
    "MACD_A": macd_s,
    "ADX_T": adx_s,
    "ADX_Z": adx_s,
    "AI_A": ai_s,
    "CCI_Z": cci_s,
    "CCI_A": cci_s,
    "DPO_Z": dpo_s,
    "MI_Z": mi_s,
    "II_Z": ii_s,
    "II_T": ii_s,
    "II_A": ii_s,
    "TRIX_T": trix_s,
    "TRIX_A": trix_s,
    "VI_T": vi_s,
    "PSAR_T": psar_s,
    "PSAR_A": psar_s,
    "ADI_T": adi_s,
    "ADI_A": adi_s,
    "OBV_T": obv_s,
    "OBV_A": obv_s,
    "EOM_Z": eom_s,
    "EOM_A": eom_s,
    "VPT_T": vpt_s,
    "VPT_A": vpt_s,
    "SMA_Z": sma_s,
    "SMA_A": sma_s,
    "WMA_Z": wma_s,
    "WMA_A": wma_s,
    "EMA_Z": ema_s,
    "EMA_A": ema_s,
}
