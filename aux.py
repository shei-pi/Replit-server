import pandas as pd

ASTYPE_COLS = ['sma_w', 'ema_w', 'wma_w', 'adi_fw', 'adi_sw', 'adi_sg', 'adx_w', 'adx_l', 'ai_w', 'ao_1', 'ao_2', 'atr_w', 'atr_l', 'bb_w', 'bb_d', 'bb_l', 'cci_w', 'cci_b', 'cci_s', 'cmf_w', 'dc_w', 'dc_l', 'dpo_w', 'dpo_s', 'eom_w', 'eom_sma', 'eom_b', 'eom_s', 'fi_w', 'fi_b', 'fi_s', 'ii_1', 'ii_2', 'ii_3', 'kc_w', 'kc_a', 'ki_w', 'ki_p1', 'ki_p2', 'ki_p3', 'kst_1', 'kst_2', 'kst_3', 'kst_4', 'kst_ns', 'kst_r1', 'kst_r2', 'kst_r3', 'kst_r4', 'kst_b', 'kst_s', 'macd_fw', 'macd_sw', 'macd_sg',
               'mfi_w', 'mfi_b', 'mfi_s', 'mi_fw', 'mi_sw', 'obv_fw', 'obv_sw', 'obv_sg', 'po_sw', 'po_fw', 'po_sg', 'pvo_sw', 'pvo_fw', 'pvo_sg', 'roc_w', 'roc_b', 'roc_s', 'rsi_w', 'rsi_b', 'rsi_s', 'so_w', 'so_sw', 'so_b', 'so_s', 'srsi_w', 'srsi_kw', 'srsi_dw', 'stc_fw', 'stc_sw', 'stc_c', 'stc_s1', 'stc_s2', 'stc_hl', 'stc_ll', 'trix_w', 'trix_sw', 'tsi_fw', 'tsi_sw', 'tsi_sig', 'tsi_s', 'ui_w', 'ui_b', 'ui_s', 'uo_1', 'uo_2', 'uo_3', 'uo_b', 'uo_s', 'vi_w', 'vpt_sma', 'wi_w', 'wi_b', 'wi_s']

ASTYPE_DICT = {c: int for c in ASTYPE_COLS}


def read_csv_to_df(loc_folder, filename):
    read_file = f'{loc_folder}/{filename}'
    df = pd.read_csv(read_file, index_col='Datetime', parse_dates=True)
    return df
