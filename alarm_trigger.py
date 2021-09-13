from indicators import INDICATORS_FUNC_DICT
from os import system
from keep_alive import keep_alive
import time
import pandas as pd
import smtplib
from ohlc import fetch_cryptodata_from_exchange
from aux import ASTYPE_DICT, DT_OUT_COLUMNS

system("pip install --upgrade pip")


def set_indicators(df, name, str_dic):
    # Diccionario de funciones:

    idx = []
    for x in range(len(str_dic[name])):

        if str_dic[name][x][0] is None:
            continue
        else:
            df = INDICATORS_FUNC_DICT[str_dic[name][x][0]](
                df, str_dic[name][x][1]
            )  # Aplica los argumentos a una función extraída
            # del diccionario de funciones (f_dic)
            idx.append(str_dic[name][x][0])  # Agrega a la lista idx los
            # indicadores
            # de a 1 (Si hay 1 agrega 1 si hay 3)

    df = df[
        df.columns.intersection(idx)
    ]  # Selecciona las columnas que figuran en la lista
    df = df.fillna(0)
    df = df.reset_index()
    df = df.drop(columns=["Datetime"])
    return df


def send_signal(coin, type):
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login("gusrab@gmail.com", "33gatosgrises")
            subject = f"{coin} {type} Signal"
            body = "You know what to do"
            msg = f"Subject: {subject}\n\n{body}"
            recipients = ["gusrab@gmail.com", "Shei3.1416@tuta.io"]
            smtp.sendmail("gusrab@gmail.com", recipients, msg)
            print(f"{coin} {type} Alarm Triggered - Email Sent")
    except Exception as e:
        print("Something went wrong... {}".format(e))


#                   -------- Binance Historical OHLC Import Tool -------


cryptos = [
    "BTC/BUSD",
    "ETH/BUSD",
    "AVAX/BUSD",
    "BNB/BUSD",
    "SOL/BUSD",
    "LUNA/BUSD",
    "ADA/BUSD",
]
sample_freq = "1m"
since = 200  # Expressed in hours
# loc_folder = 'C:/Users/USER/PycharmProjects/pythonProject/'
loc_folder = "Results/"
t_list = ("3h", "1h", "15min")
str_dic = {}

for x in range(len(cryptos)):
    for y in range(len(t_list)):
        file_n = (
            f'TA Results {cryptos[x].replace("/", "")} '
            "- {t_list[y]} 0.7.csv"
        )
        dt = pd.read_csv(f"{loc_folder}{file_n}")
        dt = dt.astype(ASTYPE_DICT)
        dt = dt.join(dt["adx_l"].rename("vpt_adx"))
        dt = dt.join(dt["std_m"].rename("mi_std"))
        dt = dt.join(dt["std_m"].rename("fi_std"))
        dt = dt.join(dt["std_m"].rename("eom_std"))
        dt = dt[DT_OUT_COLUMNS]
        dt.columns = [f" {x}" for x in dt.columns]
        dt = dt.drop_duplicates(subset=" Accuracy")
        r = len(dt) if len(dt) < 5 else 4
        a_idx = [dt[" Accuracy"].nlargest(x + 1).index[-1] for x in range(r)]
        a_list = [dt[" Accuracy"][a_idx[x]] for x in range(r)]
        e_list = [dt.loc[a_idx[x]][5] for x in range(r)]

        if (a_list[0] - a_list[-1]) > 10:
            a_idx.pop()
            a_list.pop()
            e_list.pop()

        score_list = [
            (a_list[x] - a_list[-1]) * (0.1 * e_list[x]) + e_list[x]
            for x in range(len(e_list))
        ]
        idx_max = a_idx[score_list.index(max(score_list))]

        zone = dt.loc[idx_max][8]
        trend = dt.loc[idx_max][9]
        signal = dt.loc[idx_max][10]
        str_dic[f"{cryptos[x]}{t_list[y]}"] = [
            [None, None],
            [None, None],
            [None, None],
        ]

        if zone == "Null":
            str_dic[f"{cryptos[x]}{t_list[y]}"][0][0] = None
            str_dic[f"{cryptos[x]}{t_list[y]}"][0][1] = None
        else:
            zone_f = " " + zone[:-2].lower()
            str_dic[f"{cryptos[x]}{t_list[y]}"][0][0] = zone
            str_dic[f"{cryptos[x]}{t_list[y]}"][0][1] = list(
                dt.loc[idx_max].filter(like=f"{zone_f}")
            )

        if trend == "Null":
            str_dic[f"{cryptos[x]}{t_list[y]}"][1][0] = None
            str_dic[f"{cryptos[x]}{t_list[y]}"][1][1] = None
        else:
            trend_f = " " + trend[:-2].lower()
            str_dic[f"{cryptos[x]}{t_list[y]}"][1][0] = trend
            str_dic[f"{cryptos[x]}{t_list[y]}"][1][1] = list(
                dt.loc[idx_max].filter(like=f"{trend_f}")
            )

        if signal == "Null":
            str_dic[f"{cryptos[x]}{t_list[y]}"][2][0] = None
            str_dic[f"{cryptos[x]}{t_list[y]}"][2][1] = None
        else:
            signal_f = " " + signal[:-2].lower()
            str_dic[f"{cryptos[x]}{t_list[y]}"][2][0] = signal
            str_dic[f"{cryptos[x]}{t_list[y]}"][2][1] = list(
                dt.loc[idx_max].filter(like=f"{signal_f}")
            )

cdf = pd.DataFrame([0 for x in range(len(t_list) * len(cryptos))])
s_list = [0 for x in range(len(cryptos))]

keep_alive()

while True:
    try:
        for x in range(len(cryptos)):
            n = -1
            cdf = pd.DataFrame([0 for x in range(len(t_list))])
            print(cryptos[x])
            dfx = fetch_cryptodata_from_exchange(
                loc_folder,
                exchange="binance",
                cryptos=[cryptos[x]],
                sample_freq=sample_freq,
                since_hours=since,
                page_limit=1000,
            )

            print(f"{cryptos[x]} Dataset Span: {int(len(dfx))/ 60} hours")

            for y in range(len(t_list)):
                if t_list[y] != "1min":
                    df = (
                        dfx.resample(t_list[y])
                        .agg(
                            {
                                "Open": "first",
                                "High": "max",
                                "Low": "min",
                                "Close": "last",
                                "Volume": "sum",
                            }
                        )
                        .fillna(method="ffill")
                    )
                else:
                    df = dfx

                if len(df) < 14:
                    print(
                        f"Dataset {t_list[y]} is to short to be analyzed"
                        " at the defined sampling"
                    )
                    exit()

                df = set_indicators(df, f"{cryptos[x]}{t_list[y]}", str_dic)
                n += 1
                if df.iloc[-1].sum() == len(df.columns):
                    cdf.iloc[n] = 1
                elif df.iloc[-1].sum() == -len(df.columns):
                    cdf.iloc[n] = -1

            if int(cdf.sum()) == -len(t_list) and s_list[x] == 0:
                send_signal(cryptos[x], "Buy")
                print(f"{cryptos[x]} Buy Alarm Triggered")
                s_list[x] == 1
            elif int(cdf.iloc[0]) == 1 and s_list[x] == 1:
                send_signal(cryptos[x], "Sell")
                print(f"{cryptos[x]} Sell Alarm Triggered")
                s_list[x] == 0

        time.sleep(900)
    except Exception as e:
        print("EXCEPTION on main loop {}".format(e))
        time.sleep(900)
        continue
