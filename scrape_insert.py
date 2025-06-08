# import yfinance as yf

# # Ambil data harga harian crude oil WTI
# oil = yf.Ticker("CL=F")
# data = oil.history(period="1d")  # harga 1 hari terakhir

# # Ambil harga terakhir (closing)
# harga_terbaru = data['Close'].iloc[-1]

# print("Harga minyak terbaru (WTI):", harga_terbaru)

# # https://api.freecurrencyapi.com/v1/latest?apikey=fca_live_tTroQnxXK6EB0NqV1ZP5MjpeZx3C1N4K6XXpv9Kt&currencies=IDR

# # https://api.freecurrencyapi.com/v1/latest?apikey=fca_live_tTroQnxXK6EB0NqV1ZP5MjpeZx3C1N4K6XXpv9Kt&currencies=USD&base_currency=CNY

# import requests
# import pandas as pd
# from sqlalchemy import create_engine
# from datetime import datetime

# # ========================
# # KONFIGURASI
# # ========================
# api_key = "fca_live_tTroQnxXK6EB0NqV1ZP5MjpeZx3C1N4K6XXpv9Kt"

# # USD ke IDR
# url_usd_idr = f"https://api.freecurrencyapi.com/v1/latest?apikey={api_key}&currencies=IDR"
# res_usd_idr = requests.get(url_usd_idr).json()
# kurs_usd_idr = res_usd_idr['data']['IDR']

# # CNY ke USD
# url_cny_usd = f"https://api.freecurrencyapi.com/v1/latest?apikey={api_key}&currencies=USD&base_currency=CNY"
# res_cny_usd = requests.get(url_cny_usd).json()
# kurs_cny_usd = res_cny_usd['data']['USD']

# today = datetime.today().date()
# df = pd.DataFrame([{
#     'Date': today,
#     'USD_IDR': kurs_usd_idr,
#     'CNY_USD': kurs_cny_usd
# }])

# print(df)

import requests
import yfinance as yf
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.types import Date, Float


def update_data_to_db(period="daily"):
    assert period in ["daily", "weekly"], "period harus 'daily' atau 'weekly'"

    # === Konfigurasi DB ===
    engine = create_engine("mysql+pymysql://root:@localhost/tifalanggeng")
    dtype = {"Date": Date(), "Price": Float()}
    table_suffix = "daily" if period == "daily" else "weekly"

    # Tanggal hari ini
    today = datetime.today().date()
    table_suffix = "daily" if period == "daily" else "weekly"

    # === 1. Harga Oil dari Yahoo Finance ===
    oil_df = yf.download("CL=F", period="1d", interval="1d")
    if oil_df.empty:
        print("❌ Gagal ambil data minyak.")
        return
    # Ambil nilai float dari kolom Close
    print("oil df", oil_df)
    price_oil = oil_df["Close"].iloc[-1].item()
    df_oil = pd.DataFrame([{"Date": today, "Price": price_oil}])
    print(df_oil)
    df_oil.to_sql(f"hargaoil{table_suffix}", con=engine, if_exists="append", index=False, dtype=dtype)
    print(f"✅ Harga minyak ({period}) disimpan: {price_oil}")

    # === 2. Kurs dari FreeCurrencyAPI ===
    api_key = "fca_live_tTroQnxXK6EB0NqV1ZP5MjpeZx3C1N4K6XXpv9Kt"
    try:
        url_usd_idr = f"https://api.freecurrencyapi.com/v1/latest?apikey={api_key}&currencies=IDR"
        url_cny_usd = f"https://api.freecurrencyapi.com/v1/latest?apikey={api_key}&currencies=USD&base_currency=CNY"
        res_usd_idr = requests.get(url_usd_idr).json()
        res_cny_usd = requests.get(url_cny_usd).json()

        price_usdidr = res_usd_idr['data']['IDR']
        price_cnyusd = res_cny_usd['data']['USD']

        # Simpan ke masing-masing tabel
        pd.DataFrame([{"Date": today, "Price": price_usdidr}])\
            .to_sql(f"kursidrusd{table_suffix}", con=engine, if_exists="append", index=False, dtype=dtype)

        pd.DataFrame([{"Date": today, "Price": price_cnyusd}])\
            .to_sql(f"kurscnyusd{table_suffix}", con=engine, if_exists="append", index=False, dtype=dtype)

        print(f"✅ Kurs USD/IDR ({period}): {price_usdidr}, Kurs CNY/USD: {price_cnyusd}")
    except Exception as e:
        print("❌ Gagal ambil kurs:", e)
        return

    # === 3. Prediksi Harga PP dari model XGBoost ===
    model_path = os.path.join("pickle_save_parameter_hargabeli", f"model_pred_pp_xgb_{period}.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Model PP ({period}) belum tersedia: {model_path}")
        return

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model_pp = model_data["model"]

    price_pp = model_pp.predict(pd.DataFrame({"Price_Oil": [price_oil]}))[0]
    df_pp = pd.DataFrame([{"Date": today, "Price": price_pp}])
    df_pp.to_sql(f"hargapp{table_suffix}", con=engine, if_exists="append", index=False, dtype=dtype)
    print(f"✅ Harga PP ({period}) diprediksi & disimpan: {price_pp:.2f}")

print("daily")
update_data_to_db("daily")
print("weekly")
update_data_to_db("weekly")




