
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from datetime import date
import math

def create_future_features(df, steps_ahead, tipe):
    future = df.copy(deep=True)
    for _ in range(steps_ahead):
        last_row = future.iloc[-1]
        new_date = last_row.name + pd.DateOffset(weeks=1 if tipe == "weekly" else 30)
        # st.write(f"Langkah {_+1}: Tambah tanggal {new_date.date()}")
        new_row = {}
        for lag in range(3, 0, -1):
            new_row[f"lag_{lag}"] = future.iloc[-lag]["Quantity"]
        new_row["rolling_mean_3"] = future["Quantity"].iloc[-3:].mean()
        new_row["month"] = new_date.month
        new_df = pd.DataFrame([new_row], index=[new_date])
        future = pd.concat([future, new_df])
    return future.tail(steps_ahead)[['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'month']]

def slugify(text):
    return text.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("@", "").replace("/", "_")

def kode_to_str(kode_terpilih):
    return "-".join(str(k) for k in kode_terpilih)


def main(produk_terpilih, kode_terpilih):
    st.header("Rekomendasi Pembelian Stok")
    periode = st.radio("Pilih Model untuk Demand Forecasting", ["Mingguan", "Bulanan"])
    mode = st.radio("Metode Prediksi", ["Sampai Tanggal Tertentu"])


    from datetime import date
    target_date = st.date_input("Prediksi hingga tanggal:")











    current_stock = st.number_input("Stok Saat Ini", min_value=0, step=1)
    extra_percentage = st.number_input("Persentase Penambahan Stok (%)", min_value=0.0, step=1.0)

    if st.button("Hitung Rekomendasi"):
        slug = slugify(produk_terpilih)
        kode_str = kode_to_str(kode_terpilih)
        tipe = "weekly" if periode == "Mingguan" else "monthly"
        # Load demand models
        sarima_path = f"pickle_save_parameter/model_sarima_{slug}_{kode_str}_{tipe}.pkl"
        xgb_path = f"pickle_save_parameter/model_xgboost_{slug}_{kode_str}_{tipe}.pkl"

        if not os.path.exists(sarima_path) or not os.path.exists(xgb_path):
            st.error("File model demand tidak ditemukan. Pastikan model SARIMA dan XGBoost telah dilatih.")
            return
        # Load demand models
        with open(sarima_path, 'rb') as f:
            sarima = pickle.load(f)
        with open(xgb_path, 'rb') as f:
            xgb = pickle.load(f)

        if mode == "Sampai Tanggal Tertentu":
            last_date = None
            try:
                last_date = sarima["dfsarima"].index[-1]
            except:
                last_date = xgb["historical_data"].index[-1]
            st.write("tanggal terakhir demand", last_date)

            delta_days = (pd.to_datetime(target_date) - last_date).days
            if periode == "Mingguan":
                steps_ahead = max(1, math.ceil(delta_days / 7))
                steps_ahead_hargadaily=max(1, math.ceil(delta_days / 7))
            else:  # Bulanan
                steps_ahead = max(1, math.ceil(delta_days / 30))
                steps_ahead_hargadaily=max(1, math.ceil(delta_days / 7))

            if steps_ahead <= 0:
                st.warning("Tanggal prediksi harus lebih besar dari tanggal data terakhir.")
                return

            st.write(f"📅 Akan memprediksi sebanyak **{steps_ahead} periode** ke depan ({periode.lower()})")


        # Pilih model terbaik
        model_sarima_rmse = sarima["rmse"]
        model_xgb_rmse = xgb["rmse"]
        if model_sarima_rmse < model_xgb_rmse:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            # Ambil parameter dan data historis penuh
            best_cfg = sarima["best_cfg"]
            historical_data = sarima["dfsarima"].copy(deep=True)
            

            # Latih ulang model dengan data historis penuh
            sarima_model_full = SARIMAX(
                historical_data['Quantity'],
                order=best_cfg[0],
                seasonal_order=best_cfg[1],
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarima_result_full = sarima_model_full.fit(disp=False)

            # Lakukan forecast ke depan
            forecast_demand = sarima_result_full.forecast(steps=steps_ahead)
            # st.write("📋 Parameter Model SARIMA Terbaik")
            order, seasonal_order = sarima["best_cfg"]
            # st.write(f"Non-musiman (p,d,q): {order}")
            # st.write(f"Musiman (P,D,Q,m): {seasonal_order}")
            # st.write("forecast demand", forecast_demand)

            demand_model = "SARIMA"
        else:
            full_df = xgb["historical_data"].copy(deep=True)
            # st.write(full_df)
            future_features = create_future_features(full_df, steps_ahead, tipe)
            forecast_demand = xgb["model"].predict(future_features)
            demand_model = "XGBoost"
            # st.write("📋 Parameter XGBoost Terbaik:")
            # st.json(xgb["params"])

        # Hitung Safety Stock
        import numpy as np
        std_dev = np.std(forecast_demand)
        safety_stock = 1.645 * std_dev * np.sqrt(5)

        total_forecast = np.sum(forecast_demand)
        kebutuhan_stok = total_forecast + safety_stock - current_stock

        harga_path = f"pickle_save_parameter_hargabeli/model_xgb_hargabeli_{slug}_{kode_str}_weekly.pkl"
        if not os.path.exists(harga_path):
            st.error("File model harga beli tidak ditemukan. Pastikan model XGBoost untuk harga beli telah dilatih.")
            return
        
        from sqlalchemy import create_engine
        engine = create_engine("mysql+pymysql://root:@localhost/tifalanggeng")
        query = f"""
        SELECT * 
        FROM hargaoilweekly
        """

        dfoilweekly = pd.read_sql(query, con=engine, parse_dates=['Date'])


        query = f"""
        SELECT * 
        FROM hargaoildaily
        """

        dfoildaily = pd.read_sql(query, con=engine, parse_dates=['Date'])

        query = f"""
        SELECT * 
        FROM hargappweekly
        """

        dfppweekly = pd.read_sql(query, con=engine, parse_dates=['Date'])

        query = f"""
        SELECT * 
        FROM hargappdaily
        """

        dfppdaily = pd.read_sql(query, con=engine, parse_dates=['Date'])

        query = f"""
        SELECT * 
        FROM kurscnyusdweekly
        """

        kursweekly = pd.read_sql(query, con=engine, parse_dates=['Date'])

        query = f"""
        SELECT * 
        FROM kurscnyusddaily
        """

        kursdaily = pd.read_sql(query, con=engine, parse_dates=['Date'])

        # st.write(dfppweekly)
        # st.write(kursweekly)

        dfppweeklyusd = pd.merge(dfppweekly, kursweekly, on="Date", suffixes=("_PP", "_Kurs"))
        dfppweeklyusd["Price_PP_USD"] = dfppweeklyusd["Price_PP"] * dfppweeklyusd["Price_Kurs"]

        dfppdailyusd = pd.merge(dfppdaily, kursdaily[["Date", "Price"]], on="Date", suffixes=("_PP", "_Kurs"))
        dfppdailyusd["Price_PP_USD"] = dfppdailyusd["Price_PP"] * dfppdailyusd["Price_Kurs"]
        

        # Gabungkan kedua dataset berdasarkan tanggal
        dfoilppweekly = pd.merge(dfppweeklyusd[["Date", "Price_PP_USD"]], dfoilweekly[["Date", "Price"]], on="Date", suffixes=("_PP", "_Oil"))

        # Atur Date sebagai index
        dfoilppweekly.set_index("Date", inplace=True)
        dfoilppweekly.index = pd.to_datetime(dfoilppweekly.index)

        # Sort berdasarkan tanggal (penting!)
        dfoilppweekly.sort_index(inplace=True)

            # Gabungkan kedua dataset berdasarkan tanggal
        dfoilppdaily = pd.merge(dfppdailyusd[["Date", "Price_PP_USD"]], dfoildaily[["Date", "Price"]], on="Date", suffixes=("_PP", "_Oil"))

        # Atur Date sebagai index
        dfoilppdaily.set_index("Date", inplace=True)
        dfoilppdaily.index = pd.to_datetime(dfoilppdaily.index)

        # Sort berdasarkan tanggal (penting!)
        dfoilppdaily.sort_index(inplace=True)
        # st.write(dfoilppdaily)



        # Ambil model harga beli mingguan
        with open(harga_path, "rb") as f:
            xgb_harga = pickle.load(f)
        model_harga = xgb_harga["model"]
        hist_harga = xgb_harga["historical_data"].copy(deep=True)
        # st.write("hist", hist_harga)

        # Load model prediksi PP dan Oil
        with open("pickle_save_parameter_hargabeli/model_xgb_oil_weekly.pkl", "rb") as f:
            model_oil = pickle.load(f)
        with open("pickle_save_parameter_hargabeli/model_xgb_pp_weekly.pkl", "rb") as f:
            model_pp = pickle.load(f)

        
        df_oilpp = dfoilppweekly.copy(deep=True)  # ganti dengan koneksi ke database real
        df_oilpp = df_oilpp.sort_index()
        # st.write(df_oilpp)

        start_date = hist_harga.index[-1]  # default
        st.write("tanggal terakhir harga beli", start_date)

        # Kemudian hitung selisih hari terhadap tanggal target
        delta_days = (pd.to_datetime(target_date) - start_date).days
        steps_for_price = max(1, math.ceil(delta_days / 7))

        
        # st.write("INIII", steps_for_price)
        last_row = hist_harga.iloc[-1].copy()
        # st.write("lastrow", last_row)
        future_rows = []
        

        future_dates = pd.date_range(start=start_date + pd.DateOffset(weeks=1), periods=steps_for_price, freq='W')

        # Atur Date sebagai index
        dfoilweekly.set_index("Date", inplace=True)
        dfoilweekly.index = pd.to_datetime(dfoilweekly.index)

        # Sort berdasarkan tanggal (penting!)
        dfoilweekly.sort_index(inplace=True)
        dfoilweekly.rename(columns={"Price": "Price_Oil"}, inplace=True)

        # Atur Date sebagai index
        dfppweeklyusd.set_index("Date", inplace=True)
        dfppweeklyusd.index = pd.to_datetime(dfppweeklyusd.index)

        # Sort berdasarkan tanggal (penting!)
        dfppweeklyusd.sort_index(inplace=True)
        dfppweeklyusd.drop(columns="Price_PP", inplace=True)
        dfppweeklyusd.rename(columns={"Price_PP_USD": "Price_PP"}, inplace=True)

        # Atur Date sebagai index
        dfoildaily.set_index("Date", inplace=True)
        dfoildaily.index = pd.to_datetime(dfoildaily.index)

        # Sort berdasarkan tanggal (penting!)
        dfoildaily.sort_index(inplace=True)
        dfoildaily.rename(columns={"Price": "Price_Oil"}, inplace=True)

        # Atur Date sebagai index
        dfppdailyusd.set_index("Date", inplace=True)
        dfppdailyusd.index = pd.to_datetime(dfppdailyusd.index)

        # Sort berdasarkan tanggal (penting!)
        dfppdailyusd.sort_index(inplace=True)
        dfppdailyusd.drop(columns="Price_PP", inplace=True)
        dfppdailyusd.rename(columns={"Price_PP_USD": "Price_PP"}, inplace=True)

        hist_oil = dfoilweekly.copy(deep=True)
        hist_pp = dfppweeklyusd.copy(deep=True)
        


        for i, current_date in enumerate(future_dates):
            # current_date = hist_harga.index[-1] + pd.DateOffset(weeks=i+1)
            # st.write("date sekarang", current_date)

            if current_date in df_oilpp.index:
                pp_val = df_oilpp.loc[current_date, "Price_PP_USD"]
                oil_val = df_oilpp.loc[current_date, "Price"]
                # st.write("MASUK")
                # st.write(pp_val)
                # st.write(oil_val)
            else:
                if current_date in hist_oil.index:
                    oil_val = hist_oil.loc[current_date, "Price_Oil"]
                else:
                    # Prediksi PP dan Oil dengan model autoregresif
                    hist_oil_before = hist_oil[hist_oil.index < current_date]
                    input_oil = pd.DataFrame([{
                        "Price_Oil_lag1": hist_oil_before["Price_Oil"].iloc[-1],
                        "Price_Oil_lag2": hist_oil_before["Price_Oil"].iloc[-2],
                        "rolling_mean_3": hist_oil_before["Price_Oil"].iloc[-3:].mean(),
                        "month": hist_oil_before.index[-1].month
                    }])
                    oil_val = model_oil["model"].predict(input_oil)[0]

                    hist_oil = pd.concat([
                        hist_oil,
                        pd.DataFrame([{
                            "Price_Oil": oil_val
                        }], index=[current_date])
                    ])

                    hist_oil = hist_oil.sort_index()



                if current_date in hist_pp.index:
                    pp_val = hist_pp.loc[current_date, "Price_PP"]
                else:
                    hist_pp_before = hist_pp[hist_pp.index < current_date]
                    input_pp = pd.DataFrame([{
                        "Price_PP_lag1": hist_pp_before["Price_PP"].iloc[-1],
                        "Price_PP_lag2": hist_pp_before["Price_PP"].iloc[-2],
                        "rolling_mean_3": hist_pp_before["Price_PP"].iloc[-3:].mean(),
                        "month": hist_pp_before.index[-1].month
                    }])

                    pp_val = model_pp["model"].predict(input_pp)[0]

                    hist_pp = pd.concat([
                        hist_pp,
                        pd.DataFrame([{
                            "Price_PP": pp_val
                        }], index=[current_date])
                    ])

                    hist_pp = hist_pp.sort_index()

            # Prediksi harga beli
            input_features = pd.DataFrame([{
                "Price_PP": pp_val,
                "Price_Oil": oil_val,
                "Price_Beli_lag1": last_row["Price_Beli"]
            }])
            # st.write("🔍 Iterasi ke-", i)
            # st.write(f"Tanggal: {current_date}")
            # st.write(f"Input Features: {input_features}")
            # st.write(f"🛢️ Prediksi PP: {pp_val}, Oil: {oil_val}")
            pred = model_harga.predict(input_features)[0]
            # st.write(f"Prediksi Harga Beli: {pred}")

            # Update hist_harga untuk iterasi berikutnya
            hist_harga = pd.concat([
                hist_harga,
                pd.DataFrame([{
                    "Price_PP": pp_val,
                    "Price_Oil": oil_val,
                    "Price_Beli": pred,
                    "Price_Beli_lag1": last_row["Price_Beli"]
                }], index=[current_date])
            ])

            new_row = {
                "Price_PP": pp_val,
                "Price_Oil": oil_val,
                "Price_Beli": pred
            }
            future_rows.append(new_row)
            last_row = new_row  # update untuk langkah selanjutnya


        future_harga_features = pd.DataFrame(future_rows)
        # st.write("hasil forecast harga beli", future_harga_features)
        harga_forecast = future_harga_features["Price_Beli"]
        # st.write(future_harga_features.head())
        # st.write("histharga", hist_harga)


        # future_harga_features = X_full[['Price_PP', 'Price_Oil', 'Price_Beli_lag1']].tail(steps_for_price)
        # harga_forecast = xgb_harga["model"].predict(future_harga_features)
        harga_model = "XGBoost"
        # st.write("📋 Parameter XGBoost Harga Terbaik:")
        # st.json(xgb_harga["params"])
        harga_history = xgb_harga["historical_data"].copy(deep=True)

        # Hitung tren (slope)
        # Buat indeks waktu untuk forecast ke depan (harga beli)
        start_harga = harga_history.index[-1] + pd.DateOffset(weeks=1)
        future_index_harga = pd.date_range(start=start_harga, periods=steps_for_price, freq="W")
        # st.write("Future index harga", future_index_harga)
        # Siapkan hasil slicing hanya untuk keperluan visualisasi dan analisis tren
        if periode == "Bulanan":
            if len(harga_forecast) >= 4:
                harga_forecast_plot = harga_forecast
                future_index_harga_plot = future_index_harga
            else:
                harga_forecast_plot = harga_forecast
                future_index_harga_plot = future_index_harga
                st.warning("Tidak cukup data mingguan untuk disarikan menjadi bulanan. Menampilkan semua data mingguan.")
        else:
            harga_forecast_plot = harga_forecast
            future_index_harga_plot = future_index_harga

        from sklearn.linear_model import LinearRegression
        import numpy as np

        def hitung_slope_safely(y_series):
            if isinstance(y_series, pd.Series):
                y = y_series.values
            else:
                y = np.array(y_series)

            if len(y) < 2:
                return 0.0, np.nan, np.nan  # Slope, start, end
            
            x = np.arange(len(y)).reshape(-1, 1)
            y = y.reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            slope = model.coef_[0][0]
            return slope, y[0][0], y[-1][0]


        slope_harga, harga_start, harga_end = hitung_slope_safely(harga_forecast_plot)
        y_demand = forecast_demand if isinstance(forecast_demand, np.ndarray) else forecast_demand.values
        slope_demand, demand_start, demand_end = hitung_slope_safely(y_demand)

        # harga_awal = harga_forecast.iloc[0]
        # harga_akhir = harga_forecast.iloc[-1]
        kenaikan_persen = ((harga_end - harga_start) / harga_start) * 100




        naik_harga = slope_harga > 0
        naik_demand = slope_demand > 0


        # === Rekomendasi ===
        st.subheader("📊 Rekomendasi")

        # tambahan = kebutuhan_stok * (extra_percentage / 100)
        # total_direkomendasikan = kebutuhan_stok + tambahan

        tambahan = max(0, kebutuhan_stok * (extra_percentage / 100))
        total_direkomendasikan = max(0, kebutuhan_stok + tambahan)

        if total_direkomendasikan <= 0:
            st.info(f"ℹ️ Stok saat ini ({current_stock:.0f}) mencukupi kebutuhan prediksi ({total_forecast:.0f}). Tidak perlu pembelian.")

        if total_direkomendasikan <= 0:
            st.info("ℹ️ Stok masih cukup untuk melayani pembelian")
        else:

            if naik_demand and naik_harga:
                st.success(f"✅ Disarankan menambah pembelian stok sebesar {total_direkomendasikan:.0f} unit sebelum harga naik.")
            elif naik_demand and not naik_harga:
                st.success(f"✅ Disarankan membeli stok awal sebesar {kebutuhan_stok:.0f} unit dan menambah pembelian sebesar {tambahan:.0f} unit setelah harga turun.")
            elif not naik_demand and naik_harga:
                st.warning(f"⛔ Permintaan menurun dan harga naik. Namun, disarankan tetap membeli stok sesuai kebutuhan minimum sebesar {kebutuhan_stok:.0f} unit agar tetap bisa melayani pelanggan.")
            elif not naik_demand and not naik_harga:
                st.info(f"ℹ️ Permintaan dan harga menurun. Disarankan membeli stok sesuai kebutuhan minimum sebesar {kebutuhan_stok:.0f} unit.")
            else:
                st.info("ℹ️ Rekomendasi campuran: hanya salah satu dari permintaan atau harga yang naik. Pertimbangkan kebijakan internal.")

        # === Ringkasan Angka ===
        st.write(f"📦 Total Forecast Permintaan: {total_forecast:.0f}")
        if kebutuhan_stok <= 0:
            safety_stock = 0
            kebutuhan_stok = 0
            st.write(f"🔐 Safety Stock (95%): {safety_stock:.0f}")
            st.write(f"📉 Kebutuhan Stok Awal: {kebutuhan_stok:.0f}")

        else:
            st.write(f"🔐 Safety Stock (95%): {safety_stock:.0f}")
            st.write(f"📉 Kebutuhan Stok Awal: {kebutuhan_stok:.0f}")
        st.write(f"📈 Tren Permintaan {'naik' if naik_demand else 'turun'} ({slope_demand:+.2f})")
        if not np.isnan(harga_start) and not np.isnan(harga_end):
            st.write(f"📈 Tren Harga {'naik' if naik_harga else 'turun'} (dari {harga_start:.2f} ke {harga_end:.2f}, {slope_harga:+.2f}) HARGA DALAM USD")

        # ============================
        # Estimasi Tanggal Kenaikan/Penurunan Harga Beli (Harian)
        # ============================

        harga_daily_path = f"pickle_save_parameter_hargabeli/model_xgb_hargabeli_{slug}_{kode_str}_daily.pkl"
        if os.path.exists(harga_daily_path):
            with open(harga_daily_path, "rb") as f:
                xgb_harga_daily = pickle.load(f)

        model_harga_daily = xgb_harga_daily["model"]
        hist_daily = xgb_harga_daily["historical_data"].copy(deep=True)
        # st.write("histdaily", hist_daily)


        # Load model fallback untuk Oil & PP harian
        with open("pickle_save_parameter_hargabeli/model_xgb_oil_daily.pkl", "rb") as f:
            model_oil_daily = pickle.load(f)
        with open("pickle_save_parameter_hargabeli/model_xgb_pp_daily.pkl", "rb") as f:
            model_pp_daily = pickle.load(f)

        df_oilpp_daily = dfoilppdaily.copy(deep=True)
        df_oilpp_daily = df_oilpp_daily.sort_index()

        last_date_daily = hist_daily.index[-1]

        future_dates_daily = pd.bdate_range(
            start=last_date_daily + pd.DateOffset(days=1),
            end=target_date
        )
        steps_daily = len(future_dates_daily)

        # st.write("futuredatesdaily", future_dates_daily)

        future_daily_rows = []
        last_row = hist_daily.iloc[-1].copy()

        hist_oil_daily = dfoildaily.copy(deep=True)
        hist_pp_daily = dfppdailyusd.copy(deep=True)




        for i, current_date in enumerate(future_dates_daily):
            # current_date = hist_daily.index[-1] + pd.DateOffset(days=i+1)
            # st.write("date sekarang", current_date)

            if current_date in df_oilpp_daily.index:
                pp_val = df_oilpp_daily.loc[current_date, "Price_PP_USD"]
                oil_val = df_oilpp_daily.loc[current_date, "Price"]
                # st.write("MASUK")
                # st.write(pp_val)
                # st.write(oil_val)
            else:
                if current_date in hist_oil_daily.index:
                    oil_val = hist_oil_daily.loc[current_date, "Price_Oil"]
                else:
                    # Prediksi PP dan Oil dengan model autoregresif
                    hist_oil_before = hist_oil_daily[hist_oil_daily.index < current_date]
                    input_oil = pd.DataFrame([{
                        "Price_Oil_lag1": hist_oil_before["Price_Oil"].iloc[-1],
                        "Price_Oil_lag2": hist_oil_before["Price_Oil"].iloc[-2],
                        "rolling_mean_3": hist_oil_before["Price_Oil"].iloc[-3:].mean(),
                        "month": hist_oil_before.index[-1].month
                    }])

                    oil_val = model_oil_daily["model"].predict(input_oil)[0]
                    # Update hist_oil dan hist_pp dengan hasil prediksi saat ini
                    hist_oil_daily = pd.concat([
                        hist_oil_daily,
                        pd.DataFrame([{
                            "Price_Oil": oil_val
                        }], index=[current_date])
                    ])
                    hist_oil_daily = hist_oil_daily.sort_index()


                if current_date in hist_pp_daily.index:
                    pp_val = hist_pp_daily.loc[current_date, "Price_PP"]
                else:
                    hist_pp_before = hist_pp_daily[hist_pp_daily.index < current_date]
                    input_pp = pd.DataFrame([{
                        "Price_PP_lag1": hist_pp_before["Price_PP"].iloc[-1],
                        "Price_PP_lag2": hist_pp_before["Price_PP"].iloc[-2],
                        "rolling_mean_3": hist_pp_before["Price_PP"].iloc[-3:].mean(),
                        "month": hist_pp_before.index[-1].month
                    }])
                    # st.write("inputoil", input_oil)
                    # st.write("inputpp", input_pp)
                    
                    pp_val = model_pp_daily["model"].predict(input_pp)[0]
                    # st.write("oilval", oil_val)
                    # st.write("ppval", pp_val)
                    # st.write("🔍 Iterasi ke-", i)
                    # st.write(f"Tanggal: {current_date}")

                    

                    hist_pp_daily = pd.concat([
                        hist_pp_daily,
                        pd.DataFrame([{
                            "Price_PP": pp_val
                        }], index=[current_date])
                    ])

                    hist_pp_daily = hist_pp_daily.sort_index()

            # Prediksi harga beli harian
            input_features = pd.DataFrame([{
                "Price_PP": pp_val,
                "Price_Oil": oil_val,
                "Price_Beli_lag1": last_row["Price_Beli"]
            }])
            # st.write("🔍 Iterasi ke-", i)
            # st.write(f"Tanggal: {current_date}")
            # st.write(f"Input Features: {input_features}")
            # st.write(f"🛢️ Prediksi PP: {pp_val}, Oil: {oil_val}")
            pred = model_harga_daily.predict(input_features)[0]
            # st.write(f"Prediksi Harga Beli: {pred}")

            hist_daily = pd.concat([
                hist_daily,
                pd.DataFrame([{
                    "Price_PP": pp_val,
                    "Price_Oil": oil_val,
                    "Price_Beli": pred,
                    "Price_Beli_lag1": last_row["Price_Beli"]
                }], index=[current_date])
            ])

            new_row = {
                "Price_PP": pp_val,
                "Price_Oil": oil_val,
                "Price_Beli": pred
            }
            future_daily_rows.append(new_row)
            last_row = new_row

        future_daily_df = pd.DataFrame(future_daily_rows, index=future_dates_daily)
        # st.write("futuredailydf", future_daily_df)
        harga_forecast_daily = future_daily_df["Price_Beli"]
        # st.write("histharga", hist_daily)


        # st.write("DIAGNOSA CEPAT")
        # plt.figure(figsize=(10, 4))
        # plt.plot(future_dates_daily, future_daily_df["Price_PP"], label="PP Daily")
        # plt.plot(future_dates, future_harga_features["Price_PP"], label="PP Weekly")  # asumsi future_dates = future weekly

        # plt.xlabel("Tanggal")
        # plt.ylabel("Price_PP")
        # plt.title("Perbandingan Input Price_PP Daily vs Weekly")
        # plt.legend()
        # plt.grid(True)
        # st.pyplot(plt)

        # plt.plot(hist_daily["Price_Beli"], label="Harga Harian")
        # plt.plot(hist_harga["Price_Beli"], label="Harga Mingguan")
        # plt.title("Harga Beli Training Harian vs Mingguan")
        # plt.legend()
        # st.pyplot(plt)

        from sklearn.linear_model import LinearRegression

        def slope(y):
            x = np.arange(len(y)).reshape(-1, 1)
            return LinearRegression().fit(x, y.reshape(-1, 1)).coef_[0][0]

        slope_weekly = slope(harga_forecast.values)
        slope_daily = slope(harga_forecast_daily.values)
        # st.write(f"📈 Slope forecast weekly: {slope_weekly:.4f}")
        # st.write(f"📉 Slope forecast daily: {slope_daily:.4f}")

        temp = xgb_harga_daily["historical_data"].copy(deep=True)
        # st.write(temp)
        harga_awal_daily = temp["Price_Beli"].iloc[-1]
        threshold = harga_awal_daily * 0.01  # 5%
        diffs = np.diff(harga_forecast_daily)
        # st.write(diffs)
        # up_idx = np.where(diffs > 0)[0]
        # down_idx = np.where(diffs < 0)[0]
        up_idx = np.where(diffs > threshold)[0]
        down_idx = np.where(diffs < -threshold)[0]

        first_up = up_idx[0] if len(up_idx) > 0 else np.inf
        first_down = down_idx[0] if len(down_idx) > 0 else np.inf
        # st.write(up_idx)
        # st.write(down_idx)
        
        # threshold = 1e-2

        st.subheader("📅 Estimasi Perubahan Harga (Harian)")
        if first_up < first_down:
            tanggal_perubahan = future_daily_df.index[first_up + 1]
            st.success(f"📈 Estimasi terjadi **kenaikan harga pertama** di data harian: **{tanggal_perubahan.date()}**")
        elif first_down < first_up:
            tanggal_perubahan = future_daily_df.index[first_down + 1]
            st.warning(f"📉 Estimasi terjadi **penurunan harga pertama** di data harian: **{tanggal_perubahan.date()}**")
        elif np.all(np.abs(diffs) < threshold):
            st.info("Harga diprediksi stabil (perubahan harian < 1%).")
        else:
            st.write("🔎 Tidak ditemukan perubahan harga signifikan.")

        # ==============================
        # 📈 Grafik Forecasting Harga Beli Harian (XGBoost)
        # ==============================

        st.subheader("📈 Grafik Forecasting Harga Beli Harian")

        # Simpan salinan data historis sebelum ditambah forecast
        hist_daily_actual = xgb_harga_daily["historical_data"].copy()
        X_harga_hist_daily = hist_daily_actual[["Price_PP", "Price_Oil", "Price_Beli_lag1"]]
        y_true_harga_hist = hist_daily_actual["Price_Beli"]
        y_fit_harga_hist = model_harga_daily.predict(X_harga_hist_daily)


        # Plot
        fig_daily, ax_daily = plt.subplots(figsize=(10, 4))

        # Plot historis
        ax_daily.plot(hist_daily_actual.index, y_true_harga_hist, label="Historis")

        # Plot in-sample fit
        ax_daily.plot(hist_daily_actual.index, y_fit_harga_hist, "--", label="In-Sample Fit")

        # Plot forecast
        ax_daily.plot(future_daily_df.index, harga_forecast_daily, label="Forecast", color="green")

        # Styling
        ax_daily.set_title("Forecast Harga Beli Harian (XGBoost)")
        ax_daily.set_xlabel("Tanggal")
        ax_daily.set_ylabel("Harga Beli")
        ax_daily.legend()
        ax_daily.grid(True)

        # Tampilkan ke Streamlit
        st.pyplot(fig_daily)


        # Plot
        fig_daily, ax_daily = plt.subplots(figsize=(10, 4))

        # Plot forecast
        ax_daily.plot(future_daily_df.index, harga_forecast_daily, label="Forecast", color="green")

        # Styling
        ax_daily.set_title("Forecast Harga Beli Harian (XGBoost)")
        ax_daily.set_xlabel("Tanggal")
        ax_daily.set_ylabel("Harga Beli")
        ax_daily.legend()
        ax_daily.grid(True)

        # Tampilkan ke Streamlit
        st.pyplot(fig_daily)



            



    
        # === Visualisasi ===
        # Data historis
        if demand_model == "SARIMA":
            demand_history = sarima["dfsarima"].copy(deep=True)
            y_true = demand_history['Quantity']
            y_pred = sarima["model_fit"].fittedvalues
            y_train_demand = y_pred  # <-- Tambahkan ini
        else:
            demand_history = xgb["historical_data"].copy(deep=True)
            y_true = demand_history['Quantity']
            X_all = demand_history[['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'month']]
            y_pred = xgb["model"].predict(X_all)
            y_train_demand = y_pred  # <-- Tambahkan ini

        rmse_demand = np.sqrt(mean_squared_error(y_true[-len(y_pred):], y_pred))
        mape_demand = mean_absolute_percentage_error(y_true[-len(y_pred):], y_pred) * 100
        def smape(y_true, y_pred):
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            diff = np.abs(y_true - y_pred) / denominator
            diff[denominator == 0] = 0.0  # untuk menghindari division by zero
            return np.mean(diff) * 100
        smape_demand = smape(y_true[-len(y_pred):], y_pred)


        # Harga beli
        
        X_harga_hist = harga_history[['Price_PP', 'Price_Oil', 'Price_Beli_lag1']]
        y_harga_hist = harga_history['Price_Beli']
        y_train_harga = xgb_harga["model"].predict(X_harga_hist)


        rmse_harga = np.sqrt(mean_squared_error(y_harga_hist, y_train_harga))
        mape_harga = mean_absolute_percentage_error(y_harga_hist, y_train_harga) * 100

        # if not isinstance(harga_history.index, pd.DatetimeIndex):
        #     harga_history.index = pd.to_datetime(harga_history.index)

        # if not isinstance(demand_history.index, pd.DatetimeIndex):
        #     demand_history.index = pd.to_datetime(demand_history.index)

        # st.write(harga_history)
        # st.write(demand_history)

        st.write("📉 **Evaluasi Predisi Permintaan (Train)**")
        st.write(f"RMSE: {rmse_demand:.2f}")
        # st.write(f"MAPE: {mape_demand:.2f}%")
        st.write(f"SMAPE: {smape_demand:.2f}%")

        st.write("📉 **Evaluasi Prediksi Harga Beli (Train)**")
        st.write(f"RMSE: {rmse_harga:.2f}")
        st.write(f"MAPE: {mape_harga:.2f}%")
        
        # Buat indeks waktu untuk forecast ke depan
        # start_demand = demand_history.index[-1] + pd.DateOffset(weeks=1 if periode == "Mingguan" else 30)
        # future_index_demand = pd.date_range(start=start_demand, periods=steps_ahead, freq="W" if periode == "Mingguan" else "MS")
        if periode == "Mingguan":
            start_demand = (demand_history.index[-1] + pd.DateOffset(weeks=1)).normalize()
            future_index_demand = pd.date_range(start=start_demand, periods=steps_ahead, freq="W")
        else:
            start_demand = (demand_history.index[-1] + pd.DateOffset(months=1)).replace(day=1)
            future_index_demand = pd.date_range(start=start_demand, periods=steps_ahead, freq="ME")  # Month Start

        # st.write("start_demand:", start_demand)
        # st.write("steps_ahead:", steps_ahead)
        # st.write("future_index_demand:", future_index_demand)
        # st.write(forecast_demand)

        # start_harga = harga_history.index[-1] + pd.DateOffset(weeks=1)
        # future_index_harga = pd.date_range(start=start_harga, periods=steps_for_price, freq="W")
        # if periode == "Bulanan":
        #     harga_forecast = harga_forecast[3::4]  # Ambil minggu ke-4 untuk tiap bulan
        #     future_index_harga = future_index_harga[3::4]  # Sesuaikan jumlah waktu
        # st.write("Tanggal terakhir data historis:", last_date)
        # st.write("Tanggal target:", target_date)
        # st.write("Steps ahead:", steps_ahead)

        # ========== GRAFIK DEMAND ==========
        st.subheader("📈 Grafik Forecasting Permintaan")
        fig_demand, ax_demand = plt.subplots(figsize=(10, 4))

        # Index untuk in-sample fit (SARIMA/XGBoost)
        index_fit_demand = demand_history.index[-len(y_train_demand):]
        # fitted_aligned = pd.Series(np.nan, index=demand_history.index)
        # fitted_aligned.iloc[-len(y_train_demand):] = y_train_demand
        

        ax_demand.plot(demand_history.index, demand_history['Quantity'], label='Historis')
        ax_demand.plot(index_fit_demand, y_train_demand, '--', label='In-Sample Fit')
        # ax_demand.plot(fitted_aligned.index, fitted_aligned, '--', label='In-Sample Fit')
        ax_demand.plot(future_index_demand, forecast_demand, label='Forecast')
        # st.write(future_index_demand, forecast_demand)

        ax_demand.set_title(f"Forecast Demand ({demand_model})")
        # ax_demand.set_xlabel("Time")
        ax_demand.set_xlabel("Tanggal")
        ax_demand.set_ylabel("Jumlah Permintaan")
        ax_demand.legend()
        ax_demand.grid(True)
        st.pyplot(fig_demand)


        # fig_demand, ax_demand = plt.subplots(figsize=(10, 4))

        # ax_demand.plot(future_index_demand, forecast_demand, label='Forecast')
        # # st.write(future_index_demand, forecast_demand)

        # ax_demand.set_title(f"Forecast Demand ({demand_model})")
        # # ax_demand.set_xlabel("Time")
        # ax_demand.set_xlabel("Tanggal")
        # ax_demand.set_ylabel("Jumlah Permintaan")
        # ax_demand.legend()
        # ax_demand.grid(True)
        # st.pyplot(fig_demand)




        

        # Plot tren permintaan
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        index_demand = forecast_demand.index if hasattr(forecast_demand, "index") else np.arange(len(forecast_demand))
        ax2.plot(index_demand, y_demand, label='Forecast Demand', color='green')

        # Garis slope permintaan
        x_vals_d = np.arange(len(y_demand))
        y_vals_d = slope_demand * x_vals_d + demand_start
        ax2.plot(index_demand, y_vals_d, '--', label='Linear Trend (Demand)', color='orange')

        ax2.set_title('Tren Permintaan dan Slope')
        ax2.set_ylabel('Jumlah')
        ax2.set_xlabel('Tanggal')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)























        # ========== GRAFIK HARGA BELI ==========
        st.subheader("📈 Grafik Forecasting Harga Beli")
        fig_harga, ax_harga = plt.subplots(figsize=(10, 4))

        # Index untuk in-sample fit harga beli
        index_fit_harga = harga_history.index

        ax_harga.plot(harga_history.index, y_harga_hist, label='Historis')
        ax_harga.plot(index_fit_harga, y_train_harga, '--', label='In-Sample Fit')
        ax_harga.plot(future_dates, harga_forecast_plot, label='Forecast')
        # st.write(future_dates, harga_forecast_plot)
        ax_harga.set_title(f"Forecast Harga Beli ({harga_model})")
        ax_harga.set_xlabel("Time")
        ax_harga.legend()
        ax_harga.grid(True)
        st.pyplot(fig_harga)


        # fig_harga, ax_harga = plt.subplots(figsize=(10, 4))

        # # Index untuk in-sample fit harga beli
        # index_fit_harga = harga_history.index

        # ax_harga.plot(future_dates, harga_forecast_plot, label='Forecast')
        # # st.write(future_dates, harga_forecast_plot)
        # ax_harga.set_title(f"Forecast Harga Beli ({harga_model})")
        # ax_harga.set_xlabel("Time")
        # ax_harga.legend()
        # ax_harga.grid(True)
        # st.pyplot(fig_harga)


        # Plot tren harga
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(harga_forecast_plot.index, harga_forecast_plot, label='Harga Forecast', color='blue')

        # Garis slope (regresi linear)
        x_vals = np.arange(len(harga_forecast_plot))
        y_vals = slope_harga * x_vals + harga_start
        ax1.plot(harga_forecast_plot.index, y_vals, '--', label='Linear Trend (Harga)', color='red')

        ax1.set_title('Tren Harga dan Slope')
        ax1.set_ylabel('Harga')
        ax1.set_xlabel('Tanggal')
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)



