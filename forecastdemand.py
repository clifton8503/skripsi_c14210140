def main(produk_terpilih, kode_terpilih):
    # %% import library
    import streamlit as st
    import numpy as np
    import pandas as pd
    import mysql.connector
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    import numpy as np
    import matplotlib.pyplot as plt
    from xgboost import XGBRegressor
    import optuna
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import os
    import pickle
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sqlalchemy import create_engine
    from sqlalchemy.dialects.mysql import VARCHAR, FLOAT, DATETIME, INTEGER
    import mysql.connector
    import os
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    from sklearn.model_selection import TimeSeriesSplit
    from optuna.study import MaxTrialsCallback
    from optuna.pruners import SuccessiveHalvingPruner

    # %% upload file baru
    st.header("📤 Upload Data Penjualan Baru & Retrain Model")

    uploaded_file = st.file_uploader("Upload file Excel penjualan (.xlsx)", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # === STEP 1: Baca Excel ===
            penjualan = pd.read_excel(uploaded_file)
            penjualan = penjualan.dropna()
            penjualan['Tgl Faktur'] = penjualan['Tgl Faktur'].astype(str)

            # Replace nama bulan
            penjualan['Tgl Faktur'] = penjualan['Tgl Faktur'].replace({
                'Jan': 'Jan', 'Feb': 'Feb', 'Mar': 'Mar', 'Apr': 'Apr',
                'Mei': 'May', 'Jun': 'Jun', 'Jul': 'Jul', 'Agu': 'Aug',
                'Sep': 'Sep', 'Okt': 'Oct', 'Nop': 'Nov', 'Des': 'Dec'
            }, regex=True)

            # Konversi ke datetime
            penjualan['Tgl Faktur'] = pd.to_datetime(penjualan['Tgl Faktur'], errors='coerce')

            # Konversi tipe data
            penjualan['Kuantitas'] = penjualan['Kuantitas'].str.replace('.', '', regex=False).str.replace(',00', '', regex=False).astype(int)
            penjualan['Jumlah'] = penjualan['Jumlah'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
            penjualan['Jumlah Data'] = penjualan['Jumlah Data'].str.replace('.', '', regex=False).str.replace(',00', '', regex=False).astype(int)
            penjualan['Nilai HPP'] = penjualan['Nilai HPP'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
            penjualan['Harga satuan'] = penjualan['Harga satuan'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

            # Ganti nama kolom
            penjualan.rename(columns={
                'No. Faktur': 'Invoice_Number',
                'Tgl Faktur': 'Invoice_Date',
                'Kuantitas': 'Quantity',
                'Jumlah': 'Total_Amount',
                'Unit 1 Barang': 'Unit',
                'Nilai HPP': 'nilai_hpp',
                'Jumlah Data': 'jumlah_data',
                'Keterangan Barang': 'keterangan_barang',
                'No. Barang': 'no_barang',
                'Harga satuan': 'harga_satuan'
            }, inplace=True)

            penjualan = penjualan.sort_values(by='Invoice_Date')

            # === STEP 2: Simpan ke database MySQL ===
            engine = create_engine("mysql+pymysql://root:@localhost/tifalanggeng")

            dtype = {
                'Invoice_Number': VARCHAR(50),
                'Invoice_Date': DATETIME(),
                'Quantity': INTEGER(),
                'Unit': VARCHAR(20),
                'Total_Amount': FLOAT(),
                'nilai_hpp': FLOAT(),
                'jumlah_data': INTEGER(),
                'keterangan_barang': VARCHAR(255),
                'no_barang': VARCHAR(20),
                'harga_satuan': FLOAT()
            }

            st.write(f"📊 Jumlah data baru yang akan ditambahkan: {len(penjualan)}")
            penjualan.to_sql("penjualan", con=engine, if_exists="append", index=False, dtype=dtype)
            st.success("✅ Data berhasil diunggah dan disimpan ke MySQL!")

            # === STEP 3: Trigger retrain setelah upload ===
            st.info("Retrain akan dijalankan ulang berdasarkan data baru.")

            # Optional: hapus model lama
            model_dir = "pickle_save_parameter"
            for file in os.listdir(model_dir):
                if file.endswith(".pkl"):
                    os.remove(os.path.join(model_dir, file))
            st.warning("🗑️ Model lama dihapus. Silakan pilih produk untuk retrain ulang.")

            # Optional: refresh Streamlit (pakai st.rerun)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Gagal memproses file: {e}")


    # %% dropdown

    kode_terpilih_str = ','.join(f"'{str(k)}'" for k in kode_terpilih)


    # # Koneksi ke MySQL
    # conn = mysql.connector.connect(
    #     host="localhost",
    #     user="root",
    #     password="",
    #     database="tifalanggeng"
    # )
    engine = create_engine("mysql+pymysql://root:@localhost/tifalanggeng")
    # Ambil data dari MySQL
    query = f"""
    SELECT Invoice_Date, Quantity 
    FROM penjualan 
    WHERE no_barang IN ({kode_terpilih_str})
    ORDER BY Invoice_Date
    """

    # df = pd.read_sql(query, conn, parse_dates=['Invoice_Date'])
    df = pd.read_sql(query, con=engine, parse_dates=['Invoice_Date'])
    # df.set_index('Invoice_Date', inplace=True)
    # # Tampilkan data hasil filter
    # st.write("Data Penjualan Terpilih:", df)
    # %% forecasting
    weekly_sales = df.resample('W', on='Invoice_Date')["Quantity"].sum().reset_index()
    # st.write(weekly_sales)
    monthly_sales = df.resample('ME', on='Invoice_Date')["Quantity"].sum().reset_index()

    plt.figure(figsize=(20, 5))
    plt.plot(weekly_sales["Invoice_Date"], weekly_sales["Quantity"], marker='o', linestyle='-')

    plt.xlabel("Tanggal")
    plt.ylabel("Total Penjualan")
    plt.title("DATA ASLI Grafik Penjualan per Minggu")
    plt.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(plt)

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_sales["Invoice_Date"], monthly_sales["Quantity"], marker='o', linestyle='-')

    plt.xlabel("Tanggal")
    plt.ylabel("Total Penjualan")
    plt.title("DATA ASLI Grafik Penjualan per Bulan")
    plt.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.write("Null weekly", weekly_sales.isnull().sum())
    st.write("0 weekly", (weekly_sales['Quantity'] == 0).sum())
    st.write("Null monthly", monthly_sales.isnull().sum())
    st.write("0 monthly", (monthly_sales['Quantity'] == 0).sum())
    if st.button("Start"):
        
        # weekly_sales['Quantity'] = weekly_sales['Quantity'].replace(0, np.nan)
        # weekly_sales['Quantity'] = weekly_sales['Quantity'].interpolate(method='linear')
        # monthly_sales['Quantity'] = monthly_sales['Quantity'].replace(0, np.nan)
        # monthly_sales['Quantity'] = monthly_sales['Quantity'].interpolate(method='linear')
        # libur = ['2023-04-30', '2024-04-14']
        # libur = pd.to_datetime(libur)
        # weekly_sales = weekly_sales[~weekly_sales['Invoice_Date'].isin(libur)]


        # plt.figure(figsize=(20, 5))
        # plt.plot(weekly_sales["Invoice_Date"], weekly_sales["Quantity"], marker='o', linestyle='-')

        # plt.xlabel("Tanggal")
        # plt.ylabel("Total Penjualan")
        # plt.title("Grafik Penjualan per Minggu")
        # plt.grid(True)
        # plt.xticks(rotation=45)
        # st.pyplot(plt)

        # plt.figure(figsize=(10, 5))
        # plt.plot(monthly_sales["Invoice_Date"], monthly_sales["Quantity"], marker='o', linestyle='-')

        # plt.xlabel("Tanggal")
        # plt.ylabel("Total Penjualan")
        # plt.title("Grafik Penjualan per Bulan")
        # plt.grid(True)
        # plt.xticks(rotation=45)
        # st.pyplot(plt)

        weekly_sales.set_index("Invoice_Date", inplace=True)
        monthly_sales.set_index("Invoice_Date", inplace=True)

        dfdemandxgboostweekly = weekly_sales.copy(deep=True)
        dfdemandxgboostmonthly = monthly_sales.copy(deep=True)

        Q1 = weekly_sales['Quantity'].quantile(0.25)
        Q3 = weekly_sales['Quantity'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Tandai outlier sebagai NaN
        weekly_sales['Quantity'] = weekly_sales['Quantity'].where(
            (weekly_sales['Quantity'] >= lower_bound) &
            (weekly_sales['Quantity'] <= upper_bound),
            np.nan
        )

        weekly_sales['Quantity'] = weekly_sales['Quantity'].fillna(
            weekly_sales['Quantity'].rolling(window=3, center=True, min_periods=1).mean()
        )

        Q1 = monthly_sales['Quantity'].quantile(0.25)
        Q3 = monthly_sales['Quantity'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Tandai outlier sebagai NaN
        monthly_sales['Quantity'] = monthly_sales['Quantity'].where(
            (monthly_sales['Quantity'] >= lower_bound) &
            (monthly_sales['Quantity'] <= upper_bound),
            np.nan
        )

        monthly_sales['Quantity'] = monthly_sales['Quantity'].fillna(
            monthly_sales['Quantity'].rolling(window=3, center=True, min_periods=1).mean()
        )


        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        import matplotlib.pyplot as plt
        from statsmodels.tsa.stattools import adfuller, acf, pacf
        import numpy as np


        # %% grid search function


        # === Fungsi kecil tetap dipertahankan ===
        def get_d(series, max_d=2):
            for d in range(max_d + 1):
                test_series = series.copy()
                if d > 0:
                    test_series = test_series.diff(d).dropna()
                adf_result = adfuller(test_series)
                if adf_result[1] < 0.05:
                    return d
            return max_d


        def get_D(series, m, max_D=1):
            for D in range(max_D + 1):
                test_series = series.copy()
                if D > 0:
                    test_series = test_series.diff(m * D).dropna()
                adf_result = adfuller(test_series)
                if adf_result[1] < 0.05:
                    return D
            return max_D

        def make_range(center, max_val=3):
            return sorted(list(set([max(center - 1, 0), center, min(center + 1, max_val)])))

        def sarima_grid_search(data, p_values, d_values, q_values,
                            P_values, D_values, Q_values, m_values):

            total_combinations = (
                len(p_values) * len(d_values) * len(q_values) *
                len(P_values) * len(D_values) * len(Q_values) * len(m_values)
            )
            progress_bar = st.progress(0)
            status_text = st.empty()
            evaluated = 0
            best_score = float('inf')
            best_cfg = None
            best_model = None

            for m in m_values:
                for p in p_values:
                    for d in d_values:
                        for q in q_values:
                            for P in P_values:
                                for D in D_values:
                                    for Q in Q_values:
                                        order = (p, d, q)
                                        seasonal_order = (P, D, Q, m)
                                        try:
                                            status_text.text(f"Evaluating: SARIMA{order} x {seasonal_order}")
                                            model = SARIMAX(data,
                                                            order=order,
                                                            seasonal_order=seasonal_order,
                                                            enforce_stationarity=False,
                                                            enforce_invertibility=False)
                                            
                                            result = model.fit(disp=False)
                                            # ==== RMSE sebagai metrik ====
                                            offset = d + m * D
                                            y_true = data[offset:]
                                            y_pred = result.fittedvalues[-len(y_true):]

                                            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                                            if rmse < best_score and not (p == 0 and d == 0 and q == 0 and P == 0 and D == 0 and Q == 0):
                                                best_score = rmse
                                                best_cfg = (order, seasonal_order)
                                                best_model = result

                                        except:
                                            continue

                                        # Update progress
                                        evaluated += 1
                                        progress = int(evaluated / total_combinations * 100)
                                        progress_bar.progress(progress)

            status_text.text("✅ Grid search selesai.")
            return best_cfg, best_model, best_score

        def run_sarima_auto_pipeline(series, seasonal_periods=[13, 26, 52], is_weekly=True):
            data = series.copy(deep=True)
            # st.write(data)
            # st.write(len(data))
            # === Deteksi apakah musiman tahunan layak (berdasarkan ACF[52]) ===
            from statsmodels.tsa.stattools import acf
            acf_vals = acf(data, nlags=60)
            if len(acf_vals) > 52 and abs(acf_vals[52]) > 0.2:
                st.info("✅ Pola musiman tahunan (m=52) terdeteksi, akan dievaluasi.")
            else:
                if 52 in seasonal_periods:
                    seasonal_periods.remove(52)
                    st.warning("Musiman tahunan (m=52) dilewati karena tidak terdeteksi pola signifikan.")


            # === Step 1: Tentukan non-seasonal differencing ===
            d = get_d(data)

            # === Range Parameter ===
            d_range = [d]
            if is_weekly:
                p_range = list(range(0, 11))  # 0 to 10
                q_range = list(range(0, 11))
            else:
                p_range = list(range(0, 5))   # 0 to 4
                q_range = list(range(0, 5))

            best_cfg, best_model, best_score = None, None, float('inf')

            # === Coba semua seasonal period ===
            best_params_info = None
            for m in seasonal_periods:
                # if len(data) < m * 3:
                #     st.info(f"⏭️ Skipping m = {m} karena data hanya {len(data)} baris (< {m*3})")
                #     continue  # skip m jika data tidak cukup panjang

                D = get_D(data, m)

                D_range = [D]
                P_range = list(range(0, 2))
                Q_range = list(range(0, 2))
                
                
                # === Evaluasi grid search SARIMA ===
                st.write(f"Evaluating seasonal m={m} → SARIMA grid search")

                cfg, model, score = sarima_grid_search(
                    data,
                    p_values=p_range,
                    d_values=d_range,
                    q_values=q_range,
                    P_values=P_range,
                    D_values=D_range,
                    Q_values=Q_range,
                    m_values=[m]
                )

                if score < best_score:
                    best_cfg, best_model, best_score = cfg, model, score

                    best_params_info = {
                        'm': m,
                        'p_range': p_range,
                        'd_range': d_range,
                        'q_range': q_range,
                        'P_range': P_range,
                        'D_range': D_range,
                        'Q_range': Q_range,
                        'chosen_cfg': cfg
                    }
            if best_params_info:
                st.write("Parameter Grid untuk Model Terbaik:")
                st.write(f"Seasonal period m = {best_params_info['m']}")
                st.write(f"p_range = {best_params_info['p_range']}")
                st.write(f"d_range = {best_params_info['d_range']}")
                st.write(f"q_range = {best_params_info['q_range']}")
                st.write(f"P_range = {best_params_info['P_range']}")
                st.write(f"D_range = {best_params_info['D_range']}")
                st.write(f"Q_range = {best_params_info['Q_range']}")

                # st.write(f"Parameter ARIMA yang dipilih:")
                # st.write(f"p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}")
            else:
                st.warning("Tidak ada model SARIMA yang valid ditemukan.")
            return best_cfg, best_model, best_score





        # %% pacf acf plot
        d_weekly = get_d(weekly_sales['Quantity'])
        d_monthly = get_d(monthly_sales['Quantity'])

        if d_weekly == 0:
            weekly_diff = weekly_sales['Quantity']

        else:
            weekly_diff = weekly_sales['Quantity'].diff(d_weekly).dropna()

        if d_monthly == 0:
            monthly_diff = monthly_sales['Quantity']

        else:
            monthly_diff = monthly_sales['Quantity'].diff(d_monthly).dropna()
        
        max_lags = min(20, len(monthly_diff) // 2 - 1)


        # PACF Plot untuk p
        plot_pacf(weekly_diff, lags=20)
        plt.title("WEEKLY PACF - Tentukan p")
        st.pyplot(plt)

        # ACF Plot untuk q
        plot_acf(weekly_diff, lags=20)
        plt.title("WEEKLY ACF - Tentukan q")
        st.pyplot(plt)

        # PACF Plot untuk p
        plot_pacf(monthly_diff, lags=max_lags)
        plt.title("MONTHLY PACF - Tentukan p")
        st.pyplot(plt)

        # ACF Plot untuk q
        plot_acf(monthly_diff, lags=max_lags)
        plt.title("MONTHLY ACF - Tentukan q")
        st.pyplot(plt)

        def slugify(text):
            return text.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("@", "").replace("/", "_")

        def generate_model_filename(produk_name, kode_list, periode):
            slug = slugify(produk_name)
            kode_str = "-".join(str(k) for k in kode_list)
            return f"pickle_save_parameter/model_sarima_{slug}_{kode_str}_{periode}.pkl"

        os.makedirs(r"C:\Users\RAVASHOP\Documents\kuliah\skripsi\coding\pickle_save_parameter", exist_ok=True)

        dfsarima = weekly_sales.copy(deep=True)
        dfsarimamonthly = monthly_sales.copy(deep=True)
        train = dfsarima.iloc[:int(0.8*len(dfsarima))]
        test = dfsarima.iloc[int(0.8*len(dfsarima)):]
        trainmonthly = dfsarimamonthly.iloc[:int(0.8*len(dfsarimamonthly))]
        testmonthly = dfsarimamonthly.iloc[int(0.8*len(dfsarimamonthly)):]
        # train = train.set_index('Invoice_Date') 
        # trainmonthly = trainmonthly.set_index('Invoice_Date') 
        # st.write(dfsarima)







        # %% demand forecasting arima weekly
        # best_cfg_weekly, modelweekly, scoreweekly = run_sarima_auto_pipeline(
        #     weekly_sales['Quantity'], seasonal_periods=[13, 26, 52]
        # )


        # modelweekly = SARIMAX(train['Quantity'], order=best_cfg_weekly[0], seasonal_order=best_cfg_weekly[1], enforce_stationarity=False,
        #     enforce_invertibility=False)
        # model_fitweekly = modelweekly.fit()
        filename_weekly = generate_model_filename(produk_terpilih, kode_terpilih, "weekly")

        # === Load atau training ulang model ===
        if os.path.exists(filename_weekly):
            with open(filename_weekly, 'rb') as f:
                saved = pickle.load(f)
            best_cfg_weekly = saved["best_cfg"]
            model_fitweekly = saved["model_fit"]
            st.success("Model SARIMA weekly dimuat dari file.")
            # st.write(f"Parameter ARIMA yang dipilih:")
            # st.write(f"p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}")
        else:
            best_cfg_weekly, modelweekly, _ = run_sarima_auto_pipeline(
                dfsarima, seasonal_periods=[13, 26, 52], is_weekly=True
            )
            st.write(dfsarima)
            modelweekly = SARIMAX(train['Quantity'],
                                order=best_cfg_weekly[0],
                                seasonal_order=best_cfg_weekly[1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
            model_fitweekly = modelweekly.fit()

            saved = {
                "best_cfg": best_cfg_weekly,
                "model_fit": model_fitweekly
            }

            with open(filename_weekly, 'wb') as f:
                pickle.dump(saved, f)
            st.success("Model SARIMA weekly berhasil disimpan.")

        # === Forecast dan evaluasi ===
        forecastweekly = model_fitweekly.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test['Quantity'], forecastweekly))
        mape = mean_absolute_percentage_error(test['Quantity'], forecastweekly) * 100
        smape = 100 * np.mean(2 * np.abs(forecastweekly - test['Quantity']) / (np.abs(test['Quantity']) + np.abs(forecastweekly)))

        # Tambahkan MAPE jika belum ada
        # Tambahkan MAPE jika belum ada
        updated = False
        if "mape" not in saved:
            saved["mape"] = mape
            updated = True

        if "smape" not in saved:
            saved["smape"] = smape
            updated = True

        # Tambahkan dfsarima jika belum ada
        if "dfsarima" not in saved:
            saved["dfsarima"] = dfsarima
            updated = True

        if "rmse" not in saved:
            saved["rmse"] = rmse
            updated = True

        if updated:
            with open(filename_weekly, 'wb') as f:
                pickle.dump(saved, f)
            st.info("File model diperbarui (MAPE/dfsarima/RMSE/SMAPE).")

        # === Tampilkan hasil evaluasi ===
        st.write(best_cfg_weekly[0], best_cfg_weekly[1])
        st.write(f"RMSE weekly arima: {rmse:.2f}")
        st.write(f"MAPE weekly arima: {mape:.2f}%")
        st.write(f"SMAPE weekly arima: {smape:.2f}%")


        # 1. In-sample prediction (fitted values selama periode training)
        fitted_valuesweekly = model_fitweekly.fittedvalues

        # 2. Out-of-sample prediction (forecast selama periode test sudah kamu buat)
        #    forecast_df sudah berisi prediksi hasil forecast untuk periode test

        # 3. Plot semuanya
        plt.figure(figsize=(12, 6))

        # Plot data aktual (train + test)
        plt.plot(train.index, train['Quantity'], label='Train (Actual)', color='black')
        plt.plot(test.index, test['Quantity'], label='Test (Actual)', color='gray')

        # Plot in-sample fit (selama train)
        plt.plot(fitted_valuesweekly.index, fitted_valuesweekly, label='In-Sample Prediction (Fit)', color='blue', linestyle='--')

        # Plot forecast (periode test)
        plt.plot(test.index, forecastweekly, label='Forecast (Test)', color='red', linestyle='--')

        plt.title('SARIMA In-Sample Fit + Forecast')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)

        # %% demand forecasting arima monthly
        # best_cfg_monthly, modelmonthly, scoremonthly = run_sarima_auto_pipeline(
        #     monthly_sales['Quantity'], seasonal_periods=[2,4,6,8,10,12]
        # )

        # modelmonthly = SARIMAX(trainmonthly['Quantity'], order=best_cfg_monthly[0], seasonal_order=best_cfg_monthly[1], enforce_stationarity=False,
        #     enforce_invertibility=False)
        # # modelmonthly = SARIMAX(trainmonthly['Quantity'],
        # #                 order=(1, 0, 2),            # (p, d, q)
        # #                 seasonal_order=(1, 0, 1, 4) # (P, D, Q, m)
        # #                 ,  # Tidak ada komponen musiman
        # #     enforce_stationarity=False,
        # #     enforce_invertibility=False
        # #                )
        # model_fitmonthly = modelmonthly.fit()

        filename_monthly = generate_model_filename(produk_terpilih, kode_terpilih, "monthly")

        # === Load model jika sudah ada, atau latih jika belum ===
        if os.path.exists(filename_monthly):
            with open(filename_monthly, 'rb') as f:
                saved = pickle.load(f)
            best_cfg_monthly = saved["best_cfg"]
            model_fitmonthly = saved["model_fit"]
            st.success("Model SARIMA monthly dimuat dari file.")
        else:
            best_cfg_monthly, modelmonthly, _ = run_sarima_auto_pipeline(
                dfsarimamonthly, seasonal_periods=[2, 4, 6, 8, 10, 12], is_weekly=False
            )
            modelmonthly = SARIMAX(trainmonthly['Quantity'],
                                order=best_cfg_monthly[0],
                                seasonal_order=best_cfg_monthly[1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
            model_fitmonthly = modelmonthly.fit()

            saved = {
                "best_cfg": best_cfg_monthly,
                "model_fit": model_fitmonthly
            }

            with open(filename_monthly, 'wb') as f:
                pickle.dump(saved, f)
            st.success("Model SARIMA monthly berhasil disimpan.")

        # === Evaluasi model ===
        forecastmonthly = model_fitmonthly.forecast(steps=len(testmonthly))
        rmse = np.sqrt(mean_squared_error(testmonthly['Quantity'], forecastmonthly))
        mape = mean_absolute_percentage_error(testmonthly['Quantity'], forecastmonthly) * 100
        smape = 100 * np.mean(2 * np.abs(forecastmonthly - testmonthly['Quantity']) / (np.abs(testmonthly['Quantity']) + np.abs(forecastmonthly)))

        # === Tambahkan informasi tambahan sesuai pola save parameter full ===
        updated = False
        if "mape" not in saved:
            saved["mape"] = mape
            updated = True

        if "smape" not in saved:
            saved["smape"] = smape
            updated = True

        if "dfsarima" not in saved:
            saved["dfsarima"] = dfsarimamonthly
            updated = True
        
        if "rmse" not in saved:
            saved["rmse"] = rmse
            updated = True

        if updated:
            with open(filename_monthly, 'wb') as f:
                pickle.dump(saved, f)
            st.info("File model SARIMA monthly diperbarui (MAPE/dfsarima/SMAPE/RMSE).")

        # === Tampilkan hasil evaluasi ===
        st.write(best_cfg_monthly[0], best_cfg_monthly[1])
        st.write(f"RMSE SARIMA monthly: {rmse:.2f}")
        st.write(f"MAPE SARIMA monthly: {mape:.2f}%")
        st.write(f"SMAPE monthly arima: {smape:.2f}%")



        import matplotlib.pyplot as plt

        # 1. In-sample prediction (fitted values selama periode training)
        fitted_valuesmonthly = model_fitmonthly.fittedvalues

        # 2. Out-of-sample prediction (forecast selama periode test sudah kamu buat)
        #    forecast_df sudah berisi prediksi hasil forecast untuk periode test

        # 3. Plot semuanya
        plt.figure(figsize=(12, 6))

        # Plot data aktual (train + test)
        plt.plot(trainmonthly.index, trainmonthly['Quantity'], label='Train (Actual)', color='black')
        plt.plot(testmonthly.index, testmonthly['Quantity'], label='Test (Actual)', color='gray')

        # Plot in-sample fit (selama train)
        plt.plot(fitted_valuesmonthly.index, fitted_valuesmonthly, label='In-Sample Prediction (Fit)', color='blue', linestyle='--')

        # Plot forecast (periode test)
        plt.plot(testmonthly.index, forecastmonthly, label='Forecast (Test)', color='red', linestyle='--')

        plt.title('SARIMA In-Sample Fit + Forecast')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)

        # %% save xgboost model
        def generate_xgb_model_filename(produk_name, kode_list, periode):
            slug = produk_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("@", "").replace("/", "_")
            kode_str = "-".join(str(k) for k in kode_list)
            return f"pickle_save_parameter/model_xgboost_{slug}_{kode_str}_{periode}.pkl"

        # %% demand forecasting xgboost weekly
        
        for lag in range(1, 4):
            dfdemandxgboostweekly[f'lag_{lag}'] = dfdemandxgboostweekly['Quantity'].shift(lag)

        dfdemandxgboostweekly['rolling_mean_3'] = dfdemandxgboostweekly['Quantity'].rolling(window=3).mean().shift(1)
        dfdemandxgboostweekly['month'] = dfdemandxgboostweekly.index.month
        dfdemandxgboostweekly = dfdemandxgboostweekly.dropna()

        train_size_weekly = int(len(dfdemandxgboostweekly) * 0.8)
        train_weekly = dfdemandxgboostweekly.iloc[:train_size_weekly]
        test_weekly = dfdemandxgboostweekly.iloc[train_size_weekly:]

        X_train_weekly = train_weekly.drop('Quantity', axis=1)
        y_train_weekly = train_weekly['Quantity']
        X_test_weekly = test_weekly.drop('Quantity', axis=1)
        y_test_weekly = test_weekly['Quantity']

        def objective_weekly(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                'random_state': 42
            }

            model = XGBRegressor(**params)
            model.fit(
                X_train_weekly,
                y_train_weekly,
                eval_set=[(X_test_weekly, y_test_weekly)],
                verbose=False
            )
            y_pred = model.predict(X_test_weekly)
            return np.sqrt(mean_squared_error(y_test_weekly, y_pred))
        filename_xgb_weekly = generate_xgb_model_filename(produk_terpilih, kode_terpilih, "weekly")

        # === Load model jika sudah ada ===
        if os.path.exists(filename_xgb_weekly):
            with open(filename_xgb_weekly, 'rb') as f:
                saved_weekly = pickle.load(f)
            model_weekly = saved_weekly["model"]
            best_params_weekly = saved_weekly["params"]
            mape_weekly = saved_weekly.get("mape")
            rmse_weekly = saved_weekly.get("rmse")
            X_test_weekly = saved_weekly.get("X_test", X_test_weekly)
            historical_data_weekly = saved_weekly.get("historical_data", dfdemandxgboostweekly)
            st.success("Model XGBoost weekly dimuat dari file.")
        else:
            # === Tuning dan training baru ===
            study_weekly = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=SuccessiveHalvingPruner()
            )
            early_stop_callback_weekly = MaxTrialsCallback(300, states=(optuna.trial.TrialState.COMPLETE,))
            study_weekly.optimize(objective_weekly, n_trials=300, callbacks=[early_stop_callback_weekly])
            best_params_weekly = study_weekly.best_params

            model_weekly = XGBRegressor(**best_params_weekly)
            model_weekly.fit(X_train_weekly, y_train_weekly)

            saved_weekly = {
                "model": model_weekly,
                "params": best_params_weekly,
                "X_test": X_test_weekly,
                "historical_data": dfdemandxgboostweekly  # full data untuk prediksi ke depan
            }

            with open(filename_xgb_weekly, 'wb') as f:
                pickle.dump(saved_weekly, f)
            st.success("Model XGBoost weekly berhasil disimpan.")

        # === Evaluasi model ===
        y_train_pred_weekly = model_weekly.predict(X_train_weekly)
        y_test_pred_weekly = model_weekly.predict(X_test_weekly)

        rmse_weekly = np.sqrt(mean_squared_error(y_test_weekly, y_test_pred_weekly))
        mape_weekly = mean_absolute_percentage_error(y_test_weekly, y_test_pred_weekly) * 100
        smape_weekly = 100 * np.mean(2 * np.abs(y_test_pred_weekly - y_test_weekly) / (np.abs(y_test_weekly) + np.abs(y_test_pred_weekly)))

        # Tambahkan MAPE jika belum disimpan
        updated = False
        if "mape" not in saved_weekly:
            saved_weekly["mape"] = mape_weekly
            updated = True

        if "smape" not in saved_weekly:
            saved_weekly["smape"] = smape_weekly
            updated = True

        if "rmse" not in saved_weekly:
            saved_weekly["rmse"] = rmse_weekly
            updated = True

        # Tambahkan X_test jika belum ada
        if "X_test" not in saved_weekly:
            saved_weekly["X_test"] = X_test_weekly
            updated = True

        # Tambahkan historical data jika belum ada
        if "historical_data" not in saved_weekly:
            saved_weekly["historical_data"] = dfdemandxgboostweekly
            updated = True

        if updated:
            with open(filename_xgb_weekly, 'wb') as f:
                pickle.dump(saved_weekly, f)
            st.info("File model diperbarui (MAPE/X_test/historical_data/SMAPE/RMSE).")

        # Tampilkan evaluasi
        st.write(best_params_weekly)
        st.write(f"Final RMSE XGBoost weekly: {rmse_weekly:.2f}")
        st.write(f"Final MAPE XGBoost weekly: {mape_weekly:.2f}%")
        st.write(f"SMAPE weekly XGBoost: {smape_weekly:.2f}%")




        train_pred_df_weekly = pd.DataFrame(y_train_pred_weekly, index=y_train_weekly.index, columns=['Predicted_Train'])
        test_pred_df_weekly = pd.DataFrame(y_test_pred_weekly, index=y_test_weekly.index, columns=['Predicted_Test'])

        plt.figure(figsize=(12, 6))
        plt.plot(y_train_weekly.index, y_train_weekly, label='Train (Actual)', color='black')
        plt.plot(y_test_weekly.index, y_test_weekly, label='Test (Actual)', color='gray')
        plt.plot(train_pred_df_weekly.index, train_pred_df_weekly['Predicted_Train'], label='In-Sample Prediction (XGBoost)', color='blue', linestyle='--')
        plt.plot(test_pred_df_weekly.index, test_pred_df_weekly['Predicted_Test'], label='Forecast (XGBoost)', color='red', linestyle='--')
        plt.title('XGBoost In-Sample Fit + Forecast (Weekly)')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)


        # %% demand forecasting xgboost monthly
        

        for lag in range(1, 4):
            dfdemandxgboostmonthly[f'lag_{lag}'] = dfdemandxgboostmonthly['Quantity'].shift(lag)

        dfdemandxgboostmonthly['rolling_mean_3'] = dfdemandxgboostmonthly['Quantity'].rolling(window=3).mean().shift(1)
        dfdemandxgboostmonthly['month'] = dfdemandxgboostmonthly.index.month
        dfdemandxgboostmonthly = dfdemandxgboostmonthly.dropna()

        train_size_monthly = int(len(dfdemandxgboostmonthly) * 0.8)
        train_monthly = dfdemandxgboostmonthly.iloc[:train_size_monthly]
        test_monthly = dfdemandxgboostmonthly.iloc[train_size_monthly:]

        X_train_monthly = train_monthly.drop('Quantity', axis=1)
        y_train_monthly = train_monthly['Quantity']
        X_test_monthly = test_monthly.drop('Quantity', axis=1)
        y_test_monthly = test_monthly['Quantity']

        def objective_monthly(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                'random_state': 42
            }

            model = XGBRegressor(**params)
            model.fit(
                X_train_monthly,
                y_train_monthly,
                eval_set=[(X_test_monthly, y_test_monthly)],
                verbose=False
            )
            y_pred = model.predict(X_test_monthly)
            return np.sqrt(mean_squared_error(y_test_monthly, y_pred))
                
        filename_xgb_monthly = generate_xgb_model_filename(produk_terpilih, kode_terpilih, "monthly")

        # === Cek apakah file model sudah ada ===
        if os.path.exists(filename_xgb_monthly):
            with open(filename_xgb_monthly, 'rb') as f:
                saved_monthly = pickle.load(f)
            model_monthly = saved_monthly["model"]
            best_params_monthly = saved_monthly["params"]
            X_test_monthly = saved_monthly.get("X_test", X_test_monthly)
            mape_monthly = saved_monthly.get("mape")
            rmse_monthly = saved_monthly.get("rmse")
            historical_data_monthly = saved_monthly.get("historical_data", dfdemandxgboostmonthly)
            st.success("Model XGBoost monthly dimuat dari file.")
        else:
            # === Training ulang jika belum ada ===
            study_monthly = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=SuccessiveHalvingPruner()
            )
            early_stop_callback_monthly = MaxTrialsCallback(300, states=(optuna.trial.TrialState.COMPLETE,))
            study_monthly.optimize(objective_monthly, n_trials=300, callbacks=[early_stop_callback_monthly])
            best_params_monthly = study_monthly.best_params

            model_monthly = XGBRegressor(**best_params_monthly)
            model_monthly.fit(X_train_monthly, y_train_monthly)

            saved_monthly = {
                "model": model_monthly,
                "params": best_params_monthly,
                "X_test": X_test_monthly,
                "historical_data": dfdemandxgboostmonthly
            }

            with open(filename_xgb_monthly, 'wb') as f:
                pickle.dump(saved_monthly, f)
            st.success("Model XGBoost monthly berhasil disimpan.")

        # === Evaluasi model ===
        y_train_pred_monthly = model_monthly.predict(X_train_monthly)
        y_test_pred_monthly = model_monthly.predict(X_test_monthly)

        rmse_monthly = np.sqrt(mean_squared_error(y_test_monthly, y_test_pred_monthly))
        mape_monthly = mean_absolute_percentage_error(y_test_monthly, y_test_pred_monthly) * 100
        smape_monthly = 100 * np.mean(2 * np.abs(y_test_pred_monthly - y_test_monthly) / (np.abs(y_test_monthly) + np.abs(y_test_pred_monthly)))

        # === Perbarui jika ada yang belum disimpan ===
        updated = False

        if "mape" not in saved_monthly:
            saved_monthly["mape"] = mape_monthly
            updated = True
        
        if "smape" not in saved_monthly:
            saved_monthly["smape"] = smape_monthly
            updated = True

        if "rmse" not in saved_monthly:
            saved_monthly["rmse"] = rmse_monthly
            updated = True

        if "X_test" not in saved_monthly:
            saved_monthly["X_test"] = X_test_monthly
            updated = True

        if "historical_data" not in saved_monthly:
            saved_monthly["historical_data"] = dfdemandxgboostmonthly
            updated = True

        if updated:
            with open(filename_xgb_monthly, 'wb') as f:
                pickle.dump(saved_monthly, f)
            st.info("File model XGBoost monthly diperbarui (MAPE / X_test / historical_data / SMAPE / RMSE).")

        # === Tampilkan evaluasi ===
        st.write(best_params_monthly)
        st.write(f"Final RMSE xgboost monthly: {rmse_monthly:.2f}")
        st.write(f"Final MAPE xgboost monthly: {mape_monthly:.2f}%")
        st.write(f"SMAPE monthly XGBoost: {smape_monthly:.2f}%")



        train_pred_df_monthly = pd.DataFrame(y_train_pred_monthly, index=y_train_monthly.index, columns=['Predicted_Train'])
        test_pred_df_monthly = pd.DataFrame(y_test_pred_monthly, index=y_test_monthly.index, columns=['Predicted_Test'])

        plt.figure(figsize=(12, 6))
        plt.plot(y_train_monthly.index, y_train_monthly, label='Train (Actual)', color='black')
        plt.plot(y_test_monthly.index, y_test_monthly, label='Test (Actual)', color='gray')
        plt.plot(train_pred_df_monthly.index, train_pred_df_monthly['Predicted_Train'], label='In-Sample Prediction (XGBoost)', color='blue', linestyle='--')
        plt.plot(test_pred_df_monthly.index, test_pred_df_monthly['Predicted_Test'], label='Forecast (XGBoost)', color='red', linestyle='--')
        plt.title('XGBoost In-Sample Fit + Forecast (Monthly)')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)






