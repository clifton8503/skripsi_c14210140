


def main(produk_terpilih, kode_terpilih):

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
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import os
    # Bagian untuk upload dan simpan data pembelian
    import streamlit as st
    import pandas as pd
    from sqlalchemy import create_engine
    from sqlalchemy.dialects.mysql import VARCHAR, FLOAT, DATETIME, INTEGER
    import pickle



    # Upload file pembelian baru
    st.header("📤 Upload Data Pembelian Baru & Retrain Model Harga Beli")

    uploaded_file_pembelian = st.file_uploader("Upload file Excel pembelian (.xlsx)", type=["xlsx"])

    if uploaded_file_pembelian is not None:
        try:
            pembelian = pd.read_excel(uploaded_file_pembelian)
            pembelian = pembelian.dropna()

            pembelian['Tgl Faktur'] = pembelian['Tgl Faktur'].replace({
                'Jan': 'Jan', 'Feb': 'Feb', 'Mar': 'Mar', 'Apr': 'Apr', 'Mei': 'May', 'Jun': 'Jun',
                'Jul': 'Jul', 'Agu': 'Aug', 'Sep': 'Sep', 'Okt': 'Oct', 'Nop': 'Nov', 'Des': 'Dec'
            }, regex=True)
            pembelian['Tgl Faktur'] = pd.to_datetime(pembelian['Tgl Faktur'], format='%d %b %Y')
            pembelian['Kuantitas'] = pembelian['Kuantitas'].str.replace('.', '').str.replace(',00', '').astype(int)
            pembelian['Jumlah'] = pembelian['Jumlah'].str.replace('.', '').str.replace(',', '.').astype(float)
            pembelian['Jumlah Data'] = pembelian['Jumlah Data'].str.replace('.', '').str.replace(',00', '').astype(int)
            pembelian['Harga satuan'] = pembelian['Harga satuan'].str.replace('.', '').str.replace(',', '.').astype(float)

            pembelian.rename(columns={
                'No. Faktur': 'Invoice_Number',
                'Tgl Faktur': 'Invoice_Date',
                'Kuantitas': 'Quantity',
                'Jumlah': 'Total_Amount',
                'Unit 1 Barang': 'Unit',
                'Jumlah Data': 'jumlah_data',
                'Keterangan Barang': 'keterangan_barang',
                'No. Barang': 'no_barang',
                'Harga satuan': 'harga_satuan'
            }, inplace=True)

            # Tipe data untuk to_sql
            from sqlalchemy.types import String, Float, DateTime, Integer
            dtype = {
                'Invoice_Number': VARCHAR(50),
                'Invoice_Date': DATETIME(),
                'Quantity': INTEGER(),
                'Unit': VARCHAR(20),
                'Total_Amount': FLOAT(),
                'jumlah_data': FLOAT(),
                'keterangan_barang': VARCHAR(255),
                'no_barang': VARCHAR(20),
                'harga_satuan': FLOAT()
            }
            st.write(f"📊 Jumlah data baru yang akan ditambahkan: {len(pembelian)}")
            # Simpan ke MySQL
            from sqlalchemy import create_engine
            engine = create_engine("mysql+pymysql://root:@localhost/tifalanggeng")
            pembelian.to_sql("pembelian", con=engine, if_exists="append", index=False, dtype=dtype)

            st.success(f"✅ {len(pembelian)} data berhasil diupload dan disimpan ke database!")

            # === STEP 3: Trigger retrain setelah upload ===
            st.info("Retrain akan dijalankan ulang berdasarkan data baru.")

            # Optional: hapus model lama
            model_dir = "pickle_save_parameter_hargabeli"
            for file in os.listdir(model_dir):
                if file.endswith(".pkl"):
                    os.remove(os.path.join(model_dir, file))
            st.warning("🗑️ Model lama harga beli dihapus. Silakan retrain ulang.")

            st.rerun()

        except Exception as e:
            st.error(f"❌ Gagal memproses file: {e}")

    kode_terpilih_str = ','.join(f"'{str(k)}'" for k in kode_terpilih)

    # Koneksi ke MySQL
    engine = create_engine("mysql+pymysql://root:@localhost/tifalanggeng")


    query = f"""
    SELECT Invoice_Date, harga_satuan
    FROM pembelian 
    WHERE no_barang IN ({kode_terpilih_str})
    ORDER BY Invoice_Date
    """

    filtered_datapembelian = pd.read_sql(query, con=engine, parse_dates=['Invoice_Date'])
    filtered_datapembelian["Invoice_Date"] = pd.to_datetime(filtered_datapembelian["Invoice_Date"])
    filtered_datapembelian.set_index("Invoice_Date", inplace=True)
    filtered_datapembelian = filtered_datapembelian[(filtered_datapembelian['harga_satuan'] >= 100000) & (filtered_datapembelian['harga_satuan'] <= 1000000)]


    weekly_priceinterpolasi = filtered_datapembelian["harga_satuan"].resample("W").mean()

    weekly_priceinterpolasi.index = pd.to_datetime(weekly_priceinterpolasi.index)
    weekly_priceinterpolasi = weekly_priceinterpolasi.interpolate(method='linear')
    weekly_priceinterpolasi = weekly_priceinterpolasi.reset_index()

    daily_priceinterpolasi = filtered_datapembelian["harga_satuan"].resample("D").mean()
    daily_priceinterpolasi.index = pd.to_datetime(daily_priceinterpolasi.index)
    daily_priceinterpolasi = daily_priceinterpolasi.interpolate(method='linear')
    daily_priceinterpolasi = daily_priceinterpolasi.reset_index()
    # st.write(daily_priceinterpolasi)
    query = f"""
    SELECT * 
    FROM kursidrusdweekly
    """

    kursweeklyusd = pd.read_sql(query, con=engine, parse_dates=['Date'])

    query = f"""
    SELECT * 
    FROM kursidrusddaily
    """

    kursdailyusd = pd.read_sql(query, con=engine, parse_dates=['Date'])



    kursweeklyusd = kursweeklyusd.sort_values('Date')
    weekly_priceinterpolasi = weekly_priceinterpolasi.rename(columns={"Invoice_Date": "Date"})
    weekly_priceinterusd = pd.merge(weekly_priceinterpolasi, kursweeklyusd[["Date", "Price"]], on="Date", suffixes=("_PP", "_Kurs"))
    # Konversi harga dari CNY ke USD
    weekly_priceinterusd["Price_USD"] = weekly_priceinterusd["harga_satuan"] / weekly_priceinterusd["Price"]


    kursdailyusd = kursdailyusd.sort_values('Date')
    daily_priceinterpolasi = daily_priceinterpolasi.rename(columns={"Invoice_Date": "Date"})
    daily_priceinterusd = pd.merge(daily_priceinterpolasi, kursdailyusd[["Date", "Price"]], on="Date", suffixes=("_PP", "_Kurs"))
    # Konversi harga dari CNY ke USD
    daily_priceinterusd["Price_USD"] = daily_priceinterusd["harga_satuan"] / daily_priceinterusd["Price"]






    query = f"""
    SELECT * 
    FROM hargaoilweekly
    """

    dfoilweekly = pd.read_sql(query, con=engine, parse_dates=['Date'])

    query = f"""
    SELECT * 
    FROM hargaoildaily
    """

    dfoil = pd.read_sql(query, con=engine, parse_dates=['Date'])

    # Ambil data dari MySQL
    query = f"""
    SELECT * 
    FROM kurscnyusdweekly
    """

    kursweekly = pd.read_sql(query, con=engine, parse_dates=['Date'])

    query = f"""
    SELECT * 
    FROM hargappweekly
    """

    dfppweekly = pd.read_sql(query, con=engine, parse_dates=['Date'])

    dfppweeklyusd = pd.merge(dfppweekly, kursweekly, on="Date", suffixes=("_PP", "_Kurs"))
    dfppweeklyusd["Price_PP_USD"] = dfppweeklyusd["Price_PP"] * dfppweeklyusd["Price_Kurs"]


    query = f"""
    SELECT * 
    FROM kurscnyusddaily
    """

    kursdaily = pd.read_sql(query, con=engine, parse_dates=['Date'])

    query = f"""
    SELECT * 
    FROM hargappdaily
    """

    dfpp = pd.read_sql(query, con=engine, parse_dates=['Date'])


    dfppdailyusd = pd.merge(dfpp, kursdaily[["Date", "Price"]], on="Date", suffixes=("_PP", "_Kurs"))
    dfppdailyusd["Price_PP_USD"] = dfppdailyusd["Price_PP"] * dfppdailyusd["Price_Kurs"]



    # Gabungkan kedua dataset berdasarkan tanggal
    dfoilpp = pd.merge(dfppweeklyusd[["Date", "Price_PP_USD"]], dfoilweekly[["Date", "Price"]], on="Date", suffixes=("_PP", "_Oil"))

    # Atur Date sebagai index
    dfoilpp.set_index("Date", inplace=True)
    dfoilpp.index = pd.to_datetime(dfoilpp.index)

    # Sort berdasarkan tanggal (penting!)
    dfoilpp.sort_index(inplace=True)



    # Gabungkan kedua dataset berdasarkan tanggal
    dfoilppdaily = pd.merge(dfppdailyusd[["Date", "Price_PP_USD"]], dfoil[["Date", "Price"]], on="Date", suffixes=("_PP", "_Oil"))

    # Atur Date sebagai index
    dfoilppdaily.set_index("Date", inplace=True)
    dfoilppdaily.index = pd.to_datetime(dfoilppdaily.index)

    # Sort berdasarkan tanggal (penting!)
    dfoilppdaily.sort_index(inplace=True)


    dfoilpp = dfoilpp.rename(columns={"Price_PP_USD": "Price_PP"})
    dfoilpp = dfoilpp.rename(columns={"Price": "Price_Oil"})
    dfoilpppredweekly = dfoilpp.copy(deep=True)
    dfoilppbeli = pd.merge(dfoilpp, weekly_priceinterusd[["Date", "Price_USD"]], on="Date")
    dfoilppbeli = dfoilppbeli.rename(columns={"Price_USD": "Price_Beli"})
    dfoilppbeli["Log_Price_PP"] = np.log(dfoilppbeli["Price_PP"])
    dfoilppbeli["Log_Price_Oil"] = np.log(dfoilppbeli["Price_Oil"])
    dfoilppbeli["Log_Price_Beli"] = np.log(dfoilppbeli["Price_Beli"])
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    dfoilppbeli[["Scaled_Log_PP", "Scaled_Log_Oil", "Scaled_Log_Beli"]] = scaler.fit_transform(
        dfoilppbeli[["Price_PP", "Price_Oil", "Price_Beli"]]
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set ukuran plot
    plt.figure(figsize=(24, 6))

    # Plot masing-masing nilai log yang sudah di-scale
    sns.lineplot(data=dfoilppbeli, x="Date", y="Scaled_Log_PP", label="Scaled Price PP", marker="o")
    sns.lineplot(data=dfoilppbeli, x="Date", y="Scaled_Log_Oil", label="Scaled Price Oil", marker="s")
    sns.lineplot(data=dfoilppbeli, x="Date", y="Scaled_Log_Beli", label="Scaled Price Beli", marker="^")

    # Atur tampilan
    plt.xticks(rotation=45)  # Rotasi label sumbu X agar terbaca
    plt.title("Pergerakan Harga dalam Skala Normalisasi")
    plt.xlabel("Tanggal")
    plt.ylabel("Scaled Harga (0-1)")
    plt.legend()
    plt.grid(True)  # Tambahkan grid untuk kejelasan

    # Tampilkan plot
    st.write("WEEKLY")
    st.pyplot(plt)
    # Hitung korelasi antar variabel
    correlation_matrix = dfoilppbeli[["Price_PP", "Price_Oil", "Price_Beli"]].corr()

    # Tampilkan hasilnya
    st.write(correlation_matrix)
    dfoilppbelioriginal = dfoilppbeli.copy(deep=True)
    # st.write(dfoilppbelioriginal)
    def detect_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers
    # Deteksi outlier untuk masing-masing kolom
    outliers_price_pp = detect_outliers_iqr(dfoilppbeli, 'Price_PP')
    outliers_price_oil = detect_outliers_iqr(dfoilppbeli, 'Price_Oil')
    outliers_price_beli = detect_outliers_iqr(dfoilppbeli, 'Price_Beli')
    dfoilppbeli['Date'] = pd.to_datetime(dfoilppbeli['Date'])
    dfoilppbeli.set_index('Date', inplace=True)
    # Tandai Price_Oil outliers sebagai NaN
    dfoilppbeli.loc[outliers_price_oil['Date'], 'Price_Oil'] = np.nan

    # Interpolasi linear
    dfoilppbeli['Price_Oil'] = dfoilppbeli['Price_Oil'].interpolate(method='linear')

    # Hitung ulang log dan scaled log jika perlu
    dfoilppbeli['Log_Price_Oil'] = np.log(dfoilppbeli['Price_Oil'])

    # Jika sebelumnya ada scaling MinMax pada log
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    dfoilppbeli['Scaled_Log_Oil'] = scaler.fit_transform(dfoilppbeli[['Price_Oil']])

    # (Opsional) Reset index jika ingin kembali ke format semula
    dfoilppbeli.reset_index(inplace=True)
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller

    # Konversi Date ke datetime
    dfoilppbeli["Date"] = pd.to_datetime(dfoilppbeli["Date"])
    dfoilppbeli.set_index("Date", inplace=True)

    dfoilppdaily = dfoilppdaily.rename(columns={"Price_PP_USD": "Price_PP"})
    dfoilppdaily = dfoilppdaily.rename(columns={"Price": "Price_Oil"})
    dfoilpppreddaily = dfoilppdaily.copy(deep=True)
    dfoilppbelidaily = pd.merge(dfoilppdaily, daily_priceinterusd[["Date", "Price_USD"]], on="Date")
    dfoilppbelidaily = dfoilppbelidaily.rename(columns={"Price_USD": "Price_Beli"})
    dfoilppbelidaily["Log_Price_PP"] = np.log(dfoilppbelidaily["Price_PP"])
    dfoilppbelidaily["Log_Price_Oil"] = np.log(dfoilppbelidaily["Price_Oil"])
    dfoilppbelidaily["Log_Price_Beli"] = np.log(dfoilppbelidaily["Price_Beli"])
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    dfoilppbelidaily[["Scaled_Log_PP", "Scaled_Log_Oil", "Scaled_Log_Beli"]] = scaler.fit_transform(
        dfoilppbelidaily[["Price_PP", "Price_Oil", "Price_Beli"]]
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set ukuran plot
    plt.figure(figsize=(24, 6))

    # Plot masing-masing nilai log yang sudah di-scale
    sns.lineplot(data=dfoilppbelidaily, x="Date", y="Scaled_Log_PP", label="Scaled Price PP", marker="o")
    sns.lineplot(data=dfoilppbelidaily, x="Date", y="Scaled_Log_Oil", label="Scaled Price Oil", marker="s")
    sns.lineplot(data=dfoilppbelidaily, x="Date", y="Scaled_Log_Beli", label="Scaled Price Beli", marker="^")

    # Atur tampilan
    plt.xticks(rotation=45)  # Rotasi label sumbu X agar terbaca
    plt.title("Pergerakan Harga dalam Skala Normalisasi")
    plt.xlabel("Tanggal")
    plt.ylabel("Scaled Harga (0-1)")
    plt.legend()
    plt.grid(True)  # Tambahkan grid untuk kejelasan

    # Tampilkan plot
    st.write("DAILY")
    st.pyplot(plt)
    # Hitung korelasi antar variabel
    correlation_matrix = dfoilppbelidaily[["Price_PP", "Price_Oil", "Price_Beli"]].corr()

    # Tampilkan hasilnya
    st.write(correlation_matrix)

    dfoilppbelidailyoriginal = dfoilppbelidaily.copy(deep=True)
    def detect_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers
    # Deteksi outlier untuk masing-masing kolom
    outliers_price_pp = detect_outliers_iqr(dfoilppbelidaily, 'Price_PP')
    outliers_price_oil = detect_outliers_iqr(dfoilppbelidaily, 'Price_Oil')
    outliers_price_beli = detect_outliers_iqr(dfoilppbelidaily, 'Price_Beli')
    dfoilppbelidaily['Date'] = pd.to_datetime(dfoilppbelidaily['Date'])
    dfoilppbelidaily.set_index('Date', inplace=True)
    # Tandai Price_Oil outliers sebagai NaN
    dfoilppbelidaily.loc[outliers_price_oil['Date'], 'Price_Oil'] = np.nan

    # Interpolasi linear
    dfoilppbelidaily['Price_Oil'] = dfoilppbelidaily['Price_Oil'].interpolate(method='linear')

    # Hitung ulang log dan scaled log jika perlu
    dfoilppbelidaily['Log_Price_Oil'] = np.log(dfoilppbelidaily['Price_Oil'])

    # Jika sebelumnya ada scaling MinMax pada log
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    dfoilppbelidaily['Scaled_Log_Oil'] = scaler.fit_transform(dfoilppbelidaily[['Price_Oil']])

    # (Opsional) Reset index jika ingin kembali ke format semula
    dfoilppbelidaily.reset_index(inplace=True)

    if st.button("Start"):

        # === ANALISIS VAR BERDASARKAN DATA ASLI (setelah klik Start) ===

        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.stattools import grangercausalitytests
        import matplotlib.pyplot as plt

        st.subheader("Analisis VAR dan Granger Causality")

        # === WEEKLY ===
        st.write("VAR Mingguan (Data Asli)")
        df_var_weekly = dfoilppbelioriginal[['Price_PP', 'Price_Oil', 'Price_Beli']].copy(deep=True)
        df_var_weekly = df_var_weekly.dropna()

        # Difference untuk stasioneritas
        from statsmodels.tsa.stattools import adfuller

        def check_stationarity(series, name):
            result = adfuller(series.dropna())
            pvalue = result[1]
            st.write(f"ADF p-value ({name}): {pvalue:.4f}")
            return pvalue > 0.05  # True jika tidak stasioner

        # Cek satu-satu
        non_stationer_pp = check_stationarity(df_var_weekly['Price_PP'], "Price_PP")
        non_stationer_oil = check_stationarity(df_var_weekly['Price_Oil'], "Price_Oil")
        non_stationer_beli = check_stationarity(df_var_weekly['Price_Beli'], "Price_Beli")

        # Lakukan differencing hanya jika perlu
        if non_stationer_pp:
            df_var_weekly['D_PP'] = df_var_weekly['Price_PP'].diff()
        else:
            df_var_weekly['D_PP'] = df_var_weekly['Price_PP']

        if non_stationer_oil:
            df_var_weekly['D_Oil'] = df_var_weekly['Price_Oil'].diff()
        else:
            df_var_weekly['D_Oil'] = df_var_weekly['Price_Oil']

        if non_stationer_beli:
            df_var_weekly['D_Beli'] = df_var_weekly['Price_Beli'].diff()
        else:
            df_var_weekly['D_Beli'] = df_var_weekly['Price_Beli']

        df_diff_weekly = df_var_weekly[['D_PP', 'D_Oil', 'D_Beli']].dropna()

        model_weekly = VAR(df_diff_weekly)
        lag_selection = model_weekly.select_order(10)
        optimal_lag_weekly = lag_selection.aic  # atau .bic atau .hqic
        st.text("Lag Selection Weekly:\n" + str(lag_selection.summary()))

        if optimal_lag_weekly == 0:
            st.warning("⚠️ Lag optimal berdasarkan AIC adalah 0, namun VAR(0) tidak bisa digunakan untuk IRF. Menggunakan lag=1.")
            optimal_lag_weekly = 1

        var_result_weekly = model_weekly.fit(optimal_lag_weekly)
        st.text(f"VAR({optimal_lag_weekly}) Weekly Result:\n" + str(var_result_weekly.summary()))
        
        from statsmodels.tsa.stattools import grangercausalitytests
        import io
        import sys

        def capture_granger_output(df, maxlag):
            buffer = io.StringIO()
            sys.stdout = buffer
            grangercausalitytests(df, maxlag=maxlag, verbose=True)
            sys.stdout = sys.__stdout__
            return buffer.getvalue()




        st.write("Granger Causality (Mingguan)")
        st.text("Oil → PP")
        # grangercausalitytests(df_diff_weekly[['D_PP', 'D_Oil']], maxlag=3, verbose=True)
        output_oil_pp_weekly = capture_granger_output(df_diff_weekly[['D_PP', 'D_Oil']], optimal_lag_weekly)
        st.code(output_oil_pp_weekly, language='text')
        st.text("PP → Beli")
        # grangercausalitytests(df_diff_weekly[['D_Beli', 'D_PP']], maxlag=3, verbose=True)
        output_pp_beli_weekly = capture_granger_output(df_diff_weekly[['D_Beli', 'D_PP']], optimal_lag_weekly)
        st.code(output_pp_beli_weekly, language='text')

        st.write("Impulse Response Function (IRF) - Weekly")
        irf_weekly = var_result_weekly.irf(10)
        fig_irf = irf_weekly.plot(orth=True)
        st.pyplot(fig_irf)

        st.write("Forecast Error Variance Decomposition (FEVD) - Weekly")
        fevd_weekly = var_result_weekly.fevd(10)
        fig_fevd = fevd_weekly.plot()
        st.pyplot(fig_fevd)

        import pandas as pd
        import numpy as np
        from statsmodels.tsa.api import VAR
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
        import matplotlib.pyplot as plt

        n_obs = int(len(df_diff_weekly) * 0.8)
        train, test = df_diff_weekly.iloc[:n_obs], df_diff_weekly.iloc[n_obs:]

        model = VAR(train)
        # lag_order = model.select_order().aic  # pilih lag terbaik pakai AIC
        lag_order = optimal_lag_weekly
        var_model = model.fit(lag_order)

        forecast_input = train.values[-lag_order:]
        forecast_steps = len(test)

        forecast = var_model.forecast(y=forecast_input, steps=forecast_steps)
        forecast_df = pd.DataFrame(forecast, index=test.index, columns=df_diff_weekly.columns)

        y_true = test['D_Beli']
        y_pred = forecast_df['D_Beli']

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        st.write(f'RMSE: {rmse:.4f}')
        st.write(f'MAPE: {mape:.2f}%')

        plt.figure(figsize=(10,5))
        plt.plot(y_true, label='Actual Beli')
        plt.plot(y_pred, label='Forecasted Beli', linestyle='--')
        plt.legend()
        plt.title('Forecast vs Actual Beli')
        st.pyplot(plt)

        # === DAILY ===
        st.write("VAR Harian (Data Asli)")
        df_var_daily = dfoilppbelidailyoriginal[['Price_PP', 'Price_Oil', 'Price_Beli']].copy(deep=True)
        df_var_daily = df_var_daily.dropna()

        # Difference
        from statsmodels.tsa.stattools import adfuller

        # Cek satu-satu
        non_stationer_pp = check_stationarity(df_var_daily['Price_PP'], "Price_PP")
        non_stationer_oil = check_stationarity(df_var_daily['Price_Oil'], "Price_Oil")
        non_stationer_beli = check_stationarity(df_var_daily['Price_Beli'], "Price_Beli")

        # Lakukan differencing hanya jika perlu
        if non_stationer_pp:
            df_var_daily['D_PP'] = df_var_daily['Price_PP'].diff()
        else:
            df_var_daily['D_PP'] = df_var_daily['Price_PP']

        if non_stationer_oil:
            df_var_daily['D_Oil'] = df_var_daily['Price_Oil'].diff()
        else:
            df_var_daily['D_Oil'] = df_var_daily['Price_Oil']

        if non_stationer_beli:
            df_var_daily['D_Beli'] = df_var_daily['Price_Beli'].diff()
        else:
            df_var_daily['D_Beli'] = df_var_daily['Price_Beli']

        df_diff_daily = df_var_daily[['D_PP', 'D_Oil', 'D_Beli']].dropna()

        model_daily = VAR(df_diff_daily)
        lag_selection = model_daily.select_order(10)
        optimal_lag_daily = lag_selection.aic  # atau .bic atau .hqic
        st.text("Lag Selection Daily:\n" + str(lag_selection.summary()))

        if optimal_lag_daily == 0:
            st.warning("⚠️ Lag optimal berdasarkan AIC adalah 0, namun VAR(0) tidak bisa digunakan untuk IRF. Menggunakan lag=1.")
            optimal_lag_daily = 1

        var_result_daily = model_daily.fit(optimal_lag_daily)
        st.text(f"VAR({optimal_lag_daily}) Daily Result:\n" + str(var_result_daily.summary()))

        st.write("Granger Causality (Harian)")
        st.text("Oil → PP")
        # grangercausalitytests(df_diff_daily[['D_PP', 'D_Oil']], maxlag=3, verbose=True)
        output_oil_pp_daily = capture_granger_output(df_diff_daily[['D_PP', 'D_Oil']], optimal_lag_daily)
        st.code(output_oil_pp_daily, language='text')
        st.text("PP → Beli")
        # grangercausalitytests(df_diff_daily[['D_Beli', 'D_PP']], maxlag=3, verbose=True)
        output_pp_beli_daily = capture_granger_output(df_diff_daily[['D_Beli', 'D_PP']], optimal_lag_daily)
        st.code(output_pp_beli_daily, language='text')

        st.write("Impulse Response Function (IRF) - Daily")
        irf_daily = var_result_daily.irf(10)
        fig_irf_d = irf_daily.plot(orth=True)
        st.pyplot(fig_irf_d)

        st.write("Forecast Error Variance Decomposition (FEVD) - Daily")
        fevd_daily = var_result_daily.fevd(10)
        fig_fevd_d = fevd_daily.plot()
        st.pyplot(fig_fevd_d)

        n_obs = int(len(df_diff_daily) * 0.8)
        train, test = df_diff_daily.iloc[:n_obs], df_diff_daily.iloc[n_obs:]

        model = VAR(train)
        # lag_order = model.select_order().aic  # pilih lag terbaik pakai AIC
        lag_order = optimal_lag_daily
        var_model = model.fit(lag_order)

        forecast_input = train.values[-lag_order:]
        forecast_steps = len(test)

        forecast = var_model.forecast(y=forecast_input, steps=forecast_steps)
        forecast_df = pd.DataFrame(forecast, index=test.index, columns=df_diff_daily.columns)

        y_true = test['D_Beli']
        y_pred = forecast_df['D_Beli']

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        st.write(f'RMSE: {rmse:.4f}')
        st.write(f'MAPE: {mape:.2f}%')

        plt.figure(figsize=(10,5))
        plt.plot(y_true, label='Actual Beli')
        plt.plot(y_pred, label='Forecasted Beli', linestyle='--')
        plt.legend()
        plt.title('Forecast vs Actual Beli')
        st.pyplot(plt)


        st.write("VECM")
        st.write("WEEKLY")
        import pandas as pd
        import numpy as np
        from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        import matplotlib.pyplot as plt

        def slugify(text):
            return text.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("@", "").replace("/", "_")

        def generate_model_filename(produk_name, kode_list, periode):
            slug = slugify(produk_name)
            kode_str = "-".join(str(k) for k in kode_list)
            return f"pickle_save_parameter_hargabeli/model_vecm_{slug}_{kode_str}_{periode}.pkl"

        os.makedirs(r"C:\Users\RAVASHOP\Documents\kuliah\skripsi\coding\pickle_save_parameter_hargabeli", exist_ok=True)

        df_vecmweekly = dfoilppbeli[['Price_PP', 'Price_Oil', 'Price_Beli']].copy(deep=True)
        df_vecmweekly.index.name = 'Date'  # (opsional) beri nama index untuk kejelasan

        from statsmodels.tsa.api import VAR
        model_var = VAR(df_vecmweekly)
        selected_order=model_var.select_order(10)  # hanya untuk konfirmasi
        st.text(selected_order.summary())


        train_size = int(len(df_vecmweekly) * 0.8)
        train = df_vecmweekly.iloc[:train_size]
        test = df_vecmweekly.iloc[train_size:]

        model_var = VAR(train)
        selected_order = model_var.select_order(10)
        optimal_lag = selected_order.aic  # atau .bic atau .hqic
        k_diff = optimal_lag - 1

        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        johan = coint_johansen(train, det_order=0, k_ar_diff=k_diff)

        # Hitung rank: jumlah trace statistic > critical value 95%
        rank = sum(johan.lr1 > johan.cvt[:, 1])  # cvt[:, 1] = 95% level

        st.write("rank", rank)

        from statsmodels.tsa.vector_ar.vecm import VECM

        vecm_model = VECM(train, k_ar_diff=k_diff, coint_rank=rank, deterministic="co")
        vecm_result = vecm_model.fit()
        # Lihat ringkasan hasil
        n_test = len(test)
        forecast = vecm_result.predict(steps=n_test)

        # Buat dataframe forecast dengan index sama seperti test
        forecast_df = pd.DataFrame(forecast, columns=train.columns, index=test.index)
        y_true = test['Price_Beli']
        y_pred = forecast_df['Price_Beli']

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)

        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAPE: {mape:.2%}")
        # Ambil nama kolom dari original data
        columns = train.columns.tolist()  # atau df_train.columns jika kamu split

        # Ubah fittedvalues jadi DataFrame
        fit_df = pd.DataFrame(vecm_result.fittedvalues, columns=columns)

        # Ambil hanya kolom 'Price_Beli'
        fit = fit_df['Price_Beli']

        # Samakan index dengan bagian akhir train
        fit.index = train.index[-len(fit):]
        # Data pembanding
        actual = df_vecmweekly['Price_Beli']
        train = actual[:train_size]
        test = actual[train_size:]

        # # In-sample fit dari model
        # fit = vecm_result.fittedvalues['Price_Beli']
        # fit.index = train.index[-len(fit):]  # pastikan index cocok

        # Forecast ke depan (dari hasil vecm_result.predict)
        forecast = forecast_df['Price_Beli']

        # Plot
        plt.figure(figsize=(14, 5))
        plt.plot(train, label='Train (Actual)', color='black')
        plt.plot(test, label='Test (Actual)', color='gray')
        plt.plot(fit, label='In-Sample Prediction (Fit)', linestyle='--', color='blue')
        plt.plot(forecast, label='Forecast (Test)', linestyle='--', color='red')

        plt.title("VECM In-Sample Fit + Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price_Beli")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)


        st.write("DAILY")

        df_vecmdaily = dfoilppbelidaily[['Price_PP', 'Price_Oil', 'Price_Beli']].copy(deep=True)
        df_vecmdaily.index.name = 'Date'  # (opsional) beri nama index untuk kejelasan


        from statsmodels.tsa.api import VAR
        model_var = VAR(df_vecmdaily)
        selected_order=model_var.select_order(10)  # hanya untuk konfirmasi
        st.text(selected_order.summary())


        train_size = int(len(df_vecmdaily) * 0.8)
        train = df_vecmdaily.iloc[:train_size]
        test = df_vecmdaily.iloc[train_size:]

        model_var = VAR(train)
        selected_order = model_var.select_order(10)
        optimal_lag = selected_order.aic  # atau .bic atau .hqic
        k_diff = optimal_lag - 1

        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        johan = coint_johansen(train, det_order=0, k_ar_diff=k_diff)

        # Hitung rank: jumlah trace statistic > critical value 95%
        rank = sum(johan.lr1 > johan.cvt[:, 1])  # cvt[:, 1] = 95% level

        st.write("rank", rank)

        from statsmodels.tsa.vector_ar.vecm import VECM

        vecm_model = VECM(train, k_ar_diff=k_diff, coint_rank=rank, deterministic="co")
        vecm_result = vecm_model.fit()
        # Lihat ringkasan hasil
        n_test = len(test)
        forecast = vecm_result.predict(steps=n_test)

        # Buat dataframe forecast dengan index sama seperti test
        forecast_df = pd.DataFrame(forecast, columns=train.columns, index=test.index)
        y_true = test['Price_Beli']
        y_pred = forecast_df['Price_Beli']

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)

        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAPE: {mape:.2%}")
        # Ambil nama kolom dari original data
        columns = train.columns.tolist()  # atau df_train.columns jika kamu split

        # Ubah fittedvalues jadi DataFrame
        fit_df = pd.DataFrame(vecm_result.fittedvalues, columns=columns)

        # Ambil hanya kolom 'Price_Beli'
        fit = fit_df['Price_Beli']

        # Samakan index dengan bagian akhir train
        fit.index = train.index[-len(fit):]
        # Data pembanding
        actual = df_vecmdaily['Price_Beli']
        train = actual[:train_size]
        test = actual[train_size:]

        # # In-sample fit dari model
        # fit = vecm_result.fittedvalues['Price_Beli']
        # fit.index = train.index[-len(fit):]  # pastikan index cocok

        # Forecast ke depan (dari hasil vecm_result.predict)
        forecast = forecast_df['Price_Beli']

        # Plot
        plt.figure(figsize=(14, 5))
        plt.plot(train, label='Train (Actual)', color='black')
        plt.plot(test, label='Test (Actual)', color='gray')
        plt.plot(fit, label='In-Sample Prediction (Fit)', linestyle='--', color='blue')
        plt.plot(forecast, label='Forecast (Test)', linestyle='--', color='red')

        plt.title("VECM In-Sample Fit + Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price_Beli")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)



        def generate_model_filename_xgb_hargabeli(produk_name, kode_list, periode):
            slug = produk_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("@", "").replace("/", "_")
            kode_str = "-".join(str(k) for k in kode_list)
            return f"pickle_save_parameter_hargabeli/model_xgb_hargabeli_{slug}_{kode_str}_{periode}.pkl"

        st.write("XGBoost")
        st.write("WEEKLY")
        dfhargabelixgboost = dfoilppbelioriginal[["Date", "Price_PP", "Price_Oil", "Price_Beli"]].copy(deep=True)
        # dfhargabelixgboost['Date'] = dfhargabelixgboost.index
        dfhargabelixgboost.set_index('Date', inplace=True)
        dfhargabelixgboost.index = pd.to_datetime(dfhargabelixgboost.index)
        # dfhargabelixgboost = dfhargabelixgboost.reset_index(drop=True)
        dfhargabelixgboost['Price_Beli_lag1'] = dfhargabelixgboost['Price_Beli'].shift(1)
        dfhargabelixgboost = dfhargabelixgboost.dropna()
        from sklearn.model_selection import train_test_split

        X = dfhargabelixgboost[['Price_PP', 'Price_Oil', 'Price_Beli_lag1']]
        y = dfhargabelixgboost['Price_Beli']

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        from xgboost import XGBRegressor
        import optuna
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error
        import numpy as np

        # Define objective function buat Optuna
        def objective(trial):
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
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # Hitung MAPE
            mape = mean_absolute_percentage_error(y_test, y_pred) *100
            return mape  # Karena kita mau MAPE sekecil mungkin
        
        filename_xgb_hargabeli_weekly = generate_model_filename_xgb_hargabeli(produk_terpilih, kode_terpilih, "weekly")

        # Load atau train model
        if os.path.exists(filename_xgb_hargabeli_weekly):
            with open(filename_xgb_hargabeli_weekly, "rb") as f:
                saved = pickle.load(f)
            model = saved["model"]
            best_params = saved["params"]
            X_test = saved.get("X_test", X_test)
            mape = saved.get("mape")  # jika sudah ada
            st.success("Model XGBoost weekly untuk harga beli dimuat dari file.")
        else:
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100)
            best_params = study.best_params
            model = XGBRegressor(**best_params)
            model.fit(X_train, y_train)

            saved = {
                "model": model,
                "params": best_params
            }
            with open(filename_xgb_hargabeli_weekly, "wb") as f:
                pickle.dump(saved, f)
            st.success("💾 Model XGBoost weekly untuk harga beli berhasil disimpan.")


        # Prediksi in-sample
        y_train_pred = model.predict(X_train)

        # Prediksi test (forecast)
        y_test_pred = model.predict(X_test)
        # Gabungkan prediksi
        import pandas as pd

        # Pastikan y_train dan y_test punya index waktu
        y_train_pred = pd.Series(y_train_pred, index=y_train.index)
        y_test_pred = pd.Series(y_test_pred, index=y_test.index)
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
        import numpy as np

        # Hitung RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Hitung MAPE
        mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"MAPE: {mape:.2f}%")

        relative_error = rmse / np.mean(y_test)
        st.write(f"Relative RMSE: {relative_error:.2%}")

        # Update file jika perlu (X_test, mape, historical)
        updated = False
        if "X_test" not in saved:
            saved["X_test"] = X_test
            updated = True
        if "mape" not in saved:
            saved["mape"] = mape
            updated = True
        if "historical_data" not in saved:
            saved["historical_data"] = dfhargabelixgboost  # jika mau disimpan
            updated = True
        if updated:
            with open(filename_xgb_hargabeli_weekly, 'wb') as f:
                pickle.dump(saved, f)
            st.info("File model diperbarui (X_test/mape/historical).")

        import matplotlib.pyplot as plt

        plt.figure(figsize=(14,5))

        # Garis aktual
        plt.plot(y_train, label='Train (Actual)', color='black')
        plt.plot(y_test, label='Test (Actual)', color='gray')

        # Prediksi in-sample
        plt.plot(y_train_pred, label='In-Sample Prediction (Fit)', linestyle='--', color='blue')

        # Prediksi test (forecast)
        plt.plot(y_test_pred, label='Forecast (Test)', linestyle='--', color='red')

        # Estetika plot
        plt.title("XGBoost In-Sample Fit + Forecast")
        plt.xlabel("Date")
        plt.ylabel("Quantity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        st.pyplot(plt)


        st.write("DAILY")
        dfhargabelixgboostdaily = dfoilppbelidailyoriginal[["Date", "Price_PP", "Price_Oil", "Price_Beli"]].copy(deep=True)
        dfhargabelixgboostdaily.set_index('Date', inplace=True)
        dfhargabelixgboostdaily.index = pd.to_datetime(dfhargabelixgboostdaily.index)
        dfhargabelixgboostdaily['Price_Beli_lag1'] = dfhargabelixgboostdaily['Price_Beli'].shift(1)
        dfhargabelixgboostdaily = dfhargabelixgboostdaily.dropna()
        from sklearn.model_selection import train_test_split

        X = dfhargabelixgboostdaily[['Price_PP', 'Price_Oil', 'Price_Beli_lag1']]
        y = dfhargabelixgboostdaily['Price_Beli']

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        from xgboost import XGBRegressor
        import optuna
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error
        import numpy as np

        # Define objective function buat Optuna
        def objective(trial):
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
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # Hitung MAPE
            mape = mean_absolute_percentage_error(y_test, y_pred) *100
            return mape  # Karena kita mau MAPE sekecil mungkin
        
        filename_xgb_hargabeli_daily = generate_model_filename_xgb_hargabeli(produk_terpilih, kode_terpilih, "daily")

        historical_data_daily = dfhargabelixgboostdaily.copy(deep=True)

        if os.path.exists(filename_xgb_hargabeli_daily):
            with open(filename_xgb_hargabeli_daily, "rb") as f:
                saved = pickle.load(f)
            model = saved["model"]
            best_params = saved["params"]
            X_test = saved.get("X_test", X_test)
            mape = saved.get("mape")
            st.success("Model XGBoost daily untuk harga beli dimuat dari file.")
        else:
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100)
            best_params = study.best_params

            model = XGBRegressor(**best_params)
            model.fit(X_train, y_train)

            saved = {
                "model": model,
                "params": best_params
            }

            with open(filename_xgb_hargabeli_daily, "wb") as f:
                pickle.dump(saved, f)

            st.success("Model XGBoost daily untuk harga beli berhasil disimpan.")


        # Prediksi in-sample
        y_train_pred = model.predict(X_train)

        # Prediksi test (forecast)
        y_test_pred = model.predict(X_test)
        # Gabungkan prediksi
        import pandas as pd

        # Pastikan y_train dan y_test punya index waktu
        y_train_pred = pd.Series(y_train_pred, index=y_train.index)
        y_test_pred = pd.Series(y_test_pred, index=y_test.index)
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
        import numpy as np

        # Hitung RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Hitung MAPE
        mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"MAPE: {mape:.2f}%")

        relative_error = rmse / np.mean(y_test)
        st.write(f"Relative RMSE: {relative_error:.2%}")

        updated = False
        if "X_test" not in saved:
            saved["X_test"] = X_test
            updated = True
        if "mape" not in saved:
            saved["mape"] = mape
            updated = True
        if "historical_data" not in saved:
            saved["historical_data"] = historical_data_daily
            updated = True

        if updated:
            with open(filename_xgb_hargabeli_daily, 'wb') as f:
                pickle.dump(saved, f)
            st.info("File model XGBoost daily diperbarui (MAPE/X_test/historical_data).")

        import matplotlib.pyplot as plt

        plt.figure(figsize=(14,5))

        # Garis aktual
        plt.plot(y_train, label='Train (Actual)', color='black')
        plt.plot(y_test, label='Test (Actual)', color='gray')

        # Prediksi in-sample
        plt.plot(y_train_pred, label='In-Sample Prediction (Fit)', linestyle='--', color='blue')

        # Prediksi test (forecast)
        plt.plot(y_test_pred, label='Forecast (Test)', linestyle='--', color='red')

        # Estetika plot
        plt.title("XGBoost In-Sample Fit + Forecast")
        plt.xlabel("Date")
        plt.ylabel("Quantity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        st.pyplot(plt)

        import pandas as pd
        from sklearn.linear_model import LinearRegression
        import matplotlib.pyplot as plt

        # Load data
        df = dfoilppbelioriginal[["Date", "Price_Oil", "Price_PP"]].copy(deep=True)

        # Siapkan variabel
        X = df[['Price_Oil']]  # fitur input
        y = df['Price_PP']     # target yang ingin diprediksi

        from xgboost import XGBRegressor

        model_xgb = XGBRegressor()
        model_xgb.fit(X, y)
        r2 = model_xgb.score(X, y)
        st.write(f"R² XGBoost: {r2:.3f}")

        # Buat model
        model = LinearRegression()
        model.fit(X, y)

        # Cetak koefisien regresi
        st.write(f"Intercept (a): {model.intercept_:.2f}")
        st.write(f"Slope (b): {model.coef_[0]:.2f}")

        # Contoh prediksi untuk harga minyak tertentu
        oil_terbaru = 61.29
        pp_prediksi = model.predict([[oil_terbaru]])
        st.write(f"Prediksi Harga PP jika Oil = {oil_terbaru} → {pp_prediksi[0]:,.2f}")

        # Visualisasi
        plt.figure(figsize=(10, 5))
        plt.scatter(df['Price_Oil'], df['Price_PP'], label='Data Historis')
        plt.plot(df['Price_Oil'], model.predict(X), color='red', label='Linear Fit')
        plt.xlabel('Harga Minyak (USD)')
        plt.ylabel('Harga PP (CNY/ton)')
        plt.title('Regresi Linier: Oil vs PP')
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

        # Evaluasi model
        st.write(f"R² (koefisien determinasi): {model.score(X, y):.3f}")

        # import pandas as pd
        # import numpy as np
        # from xgboost import XGBRegressor
        # from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
        # from sklearn.model_selection import train_test_split
        # import matplotlib.pyplot as plt
        # import pickle

        # def evaluate_model(y_true, y_pred):
        #     rmse = mean_squared_error(y_true, y_pred, squared=False)
        #     mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        #     return rmse, mape

                
        # st.write("🚀 [WEEKLY] Prediksi Harga PP dari Harga Minyak")

        # df_weekly = dfoilppbelioriginal[["Price_Oil", "Price_PP"]].copy(deep=True)
        # X_weekly = df_weekly[["Price_Oil"]]
        # y_weekly = df_weekly["Price_PP"]

        # X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_weekly, y_weekly, test_size=0.2, shuffle=False)

        # model_xgb_weekly = XGBRegressor()
        # model_xgb_weekly.fit(X_train_w, y_train_w)

        # y_pred_w = model_xgb_weekly.predict(X_test_w)
        # rmse_w, mape_w = evaluate_model(y_test_w, y_pred_w)

        # st.write(f"✅ Weekly - RMSE: {rmse_w:.2f}, MAPE: {mape_w:.2f}%")

        # st.write("\n🚀 [DAILY] Prediksi Harga PP dari Harga Minyak")

        # df_daily = dfoilppbelidailyoriginal[["Price_Oil", "Price_PP"]].copy(deep=True)
        # X_daily = df_daily[["Price_Oil"]]
        # y_daily = df_daily["Price_PP"]

        # X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_daily, y_daily, test_size=0.2, shuffle=False)

        # model_xgb_daily = XGBRegressor()
        # model_xgb_daily.fit(X_train_d, y_train_d)

        # y_pred_d = model_xgb_daily.predict(X_test_d)
        # rmse_d, mape_d = evaluate_model(y_test_d, y_pred_d)

        # st.write(f"✅ Daily - RMSE: {rmse_d:.2f}, MAPE: {mape_d:.2f}%")

        import pandas as pd
        import numpy as np
        from xgboost import XGBRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
        from sklearn.model_selection import train_test_split
        import matplotlib.pyplot as plt
        import pickle
        import optuna
        import os

        def evaluate_model(y_true, y_pred):
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            return rmse, mape

        def objective(trial, X_train, y_train, X_test, y_test):
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
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, preds)
            return mape

        def train_and_save_model(df_oil, df_pp, period):
            # Gabungkan berdasarkan index (pastikan tanggal cocok)
            df_combined = pd.concat([df_oil["Price"], df_pp["Price"]], axis=1, join="inner").dropna()
            df_combined.columns = ["Price_Oil", "Price_PP"]

            X = df_combined[["Price_Oil"]]
            y = df_combined["Price_PP"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

            # Path penyimpanan
            save_dir = "pickle_save_parameter_hargabeli"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"model_pred_pp_xgb_{period}.pkl")

            # Cek apakah file sudah ada
            if os.path.exists(save_path):
                st.info(f"Model prediksi PP ({period}) sudah tersedia: {save_path} — tidak dilatih ulang.")
                st.write(X)
                st.write(y)
                with open(save_path, "rb") as f:
                    model_data = pickle.load(f)

                # model = model_data["model"]
                mape = model_data["mape"]
                rmse = model_data["rmse"]
                st.write(f"{save_path} - RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
                return
            else:

                # Optuna tuning
                study = optuna.create_study(direction="minimize")
                study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=300)

                best_params = study.best_params
                best_model = XGBRegressor(**best_params)
                best_model.fit(X_train, y_train)
                preds = best_model.predict(X_test)

                rmse, mape = evaluate_model(y_test, preds)
                st.success(f"{period.upper()} - RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

                # Simpan model dan parameter
                with open(save_path, "wb") as f:
                    pickle.dump({
                        "model": best_model,
                        "params": best_params,
                        "rmse": rmse,
                        "mape": mape
                    }, f)
                st.success(f"Model disimpan: {save_path}")


        # === WEEKLY ===
        st.write("[WEEKLY] Prediksi Harga PP dari Harga Minyak")
        df_weekly_oil = dfoilweekly[["Price"]].copy(deep=True)
        df_weekly_pp = dfppweekly[["Price"]].copy(deep=True)
        train_and_save_model(df_weekly_oil, df_weekly_pp, "weekly")

        # === DAILY ===
        st.write("[DAILY] Prediksi Harga PP dari Harga Minyak")
        df_daily_oil = dfoil[["Price"]].copy(deep=True)
        df_daily_pp = dfpp[["Price"]].copy(deep=True)
        train_and_save_model(df_daily_oil, df_daily_pp, "daily")

        import pandas as pd
        import numpy as np
        import os
        import pickle
        from xgboost import XGBRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
        from sklearn.preprocessing import StandardScaler

        def prepare_features(df, target_col):
            df = df.sort_index()
            df[f'{target_col}_lag1'] = df[target_col].shift(1)
            df[f'{target_col}_lag2'] = df[target_col].shift(2)
            df['rolling_mean_3'] = df[target_col].rolling(window=3).mean()
            df['month'] = df.index.month
            df = df.dropna()
            return df

        def train_xgb_model(df, target_col, model_name, save_folder):
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"model_xgb_{model_name.lower()}.pkl")

            # Cek jika file sudah ada
            if os.path.exists(save_path):
                st.write(f"File model '{model_name}' sudah ada, skip training.")
                with open(save_path, "rb") as f:
                    model_data = pickle.load(f)

                # model = model_data["model"]
                mape = model_data["mape"]
                rmse = model_data["rmse"]
                st.write(f"{save_path} - RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
                # X_test = model_data["X_test"]
                # historical_data = model_data["historical_data"]

                return

            df_feat = prepare_features(df, target_col)

            X = df_feat[[f'{target_col}_lag1', f'{target_col}_lag2', 'rolling_mean_3', 'month']]
            y = df_feat[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

            model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100

            st.write(f"{target_col.upper()} - RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

            model_data = {
                "model": model,
                "params": model.get_params(),
                "rmse": rmse,
                "mape": mape,
                "X_test": X_test,
                "historical_data": df_feat
            }

            with open(save_path, "wb") as f:
                pickle.dump(model_data, f)

            st.write(f"Model {model_name} saved to: {save_path}")


        # ========== USAGE ==========
        # Pastikan df memiliki index datetime dan kolom Price_Oil dan Price_PP

        # df = pd.read_csv("data_harga.csv", parse_dates=["Date"], index_col="Date")
        # df = df[['Price_Oil', 'Price_PP']]  # pastikan hanya kolom yang dibutuhkan

        # train_xgb_model(df, target_col="Price_Oil", model_name="oil", save_folder="pickle_save_parameter_hargabeli")
        # train_xgb_model(df, target_col="Price_PP", model_name="pp", save_folder="pickle_save_parameter_hargabeli")

        # Data mingguan
        df_weekly = dfoilpppredweekly[['Price_Oil', 'Price_PP']]

        # Data harian
        df_daily = dfoilpppreddaily[['Price_Oil', 'Price_PP']]

        # Training model untuk weekly
        train_xgb_model(df_weekly, target_col="Price_Oil", model_name="oil_weekly", save_folder="pickle_save_parameter_hargabeli")
        train_xgb_model(df_weekly, target_col="Price_PP", model_name="pp_weekly", save_folder="pickle_save_parameter_hargabeli")

        # Training model untuk daily
        train_xgb_model(df_daily, target_col="Price_Oil", model_name="oil_daily", save_folder="pickle_save_parameter_hargabeli")
        train_xgb_model(df_daily, target_col="Price_PP", model_name="pp_daily", save_folder="pickle_save_parameter_hargabeli")

























