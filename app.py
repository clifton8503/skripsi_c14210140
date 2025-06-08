import streamlit as st
import forecastdemand
import forecasthargabeli
import rekomendasi_pembelian
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import datetime

def tampilanhome():

    # === Judul Halaman ===
    st.title("📊 Dashboard")

    # === Koneksi ke database ===
    engine = create_engine("mysql+pymysql://root:@localhost/tifalanggeng")

    # === Query data dari tabel `hargaoildaily` ===
    query = "SELECT Date, Price FROM hargaoildaily ORDER BY Date DESC"
    df_oil = pd.read_sql(query, con=engine, parse_dates=['Date'])

    # === Pastikan ada data ===
    if df_oil.empty:
        st.warning("Data harga minyak belum tersedia di database.")
    else:
        # === Harga terbaru (baris pertama setelah di-sort DESC) ===
        harga_terbaru = df_oil.iloc[0]
        st.metric("🛢️ Harga Minyak Terbaru", f"${harga_terbaru['Price']:.2f}", help=f"Tanggal: {harga_terbaru['Date'].strftime('%Y-%m-%d')}")

        # === Ambil 30 hari terakhir ===
        satu_bulan_lalu = pd.Timestamp.today() - pd.Timedelta(days=30)
        df_oil_last_month = df_oil[df_oil['Date'] >= satu_bulan_lalu].sort_values('Date')

        # === Grafik historis harga minyak ===
        plt.figure(figsize=(10, 4))
        plt.plot(df_oil_last_month['Date'], df_oil_last_month['Price'], marker='o', linestyle='-')
        plt.title("Harga Minyak Bumi (30 Hari Terakhir)")
        plt.xlabel("Tanggal")
        plt.ylabel("Harga (USD)")
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(plt)
    # Query tren penjualan harian (jumlah quantity)
        query_tren = """
        SELECT Invoice_Date, SUM(Total_Amount) AS Total_Quantity
        FROM penjualan
        WHERE Invoice_Date >= (
            SELECT MAX(Invoice_Date) FROM penjualan
        ) - INTERVAL 60 DAY
        GROUP BY Invoice_Date
        ORDER BY Invoice_Date
        """
        df_tren = pd.read_sql(query_tren, con=engine, parse_dates=['Invoice_Date'])

        # Query Top 10 Produk Terjual
        query_top_produk = """
        SELECT keterangan_barang, SUM(Total_Amount) as Total_Quantity
        FROM penjualan
        GROUP BY keterangan_barang
        ORDER BY Total_Quantity DESC
        LIMIT 10
        """
        df_top_produk = pd.read_sql(query_top_produk, con=engine)

        st.subheader("📈 Tren Penjualan Harian")

        plt.figure(figsize=(10, 4))
        plt.plot(df_tren['Invoice_Date'], df_tren['Total_Quantity'], marker='o')
        plt.title("Tren Total Penjualan per Hari")
        plt.xlabel("Tanggal")
        plt.ylabel("Total Penjualan (rupiah)")
        plt.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(plt)

        st.subheader("🏆 Top 10 Produk Terjual")

        plt.figure(figsize=(10, 4))
        plt.barh(df_top_produk['keterangan_barang'][::-1], df_top_produk['Total_Quantity'][::-1])
        plt.xlabel("Total Quantity")
        plt.title("Top 10 Produk Berdasarkan Penjualan (rupiah)")
        plt.tight_layout()
        st.pyplot(plt)

        # Ambil data kurs USD ke IDR
        query_kurs = "SELECT Date, Price FROM kursidrusddaily ORDER BY Date DESC"
        df_kurs = pd.read_sql(query_kurs, con=engine, parse_dates=['Date'])
        if not df_kurs.empty:
            kurs_terbaru = df_kurs.iloc[0]
            st.metric("💱 Kurs USD ke Rupiah Terbaru", f"Rp {kurs_terbaru['Price']:,.0f}", help=f"Tanggal: {kurs_terbaru['Date'].strftime('%Y-%m-%d')}")
        else:
            st.warning("❗ Data kurs USD-IDR belum tersedia.")

        # Filter 30 hari terakhir
        satu_bulan_lalu = pd.Timestamp.today() - pd.Timedelta(days=30)
        df_kurs_last_month = df_kurs[df_kurs['Date'] >= satu_bulan_lalu].sort_values('Date')

        st.subheader("📉 Grafik Kurs USD ke Rupiah (30 Hari Terakhir)")

        plt.figure(figsize=(10, 4))
        plt.plot(df_kurs_last_month['Date'], df_kurs_last_month['Price'], marker='o', linestyle='-')
        plt.title("Kurs USD ke Rupiah Selama 30 Hari Terakhir")
        plt.xlabel("Tanggal")
        plt.ylabel("Kurs (IDR)")
        plt.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(plt)

produk_dict = {
    "Thinwall Victory Square 2000ml (6pack@25pcs)": [101101, 101429],
    "Thinwall Victory Rect 650ml (20pack@25pcs)": [101094, 101430],
    "Thinwall Victory Square 3000ml (6pack@25pcs)": [101102, 101446],
    "Gelas 120ml BLB Natural (40slop@50pcs)": [101159],
    "Gelas HOK 16oz (40slop@50pcs)": [101116, 101475],
    "Gelas HOK 14oz (40slop@50pcs)": [101115, 101474],
    "Sendok Makan (30pack@100pcs)": [101062, 101426],
    "Gelas HOK 12oz (40slop@50pcs)": [101114, 101473],
    "Thinwall Victory Rect 500ml (20pack@25pcs)": [101093, 101437]
}

st.sidebar.title("Pilih Produk")

# Tambahkan opsi default kosong
produk_options = ["-- Pilih Produk --"] + list(produk_dict.keys())
produk_terpilih = st.sidebar.selectbox("Produk", produk_options)



if produk_terpilih != "-- Pilih Produk --":
    kode_terpilih = produk_dict[produk_terpilih]

    tab = st.sidebar.radio("Menu", ["Demand Forecasting", "Price Prediction", "Rekomendasi Pembelian", "Home"])

    if produk_terpilih != "-- Pilih Produk --":
        kode_terpilih = produk_dict[produk_terpilih]

        if tab == "Demand Forecasting":
            forecastdemand.main(produk_terpilih, kode_terpilih)
        elif tab == "Price Prediction":
            forecasthargabeli.main(produk_terpilih, kode_terpilih)
        elif tab == "Rekomendasi Pembelian":
            rekomendasi_pembelian.main(produk_terpilih, kode_terpilih)
        elif tab == "Home":
            tampilanhome()
    else:
        st.info("Silakan pilih produk terlebih dahulu di sidebar.")

else:

    st.write("Silakan pilih produk terlebih dahulu di sidebar.")

    # if tab ==

    tampilanhome()














    

