# update_data.py
from datetime import datetime
from scrape_insert import update_data_to_db  # Ganti nama_file_import dengan nama script yang berisi fungsi

update_data_to_db("weekly")   # Untuk harian
# update_data_to_db("weekly") # Untuk mingguan (bisa buat file terpisah jika mau beda jadwal)
with open("log_update.txt", "a") as f:
    f.write(f"{datetime.now()}: Data berhasil diupdate\n")
