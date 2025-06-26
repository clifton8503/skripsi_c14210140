# update_data.py
from datetime import datetime
from scrape_insert import update_data_to_db  # Ganti nama_file_import dengan nama script yang berisi fungsi

# Cek apakah hari ini bukan Sabtu (5) atau Minggu (6)
today = datetime.today()
if today.weekday() < 5:  # 0 = Monday, ..., 4 = Friday
    update_data_to_db("daily")
else:
    print("Hari ini akhir pekan (Sabtu/Minggu) — tidak update.")

