# KARST-FUND v8 — Dashboard Pembiayaan Risiko Kekeringan

Prototipe penelitian interaktif untuk simulasi kebutuhan pembiayaan bantuan air, dana cadangan, perbandingan Separate/Hybrid/Pooling, dan evaluasi target cakupan.

## Cara menjalankan di laptop
```bash
pip install -r requirements.txt
streamlit run karst_fund_app.py
```

## Cara membuat tautan publik
1. Buat repository baru di GitHub.
2. Unggah seluruh isi folder ini (termasuk folder `data/` dan `.streamlit/`).
3. Buka Streamlit Community Cloud, masuk menggunakan GitHub.
4. Pilih repository dan file utama `karst_fund_app.py`, lalu Deploy.
5. Bagikan URL aplikasi yang dihasilkan.

## Data
- Data bawaan: `data/KARST_FUND_Analisis_Kekeringan.xlsx` (berisi hasil analisis historis yang diberikan untuk proyek).
- Pengunjung dapat langsung mencoba aplikasi tanpa mengunggah Excel.
- Pengguna juga dapat mengunggah Excel lain dari sidebar, dengan sheet `Data_Bulanan_SPI` dan `Episode_Kekeringan` serta kolom `Durasi_Bulan`.

## Batasan penting
- Simulasi didasarkan pada asumsi, bukan estimasi resmi BPBD.
- SPI tingkat kabupaten bukan indikator akses air setiap kapanewon.
- Biaya per rit dan proporsi sasaran adalah parameter skenario.
- Simulasi Monte Carlo dan optimisasi integer bisa memakan waktu di hosting gratis.
- Data dan kode di repository publik dapat dilihat orang lain; jangan masukkan data rahasia.

## Perubahan kecil untuk deployment
- Menggunakan Excel bawaan secara otomatis bila tidak ada file unggahan.
- Memperjelas label gap anggaran awal dan memisahkan status kelayakan finansial dari audit aritmetika.
