# 🎯 Proyek UAS Kecerdasan Buatan: Prediksi Konversi Pembelian Pengunjung Website E-Commerce
### Repositori Resmi Kelompok 17 - Informatika A

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Focus-Machine%20Learning-orange.svg?style=for-the-badge)

Repositori ini disusun sebagai pemenuhan tugas **Ujian Akhir Semester (UAS) Mata Kuliah Kecerdasan Buatan / Artificial Intelligence**. Proyek ini mendemonstrasikan proses ujung-ke-ujung (*end-to-end*) dari implementasi *Machine Learning* untuk menyelesaikan permasalahan klasifikasi biner pada ranah bisnis digital, yaitu memprediksi intensi atau keputusan pembelian pengunjung situs web *e-commerce* (`Revenue = True` atau `False`) berdasarkan aktivitas jejak sesi (*clickstream*) mereka.

---

## 📌 Identitas Pengembang Kelompok 17
* **Anggota 1:** Cecep Faisal Ahmad (NIM: 2406033)
* **Anggota 2:** Muhammad Yusuf Faturahman (NIM: 2406002)
* **Kelas:** Informatika A
* **Program Studi:** Teknik Informatika
* **Dosen Pengampu:** Leni Fitriani, ST. M.Kom.

---

## 🛠️ Alat dan Teknologi (Tech Stack)
Proyek ini dibangun menggunakan ekosistem bahasa pemrograman **Python** dengan pustaka (*libraries*) standar industri:
* **Pengolahan Data:** `Pandas`, `NumPy`
* **Visualisasi Data:** `Matplotlib`, `Seaborn`
* **Machine Learning:** `Scikit-Learn` (Model, Preprocessing, dan Evaluasi Metrics) & `Imbalanced-Learn` (SMOTE)
* **Lingkungan Kerja:** `Jupyter Notebook` / `Google Colab`

---

## 💼 Latar Belakang & Pemahaman Bisnis (*Business Understanding*)
Dalam industri *e-commerce*, mayoritas pengunjung website hanya melakukan aktivitas penjelajahan sekilas (*window shopping*) lalu keluar tanpa bertransaksi. Mengidentifikasi pengunjung yang memiliki niat beli tinggi (*high-intent*) secara *real-time* sebelum mereka menutup halaman adalah tantangan besar. 

Proyek ini memberikan solusi berupa sistem **Predictive Marketing Analytics**. Dengan memprediksi potensi pembelian pelanggan secara instan, tim pemasar dapat menyaring audiens secara efisien. Anggaran promosi dapat dihemat dengan membatasi pemberian insentif voucher belanja atau *pop-up* diskon kilat hanya kepada segmen pengunjung yang terdeteksi "ragu-ragu namun potensial". Langkah taktis ini berfungsi optimal demi menekan biaya operasional sekaligus mendongkrak tingkat konversi penjualan (*conversion rate*) perusahaan.

---

## 📊 Bedah Mendalam Dataset (*Online Shoppers Intention Deep Dive*)

Dataset yang digunakan dalam penelitian ini adalah **Online Shoppers Purchasing Intention Dataset** dari UCI Machine Learning Repository, yang mencerminkan 12.330 sesi kunjungan unik dalam platform e-commerce selama periode satu tahun. Dataset ini sangat berharga karena tidak hanya merekam data demografis statis, melainkan perilaku navigasi dinamis pengguna dari detik ke detik.

### A. Taksonomi & Karakteristik Fitur (18 Atribut)
Fitur-fitur di dalam dataset ini dapat diklasifikasikan ke dalam 3 kategori utama:

1. **Metrik Perilaku Navigasi Halaman (*Behavioral Clickstream Metrics*):**
   * `Administrative`, `Informational`, `ProductRelated`: Menunjukkan jumlah halaman dengan fungsi spesifik yang dibuka dalam satu sesi. `Administrative` mengacu pada akun atau pelacakan, `Informational` pada profil perusahaan/kontak, dan `ProductRelated` pada etalase produk.
   * `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`: Total waktu (dalam detik) yang dihabiskan user pada jenis halaman tersebut. Fitur ini sangat fluktuatif dan merepresentasikan tingkat keterlibatan konsumen (*user engagement*).

2. **Metrik Kualitas dan Nilai Situs (*Web Analytics Metrics*):**
   * `BounceRates`: Persentase sesi di mana pengguna masuk ke suatu halaman dan langsung keluar tanpa memicu interaksi lanjutan ke halaman lain.
   * `ExitRates`: Dihitung per halaman untuk melacak seberapa sering halaman tertentu menjadi titik akhir penutupan sesi penjelajahan.
   * `PageValues`: Indikator finansial terpenting (fitur paling krusial bagi model). Atribut ini menghitung nilai rata-rata halaman web yang dikunjungi pengguna sesaat sebelum menyelesaikan transaksi pembelian (Shathya, 2025).

3. **Metrik Kontekstual, Waktu, dan Perangkat (*Contextual & Temporal Attributes*):**
   * `SpecialDay`: Mengukur kedekatan waktu kunjungan dengan hari-hari perayaan tertentu (misalnya Hari Ibu atau Valentine) atau festival belanja besar. Nilainya berkisar antara 0.0 hingga 1.0 (semakin dekat ke hari H, nilainya semakin tinggi).
   * `Month`: Bulan dilakukannya kunjungan (10 bulan terekam, tidak termasuk Januari dan April).
   * `OperatingSystems`, `Browser`, `Region`, `TrafficType`: Parameter tipe sistem operasi, jenis web browser, kode area wilayah geografis, dan sumber trafik kedatangan (misalnya iklan berbayar, pencarian organik, atau direct link).
   * `VisitorType`: Jenis kategori pengunjung situs (`New Visitor`, `Returning Visitor`, atau `Other`).
   * `Weekend`: Indikator biner (`True`/`False`) yang menandakan apakah kunjungan terjadi pada hari libur akhir pekan.

### B. Analisis Masalah Ketidakseimbangan Kelas (*The Class Imbalance Problem*)
Tantangan matematis terbesar dari dataset ini terletak pada variabel target **`Revenue`**. Distribusi kelas sangat tidak seimbang (*highly imbalanced*):
* **Kelas `False` (Tidak Membeli):** 10.422 sesi (~84.5%)
* **Kelas `True` (Membeli):** 1.908 sesi (~15.5%)

Secara teoritis, jika dataset ini langsung dimasukkan ke dalam algoritma klasifikasi tanpa penanganan khusus, model akan mengalami fenomena *akurasi semu* (bias mayoritas). Model cenderung menebak semua data sebagai `False` untuk mendapatkan akurasi 84.5%, namun gagal total dalam mendeteksi pembeli potensial (`True`). Oleh karena itu, penerapan teknik **SMOTE** pada proyek ini menjadi sebuah keharusan mutlak dalam pipa pemrosesan data (Obiedat, 2020; Nugroho H dkk., 2025).

