# UAS-KecerdasanBuatan-Kelompok-17
# 🎯 Proyek UAS Kecerdasan Buatan: Prediksi Keputusan Pembelian Klasifikasi Social Media Ads
### repositori resmi kelompok 17 - informatika a

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Focus-Machine%20Learning-orange.svg?style=for-the-badge)

Repositori ini disusun sebagai pemenuhan tugas **Ujian Akhir Semester (UAS) Mata Kuliah Kecerdasan Buatan / Artificial Intelligence**. Proyek ini mendemonstrasikan proses ujung-ke-ujung (*end-to-end*) dari implementasi *Machine Learning* untuk menyelesaikan permasalahan klasifikasi biner pada ranah pemasaran digital (*digital marketing*), yaitu memprediksi probabilitas seorang pengguna untuk membeli produk (*purchased*) setelah terpapar iklan media sosial berdasarkan atribut demografisnya.

---

## 📑 Daftar Isi
1. [Identitas Pengembang](#-identitas-pengembang-kelompok-17)
2. [Alat dan Teknologi](#️-alat-dan-teknologi-tech-stack)
3. [Latar Belakang & Pemahaman Bisnis](#-latar-belakang--pemahaman-bisnis-business-understanding)
4. [Pemahaman Data](#-pemahaman-data-data-understanding)
5. [Struktur Repositori](#-struktur-repositori)
6. [Metodologi Proyek & Alur Kerja](#-metodologi-proyek--alur-kerja-machine-learning-pipeline)
7. [Hasil dan Insight Utama](#-hasil-dan-insight-utama)
8. [Panduan Eksekusi Program](#-panduan-eksekusi-program)
9. [Rencana Pengembangan](#-rencana-pengembangan-future-work)
10. [Referensi Ilmiah (APA Style)](#-referensi-ilmiah-apa-style)

---

## 📌 Identitas Pengembang Kelompok 17
* **Anggota 1:** Muhammad Yusuf Faturahman (NIM: 2406002)
* **Anggota 2:** Cecep Faisal Ahmad (NIM: 2406033)
* **Kelas:** Informatika A
* **Program Studi:** Teknik Informatika
* **Instansi:** [Isi Nama Kampus Kamu, misal: Universitas Garut]
* **Dosen Pengampu:** Leni Fitriani, ST. M.Kom.

---

## 🛠️ Alat dan Teknologi (Tech Stack)
Proyek ini dibangun menggunakan ekosistem bahasa pemrograman **Python** dengan pustaka (*libraries*) pengolahan data standar industri:
* **Pengolahan & Manipulasi Data:** `Pandas`, `NumPy`
* **Visualisasi Data Dinamis:** `Matplotlib`, `Seaborn`
* **Sains Data & Machine Learning:** `Scikit-Learn` (Model, Preprocessing, dan Evaluasi Metrics)
* **Lingkungan Kerja:** `Jupyter Notebook` / `Google Colab`

---

## 💼 Latar Belakang & Pemahaman Bisnis (*Business Understanding*)
Dalam industri *digital marketing*, membuang-buang anggaran iklan untuk menargetkan audiens yang salah atau tidak relevan (*ad-spend waste*) merupakan salah satu inefisiensi biaya terbesar perusahaan. 

Proyek ini hadir untuk memberikan solusi berbasis **Predictive Analytics**. Dengan memprediksi potensi pembelian pelanggan sebelum kampanye iklan dieksekusi, tim pemasar dapat menyaring audiens non-potensial terlebih dahulu. Anggaran pemasaran dapat difokuskan secara optimal hanya kepada segmen pengguna dengan probabilitas konversi tinggi. Hasil akhirnya adalah meminimalkan kerugian biaya operasional dan meningkatkan nilai *Return on Ad Spend* (ROAS) perusahaan secara signifikan.

---

## 📊 Pemahaman Data (*Data Understanding*)
Dataset yang digunakan dalam proyek ini adalah data publik yang telah dianonimkan untuk menjaga privasi pengguna media sosial.
* **Sumber Data Asli:** [Kaggle - Social Network Ads Dataset](https://www.kaggle.com/datasets/d4rklucif3r/social-network-ads)
* **Format & Dimensi:** Berkas berkode `.csv` | Memuat 400 Baris Data | Terdiri atas 5 Kolom Atribut.
* **Deskripsi Fitur & Variabel:**
  * `User ID`: Identitas unik numerik pengguna (Dihapus saat pemrosesan karena tidak memiliki nilai prediktif).
  * `Gender`: Jenis kelamin pengguna (Kategorikal: *Male*, *Female*).
  * `Age`: Usia pengguna saat ini dalam satuan tahun (Numerik).
  * `EstimatedSalary`: Estimasi pendapatan/gaji tahunan pengguna dalam satuan USD (Numerik).
  * `Purchased` **[TARGET VARIABLE]**: Keputusan akhir pengguna terhadap iklan (Biner: `0` = Tidak Membeli, `1` = Membeli).

---

## 📁 Struktur Repositori
Susunan direktori di dalam proyek ini diatur secara modular agar mempermudah proses dokumentasi dan pemeriksaan kode program:

```text
UAS-KecerdasanBuatan-Kelompok-17/
│
├── data/
│   ├── dataset/
│   │   ├── baca-me.md
│   │   └── Social_Network_Ads.csv          # File dataset mentah dari Kaggle
│   └── jurnal/
│       ├── baca-me.md
│       ├── Ramadhan_2023_KNN.pdf           # File PDF Jurnal Referensi 1
│       ├── Pratama_2023_RandomForest.pdf   # File PDF Jurnal Referensi 2
│       └── [PDF Jurnal Lainnya]            # PDF Jurnal pendukung berformat APA
│
├── Laporan_uas.md                          # Laporan tertulis akademik format Markdown
├── uas_model.ipynb                         # Jupyter Notebook kode eksperimen Python
└── README.md                               # Dokumentasi utama repositori (File ini)
