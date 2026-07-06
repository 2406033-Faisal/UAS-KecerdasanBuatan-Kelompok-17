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

# Prediksi Niat Beli Konsumen E-Commerce Menggunakan Machine Learning

Repositori ini berisi dokumentasi eksperimen, kode program, dan analisis proyek data science untuk memprediksi niat pembelian pengunjung website e-commerce (*Online Shoppers Purchasing Intention*). Projek ini disusun untuk memenuhi tugas akhir/UAS mata kuliah Kecerdasan Buatan (Kelompok 17).

---

## 1. Eksplorasi Data (Exploratory Data Analysis - EDA)
Tahap awal diisi dengan menganalisis persebaran data secara visual dan mendeteksi keberadaan ketidakseimbangan kelas target. Melalui visualisasi diagram kotak (*boxplot*), terungkap bahwa fitur `PageValues` bertindak sebagai pemisah alami utama antara pengunjung transaksional dan pengunjung kasual (Shathya, 2025).

## 2. Pra-Pemrosesan Data (Data Preparation)
* **Pembersihan Data:** Menghapus baris data duplikat sebanyak 125 baris untuk menjaga keaslian variansi data operasional.
* **Encoding Data:** Mentransformasikan fitur teks kategorikal non-numerik seperti nama bulan (`Month`) dan tipe pengunjung (`VisitorType`) menjadi representasi angka digital ordinal menggunakan `LabelEncoder`.
* **Penanganan Data Tidak Seimbang (SMOTE):** Karena kelas minoritas `True` (Membeli) hanya menempati porsi kurang dari 16%, teknik SMOTE diterapkan khusus pada data latih guna mensintesis sampel baru secara artifisial hingga perbandingan kelas seimbang 50:50 (Nugroho H dkk., 2025).
* **Feature Scaling:** Seluruh atribut numerik distandarisasi menggunakan `StandardScaler` ($Z\text{-score standardization}$) agar komputasi tidak bias akibat perbedaan rentang nilai angka antar-kolom.
* **Pembagian Data:** Dataset dipisahkan secara terstratifikasi dengan proporsi 80% untuk Data Pelatihan (*Train Set*) dan 20% untuk Data Pengujian (*Test Set*).

## 3. Pemodelan AI (Modeling)
Eksperimen ini menguji dua algoritma klasifikasi dengan karakteristik kalkulasi matematika berbeda:
1. **K-Nearest Neighbors (KNN):** Dikonfigurasikan dengan parameter kedekatan $k=5$ berbasis pengukuran jarak spasial *Euclidean Distance* (Meka, 2024).
2. **Support Vector Machine (SVM):** Diimplementasikan menggunakan fungsi kernel *Radial Basis Function* (RBF) untuk memetakan ruang fitur ke dalam dimensi yang lebih tinggi demi menemukan bidang pemisah (*hyperplane*) optimal (Obiedat, 2020).

---

## 💡 Hasil dan Insight Utama
Melalui pengujian pada data uji murni yang disisihkan (20% sampel), berikut adalah rangkuman performa kedua model kecerdasan buatan setelah penyeimbangan data menggunakan SMOTE:

| Model Algoritma | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | 0.8779 | 0.6780 | 0.4188 | 0.5178 |
| **Support Vector Machine (SVM)** | **0.8923 (Unggul)** | **0.7429 (Unggul)** | **0.4764 (Unggul)** | **0.5805 (Unggul)** |

### Kesimpulan Akhir Proyek
Berdasarkan hasil komparasi di atas, algoritma **Support Vector Machine (SVM) dengan Kernel RBF** keluar sebagai model terbaik. SVM terbukti jauh lebih kokoh dalam meredam tingkat kesalahan klasifikasi salah sasaran (*False Positive*), yang ditandai dengan capaian nilai Akurasi tertinggi sebesar **89.23%** serta nilai Presisi puncak mencapai **74.29%**. Penerapan metode penyeimbangan data SMOTE terbukti krusial dalam menstabilkan performa generalisasi model (Obiedat, 2020; Nugroho H dkk., 2025).

---

## 🛠️ Panduan Eksekusi Program
Bagi penguji yang ingin mereplikasi kodingan eksperimen Kelompok 17 di komputer lokal, silakan ikuti instruksi berikut:

```bash
git clone [https://github.com/MuhammadYusufFaturahman/UAS-KecerdasanBuatan-Kelompok-17.git](https://github.com/MuhammadYusufFaturahman/UAS-KecerdasanBuatan-Kelompok-17.git)
cd UAS-KecerdasanBuatan-Kelompok-17

## 🚀 Rencana Pengembangan (Future Work)

* **Kelebihan & Keterbatasan Model:** Algoritma KNN unggul dalam memetakan batas keputusan non-linear pada data berukuran kecil hingga menengah (Meka, 2024), namun komputasinya melambat seiring bertambahnya volume data (Shathya, 2025). Sementara SVM memiliki akurasi dan presisi tinggi untuk meredam *False Positive*, namun sangat sensitif terhadap pemilihan nilai parameter penalti kernel (Obiedat, 2020).
* **Pengembangan Fitur Baru:** Mengintegrasikan metrik perilaku konsumen digital yang lebih dinamis di masa mendatang, seperti indikator promosi musiman dan durasi interaksi halaman (Alamuri & Bondalapu, 2026).
* **Eksplorasi Algoritma Lanjutan:** Menguji performa pemodelan menggunakan kelompok *Ensemble Learning* tingkat tinggi seperti arsitektur LightGBM yang dipadukan dengan teknik kombinasi oversampling-undersampling guna mengolah data tabular berskala masif secara lebih optimal (Nugroho H dkk., 2025; Shathya, 2025).

---

## 📚 Referensi Ilmiah (APA Style 7th Edition)

* Alamuri, O. R. S. G., & Bondalapu, C. K. (2026). Predicting E-Commerce Purchase Intention Using Machine Learning. *ASEAN Journal of Scientific and Technological Reports*, 29(1), e260159. https://doi.org/10.55164/ajstr.v29i1.260159
* Meka, V. N. (2024). *Advanced Predictive Modelling of E-Commerce Customer Behaviour: Integrating Machine Learning and Deep Learning Techniques* (MSc Research Project, National College of Ireland). National College of Ireland Institutional Repository.
* Nugroho H, Y. D., Zakiyabarsi, F., & Paramita, A. J. (2025). Implementasi SMOTE-ENN dan Borderline SMOTE terhadap Performa LightGBM pada Imbalanced Class. *RABIT: Jurnal Teknologi dan Sistem Informasi Univrab*, 10(1), 51–59. https://doi.org/10.36341/rabit.v10i1.5432  
* Obiedat, R. (2020). A Comparative Study of Different Data Mining Algorithms with Different Oversampling Techniques in Predicting Online Shopper Behavior. *International Journal of Advanced Trends in Computer Science and Engineering (IJATCSE)*, 9(3), 3091–3097. https://doi.org/10.30534/ijatcse/2020/164932020  
* Shathya, B. (2025). Prediction of Online Shoppers Intention using Ensemble Learning Model after Treating Imbalance and Outlier Data. *International Journal of System Design and Information Processing (IJSDIP)*, 13(3), 32–37.

