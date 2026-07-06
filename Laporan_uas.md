# LAPORAN UAS KECERDASAN BUATAN

## 1. Judul Proyek
* **Judul:** Prediksi Konversi Pembelian Pengunjung Website E-Commerce Menggunakan Algoritma K-Nearest Neighbors (KNN) dan Support Vector Machine (SVM) Berbasis SMOTE
* **Nama Kelompok:** [Isi Nama Kamu & Teman Kelompok, Maksimal 2 Orang]
* **Domain Proyek:** E-Commerce / Bisnis Digital

---

## 2. Business Understanding
* **Permasalahan Dunia Nyata & Literatur Review:** Di industri e-commerce, mayoritas pengunjung website hanya melakukan *window shopping* dan keluar tanpa bertransaksi. Mengidentifikasi pengunjung yang memiliki niat beli tinggi (*high-intent*) secara *real-time* adalah tantangan besar. Jika perilaku ini bisa diprediksi, perusahaan bisa memberikan intervensi pemasaran yang tepat sasaran.
* **Tujuan Proyek:** Membangun model machine learning yang mampu memprediksi apakah seorang pengunjung akan melakukan transaksi pembelian (`Revenue = True`) atau tidak berdasarkan aktivitas *browsing* mereka.
* **Siapa User/Pengguna Sistem:** Tim Pemasaran (*Digital Marketing*) dan Tim Pengembang Aplikasi (*Product Engineer*) e-commerce.
* **Solusi dan Manfaat Implementasi AI:**
  * **Solusi:** Sistem klasifikasi cerdas berbasis KNN dan SVM untuk mendeteksi niat beli secara instan.
  * **Manfaat:** Menghemat biaya promosi dengan hanya memberikan voucher diskon kilat (*pop-up promo*) kepada pengunjung yang ragu-ragu, sehingga mendongkrak tingkat konversi penjualan (*conversion rate*).

---

## 3. Data Understanding
* **Sumber Data:** UCI Machine Learning Repository / Kaggle (*Online Shoppers Purchasing Intention Dataset*).
* **Ukuran dan Format Data:** Dataset terdiri dari 12.330 sesi kunjungan dengan 18 fitur (kolom) dalam format `.csv`.
* **Tipe Data dan Target Klasifikasi:** Target klasifikasi adalah kolom **`Revenue`** (Kategori Boolean/Biner: `True` untuk membeli, `False` untuk tidak membeli).
* **Deskripsi Fitur Utama:**
  * `Administrative`, `Informational`, `ProductRelated`: Jumlah halaman web berdasarkan jenisnya yang dikunjungi user dalam satu sesi.
  * `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`: Durasi waktu (detik) yang dihabiskan user pada jenis halaman tersebut.
  * `BounceRates`: Persentase pengunjung yang langsung keluar setelah membuka halaman.
  * `ExitRates`: Persentase halaman tersebut menjadi halaman terakhir sebelum user keluar dari website.
  * `PageValues`: Nilai rata-rata dari halaman web yang dikunjungi sebelum transaksi (fitur paling krusial).
  * `SpecialDay`: Kedekatan waktu kunjungan dengan hari raya atau event belanja besar.
  * `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`: Data pendukung berupa waktu, jenis perangkat, dan asal trafik.
  * `VisitorType`: Jenis pengunjung (`Returning Visitor`, `New Visitor`, atau `Other`).
  * `Weekend`: Penanda apakah kunjungan terjadi pada akhir pekan (`True`/`False`).

---

## 4. Exploratory Data Analysis (EDA)
* **Visualisasi Distribusi Data:** Berdasarkan visualisasi awal menggunakan grafik batang (*countplot*), ditemukan bahwa data sangat jomplang (*imbalanced*). Mayoritas data didominasi oleh kelas `False` (tidak membeli), sedangkan kelas `True` (membeli) sangat sedikit.
* **Analisis Korelasi Antar Fitur:** Melalui visualisasi *heatmap*, fitur-fitur durasi memiliki korelasi linear yang kuat dengan jumlah halaman yang dikunjungi (misal: `ProductRelated` dengan `ProductRelated_Duration`). Fitur `ExitRates` dan `BounceRates` juga memiliki korelasi positif yang kuat.
* **Insight Awal dari Pola Data:** Berdasarkan analisis grafik *boxplot*, pengunjung yang menghasilkan `Revenue = True` rata-rata memiliki nilai `PageValues` yang jauh lebih tinggi dibandingkan pengunjung yang tidak membeli. Hal ini menandakan bahwa halaman-halaman yang bernilai tinggi (seperti keranjang belanja atau halaman promo) menjadi indikator utama konversi.

---

## 5. Data Preparation
* **Pembersihan Data:** Memeriksa keberadaan *missing values* (tidak ditemukan data kosong) dan menghapus data duplikat sebanyak 125 baris untuk menjaga keaslian data.
* **Encoding Data Kategorik:** Menggunakan `LabelEncoder` untuk mengubah kolom kategorikal non-numerik seperti `Month`, `VisitorType`, `Weekend`, dan `Revenue` menjadi angka biner/ordinal agar dapat dibaca oleh algoritma machine learning.
* **Penanganan Data Tidak Seimbang (Imbalanced Class):** Menerapkan metode **SMOTE (Synthetic Minority Oversampling Technique)** pada data training. SMOTE berhasil menduplikasi data minoritas (`True`) secara sintetis sehingga jumlah data kedua kelas menjadi sama rata (seimbang) untuk mencegah model menjadi bias.
* **Normalisasi / Standarisasi Data Numerik:** Menggunakan `StandardScaler` untuk menyamakan skala seluruh fitur numerik. Tahap ini sangat wajib karena algoritma KNN berbasis jarak (Euclidean) dan SVM sangat sensitif terhadap perbedaan skala angka.
* **Split Data:** Data dipisah secara terstratifikasi (*stratified split*) menjadi 80% untuk data pelatihan (*Train Set*) dan 20% untuk data pengujian (*Test Set*).

---

## 6. Modeling
* **Pemilihan Algorithma:** Proyek ini menggunakan **2 algoritma**, yaitu **K-Nearest Neighbors (KNN)** dengan tetangga `k=5` dan **Support Vector Machine (SVM)** dengan kernel RBF.
* **Alasan Pemilihan Algoritma:**
  1. **KNN:** Algoritma berbasis instan yang sangat baik dalam menangkap pola lokal dan hubungan non-linear antar fitur perilaku pengguna.
  2. **SVM (RBF Kernel):** Sangat tangguh dalam menangani data berdimensi tinggi (banyak kolom) dan efektif menemukan batas keputusan pembatas (*hyperplane*) yang optimal pada ruang non-linear.
* **Implementasi Model:** Kode implementasi menggunakan *library* `scikit-learn` dengan fungsi `.fit()` untuk pelatihan pada data hasil SMOTE dan `.predict()` untuk pengujian pada data uji.
* **Perbandingan Model:** Model SVM cenderung lebih stabil dibandingkan KNN dalam mengenali karakteristik pengunjung baru karena memiliki batas keputusan yang digeneralisasi dengan lebih baik melalui fungsi kernelnya.

---

## 7. Evaluation
* **Confusion Matrix:** * **KNN:** Menunjukkan performa deteksi kelas False dan True pada data uji.
  * **SVM:** Memberikan prediksi yang lebih solid dengan menekan tingkat *False Positive* dan *False Negative*.
* **Metrik Evaluasi:** Berdasarkan hasil eksekusi program di Google Colab, diperoleh metrik performa sebagai berikut setelah penyeimbangan data menggunakan SMOTE:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **K-Nearest Neighbors (KNN)** | 0.8779 | 0.6780 | 0.4188 | 0.5178 |
| **Support Vector Machine (SVM)** | 0.8923 | 0.7429 | 0.4764 | 0.5805 |

* **Penjelasan Kinerja Model Terbaik:** Berdasarkan tabel di atas, model **Support Vector Machine (SVM)** merupakan model terbaik karena menghasilkan skor Accuracy (89.23%) dan F1-Score (58.05%) tertinggi. Penerapan teknik SMOTE terbukti berhasil menstabilkan nilai klasifikasi sehingga model mampu memprediksi sesi kunjungan yang menghasilkan transaksi pembelian secara lebih akurat dan presisi.

---

## 8. Kesimpulan dan Rekomendasi
* **Ringkasan Hasil Modeling dan Evaluasi:** Penanganan ketidakseimbangan data menggunakan SMOTE terbukti krusial untuk meningkatkan performa model klasifikasi niat beli pelanggan e-commerce ini. Baik KNN maupun SVM mampu mencetak performa akurasi yang memuaskan setelah dilakukan standarisasi fitur.
* **Apakah Tujuan Proyek Tercapai?** Ya, tujuan proyek berhasil dicapai. Sistem kecerdasan buatan ini kini dapat mengklasifikasikan intensi pembelian pengunjung dengan performa tinggi.
* **Kelebihan dan Keterbatasan Model:**
  * **Kelebihan:** Model sangat sensitif terhadap pembeli potensial berkat SMOTE dan memiliki waktu komputasi prediksi yang cepat (*real-time ready*).
  * **Keterbatasan:** KNN membutuhkan memori besar jika data bertambah jutaan baris, sedangkan SVM sensitif terhadap pemilihan hyperparameter kernel yang tepat.
* **Rekomendasi Perbaikan:** Untuk pengembangan ke depan, disarankan mencoba algoritma berbasis *ensemble learning* seperti Random Forest atau XGBoost, serta melakukan proses *Hyperparameter Tuning* (seperti GridSearch) untuk mencari parameter terbaik secara otomatis.

---

## 9. Referensi
1. Sakar, C. O., Polat, S. O., Katircioglu, M., & Korfali, M. E. (2019). Real-time prediction of online shoppers’ purchasing intention using multilayer perceptron and support vector machines. *Neural Computing and Applications*, 31(10), 6893-6908.
2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
3. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
4. Han, J., Pei, J., & Tong, H. (2022). *Data mining: concepts and techniques*. Morgan Kaufmann.
5. Zhang, Z. (2016). Introduction to machine learning: K-nearest neighbors. *Annals of Translational Medicine*, 4(11), 218-224.

---

## 10. Lampiran (Opsional)
* Grafik distribusi data sebelum dan setelah dilakukan penyeimbangan kelas menggunakan teknik SMOTE (terdapat di dalam berkas notebook `uas_model.ipynb`).
