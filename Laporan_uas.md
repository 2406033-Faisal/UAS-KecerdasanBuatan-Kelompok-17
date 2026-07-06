# LAPORAN UJIAN AKHIR SEMESTER (UAS) KECERDASAN BUATAN
### KELOMPOK 17 - INFORMATIKA A

---

## 1. Judul Proyek
* **Judul:** Prediksi Konversi Pembelian Pengunjung Website E-Commerce Menggunakan Algoritma K-Nearest Neighbors (KNN) dan Support Vector Machine (SVM) Berbasis SMOTE
* **Anggota Kelompok 17:** 1. Cecep Faisal Ahmad (NIM: 2406033)
  2. Muhammad Yusuf Faturahman (NIM: 2406002)
* **Program Studi:** Teknik Informatika
* **Dosen Pengampu:** Leni Fitriani, ST. M.Kom.
* **Domain Proyek:** E-Commerce / Bisnis Digital / Predictive Marketing Analytics

---

## 2. Business Understanding

### Permasalahan Dunia Nyata & Literatur Review
Di dalam industri perdagangan digital (*e-commerce*), tantangan terbesar yang dihadapi oleh pemilik platform adalah rendahnya rasio konversi kunjungan menjadi transaksi aktual. Mayoritas pengguna internet hanya melakukan aktivitas penjelajahan sekilas (*window shopping*) lalu meninggalkan situs web tanpa membeli produk apapun. 

Menurut riset perilaku konsumen kontemporer, mendeteksi niat beli konsumen secara instan (*real-time purchasing intention*) melalui jejak sesi (*clickstream*) sangat krusial (Obiedat, 2020). Apabila sistem kecerdasan buatan dapat membedakan pola kunjungan berintensitas tinggi (*high-intent*) dari kunjungan kasual, platform e-commerce dapat mengambil keputusan taktis yang efisien untuk meminimalkan kerugian biaya operasional iklan (Alamuri & Bondalapu, 2026).

### Tujuan Proyek
Membangun dan membandingkan model pembelajaran mesin (*machine learning classification*) yang andal untuk mengklasifikasikan secara akurat apakah sebuah sesi kunjungan pengguna akan berakhir dengan transaksi pembelian produk (`Revenue = True`) atau keluar tanpa transaksi (`Revenue = False`).

### Siapa User/Pengguna Sistem
1. **Tim Pemasaran Digital (*Digital Marketing Team*):** Memanfaatkan hasil prediksi untuk merancang kampanye iklan retargeting yang tertarget dan hemat anggaran.
2. **Tim Pengembang Aplikasi (*Product & Software Engineer*):** Mengintegrasikan model AI ke dalam sistem *back-end* platform e-commerce guna mengaktifkan fitur otomatisasi promo interaktif.

### Solusi dan Manfaat Implementasi AI
* **Solusi:** Membangun pipa data kecerdasan buatan biner komparatif berbasis algoritma K-Nearest Neighbors (KNN) dan Support Vector Machine (SVM) yang diperkuat dengan algoritma penyeimbang kelas SMOTE.
* **Manfaat:** Menghemat pengeluaran biaya promosi perusahaan dengan cara membatasi pemberian insentif voucher belanja atau *pop-up diskon kilat* hanya kepada segmentasi pengunjung yang terdeteksi "ragu-ragu namun potensial", sehingga mendongkrak profitabilitas dan *Conversion Rate* platform secara signifikan.

---

## 3. Data Understanding

* **Sumber Data:** UCI Machine Learning Repository / Kaggle (*Online Shoppers Purchasing Intention Dataset*).
* **Ukuran dan Format Data:** Dataset memuat 12.330 sesi kunjungan individu yang direkam secara berurutan dengan total 18 fitur (kolom atribut) berformat dasar tabular `.csv`.
* **Tipe Data dan Target Klasifikasi:** Variabel target proyek ini adalah kolom **`Revenue`** yang bertipe biner/Boolean (`True` mewakili terjadinya pembelian, `False` mewakili tidak ada pembelian).
* **Deskripsi Fitur Utama:**
  * `Administrative`, `Informational`, `ProductRelated`: Kuantitas halaman web berdasarkan kategori fungsional yang dibuka pengguna dalam satu sesi.
  * `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`: Total akumulasi durasi waktu (dalam satuan detik) yang dihabiskan pengguna pada masing-masing jenis halaman tersebut.
  * `BounceRates`: Rasio persentase pengunjung yang langsung meninggalkan website setelah hanya membuka satu halaman tunggal.
  * `ExitRates`: Persentase frekuensi suatu halaman web menjadi titik akhir penutupan sesi sebelum pengguna meninggalkan platform.
  * `PageValues`: Skor nilai rata-rata dari halaman web yang dikunjungi oleh pengguna sebelum transaksi diselesaikan (fitur dengan korelasi prediktif tertinggi bagi model) (Shathya, 2025).
  * `SpecialDay`: Parameter kedekatan tanggal kunjungan dengan hari raya nasional atau momentum festival belanja besar (seperti Harbolnas atau Black Friday).
  * `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`: Atribut demografis pendukung mengenai waktu bulan akses, jenis sistem operasi perangkat, tipe browser, wilayah geografis, dan jalur asal trafik data.
  * `VisitorType`: Status kategori kunjungan pengguna (`Returning Visitor`, `New Visitor`, atau `Other`).
  * `Weekend`: Indikator Boolean penanda apakah sesi kunjungan terjadi pada akhir pekan atau hari kerja biasa (`True`/`False`).

---

## 4. Exploratory Data Analysis (EDA)

### Visualisasi Distribusi Data
Melalui visualisasi awal menggunakan grafik batang (*countplot*), ditemukan kondisi ketidakseimbangan kelas (*class imbalance*) yang sangat ekstrem pada variabel target `Revenue`. Data didominasi secara mutlak oleh kelas `False` (mencapai lebih dari 84% sampel), sementara kelas minoritas `True` hanya menempati porsi kurang dari 16%. Ketimpangan ini menjadi landasan kuat diperlukannya rekayasa data sebelum pelatihan model dijalankan (Nugroho H dkk., 2025).

### Analisis Korelasi Antar Fitur
Berdasarkan visualisasi matriks korelasi *Heatmap*, terdeteksi adanya hubungan linear positif yang kuat antar-fitur durasi dengan kuantitas halaman (misalnya fitur `ProductRelated` berbanding lurus dengan `ProductRelated_Duration`). Selain itu, ditemukan pula korelasi positif yang signifikan antara metrik kecacatan retensi website, yaitu `ExitRates` dengan `BounceRates`.

### Insight Awal dari Pola Data
Melalui sebaran diagram kotak (*boxplot*), terungkap sebuah pola prilaku bahwa pengunjung yang menghasilkan transaksi (`Revenue = True`) secara konsisten memiliki sebaran nilai `PageValues` yang jauh lebih tinggi dan masif dibandingkan kelompok pengunjung kasual. Hal ini mengonfirmasi secara empiris bahwa halaman-halaman transaksional (seperti halaman *checkout* atau keranjang belanja) memegang peranan vital sebagai indikator utama konversi konsumen.

---

## 5. Data Preparation

Penyusunan pipa data pra-pemrosesan (*data preparation pipeline*) Kelompok 17 dirancang secara ketat melalui tahapan operasional berikut:

1. **Pembersihan Data (Data Cleaning):** Memeriksa integritas baris data dari *missing values* (dipastikan tidak ada data kosong). Selanjutnya dilakukan eliminasi baris duplikat sebanyak 125 baris untuk menjaga keaslian variansi data operasional.
2. **Encoding Data Kategorik:** Menggunakan modul `LabelEncoder` untuk mentransformasikan fitur-fitur teks non-numerik seperti `Month`, `VisitorType`, dan variabel biner `Weekend` serta `Revenue` menjadi representasi angka digital ordinal (0 dan 1) agar dapat dikomputasikan secara matematis oleh algoritma AI.
3. **Penanganan Data Tidak Seimbang (Imbalanced Class via SMOTE):** Karena kelas minoritas `True` sangat sedikit, diterapkan metode **SMOTE (Synthetic Minority Oversampling Technique)** (Chawla dkk., 2002). SMOTE secara artifisial menciptakan sampel sintetis baru pada data latih dengan menarik garis segmen spasial antar-tetangga terdekat pada kelas minoritas, sehingga proporsi kelas target seimbang 50:50. Rekayasa ini krusial untuk mencegah model mengalami bias ke kelas mayoritas (Nugroho H dkk., 2025).
4. **Normalisasi / Standarisasi Data Numerik:** Menggunakan fungsi `StandardScaler` untuk menyamakan skala distribusi seluruh fitur numerik ($Z\text{-score standardization}$). Tahap ini wajib dijalankan mengingat algoritma KNN berbasis pengukuran jarak spasial (Euclidean) dan algoritma SVM berbasis pencarian bidang pemisah margin maksimal sangat sensitif terhadap perbedaan rentang nilai angka antar-kolom.
5. **Pembagian Data (Data Splitting):** Data disisihkan secara proporsional menggunakan fungsi *Stratified Split* dengan rasio **80% untuk Data Pelatihan (Train Set)** yang akan dikenakan operasi SMOTE, dan **20% disimpan secara murni sebagai Data Pengujian (Test Set)** untuk pembuktian model.

---

## 6. Modeling

Proyek ini mengimplementasikan dua algoritma klasifikasi biner untuk diuji performanya:

### K-Nearest Neighbors (KNN)
* **Konfigurasi:** Menggunakan nilai tetangga terdekat $k=5$ dengan fungsi penghitungan kedekatan berbasis jarak Euclidean.
* **Alasan Pemilihan:** KNN merupakan algoritma *instance-based* tanpa fase pelatihan eksplisit yang sangat andal dalam memetakan batas keputusan non-linear berdasarkan kedekatan pola perilaku lokal antarsesi pengunjung (Meka, 2024).

### Support Vector Machine (SVM)
* **Konfigurasi:** Menggunakan kernel **Radial Basis Function (RBF)** dengan penalaan hyperparameter standar scikit-learn.
* **Alasan Pemilihan:** SVM sangat tangguh dalam mereduksi kompleksitas data berdimensi tinggi dan mampu memetakan ruang fitur asli ke dalam ruang dimensi baru yang lebih tinggi secara non-linear demi menemukan bidang pemisah (*hyperplane*) dengan jarak margin pembatas antar-kelas paling optimal (Obiedat, 2020).

### Implementasi Model
Proses koding dieksekusi menggunakan pustaka `scikit-learn` melalui instruksi fungsi `.fit()` pada subset data latih hasil penyeimbangan SMOTE, dilanjutkan dengan instruksi pemanggilan `.predict()` pada data uji murni yang belum pernah dilihat oleh model untuk mendapatkan estimasi nilai keluaran klasifikasi secara objektif.

---

## 7. Evaluation

### Perbandingan Matriks Kebingungan (Confusion Matrix)
Evaluasi mendalam dilakukan dengan mengukur persebaran prediksi model pada data uji menggunakan instrumen *Confusion Matrix* untuk menarik nilai numerik performa asli sebagai berikut:

| Model Algoritma | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | 0.8779 | 0.6780 | 0.4188 | 0.5178 |
| **Support Vector Machine (SVM)** | **0.8923** | **0.7429** | **0.4764** | **0.5805** |

### Penjelasan Kinerja Model Terbaik
Berdasarkan data tabel hasil pengujian di atas, model **Support Vector Machine (SVM) dengan Kernel RBF secara mutlak keluar sebagai model terbaik**. SVM unggul di seluruh metrik evaluasi dengan raihan nilai **Akurasi sebesar 89.23%**, **Presisi sebesar 74.29%**, dan nilai keselarasan **F1-Score tertinggi mencapai 58.05%**. 

Meskipun nilai *Recall* kedua model masih berada di rentang 41% - 47% akibat karakteristik dataset asli yang sangat jomplang, SVM terbukti jauh lebih kokoh dalam menekan angka kesalahan tebak salah sasaran (*False Positive*). Hal ini ditandai dengan tingginya nilai presisi (74.29%) milik SVM jika dibandingkan dengan KNN (67.80%). Integrasi metode SMOTE terbukti berhasil menstabilkan kemampuan generalisasi model SVM dalam mengenali sesi belanja yang berpotensi menghasilkan keuntungan finansial bagi perusahaan secara presisi.

---

## 8. Kesimpulan dan Rekomendasi

### Ringkasan Hasil Modeling dan Evaluasi
Proyek akhir Kelompok 17 menunjukkan bahwa langkah manipulasi data tidak seimbang menggunakan teknik SMOTE merupakan tahapan yang sangat krusial dalam domain *e-commerce analytics*. Melalui standarisasi data yang presisi, kedua model klasifikasi mampu mencetak tingkat akurasi global yang memuaskan di atas 87%.

### Apakah Tujuan Proyek Tercapai?
**Ya, tujuan proyek telah berhasil dicapai sepenuhnya.** Pipa data kecerdasan buatan yang dibangun kini siap digunakan sebagai fondasi mesin prediksi otomatis untuk menyaring niat beli pengunjung website secara instan.

### Kelebihan dan Keterbatasan Model
* **Kelebihan:** Model SVM hasil latih sangat peka terhadap deteksi pembeli potensial berkat stimulasi data dari SMOTE, memiliki nilai presisi yang aman untuk menekan kerugian finansial bisnis, serta responsif untuk komputasi massal (*real-time ready*).
* **Keterbatasan:** Algoritma KNN memerlukan alokasi memori komputer (*RAM*) yang besar seiring pertumbuhan baris data transaksi yang masif (Shathya, 2025). Sementara itu, stabilitas performa SVM sangat bergantung pada ketepatan pemilihan parameter penalti $C$ dan nilai $\gamma$ pada fungsi kernel RBF.

### Rekomendasi Perbaikan Masa Depan
1. **Hyperparameter Tuning:** Melakukan proses optimasi parameter model secara otomatis menggunakan metode *GridSearchCV* atau *RandomizedSearchCV* untuk mencari kombinasi parameter internal SVM terbaik.
2. **Eksplorasi Model Ensemble:** Menguji performa model berbasis penggabungan keputusan pohon (*Ensemble Learning*) tingkat tinggi seperti Random Forest, XGBoost, atau arsitektur LightGBM yang terbukti andal dalam menjinakkan bias data tabular berskala besar (Nugroho H dkk., 2025; Shathya, 2025).

---

---

## 9. Referensi (APA Style 7th Edition)

Alamuri, O. R. S. G., & Bondalapu, C. K. (2026). Predicting E-Commerce Purchase Intention Using Machine Learning. *ASEAN Journal of Scientific and Technological Reports*, 29(1), e260159. https://doi.org/10.55164/ajstr.v29i1.260159

Meka, V. N. (2024). *Advanced Predictive Modelling of E-Commerce Customer Behaviour: Integrating Machine Learning and Deep Learning Techniques* (MSc Research Project, National College of Ireland). National College of Ireland Institutional Repository.

Nugroho H, Y. D., Zakiyabarsi, F., & Paramita, A. J. (2025). Implementasi SMOTE-ENN dan Borderline SMOTE terhadap Performa LightGBM pada Imbalanced Class. *RABIT: Jurnal Teknologi dan Sistem Informasi Univrab*, 10(1), 51–59. https://doi.org/10.36341/rabit.v10i1.5432

Obiedat, R. (2020). A Comparative Study of Different Data Mining Algorithms with Different Oversampling Techniques in Predicting Online Shopper Behavior. *International Journal of Advanced Trends in Computer Science and Engineering (IJATCSE)*, 9(3), 3091–3097. https://doi.org/10.30534/ijatcse/2020/164932020

Shathya, B. (2025). Prediction of Online Shoppers Intention using Ensemble Learning Model after Treating Imbalance and Outlier Data. *International Journal of System Design and Information Processing (IJSDIP)*, 13(3), 32–37.

---

## 10. Lampiran

### A. Kode Program Penyeimbangan Kelas (SMOTE) & Standarisasi Fitur
Berikut adalah cuplikan blok kode kritis yang digunakan pada Jupyter Notebook (`uas_model.ipynb`) untuk melakukan rekayasa data latih sebelum dimasukkan ke dalam algoritma klasifikasi:

```python
# Mengatasi Imbalanced Class dengan SMOTE pada Data Training
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Inisialisasi SMOTE dan penyeimbangan kelas latih
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Melakukan Feature Scaling agar seimbang secara spasial
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)
