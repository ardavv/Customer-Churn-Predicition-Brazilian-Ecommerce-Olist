# 🛒 Olist E-Commerce: Customer Churn vs. Propensity to Repeat Purchase
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1P-7nMDfGk2LDeovcO7_sQjGsMtamnlLh?usp=sharing)

## 📌 Project Overview
Proyek ini mengimplementasikan *pipeline Big Data* dan *Machine Learning end-to-end* untuk mendeteksi serta menganalisis pola retensi pelanggan pada platform Olist Brazilian E-Commerce. Tantangan terbesar dalam dataset ini adalah sifat model bisnisnya: Olist beroperasi sebagai integrator toko perantara (*storefront*) di dalam *marketplace* raksasa, sehingga ketiadaan loyalitas merek (*brand loyalty*) dari konsumen menyebabkan 98,8% populasinya merupakan *one-time buyers*. 

Melalui proyek ini, fokus dialihkan dari sekadar mengejar akurasi prediksi mentah menjadi analisis diagnostik kausalitas, serta memberikan rekomendasi perubahan paradigma bisnis dari klasifikasi **"Customer Churn"** menjadi **"Propensity to Repeat Purchase"**.

## 📊 Dataset Source
Proyek ini menggunakan data resmi dari Olist yang tersedia di Kaggle:
* **Dataset Link:** [Kaggle - Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/)

## 🏗️ Data Architecture (Medallion Architecture)
Proyek ini mengadopsi kerangka kerja **Medallion Architecture** menggunakan PySpark untuk menjamin skalabilitas pemrosesan data, dengan penerapan mitigasi *Data Leakage* yang ketat:

* **Bronze Layer (Ingestion):** Memuat seluruh data mentah dari tabel eksternal e-commerce Olist dengan skema terstruktur.
* **Silver Layer (Cleansing & Transformation):** Melakukan *text cleansing* (standarisasi aksen dengan regex), penyaringan status pesanan valid (`delivered`), dan perhitungan selisih waktu. **Catatan:** Tidak ada imputasi *missing values* di lapisan ini. Nilai kosong (Null) dibiarkan mengalir untuk diimputasi secara terisolasi pada fase Machine Learning guna mencegah kebocoran informasi masa depan (*data leakage*).
* **Gold Layer (Customer Feature Mart):** Agregasi masif pada tingkat pelanggan (*customer level*) untuk membangun *Data Mart* komprehensif yang memuat fitur Transaksional, Geografis, Logistik, dan Sentimen Rating.

## 🧪 Validation Strategy & Time-Window Split
Untuk mencegah terjadinya kebocoran data, pemrosesan fitur historis dipisahkan secara kaku dari penentuan label target menggunakan metode *Time-Window Split*:

* **Justifikasi Threshold (180 Hari):** Berdasarkan analisis empiris fungsi distribusi kumulatif (CDF) pada *Inter-Purchase Interval* dari pelanggan loyal, persentil ke-80 berada di angka 165 hari. Batas *Churn* ditetapkan pada 180 hari sebagai zona penyangga konservatif.
* **Observation Window (4 Sep 2016 - 28 Feb 2018):** Garis waktu ekstraksi fitur dikunci pada 1 Maret 2018 (*cut-off date*). Seluruh transaksi pada jendela ini digunakan untuk menyusun matriks fitur.
* **Prediction Window (1 Mar 2018 - 17 Okt 2018):** Berlangsung selama 230 hari murni untuk mengamati kemunculan transaksi baru sebagai penentu target label (`is_churn` = 1 jika tidak ada transaksi, 0 jika sebaliknya).

## ⚖️ Strict Anti-Leakage Pipeline & Imbalance Handling
Eksplorasi data akhir pada Data Mart menunjukkan tingkat ketidakseimbangan kelas yang ekstrem (98,84% Churn vs 1,16% Active). Untuk menangani hal ini secara objektif:

Seluruh tahapan preprocessing dienkapsulasi ke dalam **Imblearn Pipeline** yang dieksekusi secara terisolasi di dalam *5-Fold Stratified Cross-Validation*. Pipeline tersebut mencakup:
1. `IQROutlierCapper`: Membatasi nilai ekstrem (capping) tanpa menghapus baris.
2. `SimpleImputer`: Mengisi *missing values* menggunakan median (dihitung murni dari data *training fold*).
3. `StandardScaler`: Menyamakan skala fitur numerik.
4. `RandomUnderSampler (RUS)`: Memangkas kelas mayoritas secara acak hingga seimbang dengan kelas minoritas untuk mencegah algoritma menjadi *majority class classifier*.

## 📈 Model Performance Matrix
Model dievaluasi menggunakan metrik yang kebal terhadap ketidakseimbangan kelas (*Macro Average* & ROC-AUC). Penanganan RUS menyebabkan ukuran data pelatihan menyusut drastis, sehingga menguji kemampuan generalisasi asli dari model:

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **57.16%** | **50.29%** | **56.27%** | **37.71%** | **60.48%** |
| **Random Forest** | 56.35% | 50.20% | 54.23% | 37.27% | 56.77% |
| **XGBoost** | 54.39% | 50.08% | 51.83% | 36.34% | 55.68% |

**Insight:** *Logistic Regression* (model linier ber-varians rendah) memenangkan evaluasi ini. Pada dataset yang mengalami *data starvation* akibat undersampling dan memiliki ruang fitur yang saling tumpang tindih (*overlapping*), model *ensemble* kompleks seperti Random Forest dan XGBoost cenderung mengalami *overfitting* pada *noise* pelatihan, membuktikan ketiadaan sinyal batas keputusan yang jelas.

## 💡 Business & Interpretability Insights (SHAP Analysis)
Melalui metode *SHapley Additive exPlanations* (SHAP) pada XGBoost, penelitian ini membongkar kekuatan prediktif fitur:

1. **Dominasi Noise pada Sinyal Prediktif:** Meskipun *Global Feature Importance* menempatkan fitur `geolocation_lat`, durasi persetujuan pesanan (`purchased_approved`), dan `freight_ratio` di urutan teratas, visualisasi *SHAP Beeswarm Plot* memperlihatkan sebaran nilai yang sempit dan saling bertumpuk acak (rentang absolut -1.0 hingga 1.0). 
2. **Ketiadaan Kausalitas Statis:** Fitur dengan nilai ekstrem tinggi maupun rendah tidak secara konsisten mengarahkan probabilitas model pada kelas tertentu. Hal ini membuktikan bahwa atribut operasional statis (logistik & geografis) tidak memiliki kekuatan prediktif (*predictive power*) yang mumpuni untuk membedakan *one-time buyers* dan *repeat buyers*.
3. **Rekomendasi Strategis:** Mengingat model bisnis Olist, manajemen data sebaiknya menghentikan permodelan klasifikasi biner *Customer Churn* standar. Perusahaan direkomendasikan untuk menggunakan pemodelan **Buy-Till-You-Die (BTYD)** seperti BG/NBD atau **Survival Analysis** untuk memprediksi probabilitas kelompok minoritas bertransaksi kembali berbasis waktu.

## 🛠️ Requirements & Installation
Untuk mengeksekusi pipeline ini, pastikan *environment* Anda memiliki pustaka berikut:
```bash
pip install pyspark scikit-learn xgboost imbalanced-learn shap seaborn matplotlib folium
