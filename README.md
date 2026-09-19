# Diabetes Risk Classification

Project machine learning untuk memprediksi **risiko diabetes** (rendah/tinggi) pada pasien berdasarkan data klinis, dibangun menggunakan **Microsoft Fabric** (notebook, Data Wrangler, dan experiment tracking) dengan Python (pandas, scikit-learn).

## Tujuan

Membandingkan dua pendekatan machine learning untuk memprediksi kondisi diabetes:
1. **Model regresi** : memprediksi skor progres penyakit (`Y`) secara langsung.
2. **Model klasifikasi** : memprediksi kategori risiko (`Risk`: 0 = rendah, 1 = tinggi), hasil turunan dari `Y`.

Model klasifikasi dipilih sebagai model final karena hasilnya lebih akurat dan lebih mudah ditafsirkan untuk pengambilan keputusan praktis.

## Dataset

- **Sumber**: [Diabetes dataset — Azure Open Datasets](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-diabetes)
- **Jumlah data**: 442 pasien
- **Fitur (10)**: `AGE`, `SEX`, `BMI`, `BP` (tekanan darah), `S1`–`S6` (hasil tes darah: kolesterol, LDL, HDL, dll.)
- **Target asli**: `Y` — skor kuantitatif progres penyakit diabetes setahun setelah pengukuran awal

## Metodologi

1. **Eksplorasi data (EDA)** — statistik deskriptif, cek missing values, distribusi tiap kolom.
2. **Feature engineering** — membuat kolom biner `Risk` dari `Y`, menggunakan threshold persentil ke-75 (`Y > 211.5` → risiko tinggi).
3. **Training model** — dua eksperimen dijalankan dan dilacak via Fabric Experiment Tracking:
   - `diabetes-regression` — regresi terhadap `Y`
   - `diabetes-classification` — klasifikasi terhadap `Risk` (Logistic Regression)
4. **Evaluasi & perbandingan** model berdasarkan metrik masing-masing.

## Hasil

| Metric       | Regresi   | Klasifikasi |

| R² Score     | 0.554     |   –         |
| RMSE         | 52.95     |   –         |
| MAE          | 43.05     |   –         |
| Accuracy     | –         |   0.877     |
| Precision    | –         |   0.874     |
| Recall       | –         |   0.877     |
| F1-Score     | –         |   0.874     |
| ROC AUC      | –         |   0.920     |

**Model klasifikasi (Logistic Regression)** dipilih sebagai model final performa jauh lebih baik (ROC AUC 0.92) dan hasil prediksinya (risiko rendah/tinggi) lebih actionable dibanding angka skor mentah.

## Keterbatasan

- Ukuran dataset kecil (442 baris), generalisasi model masih terbatas.
- Threshold `Risk` (211.5) ditentukan secara statistik (persentil ke-75), bukan standar medis resmi.
- Metrik dilaporkan dari data training, belum divalidasi dengan data uji terpisah (test set) atau cross-validation.
- Model masih menggunakan parameter default, belum dilakukan hyperparameter tuning.

## Langkah Selanjutnya

- Melakukan train-test split / cross-validation untuk validasi yang lebih andal.
- Mencoba algoritma lain (Random Forest, XGBoost) untuk perbandingan performa.
- Melakukan hyperparameter tuning.

## Tools

- Microsoft Fabric (Notebook, Data Wrangler, Experiment Tracking, MLflow)
- Python: pandas, scikit-learn
- PySpark (untuk load data dari Lakehouse)

## Lisensi
Project ini menggunakan lisensi MIT — lihat file `LICENSE` untuk detail.
