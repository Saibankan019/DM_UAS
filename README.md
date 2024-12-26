NAMA : MUHAMMAD FADHLAN HAKIM  ||  NIM : A11.2022.14619

# 🌟 **Energy Load Prediction Using Machine Learning** 🌟

> **Aplikasi pembelajaran mesin untuk memprediksi kategori beban energi berdasarkan parameter lingkungan.**

---

## 📜 **Deskripsi Proyek**
Proyek ini bertujuan untuk memanfaatkan algoritme pembelajaran mesin dalam mengklasifikasikan beban energi (`energy_load`) menjadi kategori **Low**, **Medium**, dan **High**. Dataset yang digunakan mengandung informasi seperti suhu, kelembaban, kecepatan angin, dan harga energi aktual.

---

## 🛠️ **Fitur Utama**
1. **Pra-pemrosesan Data**:
   - Pemilihan atribut penting.
   - Normalisasi menggunakan `StandardScaler`.
   - Kategorisasi beban energi menggunakan teknik binning.

2. **Pelatihan Model**:
   - Menggunakan algoritme **Random Forest Classifier** untuk pelatihan dan prediksi.

3. **Evaluasi Model**:
   - Laporan klasifikasi, akurasi, dan matriks kebingungan.
   - Kurva ROC untuk klasifikasi multiclass.

4. **Visualisasi Data**:
   - Korelasi antar variabel.
   - Pentingnya fitur dalam model.
   - Distribusi kategori dan atribut penting.

5. **Model Deployment**:
   - Model disimpan dalam file `.pkl` menggunakan `joblib` untuk digunakan lebih lanjut.

---

## 📁 **Struktur Proyek**
UAS_Energy_Load_Prediction/ ├── kumpulan-data/ │ └── merged_dataset.csv # Dataset utama ├── UAS_14619_Final.py # Skrip utama ├── random_forest_energy_model.pkl # Model terlatih └── README.md # Dokumentasi proyek

---

## 📊 **Hasil Visualisasi**
### 🔹 Matriks Korelasi
![Correlation Matrix](#)
Korelasi antar atribut ditampilkan dalam heatmap untuk mengidentifikasi hubungan antar variabel.

### 🔹 Pentingnya Fitur
![Feature Importance](#)
Grafik bar untuk menggambarkan kontribusi masing-masing fitur dalam model.

### 🔹 Kurva ROC
![ROC Curve](#)
Kurva ROC untuk setiap kategori beban energi dengan nilai AUC yang disertakan.

---

## 🚀 **Cara Menjalankan Proyek**
1. **Persyaratan Lingkungan**:
   Pastikan Python dan pustaka berikut terinstal:
   - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`.

   Instal dengan perintah:
   ```bash
   pip install -r requirements.txt

