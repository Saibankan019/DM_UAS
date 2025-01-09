# DM_UAS - Analisis Data dan Model Prediksi  
  
## 1. Judul / Topik Proyek  
**Judul:** Analisis Data Energi dan Model Prediksi Beban Energi  
  
**Identitas Lengkap:**  
- Nama: MUHAMMAD FADHLAN HAKIM  
- NIM: A11.2022.14619  
- Program Studi: TEKNIK INFORMATIKA  
- Universitas: UNIVERSITAS DIAN NUSWANTORO  
- Mata Kuliah: Penambangan Data / Data Mining  
  
---  
  
## 2. Ringkasan dan Permasalahan Proyek  
Proyek ini bertujuan untuk menganalisis dataset energi dan membangun model prediksi untuk mengklasifikasikan beban energi berdasarkan beberapa fitur.  
  
### Permasalahan:  
- Bagaimana cara menganalisis data untuk menemukan pola dalam beban energi?  
- Model apa yang paling efektif untuk prediksi beban energi?  
  
### Tujuan:  
- Menganalisis dataset untuk menemukan pola.  
- Membangun model prediksi yang akurat untuk klasifikasi beban energi.  
  
### Model / Alur Penyelesaian:  
[Mulai]
|
v
[Impor Libraries]
|
v
[Membaca Dataset]
|
v
[Pra-pemrosesan Data]
|
v
[Kategorisasi Target]
|
v
[Pembagian Data]
|
v
[Pelatihan Model]
|
v
[Evaluasi Model]
|
v
[Visualisasi Hasil]
|
v
[Selesai]

---  
  
## 3. Penjelasan Dataset, EDA, dan Proses Features Dataset  
Dataset yang digunakan dalam proyek ini adalah `merged_dataset.csv`, yang terdiri dari fitur-fitur berikut:  
- `temp`: Suhu  
- `humidity`: Kelembapan  
- `wind_speed`: Kecepatan angin  
- `price actual`: Harga aktual  
- `total load actual`: Beban energi aktual  
  
### Exploratory Data Analysis (EDA):  
- **Distribusi Fitur:**  
  - Distribusi suhu, kelembapan, kecepatan angin, dan harga aktual dianalisis menggunakan histogram.  
  python

import seaborn as sns
import matplotlib.pyplot as plt
Daftar variabel yang ingin dianalisis
variables = ['temp', 'humidity', 'wind_speed', 'price actual']
Membuat histogram untuk setiap variabel
for col in variables:
sns.histplot(data[col], kde=True)
plt.title(f'Distribution of {col}')
plt.xlabel(col)
plt.ylabel('Count')
plt.show()

  
- **Korelasi:**  
  - Matriks korelasi antara fitur-fitur untuk memahami hubungan antar fitur.  
  
python
import seaborn as sns
import matplotlib.pyplot as plt
Menghitung matriks korelasi
correlation_matrix = data.corr()
Membuat heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

---  
  
## 4. Proses Learning / Modeling  
Model yang digunakan dalam proyek ini adalah **Random Forest Classifier**. Proses pelatihan dilakukan dengan langkah-langkah berikut:  
  
1. **Pembagian Data:**  
   - Data dibagi menjadi data latih dan data uji dengan rasio 80:20.  
  
python
from sklearn.model_selection import train_test_split
Pembagian data
X = data[['temp', 'humidity', 'wind_speed', 'price actual']]
y = data['energy_load_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
2. **Pelatihan Model:**  
   - Model dilatih menggunakan data latih.  
  
python
from sklearn.ensemble import RandomForestClassifier
Inisialisasi dan pelatihan model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
  
---  
  
## 5. Evaluasi Model  
Model dievaluasi menggunakan metrik berikut:  
- **Akurasi, Precision, Recall, dan F1-Score:**  
  
python
from sklearn.metrics import classification_report, accuracy_score
Prediksi menggunakan data uji
y_pred = clf.predict(X_test)
Menampilkan laporan klasifikasi
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

### Visualisasi Hasil:  
- **Confusion Matrix:**

python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
Menghitung confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

- **ROC Curve:**  

python
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
Binarisasi target untuk ROC
y_test_bin = label_binarize(y_test, classes=clf.classes_)
y_score = clf.predict_proba(X_test)
Menghitung ROC curve
fpr = {}
tpr = {}
roc_auc = {}
for i in range(len(clf.classes_)):
fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
roc_auc[i] = auc(fpr[i], tpr[i])

Plot ROC curve
for i in range(len(clf.classes_)):
plt.plot(fpr[i], tpr[i], label=f"{clf.classes_[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="best")
plt.show()


  
---  
  
## 6. Diskusi Hasil dan Kesimpulan  
Hasil analisis menunjukkan bahwa model Random Forest dapat memberikan prediksi yang cukup baik, meskipun akurasinya masih bisa ditingkatkan.  
  
### Kesimpulan:  
- Proyek ini berhasil mencapai tujuan yang ditetapkan.  
- Rekomendasi untuk penelitian lebih lanjut termasuk eksplorasi model lain dan tuning hyperparameter.  
  
---  
  
## Catatan  
- Untuk menjalankan proyek ini, pastikan Anda memiliki semua dependensi yang diperlukan, termasuk `pandas`, `numpy`, `scikit-learn`, `matplotlib`, dan `seaborn`.  

