# 🏭 Sistem Klasifikasi IKM Kota Makassar - Deep Neural Network

> **Pemodelan Deep Neural Network dalam Klasifikasi Jenis Industri IKM Kota Makassar**

Sistem klasifikasi dan rekomendasi berbasis DNN untuk mengklasifikasikan IKM ke dalam 10 sektor industri secara otomatis dengan interpretabilitas lengkap.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Sistem (Otomatis)
```bash
chmod +x run_classification.sh
./run_classification.sh
```

### 3. Atau Manual
```bash
python ikm_dnn_classification.py
```

---

## 📊 10 Sektor Industri

1. **Makanan Olahan** - Produk makanan, camilan, frozen food
2. **Roti & Kue** - Bakery, pastry, kue tradisional
3. **Minuman** - Kopi, teh, jus, minuman kemasan
4. **Tekstil & Pakaian** - Kain, batik, tenun
5. **Konveksi & Penjahitan** - Pakaian jadi, seragam
6. **Reparasi Kendaraan** - Bengkel motor/mobil
7. **Percetakan & Media** - Printing, sablon
8. **Kerajinan & Furnitur** - Mebel, kerajinan kayu
9. **Logam & Bengkel Umum** - Las, konstruksi besi
10. **Industri Rumah Tangga Lainnya** - Kategori lain

---

## ✨ Fitur Utama

- ✅ **Rule-Based Clustering Otomatis** - Mapping 10 sektor berbasis kata kunci
- ✅ **DNN Eksplisit** - Arsitektur 4 layers (128-64-32-10 neurons)
- ✅ **Rekomendasi Top-3** - Berbasis probabilitas softmax
- ✅ **Evaluasi Komprehensif** - Accuracy, F1-Score, MSE, Confusion Matrix
- ✅ **SHAP Interpretability** - Feature importance & explanations
- ✅ **Visualisasi Lengkap** - Training curves, heatmaps, activation functions

---

## 🎯 Arsitektur DNN

```
Input (4 features) 
    ↓
Hidden Layer 1 (128 neurons, ReLU, BatchNorm, Dropout 0.3)
    ↓
Hidden Layer 2 (64 neurons, ReLU, BatchNorm, Dropout 0.3)
    ↓
Hidden Layer 3 (32 neurons, ReLU, Dropout 0.2)
    ↓
Output (10 neurons, Softmax)
```

**Fitur Input:**
- Kelurahan (encoded dari alamat)
- Tenaga Kerja (numerik)
- Badan Usaha (encoded)
- Nilai Investasi (log-scaled)

---

## 📈 Output Yang Dihasilkan

### Model Files
- `models/ikm_classifier_final.keras` - Model terlatih
- `models/encoders.pkl` - Encoders untuk preprocessing

### Visualizations (300 DPI)
- `confusion_matrix.png` - Heatmap multi-class
- `training_history.png` - Loss & accuracy curves
- `shap_feature_importance.png` - Global feature ranking
- `shap_summary_plot.png` - Feature impact distribution
- `activation_functions.png` - ReLU, Sigmoid, Softmax
- `dnn_architecture.png` - Network diagram

### Data
- `outputs/data_processed.csv` - Data setelah preprocessing

---

## 🔮 Prediksi Data Baru

```python
from ikm_dnn_classification import predict_new_data

recommendations = predict_new_data(
    kelurahan="Tamalanrea",
    tenaga_kerja=5,
    badan_usaha="CV",
    nilai_investasi=50000000
)

# Output: Top-3 sektor dengan probabilitas
# [(sektor_1, 0.68), (sektor_2, 0.18), (sektor_3, 0.08)]
```

### Demo Script
```bash
python demo_prediksi.py
```

---

## 📋 Requirements

### Python Libraries
- TensorFlow 2.15.0
- scikit-learn 1.3.2
- pandas 2.1.4
- numpy 1.26.2
- SHAP 0.44.0
- matplotlib 3.8.2
- seaborn 0.13.0

### System
- RAM: 4GB minimum (8GB recommended)
- Python: 3.8+
- Disk: ~500MB

---

## 📊 Evaluasi Model

**Metrik Utama:**
- **Accuracy** - Overall correctness
- **Macro F1-Score** - Balance precision & recall (metrik utama)
- **MSE** - Magnitude of prediction errors

**Analisis:**
- Confusion Matrix heatmap
- Classification report per-class
- Training history visualization
- SHAP feature importance

---

## 🔬 Metodologi

1. **Data Loading** - Read Excel data
2. **Rule-Based Clustering** - Map ke 10 sektor (keyword-based)
3. **Feature Engineering** - Extract kelurahan, encode, normalize
4. **Model Building** - DNN dengan 3 hidden layers
5. **Training** - Adam optimizer, early stopping, learning rate reduction
6. **Evaluation** - Comprehensive metrics & visualizations
7. **Interpretability** - SHAP values untuk explainability
8. **Recommendation** - Top-3 sektor berbasis probabilitas

---

## 📁 Struktur Proyek

```
DNN/
├── ikm_dnn_classification.py    # Script utama
├── demo_prediksi.py              # Demo prediksi baru
├── run_classification.sh         # Auto runner
├── requirements.txt              # Dependencies
├── README.md                     # Quick start (file ini)
├── README_LENGKAP.md            # Dokumentasi lengkap
├── Data Penelitian IKM Makassar Fix.xlsx
├── models/                       # Model tersimpan
└── outputs/                      # Hasil & visualisasi
```

---

## 🎓 Penjelasan Konseptual

### Feedforward Neural Network
```
Input → Hidden Layers → Output
```
- Data masuk melalui input layer
- Hidden layers ekstraksi fitur kompleks
- Output layer prediksi dengan softmax

### Backpropagation
- Hitung error di output
- Propagate gradient ke belakang
- Update weights dengan gradient descent
- Minimize loss function

### Regularization (Anti-Overfitting)
- **Dropout**: Random deactivate neurons
- **Batch Normalization**: Stabilize learning
- **Early Stopping**: Stop saat validation loss naik

---

## ⚠️ Troubleshooting

**Error: TensorFlow tidak ditemukan**
```bash
pip install tensorflow==2.15.0
```

**Error: SHAP warning**
```bash
pip install shap
```

**Error: Memory error**
- Reduce batch_size dari 32 ke 16
- Gunakan fewer SHAP samples

---

## 📖 Dokumentasi Lengkap

Untuk penjelasan detail, lihat: **[README_LENGKAP.md](README_LENGKAP.md)**

Mencakup:
- Penjelasan detail setiap fungsi
- Interpretasi hasil lengkap
- Referensi akademik
- Advanced troubleshooting
- Hyperparameter tuning guide

---

## 🎯 Hasil yang Diharapkan

- **Accuracy**: 75-90% (tergantung kualitas data)
- **Macro F1-Score**: 0.70-0.85
- **Training Time**: 10-20 menit
- **SHAP Analysis**: 5-10 menit

---

## 👥 Author

**Research Team**  
Pemodelan Deep Neural Network dalam Klasifikasi Jenis Industri IKM Kota Makassar  
December 2025

---

## 📄 License

Untuk keperluan penelitian akademik.

---

**🚀 Selamat Menggunakan Sistem Klasifikasi IKM Makassar!**