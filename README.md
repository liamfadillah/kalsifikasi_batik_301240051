# 🪡 BatikScan — Klasifikasi Motif Batik Indonesia

Aplikasi web berbasis **Flask** yang menggunakan algoritma **Random Forest** 
untuk mengklasifikasikan motif batik Indonesia dari gambar.

---

## 📁 Struktur Proyek

```
batik_classifier/
├── app.py                    ← Flask backend (API + routing)
├── train_model.py            ← Script training model RF
├── download_dataset.py       ← Script download/siapkan dataset
├── requirements.txt          ← Dependensi Python
├── Procfile                  ← Untuk deployment Heroku/Railway
├── model/
│   ├── rf_batik_model.pkl    ← Model hasil training (dibuat otomatis)
│   ├── label_encoder.pkl     ← Label encoder
│   ├── img_size.pkl          ← Ukuran gambar
│   ├── class_names.pkl       ← Nama kelas
│   └── metadata.json         ← Akurasi & info model
├── static/
│   ├── uploads/              ← Gambar yang diupload user
│   ├── confusion_matrix.png  ← Visualisasi hasil training
│   ├── feature_importance.png
│   └── class_dist.png
└── templates/
    ├── index.html            ← Halaman utama
    └── about.html            ← Halaman tentang
```

---

## 🗂️ Dataset

**Motif Batik yang Didukung:**
| No | Motif         | Asal Daerah               |
|----|---------------|---------------------------|
| 1  | Parang        | Keraton Yogyakarta & Solo |
| 2  | Kawung        | Keraton Yogyakarta        |
| 3  | Mega Mendung  | Cirebon, Jawa Barat       |
| 4  | Truntum       | Keraton Solo              |
| 5  | Sido Mukti    | Keraton Solo & Yogyakarta |

**Sumber Dataset:**
- Kaggle: https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs
- GitHub: https://github.com/firman-qs/batik-classification-dataset

**Struktur Folder Dataset:**
```
dataset/
├── Parang/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── Kawung/
│   └── ...
├── Mega Mendung/
│   └── ...
├── Truntum/
│   └── ...
└── Sido Mukti/
    └── ...
```

---

## 🚀 Cara Menjalankan

### 1. Clone / Download Proyek
```bash
git clone https://github.com/username/batikscan.git
cd batikscan
```

### 2. Buat Virtual Environment (disarankan)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Siapkan Dataset
```bash
python download_dataset.py
```
> Pilih opsi 1 untuk download otomatis, atau opsi 2 untuk dataset dummy (testing).
> 
> Untuk hasil terbaik, download manual dari Kaggle lalu taruh di folder `dataset/`
> dengan struktur seperti di atas.

### 5. Training Model
```bash
python train_model.py
```
> Proses ini memakan waktu 1-5 menit tergantung ukuran dataset.
> 
> Output: `model/rf_batik_model.pkl` dan visualisasi di folder `static/`

### 6. Jalankan Aplikasi
```bash
python app.py
```
> Buka browser: **http://localhost:5000**

---

## 🧠 Algoritma Random Forest

### Ekstraksi Fitur (198 fitur/gambar)
| Fitur | Deskripsi | Jumlah |
|-------|-----------|--------|
| Histogram RGB | 3 channel × 32 bin | 96 |
| Statistik Warna | Mean & Std per channel | 6 |
| Gradient Magnitude | 4×4 blok × 2 statistik | 32 |
| Grayscale Histogram | 32 bin | 32 |
| LBP Pattern | Histogram pola lokal | 32 |
| **Total** | | **198** |

### Konfigurasi Model
```python
RandomForestClassifier(
    n_estimators=200,      # 200 pohon keputusan
    max_depth=20,          # Kedalaman maksimum pohon
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",   # Fitur acak per split
    bootstrap=True,        # Bagging
    class_weight="balanced",
    n_jobs=-1              # Paralel semua core
)
```

---

## 🌐 Deployment (Heroku / Railway)

### Buat Procfile
```
web: gunicorn app:app
```

### Push ke GitHub
```bash
git init
git add .
git commit -m "Initial commit BatikScan"
git remote add origin https://github.com/username/batikscan.git
git push -u origin main
```

### Deploy ke Railway (Gratis)
1. Buka https://railway.app
2. "New Project" → "Deploy from GitHub repo"
3. Pilih repository ini
4. Railway otomatis detect Python & deploy ✅

### Deploy ke Heroku
```bash
heroku create nama-app-anda
heroku git:remote -a nama-app-anda
git push heroku main
heroku open
```

---

## 📊 Evaluasi Model

Setelah training, cek hasil di:
- `static/confusion_matrix.png` — Matriks konfusi
- `static/feature_importance.png` — Fitur terpenting
- `static/class_dist.png` — Distribusi kelas dataset
- `model/metadata.json` — Akurasi dan info model

---

## 📚 Referensi

- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
- Dataset: https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs
- Scikit-Learn RF: https://scikit-learn.org/stable/modules/ensemble.html#forest
- Flask Docs: https://flask.palletsprojects.com/

---

## 👤 Informasi Tugas

- **Mata Kuliah**: Machine Learning / Pembelajaran Mesin
- **Algoritma**: Random Forest Classifier
- **Studi Kasus**: Klasifikasi Motif Batik Indonesia
- **Tools**: Python · Flask · Scikit-Learn · OpenCV · Bootstrap
