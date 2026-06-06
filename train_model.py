"""
train_model.py
==============
Melatih model Random Forest untuk klasifikasi motif batik Indonesia.

Jalankan SEKALI sebelum menjalankan app.py:
    python train_model.py

Output:
    model/rf_batik_model.pkl
    model/label_encoder.pkl
    model/img_size.pkl
    model/class_names.pkl
    static/confusion_matrix.png
    static/feature_importance.png
"""

import os, sys, time
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder

# ═══════════════════════════════════════════════════════════════
#  KONFIGURASI
# ═══════════════════════════════════════════════════════════════
DATASET_PATH  = "dataset"
IMG_SIZE      = 64        # resize gambar ke 64×64 px
N_ESTIMATORS  = 200       # jumlah pohon
MAX_DEPTH     = 20
MIN_SAMPLES_SPLIT = 4
MIN_SAMPLES_LEAF  = 2
RANDOM_STATE  = 42
TEST_SIZE     = 0.2

os.makedirs("model", exist_ok=True)
os.makedirs("static", exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  EKSTRAKSI FITUR
# ═══════════════════════════════════════════════════════════════
def extract_features(image_path: str, img_size: int = 64) -> np.ndarray | None:
    """
    Ekstrak vektor fitur dari sebuah gambar.
    Fitur yang digunakan:
      1. Histogram warna RGB  (3 × 32 bin = 96 nilai)
      2. Statistik warna      (mean + std per channel = 6 nilai)
      3. Fitur tekstur gradient (4×4 blok × 2 statistik = 32 nilai)
      4. Histogram grayscale  (32 bin = 32 nilai)
      5. LBP sederhana        (histogram 32 bin = 32 nilai)
    Total ≈ 198 fitur
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []

    # 1. Histogram RGB
    for ch in range(3):
        h = cv2.calcHist([img_rgb], [ch], None, [32], [0, 256])
        features.extend(cv2.normalize(h, h).flatten())

    # 2. Statistik warna per channel
    for ch in range(3):
        features.append(float(np.mean(img_rgb[:, :, ch])))
        features.append(float(np.std(img_rgb[:, :, ch])))

    # 3. Gradient magnitude, dibagi 4×4 blok
    gx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    bs  = img_size // 4
    for r in range(4):
        for c in range(4):
            blk = mag[r*bs:(r+1)*bs, c*bs:(c+1)*bs]
            features.append(float(np.mean(blk)))
            features.append(float(np.std(blk)))

    # 4. Histogram grayscale
    hg = cv2.calcHist([gray], [0], None, [32], [0, 256])
    features.extend(cv2.normalize(hg, hg).flatten())

    # 5. LBP sederhana (uniform pattern histogram)
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        shifted = np.roll(np.roll(gray, dy, 0), dx, 1)
        lbp += (gray >= shifted).astype(np.uint8)
    hl = cv2.calcHist([lbp], [0], None, [32], [0, 9])
    features.extend(cv2.normalize(hl, hl).flatten())

    return np.array(features, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  LOAD DATASET
# ═══════════════════════════════════════════════════════════════
def load_dataset(path: str, img_size: int = 64):
    if not os.path.isdir(path):
        print(f"[ERROR]  Folder '{path}' tidak ditemukan.")
        print("    Jalankan: python download_dataset.py")
        sys.exit(1)

    classes = sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ])
    if not classes:
        print("[ERROR]  Folder 'dataset/' kosong. Jalankan download_dataset.py dulu.")
        sys.exit(1)

    print(f"  Ditemukan {len(classes)} kelas: {classes}\n")

    X, y = [], []
    for cls in classes:
        cls_path = os.path.join(path, cls)
        files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        ok = 0
        for f in files:
            feat = extract_features(os.path.join(cls_path, f), img_size)
            if feat is not None:
                X.append(feat)
                y.append(cls)
                ok += 1
        print(f"  [OK] {cls:<20} {ok:>3} gambar  ({img_size}x{img_size} px, "
              f"{len(X[-1]) if X else 0} fitur)")

    return np.array(X), np.array(y), classes


# ═══════════════════════════════════════════════════════════════
#  VISUALISASI
# ═══════════════════════════════════════════════════════════════
def plot_confusion_matrix(cm, class_names, out="static/confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)*1.5),
                                    max(6, len(class_names)*1.2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix – Klasifikasi Motif Batik",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Prediksi", fontsize=12)
    ax.set_ylabel("Sebenarnya", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> Disimpan: {out}")


def plot_feature_importance(model, top_n=20, out="static/feature_importance.png"):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(top_n), imp[idx], color="#8B4513", alpha=0.85)
    ax.set_title(f"Top {top_n} Fitur Terpenting – Random Forest Batik",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Indeks Fitur")
    ax.set_ylabel("Importance")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([str(i) for i in idx], fontsize=8, rotation=45)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> Disimpan: {out}")


def plot_class_distribution(y, out="static/class_dist.png"):
    from collections import Counter
    counts = Counter(y)
    # Sort by class name to maintain alphabetical order
    sorted_counts = dict(sorted(counts.items()))
    fig, ax = plt.subplots(figsize=(max(10, len(sorted_counts) * 0.5), 6))
    bars = ax.bar(sorted_counts.keys(), sorted_counts.values(),
                  color="#8B4513")
    ax.bar_label(bars, padding=3, fontsize=9, fontweight="bold")
    ax.set_title("Distribusi Kelas Dataset Batik", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Jumlah Gambar", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  -> Disimpan: {out}")


# ═══════════════════════════════════════════════════════════════
#  MAIN TRAINING
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 62)
    print("  TRAINING MODEL RANDOM FOREST — MOTIF BATIK INDONESIA")
    print("=" * 62)

    # ── 1. Load ──────────────────────────────────────────────────
    print("\n[1/6] Memuat dan mengekstrak fitur dataset...")
    t0 = time.time()
    X, y, class_names = load_dataset(DATASET_PATH, IMG_SIZE)
    print(f"\n  Total sampel : {len(X)}")
    print(f"  Fitur/sampel : {X.shape[1]}")
    print(f"  Waktu        : {time.time()-t0:.1f}s")

    plot_class_distribution(y)

    # ── 2. Encode label ──────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ── 3. Split ─────────────────────────────────────────────────
    print(f"\n[2/6] Membagi dataset (train {int((1-TEST_SIZE)*100)}% / "
          f"test {int(TEST_SIZE*100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_enc
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 4. Build model ───────────────────────────────────────────
    print(f"\n[3/6] Membangun Random Forest ({N_ESTIMATORS} pohon)...")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
        class_weight="balanced",
    )

    # ── 5. Train ─────────────────────────────────────────────────
    print("\n[4/6] Melatih model...")
    t1 = time.time()
    rf.fit(X_train, y_train)
    print(f"  Selesai dalam {time.time()-t1:.1f}s")

    # ── 6. Evaluate ──────────────────────────────────────────────
    print("\n[5/6] Evaluasi model...")
    y_pred = rf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n  =================================")
    print(f"    Akurasi Test Set : {acc*100:6.2f}%")
    print(f"  =================================")

    # Cross-validation
    cv_scores = cross_val_score(rf, X, y_enc, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"\n  Cross-Validation (5-fold):")
    print(f"  Mean: {cv_scores.mean()*100:.2f}%  +/-  {cv_scores.std()*100:.2f}%")

    print(f"\n  Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred,
                                 target_names=le.classes_))

    # Confusion matrix & feature importance plots
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, le.classes_)
    plot_feature_importance(rf)

    # ── 7. Simpan ────────────────────────────────────────────────
    print("\n[6/6] Menyimpan model...")
    joblib.dump(rf,          "model/rf_batik_model.pkl")
    joblib.dump(le,          "model/label_encoder.pkl")
    joblib.dump(IMG_SIZE,    "model/img_size.pkl")
    joblib.dump(class_names, "model/class_names.pkl")

    # Simpan metadata
    meta = {
        "accuracy": round(float(acc), 4),
        "cv_mean":  round(float(cv_scores.mean()), 4),
        "cv_std":   round(float(cv_scores.std()), 4),
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "img_size": IMG_SIZE,
        "n_features": int(X.shape[1]),
        "classes": list(le.classes_),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    import json
    with open("model/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("  model/rf_batik_model.pkl  [OK]")
    print("  model/label_encoder.pkl   [OK]")
    print("  model/metadata.json       [OK]")

    print("\n" + "=" * 62)
    print("  [SUCCESS]  TRAINING SELESAI -- Jalankan: python app.py")
    print("=" * 62)
