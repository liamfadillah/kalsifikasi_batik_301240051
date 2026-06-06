"""
app.py — BatikScan Flask Application
=====================================
Backend klasifikasi motif batik menggunakan Random Forest.

Jalankan:
    python app.py

Buka browser:
    http://localhost:5000
"""

import os, json, base64, time
import numpy as np
import cv2
import joblib
from flask import (Flask, render_template, request,
                   jsonify, redirect, url_for)
from werkzeug.utils import secure_filename

# ═══════════════════════════════════════════════════════════════
#  FLASK CONFIG
# ═══════════════════════════════════════════════════════════════
app = Flask(__name__)
app.config["SECRET_KEY"]        = "batikscan-secret-2024"
app.config["UPLOAD_FOLDER"]     = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "bmp"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("static", exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  LOAD MODEL
# ═══════════════════════════════════════════════════════════════
MODEL_READY = False
rf_model = label_encoder = None
IMG_SIZE  = 64
CLASS_NAMES = []
META = {}

try:
    rf_model      = joblib.load("model/rf_batik_model.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    IMG_SIZE      = joblib.load("model/img_size.pkl")
    CLASS_NAMES   = joblib.load("model/class_names.pkl")
    with open("model/metadata.json") as f:
        META = json.load(f)
    MODEL_READY = True
    print(f"✅ Model dimuat — {len(CLASS_NAMES)} kelas: {CLASS_NAMES}")
    print(f"   Akurasi training: {META.get('accuracy',0)*100:.2f}%")
except Exception as e:
    print(f"⚠️  Model belum tersedia ({e}). Jalankan train_model.py dulu.")


# ═══════════════════════════════════════════════════════════════
#  INFORMASI MOTIF BATIK
# ═══════════════════════════════════════════════════════════════
BATIK_INFO = {
    "Parang": {
        "asal":      "Keraton Yogyakarta & Solo",
        "filosofi":  "Melambangkan ombak laut yang tak henti — simbol semangat "
                     "dan kekuatan yang terus mengalir tanpa putus.",
        "ciri":      "Motif diagonal berulang menyerupai pisau parang atau "
                     "ombak laut. Warna dominan coklat, hitam, dan krem.",
        "penggunaan":"Upacara adat, pernikahan, acara resmi kerajaan.",
        "warna":     "#8B1A1A",
        "emoji":     "⚔️",
    },
    "Kawung": {
        "asal":      "Keraton Yogyakarta",
        "filosofi":  "Berasal dari buah aren/kawung. Melambangkan kesucian, "
                     "harapan, dan pengendalian diri.",
        "ciri":      "Pola lingkaran atau oval tersusun simetris membentuk "
                     "motif bunga empat kelopak. Biasanya hitam-putih.",
        "penggunaan":"Busana kerajaan, pakaian adat, ritual sakral.",
        "warna":     "#4A235A",
        "emoji":     "🌸",
    },
    "Mega Mendung": {
        "asal":      "Cirebon, Jawa Barat",
        "filosofi":  "Melambangkan kesabaran dan ketenangan hati. Awan "
                     "megah mencerminkan keagungan dan kebesaran jiwa.",
        "ciri":      "Motif awan berlapis bergradasi biru-merah khas Cirebon. "
                     "Dipengaruhi motif awan China (Bangsi).",
        "penggunaan":"Batik khas pesisir, busana seni, dekorasi.",
        "warna":     "#1A3A6A",
        "emoji":     "☁️",
    },
    "Truntum": {
        "asal":      "Keraton Solo (Surakarta)",
        "filosofi":  "Diciptakan Kanjeng Ratu Kencana sebagai simbol cinta "
                     "yang tulus. 'Truntum' berarti tumbuh kembali.",
        "ciri":      "Bintang-bintang kecil berulang di atas latar gelap, "
                     "menyerupai bunga melati atau bintang di langit malam.",
        "penggunaan":"Kain pengantin, pernikahan adat Jawa.",
        "warna":     "#2C4A1A",
        "emoji":     "⭐",
    },
    "Sido Mukti": {
        "asal":      "Keraton Solo & Yogyakarta",
        "filosofi":  "'Sido' = terus-menerus, 'Mukti' = bahagia/sejahtera. "
                     "Harapan akan kebahagiaan abadi bagi yang memakainya.",
        "ciri":      "Kotak-kotak berisi motif kupu-kupu, bunga, dan garuda "
                     "bergantian. Warna sogan (coklat kemerahan) khas.",
        "penggunaan":"Busana pengantin wanita, upacara kebesaran.",
        "warna":     "#8B4513",
        "emoji":     "🦋",
    },
}

def get_batik_info(label: str) -> dict:
    return BATIK_INFO.get(label, {
        "asal":      "Indonesia",
        "filosofi":  "Salah satu kekayaan budaya batik Nusantara.",
        "ciri":      "Motif batik tradisional Indonesia.",
        "penggunaan":"Berbagai keperluan adat dan fashion.",
        "warna":     "#8B4513",
        "emoji":     "🎨",
    })


# ═══════════════════════════════════════════════════════════════
#  EKSTRAKSI FITUR (HARUS IDENTIK dengan train_model.py)
# ═══════════════════════════════════════════════════════════════
def extract_features(image_path: str, img_size: int = 64) -> np.ndarray | None:
    img = cv2.imread(image_path)
    if img is None:
        return None

    img     = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []

    # 1. Histogram RGB
    for ch in range(3):
        h = cv2.calcHist([img_rgb], [ch], None, [32], [0, 256])
        features.extend(cv2.normalize(h, h).flatten())

    # 2. Statistik warna
    for ch in range(3):
        features.append(float(np.mean(img_rgb[:, :, ch])))
        features.append(float(np.std(img_rgb[:, :, ch])))

    # 3. Gradient magnitude
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

    # 5. LBP sederhana
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        shifted = np.roll(np.roll(gray, dy, 0), dx, 1)
        lbp += (gray >= shifted).astype(np.uint8)
    hl = cv2.calcHist([lbp], [0], None, [32], [0, 9])
    features.extend(cv2.normalize(hl, hl).flatten())

    return np.array(features, dtype=np.float32)


def predict_batik(image_path: str):
    """Return (label, confidence_pct, top3_list) atau None."""
    if not MODEL_READY:
        return None, None, None

    feat = extract_features(image_path, IMG_SIZE)
    if feat is None:
        return None, None, None

    feat    = feat.reshape(1, -1)
    enc     = rf_model.predict(feat)[0]
    label   = label_encoder.inverse_transform([enc])[0]
    proba   = rf_model.predict_proba(feat)[0]

    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        {
            "label":       label_encoder.inverse_transform([i])[0],
            "probability": round(float(proba[i] * 100), 2),
        }
        for i in top3_idx
    ]
    confidence = round(float(proba[enc] * 100), 2)
    return label, confidence, top3


def allowed_file(fname: str) -> bool:
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def img_to_b64(path: str) -> str:
    ext  = path.rsplit(".", 1)[1].lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{data}"


# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html",
                           model_ready=MODEL_READY,
                           meta=META,
                           class_names=CLASS_NAMES,
                           batik_info=BATIK_INFO)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Pilih file terlebih dahulu."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Format tidak didukung. Gunakan JPG / PNG / WEBP."}), 400
    if not MODEL_READY:
        return jsonify({"error": "Model belum siap. Jalankan train_model.py."}), 503

    fname  = secure_filename(f"{int(time.time())}_{file.filename}")
    fpath  = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.save(fpath)

    label, conf, top3 = predict_batik(fpath)
    if label is None:
        return jsonify({"error": "Gagal memproses gambar."}), 500

    info = get_batik_info(label)

    return jsonify({
        "success":    True,
        "prediction": label,
        "confidence": conf,
        "top3":       top3,
        "batik_info": info,
        "image_src":  img_to_b64(fpath),
    })


@app.route("/api/status")
def api_status():
    return jsonify({
        "model_ready": MODEL_READY,
        "classes":     CLASS_NAMES,
        "meta":        META,
    })


@app.route("/about")
def about():
    return render_template("about.html",
                           meta=META,
                           class_names=CLASS_NAMES,
                           batik_info=BATIK_INFO)


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
