"""
download_dataset.py
===================
Script otomatis untuk menyiapkan dataset motif batik.

Dataset: Batik Motif Classification
Sumber  : GitHub publik (Reza-Ardiyansyah/batik-motif-dataset)
          Berisi 5 kelas motif batik Indonesia
Kelas   : Parang, Kawung, Mega Mendung, Truntum, Sido Mukti
Total   : ±250 gambar (50/kelas) — ringan, <30 MB
"""

import os, sys, zipfile, shutil, urllib.request, json

# ── URL alternatif dataset publik ─────────────────────────────────────────────
SOURCES = [
    # Opsi 1 – GitHub dataset batik ringan (fork-able)
    {
        "name": "Batik Motif Dataset (GitHub)",
        "url": "https://github.com/ardisyahputra12/batik-motif/archive/refs/heads/main.zip",
        "type": "zip",
        "inner_folder": "batik-motif-main/dataset",
    },
]

CLASSES = ["Parang", "Kawung", "Mega Mendung", "Truntum", "Sido Mukti"]
IMAGES_PER_CLASS = 50   # target minimum


# ── Helpers ───────────────────────────────────────────────────────────────────
def progress_hook(count, block, total):
    pct = min(int(count * block * 100 / total), 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct}%", end="", flush=True)


def try_download_zip(src: dict) -> bool:
    """Download zip, ekstrak, pindahkan ke dataset/"""
    url = src["url"]
    tmp = "_tmp_batik.zip"
    print(f"\n→ Mencoba: {src['name']}")
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=progress_hook)
        print()
        print("  Mengekstrak...")
        with zipfile.ZipFile(tmp, "r") as z:
            z.extractall("_tmp_extracted")
        os.remove(tmp)

        inner = os.path.join("_tmp_extracted", src["inner_folder"])
        if not os.path.isdir(inner):
            # Coba cari folder yang ada
            for root, dirs, _ in os.walk("_tmp_extracted"):
                for d in dirs:
                    if any(cls.lower() in d.lower() for cls in CLASSES):
                        inner = root
                        break

        if os.path.isdir(inner):
            os.makedirs("dataset", exist_ok=True)
            for item in os.listdir(inner):
                src_path = os.path.join(inner, item)
                dst_path = os.path.join("dataset", item)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            shutil.rmtree("_tmp_extracted", ignore_errors=True)
            return True

        shutil.rmtree("_tmp_extracted", ignore_errors=True)
        return False

    except Exception as e:
        print(f"\n  ✗ Gagal: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        shutil.rmtree("_tmp_extracted", ignore_errors=True)
        return False


def create_dummy_dataset():
    """
    Buat dataset dummy berwarna-pola untuk testing pipeline.
    Ganti dengan gambar asli sebelum presentasi!
    """
    import numpy as np, cv2

    print("\n🎨 Membuat dummy dataset motif batik untuk testing...")

    # Setiap kelas punya pola berbeda agar model bisa belajar
    patterns = {
        "Parang":       {"bg": (30,  80, 140), "fg": (220, 180,  50), "shape": "diagonal"},
        "Kawung":       {"bg": (140, 30,  80), "fg": (240, 220, 180), "shape": "circle"},
        "Mega Mendung": {"bg": (20, 100, 160), "fg": (200, 230, 255), "shape": "wave"},
        "Truntum":      {"bg": (60, 120,  40), "fg": (230, 200, 100), "shape": "star"},
        "Sido Mukti":   {"bg": (120, 60,  20), "fg": (210, 170,  80), "shape": "grid"},
    }

    N = 60   # gambar per kelas

    for cls, cfg in patterns.items():
        folder = os.path.join("dataset", cls)
        os.makedirs(folder, exist_ok=True)
        bg, fg, shape = cfg["bg"], cfg["fg"], cfg["shape"]

        for i in range(N):
            img = np.ones((128, 128, 3), np.uint8)
            img[:] = bg[::-1]   # BGR
            rng = np.random.default_rng(i)

            if shape == "diagonal":
                for k in range(-5, 15):
                    offset = k * 18 + rng.integers(-3, 3)
                    pts = np.array([[offset,0],[offset+12,0],[0,offset+12],[0,offset]], np.int32)
                    cv2.fillPoly(img, [pts], fg[::-1])

            elif shape == "circle":
                for r in range(4):
                    for c in range(4):
                        cx, cy = 16 + c*32 + rng.integers(-2,2), 16 + r*32 + rng.integers(-2,2)
                        cv2.circle(img, (cx,cy), 10, fg[::-1], -1)
                        cv2.circle(img, (cx,cy), 5, bg[::-1], -1)

            elif shape == "wave":
                for y in range(0, 128, 20):
                    pts = []
                    for x in range(0, 128, 4):
                        wave_y = int(y + 8 * np.sin((x + i*3) * 0.15))
                        pts.append([x, wave_y])
                    pts = np.array(pts, np.int32)
                    cv2.polylines(img, [pts], False, fg[::-1], 3)

            elif shape == "star":
                for r in range(4):
                    for c in range(4):
                        cx, cy = 16 + c*32 + rng.integers(-3,3), 16 + r*32 + rng.integers(-3,3)
                        for angle in range(0, 360, 60):
                            rad = np.radians(angle)
                            ex = int(cx + 12 * np.cos(rad))
                            ey = int(cy + 12 * np.sin(rad))
                            cv2.line(img, (cx,cy), (ex,ey), fg[::-1], 2)

            elif shape == "grid":
                for k in range(0, 128, 16):
                    cv2.line(img, (k, 0), (k, 128), fg[::-1], 1 + (k%32==0))
                    cv2.line(img, (0, k), (128, k), fg[::-1], 1 + (k%32==0))
                for r in range(4):
                    for c in range(4):
                        cx, cy = 16 + c*32, 16 + r*32
                        cv2.rectangle(img, (cx-6,cy-6), (cx+6,cy+6), fg[::-1], -1)

            # Tambah noise ringan
            noise = rng.integers(-15, 15, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            cv2.imwrite(os.path.join(folder, f"batik_{i+1:03d}.jpg"), img)

        print(f"  ✓ {cls}: {N} gambar dibuat")

    print("\n⚠️  Dataset dummy untuk testing pipeline.")
    print("   Untuk akurasi nyata, gunakan gambar batik asli.")
    print("   Sumber dataset asli disarankan:")
    print("   → https://www.kaggle.com/datasets/dionisiusdh/indonesian-batik-motifs")
    print("   → https://github.com/firman-qs/batik-classification-dataset")
    print(f"\n✅ Dummy dataset siap! Total: {len(patterns)*N} gambar")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(" PERSIAPAN DATASET MOTIF BATIK INDONESIA")
    print("=" * 60)

    if os.path.isdir("dataset") and any(os.listdir("dataset")):
        print("\n✅ Folder 'dataset/' sudah ada dan berisi data.")
        print("   Hapus folder 'dataset/' jika ingin mengunduh ulang.")
        sys.exit(0)

    # Coba download otomatis
    success = False
    for src in SOURCES:
        if try_download_zip(src):
            success = True
            print(f"\n✅ Dataset berhasil diunduh dari: {src['name']}")
            break

    if not success:
        print("\n⚠️  Download otomatis gagal (mungkin koneksi/URL berubah).")
        create_dummy_dataset()

    # Ringkasan akhir
    print("\n📊 Ringkasan dataset:")
    if os.path.isdir("dataset"):
        for cls in sorted(os.listdir("dataset")):
            path = os.path.join("dataset", cls)
            if os.path.isdir(path):
                count = len([f for f in os.listdir(path)
                             if f.lower().endswith(('.jpg','.jpeg','.png'))])
                print(f"   {cls}: {count} gambar")
