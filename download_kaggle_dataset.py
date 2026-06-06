import os
import shutil
import kagglehub

def main():
    print("=" * 60)
    print(" DOWNLOAD DATASET FROM KAGGLE")
    print("=" * 60)
    
    # 1. Download dataset
    print("\n-> Mengunduh dataset 'dionisiusdh/indonesian-batik-motifs' via kagglehub...")
    try:
        download_path = kagglehub.dataset_download('dionisiusdh/indonesian-batik-motifs')
        print(f"[OK] Unduhan selesai! Disimpan di cache: {download_path}")
    except Exception as e:
        print(f"[ERROR] Gagal mengunduh: {e}")
        return
        
    # 2. Periksa isi dari download_path
    print("\n-> Memeriksa struktur dataset yang diunduh...")
    for root, dirs, files in os.walk(download_path):
        level = root.replace(download_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}[DIR] {os.path.basename(root)}/ ({len(dirs)} folder, {len(files)} file)")
        if level > 2:
            break
            
    # 3. Pindahkan file ke folder 'dataset/' lokal
    target_dataset_dir = os.path.join(os.getcwd(), 'dataset')
    print(f"\n-> Memindahkan data ke folder tujuan: {target_dataset_dir}")
    os.makedirs(target_dataset_dir, exist_ok=True)
    
    # Kaggle dataset usually has folders directly inside download_path (e.g. Parang, Kawung, etc.)
    # or it might have a nested structure like 'batik-classification/dataset/...'
    # Let's find the folder that contains subfolders representing classes.
    # We want folders that have image files.
    class_folders = []
    
    # Walk to find folders that directly contain images
    for root, dirs, files in os.walk(download_path):
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if image_files and len(dirs) == 0:
            # This is a leaf folder with images. The class name is the name of this folder.
            class_name = os.path.basename(root)
            class_folders.append((class_name, root))
            
    if not class_folders:
        print("[ERROR] Tidak ditemukan folder kelas berisi gambar!")
        return
        
    print(f"Ditemukan {len(class_folders)} kelas gambar.")
    
    for class_name, src_path in class_folders:
        # Standardize class names (e.g. capitalized, replace underscores/spaces if needed)
        # Let's keep the folder name from Kaggle
        dest_path = os.path.join(target_dataset_dir, class_name)
        os.makedirs(dest_path, exist_ok=True)
        
        print(f"  -> Menyalin {class_name} dari {src_path}...")
        copied = 0
        for item in os.listdir(src_path):
            if item.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                shutil.copy2(os.path.join(src_path, item), os.path.join(dest_path, item))
                copied += 1
        print(f"    [OK] Berhasil menyalin {copied} gambar untuk kelas {class_name}")

    print("\n[OK] Pemindahan dataset selesai!")
    
    # Ringkasan akhir
    print("\n📊 Ringkasan dataset:")
    if os.path.isdir("dataset"):
        for cls in sorted(os.listdir("dataset")):
            path = os.path.join("dataset", cls)
            if os.path.isdir(path):
                count = len([f for f in os.listdir(path)
                             if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))])
                print(f"   {cls}: {count} gambar")

if __name__ == "__main__":
    main()
