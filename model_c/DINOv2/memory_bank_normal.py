import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# 1. FOLDER FOTO NORMAL YANG SUDAH ADA (pakai yang di folder train)
FOLDER_NORMAL = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\model_c\dataset\training\normal'

# 2. Berapa banyak foto normal yang dipakai buat "belajar"
# (Nggak perlu semua 5000+ foto, ambil sebagian aja biar nggak lama)
JUMLAH_SAMPEL = 1000

print("Memuat model DINOv2 (sekali download, agak lama di awal)...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()  # mode "cuma lihat", bukan belajar

# 3. Cara mengubah foto jadi format yang dipahami DINOv2
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def ambil_fitur(path_gambar):
    """Ubah 1 foto jadi 'sidik jari' angka"""
    img = Image.open(path_gambar).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        fitur = model(img_tensor)
    return fitur.squeeze().numpy()

# 4. Ambil daftar foto normal, pilih sebagian secara acak
semua_file = [f for f in os.listdir(FOLDER_NORMAL) if f.endswith(('.png', '.jpg', '.jpeg'))]
np.random.seed(123)
file_terpilih = np.random.choice(semua_file, size=min(JUMLAH_SAMPEL, len(semua_file)), replace=False)

print(f"Memproses {len(file_terpilih)} foto normal...")
daftar_fitur = []
for i, nama_file in enumerate(file_terpilih):
    path = os.path.join(FOLDER_NORMAL, nama_file)
    fitur = ambil_fitur(path)
    daftar_fitur.append(fitur)
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(file_terpilih)} selesai...")

# 5. Simpan "database sidik jari" ini biar nggak perlu diulang tiap kali
daftar_fitur = np.array(daftar_fitur)
np.save('memory_bank_normal.npy', daftar_fitur)
print(f"\nSelesai! Database tersimpan: memory_bank_normal.npy ({daftar_fitur.shape})")