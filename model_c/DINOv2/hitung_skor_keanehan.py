import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from sklearn.metrics import roc_curve, roc_auc_score  # <-- baris yang kurang, tambahkan ini

# 1. FOLDER FOTO YANG MAU DIUJI
FOLDER_VAL_NORMAL = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\model_c\dataset\validation\normal'
FOLDER_VAL_DEFECT = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\model_c\dataset\validation\defect'

print("Memuat model DINOv2...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def ambil_fitur(path_gambar):
    img = Image.open(path_gambar).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        fitur = model(img_tensor)
    return fitur.squeeze().numpy()

memory_bank = np.load('memory_bank_normal.npy')
print(f"Memory bank dimuat: {memory_bank.shape}")

def hitung_skor_keanehan(fitur_foto, memory_bank, k=5):
    jarak = np.linalg.norm(memory_bank - fitur_foto, axis=1)
    jarak_terdekat = np.sort(jarak)[:k]
    return np.mean(jarak_terdekat)

def uji_folder(folder_path, label_asli):
    hasil = []
    daftar_file = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for nama_file in daftar_file:
        path = os.path.join(folder_path, nama_file)
        fitur = ambil_fitur(path)
        skor = hitung_skor_keanehan(fitur, memory_bank)
        hasil.append({'file': nama_file, 'label_asli': label_asli, 'skor': skor})
    return hasil

# ===== BAGIAN 2: HITUNG SKOR =====
print("Menguji folder val/normal...")
hasil_normal = uji_folder(FOLDER_VAL_NORMAL, 'normal')

print("Menguji folder val/defect...")
hasil_defect = uji_folder(FOLDER_VAL_DEFECT, 'defect')

skor_normal = [h['skor'] for h in hasil_normal]
skor_defect = [h['skor'] for h in hasil_defect]

print(f"\nRata-rata skor Normal: {np.mean(skor_normal):.4f}")
print(f"Rata-rata skor Defect: {np.mean(skor_defect):.4f}")

# ===== BAGIAN 3: HITUNG THRESHOLD =====
semua_skor = np.array(skor_normal + skor_defect)
semua_label = np.array([0]*len(skor_normal) + [1]*len(skor_defect))

fpr, tpr, thresholds = roc_curve(semua_label, semua_skor)
j_scores = tpr - fpr
idx_terbaik = np.argmax(j_scores)
threshold_optimal = thresholds[idx_terbaik]

print(f"\nThreshold optimal: {threshold_optimal:.4f}")
print(f"TPR (Recall Defect): {tpr[idx_terbaik]:.4f}, FPR: {fpr[idx_terbaik]:.4f}")

auc = roc_auc_score(semua_label, semua_skor)
print(f"AUC Score: {auc:.4f}")

prediksi = (semua_skor >= threshold_optimal).astype(int)
benar = (prediksi == semua_label).sum()
print(f"Akurasi dengan threshold ini: {benar}/{len(semua_label)} ({100*benar/len(semua_label):.2f}%)")