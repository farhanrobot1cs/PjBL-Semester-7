import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# ===== SEMUA LOKASI FILE, PAKAI ALAMAT LENGKAP =====
FOLDER_PROJECT = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\model_c\anomaly_detection'
FOLDER_TRAIN_NORMAL = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\model_c\dataset\training\normal'
FOLDER_VAL_NORMAL = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\model_c\dataset\validation\normal'
FOLDER_VAL_DEFECT = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\model_c\dataset\validation\defect'

PATH_MEMORY_BANK = os.path.join(FOLDER_PROJECT, 'memory_bank_normal.npy')
PATH_THRESHOLD = os.path.join(FOLDER_PROJECT, 'anomaly_threshold.npy')

JUMLAH_SAMPEL_REFERENSI = 1000

# Pastikan folder tujuan ada
os.makedirs(FOLDER_PROJECT, exist_ok=True)

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

def hitung_skor_keanehan(fitur_foto, memory_bank, k=5):
    jarak = np.linalg.norm(memory_bank - fitur_foto, axis=1)
    jarak_terdekat = np.sort(jarak)[:k]
    return np.mean(jarak_terdekat)

# ===== LANGKAH 1: BIKIN MEMORY BANK =====
print("\n[1/3] Membangun memory bank dari foto normal...")
semua_file = [f for f in os.listdir(FOLDER_TRAIN_NORMAL) if f.endswith(('.png', '.jpg', '.jpeg'))]
np.random.seed(123)
file_terpilih = np.random.choice(semua_file, size=min(JUMLAH_SAMPEL_REFERENSI, len(semua_file)), replace=False)

daftar_fitur = []
for i, nama_file in enumerate(file_terpilih):
    path = os.path.join(FOLDER_TRAIN_NORMAL, nama_file)
    daftar_fitur.append(ambil_fitur(path))
    if (i+1) % 200 == 0:
        print(f"    {i+1}/{len(file_terpilih)} foto diproses...")

memory_bank = np.array(daftar_fitur)
np.save(PATH_MEMORY_BANK, memory_bank)
print(f"    Memory bank disimpan di: {PATH_MEMORY_BANK}")
print(f"    Ukuran: {memory_bank.shape}")

# ===== LANGKAH 2: HITUNG SKOR DI DATA VALIDASI =====
print("\n[2/3] Menghitung skor di data validasi...")

def uji_folder(folder_path, label_asli_num):
    hasil = []
    daftar_file = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for nama_file in daftar_file:
        path = os.path.join(folder_path, nama_file)
        fitur = ambil_fitur(path)
        skor = hitung_skor_keanehan(fitur, memory_bank)
        hasil.append({'file': nama_file, 'label_asli': label_asli_num, 'skor': skor})
    return hasil

hasil_normal = uji_folder(FOLDER_VAL_NORMAL, 0)
hasil_defect = uji_folder(FOLDER_VAL_DEFECT, 1)

skor_normal = [h['skor'] for h in hasil_normal]
skor_defect = [h['skor'] for h in hasil_defect]
print(f"    Rata-rata skor Normal: {np.mean(skor_normal):.4f}")
print(f"    Rata-rata skor Defect: {np.mean(skor_defect):.4f}")

# ===== LANGKAH 3: HITUNG THRESHOLD =====
print("\n[3/3] Menghitung threshold optimal...")
semua_skor = np.array(skor_normal + skor_defect)
semua_label = np.array([0]*len(skor_normal) + [1]*len(skor_defect))

fpr, tpr, thresholds = roc_curve(semua_label, semua_skor)
j_scores = tpr - fpr
idx_terbaik = np.argmax(j_scores)
threshold_optimal = float(thresholds[idx_terbaik])

np.save(PATH_THRESHOLD, threshold_optimal)
print(f"    Threshold disimpan di: {PATH_THRESHOLD}")
print(f"    Nilai threshold: {threshold_optimal:.4f}")

# ===== EVALUASI LENGKAP (Accuracy, Precision, Recall, F1, Confusion Matrix) =====
y_true = semua_label
y_pred = (semua_skor >= threshold_optimal).astype(int)

akurasi = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
auc = roc_auc_score(y_true, semua_skor)

print("\n========================================")
print("HASIL EVALUASI ANOMALY DETECTION (DINOv2)")
print("========================================")
print(f"AUC Score: {auc:.4f}")
print(f"Accuracy : {akurasi*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1-score : {f1*100:.2f}%")

print("\nConfusion Matrix:")
print("                  Predicted Normal   Predicted Defect")
print(f"Actual Normal     {cm[0][0]:<18} {cm[0][1]}")
print(f"Actual Defect     {cm[1][0]:<18} {cm[1][1]}")

print("\n[SUKSES] Model anomaly detection siap dipakai!")
print(f"File tersimpan di folder: {FOLDER_PROJECT}")