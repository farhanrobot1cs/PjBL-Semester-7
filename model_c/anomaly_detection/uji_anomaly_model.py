import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# ===== SETUP =====
PATH_MEMORY_BANK = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\model_c\anomaly_detection\memory_bank_normal.npy'
PATH_THRESHOLD = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\model_c\anomaly_detection\anomaly_threshold.npy'
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

def hitung_skor_keanehan(fitur_foto, memory_bank, k=5):
    jarak = np.linalg.norm(memory_bank - fitur_foto, axis=1)
    jarak_terdekat = np.sort(jarak)[:k]
    return np.mean(jarak_terdekat)

memory_bank = np.load(PATH_MEMORY_BANK)
threshold = float(np.load(PATH_THRESHOLD))
print(f"Threshold yang dipakai: {threshold:.4f}\n")

# ===== UJI SEMUA FOTO =====
def uji_folder(folder_path, label_asli_num):
    hasil = []
    daftar_file = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for nama_file in daftar_file:
        path = os.path.join(folder_path, nama_file)
        fitur = ambil_fitur(path)
        skor = hitung_skor_keanehan(fitur, memory_bank)
        prediksi_num = 1 if skor >= threshold else 0
        hasil.append({
            'file': nama_file,
            'label_asli': label_asli_num,
            'prediksi': prediksi_num,
            'skor': skor
        })
    return hasil

print("Menguji folder val/normal...")
hasil_normal = uji_folder(FOLDER_VAL_NORMAL, 0)  # 0 = normal

print("Menguji folder val/defect...")
hasil_defect = uji_folder(FOLDER_VAL_DEFECT, 1)  # 1 = defect

semua_hasil = hasil_normal + hasil_defect
y_true = [h['label_asli'] for h in semua_hasil]
y_pred = [h['prediksi'] for h in semua_hasil]

# ===== HITUNG METRIK =====
akurasi = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n========================================")
print("HASIL EVALUASI ANOMALY DETECTION (DINOv2)")
print("========================================")
print(f"Total foto diuji: {len(semua_hasil)}")
print(f"Accuracy : {akurasi*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1-score : {f1*100:.2f}%")

print("\nConfusion Matrix:")
print("                  Predicted Normal   Predicted Defect")
print(f"Actual Normal     {cm[0][0]:<18} {cm[0][1]}")
print(f"Actual Defect     {cm[1][0]:<18} {cm[1][1]}")

# Tampilkan yang salah
salah = [h for h in semua_hasil if h['label_asli'] != h['prediksi']]
print(f"\nJumlah prediksi SALAH: {len(salah)}")
for h in salah:
    label_str = 'defect' if h['label_asli']==1 else 'normal'
    pred_str = 'defect' if h['prediksi']==1 else 'normal'
    print(f"  {h['file']} -> asli: {label_str}, ditebak: {pred_str} (skor: {h['skor']:.4f})")
print("========================================")