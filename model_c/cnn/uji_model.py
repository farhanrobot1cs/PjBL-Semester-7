import tensorflow as tf
import numpy as np
import os

# 1. LOKASI MODEL DAN FOLDER VAL
NAMA_FILE_MODEL = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\model_c\lightweight_cnn_solder_model_v2.h5'
FOLDER_VAL_NORMAL = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\model_c\dataset\validation\normal'
FOLDER_VAL_DEFECT = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\model_c\dataset\validation\defect'

IMG_HEIGHT = 64
IMG_WIDTH = 64
class_names = ['defect', 'normal']  # sesuaikan urutan sesuai class_names pas training kamu

print("Memuat model...")
model = tf.keras.models.load_model(NAMA_FILE_MODEL)

def uji_folder(folder_path, label_asli):
    hasil = []
    daftar_file = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for nama_file in daftar_file:
        path_gambar = os.path.join(folder_path, nama_file)
        img = tf.keras.utils.load_img(path_gambar, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        pred = model.predict(img_array, verbose=0)[0]
        confidence = np.max(pred)
        prediksi = class_names[np.argmax(pred)]
        benar = (prediksi == label_asli)
        
        hasil.append({
            'file': nama_file,
            'label_asli': label_asli,
            'prediksi': prediksi,
            'confidence': confidence,
            'benar': benar
        })
    return hasil

print("\nMenguji folder val/normal...")
hasil_normal = uji_folder(FOLDER_VAL_NORMAL, 'normal')

print("Menguji folder val/defect...")
hasil_defect = uji_folder(FOLDER_VAL_DEFECT, 'defect')

semua_hasil = hasil_normal + hasil_defect
confidences = [h['confidence'] for h in semua_hasil]
jumlah_benar = sum(h['benar'] for h in semua_hasil)

print("\n========================================")
print(f"Total foto diuji: {len(semua_hasil)}")
print(f"Jumlah benar: {jumlah_benar} ({100*jumlah_benar/len(semua_hasil):.2f}%)")
print(f"Confidence rata-rata: {np.mean(confidences)*100:.2f}%")
print(f"Confidence terendah: {np.min(confidences)*100:.2f}%")
print(f"Confidence tertinggi: {np.max(confidences)*100:.2f}%")

# Cek berapa banyak yang confidence-nya "biasa aja" (di bawah 90%)
ragu2 = [h for h in semua_hasil if h['confidence'] < 0.90]
print(f"\nJumlah prediksi dengan confidence < 90%: {len(ragu2)} dari {len(semua_hasil)}")

# Tampilkan yang salah (kalau ada)
salah = [h for h in semua_hasil if not h['benar']]
print(f"\nJumlah prediksi SALAH: {len(salah)}")
for h in salah[:10]:  # tampilkan max 10 biar nggak kepanjangan
    print(f"  {h['file']} -> asli: {h['label_asli']}, ditebak: {h['prediksi']} ({h['confidence']*100:.2f}%)")
print("========================================")