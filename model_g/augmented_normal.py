import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. TENTUKAN FOLDER GAMBAR NORMAL ASLI DAN FOLDER HASILNYA
FOLDER_NORMAL_ASLI = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\final_dataset\normal' 
FOLDER_NORMAL_AUGMENTASI = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\final_dataset_robust\normal'

# 2. Atur Variasi Kecerahan Saja (Tanpa merubah bentuk agar tetap normal)
datagen_normal = ImageDataGenerator(
    brightness_range=[0.7, 1.3], # Samakan persis: 70% redup sampai 130% terang
    fill_mode='nearest'
)

if not os.path.exists(FOLDER_NORMAL_AUGMENTASI):
    os.makedirs(FOLDER_NORMAL_AUGMENTASI)

print("Membuat gambar normal menjadi robust terhadap cahaya...")
daftar_gambar = [f for f in os.listdir(FOLDER_NORMAL_ASLI) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Kita cukup membuat 1 variasi baru per gambar asli agar jumlahnya tetap seimbang
for nama_file in daftar_gambar:
    path_gambar = os.path.join(FOLDER_NORMAL_ASLI, nama_file)
    img = tf.keras.preprocessing.image.load_img(path_gambar, target_size=(64, 64))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    
    # Jalankan mesin augmentasi cahaya
    for batch in datagen_normal.flow(x, batch_size=1, 
                                     save_to_dir=FOLDER_NORMAL_AUGMENTASI, 
                                     save_prefix='robust_norm', 
                                     save_format='png'):
        break # Cukup 1 variasi per gambar

print(f"Selesai! Sekarang folder gambar Normal Anda sudah kebal cahaya di: {FOLDER_NORMAL_AUGMENTASI}")
