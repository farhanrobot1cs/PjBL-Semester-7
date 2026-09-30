import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. FOLDER SUMBER: foto defect asli yang SUDAH masuk folder train
#    (bukan yang di folder val, yang di val jangan disentuh!)
FOLDER_SUMBER = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\model_claude\dataset\training\defect_sample'

# 2. FOLDER TUJUAN: simpan hasil augmentasi LANGSUNG ke folder yang sama
FOLDER_TUJUAN = FOLDER_SUMBER

# 3. Pengaturan cara mengubah gambar
datagen_defect = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# 4. Ambil daftar foto ASLI dulu (sebelum ada tambahan hasil augmentasi)
#    Penting: jalankan skrip ini SEKALI SAJA di folder yang masih bersih,
#    supaya tidak "mengaugmentasi hasil augmentasi"
daftar_gambar = [f for f in os.listdir(FOLDER_SUMBER) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Ditemukan {len(daftar_gambar)} foto asli di folder train/defect")

# 5. Berapa banyak variasi per foto
TARGET_VARIASI_PER_GAMBAR = 150

for nama_file in daftar_gambar:
    path_gambar = os.path.join(FOLDER_SUMBER, nama_file)

    img = tf.keras.preprocessing.image.load_img(path_gambar, target_size=(64, 64))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen_defect.flow(
        x, batch_size=1,
        save_to_dir=FOLDER_TUJUAN,
        save_prefix='aug_defect',
        save_format='png'
    ):
        i += 1
        if i >= TARGET_VARIASI_PER_GAMBAR:
            break

print(f"Selesai! Cek folder: {FOLDER_TUJUAN}")
print(f"Sekarang harusnya ada sekitar {len(daftar_gambar) * TARGET_VARIASI_PER_GAMBAR} foto tambahan.")