import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. Tentukan folder tempat Anda menyimpan 24 gambar defect
# (Sesuaikan jalur folder ini dengan kondisi di PC Lab Anda)
FOLDER_DEFECT_ASLI = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\defect_sample' 
FOLDER_DEFECT_AUGMENTASI = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\augmented_defect'

# 2. Atur Konfigurasi Augmentasi secara Agresif
datagen_defect = ImageDataGenerator(
    rotation_range=15,       # Memutar gambar secara acak maksimal 15 derajat
    width_shift_range=0.1,   # Menggeser gambar ke kanan/kiri sebesar 10%
    height_shift_range=0.1,  # Menggeser gambar ke atas/bawah sebesar 10%
    brightness_range=[0.7, 1.3], # Mengubah cahaya (70% redup sampai 130% lebih terang)
    horizontal_flip=True,    # Membalik gambar secara horizontal (kiri-kanan)
    vertical_flip=True,      # Membalik gambar secara vertikal (atas-bawah)
    fill_mode='nearest'      # Mengisi kekosongan piksel akibat pergeseran/rotasi
)

# 3. Membuat folder tujuan jika belum ada
if not os.path.exists(FOLDER_DEFECT_AUGMENTASI):
    os.makedirs(FOLDER_DEFECT_AUGMENTASI)

# 4. Proses Otomatis Memperbanyak Gambar
# Kode ini akan membaca gambar di folder asli dan menyimpannya ke folder baru
print("Memulai proses augmentasi data...")

# Mengambil semua file gambar di dalam folder defect asli
daftar_gambar = [f for f in os.listdir(FOLDER_DEFECT_ASLI) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Target kita: Menduplikasi setiap gambar sebanyak 150 variasi baru
# 24 gambar asli x 150 variasi = 3.600 gambar defect baru (Siap bersaing dengan 8.000 data normal!)
TARGET_VARIASI_PER_GAMBAR = 150 

for nama_file in daftar_gambar:
    path_gambar = os.path.join(FOLDER_DEFECT_ASLI, nama_file)
    
    # Membaca gambar dan mengubahnya ke format matriks angka yang dipahami AI
    img = tf.keras.preprocessing.image.load_img(path_gambar, target_size=(64, 64))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x.reshape((1,) + x.shape) # Mengubah dimensi agar sesuai format generator
    
    # Menjalankan mesin augmentasi
    i = 0
    for batch in datagen_defect.flow(x, batch_size=1, 
                                     save_to_dir=FOLDER_DEFECT_AUGMENTASI, 
                                     save_prefix='aug_defect', 
                                     save_format='png'):
        i += 1
        if i >= TARGET_VARIASI_PER_GAMBAR:
            break # Berhenti jika satu gambar sudah menghasilkan 150 variasi baru

print(f"Selesai! Periksa folder: {FOLDER_DEFECT_AUGMENTASI}")
