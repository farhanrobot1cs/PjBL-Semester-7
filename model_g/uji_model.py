import tensorflow as tf
import numpy as np

# 1. TENTUKAN PATH MODEL BARU DAN GAMBAR YANG AKAN DIUJI
NAMA_FILE_MODEL = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\lightweight_cnn_solder_model.h5'
PATH_GAMBAR_BARU = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\folder uji\normal\2.jpg'

IMG_HEIGHT = 64
IMG_WIDTH = 64

print("Memuat model AI...")
# 2. Memuat kembali model h5 yang sudah Anda latih sebelumnya
model = tf.keras.models.load_model(NAMA_FILE_MODEL)

print("Memproses gambar uji coba...")
# 3. Membaca gambar baru dan mengubah ukurannya menjadi 64x64 piksel
img = tf.keras.utils.load_img(
    PATH_GAMBAR_BARU, target_size=(IMG_HEIGHT, IMG_WIDTH)
)

# Mengubah gambar menjadi matriks angka yang dipahami oleh TensorFlow
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Menambah dimensi batch

print("Melakukan prediksi...")
# 4. Model menebak gambar tersebut
predictions = model.predict(img_array)

# Ambil hasil prediksi index ke-0 tanpa softmax ganda
score = predictions[0]

# Nama kelas disesuaikan otomatis berdasarkan urutan alfabet folder Anda
class_names = ['Defect', 'Normal']

nama_terpilih = class_names[np.argmax(score)]
persentase_keyakinan = 100 * np.max(score)

print("\n========================================")
print(f"HASIL DETEKSI: Solder ini diprediksi -> {nama_terpilih.upper()}")
print(f"TINGKAT KEYAKINAN AI: {persentase_keyakinan:.2f}%")
print("========================================")
