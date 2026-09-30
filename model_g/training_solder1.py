import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. TENTUKAN PATH DATASET ANDA
# Ganti tanda r'...' di bawah ini dengan lokasi folder dataset_final Anda
PATH_DATASET = r'D:\Kuliah\Kuliah Luar Biasa\Semester 7\NFU FTIP Fall\Materi perkuliahan\Soldering defect project\final_dataset'

# 2. Pengaturan Parameter Gambar dan Batch
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32 # Ukuran standar agar RAM PC Lab tidak penuh

print("Membaca dan membagi dataset...")

# 3. Otomatis Membagi Data (80% untuk Belajar / Training)
train_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_DATASET,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# 4. Otomatis Membagi Data (20% untuk Ujian / Validation)
val_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_DATASET,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# 5. Optimasi Performa Loading Gambar di PC
# Menggunakan fitur AUTOTUNE agar komputer membaca gambar lebih cepat dari harddisk
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 6. Struktur Arsitektur Lightweight CNN Anda
model = models.Sequential([
    # Mengubah nilai warna piksel dari 0-255 menjadi 0-1 agar model cepat pintar
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)), 
    
    # Blok Konvolusi 1
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Blok Konvolusi 2
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Blok Klasifikasi
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),  # Mencegah model menghafal buta data augmentasi
    layers.Dense(2, activation='softmax') # Output 2 kelas: Normal atau Defect
])

# 7. Mengompilasi Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Menampilkan ringkasan model di terminal VS Code
model.summary()

# 8. Memulai Proses Pelatihan sebanyak 10 Putaran (Epochs)
print("\n[INFO] Memulai pelatihan model AI. Mohon tunggu...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# 9. Menyimpan Model Pintar Anda
model.save('model_deteksi_solder_final.h5')
print("\n[SUKSES] Model sukses dilatih dan disimpan dengan nama 'model_deteksi_solder_final.h5'")
