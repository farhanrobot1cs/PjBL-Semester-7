import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

# Gabungkan semua skor dan label aslinya
semua_skor = skor_normal + skor_defect  # dari langkah sebelumnya
semua_label = [0]*len(8.1211) + [1]*len(17.8795)  # 0=normal, 1=defect

semua_skor = np.array(semua_skor)
semua_label = np.array(semua_label)

# Cari threshold terbaik pakai ROC curve
fpr, tpr, thresholds = roc_curve(semua_label, semua_skor)

# Youden's J statistic: cari titik yang paling optimal (TPR tinggi, FPR rendah)
j_scores = tpr - fpr
idx_terbaik = np.argmax(j_scores)
threshold_optimal = thresholds[idx_terbaik]

print(f"Threshold optimal: {threshold_optimal:.4f}")
print(f"Pada threshold ini -> TPR (Recall Defect): {tpr[idx_terbaik]:.4f}, FPR: {fpr[idx_terbaik]:.4f}")

auc = roc_auc_score(semua_label, semua_skor)
print(f"AUC Score: {auc:.4f}")  # semakin dekat 1.0, semakin bagus pemisahannya

# Coba terapkan threshold ke semua data
prediksi = (semua_skor >= threshold_optimal).astype(int)
benar = (prediksi == semua_label).sum()
print(f"\nAkurasi dengan threshold ini: {benar}/{len(semua_label)} ({100*benar/len(semua_label):.2f}%)")