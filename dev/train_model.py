import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# 1. Load Data
try:
    print("Membaca dataset...")
    df = pd.read_csv('dataset_jurusan.csv')
except FileNotFoundError:
    print("❌ Error: File 'dataset_jurusan.csv' tidak ditemukan. Jalankan buat_dataset.py dulu!")
    exit()

# 2. Preprocessing (Encoding Text -> Angka)
le_sekolah = LabelEncoder()
le_jurusan = LabelEncoder()

df['asal_sekolah_enc'] = le_sekolah.fit_transform(df['asal_sekolah'])
df['jurusan_asal_enc'] = le_jurusan.fit_transform(df['jurusan_asal'])

# 3. Fitur yang dipakai untuk training
features = [
    'asal_sekolah_enc', 'jurusan_asal_enc', 
    'pkn', 'mtk', 'indo', 'ing', 'lainnya',
    'hobi_olahraga', 'hobi_musik', 'hobi_editing', 'hobi_game'
]
X = df[features]

# 4. Scaling (Agar angka 0-100 dan 0-1 setara)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Training K-Means
print("Melatih model AI...")
k = 15 # Jumlah cluster
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 6. Mapping Cluster -> Nama Jurusan Dominan
df['cluster'] = clusters
cluster_map = {}

for i in range(k):
    # Ambil jurusan paling sering muncul di cluster ini
    jurusan_top = df[df['cluster'] == i]['jurusan_kuliah'].mode()[0]
    # Ambil alternatif
    jurusan_alt = df[df['cluster'] == i]['jurusan_kuliah'].value_counts().index[:3].tolist()
    cluster_map[i] = jurusan_alt

# 7. Simpan Model & Scaler
print("Menyimpan model ke 'model_k-means.pkl'...")
data_to_save = {
    "model": kmeans,
    "scaler": scaler,
    "le_sekolah": le_sekolah,
    "le_jurusan": le_jurusan,
    "cluster_map": cluster_map
}

with open('model_k-means.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("✅ Selesai! File 'model_k-means.pkl' siap digunakan.")