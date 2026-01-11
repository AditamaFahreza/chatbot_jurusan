import pandas as pd
import random
import numpy as np

# Jumlah data
n = 1000
data = []

# Pola-pola logis (Rules) untuk generate data
print("Sedang membuat data dummy...")
for _ in range(n):
    # Acak profil siswa
    profil = random.choice(['teknik', 'kesehatan', 'sosial', 'seni', 'olahraga'])
    
    # Default values
    asal = random.choice(['SMA', 'SMK'])
    jurusan_sekolah = 'IPA' if asal == 'SMA' else random.choice(['RPL', 'TKJ', 'Mesin', 'Akuntansi'])
    if profil == 'sosial': jurusan_sekolah = 'IPS'
    
    pkn = random.randint(70, 90)
    indo = random.randint(70, 90)
    ing = random.randint(60, 90)
    mtk = random.randint(60, 90)
    lainnya = random.randint(60, 90) # Nilai produktif/peminatan
    
    # Hobi (0 = Tidak, 1 = Ya)
    h_olr, h_mus, h_edit, h_game = 0, 0, 0, 0
    
    target_jurusan = "Umum"

    # --- SETTING NILAI BERDASARKAN PROFIL ---
    if profil == 'teknik': # Calon anak IT/Teknik
        mtk = random.randint(85, 100)
        ing = random.randint(75, 95)
        h_game = random.choice([0, 1])
        h_edit = random.choice([0, 1])
        target_jurusan = random.choice(['Teknik Informatika', 'Sistem Informasi', 'Teknik Mesin'])
        
    elif profil == 'kesehatan': # Calon Dokter/Perawat
        if asal == 'SMA': jurusan_sekolah = 'IPA'
        lainnya = random.randint(85, 100) # Biologi/Kimia tinggi
        indo = random.randint(80, 95)
        target_jurusan = random.choice(['Kedokteran', 'Farmasi', 'Keperawatan', 'Gizi'])

    elif profil == 'sosial': # Calon Hukum/Manajemen
        jurusan_sekolah = 'IPS'
        pkn = random.randint(85, 100)
        indo = random.randint(85, 100)
        lainnya = random.randint(80, 95) # Ekonomi/Sosiologi
        target_jurusan = random.choice(['Hukum', 'Manajemen', 'Akuntansi', 'Ilmu Komunikasi'])

    elif profil == 'seni': # Calon DKV/Musik
        h_edit = 1 if random.random() > 0.3 else 0
        h_mus = 1 if random.random() > 0.3 else 0
        target_jurusan = random.choice(['DKV', 'Seni Musik', 'Desain Interior'])

    elif profil == 'olahraga': # Calon Atlet
        h_olr = 1
        target_jurusan = random.choice(['Ilmu Keolahragaan', 'Pendidikan Jasmani'])

    data.append([asal, jurusan_sekolah, pkn, mtk, indo, ing, lainnya, h_olr, h_mus, h_edit, h_game, target_jurusan])

# Buat DataFrame
df = pd.DataFrame(data, columns=[
    'asal_sekolah', 'jurusan_asal', 'pkn', 'mtk', 'indo', 'ing', 'lainnya', 
    'hobi_olahraga', 'hobi_musik', 'hobi_editing', 'hobi_game', 'jurusan_kuliah'
])

# Simpan ke CSV
df.to_csv('dataset_jurusan.csv', index=False)
print("✅ Selesai! File 'dataset_jurusan.csv' berhasil dibuat.")