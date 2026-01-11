import streamlit as st
import numpy as np
import pickle
import time

# --- CONFIG & LOAD MODEL ---
st.set_page_config(page_title="Chat Bot Rekomendasi Jurusan", page_icon="🤖")

@st.cache_resource
def load_model():
    try:
        with open('model_k-means.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("File 'model_k-means.pkl' tidak ditemukan. Harap jalankan 'train_model.py' terlebih dahulu.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

data = load_model()
kmeans = data["model"]
scaler = data["scaler"]
le_sekolah = data["le_sekolah"]
le_jurusan = data["le_jurusan"]
cluster_jurusan_map = data["cluster_map"]

# --- HELPER 1: REKOMENDASI HOBI (MINAT) ---
def get_hobby_recommendation(hobbies_list, custom_hobby=None):
    rekomendasi = []
    # Mapping Hobi -> Jurusan
    map_hobi = {
        '1': ['Ilmu Keolahragaan', 'Pendidikan Jasmani', 'Fisioterapi'], 
        '2': ['Seni Musik', 'Etnomusikologi', 'Seni Pertunjukan'], 
        '3': ['DKV', 'Film & Televisi', 'Multimedia', 'Arsitektur'], 
        '4': ['Teknik Informatika', 'Sistem Informasi', 'Ilmu Komputer', 'Game Dev'],
        '5': ['Sastra Inggris', 'Ilmu Komunikasi', 'Jurnalistik', 'Hubungan Internasional'], 
        '6': ['Manajemen', 'Bisnis Digital', 'Kewirausahaan', 'Akuntansi'], 
        '7': ['Teknik Mesin', 'Teknik Elektro', 'Teknik Industri', 'Teknik Otomotif']
    }
    
    for h in hobbies_list:
        if h in map_hobi:
            rekomendasi.extend(map_hobi[h])
            
    if custom_hobby:
        rekomendasi.append(f"Jurusan Terkait Minat '{custom_hobby}'")

    return list(set(rekomendasi))

# --- HELPER 2: REKOMENDASI NILAI TERTINGGI (BAKAT) ---
def get_grade_recommendation(d):
    rekomendasi = []
    
    # 1. Identifikasi Nilai Mapel Dasar & Peminatan
    scores = {
        'Matematika': d['mtk'],
        'B.Inggris': d['ing'],
        'B.Indo': d['indo'],
        'PKN': d['pkn']
    }
    
    # Tambahkan Mapel Spesifik Jurusan (n1, n2, n3)
    jurusan = d['jurusan']
    if jurusan == 'ipa':
        scores['Fisika'] = d.get('n1_step34', 0) 
        scores['Kimia'] = d['n1']
        scores['Biologi'] = d['n2']
    elif jurusan == 'ips':
        scores['Ekonomi'] = d.get('n1_step34', 0)
        scores['Geografi'] = d['n1']
        scores['Sosiologi'] = d['n2']
    elif jurusan == 'smk':
        scores['Kejuruan'] = d['lainnya']

    # 2. Cari Mapel dengan Nilai Tertinggi (Minimal 75)
    high_scores = {k: v for k, v in scores.items() if v >= 75}
    if not high_scores: 
        high_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:1]}

    top_subjects = sorted(high_scores, key=high_scores.get, reverse=True)[:2] 

    # 3. Mapping Mapel -> Jurusan
    map_mapel = {
        'Matematika': ['Teknik Informatika', 'Matematika Murni', 'Aktuaria', 'Statistika', 'Sistem Informasi'],
        'Fisika': ['Teknik Mesin', 'Teknik Elektro', 'Fisika Murni', 'Teknik Sipil', 'Arsitektur'],
        'Biologi': ['Kedokteran', 'Biologi', 'Farmasi', 'Gizi', 'Kehutanan'],
        'Kimia': ['Teknik Kimia', 'Farmasi', 'Kimia Murni', 'Teknologi Pangan'],
        'Ekonomi': ['Akuntansi', 'Manajemen', 'Ekonomi Pembangunan', 'Bisnis Digital'],
        'Sosiologi': ['Ilmu Komunikasi', 'Sosiologi', 'Hukum', 'Psikologi'],
        'Geografi': ['Geografi', 'Perencanaan Wilayah Kota', 'Pariwisata'],
        'B.Inggris': ['Sastra Inggris', 'Hubungan Internasional', 'Pariwisata'],
        'B.Indo': ['Sastra Indonesia', 'Jurnalistik', 'Ilmu Komunikasi'],
        'PKN': ['Hukum', 'Ilmu Politik'],
        'Kejuruan': ['Pendidikan Vokasi Lanjutan', 'Manajemen Bisnis']
    }

    for sub in top_subjects:
        if sub in map_mapel:
            rekomendasi.extend(map_mapel[sub])
            
    return list(set(rekomendasi))

# --- HELPER 3: PREDIKSI AI ---
def get_prediction(d):
    try:
        js_enc = le_sekolah.transform([d['asal'].lower()])[0]
        try:
            jur_enc = le_jurusan.transform([d['jurusan'].lower()])[0]
        except:
            jur_enc = le_jurusan.transform(['multimedia'])[0] 

        input_array = np.array([[
            js_enc, jur_enc, d['pkn'], d['mtk'], d['indo'], d['ing'], d['lainnya'],
            d.get('olr', 0), d.get('mus', 0), d.get('edit', 0), d.get('game', 0)
        ]])
        
        scaled = scaler.transform(input_array)
        cluster = kmeans.predict(scaled)[0]
        return cluster_jurusan_map.get(cluster, [])
    except:
        return []

# --- MAIN APP ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Perkenalkan Saya Bot Rekomendasi Jurusan. Siapa Namamu?"}]
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "step" not in st.session_state:
    st.session_state.step = 1

st.title("Chat Bot Rekomendasi :green[Jurusan Kuliah]")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ketik jawabanmu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # --- STEP 1-3.3 ---
        if st.session_state.step == 1:
            st.session_state.user_data['nama'] = prompt.split()[0].capitalize()
            full_response = f"Halo {st.session_state.user_data['nama']}! Lulusan **SMA** atau **SMK**?"
            st.session_state.step = 1.5

        elif st.session_state.step == 1.5:
            sekolah_input = prompt.lower().strip()
            if 'sma' in sekolah_input:
                st.session_state.user_data['asal'] = 'sma'
                full_response = "Jurusan **IPA** atau **IPS**?"
                st.session_state.step = 2
            elif 'smk' in sekolah_input:
                st.session_state.user_data['asal'] = 'smk'
                full_response = "Jurusan SMK-nya apa?"
                st.session_state.step = 2
            else:
                full_response = "Pilihan hanya **SMA** atau **SMK**."

        elif st.session_state.step == 2:
            st.session_state.user_data['jurusan'] = prompt.strip().lower()
            full_response = "Masukkan nilai **PKN** Kamu?"
            st.session_state.step = 3.1

        elif st.session_state.step == 3.1:
            try:
                st.session_state.user_data['pkn'] = float(prompt)
                full_response = "Sekarang Nilai **Matematika**?"
                st.session_state.step = 3.2
            except: full_response = "⚠️ Masukkan angka yang valid."
        
        elif st.session_state.step == 3.2:
            try:
                st.session_state.user_data['mtk'] = float(prompt)
                full_response = "Nilai **Bahasa Indonesia**?"
                st.session_state.step = 3.3
            except: full_response = "⚠️ Masukkan angka yang valid."
        
        elif st.session_state.step == 3.3:
            try:
                st.session_state.user_data['indo'] = float(prompt)
                full_response = "Nilai **Bahasa Inggris**?"
                st.session_state.step = 3.4
            except: full_response = "⚠️ Masukkan angka yang valid."

        elif st.session_state.step == 3.4:
            try:
                st.session_state.user_data['ing'] = float(prompt)
                jur = st.session_state.user_data['jurusan']
                if jur == 'ipa':
                    full_response = "Berapa nilai **Fisika**?"
                    st.session_state.step = 4.1 
                elif jur == 'ips':
                    full_response = "Berapa nilai **Ekonomi**?"
                    st.session_state.step = 4.1 
                else: 
                    full_response = "Berapa nilai **Produktif/Kejuruan**?"
                    st.session_state.step = 4.4
            except: full_response = "⚠️ Masukkan angka yang valid."

        # --- STEP 4 ---
        elif st.session_state.step == 4.1:
            try:
                st.session_state.user_data['n1_step34'] = float(prompt) 
                lbl = "Kimia" if st.session_state.user_data['jurusan'] == 'ipa' else "Geografi"
                full_response = f"Berapa nilai **{lbl}**?"
                st.session_state.step = 4.2
            except: full_response = "⚠️ Masukkan angka yang valid."

        elif st.session_state.step == 4.2:
            try:
                st.session_state.user_data['n1'] = float(prompt) 
                lbl = "Biologi" if st.session_state.user_data['jurusan'] == 'ipa' else "Sosiologi"
                full_response = f"Berapa nilai **{lbl}**?"
                st.session_state.step = 4.3
            except: full_response = "⚠️ Masukkan angka yang valid."

        elif st.session_state.step == 4.3:
            try:
                st.session_state.user_data['n2'] = float(prompt) 
                st.session_state.user_data['n3'] = 0 
                
                val_1 = st.session_state.user_data['n1_step34']
                val_2 = st.session_state.user_data['n1']
                val_3 = st.session_state.user_data['n2']
                st.session_state.user_data['lainnya'] = (val_1 + val_2 + val_3) / 3
                
                full_response = "Terakhir! Apa Hobimu? (Pilih angka, boleh lebih dari 1)\n\n" \
                                "1. Olahraga\n2. Musik\n3. Desain/Art\n4. Komputer/Game\n" \
                                "5. Menulis/Bicara\n6. Bisnis\n7. Teknik/Mesin\n8. Lainnya"
                st.session_state.step = 5
            except: full_response = "⚠️ Masukkan angka yang valid."

        elif st.session_state.step == 4.4:
            try:
                st.session_state.user_data['lainnya'] = float(prompt)
                full_response = "Terakhir! Apa Hobimu? (Pilih angka, boleh lebih dari 1 pisahkan dengan koma (cth : 1,2))\n\n" \
                                "1. Olahraga\n2. Musik\n3. Desain/Art\n4. Komputer/Game\n" \
                                "5. Menulis/Bicara\n6. Bisnis\n7. Teknik/Mesin\n8. Lainnya"
                st.session_state.step = 5
            except: full_response = "⚠️ Masukkan angka yang valid."

        # --- STEP 5 & 6 (LOGIC FINAL) ---
        elif st.session_state.step == 5:
            input_hobbies = prompt.split(',')
            hobbies_clean = [h.strip() for h in input_hobbies]
            st.session_state.user_data['hobbies_raw'] = hobbies_clean
            
            if '8' in hobbies_clean or '99' in hobbies_clean:
                full_response = "Apa hobi spesifik kamu?"
                st.session_state.step = 5.5
            else:
                st.session_state.step = 6

        elif st.session_state.step == 5.5:
            st.session_state.user_data['custom_hobby'] = prompt
            st.session_state.step = 6

        if st.session_state.step == 6:
            # 1. Update data hobi boolean
            h = {'olr':0, 'mus':0, 'edit':0, 'game':0}
            raw = st.session_state.user_data.get('hobbies_raw', [])
            if '1' in raw: h['olr'] = 1
            if '2' in raw: h['mus'] = 1
            if '3' in raw: h['edit'] = 1
            if '4' in raw: h['game'] = 1
            st.session_state.user_data.update(h)

            # 2. GET RECOMMENDATIONS
            rec_hobi = get_hobby_recommendation(raw, st.session_state.user_data.get('custom_hobby'))
            rec_nilai = get_grade_recommendation(st.session_state.user_data)
            rec_ai = get_prediction(st.session_state.user_data)

            # 3. INTERSECTION STRATEGY
            final_list = []
            
            # Prioritas 1: Sweet Spot (Hobi + Nilai Tinggi)
            sweet_spot = [jur for jur in rec_hobi if jur in rec_nilai]
            final_list.extend(sweet_spot)
            
            # Prioritas 2: Sisa Hobi
            for jur in rec_hobi:
                if jur not in final_list:
                    final_list.append(jur)
                    
            # Prioritas 3: Sisa Nilai Tinggi
            for jur in rec_nilai:
                if jur not in final_list:
                    final_list.append(jur)
            
            # Prioritas 4: AI
            for jur in rec_ai:
                if jur not in final_list:
                    final_list.append(jur)

            # --- LIMIT JADI 3 TERATAS SAJA ---
            final_list = final_list[:3]

            full_response = f"Analisis selesai! Berikut **Top {len(final_list)} Jurusan** yang paling cocok untukmu:\n\n"
            
            for i, jur in enumerate(final_list, 1):
                # Kasih tanda bintang jika itu adalah 'Sweet Spot'
                keterangan = "⭐ (Sangat Cocok)" if jur in sweet_spot else ""
                full_response += f"{i}. **{jur}** {keterangan}\n"
            
            full_response += "\n\n**Mau coba lagi? (Ya/Tidak)**"
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.step = 7

        elif st.session_state.step == 7:
            if "ya" in prompt.lower():
                st.session_state.user_data = {}
                st.session_state.step = 1
                full_response = "Oke, mari mulai lagi! Siapa namamu?"
                st.session_state.messages = [{"role": "assistant", "content": full_response}]
                st.rerun()
            else:
                full_response = "Terima kasih! Semoga sukses. 👋"

        if st.session_state.step not in [6, 7] and full_response:
            for chunk in full_response.split():
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})