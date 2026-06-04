import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# KAI Logo URL (Direct Image Link)
KAI_LOGO = "https://upload.wikimedia.org/wikipedia/commons/5/56/Logo_PT_Kereta_Api_Indonesia_%28Persero%29_2020.svg"

st.set_page_config(
    page_title="DSS Pemilihan Kereta Api - Metode AHP",
    page_icon=KAI_LOGO,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .kai-header {
        background: linear-gradient(135deg, #0D2E5C 0%, #163B6E 45%, #F26522 100%);
        padding: 28px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .kai-header img { 
        height: 58px;
        background: rgba(255, 255, 255, 0.9);
        padding: 6px 10px;
        border-radius: 8px;
    }
    .kai-header-text h1 {
        color: #fff; font-size: 26px; font-weight: 700;
        margin: 0; letter-spacing: -0.3px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .kai-header-text p {
        color: rgba(255,255,255,0.9); font-size: 14px;
        margin: 4px 0 0 0;
    }
    
    .step-badge {
        display: inline-block;
        background: #F26522; color: #fff;
        width: 28px; height: 28px; border-radius: 50%;
        text-align: center; line-height: 28px;
        font-weight: 600; font-size: 14px; margin-right: 8px;
    }
    .step-badge.inactive { background: #555; }
    .step-badge.done { background: #2ecc71; }
    
    .section-title {
        font-size: 20px; font-weight: 600; color: #1a1a2e;
        border-left: 4px solid #F26522; padding-left: 12px;
        margin: 24px 0 16px 0;
    }
    
    .rank-card {
        border-radius: 10px; padding: 16px 20px;
        margin-bottom: 8px; display: flex;
        align-items: center; gap: 16px;
    }
    .rank-card.gold { background: linear-gradient(135deg, #F7C948, #F5A623); color: #fff; }
    .rank-card.silver { background: linear-gradient(135deg, #BCC6CC, #8E9EAB); color: #fff; }
    .rank-card.bronze { background: linear-gradient(135deg, #C9956B, #A0724A); color: #fff; }
    .rank-card.normal { background: #f0f2f6; color: #1a1a2e; }
    
    .rank-number { font-size: 28px; font-weight: 700; min-width: 40px; text-align: center; }
    .rank-name { font-size: 16px; font-weight: 600; flex: 1; }
    .rank-score { font-size: 16px; font-weight: 500; }

    div[data-testid="stSidebar"] { background: #1a1a2e; }
    div[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    div[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def loadData():
    df = pd.read_csv('Data KAI PSE-YK,LPN.csv', sep=';', dtype={'Harga (Rp)': str})
    df['Harga (Rp)'] = df['Harga (Rp)'].str.replace('.', '').astype(int)
    return df

def perbandingan_berpasangan_ahp(values, is_lower_better=True):
    n = len(values)
    matriks = np.ones((n, n))
    safe_values = [max(v, 0.1) for v in values]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if is_lower_better:
                    rasio = safe_values[j] / safe_values[i]
                else:
                    rasio = safe_values[i] / safe_values[j]
                
                if rasio >= 1:
                    if rasio <= 1.1: matriks[i, j] = 1
                    elif rasio <= 1.3: matriks[i, j] = 2
                    elif rasio <= 1.5: matriks[i, j] = 3
                    elif rasio <= 1.8: matriks[i, j] = 4
                    elif rasio <= 2.1: matriks[i, j] = 5
                    elif rasio <= 2.5: matriks[i, j] = 6
                    elif rasio <= 3.0: matriks[i, j] = 7
                    elif rasio <= 3.5: matriks[i, j] = 8
                    else: matriks[i, j] = 9
                else:
                    matriks[i, j] = 1 / matriks[j, i]
    return matriks

def hitung_vektor_prioritas(matriks_kriteria):
    jumlah_kolom = matriks_kriteria.sum(axis=0)
    normalisasi_kriteria = matriks_kriteria / jumlah_kolom
    bobot_kriteria = normalisasi_kriteria.mean(axis=1)
    hasil_kali = np.dot(matriks_kriteria, bobot_kriteria)
    lambda_max = np.mean(hasil_kali / bobot_kriteria)
    return bobot_kriteria, lambda_max, normalisasi_kriteria

def main():
    st.markdown(f'''
    <div class="kai-header">
        <img src="{KAI_LOGO}" alt="KAI Logo">
        <div class="kai-header-text">
            <h1>Sistem Pendukung Keputusan Pemilihan Kereta Api</h1>
            <p>Metode Analytical Hierarchy Process (AHP) &mdash; Rute: Pasar Senen (PSE) &rarr; Yogyakarta (YK) / Lempuyangan (LPN)</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    data = loadData()
    
    if 'step' not in st.session_state: st.session_state.step = 1
    if 'stasiun_terpilih' not in st.session_state: st.session_state.stasiun_terpilih = None
    if 'alternatif' not in st.session_state: st.session_state.alternatif = []
    if 'bobot_kriteria' not in st.session_state: st.session_state.bobot_kriteria = None
    if 'skor_akhir' not in st.session_state: st.session_state.skor_akhir = None
    
    # Inisialisasi awal matriks kriteria untuk data_editor Step 3
    if 'matriks_berpasangan_df' not in st.session_state:
        kriteria = ["Biaya", "Jenis Rangkaian", "Waktu Sampai", "Waktu Tempuh", "Jml Pemberhentian"]
        default_matriks = np.ones((5, 5))
        st.session_state.matriks_berpasangan_df = pd.DataFrame(default_matriks, index=kriteria, columns=kriteria)
    
    with st.sidebar:
        st.image(KAI_LOGO, width=120)
        st.markdown("#### Langkah Analisis")
        
        steps = [
            "Filter Stasiun Tujuan",
            "Pilih 5 Alternatif",
            "Tentukan Bobot Kriteria",
            "Hitung dengan AHP",
            "Lihat Hasil Ranking"
        ]
        
        for i, step in enumerate(steps, 1):
            if i < st.session_state.step: badge = "done"
            elif i == st.session_state.step: badge = ""
            else: badge = "inactive"
            st.markdown(f'<span class="step-badge {badge}">{i}</span> {step}', unsafe_allow_html=True)
        
        st.divider()
        st.caption("Metode: Analytical Hierarchy Process (AHP)")
        st.caption("© Copyright 2026. All rights reserved Kelompok 10 SCPK KAI")
    
    if st.session_state.step == 1: langkah1_pilih_stasiun(data)
    elif st.session_state.step == 2: langkah2_pilih_alternatif(data)
    elif st.session_state.step == 3: langkah3_bobot_kriteria()
    elif st.session_state.step == 4: langkah4_perhitungan_ahp()
    elif st.session_state.step == 5: langkah5_hasil()

def langkah1_pilih_stasiun(data):
    st.markdown('<div class="section-title">Pilih Stasiun Tujuan</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("YOGYAKARTA (YK)", use_container_width=True, type="primary"):
            st.session_state.stasiun_terpilih = "YK"
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("LEMPUYANGAN (LPN)", use_container_width=True, type="primary"):
            st.session_state.stasiun_terpilih = "LPN"
            st.session_state.step = 2
            st.rerun()
            
    st.markdown("---")
    st.markdown('<div class="section-title">Preview Data Kereta</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Yogyakarta (YK)", "Lempuyangan (LPN)"])
    
    with tab1:
        yk_trains = data[data['KodeStasiunTujuan'] == "YK"].drop_duplicates(subset=['Nama Kereta', 'Kelas'])
        if len(yk_trains) > 0:
            display_df = yk_trains[['Nama Kereta', 'Kelas', 'Jenis Rangkaian', 'Harga (Rp)', 'Durasi (menit)', 'Jumlah Stasiun Pemberhentian']].copy()
            display_df['Harga (Rp)'] = display_df['Harga (Rp)'].apply(lambda x: f"Rp{x:,.0f}")
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("Tidak ada data kereta ke Yogyakarta")
            
    with tab2:
        lpn_trains = data[data['KodeStasiunTujuan'] == "LPN"].drop_duplicates(subset=['Nama Kereta', 'Kelas'])
        if len(lpn_trains) > 0:
            display_df = lpn_trains[['Nama Kereta', 'Kelas', 'Jenis Rangkaian', 'Harga (Rp)', 'Durasi (menit)', 'Jumlah Stasiun Pemberhentian']].copy()
            display_df['Harga (Rp)'] = display_df['Harga (Rp)'].apply(lambda x: f"Rp{x:,.0f}")
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("Tidak ada data kereta ke Lempuyangan")

def langkah2_pilih_alternatif(data):
    st.markdown('<div class="section-title">Pilih Kelas & Alternatif Kereta</div>', unsafe_allow_html=True)
    base_filtered_df = data[data['KodeStasiunTujuan'] == st.session_state.stasiun_terpilih].drop_duplicates(subset=['Nama Kereta', 'Kelas'])
    st.info(f"Stasiun tujuan: **{st.session_state.stasiun_terpilih}**")
    
    kelas_options = ["Semua Kelas"] + sorted(list(base_filtered_df['Kelas'].unique()))
    selected_kelas = st.radio("Filter Kelas Kereta:", kelas_options, horizontal=True)
    
    filtered_df = base_filtered_df if selected_kelas == "Semua Kelas" else base_filtered_df[base_filtered_df['Kelas'] == selected_kelas]
    st.caption(f"Tersedia {len(filtered_df)} pilihan kereta. Silakan pilih tepat 5 kereta.")
    
    daftar_alternatif = []
    for idx, row in filtered_df.iterrows():
        alt_text = f"{row['Nama Kereta']} | {row['Kelas']} | Rp{row['Harga (Rp)']:,.0f} | {row['Durasi (menit)']} mnt | {row['Jumlah Stasiun Pemberhentian']} st | {row['Jam Sampai']} | {row['Jenis Rangkaian']}"
        jam, menit = map(int, str(row['Jam Sampai']).split(':'))
        waktu_sampai = jam * 60 + menit
        
        r = str(row['Jenis Rangkaian']).lower()
        if 'new generasion' in r and 'eksekutif' in r: rangkaian_score = 5
        elif 'new generasion' in r: rangkaian_score = 4
        elif 'eksekutif' in r or 'premium' in r: rangkaian_score = 3
        elif 'modifikasi' in r: rangkaian_score = 2
        else: rangkaian_score = 1
            
        daftar_alternatif.append({
            'key': idx, 'text': alt_text, 'nama': row['Nama Kereta'], 'kelas': row['Kelas'],
            'harga': row['Harga (Rp)'], 'durasi': row['Durasi (menit)'], 'stasiun': row['Jumlah Stasiun Pemberhentian'],
            'rangkaian': rangkaian_score, 'waktu_sampai': waktu_sampai, 'rangkaian_nama': row['Jenis Rangkaian'], 'jam_sampai': row['Jam Sampai']
        })
    
    dipilih = st.multiselect("Pilih 5 alternatif kereta:", options=[alt['text'] for alt in daftar_alternatif])
    col1, col2 = st.columns(2)
    
    with col1:
        if len(dipilih) == 5:
            if st.button("Simpan Pilihan", use_container_width=True, type="primary"):
                st.session_state.alternatif = [alt for sel in dipilih for alt in daftar_alternatif if alt['text'] == sel]
                st.session_state.step = 3
                st.rerun()
        else:
            st.button("Simpan Pilihan", disabled=True, use_container_width=True)
    with col2:
        if st.button("↺ Reset Pilihan", use_container_width=True): st.rerun()
        
    st.markdown("---")
    st.metric("Jumlah Terpilih", f"{len(dipilih)} / 5")
    if dipilih:
        for i, sel in enumerate(dipilih, 1): st.success(f"{i}. {sel}")

def langkah3_bobot_kriteria():
    st.markdown('<div class="section-title">Tentukan Bobot Kriteria (Matriks Berpasangan Fleksibel)</div>', unsafe_allow_html=True)
    
    st.info("""
    💡 **Cara Pengisian:**
    1. Isi langsung nilai tingkat kepentingan kriteria pada tabel di bawah (Skala Saaty: **1 sampai 9**).
    2. Jika **Baris lebih penting dari Kolom**, masukkan angka **2 s/d 9** (Misal Biaya vs Jenis Rangkaian = 3).
    3. Jika **Kolom lebih penting dari Baris**, masukkan pecahan desimalnya (Misal **0.33**, **0.5**, dll).
    4. **Sistem otomatis menghitung sel kebalikannya secara real-time (1 / nilai dinput).**
    5. **DIAGONAL UTAMA TERKUNCI MUTLAK:** Perbandingan kriteria dengan dirinya sendiri (diagonal utama) wajib bernilai **1.0**. Jika Anda mencoba mengubahnya, nilai otomatis dibatalkan & dipaksa kembali ke **1.0**.
    """)
    
    kriteria = ["Biaya", "Jenis Rangkaian", "Waktu Sampai", "Waktu Tempuh", "Jml Pemberhentian"]
    df_saat_ini = st.session_state.matriks_berpasangan_df.copy()
    
    st.markdown("### 📊 Tabel Matriks Perbandingan Berpasangan")
    
    df_diedit = st.data_editor(
        df_saat_ini,
        disabled=["Index"], 
        num_rows="fixed",
        column_config={
            c: st.column_config.NumberColumn(min_value=0.01, max_value=9.0, format="%.3f") for c in kriteria
        },
        use_container_width=True
    )
    
    matriks_diubah = False
    reset_diagonal = False
    
    for i in range(len(kriteria)):
        for j in range(len(kriteria)):
            if i == j:
                if df_diedit.iloc[i, j] != 1.0:
                    df_diedit.iloc[i, j] = 1.0
                    reset_diagonal = True
                    matriks_diubah = True
            else:
                if df_diedit.iloc[i, j] != df_saat_ini.iloc[i, j]:
                    val = df_diedit.iloc[i, j]
                    if val <= 0: val = 0.01  
                    df_diedit.iloc[j, i] = round(1 / val, 4)
                    matriks_diubah = True
                elif df_diedit.iloc[j, i] != df_saat_ini.iloc[j, i]:
                    val = df_diedit.iloc[j, i]
                    if val <= 0: val = 0.01
                    df_diedit.iloc[i, j] = round(1 / val, 4)
                    matriks_diubah = True

    if matriks_diubah:
        st.session_state.matriks_berpasangan_df = df_diedit
        if reset_diagonal:
            st.toast("⚠️ Perbandingan dengan kriteria sendiri dikunci mutlak sebesar 1.0!", icon="🚫")
        st.rerun()

    matriks = df_diedit.to_numpy()
    columns1, columns2 = st.columns([1, 1])
    
    with columns1:
        if st.button("Hitung Bobot Kriteria & Konsistensi", use_container_width=True, type="primary"):
            bobot, lambda_max, matriks_ternormalisasi = hitung_vektor_prioritas(matriks)
            n = len(kriteria)
            ci = (lambda_max - n) / (n - 1) if n > 1 else 0
            ri = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
            cr = ci / ri[n-1] if ri[n-1] != 0 else 0
            
            st.session_state.bobot_kriteria = {
                'bobot': bobot, 'kriteria': kriteria, 'cr': cr, 'matriks': matriks, 
                'matriks_ternormalisasi': matriks_ternormalisasi, 'lambda_max': lambda_max, 'ci': ci
            }
    with columns2:
        if st.button("↺ Reset Tabel ke Default (1.0)", use_container_width=True):
            st.session_state.matriks_berpasangan_df = pd.DataFrame(np.ones((5, 5)), index=kriteria, columns=kriteria)
            st.session_state.bobot_kriteria = None
            st.rerun()
        
    if st.session_state.bobot_kriteria is not None:
        hasil_bobot = st.session_state.bobot_kriteria
        st.markdown('<div class="section-title">Hasil Analisis Bobot Kriteria</div>', unsafe_allow_html=True)
        
        # Menampilkan tabel normalisasi & bobot seperti kode awal
        st.markdown('##### Matriks Normalisasi & Bobot (W)')
        norm_df = pd.DataFrame(hasil_bobot['matriks_ternormalisasi'], index=kriteria, columns=kriteria)
        norm_df['Bobot (W)'] = hasil_bobot['bobot']
        st.dataframe(norm_df.style.format("{:.4f}"), use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Lambda Max (λ max)", f"{round(hasil_bobot['lambda_max'], 4)}")
        c2.metric("Consistency Index (CI)", f"{round(hasil_bobot['ci'], 4)}")
        c3.metric("Consistency rasio (CR)", f"{round(hasil_bobot['cr'], 4)}")
        
        if hasil_bobot['cr'] <= 0.1:
            st.success("✅ Matriks Konsisten! (CR ≤ 0.1). Silakan lanjut ke langkah berikutnya.")
            if st.button("Lanjut ke Perhitungan AHP ➡️", use_container_width=True):
                st.session_state.step = 4
                st.rerun()
        else:
            st.warning("⚠️ Matriks Anda tidak konsisten! (CR > 0.1). Silakan sesuaikan kembali nilai di tabel di atas agar logis.")

def langkah4_perhitungan_ahp():
    """Step 4: Mengembalikan detail matriks perbandingan alternatif per kriteria seperti file awal"""
    st.markdown('<div class="section-title">Perhitungan AHP (Matriks Ternilai Alternatif per Kriteria)</div>', unsafe_allow_html=True)
    if st.session_state.bobot_kriteria is None or not st.session_state.alternatif:
        st.error("Lengkapi langkah sebelumnya!"); return
        
    daftar_alternatif = st.session_state.alternatif
    bobot_kriteria = st.session_state.bobot_kriteria['bobot']
    
    st.info(f"Analisis Berpasangan untuk {len(daftar_alternatif)} alternatif pilihan ke stasiun {st.session_state.stasiun_terpilih}")
    
    nilai_harga = [alt['harga'] for alt in daftar_alternatif]
    nilai_rangkaian = [alt['rangkaian'] for alt in daftar_alternatif]
    nilai_waktu_sampai = [alt['waktu_sampai'] for alt in daftar_alternatif]
    nilai_durasi = [alt['durasi'] for alt in daftar_alternatif]
    nilai_stasiun = [alt['stasiun'] for alt in daftar_alternatif]
    nama_alternatif = [f"{alt['nama']} ({alt['kelas']})" for alt in daftar_alternatif]
    
    # 1. Biaya (Harga)
    st.markdown("### 1. Perbandingan Berdasarkan Biaya (Harga)")
    harga_matriks = perbandingan_berpasangan_ahp(nilai_harga, is_lower_better=True)
    bobot_harga, _, harga_norm = hitung_vektor_prioritas(harga_matriks)
    harga_df = pd.DataFrame(harga_matriks, index=nama_alternatif, columns=nama_alternatif)
    st.dataframe(harga_df.style.format("{:.2f}"), use_container_width=True)
    with st.expander("Lihat Matriks Normalisasi & Bobot (W) - Biaya"):
        harga_norm_df = pd.DataFrame(harga_norm, index=nama_alternatif, columns=nama_alternatif)
        harga_norm_df['Bobot (W)'] = bobot_harga
        st.dataframe(harga_norm_df.style.format("{:.4f}"), use_container_width=True)
                           
    # 2. Jenis Rangkaian
    st.markdown("### 2. Perbandingan Berdasarkan Jenis Rangkaian")
    rangkaian_matriks = perbandingan_berpasangan_ahp(nilai_rangkaian, is_lower_better=False)
    bobot_rangkaian, _, rangkaian_norm = hitung_vektor_prioritas(rangkaian_matriks)
    rangkaian_df = pd.DataFrame(rangkaian_matriks, index=nama_alternatif, columns=nama_alternatif)
    st.dataframe(rangkaian_df.style.format("{:.2f}"), use_container_width=True)
    with st.expander("Lihat Matriks Normalisasi & Bobot (W) - Jenis Rangkaian"):
        rangkaian_norm_df = pd.DataFrame(rangkaian_norm, index=nama_alternatif, columns=nama_alternatif)
        rangkaian_norm_df['Bobot (W)'] = bobot_rangkaian
        st.dataframe(rangkaian_norm_df.style.format("{:.4f}"), use_container_width=True)
                           
    # 3. Waktu Sampai
    st.markdown("### 3. Perbandingan Berdasarkan Waktu Sampai")
    waktu_sampai_matriks = perbandingan_berpasangan_ahp(nilai_waktu_sampai, is_lower_better=True)
    bobot_waktu_sampai, _, ws_norm = hitung_vektor_prioritas(waktu_sampai_matriks)
    ws_df = pd.DataFrame(waktu_sampai_matriks, index=nama_alternatif, columns=nama_alternatif)
    st.dataframe(ws_df.style.format("{:.2f}"), use_container_width=True)
    with st.expander("Lihat Matriks Normalisasi & Bobot (W) - Waktu Sampai"):
        ws_norm_df = pd.DataFrame(ws_norm, index=nama_alternatif, columns=nama_alternatif)
        ws_norm_df['Bobot (W)'] = bobot_waktu_sampai
        st.dataframe(ws_norm_df.style.format("{:.4f}"), use_container_width=True)

    # 4. Waktu Tempuh
    st.markdown("### 4. Perbandingan Berdasarkan Waktu Tempuh")
    durasi_matriks = perbandingan_berpasangan_ahp(nilai_durasi, is_lower_better=True)
    bobot_durasi, _, durasi_norm = hitung_vektor_prioritas(durasi_matriks)
    durasi_df = pd.DataFrame(durasi_matriks, index=nama_alternatif, columns=nama_alternatif)
    st.dataframe(durasi_df.style.format("{:.2f}"), use_container_width=True)
    with st.expander("Lihat Matriks Normalisasi & Bobot (W) - Waktu Tempuh"):
        durasi_norm_df = pd.DataFrame(durasi_norm, index=nama_alternatif, columns=nama_alternatif)
        durasi_norm_df['Bobot (W)'] = bobot_durasi
        st.dataframe(durasi_norm_df.style.format("{:.4f}"), use_container_width=True)

    # 5. Jumlah Pemberhentian
    st.markdown("### 5. Perbandingan Berdasarkan Jumlah Pemberhentian")
    stasiun_matriks = perbandingan_berpasangan_ahp(nilai_stasiun, is_lower_better=True)
    bobot_stasiun, _, stasiun_norm = hitung_vektor_prioritas(stasiun_matriks)
    stasiun_df = pd.DataFrame(stasiun_matriks, index=nama_alternatif, columns=nama_alternatif)
    st.dataframe(stasiun_df.style.format("{:.2f}"), use_container_width=True)
    with st.expander("Lihat Matriks Normalisasi & Bobot (W) - Jumlah Pemberhentian"):
        stasiun_norm_df = pd.DataFrame(stasiun_norm, index=nama_alternatif, columns=nama_alternatif)
        stasiun_norm_df['Bobot (W)'] = bobot_stasiun
        st.dataframe(stasiun_norm_df.style.format("{:.4f}"), use_container_width=True)
    
    # Ringkasan Prioritas Alternatif per Kriteria
    st.markdown('<div class="section-title">Prioritas Alternatif per Kriteria (Summary)</div>', unsafe_allow_html=True)
    prioritas_data = []
    for i, alt in enumerate(daftar_alternatif):
        prioritas_data.append({
            'Alternatif': f"{alt['nama']} ({alt['kelas']})",
            'Biaya': f"{bobot_harga[i]:.4f}",
            'Jenis Rangkaian': f"{bobot_rangkaian[i]:.4f}",
            'Waktu Sampai': f"{bobot_waktu_sampai[i]:.4f}",
            'Waktu Tempuh': f"{bobot_durasi[i]:.4f}",
            'Jml Pemberhentian': f"{bobot_stasiun[i]:.4f}"
        })
    st.dataframe(pd.DataFrame(prioritas_data), use_container_width=True)
    
    # Sintesis Global / Hitung Skor Akhir
    st.markdown('<div class="section-title">Sintesis Global (Skor Akhir)</div>', unsafe_allow_html=True)
    daftar_skor = []
    for i, alt in enumerate(daftar_alternatif):
        skor = (bobot_kriteria[0] * bobot_harga[i] + 
                 bobot_kriteria[1] * bobot_rangkaian[i] +
                 bobot_kriteria[2] * bobot_waktu_sampai[i] +
                 bobot_kriteria[3] * bobot_durasi[i] +
                 bobot_kriteria[4] * bobot_stasiun[i])
        
        detail_teks = (f"({bobot_kriteria[0]:.4f} × {bobot_harga[i]:.4f}) + "
                       f"({bobot_kriteria[1]:.4f} × {bobot_rangkaian[i]:.4f}) + "
                       f"({bobot_kriteria[2]:.4f} × {bobot_waktu_sampai[i]:.4f}) + "
                       f"({bobot_kriteria[3]:.4f} × {bobot_durasi[i]:.4f}) + "
                       f"({bobot_kriteria[4]:.4f} × {bobot_stasiun[i]:.4f})")
                       
        daftar_skor.append({
            'Alternatif': f"{alt['nama']} ({alt['kelas']})",
            'Skor': skor,
            'Detail': detail_teks
        })
    
    final_df = pd.DataFrame(daftar_skor).sort_values('Skor', ascending=False)
    
    show_detail = st.checkbox("Tampilkan Detail Rumus Perhitungan Sintesis")
    if show_detail:
        display_df = final_df.copy()
        display_df['Skor'] = display_df['Skor'].apply(lambda x: f"{x:.4f}")
        st.dataframe(display_df.rename(columns={'Detail': 'Detail (Σ Bobot Kriteria × Bobot Alternatif)'}), use_container_width=True)
    else:
        st.dataframe(final_df[['Alternatif', 'Skor']].style.format({"Skor": "{:.4f}"}), use_container_width=True)
        
    st.session_state.skor_akhir = daftar_skor
    
    columns1, columns2 = st.columns(2)
    with columns1:
        if st.button("Lihat Hasil Ranking ➡️", use_container_width=True, type="primary"):
            st.session_state.step = 5
            st.rerun()
    with columns2:
        if st.button("← Kembali ke Bobot Kriteria", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

def langkah5_hasil():
    st.markdown('<div class="section-title">Hasil Ranking Alternatif</div>', unsafe_allow_html=True)
    if st.session_state.skor_akhir is None: st.error("Hitung AHP dulu!"); return
    
    sorted_scores = sorted(st.session_state.skor_akhir, key=lambda x: x['Skor'], reverse=True)
    rank_classes = ['gold', 'silver', 'bronze']
    for i, score in enumerate(sorted_scores):
        cls = rank_classes[i] if i < 3 else 'normal'
        st.markdown(f'<div class="rank-card {cls}"><div class="rank-number">#{i+1}</div><div class="rank-name">{score["Alternatif"]}</div><div class="rank-score">Skor: {score["Skor"]:.4f}</div></div>', unsafe_allow_html=True)
        with st.expander(f"Detail Perhitungan #{i+1} — {score['Alternatif']}"):
            st.code(score['Detail'], language="text")
            
    st.markdown('<div class="section-title">Visualisasi Perbandingan Skor</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0E1117'); ax.set_facecolor('#0E1117')
    names = [s['Alternatif'] for s in sorted_scores]
    scores = [s['Skor'] for s in sorted_scores]
    bars = ax.barh(names, scores, color=['#F5A623', '#8E9EAB', '#A0724A'] + ['#F26522']*(len(scores)-3))
    ax.invert_yaxis(); ax.tick_params(colors='#e0e0e0')
    
    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
               f'{s:.4f}', va='center', fontsize=10, color='#e0e0e0')
               
    st.pyplot(fig)
    
    st.markdown("---")
    st.markdown('<div class="section-title">Rekomendasi</div>', unsafe_allow_html=True)
    st.success(f"**Berdasarkan perhitungan AHP, rekomendasi terbaik adalah:** \n\n **{sorted_scores[0]['Alternatif']}** dengan skor **{sorted_scores[0]['Skor']:.4f}**")
    
    columns1, columns2 = st.columns(2)
    with columns1:
        if st.button("Mulai Lagi dari Awal ↺", use_container_width=True):
            for k in ['step', 'stasiun_terpilih', 'alternatif', 'bobot_kriteria', 'skor_akhir', 'matriks_berpasangan_df']: 
                if k in st.session_state: del st.session_state[k]
            st.rerun()
    with columns2:
        if st.button("← Kembali ke Perhitungan", use_container_width=True):
            st.session_state.step = 4
            st.rerun()

if __name__ == "__main__":
    main()