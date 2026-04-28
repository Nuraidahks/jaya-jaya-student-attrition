import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURASI HALAMAN & MODEL
# ==========================================
st.set_page_config(page_title="Jaya Jaya Institut Analytics", page_icon="🎓", layout="wide")

@st.cache_resource
def load_model():
    # Pastikan lokasi file .pkl sudah benar sesuai struktur repositori Anda
    return joblib.load('model/student_model_tuned.pkl')

model = load_model()

# Mengambil daftar fitur aktual yang dipelajari model saat training (26 fitur)
expected_features = model.feature_names_in_

# ==========================================
# 2. HEADER APLIKASI
# ==========================================
st.title("🎓 Jaya Jaya Institut: Student Attrition Early Warning System")
st.markdown(f"Sistem prediktif ini menggunakan **{len(expected_features)} indikator utama** untuk mendeteksi potensi *dropout* mahasiswa secara dini.")
st.markdown("---")

tab1, tab2 = st.tabs(["🔍 Prediksi Individu", "📊 Insight Algoritma"])

# ==========================================
# 3. TAB 1: FORM PREDIKSI (DYNAMIC INPUTS)
# ==========================================
with tab1:
    st.subheader("Input Profil Mahasiswa Secara Lengkap")
    st.info("Silakan isi parameter di bawah ini. Sistem secara otomatis akan menyeleksi fitur yang relevan dengan algoritma.")
    
    # --- KELOMPOK 1: Demografi ---
    with st.expander("👤 1. Demografi & Latar Belakang", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Jenis Kelamin (Gender)", options=[0, 1], format_func=lambda x: "Wanita (0)" if x==0 else "Pria (1)")
            age_at_enrollment = st.number_input("Usia Saat Mendaftar", min_value=15, max_value=80, value=20)
            marital_status = st.selectbox("Status Pernikahan", options=[1, 2, 3, 4, 5, 6], format_func=lambda x: f"Kategori {x}")
        with col2:
            nacionality = st.number_input("Kewarganegaraan (Nacionality Kode)", min_value=1, max_value=200, value=1)
            international = st.selectbox("Mahasiswa Internasional?", options=[0, 1], format_func=lambda x: "Tidak (0)" if x==0 else "Ya (1)")
            displaced = st.selectbox("Pendatang (Displaced)?", options=[0, 1], format_func=lambda x: "Tidak (0)" if x==0 else "Ya (1)")
        with col3:
            special_needs = st.selectbox("Kebutuhan Khusus?", options=[0, 1], format_func=lambda x: "Tidak (0)" if x==0 else "Ya (1)")
            mothers_qual = st.number_input("Kualifikasi Ibu (Kode)", min_value=1, max_value=50, value=1)
            fathers_qual = st.number_input("Kualifikasi Ayah (Kode)", min_value=1, max_value=50, value=1)

    # --- KELOMPOK 2: Akademik Awal & Ekonomi ---
    with st.expander("🎓 2. Akademik Awal & Finansial", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            course = st.number_input("Program Studi (Kode Course)", min_value=1, max_value=9999, value=9147)
            daytime_evening = st.selectbox("Kelas Siang/Malam", options=[0, 1], format_func=lambda x: "Malam (0)" if x==0 else "Siang (1)")
            prev_qual = st.number_input("Kualifikasi Sebelumnya (Kode)", min_value=1, max_value=50, value=1)
        with col2:
            app_mode = st.number_input("Jalur Pendaftaran (Kode)", min_value=1, max_value=60, value=1)
            app_order = st.number_input("Urutan Pilihan Jurusan", min_value=0, max_value=10, value=1)
            prev_qual_grade = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=130.0)
        with col3:
            admission_grade = st.number_input("Nilai Masuk (Admission Grade)", min_value=0.0, max_value=200.0, value=120.0)
            scholarship = st.selectbox("Penerima Beasiswa?", options=[0, 1], format_func=lambda x: "Tidak (0)" if x==0 else "Ya (1)")
            tuition = st.selectbox("Status SPP (Up to date)?", options=[0, 1], format_func=lambda x: "Menunggak (0)" if x==0 else "Lancar (1)")
            debtor = st.selectbox("Punya Hutang/Tunggakan?", options=[0, 1], format_func=lambda x: "Tidak (0)" if x==0 else "Ya (1)")

    # --- KELOMPOK 3: Performa Semester 1 & 2 ---
    with st.expander("📚 3. Performa Akademik (Semester 1 & 2)", expanded=True):
        st.markdown("**Semester 1**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        s1_cred = c1.number_input("SKS Diakui (S1)", value=0)
        s1_enrolled = c2.number_input("SKS Diambil (S1)", value=6)
        s1_evals = c3.number_input("Evaluasi (S1)", value=8)
        s1_approved = c4.number_input("SKS Lulus (S1)", value=6)
        s1_grade = c5.number_input("IPK (S1)", value=12.0)
        s1_wo_evals = c6.number_input("Tanpa Evaluasi (S1)", value=0)

        st.markdown("**Semester 2**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        s2_cred = c1.number_input("SKS Diakui (S2)", value=0)
        s2_enrolled = c2.number_input("SKS Diambil (S2)", value=6)
        s2_evals = c3.number_input("Evaluasi (S2)", value=8)
        s2_approved = c4.number_input("SKS Lulus (S2)", value=6)
        s2_grade = c5.number_input("IPK (S2)", value=12.0)
        s2_wo_evals = c6.number_input("Tanpa Evaluasi (S2)", value=0)

    st.markdown("---")
    
    if st.button("🚀 Jalankan Prediksi Algoritma", use_container_width=True):
        
        # --- PROSES FEATURE ENGINEERING ---
        if scholarship == 0 and debtor == 1: fin_risk = 2 # Tinggi
        elif scholarship == 1 and debtor == 0: fin_risk = 1 # Rendah
        else: fin_risk = 0 # Menengah
        
        total_approved = s1_approved + s2_approved
        
        if age_at_enrollment <= 21: age_group = 0 
        elif age_at_enrollment <= 30: age_group = 1 
        else: age_group = 2 

        # --- MEMBUAT DATAFRAME KESELURUHAN ---
        input_data = {
            'Marital_status': marital_status, 'Application_mode': app_mode, 'Application_order': app_order,
            'Course': course, 'Daytime_evening_attendance': daytime_evening, 'Previous_qualification': prev_qual,
            'Previous_qualification_grade': prev_qual_grade, 'Nacionality': nacionality, 
            'Mothers_qualification': mothers_qual, 'Fathers_qualification': fathers_qual,
            'Admission_grade': admission_grade, 'Displaced': displaced, 'Educational_special_needs': special_needs,
            'Debtor': debtor, 'Tuition_fees_up_to_date': tuition, 'Gender': gender,
            'Scholarship_holder': scholarship, 'Age_at_enrollment': age_at_enrollment, 'International': international,
            'Curricular_units_1st_sem_credited': s1_cred, 'Curricular_units_1st_sem_enrolled': s1_enrolled,
            'Curricular_units_1st_sem_evaluations': s1_evals, 'Curricular_units_1st_sem_approved': s1_approved,
            'Curricular_units_1st_sem_grade': s1_grade, 'Curricular_units_1st_sem_without_evaluations': s1_wo_evals,
            'Curricular_units_2nd_sem_credited': s2_cred, 'Curricular_units_2nd_sem_enrolled': s2_enrolled,
            'Curricular_units_2nd_sem_evaluations': s2_evals, 'Curricular_units_2nd_sem_approved': s2_approved,
            'Curricular_units_2nd_sem_grade': s2_grade, 'Curricular_units_2nd_sem_without_evaluations': s2_wo_evals,
            'Financial_Risk': fin_risk, 'Total_Approved_Units': total_approved, 'Age_Group': age_group
        }
        
        df_input = pd.DataFrame([input_data])

        # --- INTELLIGENT FEATURE SYNCHRONIZATION ---
        try:
            # 1. Buang kolom yang tidak digunakan oleh model Anda di notebook
            for col in df_input.columns:
                if col not in expected_features:
                    df_input = df_input.drop(columns=[col])
                    
            # 2. Jika ada kolom model yang tidak sengaja terhapus, tambahkan dengan nilai default 0
            for col in expected_features:
                if col not in df_input.columns:
                    df_input[col] = 0
                    
            # 3. Urutkan posisi kolom agar 100% sama dengan saat model dilatih
            df_input = df_input[expected_features]
            
            # --- EKSEKUSI PREDIKSI BINARY (DROPOUT VS GRADUATE) ---
            prediction = model.predict(df_input)[0]
            probabilities = model.predict_proba(df_input)[0]
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.subheader("Hasil Keputusan")
                # Jika model mendeteksi kelas 0 (Dropout)
                if prediction == 0:
                    st.error("🔴 DROPOUT")
                    st.caption("Peringkat Risiko: KRITIS")
                # Jika model mendeteksi kelas 1 (Graduate)
                else:
                    st.success("🟢 GRADUATE")
                    st.caption("Peringkat Risiko: AMAN")
                    
            with res_col2:
                st.subheader("Keyakinan Model")
                st.write(f"Risiko Dropout: **{probabilities[0]:.1%}**")
                st.progress(float(probabilities[0]))
                
                st.write(f"Potensi Lulus (Graduate): **{probabilities[1]:.1%}**")
                st.progress(float(probabilities[1]))
                
        except Exception as e:
            st.error(f"Terjadi kesalahan sinkronisasi fitur: {e}")

# ==========================================
# 4. TAB 2: INSIGHT MODEL (FEATURE IMPORTANCE)
# ==========================================
with tab2:
    st.subheader("Bagaimana Algoritma Mengambil Keputusan?")
    st.write("Grafik di bawah ini menunjukkan indikator historis yang paling krusial dalam menentukan kelulusan mahasiswa berdasarkan pelatihan algoritma Random Forest.")
    
    try:
        # Menghitung Feature Importance statis agar konsisten dengan Notebook
        importances = model.feature_importances_
        
        # Membuat DataFrame untuk Plotting
        fi_df = pd.DataFrame({'Fitur': expected_features, 'Pengaruh (%)': importances * 100})
        # Urutkan berdasarkan Pengaruh (%) dari tinggi ke rendah, ambil Top 10
        fi_df = fi_df.sort_values(by='Pengaruh (%)', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # Gunakan warna seragam agar terlihat profesional
        sns.barplot(x='Pengaruh (%)', y='Fitur', data=fi_df, palette='viridis', ax=ax)
        ax.set_title("Top 10 Indikator Risiko", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Grafik Feature Importance belum dapat ditampilkan: {e}")
